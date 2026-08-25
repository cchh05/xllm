/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "lora_swap_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include "lora_block_pool.h"
#include "lora_kv_dependency_tree.h"

namespace xllm {

namespace {
int64_t now_micros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

LoRASwapManager::LoRASwapManager(LoRABlockPool* pool,
                                 LoRAKVDependencyTree* tree,
                                 const Options& opts)
    : pool_(pool), tree_(tree), opts_(opts) {
  CHECK(pool_ != nullptr) << "LoRASwapManager: pool_ null";
  CHECK(tree_ != nullptr) << "LoRASwapManager: tree_ null";
  CHECK_LT(opts_.hbm_low_threshold_pct, opts_.hbm_high_threshold_pct)
      << "LoRASwapManager: hbm_low must be < hbm_high";

  // swap_stream_ is lazy-initialized in P2.C+ when the first swap is
  // issued, because it needs the model's device index / stream pool
  // handle which LoRARuntime obtains later.

  LOG(INFO) << "[LoRASwapManager] init hbm_high="
            << opts_.hbm_high_threshold_pct
            << "% hbm_low=" << opts_.hbm_low_threshold_pct
            << "% tick_ms=" << opts_.tick_ms
            << " max_host_bytes=" << opts_.max_host_mirror_bytes
            << " swap_batch=" << opts_.swap_batch;
}

LoRASwapManager::~LoRASwapManager() {
  stop_tick_thread();

#if defined(USE_NPU)
  // Free any pending aclrtEvents. Adapters map cleanup runs its own dtor
  // for the map entries; here we just destroy dangling event handles.
  std::lock_guard<std::mutex> g(mu_);
  for (auto& [id, entry] : adapters_) {
    if (entry.pending_event != nullptr) {
      aclrtDestroyEvent(entry.pending_event);
      entry.pending_event = nullptr;
    }
  }
#endif
}

// ---------- Adapter lifecycle ----------

void LoRASwapManager::register_adapter(uint64_t int_id,
                                       const std::string& lora_name,
                                       std::vector<int32_t> blocks,
                                       size_t slab_bytes) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = adapters_.find(int_id);
  if (it != adapters_.end()) {
    LOG(WARNING) << "[LoRASwapManager] register_adapter: duplicate int_id="
                 << int_id << " name='" << lora_name << "', ignoring";
    return;
  }
  AdapterEntry e;
  e.lora_name = lora_name;
  e.hbm_blocks = std::move(blocks);
  e.slab_bytes = slab_bytes;
  e.residency = Residency::HBM;
  e.prob_estimate = 1.0;  // Phase 2 placeholder; Phase 3 estimator overwrites.
  e.last_swap_us = now_micros();
  adapters_.emplace(int_id, std::move(e));
  LOG(INFO) << "[LoRASwapManager] register_adapter id=" << int_id << " name='"
            << lora_name << "' n_blocks=" << adapters_[int_id].hbm_blocks.size()
            << " slab_bytes=" << slab_bytes;
}

std::vector<int32_t> LoRASwapManager::unregister_adapter(uint64_t int_id) {
  std::vector<int32_t> out;

#if defined(USE_NPU)
  // First pass under lock: snapshot the pending_event (if any) so we can
  // release the lock while sync-waiting. Otherwise the tick thread could
  // block waiting to acquire mu_ during our wait.
  aclrtEvent pending_snapshot = nullptr;
  {
    std::lock_guard<std::mutex> g(mu_);
    auto it = adapters_.find(int_id);
    if (it == adapters_.end()) {
      LOG(WARNING) << "[LoRASwapManager] unregister_adapter: unknown id="
                   << int_id;
      return out;
    }
    pending_snapshot = it->second.pending_event;
    it->second.pending_event = nullptr;  // take ownership
  }

  if (pending_snapshot != nullptr) {
    LOG(INFO) << "[LoRASwapManager] unregister_adapter id=" << int_id
              << " sync-waiting for in-flight swap event to complete";
    auto ret = aclrtSynchronizeEvent(pending_snapshot);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "[LoRASwapManager] unregister_adapter aclrtSynchronizeEvent"
                 << " failed ret=" << ret;
    }
    aclrtDestroyEvent(pending_snapshot);
  }
#endif

  // Second pass: pull the entry out under lock and hand blocks back.
  std::lock_guard<std::mutex> g(mu_);
  auto it = adapters_.find(int_id);
  if (it == adapters_.end()) {
    // Concurrent unregister_adapter for same id (shouldn't happen but
    // defensive). Nothing more to do.
    return out;
  }
  out = std::move(it->second.hbm_blocks);
  if (it->second.slab_bytes > 0 && it->second.residency != Residency::HBM &&
      total_host_mirror_bytes_ >= it->second.slab_bytes) {
    total_host_mirror_bytes_ -= it->second.slab_bytes;
  }
  LOG(INFO) << "[LoRASwapManager] unregister_adapter id=" << int_id << " name='"
            << it->second.lora_name << "' returning n_blocks=" << out.size();
  adapters_.erase(it);
  return out;
}

// ---------- Tick lifecycle ----------

void LoRASwapManager::start_tick_thread() {
  bool expected = false;
  if (!tick_started_.compare_exchange_strong(expected, true)) {
    return;  // already started
  }
  stop_.store(false);
  tick_thread_ = std::thread([this] { tick_loop(); });
  LOG(INFO) << "[LoRASwapManager] tick thread started period_ms="
            << opts_.tick_ms;
}

void LoRASwapManager::stop_tick_thread() {
  if (!tick_started_.load()) return;
  stop_.store(true);
  if (tick_thread_.joinable()) {
    tick_thread_.join();
  }
  tick_started_.store(false);
  LOG(INFO) << "[LoRASwapManager] tick thread stopped";
}

// ---------- Tick loop (P2.C fills in the real work) ----------

void LoRASwapManager::tick_loop() {
  const auto period = std::chrono::milliseconds(opts_.tick_ms);
#if defined(USE_NPU)
  // aclrtGetMemInfo requires a device context on the calling thread.
  // Bind this tick thread to device 0 (rank 0 uses device 0 in TP
  // layout; per-rank setDevice would be needed for worker rank ticks).
  aclrtSetDevice(0);
#endif
  while (!stop_.load()) {
    const int64_t start_us = now_micros();
    reclaim_completed_swaps();
    // Update stats snapshot under mu_ before making decisions.
    {
      std::lock_guard<std::mutex> g(mu_);
      stats_.last_tick_us = start_us;
      stats_.n_hbm_adapters = 0;
      stats_.n_main_mem_adapters = 0;
      stats_.n_hbm_bytes = 0;
      stats_.n_main_mem_bytes = 0;
      for (const auto& [id, e] : adapters_) {
        if (e.residency == Residency::HBM ||
            e.residency == Residency::LOADING) {
          stats_.n_hbm_adapters++;
          stats_.n_hbm_bytes += e.slab_bytes;
        } else if (e.residency == Residency::MAIN_MEM ||
                   e.residency == Residency::EVICTING) {
          stats_.n_main_mem_adapters++;
          stats_.n_main_mem_bytes += e.slab_bytes;
        }
      }
      stats_.hbm_util_pct = read_hbm_util_pct();
    }
    // Every-tick trace for Phase 3 bench diag. Off by default; enable
    // with --v=1 at the xllm CLI.
    VLOG(1) << "[LoRASwapManager] tick hbm_util=" << stats_.hbm_util_pct
            << "% n_hbm=" << stats_.n_hbm_adapters
            << " n_main=" << stats_.n_main_mem_adapters
            << " hbm_bytes=" << stats_.n_hbm_bytes
            << " main_bytes=" << stats_.n_main_mem_bytes
            << " swap_in_total=" << stats_.n_swap_in_count
            << " swap_out_total=" << stats_.n_swap_out_count;
    do_swap_decisions(start_us);
    std::this_thread::sleep_for(period);
  }
}

void LoRASwapManager::do_swap_decisions(int64_t now_us) {
  // Snapshot state under mu_, choose swap-in/out targets, then release
  // mu_ before invoking the async kernels (P2.C+ real, P2.C stub).
  std::vector<uint64_t> evict_ids;
  std::vector<uint64_t> load_ids;
  bool decisions_logged = false;
  {
    std::lock_guard<std::mutex> g(mu_);
    if (adapters_.empty()) return;
    const double hbm_util = stats_.hbm_util_pct;
    const bool over_high = hbm_util > opts_.hbm_high_threshold_pct;
    const bool under_low = hbm_util < opts_.hbm_low_threshold_pct;
    if (!over_high && !under_low) return;
    // Estimate batch size from tree: use n_hbm_adapters as a proxy in
    // Phase 2 (Phase 3 wires the real scheduler bs). Bounded [1, 32].
    const double bs = std::max<double>(
        1.0,
        std::min<double>(32.0, static_cast<double>(stats_.n_hbm_adapters)));
    const double low_lora = compute_low_lora(bs);
    const size_t n_hbm = stats_.n_hbm_adapters;
    const double lora_eval = compute_lora_eval(n_hbm, low_lora);
    // Score every adapter; separate HBM (eviction candidates) from
    // MAIN_MEM (swap-in candidates). EVICTING/LOADING are already
    // in-flight and are skipped.
    std::vector<std::pair<double, uint64_t>> hbm_scores;
    std::vector<std::pair<double, uint64_t>> main_scores;
    for (const auto& [id, e] : adapters_) {
      const double score = compute_swap_score(e, lora_eval, now_us);
      if (e.residency == Residency::HBM) {
        hbm_scores.emplace_back(score, id);
      } else if (e.residency == Residency::MAIN_MEM) {
        main_scores.emplace_back(score, id);
      }
    }
    // Ascending for eviction (lowest score = worst = evict first).
    std::sort(hbm_scores.begin(), hbm_scores.end());
    // Descending for swap-in (highest score = best = load first).
    std::sort(main_scores.begin(), main_scores.end(), [](auto& a, auto& b) {
      return a.first > b.first;
    });
    if (over_high) {
      // Evict up to swap_batch adapters. Skip adapters that the tree
      // says are still referenced by admitted requests (ref_count > 0).
      uint32_t picked = 0;
      for (auto& [score, id] : hbm_scores) {
        if (picked >= opts_.swap_batch) break;
        const auto& e = adapters_[id];
        // Consult tree ref_count to avoid evicting an active adapter.
        // Phase 2 uses a coarse lookup; Phase 3 refines w/ per-request
        // arrival prob.
        // (tree_->lookup would go here; kept out of P2.C to avoid
        // header cycles. Ref_count guard lands in P2.C+ w/ full API.)
        (void)e;
        evict_ids.push_back(id);
        ++picked;
      }
    } else if (under_low) {
      uint32_t picked = 0;
      for (auto& [score, id] : main_scores) {
        if (picked >= opts_.swap_batch) break;
        load_ids.push_back(id);
        ++picked;
      }
    }
    if (!evict_ids.empty() || !load_ids.empty()) {
      LOG(INFO) << "[LoRASwapManager] tick decision hbm_util=" << hbm_util
                << "% n_hbm=" << n_hbm
                << " n_main=" << stats_.n_main_mem_adapters
                << " evict=" << evict_ids.size() << " load=" << load_ids.size();
      decisions_logged = true;
    }
  }
  // Fire async kernels outside mu_. Each helper reacquires mu_ briefly
  // to transition residency HBM->EVICTING or MAIN_MEM->LOADING and to
  // record the aclrtEvent (P2.C+).
  for (uint64_t id : evict_ids) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = adapters_.find(id);
    if (it != adapters_.end() && it->second.residency == Residency::HBM) {
      swap_out_async(id, it->second);
    }
  }
  for (uint64_t id : load_ids) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = adapters_.find(id);
    if (it != adapters_.end() && it->second.residency == Residency::MAIN_MEM) {
      swap_in_async(id, it->second);
    }
  }
  (void)decisions_logged;
}

void LoRASwapManager::reclaim_completed_swaps() {
  // Phase 2.C+ fills this. For P2.B skeleton, no-op.
}

void LoRASwapManager::swap_out_async(uint64_t /*int_id*/, AdapterEntry& /*e*/) {
  // Phase 2.C+ fills this.
}

void LoRASwapManager::swap_in_async(uint64_t /*int_id*/, AdapterEntry& /*e*/) {
  // Phase 2.C+ fills this.
}

// ---------- Cost model (paper Eq.3-6) ----------

// Eq.3: expected active LoRAs given per-adapter arrival prob and BS.
// Low_lora = Sum_i (1 - (1 - prob_i)^BS). Requires mu_ held by caller.
double LoRASwapManager::compute_low_lora(double bs) const {
  double sum = 0.0;
  for (const auto& [id, e] : adapters_) {
    const double p = std::max(0.0, std::min(1.0, e.prob_estimate));
    sum += (1.0 - std::pow(1.0 - p, bs));
  }
  return sum;
}

// Eq.4: LoRA_Eval = max(1, low_lora / n_hbm_adapters).
double LoRASwapManager::compute_lora_eval(size_t n_hbm, double low_lora) const {
  if (n_hbm == 0) return 1.0;
  const double e = low_lora / static_cast<double>(n_hbm);
  return std::max(1.0, e);
}

// Eq.5: Retain_Eval = cost * prob * (1 - sigmoid(t_since_use_us)).
// cost is a byte-normalized swap cost proxy; sigmoid uses paper §3.2
// scaled t argument. Higher retain_eval = more valuable to keep in HBM.
// Requires mu_ held by caller.
double LoRASwapManager::compute_retain_eval(const AdapterEntry& e,
                                            int64_t now_us) const {
  const double dt_sec = static_cast<double>(now_us - e.last_swap_us) / 1e6;
  // Paper §3.2: sigmoid(dt / tau) with tau on the order of seconds.
  // Use tau=5s so dt=5s -> sigmoid(1) ~= 0.73, tail beyond 30s ~= 1.
  const double tau = 5.0;
  const double x = dt_sec / tau;
  const double sigmoid = 1.0 / (1.0 + std::exp(-x));
  // cost proxy: normalized slab bytes. Larger adapter = higher swap
  // cost = higher retain incentive.
  const double cost =
      static_cast<double>(e.slab_bytes) / (256.0 * 1024.0 * 1024.0);
  const double p = std::max(0.0, std::min(1.0, e.prob_estimate));
  return cost * p * (1.0 - sigmoid);
}

// Eq.6: Eval_i = LoRA_Eval * Retain_Eval. Higher = keep in HBM.
double LoRASwapManager::compute_swap_score(const AdapterEntry& e,
                                           double lora_eval,
                                           int64_t now_us) const {
  return lora_eval * compute_retain_eval(e, now_us);
}

// ---------- HBM monitor ----------

double LoRASwapManager::read_hbm_util_pct() const {
  // Try device HBM first via aclrtGetMemInfo. If that fails (missing
  // device context on this thread, or driver returns 0), fall back
  // to a LoRA-slab-bytes proxy so the swap decision path can still
  // exercise under pressure. Caller must hold mu_ if reading stats_.
#if defined(USE_NPU)
  size_t free_memory = 0;
  size_t total_memory = 0;
  const auto ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
  if (ret == ACL_SUCCESS && total_memory > 0) {
    const size_t used = total_memory - free_memory;
    return 100.0 * static_cast<double>(used) /
           static_cast<double>(total_memory);
  }
#endif
  // Fallback proxy: LoRA slab used / (pool total slab bytes).
  if (pool_ == nullptr) return 0.0;
  const uint64_t block_bytes = pool_->block_bytes();
  const uint64_t total_blocks = pool_->num_total_blocks();
  const uint64_t slab_total = block_bytes * total_blocks;
  if (slab_total == 0) return 0.0;
  size_t used_bytes = 0;
  for (const auto& [id, e] : adapters_) {
    if (e.residency == Residency::HBM || e.residency == Residency::LOADING) {
      used_bytes += e.slab_bytes;
    }
  }
  return 100.0 * static_cast<double>(used_bytes) /
         static_cast<double>(slab_total);
}

// ---------- Diagnostics ----------

LoRASwapManager::Stats LoRASwapManager::get_stats() const {
  std::lock_guard<std::mutex> g(mu_);
  return stats_;
}

std::string LoRASwapManager::dump_state() const {
  std::lock_guard<std::mutex> g(mu_);
  std::ostringstream oss;
  oss << "[LoRASwapManager] state: n_adapters=" << adapters_.size()
      << " hbm_util_pct=" << stats_.hbm_util_pct
      << " host_mirror_bytes=" << total_host_mirror_bytes_ << "\n";
  for (const auto& [id, e] : adapters_) {
    const char* res =
        (e.residency == Residency::HBM)
            ? "HBM"
            : (e.residency == Residency::MAIN_MEM
                   ? "MAIN"
                   : (e.residency == Residency::EVICTING ? "EVICT" : "LOAD"));
    oss << "  id=" << id << " name='" << e.lora_name << "' res=" << res
        << " n_blocks=" << e.hbm_blocks.size() << " slab_bytes=" << e.slab_bytes
        << " prob=" << e.prob_estimate << "\n";
  }
  return oss.str();
}

}  // namespace xllm
