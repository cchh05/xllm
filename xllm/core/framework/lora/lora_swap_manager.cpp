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

#include <chrono>
#include <cmath>
#include <sstream>
#include <utility>

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
  while (!stop_.load()) {
    const int64_t start_us = now_micros();
    // Phase 2 skeleton: reclaim finished swaps, but do NOT yet issue new
    // swap decisions. P2.C wires do_swap_decisions.
    reclaim_completed_swaps();
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
    std::this_thread::sleep_for(period);
  }
}

void LoRASwapManager::do_swap_decisions(int64_t /*now_us*/) {
  // Phase 2.C fills this. For P2.B skeleton, no-op.
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

// ---------- Cost model (P2.C fills the real math) ----------

double LoRASwapManager::compute_low_lora(double /*bs*/) const {
  return 0.0;  // P2.C impl
}

double LoRASwapManager::compute_lora_eval(size_t /*n_hbm*/,
                                          double /*low_lora*/) const {
  return 0.0;  // P2.C impl
}

double LoRASwapManager::compute_retain_eval(const AdapterEntry& /*e*/,
                                            int64_t /*now_us*/) const {
  return 0.0;  // P2.C impl
}

double LoRASwapManager::compute_swap_score(const AdapterEntry& /*e*/,
                                           double /*lora_eval*/,
                                           int64_t /*now_us*/) const {
  return 0.0;  // P2.C impl
}

// ---------- HBM monitor ----------

double LoRASwapManager::read_hbm_util_pct() const {
#if defined(USE_NPU)
  size_t free_memory = 0;
  size_t total_memory = 0;
  const auto ret = aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
  if (ret != ACL_SUCCESS || total_memory == 0) return 0.0;
  const size_t used = total_memory - free_memory;
  return 100.0 * static_cast<double>(used) / static_cast<double>(total_memory);
#else
  return 0.0;
#endif
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
