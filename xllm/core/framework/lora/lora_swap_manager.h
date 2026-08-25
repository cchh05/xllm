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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(USE_NPU)
#include <acl/acl.h>
#endif

namespace xllm {

class LoRABlockPool;
class LoRAKVDependencyTree;

// [FASTLIBRA Phase 2] LoRASwapManager
//
// Applies FASTLIBRA (arxiv 2505.03756) Eq.3-6 cost model to swap LoRA
// adapters between HBM (LoRABlockPool slab) and host pinned memory.
// Runs a 100ms tick thread per rank; decision is deterministic on
// identical (tree state + HBM state + adapter prob estimate) so 4
// ranks converge to the same swap set without explicit cross-rank RPC.
//
// Design invariants (paper §3.2):
// - Swap-out only from HBM adapters whose LoRA dependency tree ref_count
//   is 0 (no in-flight requests) AND whose Retain_Eval is lowest.
// - Swap-in only from main-mem adapters whose Retain_Eval is highest,
//   as long as HBM utilization stays under high_threshold.
// - HBM high/low water threshold (default 95%/70%) creates a hysteresis
//   band that prevents ping-pong.
//
// Phase 2 scope: LoRA-level swap only. KV block swap deferred to Phase 4+.
// Arrival rate estimator hardcodes prob_i=1.0; real estimator lands in
// Phase 3 bench harness.
class LoRASwapManager {
 public:
  enum class Residency : uint8_t {
    HBM = 0,       // in slab, ready for forward
    MAIN_MEM = 1,  // in host pinned mirror, swap-in required to use
    EVICTING = 2,  // in-flight D2H copy from slab to host mirror
    LOADING = 3,   // in-flight H2D copy from host mirror to slab
  };

  struct Options {
    // HBM utilization thresholds (percent, 0-100). Tick loop swaps out
    // when HBM > high, swaps in when HBM < low. Band prevents ping-pong.
    uint32_t hbm_high_threshold_pct = 95;
    uint32_t hbm_low_threshold_pct = 70;
    // Tick loop period. Paper Table 1 uses 100ms; smaller = faster
    // reaction to burst but more overhead.
    uint32_t tick_ms = 100;
    // Cap on total host pinned bytes to avoid unbounded host mem growth.
    // Default 12 GiB accommodates ~30 adapters at 400MB each.
    uint64_t max_host_mirror_bytes = 12ULL * 1024 * 1024 * 1024;
    // Max blocks to swap in a single tick pass. Prevents one tick from
    // hogging the swap stream too long.
    uint32_t swap_batch = 32;
  };

  struct Stats {
    size_t n_hbm_adapters = 0;
    size_t n_main_mem_adapters = 0;
    size_t n_hbm_bytes = 0;
    size_t n_main_mem_bytes = 0;
    size_t n_swap_in_count = 0;
    size_t n_swap_out_count = 0;
    double hbm_util_pct = 0.0;
    int64_t last_tick_us = 0;
  };

  LoRASwapManager(LoRABlockPool* pool,
                  LoRAKVDependencyTree* tree,
                  const Options& opts);
  // Destructor defined in .cpp so unique_ptr<Stream> sees complete Stream.
  ~LoRASwapManager();

  LoRASwapManager(const LoRASwapManager&) = delete;
  LoRASwapManager& operator=(const LoRASwapManager&) = delete;

  // ---------- Adapter lifecycle ----------
  // Called from LoRARuntime install path after LoRABlockPool blocks have
  // been claimed. Ownership of the block ids is tracked by the swap manager
  // for the adapter's lifetime; on unregister we hand them back to the
  // pool. `slab_bytes` is the sum of all block bytes for this adapter,
  // used to size the host pinned mirror when eviction happens.
  void register_adapter(uint64_t int_id,
                        const std::string& lora_name,
                        std::vector<int32_t> blocks,
                        size_t slab_bytes);

  // Called from LoRARuntime unload path *before* freeing blocks back to
  // pool. If an in-flight swap is pending, sync-waits for the aclrtEvent
  // to complete (Option A per Phase 2 design), then returns the block
  // ids so the caller can free them.
  std::vector<int32_t> unregister_adapter(uint64_t int_id);

  // Option A install-time OOM handler: evict up to n_blocks_needed
  // worth of adapter blocks (swap them out to host mirror + free
  // blocks back to pool). Called synchronously from LoRARuntime
  // install path when pool.allocate fails. Skips adapters with
  // ref_count > 0 (active requests). Returns number of blocks freed.
  size_t try_evict_for_install(uint32_t n_blocks_needed);

  // ---------- Tick control ----------
  // Start the 100ms decision thread. Idempotent — subsequent calls no-op.
  void start_tick_thread();
  // Signal + join the tick thread. Called from destructor; idempotent.
  void stop_tick_thread();

  // ---------- Diagnostics ----------
  Stats get_stats() const;
  // Multi-line dump of per-adapter state for manual verification.
  // Not stable output for parsing.
  std::string dump_state() const;

 private:
  // Per-adapter bookkeeping. Held under mu_.
  struct AdapterEntry {
    std::string lora_name;
    std::vector<int32_t> hbm_blocks;  // ids into LoRABlockPool slab
    size_t slab_bytes = 0;
    // Host pinned mirror. Materialized lazily on first eviction and
    // reused. torch::Tensor with device=CPU + pinned_memory=true.
    torch::Tensor main_mem_mirror;
    Residency residency = Residency::HBM;
    // Paper prob_i. Phase 2 hardcodes 1.0; Phase 3 estimator overwrites.
    double prob_estimate = 1.0;
    int64_t last_swap_us = 0;
#if defined(USE_NPU)
    // In-flight swap tracking. Nonnull only while residency is
    // EVICTING or LOADING. Signaled by the async memcpy on swap_stream_.
    aclrtEvent pending_event = nullptr;
#endif
  };

  // ---------- Cost model (paper Eq.3-6) ----------
  // Eq.3: expected active LoRAs in current batch given prob estimates.
  double compute_low_lora(double bs) const;
  // Eq.4: per-adapter LoRA_Eval = max(1, low_lora / n_hbm_adapters).
  double compute_lora_eval(size_t n_hbm_adapters, double low_lora) const;
  // Eq.5: per-adapter Retain_Eval = cost * prob * (1 - sigmoid(t_since_use)).
  double compute_retain_eval(const AdapterEntry& e, int64_t now_us) const;
  // Eq.6: composite score. Higher score = keep, lower = evict candidate.
  double compute_swap_score(const AdapterEntry& e,
                            double lora_eval,
                            int64_t now_us) const;

  // ---------- Tick loop ----------
  void tick_loop();
  void do_swap_decisions(int64_t now_us);
  void swap_out_async(uint64_t int_id, AdapterEntry& e);
  void swap_in_async(uint64_t int_id, AdapterEntry& e);
  // Poll in-flight events non-blocking; complete state transitions.
  void reclaim_completed_swaps();
  // Read HBM utilization via Device::get_device_mem.
  double read_hbm_util_pct() const;

  LoRABlockPool* pool_;         // non-owning
  LoRAKVDependencyTree* tree_;  // non-owning
  Options opts_;

  mutable std::mutex mu_;
  std::unordered_map<uint64_t, AdapterEntry> adapters_;
  uint64_t total_host_mirror_bytes_ = 0;

  // Dedicated stream for LoRA D2H/H2D async copies. Lazy-created in
  // P2.C+ when the first swap fires; stored as void* here to avoid
  // pulling Stream's full definition into every user of this header.
  // Cast back to xllm::Stream* in .cpp where the type is complete.
  void* swap_stream_ = nullptr;

  // Tick thread lifecycle.
  std::atomic<bool> stop_{false};
  std::atomic<bool> tick_started_{false};
  std::thread tick_thread_;

  // Cumulative stats (public via get_stats()).
  Stats stats_;
};

}  // namespace xllm
