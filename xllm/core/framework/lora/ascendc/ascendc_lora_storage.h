/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AscendCLoRAStorage — singleton holding N=8 pre-stacked adapter slabs
 * for AscendC bgmv/sgmv kernels. Populated via LoRARuntime install/unload
 * callbacks so per-forward path has zero organize overhead.
 *
 * Storage layout per (layer, proj):
 *   A_stacked:  [N, R, H_in]   fp16 device
 *   B_stacked:  [N, H_out, R]  fp16 device
 *   scaling:    float          (shared per adapter, we currently assume
 *                               all N adapters share same scaling — if not,
 *                               fold into A weight)
 *
 * int_id → slot mapping: int_id_to_slot_[int_id] gives 0..N-1 index used
 * to build per-token 'indices' tensor at forward time.
 *
 * Adding a 9th adapter when N slots occupied → returns false (caller
 * falls back to per-seq loop and logs a warning).
 *
 * Amend (Fix Y + Fix Z, 2026-08-27):
 *   Y: shared per-rank buf pool (Q/K/V fp32 accumulators) pre-alloc on
 *      first install; forward hot path calls buf_q()/k()/v() + slice +
 *      zero_() — mutex-free view op, no HBM alloc churn.
 *   Z: std::mutex → std::shared_mutex; build_indices_cpu takes shared
 *      lock so concurrent forward readers do not serialize. int_id map
 *      pre-reserved to prevent rehash under concurrent readers.
 *
 * Amend (Fix W, 2026-08-28):
 *   W: device-side slot lookup table (slot_lookup_dev_) populated at
 *      register time. Forward hot path replaces build_indices_cpu + .to()
 *      with slot_lookup_dev_.index_select(0, adapter_ids_per_token) —
 *      pure device op, no host→device sync memcpy. Required for
 *      compatibility with --enable_graph=true (CANN forbids sync memcpy
 *      inside graph capture; see 07-06 memory + 08-28 3-fallback bench
 *      root cause). Unregister does NOT invalidate the device slot;
 *      forward-path aid==0 host guard already blocks base-mixed batches
 *      from entering the device lookup path.
 */
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

class AscendCLoRAStorage {
 public:
  static constexpr int kNMaxActive = 8;
  static constexpr int kBufMaxR = 64;
  // Fix W: device slot lookup table upper bound. int_id assigned by
  // LoRARegistry starts at 1 and increments; production SaaS scenarios
  // typically stay <100 (see xllm-fastlibra memory). 1024 gives 10x safety
  // margin at 8 KB per rank cost — negligible.
  static constexpr int64_t kMaxIntId = 1024;

  static AscendCLoRAStorage& instance();

  // Register adapter's (layer, proj) A/B into a fixed slot. Returns
  // slot index (0..N-1) on success, -1 if all N slots taken.
  // idempotent for same int_id (returns previously-assigned slot).
  int register_adapter(uint64_t int_id,
                       int layer_idx,
                       const std::string& proj,
                       const torch::Tensor& A,
                       const torch::Tensor& B,
                       float scaling);

  // Remove int_id from all (layer, proj) storage; slot becomes reusable.
  void unregister_adapter(uint64_t int_id);

  struct StackedView {
    torch::Tensor A_stacked;  // [N, R, H_in]
    torch::Tensor B_stacked;  // [N, H_out, R]
    float scaling = 1.0f;
    bool valid = false;
    // Incremented on every register/unregister that touches this (layer,proj).
    // Callers may cache StackedView across forwards and revalidate cheaply by
    // comparing version. Not yet used (all ops synchronous today) but keeps a
    // hook for future async prefetch without breaking API.
    int64_t version = 0;
  };

  // Fetch pre-stacked slabs for (layer, proj). Returns valid=false if
  // no adapter registered for this key.
  StackedView get_stacked(int layer_idx, const std::string& proj) const;

  // Look up slot indices for a batch of adapter ids. Returns int32 tensor
  // [batch_tokens] on CPU (caller .npu() before kernel launch).
  // For int_id=0 (base) or int_id not in storage, returns -1 (kernel
  // should mask or caller pre-filters).
  // NOTE: This is the legacy host-side path. slow_path prefers Fix W
  // slot_lookup_dev() which is graph-capture safe.
  torch::Tensor build_indices_cpu(
      const std::vector<uint64_t>& int_ids_per_token) const;

  // slot count currently in use
  int active_count() const;

  // Fix Y: shared buf pool for slow_path Q/K/V fp32 accumulators. Grow-only,
  // pre-allocated on the first install. Forward hot path calls buf_q()/k()/v()
  // and does slice + zero_() — mutex-free view op on a device tensor.
  //
  // Buf shape: [buf_max_t_, kBufMaxR] fp32. Slot Q/K/V get independent tensors.
  // Slice narrowing to (total_tokens, R) at forward time is a zero-copy view.
  //
  // Thread safety: buf_{q,k,v}_shared_ is set only inside register_adapter
  // (under unique_lock). Once set, the tensor handle is const and safe to read
  // from concurrent forward readers without lock. slice/zero_() operate on the
  // device tensor via ATen, which is thread-safe for read/write on different
  // slices per-forward.
  const torch::Tensor& buf_q() const { return buf_q_shared_; }
  const torch::Tensor& buf_k() const { return buf_k_shared_; }
  const torch::Tensor& buf_v() const { return buf_v_shared_; }
  // Commit C: MoE column (gate/up) + row (down/o) independent bufs to avoid
  // race with QKV bufs under xllm layer_synchronizer multi-stream forward.
  const torch::Tensor& buf_gate() const { return buf_gate_shared_; }
  const torch::Tensor& buf_up() const { return buf_up_shared_; }
  const torch::Tensor& buf_down() const { return buf_down_shared_; }
  int64_t buf_max_t() const { return buf_max_t_; }

  // Fix W: device-side slot lookup table. Shape [kMaxIntId] int64, populated
  // at register time (install-path, graph-capture safe). Forward slow_path
  // does `slot_lookup_dev().index_select(0, adapter_ids_per_token)` — pure
  // device op, no host→device sync memcpy.
  //
  // Thread safety: allocated + written only inside register_adapter under
  // unique_lock. Forward readers only take the tensor handle (const ref);
  // index_select is a pure functional op producing a new tensor. Unregister
  // does NOT invalidate the device slot (see class comment for rationale).
  const torch::Tensor& slot_lookup_dev() const { return slot_lookup_dev_; }

 private:
  AscendCLoRAStorage() {
    // Fix Z: reserve to prevent rehash on register while concurrent readers
    // hold shared_lock. 8 slots * 4 = 32 buckets safety margin.
    int_id_to_slot_.reserve(kNMaxActive * 4);
  }

  // Fix Y helper: allocate/grow shared buf pool if the requested max_t
  // exceeds current capacity. Must be called under unique_lock (only from
  // register_adapter which already holds the write lock).
  void ensure_bufs_ready_locked(torch::Device device, int64_t max_t);

  // Fix W helper: allocate slot_lookup_dev_ on first call (init to -1),
  // then write slot at position int_id. Must be called under unique_lock.
  // int_id >= kMaxIntId is silently ignored — caller checks range or
  // storage stays consistent with legacy int_id_to_slot_ map (host).
  void update_slot_lookup_locked(uint64_t int_id,
                                 int32_t slot,
                                 torch::Device device);

  using KeyType = std::pair<int, std::string>;
  struct KeyHash {
    size_t operator()(const KeyType& k) const noexcept {
      return std::hash<int>()(k.first) ^
             (std::hash<std::string>()(k.second) << 1);
    }
  };

  // int_id -> slot idx
  std::unordered_map<uint64_t, int32_t> int_id_to_slot_;
  // (layer, proj) -> stacked storage
  std::unordered_map<KeyType, StackedView, KeyHash> storage_;

  // Fix Y: shared buf pool (Q/K/V fp32 accumulators), pre-alloc grow-only.
  torch::Tensor buf_q_shared_;
  torch::Tensor buf_k_shared_;
  torch::Tensor buf_v_shared_;
  // Commit C: MoE bufs
  torch::Tensor buf_gate_shared_;
  torch::Tensor buf_up_shared_;
  torch::Tensor buf_down_shared_;
  int64_t buf_max_t_ = 0;

  // Fix W: device-side slot lookup table [kMaxIntId] int64, init -1.
  torch::Tensor slot_lookup_dev_;

  // Fix Z: shared_mutex replaces std::mutex. Writers (register/unregister)
  // take unique_lock; readers (get_stacked/build_indices_cpu/active_count)
  // take shared_lock so concurrent forward paths do not serialize.
  mutable std::shared_mutex mu_;
};

}  // namespace xllm
