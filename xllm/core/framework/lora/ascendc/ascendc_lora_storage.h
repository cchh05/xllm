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
 */
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

class AscendCLoRAStorage {
 public:
  static constexpr int kNMaxActive = 8;

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
  torch::Tensor build_indices_cpu(
      const std::vector<uint64_t>& int_ids_per_token) const;

  // slot count currently in use
  int active_count() const;

 private:
  AscendCLoRAStorage() = default;

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

  mutable std::mutex mu_;
};

}  // namespace xllm
