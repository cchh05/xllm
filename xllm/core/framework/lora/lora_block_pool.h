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

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace xllm {

// [FASTLIBRA Phase 0] LoRABlockPool
//
// Slab allocator for LoRA adapter A/B matrices, block-shaped for future
// unification with KV BlockManager (Phase 1 dependency tree). Each block is a
// contiguous device-side buffer of fixed byte size; adapters materialize
// their per-(layer, proj) A / B tensors by claiming N blocks each.
//
// Design intent (Phase 0):
// - Replace ad-hoc `torch::empty(device_opts)` in LoRARuntime install path
//   with block-based allocation from a pre-reserved slab.
// - Keep API surface analogous to xllm::BlockManager (`allocate` /
//   `free` / `num_free_blocks`) so Phase 1 can trivially compose this pool
//   under a shared dependency tree.
// - No cross-request sharing yet; each install claims blocks, unload frees.
// - Block size is a runtime option (chosen at pool init to fit the largest
//   expected per-tensor A/B under the given rank cap × hidden dim).
//
// Non-goals for Phase 0:
// - No prefix cache, no reference count sharing across adapters, no LRU.
//   Those are Phase 1+ concerns.
// - Not thread-safe against concurrent install from >1 executor thread
//   (LoRARuntime only has one executor thread per rank, so this is fine).
class LoRABlockPool {
 public:
  struct Options {
    // Total number of blocks reserved at pool init.
    uint32_t num_blocks = 0;
    // Size of each block in bytes. Must be >= any single A or B tensor
    // this pool is asked to store.
    uint64_t block_bytes = 0;
    // Device to allocate the slab on. Currently only NPU
    // (torch::kPrivateUse1) is supported.
    torch::Device device = torch::Device(torch::kCPU);
  };

  explicit LoRABlockPool(const Options& options);
  ~LoRABlockPool();

  LoRABlockPool(const LoRABlockPool&) = delete;
  LoRABlockPool& operator=(const LoRABlockPool&) = delete;

  // Allocate a contiguous run of `n` blocks. Returns the vector of block ids
  // on success, or empty vector on OOM. Ids are dense 0..num_blocks-1.
  // Blocks within the returned run are not necessarily contiguous in index
  // space; each block backs a `block_bytes` slab that can be viewed as a
  // torch::Tensor via `tensor_view`.
  std::vector<int32_t> allocate(uint32_t n);

  // Free a block by id. Free of an unowned id is a no-op with a warning.
  void free(int32_t block_id);

  // Free a run of block ids. Convenience over per-id free.
  void free(const std::vector<int32_t>& block_ids);

  // Return a torch::Tensor view over the given block, reshaped to `sizes`
  // and given `dtype`. The view aliases the underlying slab: modifying the
  // returned tensor mutates the slab. Caller must ensure `nbytes(sizes, dtype)
  // <= block_bytes`, else this returns an undefined tensor.
  torch::Tensor tensor_view(int32_t block_id,
                            torch::IntArrayRef sizes,
                            torch::ScalarType dtype);

  uint32_t num_free_blocks() const;
  uint32_t num_used_blocks() const;
  uint32_t num_total_blocks() const { return num_total_blocks_; }
  uint64_t block_bytes() const { return block_bytes_; }
  const torch::Device& device() const { return device_; }

 private:
  // Underlying slab as one big device tensor, sized (num_blocks *
  // block_bytes) bytes. Views handed out via `tensor_view` alias into this.
  torch::Tensor slab_;
  uint32_t num_total_blocks_ = 0;
  uint64_t block_bytes_ = 0;
  torch::Device device_;

  // Free-list guarded by a mutex. In-use bitmap kept separately so double-
  // free / free-of-unowned surfaces as a WARN log rather than corruption.
  mutable std::mutex mu_;
  std::vector<int32_t> free_list_;  // stack of currently-free block ids
  std::vector<uint8_t> in_use_;     // in_use_[i] == 1 iff block i is allocated
};

}  // namespace xllm
