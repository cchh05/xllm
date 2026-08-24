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

#include "lora_block_pool.h"

#include <glog/logging.h>

#include <cstddef>

namespace xllm {

LoRABlockPool::LoRABlockPool(const Options& options)
    : num_total_blocks_(options.num_blocks),
      block_bytes_(options.block_bytes),
      device_(options.device) {
  CHECK_GT(num_total_blocks_, 0u) << "LoRABlockPool: num_blocks must be > 0";
  CHECK_GT(block_bytes_, 0u) << "LoRABlockPool: block_bytes must be > 0";
  CHECK_EQ(block_bytes_ % sizeof(uint16_t), 0u)
      << "LoRABlockPool: block_bytes must be multiple of 2 bytes";

  // Slab is a flat 1-D bfloat16 tensor sized (num_blocks * block_bytes / 2)
  // elements. We keep the storage element type at bfloat16 because it is
  // the shipping default for LoRA A/B on Qwen3.5 hybrid; tensor_view() is
  // free to reinterpret via .view() if callers want other dtypes of the
  // same width. Wider dtypes need a byte-based slab; not needed in Phase 0.
  const int64_t total_bytes = static_cast<int64_t>(num_total_blocks_) *
                              static_cast<int64_t>(block_bytes_);
  const int64_t num_elems =
      total_bytes / static_cast<int64_t>(sizeof(uint16_t));
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  slab_ = torch::empty({num_elems}, opts);

  free_list_.reserve(num_total_blocks_);
  for (int32_t i = static_cast<int32_t>(num_total_blocks_) - 1; i >= 0; --i) {
    free_list_.push_back(i);
  }
  in_use_.assign(num_total_blocks_, 0);

  LOG(INFO) << "[LoRABlockPool] init num_blocks=" << num_total_blocks_
            << " block_bytes=" << block_bytes_
            << " total_slab_bytes=" << total_bytes << " device=" << device_;
}

LoRABlockPool::~LoRABlockPool() {
  const uint32_t in_use = num_used_blocks();
  if (in_use > 0) {
    LOG(WARNING) << "[LoRABlockPool] shutdown with " << in_use
                 << " blocks still in use (leaked)";
  }
}

std::vector<int32_t> LoRABlockPool::allocate(uint32_t n) {
  std::vector<int32_t> out;
  if (n == 0) return out;
  std::lock_guard<std::mutex> g(mu_);
  if (free_list_.size() < n) {
    LOG(ERROR) << "[LoRABlockPool] OOM: requested=" << n
               << " available=" << free_list_.size();
    return out;
  }
  out.reserve(n);
  for (uint32_t k = 0; k < n; ++k) {
    const int32_t id = free_list_.back();
    free_list_.pop_back();
    in_use_[static_cast<size_t>(id)] = 1;
    out.push_back(id);
  }
  return out;
}

void LoRABlockPool::free(int32_t block_id) {
  if (block_id < 0 || block_id >= static_cast<int32_t>(num_total_blocks_)) {
    LOG(WARNING) << "[LoRABlockPool] free(): out-of-range block_id="
                 << block_id;
    return;
  }
  std::lock_guard<std::mutex> g(mu_);
  if (in_use_[static_cast<size_t>(block_id)] == 0) {
    LOG(WARNING) << "[LoRABlockPool] free(): double-free of block_id="
                 << block_id;
    return;
  }
  in_use_[static_cast<size_t>(block_id)] = 0;
  free_list_.push_back(block_id);
}

void LoRABlockPool::free(const std::vector<int32_t>& block_ids) {
  for (int32_t id : block_ids) {
    free(id);
  }
}

torch::Tensor LoRABlockPool::tensor_view(int32_t block_id,
                                         torch::IntArrayRef sizes,
                                         torch::ScalarType dtype) {
  CHECK_GE(block_id, 0) << "LoRABlockPool::tensor_view: block_id < 0";
  CHECK_LT(block_id, static_cast<int32_t>(num_total_blocks_))
      << "LoRABlockPool::tensor_view: block_id out of range";
  // Compute caller element count and required bytes.
  int64_t caller_elems = 1;
  for (int64_t d : sizes) {
    caller_elems *= d;
  }
  const int64_t elem_size_bytes = torch::elementSize(dtype);
  const int64_t caller_bytes = caller_elems * elem_size_bytes;
  CHECK_LE(caller_bytes, static_cast<int64_t>(block_bytes_))
      << "LoRABlockPool::tensor_view: caller shape " << sizes << " needs "
      << caller_bytes << " bytes, exceeds block_bytes " << block_bytes_;
  const int64_t start_elem_bf16 = static_cast<int64_t>(block_id) *
                                  static_cast<int64_t>(block_bytes_) /
                                  static_cast<int64_t>(sizeof(uint16_t));
  const int64_t caller_elems_bf16 =
      (caller_bytes + static_cast<int64_t>(sizeof(uint16_t)) - 1) /
      static_cast<int64_t>(sizeof(uint16_t));
  torch::Tensor slice_bf16 =
      slab_.narrow(0, start_elem_bf16, caller_elems_bf16);
  if (dtype == torch::kBFloat16) {
    return slice_bf16.view(sizes);
  }
  return slice_bf16.view(dtype).view(sizes);
}

uint32_t LoRABlockPool::num_free_blocks() const {
  std::lock_guard<std::mutex> g(mu_);
  return static_cast<uint32_t>(free_list_.size());
}

uint32_t LoRABlockPool::num_used_blocks() const {
  std::lock_guard<std::mutex> g(mu_);
  return num_total_blocks_ - static_cast<uint32_t>(free_list_.size());
}

}  // namespace xllm
