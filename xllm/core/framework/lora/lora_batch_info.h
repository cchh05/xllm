/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// LoRABatchInfo: pre-computed metadata for Punica-style batched
// shrink/expand kernels. Built once per forward at batch-build time and
// consumed by every LoRA*ParallelLinear wrapper in that forward.
//
// The metadata answers three questions the wrapper needs to fuse all
// adapters into two grouped-matmul launches:
//
// 1. Which tokens use which adapter?
//    - active_adapter_ids: [num_active_adapters] host vector, sorted
//      ascending by adapter_id; adapter_id == 0 (base) never appears.
//    - token_count: [num_active_adapters + has_base] int64 tensor on
//      device. Order matches the sorted permutation output: base tokens
//      (if any) come last as a single group, so grouped_matmul can skip
//      them via group_list padding of 0 for empty groups.
//    - Note: base tokens are NOT permuted into the LoRA groups. They stay
//      in their original position; only LoRA tokens are gathered.
//
// 2. What is the sort order?
//    - lora_perm: [num_lora_tokens] int64 tensor on device. Applied to
//      the input hidden state to produce a densely-packed [num_lora_tokens
//      , hidden] view where tokens for adapter 1 come first, then adapter
//      2, etc.
//    - lora_inv_perm: [num_lora_tokens] int64 tensor on device. Applied
//      to the delta output to scatter it back to the original token order
//      before adding to base output.
//    - lora_token_mask: [total_tokens] bool tensor on device. True where
//      the token uses a non-zero adapter. Used to gather/scatter.
//
// 3. Fast-path signals?
//    - is_pure_base: true iff every token in the batch has adapter_id==0.
//      Wrapper short-circuits to `return base_output` and skips all
//      Punica machinery.
//    - is_uniform_lora: true iff every non-base token uses the same
//      adapter_id (`uniform_adapter_id`). Wrapper takes the existing
//      single-adapter fast path (one dense matmul, no permutation).
//    - When !is_pure_base && !is_uniform_lora, wrapper enters Punica
//      batched path.
//
// Construction: `build_lora_batch_info(adapter_ids, q_seq_lens_vec,
// device)` — see lora_batch_info.cpp. Called by
// ModelInputParams::to(device) or the model's forward entry.
//
// Cost: O(total_tokens) host-side sort + one small H2D copy of int64
// tensors (10-100 KB total). Amortized across ~64 layers x 4 projs =
// ~256 grouped_matmul calls per forward, so <10 us per call.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm {

struct LoRABatchInfo {
  // === Host-side metadata (small, cheap to construct) ===

  // Sorted ascending list of adapter_ids present in this batch that are
  // non-zero. Order matches per_adapter_token_count.
  // Length == num_active_adapters. Empty when is_pure_base.
  std::vector<uint64_t> active_adapter_ids;

  // Per-active-adapter total token count (used to build group_list).
  // Length == num_active_adapters. Sum == num_lora_tokens.
  std::vector<int64_t> per_adapter_token_count;

  // Total number of tokens in the batch (LoRA + base).
  int64_t total_tokens = 0;

  // Number of tokens whose adapter_id != 0.
  int64_t num_lora_tokens = 0;

  // === Fast-path signals ===

  // True when every token has adapter_id == 0. LoRA wrapper skips
  // entirely and returns base output.
  bool is_pure_base = true;

  // True when every LoRA token uses the same adapter (and there are no
  // base tokens interleaved, OR base tokens are also allowed but then
  // uniform_adapter_id refers only to the non-base subset). Wrapper uses
  // the existing dense-matmul fast path in this case.
  bool is_uniform_lora = false;

  // Only meaningful when is_uniform_lora == true.
  uint64_t uniform_adapter_id = 0;

  // === Device-side tensors (built lazily on to(device)) ===

  // Per-token adapter mapping. -1 for base tokens (no LoRA), otherwise
  // dense slot index into active_adapter_ids (0..num_active_adapters-1).
  // Shape: [total_tokens], dtype: int32, on device.
  // Mirrors vLLM Punica `token_lora_mapping` semantics.
  // Undefined when is_pure_base.
  torch::Tensor token_lora_mapping;

  // Permutation to gather LoRA tokens (adapter_id != 0) from the input
  // hidden state into a densely-packed layout sorted by adapter_id.
  // Shape: [num_lora_tokens], dtype: int64, on the model device.
  // Undefined when is_pure_base.
  torch::Tensor lora_perm;

  // Inverse permutation to scatter the LoRA delta back to the original
  // token positions.
  // Shape: [num_lora_tokens], dtype: int64, on the model device.
  // Undefined when is_pure_base.
  torch::Tensor lora_inv_perm;

  // group_list for npu_grouped_matmul: per-adapter token counts in the
  // sorted order.
  // Shape: [num_active_adapters], dtype: int64, on the model device.
  // Undefined when is_pure_base.
  torch::Tensor group_list;

  // Cumulative sum of group_list, useful for kernels that need per-adapter
  // start offset (like vLLM's `lora_token_start_loc`).
  // Shape: [num_active_adapters + 1], dtype: int64, first element = 0.
  // Undefined when is_pure_base.
  torch::Tensor group_start_loc;

  // === CPU-side flag for kernel launch early-exit ===

  // Set on CPU (not device). Wrapper reads this via .item() before any
  // NPU kernel launch to skip when the entire batch is pure base — even
  // cheaper than a device-side branch.
  // vLLM keeps this off-device so torch.compile doesn't specialize.
  torch::Tensor no_lora_flag_cpu;

  // === Base token handling ===

  // Whether the batch mixes base (adapter_id==0) and LoRA tokens.
  // When true, wrapper adds delta only to positions that had a LoRA
  // adapter — base positions get zero delta.
  bool has_base_tokens = false;
};

// Build LoRABatchInfo from per-sequence adapter_ids and q_seq_lens_vec.
//
// adapter_ids: length N, each entry is the adapter_id for sequence i
//   (0 == base, non-zero == LoRA).
// q_seq_lens_vec: length N, each entry is the token count of sequence i.
//   Sum determines total_tokens.
// device: target device for device-side tensors. When kCPU, tensors are
//   left undefined (caller may promote later via to(device)).
//
// Returns a heap-allocated LoRABatchInfo. Ownership passes to the caller
// (typically stored in ModelInputParams or LoRAContextFrame).
std::unique_ptr<LoRABatchInfo> build_lora_batch_info(
    const std::vector<uint64_t>& adapter_ids,
    const std::vector<int32_t>& q_seq_lens_vec,
    torch::Device device);

}  // namespace xllm
