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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "framework/lora/lora_runtime.h"

namespace xllm {
namespace layer {

// -----------------------------------------------------------------------------
// LoRA grouped-matmul helper.
//
// The stock slow-path forward in the LoRA Linear wrappers
// (lora_qkv/column/row_parallel_linear.cpp) walks the batch sequence-by-
// sequence, issues two small torch::matmul calls per (seq, proj), and pastes
// the delta back with y.slice(...).add_(...). On NPU the launch + slice
// memcpy cost dominates the actual FLOPs -- roughly the same failure mode as
// the SGMV NPU spike (project_xllm_multilora_punica_infeasible_2026_07_28
// memory) and matches the ~31% (30B) / ~12.9% (122B) slow-path overhead
// baseline.
//
// This helper folds the per-sequence loop into a pair of npu_grouped_matmul
// calls plus a single index_add scatter, matching how fused_moe.cpp
// dispatches expert weights. The wrappers own the fast-path branch
// (distinct_count <= 1) and only call in here when the batch mixes >= 2
// distinct non-zero adapters.
//
// TP shard slicing (A on in-dim for row-parallel; B on out-dim for column-
// parallel + qkv) still happens inside the wrapper before A_list / B_list
// are passed in, so this helper is TP-shape-agnostic.
// -----------------------------------------------------------------------------
struct GroupedLoraSpec {
  // Distinct non-zero adapter_ids in the batch, in the same order they are
  // packed as groups for npu_grouped_matmul (group_idx = index into this
  // vector).
  std::vector<uint64_t> distinct_aids;

  // Token count per group, dtype int64, shape [N_groups], on the same device
  // as x. Matches group_list_type=1 (per-group sizes) which is what
  // fused_moe.cpp uses in production. Sum equals x_permuted.size(0).
  torch::Tensor group_list_int64;

  // Input reordered so tokens routing to the same adapter are contiguous.
  // Shape [total_lora_tokens, hidden_local], contiguous, same device as
  // input. Excludes base-only tokens (adapter_id == 0).
  torch::Tensor x_permuted;

  // For each of the total_lora_tokens rows above, the position it originally
  // occupied in the un-permuted [T] batch. Used by scatter_add_lora_delta_
  // to write results back. Dtype int64, shape [total_lora_tokens].
  torch::Tensor original_row_indices;

  // How many tokens in the original batch had adapter_id == 0 (base-only).
  // Not consumed by the helper itself; wrappers use it for observability.
  int64_t base_only_tokens = 0;
};

// Build a GroupedLoraSpec from the current batch. Guaranteed by the caller
// to observe at least 2 distinct non-zero adapters -- the wrappers short-
// circuit the fast path (distinct_count <= 1) before calling in.
//
// Callers pass q_seq_lens directly rather than through the LoRAContextFrame
// so this stays a pure function and is easier to unit-test.
GroupedLoraSpec build_grouped_lora_spec(
    const std::vector<uint64_t>& adapter_ids,
    const std::vector<int32_t>& q_seq_lens,
    const torch::Tensor& input);

// One-proj batched delta compute. Inputs are the per-adapter A / B slices
// (already sharded for TP by the caller), one entry per group in the same
// order as spec.distinct_aids. r_max is the dynamic pad target -- the
// helper pads shorter-rank A / B with zeros so npu_grouped_matmul sees
// homogeneous shapes ([N_groups, hidden_local, r_max] and [N_groups, r_max,
// out_local] stacked). For the common case r_i == r_max the pad is a no-op.
//
// Returns [total_lora_tokens, out_local] in the same permutation as
// spec.x_permuted, dtype matches the input tensor.
//
// scaling_list is applied as a per-group fp32 scalar after the second
// grouped matmul.
torch::Tensor apply_grouped_lora_delta(const GroupedLoraSpec& spec,
                                       const std::vector<torch::Tensor>& A_list,
                                       const std::vector<torch::Tensor>& B_list,
                                       const std::vector<float>& scaling_list,
                                       int64_t hidden_local,
                                       int64_t out_local,
                                       int64_t r_max);

// Scatter the batched delta back to the original per-token positions in y.
// Uses torch::Tensor::index_add_ so the write is a single kernel launch
// instead of N per-seq y.slice().add_() calls.
void scatter_add_lora_delta_(torch::Tensor& y,
                             const torch::Tensor& delta_permuted,
                             const GroupedLoraSpec& spec);

// -----------------------------------------------------------------------------
// Test hooks. Compiled in when XLLM_LORA_ENABLE_TEST_HOOKS is defined by the
// test target. Production builds do not see these.
// -----------------------------------------------------------------------------
#ifdef XLLM_LORA_ENABLE_TEST_HOOKS
int64_t grouped_matmul_call_count_for_test();
void reset_grouped_matmul_call_count_for_test();
#endif

}  // namespace layer
}  // namespace xllm
