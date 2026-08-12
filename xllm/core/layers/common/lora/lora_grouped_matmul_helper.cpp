/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_grouped_matmul_helper.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kernels/ops_api.h"
#include "kernels/param.h"

namespace xllm {
namespace layer {

namespace {

#ifdef XLLM_LORA_ENABLE_TEST_HOOKS
std::atomic<int64_t> g_grouped_matmul_call_count{0};
#endif

// Small helper: pad a [r, C] tensor to [r_max, C] by concatenating a zero
// block on dim 0. When r == r_max returns t unchanged so the common
// homogeneous-rank case is zero-copy.
torch::Tensor pad_rows_zero(const torch::Tensor& t, int64_t target_rows) {
  const int64_t rows = t.size(0);
  CHECK_LE(rows, target_rows) << "pad_rows_zero: target < current";
  if (rows == target_rows) {
    return t.contiguous();
  }
  auto shape = t.sizes().vec();
  shape[0] = target_rows - rows;
  auto pad = torch::zeros(shape, t.options());
  return torch::cat({t, pad}, /*dim=*/0).contiguous();
}

// Same idea but on dim 1: pad a [R, r] tensor to [R, r_max]. Used for B
// slices stored as [out_local, rank].
torch::Tensor pad_cols_zero(const torch::Tensor& t, int64_t target_cols) {
  const int64_t cols = t.size(1);
  CHECK_LE(cols, target_cols) << "pad_cols_zero: target < current";
  if (cols == target_cols) {
    return t.contiguous();
  }
  auto shape = t.sizes().vec();
  shape[1] = target_cols - cols;
  auto pad = torch::zeros(shape, t.options());
  return torch::cat({t, pad}, /*dim=*/1).contiguous();
}

}  // namespace

GroupedLoraSpec build_grouped_lora_spec(
    const std::vector<uint64_t>& adapter_ids,
    const std::vector<int32_t>& q_seq_lens,
    const torch::Tensor& input) {
  CHECK_EQ(adapter_ids.size(), q_seq_lens.size())
      << "adapter_ids and q_seq_lens must be seq-indexed together";

  // 1) Enumerate distinct non-zero adapters in first-seen order.
  std::vector<uint64_t> distinct_aids;
  std::unordered_map<uint64_t, int32_t> aid_to_group;
  distinct_aids.reserve(adapter_ids.size());
  for (uint64_t aid : adapter_ids) {
    if (aid == 0) continue;
    if (aid_to_group.find(aid) == aid_to_group.end()) {
      aid_to_group.emplace(aid, static_cast<int32_t>(distinct_aids.size()));
      distinct_aids.push_back(aid);
    }
  }
  const int64_t n_groups = static_cast<int64_t>(distinct_aids.size());

  // 2) Walk the seqs a second time. Bucket per-token original row index
  //    per group and record base-only tokens. Bucketing preserves natural
  //    intra-group order which keeps scatter_add deterministic.
  std::vector<std::vector<int64_t>> per_group_rows(n_groups);
  int64_t tok_off = 0;
  int64_t base_only_tokens = 0;
  for (size_t seq_idx = 0; seq_idx < adapter_ids.size(); ++seq_idx) {
    const int32_t seq_len = q_seq_lens[seq_idx];
    if (seq_len <= 0) continue;
    const uint64_t aid = adapter_ids[seq_idx];
    if (aid == 0) {
      base_only_tokens += seq_len;
      tok_off += seq_len;
      continue;
    }
    const int32_t g = aid_to_group.at(aid);
    auto& bucket = per_group_rows[g];
    bucket.reserve(bucket.size() + seq_len);
    for (int32_t k = 0; k < seq_len; ++k) {
      bucket.push_back(tok_off + k);
    }
    tok_off += seq_len;
  }

  // 3) Materialise group_list [N_groups] int64, original_row_indices
  //    [total_lora_tokens] int64, both on input.device(). Use torch::from_
  //    blob into a temp CPU buffer then .to(device) so we do not build the
  //    tensor row-by-row (device-side vector op).
  std::vector<int64_t> group_sizes;
  group_sizes.reserve(n_groups);
  int64_t total_lora_tokens = 0;
  for (const auto& bucket : per_group_rows) {
    group_sizes.push_back(static_cast<int64_t>(bucket.size()));
    total_lora_tokens += bucket.size();
  }
  std::vector<int64_t> flat_rows;
  flat_rows.reserve(total_lora_tokens);
  for (const auto& bucket : per_group_rows) {
    flat_rows.insert(flat_rows.end(), bucket.begin(), bucket.end());
  }

  const auto device = input.device();
  auto group_list_cpu =
      torch::from_blob(group_sizes.data(),
                       {static_cast<int64_t>(group_sizes.size())},
                       torch::TensorOptions().dtype(torch::kInt64));
  auto rows_cpu = torch::from_blob(flat_rows.data(),
                                   {static_cast<int64_t>(flat_rows.size())},
                                   torch::TensorOptions().dtype(torch::kInt64));

  GroupedLoraSpec spec;
  spec.distinct_aids = std::move(distinct_aids);
  spec.group_list_int64 = group_list_cpu.clone().to(device);
  spec.original_row_indices = rows_cpu.clone().to(device);
  spec.base_only_tokens = base_only_tokens;

  // 4) x_permuted: input.index_select(0, original_row_indices). Contiguous
  //    output guaranteed by index_select semantics.
  spec.x_permuted = input.index_select(0, spec.original_row_indices);

  return spec;
}

torch::Tensor apply_grouped_lora_delta(const GroupedLoraSpec& spec,
                                       const std::vector<torch::Tensor>& A_list,
                                       const std::vector<torch::Tensor>& B_list,
                                       const std::vector<float>& scaling_list,
                                       int64_t hidden_local,
                                       int64_t out_local,
                                       int64_t r_max) {
  const int64_t n_groups = static_cast<int64_t>(spec.distinct_aids.size());
  CHECK_EQ(static_cast<int64_t>(A_list.size()), n_groups);
  CHECK_EQ(static_cast<int64_t>(B_list.size()), n_groups);
  CHECK_EQ(static_cast<int64_t>(scaling_list.size()), n_groups);
  CHECK_GT(n_groups, 0);
  CHECK_GT(r_max, 0);

#ifdef XLLM_LORA_ENABLE_TEST_HOOKS
  g_grouped_matmul_call_count.fetch_add(1, std::memory_order_relaxed);
#endif

  const auto dtype = spec.x_permuted.scalar_type();
  const auto device = spec.x_permuted.device();

  // 1) Stack A into [N_groups, hidden_local, r_max]. Each A_list[i] is
  //    stored as [r_i, hidden_local]. group_gemm expects x @ weight, so
  //    weight rows must be the K axis and columns the N axis; we transpose
  //    A to [hidden_local, r_i] then pad on dim 1 to r_max.
  std::vector<torch::Tensor> A_padded;
  A_padded.reserve(n_groups);
  for (int64_t g = 0; g < n_groups; ++g) {
    CHECK_EQ(A_list[g].size(1), hidden_local)
        << "A[" << g << "] hidden mismatch";
    CHECK_LE(A_list[g].size(0), r_max);
    A_padded.emplace_back(
        pad_cols_zero(A_list[g].transpose(0, 1).contiguous(), r_max));
  }
  auto A_stacked = torch::stack(A_padded, /*dim=*/0);  // [G, hid, r_max]

  // 2) First group_gemm: tmp = x @ A_stacked -> [total_lora_tokens, r_max]
  //    Uses the simple (non-quant) fused_moe.cpp pattern:
  //      split_item=2, group_type=0, group_list_type=1.
  xllm::kernel::GroupGemmParams gg1;
  std::vector<torch::Tensor> x_list_1 = {spec.x_permuted};
  std::vector<torch::Tensor> w_list_1 = {A_stacked};
  gg1.x_list = torch::TensorList(x_list_1);
  gg1.weight_list = torch::TensorList(w_list_1);
  gg1.group_list = spec.group_list_int64;
  gg1.split_item = 2;
  gg1.group_type = 0;
  gg1.group_list_type = 1;
  auto tmp = xllm::kernel::group_gemm(gg1);  // [total_lora_tokens, r_max]

  // 3) Apply per-group scaling. Build a per-token scale [total_lora_tokens]
  //    on device from group_list_int64 + scaling_list, then broadcast-mul.
  std::vector<float> per_token_scale_cpu;
  per_token_scale_cpu.reserve(spec.x_permuted.size(0));
  int64_t running = 0;
  auto group_sizes_cpu = spec.group_list_int64.to(torch::kCPU).contiguous();
  const int64_t* gs_ptr = group_sizes_cpu.data_ptr<int64_t>();
  for (int64_t g = 0; g < n_groups; ++g) {
    const int64_t sz = gs_ptr[g];
    for (int64_t k = 0; k < sz; ++k) {
      per_token_scale_cpu.push_back(scaling_list[g]);
    }
    running += sz;
  }
  CHECK_EQ(running, spec.x_permuted.size(0));
  auto per_token_scale =
      torch::from_blob(per_token_scale_cpu.data(),
                       {static_cast<int64_t>(per_token_scale_cpu.size())},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .clone()
          .to(device);
  tmp = tmp.to(torch::kFloat32) * per_token_scale.unsqueeze(1);
  tmp = tmp.to(dtype);

  // 4) Stack B into [N_groups, r_max, out_local]. B_list[i] is stored as
  //    [out_local, r_i]. For group_gemm the (K,N) axes of weight are
  //    (r_max, out_local), so transpose to [r_i, out_local] then pad on
  //    dim 0 to r_max. Padding here matches the pad we did on A: the
  //    unused rows of the padded A produce zero contributions in tmp, so
  //    padded rows of B will get multiplied by zero regardless.
  std::vector<torch::Tensor> B_padded;
  B_padded.reserve(n_groups);
  for (int64_t g = 0; g < n_groups; ++g) {
    CHECK_EQ(B_list[g].size(0), out_local)
        << "B[" << g << "] out_local mismatch";
    CHECK_LE(B_list[g].size(1), r_max);
    B_padded.emplace_back(
        pad_rows_zero(B_list[g].transpose(0, 1).contiguous(), r_max));
  }
  auto B_stacked = torch::stack(B_padded, /*dim=*/0);  // [G, r_max, out]

  // 5) Second group_gemm: delta = tmp @ B_stacked -> [total_lora_tokens,
  //    out_local].
  xllm::kernel::GroupGemmParams gg2;
  std::vector<torch::Tensor> x_list_2 = {tmp};
  std::vector<torch::Tensor> w_list_2 = {B_stacked};
  gg2.x_list = torch::TensorList(x_list_2);
  gg2.weight_list = torch::TensorList(w_list_2);
  gg2.group_list = spec.group_list_int64;
  gg2.split_item = 2;
  gg2.group_type = 0;
  gg2.group_list_type = 1;
  auto delta = xllm::kernel::group_gemm(gg2);
  return delta;
}

void scatter_add_lora_delta_(torch::Tensor& y,
                             const torch::Tensor& delta_permuted,
                             const GroupedLoraSpec& spec) {
  CHECK_EQ(delta_permuted.size(0), spec.original_row_indices.size(0));
  y.index_add_(0, spec.original_row_indices, delta_permuted.to(y.dtype()));
}

#ifdef XLLM_LORA_ENABLE_TEST_HOOKS
int64_t grouped_matmul_call_count_for_test() {
  return g_grouped_matmul_call_count.load(std::memory_order_relaxed);
}

void reset_grouped_matmul_call_count_for_test() {
  g_grouped_matmul_call_count.store(0, std::memory_order_relaxed);
}
#endif

}  // namespace layer
}  // namespace xllm
