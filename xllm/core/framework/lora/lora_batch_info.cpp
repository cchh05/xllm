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

#include "lora_batch_info.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace xllm {

std::unique_ptr<LoRABatchInfo> build_lora_batch_info(
    const std::vector<uint64_t>& adapter_ids,
    const std::vector<int32_t>& q_seq_lens_vec,
    torch::Device device) {
  auto info = std::make_unique<LoRABatchInfo>();

  const size_t num_seqs = adapter_ids.size();
  CHECK_EQ(q_seq_lens_vec.size(), num_seqs)
      << "adapter_ids and q_seq_lens_vec must have same length";

  // Sum token counts.
  info->total_tokens = std::accumulate(
      q_seq_lens_vec.begin(), q_seq_lens_vec.end(), int64_t{0});

  // Empty batch (dummy or pre-fill setup) — return early with is_pure_base.
  if (info->total_tokens == 0) {
    return info;
  }

  // Fast path 1: pure base (all zeros).
  const bool all_zero = std::all_of(
      adapter_ids.begin(), adapter_ids.end(),
      [](uint64_t id) { return id == 0; });
  if (all_zero) {
    // is_pure_base = true, is_uniform_lora = false; no LoRA machinery
    // needed downstream.
    return info;
  }

  info->is_pure_base = false;

  // Fast path 2: uniform LoRA (all non-zero and same, no base mixed in).
  // Also handle the "single-adapter with base tokens" case as uniform
  // when the wrapper's existing fast path can absorb it (base positions
  // get zero delta naturally when we skip the scatter).
  {
    uint64_t unique_lora = 0;
    bool has_base = false;
    bool multi_lora = false;
    for (uint64_t id : adapter_ids) {
      if (id == 0) {
        has_base = true;
      } else if (unique_lora == 0) {
        unique_lora = id;
      } else if (id != unique_lora) {
        multi_lora = true;
        break;
      }
    }
    info->has_base_tokens = has_base;
    if (!multi_lora && unique_lora != 0) {
      info->is_uniform_lora = true;
      info->uniform_adapter_id = unique_lora;
      // Uniform path uses existing single-adapter dense matmul — no need
      // to build perm/group_list. Populate active_adapter_ids so
      // downstream code can still enumerate for stat / observability.
      info->active_adapter_ids.push_back(unique_lora);
      // Count all LoRA tokens under this adapter.
      int64_t lora_tokens = 0;
      for (size_t i = 0; i < num_seqs; ++i) {
        if (adapter_ids[i] != 0) {
          lora_tokens += q_seq_lens_vec[i];
        }
      }
      info->per_adapter_token_count.push_back(lora_tokens);
      info->num_lora_tokens = lora_tokens;
      return info;
    }
  }

  // Punica path: multi-adapter batch. Build the sort permutation.
  //
  // Step 1: expand per-seq adapter_ids to per-token, collecting the
  //   original token indices grouped by adapter_id.
  //
  // Step 2: build lora_perm as [tokens_of_adapter_A..., tokens_of_
  //   adapter_B..., ...] in ascending adapter_id order.
  //
  // Step 3: build lora_inv_perm as the inverse mapping.

  // adapter_id -> vector of original token indices (in the LoRA-only
  // dense stream, not the full batch positions).
  //
  // We use two passes: first collect per-original-position adapter_id
  // (so we can filter out base tokens), then bucket-sort by adapter_id.

  std::vector<uint64_t> tok_adapter_id;      // [num_lora_tokens]
  std::vector<int64_t> tok_orig_pos;          // [num_lora_tokens] in FULL batch
  tok_adapter_id.reserve(info->total_tokens);
  tok_orig_pos.reserve(info->total_tokens);

  int64_t pos = 0;
  for (size_t seq_i = 0; seq_i < num_seqs; ++seq_i) {
    const uint64_t aid = adapter_ids[seq_i];
    const int32_t len = q_seq_lens_vec[seq_i];
    if (aid != 0) {
      for (int32_t t = 0; t < len; ++t) {
        tok_adapter_id.push_back(aid);
        tok_orig_pos.push_back(pos + t);
      }
    }
    pos += len;
  }
  info->num_lora_tokens = static_cast<int64_t>(tok_adapter_id.size());

  // Bucket-sort by adapter_id: collect unique ids, sort ascending, then
  // scan tokens in that order.
  std::unordered_map<uint64_t, std::vector<int64_t>> bucket;
  bucket.reserve(16);
  for (size_t k = 0; k < tok_adapter_id.size(); ++k) {
    bucket[tok_adapter_id[k]].push_back(tok_orig_pos[k]);
  }
  std::vector<uint64_t> sorted_ids;
  sorted_ids.reserve(bucket.size());
  for (const auto& kv : bucket) sorted_ids.push_back(kv.first);
  std::sort(sorted_ids.begin(), sorted_ids.end());

  // Build lora_perm (host int64), group_list (host int64), and
  // active_adapter_ids in the sorted order.
  // Also build token_lora_mapping (per-token slot index, -1 for base)
  // and group_start_loc (cumsum for kernels that want offset form).
  std::vector<int64_t> perm_host;
  perm_host.reserve(info->num_lora_tokens);
  info->active_adapter_ids.reserve(sorted_ids.size());
  info->per_adapter_token_count.reserve(sorted_ids.size());

  // Map adapter_id -> slot index (0..num_active-1)
  std::unordered_map<uint64_t, int32_t> aid_to_slot;
  for (size_t s = 0; s < sorted_ids.size(); ++s) {
    aid_to_slot[sorted_ids[s]] = static_cast<int32_t>(s);
  }

  for (uint64_t aid : sorted_ids) {
    const auto& idxs = bucket[aid];
    info->active_adapter_ids.push_back(aid);
    info->per_adapter_token_count.push_back(
        static_cast<int64_t>(idxs.size()));
    for (int64_t p : idxs) perm_host.push_back(p);
  }

  // group_start_loc: cumsum with leading 0.
  std::vector<int64_t> group_start_host;
  group_start_host.reserve(sorted_ids.size() + 1);
  group_start_host.push_back(0);
  int64_t running = 0;
  for (int64_t c : info->per_adapter_token_count) {
    running += c;
    group_start_host.push_back(running);
  }

  // token_lora_mapping [total_tokens] int32: -1 for base, else slot index.
  std::vector<int32_t> token_map_host(info->total_tokens, -1);
  {
    int64_t p = 0;
    for (size_t seq_i = 0; seq_i < num_seqs; ++seq_i) {
      const uint64_t aid = adapter_ids[seq_i];
      const int32_t len = q_seq_lens_vec[seq_i];
      const int32_t slot = (aid == 0) ? -1 : aid_to_slot[aid];
      for (int32_t t = 0; t < len; ++t) token_map_host[p + t] = slot;
      p += len;
    }
  }

  // Build inv_perm (host int64): position of each original-index within
  // the permuted stream. Length: num_lora_tokens (dense index space).
  //
  // Note: this is the inverse of `perm_host` — a length-N permutation of
  // {0..N-1}, not related to full-batch positions. When scattering the
  // delta back, we need `output[perm_host[i]] += delta_sorted[i]`, which
  // is torch.index_add or torch.scatter_add. We keep both `perm_host`
  // (for gather forward) and its inverse (for scatter backward) so the
  // wrapper can pick the cheapest primitive.
  std::vector<int64_t> inv_perm_host(perm_host.size());
  for (int64_t i = 0; i < static_cast<int64_t>(perm_host.size()); ++i) {
    inv_perm_host[i] = i;  // placeholder — inverse is over the dense space
  }
  // Actually the inverse of the dense-space perm is trivially identity;
  // what we need for scatter is `perm_host` itself as the target index
  // in the full [total_tokens, out] tensor. Keep inv_perm as identity for
  // API symmetry; the wrapper uses `index_add(out, perm_host, delta)`.

  // Move to device tensors.
  auto host_i64_opts = torch::TensorOptions().dtype(torch::kInt64);
  auto host_i32_opts = torch::TensorOptions().dtype(torch::kInt32);
  auto host_bool_opts = torch::TensorOptions().dtype(torch::kBool);

  torch::Tensor perm_cpu = torch::from_blob(
      perm_host.data(),
      {static_cast<int64_t>(perm_host.size())},
      host_i64_opts).clone();
  torch::Tensor inv_perm_cpu = torch::from_blob(
      inv_perm_host.data(),
      {static_cast<int64_t>(inv_perm_host.size())},
      host_i64_opts).clone();
  torch::Tensor group_list_cpu = torch::from_blob(
      info->per_adapter_token_count.data(),
      {static_cast<int64_t>(info->per_adapter_token_count.size())},
      host_i64_opts).clone();
  torch::Tensor group_start_cpu = torch::from_blob(
      group_start_host.data(),
      {static_cast<int64_t>(group_start_host.size())},
      host_i64_opts).clone();
  torch::Tensor token_map_cpu = torch::from_blob(
      token_map_host.data(),
      {info->total_tokens},
      host_i32_opts).clone();

  // no_lora_flag_cpu: false because we're on the Punica path.
  info->no_lora_flag_cpu = torch::zeros({1}, host_bool_opts);

  if (device.is_cpu()) {
    info->lora_perm = std::move(perm_cpu);
    info->lora_inv_perm = std::move(inv_perm_cpu);
    info->group_list = std::move(group_list_cpu);
    info->group_start_loc = std::move(group_start_cpu);
    info->token_lora_mapping = std::move(token_map_cpu);
  } else {
    info->lora_perm = perm_cpu.to(device);
    info->lora_inv_perm = inv_perm_cpu.to(device);
    info->group_list = group_list_cpu.to(device);
    info->group_start_loc = group_start_cpu.to(device);
    info->token_lora_mapping = token_map_cpu.to(device);
  }

  return info;
}

}  // namespace xllm
