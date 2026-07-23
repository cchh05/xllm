/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_row_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/lora/lora_config.h"
#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {

LoRARowParallelLinearImpl::LoRARowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const std::string& proj_name)
    : proj_name_(proj_name),
      in_features_(in_features),
      out_features_(out_features) {
  // NOT register_module'd on this wrapper: keeps checkpoint keys unchanged.
  // (Same rationale as LoRAQKVParallelLinearImpl.)
  base_ = RowParallelLinear(in_features,
                            out_features,
                            bias,
                            input_is_parallelized,
                            enable_result_reduction,
                            quant_args,
                            process_group,
                            options);
  auto* pg = base_->process_group();
  tp_world_size_ = (pg != nullptr) ? pg->world_size() : 1;
  tp_rank_ = (pg != nullptr) ? pg->rank() : 0;
  in_features_local_ =
      (tp_world_size_ > 1) ? (in_features_ / tp_world_size_) : in_features_;
}

torch::Tensor LoRARowParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  // Under TP>1, applying the LoRA delta correctly requires an extra rank-dim
  // all-reduce (see below). That collective has non-trivial HCCL launch cost
  // on NPU; deployments that only serve adapters whose target_modules are
  // column-parallel (q/k/v_proj, gate/up_proj) can opt out to reclaim the
  // throughput. The flag is a hard gate: when false the wrapper degrades to
  // pre-fix behaviour and o_proj / down_proj deltas silently no-op under TP>1.
  if (tp_world_size_ > 1 && !FLAGS_enable_lora_row_parallel_all_reduce) {
    return y;
  }

  // Row-parallel LoRA delta.
  //
  // Base RowParallelLinear (input_is_parallelized=true) receives x_local
  // with shape [T, in_features_local] (already sharded along in-dim) and
  // all-reduces the base output y across TP ranks to full [T, out].
  //
  // For the LoRA delta we mirror SGLang's RowParallelLinearWithLoRA:
  //   * A is stored full-width [r, in_features]; each rank slices its own
  //     shard A_local = A[:, tp_rank * in_local : (tp_rank+1) * in_local]
  //   * B is replicated (kept full [out, r]) — cheap because r is small
  //   * per rank: tmp_local = x_local @ A_local^T   -> [T, r]  (partial)
  //   * all-reduce tmp_local over TP  -> [T, r]  (full sum, reconstructs A@x)
  //   * delta = tmp_reduced @ B^T * scaling         -> [T, out]  (replicated)
  //   * y already all-reduced by base, so y + delta is correct.
  //
  // Cost win: we all-reduce a rank-dim tensor [T, r=16], not a hidden-dim
  // tensor [T, out]. For Qwen3-30B r=16 vs out=2048 that is 128x smaller.
  const auto* ctx = current_lora_context();
  if (ctx == nullptr || ctx->adapter_ids == nullptr ||
      ctx->q_seq_lens_vec == nullptr || ctx->layer_index < 0) {
    return y;
  }
  const auto& adapter_ids = *ctx->adapter_ids;
  const auto& q_seq_lens = *ctx->q_seq_lens_vec;
  if (adapter_ids.empty() || adapter_ids.size() != q_seq_lens.size()) {
    return y;
  }

  bool any_nonzero = false;
  for (auto id : adapter_ids) {
    if (id != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) return y;

  auto* pg = base_->process_group();

  // Fast path: batch uses a single adapter and no base-only seq. Shrink+
  // expand on the full batch, one all-reduce for the whole batch instead
  // of one per seq.
  {
    uint64_t sole_aid = 0;
    bool single_adapter = true;
    for (auto id : adapter_ids) {
      if (id == 0) {
        single_adapter = false;
        break;
      }
      if (sole_aid == 0) {
        sole_aid = id;
      } else if (id != sole_aid) {
        single_adapter = false;
        break;
      }
    }
    if (single_adapter && sole_aid != 0) {
      auto& runtime = LoRARuntime::instance();
      const auto* pd =
          runtime.get_per_proj_delta(sole_aid, ctx->layer_index, proj_name_);
      if (pd == nullptr) return y;
      // A: [r, in_full]; slice to local in-shard for TP>1.
      torch::Tensor A_local = pd->A;
      if (tp_world_size_ > 1 && pd->A.size(1) > in_features_local_) {
        const int64_t start = tp_rank_ * in_features_local_;
        A_local = pd->A.slice(1, start, start + in_features_local_);
      }
      // tmp_local: [T, in_local] @ [in_local, r] -> [T, r]  (partial)
      auto tmp = torch::matmul(input, A_local.transpose(0, 1));
      if (tp_world_size_ > 1 && pg != nullptr) {
        tmp = xllm::parallel_state::reduce(tmp, pg);
      }
      // delta: [T, r] @ [r, out] -> [T, out]  (replicated; matches y)
      auto delta = torch::matmul(tmp, pd->B.transpose(0, 1));
      delta = (delta * pd->scaling).to(y.dtype());
      y.add_(delta);
      return y;
    }
  }

  // Slow path: per-seq, one all-reduce per adapter-bearing seq. Correct but
  // costly under interleaved base+adapter batching; adapter-affinity
  // batching at the gateway is the recommended mitigation.
  auto& runtime = LoRARuntime::instance();
  int64_t tok_off = 0;
  for (size_t seq_idx = 0; seq_idx < adapter_ids.size(); ++seq_idx) {
    const int32_t seq_len = q_seq_lens[seq_idx];
    if (seq_len <= 0) continue;
    const uint64_t aid = adapter_ids[seq_idx];
    if (aid == 0) {
      tok_off += seq_len;
      continue;
    }

    const auto* pd =
        runtime.get_per_proj_delta(aid, ctx->layer_index, proj_name_);
    if (pd == nullptr) {
      tok_off += seq_len;
      continue;
    }

    auto x_seq = input.slice(0, tok_off, tok_off + seq_len);
    torch::Tensor A_local = pd->A;
    if (tp_world_size_ > 1 && pd->A.size(1) > in_features_local_) {
      const int64_t start = tp_rank_ * in_features_local_;
      A_local = pd->A.slice(1, start, start + in_features_local_);
    }
    auto tmp = torch::matmul(x_seq, A_local.transpose(0, 1));
    if (tp_world_size_ > 1 && pg != nullptr) {
      tmp = xllm::parallel_state::reduce(tmp, pg);
    }
    auto delta = torch::matmul(tmp, pd->B.transpose(0, 1));
    delta = (delta * pd->scaling).to(y.dtype());

    y.slice(0, tok_off, tok_off + seq_len).add_(delta);
    tok_off += seq_len;
  }
  return y;
}

void LoRARowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

}  // namespace layer
}  // namespace xllm
