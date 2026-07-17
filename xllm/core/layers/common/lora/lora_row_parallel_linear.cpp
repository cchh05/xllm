/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_row_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"

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
}

torch::Tensor LoRARowParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  // TP>1 row-parallel LoRA temporarily skipped: input is already sharded
  // along in-dim, so applying pd->A (full in_features) would broadcast-error.
  // A correct fix requires either (a) slicing A on in-dim + all-reducing the
  // partial delta, or (b) storing TP-sharded A per rank. Both are non-trivial
  // and deferred to P1. For now, TP>1 row-parallel returns base output only.
  // Attention o_proj and MLP down_proj miss their LoRA delta contribution
  // in TP>1 mode. QKV (col-parallel input, replicated in-dim) and gate/up
  // (col-parallel out-dim) still apply their deltas correctly.
  if (base_->process_group() != nullptr &&
      base_->process_group()->world_size() > 1) {
    return y;
  }

  // M10 per-request per-proj real LoRA. Row-parallel wrapper is a single
  // proj (o_proj or down_proj) so no output-concat like QKV.
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
    auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
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
