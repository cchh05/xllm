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
#include "layers/common/lora/lora_grouped_matmul_helper.h"

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
    const std::string& proj_name,
    const LinearExtraArgs& linear_extra_args)
    : proj_name_(proj_name),
      in_features_(in_features),
      out_features_(out_features) {
  // Base's own AR is disabled when the fused-AR path is on so the wrapper
  // can fold the LoRA delta into the base's partial output and reduce both
  // together in a single collective.
  fused_ar_ = FLAGS_enable_lora_row_parallel_fused_ar;
  const bool base_reduce = fused_ar_ ? false : enable_result_reduction;
  // NOT register_module'd on this wrapper: keeps checkpoint keys unchanged.
  // (Same rationale as LoRAQKVParallelLinearImpl.)
  base_ = RowParallelLinear(in_features,
                            out_features,
                            bias,
                            input_is_parallelized,
                            base_reduce,
                            quant_args,
                            process_group,
                            options,
                            linear_extra_args);
  auto* pg = base_->process_group();
  tp_world_size_ = (pg != nullptr) ? pg->world_size() : 1;
  tp_rank_ = (pg != nullptr) ? pg->rank() : 0;
  in_features_local_ =
      (tp_world_size_ > 1) ? (in_features_ / tp_world_size_) : in_features_;
  // Cache the intent to all-reduce the base output ourselves in fused mode.
  // Under TP=1 there is no reduction to do at all.
  wrapper_owns_reduction_ = fused_ar_ && tp_world_size_ > 1;
}

torch::Tensor LoRARowParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  auto* pg = base_->process_group();

  // Fast-out: TP>1 with row-parallel AR disabled and NOT in fused mode.
  // The wrapper skips the delta entirely so o_proj / down_proj LoRA silently
  // no-ops, matching pre-a9d6ad74 behaviour.
  if (tp_world_size_ > 1 && !fused_ar_ &&
      !FLAGS_enable_lora_row_parallel_all_reduce) {
    return y;
  }

  // In fused-AR mode the base was constructed with enable_result_reduction=
  // false, so `y` here is a per-rank partial-sum on the out-dim. The wrapper
  // will add the LoRA delta (also a per-rank partial on out-dim) and
  // all-reduce the sum in one collective. Under TP=1 there is no partial
  // and nothing to reduce; the code below still works because
  // wrapper_owns_reduction_ = false in that case.

  // Row-parallel LoRA delta.
  //
  // Fused-AR path (default when enable_lora_row_parallel_fused_ar=true):
  //   * A is sliced on in-dim per rank      -> A_local [r, in_local]
  //   * B is replicated at full [out, r]
  //   * tmp_local = x_local @ A_local^T                    [T, r] partial-A
  //   * local_delta = tmp_local @ B^T * scaling            [T, out] partial on
  //   out-dim
  //     (mathematically B @ A_local @ x_local; the sum-over-ranks is deferred)
  //   * combined = y (base partial) + local_delta
  //   * output = all_reduce(combined) -> full base + full delta in ONE
  //   collective
  //
  // Legacy path (fallback, enable_lora_row_parallel_all_reduce=true,
  // fused_ar_=false):
  //   * A slice as above, tmp_local same
  //   * all-reduce on rank-dim tmp [T, r=16]
  //   * delta = tmp_full @ B^T * scaling  (replicated)
  //   * y is already reduced by base, y += delta
  //
  // Cost win of fused vs legacy: 1 AR per proj vs 2. On NPU HCCL where
  // launch latency dominates, this halves the collective count and reclaims
  // most of the ~17% overhead the legacy path incurs.
  const auto* ctx = current_lora_context();
  if (ctx == nullptr || ctx->adapter_ids == nullptr ||
      ctx->q_seq_lens_vec == nullptr || ctx->layer_index < 0) {
    // No LoRA context: nothing to add. Still need to reduce y if fused_ar.
    if (wrapper_owns_reduction_ && pg != nullptr) {
      y = xllm::parallel_state::reduce(y, pg);
    }
    return y;
  }
  const auto& adapter_ids = *ctx->adapter_ids;
  const auto& q_seq_lens = *ctx->q_seq_lens_vec;
  if (adapter_ids.empty() || adapter_ids.size() != q_seq_lens.size()) {
    if (wrapper_owns_reduction_ && pg != nullptr) {
      y = xllm::parallel_state::reduce(y, pg);
    }
    return y;
  }

  bool any_nonzero = false;
  for (auto id : adapter_ids) {
    if (id != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) {
    // Pure-base batch: still need to reduce y in fused mode.
    if (wrapper_owns_reduction_ && pg != nullptr) {
      y = xllm::parallel_state::reduce(y, pg);
    }
    return y;
  }

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
      if (pd == nullptr) {
        if (wrapper_owns_reduction_ && pg != nullptr) {
          y = xllm::parallel_state::reduce(y, pg);
        }
        return y;
      }
      // A: [r, in_full]; slice to local in-shard for TP>1.
      torch::Tensor A_local = pd->A;
      if (tp_world_size_ > 1 && pd->A.size(1) > in_features_local_) {
        const int64_t start = tp_rank_ * in_features_local_;
        A_local = pd->A.slice(1, start, start + in_features_local_);
      }
      // tmp_local: [T, in_local] @ [in_local, r] -> [T, r]  (partial)
      auto tmp = torch::matmul(input, A_local.transpose(0, 1));
      torch::Tensor delta;
      if (wrapper_owns_reduction_) {
        // Fused mode: keep tmp partial. Compute per-rank partial delta on
        // out-dim and add to y (also partial on out-dim), then reduce once.
        delta = torch::matmul(tmp, pd->B.transpose(0, 1));
      } else {
        // Legacy: reduce rank-dim tmp first, then expand to out-dim.
        if (tp_world_size_ > 1 && pg != nullptr) {
          tmp = xllm::parallel_state::reduce(tmp, pg);
        }
        delta = torch::matmul(tmp, pd->B.transpose(0, 1));
      }
      delta = (delta * pd->scaling).to(y.dtype());
      y.add_(delta);
      if (wrapper_owns_reduction_ && pg != nullptr) {
        y = xllm::parallel_state::reduce(y, pg);
      }
      return y;
    }
  }

  // Grouped-matmul slow path (opt-in via --enable_lora_grouped_matmul).
  //
  // Only enabled when wrapper_owns_reduction_ is true (fused-AR mode). In
  // legacy mode a rank-dim reduce sits *between* the two matmuls, so the
  // grouped helper would need to expose an internal hook -- not worth the
  // extra API surface for a code path we do not push to production.
  if (FLAGS_enable_lora_grouped_matmul && wrapper_owns_reduction_) {
    auto& runtime_gm = LoRARuntime::instance();
    auto spec = build_grouped_lora_spec(adapter_ids, q_seq_lens, input);
    const int64_t n_groups = static_cast<int64_t>(spec.distinct_aids.size());
    std::vector<torch::Tensor> A_list, B_list;
    std::vector<float> scale_list;
    A_list.reserve(n_groups);
    B_list.reserve(n_groups);
    scale_list.reserve(n_groups);
    int64_t r_max = 0;
    // Row-parallel: A is sliced on dim=1 (in-features) per rank; B is
    // replicated at full [out_features_, r]. delta stays partial on out-
    // dim; the final wrapper_owns_reduction_ reduce below handles the
    // combined base+delta all-reduce.
    auto slice_A_shard = [&](const torch::Tensor& A) -> torch::Tensor {
      if (tp_world_size_ > 1 && A.size(1) > in_features_local_) {
        const int64_t start = tp_rank_ * in_features_local_;
        return A.slice(1, start, start + in_features_local_).contiguous();
      }
      return A.contiguous();
    };
    auto zero_A = torch::zeros({1, in_features_local_}, input.options());
    auto zero_B = torch::zeros({out_features_, 1}, input.options());
    for (uint64_t aid : spec.distinct_aids) {
      const auto* pd =
          runtime_gm.get_per_proj_delta(aid, ctx->layer_index, proj_name_);
      if (pd != nullptr) {
        A_list.emplace_back(slice_A_shard(pd->A));
        B_list.emplace_back(pd->B.contiguous());
        scale_list.emplace_back(pd->scaling);
        r_max = std::max(r_max, static_cast<int64_t>(pd->r));
      } else {
        A_list.emplace_back(zero_A);
        B_list.emplace_back(zero_B);
        scale_list.emplace_back(0.0f);
      }
    }
    if (r_max == 0) {
      // No real delta. Still need the deferred base reduce.
      if (wrapper_owns_reduction_ && pg != nullptr) {
        y = xllm::parallel_state::reduce(y, pg);
      }
      return y;
    }
    auto delta = apply_grouped_lora_delta(spec,
                                          A_list,
                                          B_list,
                                          scale_list,
                                          in_features_local_,
                                          out_features_,
                                          r_max);
    scatter_add_lora_delta_(y, delta, spec);
    if (wrapper_owns_reduction_ && pg != nullptr) {
      y = xllm::parallel_state::reduce(y, pg);
    }
    return y;
  }

  // Slow path: per-seq, one all-reduce per adapter-bearing seq (legacy) or
  // deferred to a single reduction at the end (fused). Fused mode wins big
  // here — legacy issues N collectives per proj per layer for an N-seq
  // batch; fused issues 1. Interleaved base + adapter seqs are still the
  // main cost driver; adapter-affinity batching at the gateway helps.
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
    torch::Tensor delta;
    if (wrapper_owns_reduction_) {
      // Fused: keep tmp partial; delta stays partial on out-dim; combined
      // reduce happens after the loop.
      delta = torch::matmul(tmp, pd->B.transpose(0, 1));
    } else {
      if (tp_world_size_ > 1 && pg != nullptr) {
        tmp = xllm::parallel_state::reduce(tmp, pg);
      }
      delta = torch::matmul(tmp, pd->B.transpose(0, 1));
    }
    delta = (delta * pd->scaling).to(y.dtype());

    y.slice(0, tok_off, tok_off + seq_len).add_(delta);
    tok_off += seq_len;
  }
  if (wrapper_owns_reduction_ && pg != nullptr) {
    y = xllm::parallel_state::reduce(y, pg);
  }
  return y;
}

void LoRARowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

// 2-arg overload for FlashComm1 sequence-parallel dispatch.
// Delegates to the single-arg forward; LoRA delta does not participate in
// sequence-parallel reduce.
torch::Tensor LoRARowParallelLinearImpl::forward(
    torch::Tensor input,
    xllm::RowParallelReduceMode reduce_mode) {
  return this->forward(input);
}

}  // namespace layer
}  // namespace xllm
