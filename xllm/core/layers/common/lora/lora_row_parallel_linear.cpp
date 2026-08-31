/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_row_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>

#include "framework/lora/lora_config.h"
#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"
#if defined(USE_NPU)
#include "framework/lora/ascendc/ascendc_lora_storage.h"
#include "framework/lora/ascendc/xllm_lora_ascendc_binding.h"
#endif
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
      y = y + delta;  // NPU add_ silent no-op fix
      if (wrapper_owns_reduction_ && pg != nullptr) {
        y = xllm::parallel_state::reduce(y, pg);
      }
      return y;
    }
  }

  // Slow path: per-seq, one all-reduce per adapter-bearing seq (legacy) or
  // deferred to a single reduction at the end (fused). Fused mode wins big
  // here — legacy issues N collectives per proj per layer for an N-seq
  // batch; fused issues 1. Interleaved base + adapter seqs are still the
  // main cost driver; adapter-affinity batching at the gateway helps.
#if defined(USE_NPU)
  // Sprint γ+1 (2026-09-01): multi-rank multi-adapter row-parallel dispatch.
  // Iterates rank buckets and launches one bgmv_shrink+expand per bucket.
  //
  // Row A_stacked [N, R, H_in_full] is TP-sharded on dim=2 per tp_rank.
  // B_stacked [N, H_out, R] is not sharded.
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr) break;
    static bool first_attempt_row = true;
    if (first_attempt_row) {
      first_attempt_row = false;
      LOG(INFO) << "[AscendC Row multishape] first attempt proj=" << proj_name_;
    }
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) break;
    if (!input.device().is_privateuseone()) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto bv = storage.get_bucketed(ctx->layer_index, proj_name_);
    if (bv.num_buckets() == 0) break;
    if (!storage.slot_lookup_dev().defined() ||
        !storage.bucket_lookup_dev().defined()) {
      break;
    }
    if (ctx->adapter_ids_per_token == nullptr ||
        !ctx->adapter_ids_per_token->defined()) {
      break;
    }
    if (ctx->adapter_ids_per_token->numel() != input.size(0)) break;

    auto per_tok_slot =
        storage.slot_lookup_dev().index_select(0, *ctx->adapter_ids_per_token);
    auto per_tok_bucket = storage.bucket_lookup_dev().index_select(
        0, *ctx->adapter_ids_per_token);
    if ((per_tok_slot < 0).any().to(torch::kCPU).item<bool>()) break;

    LOG_EVERY_N(INFO, 100) << "[AscendC Row multishape] enter proj="
                           << proj_name_ << " aids=" << adapter_ids.size()
                           << " layer=" << ctx->layer_index
                           << " buckets=" << bv.num_buckets();

    const auto buf_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

    for (int b = 0; b < bv.num_buckets(); ++b) {
      const auto& view = bv.buckets[b];
      if (!view.valid) continue;
      const int64_t R = view.A_stacked.size(1);
      if (R > 64) continue;

      // TP shard A along dim=2 (H_in_full → in_features_local)
      const int64_t A_hin = view.A_stacked.size(2);
      const int64_t in_local = static_cast<int64_t>(in_features_local_);
      torch::Tensor A_stacked_local = view.A_stacked;
      if (tp_world_size_ > 1 && A_hin > in_local) {
        const int64_t num_shards = A_hin / in_local;
        if (num_shards <= 0) continue;
        const int64_t shard_idx = static_cast<int64_t>(tp_rank_) * num_shards /
                                  static_cast<int64_t>(tp_world_size_);
        const int64_t start = shard_idx * in_local;
        A_stacked_local = view.A_stacked.slice(2, start, start + in_local);
      } else if (A_hin != input.size(1)) {
        continue;
      }

      auto mask = (per_tok_bucket == static_cast<int64_t>(b));
      auto sub_token_ids = torch::nonzero(mask).squeeze(-1);
      if (sub_token_ids.numel() == 0) continue;

      auto sub_input = input.index_select(0, sub_token_ids);
      auto sub_slot = per_tok_slot.index_select(0, sub_token_ids);
      const int64_t sub_B = sub_input.size(0);
      const int64_t out_size = view.B_stacked.size(1);

      auto sub_buf = torch::zeros({sub_B, R}, buf_options);
      auto sub_y = torch::zeros({sub_B, out_size}, input.options());
      xllm::bgmv_shrink(sub_input,
                        A_stacked_local,
                        sub_slot,
                        sub_buf,
                        static_cast<double>(view.scaling));
      xllm::bgmv_expand(sub_buf, view.B_stacked, sub_slot, sub_y, 0, out_size);
      y.index_add_(0, sub_token_ids, sub_y);
    }
    return y;
  } while (false);
#endif  // USE_NPU

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
