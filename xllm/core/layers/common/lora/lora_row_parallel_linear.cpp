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
  // Commit C (2026-08-28) - AscendC LoRA slow_path for row parallel (down/o).
  // Replaces per-seq for-loop with 1 batched bgmv (single proj_name_). Reuses
  // storage buf_down (independent from QKV bufs) + slot_lookup_dev.
  // Note: Row parallel shards A along dim=2 (input hidden) via tp_rank.
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr)
      break;  // FIX_F5B: comprehensive sprint gamma disable
    static bool first_attempt_row = true;
    if (first_attempt_row) {
      first_attempt_row = false;
      LOG(INFO) << "[AscendC Row slow_path] first attempt (env checked) proj="
                << proj_name_;
    }
    static std::atomic<int64_t> row_enter_reached{0};
    static std::atomic<int64_t> row_aids_size_le_1{0};
    static std::atomic<int64_t> row_defined_break{0};
    static std::atomic<int64_t> row_numel_break{0};
    static std::atomic<int64_t> row_buf_break{0};
    static std::atomic<int64_t> row_kernel_launch{0};
    row_enter_reached.fetch_add(1);
    LOG_EVERY_N(ERROR, 500) << "[Row_STAT] enter=" << row_enter_reached.load()
                            << " aids_le_1=" << row_aids_size_le_1.load()
                            << " defined_break=" << row_defined_break.load()
                            << " numel_break=" << row_numel_break.load()
                            << " buf_break=" << row_buf_break.load()
                            << " kernel_launch=" << row_kernel_launch.load();
    LOG_FIRST_N(ERROR, 20)
        << "[AscendC Row guard_diag P1]"
        << " env=" << (std::getenv("USE_ASCENDC_LORA") ? "set" : "unset")
        << " aids_size=" << adapter_ids.size()
        << " device_npu=" << input.device().is_privateuseone()
        << " proj=" << proj_name_ << " layer=" << ctx->layer_index
        << " input_hidden=" << input.size(1)
        << " total_tokens=" << input.size(0);
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) {
      row_aids_size_le_1.fetch_add(1);
      break;
    }
    if (!input.device().is_privateuseone()) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto view = storage.get_stacked(ctx->layer_index, proj_name_);
    LOG_FIRST_N(ERROR, 20) << "[AscendC Row guard_diag P2]"
                           << " layer=" << ctx->layer_index
                           << " proj=" << proj_name_
                           << " view_valid=" << view.valid;
    if (!view.valid) break;

    const int64_t R = view.A_stacked.size(1);
    const int64_t A_hin_early = view.A_stacked.size(2);
    const int64_t in_local_early = static_cast<int64_t>(in_features_local_);
    LOG_FIRST_N(ERROR, 20) << "[AscendC Row guard_diag P3]"
                           << " layer=" << ctx->layer_index
                           << " proj=" << proj_name_ << " R=" << R
                           << " view_dim=" << view.A_stacked.dim()
                           << " view_R=" << view.A_stacked.size(1)
                           << " A_hin=" << A_hin_early
                           << " in_local=" << in_local_early
                           << " tp_world_size=" << tp_world_size_
                           << " input_hidden=" << input.size(1);
    if (R > 64) break;
    if (view.A_stacked.dim() != 3) break;
    if (view.A_stacked.size(1) != R) break;
    // Row A shape: [N, R, H_in_full]. TP shards A along dim=2 by tp_rank.
    // A_local width should match input.size(1) (already TP-sharded input).
    const int64_t A_hin = view.A_stacked.size(2);
    const int64_t in_local = static_cast<int64_t>(in_features_local_);
    if (tp_world_size_ > 1 && A_hin > in_local) {
      // A pre-shard: caller will slice per-rank below
    } else if (A_hin != input.size(1)) {
      break;
    }

    bool all_registered = true;
    for (size_t s = 0; s < adapter_ids.size(); ++s) {
      if (adapter_ids[s] == 0) {
        all_registered = false;
        break;
      }
    }
    LOG_FIRST_N(ERROR, 20)
        << "[AscendC Row guard_diag P4]"
        << " layer=" << ctx->layer_index << " proj=" << proj_name_
        << " all_reg=" << all_registered
        << " aids_pt_ptr=" << (ctx->adapter_ids_per_token ? "ok" : "null")
        << " aids_pt_def="
        << (ctx->adapter_ids_per_token && ctx->adapter_ids_per_token->defined()
                ? "def"
                : "undef")
        << " slot_lookup_def="
        << (storage.slot_lookup_dev().defined() ? "def" : "undef")
        << " aids_pt_numel="
        << (ctx->adapter_ids_per_token && ctx->adapter_ids_per_token->defined()
                ? ctx->adapter_ids_per_token->numel()
                : -1)
        << " total_tokens=" << input.size(0)
        << " buf_down=" << (storage.buf_down().defined() ? "def" : "undef")
        << " buf_max_t=" << storage.buf_max_t();
    if (!all_registered) break;

    if (ctx->adapter_ids_per_token == nullptr ||
        !ctx->adapter_ids_per_token->defined() ||
        !storage.slot_lookup_dev().defined()) {
      row_defined_break.fetch_add(1);
      break;
    }

    const int64_t total_tokens = input.size(0);
    if (ctx->adapter_ids_per_token->numel() != total_tokens) {
      row_numel_break.fetch_add(1);
      break;
    }
    if (!storage.buf_down().defined()) {
      row_buf_break.fetch_add(1);
      break;
    }
    if (storage.buf_max_t() < total_tokens) {
      row_buf_break.fetch_add(1);
      break;
    }

    LOG_EVERY_N(INFO, 100) << "[AscendC Row slow_path] enter, aids="
                           << adapter_ids.size()
                           << " layer=" << ctx->layer_index
                           << " proj=" << proj_name_ << " R=" << R;

    row_kernel_launch.fetch_add(1);
    auto indices =
        storage.slot_lookup_dev().index_select(0, *ctx->adapter_ids_per_token);

    // TP shard A along dim=2 (H_in_full -> in_features_local)
    torch::Tensor A_stacked_local = view.A_stacked;
    if (tp_world_size_ > 1 && A_hin > in_local) {
      const int64_t num_shards = A_hin / in_local;
      if (num_shards <= 0) break;
      const int64_t shard_idx = static_cast<int64_t>(tp_rank_) * num_shards /
                                static_cast<int64_t>(tp_world_size_);
      const int64_t start = shard_idx * in_local;
      A_stacked_local = view.A_stacked.slice(2, start, start + in_local);
    }

    // B: [N, H_out, R]. Row parallel: B is full output (no TP shard on B).
    const int64_t out_size = view.B_stacked.size(1);

    auto buf_d =
        storage.buf_down().slice(0, 0, total_tokens).slice(1, 0, R).zero_();
    xllm::bgmv_shrink(input,
                      A_stacked_local,
                      indices,
                      buf_d,
                      static_cast<double>(view.scaling));
    xllm::bgmv_expand(buf_d, view.B_stacked, indices, y, 0, out_size);
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
