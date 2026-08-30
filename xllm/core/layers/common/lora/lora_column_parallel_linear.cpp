/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

#include "layers/common/lora/lora_column_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <atomic>

#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"
#if defined(USE_NPU)
#include "framework/lora/ascendc/ascendc_lora_storage.h"
#include "framework/lora/ascendc/xllm_lora_ascendc_binding.h"
#endif

namespace xllm {
namespace layer {

LoRAColumnParallelLinearImpl::LoRAColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const std::string& proj_name,
    const LinearExtraArgs& linear_extra_args)
    : proj_name_(proj_name),
      in_features_(in_features),
      out_features_(out_features),
      is_fused_gate_up_(proj_name == "gate_up_proj") {
  base_ = ColumnParallelLinear(in_features,
                               out_features,
                               bias,
                               gather_output,
                               quant_args,
                               process_group,
                               options,
                               linear_extra_args);

  // out_features here is the total; per-rank it is out_features / world_size.
  tp_rank_ = process_group ? process_group->rank() : 0;
  tp_world_size_ = process_group ? process_group->world_size() : 1;
  const int64_t world_size = tp_world_size_;
  out_size_local_ = out_features / std::max<int64_t>(1, world_size);
  inter_size_local_ = is_fused_gate_up_ ? out_size_local_ / 2 : out_size_local_;
}

torch::Tensor LoRAColumnParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  const auto* ctx = current_lora_context();
  // Commit C diag: verify LoRA context propagation to shared_experts_ path
  LOG_FIRST_N(ERROR, 20) << "[LoRAColumn_diag] proj=" << proj_name_
                         << " ctx=" << (ctx ? "ok" : "null") << " aids_ptr="
                         << (ctx && ctx->adapter_ids ? "ok" : "null")
                         << " layer=" << (ctx ? ctx->layer_index : -999);
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

  // P1a Phase 0 fast path: single-adapter batch (99% prod). Skip per-seq
  // slice/matmul loop; do 2 matmuls per proj on full [T, hidden].
  {
    uint64_t sole_aid = 0;
    bool single_adapter = true;
    for (auto id : adapter_ids) {
      if (id == 0) {
        single_adapter = false;  // base-only seq disqualifies fast path
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
      if (is_fused_gate_up_) {
        const auto* gate_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "gate_proj");
        const auto* up_pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "up_proj");
        if (gate_pd == nullptr && up_pd == nullptr) return y;
        auto shrink_expand =
            [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
          if (pd == nullptr) {
            return torch::zeros({input.size(0), inter_size_local_},
                                input.options());
          }
          auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
          torch::Tensor B_local = pd->B;
          if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
            const int64_t start = tp_rank_ * inter_size_local_;
            B_local = pd->B.slice(0, start, start + inter_size_local_);
          }
          auto d = torch::matmul(tmp, B_local.transpose(0, 1));
          return (d * pd->scaling).to(input.dtype());
        };
        auto gate_delta = shrink_expand(gate_pd);
        auto up_delta = shrink_expand(up_pd);
        auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
        y = y + fused_delta;  // NPU add_ silent no-op fix
        return y;
      } else {
        // Single-proj branch (reserved). Not exercised in Qwen family
        // currently, but mirror the fast path for symmetry.
        const auto* pd =
            runtime.get_per_proj_delta(sole_aid, ctx->layer_index, proj_name_);
        if (pd == nullptr) return y;
        auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
          const int64_t start = tp_rank_ * out_size_local_;
          B_local = pd->B.slice(0, start, start + out_size_local_);
        }
        auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
        delta = (delta * pd->scaling).to(y.dtype());
        y = y + delta;  // NPU add_ silent no-op fix
        return y;
      }
    }
  }

#if defined(USE_NPU)
  // Commit C (2026-08-28) - AscendC LoRA slow_path for fused gate_up column
  // parallel (MoE). Replaces per-seq for-loop with 2 batched bgmv (gate+up
  // independent kernel pair, concat via offset). Reuses storage buf_gate /
  // buf_up (independent from QKV bufs) + slot_lookup_dev (Fix W device
  // indices, graph-capture safe).
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr)
      break;  // FIX_F5B: comprehensive sprint gamma disable
    static bool first_attempt_col = true;
    if (first_attempt_col) {
      first_attempt_col = false;
      LOG(INFO) << "[AscendC Column slow_path] first attempt (env checked)";
    }
    static std::atomic<int64_t> col_enter_reached{0};
    static std::atomic<int64_t> col_aids_size_le_1{0};
    static std::atomic<int64_t> col_defined_break{0};
    static std::atomic<int64_t> col_numel_break{0};
    static std::atomic<int64_t> col_buf_break{0};
    static std::atomic<int64_t> col_kernel_launch{0};
    col_enter_reached.fetch_add(1);
    LOG_EVERY_N(ERROR, 500)
        << "[Column_STAT] enter=" << col_enter_reached.load()
        << " aids_le_1=" << col_aids_size_le_1.load()
        << " defined_break=" << col_defined_break.load()
        << " numel_break=" << col_numel_break.load()
        << " buf_break=" << col_buf_break.load()
        << " kernel_launch=" << col_kernel_launch.load();
    LOG_FIRST_N(ERROR, 20)
        << "[AscendC Column guard_diag P1]"
        << " env=" << (std::getenv("USE_ASCENDC_LORA") ? "set" : "unset")
        << " aids_size=" << adapter_ids.size()
        << " device_npu=" << input.device().is_privateuseone()
        << " fused_gate_up=" << is_fused_gate_up_
        << " layer=" << ctx->layer_index << " input_hidden=" << input.size(1)
        << " total_tokens=" << input.size(0);
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) {
      col_aids_size_le_1.fetch_add(1);
      break;
    }
    if (!input.device().is_privateuseone()) break;
    if (!is_fused_gate_up_) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto gate_view = storage.get_stacked(ctx->layer_index, "gate_proj");
    auto up_view = storage.get_stacked(ctx->layer_index, "up_proj");
    LOG_FIRST_N(ERROR, 20) << "[AscendC Column guard_diag P2]"
                           << " layer=" << ctx->layer_index
                           << " gate_valid=" << gate_view.valid
                           << " up_valid=" << up_view.valid;
    if (!gate_view.valid || !up_view.valid) break;

    const int64_t R = gate_view.A_stacked.size(1);
    LOG_FIRST_N(ERROR, 20) << "[AscendC Column guard_diag P3]"
                           << " layer=" << ctx->layer_index << " R=" << R
                           << " gate_dim=" << gate_view.A_stacked.dim()
                           << " up_dim=" << up_view.A_stacked.dim()
                           << " gate_R=" << gate_view.A_stacked.size(1)
                           << " up_R=" << up_view.A_stacked.size(1)
                           << " gate_A_hidden=" << gate_view.A_stacked.size(2)
                           << " input_hidden=" << input.size(1);
    if (R > 64) break;
    if (gate_view.A_stacked.dim() != 3 || up_view.A_stacked.dim() != 3) break;
    if (gate_view.A_stacked.size(1) != R || up_view.A_stacked.size(1) != R)
      break;
    if (gate_view.A_stacked.size(2) != input.size(1)) break;

    bool all_registered = true;
    for (size_t s = 0; s < adapter_ids.size(); ++s) {
      if (adapter_ids[s] == 0) {
        all_registered = false;
        break;
      }
    }
    LOG_FIRST_N(ERROR, 20)
        << "[AscendC Column guard_diag P4]"
        << " layer=" << ctx->layer_index << " all_reg=" << all_registered
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
        << " buf_gate=" << (storage.buf_gate().defined() ? "def" : "undef")
        << " buf_up=" << (storage.buf_up().defined() ? "def" : "undef")
        << " buf_max_t=" << storage.buf_max_t();
    if (!all_registered) break;

    if (ctx->adapter_ids_per_token == nullptr ||
        !ctx->adapter_ids_per_token->defined() ||
        !storage.slot_lookup_dev().defined()) {
      col_defined_break.fetch_add(1);
      break;
    }

    const int64_t total_tokens = input.size(0);
    if (ctx->adapter_ids_per_token->numel() != total_tokens) {
      col_numel_break.fetch_add(1);
      break;
    }
    if (!storage.buf_gate().defined() || !storage.buf_up().defined()) {
      col_buf_break.fetch_add(1);
      break;
    }
    if (storage.buf_max_t() < total_tokens) {
      col_buf_break.fetch_add(1);
      break;
    }

    LOG_EVERY_N(INFO, 100) << "[AscendC Column slow_path] enter, aids="
                           << adapter_ids.size()
                           << " layer=" << ctx->layer_index << " R=" << R;

    col_kernel_launch.fetch_add(1);
    auto indices =
        storage.slot_lookup_dev().index_select(0, *ctx->adapter_ids_per_token);

    auto shard_B_col = [&](const torch::Tensor& B_full) {
      if (tp_world_size_ <= 1 || B_full.size(1) <= inter_size_local_)
        return B_full;
      const int64_t num_shards = B_full.size(1) / inter_size_local_;
      if (num_shards <= 0) return B_full;
      const int64_t shard_idx = tp_rank_ * num_shards / tp_world_size_;
      const int64_t start = shard_idx * inter_size_local_;
      return B_full.slice(1, start, start + inter_size_local_);
    };
    auto gateB = shard_B_col(gate_view.B_stacked);
    auto upB = shard_B_col(up_view.B_stacked);

    auto buf_g =
        storage.buf_gate().slice(0, 0, total_tokens).slice(1, 0, R).zero_();
    auto buf_u =
        storage.buf_up().slice(0, 0, total_tokens).slice(1, 0, R).zero_();
    xllm::bgmv_shrink(input,
                      gate_view.A_stacked,
                      indices,
                      buf_g,
                      static_cast<double>(gate_view.scaling));
    xllm::bgmv_shrink(input,
                      up_view.A_stacked,
                      indices,
                      buf_u,
                      static_cast<double>(up_view.scaling));

    xllm::bgmv_expand(buf_g, gateB, indices, y, 0, inter_size_local_);
    xllm::bgmv_expand(
        buf_u, upB, indices, y, inter_size_local_, inter_size_local_);
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

    auto x_seq = input.slice(0, tok_off, tok_off + seq_len);

    if (is_fused_gate_up_) {
      // Fused gate_up: lookup gate_proj + up_proj deltas, concat.
      const auto* gate_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "gate_proj");
      const auto* up_pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, "up_proj");
      if (gate_pd == nullptr && up_pd == nullptr) {
        tok_off += seq_len;
        continue;
      }
      auto make_delta = [&](const LoRARuntime::ProjDelta* pd) -> torch::Tensor {
        if (pd == nullptr) {
          return torch::zeros({x_seq.size(0), inter_size_local_},
                              x_seq.options());
        }
        auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
        // TP shard: adapter's B is [inter_full, rank]. This rank owns
        // rows [tp_rank * inter_size_local_, +inter_size_local_).
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > inter_size_local_) {
          const int64_t start = tp_rank_ * inter_size_local_;
          B_local = pd->B.slice(0, start, start + inter_size_local_);
        }
        auto d = torch::matmul(tmp, B_local.transpose(0, 1));
        return (d * pd->scaling).to(x_seq.dtype());
      };
      auto gate_delta = make_delta(gate_pd);
      auto up_delta = make_delta(up_pd);
      auto fused_delta = torch::cat({gate_delta, up_delta}, /*dim=*/-1);
      y.slice(0, tok_off, tok_off + seq_len).add_(fused_delta);
    } else {
      // Single proj (future use — reserved). Not applicable to Qwen for now.
      const auto* pd =
          runtime.get_per_proj_delta(aid, ctx->layer_index, proj_name_);
      if (pd == nullptr) {
        tok_off += seq_len;
        continue;
      }
      auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
      // TP shard: adapter's B is [out_full, rank]. This rank owns
      // rows [tp_rank * out_size_local_, +out_size_local_).
      torch::Tensor B_local = pd->B;
      if (tp_world_size_ > 1 && pd->B.size(0) > out_size_local_) {
        const int64_t start = tp_rank_ * out_size_local_;
        B_local = pd->B.slice(0, start, start + out_size_local_);
      }
      auto delta = torch::matmul(tmp, B_local.transpose(0, 1));
      delta = (delta * pd->scaling).to(y.dtype());
      y.slice(0, tok_off, tok_off + seq_len).add_(delta);
    }
    tok_off += seq_len;
  }
  return y;
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  base_->load_state_dict(state_dict, prefixes);
}

void LoRAColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  base_->load_state_dict(state_dict, shard_tensor_count, shard_sizes);
}

}  // namespace layer
}  // namespace xllm
