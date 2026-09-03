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
  // Sprint γ+1 (2026-09-01): multi-rank multi-adapter dispatch. Replaces
  // sprint γ' single-slab do-while. Buckets tokens by rank via
  // bucket_lookup_dev and launches one bgmv_shrink+expand per bucket.
  //
  // Fused gate_up column parallel (MoE): each bucket runs 2 kernels
  // (gate + up), independent scaling per bucket's slab metadata.
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr) break;
    static bool first_attempt_col = true;
    if (first_attempt_col) {
      first_attempt_col = false;
      LOG(INFO) << "[AscendC Column multishape] first attempt";
    }
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) break;
    if (!input.device().is_privateuseone()) break;
    if (!is_fused_gate_up_) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto gate_b = storage.get_bucketed(ctx->layer_index, "gate_proj");
    auto up_b = storage.get_bucketed(ctx->layer_index, "up_proj");
    if (gate_b.num_buckets() == 0 && up_b.num_buckets() == 0) break;
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

    LOG_EVERY_N(INFO, 100) << "[AscendC Column multishape] enter aids="
                           << adapter_ids.size()
                           << " layer=" << ctx->layer_index
                           << " gate_buckets=" << gate_b.num_buckets()
                           << " up_buckets=" << up_b.num_buckets();

    auto shard_B_col = [&](const torch::Tensor& B_full) {
      if (tp_world_size_ <= 1 || B_full.size(1) <= inter_size_local_)
        return B_full;
      const int64_t num_shards = B_full.size(1) / inter_size_local_;
      if (num_shards <= 0) return B_full;
      const int64_t shard_idx = tp_rank_ * num_shards / tp_world_size_;
      const int64_t start = shard_idx * inter_size_local_;
      return B_full.slice(1, start, start + inter_size_local_);
    };

    const auto buf_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device());

    // Iterate buckets present in EITHER gate or up (they should have
    // parallel bucket_id since same adapter registers both). Use the
    // shared bucket_id space.
    const int max_buckets = std::max(gate_b.num_buckets(), up_b.num_buckets());
    for (int b = 0; b < max_buckets; ++b) {
      bool gate_valid = (b < gate_b.num_buckets()) && gate_b.buckets[b].valid;
      bool up_valid = (b < up_b.num_buckets()) && up_b.buckets[b].valid;
      if (!gate_valid && !up_valid) continue;

      const int64_t R = gate_valid ? gate_b.buckets[b].A_stacked.size(1)
                                   : up_b.buckets[b].A_stacked.size(1);
      if (R > 64) continue;
      if (gate_valid && gate_b.buckets[b].A_stacked.size(2) != input.size(1))
        continue;
      if (up_valid && up_b.buckets[b].A_stacked.size(2) != input.size(1))
        continue;

      auto mask = (per_tok_bucket == static_cast<int64_t>(b));
      auto sub_token_ids = torch::nonzero(mask).squeeze(-1);
      if (sub_token_ids.numel() == 0) continue;

      auto sub_input = input.index_select(0, sub_token_ids);
      auto sub_slot = per_tok_slot.index_select(0, sub_token_ids);
      const int64_t sub_B = sub_input.size(0);

      if (gate_valid) {
        const auto& gate_view = gate_b.buckets[b];
        auto sub_buf = torch::zeros({sub_B, R}, buf_options);
        auto gateB = shard_B_col(gate_view.B_stacked);
        auto sub_y = torch::zeros({sub_B, inter_size_local_}, input.options());
        xllm::bgmv_shrink(sub_input,
                          gate_view.A_stacked,
                          sub_slot,
                          sub_buf,
                          static_cast<double>(gate_view.scaling));
        xllm::bgmv_expand(
            sub_buf, gateB, sub_slot, sub_y, 0, inter_size_local_);
        y.slice(1, 0, inter_size_local_).index_add_(0, sub_token_ids, sub_y);
      }
      if (up_valid) {
        const auto& up_view = up_b.buckets[b];
        auto sub_buf = torch::zeros({sub_B, R}, buf_options);
        auto upB = shard_B_col(up_view.B_stacked);
        auto sub_y = torch::zeros({sub_B, inter_size_local_}, input.options());
        xllm::bgmv_shrink(sub_input,
                          up_view.A_stacked,
                          sub_slot,
                          sub_buf,
                          static_cast<double>(up_view.scaling));
        xllm::bgmv_expand(sub_buf, upB, sub_slot, sub_y, 0, inter_size_local_);
        y.slice(1, inter_size_local_, 2 * inter_size_local_)
            .index_add_(0, sub_token_ids, sub_y);
      }
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
      auto y_view_fused = y.slice(0, tok_off, tok_off + seq_len);
      y_view_fused.copy_(y_view_fused + fused_delta);
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
      auto y_view_col = y.slice(0, tok_off, tok_off + seq_len);
      y_view_col.copy_(y_view_col + delta);
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
