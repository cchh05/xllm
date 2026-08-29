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

#include "layers/common/lora/lora_qkv_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"

#if defined(USE_NPU)
#include <cstdlib>

#include "framework/lora/ascendc/ascendc_lora_storage.h"
#include "framework/lora/ascendc/xllm_lora_ascendc_binding.h"
#endif

namespace xllm {
namespace layer {

LoRAQKVParallelLinearImpl::LoRAQKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args,
    bool q_has_gate)
    : hidden_size_(hidden_size) {
  // Cache TP topology for LoRA weight sharding logic.
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_world_size_ = parallel_args.tp_group_->world_size();

  // Match base's per-partition sizing exactly. When q_has_gate=true
  // (Qwen3-Next attn_output_gate), num_heads passed in already accounts
  // for the fused q+gate lanes (num_heads_attn * 2), so q_size_local is
  // naturally the fused lane width. The PEFT adapter for such a model
  // must also carry a B_q sized to the fused lane; if it does not, the
  // shape check in set_lora_weights / per-proj slicing will trip.
  //   q_size_local  = num_heads * head_size          (already tp-partitioned)
  //   kv_size_local = num_kv_heads * head_size       (with kv_head_replicas)
  q_size_local_ = num_heads * head_size;
  kv_size_local_ = num_kv_heads * head_size;
  out_size_local_ = q_size_local_ + 2 * kv_size_local_;
  q_has_gate_ = q_has_gate;

  // Base linear owns the vanilla forward + weight loading path. NOT
  // register_module'd here on purpose — see the note in the header. Base
  // is held as a plain member; attention's load_state_dict("qkv_proj.")
  // routes through this wrapper's load_state_dict, which forwards to
  // base_->load_state_dict, so the on-disk state_dict key layout stays
  // "qkv_proj.weight" (identical to vanilla xllm).
  base_ = QKVParallelLinear(hidden_size,
                            num_heads,
                            num_kv_heads,
                            head_size,
                            num_kv_head_replicas,
                            bias,
                            gather_output,
                            parallel_args,
                            options,
                            quant_args);
}

torch::Tensor LoRAQKVParallelLinearImpl::forward(torch::Tensor input) {
  auto y = base_->forward(input);

  // Legacy hardcoded path (Spike Day 5b). Kept behind lora_active_ so
  // unit tests still work without a LoRARuntime. Real production path
  // uses the per-request routing below.
  if (lora_active_ && lora_rank_ > 0) {
    auto lora_intermediate = torch::matmul(input, lora_a_.transpose(0, 1));
    auto delta = torch::matmul(lora_intermediate, lora_b_.transpose(0, 1));
    y = y + (delta * lora_scaling_).to(y.dtype());
  }

  // M10 per-request per-proj real LoRA. QKV wrapper concatenates Q/K/V
  // deltas along the last dim to match the base's fused output.
  const auto* ctx = current_lora_context();
  // DEBUG P1G: dump wrapper's view of context (rate limited)
  LOG_EVERY_N(ERROR, 200) << "[P1G_QKV_FWD] ctx=" << (ctx ? "ok" : "null")
                          << " aids_ptr="
                          << (ctx && ctx->adapter_ids ? "ok" : "null")
                          << " layer_idx=" << (ctx ? ctx->layer_index : -99)
                          << " aids_size="
                          << (ctx && ctx->adapter_ids ? ctx->adapter_ids->size()
                                                      : 0);
  if (ctx == nullptr || ctx->adapter_ids == nullptr ||
      ctx->q_seq_lens_vec == nullptr || ctx->layer_index < 0) {
    return y;
  }
  const auto& adapter_ids = *ctx->adapter_ids;
  const auto& q_seq_lens = *ctx->q_seq_lens_vec;
  if (adapter_ids.empty() || adapter_ids.size() != q_seq_lens.size()) {
    return y;
  }

  // Fast path: batch is pure-base.
  bool any_nonzero = false;
  for (auto id : adapter_ids) {
    if (id != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) return y;

  // P1a Phase 0 fast path: batch uses one non-zero adapter (99% prod).
  // Skip per-seq slice/matmul loop; do 2 matmuls on full [T, hidden]:
  //   tmp = x @ A^T                    [T, r]
  //   delta_q = tmp @ B_q_local^T      [T, q_local]
  //   delta_k = tmp @ B_k_local^T      [T, kv_local]
  //   delta_v = tmp @ B_v_local^T      [T, kv_local]
  //   y[:, q_local .. q_local+2*kv_local] += cat([q,k,v], -1)
  // For base-only sequences interleaved with adapter sequences we still
  // fall back to per-seq (see below); this fast path requires *all*
  // non-zero ids match one adapter and no base-only seq is present.
  {
    uint64_t sole_aid = 0;
    bool single_adapter = true;
    for (auto id : adapter_ids) {
      if (id == 0) {
        single_adapter = false;  // any base-only seq disqualifies fast path
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
      const auto* q_pd =
          runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "q_proj");
      const auto* k_pd =
          runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "k_proj");
      const auto* v_pd =
          runtime.get_per_proj_delta(sole_aid, ctx->layer_index, "v_proj");
      if (q_pd == nullptr && k_pd == nullptr && v_pd == nullptr) {
        return y;
      }
      // A is the same for Q/K/V? No — three independent A matrices. But
      // each proj has its own A. We do 3 shrinks + 3 expands (each on
      // the full [T, hidden] input, no slicing).
      auto shrink_expand = [&](const LoRARuntime::ProjDelta* pd,
                               int64_t out_local) -> torch::Tensor {
        if (pd == nullptr) {
          return torch::zeros({input.size(0), out_local}, input.options());
        }
        // tmp: [T, hidden] @ [hidden, r] -> [T, r]
        auto tmp = torch::matmul(input, pd->A.transpose(0, 1));
        // B [out_full, r] -> slice rows to local: [out_local, r].
        // For q_proj under TP: out_full = q_hidden_total, sliced evenly
        // per tp_rank. For k/v_proj when TP > num_kv_heads (GQA replicas),
        // out_full = kv_hidden_total which is smaller than out_local *
        // tp_world_size; several ranks share the same kv head. Compute
        // the actual shard count from B.size(0) / out_local so each rank
        // picks the correct slice.
        torch::Tensor B_local = pd->B;
        if (tp_world_size_ > 1 && pd->B.size(0) > out_local) {
          const int64_t num_shards = pd->B.size(0) / out_local;
          const int64_t shard_idx =
              (num_shards > 0) ? (tp_rank_ * num_shards / tp_world_size_) : 0;
          const int64_t start = shard_idx * out_local;
          B_local = pd->B.slice(0, start, start + out_local);
        }
        // d: [T, r] @ [r, out_local] -> [T, out_local]
        auto d = torch::matmul(tmp, B_local.transpose(0, 1));
        return (d * pd->scaling).to(input.dtype());
      };
      auto q_delta = shrink_expand(q_pd, q_size_local_);
      auto k_delta = shrink_expand(k_pd, kv_size_local_);
      auto v_delta = shrink_expand(v_pd, kv_size_local_);
      auto qkv_delta = torch::cat({q_delta, k_delta, v_delta}, /*dim=*/-1);
      y.add_(qkv_delta);
      return y;
    }
  }

  // Fallback per-seq apply: slice y along dim=0 by q_seq_lens, look up
  // (int_id, layer, {q_proj,k_proj,v_proj}), stack Q/K/V deltas into
  // one [seq_tokens, q_size + 2*kv_size] slab, add to that slice.

#if defined(USE_NPU)
  // Commit B — AscendC LoRA slow_path (mixed adapter batch).
  //
  // Replaces the per-seq for loop below with 3 batched bgmv_shrink +
  // 3 batched bgmv_expand calls when:
  //   USE_ASCENDC_LORA env is set
  //   adapter_ids.size() > 1 (fast_path above handles == 1)
  //   input.device().is_privateuseone() (NPU only)
  //   Storage has an entry for (layer, q_proj) AND (layer, k_proj) AND
  //     (layer, v_proj)
  //   R <= 64 (kernel constraint, per vllm-ascend punica_npu.py)
  //   All seqs have a registered aid (base-only seq mixed → per-seq)
  //
  // The three per-proj B slabs are TP-sharded per this rank via a view
  // slice: fast_path uses the same "B.size(0) / out_local → shard_idx"
  // formula (see line ~178 above) for a 2D pd->B [H_out_full, R]; here
  // AscendCLoRAStorage::StackedView::B_stacked is 3D
  // [num_adapters, H_out_full, R] so the slice is along dim=1 instead of 0.
  //
  // Known-optimization-followup: `torch::zeros` per forward for buf_{q,k,v}
  // fp32 accumulators. HBM alloc churn may dominate at high decode QPS.
  // If Step 4 bench misses KPI, move buf allocation into
  // AscendCLoRAStorage (per-layer pre-alloc + zero_() reuse).
  do {
    if (std::getenv("DISABLE_ASCENDC_SPRINT_GAMMA") != nullptr)
      break;  // FIX_F5B: comprehensive sprint gamma disable
    // Diag: first-attempt log fires exactly once per process so absence of
    // this line in the server log means the slow_path AscendC branch was
    // never even considered (e.g. the whole #ifdef USE_NPU wasn't compiled
    // in, or the flow never reached this point).
    {
      static bool first_attempt = true;
      if (first_attempt) {
        first_attempt = false;
        LOG(INFO) << "[AscendC LoRA slow_path] first attempt (env checked)";
      }
    }
    if (std::getenv("USE_ASCENDC_LORA") == nullptr) break;
    if (adapter_ids.size() <= 1) break;
    if (!input.device().is_privateuseone()) break;

    auto& storage = AscendCLoRAStorage::instance();
    auto qv = storage.get_stacked(ctx->layer_index, "q_proj");
    auto kv = storage.get_stacked(ctx->layer_index, "k_proj");
    auto vv = storage.get_stacked(ctx->layer_index, "v_proj");
    if (!qv.valid || !kv.valid || !vv.valid) break;

    const int64_t R = qv.A_stacked.size(1);
    if (R > 64) break;

    // Shape guards — early exit is better than device crash if storage
    // ever produced malformed slabs. register_adapter should refuse
    // zero-sized inputs so this is truly defense-in-depth.
    if (qv.A_stacked.dim() != 3 || kv.A_stacked.dim() != 3 ||
        vv.A_stacked.dim() != 3) {
      break;
    }
    if (qv.A_stacked.size(1) != R || kv.A_stacked.size(1) != R ||
        vv.A_stacked.size(1) != R) {
      break;  // Q/K/V rank mismatch
    }
    if (qv.A_stacked.size(2) != input.size(1)) {
      break;  // hidden_in mismatch
    }

    LOG_EVERY_N(INFO, 100) << "[AscendC LoRA slow_path] enter, aids="
                           << adapter_ids.size()
                           << " layer=" << ctx->layer_index << " R=" << R;

    // Per-token adapter slot indices. int_id 0 (base) or unregistered id
    // → we fall back to per-seq (skeleton simplicity; a base-only seq
    // interleaved with adapter seqs is uncommon in mixed batches).
    const int64_t total_tokens = input.size(0);
    std::vector<int64_t> idx_host(total_tokens, -1);
    int64_t off = 0;
    bool all_registered = true;
    for (size_t s = 0; s < adapter_ids.size(); ++s) {
      const int32_t slen = q_seq_lens[s];
      if (slen <= 0) continue;
      const uint64_t aid = adapter_ids[s];
      if (aid == 0) {
        all_registered = false;
        break;
      }
      auto slot_cpu = storage.build_indices_cpu({aid});
      const int64_t slot = slot_cpu.data_ptr<int64_t>()[0];
      if (slot < 0) {
        all_registered = false;
        break;
      }
      for (int32_t t = 0; t < slen; ++t) idx_host[off + t] = slot;
      off += slen;
    }
    if (!all_registered) break;
    auto indices =
        torch::from_blob(idx_host.data(), {total_tokens}, torch::kInt64)
            .clone()
            .to(input.device());

    // TP shard B_stacked along dim=1 (H_out_full → out_local). Formula
    // mirrors the fast_path slice (line ~172 above) with the extra
    // leading N dim shifted (0 → 1). num_shards >= 1 for a valid shard
    // by construction (register_adapter refuses zero-sized B).
    auto shard_B = [&](const torch::Tensor& B_full, int64_t out_local) {
      if (tp_world_size_ <= 1 || B_full.size(1) <= out_local) return B_full;
      const int64_t num_shards = B_full.size(1) / out_local;
      if (num_shards <= 0) return B_full;
      const int64_t shard_idx = tp_rank_ * num_shards / tp_world_size_;
      const int64_t start = shard_idx * out_local;
      return B_full.slice(1, start, start + out_local);
    };
    auto qB = shard_B(qv.B_stacked, q_size_local_);
    auto kB = shard_B(kv.B_stacked, kv_size_local_);
    auto vB = shard_B(vv.B_stacked, kv_size_local_);

    // 3 bgmv_shrink (Q/K/V have independent A). Buf is fp32 per kernel
    // spec — shrink writes accumulator in fp32; expand reads it back.
    const auto buf_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto buf_q = torch::zeros({total_tokens, R}, buf_options);
    auto buf_k = torch::zeros({total_tokens, R}, buf_options);
    auto buf_v = torch::zeros({total_tokens, R}, buf_options);
    xllm::bgmv_shrink(
        input, qv.A_stacked, indices, buf_q, static_cast<double>(qv.scaling));
    xllm::bgmv_shrink(
        input, kv.A_stacked, indices, buf_k, static_cast<double>(kv.scaling));
    xllm::bgmv_shrink(
        input, vv.A_stacked, indices, buf_v, static_cast<double>(vv.scaling));

    // 3 bgmv_expand writing into y at Q/K/V segment offsets. The kernel
    // accumulates in-place, so any base output already sitting in y is
    // preserved (y = base_forward result at line 76).
    xllm::bgmv_expand(buf_q, qB, indices, y, /*offset=*/0, q_size_local_);
    xllm::bgmv_expand(buf_k, kB, indices, y, q_size_local_, kv_size_local_);
    xllm::bgmv_expand(
        buf_v, vB, indices, y, q_size_local_ + kv_size_local_, kv_size_local_);
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

    const auto* q_pd =
        runtime.get_per_proj_delta(aid, ctx->layer_index, "q_proj");
    const auto* k_pd =
        runtime.get_per_proj_delta(aid, ctx->layer_index, "k_proj");
    const auto* v_pd =
        runtime.get_per_proj_delta(aid, ctx->layer_index, "v_proj");
    if (q_pd == nullptr && k_pd == nullptr && v_pd == nullptr) {
      tok_off += seq_len;
      continue;
    }

    auto x_seq = input.slice(0, tok_off, tok_off + seq_len);
    // Compute per-proj delta on the fused output. Use zeros as filler
    // for missing proj slots so the concat shape stays correct.
    // TP shard: dummy adapter's B is stored at full out size, but this
    // rank only handles [tp_rank * out_size, (tp_rank+1) * out_size).
    // Slice B to the local shard before matmul so the delta shape
    // matches base linear's local output.
    auto make_delta = [&](const LoRARuntime::ProjDelta* pd, int64_t out_size) {
      if (pd == nullptr) {
        return torch::zeros({x_seq.size(0), out_size}, x_seq.options());
      }
      auto tmp = torch::matmul(x_seq, pd->A.transpose(0, 1));
      // B is [out_full, rank]. Slice per shard so this rank picks its
      // portion of the output. For k/v_proj with GQA replicas the shard
      // count is smaller than tp_world_size, so derive it from B.size(0).
      torch::Tensor B_local = pd->B;
      if (tp_world_size_ > 1 && pd->B.size(0) > out_size) {
        const int64_t num_shards = pd->B.size(0) / out_size;
        const int64_t shard_idx =
            (num_shards > 0) ? (tp_rank_ * num_shards / tp_world_size_) : 0;
        const int64_t start = shard_idx * out_size;
        B_local = pd->B.slice(0, start, start + out_size);
      }
      auto d = torch::matmul(tmp, B_local.transpose(0, 1));
      return (d * pd->scaling).to(x_seq.dtype());
    };
    auto q_delta = make_delta(q_pd, q_size_local_);
    auto k_delta = make_delta(k_pd, kv_size_local_);
    auto v_delta = make_delta(v_pd, kv_size_local_);
    auto qkv_delta = torch::cat({q_delta, k_delta, v_delta}, /*dim=*/-1);

    y.slice(0, tok_off, tok_off + seq_len).add_(qkv_delta);
    tok_off += seq_len;
  }
  return y;
}

void LoRAQKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  // Passthrough: base handles the fused q/k/v load. We do not touch the
  // LoRA A/B tensors here — those are set separately via
  // set_lora_weights (Spike) or load_lora_state_dict (adapter manager).
  base_->load_state_dict(state_dict, prefixes);
}

void LoRAQKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  base_->load_state_dict(state_dict);
}

void LoRAQKVParallelLinearImpl::set_lora_weights(const torch::Tensor& lora_a,
                                                 const torch::Tensor& lora_b,
                                                 double scaling) {
  // Sanity checks — surface shape mismatches early instead of getting
  // an opaque matmul dispatch error at forward time.
  CHECK_EQ(lora_a.dim(), 2)
      << "lora_a must be 2-D [rank, hidden], got dim=" << lora_a.dim();
  CHECK_EQ(lora_b.dim(), 2)
      << "lora_b must be 2-D [out_local, rank], got dim=" << lora_b.dim();
  CHECK_EQ(lora_a.size(1), hidden_size_)
      << "lora_a hidden dim mismatch: got " << lora_a.size(1) << ", expected "
      << hidden_size_;
  CHECK_EQ(lora_b.size(0), out_size_local_)
      << "lora_b out_local dim mismatch: got " << lora_b.size(0)
      << ", expected " << out_size_local_ << " (q_local=" << q_size_local_
      << " + 2*kv_local=" << kv_size_local_ << ")";
  CHECK_EQ(lora_a.size(0), lora_b.size(1))
      << "lora_a rank " << lora_a.size(0) << " != lora_b rank "
      << lora_b.size(1);

  lora_a_ = lora_a.contiguous();
  lora_b_ = lora_b.contiguous();
  lora_rank_ = lora_a.size(0);
  lora_scaling_ = scaling;
  lora_active_ = true;

  LOG(INFO) << "LoRAQKVParallelLinear: activated adapter with rank="
            << lora_rank_ << ", scaling=" << lora_scaling_
            << ", tp_rank=" << tp_rank_ << "/" << tp_world_size_;
}

}  // namespace layer
}  // namespace xllm
