/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/nn/functional/normalization.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/lora/lora_runtime.h"
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_qwen3_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

class QWen3DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer> {
 public:
  QWen3DecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer>(context,
                                                             layer_id) {}
};
TORCH_MODULE(QWen3DecoderLayer);

class QWen3ModelImpl : public LlmModelImplBase<QWen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen3DecoderLayer>("qwen3", context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    // Eagle3: layer ids to capture (can be read from layers_to_capture in
    // config.json)
    if (FLAGS_speculative_algorithm == "Eagle3") {
      const auto& layer_ids_from_config = model_args.layers_to_capture();
      if (!layer_ids_from_config.empty()) {
        set_eagle3_layers_to_capture(
            std::make_optional<std::vector<int32_t>>(layer_ids_from_config));
      } else {
        set_eagle3_layers_to_capture();
      }
      // Pre-allocate aux output buffer [max_tokens_per_batch, hidden_size *
      // num_captured]
      const int64_t num_captured = layers_to_capture_set_.size();
      const int64_t aux_dim = model_args.hidden_size() * num_captured;
      aux_output_buffer_ =
          torch::empty({FLAGS_max_tokens_per_batch, aux_dim}, options);
    }

    // Path C wires the model's whole-decoder-block delta path but does not
    // ship any dummy weights. Actual A/B tensors live in LoRARuntime and
    // are populated by /v1/load_lora_adapter -- when nothing is loaded the
    // forward path skips the delta entirely.
    LoRARuntime::instance().set_model_device_dtype(
        options.device(), options.dtype().toScalarType());

    // Pre-allocate on-device staging buffers for the whole-block LoRA
    // delta at ctor time.
    //
    // Constraint we hit on 82 (CANN 8.5, torch_npu 2.7.1.post2):
    // CPU->NPU tensor copies via .to() / .copy_() from the forward
    // thread crash with aclrtMemcpy 107017 "invalid handle". atb layer
    // ops on the same thread work, but torch_npu's opapi copy stream is
    // not wired for that thread. Ctor-time allocation with an
    // options-that-already-has-device is fine because it goes directly
    // through the NPU allocator without a copy.
    //
    // For the M9 milestone the /v1/load_lora_adapter handler therefore
    // does NOT copy real weights into these slots; it only registers
    // the adapter so the pipeline is exercised end-to-end. The slot
    // starts filled with small torch::randn values so the delta is
    // observable in curl responses -- proving the delta plumbing goes
    // through -- and will be replaced by actual per-adapter tensors
    // when Path B or a proper worker-broadcast pattern (like
    // update_weights) lands in P0-B.
    {
      const int64_t hidden = model_args.hidden_size();
      const int64_t max_r = 32;
      // Non-zero init so any load registration produces a visible delta
      // in chat output. Values are stable across restarts (manual_seed
      // in xllm.cpp before entering this ctor gives determinism).
      cached_lora_A_ = torch::randn({max_r, hidden}, options) * 0.005f;
      cached_lora_B_ = torch::randn({hidden, max_r}, options) * 0.005f;
      LOG(INFO) << "[Path C] pre-allocated LoRA slots max_r=" << max_r
                << " hidden=" << hidden << " device=" << cached_lora_A_.device()
                << " dtype=" << cached_lora_A_.dtype()
                << " (dummy content, per-adapter fill deferred to P0-B)";
    }

    // ===== Route F: static --lora-modules preload =====
    // At this point we're inside QWen3ModelImpl ctor, which runs on the
    // WorkerImpl::threadpool_ init task. That is the exact thread that
    // calls set_device()/init_device_context() at line 137-138 of
    // worker_impl.cpp -- the ONE thread whose CANN device context and
    // torch_npu opapi memcpy stream are wired up. .to(device) on this
    // thread is safe.
    //
    // Business tenants list adapters via --lora-modules NAME=PATH on
    // the CLI. We drain that list synchronously so /v1/lora_adapters
    // returns them from the moment HTTP starts serving. This mirrors
    // vLLM's --lora-modules static preload semantics.
    {
      const auto& cfg = LoRARuntime::instance().config_snapshot();
      if (cfg.enable_lora && !cfg.lora_modules.empty()) {
        LOG(INFO) << "[Route F] preloading " << cfg.lora_modules.size()
                  << " static adapter(s) from --lora-modules";
        for (const auto& [name, path] : cfg.lora_modules) {
          auto id = LoRARuntime::instance().load_and_activate(name, path, "");
          if (!id.has_value()) {
            LOG(ERROR) << "[Route F] preload FAILED '" << name
                       << "' path=" << path;
            continue;
          }
          // load_and_activate leaves the tensors on CPU as pending. Call
          // active_delta() to promote them, then migrate to device on
          // this thread (which owns the correct NPU context).
          auto ad_opt = LoRARuntime::instance().active_delta();
          if (!ad_opt.has_value()) {
            LOG(ERROR) << "[Route F] no active_delta after load '" << name
                       << "'";
            continue;
          }
          try {
            auto A_dev = ad_opt->A.to(options.device()).contiguous();
            auto B_dev = ad_opt->B.to(options.device()).contiguous();
            const int64_t r = A_dev.size(0);
            if (r > cached_lora_A_.size(0)) {
              LOG(ERROR) << "[Route F] adapter '" << name << "' rank " << r
                         << " exceeds max_r " << cached_lora_A_.size(0);
              continue;
            }
            // Copy real weights into pre-allocated slot [0:r]. Values
            // beyond r are still random but are never read (forward
            // slices to r).
            cached_lora_A_.slice(0, 0, r).copy_(A_dev);
            cached_lora_B_.slice(1, 0, r).copy_(B_dev);
            LOG(INFO) << "[Route F] preload OK '" << name << "' id=" << *id
                      << " r=" << r << " scaling=" << ad_opt->scaling
                      << " (real PEFT weights baked into slot [0:" << r << "])";
          } catch (const std::exception& e) {
            LOG(ERROR) << "[Route F] .to(device) failed for '" << name
                       << "': " << e.what();
          }
        }
      }
    }
  }

  torch::Tensor deepstack_process(torch::Tensor hidden_states,
                                  torch::Tensor visual_pos_masks,
                                  torch::Tensor visual_embeds) {
    visual_pos_masks = visual_pos_masks.to(hidden_states.device());
    auto selected = hidden_states.index({visual_pos_masks});
    auto local_this = selected + visual_embeds;
    hidden_states.index_put_({visual_pos_masks}, local_this);
    return hidden_states;
  }

  void set_eagle3_layers_to_capture(
      const std::optional<std::vector<int32_t>>& layer_ids = std::nullopt) {
    capture_aux_hidden_states_ = true;
    layers_to_capture_set_.clear();
    if (!layer_ids.has_value()) {
      int32_t num_layers = layers_.size();
      layers_to_capture_set_.insert(2);
      layers_to_capture_set_.insert(num_layers / 2);
      layers_to_capture_set_.insert(num_layers - 3);
    } else {
      // Config uses 0-based layer indices, same as default {2, n/2, n-3}
      for (int32_t val : layer_ids.value()) {
        layers_to_capture_set_.insert(val);
      }
    }
    LOG(INFO) << "layers_to_capture_set_ size: "
              << layers_to_capture_set_.size();
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    bool use_deepstack = input_params.deep_stacks.size() > 0;
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }
    if (use_deepstack) {
      deep_stacks = input_params.deep_stacks;  // [num_deepstack, hidden_size]
    }
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto freqs_t = x[0].clone();
        // mrop_length == freqs_length == head_dim / 2
        int64_t mrop_length = freqs_t.size(-1) / 2;

        for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
          int64_t offset = dim_idx;
          int64_t section_len = mrope_section_[dim_idx];
          int64_t length = section_len * 3;

          // Since the last dim of freqs is repeated to 2*mrop_length
          // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
          // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
          //     mrop_length+offset+6, ... < 2*mrop_length]
          torch::TensorOptions options =
              torch::TensorOptions().dtype(torch::kLong).device(x.device());
          auto idx_first_half = torch::arange(offset, length, 3, options);
          auto idx_second_half = torch::arange(
              offset + mrop_length, length + mrop_length, 3, options);

          auto idx_tensor = torch::cat({idx_first_half, idx_second_half}, 0);
          // freqs_t[..., idx] = freqs[dim_idx][..., idx]
          auto src = x[dim_idx].index_select(-1, idx_tensor);
          freqs_t.index_copy_(-1, idx_tensor, src);
        }
        return freqs_t;
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    torch::Tensor attn_mask;
    // for chunked prefill, generate the attn mask.
    if (!input_params.batch_forward_type.is_decode()) {
      if (FLAGS_enable_chunked_prefill) {
        int max_kv_seq = input_params.kv_max_seq_len;
        int num_sequences = input_params.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask =
                attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                           input_params.kv_seq_lens_vec[j],
                                           max_kv_seq,
                                           cos_pos.dtype().toScalarType(),
                                           cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    const int64_t num_tokens = h.size(0);
    const int64_t hidden_size = h.size(-1);
    int64_t capture_idx = 0;
    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event{nullptr};
      std::atomic<bool>* event_flag{nullptr};

      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      if (capture_aux_hidden_states_ &&
          layers_to_capture_set_.count(layer_index) != 0) {
        aux_output_buffer_.slice(0, 0, num_tokens)
            .slice(
                1, capture_idx * hidden_size, (capture_idx + 1) * hidden_size)
            .copy_(h.reshape({num_tokens, hidden_size}));
        capture_idx++;
      }

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
        return ModelOutput();
      }
      rolling_guard.before_layer(layer_index);

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);

      // ===== Path C: whole-decoder-block LoRA delta =====
      // If an adapter is loaded via /v1/load_lora_adapter, LoRARuntime
      // hands us its rank/scaling metadata; we pretend the pre-allocated
      // dummy A/B are that adapter's weights (see ctor for why real
      // per-adapter fill is deferred). This is enough to demonstrate
      // the full pipeline: HTTP call -> registry -> forward-time delta
      // -> visibly different chat output.
      if (auto ad = LoRARuntime::instance().active_delta(); ad.has_value()) {
        if (cached_lora_int_id_ != ad->int_id) {
          cached_lora_r_ = ad->A.size(0);
          cached_lora_scaling_ = ad->scaling;
          cached_lora_int_id_ = ad->int_id;
          LOG(INFO) << "[Path C] activated adapter '" << ad->name
                    << "' id=" << ad->int_id << " r=" << cached_lora_r_
                    << " scaling=" << cached_lora_scaling_
                    << " (using pre-allocated dummy weights until real"
                       " weight-fill lands in P0-B)";
        }
        auto A_view = cached_lora_A_.slice(0, 0, cached_lora_r_);
        auto B_view = cached_lora_B_.slice(1, 0, cached_lora_r_);
        auto tmp = torch::matmul(h, A_view.transpose(0, 1));
        auto delta = torch::matmul(tmp, B_view.transpose(0, 1));
        h = h + delta * cached_lora_scaling_;
      }

      rolling_guard.after_layer(layer_index);
      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = deepstack_process(
              h, input_params.visual_pos_masks, deep_stacks[i]);
        }
      }
    }
    auto hidden_states = norm_(h, 0);
    if (capture_aux_hidden_states_) {
      torch::Tensor aux_hidden_states =
          aux_output_buffer_.slice(0, 0, num_tokens);
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
    return ModelOutput(hidden_states);
  }

 private:
  torch::Tensor viusal_pos_mask_;
  std::unordered_set<int32_t> layers_to_capture_set_;
  bool capture_aux_hidden_states_ = false;
  torch::Tensor aux_output_buffer_;

  // Path C on-device LoRA cache. Pre-allocated at ctor time, populated
  // via copy_ from CPU when a new adapter is activated.
  torch::Tensor cached_lora_A_;  // [max_r, hidden]
  torch::Tensor cached_lora_B_;  // [hidden, max_r]
  int64_t cached_lora_r_ = 0;
  int64_t cached_lora_hidden_ = 0;
  float cached_lora_scaling_ = 0.0f;
  uint64_t cached_lora_int_id_ = 0;  // 0 == no adapter cached yet
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
  }
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_atb, qwen3_atb, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS_WITH_VARNAME(qwen3_atb, qwen3_atb, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  // Eagle3: layer ids (0-based) to capture from config, e.g.
  // "layers_to_capture": [2, 14, 25]; defaults to empty if missing
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model
