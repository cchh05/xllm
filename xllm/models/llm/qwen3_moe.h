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

#include "core/common/rec_model_utils.h"
#include "core/framework/model/model_output.h"
#include "core/layers/qwen3_moe_decoder_layer.h"
#include "framework/lora/lora_context.h"
#include "framework/lora/lora_runtime.h"
#include "llm_model_base.h"
#if defined(USE_NPU)
#include "core/common/global_flags.h"
#include "core/layers/common/attention_mask.h"
#endif

namespace xllm {

class Qwen3MoeModelImpl : public LlmModelImplBase<layer::Qwen3MoeDecoderLayer> {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::Qwen3MoeDecoderLayer>(
            "qwen3_moe",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    layers_.reserve(model_args.n_layers());
    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    // register submodules
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::Qwen3MoeDecoderLayer(context, i);
      layers_.push_back(layer);
    }

#if defined(USE_NPU)
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);
#endif

    // fix #4: static LoRA adapter preload (mirror of qwen3.h ctor block).
    // Attention-only adapters are handled by the Qwen2Attention wrappers;
    // any experts.* keys in adapter safetensors are ignored by the loader
    // (P1 will add expert-LoRA support).
    if (LoRARuntime::instance().enabled()) {
      LoRARuntime::instance().set_model_device_dtype(
          options.device(), options.dtype().toScalarType());
      const auto& modules = LoRARuntime::instance().config().lora_modules;
      if (!modules.empty()) {
        LOG(INFO) << "[qwen3_moe M10] preloading " << modules.size()
                  << " per-proj static adapter(s) on " << options.device();
        auto parallel_args = context.get_parallel_args();
        const int32_t tp_size =
            std::max(1, parallel_args.world_size() / parallel_args.dp_size());
        const int32_t tp_rank = parallel_args.rank() % tp_size;
        int ok = 0, failed = 0;
        // Compute MoE expert partition (matches FusedMoEImpl::load_experts).
        const int32_t num_experts_total =
            static_cast<int32_t>(model_args.num_experts());
        const int32_t ep_size = std::max(1, parallel_args.ep_size());
        const int32_t num_experts_per_rank = num_experts_total / ep_size;
        // ep_rank -- when ep_size == 1, this is 0 for all TP ranks.
        const int32_t ep_rank = (ep_size > 1) ? parallel_args.rank() : 0;
        const int32_t start_expert_id = ep_rank * num_experts_per_rank;
        const int32_t moe_intermediate =
            static_cast<int32_t>(model_args.moe_intermediate_size());
        for (const auto& [name, path] : modules) {
          auto id =
              LoRARuntime::instance().install_static_adapter_on_device_per_proj(
                  name,
                  path,
                  /*base_model_name=*/"",
                  options.device(),
                  options.dtype().toScalarType(),
                  TPInfo{tp_size, tp_rank});
          if (id.has_value()) {
            ++ok;
            LOG(INFO) << "[qwen3_moe M10] preloaded '" << name
                      << "' id=" << *id;
          } else {
            ++failed;
            LOG(ERROR) << "[qwen3_moe M10] failed '" << name << "'";
          }

          // Also install MoE expert LoRA tensors (experts.{E}.{proj}).
          // Uses the same name/path -> same int_id via registry idempotency.
          auto moe_id =
              LoRARuntime::instance().install_static_adapter_on_moe_experts(
                  name,
                  path,
                  /*base_model_name=*/"",
                  options.device(),
                  options.dtype().toScalarType(),
                  TPInfo{tp_size, tp_rank},
                  num_experts_total,
                  num_experts_per_rank,
                  start_expert_id,
                  moe_intermediate);
          if (moe_id.has_value()) {
            LOG(INFO) << "[qwen3_moe M10-MoE] preloaded expert-LoRA for '"
                      << name << "' id=" << *moe_id;
          }
        }
        LOG(INFO) << "[qwen3_moe M10] preload done: ok=" << ok
                  << " failed=" << failed;
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

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      // mrop_length == freqs_length == head_dim / 2
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);

      for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;  // H -> offset=1, W -> offset=2
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;

        // Since the last dim of freqs is repeated to 2*mrop_length
        // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
        // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
        //     mrop_length+offset+6, ... < 2*mrop_length]
        auto idx_first_half = torch::arange(offset, length, 3, torch::kLong);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, torch::kLong);
        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
        // freqs_t[..., idx] = freqs[dim_idx][..., idx]
        auto src = x[dim_idx].index_select(-1, idx_tensor);
        freqs_t.index_copy_(-1, idx_tensor, src);
      }
      return freqs_t;
    };
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(-1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(-1)}));
    return std::make_pair(cos_pos, sin_pos);
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    ModelInputParams modified_input_params = input_params;

    // M10 per-proj LoRA (mirror of LlmModelImplBase::forward push).
    // Qwen3MoE forward overrides the base's forward, so LoRAContext must
    // be pushed here or wrappers never see adapter routing info.
    LoRAContextFrame lora_frame;
    lora_frame.adapter_ids = &modified_input_params.adapter_ids;
    lora_frame.q_seq_lens_vec = &modified_input_params.q_seq_lens_vec;
    // Phase A W2 v2: device-side per-token adapter tensor (built in
    // ModelInputParams::to(device); may be undefined() for pure-base).
    lora_frame.adapter_ids_per_token = &modified_input_params.adapter_ids_per_token;
    lora_frame.layer_index = -1;
    LoRAScope _lora_scope(lora_frame);

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    auto deep_stacks = input_params.deep_stacks;
    int deep_stack_size = deep_stacks.size();
    if (!modified_input_params.attn_metadata) {
#if defined(USE_NPU)
      max_seq_len_ =
          std::max(modified_input_params.kv_max_seq_len, max_seq_len_);
      torch::Tensor attn_mask;
      if (FLAGS_enable_chunked_prefill) {
        const int32_t max_kv_seq = modified_input_params.kv_max_seq_len;
        const int32_t num_sequences = modified_input_params.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);
          for (int32_t j = 0; j < num_sequences; ++j) {
            auto mask = attn_mask_.gen_append_mask(
                modified_input_params.q_seq_lens_vec[j],
                modified_input_params.kv_seq_lens_vec[j],
                max_kv_seq,
                h.dtype().toScalarType(),
                h.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        } else {
          attn_mask = attn_mask_.get_attn_mask(
              max_seq_len_, h.dtype().toScalarType(), h.device());
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            max_seq_len_, h.dtype().toScalarType(), h.device());
      }
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(
                  modified_input_params, model_args_, attn_mask));
#else
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(modified_input_params,
                                                     model_args_));
#endif
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);
    bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    const LlmRecMultiRoundParams* llmrec_params = nullptr;
    if (is_rec_multi_round_mode() &&
        modified_input_params.has_llmrec_params()) {
      llmrec_params = modified_input_params.llmrec_params();
      CHECK_EQ(llmrec_params->full_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_k_caches per layer.";
      CHECK_EQ(llmrec_params->full_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires full_v_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_k_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_k_caches per layer.";
      CHECK_EQ(llmrec_params->unshared_v_caches.size(), layers_.size())
          << "Rec multi-round mode requires unshared_v_caches per layer.";
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      set_lora_context_layer(static_cast<int32_t>(i));
      if (llmrec_params != nullptr) {
        attn_metadata.full_k_cache = llmrec_params->full_k_caches[i];
        attn_metadata.full_v_cache = llmrec_params->full_v_caches[i];
        attn_metadata.unshared_k_cache = llmrec_params->unshared_k_caches[i];
        attn_metadata.unshared_v_cache = llmrec_params->unshared_v_caches[i];
      }
#if defined(USE_CUDA) || defined(USE_MUSA)
      attn_metadata.plan_info->layer_id = i;
#endif
      auto& layer = layers_[i];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[i],
                modified_input_params);

      if (deep_stack_size && i < deep_stack_size) {
        h = deepstack_process(h, input_params.visual_pos_masks, deep_stacks[i]);
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

  StateDict update_expert_dict(const StateDict& state_dict) {
    auto dict = std::unordered_map<std::string, torch::Tensor>(
        state_dict.begin(), state_dict.end());
    std::unordered_map<std::string, torch::Tensor> expert_dict;
    for (auto& [name, expert_gate_up] : dict) {
      if (name.find(".mlp.experts.gate_up_proj") == std::string::npos) {
        continue;
      }

      const std::string prefix = name.substr(0, name.find("gate_up_proj"));
      const std::string down_name = prefix + "down_proj";
      auto expert_down = state_dict.get_tensor(down_name);
      CHECK(expert_down.defined()) << "not find down_proj: " << down_name;

      const int32_t num_experts = expert_gate_up.size(0);
      auto chunks = expert_gate_up.chunk(2, -1);
      torch::Tensor expert_gate = chunks[0].permute({0, 2, 1});
      torch::Tensor expert_up = chunks[1].permute({0, 2, 1});
      expert_down = expert_down.permute({0, 2, 1});

      for (int j = 0; j < num_experts; ++j) {
        const std::string expert_key = prefix + std::to_string(j) + ".";
        expert_dict[expert_key + "gate_proj.weight"] = expert_gate[j];
        expert_dict[expert_key + "up_proj.weight"] = expert_up[j];
        expert_dict[expert_key + "down_proj.weight"] = expert_down[j];
      }
    }
    if (expert_dict.empty()) {
      return state_dict;
    }
    dict.merge(expert_dict);
    return StateDict(std::move(dict));
  }

  void load_state_dict(const StateDict& state_dict) override {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    auto new_state_dict = update_expert_dict(state_dict);
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(new_state_dict.get_dict_with_prefix(
          "layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

#if defined(USE_NPU)
 private:
  layer::AttentionMask attn_mask_;
#endif
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl : public LlmForCausalLMImplBase<Qwen3MoeModel> {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Qwen3MoeModel>(context) {}
};
TORCH_MODULE(Qwen3MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_moe, Qwen3MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
// https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
REGISTER_MODEL_ARGS(qwen3_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 48);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(num_experts, "num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  // arguments to be compatible with other fused moe models
  LOAD_ARG_OR(n_routed_experts, "num_experts", 128);
  SET_ARG(n_shared_experts, 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0);
});
}  // namespace xllm
