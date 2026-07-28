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

#include <torch/torch.h>

#include "activation.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "linear.h"
#include "lora/lora_column_parallel_linear.h"
#include "lora/lora_row_parallel_linear.h"

namespace xllm {
namespace layer {

class DenseMLPImpl : public torch::nn::Module {
 public:
  DenseMLPImpl() = default;
  DenseMLPImpl(int64_t hidden_size,
               int64_t intermediate_size,
               bool is_gated,
               bool has_bias,
               const std::string& hidden_act,
               bool enable_result_reduction,
               const QuantArgs& quant_args,
               ProcessGroup* process_group,
               const torch::TensorOptions& options,
               // Optional prefix for the underlying LoRA wrapper proj names.
               // "" (default) = regular per-layer MLP; the wrappers register
               // as "gate_up_proj" / "down_proj" and read deltas keyed by
               // "gate_proj" / "up_proj" / "down_proj". Pass e.g.
               // "shared_expert." so this DenseMLP instance (used as the
               // MoE shared expert in Qwen3.5-122B / DeepSeek-V2/V3 /
               // GLM4-MoE) reads deltas keyed by
               // "shared_expert.gate_proj" / "shared_expert.up_proj" /
               // "shared_expert.down_proj" and does not collide with the
               // regular per-layer MLP entries in per_proj_device_pool_.
               const std::string& lora_proj_prefix = "");

  torch::Tensor forward(const torch::Tensor& hidden_states);

  void load_state_dict(const StateDict& state_dict);
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& gate_up_name,
                       const std::string& down_name);

  // Get FP8 input scale from gate_up_proj for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_fp8_input_scale() const;

 private:
  bool is_gated_;
  int64_t intermediate_size_;
  ProcessGroup* process_group_;
  LoRAColumnParallelLinear gate_up_proj_{nullptr};
  LoRARowParallelLinear down_proj_{nullptr};
  Activation act_{nullptr};
  bool is_smoothquant_;
  std::string hidden_act_;
};
TORCH_MODULE(DenseMLP);

}  // namespace layer
}  // namespace xllm