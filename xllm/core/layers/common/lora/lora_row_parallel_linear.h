/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE
==============================================================================*/

// Composition-based LoRA wrapper around RowParallelLinear.
//
// Used for o_proj (attention output projection) and down_proj (MLP output).
// Both are Row-parallel (input already TP-sharded, output reduced).
//
// LoRA math (single proj):
//   y = base(x)
//   delta = B @ A @ x           A: [r, in]     B: [out, r]
//   return y + delta * scaling
//
// TP handling: A is input-sharded (matches base's row-shard), B is replicated.
// Since input is already partial along in-features, A@x gives a partial
// intermediate; the final all-reduce on base output already covers our delta
// too if we add before reduce — but for simplicity we add after reduce and
// keep A/B replicated. First-order correct for r << in_features; TP-sharded
// A/B is a P1 optimization.

#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRARowParallelLinearImpl : public torch::nn::Module {
 public:
  LoRARowParallelLinearImpl() = default;

  // Signature mirrors RowParallelLinearImpl exactly (drop-in replace).
  // Extra parameters:
  //   proj_name: which PEFT target module this wrapper serves. Passed to
  //     LoRARuntime::get_per_proj_delta at forward time. Must be one of
  //     "o_proj" or "down_proj" for Qwen family.
  LoRARowParallelLinearImpl(int64_t in_features,
                            int64_t out_features,
                            bool bias,
                            bool input_is_parallelized,
                            bool enable_result_reduction,
                            const QuantArgs& quant_args,
                            ProcessGroup* process_group,
                            const torch::TensorOptions& options,
                            const std::string& proj_name);

  torch::Tensor forward(torch::Tensor input);

  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const {
    stream << name() << " (LoRA-wrapped/" << proj_name_
           << ")  base_weight=" << base_->weight().sizes();
  }

 private:
  RowParallelLinear base_{nullptr};
  std::string proj_name_;
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;
};
TORCH_MODULE(LoRARowParallelLinear);

}  // namespace layer
}  // namespace xllm
