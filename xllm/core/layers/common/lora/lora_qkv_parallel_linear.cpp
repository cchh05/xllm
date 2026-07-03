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
    const QuantArgs& quant_args)
    : hidden_size_(hidden_size) {
  // Cache TP topology for LoRA weight sharding logic.
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_world_size_ = parallel_args.tp_group_->world_size();

  // Match base's per-partition sizing exactly:
  //   q_size_local  = num_heads * head_size          (already tp-partitioned)
  //   kv_size_local = num_kv_heads * head_size       (with kv_head_replicas)
  q_size_local_ = num_heads * head_size;
  kv_size_local_ = num_kv_heads * head_size;
  out_size_local_ = q_size_local_ + 2 * kv_size_local_;

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
  // Fast path: no active adapter, just return base output.
  auto y = base_->forward(input);
  if (!lora_active_ || lora_rank_ == 0) {
    return y;
  }

  // LoRA delta path.
  //   input : [..., hidden_size]
  //   lora_a: [rank, hidden]        replicated
  //   lora_b: [out_local, rank]     col-shard (already sliced to this rank)
  //
  // We compute delta with plain matmuls; naive kernel, correct-first.
  // shrink: input @ A^T -> [..., rank]
  auto lora_intermediate = torch::matmul(input, lora_a_.transpose(0, 1));
  // expand: intermediate @ B^T -> [..., out_local]
  auto delta = torch::matmul(lora_intermediate, lora_b_.transpose(0, 1));
  // Cast delta to base output dtype (base may be FP8/int8 quantized while
  // LoRA is fp16/bf16); scale as alpha/rank (may be premultiplied into B
  // for real adapter loads, in which case scaling=1.0 is passed).
  return y + (delta * lora_scaling_).to(y.dtype());
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
