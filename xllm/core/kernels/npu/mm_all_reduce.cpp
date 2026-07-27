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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor mm_all_reduce_base(const torch::Tensor& input,
                                 const torch::Tensor& weight_t,
                                 const std::string& hcom_name,
                                 const std::optional<torch::Tensor>& bias) {
  // Fused MatMul + AllReduce implemented by torch_npu / CANN.
  //
  // Equivalent semantics:
  //   tmp = input @ weight_t
  //   if (bias) tmp = tmp + bias
  //   output = all_reduce(tmp, sum)
  //
  // Executed as a single kernel launch on the NPU, saving the ~1ms HCCL
  // setup latency the two-op version pays per collective. This is the
  // recovery path for the +17% single-adapter / +56% mixed-batch overhead
  // that the row-parallel LoRA all-reduce fix (a9d6ad74) introduced on
  // 30B-A3B TP=4 workloads.
  //
  // Inputs:
  //   input   [T, in_local]     — activation, sharded on in-dim across TP
  //   weight_t [in_local, out]  — weight, PRE-TRANSPOSED (caller
  //   responsibility)
  //                               so this API stays a thin wrapper. The base
  //                               RowParallelLinear stores weight as
  //                               [out, in_local], so callers should pass
  //                               weight_.t().contiguous() (v1) or a cached
  //                               transposed view (v2 follow-up).
  //   hcom_name                 — HCCL communicator name string, obtained via
  //                               ProcessGroup::get_hccl_comm_name(rank).
  //   bias                      — optional bias, added rank-0-only by
  //   convention
  //                               (caller supplies std::nullopt on ranks > 0).
  //
  // The underlying op supports antiquant / dequant scales for quantized paths;
  // this wrapper exposes only the bf16/fp16 unquantized signature for now.
  // Smoothquant and FP8 paths keep the legacy matmul + reduce fallback.
  return at_npu::native::custom_ops::npu_mm_all_reduce_base(
      input,
      weight_t,
      c10::string_view(hcom_name),
      c10::string_view("sum"),
      bias);
}

}  // namespace xllm::kernel::npu
