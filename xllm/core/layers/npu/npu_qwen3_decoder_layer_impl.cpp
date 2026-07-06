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

#include "npu_qwen3_decoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <atomic>
#include <map>
#include <sstream>

#include "common/global_flags.h"
#include "common/rec_model_utils.h"
#include "core/framework/lora/lora_config.h"

// #include "attn_mask.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 56;

void NpuQwen3DecoderLayerImpl::param_from_args(
    atb_speed::qwen::QwenLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool isPrefill) {
  param.isFA = false;
  // Enable SwiGLU activation, as used in LLaMA
  param.enableSwiGLU = true;
  // Enable LCOC for prefill phase, similar to LLaMA
  // NOTE: Currently, single-process startup requires setting enableLcoc to
  // false, which leads to performance degradation. param.enableLcoc = false;
  // //isPrefill
  param.enableLcoc = false;
  param.rmsnormQKNorm = true;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSplitFuse = FLAGS_enable_chunked_prefill && isPrefill;
  // Path B Week 3: enable atb native LoRA channel. Even without any real
  // adapter loaded, this attaches 15 additional runtime tensor slots
  // (in_seq_len_cum_sum + 8 attn A/B + 6 mlp A/B) which we bind to
  // placeholder tensors in build_node_variant_pack. AddQNormLinearNode /
  // AddKNormLinearNode / AddVNormLinearNode etc. add optional lora
  // sub-nodes when supportLora=true (=enableLora). GMM = false for now:
  // MVP is single adapter per batch; multi-adapter mixed batching (which
  // needs GroupMatmul) is P0-C1 stage 3.
  param.enableLora = FLAGS_enable_lora;
  param.loraEnableGMM = false;
  param.enableXattention = is_rec_multi_round_mode();

  // Path B: K (slot 1) and V (slot 2) linears actually execute in NoPack,
  // so their transposeType must be NOT_TRANSPOSE (matching Q). Prior INVALID
  // was a placeholder valid only for the Pack path where K/V slots were unused.
  param.linearTransposeType = {static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE),
                               static_cast<int>(TransposeType::INVALID),
                               static_cast<int>(TransposeType::NOT_TRANSPOSE)};
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / parallel_args.world_size();
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.backend = FLAGS_communication_backend;
  param.enableLogN = false;
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend};
  param.linearHasBias = {0, 0, 0, 0};
  param.useQKNorm = true;

  param.numHiddenLayers = args.n_layers();
  param.enableIntraLayerAddNorm = true;
  param.enableInterLayerAddNorm = false;
  param.enablePreFetchWeight = FLAGS_enable_prefetch_weight;
  param.enableAclGraphPagedAttention = FLAGS_enable_graph && !isPrefill;
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);

  if (isPrefill) {
    param.enableAclnnRmsNorm =
        param.enableIntraLayerAddNorm && quantize_type_.empty()
            ? false
            : quantize_type_.empty();
    // for prefix cache without chunked prefill.
    if (FLAGS_enable_prefix_cache && !FLAGS_enable_chunked_prefill &&
        FLAGS_block_size != 128) {
      LOG(ERROR) << "try to enable prefix cache without chunked prefill but "
                    "failed, because the block_size is required to be 128.";
    }
    param.isPrefixCacheWithoutChunk = FLAGS_enable_prefix_cache &&
                                      !FLAGS_enable_chunked_prefill &&
                                      FLAGS_block_size == 128;
  }
}

void NpuQwen3DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::QwenLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {parallel_args.rank(),
                              parallel_args.world_size(),
                              FLAGS_communication_backend,
                              FLAGS_rank_tablefile,
                              nullptr,
                              ""};
}

void NpuQwen3DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::qwen::QwenLayerParam& param) {
  if (quantize_type_.empty()) {
    // Path B: K (slot 1) and V (slot 2) linears actually execute in
    // NoPack, so their descs must reflect BF16 -- not INVALID -- to
    // route them to GetLinearQuantType's NO_QUANT branch. Slot 5
    // (mlp gate/up second half) stays INVALID because the current
    // MLP fusion still packs gate+up.
    param.linearDescs = {static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::BFLOAT16),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    // Path B: force atb_speed NoPack branch. CheckPack returns false when
    // packQuantType is not in the pack allowlist (ALL_FP/ALL_W8A8/...);
    // MIX_W8A8 is the smallest lie that keeps BF16 quant math untouched
    // (the actual dtype comes from linearDescs above), while forcing the
    // three-linear NoPack QKV path so LoRA slots line up.
    // Path B: only attention QKV needs NoPack (MIX_W8A8). MLP gate+up are
    // still fused into IN_MLP_W2_WEIGHT by the loader, so MLP must stay in
    // its Pack path -- keep [1] = PACK_QUANT_UNDEFINED so CheckPack picks up
    // the linearDescs single-BF16 rule and returns true for MLP.
    param.packQuantType = {static_cast<int>(PackType::MIX_W8A8),
                           static_cast<int>(PackType::PACK_QUANT_UNDEFINED)};
    param.linearQuantType = {static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};
  } else {
    param.linearDescs = {static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::W8A8),
                         static_cast<int>(LinearTypeV2::INVALID),
                         static_cast<int>(LinearTypeV2::BFLOAT16)};
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8),
                           static_cast<int>(PackType::ALL_W8A8)};
    param.linearQuantType = {static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP)};
  }
}

NpuQwen3DecoderLayerImpl::NpuQwen3DecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  prefill_tensor_storage_.resize(4);
  decode_tensor_storage_.resize(4);
  prefill_vector_storage_.resize(1);
  decode_vector_storage_.resize(1);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);

  // Path B Week 3: pre-allocate correctly-shaped zero tensors for LoRA
  // A/B slots. When enableLora=true and no real adapter loaded, these
  // are all zero -> lora_delta = 0 -> baseline preserved.
  {
    const int64_t rank = FLAGS_max_lora_rank > 0 ? FLAGS_max_lora_rank : 32;
    const int64_t hidden = model_args.hidden_size();
    const int64_t inter = model_args.intermediate_size();
    // Compute kv_hidden per rank based on TP world_size and n_kv_heads.
    const auto kv_head_opt = model_args.n_kv_heads();
    const int64_t n_kv_heads =
        static_cast<int64_t>(kv_head_opt.value()) / parallel_args.world_size();
    const int64_t head_dim = model_args.head_dim();
    const int64_t kv_hidden = n_kv_heads * head_dim;
    LOG(ERROR) << "[V42_MARKER_LORA_CTOR] enable_lora=" << FLAGS_enable_lora
               << " kv_hidden=" << kv_hidden << " rank=" << rank
               << " hidden=" << hidden;

    auto lora_opts = torch::TensorOptions().dtype(dtype_).device(device_);
    // Path B Week 3 empirical: try PEFT-standard B layout [n, r]. Real
    // PEFT adapter (adamkarvonen taboo-ship) stores B as [out_features,
    // rank] = e.g. [4096, 32] for q_proj. atb doc says [r, n] but real
    // adapter shape suggests atb accepts PEFT storage as-is.
    // Path B Week 3: pre-alloc zero LoRA tensors matching atb doc layout.
    // A: [r, k] (transposeB=true), B: [r, n] (transposeB=false).
    // Note: forward crashes with err 8 in ElewiseOperation node[3] of
    // LinearWithLora K linear regardless of these shape choices.
    // Root cause remains under investigation.
    // atb LinearOperation reads weight.shape[1] as "k" (input dim), so B
    // must be [n, k] = [out_features, rank] = PEFT-standard storage,
    // NOT [r, n] as the xllm_atb_layers doc claims. Confirmed by Step 1
    // shape dump: activation_last_dim=32 (rank) but weight.shape[1]=1024
    // (kv_hidden) triggered "inTensor0 k = 32, inTensor1 k = 1024" err 8.
    // atb LinearOperation reads weight.shape[1] as "k" (input dim), so B
    // must be [n, k] = [out_features, rank] = PEFT-standard storage,
    // NOT [r, n] as the xllm_atb_layers doc claims. Confirmed by Step 1
    // shape dump: activation_last_dim=32 (rank) but weight.shape[1]=1024
    // (kv_hidden) triggered "inTensor0 k = 32, inTensor1 k = 1024" err 8.
    at_lora_A_qkv_ = torch::zeros({rank, hidden}, lora_opts);       // [r, k]
    at_lora_B_q_ = torch::zeros({rank, hidden}, lora_opts);         // [r, n]
    at_lora_B_kv_ = torch::zeros({rank, kv_hidden}, lora_opts);     // [r, n]
    at_lora_A_dense_ = torch::zeros({rank, hidden}, lora_opts);     // [r, k]
    at_lora_B_dense_ = torch::zeros({rank, hidden}, lora_opts);     // [r, n]
    at_lora_A_mlp_gu_ = torch::zeros({rank, hidden}, lora_opts);    // [r, k]
    at_lora_B_mlp_gu_ = torch::zeros({rank, inter}, lora_opts);     // [r, n]
    at_lora_A_mlp_down_ = torch::zeros({rank, inter}, lora_opts);   // [r, k]
    at_lora_B_mlp_down_ = torch::zeros({rank, hidden}, lora_opts);  // [r, n]

    // Path B Week 3 Step 2 test: cast LoRA weights to FRACTAL_NZ so
    // that lora_b_out has same format as base_out during ElewiseAdd.
    // Base weights are all NZ-cast in loader (qwen3_decoder_loader.cpp).
    at_lora_A_qkv_ =
        at_npu::native::npu_format_cast(at_lora_A_qkv_, ACL_FORMAT_FRACTAL_NZ);
    at_lora_B_q_ =
        at_npu::native::npu_format_cast(at_lora_B_q_, ACL_FORMAT_FRACTAL_NZ);
    at_lora_B_kv_ =
        at_npu::native::npu_format_cast(at_lora_B_kv_, ACL_FORMAT_FRACTAL_NZ);
    at_lora_A_dense_ = at_npu::native::npu_format_cast(at_lora_A_dense_,
                                                       ACL_FORMAT_FRACTAL_NZ);
    at_lora_B_dense_ = at_npu::native::npu_format_cast(at_lora_B_dense_,
                                                       ACL_FORMAT_FRACTAL_NZ);
    at_lora_A_mlp_gu_ = at_npu::native::npu_format_cast(at_lora_A_mlp_gu_,
                                                        ACL_FORMAT_FRACTAL_NZ);
    at_lora_B_mlp_gu_ = at_npu::native::npu_format_cast(at_lora_B_mlp_gu_,
                                                        ACL_FORMAT_FRACTAL_NZ);
    at_lora_A_mlp_down_ = at_npu::native::npu_format_cast(
        at_lora_A_mlp_down_, ACL_FORMAT_FRACTAL_NZ);
    at_lora_B_mlp_down_ = at_npu::native::npu_format_cast(
        at_lora_B_mlp_down_, ACL_FORMAT_FRACTAL_NZ);

    // Path B Week 3: keep LoRA tensors in ND (default) format. FRACTAL_NZ
    // padding rewrites shape to [in/16, out/16, 16, 16], which breaks atb's
    // 2-dim shape check on the lora sub-graph InferShape (base_out is 2-dim
    // [batch, hidden] and expects lora_b_out to be exactly 2-dim). ND format
    // preserves [rank, hidden] as-reported.

    // seq_len_cum_sum: atb reads value from hostData (int64 pointer);
    // tensor itself lives on device just as shape/dtype descriptor.
    // Follow the kv_seq_lens pattern: device tensor + parallel std::vector
    // updated at forward time (avoids CPU->NPU copy in forward).
    at_seq_len_cum_sum_ = torch::ones(
        {1}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
    seq_len_cum_sum_ = atb_speed::Utils::AtTensor2Tensor(at_seq_len_cum_sum_);
    seq_len_cum_sum_vec_.assign(1, 1);
  }
  if (FLAGS_enable_manual_loader) {
    loader_ = std::make_unique<Qwen3DecoderManualLoader>(
        WEIGHT_COUNT_PER_LAYER,
        context,
        prefill_param_.enableIntraLayerAddNorm ||
            prefill_param_.enableInterLayerAddNorm);
  } else {
    loader_ = std::make_unique<Qwen3DecoderLoader>(
        WEIGHT_COUNT_PER_LAYER,
        context,
        prefill_param_.enableIntraLayerAddNorm ||
            prefill_param_.enableInterLayerAddNorm);
  }
}

int64_t NpuQwen3DecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "qwen3_decoder_layer";
  model_name_ = "qwen3";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t NpuQwen3DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::QwenLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::QwenDecoderLayer decoder_layer(param);
  decoder_layer.BuildGraph(&operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3DecoderLayerImpl::forward(torch::Tensor& x,
                                                torch::Tensor& cos_pos,
                                                torch::Tensor& sin_pos,
                                                torch::Tensor& attn_mask,
                                                KVCache& kv_cache,
                                                ModelInputParams& input_params,
                                                aclrtEvent* event,
                                                std::atomic<bool>* event_flag,
                                                int node_id) {
  atb::Status st;
  if (!input_params.batch_forward_type.is_decode()) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/true,
                            node_id,
                            /*use_graph_decode_input=*/false);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    const bool use_graph_decode_input =
        FLAGS_enable_graph && input_params.graph_buffer.tiling_data.defined();
    auto& decode_node =
        use_graph_decode_input ? decode_graph_node_ : decode_eager_node_;
    build_node_variant_pack(decode_node,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            /*is_prefill=*/false,
                            node_id,
                            use_graph_decode_input);
    st = execute_node(decode_node, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuQwen3DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    int node_id,
    bool use_graph_decode_input) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      input_params.kv_seq_lens_vec.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(input_params.block_tables);

  int input_idx = WEIGHT_COUNT_PER_LAYER + 11;
  if (is_rec_multi_round_mode()) {
    const auto* llmrec = input_params.llmrec_params();

    if (is_prefill) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    } else {
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_k_caches.size());
      CHECK_LT(static_cast<size_t>(node_id), llmrec->unshared_v_caches.size());
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_k_caches[node_id]);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
          atb_speed::Utils::AtTensor2Tensor(llmrec->unshared_v_caches[node_id]);
    }
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) = placeholder_;
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_k_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->shared_v_caches[node_id]);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->beam_width_tensor);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(llmrec->current_round_tensor);
    input_idx = WEIGHT_COUNT_PER_LAYER + 15;
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
        atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(input_params.new_cache_slots);
  }

  if (is_prefill &&
      (FLAGS_enable_chunked_prefill || FLAGS_enable_prefix_cache)) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        input_params.q_seq_lens_vec.data();
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph_buffer.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.graph_buffer.tiling_data);
  }

  // Path B Week 3: bind LoRA runtime tensors after the default runtime
  // block. When enableLora=true, atb qwen3 layer expects 15 additional
  // slots at [input_idx..input_idx+14]:
  //   +0: in_seq_len_cum_sum      (int64 host tensor)
  //   +1..+2: in_qkv_lora_a_0/b_0 (Q proj)
  //   +3..+4: in_qkv_lora_a_1/b_1 (K proj)
  //   +5..+6: in_qkv_lora_a_2/b_2 (V proj)
  //   +7..+8: in_qkv_dense_lora_a/b (o proj)
  //   +9..+10: in_mlp_lora_a_0/b_0 (gate proj)
  //   +11..+12: in_mlp_lora_a_1/b_1 (up proj)
  //   +13..+14: in_mlp_down_lora_a/b (down proj)
  //
  // MVP first pass: bind everything to `placeholder_` (torch::zeros({1}))
  // so atb graph can build. When atb sees a placeholder shape [1] it
  // still reads through the LoRA sub-graph nodes but their contribution
  // is effectively zero (matmul(x, zeros) = 0). Semantically identical
  // to base model. Real per-adapter tensor binding follows in a later
  // commit once M2 loader emits per-proj A/B pairs and LoRARuntime
  // exposes them here.
  if (prefill_param_.enableLora) {
    // seq_len_cum_sum: single-element {n_tokens} for single-adapter mode.
    const int64_t n_tokens = x.size(0);
    seq_len_cum_sum_vec_[0] = n_tokens;
    // Regenerate atb tensor descriptors every forward. atb tensor descs
    // seem to get zeroed after graph setup, so re-materialize from the
    // at::Tensor storage each time.
    seq_len_cum_sum_ = atb_speed::Utils::AtTensor2Tensor(at_seq_len_cum_sum_);
    seq_len_cum_sum_.hostData = seq_len_cum_sum_vec_.data();
    node.variantPack.inTensors.at(input_idx++) = seq_len_cum_sum_;

    lora_A_qkv_ = atb_speed::Utils::AtTensor2Tensor(at_lora_A_qkv_);
    lora_B_q_ = atb_speed::Utils::AtTensor2Tensor(at_lora_B_q_);
    lora_B_kv_ = atb_speed::Utils::AtTensor2Tensor(at_lora_B_kv_);
    lora_A_dense_ = atb_speed::Utils::AtTensor2Tensor(at_lora_A_dense_);
    lora_B_dense_ = atb_speed::Utils::AtTensor2Tensor(at_lora_B_dense_);
    lora_A_mlp_gu_ = atb_speed::Utils::AtTensor2Tensor(at_lora_A_mlp_gu_);
    lora_B_mlp_gu_ = atb_speed::Utils::AtTensor2Tensor(at_lora_B_mlp_gu_);
    lora_A_mlp_down_ = atb_speed::Utils::AtTensor2Tensor(at_lora_A_mlp_down_);
    lora_B_mlp_down_ = atb_speed::Utils::AtTensor2Tensor(at_lora_B_mlp_down_);

    // V42_DUMP_KWEIGHT_LORA: emit only from layer 0 first forward
    // (avoid 36x noise). Compare base K weight (post-loader NZ cast)
    // vs lora_B_kv_ (post-Step2 NZ cast) desc as atb sees them.
    static std::atomic<bool> dumped{false};
    if (!dumped.exchange(true)) {
      const auto& kw = atb_weight_tensors_[10];  // IN_K_WEIGHT
      auto dumpDesc = [](const char* name, const atb::Tensor& t) {
        std::stringstream ss;
        ss << name << " format=" << (int)t.desc.format
           << " dtype=" << (int)t.desc.dtype
           << " dimNum=" << t.desc.shape.dimNum << " dims=[";
        for (uint64_t i = 0; i < t.desc.shape.dimNum; ++i) {
          if (i) ss << ",";
          ss << t.desc.shape.dims[i];
        }
        ss << "] deviceData=" << t.deviceData;
        LOG(ERROR) << "[V42_DUMP_KWEIGHT_LORA] " << ss.str();
      };
      dumpDesc("input_x (activation)", internal_tensors_);
      dumpDesc("IN_K_WEIGHT (base)", kw);
      dumpDesc("lora_A_qkv (LoRA A shared)", lora_A_qkv_);
      dumpDesc("lora_B_kv (LoRA B for K/V)", lora_B_kv_);
      dumpDesc("lora_B_q (LoRA B for Q)", lora_B_q_);
      // Also raw at:: sizes (torch view)
      LOG(ERROR) << "[V42_DUMP_KWEIGHT_LORA] at_lora_A_qkv_.sizes()="
                 << at_lora_A_qkv_.sizes()
                 << " at_lora_B_kv_.sizes()=" << at_lora_B_kv_.sizes();
    }

    // Q proj: A_qkv, B_q
    node.variantPack.inTensors.at(input_idx++) = lora_A_qkv_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_q_;
    // K proj: A_qkv, B_kv
    node.variantPack.inTensors.at(input_idx++) = lora_A_qkv_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_kv_;
    // V proj: A_qkv, B_kv
    node.variantPack.inTensors.at(input_idx++) = lora_A_qkv_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_kv_;
    // o proj (dense)
    node.variantPack.inTensors.at(input_idx++) = lora_A_dense_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_dense_;
    // MLP gate
    node.variantPack.inTensors.at(input_idx++) = lora_A_mlp_gu_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_mlp_gu_;
    // MLP up
    node.variantPack.inTensors.at(input_idx++) = lora_A_mlp_gu_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_mlp_gu_;
    // MLP down
    node.variantPack.inTensors.at(input_idx++) = lora_A_mlp_down_;
    node.variantPack.inTensors.at(input_idx++) = lora_B_mlp_down_;
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
