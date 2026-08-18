/* Copyright 2025-2026 The xLLM Authors.

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

#include "npu_qwen3_moe_decoder_layer_impl.h"

#include <gflags/gflags.h>

#include <unordered_set>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/lora/lora_config.h"  // Option D patch #1: FLAGS_enable_lora
#include "core/framework/lora/lora_runtime.h"  // D-A-1: get_per_proj_delta for attn LoRA
namespace xllm {
namespace layer {

// D-A-1 attn-only LoRA patch #2: reserve 9 additional slot (1 lora_common
// + 8 lora_attn) after 55 base weights. Loop uses node.operation->GetInputNum()
// dynamic gate so LoRA-off (55 slots) and LoRA-on (64 slots) both work without
// vector out_of_range (avoids c3 patch #2 unconditional 越界 trap).
static constexpr uint64_t BASE_WEIGHT_COUNT = 55;
// Day 2-C fix (R3-alt3 experts LoRA down-only path Y): attn 9 slot + experts 2
// slot = 11. Day 1-B originally had 6 experts slot (gate/up/down A+B), but
// atb_speed side only wires down (Day 2-C, path Y first-cut). Match count to
// avoid outer scope GetTensorIdx returning UINT32_MAX for undeclared gate/up
// names. Path X (Day 6-7) will re-add gate/up + LoRARuntime fusion (torch.cat
// A_gate|A_up).
static constexpr uint64_t LORA_SLOT_COUNT = 11;
enum LoraSlotOffset : int {
  LORA_IN_SEQ_LEN_CUM_SUM = 0,
  LORA_QKV_A_0 = 1,
  LORA_QKV_B_0 = 2,
  LORA_QKV_A_1 = 3,
  LORA_QKV_B_1 = 4,
  LORA_QKV_A_2 = 5,
  LORA_QKV_B_2 = 6,
  LORA_QKV_DENSE_A = 7,
  LORA_QKV_DENSE_B = 8,
  // Down-only experts LoRA (path Y first-cut, gate/up 是 path X Day 6-7)
  LORA_MOE_DOWN_A = 9,   // A_down [E_per_rank, in_local, r]
  LORA_MOE_DOWN_B = 10,  // B_down [E_per_rank, r, hidden]
};
// Path γ v2 layout fix: atb qwen3_moe candidate order is base(55) + runtime(16)
// + lora(11). WEIGHT_COUNT_PER_LAYER = BASE_WEIGHT_COUNT only, so loader/resize
// allocate 55 base slots. Runtime tensor block starts at BASE_WEIGHT_COUNT
// (index 55), lora block starts at BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT
// (index 71).
static constexpr uint64_t RUNTIME_TENSOR_COUNT = 16;
static const uint64_t WEIGHT_COUNT_PER_LAYER = BASE_WEIGHT_COUNT;

NpuQwen3MoeDecoderLayerImpl::NpuQwen3MoeDecoderLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  // int ep_rank = prefill_param_.rank /  ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  loader_ = std::make_unique<Qwen3MoeDecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
  initialize_tensors(options);
}

void NpuQwen3MoeDecoderLayerImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  loader_->resize_experts_weights(num_experts_per_partition_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  quant_add_norm_scaling_ =
      torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  quant_add_norm_offset_ =
      torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
}

void NpuQwen3MoeDecoderLayerImpl::param_from_args(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void NpuQwen3MoeDecoderLayerImpl::initialize_basic_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.isPrefill = is_prefill;

  // prefill only feature
  param.enableLcoc = is_prefill;  // false;
  param.enableSplitFuse =
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache()) &&
      is_prefill;

  // decode only feature
  param.enableAclGraphPagedAttention =
      ::xllm::ExecutionConfig::get_instance().enable_graph() && !is_prefill;
  param.enableInitRoutingV3 = !is_prefill;

  // Can be applied to prefill, but has not been tested yet
  param.enableFusedReducesumDiv = !is_prefill;
  param.enableAclnnExternelAddRmsNorm =
      ::xllm::KernelConfig::get_instance().enable_intralayer_addnorm() &&
      !is_prefill;
  param.enableAclnnAddRmsNorm = !is_prefill;

  param.swigluBackend = atb_speed::common::OpBackend::ACLNN;
  param.mlpLinearTransposeType = {static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID)};
  if (quantize_type_.empty()) {
    param.moeLinearTransposeType = {static_cast<int>(TransposeType::TRANSPOSE),
                                    static_cast<int>(TransposeType::TRANSPOSE),
                                    static_cast<int>(TransposeType::INVALID),
                                    static_cast<int>(TransposeType::TRANSPOSE)};
  } else {
    param.moeLinearTransposeType = {
        static_cast<int>(TransposeType::TRANSPOSE),
        static_cast<int>(TransposeType::NOT_TRANSPOSE),
        static_cast<int>(TransposeType::INVALID),
        static_cast<int>(TransposeType::TRANSPOSE)};
  }
  param.normEps = args.rms_norm_eps();
  param.rank = parallel_args.rank();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  // param.rankTableFile =
  // ::xllm::EPLBConfig::get_instance().rank_tablefile();

  param.layerId = layer_id_;
  param.numHiddenLayers = 0;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;                    // MTP
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  param.useQKNorm = true;
  param.rmsnormQKNorm = true;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = std::max(
      1, static_cast<int>(optionalValue.value()) / parallel_args.world_size());
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.attnLinearTransposeType = {static_cast<int>(TransposeType::TRANSPOSE),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::TRANSPOSE),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::INVALID)};
  param.worldSize = parallel_args.world_size();

  if (is_prefill) {
    param.enableAclnnRmsNorm = quantize_type_.empty();
  }
}

void NpuQwen3MoeDecoderLayerImpl::initialize_attention_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.enableFA3 = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  // Option D patch #1: opt-in atb_speed MoE decoder LoRA branch (if any).
  // NOTE: No patch #2 (WEIGHT_COUNT + N) here — c3 patch #2 unconditional +15
  // caused vector out_of_range when enableLora=false; Option D test is
  // "does atb_speed MoE side accept enableLora=true and still build graph?"
  param.enableLora = FLAGS_enable_lora;  // legacy compat
  // R3-alt3 Y-alt2: independent attn/experts LoRA gate based on pool state.
  // If no attn adapter installed, keep enableAttnLora=false so the attn
  // graph does not consume undefined LoRA A/B slots (Path 2 dead-end class).
  if (FLAGS_enable_lora) {
    auto& rt = LoRARuntime::instance();
    param.enableAttnLora = rt.has_any_attn_adapter();
    param.enableExpertsLora = rt.has_any_experts_adapter();
  }
  param.processLogits = "normalization";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree = 1;
  param.enableFusedRouting = 1;
  param.numOfExperts = args.num_experts();
  param.maskStartIdx = 0;
  param.routingMethod = "integratedSoftmaxTopK";
  param.quantGroupSize = 0;

  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableCVOverlap = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend(),
      ::xllm::EPLBConfig::get_instance().rank_tablefile(),
      nullptr,
      ""};

  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};

    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP)};
  } else {
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};
    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT)};
  }
}

void NpuQwen3MoeDecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }
  init_layer();
}

int64_t NpuQwen3MoeDecoderLayerImpl::init_layer() {
  name_ = "qwen3_moe_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "Qwen3_Moe";

  // 2-graph refactor: force base variant params to have NO LoRA to guarantee
  // base variant atb graph contains zero LoRA nodes (Path γ v2 verified path).
  // This overrides Y-alt2 conditional `has_any_*_adapter()` which may return
  // true post-preload — but init_layer is called from model ctor BEFORE
  // preload (see model ctor sequence: decoder ctor → init_layer → preload).
  // Explicit override is defense-in-depth.
  prefill_param_.enableLora = false;
  prefill_param_.enableAttnLora = false;
  prefill_param_.enableExpertsLora = false;
  prefill_param_.loraEnableGMM = false;
  decode_graph_param_.enableLora = false;
  decode_graph_param_.enableAttnLora = false;
  decode_graph_param_.enableExpertsLora = false;
  decode_graph_param_.loraEnableGMM = false;
  decode_eager_param_.enableLora = false;
  decode_eager_param_.enableAttnLora = false;
  decode_eager_param_.enableExpertsLora = false;
  decode_eager_param_.loraEnableGMM = false;

  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  // 2-graph refactor: build LoRA variants when --enable_lora on.
  // Same param as base except LoRA flags forced true — atb builds graph with
  // LoRA GMM_A + GMM_B + Add nodes. Independent op tree from base variant.
  // NOTE M3c: enableAttnLora forced false because moe_zero/moe_strong are
  // experts-only. If attn LoRA needed later (mixed adapter), rebuild variant
  // with enableAttnLora=true (may need 4 variants total per phase).
  if (FLAGS_enable_lora) {
    prefill_param_lora_ = prefill_param_;
    prefill_param_lora_.enableLora = true;
    prefill_param_lora_.enableAttnLora = false;
    prefill_param_lora_.enableExpertsLora = true;
    prefill_param_lora_.loraEnableGMM = true;

    decode_graph_param_lora_ = decode_graph_param_;
    decode_graph_param_lora_.enableLora = true;
    decode_graph_param_lora_.enableAttnLora = false;
    decode_graph_param_lora_.enableExpertsLora = true;
    decode_graph_param_lora_.loraEnableGMM = true;

    decode_eager_param_lora_ = decode_eager_param_;
    decode_eager_param_lora_.enableLora = true;
    decode_eager_param_lora_.enableAttnLora = false;
    decode_eager_param_lora_.enableExpertsLora = true;
    decode_eager_param_lora_.loraEnableGMM = true;

    CHECK_OPERATION_STATUS_RETURN(
        init_node(prefill_node_lora_, prefill_param_lora_));
    CHECK_OPERATION_STATUS_RETURN(
        init_node(decode_graph_node_lora_, decode_graph_param_lora_));
    CHECK_OPERATION_STATUS_RETURN(
        init_node(decode_eager_node_lora_, decode_eager_param_lora_));
    lora_variants_built_ = true;
    LOG(INFO) << "[2graph] layer=" << layer_id_
              << " built base + LoRA variants (6 nodes total)";
  } else {
    LOG(INFO) << "[2graph] layer=" << layer_id_
              << " built base variants only (3 nodes, --enable_lora off)";
  }

  return atb::NO_ERROR;
}

int64_t NpuQwen3MoeDecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  if (layer_id_ == 0) {
    LOG(INFO) << "[atb-debug-30b] layer=" << layer_id_
              << " numOfExperts=" << param.numOfExperts
              << " numOfDeviceExperts=" << param.numOfDeviceExperts
              << " deviceExpert.size=" << param.deviceExpert.size()
              << " hasMoe=" << param.hasMoe
              << " hasSharedExpert=" << param.hasSharedExpert
              << " enableAclnnExternelAddRmsNorm="
              << param.enableAclnnExternelAddRmsNorm
              << " enableAclnnAddRmsNorm=" << param.enableAclnnAddRmsNorm
              << " packQuantType.size=" << param.packQuantType.size();
    LOG(INFO) << "[atb-debug-30b-tp] rank=" << param.tensorParallelInfo.rank
              << " worldSize=" << param.tensorParallelInfo.worldSize
              << " commDomain=" << param.tensorParallelInfo.commDomain
              << " hcommInfo_null="
              << (param.tensorParallelInfo.hcommInfo == nullptr)
              << " mapping.isInitialized=" << param.mapping.isInitialized_;
  }
  atb_speed::qwen::MoeDecoderLayer(param, &operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());
  size_t inTensorId = 1;

  // D-A-1 dynamic gate: LoRA-off atb_speed needs 55 slot, LoRA-on needs 64.
  // Use GetInputNum() so both scenarios work without vector越界 (c3 patch #2
  // trap).
  const size_t actual_input_num = node.operation->GetInputNum();
  const size_t weight_loop_end =
      std::min(actual_input_num, static_cast<size_t>(WEIGHT_COUNT_PER_LAYER));
  for (size_t weightTensorId = 0; weightTensorId < weight_loop_end;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(node.outTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3MoeDecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  atb::Status st;

  // 2-graph M2a dispatch: select base_ vs lora_ variant based on adapter_id.
  // - Base curl (adapter_id==0): base variant, atb graph has NO LoRA nodes
  // - LoRA curl (adapter_id!=0): lora variant, atb graph has LoRA
  // GMM_A+GMM_B+Add Constraint: batch must be homogeneous (all base or all same
  // adapter).
  //             Scheduler affinity gate enforces this at K=1.
  const bool use_lora_variant = FLAGS_enable_lora && lora_variants_built_ &&
                                !input_params.adapter_ids.empty() &&
                                input_params.adapter_ids[0] != 0;
  if (layer_id_ == 0) {
    LOG(INFO) << "[2graph-dispatch] use_lora_variant=" << use_lora_variant
              << " adapter_ids_size=" << input_params.adapter_ids.size()
              << " adapter_id[0]="
              << (input_params.adapter_ids.empty()
                      ? 0
                      : input_params.adapter_ids[0])
              << " lora_variants_built=" << lora_variants_built_;
  }

  if (!input_params.meta.batch_forward_type.is_decode()) {
    auto& prefill_target =
        use_lora_variant ? prefill_node_lora_ : prefill_node_;
    build_node_variant_pack(prefill_target,
                            x,
                            residual,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true,
                            false);
    st = execute_node(prefill_target, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute prefill layer fail, error code: " << st;
  } else {
    const bool use_graph_decode_input =
        ::xllm::ExecutionConfig::get_instance().enable_graph() &&
        input_params.graph.tiling_data.defined();
    atb_speed::Model::Node* decode_target;
    if (use_lora_variant) {
      decode_target = use_graph_decode_input ? &decode_graph_node_lora_
                                             : &decode_eager_node_lora_;
    } else {
      decode_target =
          use_graph_decode_input ? &decode_graph_node_ : &decode_eager_node_;
    }
    build_node_variant_pack(*decode_target,
                            x,
                            residual,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false,
                            use_graph_decode_input);
    st = execute_node(*decode_target, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void NpuQwen3MoeDecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill,
    bool use_graph_decode_input) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.parallel.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(input_params.expert.expert_array);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
  }
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.block_tables);
  }
  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.new_cache_slots);
  }

  input_idx = WEIGHT_COUNT_PER_LAYER + 16;
  if (is_prefill &&
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        const_cast<int32_t*>(input_params.attention.host.q_seq_lens.data());
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
  }

  if (input_params.meta.batch_forward_type.is_decode() &&
      ::xllm::KernelConfig::get_instance().enable_intralayer_addnorm() &&
      residual.has_value()) {
    // input
    auto& residual_tensor = residual.value();
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(residual_tensor);
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(quant_add_norm_scaling_);
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(quant_add_norm_offset_);

    // output
    auto residual_tensor_ = atb_speed::Utils::AtTensor2Tensor(residual_tensor);
    node.variantPack.outTensors.at(1) = residual_tensor_;
  }

  // D-A-1 A2-a: unified LoRA slot fill (base + adapter case).
  // atb_speed supportLora=true graph 需要**全部 9 slot 有真 shape tensor**,
  // 否则 setup_layer_node RmsNorm invalid dim=1 (error code 9).
  // Base curl (no adapter): fill zero A/B with shape from pool reference
  // (delta=0 no-op). Adapter curl: fill real A/B from get_per_proj_delta.
  // R3-alt3 Y-alt2 offset fix: gate attn fill on has_any_attn_adapter() —
  // when no attn adapter installed, atb SparseMoe has no attn LoRA slots
  // (only experts slots at offset 0-1), so writing to LORA_QKV_A_0 (slot 1)
  // would collide with LORA_MOE_DOWN_B, corrupting experts LoRA delta.
  auto& rt_lora = LoRARuntime::instance();

  // 2-graph M3a: fill slot 71 (lora_common = in_seq_len_cum_sum) whenever
  // Node is a LoRA variant (GetInputNum > 71). Unconditional wrt adapter type
  // because atb LoRA variant graph always has this slot when any lora block
  // (attn or experts) is on.
  if (FLAGS_enable_lora && node.operation->GetInputNum() >
                               BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT) {
    const int32_t off_m3a = BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT;
    node.variantPack.inTensors.at(off_m3a + LORA_IN_SEQ_LEN_CUM_SUM) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_cu_seq_lens);
    node.variantPack.inTensors.at(off_m3a + LORA_IN_SEQ_LEN_CUM_SUM).hostData =
        const_cast<int32_t*>(input_params.attention.host.q_cu_seq_lens.data());
    if (layer_id_ == 0) {
      LOG(INFO) << "[2graph-m3a-fill71] cum_sum filled, GetInputNum="
                << node.operation->GetInputNum();
    }

    // 2-graph M3c: fill slot 72-73 (experts_lora_down A/B) with real moe_delta.
    // NOTE: LoRA variant param has enableAttnLora=false, so atb candidate skips
    // lora_attn slots (72-79). Experts_lora_down slots are at 72/73 (right
    // after lora_common at 71). Offset within LoRA block: A_off=1, B_off=2.
    // Base curl (adapter_id==0) fallback to id=1 (first preloaded).
    if (rt_lora.has_any_experts_adapter()) {
      const bool is_base_curl_m3c =
          input_params.adapter_ids.empty() || input_params.adapter_ids[0] == 0;
      const uint64_t adapter_id_m3c =
          is_base_curl_m3c ? 1 : input_params.adapter_ids[0];
      auto& rt_m3c = LoRARuntime::instance();
      const auto* moe_delta =
          rt_m3c.get_moe_expert_delta(adapter_id_m3c, layer_id_);

      // Non-invasive diag (Action A): no .item()/.sum() (which force NPU->CPU
      // sync + may break stream state in forward path). Only log
      // ptr/shape/dtype.
      if (layer_id_ == 0) {
        LOG(ERROR) << "[m3-noninvasive-diag] input_adapter_id_0="
                   << (input_params.adapter_ids.empty()
                           ? 0
                           : input_params.adapter_ids[0])
                   << " is_base_curl=" << is_base_curl_m3c
                   << " lookup_adapter_id=" << adapter_id_m3c << " A_down_ptr="
                   << (moe_delta && moe_delta->A_down.defined()
                           ? moe_delta->A_down.data_ptr()
                           : nullptr)
                   << " A_down_shape="
                   << (moe_delta && moe_delta->A_down.defined()
                           ? moe_delta->A_down.sizes()
                           : c10::IntArrayRef())
                   << " A_down_dtype="
                   << (moe_delta && moe_delta->A_down.defined()
                           ? moe_delta->A_down.dtype()
                           : caffe2::TypeMeta());
      }

      if (moe_delta && moe_delta->A_down.defined() &&
          moe_delta->B_down.defined()) {
        // Slot 72 (index 1 within LoRA block, no attn slots)
        node.variantPack.inTensors.at(off_m3a + 1) =
            atb_speed::Utils::AtTensor2Tensor(moe_delta->A_down);
        // Slot 73 (index 2 within LoRA block)
        node.variantPack.inTensors.at(off_m3a + 2) =
            atb_speed::Utils::AtTensor2Tensor(moe_delta->B_down);
        if (layer_id_ == 0) {
          LOG(INFO) << "[2graph-m3c-fill72-73] adapter_id=" << adapter_id_m3c
                    << " is_base_curl=" << is_base_curl_m3c
                    << " A_down.shape=" << moe_delta->A_down.sizes()
                    << " B_down.shape=" << moe_delta->B_down.sizes();
        }
      }
    }
  }

  if (FLAGS_enable_lora && rt_lora.has_any_attn_adapter() &&
      node.operation->GetInputNum() >
          BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT) {
    const int32_t off = BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT;

    // 1. in_seq_len_cum_sum (unconditional, batch routing metadata not
    // LoRA-specific)
    node.variantPack.inTensors.at(off + LORA_IN_SEQ_LEN_CUM_SUM) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_cu_seq_lens);
    node.variantPack.inTensors.at(off + LORA_IN_SEQ_LEN_CUM_SUM).hostData =
        const_cast<int32_t*>(input_params.attention.host.q_cu_seq_lens.data());

    // 2. Determine A/B source: real adapter delta (if adapter_id != 0) or
    // zero-fill
    auto& rt = LoRARuntime::instance();
    const LoRARuntime::ProjDelta *q_delta = nullptr, *k_delta = nullptr,
                                 *v_delta = nullptr, *o_delta = nullptr;
    bool use_real_lora = false;
    uint64_t adapter_id = 0;

    if (!input_params.adapter_ids.empty() && input_params.adapter_ids[0] != 0) {
      adapter_id = input_params.adapter_ids[0];
      for (size_t s = 1; s < input_params.adapter_ids.size(); ++s) {
        CHECK_EQ(input_params.adapter_ids[s], adapter_id)
            << "D-A-1 attn LoRA supports single-adapter batch only";
      }
      q_delta = rt.get_per_proj_delta(adapter_id, layer_id_, "q_proj");
      k_delta = rt.get_per_proj_delta(adapter_id, layer_id_, "k_proj");
      v_delta = rt.get_per_proj_delta(adapter_id, layer_id_, "v_proj");
      o_delta = rt.get_per_proj_delta(adapter_id, layer_id_, "o_proj");
      if (q_delta && k_delta && v_delta && o_delta) {
        use_real_lora = true;
      }
    }

    if (use_real_lora) {
      // 3a. Real LoRA fill
      if (layer_id_ == 0) {
        LOG(INFO) << "[atb-debug-lora-shape] REAL adapter=" << adapter_id
                  << " q_A=" << q_delta->A.sizes()
                  << " q_B=" << q_delta->B.sizes()
                  << " k_A=" << k_delta->A.sizes()
                  << " k_B=" << k_delta->B.sizes()
                  << " v_A=" << v_delta->A.sizes()
                  << " v_B=" << v_delta->B.sizes()
                  << " o_A=" << o_delta->A.sizes()
                  << " o_B=" << o_delta->B.sizes()
                  << " scaling=" << q_delta->scaling << " r=" << q_delta->r;
      }
      node.variantPack.inTensors.at(off + LORA_QKV_A_0) =
          atb_speed::Utils::AtTensor2Tensor(q_delta->A);
      node.variantPack.inTensors.at(off + LORA_QKV_B_0) =
          atb_speed::Utils::AtTensor2Tensor(q_delta->B);
      node.variantPack.inTensors.at(off + LORA_QKV_A_1) =
          atb_speed::Utils::AtTensor2Tensor(k_delta->A);
      node.variantPack.inTensors.at(off + LORA_QKV_B_1) =
          atb_speed::Utils::AtTensor2Tensor(k_delta->B);
      node.variantPack.inTensors.at(off + LORA_QKV_A_2) =
          atb_speed::Utils::AtTensor2Tensor(v_delta->A);
      node.variantPack.inTensors.at(off + LORA_QKV_B_2) =
          atb_speed::Utils::AtTensor2Tensor(v_delta->B);
      node.variantPack.inTensors.at(off + LORA_QKV_DENSE_A) =
          atb_speed::Utils::AtTensor2Tensor(o_delta->A);
      node.variantPack.inTensors.at(off + LORA_QKV_DENSE_B) =
          atb_speed::Utils::AtTensor2Tensor(o_delta->B);
    } else {
      // 3b. Zero-fill (base curl or partial coverage): shape from pool
      // reference. Try adapter_id=1 (attnctrl was preloaded first) as shape
      // reference.
      const auto* ref_delta = rt.get_per_proj_delta(1, layer_id_, "q_proj");
      if (ref_delta == nullptr) {
        LOG_IF(WARNING, layer_id_ == 0)
            << "[atb-debug-lora-shape] NO REFERENCE adapter for zero-fill, "
            << "adapter_ids_size=" << input_params.adapter_ids.size()
            << " adapter_id[0]="
            << (input_params.adapter_ids.empty() ? 0
                                                 : input_params.adapter_ids[0])
            << " - atb_speed will see [1] dummy A/B, expect crash";
      } else {
        auto A_shape = ref_delta->A.sizes();
        auto B_shape = ref_delta->B.sizes();
        auto dtype = ref_delta->A.dtype();
        auto device = ref_delta->A.device();
        auto zero_A = torch::zeros(
            A_shape, torch::TensorOptions().dtype(dtype).device(device));
        auto zero_B = torch::zeros(
            B_shape, torch::TensorOptions().dtype(dtype).device(device));
        if (layer_id_ == 0) {
          LOG(INFO) << "[atb-debug-lora-shape] ZERO fill (base/partial) "
                       "adapter_ids_size="
                    << input_params.adapter_ids.size() << " ref_A=" << A_shape
                    << " ref_B=" << B_shape << " dtype=" << dtype;
        }
        node.variantPack.inTensors.at(off + LORA_QKV_A_0) =
            atb_speed::Utils::AtTensor2Tensor(zero_A);
        node.variantPack.inTensors.at(off + LORA_QKV_B_0) =
            atb_speed::Utils::AtTensor2Tensor(zero_B);
        node.variantPack.inTensors.at(off + LORA_QKV_A_1) =
            atb_speed::Utils::AtTensor2Tensor(zero_A);
        node.variantPack.inTensors.at(off + LORA_QKV_B_1) =
            atb_speed::Utils::AtTensor2Tensor(zero_B);
        node.variantPack.inTensors.at(off + LORA_QKV_A_2) =
            atb_speed::Utils::AtTensor2Tensor(zero_A);
        node.variantPack.inTensors.at(off + LORA_QKV_B_2) =
            atb_speed::Utils::AtTensor2Tensor(zero_B);
        node.variantPack.inTensors.at(off + LORA_QKV_DENSE_A) =
            atb_speed::Utils::AtTensor2Tensor(zero_A);
        node.variantPack.inTensors.at(off + LORA_QKV_DENSE_B) =
            atb_speed::Utils::AtTensor2Tensor(zero_B);
      }
    }
  }

  // R3-alt3 Y-alt2 offset fix: experts MLP LoRA fill with dynamic offset.
  // atb SparseMoe slot layout depends on enableAttnLora + enableExpertsLora:
  //   - attn OFF + experts ON: slot 0-1 = experts down A/B (no cum_sum, no qkv)
  //   - attn ON + experts ON:  slot 0 = cum_sum, 1-8 = qkv, 9-10 = experts down
  //   A/B
  // Gate on adapter_id!=0 + GetInputNum() > BASE_WEIGHT_COUNT (any lora slot
  // exists).
  if (FLAGS_enable_lora && !input_params.adapter_ids.empty() &&
      input_params.adapter_ids[0] != 0 && rt_lora.has_any_experts_adapter() &&
      node.operation->GetInputNum() >
          BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT) {
    uint64_t adapter_id = input_params.adapter_ids[0];
    auto& rt = LoRARuntime::instance();
    const auto* moe_delta = rt.get_moe_expert_delta(adapter_id, layer_id_);

    // Shape verification log (layer 0 only)
    if (layer_id_ == 0) {
      if (moe_delta && moe_delta->A_gate.defined()) {
        LOG(INFO) << "[atb-debug-moe-lora-shape] adapter=" << adapter_id
                  << " layer=" << layer_id_
                  << " A_gate=" << moe_delta->A_gate.sizes()
                  << " B_gate=" << moe_delta->B_gate.sizes()
                  << " A_up=" << moe_delta->A_up.sizes()
                  << " B_up=" << moe_delta->B_up.sizes()
                  << " A_down=" << moe_delta->A_down.sizes()
                  << " B_down=" << moe_delta->B_down.sizes()
                  << " scaling=" << moe_delta->scaling << " r=" << moe_delta->r;
      } else {
        LOG(INFO) << "[atb-debug-moe-lora-shape] adapter=" << adapter_id
                  << " no moe_expert_delta available (attn-only or empty pool)";
      }
    }

    if (moe_delta && moe_delta->A_down.defined() &&
        moe_delta->B_down.defined()) {
      // Down-only fill (path Y). Gate/up deferred to path X (Day 6-7).
      // Path γ v2: LoRA slots start at BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT
      // (index 71) to mirror atb qwen3_moe candidate order base+runtime+lora.
      // Attn-only ("has_any_attn") case: cum_sum(0), qkv(1-8), experts(9-10).
      // Experts-only case (no attn adapter): cum_sum(0), experts down A/B at
      // (1, 2).
      const int32_t off = BASE_WEIGHT_COUNT + RUNTIME_TENSOR_COUNT;
      const int32_t experts_a_off =
          rt_lora.has_any_attn_adapter() ? LORA_MOE_DOWN_A : 1;
      const int32_t experts_b_off =
          rt_lora.has_any_attn_adapter() ? LORA_MOE_DOWN_B : 2;
      node.variantPack.inTensors.at(off + experts_a_off) =
          atb_speed::Utils::AtTensor2Tensor(moe_delta->A_down);
      node.variantPack.inTensors.at(off + experts_b_off) =
          atb_speed::Utils::AtTensor2Tensor(moe_delta->B_down);
    }
    // If moe_delta null / A_gate undefined: leave slots as default (ctor [1]).
    // atb_speed SparseMoe side won't consume experts LoRA slots until Day 2
    // Part 3 wire's up supportLora branch anyway.
  }

  // Copy staged weights (55 base + 15 LoRA if enableLora). Loop bounded by
  // actual GetInputNum() to avoid越界 if LoRA off (atb_speed uses 55 slot).
  const size_t total_input_num = node.operation->GetInputNum();
  const size_t weight_copy_end =
      std::min(total_input_num, static_cast<size_t>(WEIGHT_COUNT_PER_LAYER));
  for (size_t i = 0; i < weight_copy_end; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}
}  // namespace layer
}  // namespace xllm
