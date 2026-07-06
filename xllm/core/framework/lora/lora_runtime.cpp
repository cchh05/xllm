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

#include "lora_runtime.h"

#include <acl/acl.h>
#include <glog/logging.h>

#include <mutex>

namespace xllm {

LoRARuntime& LoRARuntime::instance() {
  static LoRARuntime g;
  return g;
}

void LoRARuntime::init(const LoRAConfig& config) {
  std::lock_guard g(materialise_mu_);
  config_ = config;
  loader_ = std::make_unique<LoRAAdapterLoader>(config_);
  LOG(INFO) << "[LoRARuntime] initialised, enable=" << config_.enable_lora;
}

bool LoRARuntime::enabled() const { return config_.enable_lora; }

void LoRARuntime::set_model_device_dtype(torch::Device device,
                                         torch::ScalarType dtype) {
  std::lock_guard g(materialise_mu_);
  model_device_ = device;
  model_dtype_ = dtype;
  LOG(INFO) << "[LoRARuntime] model registered device=" << device
            << " dtype=" << dtype;
}

bool LoRARuntime::pick_whole_block_ab(const LoRAAdapter& adapter,
                                      torch::ScalarType dtype,
                                      torch::Tensor* A_out,
                                      torch::Tensor* B_out) const {
  // Path C is a *whole-decoder-block* delta -- one A/B for the whole
  // model. A real PEFT adapter has A/B per (layer, module). For MVP we
  // pick a first (module, layer=0) pair whose shapes match [rank, hidden]
  // / [hidden, rank] and treat that as the model-level delta. Real
  // per-proj / per-layer routing is P0-C.
  //
  // Preference order: q_proj > gate_proj > any first pair.
  static const std::vector<std::string> kPreferSubstr = {
      "layers.0.self_attn.q_proj",
      "self_attn.q_proj",
      "layers.0",
  };
  auto find_pair = [&](const std::string& subkey_hint,
                       torch::Tensor* A,
                       torch::Tensor* B) -> bool {
    for (const auto& [key, tensor] : adapter.tensors) {
      if (key.size() < 2) continue;
      const std::string subkey = key.substr(0, key.size() - 2);
      const std::string tail = key.substr(key.size() - 2);
      if (!subkey_hint.empty() && subkey.find(subkey_hint) == std::string::npos)
        continue;
      if (tail == "#A") {
        auto it = adapter.tensors.find(subkey + "#B");
        if (it == adapter.tensors.end()) continue;
        *A = tensor;
        *B = it->second;
        return true;
      }
    }
    return false;
  };
  for (const auto& hint : kPreferSubstr) {
    if (find_pair(hint, A_out, B_out)) break;
  }
  if (!A_out->defined() || !B_out->defined()) {
    if (!find_pair("", A_out, B_out)) {
      LOG(ERROR) << "[LoRARuntime] adapter '" << adapter.request.lora_name
                 << "' has no usable A/B pair";
      return false;
    }
  }

  // Cast to model dtype but STAY ON CPU. HtoD copy happens below via
  // raw aclrtMemcpy (Route B).
  *A_out = A_out->to(dtype).contiguous();
  *B_out = B_out->to(dtype).contiguous();
  return true;
}

// Route B helper: raw aclrtMemcpy from a contiguous CPU tensor into a
// freshly-allocated NPU tensor. Bypasses torch_npu's opapi memcpy stream
// (which is per-thread) so this works from any thread that has run
// aclrtSetDevice for the target device. Mirrors vLLM's use of raw
// cudaMemcpy on CUDA.
static bool cpu_to_npu_via_aclrt(const torch::Tensor& cpu_src,
                                 torch::Device device,
                                 torch::ScalarType dtype,
                                 torch::Tensor* dev_out,
                                 std::string* err_out) {
  if (!cpu_src.is_contiguous()) {
    *err_out = "source tensor not contiguous";
    return false;
  }
  aclError set_err = aclrtSetDevice(static_cast<int32_t>(device.index()));
  if (set_err != ACL_ERROR_NONE) {
    *err_out = "aclrtSetDevice(" + std::to_string(device.index()) +
               ") failed: " + std::to_string(set_err);
    return false;
  }
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor dst;
  try {
    dst = torch::empty(cpu_src.sizes(), opts);
  } catch (const std::exception& e) {
    *err_out = std::string("torch::empty on device failed: ") + e.what();
    return false;
  }
  const size_t nbytes = static_cast<size_t>(cpu_src.nbytes());
  aclError err = aclrtMemcpy(dst.data_ptr(),
                             nbytes,
                             cpu_src.data_ptr(),
                             nbytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
  if (err != ACL_ERROR_NONE) {
    *err_out = "aclrtMemcpy H2D failed: " + std::to_string(err);
    return false;
  }
  aclError sync_err = aclrtSynchronizeDevice();
  if (sync_err != ACL_ERROR_NONE) {
    *err_out = "aclrtSynchronizeDevice failed: " + std::to_string(sync_err);
    return false;
  }
  *dev_out = dst;
  return true;
}

std::optional<uint64_t> LoRARuntime::load_and_activate(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse to load '" << lora_name
               << "'";
    return std::nullopt;
  }
  torch::ScalarType dtype = torch::kFloat32;
  torch::Device device = torch::kCPU;
  {
    std::lock_guard g(materialise_mu_);
    if (!model_dtype_.has_value() || !model_device_.has_value()) {
      LOG(ERROR) << "[LoRARuntime] model has not registered device/dtype yet;"
                    " reject load '"
                 << lora_name << "'";
      return std::nullopt;
    }
    dtype = *model_dtype_;
    device = *model_device_;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  torch::Tensor A_cpu, B_cpu;
  if (!pick_whole_block_ab(*adapter_opt, dtype, &A_cpu, &B_cpu)) {
    return std::nullopt;
  }

  // Route B: raw aclrtMemcpy. Works from any thread (worker threadpool,
  // API handler, model-init) as long as aclrtSetDevice has been called.
  // This is the community-aligned pattern -- vLLM uses cudaMemcpy for
  // adapter uploads on CUDA for exactly the same reason.
  torch::Tensor A_dev, B_dev;
  {
    std::string err_a, err_b;
    if (!cpu_to_npu_via_aclrt(A_cpu, device, dtype, &A_dev, &err_a)) {
      LOG(ERROR) << "[LoRARuntime] Route B copy A failed for '" << lora_name
                 << "': " << err_a;
      return std::nullopt;
    }
    if (!cpu_to_npu_via_aclrt(B_cpu, device, dtype, &B_dev, &err_b)) {
      LOG(ERROR) << "[LoRARuntime] Route B copy B failed for '" << lora_name
                 << "': " << err_b;
      return std::nullopt;
    }
  }

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  {
    std::lock_guard g(materialise_mu_);
    active_ =
        ActiveDelta{A_dev, B_dev, adapter_opt->scaling, lora_name, *id_opt};
  }
  LOG(INFO) << "[LoRARuntime] activated '" << lora_name << "' id=" << *id_opt
            << " A.shape=" << A_dev.sizes() << " B.shape=" << B_dev.sizes()
            << " scaling=" << adapter_opt->scaling
            << " device=" << A_dev.device() << " (Route B / aclrtMemcpy)";
  return id_opt;
}

bool LoRARuntime::unload(const std::string& lora_name) {
  bool ok = registry_.unregister(lora_name);
  {
    std::lock_guard g(materialise_mu_);
    if (active_ && active_->name == lora_name) {
      active_.reset();
      LOG(INFO) << "[LoRARuntime] deactivated active '" << lora_name << "'";
    }
  }
  return ok;
}

std::optional<LoRARuntime::ActiveDelta> LoRARuntime::active_delta() {
  std::lock_guard g(materialise_mu_);
  return active_;
}

}  // namespace xllm
