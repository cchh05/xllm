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

bool LoRARuntime::pick_whole_block_ab(const LoRAAdapter& adapter,
                                      torch::Device device,
                                      torch::ScalarType dtype,
                                      torch::Tensor* A_out,
                                      torch::Tensor* B_out) const {
  // Path C is a *whole-decoder-block* delta -- one A/B for the whole model.
  // A real PEFT adapter has A/B per (layer, module). For MVP we pick the
  // first (module, layer=0) pair we find whose shapes match [rank, hidden]
  // / [hidden, rank]. This is intentionally coarse; real per-proj routing
  // is P0-C.
  //
  // Preference order: q_proj > gate_proj > any first pair. We match the
  // canonical-name convention "<subkey>#A" / "<subkey>#B" set by the
  // loader.
  static const std::vector<std::string> kPreferSubstr = {
      "layers.0.self_attn.q_proj",
      "self_attn.q_proj",
      "layers.0",
  };
  auto find_pair = [&](const std::string& subkey_hint,
                       torch::Tensor* A,
                       torch::Tensor* B) -> bool {
    for (const auto& [key, tensor] : adapter.tensors) {
      // key looks like "<subkey>#A" or "<subkey>#B"
      if (key.size() < 2) continue;
      const std::string subkey = key.substr(0, key.size() - 2);
      const std::string tail = key.substr(key.size() - 2);
      if (!subkey_hint.empty() && subkey.find(subkey_hint) == std::string::npos)
        continue;
      if (tail == "#A") {
        // Look for matching #B under same subkey.
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
    // Give up: try any first pair.
    if (!find_pair("", A_out, B_out)) {
      LOG(ERROR) << "[LoRARuntime] adapter '" << adapter.request.lora_name
                 << "' has no usable A/B pair";
      return false;
    }
  }

  // Move to device with the model's dtype.
  *A_out = A_out->to(device).to(dtype).contiguous();
  *B_out = B_out->to(device).to(dtype).contiguous();
  return true;
}

std::optional<uint64_t> LoRARuntime::load_and_activate(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse to load '" << lora_name
               << "'";
    return std::nullopt;
  }
  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  // Pick whole-block A/B before we touch the registry so a bad adapter
  // does not leave a phantom entry.
  torch::Tensor A, B;
  if (!pick_whole_block_ab(*adapter_opt, device, dtype, &A, &B)) {
    return std::nullopt;
  }

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  ActiveDelta ad;
  ad.A = A;
  ad.B = B;
  ad.scaling = adapter_opt->scaling;
  ad.name = lora_name;
  ad.int_id = *id_opt;

  {
    std::lock_guard g(materialise_mu_);
    active_ = std::move(ad);
  }
  LOG(INFO) << "[LoRARuntime] activated '" << lora_name << "' id=" << *id_opt
            << " A.shape=" << A.sizes() << " B.shape=" << B.sizes()
            << " scaling=" << adapter_opt->scaling;
  return id_opt;
}

bool LoRARuntime::unload(const std::string& lora_name) {
  bool ok = registry_.unregister(lora_name);
  {
    std::lock_guard g(materialise_mu_);
    if (active_ && active_->name == lora_name) {
      active_.reset();
      LOG(INFO) << "[LoRARuntime] deactivated '" << lora_name << "'";
    }
  }
  return ok;
}

std::optional<LoRARuntime::ActiveDelta> LoRARuntime::active_delta() const {
  std::lock_guard g(materialise_mu_);
  return active_;
}

}  // namespace xllm
