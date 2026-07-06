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

  // PICK_NO_CAST_APPLIED: skipping in-place .to(dtype) here — safetensors
  // mmap tensor + torch::to on CPU crashes torch_npu's opapi allocator on
  // ctor thread. Callers (install_static / load_and_activate) do their
  // own clone + cast now.
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
  {
    std::lock_guard g(materialise_mu_);
    if (!model_dtype_.has_value()) {
      LOG(ERROR) << "[LoRARuntime] model has not registered dtype yet; "
                    "reject load '"
                 << lora_name << "'";
      return std::nullopt;
    }
    dtype = *model_dtype_;
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

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  {
    std::lock_guard g(materialise_mu_);
    pending_ =
        PendingDelta{A_cpu, B_cpu, adapter_opt->scaling, lora_name, *id_opt};
    // Clear the previous active adapter -- next forward will materialise
    // the new one on device.
    active_.reset();
  }
  LOG(INFO) << "[LoRARuntime] queued '" << lora_name << "' id=" << *id_opt
            << " A_cpu.shape=" << A_cpu.sizes()
            << " B_cpu.shape=" << B_cpu.sizes()
            << " scaling=" << adapter_opt->scaling
            << " (device migration deferred to first forward)";
  return id_opt;
}

std::optional<uint64_t> LoRARuntime::install_static_adapter_on_device(
    const std::string& lora_name,
    const std::string& lora_path,
    const std::string& base_model_name,
    torch::Device device,
    torch::ScalarType dtype) {
  if (!enabled()) {
    LOG(ERROR) << "[LoRARuntime] not enabled; refuse to install '" << lora_name
               << "'";
    return std::nullopt;
  }
  if (!loader_) {
    LOG(ERROR) << "[LoRARuntime] loader not initialised";
    return std::nullopt;
  }

  LoRARequest req{lora_name, /*int_id=*/0, lora_path, base_model_name};
  auto adapter_opt = loader_->load(req);
  if (!adapter_opt) return std::nullopt;

  // PICK_BRACKET_APPLIED: log before/after to isolate the crash location.
  LOG(INFO) << "[LoRARuntime] BEFORE pick_whole_block_ab for '" << lora_name
            << "'";
  torch::Tensor A_cpu, B_cpu;
  if (!pick_whole_block_ab(*adapter_opt, dtype, &A_cpu, &B_cpu)) {
    LOG(ERROR) << "[LoRARuntime] pick_whole_block_ab returned false";
    return std::nullopt;
  }
  LOG(INFO) << "[LoRARuntime] AFTER pick_whole_block_ab: A=" << A_cpu.sizes()
            << " B=" << B_cpu.sizes() << " A.dtype=" << A_cpu.dtype()
            << " A.device=" << A_cpu.device();

  // KEY DIFFERENCE vs load_and_activate: perform the CPU->NPU migration
  // on this thread, which is the model ctor thread and thus has a valid
  // NPU context (V60 experiment 2026-07-06 confirmed).
  // STATIC_TO_CLONE_APPLIED: safetensors-loaded tensors have mmap backing
  // that crashes torch_npu's opapi copy from ctor thread. Clone to a fresh
  // cpu allocator buffer, then .to(device).
  LOG(INFO) << "[LoRARuntime] static preload '" << lora_name
            << "' A_cpu.dtype=" << A_cpu.dtype() << " sizes=" << A_cpu.sizes()
            << " storage=" << A_cpu.storage().nbytes() << "B";
  // DIRECT_TO_APPLIED: even .clone() crashes on safetensors mmap tensors on
  // ctor thread. Try single-step .to(device, dtype).
  //
  // Strategy: build a fresh CPU tensor via torch::empty (same allocator as
  // V60 randn) and copy_ from mmap tensor - this materializes into normal
  // heap. Then .to(device) proceeds.
  torch::Tensor A_dev, B_dev;
  try {
    LOG(INFO) << "[LoRARuntime] materializing A_cpu via torch::empty+copy_";
    auto cpu_opts =
        torch::TensorOptions().dtype(A_cpu.dtype()).device(torch::kCPU);
    torch::Tensor A_cpu_owned = torch::empty(A_cpu.sizes(), cpu_opts);
    A_cpu_owned.copy_(A_cpu);
    torch::Tensor B_cpu_owned = torch::empty(B_cpu.sizes(), cpu_opts);
    B_cpu_owned.copy_(B_cpu);
    LOG(INFO) << "[LoRARuntime] materialized; casting + moving to " << device;
    A_dev = A_cpu_owned.to(device, dtype);
    LOG(INFO) << "[LoRARuntime] A_dev on " << A_dev.device()
              << " dtype=" << A_dev.dtype() << " sizes=" << A_dev.sizes();
    B_dev = B_cpu_owned.to(device, dtype);
    LOG(INFO) << "[LoRARuntime] B_dev on " << B_dev.device()
              << " dtype=" << B_dev.dtype() << " sizes=" << B_dev.sizes();
  } catch (const std::exception& e) {
    LOG(ERROR) << "[LoRARuntime] .to(device) failed for static adapter '"
               << lora_name << "': " << e.what();
    return std::nullopt;
  }

  const auto id_opt = registry_.register_adapter(req);
  if (!id_opt) return std::nullopt;

  {
    std::lock_guard g(materialise_mu_);
    ActiveDelta ad;
    ad.A = A_dev;
    ad.B = B_dev;
    ad.scaling = adapter_opt->scaling;
    ad.name = lora_name;
    ad.int_id = *id_opt;
    active_ = std::move(ad);
    // Static preload wins over any pending -- unload() and future
    // load_and_activate() calls behave normally.
    pending_.reset();
  }
  LOG(INFO) << "[LoRARuntime] installed static adapter '" << lora_name
            << "' id=" << *id_opt << " A_device=" << A_dev.device()
            << " A.shape=" << A_dev.sizes() << " B.shape=" << B_dev.sizes()
            << " scaling=" << adapter_opt->scaling;
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
    if (pending_ && pending_->name == lora_name) {
      pending_.reset();
      LOG(INFO) << "[LoRARuntime] deactivated pending '" << lora_name << "'";
    }
  }
  return ok;
}

std::optional<LoRARuntime::ActiveDelta> LoRARuntime::active_delta() {
  std::lock_guard g(materialise_mu_);

  // Fast path: nothing to do.
  if (!pending_ && !active_) return std::nullopt;

  // Promote pending -> active. We DELIBERATELY leave the tensors on CPU
  // here even though the model wants them on device: the actual .to(npu)
  // has to happen on the worker's forward thread AND under the atb path
  // which has aclrtSetDevice set. Doing it here (from the LoRARuntime
  // mutex) even on the forward thread crashes with aclrtMemcpy 107017
  // because torch_npu's opapi copy stream is not attached.
  //
  // The caller (qwen3.h forward loop) does the .to(h.device()) inline:
  // that copy runs in the atb-managed NPU stream and works.
  if (pending_) {
    ActiveDelta ad;
    ad.A = pending_->A_cpu;  // still on CPU
    ad.B = pending_->B_cpu;  // still on CPU
    ad.scaling = pending_->scaling;
    ad.name = pending_->name;
    ad.int_id = pending_->int_id;
    active_ = std::move(ad);
    pending_.reset();
    LOG(INFO) << "[LoRARuntime] promoted '" << active_->name
              << "' id=" << active_->int_id
              << " (CPU-side; caller performs .to(device))";
  }
  return active_;
}

}  // namespace xllm
