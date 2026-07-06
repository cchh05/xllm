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

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "adapter_loader.h"
#include "lora_config.h"
#include "lora_registry.h"

namespace xllm {

// LoRARuntime is the process-global singleton that ties the LoRA modules
// together for the model forward path. It is deliberately narrow-featured
// so P0-A can end-to-end demonstrate load-via-API without waiting on M4
// (LoraCache) or M8 (Scheduler).
//
// Concretely:
//   - A single model-level dummy A/B pair (Path C semantics) can be
//     dynamically installed by loading an adapter.
//   - The A/B tensors are moved to the model's inference device once at
//     load time; there is no eviction / hot-swap yet.
//   - forward() paths call `active_delta()` and, if it returns a valid
//     value, use those tensors as the whole-block LoRA delta.
//
// Multi-tenant, per-request routing is P0-B / P0-C work. This singleton
// only holds "the currently applied adapter" -- the last one loaded wins.
// That's enough to prove the full end-to-end HTTP -> loader -> registry ->
// forward path from a single curl call.
class LoRARuntime {
 public:
  static LoRARuntime& instance();

  // One-time init; must be called before any adapter API is used. Copies
  // config by value so later registry / cache modules can inspect it
  // without another lookup.
  void init(const LoRAConfig& config);

  bool enabled() const;

  // Called once by the model (QWen3ModelImpl ctor) so subsequent HTTP
  // calls to /v1/load_lora_adapter know what device / dtype to
  // materialise weights on. Idempotent; last caller wins (single model
  // per engine today).
  void set_model_device_dtype(torch::Device device, torch::ScalarType dtype);

  // Load an adapter from a filesystem path, parse its PEFT files, pick a
  // whole-block A/B pair, cast to the model's dtype, and register it.
  //
  // The tensors stay on CPU here -- the actual device migration happens
  // lazily on the first active_delta() call from the model forward
  // thread, which owns the correct NPU context.
  //
  // The most recently loaded adapter becomes the pending one and will
  // be promoted to active on the next forward.
  std::optional<uint64_t> load_and_activate(const std::string& lora_name,
                                            const std::string& lora_path,
                                            const std::string& base_model_name);

  // Deactivate an adapter by name. If it was the active one, active_delta
  // will subsequently return std::nullopt.
  bool unload(const std::string& lora_name);
  const LoRAConfig& config_snapshot() const { return config_; }

  LoRARegistry& registry() { return registry_; }
  const LoRARegistry& registry() const { return registry_; }

  // The active whole-block delta tensors. std::nullopt = no adapter, the
  // forward path should just skip its delta step.
  //
  // Note: tensors are populated on CPU by the load path (which typically
  // runs on the API thread, without an NPU context) and lazily migrated
  // to device on the first active_delta() call from the model forward
  // thread (which owns the correct NPU context). This avoids the
  // aclrtMemcpy-invalid-handle failure you hit when a background thread
  // tries to allocate device memory it does not own.
  struct ActiveDelta {
    torch::Tensor A;  // [rank, hidden] on model_device
    torch::Tensor B;  // [hidden, rank] on model_device
    float scaling;
    std::string name;
    uint64_t int_id;
  };
  std::optional<ActiveDelta> active_delta();

 private:
  LoRARuntime() = default;

  // Called with materialise_mu_ held. Picks a plausible A/B pair from the
  // adapter's canonicalised tensor set. Result is CPU-side, dtype-cast to
  // the model dtype but kept off-device.
  bool pick_whole_block_ab(const LoRAAdapter& adapter,
                           torch::ScalarType dtype,
                           torch::Tensor* A_out,
                           torch::Tensor* B_out) const;

  // Pending / not-yet-migrated CPU tensors, seeded by load_and_activate.
  // active_delta() moves these to device on the forward thread.
  struct PendingDelta {
    torch::Tensor A_cpu;
    torch::Tensor B_cpu;
    float scaling;
    std::string name;
    uint64_t int_id;
  };
  std::optional<PendingDelta> pending_;

  mutable std::mutex materialise_mu_;
  LoRAConfig config_;
  LoRARegistry registry_;
  std::unique_ptr<LoRAAdapterLoader> loader_;

  // Recorded by the model at forward-init time so the HTTP handler knows
  // where to place freshly-loaded LoRA tensors. std::nullopt means no
  // model has registered yet.
  std::optional<torch::Device> model_device_;
  std::optional<torch::ScalarType> model_dtype_;

  // Currently-active adapter's device tensors. Guarded by materialise_mu_.
  std::optional<ActiveDelta> active_;
};

}  // namespace xllm
