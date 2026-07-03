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

  // Load an adapter from a filesystem path, register it, and materialise
  // its whole-block A/B tensors on `device`. Returns int_id on success.
  //
  // `whole_block_A_key` / `whole_block_B_key` name the two tensor entries
  // inside the loaded LoRAAdapter that should be used as the model-level
  // delta. When they are missing (rare in practice) we fall back to the
  // first suitable A/B pair we find so the demo path stays alive.
  //
  // The most recently loaded adapter becomes the active one.
  std::optional<uint64_t> load_and_activate(const std::string& lora_name,
                                            const std::string& lora_path,
                                            const std::string& base_model_name,
                                            torch::Device device,
                                            torch::ScalarType dtype);

  // Deactivate an adapter by name. If it was the active one, active_delta
  // will subsequently return std::nullopt.
  bool unload(const std::string& lora_name);

  LoRARegistry& registry() { return registry_; }
  const LoRARegistry& registry() const { return registry_; }

  // The active whole-block delta tensors. std::nullopt = no adapter, the
  // forward path should just skip its delta step.
  struct ActiveDelta {
    torch::Tensor A;  // [rank, hidden]
    torch::Tensor B;  // [hidden, rank]
    float scaling;
    std::string name;
    uint64_t int_id;
  };
  std::optional<ActiveDelta> active_delta() const;

 private:
  LoRARuntime() = default;

  // Called with materialise_mu_ held. Picks a plausible A/B pair from the
  // adapter's canonicalised tensor set. See notes in impl.
  bool pick_whole_block_ab(const LoRAAdapter& adapter,
                           torch::Device device,
                           torch::ScalarType dtype,
                           torch::Tensor* A_out,
                           torch::Tensor* B_out) const;

  mutable std::mutex materialise_mu_;
  LoRAConfig config_;
  LoRARegistry registry_;
  std::unique_ptr<LoRAAdapterLoader> loader_;

  // Currently-active adapter's device tensors. Guarded by materialise_mu_.
  std::optional<ActiveDelta> active_;
};

}  // namespace xllm
