/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "models_service_impl.h"

#include <nlohmann/json.hpp>
#include <string>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/framework/lora/lora_runtime.h"
#include "models.pb.h"

namespace xllm {

ModelsServiceImpl::ModelsServiceImpl(
    const std::vector<std::string>& model_names,
    const std::vector<std::string>& model_versions)
    : model_names_(model_names),
      model_versions_(model_versions),
      created_(absl::ToUnixSeconds(absl::Now())) {}

bool ModelsServiceImpl::list_models(const proto::ModelListRequest* request,
                                    proto::ModelListResponse* response) {
  for (const auto& model_id : model_names_) {
    auto* model_card = response->add_data();
    model_card->set_id(model_id);
    model_card->set_created(created_);
    model_card->set_object("model");
    model_card->set_owned_by("xllm");
  }
  return true;
}

std::string ModelsServiceImpl::list_model_versions() {
  nlohmann::json model_states_array = nlohmann::json::array();

  // Base models: emitted with version from launch-time registry.
  for (size_t i = 0; i < model_versions_.size(); ++i) {
    nlohmann::json model_state;
    // The repository index reports the model directory name (carried by
    // model_versions_), so it stays stable regardless of a user-provided
    // --model_id.
    model_state["name"] = model_versions_[i];
    model_state["version"] = model_versions_[i];
    model_state["state"] = "READY";
    model_state["reason"] = "normal";
    model_states_array.push_back(model_state);
  }

  // LoRA adapters: enumerate live registry alongside the base models so
  // /v1/repository/index reflects both. Version is the numeric int_id
  // stringified so downstream tools that treat version as string still work.
  const auto adapters = LoRARuntime::instance().registry().list();
  for (const auto& a : adapters) {
    nlohmann::json adapter_state;
    adapter_state["name"] = a.lora_name;
    adapter_state["version"] = std::to_string(a.lora_int_id);
    adapter_state["state"] = "READY";
    adapter_state["reason"] = "lora_adapter";
    model_states_array.push_back(adapter_state);
  }

  return model_states_array.dump();
}

}  // namespace xllm
