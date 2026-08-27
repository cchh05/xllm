/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ascendc_lora_storage.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

namespace xllm {

AscendCLoRAStorage& AscendCLoRAStorage::instance() {
  static AscendCLoRAStorage inst;
  return inst;
}

int AscendCLoRAStorage::register_adapter(uint64_t int_id,
                                         int layer_idx,
                                         const std::string& proj,
                                         const torch::Tensor& A,
                                         const torch::Tensor& B,
                                         float scaling) {
  std::lock_guard<std::mutex> lock(mu_);

  // Sync-copy invariant: LoRARuntime::install_static_adapter_on_device_per_proj
  // is called from the master install path with the adapter tensors already
  // materialized on NPU. If a future refactor moves install to a background
  // stream, this assert forces the caller to add explicit stream sync.
  TORCH_CHECK(A.is_privateuseone() || A.device().is_privateuseone(),
              "AscendCLoRAStorage::register_adapter: A must be on NPU device");

  int slot = -1;
  auto it = int_id_to_slot_.find(int_id);
  if (it != int_id_to_slot_.end()) {
    slot = it->second;
  } else {
    // Find first empty slot
    std::vector<bool> used(kNMaxActive, false);
    for (const auto& kv : int_id_to_slot_) {
      if (kv.second >= 0 && kv.second < kNMaxActive) used[kv.second] = true;
    }
    for (int i = 0; i < kNMaxActive; ++i) {
      if (!used[i]) {
        slot = i;
        break;
      }
    }
    if (slot < 0) {
      LOG_EVERY_N(WARNING, 100) << "[AscendCLoRAStorage] all " << kNMaxActive
                                << " slots occupied, refuse int_id=" << int_id
                                << " (fallback to per-seq)";
      return -1;
    }
    int_id_to_slot_[int_id] = slot;
  }

  KeyType key{layer_idx, proj};
  auto& view = storage_[key];
  if (!view.valid) {
    // First adapter for this (layer, proj): allocate stacked slabs
    // A: [R, H_in], B: [H_out, R] → stacked [N, R, H_in] / [N, H_out, R]
    const auto opts = A.options();
    view.A_stacked = torch::zeros({kNMaxActive, A.size(0), A.size(1)}, opts);
    view.B_stacked = torch::zeros({kNMaxActive, B.size(0), B.size(1)}, opts);
    view.scaling = scaling;
    view.valid = true;
  }

  // Slot in the stacked slab
  view.A_stacked[slot].copy_(A);
  view.B_stacked[slot].copy_(B);
  view.version++;

  return slot;
}

void AscendCLoRAStorage::unregister_adapter(uint64_t int_id) {
  std::lock_guard<std::mutex> lock(mu_);
  // Invariant: unregister is called from the LoRARuntime unload path after
  // ref_count==0 (RAII unpin, see
  // xllm-lora-registry-cross-rank-int-id-race-2026-08-25 memory). No forward
  // can hold a StackedView pointer that references this slot at this moment, so
  // zero-out is race-free.
  auto it = int_id_to_slot_.find(int_id);
  if (it == int_id_to_slot_.end()) return;
  int slot = it->second;
  int_id_to_slot_.erase(it);
  for (auto& kv : storage_) {
    if (kv.second.valid) {
      kv.second.A_stacked[slot].zero_();
      kv.second.B_stacked[slot].zero_();
      kv.second.version++;
    }
  }
}

AscendCLoRAStorage::StackedView AscendCLoRAStorage::get_stacked(
    int layer_idx,
    const std::string& proj) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = storage_.find({layer_idx, proj});
  if (it == storage_.end()) return {};
  return it->second;
}

torch::Tensor AscendCLoRAStorage::build_indices_cpu(
    const std::vector<uint64_t>& int_ids_per_token) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto out = torch::empty({static_cast<int64_t>(int_ids_per_token.size())},
                          torch::kInt64);
  auto out_ptr = out.data_ptr<int64_t>();
  for (size_t i = 0; i < int_ids_per_token.size(); ++i) {
    uint64_t iid = int_ids_per_token[i];
    if (iid == 0) {
      out_ptr[i] = -1;  // base tokens: caller must mask, or use safe slot 0
      continue;
    }
    auto it = int_id_to_slot_.find(iid);
    out_ptr[i] = (it != int_id_to_slot_.end()) ? it->second : -1;
  }
  return out;
}

int AscendCLoRAStorage::active_count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(int_id_to_slot_.size());
}

}  // namespace xllm
