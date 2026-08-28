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

void AscendCLoRAStorage::ensure_bufs_ready_locked(torch::Device device,
                                                  int64_t max_t) {
  if (buf_max_t_ >= max_t && buf_q_shared_.defined()) return;
  // Grow-only 2x factor to amortize future spikes.
  buf_max_t_ = std::max<int64_t>(buf_max_t_, max_t * 2);
  const auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  buf_q_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  buf_k_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  buf_v_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  // Commit C: MoE gate/up/down bufs, independent to avoid concurrent op race
  buf_gate_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  buf_up_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  buf_down_shared_ = torch::zeros({buf_max_t_, kBufMaxR}, opts);
  LOG(INFO) << "[AscendCLoRAStorage] buf pool alloc max_t=" << buf_max_t_
            << " R_max=" << kBufMaxR
            << " (Q/K/V + gate/up/down fp32, per-rank shared, Commit C)";
}

void AscendCLoRAStorage::update_slot_lookup_locked(uint64_t int_id,
                                                   int32_t slot,
                                                   torch::Device device) {
  if (int_id >= kMaxIntId) return;  // silently ignore out-of-range
  if (!slot_lookup_dev_.defined()) {
    // First register: allocate device tensor initialized to -1 (sentinel).
    // This runs at install-path time (not in graph capture), so a device
    // allocation + host-driven fill via torch::full is graph-safe.
    const auto opts =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    slot_lookup_dev_ = torch::full({kMaxIntId}, static_cast<int64_t>(-1), opts);
    LOG(INFO) << "[AscendCLoRAStorage] slot_lookup_dev alloc [" << kMaxIntId
              << "] int64 on device (Fix W)";
  }
  // Write slot at index int_id. index_put_ is device-side; the RHS scalar
  // tensor is built on device via .options() so no host→device copy.
  auto slot_scalar =
      torch::tensor(static_cast<int64_t>(slot), slot_lookup_dev_.options());
  slot_lookup_dev_.index_put_({static_cast<int64_t>(int_id)}, slot_scalar);
}

int AscendCLoRAStorage::register_adapter(uint64_t int_id,
                                         int layer_idx,
                                         const std::string& proj,
                                         const torch::Tensor& A,
                                         const torch::Tensor& B,
                                         float scaling) {
  std::unique_lock<std::shared_mutex> lock(mu_);

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

  // Fix Y: ensure shared buf pool is ready. First install triggers alloc;
  // subsequent installs are a no-op unless max_t grew. Hard-coded 50000
  // matches --max_tokens_per_batch=50000 prod config (Sprint scope).
  ensure_bufs_ready_locked(A.device(), /*max_t=*/50000);

  // Fix W: mirror the slot mapping into the device-side lookup table so
  // slow_path forward can build per-token indices without host→device copy
  // (required for graph-capture safety, see class comment).
  update_slot_lookup_locked(int_id, static_cast<int32_t>(slot), A.device());

  LOG(INFO) << "[AscendCLoRAStorage] register int_id=" << int_id
            << " layer=" << layer_idx << " proj=" << proj << " slot=" << slot
            << " A=[" << A.size(0) << "," << A.size(1) << "] B=[" << B.size(0)
            << "," << B.size(1) << "] scaling=" << scaling;

  return slot;
}

void AscendCLoRAStorage::unregister_adapter(uint64_t int_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  // Invariant: unregister is called from the LoRARuntime unload path after
  // ref_count==0 (RAII unpin, see
  // xllm-lora-registry-cross-rank-int-id-race-2026-08-25 memory). No forward
  // can hold a StackedView pointer that references this slot at this moment, so
  // zero-out is race-free.
  //
  // Fix W note: slot_lookup_dev_ is intentionally NOT invalidated here. The
  // forward slow_path host-side `aid == 0` guard blocks base-mixed batches
  // from entering the device index_select path; a request that legitimately
  // hits the slow_path must have aid != 0 AND be registered (LoRARegistry
  // install hook fires before scheduler admits the batch). Skipping device
  // invalidation avoids a device write while forward readers are running
  // (index_select on a stale slot returns the pre-unregister value, which
  // is safe as long as no forward legitimately targets a torn-down adapter
  // — that scenario would already race the storage_[layer,proj] zero_()
  // above and is out of scope for this sprint).
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
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = storage_.find({layer_idx, proj});
  if (it == storage_.end()) return {};
  return it->second;
}

torch::Tensor AscendCLoRAStorage::build_indices_cpu(
    const std::vector<uint64_t>& int_ids_per_token) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
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
  std::shared_lock<std::shared_mutex> lock(mu_);
  return static_cast<int>(int_id_to_slot_.size());
}

}  // namespace xllm
