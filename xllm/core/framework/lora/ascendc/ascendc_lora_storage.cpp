/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-rank multi-adapter storage (sprint γ+1). See header for the
 * design rationale (rank-only bucketing based on the observation that
 * adapters have a single rank across all layer/proj registers).
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

void AscendCLoRAStorage::update_lookup_locked(uint64_t int_id,
                                              int32_t bucket_id,
                                              int32_t slot,
                                              torch::Device device) {
  if (int_id >= kMaxIntId) return;

  if (!slot_lookup_dev_.defined()) {
    const auto opts =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    slot_lookup_dev_ = torch::full({kMaxIntId}, static_cast<int64_t>(-1), opts);
    bucket_lookup_dev_ =
        torch::full({kMaxIntId}, static_cast<int64_t>(-1), opts);
    LOG(INFO) << "[AscendCLoRAStorage] slot+bucket lookup dev alloc ["
              << kMaxIntId << "] int64 on device (sprint γ+1)";
  }

  auto slot_scalar =
      torch::tensor(static_cast<int64_t>(slot), slot_lookup_dev_.options());
  auto bucket_scalar = torch::tensor(static_cast<int64_t>(bucket_id),
                                     bucket_lookup_dev_.options());
  slot_lookup_dev_.index_put_({static_cast<int64_t>(int_id)}, slot_scalar);
  bucket_lookup_dev_.index_put_({static_cast<int64_t>(int_id)}, bucket_scalar);
}

void AscendCLoRAStorage::clear_lookup_locked(uint64_t int_id) {
  if (int_id >= kMaxIntId) return;
  if (!slot_lookup_dev_.defined()) return;
  auto sentinel =
      torch::tensor(static_cast<int64_t>(-1), slot_lookup_dev_.options());
  slot_lookup_dev_.index_put_({static_cast<int64_t>(int_id)}, sentinel);
  bucket_lookup_dev_.index_put_({static_cast<int64_t>(int_id)}, sentinel);
}

int AscendCLoRAStorage::find_or_create_rank_bucket_locked(int64_t rank) {
  for (int i = 0; i < static_cast<int>(rank_of_bucket_.size()); ++i) {
    if (rank_of_bucket_[i] == rank) return i;
  }
  if (static_cast<int>(rank_of_bucket_.size()) >= kMaxBuckets) {
    LOG(WARNING) << "[AscendCLoRAStorage] rank bucket table full ("
                 << kMaxBuckets << " buckets), refuse rank=" << rank;
    return -1;
  }
  rank_of_bucket_.push_back(rank);
  int new_bucket = static_cast<int>(rank_of_bucket_.size()) - 1;
  LOG(INFO) << "[AscendCLoRAStorage] new rank bucket_id=" << new_bucket
            << " rank=" << rank;
  return new_bucket;
}

int AscendCLoRAStorage::find_free_slot_in_bucket_locked(
    int32_t bucket_id) const {
  std::vector<bool> used(kNMaxActive, false);
  for (const auto& kv : int_id_to_location_) {
    if (kv.second.bucket_id == bucket_id && kv.second.slot >= 0 &&
        kv.second.slot < kNMaxActive) {
      used[kv.second.slot] = true;
    }
  }
  for (int i = 0; i < kNMaxActive; ++i) {
    if (!used[i]) return i;
  }
  return -1;
}

bool AscendCLoRAStorage::ensure_slab_allocated_locked(int layer_idx,
                                                      const std::string& proj,
                                                      int32_t bucket_id,
                                                      int64_t rank,
                                                      const torch::Tensor& A,
                                                      const torch::Tensor& B,
                                                      float scaling) {
  KeyType key{layer_idx, proj};
  auto& bucketed = storage_[key];

  // Ensure the buckets vector has enough entries up to bucket_id.
  while (static_cast<int>(bucketed.buckets.size()) <= bucket_id) {
    bucketed.buckets.emplace_back();
    bucketed.rank_of_bucket.push_back(-1);
  }

  auto& view = bucketed.buckets[bucket_id];
  if (!view.valid) {
    // First adapter with this rank at this (layer, proj). Allocate slab
    // using this adapter's hidden_in/hidden_out.
    const auto opts = A.options();
    view.A_stacked = torch::zeros({kNMaxActive, rank, A.size(1)}, opts);
    view.B_stacked = torch::zeros({kNMaxActive, B.size(0), rank}, opts);
    view.scaling = scaling;
    view.valid = true;
    bucketed.rank_of_bucket[bucket_id] = rank;
    LOG(INFO) << "[AscendCLoRAStorage] new slab (layer=" << layer_idx
              << ", proj=" << proj << ", bucket=" << bucket_id
              << ", rank=" << rank << ") A=[" << kNMaxActive << "," << rank
              << "," << A.size(1) << "] B=[" << kNMaxActive << "," << B.size(0)
              << "," << rank << "]";
    return true;
  }

  // Slab exists: verify shape. Same rank is guaranteed by bucket_id.
  // hidden_in and hidden_out must also match (they are model+proj properties
  // that should be identical across adapters for the same (layer, proj)).
  if (view.A_stacked.size(2) != A.size(1) ||
      view.B_stacked.size(1) != B.size(0)) {
    LOG(WARNING) << "[AscendCLoRAStorage] slab shape mismatch (layer="
                 << layer_idx << ", proj=" << proj << ", bucket=" << bucket_id
                 << ") existing_A_Hin=" << view.A_stacked.size(2)
                 << " new_A_Hin=" << A.size(1)
                 << " existing_B_Hout=" << view.B_stacked.size(1)
                 << " new_B_Hout=" << B.size(0) << ", refuse";
    return false;
  }
  return true;
}

int AscendCLoRAStorage::register_adapter(uint64_t int_id,
                                         int layer_idx,
                                         const std::string& proj,
                                         const torch::Tensor& A,
                                         const torch::Tensor& B,
                                         float scaling) {
  std::unique_lock<std::shared_mutex> lock(mu_);

  TORCH_CHECK(A.is_privateuseone() || A.device().is_privateuseone(),
              "AscendCLoRAStorage::register_adapter: A must be on NPU device");

  const int64_t rank = A.size(0);

  // Idempotent: if int_id already recorded, verify rank consistency.
  auto loc_it = int_id_to_location_.find(int_id);
  if (loc_it != int_id_to_location_.end()) {
    if (loc_it->second.rank != rank) {
      LOG(WARNING) << "[AscendCLoRAStorage] int_id=" << int_id
                   << " re-registers with different rank (new=" << rank
                   << " vs recorded=" << loc_it->second.rank << "), refusing";
      return -1;
    }
  }

  int bucket_id = find_or_create_rank_bucket_locked(rank);
  if (bucket_id < 0) return -1;

  // Allocate/verify slab at (layer, proj, bucket_id).
  if (!ensure_slab_allocated_locked(
          layer_idx, proj, bucket_id, rank, A, B, scaling)) {
    return -1;
  }

  // Determine slot. If int_id already has a location, reuse its slot
  // (adapter registers into every layer/proj using the same slot number).
  int slot = -1;
  if (loc_it != int_id_to_location_.end()) {
    slot = loc_it->second.slot;
  } else {
    slot = find_free_slot_in_bucket_locked(static_cast<int32_t>(bucket_id));
    if (slot < 0) {
      LOG_EVERY_N(WARNING, 100) << "[AscendCLoRAStorage] bucket " << bucket_id
                                << " (rank=" << rank << ") all " << kNMaxActive
                                << " slots occupied, refuse int_id=" << int_id;
      return -1;
    }
  }

  auto& view = storage_[{layer_idx, proj}].buckets[bucket_id];
  view.A_stacked[slot].copy_(A);
  view.B_stacked[slot].copy_(B);
  view.version++;

  if (loc_it == int_id_to_location_.end()) {
    int_id_to_location_[int_id] = {
        static_cast<int32_t>(bucket_id), static_cast<int32_t>(slot), rank};
    update_lookup_locked(int_id,
                         static_cast<int32_t>(bucket_id),
                         static_cast<int32_t>(slot),
                         A.device());
  }

  LOG_EVERY_N(INFO, 20) << "[AscendCLoRAStorage] register int_id=" << int_id
                        << " layer=" << layer_idx << " proj=" << proj
                        << " bucket=" << bucket_id << " slot=" << slot
                        << " rank=" << rank << " A=[" << A.size(0) << ","
                        << A.size(1) << "] B=[" << B.size(0) << "," << B.size(1)
                        << "] scaling=" << scaling;

  return slot;
}

void AscendCLoRAStorage::unregister_adapter(uint64_t int_id) {
  std::unique_lock<std::shared_mutex> lock(mu_);
  auto it = int_id_to_location_.find(int_id);
  if (it == int_id_to_location_.end()) return;

  int32_t bucket_id = it->second.bucket_id;
  int32_t slot = it->second.slot;
  int_id_to_location_.erase(it);

  for (auto& kv : storage_) {
    auto& bucketed = kv.second;
    if (bucket_id >= 0 && bucket_id < bucketed.num_buckets()) {
      auto& view = bucketed.buckets[bucket_id];
      if (view.valid && slot >= 0 && slot < kNMaxActive) {
        view.A_stacked[slot].zero_();
        view.B_stacked[slot].zero_();
        view.version++;
      }
    }
  }

  clear_lookup_locked(int_id);
}

AscendCLoRAStorage::BucketedStackedView AscendCLoRAStorage::get_bucketed(
    int layer_idx,
    const std::string& proj) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  auto it = storage_.find({layer_idx, proj});
  if (it == storage_.end()) return {};
  return it->second;
}

int AscendCLoRAStorage::active_count() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return static_cast<int>(int_id_to_location_.size());
}

int64_t AscendCLoRAStorage::bucket_rank(int32_t bucket_id) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  if (bucket_id < 0 || bucket_id >= static_cast<int>(rank_of_bucket_.size())) {
    return -1;
  }
  return rank_of_bucket_[bucket_id];
}

}  // namespace xllm
