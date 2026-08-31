/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AscendCLoRAStorage — multi-rank multi-adapter storage (sprint γ+1).
 *
 * DESIGN NOTE (sprint γ+1 v1):
 *   Adapters in practice have a single rank across all layer/proj registers
 *   (adapter_config.json has one "r" value). hidden_in is a model property,
 *   also constant across all (layer, proj) for a given adapter. hidden_out
 *   varies per proj (q_proj vs v_proj etc.) but is the same across all
 *   adapters at the same (layer, proj). Therefore we bucket ONLY by rank:
 *   adapters with the same rank share a slot number globally; each
 *   (layer, proj) then holds its own StackedView with slab-shape derived
 *   from that (layer, proj)'s natural hidden_in/hidden_out. Different-
 *   rank adapters live in different buckets.
 *
 *   Slot occupancy is bucket-local: each rank bucket has kNMaxActive=8
 *   slots. An adapter takes one slot in exactly one bucket (its rank)
 *   and reuses that slot across all its (layer, proj) registers.
 *
 * Storage layout per (layer, proj, rank):
 *   A_stacked [N=8, R, H_in]  bfloat16 device
 *   B_stacked [N=8, H_out, R] bfloat16 device
 *
 * int_id → AdapterLocation:
 *   {bucket_id (= rank_bucket), slot in that bucket, rank}
 *
 * Forward path (wrapper):
 *   1. Token t → int_id[t] → (slot, bucket_id) via device lookup tables.
 *   2. Group tokens by bucket_id.
 *   3. Per bucket, launch bgmv_shrink/expand on that bucket's slab.
 *   4. Scatter results back to y.
 *
 * Historical: sprint γ' assumed all adapters at (layer, proj) share the
 * same rank slab. Crashed pod when 2nd adapter had different rank. See
 * project-xllm-v010-r11-binpack-2026-08-31 for the crash story.
 */
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// Rank is the sole shape discriminator for bucketing. hidden_in / hidden_out
// are derived per (layer, proj) at slab-allocation time.
struct RankBucketKey {
  int64_t rank = 0;

  bool operator==(const RankBucketKey& o) const noexcept {
    return rank == o.rank;
  }
};

struct RankBucketKeyHash {
  size_t operator()(const RankBucketKey& k) const noexcept {
    return std::hash<int64_t>()(k.rank);
  }
};

// Where a registered int_id lives. bucket_id is per-rank (assigned in
// registration order per storage-global bucket table). slot is bucket-local
// (0..kNMaxActive-1). rank is stored for shape verification and unregister.
struct AdapterLocation {
  int32_t bucket_id = -1;
  int32_t slot = -1;
  int64_t rank = 0;
};

class AscendCLoRAStorage {
 public:
  static constexpr int kNMaxActive = 8;
  // Cap on distinct ranks (buckets). Beyond this, register falls through
  // to baseline slow_path. Prod expects <= 4 distinct ranks.
  static constexpr int kMaxBuckets = 8;
  static constexpr int64_t kMaxIntId = 1024;

  static AscendCLoRAStorage& instance();

  // Register adapter tensors for (layer, proj) into a rank-bucket slot.
  // Returns slot on success, -1 on failure (bucket full, bucket cap hit,
  // shape inconsistency, etc.). Idempotent for same int_id + same rank.
  int register_adapter(uint64_t int_id,
                       int layer_idx,
                       const std::string& proj,
                       const torch::Tensor& A,
                       const torch::Tensor& B,
                       float scaling);

  // Remove int_id: free the slot in its rank bucket and zero the slab
  // entries across all (layer, proj) that stored this adapter.
  void unregister_adapter(uint64_t int_id);

  struct StackedView {
    torch::Tensor A_stacked;  // [N, R, H_in]
    torch::Tensor B_stacked;  // [N, H_out, R]
    float scaling = 1.0f;
    bool valid = false;
    int64_t version = 0;
  };

  // Per (layer, proj) holds one StackedView per rank bucket. bucket_of_rank
  // maps a rank value to its bucket_id (parallel to buckets vector).
  struct BucketedStackedView {
    std::vector<StackedView> buckets;     // indexed by bucket_id
    std::vector<int64_t> rank_of_bucket;  // parallel to buckets

    int num_buckets() const { return static_cast<int>(buckets.size()); }
  };

  // Fetch all rank buckets for a (layer, proj). Empty view if nothing
  // registered.
  BucketedStackedView get_bucketed(int layer_idx,
                                   const std::string& proj) const;

  int active_count() const;

  // Device-side lookup tables (Fix W generalized).
  //   slot_lookup_dev_   [kMaxIntId] int64 → slot in the bucket (0..N-1)
  //   bucket_lookup_dev_ [kMaxIntId] int64 → bucket_id (rank bucket)
  // Both indexed by int_id. int_id=0 or unregistered → -1 (sentinel).
  const torch::Tensor& slot_lookup_dev() const { return slot_lookup_dev_; }
  const torch::Tensor& bucket_lookup_dev() const { return bucket_lookup_dev_; }

  // Return the rank associated with a bucket_id (host-side, for wrapper
  // when it needs to allocate per-bucket buf tensors).
  int64_t bucket_rank(int32_t bucket_id) const;

 private:
  AscendCLoRAStorage() { int_id_to_location_.reserve(kMaxIntId); }

  void update_lookup_locked(uint64_t int_id,
                            int32_t bucket_id,
                            int32_t slot,
                            torch::Device device);
  void clear_lookup_locked(uint64_t int_id);

  // Find or create a global rank bucket. Returns bucket_id or -1 if
  // kMaxBuckets is exhausted.
  int find_or_create_rank_bucket_locked(int64_t rank);

  // Find the first unused slot in the given rank bucket. Scans
  // int_id_to_location_ for entries matching bucket_id. O(kMaxIntId).
  int find_free_slot_in_bucket_locked(int32_t bucket_id) const;

  // Ensure the (layer, proj, bucket_id) slab is allocated with the given
  // A/B tensor shape. If slab exists but shape mismatches, log and
  // return false (register_adapter will refuse the registration).
  bool ensure_slab_allocated_locked(int layer_idx,
                                    const std::string& proj,
                                    int32_t bucket_id,
                                    int64_t rank,
                                    const torch::Tensor& A,
                                    const torch::Tensor& B,
                                    float scaling);

  using KeyType = std::pair<int, std::string>;
  struct KeyHash {
    size_t operator()(const KeyType& k) const noexcept {
      return std::hash<int>()(k.first) ^
             (std::hash<std::string>()(k.second) << 1);
    }
  };

  std::unordered_map<uint64_t, AdapterLocation> int_id_to_location_;
  std::unordered_map<KeyType, BucketedStackedView, KeyHash> storage_;

  // Global rank bucket table (rank_of_bucket_[bucket_id] = rank).
  std::vector<int64_t> rank_of_bucket_;

  torch::Tensor slot_lookup_dev_;
  torch::Tensor bucket_lookup_dev_;

  mutable std::shared_mutex mu_;
};

}  // namespace xllm
