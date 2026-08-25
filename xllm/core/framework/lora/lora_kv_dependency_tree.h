/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// [FASTLIBRA Phase 1] LoRAKVDependencyTree
//
// A unified trie describing the LoRA <-> KV block dependency implied by
// active adapters serving live requests. Structure:
//
//   VirtualRoot
//     |
//     +-- LoRA(adapter_id="attn_dummy", residency=HBM, ref_count=2)
//     |     |
//     |     +-- KVSubtree(token_seq_hash=0xABCD, layer=0, block_id=42)
//     |     +-- KVSubtree(token_seq_hash=0xABCD, layer=1, block_id=43)
//     |     +-- ...
//     |
//     +-- LoRA(adapter_id="moe_zero", residency=MainMem, ref_count=0)
//           +-- KVSubtree(...)
//
// Invariant (Phase 2 will enforce): swap-out only from HBM *leaves*,
// swap-in only from main-mem *roots*. Because a KV node is always a child
// of the LoRA it was allocated under, this invariant guarantees that a KV
// block resident in HBM always has its owning LoRA also resident in HBM.
// This is the core FASTLIBRA property that eliminates the "invalid KV"
// class of pathology (paper reports vLLM saw 46.5% invalid KVs).
//
// Phase 1 scope: build the tree + wire hooks. No swap decisions, no
// eviction. Traversal APIs `find_leaves_in_hbm` / `find_roots_in_main_mem`
// are exposed as Phase 2 stubs; they return empty until Phase 2 wires
// residency transitions.

class LoRAKVNode {
 public:
  enum class Type : uint8_t {
    VirtualRoot = 0,
    LoRA = 1,
    KVSubtree = 2,
  };

  enum class Residency : uint8_t {
    HBM = 0,
    MainMem = 1,
    Absent = 2,
  };

  Type type = Type::VirtualRoot;
  std::string label;         // LoRA: adapter_id; KV: hex(hash) preview.
  uint64_t label_hash = 0;   // LoRA: str_hash; KV: token_seq_hash.
  int64_t block_id = -1;     // Backing block in its owning pool; -1 for root.
  int32_t block_layer = -1;  // KV only: layer index; -1 for LoRA nodes.
  size_t node_bytes = 0;

  // Usage stats (Phase 2 cost model input).
  size_t visit_freq = 0;
  int64_t last_use_us = 0;
  int ref_count = 0;

  Residency residency = Residency::HBM;

  // Tree links: children own their storage, parent is a back-reference.
  LoRAKVNode* parent = nullptr;
  std::vector<std::unique_ptr<LoRAKVNode>> children;

  // Fast child lookup by label_hash. Pointers alias into `children`.
  std::unordered_map<uint64_t, LoRAKVNode*> child_index;
};

class LoRAKVDependencyTree {
 public:
  // Process-wide singleton, mirrors LoRARuntime instance pattern.
  static LoRAKVDependencyTree& instance();

  LoRAKVDependencyTree(const LoRAKVDependencyTree&) = delete;
  LoRAKVDependencyTree& operator=(const LoRAKVDependencyTree&) = delete;

  // ------------------- LoRA lifecycle -------------------
  // Called from LoRARuntime install path. block_id is the first block in
  // the pool run backing this adapter; bytes is the total slab bytes.
  // Idempotent: repeat registration of the same adapter_id is a WARN log.
  void register_lora(const std::string& adapter_id,
                     int64_t block_id,
                     size_t bytes);
  // Called from LoRARuntime unload path. Removes the LoRA node and all
  // KV children hanging under it. No-op with WARN if unknown.
  void unregister_lora(const std::string& adapter_id);

  // ------------------- KV lifecycle -------------------
  // Called from LLMWorkerImpl KV write path. active_adapter=="" means the
  // request is not using LoRA, in which case the KV node is NOT registered
  // in the tree (see design note in project memory).
  void register_kv(const std::string& active_adapter,
                   uint64_t token_seq_hash,
                   int64_t block_id,
                   int32_t layer,
                   size_t bytes);
  // Bump usage stats on a KV node. No-op if unknown.
  void mark_kv_use(uint64_t token_seq_hash);
  // Remove a KV node. No-op with WARN if unknown.
  void unregister_kv(uint64_t token_seq_hash);

  // ------------------- Reference counting -------------------
  // Called from ContinuousScheduler admit_request / finish_request. Only
  // valid for adapter_id != "". Unknown adapter_id is a WARN.
  void inc_ref(const std::string& adapter_id);
  void dec_ref(const std::string& adapter_id);

  // Read ref_count for the given adapter. Returns -1 if unknown.
  // Used by LoRASwapManager to skip evicting active adapters.
  int get_ref_count(const std::string& adapter_id) const;

  // ------------------- Traversal (Phase 2 will consume) -------------------
  // Eviction candidates: KV leaves currently in HBM.
  std::vector<LoRAKVNode*> find_leaves_in_hbm();
  // Swap-in candidates: LoRA roots currently in main memory.
  std::vector<LoRAKVNode*> find_roots_in_main_mem();

  // ------------------- Diagnostics -------------------
  struct Stats {
    size_t n_lora = 0;
    size_t n_kv = 0;
    size_t n_hbm_bytes = 0;
    size_t n_main_bytes = 0;
  };
  Stats get_stats() const;
  // Multi-line indented dump of the current topology. For manual review
  // during Phase 1 verify; not stable output for parsing.
  std::string dump_topology() const;

 private:
  LoRAKVDependencyTree();
  ~LoRAKVDependencyTree() = default;

  static uint64_t str_hash(const std::string& s);
  static std::string hex_preview(uint64_t h);

  // Must be held for any tree mutation or traversal.
  mutable std::mutex mu_;

  std::unique_ptr<LoRAKVNode> root_;  // Type::VirtualRoot, owned.
  std::unordered_map<std::string, LoRAKVNode*> lora_index_;
  std::unordered_map<uint64_t, LoRAKVNode*> kv_index_;
};

}  // namespace xllm
