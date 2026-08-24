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

#include "lora_kv_dependency_tree.h"

#include <glog/logging.h>

#include <chrono>
#include <functional>
#include <sstream>
#include <utility>

namespace xllm {

namespace {

int64_t now_micros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Detach a child unique_ptr from its parent's children vector, transferring
// ownership out. Returns nullptr if `child` is not actually a child of
// `parent`.
std::unique_ptr<LoRAKVNode> detach_child(LoRAKVNode* parent,
                                         LoRAKVNode* child) {
  if (parent == nullptr || child == nullptr) return nullptr;
  for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
    if (it->get() == child) {
      std::unique_ptr<LoRAKVNode> out = std::move(*it);
      parent->children.erase(it);
      return out;
    }
  }
  return nullptr;
}

}  // namespace

// ----------------------------- Singleton -----------------------------

LoRAKVDependencyTree& LoRAKVDependencyTree::instance() {
  static LoRAKVDependencyTree kInstance;
  return kInstance;
}

LoRAKVDependencyTree::LoRAKVDependencyTree() {
  root_ = std::make_unique<LoRAKVNode>();
  root_->type = LoRAKVNode::Type::VirtualRoot;
  root_->label = "__virtual_root__";
  root_->residency = LoRAKVNode::Residency::HBM;
  LOG(INFO) << "[LoRAKVDepTree] virtual root created";
}

// ----------------------------- Hashing -----------------------------

uint64_t LoRAKVDependencyTree::str_hash(const std::string& s) {
  return std::hash<std::string>{}(s);
}

std::string LoRAKVDependencyTree::hex_preview(uint64_t h) {
  std::ostringstream oss;
  oss << std::hex << h;
  return oss.str();
}

// ----------------------------- LoRA lifecycle -----------------------------

void LoRAKVDependencyTree::register_lora(const std::string& adapter_id,
                                         int64_t block_id,
                                         size_t bytes) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = lora_index_.find(adapter_id);
  if (it != lora_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] register_lora: adapter '" << adapter_id
                 << "' already registered, ignoring";
    return;
  }
  auto node = std::make_unique<LoRAKVNode>();
  node->type = LoRAKVNode::Type::LoRA;
  node->label = adapter_id;
  node->label_hash = str_hash(adapter_id);
  node->block_id = block_id;
  node->node_bytes = bytes;
  node->last_use_us = now_micros();
  node->residency = LoRAKVNode::Residency::HBM;
  node->parent = root_.get();

  LoRAKVNode* raw = node.get();
  root_->child_index[node->label_hash] = raw;
  root_->children.push_back(std::move(node));
  lora_index_[adapter_id] = raw;

  LOG(INFO) << "[LoRAKVDepTree] register_lora id=" << adapter_id
            << " block=" << block_id << " bytes=" << bytes;
}

void LoRAKVDependencyTree::unregister_lora(const std::string& adapter_id) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = lora_index_.find(adapter_id);
  if (it == lora_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] unregister_lora: unknown adapter '"
                 << adapter_id << "'";
    return;
  }
  LoRAKVNode* node = it->second;
  // Remove any KV children from kv_index_ before dropping the subtree.
  size_t kv_removed = 0;
  for (const auto& child : node->children) {
    if (child->type == LoRAKVNode::Type::KVSubtree) {
      kv_index_.erase(child->label_hash);
      ++kv_removed;
    }
  }
  root_->child_index.erase(node->label_hash);
  detach_child(root_.get(), node);
  lora_index_.erase(it);

  LOG(INFO) << "[LoRAKVDepTree] unregister_lora id=" << adapter_id
            << " kv_children_removed=" << kv_removed;
}

// ----------------------------- KV lifecycle -----------------------------

void LoRAKVDependencyTree::register_kv(const std::string& active_adapter,
                                       uint64_t token_seq_hash,
                                       int64_t block_id,
                                       int32_t layer,
                                       size_t bytes) {
  if (active_adapter.empty()) {
    // No LoRA on this request -- Phase 1 policy: skip tree registration.
    return;
  }
  std::lock_guard<std::mutex> g(mu_);
  auto lit = lora_index_.find(active_adapter);
  if (lit == lora_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] register_kv: unknown active_adapter '"
                 << active_adapter << "', skipping KV attach hash=0x"
                 << hex_preview(token_seq_hash);
    return;
  }
  auto kit = kv_index_.find(token_seq_hash);
  if (kit != kv_index_.end()) {
    // KV entry with this hash already exists -- treat as a use and bump.
    LoRAKVNode* existing = kit->second;
    existing->visit_freq += 1;
    existing->last_use_us = now_micros();
    return;
  }
  LoRAKVNode* parent = lit->second;
  auto node = std::make_unique<LoRAKVNode>();
  node->type = LoRAKVNode::Type::KVSubtree;
  node->label = hex_preview(token_seq_hash);
  node->label_hash = token_seq_hash;
  node->block_id = block_id;
  node->block_layer = layer;
  node->node_bytes = bytes;
  node->visit_freq = 1;
  node->last_use_us = now_micros();
  node->residency = LoRAKVNode::Residency::HBM;
  node->parent = parent;

  LoRAKVNode* raw = node.get();
  parent->child_index[token_seq_hash] = raw;
  parent->children.push_back(std::move(node));
  kv_index_[token_seq_hash] = raw;

  VLOG(1) << "[LoRAKVDepTree] register_kv hash=0x"
          << hex_preview(token_seq_hash) << " lora=" << active_adapter
          << " block=" << block_id << " layer=" << layer << " bytes=" << bytes;
}

void LoRAKVDependencyTree::mark_kv_use(uint64_t token_seq_hash) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = kv_index_.find(token_seq_hash);
  if (it == kv_index_.end()) {
    return;
  }
  it->second->visit_freq += 1;
  it->second->last_use_us = now_micros();
}

void LoRAKVDependencyTree::unregister_kv(uint64_t token_seq_hash) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = kv_index_.find(token_seq_hash);
  if (it == kv_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] unregister_kv: unknown hash=0x"
                 << hex_preview(token_seq_hash);
    return;
  }
  LoRAKVNode* node = it->second;
  LoRAKVNode* parent = node->parent;
  if (parent != nullptr) {
    parent->child_index.erase(token_seq_hash);
    detach_child(parent, node);
  }
  kv_index_.erase(it);
}

// ----------------------------- Ref counting -----------------------------

void LoRAKVDependencyTree::inc_ref(const std::string& adapter_id) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = lora_index_.find(adapter_id);
  if (it == lora_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] inc_ref: unknown adapter '" << adapter_id
                 << "'";
    return;
  }
  it->second->ref_count += 1;
  it->second->last_use_us = now_micros();
}

void LoRAKVDependencyTree::dec_ref(const std::string& adapter_id) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = lora_index_.find(adapter_id);
  if (it == lora_index_.end()) {
    LOG(WARNING) << "[LoRAKVDepTree] dec_ref: unknown adapter '" << adapter_id
                 << "'";
    return;
  }
  if (it->second->ref_count <= 0) {
    LOG(WARNING) << "[LoRAKVDepTree] dec_ref: ref_count already 0 for '"
                 << adapter_id << "'";
    return;
  }
  it->second->ref_count -= 1;
}

// ----------------------------- Traversal -----------------------------

std::vector<LoRAKVNode*> LoRAKVDependencyTree::find_leaves_in_hbm() {
  std::vector<LoRAKVNode*> out;
  std::lock_guard<std::mutex> g(mu_);
  for (const auto& kv : kv_index_) {
    LoRAKVNode* n = kv.second;
    if (n->residency == LoRAKVNode::Residency::HBM && n->children.empty()) {
      out.push_back(n);
    }
  }
  return out;
}

std::vector<LoRAKVNode*> LoRAKVDependencyTree::find_roots_in_main_mem() {
  std::vector<LoRAKVNode*> out;
  std::lock_guard<std::mutex> g(mu_);
  for (const auto& kv : lora_index_) {
    LoRAKVNode* n = kv.second;
    if (n->residency == LoRAKVNode::Residency::MainMem) {
      out.push_back(n);
    }
  }
  return out;
}

// ----------------------------- Diagnostics -----------------------------

LoRAKVDependencyTree::Stats LoRAKVDependencyTree::get_stats() const {
  std::lock_guard<std::mutex> g(mu_);
  Stats s;
  s.n_lora = lora_index_.size();
  s.n_kv = kv_index_.size();
  for (const auto& kv : lora_index_) {
    const LoRAKVNode* n = kv.second;
    if (n->residency == LoRAKVNode::Residency::HBM) {
      s.n_hbm_bytes += n->node_bytes;
    } else if (n->residency == LoRAKVNode::Residency::MainMem) {
      s.n_main_bytes += n->node_bytes;
    }
  }
  for (const auto& kv : kv_index_) {
    const LoRAKVNode* n = kv.second;
    if (n->residency == LoRAKVNode::Residency::HBM) {
      s.n_hbm_bytes += n->node_bytes;
    } else if (n->residency == LoRAKVNode::Residency::MainMem) {
      s.n_main_bytes += n->node_bytes;
    }
  }
  return s;
}

std::string LoRAKVDependencyTree::dump_topology() const {
  std::lock_guard<std::mutex> g(mu_);
  std::ostringstream oss;
  oss << "[LoRAKVDepTree] topology:\n";
  oss << "  root (n_lora=" << lora_index_.size() << " n_kv=" << kv_index_.size()
      << ")\n";
  for (const auto& lora_child : root_->children) {
    const LoRAKVNode* lora = lora_child.get();
    const char* res =
        (lora->residency == LoRAKVNode::Residency::HBM)
            ? "HBM"
            : (lora->residency == LoRAKVNode::Residency::MainMem ? "MAIN"
                                                                 : "ABSENT");
    oss << "    LoRA[" << lora->label << "] block=" << lora->block_id
        << " bytes=" << lora->node_bytes << " ref=" << lora->ref_count
        << " freq=" << lora->visit_freq << " residency=" << res
        << " kv_children=" << lora->children.size() << "\n";
    // Print up to 4 KV children per LoRA to keep dump bounded.
    size_t printed = 0;
    for (const auto& kv_child : lora->children) {
      const LoRAKVNode* kv = kv_child.get();
      oss << "      KV[0x" << hex_preview(kv->label_hash)
          << "] layer=" << kv->block_layer << " block=" << kv->block_id
          << " bytes=" << kv->node_bytes << " freq=" << kv->visit_freq << "\n";
      if (++printed >= 4 && lora->children.size() > 4) {
        oss << "      ... (" << (lora->children.size() - 4) << " more)\n";
        break;
      }
    }
  }
  return oss.str();
}

}  // namespace xllm
