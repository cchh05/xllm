#pragma once

// Hidden-state dump helper for backend-vs-backend layer diff.
//
// Enable at runtime with env var XLLM_DUMP_LAYER=<dir>. When set:
//   - For each layer i, appends one JSON line to <dir>/<backend>_summary.jsonl
//     with (i, shape, dtype, device, mean, std, absmax, nan_count, first32).
//   - If XLLM_DUMP_LAYER_FULL=1 is also set, additionally saves the full CPU
//     tensor to <dir>/<backend>_layer_<i:02d>.pt (torch::save).
//
// Off (env unset) it is truly zero cost: single getenv on first call, cached.

#include <torch/torch.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace xllm::layer_dump {

struct Config {
  bool enabled = false;
  bool full_tensor = false;
  std::string dir;
};

inline const Config& config() {
  static Config cfg = []() {
    Config c;
    const char* d = std::getenv("XLLM_DUMP_LAYER");
    if (d && *d) {
      c.enabled = true;
      c.dir = d;
    }
    const char* f = std::getenv("XLLM_DUMP_LAYER_FULL");
    if (f && *f && std::strcmp(f, "0") != 0) c.full_tensor = true;
    return c;
  }();
  return cfg;
}

inline std::mutex& mu() {
  static std::mutex m;
  return m;
}

inline void dump_layer(const std::string& backend,
                       int32_t layer_index,
                       const torch::Tensor& h) {
  const auto& cfg = config();
  if (!cfg.enabled) return;
  if (!h.defined()) return;

  // Move to CPU float for stats to avoid NPU-side scalar ops.
  torch::Tensor h_cpu;
  try {
    h_cpu = h.detach().to(torch::kCPU).to(torch::kFloat32);
  } catch (const std::exception& e) {
    std::lock_guard g(mu());
    std::ofstream out(cfg.dir + "/" + backend + "_summary.jsonl",
                      std::ios::app);
    out << "{\"layer\":" << layer_index << ",\"error\":\"" << e.what()
        << "\"}\n";
    return;
  }

  const auto sizes = h_cpu.sizes();
  const int64_t n = h_cpu.numel();
  double mean = h_cpu.mean().item<double>();
  double std = h_cpu.std(false).item<double>();
  double absmax = h_cpu.abs().max().item<double>();
  int64_t nan_count = torch::isnan(h_cpu).sum().item<int64_t>();

  auto flat = h_cpu.flatten();
  const int64_t k = std::min<int64_t>(32, flat.numel());
  auto head = flat.slice(0, 0, k);

  std::ostringstream shape_ss;
  shape_ss << "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    shape_ss << sizes[i] << (i + 1 == sizes.size() ? "" : ",");
  }
  shape_ss << "]";

  std::ostringstream head_ss;
  head_ss << "[";
  for (int64_t i = 0; i < k; ++i) {
    head_ss << head[i].item<float>() << (i + 1 == k ? "" : ",");
  }
  head_ss << "]";

  std::lock_guard g(mu());
  std::ofstream out(cfg.dir + "/" + backend + "_summary.jsonl", std::ios::app);
  out << "{\"layer\":" << layer_index << ",\"shape\":" << shape_ss.str()
      << ",\"numel\":" << n << ",\"dtype\":\"" << torch::toString(h.dtype())
      << "\""
      << ",\"device\":\"" << h.device().str() << "\""
      << ",\"mean\":" << mean << ",\"std\":" << std << ",\"absmax\":" << absmax
      << ",\"nan_count\":" << nan_count << ",\"first32\":" << head_ss.str()
      << "}\n";

  if (cfg.full_tensor) {
    std::ostringstream fname;
    fname << cfg.dir << "/" << backend << "_layer_";
    fname.width(2);
    fname.fill(0);
    fname << layer_index;
    fname << ".pt";
    try {
      auto to_save = h_cpu.contiguous();
      torch::save(to_save, fname.str());
    } catch (const std::exception& e) {
      std::ofstream err(cfg.dir + "/" + backend + "_summary.jsonl",
                        std::ios::app);
      err << "{\"layer\":" << layer_index << ",\"save_error\":\"" << e.what()
          << "\"}\n";
    }
  }
}

}  // namespace xllm::layer_dump
