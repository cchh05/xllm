/*
 * Copyright (c) 2026 xllm authors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * xllm_lora_ascendc_binding.cpp — torch_library binding for vllm-ascend
 * AscendC bgmv/sgmv LoRA kernels. Loads libvllm_ascend_kernels.so at
 * first-call via dlopen (kernel .so must be on the runtime linker path;
 * production packaging exports LD_LIBRARY_PATH before launch).
 *
 * Kernel argtype tables (from vllm-ascend 0.11.0+, verified via ctypes
 * PoC 2026-08-27):
 *   bgmv_shrink_impl:
 *     (dtype, stream, x, w, idx, idx_size, y,
 *      batch, tpc, hidden, rank, scale) — 12 params
 *   bgmv_expand_impl:
 *     (dtype, stream, x, w, idx, idx_size, y, y_out,
 *      batch, tpc, rank, slice_size, slice_offset, full_dim) — 14 params
 *   sgmv_shrink_impl:
 *     (dtype, stream, x, w, lora_idx, lora_idx_size, seq_len, seq_len_size,
 *      y, batch, tpc, hidden, rank, scale) — 14 params
 *   sgmv_expand_impl:
 *     (dtype, stream, x, w, lora_idx, lora_idx_size, seq_len, seq_len_size,
 *      y, y_out, batch, tpc, rank, slice_size, slice_offset, full_dim) — 16
 */

#include <acl/acl.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <mutex>

namespace xllm {

// enum AscendType {FP16=0, BF16=1, FP32=2} — from vllm-ascend types.h
enum AscendType : int { AT_FP16 = 0, AT_BF16 = 1, AT_FP32 = 2 };

using BgmvShrinkFn = void (*)(int dtype,
                              void* stream,
                              void* x,
                              void* w,
                              void* idx,
                              unsigned int idx_size,
                              void* y,
                              unsigned int batch,
                              unsigned int tpc,
                              unsigned int hidden,
                              unsigned int rank,
                              float scale);

using BgmvExpandFn = void (*)(int dtype,
                              void* stream,
                              void* x,
                              void* w,
                              void* idx,
                              unsigned int idx_size,
                              void* y,
                              void* y_out,
                              unsigned int batch,
                              unsigned int tpc,
                              unsigned int rank,
                              unsigned int slice_size,
                              unsigned int slice_offset,
                              unsigned int full_dim);

using SgmvShrinkFn = void (*)(int dtype,
                              void* stream,
                              void* x,
                              void* w,
                              void* lora_idx,
                              unsigned int lora_idx_size,
                              void* seq_len,
                              unsigned int seq_len_size,
                              void* y,
                              unsigned int batch,
                              unsigned int tpc,
                              unsigned int hidden,
                              unsigned int rank,
                              float scale);

using SgmvExpandFn = void (*)(int dtype,
                              void* stream,
                              void* x,
                              void* w,
                              void* lora_idx,
                              unsigned int lora_idx_size,
                              void* seq_len,
                              unsigned int seq_len_size,
                              void* y,
                              void* y_out,
                              unsigned int batch,
                              unsigned int tpc,
                              unsigned int rank,
                              unsigned int slice_size,
                              unsigned int slice_offset,
                              unsigned int full_dim);

namespace {

constexpr const char* kBgmvShrinkSym =
    "_ZN11vllm_ascend16bgmv_shrink_implENS_10AscendTypeEPvS1_S1_S1_jS1_jjjjf";
constexpr const char* kBgmvExpandSym =
    "_ZN11vllm_ascend16bgmv_expand_implENS_10AscendTypeEPvS1_S1_S1_jS1_S1_"
    "jjjjjj";
constexpr const char* kSgmvShrinkSym =
    "_ZN11vllm_ascend16sgmv_shrink_implENS_10AscendTypeEPvS1_S1_S1_jS1_jS1_"
    "jjjjf";
constexpr const char* kSgmvExpandSym =
    "_ZN11vllm_ascend16sgmv_expand_implENS_10AscendTypeEPvS1_S1_S1_jS1_jS1_S1_"
    "jjjjjj";

struct KernelSymbols {
  void* handle = nullptr;
  BgmvShrinkFn bgmv_shrink = nullptr;
  BgmvExpandFn bgmv_expand = nullptr;
  SgmvShrinkFn sgmv_shrink = nullptr;
  SgmvExpandFn sgmv_expand = nullptr;
};

KernelSymbols& syms() {
  static KernelSymbols s;
  return s;
}

std::once_flag load_once;

void load_kernels() {
  auto& s = syms();
  // Try candidate paths for libvllm_ascend_kernels.so.
  const char* env_path = std::getenv("VLLM_ASCEND_KERNELS_SO");
  const char* candidates[] = {
      env_path ? env_path : "",
      "/usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend/"
      "libvllm_ascend_kernels.so",
      "libvllm_ascend_kernels.so",  // rely on LD_LIBRARY_PATH
  };
  for (const auto* p : candidates) {
    if (p && *p) {
      s.handle = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
      if (s.handle) {
        LOG(INFO) << "[AscendCLoRA] dlopen " << p << " OK";
        break;
      }
    }
  }
  TORCH_CHECK(s.handle,
              "[AscendCLoRA] failed to dlopen libvllm_ascend_kernels.so; set "
              "VLLM_ASCEND_KERNELS_SO or LD_LIBRARY_PATH. dlerror=",
              dlerror() ? dlerror() : "unknown");

  s.bgmv_shrink =
      reinterpret_cast<BgmvShrinkFn>(dlsym(s.handle, kBgmvShrinkSym));
  s.bgmv_expand =
      reinterpret_cast<BgmvExpandFn>(dlsym(s.handle, kBgmvExpandSym));
  s.sgmv_shrink =
      reinterpret_cast<SgmvShrinkFn>(dlsym(s.handle, kSgmvShrinkSym));
  s.sgmv_expand =
      reinterpret_cast<SgmvExpandFn>(dlsym(s.handle, kSgmvExpandSym));
  TORCH_CHECK(s.bgmv_shrink && s.bgmv_expand && s.sgmv_shrink && s.sgmv_expand,
              "[AscendCLoRA] dlsym missing some symbol (0.11.0+ required)");
}

void ensure_loaded() { std::call_once(load_once, load_kernels); }

int torch_dtype_to_ascend(at::ScalarType t) {
  switch (t) {
    case torch::kFloat16:
      return AT_FP16;
    case torch::kBFloat16:
      return AT_BF16;
    case torch::kFloat32:
      return AT_FP32;
    default:
      TORCH_CHECK(false, "AscendCLoRA: unsupported dtype ", t);
  }
}

unsigned int tokens_per_core(int64_t batch) {
  static int64_t cached_aiv = 0;
  if (cached_aiv <= 0) {
    aclGetDeviceCapability(0, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cached_aiv);
    if (cached_aiv <= 0) cached_aiv = 48;  // A3 default
  }
  return static_cast<unsigned int>(
      std::max<int64_t>(1, (batch + cached_aiv - 1) / cached_aiv));
}

void* current_stream() { return c10_npu::getCurrentNPUStream().stream(); }

}  // namespace

// -- Op wrappers ---------------------------------------------------------

void bgmv_shrink(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor indices,
                 torch::Tensor y,
                 double scale) {
  ensure_loaded();
  TORCH_CHECK(x.dim() == 2, "x must be [B, H_in]");
  TORCH_CHECK(w.dim() == 3 || w.dim() == 4,
              "w must be [N, R, H_in] or [N, 1, R, H_in]");
  TORCH_CHECK(y.dim() == 2, "y must be [B, R] fp32");
  TORCH_CHECK(indices.dim() == 1 && indices.size(0) == x.size(0),
              "indices must be [B] and match x.size(0)");
  auto dtype = torch_dtype_to_ascend(x.scalar_type());
  auto stream = current_stream();
  const unsigned int B = static_cast<unsigned int>(x.size(0));
  const unsigned int H = static_cast<unsigned int>(x.size(1));
  const unsigned int R = static_cast<unsigned int>(y.size(1));
  syms().bgmv_shrink(dtype,
                     stream,
                     x.data_ptr(),
                     w.data_ptr(),
                     indices.data_ptr(),
                     B,
                     y.data_ptr(),
                     B,
                     tokens_per_core(B),
                     H,
                     R,
                     static_cast<float>(scale));
}

void bgmv_expand(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor indices,
                 torch::Tensor y,
                 int64_t slice_offset,
                 int64_t slice_size) {
  ensure_loaded();
  TORCH_CHECK(x.dim() == 2, "x must be [B, R] fp32");
  TORCH_CHECK(w.dim() == 3 || w.dim() == 4,
              "w must be [N, H_out, R] or [N, 1, H_out, R]");
  TORCH_CHECK(y.dim() == 2, "y must be [B, y_size]");
  TORCH_CHECK(indices.dim() == 1 && indices.size(0) == x.size(0),
              "indices must be [B]");
  auto dtype = torch_dtype_to_ascend(y.scalar_type());
  auto stream = current_stream();
  const unsigned int B = static_cast<unsigned int>(x.size(0));
  const unsigned int R = static_cast<unsigned int>(x.size(1));
  const unsigned int full_dim = static_cast<unsigned int>(y.size(1));
  syms().bgmv_expand(dtype,
                     stream,
                     x.data_ptr(),
                     w.data_ptr(),
                     indices.data_ptr(),
                     B,
                     y.data_ptr(),
                     y.data_ptr(),
                     B,
                     tokens_per_core(B),
                     R,
                     static_cast<unsigned int>(slice_size),
                     static_cast<unsigned int>(slice_offset),
                     full_dim);
}

void sgmv_shrink(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor lora_indices,
                 torch::Tensor seq_lens,
                 torch::Tensor y,
                 double scale) {
  ensure_loaded();
  auto dtype = torch_dtype_to_ascend(x.scalar_type());
  auto stream = current_stream();
  const unsigned int T = static_cast<unsigned int>(x.size(0));
  const unsigned int H = static_cast<unsigned int>(x.size(1));
  const unsigned int R = static_cast<unsigned int>(y.size(1));
  syms().sgmv_shrink(dtype,
                     stream,
                     x.data_ptr(),
                     w.data_ptr(),
                     lora_indices.data_ptr(),
                     static_cast<unsigned int>(lora_indices.size(0)),
                     seq_lens.data_ptr(),
                     static_cast<unsigned int>(seq_lens.size(0)),
                     y.data_ptr(),
                     T,
                     tokens_per_core(T),
                     H,
                     R,
                     static_cast<float>(scale));
}

void sgmv_expand(torch::Tensor x,
                 torch::Tensor w,
                 torch::Tensor lora_indices,
                 torch::Tensor seq_lens,
                 torch::Tensor y,
                 int64_t slice_offset,
                 int64_t slice_size) {
  ensure_loaded();
  auto dtype = torch_dtype_to_ascend(y.scalar_type());
  auto stream = current_stream();
  const unsigned int T = static_cast<unsigned int>(x.size(0));
  const unsigned int R = static_cast<unsigned int>(x.size(1));
  const unsigned int full_dim = static_cast<unsigned int>(y.size(1));
  syms().sgmv_expand(dtype,
                     stream,
                     x.data_ptr(),
                     w.data_ptr(),
                     lora_indices.data_ptr(),
                     static_cast<unsigned int>(lora_indices.size(0)),
                     seq_lens.data_ptr(),
                     static_cast<unsigned int>(seq_lens.size(0)),
                     y.data_ptr(),
                     y.data_ptr(),
                     T,
                     tokens_per_core(T),
                     R,
                     static_cast<unsigned int>(slice_size),
                     static_cast<unsigned int>(slice_offset),
                     full_dim);
}

TORCH_LIBRARY(_C_xllm_lora, m) {
  m.def(
      "bgmv_shrink(Tensor! x, Tensor! w, Tensor! indices, Tensor! y, float "
      "scale) -> ()");
  m.def(
      "bgmv_expand(Tensor! x, Tensor! w, Tensor! indices, Tensor! y, int "
      "slice_offset, int slice_size) -> ()");
  m.def(
      "sgmv_shrink(Tensor! x, Tensor! w, Tensor! lora_indices, Tensor! "
      "seq_lens, Tensor! y, float scale) -> ()");
  m.def(
      "sgmv_expand(Tensor! x, Tensor! w, Tensor! lora_indices, Tensor! "
      "seq_lens, Tensor! y, int slice_offset, int slice_size) -> ()");
}

TORCH_LIBRARY_IMPL(_C_xllm_lora, PrivateUse1, m) {
  m.impl("bgmv_shrink", &bgmv_shrink);
  m.impl("bgmv_expand", &bgmv_expand);
  m.impl("sgmv_shrink", &sgmv_shrink);
  m.impl("sgmv_expand", &sgmv_expand);
}

TORCH_LIBRARY_IMPL(_C_xllm_lora, CPU, m) {
  // Intentionally empty. No CPU impl: production forward path only routes
  // NPU tensors through _C_xllm_lora ops; a CPU stub adds no coverage but
  // pulls in torch 2.9 vs 2.10 boxed-function ABI differences we don't
  // want to track. Torch dispatcher will raise unimplemented-kernel on
  // any CPU call, which is the correct behavior.
}

}  // namespace xllm

// Linker anchor. TORCH_LIBRARY / TORCH_LIBRARY_IMPL register the ops via
// static ctors in this translation unit, but with nothing else pulling
// binding.o into the final link the whole TU is eligible for
// --gc-sections dead-strip and the ops silently disappear
// (_jit_get_operation returns (None, None)). Exposing this extern "C" symbol
// gives lora_runtime.cpp a link-time reference to hold binding.o alive.
extern "C" void _xllm_c_xllm_lora_anchor() {}
