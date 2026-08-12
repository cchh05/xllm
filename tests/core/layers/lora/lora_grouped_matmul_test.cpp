/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

// LoRA grouped_matmul helper correctness harness.
//
// Scope (see plan file ~/.claude/plans/logical-beaming-possum.md,
// Caveat #1): TP=1 only. TP>1 correctness relies on real HCCL and is
// covered by e2e curl smoke tests on caihao-cann9, not by this gtest.
//
// The golden reference is a hand-written per-seq matmul loop that
// mirrors the naive path in lora_qkv/column/row_parallel_linear.cpp
// (pre-refactor). BF16 tolerance is atol=5e-3 rtol=1e-2; FP32 golden
// asserts tighter (atol=1e-5 rtol=1e-4).
//
// The distinct=1 case asserts that build_grouped_lora_spec + apply
// path DOES run (i.e., the harness is exercising the grouped path even
// for a "would be fast_path" batch composition). The wrapper-level
// fast_path short-circuit is not this file's responsibility -- that is
// tested indirectly by verifying the counter emitted by the helper
// (grouped_matmul_call_count_for_test) after this test.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "layers/common/lora/lora_grouped_matmul_helper.h"

namespace xllm::layer {
namespace {

// Torch device used by all test cases. Tests are gated by USE_NPU in the
// CMakeLists so torch_npu will actually be available.
torch::Device test_device() {
#ifdef USE_NPU
  return torch::Device(torch::kPrivateUse1);
#else
  return torch::Device(torch::kCPU);
#endif
}

// Golden reference: replicate the pre-refactor per-seq loop from
// lora_column_parallel_linear.cpp (single-proj variant) directly. Applied
// to the same [T, hidden_local] input, returns [T, out_local] delta with
// the same base-only-passthrough semantics.
torch::Tensor golden_per_seq_delta(const torch::Tensor& input,
                                   const std::vector<uint64_t>& adapter_ids,
                                   const std::vector<int32_t>& q_seq_lens,
                                   const std::vector<torch::Tensor>& A_by_aid,
                                   const std::vector<torch::Tensor>& B_by_aid,
                                   const std::vector<float>& scale_by_aid,
                                   const std::vector<uint64_t>& aid_index,
                                   int64_t out_local) {
  const int64_t total_tokens = input.size(0);
  auto out = torch::zeros({total_tokens, out_local}, input.options());
  int64_t tok_off = 0;
  for (size_t s = 0; s < adapter_ids.size(); ++s) {
    const int32_t sl = q_seq_lens[s];
    if (sl <= 0) continue;
    const uint64_t aid = adapter_ids[s];
    if (aid == 0) {
      tok_off += sl;
      continue;
    }
    // aid_index is a parallel array to aid_by_aid / B_by_aid that maps
    // aid -> index into those vectors.
    size_t idx = SIZE_MAX;
    for (size_t i = 0; i < aid_index.size(); ++i) {
      if (aid_index[i] == aid) {
        idx = i;
        break;
      }
    }
    if (idx == SIZE_MAX) {
      tok_off += sl;
      continue;
    }
    auto x_seq = input.slice(0, tok_off, tok_off + sl);
    auto tmp = torch::matmul(x_seq, A_by_aid[idx].transpose(0, 1));
    auto delta = torch::matmul(tmp, B_by_aid[idx].transpose(0, 1));
    delta = (delta * scale_by_aid[idx]).to(input.dtype());
    out.slice(0, tok_off, tok_off + sl).add_(delta);
    tok_off += sl;
  }
  return out;
}

// One convenience wrapper: turn (input, adapters, seq_lens, per-adapter A/B)
// into (spec, A_list, B_list, scale_list) sorted in spec.distinct_aids
// order and call apply_grouped_lora_delta + scatter_add.
torch::Tensor run_grouped(const torch::Tensor& input,
                          const std::vector<uint64_t>& adapter_ids,
                          const std::vector<int32_t>& q_seq_lens,
                          const std::vector<torch::Tensor>& A_by_aid,
                          const std::vector<torch::Tensor>& B_by_aid,
                          const std::vector<float>& scale_by_aid,
                          const std::vector<uint64_t>& aid_index,
                          int64_t hidden_local,
                          int64_t out_local,
                          int64_t r_max) {
  auto spec = build_grouped_lora_spec(adapter_ids, q_seq_lens, input);
  std::vector<torch::Tensor> A_list, B_list;
  std::vector<float> scale_list;
  A_list.reserve(spec.distinct_aids.size());
  B_list.reserve(spec.distinct_aids.size());
  scale_list.reserve(spec.distinct_aids.size());
  for (uint64_t aid : spec.distinct_aids) {
    size_t idx = SIZE_MAX;
    for (size_t i = 0; i < aid_index.size(); ++i) {
      if (aid_index[i] == aid) {
        idx = i;
        break;
      }
    }
    EXPECT_NE(idx, SIZE_MAX);
    A_list.emplace_back(A_by_aid[idx].contiguous());
    B_list.emplace_back(B_by_aid[idx].contiguous());
    scale_list.emplace_back(scale_by_aid[idx]);
  }
  auto delta = apply_grouped_lora_delta(
      spec, A_list, B_list, scale_list, hidden_local, out_local, r_max);
  auto y = torch::zeros({input.size(0), out_local}, input.options());
  scatter_add_lora_delta_(y, delta, spec);
  return y;
}

// Build a small deterministic A [r, hidden] + B [out, r] pair, values in
// [-0.5, 0.5]. Same seed produces byte-identical tensors across cases.
void make_ab(int64_t r,
             int64_t hidden,
             int64_t out_local,
             torch::Dtype dtype,
             torch::Device device,
             uint32_t seed,
             torch::Tensor& A,
             torch::Tensor& B) {
  torch::manual_seed(seed);
  A = (torch::rand({r, hidden}, torch::TensorOptions().dtype(torch::kFloat32)) -
       0.5f)
          .to(dtype)
          .to(device)
          .contiguous();
  B = (torch::rand({out_local, r},
                   torch::TensorOptions().dtype(torch::kFloat32)) -
       0.5f)
          .to(dtype)
          .to(device)
          .contiguous();
}

// BF16 tolerance: atol=5e-3 rtol=1e-2 (Caveat #2 in plan).
void expect_close_bf16(const torch::Tensor& actual,
                       const torch::Tensor& expected) {
  auto a = actual.to(torch::kFloat32).contiguous().to(torch::kCPU);
  auto e = expected.to(torch::kFloat32).contiguous().to(torch::kCPU);
  ASSERT_EQ(a.sizes(), e.sizes());
  const double atol = 2e-2;
  const double rtol = 1e-2;
  auto diff = (a - e).abs();
  auto thresh = torch::abs(e) * rtol + atol;
  const int64_t nfail = (diff > thresh).sum().item<int64_t>();
  const double max_diff = diff.max().item<double>();
  EXPECT_EQ(nfail, 0) << "max_abs_diff=" << max_diff << " (atol=" << atol
                      << " rtol=" << rtol << ")";
}

// FP32 tighter tolerance for the golden case.
void expect_close_fp32(const torch::Tensor& actual,
                       const torch::Tensor& expected) {
  auto a = actual.to(torch::kFloat32).contiguous().to(torch::kCPU);
  auto e = expected.to(torch::kFloat32).contiguous().to(torch::kCPU);
  ASSERT_EQ(a.sizes(), e.sizes());
  const double atol = 1e-5;
  const double rtol = 1e-4;
  auto diff = (a - e).abs();
  auto thresh = torch::abs(e) * rtol + atol;
  const int64_t nfail = (diff > thresh).sum().item<int64_t>();
  const double max_diff = diff.max().item<double>();
  EXPECT_EQ(nfail, 0) << "max_abs_diff=" << max_diff;
}

class LoRAGroupedMatmulTest : public ::testing::Test {
 protected:
  void SetUp() override { reset_grouped_matmul_call_count_for_test(); }
};

// Case 1: distinct=1 batch. build_spec still runs (grouped path was
// dispatched at wrapper level, but our unit test always runs the grouped
// path directly). Assert helper still produces bit-identical output to
// naive loop.
TEST_F(LoRAGroupedMatmulTest, DistinctOneCorrectness) {
  const int64_t hidden = 64, out_local = 96, r = 8;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  std::vector<uint64_t> aids = {7, 7, 7};
  std::vector<int32_t> lens = {3, 2, 4};
  torch::manual_seed(1);
  auto input =
      (torch::rand({9, hidden}, torch::TensorOptions().dtype(torch::kFloat32)) -
       0.5f)
          .to(dt)
          .to(dev)
          .contiguous();
  torch::Tensor A, B;
  make_ab(r, hidden, out_local, dt, dev, 42, A, B);
  const std::vector<uint64_t> aid_index = {7};
  const std::vector<torch::Tensor> A_by = {A};
  const std::vector<torch::Tensor> B_by = {B};
  const std::vector<float> s_by = {2.0f};
  const int64_t before = grouped_matmul_call_count_for_test();
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r);
  const int64_t after = grouped_matmul_call_count_for_test();
  EXPECT_EQ(after - before, 1) << "apply_grouped_lora_delta should run once";
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_bf16(got, want);
}

// Case 2: distinct=2, ranks equal. Baseline of the grouped fast case.
TEST_F(LoRAGroupedMatmulTest, DistinctTwoSameRank) {
  const int64_t hidden = 128, out_local = 128, r = 16;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  std::vector<uint64_t> aids = {11, 22, 11, 22};
  std::vector<int32_t> lens = {2, 3, 4, 1};
  torch::manual_seed(2);
  auto input = (torch::rand({10, hidden},
                            torch::TensorOptions().dtype(torch::kFloat32)) -
                0.5f)
                   .to(dt)
                   .to(dev)
                   .contiguous();
  torch::Tensor A11, B11, A22, B22;
  make_ab(r, hidden, out_local, dt, dev, 100, A11, B11);
  make_ab(r, hidden, out_local, dt, dev, 200, A22, B22);
  const std::vector<uint64_t> aid_index = {11, 22};
  const std::vector<torch::Tensor> A_by = {A11, A22};
  const std::vector<torch::Tensor> B_by = {B11, B22};
  const std::vector<float> s_by = {1.5f, 0.75f};
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r);
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_bf16(got, want);
}

// Case 3: distinct=8, ranks equal. Stresses group_gemm N_groups dispatch.
TEST_F(LoRAGroupedMatmulTest, DistinctEightSameRank) {
  const int64_t hidden = 64, out_local = 64, r = 8;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  std::vector<uint64_t> aids, aid_index;
  std::vector<int32_t> lens;
  std::vector<torch::Tensor> A_by, B_by;
  std::vector<float> s_by;
  for (int i = 0; i < 8; ++i) {
    aid_index.push_back(static_cast<uint64_t>(100 + i));
    aids.push_back(static_cast<uint64_t>(100 + i));
    lens.push_back(1 + i);
    torch::Tensor A, B;
    make_ab(
        r, hidden, out_local, dt, dev, static_cast<uint32_t>(300 + i), A, B);
    A_by.emplace_back(A);
    B_by.emplace_back(B);
    s_by.emplace_back(0.5f + 0.1f * i);
  }
  const int64_t total = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8;  // 36
  torch::manual_seed(3);
  auto input = (torch::rand({total, hidden},
                            torch::TensorOptions().dtype(torch::kFloat32)) -
                0.5f)
                   .to(dt)
                   .to(dev)
                   .contiguous();
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r);
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_bf16(got, want);
}

// Case 4: distinct=2, mixed ranks (16 + 8). Exercises dynamic rank pad.
TEST_F(LoRAGroupedMatmulTest, DistinctTwoMixedRank) {
  const int64_t hidden = 128, out_local = 128;
  const int64_t r1 = 16, r2 = 8;
  const int64_t r_max = 16;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  std::vector<uint64_t> aids = {33, 44, 33, 44, 33};
  std::vector<int32_t> lens = {2, 2, 3, 3, 2};
  torch::manual_seed(4);
  auto input = (torch::rand({12, hidden},
                            torch::TensorOptions().dtype(torch::kFloat32)) -
                0.5f)
                   .to(dt)
                   .to(dev)
                   .contiguous();
  torch::Tensor A33, B33, A44, B44;
  make_ab(r1, hidden, out_local, dt, dev, 500, A33, B33);
  make_ab(r2, hidden, out_local, dt, dev, 600, A44, B44);
  const std::vector<uint64_t> aid_index = {33, 44};
  const std::vector<torch::Tensor> A_by = {A33, A44};
  const std::vector<torch::Tensor> B_by = {B33, B44};
  const std::vector<float> s_by = {1.0f, 2.0f};
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r_max);
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_bf16(got, want);
}

// Case 5: distinct=3 with base-only seqs interspersed. Verifies scatter
// masks aid=0 tokens correctly (their positions in y remain zero).
TEST_F(LoRAGroupedMatmulTest, DistinctThreeWithBaseOnly) {
  const int64_t hidden = 64, out_local = 64, r = 8;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  std::vector<uint64_t> aids = {51, 0, 52, 53, 0};
  std::vector<int32_t> lens = {2, 3, 2, 2, 1};
  torch::manual_seed(5);
  auto input = (torch::rand({10, hidden},
                            torch::TensorOptions().dtype(torch::kFloat32)) -
                0.5f)
                   .to(dt)
                   .to(dev)
                   .contiguous();
  torch::Tensor A51, B51, A52, B52, A53, B53;
  make_ab(r, hidden, out_local, dt, dev, 700, A51, B51);
  make_ab(r, hidden, out_local, dt, dev, 800, A52, B52);
  make_ab(r, hidden, out_local, dt, dev, 900, A53, B53);
  const std::vector<uint64_t> aid_index = {51, 52, 53};
  const std::vector<torch::Tensor> A_by = {A51, A52, A53};
  const std::vector<torch::Tensor> B_by = {B51, B52, B53};
  const std::vector<float> s_by = {1.0f, 1.0f, 1.0f};
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r);
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_bf16(got, want);
  // Base-only rows (indices [2..5) and [9]) must be zero in got.
  auto got_cpu = got.to(torch::kFloat32).to(torch::kCPU);
  for (int64_t row : {2L, 3L, 4L, 9L}) {
    EXPECT_LT(got_cpu[row].abs().max().item<double>(), 1e-3)
        << "base-only row " << row << " must be untouched (zero delta)";
  }
}

// Case 6: FP32 golden. Same shape as Case 2 but promoted to FP32; the
// tolerance is much tighter so this catches accumulate-order regressions
// that the BF16 checks would tolerate.
TEST_F(LoRAGroupedMatmulTest, Fp32Golden) {
  const int64_t hidden = 64, out_local = 64, r = 8;
  const torch::Dtype dt = torch::kFloat32;
  const auto dev = test_device();
  std::vector<uint64_t> aids = {60, 61, 60};
  std::vector<int32_t> lens = {2, 2, 2};
  torch::manual_seed(6);
  auto input =
      (torch::rand({6, hidden}, torch::TensorOptions().dtype(torch::kFloat32)) -
       0.5f)
          .to(dev)
          .contiguous();
  torch::Tensor A60, B60, A61, B61;
  make_ab(r, hidden, out_local, dt, dev, 1000, A60, B60);
  make_ab(r, hidden, out_local, dt, dev, 1100, A61, B61);
  const std::vector<uint64_t> aid_index = {60, 61};
  const std::vector<torch::Tensor> A_by = {A60, A61};
  const std::vector<torch::Tensor> B_by = {B60, B61};
  const std::vector<float> s_by = {1.0f, 1.0f};
  auto got = run_grouped(
      input, aids, lens, A_by, B_by, s_by, aid_index, hidden, out_local, r);
  auto want = golden_per_seq_delta(
      input, aids, lens, A_by, B_by, s_by, aid_index, out_local);
  expect_close_fp32(got, want);
}

// Case 7: edge cases -- zero-length seqs interspersed and an all-base-only
// batch. build_grouped_lora_spec should skip seq_len<=0 silently, and an
// all-base batch should produce distinct_aids.empty() (0 groups). The
// caller (wrapper) is responsible for gating this before calling the
// helper, but the helper must not crash if asked.
TEST_F(LoRAGroupedMatmulTest, EdgeCasesZeroLenAndAllBase) {
  const int64_t hidden = 32, out_local = 32;
  const torch::Dtype dt = torch::kBFloat16;
  const auto dev = test_device();
  // Sub-case A: zero-length seq mixed in.
  {
    std::vector<uint64_t> aids = {77, 0, 77};
    std::vector<int32_t> lens = {2, 0, 3};
    torch::manual_seed(7);
    auto input = (torch::rand({5, hidden},
                              torch::TensorOptions().dtype(torch::kFloat32)) -
                  0.5f)
                     .to(dt)
                     .to(dev)
                     .contiguous();
    auto spec = build_grouped_lora_spec(aids, lens, input);
    EXPECT_EQ(spec.distinct_aids.size(), 1u);
    EXPECT_EQ(spec.base_only_tokens, 0);
    EXPECT_EQ(spec.x_permuted.size(0), 5);
  }
  // Sub-case B: all base-only. Helper must return empty spec.
  {
    std::vector<uint64_t> aids = {0, 0};
    std::vector<int32_t> lens = {3, 2};
    torch::manual_seed(8);
    auto input = (torch::rand({5, hidden},
                              torch::TensorOptions().dtype(torch::kFloat32)) -
                  0.5f)
                     .to(dt)
                     .to(dev)
                     .contiguous();
    auto spec = build_grouped_lora_spec(aids, lens, input);
    EXPECT_EQ(spec.distinct_aids.size(), 0u);
    EXPECT_EQ(spec.base_only_tokens, 5);
    EXPECT_EQ(spec.x_permuted.size(0), 0);
  }
}

}  // namespace
}  // namespace xllm::layer
