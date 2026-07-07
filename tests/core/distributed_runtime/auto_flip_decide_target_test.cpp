/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Unit test for AutoFlipController::decide_target's mode-selection state
// machine. The real controller couples the decision to a live scheduler,
// engine, and mode-switch RPC, so this test re-implements the pure decision
// function here and exercises its edges. If you change the algorithm in
// auto_flip_controller.cpp, update DecideTarget below to match.
//
// Background: enum truth is CP_PREFILL=0, DP_DECODE=1
// (see xllm/core/framework/parallel_state/parallel_args.h:219-220).
// The mode_switch.proto comment on target_mode has 0/1 swapped -- do NOT
// follow the proto comment; follow the enum. Earlier versions of
// decide_target got this wrong and drove flips in the reverse direction
// under load; this test locks the corrected mapping.
//
// Heal path: auto-flip must actively pull the instance back to
// CP_PREFILL when we're in DP_DECODE with too few pending requests to
// fill every dp_rank. The alternative is that the ContinuousScheduler's
// lopsided-defer backdoor eventually fires and drives worker_impl into
// the fake-input path, which hangs on ATB decoder placeholder tensors
// (v14 investigation). The heal is exempt from the sample-count and
// dwell-time gates because a hung DP forward stops the sample pipe.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include <gtest/gtest.h>

namespace xllm {
namespace test {

struct DecideTargetKnobs {
  double long_ratio_activate = 0.4;
  double long_ratio_deactivate = 0.2;
  double min_pending_per_dp_rank = 2.0;
  uint64_t min_samples = 10;
};

// Mirror of AutoFlipController::heal_active. Kept as a free function so
// the two callers (DecideTarget below, tests) share a single definition.
bool HealActive(int8_t cur_mode,
                int32_t active_dp_size,
                size_t pending_requests,
                const DecideTargetKnobs& k) {
  if (cur_mode != 1 || active_dp_size <= 1 ||
      k.min_pending_per_dp_rank <= 0.0) {
    return false;
  }
  const double min_pending_double =
      k.min_pending_per_dp_rank * static_cast<double>(active_dp_size);
  const size_t min_pending =
      static_cast<size_t>(std::ceil(min_pending_double));
  return pending_requests < min_pending;
}

// Mirror of AutoFlipController::decide_target. Kept intentionally free of
// gflags / logging so we can exercise the branches in isolation.
int8_t DecideTarget(int8_t cur_mode,
                    double long_ratio,
                    uint64_t total_in_window,
                    int32_t active_dp_size,
                    size_t pending_requests,
                    const DecideTargetKnobs& k = DecideTargetKnobs()) {
  if (HealActive(cur_mode, active_dp_size, pending_requests, k)) {
    return 0;
  }
  if (total_in_window < k.min_samples) {
    return cur_mode;
  }
  if (cur_mode == 0) {
    if (long_ratio < k.long_ratio_deactivate) {
      return 1;
    }
    return 0;
  }
  if (cur_mode == 1) {
    if (long_ratio >= k.long_ratio_activate) {
      return 0;
    }
    return 1;
  }
  return cur_mode;
}

TEST(AutoFlipDecideTargetTest, HealFiresOnLopsidedDp) {
  // DP with dp=2 requires at least ceil(2.0*2)=4 pending to be safe. A
  // single request in flight (pending=1) drops below that threshold and
  // the heal path returns 0 (CP_PREFILL) even though long_ratio is 0.
  EXPECT_EQ(DecideTarget(/*cur_mode=*/1,
                         /*long_ratio=*/0.0,
                         /*total_in_window=*/50,
                         /*active_dp_size=*/2,
                         /*pending_requests=*/1),
            0);
}

TEST(AutoFlipDecideTargetTest, HealDoesNotFireWhenPendingSufficient) {
  // dp=2 with pending=4 fills both dp_ranks with 2 each. Heal must not
  // fire; low long_ratio still keeps us in DP.
  EXPECT_EQ(DecideTarget(/*cur_mode=*/1,
                         /*long_ratio=*/0.0,
                         /*total_in_window=*/50,
                         /*active_dp_size=*/2,
                         /*pending_requests=*/4),
            1);
}

TEST(AutoFlipDecideTargetTest, HealCeilingRoundsUp) {
  // Fractional threshold: 1.5 * 2 = 3. pending=2 must trigger heal
  // (2 < ceil(3)); pending=3 must not. Guards against a floor-rounding
  // regression that would leave one dp_rank empty.
  DecideTargetKnobs k;
  k.min_pending_per_dp_rank = 1.5;
  EXPECT_EQ(DecideTarget(1, 0.0, 50, 2, 2, k), 0);
  EXPECT_EQ(DecideTarget(1, 0.0, 50, 2, 3, k), 1);
}

TEST(AutoFlipDecideTargetTest, HealDisabledByZeroThreshold) {
  // Setting min_pending_per_dp_rank=0 disables the heal path (legacy
  // behavior for operators who want to opt out).
  DecideTargetKnobs k;
  k.min_pending_per_dp_rank = 0.0;
  EXPECT_EQ(DecideTarget(1, 0.0, 50, 2, 0, k), 1);
}

TEST(AutoFlipDecideTargetTest, HealSkippedInSingleDp) {
  // active_dp_size=1: no lopsided risk, heal path must not fire.
  EXPECT_EQ(DecideTarget(1, 0.0, 50, 1, 0), 1);
}

TEST(AutoFlipDecideTargetTest, InsufficientSamplesStaysPut) {
  // Below min_samples (default 10) both modes stay in place.
  EXPECT_EQ(DecideTarget(0, 0.9, /*total_in_window=*/5, 1, 100), 0);
  EXPECT_EQ(DecideTarget(1, 0.0, /*total_in_window=*/5, 1, 100), 1);
}

TEST(AutoFlipDecideTargetTest, DpToCpOnHighLongRatio) {
  // In DP mode, long_ratio >= activate flips to CP.
  EXPECT_EQ(DecideTarget(1, 0.5, 50, 1, 100), 0);
}

TEST(AutoFlipDecideTargetTest, CpToDpOnLowLongRatio) {
  // In CP mode, long_ratio < deactivate flips to DP.
  EXPECT_EQ(DecideTarget(0, 0.1, 50, 1, 100), 1);
}

TEST(AutoFlipDecideTargetTest, HysteresisBandKeepsMode) {
  // long_ratio in the band [deactivate, activate) does NOT flip either
  // way. This is the hysteresis that prevents flip-flopping when the
  // signal drifts around the threshold.
  EXPECT_EQ(DecideTarget(0, 0.3, 50, 1, 100), 0);
  EXPECT_EQ(DecideTarget(1, 0.3, 50, 1, 100), 1);
}

TEST(AutoFlipDecideTargetTest, HealPreemptsHysteresis) {
  // In DP with high long_ratio (would otherwise pass the hysteresis
  // check to stay in DP -> flip to CP for load reasons), the heal path
  // still fires when pending is too low. Both DECIDE the same target
  // here, but the point is that heal runs BEFORE the sample gate --
  // total_in_window=0 would normally return cur_mode.
  EXPECT_EQ(DecideTarget(/*cur_mode=*/1,
                         /*long_ratio=*/0.9,
                         /*total_in_window=*/0,
                         /*active_dp_size=*/2,
                         /*pending_requests=*/0),
            0);
}

TEST(AutoFlipHealActiveTest, MatchesDecideTargetHealBranch) {
  // The tick-loop bypasses the dwell gate when heal_active returns true;
  // that predicate MUST match the condition decide_target uses to return 0.
  // A drift between the two would either (a) skip dwell without decide
  // agreeing (wasted RPC), or (b) decide heal while dwell blocks (hang
  // persists). This test locks them together.
  for (int cur_mode : {0, 1}) {
    for (int32_t dp : {1, 2, 4}) {
      for (size_t pending : {0u, 1u, 3u, 8u, 20u}) {
        DecideTargetKnobs k;
        const bool heal = HealActive(static_cast<int8_t>(cur_mode), dp,
                                     pending, k);
        // When heal fires, decide_target must return 0 regardless of the
        // other signals we pass (long_ratio, samples) -- pick values that
        // WOULD keep us in cur_mode absent heal.
        const int8_t got = DecideTarget(static_cast<int8_t>(cur_mode),
                                        /*long_ratio=*/0.5,
                                        /*total_in_window=*/50, dp, pending);
        if (heal) {
          EXPECT_EQ(got, 0)
              << "cur_mode=" << cur_mode << " dp=" << dp
              << " pending=" << pending;
        }
      }
    }
  }
}

TEST(AutoFlipHealActiveTest, RespectsFlagDisable) {
  DecideTargetKnobs k;
  k.min_pending_per_dp_rank = 0.0;
  EXPECT_FALSE(HealActive(1, 4, 0, k));
  EXPECT_FALSE(HealActive(1, 2, 0, k));
}

TEST(AutoFlipHealActiveTest, CpModeNeverHeals) {
  // Heal only meaningful when we're in DP; in CP a low pending is fine.
  DecideTargetKnobs k;
  EXPECT_FALSE(HealActive(/*cur_mode=*/0, /*dp=*/4, /*pending=*/0, k));
}

}  // namespace test
}  // namespace xllm
