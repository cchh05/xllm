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

// Probe v2: validate that two CollectiveCommunicator instances (one CP=N
// dp_size=1, one DP=N cp_size=1) can coexist in the same process and be
// used interleavingly without HCCL state corruption. This is the gating
// experiment for the "dual-graph" approach to runtime CP<->DP switching:
// instead of destroy+recreate on every switch (~600ms), keep both
// communicators alive and only flip which one is "active" for each
// request (~ms switch).
//
// Build is gated behind USE_NPU; run with torchrun on a real 4-card NPU
// box (910_82) -- same harness as comm_switch_probe.
//
// Usage (4 cards):
//   bash tests/probes/run_dual_graph_probe.sh --rounds=20
//
// What we measure per round:
//   * forward_ms on the CP communicator (cp_size=N, dp_size=1)
//   * forward_ms on the DP communicator (cp_size=1, dp_size=N)
//   * NPU memory growth across all rounds
//
// Pass criteria (mapping back to plan b-cozy-valley.md):
//   PASS  - both comms callable in any interleaving order, mem stable,
//           numerical values correct on both sides; sets up the dual-
//           graph architecture as the v1 implementation route.
//   FAIL  - HCCL errors or numerical corruption when crossing comms;
//           dual-graph route is dead, fall back to destroy+recreate
//           with a longer drain window.
//
// What this probe DOES NOT cover:
//   * ATB backend (probe runs TORCH backend like its v1 sibling). ATB
//     dual-commDomain is a separate question, will need a v3 probe.
//   * Model layer (no ATB graph reinit, no decoder forward). The probe
//     confirms the comms-level coexistence; layer-level dual-graph is
//     a follow-up.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"

DEFINE_int32(rounds, 20, "Number of interleaved CP/DP forward rounds.");
DEFINE_string(master_addr,
              "127.0.0.1:40000",
              "Base TCPStore endpoint. The two communicators each grab a "
              "non-overlapping port window starting at this base.");
DEFINE_int32(port_stride,
             256,
             "Port window per communicator. Same rationale as in "
             "comm_switch_probe (covers inner fanout + TIME_WAIT slack).");
DEFINE_int32(world_size,
             4,
             "World size; matches torchrun --nproc_per_node N.");
DEFINE_int32(forward_numel,
             1024,
             "Tensor size for each smoke allreduce.");
DEFINE_int32(mem_growth_threshold_mb,
             200,
             "Fail if NPU free-memory drop after all rounds exceeds this "
             "many MB. Higher than the v1 probe's 100 MB because we hold "
             "two comm sets resident (one CP, one DP).");

namespace {

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int32_t require_int_env(const char* name) {
  const char* v = std::getenv(name);
  CHECK(v != nullptr && *v != '\0')
      << "Env var " << name
      << " is required (torchrun sets it; export RANK explicitly otherwise)";
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

int32_t read_int_env(const char* name, int32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return fallback;
  }
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

void split_addr(const std::string& addr,
                std::string* host,
                int32_t* port) {
  const auto colon = addr.find(':');
  CHECK_NE(colon, std::string::npos)
      << "master_addr must be host:port, got " << addr;
  *host = addr.substr(0, colon);
  *port = static_cast<int32_t>(std::strtol(addr.c_str() + colon + 1,
                                           /*endptr=*/nullptr,
                                           /*base=*/10));
  CHECK_GT(*port, 0) << "master_addr port must be > 0, got " << addr;
}

std::string make_addr(const std::string& host, int32_t port) {
  return host + ":" + std::to_string(port);
}

// Pick the most useful ProcessGroup from a ParallelArgs bag for a smoke
// allreduce. Mirrors comm_switch_probe's logic so the two probes' PASS
// signals stay comparable.
xllm::ProcessGroup* pick_group(const xllm::ParallelArgs* parallel_args) {
  if (parallel_args == nullptr) {
    return nullptr;
  }
  if (parallel_args->dp_local_process_group_ != nullptr) {
    return parallel_args->dp_local_process_group_;
  }
  if (parallel_args->process_group_ != nullptr) {
    return parallel_args->process_group_;
  }
  return parallel_args->tp_group_;
}

// Run a fixed-value allreduce and verify the sum equals world_size *
// (rank+1) baseline. Fails fast on numerical mismatch -- if the two
// communicators were corrupting each other's state we'd see this gate
// trip on the round where the bad value lands.
void smoke_forward(xllm::ProcessGroup* pg,
                   const torch::Device& device,
                   int32_t numel,
                   const std::string& tag) {
  CHECK(pg != nullptr) << tag << ": ProcessGroup is null";
  const int32_t rank = pg->rank();
  const int32_t group_size = pg->world_size();
  const float my_value = static_cast<float>(rank + 1);
  const float expected =
      static_cast<float>(group_size * (group_size + 1)) / 2.0f;

  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(device)
                  .requires_grad(false);
  torch::Tensor t = torch::full({numel}, my_value, opts);
  pg->allreduce(t);
  const float got = t.cpu().data_ptr<float>()[0];
  CHECK_LT(std::abs(got - expected), 1e-3f)
      << tag << " allreduce mismatch: got=" << got
      << " expected=" << expected
      << " (cross-comm state corruption?)";
}

std::unique_ptr<xllm::CollectiveCommunicator> build_communicator(
    int32_t global_rank,
    int32_t world_size,
    int32_t dp_size,
    int32_t cp_size,
    const std::string& master_addr,
    const torch::Device& device) {
  auto comm = std::make_unique<xllm::CollectiveCommunicator>(
      /*global_rank=*/global_rank,
      /*world_size=*/world_size,
      /*dp_size=*/dp_size,
      /*ep_size=*/1,
      /*cp_size=*/cp_size);
  comm->create_process_groups(master_addr, device);
  return comm;
}

}  // namespace

int main(int32_t argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  // Same singleton init as comm_switch_probe. Without this the
  // CollectiveCommunicator constructor hits the ATB code path and
  // skips ProcessGroup creation, masking the probe's signal.
  xllm::KernelConfig::get_instance().from_flags();
  xllm::ParallelConfig::get_instance().from_flags();

  const int32_t world_size = read_int_env("WORLD_SIZE", FLAGS_world_size);
  const int32_t global_rank = require_int_env("RANK");
  const int32_t local_rank = read_int_env("LOCAL_RANK", global_rank);
  CHECK_GT(world_size, 1);
  CHECK_LT(local_rank, xllm::Device::device_count());

  std::string base_host;
  int32_t base_port = 0;
  split_addr(FLAGS_master_addr, &base_host, &base_port);

  xllm::Device dev(local_rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  const int64_t baseline_free_mb = dev.free_memory() / (1024 * 1024);

  LOG(INFO) << "[probe2] start rank=" << global_rank
            << " local_rank=" << local_rank
            << " world_size=" << world_size
            << " rounds=" << FLAGS_rounds
            << " base=" << FLAGS_master_addr
            << " baseline_free_mb=" << baseline_free_mb;

  // Build BOTH comms up-front. This is the moment of truth -- if HCCL
  // refuses two simultaneous comms on the same device, we'll hear about
  // it here, not in the loop below.
  const std::string cp_addr =
      make_addr(base_host, base_port);
  const std::string dp_addr =
      make_addr(base_host, base_port + FLAGS_port_stride);

  const int64_t t_setup_0 = now_ms();
  auto comm_cp = build_communicator(global_rank, world_size,
                                    /*dp_size=*/1,
                                    /*cp_size=*/world_size,
                                    cp_addr, device);
  const int64_t t_setup_1 = now_ms();
  auto comm_dp = build_communicator(global_rank, world_size,
                                    /*dp_size=*/world_size,
                                    /*cp_size=*/1,
                                    dp_addr, device);
  const int64_t t_setup_2 = now_ms();

  const int64_t after_setup_free_mb = dev.free_memory() / (1024 * 1024);
  LOG(INFO) << "[probe2] both_comms_built"
            << " cp_setup_ms=" << (t_setup_1 - t_setup_0)
            << " dp_setup_ms=" << (t_setup_2 - t_setup_1)
            << " mem_drop_mb=" << (baseline_free_mb - after_setup_free_mb);

  xllm::ProcessGroup* cp_pg = pick_group(comm_cp->parallel_args());
  xllm::ProcessGroup* dp_pg = pick_group(comm_dp->parallel_args());
  if (cp_pg == nullptr || dp_pg == nullptr) {
    LOG(WARNING)
        << "[probe2] one of the comms has no usable ProcessGroup; the "
           "TORCH backend was likely not selected. "
           "Pass --npu_kernel_backend=TORCH explicitly.";
    return 4;
  }

  // Interleaving pattern is intentionally not strict alternation: a
  // straight CP/DP/CP/DP would not catch state corruption that depends
  // on consecutive same-comm calls. We use repeats so each comm gets
  // both "fresh from setup" and "right after the other comm" treatment.
  const std::vector<char> pattern = {
      'C', 'C', 'D', 'C', 'D', 'D', 'C', 'D', 'D', 'C',
      'D', 'C', 'C', 'C', 'D', 'D', 'D', 'C', 'D', 'C'};

  int64_t total_cp_forward_ms = 0;
  int64_t total_dp_forward_ms = 0;
  int32_t cp_calls = 0;
  int32_t dp_calls = 0;
  int32_t failures = 0;

  for (int32_t round = 0; round < FLAGS_rounds; ++round) {
    for (char step : pattern) {
      try {
        const int64_t t0 = now_ms();
        if (step == 'C') {
          smoke_forward(cp_pg, device, FLAGS_forward_numel, "CP");
          total_cp_forward_ms += now_ms() - t0;
          ++cp_calls;
        } else {
          smoke_forward(dp_pg, device, FLAGS_forward_numel, "DP");
          total_dp_forward_ms += now_ms() - t0;
          ++dp_calls;
        }
      } catch (const std::exception& e) {
        ++failures;
        LOG(ERROR) << "[probe2] round=" << round << " step=" << step
                   << " EXCEPTION: " << e.what();
      }
    }
    if ((round + 1) % 5 == 0) {
      const int64_t free_now_mb = dev.free_memory() / (1024 * 1024);
      LOG(INFO) << "[probe2] round=" << round
                << " cp_calls=" << cp_calls
                << " avg_cp_fwd_ms="
                << (cp_calls > 0 ? total_cp_forward_ms / cp_calls : 0)
                << " dp_calls=" << dp_calls
                << " avg_dp_fwd_ms="
                << (dp_calls > 0 ? total_dp_forward_ms / dp_calls : 0)
                << " failures=" << failures
                << " mem_drop_mb=" << (baseline_free_mb - free_now_mb);
    }
  }

  // Tear down in reverse order. We deliberately destroy CP first to
  // see if HCCL handles the asymmetric teardown -- if it complains,
  // we'll know to expect issues when the production switch keeps
  // comm_dp alive while killing comm_cp.
  comm_cp.reset();
  dev.synchronize_default_stream();
  comm_dp.reset();
  dev.synchronize_default_stream();

  const int64_t final_free_mb = dev.free_memory() / (1024 * 1024);
  const int64_t total_drop_mb = baseline_free_mb - final_free_mb;

  LOG(INFO) << "[probe2] summary"
            << " rounds=" << FLAGS_rounds
            << " cp_calls=" << cp_calls
            << " dp_calls=" << dp_calls
            << " failures=" << failures
            << " avg_cp_fwd_ms="
            << (cp_calls > 0 ? total_cp_forward_ms / cp_calls : 0)
            << " avg_dp_fwd_ms="
            << (dp_calls > 0 ? total_dp_forward_ms / dp_calls : 0)
            << " final_free_mb=" << final_free_mb
            << " total_drop_mb=" << total_drop_mb;

  if (failures > 0) {
    LOG(ERROR) << "[probe2] FAIL: " << failures
               << " forward(s) errored. dual-graph route is at risk; "
                  "see logs for cross-comm corruption signatures.";
    return 2;
  }
  if (total_drop_mb > FLAGS_mem_growth_threshold_mb) {
    LOG(ERROR) << "[probe2] FAIL: NPU free memory dropped " << total_drop_mb
               << " MB; suspect leak in dual-comm coexistence. "
               << "Threshold=" << FLAGS_mem_growth_threshold_mb << " MB.";
    return 3;
  }

  LOG(INFO) << "[probe2] PASS: dual CP+DP comms coexist; "
            << FLAGS_rounds << " rounds, " << (cp_calls + dp_calls)
            << " calls, 0 failures. Dual-graph architecture is feasible "
               "at the comm layer.";
  return 0;
}
