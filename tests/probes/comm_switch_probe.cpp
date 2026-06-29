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

// Probe: validate that the HCCL + ATB collective communicators can be
// destroyed and rebuilt at runtime, switching between CP=N (cp_size=N,
// dp_size=1) and DP=N (cp_size=1, dp_size=N) over the same world.
//
// This probe is the risk-clearing experiment for the larger "P-pool runtime
// CP<->DP" feature (plan: ~/.claude/plans/b-cozy-valley.md). Build is gated
// behind XLLM_BUILD_PROBES=ON so the binary stays out of CI; run with mpirun
// on a real NPU box (e.g. 910_82, 4 cards).
//
// Usage (4 cards):
//   mpirun -n 4 ./comm_switch_probe \
//       --rank_tablefile=/path/to/82_4card.json \
//       --master_addr=127.0.0.1:29500 \
//       --iters=50
//
// What we measure per iteration:
//   * setup_ms / forward_ms / destroy_ms for both CP and DP phases
//   * NPU memory growth (free_memory delta) to catch HCCL/ATB leaks
//
// Pass criteria (see plan file Section "探针通过/不通过的处置路径"):
//   A1 PASS  - setup < 2s, destroy < 1s, mem growth < 100 MB end-to-end
//   A2 SLOW  - succeeds but exceeds budget; flag for "dual-group" fallback
//   A3 FAIL  - HCCL/ATB error during recreate; flag for "always-on dual" path
//   A4 FAIL  - second InitGlobalCommDomain rejects; needs ATB API change
//
// Important caveat about ATB cleanup:
//   resetting the unique_ptr<CollectiveCommunicator> only frees the xllm-
//   owned ProcessGroup objects (each of which calls HcclCommDestroy on its
//   own comm_). The ATB-side commDomain registered through
//   `mapping.InitGlobalCommDomain` and the HcclComm cached in
//   `atb_speed::ExternalCommManager` survive — that singleton has no
//   user-visible Remove*Domain API today. The probe surfaces this in two
//   ways: rising free-memory drop across iters, or the second iter's
//   InitGlobalCommDomain throwing (outcome A4).

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"

DEFINE_int32(iters, 50, "Number of CP<->DP switch iterations.");
DEFINE_string(master_addr,
              "127.0.0.1:29500",
              "Master addr (host:port) used as the BASE TCPStore endpoint. "
              "Each iteration shifts the port by iter * port_stride to avoid "
              "TIME_WAIT collisions and stale TCPStore reuse across rebuilds.");
DEFINE_int32(port_stride,
             64,
             "Port shift between iterations. Must be larger than the inner "
             "fanout used by CollectiveCommunicator::create_process_groups "
             "(world_size + dp_size + tp_size + single-rank-per-rank). The "
             "default 64 covers world_size up to 8 comfortably.");
DEFINE_int32(world_size,
             4,
             "World size for the probe; matches mpirun -n N. Both CP=N and "
             "DP=N phases use this many ranks.");
DEFINE_int32(forward_iters,
             3,
             "Number of smoke allreduce calls per phase per iteration.");
DEFINE_int32(forward_numel,
             1024,
             "Number of float elements in each smoke allreduce tensor.");
DEFINE_int32(mem_growth_threshold_mb,
             100,
             "Fail the probe if NPU free-memory drop after all iterations "
             "exceeds this many MB.");

namespace {

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Read RANK / LOCAL_RANK / WORLD_SIZE from environment, falling back to
// gflags. mpirun / torchrun both export these. We require RANK to be present
// because a default of 0 would silently put every process on the same rank
// and deadlock at TCPStore rendezvous.
int32_t read_int_env(const char* name, int32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return fallback;
  }
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

int32_t require_int_env(const char* name) {
  const char* v = std::getenv(name);
  CHECK(v != nullptr && *v != '\0')
      << "Env var " << name
      << " is required (mpirun / torchrun set it; export RANK / WORLD_SIZE "
         "explicitly otherwise)";
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

// Split a "host:port" string. CollectiveCommunicator's TCPStore wiring
// expects the same string back, just with the port stepped each iteration.
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

std::string get_master_addr() {
  const char* host = std::getenv("MASTER_ADDR");
  const char* port = std::getenv("MASTER_PORT");
  if (host != nullptr && port != nullptr && *host != '\0' && *port != '\0') {
    return std::string(host) + ":" + port;
  }
  return FLAGS_master_addr;
}

// Run a fixed-value allreduce on the given process group and verify the
// reduction sums to world_size * (rank+1) baseline. The actual numerical
// check is best-effort: any mismatch fails fast via CHECK so a regression
// in HCCL after recreate is loud, not silent.
void run_smoke_forward(const xllm::ParallelArgs* parallel_args,
                       const torch::Device& device,
                       int32_t numel,
                       int32_t iters,
                       const std::string& tag) {
  CHECK(parallel_args != nullptr) << tag << ": parallel_args is null";

  // Pick an available group from the bag. Prefer dp_local_process_group_ when
  // dp_size > 1 (DP phase), otherwise fall back to process_group_ which spans
  // the whole world. tp_group_ is the next fallback for the CP phase since
  // process_group_ may stay null on the ATB-only init path.
  xllm::ProcessGroup* pg = nullptr;
  if (parallel_args->dp_local_process_group_ != nullptr) {
    pg = parallel_args->dp_local_process_group_;
  } else if (parallel_args->process_group_ != nullptr) {
    pg = parallel_args->process_group_;
  } else if (parallel_args->tp_group_ != nullptr) {
    pg = parallel_args->tp_group_;
  }
  if (pg == nullptr) {
    LOG(WARNING) << tag
                 << ": no ProcessGroup available for smoke forward; "
                    "skipping numerical check (likely ATB-only path)";
    return;
  }

  const int32_t rank = pg->rank();
  const int32_t group_size = pg->world_size();
  const float my_value = static_cast<float>(rank + 1);
  const float expected =
      static_cast<float>(group_size * (group_size + 1)) / 2.0f;

  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(device)
                  .requires_grad(false);

  for (int32_t i = 0; i < iters; ++i) {
    torch::Tensor t = torch::full({numel}, my_value, opts);
    pg->allreduce(t);
    // Bring one element back for the assertion. cheap correctness gate that
    // would catch comm corruption after a rebuild.
    const float got = t.cpu().data_ptr<float>()[0];
    CHECK_LT(std::abs(got - expected), 1e-3f)
        << tag << " allreduce mismatch on iter " << i << ": got=" << got
        << " expected=" << expected;
  }
}

// Build a CollectiveCommunicator for the requested mode. Encapsulates the
// CP=N / DP=N split so the iteration loop stays readable.
std::unique_ptr<xllm::CollectiveCommunicator> build_communicator(
    int32_t global_rank,
    int32_t world_size,
    int32_t dp_size,
    int32_t cp_size,
    const std::string& master_addr,
    const torch::Device& device) {
  // ep_size=1 keeps the probe focused on attention CP/DP. MoE EP rebuild is
  // out of v1 scope; once attention path is proven, EP can be retested with
  // a dedicated probe.
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
  // Init logging before flag parsing so any warnings during parsing land in
  // stderr immediately; FLAGS_logtostderr is a glog flag the parser knows
  // about, so user-supplied --logtostderr=true still wins.
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  // RANK / WORLD_SIZE must come from the launcher. A default-zero rank
  // silently deadlocks TCPStore because every process tries to be rank 0.
  const int32_t world_size =
      read_int_env("WORLD_SIZE", FLAGS_world_size);
  const int32_t global_rank = require_int_env("RANK");
  const int32_t local_rank = read_int_env("LOCAL_RANK", global_rank);
  const std::string base_master_addr = get_master_addr();

  CHECK_GT(world_size, 1) << "Probe needs world_size >= 2";
  CHECK_LT(local_rank, xllm::Device::device_count())
      << "local_rank=" << local_rank
      << " exceeds device_count=" << xllm::Device::device_count();

  std::string base_host;
  int32_t base_port = 0;
  split_addr(base_master_addr, &base_host, &base_port);

  xllm::Device dev(local_rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();

  const int64_t baseline_free_mb = dev.free_memory() / (1024 * 1024);
  LOG(INFO) << "[probe] start rank=" << global_rank
            << " local_rank=" << local_rank
            << " world_size=" << world_size
            << " iters=" << FLAGS_iters
            << " base_addr=" << base_master_addr
            << " port_stride=" << FLAGS_port_stride
            << " baseline_free_mb=" << baseline_free_mb;

  int64_t total_setup_ms = 0;
  int64_t total_destroy_ms = 0;
  int64_t worst_setup_ms = 0;
  int64_t worst_destroy_ms = 0;
  int32_t failures = 0;

  for (int32_t i = 0; i < FLAGS_iters; ++i) {
    // Each iteration uses two non-overlapping port windows (one for the CP
    // phase, one for the DP phase) so a TIME_WAIT TCPStore from the previous
    // iter cannot collide with the next. The window size FLAGS_port_stride
    // must exceed the max inner port fanout in
    // CollectiveCommunicator::create_process_groups.
    const int32_t cp_port = base_port + (2 * i) * FLAGS_port_stride;
    const int32_t dp_port = base_port + (2 * i + 1) * FLAGS_port_stride;
    const std::string cp_addr = make_addr(base_host, cp_port);
    const std::string dp_addr = make_addr(base_host, dp_port);

    try {
      // ---------- Phase A: CP=world_size, dp_size=1 ----------
      const int64_t a0 = now_ms();
      auto comm_cp = build_communicator(
          global_rank, world_size, /*dp_size=*/1,
          /*cp_size=*/world_size, cp_addr, device);
      const int64_t a1 = now_ms();
      run_smoke_forward(comm_cp->parallel_args(),
                        device,
                        FLAGS_forward_numel,
                        FLAGS_forward_iters,
                        /*tag=*/"CP");
      const int64_t a2 = now_ms();
      comm_cp.reset();
      dev.synchronize_default_stream();
      const int64_t a3 = now_ms();

      // ---------- Phase B: DP=world_size, cp_size=1 ----------
      const int64_t b0 = now_ms();
      auto comm_dp = build_communicator(
          global_rank, world_size, /*dp_size=*/world_size,
          /*cp_size=*/1, dp_addr, device);
      const int64_t b1 = now_ms();
      run_smoke_forward(comm_dp->parallel_args(),
                        device,
                        FLAGS_forward_numel,
                        FLAGS_forward_iters,
                        /*tag=*/"DP");
      const int64_t b2 = now_ms();
      comm_dp.reset();
      dev.synchronize_default_stream();
      const int64_t b3 = now_ms();

      const int64_t cp_setup = a1 - a0;
      const int64_t cp_forward = a2 - a1;
      const int64_t cp_destroy = a3 - a2;
      const int64_t dp_setup = b1 - b0;
      const int64_t dp_forward = b2 - b1;
      const int64_t dp_destroy = b3 - b2;

      total_setup_ms += cp_setup + dp_setup;
      total_destroy_ms += cp_destroy + dp_destroy;
      worst_setup_ms = std::max(worst_setup_ms, std::max(cp_setup, dp_setup));
      worst_destroy_ms =
          std::max(worst_destroy_ms, std::max(cp_destroy, dp_destroy));

      const int64_t free_now_mb = dev.free_memory() / (1024 * 1024);
      const int64_t mem_drop_mb = baseline_free_mb - free_now_mb;

      LOG(INFO) << "[probe] iter=" << i
                << " cp_setup=" << cp_setup << "ms"
                << " cp_forward=" << cp_forward << "ms"
                << " cp_destroy=" << cp_destroy << "ms"
                << " dp_setup=" << dp_setup << "ms"
                << " dp_forward=" << dp_forward << "ms"
                << " dp_destroy=" << dp_destroy << "ms"
                << " free_mb=" << free_now_mb
                << " mem_drop_mb=" << mem_drop_mb;
    } catch (const std::exception& e) {
      ++failures;
      LOG(ERROR) << "[probe] iter=" << i << " EXCEPTION: " << e.what();
      // Keep going so we can quantify how often it fails, not just first.
    }
  }

  const int64_t final_free_mb = dev.free_memory() / (1024 * 1024);
  const int64_t total_drop_mb = baseline_free_mb - final_free_mb;

  LOG(INFO) << "[probe] summary"
            << " iters=" << FLAGS_iters
            << " failures=" << failures
            << " avg_setup_ms="
            << (FLAGS_iters > 0 ? total_setup_ms / (2 * FLAGS_iters) : 0)
            << " avg_destroy_ms="
            << (FLAGS_iters > 0 ? total_destroy_ms / (2 * FLAGS_iters) : 0)
            << " worst_setup_ms=" << worst_setup_ms
            << " worst_destroy_ms=" << worst_destroy_ms
            << " final_free_mb=" << final_free_mb
            << " total_drop_mb=" << total_drop_mb;

  if (failures > 0) {
    LOG(ERROR) << "[probe] FAIL: " << failures
               << " iteration(s) errored. See logs above; map outcome to "
                  "A3/A4 in plan file.";
    return 2;
  }
  if (total_drop_mb > FLAGS_mem_growth_threshold_mb) {
    LOG(ERROR) << "[probe] FAIL: NPU free memory dropped " << total_drop_mb
               << " MB across " << FLAGS_iters
               << " switches; suspect HCCL/ATB leak. Threshold="
               << FLAGS_mem_growth_threshold_mb << " MB.";
    return 3;
  }

  LOG(INFO) << "[probe] PASS: " << FLAGS_iters
            << " CP<->DP switches succeeded; map outcome to A1 (or A2 if "
               "setup/destroy timings exceed budget).";
  return 0;
}
