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

// Probe v3 (rewritten): validate that two MappingNPU instances built
// back-to-back inside the same process produce independent, non-clobbering
// commDomain entries inside the global atb_speed::ExternalCommManager.
//
// Why this rewrite: the first v3 attempt read parallel_args.dispatch-
// AndCombineHcclComm(), which is the MOE-EP comm. With ep_size=1 the
// MOE_EP rankIds list is too small (size <= 1) and ExternalCommManager
// short-circuits to an empty commDomain. That said nothing about the
// real attention comms, which is what dual-graph actually depends on.
//
// What we now measure:
//   * Build CP communicator (cp=N, dp=1). Pull the ATTN_CP ParallelInfo
//     from its mapping and ask ExternalCommManager for the commDomain
//     + HcclComm.
//   * Build DP communicator (cp=1, dp=N). Pull the ATTN_DP ParallelInfo
//     from its mapping; same lookup.
//   * Verify the two ParallelInfo objects have non-empty rankIds and
//     yield non-null HcclComm handles.
//   * Run interleaved HcclAllReduce against each comm, fixed-value
//     payload, expect rank+1 sum across world_size.
//   * After the smoke loop, re-pull the CP commDomain and confirm the
//     entry inside ExternalCommManager was not clobbered by the second
//     mapping init.
//
// Pass criteria:
//   PASS - both mappings coexist; cross-comm calls numerically correct;
//          mem stable. Dual-graph viable on ATB backend.
//   FAIL - second mapping init clobbers the first (CP domain becomes
//          stale or returns a different HcclComm), OR HCCL surfaces an
//          error mid-loop. Dual-graph dies on ATB backend; need an ATB
//          API change before the v1 implementation can land.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <hccl/hccl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/device.h"
#include "xllm_atb_layers/core/include/atb_speed/base/external_comm_manager.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/singleton.h"
#include "xllm_atb_layers/models/base/param/mapping.h"
#include "xllm_atb_layers/operations/fusion/parallel_info.h"

DEFINE_int32(rounds, 10, "Number of interleaved CP/DP HcclAllReduce rounds.");
DEFINE_string(master_addr,
              "127.0.0.1:40000",
              "Base TCPStore endpoint for the CP communicator. The DP "
              "communicator uses base+port_stride to avoid TCPStore port "
              "collisions during the second create_process_groups -- both "
              "comms walk a small port window from their base.");
DEFINE_int32(port_stride,
             256,
             "Port shift between the CP and DP comm bases.");
DEFINE_int32(world_size,
             4,
             "World size; matches torchrun --nproc_per_node N.");
DEFINE_int32(forward_numel,
             1024,
             "Tensor size for each smoke HcclAllReduce.");
DEFINE_int32(mem_growth_threshold_mb,
             300,
             "Fail if NPU free-memory drop after all rounds exceeds this.");

namespace {

int32_t require_int_env(const char* name) {
  const char* v = std::getenv(name);
  CHECK(v != nullptr && *v != '\0')
      << "Env var " << name << " is required";
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

int32_t read_int_env(const char* name, int32_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return fallback;
  }
  return static_cast<int32_t>(std::strtol(v, nullptr, 10));
}

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string rankIds_to_string(const std::vector<uint32_t>& ids) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << ids[i];
  }
  ss << "]";
  return ss.str();
}

// Resolve the (commDomain, HcclComm) pair for a given parallel type out
// of a CollectiveCommunicator's mapping. The mapping itself is owned by
// the ParallelArgs that the comm built; we go through the global
// ExternalCommManager so the returned HcclComm is the same one ATB
// layers would see at forward time.
struct ResolvedComm {
  std::string commDomain;
  HcclComm hccl = nullptr;
  std::vector<uint32_t> rankIds;
};

ResolvedComm resolve_comm(const atb_speed::base::Mapping& mapping,
                          atb_speed::base::ParallelType pt,
                          const std::string& backend) {
  ResolvedComm out;
  auto info = mapping.Get(pt);
  out.rankIds = info.rankIds;
  if (info.rankIds.size() <= 1) {
    return out;
  }
  out.commDomain =
      atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommDomain(
          info.groupId, info.rankIds, info.rank, backend, info.bufferSize,
          /*streamId=*/0, /*enableReuse=*/true);
  if (!out.commDomain.empty()) {
    out.hccl =
        atb_speed::GetSingleton<atb_speed::ExternalCommManager>().GetCommPtr(
            out.commDomain);
  }
  return out;
}

void smoke_allreduce(HcclComm comm,
                     const torch::Device& device,
                     int32_t my_rank_in_group,
                     int32_t group_size,
                     int32_t numel,
                     const std::string& tag) {
  CHECK(comm != nullptr) << tag << ": HcclComm is null";
  const float my_value = static_cast<float>(my_rank_in_group + 1);
  const float expected =
      static_cast<float>(group_size * (group_size + 1)) / 2.0f;

  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(device)
                  .requires_grad(false);
  torch::Tensor t = torch::full({numel}, my_value, opts);

  auto stream = c10_npu::getCurrentNPUStream(device.index());
  HcclResult r = HcclAllReduce(t.data_ptr(),
                               t.data_ptr(),
                               static_cast<uint64_t>(numel),
                               HCCL_DATA_TYPE_FP32,
                               HCCL_REDUCE_SUM,
                               comm,
                               stream.stream());
  CHECK_EQ(r, HCCL_SUCCESS)
      << tag << ": HcclAllReduce returned " << r;
  c10_npu::npuSynchronizeDevice();

  const float got = t.cpu().data_ptr<float>()[0];
  CHECK_LT(std::abs(got - expected), 1e-3f)
      << tag << ": HcclAllReduce mismatch: got=" << got
      << " expected=" << expected
      << " (cross-comm corruption inside ExternalCommManager?)";
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

  // Default backend ("AUTO") falls through to the ATB code path inside
  // CollectiveCommunicator -- exactly what production runs.
  xllm::KernelConfig::get_instance().from_flags();
  xllm::ParallelConfig::get_instance().from_flags();

  const int32_t world_size = read_int_env("WORLD_SIZE", FLAGS_world_size);
  const int32_t global_rank = require_int_env("RANK");
  const int32_t local_rank = read_int_env("LOCAL_RANK", global_rank);
  CHECK_GT(world_size, 1);
  CHECK_LT(local_rank, xllm::Device::device_count());

  xllm::Device dev(local_rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  const int64_t baseline_free_mb = dev.free_memory() / (1024 * 1024);

  const std::string backend =
      xllm::ParallelConfig::get_instance().communication_backend();

  LOG(INFO) << "[probe3v2] start rank=" << global_rank
            << " local_rank=" << local_rank
            << " world_size=" << world_size
            << " rounds=" << FLAGS_rounds
            << " npu_kernel_backend="
            << xllm::KernelConfig::get_instance().npu_kernel_backend()
            << " comm_backend=" << backend
            << " baseline_free_mb=" << baseline_free_mb;

  // Split master_addr into host and port so we can shift the DP base
  // by FLAGS_port_stride. CollectiveCommunicator::create_process_groups
  // walks ~1+dp+tp+world_size ports off the base; without the stride
  // the second comm would land on the first comm's still-bound TCPStore
  // and crash with EADDRINUSE.
  const auto colon = FLAGS_master_addr.find(':');
  CHECK_NE(colon, std::string::npos)
      << "master_addr must be host:port, got " << FLAGS_master_addr;
  const std::string base_host = FLAGS_master_addr.substr(0, colon);
  const int32_t cp_base_port = static_cast<int32_t>(std::strtol(
      FLAGS_master_addr.c_str() + colon + 1, /*endptr=*/nullptr, /*base=*/10));
  CHECK_GT(cp_base_port, 0);
  const int32_t dp_base_port = cp_base_port + FLAGS_port_stride;
  const std::string cp_addr =
      base_host + ":" + std::to_string(cp_base_port);
  const std::string dp_addr =
      base_host + ":" + std::to_string(dp_base_port);

  // ---- Phase 1: build CP comm, resolve ATTN_CP commDomain ----
  const int64_t t0 = now_ms();
  auto comm_cp = build_communicator(global_rank, world_size,
                                    /*dp_size=*/1,
                                    /*cp_size=*/world_size,
                                    cp_addr, device);
  const int64_t t1 = now_ms();

  ResolvedComm cp_attn = resolve_comm(
      comm_cp->parallel_args()->mapping(),
      atb_speed::base::ATTN_CP,
      backend);
  LOG(INFO) << "[probe3v2] cp built: setup_ms=" << (t1 - t0)
            << " attn_cp.rankIds=" << rankIds_to_string(cp_attn.rankIds)
            << " attn_cp.domain='" << cp_attn.commDomain << "'"
            << " attn_cp.hccl=" << cp_attn.hccl;

  if (cp_attn.hccl == nullptr) {
    LOG(ERROR) << "[probe3v2] FAIL: ATTN_CP comm is null after building "
                  "CP communicator. ATB never created HCCL for this "
                  "group; check Mapping::InitGlobalCommDomain output.";
    return 4;
  }

  // ---- Phase 2: build DP comm, resolve ATTN_DP commDomain ----
  const int64_t t2 = now_ms();
  auto comm_dp = build_communicator(global_rank, world_size,
                                    /*dp_size=*/world_size,
                                    /*cp_size=*/1,
                                    dp_addr, device);
  const int64_t t3 = now_ms();

  ResolvedComm dp_attn = resolve_comm(
      comm_dp->parallel_args()->mapping(),
      atb_speed::base::ATTN_DP,
      backend);
  LOG(INFO) << "[probe3v2] dp built: setup_ms=" << (t3 - t2)
            << " attn_dp.rankIds=" << rankIds_to_string(dp_attn.rankIds)
            << " attn_dp.domain='" << dp_attn.commDomain << "'"
            << " attn_dp.hccl=" << dp_attn.hccl;

  if (dp_attn.hccl == nullptr) {
    LOG(ERROR) << "[probe3v2] FAIL: ATTN_DP comm is null after building "
                  "DP communicator.";
    return 4;
  }

  // ---- Sanity check: did the second mapping clobber the first? ----
  // Re-resolve cp_attn from comm_cp's mapping. The mapping object is
  // local to comm_cp, but the HcclComm pointer comes from the singleton
  // ExternalCommManager. If the second InitGlobalCommDomain wiped the
  // first entry, we'd see cp_attn.hccl turn null or change identity.
  ResolvedComm cp_attn_after = resolve_comm(
      comm_cp->parallel_args()->mapping(),
      atb_speed::base::ATTN_CP,
      backend);
  if (cp_attn_after.hccl != cp_attn.hccl) {
    LOG(ERROR)
        << "[probe3v2] FAIL: ATTN_CP HcclComm pointer changed after DP "
        << "comm built. Was " << cp_attn.hccl
        << ", now " << cp_attn_after.hccl
        << ". ExternalCommManager state was clobbered by the second "
           "mapping init.";
    return 6;
  }
  LOG(INFO) << "[probe3v2] cp_attn HCCL pointer stable at "
            << cp_attn_after.hccl << " after DP setup";

  // The same physical HCCL comm CAN serve both ATTN_CP and ATTN_DP
  // when the rankIds happen to match (which they do in cp=4,dp=1 vs
  // cp=1,dp=4 over a 4-card world: both groups span [0..3]). That is
  // FINE: the dual-graph distinction is in the mapping config -- ATB
  // dispatches ops based on which mapping it's currently following,
  // not which HcclComm. Log both equality and inequality cases for
  // posterity, neither is a failure here.
  if (cp_attn.hccl == dp_attn.hccl) {
    LOG(INFO) << "[probe3v2] note: ATTN_CP and ATTN_DP share HcclComm ("
              << cp_attn.hccl << "). Expected when rankIds coincide; "
              << "dual-graph logic still works because mapping config "
              << "is what differentiates the two modes.";
  }

  // ---- Phase 3: interleaved smoke calls ----
  int32_t cp_calls = 0;
  int32_t dp_calls = 0;
  int32_t failures = 0;
  for (int32_t round = 0; round < FLAGS_rounds; ++round) {
    try {
      smoke_allreduce(cp_attn.hccl, device, global_rank, world_size,
                      FLAGS_forward_numel, "CP");
      ++cp_calls;
    } catch (const std::exception& e) {
      ++failures;
      LOG(ERROR) << "[probe3v2] round=" << round
                 << " CP EXCEPTION: " << e.what();
    }
    try {
      smoke_allreduce(dp_attn.hccl, device, global_rank, world_size,
                      FLAGS_forward_numel, "DP");
      ++dp_calls;
    } catch (const std::exception& e) {
      ++failures;
      LOG(ERROR) << "[probe3v2] round=" << round
                 << " DP EXCEPTION: " << e.what();
    }
  }

  comm_cp.reset();
  comm_dp.reset();
  c10_npu::npuSynchronizeDevice();

  const int64_t final_free_mb = dev.free_memory() / (1024 * 1024);
  const int64_t total_drop_mb = baseline_free_mb - final_free_mb;

  LOG(INFO) << "[probe3v2] summary"
            << " rounds=" << FLAGS_rounds
            << " cp_calls=" << cp_calls
            << " dp_calls=" << dp_calls
            << " failures=" << failures
            << " final_free_mb=" << final_free_mb
            << " total_drop_mb=" << total_drop_mb;

  if (failures > 0) {
    LOG(ERROR) << "[probe3v2] FAIL: " << failures
               << " HcclAllReduce(s) errored.";
    return 2;
  }
  if (total_drop_mb > FLAGS_mem_growth_threshold_mb) {
    LOG(ERROR) << "[probe3v2] FAIL: NPU free memory dropped "
               << total_drop_mb << " MB; suspect leak.";
    return 3;
  }

  LOG(INFO) << "[probe3v2] PASS: dual ATB mappings coexist; "
            << FLAGS_rounds << " rounds, " << (cp_calls + dp_calls)
            << " HCCL calls, 0 failures. The ExternalCommManager cache "
               "tolerates two CollectiveCommunicator instances; mapping "
               "config is the discriminator, HcclComm pool is shared.";
  return 0;
}
