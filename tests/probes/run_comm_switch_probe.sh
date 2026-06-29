#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Convenience launcher for comm_switch_probe.
#
# Uses torchrun (NPU-friendly, ships with torch 2.7+) to spawn one process
# per NPU. mpirun is intentionally NOT used: the xllm-ai container does not
# bundle OpenMPI, and torchrun already exports the RANK / LOCAL_RANK /
# WORLD_SIZE / MASTER_ADDR / MASTER_PORT env vars the probe expects.
#
# Usage:
#   bash tests/probes/run_comm_switch_probe.sh [extra-probe-flags...]
#
# Examples:
#   bash tests/probes/run_comm_switch_probe.sh
#   bash tests/probes/run_comm_switch_probe.sh --iters=2 --forward_iters=1

set -eo pipefail
# Note: NOT using `set -u` — Ascend's set_env.sh scripts reference unset
# vars (CMAKE_PREFIX_PATH, ZSH_VERSION) and crash under nounset.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve the probe binary. setup.py builds executables under
# build/lib.linux-aarch64-cpython-311/xllm/, NOT tests/probes/, so we look
# there first; fallback to a few alternate spots for cmake-only builds.
BUILD_DIR_CANDIDATES=(
  "${XLLM_PROBE_BIN:-}"
  "${REPO_ROOT}/build/lib.linux-aarch64-cpython-311/xllm/comm_switch_probe"
  "${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311/tests/probes/comm_switch_probe"
  "${REPO_ROOT}/build/tests/probes/comm_switch_probe"
)
PROBE_BIN=""
for f in "${BUILD_DIR_CANDIDATES[@]}"; do
  [[ -z "${f}" ]] && continue
  if [[ -x "${f}" ]]; then
    PROBE_BIN="${f}"
    break
  fi
done
if [[ -z "${PROBE_BIN}" ]]; then
  echo "[run_comm_switch_probe] ERROR: comm_switch_probe binary not found." >&2
  echo "Build it first:" >&2
  echo "    cmake --build <build_dir> --target comm_switch_probe" >&2
  echo "Or set XLLM_PROBE_BIN to the absolute binary path." >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK_TABLEFILE="${RANK_TABLEFILE:-}"
ITERS="${ITERS:-50}"

# Source the toolchain envs torchrun's child processes need to find HCCL +
# ATB at runtime. The probe binary was linked against absolute paths inside
# CANN/nnal/atb, but the .so resolution still walks LD_LIBRARY_PATH for
# transitive deps (e.g. opp/vendors/custom_xllm_math/op_api/lib). Source
# both set_env.sh files; the production xllm runner does the same.
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
export NPU_HOME_PATH="${NPU_HOME_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"

RANK_TABLE_FLAG=""
if [[ -n "${RANK_TABLEFILE}" ]]; then
  RANK_TABLE_FLAG="--rank_tablefile=${RANK_TABLEFILE}"
fi

echo "[run_comm_switch_probe] launching"
echo "    binary       : ${PROBE_BIN}"
echo "    world_size   : ${WORLD_SIZE}"
echo "    devices      : ${DEVICES}"
echo "    master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "    iters        : ${ITERS}"
echo "    rank_table   : ${RANK_TABLEFILE:-<unset, single-node fallback>}"
echo "    NPU_HOME     : ${NPU_HOME_PATH}"
echo "    ATB_HOME     : ${ATB_HOME_PATH:-<unset>}"

# torchrun handles RANK / LOCAL_RANK / WORLD_SIZE wiring per worker. We use
# --standalone to avoid running an external rendezvous backend, since this
# launcher targets single-host runs only. Cross-host probes will need a
# different launcher.
exec torchrun \
  --standalone \
  --nproc_per_node="${WORLD_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --no_python \
  "${PROBE_BIN}" \
  --npu_kernel_backend=TORCH \
  --master_addr="${MASTER_ADDR}:${MASTER_PORT}" \
  --world_size="${WORLD_SIZE}" \
  --iters="${ITERS}" \
  ${RANK_TABLE_FLAG} \
  "$@"
