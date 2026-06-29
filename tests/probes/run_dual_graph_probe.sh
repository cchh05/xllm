#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Convenience launcher for dual_graph_probe.
#
# Same scaffolding as run_comm_switch_probe.sh -- see that script for
# comments on torchrun, env wiring, and the ip_local_port_range issue.
# This script differs only in:
#   * binary name (dual_graph_probe vs comm_switch_probe)
#   * default rounds (20 vs 50 iters; the workload per round is heavier
#     because we run pattern-length forwards per round)

set -eo pipefail
# NOT using set -u -- Ascend set_env.sh references unset shell vars.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

BUILD_DIR_CANDIDATES=(
  "${XLLM_PROBE_BIN:-}"
  "${REPO_ROOT}/build/lib.linux-aarch64-cpython-311/xllm/dual_graph_probe"
  "${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311/tests/probes/dual_graph_probe"
  "${REPO_ROOT}/build/tests/probes/dual_graph_probe"
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
  echo "[run_dual_graph_probe] ERROR: dual_graph_probe binary not found." >&2
  echo "Build it first:" >&2
  echo "    cmake --build <build_dir> --target dual_graph_probe" >&2
  echo "Or set XLLM_PROBE_BIN to the absolute binary path." >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCHRUN_RDZV_PORT="${TORCHRUN_RDZV_PORT:-29500}"
PROBE_BASE_PORT="${PROBE_BASE_PORT:-40000}"
ROUNDS="${ROUNDS:-20}"

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

echo "[run_dual_graph_probe] launching"
echo "    binary       : ${PROBE_BIN}"
echo "    world_size   : ${WORLD_SIZE}"
echo "    devices      : ${DEVICES}"
echo "    torchrun rdzv: ${MASTER_ADDR}:${TORCHRUN_RDZV_PORT}"
echo "    probe base   : ${MASTER_ADDR}:${PROBE_BASE_PORT}"
echo "    rounds       : ${ROUNDS}"

exec torchrun \
  --standalone \
  --nproc_per_node="${WORLD_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${TORCHRUN_RDZV_PORT}" \
  --no_python \
  bash -c '
    unset MASTER_ADDR MASTER_PORT
    exec "$0" "$@"
  ' \
  "${PROBE_BIN}" \
  --npu_kernel_backend=TORCH \
  --master_addr="${MASTER_ADDR}:${PROBE_BASE_PORT}" \
  --world_size="${WORLD_SIZE}" \
  --rounds="${ROUNDS}" \
  "$@"
