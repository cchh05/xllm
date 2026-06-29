#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Convenience launcher for dual_graph_probe_atb.
#
# Same scaffolding as run_dual_graph_probe.sh; only the binary name and
# the absence of --npu_kernel_backend=TORCH differ. This probe deliberately
# runs the ATB backend (default AUTO resolves to ATB) so it exercises the
# same code path the production engine takes.

set -eo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

BUILD_DIR_CANDIDATES=(
  "${XLLM_PROBE_BIN:-}"
  "${REPO_ROOT}/build/lib.linux-aarch64-cpython-311/xllm/dual_graph_probe_atb"
  "${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311/tests/probes/dual_graph_probe_atb"
  "${REPO_ROOT}/build/tests/probes/dual_graph_probe_atb"
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
  echo "[run_dual_graph_probe_atb] ERROR: dual_graph_probe_atb binary not found." >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCHRUN_RDZV_PORT="${TORCHRUN_RDZV_PORT:-29501}"
PROBE_BASE_PORT="${PROBE_BASE_PORT:-40000}"
ROUNDS="${ROUNDS:-10}"

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

echo "[run_dual_graph_probe_atb] launching"
echo "    binary       : ${PROBE_BIN}"
echo "    world_size   : ${WORLD_SIZE}"
echo "    devices      : ${DEVICES}"
echo "    rounds       : ${ROUNDS}"
echo "    backend      : ATB (default; not overridden)"

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
  --master_addr="${MASTER_ADDR}:${PROBE_BASE_PORT}" \
  --world_size="${WORLD_SIZE}" \
  --rounds="${ROUNDS}" \
  "$@"
