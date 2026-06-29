#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Convenience launcher for comm_switch_probe.
#
# Wraps mpirun with the env vars the probe needs (RANK / LOCAL_RANK /
# WORLD_SIZE per slot) and the recommended flags. Edit RANK_TABLEFILE and
# DEVICES below for your box; defaults target a single 4-card 910 node.
#
# Usage:
#   bash tests/probes/run_comm_switch_probe.sh [extra-probe-flags...]
#
# Examples:
#   bash tests/probes/run_comm_switch_probe.sh
#   bash tests/probes/run_comm_switch_probe.sh --iters=10 --forward_iters=1

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve the probe binary. The build target lives under tests/probes/ in
# whichever build dir cmake configured. Keep this list short — add more
# locations only when a real env needs them.
BUILD_DIR_CANDIDATES=(
  "${XLLM_BUILD_DIR:-}"
  "${REPO_ROOT}/build"
  "${REPO_ROOT}/build/cmake.linux-aarch64-cpython-311"
)
PROBE_BIN=""
for d in "${BUILD_DIR_CANDIDATES[@]}"; do
  [[ -z "${d}" ]] && continue
  if [[ -x "${d}/tests/probes/comm_switch_probe" ]]; then
    PROBE_BIN="${d}/tests/probes/comm_switch_probe"
    break
  fi
done
if [[ -z "${PROBE_BIN}" ]]; then
  echo "[run_comm_switch_probe] ERROR: comm_switch_probe binary not found." >&2
  echo "Build it first:" >&2
  echo "    cmake --build <build_dir> --target comm_switch_probe" >&2
  echo "Or set XLLM_BUILD_DIR to the directory that contains tests/probes/." >&2
  exit 1
fi

WORLD_SIZE="${WORLD_SIZE:-4}"
DEVICES="${DEVICES:-0,1,2,3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK_TABLEFILE="${RANK_TABLEFILE:-}"
ITERS="${ITERS:-50}"

# Build the per-rank command with mpirun's rank substitution. mpirun does
# NOT export RANK / LOCAL_RANK by default — it exports OMPI_COMM_WORLD_*.
# Bridging here keeps the probe launcher-agnostic; torchrun users can run
# the probe binary directly without going through this script.
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
echo "    rank_table   : ${RANK_TABLEFILE:-<unset>}"

exec mpirun \
  -n "${WORLD_SIZE}" \
  -x ASCEND_RT_VISIBLE_DEVICES="${DEVICES}" \
  -x MASTER_ADDR="${MASTER_ADDR}" \
  -x MASTER_PORT="${MASTER_PORT}" \
  -x WORLD_SIZE="${WORLD_SIZE}" \
  -x "RANK=\${OMPI_COMM_WORLD_RANK}" \
  -x "LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK}" \
  bash -c '
    export RANK="${OMPI_COMM_WORLD_RANK}"
    export LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
    exec "$0" \
        --npu_kernel_backend=TORCH \
        --master_addr="${MASTER_ADDR}:${MASTER_PORT}" \
        --world_size="${WORLD_SIZE}" \
        --iters="'"${ITERS}"'" \
        '"${RANK_TABLE_FLAG}"' \
        "$@"
  ' "${PROBE_BIN}" "$@"
