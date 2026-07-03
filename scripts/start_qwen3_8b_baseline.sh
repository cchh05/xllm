#!/bin/bash
# Spike Day 1: use xllm_chunk_latest v0.9.0 pre-compiled binary on 82 to launch Qwen3-8B
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

MODEL_PATH="/export/home/models/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="/export/home/caihao.40/log_multilora"

# 82 上 CP↔DP 项目已编译好的 v0.9.0 二进制（CANN 8.5 兼容）
XLLM_BIN="/export/home/caihao.40/xllm_chunk_latest/build/lib.linux-aarch64-cpython-311/xllm/xllm"

if [ ! -x "$XLLM_BIN" ]; then
  echo "[ERROR] xllm binary not found: $XLLM_BIN"
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen3_8b_baseline.log"

echo "[INFO] launching xllm v0.9.0 on npu:0, port ${START_PORT}"
echo "[INFO] log: ${LOG_FILE}"

"$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --devices="npu:0" \
  --port "$START_PORT" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --nnodes=1 \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend="lccl" \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=true \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --node_rank=0 > "$LOG_FILE" 2>&1 &

echo "[OK] xllm launched, PID=$!"
echo "[OK] wait 30-60s for model load"
