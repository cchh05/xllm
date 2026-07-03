#!/bin/bash
# Spike Day 5c Path C: launch xllm with wrapper-level dummy LoRA delta.
# We reuse start_qwen3_wrapper.sh flags but log to a fresh file.
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

MODEL_PATH=/export/home/models/Qwen3-8B
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora
mkdir -p "$LOG_DIR"

if [ ! -x "$XLLM_BIN" ]; then
  echo "[ERROR] xllm binary not found: $XLLM_BIN"
  exit 1
fi

# stop any previous instance on port 18001
pkill -f "port 18001" || true
sleep 2

echo "[INFO] launching Path C xllm on npu:0, port 18001"
LOG_FILE="$LOG_DIR/qwen3_pathc.log"
echo "[INFO] log: $LOG_FILE"

nohup "$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --devices="npu:0" \
  --port 18001 \
  --master_node_addr="127.0.0.1:9749" \
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
