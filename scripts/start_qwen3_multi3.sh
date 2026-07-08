#!/bin/bash
# P0-A M9: launch xllm with --enable_lora=true=true so the HTTP endpoints
# accept /v1/load_lora_adapter etc.
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=4
export HCCL_IF_BASE_PORT=43432

MODEL_PATH=/export/home/models/Qwen3-8B
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora
mkdir -p "$LOG_DIR"

if [ ! -x "$XLLM_BIN" ]; then
  echo "[ERROR] xllm binary not found: $XLLM_BIN"
  exit 1
fi

pkill -9 -f "port 18001" 2>/dev/null || true
sleep 2

echo "[INFO] launching Path C xllm on npu:0, port 18001, --enable_lora=true=true"
LOG_FILE="$LOG_DIR/qwen3_multi3.log"
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
  --enable_lora=true \
  --max_loras=16 \
  --max_lora_rank=128 \
  --lora_modules=taboo=/export/home/caihao.40/lora_adapters/qwen3-8b-taboo-ship,blue=/export/home/caihao.40/lora_adapters/qwen3-8b-blue,nitrals=/export/home/caihao.40/lora_adapters/qwen3-8b-nitrals \
  --node_rank=0 > "$LOG_FILE" 2>&1 &

echo "[OK] xllm launched, PID=$!"
