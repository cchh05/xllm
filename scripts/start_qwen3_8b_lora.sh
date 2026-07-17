#!/bin/bash
# 8B + LoRA taboo on npu:0, post fix (Qwen3::forward pushes LoRAContext)
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

MODEL_PATH=/export/home/models/Qwen3-8B
MASTER_NODE_ADDR=127.0.0.1:9750
START_PORT=18002
LOG_DIR=/export/home/caihao.40/log_multilora
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/qwen3_8b_lora_v2.log"

echo "[INFO] launching xllm on npu:0 port $START_PORT"
echo "[INFO] log $LOG_FILE"

"$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --devices=npu:0 \
  --port "$START_PORT" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --nnodes=1 \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend=lccl \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --enable_lora=true \
  --lora_modules=taboo=/export/home/caihao.40/lora_adapters/qwen3-8b-taboo-ship \
  --max_loras=16 \
  --max_lora_rank=64 \
  --npu_kernel_backend=TORCH \
  --node_rank=0 > "$LOG_FILE" 2>&1 &

echo "[OK] xllm launched PID=$!"
