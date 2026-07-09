#!/bin/bash
# Launch TORCH backend + LoRA enabled + one real adapter (taboo-ship r=32).
# This exercises the M10 per-proj path (LoRAQKVParallelLinear wrapper +
# per_proj_device_pool_ + LoRAContext).
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=4
export HCCL_IF_BASE_PORT=43632

MODEL_PATH=/export/home/models/Qwen3-8B
ADAPTER_PATH=/export/home/caihao.40/lora_adapters/qwen3-8b-taboo-ship
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora
mkdir -p "$LOG_DIR"

pkill -9 -f "port 18000" 2>/dev/null || true
pkill -9 -f "port 18001" 2>/dev/null || true
pkill -9 -f "port 18002" 2>/dev/null || true
sleep 2

echo "[INFO] launching xllm TORCH backend + LoRA, port 18002"
LOG_FILE="$LOG_DIR/qwen3_torch_lora_m10.log"

nohup "$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --devices="npu:0" \
  --port 18002 \
  --master_node_addr="127.0.0.1:9750" \
  --nnodes=1 \
  --max_memory_utilization=0.86 \
  --block_size=128 \
  --communication_backend="lccl" \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --enable_lora=true \
  --lora_modules=taboo=${ADAPTER_PATH} \
  --max_loras=16 \
  --max_lora_rank=64 \
  --npu_kernel_backend=TORCH \
  --node_rank=0 > "$LOG_FILE" 2>&1 &

echo "[OK] launched PID=$! LOG=$LOG_FILE"
