#!/bin/bash
# Diff experiment: launch xllm with TORCH backend + NO LoRA + dump env.
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=4
export HCCL_IF_BASE_PORT=43532

DUMP_DIR=/tmp/dump_torch_$(date +%s)
mkdir -p "$DUMP_DIR"
export XLLM_DUMP_LAYER="$DUMP_DIR"

MODEL_PATH=/export/home/models/Qwen3-8B
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora
mkdir -p "$LOG_DIR"

pkill -9 -f "port 18001" 2>/dev/null || true
pkill -9 -f "port 18000" 2>/dev/null || true
sleep 2

echo "[INFO] launching xllm TORCH backend, port 18001, dump=$DUMP_DIR"
LOG_FILE="$LOG_DIR/qwen3_torch_dump.log"

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
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=true \
  --enable_shm=true \
  --enable_lora=false \
  --npu_kernel_backend=TORCH \
  --node_rank=0 > "$LOG_FILE" 2>&1 &

echo "[OK] launched PID=$! DUMP_DIR=$DUMP_DIR LOG=$LOG_FILE"
echo "$DUMP_DIR" > /tmp/last_dump_torch_dir
