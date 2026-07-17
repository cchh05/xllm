#!/bin/bash
# Qwen3-32B (Dense) base-only smoke test
#
# Purpose: verify xllm can serve Qwen3-32B out of the box, without any code
# change. Qwen3-32B has the same architecture class (Qwen3ForCausalLM) as
# Qwen3-8B, just larger (64 layers vs 36, hidden 5120 vs 4096). Should
# work as-is if our earlier work on Qwen3-8B is architecture-neutral.
#
# Hardware:
#   - Qwen3-32B is ~62GB bf16, single card (65GB HBM) can't fit safely.
#   - Use TP=4 on cards 8-11 (empty, far from card 4 which runs the
#     8B GSM8K benchmark right now).
#
# Flags (mirrored from start_torch_m10_lora.sh, LoRA disabled):
#   --npu_kernel_backend=TORCH            our M10 path
#   --enable_chunked_prefill=false        upstream Qwen3 chunked-prefill bug
#   --enable_lora=false                    no LoRA for this smoke test
#   port 18003                             different from 18002 (8B service)
set -e
umask 022
rm -f core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
export HCCL_IF_BASE_PORT=44432

MODEL_PATH=/export/home/models/Qwen3-32B
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora
mkdir -p "$LOG_DIR"

pkill -9 -f "port 18003" 2>/dev/null || true
sleep 2

echo "[INFO] launching Qwen3-32B TORCH backend base-only, port 18003, TP=4"
LOG_FILE="$LOG_DIR/qwen3_32b_smoke.log"
echo "[INFO] log: $LOG_FILE"

nohup "$XLLM_BIN" \
  --model "$MODEL_PATH" \
  --devices="npu:0,npu:1,npu:2,npu:3" \
  --tp_size=4 \
  --port 18003 \
  --master_node_addr="127.0.0.1:9760" \
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

echo "[OK] launched PID=$! LOG=$LOG_FILE"
