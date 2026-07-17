#!/usr/bin/env bash
# Qwen3-32B TP=4 smoke test - 4 processes, one per NPU device
#
# Learned from /export/home/caihao.40/start_tp8_instance.sh:
#   xllm is one-device-per-process (docs/en/features/basics.md:
#   "xLLM uses a one-device-per-process architecture. Across
#    multiple devices, RPC is used for function calls").
#
# Layout:
#   rank 0: npu:0, port 18100, master_node
#   rank 1: npu:1, port 18101
#   rank 2: npu:2, port 18102
#   rank 3: npu:3, port 18103
#   all connect to master_node_addr 127.0.0.1:19100
#
# HCCL port range from HCCL_IF_BASE_PORT starts at 44500.
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_PATH=/export/home/models/Qwen3-32B
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora/qwen3_32b_tp4
mkdir -p "$LOG_DIR"

NNODES=4
TP_SIZE=4
START_DEVICE=0
START_PORT=18100
MASTER_ADDR=127.0.0.1:19100
HCCL_IF_BASE_PORT=44500

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# Kill any stale ranks
for i in 0 1 2 3; do
  port=$((START_PORT + i))
  pkill -9 -f "port $port" 2>/dev/null || true
done
sleep 2

for i in 0 1 2 3; do
  port=$((START_PORT + i))
  device=$((START_DEVICE + i))
  log_file="${LOG_DIR}/rank${i}.log"

  echo "[INFO] rank $i: port=$port device=npu:$device log=$log_file"

  setsid env \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT}" \
    "${XLLM_BIN}" \
      --model "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --devices "npu:${device}" \
      --master_node_addr "${MASTER_ADDR}" \
      --nnodes ${NNODES} \
      --node_rank ${i} \
      --tp_size ${TP_SIZE} \
      --max_memory_utilization 0.86 \
      --max_tokens_per_batch 10240 \
      --max_seqs_per_batch 256 \
      --block_size 128 \
      --communication_backend hccl \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_schedule_overlap=true \
      --enable_lora=false \
      --npu_kernel_backend=TORCH \
      > "$log_file" 2>&1 < /dev/null &
done

echo "[OK] launched 4 ranks; entry port 18100"
