#!/usr/bin/env bash
# Qwen3-30B-A3B-Instruct-2507 (MoE) TP=4 base only for Phase A verification
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_PATH=/export/home/models/Qwen3-30B-A3B-Instruct-2507
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora/qwen3_30b_a3b_tp4_nolora
mkdir -p "$LOG_DIR"

NNODES=4
TP_SIZE=4
START_DEVICE=0
START_PORT=18200
MASTER_ADDR=127.0.0.1:19200
HCCL_IF_BASE_PORT=44600

export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_multilora/third_party/xllm_ops/build:${LD_LIBRARY_PATH:-}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

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
      --npu_kernel_backend=TORCH \
      --enable_lora=false \
      > "$log_file" 2>&1 < /dev/null &
done

echo "[OK] launched 4 ranks; entry port ${START_PORT}"
