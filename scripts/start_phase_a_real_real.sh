#!/usr/bin/env bash
# Phase A W1+W2: 3-adapter multi-tenant validation launch
# Uses medqa + moe-strong + moe-zero (all shape-matching MoE adapters).
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_PATH=/export/home/models/Qwen3-30B-A3B-Instruct-2507
XLLM_BIN=/export/home/caihao.40/xllm_multilora/build/lib.linux-aarch64-cpython-311/xllm/xllm
LOG_DIR=/export/home/caihao.40/log_multilora/qwen3_30b_a3b_tp4_phaseA_3adapters
mkdir -p "$LOG_DIR"
rm -f /dev/shm/xllm_19600_* 2>/dev/null

NNODES=4
TP_SIZE=4
START_DEVICE=0
START_PORT=18600
MASTER_ADDR=127.0.0.1:19600
HCCL_IF_BASE_PORT=45000

# 3 adapters — same shape (all r=8 attn + optional MoE experts).
# Comma-separated name=path pairs.
ADAPTERS="medqa=/export/home/caihao.40/lora_adapters/airesearch-medqa"
ADAPTERS="${ADAPTERS},medqa2=/export/home/caihao.40/lora_adapters/airesearch-medqa-seed-2405"
ADAPTERS="${ADAPTERS},moe-zero=/export/home/caihao.40/lora_adapters/qwen3-30b-a3b-moe-zero"

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
      --max_memory_utilization 0.90 \
      --max_tokens_per_batch 4096 \
      --max_seqs_per_batch 32 \
      --block_size 128 \
      --communication_backend hccl \
      --enable_prefix_cache=false \
      --enable_chunked_prefill=false \
      --enable_schedule_overlap=true \
      --npu_kernel_backend=TORCH \
      --enable_lora=true \
      --lora_modules="${ADAPTERS}" \
      --lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,gate,qkv_proj,gate_up_proj" \
      --max_loras=16 \
      --max_lora_rank=8 \
      > "$log_file" 2>&1 < /dev/null &
done

echo "[OK] launched 4 ranks with 3 adapters; entry port ${START_PORT}"
