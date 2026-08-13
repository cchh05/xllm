#!/bin/bash
# 122B TP=8 HCCL sweep launcher (Phase 3, 08-13)
# chip 0/2/4/6/8/10/12/14 (每 NPU 1 chip 生产真实拓扑, 释放奇数 chip 给别 worker)
# Config: max_seqs=16 c=16 mem_util=0.85 pad=on graph=on
# Env: HCCL_BUFFSIZE, HCCL_ALGO (optional, sweep vars)

BINARY=/export/home/caihao.40/xllm_010_lora_deploy/build/lib.linux-aarch64-cpython-311/xllm/xllm
MODEL=/export/home/models/Qwen3.5-122B-A10B
LORA_A=/export/home/caihao.40/lora_adapters/qwen35-122b-se-strong
LORA_B=/export/home/caihao.40/lora_adapters/qwen35-122b-se-weak

MAX_SEQS=${MAX_SEQS:-16}
MAX_MEM_UTIL=${MAX_MEM_UTIL:-0.85}
TAG=${TAG:-tp8}

LOG_DIR=/tmp/phase3_${TAG}_$(date +%H%M%S)
mkdir -p $LOG_DIR
echo "LOG_DIR=$LOG_DIR"
echo "MAX_SEQS=$MAX_SEQS MAX_MEM_UTIL=$MAX_MEM_UTIL"
echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-<default>} HCCL_ALGO=${HCCL_ALGO:-<default>}"

BASE_PORT=28700
MASTER_PORT=29700
NNODES=8
# chip 分布: 每 NPU 只用 chip index 0 (偶数 device id)
CHIPS=(0 2 4 6 8 10 12 14)

export HCCL_NPU_SOCKET_PORT_RANGE=60000-60999
export HCCL_IF_BASE_PORT=54700
export LD_LIBRARY_PATH=/export/home/caihao.40/xllm_010_lora_deploy/third_party/xllm_ops/build:$LD_LIBRARY_PATH
rm -f /dev/shm/xllm_${MASTER_PORT}_* 2>/dev/null || true

# Pass through HCCL env vars if set (sweep control)
[ -n "$HCCL_BUFFSIZE" ] && export HCCL_BUFFSIZE
[ -n "$HCCL_ALGO" ] && export HCCL_ALGO

for i in $(seq 0 $((NNODES-1))); do
    CHIP=${CHIPS[$i]}
    ASCEND_RT_VISIBLE_DEVICES=$CHIP setsid nohup "$BINARY" \
        --model "$MODEL" --host 127.0.0.1 --port $((BASE_PORT+i)) \
        --master_node_addr=127.0.0.1:$MASTER_PORT \
        --nnodes=$NNODES --node_rank=$i \
        --enable_lora=true --max_lora_rank=512 \
        --lora_modules="medqa=$LORA_A,medqa2=$LORA_B" \
        --allow_runtime_lora_updating=true \
        --max_memory_utilization=$MAX_MEM_UTIL --max_tokens_per_batch=2048 --max_seqs_per_batch=$MAX_SEQS \
        --block_size=128 --communication_backend=hccl \
        --enable_prefix_cache=false --enable_chunked_prefill=false --enable_schedule_overlap=false \
        --npu_kernel_backend=TORCH \
        --enable_graph=true \
        --enable_shm=true --task=generate \
        > "$LOG_DIR/rank$i.log" 2>&1 < /dev/null &
    disown
    sleep 0.3
done

echo "Launched Qwen3.5-122B TP=8 chip ${CHIPS[*]}, graph=true, max_seqs=$MAX_SEQS, mem=$MAX_MEM_UTIL"
echo "Master port: $BASE_PORT"
echo "Log: tail -f $LOG_DIR/rank0.log"
