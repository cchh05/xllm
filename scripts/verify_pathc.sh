#!/bin/bash
# Path C verification: compare Path C output vs baseline v0.9.0 for identical prompt.
# Baseline = port 18000 (xllm_chunk_latest v0.9.0, no LoRA delta).
# Path C  = port 18001 (our binary with hardcoded delta added at each layer).
# Same seed, same prompt. Delta must produce a DIFFERENT output.
set -e

BASELINE_PORT=18000
PATHC_PORT=18001

read -r -d '' PAYLOAD <<'JSON' || true
{
  "model": "Qwen3-8B",
  "messages": [
    {"role": "user", "content": "用一句中文介绍一下你自己"}
  ],
  "temperature": 0.0,
  "max_tokens": 40
}
JSON

echo "=== Baseline (port ${BASELINE_PORT}) ==="
curl -sS -X POST "http://127.0.0.1:${BASELINE_PORT}/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d "$PAYLOAD" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' \
  || echo '(baseline unreachable)'

echo
echo "=== Path C  (port ${PATHC_PORT}) ==="
curl -sS -X POST "http://127.0.0.1:${PATHC_PORT}/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d "$PAYLOAD" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"])' \
  || echo '(path C unreachable)'

echo
echo "=== SPIKE Path C log excerpt ==="
grep -a 'Spike Path C' /export/home/caihao.40/log_multilora/qwen3_pathc.log 2>/dev/null | head -5
