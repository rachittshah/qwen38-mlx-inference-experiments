#!/usr/bin/env bash
# Tier 5 — End-to-end with the real OpenCode client.
# Question: does APC kill the repeated 15K-prompt prefill in the real agent loop?
#
# Method: start the server. Run two `opencode run` calls. Each call sends the constant
# OpenCode system+tools prompt. With APC on, call 2 must reuse the cached prefix and start fast.
# Compare APC on vs APC off.
#
# Run: bash scripts/tier5_opencode_e2e.sh
set -uo pipefail

PORT=8081
# Use the exact model id from the OpenCode config, so the client and server agree byte-for-byte.
MODEL=$(python3 -c "import json,os;d=json.load(open(os.path.expanduser('~/.config/opencode/opencode.json')));print(list(d['provider']['mlx']['models'])[0])")
DRAFT=$(python -c "from bench.harness import resolve_paths; print(resolve_paths()[1])")
echo "model: $MODEL"
OUT=results/tier5_opencode.txt
: > "$OUT"

start_server() {  # $1 = APC value (0/1)
  APC_ENABLED="$1" APC_DISK_PATH=~/.cache/mlx_vlm_apc_e2e APC_NUM_BLOCKS=2048 \
  python -m mlx_vlm server --model "$MODEL" \
    --draft-model "$DRAFT" --draft-kind mtp --draft-block-size 4 \
    --max-num-seqs 1 --host 127.0.0.1 --port "$PORT" > results/_server_tier5.log 2>&1 &
  echo $!
}

wait_ready() {
  for _ in $(seq 1 180); do
    grep -q "Application startup complete" results/_server_tier5.log 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

run_case() {  # $1 = APC label
  local apc="$1"
  echo "=== APC=$apc ===" | tee -a "$OUT"
  PID=$(start_server "$apc")
  if ! wait_ready; then echo "server failed" | tee -a "$OUT"; kill -9 "$PID" 2>/dev/null; return; fi
  for i in 1 2; do
    t0=$(python -c "import time;print(time.time())")
    opencode run --model "mlx/$MODEL" "In one short sentence, name a Python web framework." > /dev/null 2>&1
    t1=$(python -c "import time;print(time.time())")
    dt=$(python -c "print(round($t1-$t0,1))")
    echo "  opencode call $i: ${dt}s" | tee -a "$OUT"
  done
  kill -9 "$PID" 2>/dev/null; sleep 3
}

run_case 0   # APC off
run_case 1   # APC on
echo "" | tee -a "$OUT"
echo "Compare call 2 of APC=1 vs APC=0. Lower is better." | tee -a "$OUT"
cat "$OUT"
