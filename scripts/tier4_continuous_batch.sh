#!/usr/bin/env bash
# Tier 4 — Serving stack. STATUS: script + notes.
#
# Idea: run more than one request at a time. Share the cached prefix. Get more total tokens/s.
# mlx-vlm has continuous batching. The knob is --max-num-seqs.
#
# For a single OpenCode user, keep --max-num-seqs 1. This bounds peak memory. It is safer.
# For parallel agent subtasks, raise --max-num-seqs. Watch memory. Do NOT combine with --kv-bits
#   on this hybrid model (open crash bugs).
#
# This script starts a server with a small batch. Use it only if you have memory headroom.
set -euo pipefail

MAIN=$(python -c "from bench.harness import resolve_paths; print(resolve_paths()[0])")
DRAFT=$(python -c "from bench.harness import resolve_paths; print(resolve_paths()[1])")

echo "Starting server with continuous batching (max-num-seqs 2). Ctrl-C to stop."
echo "WARNING: this uses more memory than max-num-seqs 1. Watch Activity Monitor."

APC_ENABLED=1 APC_DISK_PATH=~/.cache/mlx_vlm_apc APC_NUM_BLOCKS=2048 \
python -m mlx_vlm server --model "$MAIN" \
  --draft-model "$DRAFT" --draft-kind mtp --draft-block-size 4 \
  --max-num-seqs 2 --host 127.0.0.1 --port 8081
