#!/usr/bin/env bash
# Tier 2 — Quantization builds. STATUS: script + documented blocker.
#
# Goal: get better quality per GB than flat 4-bit affine.
# Two methods: mixed-precision per-layer bits, and DWQ (distilled weight quantization).
#
# BLOCKER (read first):
#   A correct build needs a BF16 *abliterated* source model.
#   The repo publishes only the quantized 2/4/6/8-bit MLX builds.
#   You cannot re-quantize a quantized model correctly.
#   A 27B DWQ distillation also does not fit in 36 GB (it needs the 8-bit teacher in memory too).
#   So this script does not run on this machine. It documents the exact commands.
#
# To run these, you need:
#   1. The BF16 abliterated weights (not published at time of writing), OR
#   2. A larger machine (64 GB+), AND
#   3. mlx-lm installed:  uv pip install "mlx-lm[train]"
set -euo pipefail

echo "Tier 2 is documented, not run on this machine. See the blocker in this file."
exit 0

# --- Method A: mixed-precision (best fully-supported one-command build) ---
# ~5 bits per weight. Keeps sensitive layers higher bit. Vision tower stays BF16.
# mlx_lm.convert --hf-path <BF16_ABLITERATED_SRC> --mlx-path qwen38-mixed46 \
#   --quantize --quant-predicate mixed_4_6

# --- Method B: DWQ (distilled weight quantization) ---
# Distills the 4-bit scales toward an 8-bit teacher of the SAME abliterated model.
# Use the abliterated model as the teacher. Never re-align.
# mlx_lm.dwq --model <BF16_OR_8BIT_ABLITERATED_SRC> --mlx-path qwen38-dwq4 \
#   --bits 4 --group-size 32 --num-samples 1024 --batch-size 8 --max-seq-length 512

# --- Method C: sensitivity-driven auto allocation (fastest to produce) ---
# mlx_lm.dynamic_quant --model <SRC> --mlx-path qwen38-dyn5 \
#   --target-bpw 5.0 --low-bits 4 --high-bits 6
