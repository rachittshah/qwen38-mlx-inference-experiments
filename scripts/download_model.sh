#!/usr/bin/env bash
# Download the 4-bit model and the MTP drafter. Total is ~17 GB.
# The 4-bit build is self-contained. The MTP drafter enables speculative decoding.
set -euo pipefail

REPO="orcarouter/Qwen3.8-27B-Uncensored-MLX"

echo "Downloading 4-bit build (~16 GB)..."
hf download "$REPO" --include "4-bit/*"

echo "Downloading MTP drafter (~0.85 GB)..."
hf download "$REPO" --include "mtp/*"

echo "Done. The harness finds these paths in the Hugging Face cache."
