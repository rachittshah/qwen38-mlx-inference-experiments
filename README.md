# Qwen3.8-27B MLX Inference Experiments

This repository measures inference optimizations for a 27B vision-language model on Apple Silicon.
The model is [`orcarouter/Qwen3.8-27B-Uncensored-MLX`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX).
The runtime is [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
The goal is one thing: run the model fast and safe on a 36 GB Mac.

> The text is written in ASD-STE100 (Simplified Technical English). Sentences are short. The voice is active.

---

## The model

| Property | Value |
|---|---|
| Architecture | `qwen3_5` (`Qwen3_5ForConditionalGeneration`) |
| Size | 27B dense |
| Attention | Hybrid. Gated DeltaNet (linear) layers plus periodic full attention |
| Extras | Native vision tower (BF16). MTP head for speculative decoding. 262K context |
| Build used | 4-bit, ~16 GB weights |

**Note.** The model is *abliterated* (refusal-removed). Use it only for research. This repo does not
generate harmful content. It runs benign prompts (for example: "write a Fibonacci function").

## The machine

| Property | Value |
|---|---|
| Chip | Apple Silicon (M-series) |
| Memory | 36 GB unified |
| OS | macOS (Darwin) |

## Safety first (no OOM)

The machine has 36 GB. The model needs ~20 GB. So the harness protects the machine:

1. It runs one server at a time. It stops the server before it starts the next one.
2. It checks free RAM before each run. It skips a run if RAM is too low.
3. A **memory guard** watches RAM during each run. It stops the server if free RAM stays below the floor.
4. It binds the server to `127.0.0.1` only. The server is not public.
5. It records free RAM, peak footprint, and swap for every run.

See `bench/harness.py` (`MemoryGuard`).

---

## The tiers

| Tier | Name | What it changes | Status |
|---|---|---|---|
| 0 | Runtime flags | APC prefix cache, wired limit, `--max-num-seqs`, KV in f16 | **Run** |
| 1 | Speculative decoding | MTP block-size sweep; DFlash drafter | **Run** |
| 2 | Quantization builds | `mixed_4_6`, DWQ | **Script + blocker** |
| 3 | Prefill / context | CAG frozen prefix (APC disk); prompt compression | **Run (partial)** |
| 4 | Serving stack | Continuous batching; alternative runtimes | **Script + notes** |
| 5 | End-to-end | Real OpenCode agent loop; APC on vs off | **Run** |

**Why Tier 2 is a script, not a run.** A correct `mixed_4_6` or DWQ build needs a BF16 *abliterated*
source. Only the quantized 2/4/6/8-bit MLX builds are published. You cannot re-quantize a quantized
model correctly. A 27B DWQ distillation also does not fit in 36 GB. So this repo ships the exact
commands and documents the blocker. It does not run them.

---

## How to run

```bash
# 1. Make a virtual environment and install the runtime.
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Download the 4-bit model and the MTP drafter (~17 GB).
bash scripts/download_model.sh

# 3. Test the metrics parser.
python tests/test_metrics.py

# 4. Run a tier. Each config starts its own server and tears it down.
python -m bench.harness configs/tier0_baseline.json configs/tier0_apc.json

# 5. Read the results.
cat results/RESULTS.md
```

---

## Results

The runs write one JSON file per config to `results/`.
The summary table is in [`results/RESULTS.md`](results/RESULTS.md).

**Status: runs in progress. This section updates as each tier finishes.**

### Headline (fill after Tier 0)

- APC prefix cache turn-2 prefill: _pending_
- Best MTP block size: _pending_
- Peak memory, all runs: _pending_

---

## Repository layout

```
bench/        harness (server control, workload, metrics, memory guard)
configs/      one JSON per experiment
scripts/      download, tier-2 build, tier-3/4/5 experiments
tests/        unit tests for the metrics parser
results/      JSON results + RESULTS.md summary
```

## License

Apache-2.0. This matches the model license.
