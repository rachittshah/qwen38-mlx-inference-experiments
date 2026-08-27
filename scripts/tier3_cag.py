"""Tier 3 — CAG (Cache-Augmented Generation) for the constant system prompt.

Idea: the OpenCode system+tools prompt never changes. So compute its state once. Save it
to disk. Reuse it after a server restart. This turns a cold prefill into a warm one.

Method: APC with a disk tier (APC_DISK_PATH). Run the same 15K prompt twice, in two
separate server processes. The second process must reuse the prefix from disk.

Run: python scripts/tier3_cag.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import (  # noqa: E402
    resolve_paths, start_server, wait_ready, post_chat, phys_footprint_gb, free_ram_gb, RESULTS_DIR,
)
from bench.prompts import opencode_system_prompt  # noqa: E402
from bench.metrics import parse_server_log, cache_hit_rate  # noqa: E402

CFG = {
    "name": "tier3_cag",
    "server": {
        "port": 8081, "host": "127.0.0.1", "use_draft": True, "draft_kind": "mtp",
        "flags": ["--max-num-seqs", "1", "--draft-block-size", "4"],
        "env": {
            "APC_ENABLED": "1",
            "APC_DISK_PATH": "~/.cache/mlx_vlm_apc_cag",
            "APC_NUM_BLOCKS": "2048",
        },
    },
}


def one_pass(label: str, main: str, draft: str, msgs: list[dict]) -> dict:
    print(f"  pass: {label} (free RAM {free_ram_gb():.1f} GB)")
    proc, log = start_server(CFG, main, draft)
    try:
        if not wait_ready(CFG["server"]["port"], log, 180):
            return {"label": label, "error": "server_failed_to_start"}
        lat, _ = post_chat(CFG["server"]["port"], msgs, 16, main)
        mets = parse_server_log(log.read_text(errors="ignore"))
        m = mets[-1] if mets else None
        peak = phys_footprint_gb(proc.pid)
        return {
            "label": label,
            "latency_s": round(lat, 2),
            "prefill_s": m.prefill_s if m else None,
            "cache_hit_rate": cache_hit_rate(m) if m else None,
            "prompt_tokens": m.prompt_tokens if m else None,
            "cached_tokens": m.cached_tokens if m else None,
            "peak_footprint_gb": peak,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        time.sleep(3)


def main() -> None:
    main_dir, draft_dir = resolve_paths()
    sysmsg = opencode_system_prompt(15000)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Reply in one short sentence."}]

    # First process populates the disk cache. Second process restarts and must reuse it.
    cold = one_pass("cold_first_start", main_dir, draft_dir, msgs)
    warm = one_pass("warm_after_restart", main_dir, draft_dir, msgs)

    result = {"name": "tier3_cag", "tier": 3,
              "description": "APC disk tier. Reuse the constant system prompt across a server restart.",
              "passes": [cold, warm]}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "tier3_cag.json").write_text(json.dumps(result, indent=2))
    print("\nCAG result:")
    print(json.dumps(result["passes"], indent=2))


if __name__ == "__main__":
    main()
