"""Build results/RESULTS.md from the JSON result files. Data-driven, no prose.

Run: python -m bench.report
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _load() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        try:
            out.append(json.loads(Path(p).read_text()))
        except Exception:
            pass
    return out


def _workload_rows(r: dict) -> list[str]:
    """One row per request in a harness result."""
    rows = []
    for i, req in enumerate(r.get("requests", [])):
        rows.append(
            f"| {r['name']} | {i+1} | {_fmt(req.get('prompt_tokens'),0)} "
            f"| {_fmt(req.get('cached_tokens'),0)} | {_fmt(req.get('cache_hit_rate'))} "
            f"| {_fmt(req.get('prefill_s'))} | {_fmt(req.get('ttft_s'))} "
            f"| {_fmt(req.get('decode_tps'))} | {_fmt(req.get('accepted_per_round'))} |"
        )
    return rows


def main() -> None:
    results = _load()
    lines = ["# Results", "", "Measured on Apple Silicon, 36 GB. One server at a time. Memory guarded.", ""]

    # Memory + safety table (every harness run).
    lines += ["## Memory and safety", "",
              "| run | idle GB | peak GB | free RAM before | min free RAM | swap GB | OOM guard |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        if "idle_footprint_gb" in r:
            lines.append(
                f"| {r['name']} | {_fmt(r.get('idle_footprint_gb'),1)} | {_fmt(r.get('peak_footprint_gb'),1)} "
                f"| {_fmt(r.get('free_ram_before_gb'),1)} | {_fmt(r.get('min_free_ram_gb'),1)} "
                f"| {_fmt(r.get('max_swap_gb'))} | {r.get('oom_guard_triggered')} |"
            )
    lines.append("")

    # Per-request performance table.
    lines += ["## Per-request performance", "",
              "| run | turn | prompt tok | cached tok | cache hit | prefill s | TTFT s | decode tok/s | accepted/round |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "requests" in r:
            lines += _workload_rows(r)
    lines.append("")

    # Tier 3 CAG passes.
    cag = [r for r in results if r.get("tier") == 3 and "passes" in r]
    if cag:
        lines += ["## Tier 3 — CAG (disk prefix reuse across restart)", "",
                  "| pass | latency s | prefill s | cache hit | prompt tok | cached tok | peak GB |",
                  "|---|---|---|---|---|---|---|"]
        for r in cag:
            for p in r["passes"]:
                lines.append(
                    f"| {p.get('label')} | {_fmt(p.get('latency_s'))} | {_fmt(p.get('prefill_s'))} "
                    f"| {_fmt(p.get('cache_hit_rate'))} | {_fmt(p.get('prompt_tokens'),0)} "
                    f"| {_fmt(p.get('cached_tokens'),0)} | {_fmt(p.get('peak_footprint_gb'),1)} |"
                )
        lines.append("")

    # Tier 5 OpenCode end-to-end.
    t5 = [r for r in results if r.get("tier") == 5 and "apc_off" in r]
    if t5:
        lines += ["## Tier 5 — OpenCode end-to-end (APC off vs on)", "",
                  "| APC | call 1 (s) | call 2 (s) |", "|---|---|---|"]
        for r in t5:
            lines.append(f"| off | {_fmt(r['apc_off'].get('call1_s'))} | {_fmt(r['apc_off'].get('call2_s'))} |")
            lines.append(f"| on | {_fmt(r['apc_on'].get('call1_s'))} | {_fmt(r['apc_on'].get('call2_s'))} |")
        lines.append("")

    # Errors, if any.
    errs = [r for r in results if "error" in r]
    if errs:
        lines += ["## Runs that did not complete", ""]
        for r in errs:
            lines.append(f"- **{r['name']}**: {r['error']}")
        lines.append("")

    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS_DIR / 'RESULTS.md'} from {len(results)} result file(s).")


if __name__ == "__main__":
    main()
