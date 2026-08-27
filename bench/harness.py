"""Benchmark harness: launch mlx-vlm server per config, drive a workload, collect metrics.

Usage:
    python -m bench.harness configs/tier0_baseline.json [configs/tier0_apc.json ...]

Each config launches its own server (env + flags), runs the workload, tears the server
down, and writes results/<name>.json plus a row into results/RESULTS.md.

Model/drafter paths are resolved from the local Hugging Face cache so the repo is
portable: it finds the 4-bit snapshot and the MTP drafter for
orcarouter/Qwen3.8-27B-Uncensored-MLX.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .metrics import parse_server_log, cache_hit_rate
from .prompts import opencode_system_prompt, USER_TURNS, SHORT_PROMPT

REPO_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--orcarouter--Qwen3.8-27B-Uncensored-MLX/snapshots"
)
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def free_ram_gb() -> float:
    """Return free RAM in GB. Free = free + inactive pages (macOS reclaimable)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        page = 4096
        vals = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                vals[k.strip()] = v.strip().rstrip(".")
        free = int(vals.get("Pages free", "0")) + int(vals.get("Pages inactive", "0"))
        return free * page / 1e9
    except Exception:
        return 999.0  # fail open; never block a run on a parse error


def swap_used_gb() -> float:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10
        ).stdout
        # "total = 3072.00M  used = 2445.12M  free = 626.88M"
        used = [t for t in out.split() if t][4]  # value after 'used ='
        return _parse_size_gb(used)
    except Exception:
        return 0.0


class MemoryGuard(threading.Thread):
    """Watchdog: kill the server before the machine runs out of memory.

    Trigger: free RAM stays below `floor_gb` for `patience` samples in a row.
    On trigger, terminate the server process and record the abort. This keeps the
    machine safe (no OOM, no hard swap spiral) during every experiment.
    """

    def __init__(self, proc: subprocess.Popen, floor_gb: float = 2.0, patience: int = 3):
        super().__init__(daemon=True)
        self.proc, self.floor_gb, self.patience = proc, floor_gb, patience
        self.triggered = False
        self._stop = threading.Event()
        self.min_free_seen = 999.0

    def run(self) -> None:
        low = 0
        while not self._stop.is_set():
            free = free_ram_gb()
            self.min_free_seen = min(self.min_free_seen, free)
            low = low + 1 if free < self.floor_gb else 0
            if low >= self.patience:
                self.triggered = True
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                return
            self._stop.wait(2.0)

    def stop(self) -> None:
        self._stop.set()


def resolve_paths() -> tuple[str, str | None]:
    """Return (main_4bit_dir, mtp_drafter_dir_or_None) from the HF cache."""
    mains = sorted(glob.glob(f"{REPO_CACHE}/*/4-bit"))
    main = next(
        (d for d in mains if os.path.exists(os.path.join(d, "model.safetensors.index.json"))),
        None,
    )
    if not main:
        sys.exit(
            "Could not find the 4-bit model in the HF cache. Download it first:\n"
            "  hf download orcarouter/Qwen3.8-27B-Uncensored-MLX --include '4-bit/*'"
        )
    drafts = sorted(glob.glob(f"{REPO_CACHE}/*/mtp"))
    draft = next((d for d in drafts if os.path.exists(os.path.join(d, "model.safetensors"))), None)
    return main, draft


def start_server(cfg: dict, main: str, draft: str | None) -> tuple[subprocess.Popen, Path]:
    s = cfg["server"]
    port = s["port"]
    log = RESULTS_DIR / f"_server_{cfg['name']}.log"
    log.parent.mkdir(exist_ok=True)
    argv = [
        sys.executable, "-m", "mlx_vlm", "server",
        "--model", main,
        "--host", s.get("host", "127.0.0.1"), "--port", str(port),
    ]
    if s.get("use_draft") and draft:
        argv += ["--draft-model", draft, "--draft-kind", s.get("draft_kind", "mtp")]
    argv += s.get("flags", [])

    env = dict(os.environ)
    # expanduser handles APC_DISK_PATH="~/..."; it is a no-op for non-path values.
    env.update({k: os.path.expanduser(str(v)) for k, v in s.get("env", {}).items()})

    fh = open(log, "w")
    proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc, log


def wait_ready(port: int, log: Path, timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        txt = log.read_text(errors="ignore") if log.exists() else ""
        if "Application startup complete" in txt:
            return True
        if "address already in use" in txt.lower() or "Traceback (most recent" in txt:
            return False
        time.sleep(1)
    return False


def post_chat(port: int, messages: list[dict], max_tokens: int, model: str) -> tuple[float, dict]:
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return time.time() - t0, data


def phys_footprint_gb(pid: int) -> float | None:
    """True physical footprint incl. Metal/GPU buffers (RSS undercounts MLX badly)."""
    try:
        out = subprocess.run(
            ["/usr/bin/vmmap", "-summary", str(pid)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        for line in out.splitlines():
            if "Physical footprint:" in line and "peak" not in line.lower():
                val = line.split(":")[1].strip()  # e.g. "19.5G"
                return _parse_size_gb(val)
    except Exception:
        return None
    return None


def _parse_size_gb(s: str) -> float:
    s = s.strip().upper()
    if s.endswith("G"):
        return float(s[:-1])
    if s.endswith("M"):
        return float(s[:-1]) / 1024
    if s.endswith("K"):
        return float(s[:-1]) / 1024 / 1024
    return float(s) / 1e9


def run_config(cfg: dict, main: str, draft: str | None) -> dict:
    port = cfg["server"]["port"]
    print(f"\n=== [{cfg['name']}] tier {cfg.get('tier')} — {cfg.get('description','')}")

    # Preflight: check free RAM. The 4-bit model needs ~20 GB. Skip if too low.
    free_before = free_ram_gb()
    required = cfg.get("required_free_gb", 18.0)
    print(f"    free RAM before start: {free_before:.1f} GB (need >= {required})")
    if free_before < required:
        return {"name": cfg["name"], "error": "insufficient_memory",
                "free_ram_gb": round(free_before, 1), "required_gb": required}

    proc, log = start_server(cfg, main, draft)
    guard = MemoryGuard(proc, floor_gb=cfg.get("mem_floor_gb", 2.0))
    try:
        if not wait_ready(port, log, cfg["server"].get("startup_timeout", 180)):
            tail = "\n".join(log.read_text(errors="ignore").splitlines()[-8:])
            return {"name": cfg["name"], "error": "server_failed_to_start", "log_tail": tail}

        guard.start()  # watch memory for the whole workload
        idle_mem = phys_footprint_gb(proc.pid)
        w = cfg["workload"]
        model_id = main  # server keys the preloaded model by its path
        turn_latencies = []

        if w["type"] == "repeat_prefix":
            # A genuine multi-turn conversation: the prompt GROWS each turn (system +
            # turn1 + turn2 ...), so turn N contains turn N-1 as a strict prefix. This is
            # the "growing prompt" case APC reuses — turn 2+ should hit the cached prefix.
            sysmsg = opencode_system_prompt(w.get("system_target_tokens", 15000))
            msgs = [{"role": "system", "content": sysmsg}]
            for i in range(w.get("turns", 2)):
                msgs.append({"role": "user", "content": USER_TURNS[i % len(USER_TURNS)]})
                lat, data = post_chat(port, msgs, w.get("max_tokens", 64), model_id)
                turn_latencies.append(round(lat, 2))
                reply = data["choices"][0]["message"]["content"]
                msgs.append({"role": "assistant", "content": reply})
                print(f"    turn {i+1}: {lat:.2f}s end-to-end")
        elif w["type"] == "short":
            for i in range(w.get("turns", 1)):
                msgs = [{"role": "user", "content": SHORT_PROMPT}]
                lat, _ = post_chat(port, msgs, w.get("max_tokens", 128), model_id)
                turn_latencies.append(round(lat, 2))
                print(f"    run {i+1}: {lat:.2f}s end-to-end")

        peak_mem = phys_footprint_gb(proc.pid)
        metrics = parse_server_log(log.read_text(errors="ignore"))
        per_req = [m.to_dict() | {"cache_hit_rate": cache_hit_rate(m)} for m in metrics]
        return {
            "name": cfg["name"], "tier": cfg.get("tier"),
            "description": cfg.get("description", ""),
            "server_flags": cfg["server"].get("flags", []),
            "server_env": cfg["server"].get("env", {}),
            "idle_footprint_gb": idle_mem, "peak_footprint_gb": peak_mem,
            "free_ram_before_gb": round(free_before, 1),
            "min_free_ram_gb": round(guard.min_free_seen, 1),
            "swap_used_gb": round(swap_used_gb(), 2),
            "oom_guard_triggered": guard.triggered,
            "turn_latencies_s": turn_latencies,
            "requests": per_req,
        }
    except Exception as e:
        return {"name": cfg["name"], "error": f"{type(e).__name__}: {e}",
                "oom_guard_triggered": guard.triggered}
    finally:
        guard.stop()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)  # let Metal release unified memory before the next config


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: python -m bench.harness <config.json> [config2.json ...]")
    main_dir, draft_dir = resolve_paths()
    print(f"model: {main_dir}\ndraft: {draft_dir}")
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []
    for path in argv:
        cfg = json.loads(Path(path).read_text())
        res = run_config(cfg, main_dir, draft_dir)
        all_results.append(res)
        (RESULTS_DIR / f"{cfg['name']}.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {len(all_results)} result file(s) to {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
