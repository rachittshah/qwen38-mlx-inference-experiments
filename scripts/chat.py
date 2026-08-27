#!/usr/bin/env python3
"""Low-memory streaming print harness for the mlx-vlm server.

Why this exists: OpenCode sends a ~15K-token system+tools prompt and asks for up to
8192 output tokens. On a 36 GB Mac that grows the KV cache until Metal runs out of memory.
This client is the opposite. It sends only your short prompt. It caps output small. It is
stateless by default, so the KV cache stays tiny. It uses the Python standard library only,
so the client process itself is a few MB.

It streams tokens and prints them as they arrive.

Examples:
    python scripts/chat.py "Write a haiku about GPUs."
    python scripts/chat.py --max-tokens 128 "Name three prime numbers."
    python scripts/chat.py                      # interactive REPL (Ctrl-D to quit)
    python scripts/chat.py --history 2          # keep a small 2-turn memory window
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

DEFAULT_URL = "http://127.0.0.1:8081/v1/chat/completions"


def default_model() -> str:
    """Read the model id from the OpenCode config, else MLX_MODEL env, else empty."""
    try:
        cfg = json.load(open(os.path.expanduser("~/.config/opencode/opencode.json")))
        return list(cfg["provider"]["mlx"]["models"])[0]
    except Exception:
        return os.environ.get("MLX_MODEL", "")


def stream_chat(url: str, model: str, messages: list, max_tokens: int, temperature: float) -> str:
    """POST a streaming chat completion. Print each delta. Return the full text."""
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": True,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:                       # iterate SSE lines as they arrive
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    text += delta
    except urllib.error.URLError as e:
        sys.exit(f"\n[error] cannot reach server at {url}: {e}\n"
                 f"Start it first, then retry.")
    print()
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-memory streaming print harness.")
    ap.add_argument("prompt", nargs="*", help="prompt text; omit for interactive REPL")
    ap.add_argument("--max-tokens", type=int, default=256, help="output cap (small = low memory)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", default="", help="optional short system prompt (default: none)")
    ap.add_argument("--history", type=int, default=0,
                    help="turns of context to keep (0 = stateless, lowest memory)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=default_model())
    a = ap.parse_args()
    if not a.model:
        sys.exit("no model id. Pass --model <path> or set MLX_MODEL.")

    base = [{"role": "system", "content": a.system}] if a.system else []
    hist: list = []

    def ask(user: str) -> None:
        messages = base + hist + [{"role": "user", "content": user}]
        out = stream_chat(a.url, a.model, messages, a.max_tokens, a.temperature)
        if a.history > 0:
            hist.extend([{"role": "user", "content": user},
                         {"role": "assistant", "content": out}])
            # keep only the last N turns (2 messages per turn) to bound the context
            del hist[: max(0, len(hist) - 2 * a.history)]

    if a.prompt:
        ask(" ".join(a.prompt))
    else:
        print(f"low-mem chat  (history={a.history}, max_tokens={a.max_tokens})  Ctrl-D to quit")
        while True:
            try:
                user = input("\n>>> ")
            except EOFError:
                print()
                break
            if user.strip():
                ask(user)


if __name__ == "__main__":
    main()
