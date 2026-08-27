"""Parse mlx-vlm server log lines into structured per-request metrics.

The mlx-vlm OpenAI-compatible server emits INFO lines like::

    Prefill completed: request=6c95 prompt_tokens=20 cached_tokens=0 elapsed=0.195s rate=102.3 tok/s
    Decode started: request=6c95 time_to_first_token=1.480s
    Decode completed: request=6c95 generated_tokens=5 elapsed=0.325s rate=12.3 tok/s finish_reason=stop
    Request completed: endpoint=/chat/completions model=... prompt_tokens=20 generated_tokens=5 \
        elapsed=1.857s prefill=102.3 tok/s decode=15.4 tok/s finish_reason=stop in_flight=0
    Speculative decoding: 3.15 accepted tokens/round (2.15 accepted drafts/round, 71.7% of drafted, \
        avg draft 3.00) over 20 rounds

This module turns that text into a list of :class:`RequestMetrics`. It is the only
pure-logic piece of the harness, so it carries the unit tests (see tests/test_metrics.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# One regex per metric field we care about. Kept permissive (\S+ / [\d.]+) so a
# format tweak in a future mlx-vlm release degrades to "field missing", not a crash.
_RE = {
    "prefill": re.compile(
        r"Prefill completed: request=(?P<req>\S+).*?prompt_tokens=(?P<prompt_tokens>\d+)"
        r".*?cached_tokens=(?P<cached_tokens>\d+).*?elapsed=(?P<prefill_s>[\d.]+)s"
        r".*?rate=(?P<prefill_tps>[\d.]+)"
    ),
    "ttft": re.compile(
        r"Decode started: request=(?P<req>\S+).*?time_to_first_token=(?P<ttft_s>[\d.]+)s"
    ),
    "decode": re.compile(
        r"Decode completed: request=(?P<req>\S+).*?generated_tokens=(?P<gen_tokens>\d+)"
        r".*?elapsed=(?P<decode_s>[\d.]+)s.*?rate=(?P<decode_tps>[\d.]+)"
        r".*?finish_reason=(?P<finish>\S+)"
    ),
    "spec": re.compile(
        r"Speculative decoding: (?P<accepted_per_round>[\d.]+) accepted tokens/round"
        r".*?(?P<pct_accepted>[\d.]+)% of drafted.*?avg draft (?P<avg_draft>[\d.]+)"
        r".*?over (?P<rounds>\d+) rounds"
    ),
}


@dataclass
class RequestMetrics:
    """Metrics for one generation request, assembled from several log lines."""

    request: str
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    prefill_s: Optional[float] = None
    prefill_tps: Optional[float] = None
    ttft_s: Optional[float] = None
    generated_tokens: Optional[int] = None
    decode_s: Optional[float] = None
    decode_tps: Optional[float] = None
    finish_reason: Optional[str] = None
    # Speculative-decoding stats are per-request when a drafter is attached.
    accepted_per_round: Optional[float] = None
    pct_accepted: Optional[float] = None
    avg_draft: Optional[float] = None
    spec_rounds: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(d: dict) -> dict:
    """Cast the regex string groups to int/float where the field name implies it."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif k in {"prompt_tokens", "cached_tokens", "gen_tokens", "rounds"}:
            out[k] = int(v)
        elif any(k.endswith(s) for s in ("_s", "_tps", "_round", "_draft", "_accepted")):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def parse_server_log(text: str) -> list[RequestMetrics]:
    """Parse a full server-log string into per-request metrics, in first-seen order.

    Speculative lines carry no request id, so each is attached to the most recently
    seen request (the server emits them immediately after that request's decode).
    """
    by_req: dict[str, RequestMetrics] = {}
    order: list[str] = []
    last_req: Optional[str] = None

    def get(req: str) -> RequestMetrics:
        nonlocal last_req
        if req not in by_req:
            by_req[req] = RequestMetrics(request=req)
            order.append(req)
        last_req = req
        return by_req[req]

    for line in text.splitlines():
        if m := _RE["prefill"].search(line):
            g = _coerce(m.groupdict())
            rm = get(g["req"])
            rm.prompt_tokens, rm.cached_tokens = g["prompt_tokens"], g["cached_tokens"]
            rm.prefill_s, rm.prefill_tps = g["prefill_s"], g["prefill_tps"]
        elif m := _RE["ttft"].search(line):
            g = _coerce(m.groupdict())
            get(g["req"]).ttft_s = g["ttft_s"]
        elif m := _RE["decode"].search(line):
            g = _coerce(m.groupdict())
            rm = get(g["req"])
            rm.generated_tokens, rm.decode_s = g["gen_tokens"], g["decode_s"]
            rm.decode_tps, rm.finish_reason = g["decode_tps"], g["finish"]
        elif m := _RE["spec"].search(line):
            g = _coerce(m.groupdict())
            if last_req is not None:
                rm = by_req[last_req]
                rm.accepted_per_round = g["accepted_per_round"]
                rm.pct_accepted = g["pct_accepted"]
                rm.avg_draft = g["avg_draft"]
                rm.spec_rounds = g["rounds"]

    return [by_req[r] for r in order]


def cache_hit_rate(m: RequestMetrics) -> Optional[float]:
    """Fraction of prompt tokens served from the prefix cache (APC), 0..1."""
    if m.prompt_tokens:
        return (m.cached_tokens or 0) / m.prompt_tokens
    return None
