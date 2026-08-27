"""Unit tests for the server-log metrics parser.

Run: python -m pytest tests/ -q   (or: python tests/test_metrics.py)
These pin the parsing contract the whole harness depends on, using real log lines
captured from mlx-vlm 0.6.17.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.metrics import parse_server_log, cache_hit_rate  # noqa: E402


REAL_LOG = """
2026-08-27 11:54:26 - INFO - Prefill started: request=6c9587c50 prompt_tokens=15250 images=0
2026-08-27 11:54:27 - INFO - Prefill completed: request=6c9587c50 prompt_tokens=15250 cached_tokens=0 elapsed=74.2s rate=205.5 tok/s
2026-08-27 11:54:27 - INFO - Decode started: request=6c9587c50 time_to_first_token=75.6s
2026-08-27 11:56:23 - INFO - Decode completed: request=6c9587c50 generated_tokens=120 elapsed=9.7s rate=12.4 tok/s finish_reason=stop
Speculative decoding: 3.15 accepted tokens/round (2.15 accepted drafts/round, 71.7% of drafted, avg draft 3.00) over 20 rounds
2026-08-27 11:56:23 - INFO - Request completed: endpoint=/chat/completions prompt_tokens=15250 generated_tokens=120 elapsed=85.3s prefill=205.5 tok/s decode=12.4 tok/s finish_reason=stop in_flight=0
"""

# A second request where APC served most of the prompt from cache (warm turn).
APC_WARM_LOG = """
Prefill completed: request=abc123 prompt_tokens=15300 cached_tokens=15248 elapsed=0.3s rate=170.0 tok/s
Decode started: request=abc123 time_to_first_token=0.9s
Decode completed: request=abc123 generated_tokens=64 elapsed=5.1s rate=12.5 tok/s finish_reason=stop
"""


def test_parses_single_request_all_fields():
    [m] = parse_server_log(REAL_LOG)
    assert m.request == "6c9587c50"
    assert m.prompt_tokens == 15250
    assert m.cached_tokens == 0
    assert m.prefill_s == 74.2
    assert m.prefill_tps == 205.5
    assert m.ttft_s == 75.6
    assert m.generated_tokens == 120
    assert m.decode_tps == 12.4
    assert m.finish_reason == "stop"


def test_attaches_speculative_stats_to_last_request():
    [m] = parse_server_log(REAL_LOG)
    assert m.accepted_per_round == 3.15
    assert m.pct_accepted == 71.7
    assert m.avg_draft == 3.0
    assert m.spec_rounds == 20


def test_cache_hit_rate_cold_vs_warm():
    [cold] = parse_server_log(REAL_LOG)
    [warm] = parse_server_log(APC_WARM_LOG)
    assert cache_hit_rate(cold) == 0.0
    assert cache_hit_rate(warm) > 0.99  # APC reused ~all of the 15K prefix


def test_multiple_requests_preserve_order():
    ms = parse_server_log(REAL_LOG + APC_WARM_LOG)
    assert [m.request for m in ms] == ["6c9587c50", "abc123"]


def test_missing_fields_do_not_crash():
    ms = parse_server_log("garbage line\nanother line with request= but nothing else")
    assert ms == []


if __name__ == "__main__":
    # Lightweight runner so the file works without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
