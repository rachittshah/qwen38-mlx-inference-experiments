# Results

Measured on Apple Silicon, 36 GB. One server at a time. Memory guarded.

## Memory and safety

| run | idle GB | peak GB | free RAM before | min free RAM | swap GB | OOM guard |
|---|---|---|---|---|---|---|
| tier0_apc | 16.2 | 22.3 | 24.0 | 0.3 | 0.00 | False |
| tier0_baseline | 16.2 | 19.8 | 24.0 | 0.1 | 0.00 | False |
| tier1_dflash | 18.9 | 20.5 | 19.0 | 0.1 | 0.00 | False |
| tier1_mtp_b2 | 16.2 | 17.2 | 16.0 | 0.1 | 0.00 | False |
| tier1_mtp_b4 | 16.2 | 17.8 | 17.0 | 0.1 | 0.00 | False |
| tier1_mtp_b6 | 16.2 | 17.8 | 18.0 | 0.2 | 0.00 | False |
| tier1_mtp_b8 | 16.2 | 18.1 | 19.0 | 0.1 | 0.00 | False |
| tier1_nodraft | 15.3 | 15.6 | 24.0 | 0.2 | 0.00 | False |

## Per-request performance

| run | turn | prompt tok | cached tok | cache hit | prefill s | TTFT s | decode tok/s | accepted/round |
|---|---|---|---|---|---|---|---|---|
| tier0_apc | 1 | 15264 | 15248 | 1.00 | 0.50 | 1.57 | 18.10 | - |
| tier0_apc | 2 | 15313 | 15264 | 1.00 | 0.85 | 0.93 | 14.80 | - |
| tier0_apc | 3 | 15368 | 15313 | 1.00 | 1.74 | 1.82 | 8.30 | - |
| tier0_baseline | 1 | 15264 | 0 | 0.00 | 128.91 | 129.78 | 10.20 | - |
| tier0_baseline | 2 | 15318 | 0 | 0.00 | 172.87 | 172.91 | 13.10 | - |
| tier0_baseline | 3 | 15394 | 0 | 0.00 | 146.26 | 146.31 | 13.30 | - |
| tier1_dflash | 1 | 25 | 0 | 0.00 | 1.19 | 1.19 | 39.00 | - |
| tier1_dflash | 2 | 25 | 0 | 0.00 | 0.19 | 0.20 | 45.30 | - |
| tier1_mtp_b2 | 1 | 25 | 0 | 0.00 | 0.20 | 1.15 | 34.70 | - |
| tier1_mtp_b2 | 2 | 25 | 0 | 0.00 | 0.19 | 0.20 | 36.50 | - |
| tier1_mtp_b4 | 1 | 25 | 0 | 0.00 | 0.20 | 1.19 | 41.80 | - |
| tier1_mtp_b4 | 2 | 25 | 0 | 0.00 | 0.19 | 0.20 | 39.20 | - |
| tier1_mtp_b6 | 1 | 25 | 0 | 0.00 | 0.19 | 1.10 | 39.00 | - |
| tier1_mtp_b6 | 2 | 25 | 0 | 0.00 | 0.20 | 0.21 | 36.90 | - |
| tier1_mtp_b8 | 1 | 25 | 0 | 0.00 | 0.20 | 1.07 | 30.10 | - |
| tier1_mtp_b8 | 2 | 25 | 0 | 0.00 | 0.19 | 0.20 | 30.00 | - |
| tier1_nodraft | 1 | 25 | 0 | 0.00 | 0.20 | 0.79 | 22.20 | - |
| tier1_nodraft | 2 | 25 | 0 | 0.00 | 0.18 | 0.23 | 22.10 | - |

## Tier 3 — CAG (disk prefix reuse across restart)

| pass | latency s | prefill s | cache hit | prompt tok | cached tok | peak GB |
|---|---|---|---|---|---|---|
| cold_first_start | 95.13 | 93.63 | 0.00 | 15253 | 0 | 20.5 |
| warm_after_restart | 1.99 | 0.29 | 1.00 | 15253 | 15237 | 20.5 |

## Tier 5 — OpenCode end-to-end (APC off vs on)

| APC | call 1 (s) | call 2 (s) |
|---|---|---|
| off | 107.10 | 143.50 |
| on | 143.20 | 2.70 |
