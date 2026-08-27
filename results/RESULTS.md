# Results

Measured on Apple Silicon, 36 GB. One server at a time. Memory guarded.

## Memory and safety

| run | idle GB | peak GB | free RAM before | min free RAM | swap GB | OOM guard |
|---|---|---|---|---|---|---|
| tier0_apc | 16.2 | 22.3 | 24.0 | 0.3 | 0.00 | False |
| tier0_baseline | 16.2 | 19.8 | 24.0 | 0.1 | 0.00 | False |

## Per-request performance

| run | turn | prompt tok | cached tok | cache hit | prefill s | TTFT s | decode tok/s | accepted/round |
|---|---|---|---|---|---|---|---|---|
| tier0_apc | 1 | 15264 | 15248 | 1.00 | 0.50 | 1.57 | 18.10 | - |
| tier0_apc | 2 | 15313 | 15264 | 1.00 | 0.85 | 0.93 | 14.80 | - |
| tier0_apc | 3 | 15368 | 15313 | 1.00 | 1.74 | 1.82 | 8.30 | - |
| tier0_baseline | 1 | 15264 | 0 | 0.00 | 128.91 | 129.78 | 10.20 | - |
| tier0_baseline | 2 | 15318 | 0 | 0.00 | 172.87 | 172.91 | 13.10 | - |
| tier0_baseline | 3 | 15394 | 0 | 0.00 | 146.26 | 146.31 | 13.30 | - |
