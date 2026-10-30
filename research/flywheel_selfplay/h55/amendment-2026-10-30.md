# H55 pre-outcome execution amendment — 2026-10-30

This amendment is fixed before any H55 outcome cell. The full rationale, timing receipt, resource readings, training-budget invariants, and terminal DL-Val order are recorded in the dated execution-amendment section of `PREREGISTRATION.md`.

Timing-only EMA-25 seed 12 passed its independent audit at source commit `d5bf0c7406e566065aab43440cb015fa6d7b46ed`. Coordinator cell wall was 230.237 seconds; the fixed training projection is 1,105.1 seconds. No fold-B or DL-Val data were read.

The total outcome-wave wall cap includes all six outcome cells, their setup/scoring, and lock waiting. It is a hard stop; no efficacy-based extension or budget change is allowed.

total_gpu_wall_cap_seconds: 1800

All preregistered arm definitions, seeds 12–14, round/update counts, fold-A-only training, admission rule, CUDA lock and memory guards, DOPE pin, CPU/RSS/disk limits, and the at-most-one-terminal-read-per-seed rule remain as stated in `PREREGISTRATION.md`.
