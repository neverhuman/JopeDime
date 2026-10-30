# H55 pre-outcome execution amendment — 2026-10-30

This amendment is fixed before any H55 outcome cell. The full rationale, timing receipt, resource readings, training-budget invariants, and terminal DL-Val order are recorded in the dated execution-amendment section of `PREREGISTRATION.md`.

Timing-only EMA-25 seed 12 passed its independent audit at source commit `d5bf0c7406e566065aab43440cb015fa6d7b46ed`. Coordinator cell wall was 230.237 seconds; the fixed training projection is 1,105.1 seconds. No fold-B or DL-Val data were read.

The total locked-cell GPU wall cap includes all six outcome cells and their setup/scoring, summed across cells. Lock waiting is tracked separately and is subject to the three-hour cumulative contention limit; it does not count as GPU cell wall. The coordinator enforces the remaining GPU budget after each cell and stops before launching a cell if the cap is exhausted. This is a hard stop; no efficacy-based extension or budget change is allowed.

total_gpu_wall_cap_seconds: 1800

All preregistered arm definitions, seeds 12–14, round/update counts, fold-A-only training, admission rule, CUDA lock and memory guards, DOPE pin, CPU/RSS/disk limits, and the at-most-one-terminal-read-per-seed rule remain as stated in `PREREGISTRATION.md`.

Clarification recorded before outcomes: the first outcome coordinator attempt stopped before cell start after 903.6 seconds of lock wait because its deadline arithmetic confused wait time with GPU execution time. Its incomplete infra-void receipt is preserved under `~/jope-data/h55/h55-wave/outcomes/wave.json`; it contains no completed or partial H55 cell. The corrected coordinator carries the 903.6-second prior wait into the three-hour lock-wait budget and applies the unchanged 1,800-second cap to summed locked-cell runtime only.
