# H109 CPU preparation

H109 preregisters a paired comparison of context-uniform CONTROL, family-balanced FBAL, and family-balanced plus warmup/cosine-restarts FBAL+WCR. Read [PREREG.md](PREREG.md) before any future execution.

`runner.py` provides deterministic DL-Train-only family grouping, matched four-microbatch gradient accumulation, fixed learning-rate schedules, and per-cell/per-attempt scratch staging for a read-only r882 source. `python runner.py` runs a synthetic CPU dry run; optional `--scratch-root DIR` additionally exercises copy isolation with a temporary synthetic placeholder, never an r882 payload.

The captured synthetic CPU dry-run receipt is stored outside the repository at `/home/ubuntu/jope_plan/ns/receipts/H109_cpu_dry_run.json`.

This is CPU preparation only. No data payload, checkpoint, timing cell, outcome, GPU, or FIFO ticket was used. GPU execution is separately gated until H99, H100, and H105 clear FIFO and the owner explicitly authorizes it. The result is not a model outcome, qualification, or serving promotion.

## CPU acceptance

```sh
nice -n 15 python -m pytest -q research/flywheel_selfplay/h109_family_balanced_accum/test_h109.py
nice -n 15 python research/flywheel_selfplay/h109_family_balanced_accum/runner.py
```
