# H153 SAM / ASAM

## Hypothesis and setup

The H153 hypothesis was that SAM (rho=0.05) or ASAM (rho=0.5) could improve the adapter’s native utility on the 91-context DL-Val panel relative to CTRL and frozen r882. The preregistration fixed three seeds, 40 updates per cell, five family folds, a TRAIN-only descriptive readout, CPU resource limits, and an exact 91-context baseline-first gate. See [PREREG.md](PREREG.md), [manifest.json](manifest.json), and [fold_manifest.json](fold_manifest.json).

## Outcome

The real constructor-only smoke passed on xbabe2 at nice 19 with zero scored contexts and zero optimizer updates. The single registered run attempt then stopped before the exact r882 baseline evaluation: `runtime.Audit` refused the custody-parent `launch.json` read because the allowlist included the run output directory but not that specific parent manifest. The run monitor recorded INCOMPLETE; the audit recorded zero optimizer updates, zero protected reads, and CUDA uninitialized. See [attempt.json](attempt.json) and the hash-pinned local custody artifacts it names.

## Key metrics and verdict

- Attempts: 1; retries: 0 (the preregistration forbids retry).
- Exact r882 DL-Val baseline: NOT REACHED.
- Scored contexts: 0; optimizer updates: 0.
- Utility, gain, confidence interval, and Elo: not measured.
- Verdict: **INCOMPLETE; efficacy unmeasured.**

`metrics.json` records the terminal status without imputing an outcome. No serving or model-promotion decision follows.

## Next step

Preserve H153 as INCOMPLETE. Correct the launch-manifest audit boundary in a future preregistered wave and review it before any execution; do not restart this H153 attempt.
