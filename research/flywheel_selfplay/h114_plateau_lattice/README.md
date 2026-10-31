# H114 — Flywheel plateau lattice audit (CPU, log-only)

**Verdict: DIAGNOSTIC — the live-loop validation plateau is quantized.** No arm,
no model, no GPU, no data payload. Elo N/A (log-only diagnostic, no new contender).

## Hypothesis

The regression flywheel (xbabe1 `flywheel_gp_fused`, run-20261030; 1,219 train /
91 regression-validation contexts) has rejected every candidate since r882
(best 1.1377942). If the validation utility moves only when the argmin selection
on one of 91 contexts flips, candidates collapse onto a few discrete values and
the "plateau" is a lattice trap rather than a smooth ceiling.

## Setup

Read-only parse of `adapter_admissions.jsonl` (1,306 rows, r775–r2080, SHA-256 in
[metrics.json](metrics.json)) at 2026-10-31 12:25Z. No checkpoint, pool, DL-Val,
sealed or holdout payload opened; the live loop (PID 2297929) was not touched.
DOPE frozen (c51a56d0 in the loop's evaluation contract).

## Outcome

| Metric | Value |
|---|---|
| Accepted rows | r782 (tie), r785, r859, r882 (`accept_tie`, same utility as r859) |
| Post-r882 candidates | 1,198 |
| Distinct utilities post-r882 | **20** |
| Mass on two lattice points (1.135249, 1.134434) | 60.9% |
| Highest post-r882 value | 1.1377131 (hit 21 times), **8.1e-5 below best** |
| Candidates ≥ best | 0 |
| Median / p90 | 1.1344341 / 1.1354388 |
| Smallest / median step between distinct values | 2.5e-5 / 1.9e-4 |

The distribution is stationary across windows (r883–1200, 1200–1600, 1600–2000,
2000–2080: max 1.13771/1.13771/1.13771/1.13755, median 1.13443 in every window).
This extends H49 (distinct adapters, identical val utility) and H50 (79.7%
exact-zero utility change at radius 0.01) to the full overnight loop.

## Red-team notes

- r882 itself is a tie-accept of r859; best is the max of ~108 draws in r775–r882,
  so it carries winner's-curse optimism. Waves must keep comparing on fresh
  family-grouped folds (they do), never on the loop's 91 contexts.
- A strict margin of +0.001 over 1.13779 (1.1388) was never observed in 1,306
  loop rounds; on this lattice it needs several simultaneous argmin flips.
- No leakage, sealed/holdout use or DOPE change found in the log.

## Next step

More restore-and-perturb rounds on the same 91-context argmin utility are
unlikely to move the plateau. Candidates for the next GPU slot after the current
FIFO (RNK-12, H99, H100, H105, H109): (a) a continuous admission diagnostic
(mean regret over all candidates, not argmin utility) reported beside the
official utility, (b) larger-step / multi-step candidate generation that can
leave the r882 basin, judged on fresh family folds. Any change to the live
loop's admission rule is an owner decision.
