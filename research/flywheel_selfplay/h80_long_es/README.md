# H80 — Long direct-utility ES from r882 (`ES` vs `ESH`) — closeout

**Status:** complete, **NOT ADMITTED (both arms)**. Closed 2026-10-31 ~03:13 AM MDT (09:13Z).
Preregistration: [PREREG.md](PREREG.md) (sha `1abf747a…`, frozen before outcomes). Metrics: [metrics.json](metrics.json).

## Hypothesis
The flywheel plateau at r882 (best val 1.1378) is a candidate-generation problem (H61: 0/240 accepts).
A longer direct-utility search, antithetic ES with 16 generations × population 32 on Fold A, either plain (`ES`)
or with a family-harm penalty (`ESH`, lambda 1.0), finds an adapter that beats r882 on untouched Fold B.

## Setup
- Offline r882 COPY (`0084479a…`), only the 29,211 adapter params move; DOPE `c51a56d0` verified before and after (unchanged).
- Real-S3 H50 packet, seed-4900 family-grouped folds; ES fitness on Fold A only; Fold B (41 families / 218 contexts) scored once per final candidate.
- Seeds 21/22/23 per arm plus one NULL (`sigma 0.001`, seed 21, diagnostic only). Final mean-theta only, no best-of selection.
- Gate (per arm, independent): pooled mean LCB95 > 0.001 strictly, family median ≥ 0, harm (families < −0.01) ≤ 10%, ties fail.
- GPU under `qualification.lock` FIFO (released after every cell). 7/7 cells + Fold-B, 7,498.9 s of 9,000 s cap, 0 voids, 0 DL-Val reads (no arm admitted), 0 holdout/sealed payloads.

## Outcome (Fold B, 41 families, 3 seeds)
| Arm | pooled family median gain | 95% CI (median) | pooled mean gain | mean LCB95 | harm | Elo vs r882 per seed (21/22/23) | median Elo | admitted |
|---|---|---|---|---|---|---|---|---|
| ES  | 0.0000 | [−0.0045, +0.0120] | +0.0133 | −0.0336 | 29.3% | 1517 / 1543 / 1500 | **1517 (+17)** | no |
| ESH | +0.0034 | [0.0000, +0.0098] | +0.0148 | −0.0357 | 19.5% | 1551 / 1578 / 1560 | **1560 (+60)** | no |
| NULL (diag) | −0.0105 | — | −0.0266 | — | 51.2% | ≈1395 (−105) | — | n/a |

## Red-team
- **Leakage / split use:** fitness Fold A only; Fold B read once per candidate (6 reads); DL-Val never opened; holdout 0. r882 was trained on all 1,219 train contexts including Fold B, so Fold-B deltas are in-sample for r882 (biased *against* the candidates, not for them).
- **Seed noise:** ESH is positive in Elo on all 3 seeds (+51/+78/+60) and well above the NULL (−105), so the direction is not seed noise; but per-seed mean LCB95 is negative on every seed and family medians are 0 on 2 of 3 seeds.
- **Metric direction:** gains are utility (higher better); Elo from family win/loss/tie vs r882.
- **Cherry-picking / ties:** no arm selection between ES and ESH; ties counted as fail; final mean-theta only.
- **DOPE / resources:** DOPE checksum unchanged; RSS ≤ 3.95 GiB per cell, CPU ≤ 79%; disk gate 94.5% caused ~25 min of waits but no void.

## Verdict
No admission. ESH (family-harm-penalized ES) gives the most consistent Elo gain seen so far from r882 (+60 median, all seeds positive),
but it fails the harm-tail gate (19.5% of families lose > 0.01) and the LCB95 gate. This is not a breakthrough and not adoptable into the live loop.

## Next step
ESH's harm tail is the binding constraint. A follow-up should raise the harm penalty (lambda 2–4) or use a hard family-loss constraint,
and evaluate on DL-Val only if Fold-B admission passes. Queue it after H90/H93/H82/H95 in the FIFO; do not run it in the live flywheel.
