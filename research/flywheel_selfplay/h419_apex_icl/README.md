# H419 — APEX-ICL, our own in-context tabular regressor (in progress)

**This is a status log, not a closeout.** No prediction on a test cohort (stored P201b matrices, APEX-DEV, CONFIRM)
exists. Plan: [`PREREG.md`](PREREG.md), [`PREREG_ADDENDUM_1.md`](PREREG_ADDENDUM_1.md).

## Runs
| Run | Host | Size | State |
| --- | --- | --- | --- |
| `ICL_S_r2`, `ICL_M_r2` | xbabe3 3090, xbabe1 4090 | 7.56M, 23.66M | synthetic prior only; stopped at about 1.3 h; their checkpoints start r3 |
| `ICL_S_r3` | xbabe3 3090 | 7.56M | 22 h run, started 2026-11-10T10:52Z; column injection + FIT-ICL real-table tasks |
| `ICL_M_r3` | xbabe1 4090 | 23.66M | 46 h run, started 2026-11-10T10:52Z; same data mix |

## Tracked metrics
- **Real-table monitor:** 106 core FIT-SEL contexts, SEARCH rows only; query = fold 0 of `kfold_labels`, context =
  folds 1–4. u = −ln(RMSE / RMSE of the context-mean predictor). Ridge reference on the same rows: mean u 1.061,
  median u 0.630; ridge is near-exact (u > 3) on 7.5% of the contexts, which is why the mean is not the target.
- **Synthetic held-out tasks**, with and without injected columns, against ridge.

| Time (UTC) | Run, hours into r3 | Real win rate vs ridge | Real median u (ridge 0.630) | Real mean u (ridge 1.061) | Synthetic u (ridge) | Synthetic with injection (ridge) |
| --- | --- | --- | --- | --- | --- | --- |
| 10:50 | S, r2 checkpoint (smoke) | 0.462 | 0.585 | 0.784 | 0.247 (0.149) | 0.236 (0.109) |
| 11:13 | S, 0.33 h | 0.481 | 0.526 | 0.800 | 0.266 (0.149) | 0.241 (0.109) |
| 11:33 | S, 0.67 h | 0.491 | 0.597 | 0.818 | 0.277 (0.149) | 0.272 (0.109) |
| 11:13 | M, 0.33 h | 0.547 | 0.643 | 0.931 | 0.292 (0.132) | 0.265 (0.100) |
| 11:33 | M, 0.67 h | 0.575 | 0.645 | 0.917 | 0.297 (0.132) | 0.270 (0.100) |

**Reading at 11:34Z.**
- The larger model is ahead of ridge on real tables by win rate (0.575) and barely by median (0.645 against 0.630).
- **Negatives:** the smaller model is still behind ridge on real tables on every measure (win rate 0.491, median
  0.597, mean 0.818). Both models are behind ridge on the mean. The larger model's mean fell between the two readings
  (0.931 to 0.917) while its win rate rose. Forty minutes of the new data mix is too short to credit the column
  injection or the real-table tasks with anything; the real-table share was still ramping (0.25–0.29 of batches).
- The on-track rule (addendum 1: win rate ≥ 0.55 and median above its start value by hour 12) has not come due.

## Mistakes and corrections so far
- `ICL_*_r1` were stopped seconds after launch: the largest training shapes had not been tested. The stress test
  then found a CUDA attention limit at 65,536 sequences (fixed by chunking) and an out-of-memory shape for the M size
  on the 4090 (fixed by a row-token cap).
- No learning-rate sweep and no optimizer comparison were run, against the design review's plan.
