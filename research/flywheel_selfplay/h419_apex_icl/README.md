# H419 — APEX-ICL, our own in-context tabular regressor (in progress)

**This is a status log, not a closeout.** No prediction on a test cohort (stored P201b matrices, APEX-DEV, CONFIRM)
exists. Plan: [`PREREG.md`](PREREG.md), [`PREREG_ADDENDUM_1.md`](PREREG_ADDENDUM_1.md).

## Runs
| Run | Host | Size | State |
| --- | --- | --- | --- |
| `ICL_S_r2`, `ICL_M_r2` | xbabe3 3090, xbabe1 4090 | 7.56M, 23.66M | synthetic prior only; stopped at about 1.3 h; their checkpoints start r3 |
| `ICL_S_r3` | xbabe3 3090 | 7.56M | stopped at 0.77 h (11:38Z) for the 104M run (addendum 2); checkpoint kept |
| `ICL_M_r3` | xbabe1 4090 | 23.66M | 46 h run, started 2026-11-10T10:52Z; same data mix |
| `ICL_L_r4` | xbabe3 3090 | 104.36M | 72 h run, started 2026-11-10T11:40Z from scratch (addendum 2); same data mix |
| `ICL_L_g1_w1`, `ICL_L_g1_w2` | xbabe0 3090, xbabe2 3090 | 104.36M | started 12:29Z as averaging workers cloned from `ICL_L_r4` at step 5,531 (addendum 3); the xbabe3 run joins after the cut-over checks |
| `ICL_M_x4_inj`, `ICL_M_x4_noinj` | xbabe4 Apple GPU | 23.66M | column-injection ablation, two arms, stopped at 24 h each (addendum 4) |

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

| 12:13 | M, 1.33 h | 0.585 | 0.633 | 0.940 | 0.300 (0.132) | — |
| 12:01 | L (104M), 0.33 h | 0.019 | 0.013 | 0.033 | 0.007 | — |

**Reboots of 12:02Z (xbabe2) and 11:56Z / 12:05Z (xbabe0), not by this wave.** Neither training run was on a
rebooted host. `ICL_M_r3` (step 31,010 at 1.42 h, checkpoint 12:13:10Z) and `ICL_L_r4` (step 5,263 at 0.63 h,
checkpoint 12:00:27Z) kept running; **no training time was lost**. The session and its waiters were lost and
restarted. After the reboots the RTX 3090s of xbabe0 and xbabe2 are usable; addendum 3 puts them on the 104M run.

**Reading at 12:13Z.** The 24M run: win rate 0.585 (up from 0.575), median 0.633 (down from 0.645, now level with
ridge at 0.630), mean 0.940 (ridge 1.061). The 104M run at 0.33 h has not started to predict (win rate 0.02), as a
larger model is expected to at that point; this is neither good nor bad news yet.

**Reading at 11:34Z.**
- The larger model is ahead of ridge on real tables by win rate (0.575) and barely by median (0.645 against 0.630).
- **Negatives:** the smaller model is still behind ridge on real tables on every measure (win rate 0.491, median
  0.597, mean 0.818). Both models are behind ridge on the mean. The larger model's mean fell between the two readings
  (0.931 to 0.917) while its win rate rose. Forty minutes of the new data mix is too short to credit the column
  injection or the real-table tasks with anything; the real-table share was still ramping (0.25–0.29 of batches).
- The on-track rule (addendum 1: win rate ≥ 0.55 and median above its start value by hour 12) has not come due.

## Why xbabe4 runs an ablation and not a fourth 104M worker
- The Apple GPU trains the 104M model at about 0.36 times a 3090 per step (1.4–2.0 s against about 0.5 s). In an
  equal-weight average it would pull the group back each round, so it stays out of the group.
- The owner's direction for this wave is synthetic column injection, and nothing yet measured its contribution.
  xbabe4 therefore runs the published 24M recipe with and without injection, side by side on the same hardware
  (addendum 4). The arm with injection is also the only replicate of the 24M run, so it gives a first noise estimate.
- Limit: at about 19 tasks per second per arm, 24 hours cover only the first 1.6 million tasks. A clear early effect
  can show; a small or late one cannot.

## Mistakes and corrections so far
- `ICL_*_r1` were stopped seconds after launch: the largest training shapes had not been tested. The stress test
  then found a CUDA attention limit at 65,536 sequences (fixed by chunking) and an out-of-memory shape for the M size
  on the 4090 (fixed by a row-token cap).
- No learning-rate sweep and no optimizer comparison were run, against the design review's plan.
