# H417 — APEX corpus and baseline (closeout)

**Verdict: `BASELINE_RECORDED`.** All four harness gates of PREREG §7 pass. This wave is descriptive: no verdict
depends on a value. Everything here is TRAIN-side (APEX-DEV and APEX-FIT-SEL, families never used before).
dev91, DL-Val, DL-Holdout, sealed data and APEX-CONFIRM were not read.

Plan: [`PREREG.md`](PREREG.md), [`PREREG_ADDENDUM_1.md`](PREREG_ADDENDUM_1.md). Numbers: [`metrics.json`](metrics.json).

## 1. What was run
The stored H416 `SXB` recipe, unchanged (three pooled search runs GS / GE / GL, the 20-preset library, design
`BCAR.mix`), on 221 fresh families: 115 `DEV` and 106 `FIT-SEL` (core tier: rows ≥ 800, 3 ≤ width ≤ 400).
Utility u = −ln(RMSE / RMSE of the SEARCH-mean predictor) on the 30% AUDIT rows; a gain of +0.05 u is about 5% RMSE.

## 2. Baseline (221 families)
| Entrant | Family-mean u | Against `X0.GS`: clipped gain | Sign-flip p | Families W/T/L | Elo (`X0.GS` = 1500) |
| --- | --- | --- | --- | --- | --- |
| `X0.GS` (one stock run) | 1.998 | — | — | — | 1500 |
| `GE.stock` | 2.035 | +0.014 | 0.043 | 111/26/84 | 1532 |
| `GL.stock` | 1.977 | +0.005 | 0.29 | 126/3/92 | 1553 |
| `BAG2` | 2.068 | +0.007 | 0.19 | 111/2/108 | 1510 |
| `BAG3` | 2.097 | +0.028 | 0.0017 | 128/0/93 | 1568 |
| `KFB` (H415) | 2.092 | +0.058 | < 1e-6 | 166/1/54 | 1671 |
| `HFR` | 2.080 | +0.051 | < 1e-6 | 168/3/50 | 1703 |
| **`SXB` (H416)** | 2.036 | **+0.051** | < 1e-6 | 166/2/53 | **1717** |

- **`SXB` against `KFB` replicates on new families:** ΔElo **+45.8** [+23.0, +68.3] (family bootstrap, seed 417150,
  10,000 draws), families 139/3/79. On the P201b TRAIN board it was +45.1.
- **The mean still does not favour `SXB`:** mean gain −0.056 u against `KFB`, clipped mean −0.001 (p 0.59). `SXB` wins
  more families; `KFB` wins a few families by a lot. Same picture as in H416.
- **Per cohort:**

  | Cohort | n | `SXB` − `X0.GS` clipped (p) | `SXB` Elo | `KFB` Elo | `SXB` − `KFB` ΔElo [95% CI] |
  | --- | --- | --- | --- | --- | --- |
  | `DEV` | 115 | +0.045 (3.1e-4) | 1733 | 1677 | +55.9 [+25.2, +86.6] |
  | `FIT-SEL` | 106 | +0.058 (1.8e-5) | 1700 | 1665 | +35.1 [+0.9, +68.5] |

- **The whole stack is worth less here than on the old families.** `SXB` over one stock run is +0.042 u on `DEV`
  (about 4% RMSE); on P201b the same recipe was 11% over its locked baseline. The baselines differ, so this is not
  a like-for-like comparison, but the fresh families leave less on the table for pooling and stacking.
- 0 invalid outputs and 0 unit errors in all 221 contexts.

## 3. Selection headroom (`FIT-SEL`, 106 families; in-sample upper bounds, never deployable)
| Quantity | Family-mean u | Minus `SXB`: clipped | p | W/T/L | Median |
| --- | --- | --- | --- | --- | --- |
| `SXB` | 2.173 | — | — | — | — |
| Best single program, chosen on AUDIT | 2.383 | +0.010 | 0.20 | 63/1/42 | +0.004 |
| Caruana weights fitted on AUDIT | 2.451 | **+0.072** | < 1e-6 | 98/1/7 | +0.027 |

- `SXB` already matches the best single program that hindsight could pick.
- A perfect selector over the existing programs could add at most about +0.07 u clipped (median +0.03). That bounds
  the learned-stacker wave; a realistic share of it is a few percent RMSE at best. Gains beyond it need new learners.
- 64 usable programs per context on average.

## 4. Screener gate G0 (24 `FIT-SEL` contexts; rule fixed in PREREG §6.2)
| Arm | Family-mean AUDIT u | Family-mean best SEARCH u |
| --- | --- | --- |
| `SCR` — stock neural screener `H0.1-A` | 2.344 | 1.975 |
| `RND` — random scores | 2.405 | 2.056 |
| `ORC` — true SEARCH utility of all 48 proposals | 2.450 | 2.134 |

| Contrast | Clipped mean | Sign-flip p | W/T/L |
| --- | --- | --- | --- |
| `ORC` − `SCR` | +0.054 | 0.050 | 17/0/7 |
| `RND` − `SCR` | +0.034 | 0.077 | 15/1/8 |
| `ORC` − `RND` | +0.033 | 0.17 | 13/0/11 |

- **Rule outcome: the search-surrogate wave is built this cycle** (`ORC` − `SCR` ≥ +0.02 and `ORC` wins ≥ 60%:
  +0.054 and 70.8%).
- **Negative for the current screener:** on these fresh families the neural screener is not better than random
  scores, and points the other way (15 of 24 contexts lost to random, p 0.077). 24 contexts cannot settle that, but
  there is no sign that it helps.
- `ORC` costs 8 times the evaluations per generation; it is a bound, not a recipe.

## 5. Harness gates (PREREG §7) — [`gates_h417.sh`](gates_h417.sh)
| Gate | Result |
| --- | --- |
| 1. One stage record per context in each search arm; every deploy unit completed | pass: 221 / 221 / 221 arm records equal to the configured contexts; 221 deploy records, one line each, 0 errors |
| 2. 0 same-family bank hits; ledgers consistent; 0 read-guard refusals | pass: 0 hits and 0 in-process refusals in 9 deploy and 12 arm status files |
| 3. `KCAR100.bag` recomputed from the matrices equals the unit's `KFB` value | pass: max difference 0 |
| 4. DOPE `c51a56d0` unchanged around the view encoding | pass |

Also checked: the 10 units that were in flight at a restart were recomputed from memo hits and equal their stored
records exactly (max difference 0).

## 6. Deviations and negatives
- **Gate order.** PREREG §7 says no value is read if a gate fails. The published stage had no gate step, and I ran
  the readout before writing `gates_h417.sh`. All gates pass, so no value was read that should not have been, but
  the order was wrong and the gate script is post hoc.
- **Display error in the readout text.** The "% RMSE" column prints a fraction (0.04 means 4%). The JSON values are
  right; `metrics.json` names the field `pct_rmse_fraction`.
- **G0 ran on xbabe3, not xbabe0,** and was paused by the CI rule there (11 pauses during the run). Evaluations in
  flight during a pause can time out. Timeouts are not stored in the memo, so the per-arm count I promised cannot
  be given. All three arms ran inside the same units and shared the pauses.
- **Deploy sharding (addendum 1) and two reboots of xbabe0** (11:56Z, 12:05Z; not by this wave). The deploy resumed
  with the same arguments; about 0.7 hours were lost. Engine statistics of the processes that died were lost with
  them; worker-seconds come from the stage records and are complete.
- **Time estimates were wrong twice.** Search and deploy are bound by one Python thread per process. The first
  single-process deploy wasted about an hour.
- **Half the planned sample.** Only the core tier is deployed (221 families, not about 460), so effects below
  about 2% RMSE are not detectable on `DEV`.
- The cross-process refusal log holds 1 line, as it did in the stored H416 run; it predates this wave.

## 7. Compute
457 worker-hours: search 210 (GS 64, GE 64, GL 82), deploy 247 (fold models 99, HF 83, library 43, refits 22).
G0: 60,466 search evaluations in 1.6 hours of wall-clock.

## 8. What later waves take from this
- `DEV`: per-context `SXB` and `X0.GS` values are the paired baselines (H420 uses them now, without a recompute).
- `FIT-SEL`: the stored matrices train the learned stacker; §3 is its ceiling.
- G0: the search-surrogate wave is in scope. The first question for it is whether any learned ranking beats random.

## 9. Run log (UTC, 2026-11-10)
| Time | Item |
| --- | --- |
| 07:01–09:53 | Search runs GS / GE / GL on 221 contexts (xbabe0) |
| 09:10–10:49 | Gate G0 on xbabe3 |
| 09:54 | Deploy started as one process; 25 contexts in 64 minutes |
| 10:58 | Deploy restarted as 8 context-disjoint processes (addendum 1) |
| 11:56, 12:05 | xbabe0 rebooted; 72 contexts finished at that point |
| 12:19 | Deploy resumed |
| 14:56 | Deploy complete, 221 of 221 |
| 15:49–15:57 | Pulled (683 files, sha-compared), readout, headroom, gates, G0 pulled from xbabe3 and read out |
