# H357 — XSTACK-CL confirmation on the fresh TRAIN panel P201c

**Verdict: `XSTACK_CL_NOT_CONFIRMED`** (PREREG §4). On 194 fresh TRAIN families, adding the 4 classical learners to the
locked H335 GP stack (X_CL) helped slightly and in the same direction as on P201b. The primary sign-flip test missed
α: p = 0.074 against the 0.025 threshold. The adoption-grade secondary also failed. No dev91, DL-Val, DL-Test or sealed
path was read or mounted.

- PREREG: [`PREREG.md`](PREREG.md), commit `568e1490db` on `v1/next`, published before any P201c view, GP run, fit or
  score. Three implementation-only amendments, all published before the verdict existed (below).
- Owner: `cl_fwriter`, Claude Opus 5.5, on xbabe2 (orchestration, library) and xbabe0 (GP, scoring).
- Numbers: [`metrics.json`](metrics.json) and [`verdict.json`](verdict.json).

## What was tested
- **X0** = the locked H335 recipe output `AGG.g50.CV3@12` (LOCK `4d58dd37`), rerun from scratch on P201c: Seed F,
  then Evo A for 50 generations, search seed 357101.
- **X_CL** = X0's stack members plus `catboost.v1`, `lightgbm.v1`, `xgboost.v1` and `random_forest.v1`
  (reference-board configs), weighted by the frozen H317 S1-direct stacker on SEARCH only, `C_LIB = 0`, AUDIT scored
  once.
- **Why X_CL.** It was the best arm descriptively in H354 on P201b (+0.0093, p = 1e-4, 132/201 families). It was
  picked after seeing those results, so P201b was exploratory and only P201c counts.

## Results (P201c, 194 TRAIN families, one context each)

| Quantity | Value |
| --- | --- |
| Family-mean AUDIT u, X0 → X_CL | 1.89155 → 1.89558 |
| Gain (unclipped) | +0.0040 (0.40% RMSE cut) |
| Clipped T (±0.5) | +0.0050 |
| One-sided sign-flip p (2^20 MC) | **0.074** (needs < 0.025) |
| LOFO min / family median | +0.0044 / +0.0017 (both pass) |
| Families won / tied / lost | 123 / 19 / 52 |
| Contexts losing more than 0.05 | 4 (worst −0.695) |
| Library weight mass | mean 0.120, median 0.082; > 0 in 98% of contexts |
| Adoption secondary | LCB95 −0.0056 (needs > 0.001), unclipped p 0.213, harm > 0.01 in 6.7% of families → fail |

**Primary:** T > 0, LOFO > 0 and median ≥ 0 all hold, but p = 0.074, so the primary fails. The mirrored test
(p = 0.93) is far from showing X_CL inferior.

**TRAIN AUDIT-Elo on P201c** (anchor X0 = 1500, family bootstrap; descriptive; not the dev91 scale):

| Entrant | Elo | 95% CI |
| --- | --- | --- |
| X_CL | 1582 | [1556, 1608] |
| X0 = H335 AGG | 1500 | anchor |
| catboost alone | 1324 | [1261, 1381] |
| xgboost alone | 1215 | [1145, 1276] |
| random_forest alone | 1205 | [1138, 1262] |
| lightgbm alone | 1132 | [1058, 1196] |

X_CL wins most head-to-head comparisons; the Elo reflects its 123 of 194 family wins. But the
average gain is small next to the spread of per-family differences, so the contrast is not significant.

## What we learned
1. **The P201b gain shrank by more than half on fresh data:** +0.0093 → +0.0040. That fits the selection effect
   PREREG §1 anticipated: X_CL was chosen as the best of three arms on P201b.
2. **The large losses do not come from the classical learners' predictions.** In the three worst contexts
   (`19f0714d` −0.695, `eeff1306` −0.252, `217e0d60` −0.153) the library weight is about zero (8.5e-4, 4e-43, 1.6e-3).
   No member there was mean-substituted, and none is an Amendment-1 extra-AUDIT context.
   - What changes is the GP members' weights. Adding library columns to the candidate set changes the S1-direct
     solution over the GP members.
   - Two of these contexts are near-perfect fits (X0 u = 8.37 and 7.29). There, small weight shifts become large
     log-RMSE losses.
   - Without the single worst context, the mean gain is +0.0077 (descriptive only).
3. **Implication for the next concept.** Any XSTACK-style extension should leave the GP sub-stack's weights fixed
   and stack the library against the fixed X0 prediction (a two-level stack), so the library cannot disturb a
   near-degenerate GP solution. That is a new hypothesis and would need its own PREREG and panel. P201c is now
   used.

## Amendments and attempts (all implementation only; no rule, tolerance, arm or statistic changed)
- **§3.1 Amendment 1 (`a2f22b9ead`).** X_CL can weight GP members that X0 left at zero, and those had no AUDIT record.
  `audit-extra` on xbabe0 evaluated exactly those members with the frozen GP engine: 2 contexts, 5 members, all
  completed. It ran after the first scoring attempt stopped with a `KeyError` before writing or printing anything.
- **§3.2 Amendment 2 (`5636f558b2`).** On xbabe2 the X0 harness check failed on 1 of 194 contexts (`53e44260`,
  |Δw| 6.1e-5, |Δu| 3.0e-7 against a tolerance of 1e-9).
  - An X0-only diagnosis showed the frozen Frank-Wolfe stacker is not bit-reproducible across CPUs on
    near-degenerate stacks. On xbabe0, the GP's own host, it is exact.
  - Scoring therefore moved to xbabe0. The xbabe2 outputs are kept in `attempt1_xbabe2_unread/`, and their X_CL values
    were never read.
- **§3.3 Fix (`0eb34f91f3`).** The first xbabe0 run passed the X0 check exactly (|Δw| = |Δu| = 0.0). `decide` was then
  refused by the H317 guard, which blocks the H335 evidence-tree path of `h335_stats.py`.
  - The fix loads a sha-pinned copy of `h335_stats.py` and reruns `arms` and `decide` in full into `score2`.
  - The rerun's `arms.jsonl` and `x0_check.json` are **byte-identical** to the refused attempt's, which supports
    determinism.

## Harness and integrity checks
- **X0 check:** 194/194 contexts exact (|Δw| = |Δu| = 0.0) on xbabe0, in both runs.
- **GP:** 194/194 records, pulled back to xbabe2 with every file's sha256 compared. SEARCH views encoded with the
  frozen H317 recipe; DOPE `c51a56d0` loaded read-only, sha checked before and after.
- **Library:** 774 of 776 learner × context jobs completed (SEARCH OOF and SEARCH → AUDIT fit each), in the bwrap sandbox with no network. The 2 failures are random_forest and
  xgboost on `635612f6`, handled by the H317 rule. SEARCH gate passes: catboost 193, random_forest 192, lightgbm 190,
  xgboost 188.
- **Scoring:** code (3 files) and library outputs (776 files) sha-checked on xbabe0 before use.
- **Data access:** dev91 rows read: 0. P201c is disjoint from P205 and P201b (asserted).

## Scope
- This is a TRAIN-only result. `dev91-audit-v1` and the reference board are unchanged.
- XSTACK-CL is **not** an entrant on the fixed dev91 board; that would be a separate owner decision, and this result
  does not support one.
- The flywheel writer stayed STOPPED.

## Reproduce
```sh
bash stage_h357.sh TAG && bash stage_gpc_h357.sh TAG        # frozen read-only copies (H317 guard)
export H357_CODE=~/h357-stage/code-TAG H357_GPC=~/h357-stage/gpc-TAG
R=$H357_CODE/run_c_h357.sh
$R panel; $R encode; $R stage; $R launch; $R pull; $R prep; $R lib            # GP on xbabe0, library on xbabe2
$R auditreq; $R stagex; $R auditx; $R pullx                                   # Amendment 1
$R stagescore; $R score0; $R pullscore                                        # Amendments 2 and 3.3
```
The verdict run used `code-run4` (CODE.sha256 digest `270a44fd…`) and `gpc-run4` (`42f66c5c…`), from 11:23:47 to
11:24Z on xbabe0. The GP used `gpc-run1`, and Amendment 1's `audit-extra` used `gpc-run2`.
