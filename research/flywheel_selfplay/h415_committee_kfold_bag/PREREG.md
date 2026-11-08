# H415 PREREG — a committee of trained screeners, and a 5-fold cross-fit bag over the whole population (TRAIN P201e)

**Status: part A.** Published on `origin/v1/next` before any H415 screener fit, P201e SEARCH-view encode, GP run or
deploy output. The P201e panel rule (§3) was computed first; it reads only the TRAIN row index.

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H415_claim.txt` (2026-11-08T16:45Z).
- **Authority:** owner brief, 2026-11-08.
  - Start a new flywheel research wave.
  - Read what worked in the latest successful iterations (H335 AGG and later TRAIN wins).
  - Then propose and run a stronger, TRAIN-only training concept aimed at maximizing Elo.
- **Hard constraints (owner):**
  - Train only on training data.
  - `dev91-audit-v1` is score-once only. H415 reads **no** dev91, DL-Val, DL-Test or sealed byte, and rescores no
    holdout model.
  - Do not restart board scoring.
  - Publish on `v1/next`.
  - Do not kill other lanes or touch `codex/*`, `claude/*` or `upgrade/*` branches.

## 1. What the latest successful iterations established (stored results only)
| Wave | Result | What it means for H415 |
| --- | --- | --- |
| H335 | `AGG.g50.CV3@12` (seed F + 50 generations of K = 6 live / 48 screened + a 12-member stack) beat the incumbent GP 17/19 families on dev91 (17.2% RMSE) | The search recipe is strong; H415 keeps it unchanged as the search inside every arm. |
| H364 | A frozen second level over 4 classical learners: +0.0036 on P201d (p 0.011) | Library members help a little when GP weights are frozen. |
| H381 | A grown, locked S3 screener (E2L) in the loop: +2.5% RMSE on 120 TRAIN families, p 0.23 | One differently trained screener alone is not enough. |
| H410 | Pooling the X0 and E2 runs (two screeners) + a strong-learner library: **+8.1% RMSE on 120 TRAIN families (p 0.0006)**. Widening one run's own stack: −0.9%. | Population diversity from *different screeners* pays; widening one run does not. |
| H410 dev91-5 (tuned on, direction only) | Pooled re-selection collapsed on one context where the 2-fold, ~560-row SEARCH fitness ranked the best AUDIT program 105th of 124. The scale-L screener C1 was the strongest single arm (+0.121). | **Selection fidelity** is the next bottleneck. The L screener deserves its first TRAIN test. |

## 2. Thesis and concept
**Thesis.** The deployed model should be a bag over a population searched by **several differently trained
screeners**, selected and fit by **5-fold cross-fitting on all SEARCH rows** instead of the 2-fold half-data fitness.

**Concept**, called KFB:
1. **Train the committee.** Add a scale-L screener with complete 5-fold TRAIN coverage. H410 trained only fold 0.
2. **Search three times per context.** Run the locked H335 AGG search once with each screener: S (`H0.1-A`), grown S3
   (`H381-E2-S3fz`) and L (`H415-L`).
3. **Pool** every evaluated program of the three runs, plus the H410 strong-learner library.
4. **Re-score at higher fidelity.** The top of the pool is re-scored by y-stratified 5-fold cross-fitting on the SEARCH
   rows: each fit sees 80% of them, against 50% in the search's 2-fold fitness.
5. **Select and deploy.** A Caruana ensemble is selected on those 80%-fidelity out-of-fold predictions and deployed as
   the **5-fold bag** of each member's fold models (AutoGluon-style; no full refit).

Everything is fit and selected on SEARCH rows. AUDIT rows score each declared output exactly once.

## 3. Panel
- **P201e:** the **fifth** seeded context (`h335_core._ctx_order`) of every TRAIN family with ≥ 5 contexts.
  - 185 contexts in 185 families; cids sha256 `4ba54ff659340b786504be21aef57d112beb6e06e98e66516433d4d613d32ea0`.
  - File: `$H317_DATA/h415/P201e.json` (create-once, 0444).
  - Disjoint from P205, P201b, P201c and P201d, which is asserted.
- **Disclosure:** the families are the same TRAIN families as P201b–d, and datasets can recur as other contexts.
  P201e is fresh at the context level, not the dataset level.
- **What existed before this PREREG:** the panel file alone. No view, fit, GP run or score exists for P201e.
- **Smoke-only contexts** (never scored as H415 evidence):
  - `ID2`, the first 2 S60 contexts, for the harness identity check in §8;
  - `DEV2E`, the 2 contexts H381 used as its never-scored smoke (P201d positions 60–61).

## 4. Training: the H415-L screener (TRAIN only)
- **Recipe:** H317 `H0.2-L` exactly, the recipe H410 C1 used: scale L, opt1, A4 views, A525 subset, 3,000 steps, budget
  slate 128 / top_k 48 / batch 16, seeds 317101 and 317102. It runs through the pinned
  `rapid_hypothesis_lab.py oof --keep-models --confirm-true-targets` (sha `31161bec…`).
- **New fits:** folds 1–4 on the xbabe1 RTX 4090, taken under `flock -n qualification.lock`. Fold 0 is the stored H410
  C1 pair (same recipe and code, models `c388c637…` / `2ca8c30e…`), reused rather than refit.
- **Family folds:** `common_h317.family_folds`. A P201e context is screened only by the fold pair that held its family
  out (the per-fold out-of-fold screener of `gp_lab.build_runtime`, mean of the 2 seeds).
- **Recorded per fold, nothing gated on it:** top-1 utility on the held-out contexts, against the stored `H0.1-A`.
- **DOPE** `c51a56d0` is checked before and after training.

## 5. GP arms (xbabe0 docker sandbox; `gp_h415.py` = H381's `unit_lf` with lf_rows 0)
| Arm | Screener (per-fold out-of-fold pair) | Role |
| --- | --- | --- |
| `GS` | `H0.1-A` (stored) | **X0e** = the locked H335 recipe on P201e, the baseline |
| `GE` | `H381-E2-S3fz` (stored) | committee member |
| `GL` | `H415-L` (§4) | committee member; L screener in the loop |

- **Shared settings:** seed 415101; seed F + Evo A K = 6 live / P = 48 screened, 50 generations; stock outputs
  `AGG.g50.{single, CV3@12}`; superset worker `2a848c8f`; 60 s per-eval CPU limit; 1 thread per worker.
- **Sharding:** ≤ 108 cores in total on xbabe0, nice 19.
  - `GS` and `GE` run concurrently, about 48 cores each (6 shards × 8 workers).
  - `GL` gets up to 96 cores as they free up.
- **What differs between arms:** only the screener directory.
- **Order:** `GS` and `GE` start after the P201e SEARCH views are staged. `GL` starts when `H415-L` is complete.

## 6. Deploy unit (per P201e context; code `deploy_h415.py`, published before any deploy run, §8)
**Inputs.** Replay `GS`, `GE` and `GL` from their concatenated memos. Every evaluation is a memo hit, so the replay
costs 0 worker runs and reproduces each run's stock output, which is checked. Pools: `P2 = GS ∪ GE` and
`P3 = GS ∪ GE ∪ GL`.

**H410 builders, code unchanged** (`deploy_h410.py` imported at its published bytes):
- the stock outputs of each run, and `P2.RUNBAG` / `P3.RUNBAG`;
- for `P2` and `P3`: `W24`, `W48`, `CAR`, `NNLS` and the bagged twins `BAG_*`;
- the library: `LIB2` = `library_candidates(P2)` and `LIB3` = `library_candidates(P3)` (presets + target-transform
  variants + polish neighbours), SEARCH-evaluated at a 300 s limit, then the same builders over `P2+LIB2` and
  `P3+LIB3`.

The `BAG_<b>` → `<b>` fallback (no bag-eligible member) is applied as in H410.

**KFB, new, fixed now:**
1. **Candidates H.** The top 48 passing programs of `P3+LIB3` by fitness (`stacking_h317.passing_sorted`), plus the
   2 best passing programs of every head operator not already in (head = the pipe head op; `portfolio` for
   portfolios). H is capped at 64, dropping the worst-fitness additions first.
2. **Folds.** Five y-stratified folds of the SEARCH rows: sort by y (stable); in each consecutive block of 5 rows,
   assign a seeded permutation of 0–4 (`stream_seed(415201, "kf", cid)`); a short last block takes a seeded subset.
3. **High-fidelity OOF.** For j = 0..4, an alternative SEARCH envelope `kf<j>` has fold0 = the SEARCH rows not in F_j
   and fold1 = F_j. It contains SEARCH rows only. The worker's fold-1 predictions (fit on 80% of SEARCH) are the
   out-of-fold predictions on F_j. Evaluated inside a selection scope, 300 s limit. A program is usable only if all
   5 folds completed with finite predictions.
4. **Selection.** Caruana forward selection with replacement on the out-of-fold matrix of the usable programs, 100
   steps, from an empty ensemble.
   - Objective: ln(RMSE / b_kf), where b_kf is the RMSE of the per-fold training-mean predictor.
   - Weights = pick counts / 100.
5. **KFB deploy.** Each member's AUDIT prediction is the mean of its 5 fold models' AUDIT predictions. A fold model is
   trained on the same 80% of SEARCH rows, through an envelope `akf<j>` with fold0 = SEARCH \ F_j and fold1 = AUDIT.
   - The output is the weighted mean.
   - It is requested once, outside every selection scope, under its own ledger key.
   - Only fold-1 predictions are read. The worker's reverse fit inside `akf<j>` (fold 1 → fold 0), like the reverse
     fit inside the stock AUDIT envelope, is never read.
   - If a member's fold model fails, its mean is taken over the completed folds. If all 5 fail, its full-refit AUDIT
     prediction is used. Both cases are counted and reported.
6. **Outputs.**
   - `P3+LIB3.KFB`: **the H415 candidate**.
   - `P3+LIB3.HFR`: the same members and weights, deployed as the stock full-SEARCH refit (exploratory).
   - `P3+LIB3.KFB1`: the single usable program with the lowest out-of-fold RMSE, KFB-deployed (exploratory).

## 7. Readout, tests and verdict (TRAIN P201e, 185 families = 185 contexts)
- **Utility:** AUDIT u = −ln(RMSE_audit / RMSE of the SEARCH-mean predictor) (`audit_utility_from_rmse`). It is
  comparable across every output of a context.
- **Primary.** `P3+LIB3.KFB` vs **X0e** (`GS` stock `AGG.g50.CV3@12`), by `h335_stats.primary_test`:
  - d clipped to ±0.5 per context;
  - sign-flip p < 0.025 (Monte Carlo as in H335 Phase T);
  - LOFO min > 0, median ≥ 0, T > 0.
- **Secondaries.** Holm at 0.05 over the three sign-flip p-values, each also needing LOFO min > 0 and median ≥ 0:
  - **S1:** `P3+LIB3.KFB` vs `P3+LIB3.BAG_CAR`. Does cross-fitted selection and bagging beat H410's best builder on the
    same pool?
  - **S2:** `P2+LIB2.BAG_CAR` vs X0e. A fresh replication of H410's TRAIN result.
  - **S3:** `GL` stock vs X0e. The L screener in the loop, its first TRAIN test.
- **Descriptive, every output:**
  - unclipped family-mean gain and % RMSE;
  - W/T/L at ±1e-4, LOFO and family-bootstrap LCB95 (10,000 draws, seed 415150);
  - the raw H317 adoption gate: LCB95 > 0.001, median ≥ 0, ≤ 10% of families harmed by > 0.01, sign-flip, LOFO;
  - **TRAIN AUDIT-Elo** (`h335_stats.elo_panel`, anchor X0e = 1500, family bootstrap CI) over the declared outputs:
    X0e, GE stock, GL stock, `P2.RUNBAG`, `P3.RUNBAG`, `P2+LIB2.BAG_CAR`, `P3+LIB3.BAG_CAR`, `P3+LIB3.W48`,
    `P3+LIB3.KFB`, `P3+LIB3.HFR`, `P3+LIB3.KFB1`, `LIB.single`;
  - search, library, high-fidelity and AUDIT worker CPU seconds per output (host-speed check as in H381).
- **Verdicts:**
  - `KFB_CONFIRMED` if the primary passes. `KFB_CONFIRMED_BEATS_H410` if S1 also passes.
  - `KFB_NOT_CONFIRMED` otherwise, with the component that carried any gain named from S1–S3.
  - Missing or failed outputs are not passes. A context with an invalid output scores 0 gain for that output and is
    counted.
- **Next step, not run in H415, owner decision:** a confirmed KFB is the candidate for **one** locked dev91 once-score
  under its own PREREG, LOCK and receipt (H369 pattern). H415 itself never reads dev91.

## 8. Integrity, harness checks and mechanics
- **Code.** New files only, in `research/flywheel_selfplay/h415_committee_kfold_bag/`.
  - The frozen H317 pins, the H335 files, `gp_h381.py` and `deploy_h410.py` are staged at their published bytes and
    sha-checked (`stage_h415.sh`).
  - Every scored run uses code staged from `origin/v1/next` (`stage_h415.sh pub`).
  - `deploy_h415.py` and `train_h415.py` are published in a code amendment before any P201e deploy unit runs. The
    constants are those of §6–7; the amendment fixes code, not choices.
- **Harness checks before any P201e GP or deploy output is read:**
  1. **ID2 replay:** `gp_h415.py` with seed 364101, `H0.1-A` and a copy of the H364 memo reproduces the stored X0
     `AGG.g50.CV3@12` on 2 S60 contexts exactly (Δu = 0, identical spec, 0 physical worker runs).
  2. **Deploy smoke on DEV2E** (never scored):
     - the replays reproduce the runs' stock outputs (Δu = 0);
     - every `kf<j>` envelope has SEARCH rows only and its fold sizes differ by ≤ 1;
     - the AUDIT ledger is consistent, with 0 same-family bank hits.
- **Guards and closed data.**
  - Sandbox: the H335 bwrap / docker sandboxes, which bind only the 1,219 TRAIN tables.
  - H317 guard v2: 0 refusals.
  - DOPE `c51a56d0` unchanged.
  - No dev91 path is mounted, read or listed. The flywheel writer stays stopped.
- **No stored entrant is rerun.** X0e is the locked recipe on a new panel, the necessary baseline. The stored H410 C1
  fold-0 pair is reused.
- **Hosts and budget:**
  - xbabe0: ≤ 108 of 128 cores; host CPU ≤ 85%; MemAvailable ≥ 20 GiB.
  - xbabe1 RTX 4090: the screener folds only.
  - xbabe2: control and the 73-second-class view encode, ≤ 8 threads.
  - Compute deadline 2026-11-09T06:00Z. If time runs short, drop in this order: `KFB1`, `HFR`, `P2` builders other
    than `P2+LIB2.BAG_CAR`.
- **Records.**
  - Code, PREREG, amendment, README and `metrics.json` go on `v1/next`.
  - Run directories go to a create-only S3 archive (exact-version GET verified).
  - Same-agent review receipts in `~/jope_plan/evidence/H415/` (not an independent review).
