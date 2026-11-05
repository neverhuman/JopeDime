# H369 — once-score the H364 XSTACK-2L entrant on the fixed holdout board `dev91-audit-v1`

- **Owner:** `cl_fwriter`, Claude Opus 5.5.
- **Request:** owner, 2026-11-05 ~13:20Z: "score all new waves on the same holdout and update the ELO scores".
- **Commit order.** This PREREG, `d91_h369.py` and `run_h369.sh` are committed to `origin/v1/next` **before** the
  one-shot unlock. Nothing is chosen on dev91.

## 1. Entrant (frozen before any dev91 read)
- **Board id:** `h364.XSTACK_2L`.
- **Definition.** H364's two-level stack on top of the locked H335 GP output, unchanged:
  - code: `h354_core.py`, `h364_core.py` and `lib_oof_h354.py`, byte-equal to `origin/v1/next`;
  - design: part-B `DESIGN.json` (`f052a59f35`, `c0.35_l0.1_d0_tinf`).
- **Formula.** Per (context, search seed): ŷ = X0 + Σⱼ βⱼ (Lⱼ − X0).
  - β ≥ 0, Σβ ≤ 0.35, ridge 0.10 toward GP-only.
  - β is fit on SEARCH only and cross-fitted. It is used only if it beats X0 out-of-fold on both SEARCH folds;
    otherwise ŷ = X0 exactly.
- **X0** is H335 Stage V's stored `AGG.g50.CV3@12` output for that seed: the existing board entrant `h335.H335_AGG`,
  with its stored weights, never re-solved. GP records, memo and envelopes are pulled read-only from xbabe0 and are
  sha-identical (455 files).
- **Library** Lⱼ: catboost, lightgbm, xgboost and random_forest (`.v1`).
  - **AUDIT predictions:** the **stored** `dev91-audit-v1` records (prediction sha checked). No reference model and no
    board entrant is refit, re-run or rescored.
  - **SEARCH out-of-fold predictions:** `lib_oof_h354.py --audit stored`, which fits on dev91 **SEARCH rows only**
    (the GP's 2-fold SEARCH split). It runs in the H354 bwrap sandbox: no network, inputs only.
- **Board value per context:** the mean over the two H335 search seeds (335101, 335102), matching every `h335.*` GP
  entrant.

## 2. Harness checks (any failure ⇒ `INCOMPLETE`, no board row)
1. **Alignment, ≤ 1e-9 relative**, for every context and seed. These must agree:
   - the GP records;
   - the board AUDIT envelope: rows, folds, X, y;
   - the AUDIT baseline;
   - the member AUDIT RMSEs;
   - each stored library prediction's RMSE and baseline.
2. **X0 recompute, within 1e-6** (H364 §7.1): the stored SEARCH log-ratios and AUDIT utility, and **the board's
   `h335.H335_AGG` per-seed value**.
3. **Coverage:** library records exist for all 91 × 4 (learner, context) pairs. A failed learner fit is excluded on
   SEARCH, as in H364.

## 3. Output
- `scores/h364.XSTACK_2L.jsonl`, written create-once. The ladder is regenerated with `tools/ladder.py` from stored
  scores only. Anchor `h335.H257_REPLICA` = 1500.
- A descriptive paired summary vs `h335.H335_AGG`:
  - context gains;
  - family mean and LCB;
  - families won, tied and lost.
- **No adoption claim.** dev91 is a reuse cohort (H257 tuned on it). The H364 constants were chosen on TRAIN
  (P201b + P201c) only.

## 4. Not scored in H369 (reasons)
- **H354/H357 `X_CL`.** It re-solves the first-level weights, which needs AUDIT predictions of GP candidates that X0
  gave zero weight. Those are new GP worker AUDIT runs on dev91, not available from stored records. It also failed its
  TRAIN confirmation (H357 p = 0.074).
- **H354 `X_LLM` / `X_ALL`.** They need the four LLM regressors' SEARCH out-of-fold predictions plus the same extra GP
  AUDIT runs. H354 did not advance them.
- **Other lanes' waves (H352–H368).** Owned by other agents; not GP-stack entrants of this board.

## 5. Unlock
- The guard is H317 v2 (installed by importing `h364_core`). `d91_h369.py` points its Stage V unlock at
  `$H369_DATA/stage_v/LOCK.json` and the create-once receipt `ns/receipts/H369_LOCK_sha256.txt`, as H335 did.
- `LOCK.json` lists the sha256 of every staged code file, the four H317 modules, the pulled GP file list and
  `BOARD.json`.
- `unlock` is one-shot. Later steps re-check the LOCK (`resume_stage_v`). Sealed, holdout and DL-Test paths stay
  refused.
