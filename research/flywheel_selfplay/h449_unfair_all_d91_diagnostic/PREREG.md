# H449 Pre-Registration — Quarantined Matched Comparison: Complete `ALL-NON-DEV91` (`0%` `dev91`) vs. `ENTIRE` (`100%` `dev91`) + Exact Token & Compute Cost Ledger

> **QUARANTINE & CONTAMINATION WARNING (`DIAGNOSTIC ONLY`)**
> `H449` runs a strictly matched two-regime comparison at explicit owner request (`2026-11-12T06:15Z` / `10:40Z`):
> 1. **`ALL-NON-DEV91-EXHAUSTIVE` (`0%` `dev91` exposure):** `104M + 24M` plain `ICL` warm-continued on GPU (`xbabe1` `RTX 4090` + `xbabe3` `RTX 3090`, `1.0B` cell-tokens per model) on `all_non_d91_exhaustive_bank.npz` (`3,392` valid complete tables across `2,430` distinct non-sealed, non-`dev91` real regression datasets / `2,328` families: all `661` `APEX` non-`CONFIRM` `.jtt1` + all `1,219` `DL-Train` `.jtt1` + all `1,512` remaining non-sealed, non-`dev91` S3 `T1` complete `train.csv ∪ test.csv` datasets; `0` `dev91`, `0` `APEX-CONFIRM`, `0` `DL-Holdout`), scored as `h449.FAIR_ALL_NON_D91_STACK` and `h449.FAIR_ALL_NON_D91_ICL_ONLY`.
> 2. **`ENTIRE` (`100%` `dev91` exposure):** Initialized from `all_non_d91_{L,M}` and fine-tuned strictly inside `--network none` Stage-V Docker on `xbabe0` on `everything_all_d91_bank.npz` (`all_non_d91_exhaustive_bank.npz` (`3,392`) $\cup$ `d91_cousin_s3_bank.npz` (`42` `dev91` S3 cousins) $\cup$ all `91` complete `dev91` `[X_search ∪ X_audit, y_search ∪ y_audit]` tables = `3,525` complete tables across `2,491` distinct real datasets), scored as `h449.UNFAIR_ALL_D91_STACK` and `h449.UNFAIR_ALL_D91_ICL_ONLY`.
>
> - **NEVER promoted** to any canonical checkpoint store (`/home/ubuntu/jope-data/h335-cl/neural`, S3 canonical `models`/`learning`).
> - **NEVER written** to canonical `research/elo/reference-board/boards/dev91-audit-v1/scores/` or `/home/ubuntu/jope-data/h335-stagev/refboard/`.
> - **Hard-gated (`gates_h449_prelock.sh`)** to refuse prelock and unlock until `H448` (`scores/h448.APEX_FAIR_CHAMPION.jsonl` + S3 `GET`-verified `ARCHIVE.json`) is published on `origin/v1/next` AND a separate owner Stage-V receipt (`/home/ubuntu/jope_plan/ns/receipts/H449_UNFAIR_LOCK_sha256.txt`) is recorded.
> - **Zero Holdout Opens:** `APEX-CONFIRM` (`142` families) and `DL-Holdout` / `sealed` remain at **`0` opens**.

## 1. Experimental Arms & Evaluation Contract
At test time on `dev91-audit-v1` (`91` contexts / `19` families), **every arm obeys the identical `(X_search, y_search, X_audit) -> y_hat_audit` evaluation contract** (`0` `y_audit` values passed in context at test time). Each arm is ranked individually against the 33-entrant post-`H448` board (including `kumo_tabular_large_e8.v1` at `2063.9` and `h448.APEX_FAIR_CHAMPION`) on both `all_91` and `H425` `without_feature` clear contexts:

1. **`h448.APEX_FAIR_CHAMPION` (`Wave H448` Official Canonical Fair Entrant):** `0%` `dev91`, `P201b`-disjoint training bank.
2. **`h449.FAIR_ALL_NON_D91_STACK` & `h449.FAIR_ALL_NON_D91_ICL_ONLY` (`0%` `dev91` Exhaustive Non-`dev91` Control):**
   - Trained on `all_non_d91_exhaustive_bank.npz` (`3,392` complete real tables / `2,430` datasets, `0%` `dev91`), `1.0B` cell-tokens per model on GPU.
3. **`h449.UNFAIR_ALL_D91_STACK` & `h449.UNFAIR_ALL_D91_ICL_ONLY` (`100%` `dev91` Quarantined Upper-Bound):**
   - Initialized from `all_non_d91_{L,M}` and fine-tuned on `everything_all_d91_bank.npz` (`3,525` complete tables incl. `100%` of `dev91` `[X_search ∪ X_audit, y_all]` + `42` `dev91` S3 cousins, with `6` pre-cached 72/28 folds per `dev91` table covering 100% of `dev91` rows as both context and query).

## 2. Token & Compute Cost Ledger (`token_compute_ledger.py`)
Produces create-once `TOKEN_AND_COMPUTE_COST_LEDGER.json` and `TOKEN_AND_COMPUTE_COST_LEDGER.md` recording:
- Stage-1 ancestral pretraining (`fit_icl_bank.npz` + `PROBS_V3S`) tokens, FLOPs, and GPU-hours calibrated to exact checkpoint `state["datasets"]`,
- Stage-2 GPU `NON-DEV91` adaptation exact step-level `cum_row_tokens`, `cum_cell_tokens`, `cum_real_cell_tokens`, `approx_train_flops`, and GPU-hours,
- Stage-3 Stage-V `ENTIRE` fine-tuning exact step-level `cum_row_tokens`, `cum_cell_tokens`, `cum_real_cell_tokens`, `cum_d91_cell_tokens`, `approx_train_flops`, and CPU-hours,
- Test-time `dev91` inference cell-tokens & compute across our arms and NVIDIA Kumo Large (`215.7M` params) / Kumo Small (`28.5M` params) / TabFM.
