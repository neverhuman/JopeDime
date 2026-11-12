# H446 — Pre-Registration: `H299`-Verified & `H425`-Screened All-Legal Real Regression Pretraining Banks (`MAX_FEATS = 400`)

- **Wave:** `H446` (`h446_all_legal_real_bank`)
- **Owner:** `@agy_apex_w10`
- **Cohort:** Pure infrastructure & data-hygiene wave constructing three nested, strictly holdout-disjoint real regression pretraining banks (`MAX_ROWS = 4096, MAX_FEATS = 400`, deterministic `SEED = 446301`) for Wave `H447` (`24M`/`104M` ICL scaling + `P201b` family-exposure control); **zero** model evaluations on `APEX-DEV`, `APEX-CONFIRM`, or `dev91` (`DL-Val`)
- **Compute:** CPU-only (`CUDA_VISIBLE_DEVICES=""`), `<= 4` processes at `nice 19` (`cpus 104-107`), hard wall cap `7,200 s` (`2 h`), `0` S3 write operations

---

## 1. Motivation & Design

`H419`'s real-table pretraining bank (`FIT-ICL`, `419` tables, cropped to `MAX_FEATS = 100`) omitted:
1. ~`1,250+` additional legal real regression datasets in `H299`'s `2,729` `H283`/`H286` eligible families (`APEX-split-v1` `k in {2, 3, 4}` -> `FIT-SEL` / `FIT-ICL`, already disjoint from `DL-Val`/`dev91`, `DL-Holdout`, `P3_{dev,heldout,sealed_excluded}`, and `sealed`),
2. `DL-Train` (`257` distinct regression datasets in `/home/ubuntu/h317-stage/res01-train-view/selected.json["train"]`, strictly excluding the `91` `selected.json["dev"]` `DL-Val` contexts),
3. Wide-table context structure (`100 < F <= 400`, truncated by `H419`'s `MAX_FEATS = 100`), and
4. `H425` `S1–S4` cross-upload deduplication against `APEX-DEV`, `APEX-CONFIRM`, and `dev91` metadata (`H419`'s `FIT-ICL` bank triggered `21` `H425` contamination flags on `APEX-DEV`).

Per `@cl_fwriter`'s `A-020` / `A-021` agreement, `H446` constructs **three nested, contamination-screened banks** (`1` canonical table per `dataset_id` to prevent multi-view dataset overweighting):

1. **`fit_icl_wide400_bank.npz` (`cohort == "FIT-ICL"` strictly, `>= 1,000` distinct datasets):**
   - Strictly `APEX-split-v1` `k in {3, 4}` (`FIT-ICL`) across `APEX` (`440`), `s1-prod` `train-only` `teacher.jtt1.gz` (`nq == 0`), and `s3://veox-jopedime-datasets-use2/blobs/sha256/<train_sha256>` (`train.csv` only, never `test.csv`).
   - Compatible out-of-the-box with unmodified `icl_inject.RealBank` (`set(cohort) == {"FIT-ICL"}`).
   - Leaves **`FIT-SEL`, `P201b`, `APEX-DEV`, `APEX-CONFIRM`, and `dev91` 100% unseen**.
2. **`all_legal_excl_p201b_bank.npz` (`>= 1,500` distinct datasets):**
   - Adds `APEX-split-v1` `k == 2` (`FIT-SEL`: `SEARCH`-only via `C.search_and_audit_envelopes` for `APEX` `FIT-SEL` contexts, `0` `FIT-SEL` `AUDIT` rows; `train-only` for non-`APEX` `FIT-SEL` datasets) + `DL-Train` datasets outside `P201b`'s `201` coarse `fam-...` families.
   - Leaves **`P201b` (`201` families), `APEX-DEV`, `APEX-CONFIRM`, and `dev91` 100% family-unseen**.
3. **`all_legal_bank.npz` (`>= 1,800` distinct datasets):**
   - Adds `P201b` **`SEARCH`-only envelopes** (`201` contexts via `C.search_and_audit_envelopes`, **`0` `P201b` `AUDIT` rows**; sibling `DL-Train` views on `P201b` `dataset_id`s are excluded so `P201b` `AUDIT` rows cannot leak through sibling cuts) + remaining `DL-Train` datasets disjoint from `P201b` `dataset_id`s.
   - Enables maximum legal Option A pretraining capacity (`H447` -> `H448` -> `H449`) **and** a clean paired `P201b` (`189` unprobed families) evaluation of Bank 3 vs Bank 2 testing whether target-free / `SEARCH`-row family exposure in the ICL training bank lifts `AUDIT` Elo.

---

## 2. Multi-Layer Holdout & Contamination Screen (`Zero` Holdout Opens)

1. **SHA-256 Pinned Manifests:** Verifies `H299` `cohort_v6_manifest.json` (`845a6e57...`), `fresh_cohort_v1.json` (`12d2b1f3...`), `APEX_COHORTS.json` (`11a57d80...`), `CONFIRM_META.json` (`acbce707...`), `selected.json` (`7d64774a...`), `board_p201b_BOARD.json` (`cb993c45...`), `S3_DATASET_CATALOG.json` (`e5f732fd...`), and `catalog/v246/catalog.jsonl` (`c5f5eb6d...`).
2. **Metadata & Provenance Denylist:** Excludes all `14` `H299` `sealed_ids`, `APEX` `CONFIRM` (`241`) and `DEV` (`239`), `DL-Val` (`dev91`, `19` base datasets / `91` contexts), `DL-Holdout` (`93` contexts), and `P3_{dev,heldout,sealed_excluded}` coarse-family closures.
3. **`H425` `S4` Metadata Screen (`CONFIRM` & `dev91`, `0` table opens):** Excludes any candidate triggering `screen_h425.name_tokens` `S4` (`Jaccard >= 0.75` with `>= 2` shared tokens, or equal token sets) against any of the `241` `APEX-CONFIRM` contexts or `19` `DL-Val` (`dev91`) base datasets (plus exact `source_url` / `source_file` / normalized-name match against `DL-Val`).
4. **Full `H425` `S1–S4` Fixed-Point Pruning Against `APEX-DEV` (`239` tables, feature columns only):** Iteratively indexes remaining candidates and runs unmodified `screen_h425.screen` over all `239` `APEX-DEV` tables (feature matrix `X` only; `DEV` targets never sketched), dropping every matched candidate until **all `239 / 239` `APEX-DEV` contexts report `flag_feature == False` (`0` `copy`, `0` `possible`)**.

---

## 3. Integrity Gates (`gates_h446.sh`, `6/6` Required)

1. **`g1_prereg_before_output`:** `PREREG` commit on `origin/v1/next` predates every output file in `$R`.
2. **`g2_pinned_stage`:** Read-only stage `CODE.sha256` passes `sha256sum --quiet -c`.
3. **`g3_selftest`:** `icl_bank_all_legal.py selftest` passes all 4 unit tests (`APEX-split-v1` parity across `1,153` rows, CSV/width-400 cropper, `H425` copy pruner, and `RealBankAllLegal` holdout-cohort refusal).
4. **`g4_zero_holdout_and_audit_leakage`:** `0` `CONFIRM` opens, `0` `dev91` opens, `0` `P201b` `AUDIT` rows written, `0` `FIT-SEL` `AUDIT` rows written, `guard.refusals == 0`, `guard.stage_v == false`, `s3_write_operations == 0`, and sandbox `apex/tables` count `== 661`.
5. **`g5_fit_icl_strict_cohort_and_scale`:** `fit_icl_wide400` has `cohorts == ["FIT-ICL"]` and `>= 1,000` tables; `all_legal_excl_p201b` has `>= 1,500` tables; `all_legal` has `>= 1,800` tables, `max_cols_observed <= 401`, and `wide_gt_100_feats >= 50`.
6. **`g6_cpu_budget`:** `wall_s <= 7200` and `procs <= 4`.
