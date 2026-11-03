# H273 (`h273_rerank8_cleanroom_veto`) — Preregistration

- **Wave ID**: `h273-rerank8-cleanroom-veto-20261103` (`H273`)
- **Study Path**: `research/flywheel_selfplay/h273_rerank8_cleanroom_veto`
- **Owner**: `Argon`
- **Assignment**: `/home/ubuntu/nudges/hw_t1925_argon_h273.md`
- **Claim**: `/home/ubuntu/jope_plan/ns/receipts/H273_claim.txt`
- **Status at Preregistration**: `PREREGISTERED` (`outcomes_read = 0`)
- **Fold Manifest S3 Pointer**: `fold_manifest.s3.json` (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H273/prereg/fold_manifest.json`, `VersionId = oCb50mvrbyR0CCN_nhxc16GuBOpfzSQZ`, `sha256 = 2900c7c7742eb3639ff27b4b75625cb90f6794cb02555ff945ab110c376d8d2d`, `694,857` bytes, `outcomes_read = 0`)

---

## 1. Scientific Rationale & Clean-Room Provenance

- **Why `H273`**: `H270` (`aee5dd042`, `rd_opus`) reported `RERANK8 - CONTROL = +0.0523 [+0.0171, +0.1060]` (`Elo = 1562`, `DL-Val91 = +0.0346`) by re-ranking `r882`'s `K = 8` lowest-cost candidates instead of `K = 4`. However, `H270` was a single run in one lane, and `H272` (`rd_opus`) re-runs that same codebase on fresh seeds.
- **`H273` Role**:
  1. **Independent-lane clean-room replication**: `runner.py` is written from `H270` `PREREG.md` / `README.md` text specification **ONLY**, generalizing Argon's `H266`/`H269` `K = 4` implementation to `K = 8` (`568` feature columns). It does **not** import or copy `topk.py`, `h265_core.py`, or `h270/runner.py`.
  2. **Placebo with teeth**: Because `RERANK8_SHUF` with the strict `H257` inner-`FIT` selector fell back to `r882` in `43/45` cells in `H270` (testing selector rejection rather than unconditional ranker predictions under shuffled labels), `H273` adds **`RERANK8_CR_SHUF_UNC`** (`fallback` disabled; finite margins `0.0..1.0` and pure `margin = 0.0` top-1 argmax `RERANK8_CR_SHUF_UNC_M0`) alongside `RERANK8_CR_UNC`.
  3. **Outcome-free invalid-gate veto (`RERANK8_CR_VETO`)**: Adds the frozen `H269` outcome-free inner-`FIT` `LGBMClassifier` (`class_weight = {0: 1.0, 1: 10.0}`, fixed `tau = 0.10`) to veto any non-`r882` pick (`j > 0`) with `P(invalid) > 0.10` back to `r882` (`j = 0`).

---

## 2. Frozen Constants & Arms

- **Candidate slice (`K = 8`)**: `r882`'s `8` lowest-cost candidates (`np.argsort(rcost, kind="mergesort")[:8]`), `568` outcome-free float32 features (`28` score/pairwise cost-diff cols + `12` cosine/op cols + `412` program cols + `24` `sem24_z` cols + `16` `proj16_z` cols + `76` context descriptor cols).
- **Relevance labels (`compute_true_topk_relevance`)**: Gate-failed candidates (`gate == False`, utility `-2.0`) receive `0` (`worst`); valid candidates are ranked by ascending true revealed cost (`fitness`, tie-broken by ascending slot `j in 0..7`), receiving `7, 6, ..., 0`.
- **Ranker grid**: `LGBMRanker` (`objective="lambdarank"`, `lambdarank_truncation_level=8`, `max_depth=3`, `num_leaves=7`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `lr in {0.03, 0.05, 0.10}`, `n_estimators in {50, 100, 200}`, `margin in {0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, inf}`).
- **Strict inner-`FIT` selector (`H257` rule)**: `4`-fold inner family-grouped CV on the `164` `FIT` families; falls back to `r882` (`margin = inf`) unless inner-`FIT` OOF equal-family gain `> +0.001` with `2,000`-draw paired family bootstrap `LCB95 > 0.0` (ties keep `r882`).
- **Arms**:
  1. `CONTROL`: `r882` top-1 (`slot 0`).
  2. `RERANK8_CR` (confirmatory, clean-room): `K = 8` (`568` cols) + strict inner-`FIT` selector.
  3. `RERANK8_CR_SHUF` (strict within-family label shuffle placebo): within-family permutation of `rel_8` on `FIT` + strict inner-`FIT` selector.
  4. `RERANK8_CR_UNC` / `RERANK8_CR_SHUF_UNC` (`+ M0`): unconditional versions with `margin = inf` fallback disabled so the placebo actively re-ranks candidates.
  5. `RERANK8_CR_VETO` (descriptive): `RERANK8_CR` + frozen `H269` `LGBMClassifier` invalid-gate veto (`tau = 0.10`) trained on inner `FIT` only.

---

## 3. Two-Stage Evaluation & Decision Rule (`Holm m = 2`)

1. **Stage 1 Reproduction Gate (before fresh seeds)**:
   - Run `RERANK8_CR` at `H270` seeds (`fold_seeds = 270140, 270141, 270142`, `model_seeds = 270101, 270102, 270103`, `boot_seed = 270150`, `45` cells).
   - Report per-cell ASSESS pick agreement (`% identical` over all `10,971` cell-context picks) and `PRIMARY` median family gain vs `H270` (`+0.0523 [+0.0171, +0.1060]`). Agreement does not need to be bit-exact; report it honestly.
2. **Stage 2 Fresh-Seed Confirmatory Evaluation**:
   - **PRIMARY (`DL-Train`: `205` families, `1,219` contexts)**: Fresh `fold_seeds = 273140, 273141, 273142` $\times$ `5` family-grouped folds $\times$ `model_seeds = 273101, 273102, 273103` (`45` cells/arm), `10,000`-draw paired family bootstrap (`boot_seed = 273150`), per-seed median over fold seeds + `95%` CI.
   - **SECONDARY (`DL-Val91`: `19` families, `91` contexts)**: `fold_seed = 273140`, `model_seeds = 273101, 273102, 273103` (`5`-cell mixture per seed); `Native91 = 1.1377941898164525` (`91/91` bit-exact) reproduced first.
   - **Multiplicity**: `Holm m = 2` over `{PRIMARY, SECONDARY}` (`97.5%` simultaneous CI `[1.25%, 98.75%]`).
   - **Verdict (`REPLICATED` vs `NOT_REPLICATED`)**:
     - `REPLICATED` iff `RERANK8_CR` (or `RERANK8_CR_VETO`) satisfies all 6 gates:
       1. `PRIMARY` median family gain `> +0.002`
       2. `Holm m=2 LCB95 > 0.0` (and `CI95 LCB > 0.0`)
       3. `3/3` model seeds positive on `PRIMARY`
       4. Beats `SHUF` (`> RERANK8_CR_SHUF` strict AND `> RERANK8_CR_SHUF_UNC` unconditional, with `LCB95 > 0.0`)
       5. `SECONDARY` (`DL-Val91`) gain `> 0.0` on `3/3` model seeds (and `Holm m=2 LCB95 > 0.0`)
       6. `DL-Train` `-2.0` invalid picks (`45` cells) `<= CONTROL`'s (`153`).
     - Else `NOT_REPLICATED` (`NO_ADVANCE`).
   - **Packaging Rule**: Upload full outputs (`metrics_full.json`, `details.json`, `panels.npz`, `fit_selections_lock.json`, `stage1_h270_repro.json`, `baseline.json`) to S3 first (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H273/attempt-001/`), and commit only small text (`metrics.json <= 50 KB`, `fold_manifest.s3.json`) to Git.
