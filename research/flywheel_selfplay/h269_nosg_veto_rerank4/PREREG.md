# H269 Preregistration — `h269_nosg_veto_rerank4`

- **Wave ID**: `h269-nosg-veto-rerank4-20261103` (`H269`)
- **Study Path**: `research/flywheel_selfplay/h269_nosg_veto_rerank4`
- **Owner**: `Argon`
- **Status**: `PREREGISTERED` (`outcomes_read = 0`)
- **Deadlines**: Prereg `<= 19:15Z`, Compute stop `<= 20:50Z`, Closeout `<= 21:35Z`

## 1. Scientific Motivation & Hypothesis

In `H266` (`bf68683ee`), full `560`-column `RERANK4` replicated on `DL-Val91` (`+0.012861`, `Holm p = 0.0016`, `3/3` seeds, `0` `-2.0` invalid gates) and achieved positive `DL-Train` `205`-family OOF utility (`+0.018938`, `3/3` seeds, `Holm p = 0.0495`), but missed `LCB95 > 0` (`95% CI [-0.004329, +0.051845]`) under the strict `H257` inner-`FIT` selector while incurring `55–56` `-2.0` invalid picks on `DL-Train` (vs `41` for `CONTROL = r882`).

Crucially, `H266`'s descriptive feature-block ablation `NO_SCOREGEOM` (`528` columns, dropping all `32` `r882` scalar score/gap/cosine columns `0..31`) was the **only** arm on `fold_seed = 266140` with `DL-Train` OOF `LCB95 > 0` (`+0.012210`, `95% CI [+0.000462, +0.047014]`, `p = 0.0228`, `3/3` seeds) and `DL-Val91` gain `+0.020092` (`p = 0.0068`, `3/3` seeds), whereas `SCORE_ONLY_20` (`+0.000000`, `0/15` cells admitted) collapsed completely. Because `NO_SCOREGEOM` was 1 of 5 post-hoc ablations on a single fold seed, `H269` performs a preregistered confirmatory test on **3 fresh fold seeds (`269140, 269141, 269142`)** and **3 fresh model seeds (`269101, 269102, 269103`)**, paired with an outcome-free inner-`FIT` `LGBMClassifier` `-2.0` invalid-gate veto (`NOSG_VETO`).

## 2. Frozen Representation, Models & Arms

- **Frozen Feature Set (`NO_SCOREGEOM_COLS = np.arange(32, 560)`, `528` cols)**:
  - Exact `H265`/`H266` `build_top4_feature_matrix_for_context` minus `cols 0..31` (retaining `prog_top4` `412` + `sem_z_top4` `24` + `proj_z_top4` `16` + `ctx_vec76` `76` = `528` strictly outcome-free features).
- **Frozen `LGBMRanker` & Grids**:
  - `objective="lambdarank"`, `max_depth=3`, `num_leaves=7`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `lr in {0.03, 0.05, 0.10}`, `n_estimators in {50, 100, 200}`, `margin in {0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, inf}`.
- **Frozen `LGBMClassifier` Invalid-Gate Veto Model (`VETO_CLF_PARAMS`)**:
  - Trained on inner `FIT` candidate rows only (`528` outcome-free features) predicting `y_inv = 1[not gate]` ("candidate is a `-2.0` invalid gate"):
    `objective="binary"`, `max_depth=3`, `num_leaves=7`, `min_child_samples=15`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `class_weight={0: 1.0, 1: 10.0}`, `deterministic=True`, `force_row_wise=True`.
  - Any non-`r882` pick (`j > 0`) with `P(invalid) > tau` falls back to `r882` (`j = 0`), with `tau` selected on inner `FIT` only from frozen grid `VETO_TAU_GRID = {0.05, 0.10, 0.20}` and `margin` from `MARGIN_GRID = {0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, inf}` subject to inner-`FIT` OOF invalid-gate non-regression (`<= CONTROL`, Tier 1 zero new valid-to-invalid switches, Tier 2 strict invalid reduction) and strict `H257` admission (`inner_fit_gain > +0.001 + 1e-12` and `inner_lcb95 > 0.0`, else `m* = inf`).
- **Arms**:
  1. `CONTROL`: `r882` top-1 candidate (`slot 0`).
  2. `NOSG`: Frozen `H266` `NO_SCOREGEOM` (`528` cols, exact `H266` inner-`FIT` selector).
  3. `NOSG_VETO`: `NOSG` + inner-`FIT` `LGBMClassifier` `-2.0` invalid-gate veto (`528` cols).
  4. `NOSG_SHUF`: Within-family label shuffle placebo (`528` cols).
  5. `RERANK4`: Full `560`-col `H265`/`H266` re-ranker (reference only).

## 3. Endpoints, Multiplicity (`Holm m = 3`), and Decision Rule

- **PRIMARY (`DL-Train` OOF, `205` families / `1,219` contexts)**: Repeated nested `5`-fold family-grouped CV across `fold_seeds = (269140, 269141, 269142)` $\times$ `5` folds $\times$ `model_seeds = (269101, 269102, 269103)` (`45` cells per arm); single-pick equal-family utility minus `CONTROL` (`10,000`-draw paired family bootstrap, `boot_seed = 269150`, per-seed medians across fold seeds).
- **SECONDARY (`DL-Val91`, `19` families / `91` contexts)**: Fresh-seed evaluation (`model_seeds = 269101–269103`, `fold_seed = 269140`) after reproducing `Native91 = 1.1377941898164525` (`91/91` bit-exact). Disclosed that `DL-Val91` is heavily reused across prior waves; `DL-Train` OOF on fresh folds is the decision endpoint.
- **Multiplicity (`Holm m = 3`)**: Step-down Holm-Bonferroni over `{NOSG DL-Train OOF, NOSG_VETO DL-Train OOF, NOSG_VETO DL-Val91}`.
- **ADVANCE Criteria**: `ADVANCE` iff an arm (`NOSG` or `NOSG_VETO`) satisfies all of:
  1. `PRIMARY` (`DL-Train` OOF) median family gain `> +0.0020`,
  2. Step-down `Holm` (`m = 3`) `LCB > 0.0` (and `Holm p < 0.05`),
  3. `3 / 3` model seeds positive on `PRIMARY`,
  4. Beats `NOSG_SHUF` (`gain > 0` and one-sided `p < 0.05`),
  5. `DL-Val91` gain `> 0.0` on `3 / 3` seeds, and
  6. `DL-Train` `-2.0` invalid picks `<= CONTROL`'s across all `3` seeds.
  Else `NO_ADVANCE` (fallback identity / ties = `NO_ADVANCE`).
