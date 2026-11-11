# H433 raw_formula_features: pre-registration

- **Written:** 2026-11-11, before any H433 column was computed on a real table, by `cl_fwriter` (Claude, model
  `claude-fable-5-2`, screen `agy_research_wave` on xbabe2).
- **Authority:** the session user's instruction of 2026-11-11 about 15:15Z to run and test the CW designs as
  incremental research waves, in close contact with the Gemini lane. Claim receipt
  `~/jope_plan/ns/receipts/H433_claim.txt`. Split agreed with `agy_apex_w10` on AGENT_CHAT at 15:41Z.
- **Design document:** `/home/ubuntu/jope_plan/waves_claude_20261111/CW2_FORMULA.md`. Where this file differs, this
  file binds.
- **Relation to H430:** the same panels, bank files, column format and stack step. H430's column files are used
  read-only for one incremental reading.

`u` is −ln(RMSE / RMSE of the mean predictor). +0.01 u is about 1% lower RMSE.

## 1. Question
The worker min-max scales every column before any stage, so a ratio of two raw columns cannot be built and other
pair features exist only for the columns most correlated with the target. Does discovering two-column arithmetic on
the raw values, inside each fold, and handing it to ordinary learners raise the stack? Is it the structure that
helps, or would far more model capacity do the same?

## 2. Hypotheses
- **H1 (benefit).** The stored stack plus the four formula columns beats the stored stack on P201b under the H335
  primary test.
- **H2 (structure, not capacity).** The formula arm's clipped gain is at least twice that of a capacity-only control
  arm.
- **H3 (increment).** The formula columns still add to a stack that already holds H430's three key columns.

## 3. What motivated the design, and what that does to the P201b reading
- The design came from the probe of ten P201b tables with the largest Kumo gaps
  (`/home/ubuntu/jope_plan/waves_claude_20261111/EVIDENCE_structure_probe.md`). Four of them turn on a difference,
  product or ratio of two raw columns.
- P201b is therefore not blind for those ten tables. Every P201b result is reported on all 201 contexts and without
  the ten probed contexts (`PROBED_P201B.txt` in the H430 stage).
- P201b shares 185 families with P201e, the design panel of the stored stack. It is a development panel.
- FIT-SEL was not looked at in designing the features. H430's FIT-SEL design check had been read (keys only) when
  this file was written.

## 4. Method (fixed by the published code, `formula_core.py`)
All fitting uses SEARCH rows only, inside the stored 5-fold labels. Missing cells are filled with the fit-row median
before any arithmetic.

| Piece | Rule |
| --- | --- |
| Parents | All columns when there are at most 24; otherwise the 24 with the best single-column smoother score |
| Candidates | For each pair: difference, sum, product, both ratios, log-ratio. For each parent: log, square, square root, reciprocal |
| Guards | A ratio or reciprocal is skipped when the smallest absolute denominator is below 1e-6 of its median; a log needs one sign. Ratios, logs and roots are winsorised to the fit rows' 0.1%–99.9% range; differences, sums, products and squares are not |
| Score | u of a 32-bin equal-frequency smoother (bin means, linear interpolation between bin centres), out of fold on an inner 3-fold split |
| Kept | Gain over both parents ≥ 0.10 u, score ≥ 0.5 u, and gain above the 95th percentile of the largest gain found on each of 20 permuted targets; the 12 best by gain |
| Depth two | The four best survivors combined once more with the eight best parents under the same rule |
| `form_gbm` | LightGBM 4.6.0 (the H430 setting) on the features plus the survivors |
| `form_gam` | Ridge (generalised cross-validation) on restricted cubic splines (8 knots) of the survivors and of the 12 best raw columns |
| `form_sr` | Sparse linear formula over survivors, the 12 best raw columns and pairwise products of survivors: forward selection, least-squares refit at each step, stop when BIC stops improving, at most 12 terms. The formula text is stored |
| `form_interp` | One-dimensional interpolation of the target on the best survivor (moving average over 1, 3, 7, 15 or 31 neighbouring values, chosen by leave-one-out RMSE; linear continuation beyond the ends) plus ridge on the remainder; used only when its leave-one-out u is ≥ 3.0, otherwise equal to `form_gam` |
| `gbm_plain` (control) | The same LightGBM on the features alone |
| `cap` (control) | Capacity only: LightGBM with up to 3,000 rounds, learning rate 0.03, 127 leaves, 2 rows per leaf, averaged with an RBF kernel ridge on all fit rows (bandwidth from 0.25, 1, 4 times the median heuristic; ridge by generalised cross-validation) |

When no feature survives in a fold, the four `form_*` columns equal `gbm_plain` there.

Two choices differ from the design document and were made on synthetic checks only, before any real table:
forward selection replaces thresholded least squares (unstable with near-collinear survivors), and the interpolation
uses the raw feature values rather than 128 bin means.

| Arm | Candidates |
| --- | --- |
| BASE | The stored stack value |
| F4 (primary) | The stored 64 + `form_gbm`, `form_gam`, `form_sr`, `form_interp` |
| G1 (control) | The stored 64 + `gbm_plain` |
| C1 (control) | The stored 64 + `cap` |
| on-K3 readings | The same four arms where BASE is the stored 64 plus H430's `key_direct`, `key_gbm`, `factor_glm`, rebuilt with the stored builder |

## 5. Stages, tests and verdicts
| Stage | Data | Rule | Verdicts |
| --- | --- | --- | --- |
| A. Design check | FIT-SEL, 106 contexts (the H430 bank and matrices) | No verdict on benefit. If F4 vs BASE is harmful (clipped mean below 0 with mirror sign-flip p < 0.025), the wave stops and P201b is not run | `FORMULA_HARM_DESIGN` or continue |
| B0. Census | P201b column records (SEARCH rows only) | `F`: a feature survived in at least 4 of 5 folds, with best gain ≥ 0.3 u and best score ≥ 1.0 u | counts reported |
| B1. Replication, primary | P201b, 201 contexts | F4 vs BASE: `h335_stats.primary_test` (gains clipped at ±0.5; clipped family mean > 0; one-sided sign-flip p < 0.025; leave-one-family-out minimum > 0; family median ≥ 0) | `REP_CONFIRMED`, `REP_NULL`, `REP_HARM` |
| B2. Structure or capacity | P201b | F4's clipped gain over BASE compared with C1's | `STRUCTURE` (F4 confirmed and its clipped gain ≥ 2 × C1's, or C1's ≤ 0), `CAPACITY` (C1 passes the primary test and F4's clipped gain < 2 × C1's), `UNDECIDED` otherwise |
| B3. Increment | P201b, on-K3 readings | F4 vs BASE with H430's key columns in both | `INCREMENT_CONFIRMED`, `INCREMENT_NULL` |
| B4. Secondaries | P201b | Share of the positive Kumo L − SXB gap on `F` tables and the fraction of it closed by F4; mean absolute change of the stack on unflagged tables (prediction: ≤ 0.003 u); all of B1 without the ten probed contexts | reported |

No DEV read is part of this wave. By agreement with `agy_apex_w10` (AGENT_CHAT 16:00Z), DEV is left to one later
pre-registered wave agreed with `cl_apex`, after the TRAIN-panel results of H430, H433 and the Gemini lane are known.
Kumo L and TabPFN 3.5 values are stored TRAIN-board reference rows and never stack members.

## 6. Budget and stop rules
- xbabe2, CPU only, 8 single-thread processes at nice 19, at most 24 CPU-hours in total, no GPU. The capacity control
  is the main cost.
- A context that exceeds 2,400 s, or raises an error, is written as not ok and scores BASE in every arm. If more
  than 5% of a panel's contexts are not ok, the stage carries no verdict.
- Code changes after publication are limited to fixing a crash or a non-finite output, published as an amendment
  before the affected stage continues; no finished context is recomputed.
- Nothing is rerun after a verdict.

## 7. Not in this wave
- No DEV, CONFIRM, dev91, DL-Val or DL-Holdout read.
- No change to the stored stack, the search grammar, the Rust worker or any other lane's files.
- Formulas of more than three columns; any Rust port (a follow-up specification only).

## 8. Files
`formula_core.py`, `cols_h433.py`, `stack_h433.py`, `census_h433.py`, `test_formula_core.py`, `stage_h433.sh`,
`driver_h433.sh`; `keys_core.py` (learners) and `PROBED_P201B.txt` come from the H430 stage. Before publication:
`test_formula_core.py` passes 4 of 4 on synthetic tables (a squared difference: interpolation u 8.5 against 1.8 for
plain trees and 2.0 for the capacity control; a ratio; no survivor in 20 structureless draws; finite output far
outside the fit range), and the column writer ran end to end on a synthetic two-context bank.
