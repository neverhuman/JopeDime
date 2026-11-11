# H436 key_lookup_model_backoff: pre-registration

- **Written:** 2026-11-11, before any H436 column was computed on a real table, by `cl_fwriter` (Claude, model
  `claude-fable-5-2`, screen `agy_research_wave` on xbabe2).
- **Authority:** the session user's instruction of 2026-11-11 to run the CW designs as incremental waves, and
  "Proceed" after the H430 report. Claim receipt `~/jope_plan/ns/receipts/H436_claim.txt`.
- **Parent:** H430 `exact_key_encoders`, closeout section 4 (two defects).

`u` is −ln(RMSE / RMSE of the mean predictor). +0.01 u is about 1% lower RMSE.

## 1. Question
H430's lookup column falls back to the fit mean (or to interpolation between keys) when a key value is unknown. A key
value with a single SEARCH row is unknown whenever that row is held out, so the column's out-of-fold error is
dominated by a few rows, and the stack builder under-weights a column that is exact on AUDIT. Its ridge fallback also
had no output guard. Does a lookup that falls back to a tree model's prediction, with guarded output, do better in
the stack than H430's lookup?

## 2. What is already known, and what that does to the reading
- Both defects were found on P201b, and the fix was designed from them. **P201b is a fix verification here, not a
  confirmation.** FIT-SEL was read by H430 and H433 as a design cohort.
- The stand-in harness run of the stack step on FIT-SEL (four contexts with H430's plain LightGBM column copied
  into the new column names) showed the common base of this wave on FIT-SEL: mean u 2.2192 against 2.2142 for the
  stored stack. No H436 column existed.
- The only blind reading available later is the joint DEV wave.

## 3. Method (fixed by the published code, `hybrid_core.py`)
All fitting uses SEARCH rows only, inside the stored 5-fold labels.

| Piece | Rule |
| --- | --- |
| Backoff model | LightGBM with the H430 setting on the raw features; its inner 5-fold out-of-fold prediction of the fit rows is the backoff for scoring keys |
| Eligible key | As in H430: a column, or a pair among the 12 best single keys, with at most n/2 levels and at least half of the fit rows in levels that occur twice or more |
| Encoding | `(S_g + m·model) / (n_g + m)`: the group mean shrunk toward the model's prediction for that row, `m` from {0, 0.5, 2, 8} by leave-one-out RMSE; a row alone in its level gets the model's prediction |
| Key chosen | The lowest leave-one-out RMSE; used only if that is below 0.98 of the model's own out-of-fold RMSE |
| `key_hybrid` | The encoding where the key value is known among the fit rows, the model's prediction where it is not; equal to the model when no key is chosen |
| `ridge_z` | Ridge on z-scored features, penalty by generalised cross-validation (H430's fallback, now its own column) |
| `gbm_plain` (control) | The backoff model alone |
| Guard | Every column is clipped to the fit target range widened by half its width |

| Arm | Candidates |
| --- | --- |
| B | The stored stack value |
| BASE | The stored 64 + H430's `key_gbm` and `factor_glm`, rebuilt with the stored builder |
| KH (primary) | BASE's candidates + `key_hybrid` |
| KD | BASE's candidates + H430's `key_direct` (this is H430's arm K3) |
| KHR | BASE's candidates + `key_hybrid` + `ridge_z` |
| G1 (control) | BASE's candidates + `gbm_plain` |

## 4. Stages, tests and verdicts
| Stage | Data | Rule | Verdicts |
| --- | --- | --- | --- |
| A. Design check | FIT-SEL, 106 contexts | No verdict on benefit. If KH vs B is harmful (clipped mean below 0 with mirror sign-flip p < 0.025), the wave stops | `HYBRID_HARM_DESIGN` or continue |
| B1. Primary | P201b, 201 contexts | KH vs KD: `h335_stats.primary_test` (gains clipped at ±0.5; clipped family mean > 0; one-sided sign-flip p < 0.025; leave-one-family-out minimum > 0; family median ≥ 0) | `FIX_CONFIRMED`, `FIX_NULL`, `FIX_WORSE` (KD better under the mirrored test) |
| B2. Safety | P201b and FIT-SEL | The smallest KH − B over all contexts is above −0.25 u, and no H436 column alone scores below −1 u on AUDIT | `SAFE`, `UNSAFE` |
| B3. Secondaries | P201b | KH vs B under the primary test; fraction of the positive Kumo L − SXB gap closed on H430's 16 `K` tables (H430's arm closed 27.8%; prediction ≥ 50%); the values on `3b0303bb38…` and `7914260659…`; KHR vs KH (does the ridge column add); G1 vs BASE; KH vs KD without the ten probed contexts | reported |

Kumo L values are stored TRAIN-board reference rows and never stack members. No DEV, CONFIRM, dev91, DL-Val or
DL-Holdout read.

## 5. Budget and stop rules
- xbabe2, CPU only, 8 single-thread processes at nice 19, at most 8 CPU-hours, no GPU.
- A context that exceeds 1,200 s, or raises an error, is written as not ok and scores BASE in every arm. More than 5%
  of a panel not ok: no verdict for that stage.
- Code changes after publication are limited to fixing a crash or a non-finite output, published as an amendment
  before the affected stage continues; no finished context is recomputed. Nothing is rerun after a verdict.

## 6. Files
`hybrid_core.py`, `cols_h436.py`, `stack_h436.py`, `test_hybrid_core.py`, `stage_h436.sh`, `driver_h436.sh`;
`keys_core.py` and `PROBED_P201B.txt` come from the H430 stage. Before publication: `test_hybrid_core.py` passes 4 of
4 on synthetic tables (out-of-fold u 2.82 against 1.85 for H430's lookup when some key values have one fit row;
exact where the key is known; equal to the model when no key helps; every column inside the guarded range for query
rows 50 times outside the fit range), and the stack step ran on FIT-SEL with stand-in columns.
