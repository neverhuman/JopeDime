# H430 exact_key_encoders: pre-registration

- **Written:** 2026-11-11, before any H430 column was computed on a real table, by `cl_fwriter` (Claude, model
  `claude-fable-5-2`, screen `agy_research_wave` on xbabe2).
- **Authority:** the session user's instruction of 2026-11-11 about 15:15Z to run and test the five CW designs as
  incremental research waves, in close contact with the Gemini lane. Claim receipt
  `~/jope_plan/ns/receipts/H430_claim.txt`.
- **Design document:** `/home/ubuntu/jope_plan/waves_claude_20261111/CW1_KEYS.md`. Where this file differs, this file
  binds.

`u` is −ln(RMSE / RMSE of the mean predictor). +0.01 u is about 1% lower RMSE.

## 1. Question
Our stack treats every column as a scaled continuous number and has no encoder that uses a column's exact values.
Does adding exact-value target encodings, as fitted stack columns, raise the stack?

## 2. Hypotheses
- **H1 (prevalence).** At least 10 of the 201 TRAIN P201b tables are key-determined (rule in section 5), and those
  tables carry at least 20% of the positive Kumo L − SXB gap on that board.
- **H2 (benefit).** The stored stack plus the three H430 columns beats the stored stack on P201b under the H335
  primary test, and beats the same stack with a plain tree column instead.
- **H3 (transfer).** The same holds on the APEX DEV cohort against H424's arm `I`. This is tested only if `cl_apex`
  agrees to one DEV read.

## 3. What motivated the design, and what that does to the P201b reading
- The design came from a read-only probe of ten P201b tables with the largest Kumo gaps
  (`/home/ubuntu/jope_plan/waves_claude_20261111/EVIDENCE_structure_probe.md`). Three of them are key tables.
- So P201b is not blind for those ten tables. Every P201b result is reported twice: on all 201 contexts, and without
  the ten probed contexts (`3b0303bb3885`, `791426065964`, `589eab8b1034`, `fd23dd59317e`, `9fcd56c6e2c2`,
  `acdd9d992987`, `26abcd69a561`, `51e71f0b3089`, `cc7e569a6f61`, `1d1fcc386cce`).
- P201b also shares 185 families with P201e, the design panel of the stored stack. It is a development panel.
- FIT-SEL and DEV were not looked at in designing the encoders.

## 4. Method (fixed by the published code)
All fitting uses SEARCH rows only, inside the stored 5-fold labels. Constants are in `keys_core.py`.

| Piece | Rule |
| --- | --- |
| Eligible key | A column, or a pair among the 12 best single keys, with at most n/2 levels and at least half of the fit rows in levels that occur twice or more |
| Encoding | Shrunk group mean `(S_g + m·μ)/(n_g + m)`, `m` from {0, 0.5, 2, 8} by leave-one-out RMSE |
| Unseen level | Linear interpolation between neighbouring key values when that beats the mean under group-level leave-out; for a pair, the better single key's encoding; otherwise the fit mean |
| Pair kept | Only if its leave-one-out u beats both of its columns by 0.05; at most 30 pairs |
| `key_direct` | The best key's encoding when its leave-one-out u is at least 1.0; otherwise ridge on standardised features |
| `key_gbm` | LightGBM 4.6.0 (600 rounds at most, learning rate 0.05, 31 leaves, 5 rows per leaf, row and column sampling 0.8, rounds from a 15% inner holdout) on the features plus the nested out-of-fold encoding and group size of the 8 best keys |
| `factor_glm` | Ridge (generalised cross-validation) on one-hot columns of at most 20 levels, up to 6 factor crosses, and restricted cubic splines of the other columns; link from identity, shifted log, Yeo-Johnson by inner 3-fold RMSE; smearing correction |
| `gbm_plain` (control) | The same LightGBM without encodings |
| `key_noise` (control) | The same as `key_gbm` with encodings built from a permuted target |

Per context the columns are written in the H424 column format: out-of-fold SEARCH prediction, bagged AUDIT
prediction, refit AUDIT prediction (`cols_h430.py`). The stack step (`stack_h430.py`) adds columns to the stored 64
candidates and rebuilds the weights with the stored builder (`meta_h416` BCAR through `cache_h422.bcar`), with the
stored mix of bag and refit predictions.

| Arm | Candidates |
| --- | --- |
| BASE | The stored stack value (P201b, FIT-SEL). On DEV: the stored 64 plus H424's in-context column, rebuilt |
| K3 (primary) | BASE's candidates + `key_direct`, `key_gbm`, `factor_glm` |
| G1 (control) | BASE's candidates + `gbm_plain` |
| N1 (control) | BASE's candidates + `key_noise` |

## 5. Stages, tests and verdicts
| Stage | Data | Rule | Verdicts |
| --- | --- | --- | --- |
| A. Design check | FIT-SEL, 106 contexts (bank `/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz`, matrices `~/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL`, tag `TBOTH`) | No verdict on benefit. If K3 vs BASE is harmful (clipped mean below 0 with mirror sign-flip p < 0.025), the wave stops and P201b is not run | `KEYS_HARM_DESIGN` or continue |
| B0. Census | P201b column records (SEARCH rows only) | `K`: the best key reached leave-one-out u ≥ 1.0 in at least 4 of 5 folds and the out-of-fold u of `key_direct` is ≥ 2.0. `C`: at least half the columns have at most 20 levels and `factor_glm`'s out-of-fold u is ≥ 0.3 above `gbm_plain`'s (and not `K`) | `KEYS_RARE` (fewer than 10 `K` tables: no further claim), `CENSUS_CONCENTRATED` (`K` tables carry ≥ 20% of the positive Kumo L − SXB gap), `CENSUS_DIFFUSE` |
| B1. Replication, primary | P201b, 201 contexts (bank built by `bank_p201b.py` from the TRAIN board envelopes; matrices `~/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM`, tag `P201B`) | K3 vs BASE: `h335_stats.primary_test` (gains clipped at ±0.5; clipped family mean > 0; one-sided sign-flip p < 0.025; leave-one-family-out minimum > 0; family median ≥ 0), and K3's clipped mean above G1's | `REP_CONFIRMED`, `REP_NULL`, `REP_HARM` |
| B2. Secondaries | P201b | Fraction of the positive Kumo L − SXB gap closed on `K` tables (prediction: ≥ 50%); mean absolute change of the stack on unflagged tables (prediction: ≤ 0.003 u); N1 vs BASE (prediction: no gain); all of B1 without the ten probed contexts | reported |
| C. DEV, one read | 115 core contexts in 110 clusters; only after `cl_apex` agrees | K3 vs BASE with H426 clusters as units, same test | `KEYS_CONFIRMED`, `KEYS_NULL`, `KEYS_HARM` |

Kumo L and TabPFN 3.5 values are the stored TRAIN-board reference rows (FMGAP Part A). They are never stack members.

## 6. Budget and stop rules
- xbabe2, CPU only, at most 8 single-thread processes at nice 19, at most 12 CPU-hours in total, no GPU.
- A context that exceeds 900 s, or raises an error, is written as not ok and scores BASE in every arm. If more than
  5% of a panel's contexts are not ok, the stage is reported as incomplete and carries no verdict.
- Code changes after this file is published are limited to fixing a crash or a non-finite output; each one is
  published as an amendment before the affected stage is rerun, and no finished context is recomputed.
- Nothing is rerun after a verdict.

## 7. Not in this wave
- No CONFIRM, dev91, DL-Val or DL-Holdout read. No DEV read without `cl_apex`'s agreement.
- No change to the stored stack, the search grammar, the Rust worker or any other lane's files.
- The two-column arithmetic features found by the same probe belong to the next wave (CW-2).

## 8. Files
`keys_core.py`, `cols_h430.py`, `stack_h430.py`, `census_h430.py`, `bank_p201b.py`, `test_keys_core.py`,
`smoke_h430.py`. Their sha256 values are in the stage's `CODE.sha256`. Before publication: `test_keys_core.py` passes
7 of 7 on synthetic tables, `smoke_h430.py` passes, and the stack step with no added column reproduces the stored
P201b weights exactly on 201 of 201 contexts (largest weight difference 0, largest u difference 4e-10).
