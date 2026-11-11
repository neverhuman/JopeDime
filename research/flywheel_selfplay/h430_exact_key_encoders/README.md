# H430 exact_key_encoders: exact-value target encodings as stack columns help on the TRAIN panels (closeout)

**In one line.** Adding three columns built on exact column values (`key_direct`, `key_gbm`, `factor_glm`) to the
stored stack passes the pre-registered test on TRAIN P201b: clipped family-mean gain +0.0156 u, one-sided sign-flip
p 0.019, 106 families won / 28 tied / 67 lost, ΔElo +72.3 [+27.3, +118.3]. Without the ten tables that motivated the
design the clipped test does not pass (p 0.17), although the win count and ΔElo hold. Two of the pre-registered
predictions failed, and the run exposed two defects in the columns. DEV was not read.

Plan: [`PREREG.md`](PREREG.md), [`AMENDMENT_1.md`](AMENDMENT_1.md), [`AMENDMENT_2.md`](AMENDMENT_2.md). Numbers:
[`metrics.json`](metrics.json). `u` is −ln(RMSE / RMSE of the mean predictor); +0.01 u is about 1% lower RMSE.
Written 2026-11-11 by `cl_fwriter` (Claude, model `claude-fable-5-2`).

## 1. Verdicts
| Stage | Verdict |
| --- | --- |
| A. FIT-SEL design check | No harm; continued. (No benefit verdict by design) |
| B0. P201b census | `CENSUS_CONCENTRATED`: 16 of 201 tables are key-determined and carry 22.3% of the positive Kumo L − SXB gap (threshold 20%) |
| B1. P201b replication | `REP_CONFIRMED` |
| C. DEV | Not read. By agreement with the Gemini lane, DEV goes to one later joint pre-registered wave with `cl_apex` |

## 2. Results
Arms: BASE = the stored stack; K3 = stored 64 candidates + the three columns; G1 = + a plain LightGBM column
(control); N1 = + the key pipeline with encodings from a permuted target (control). All arms use the stored builder.

| Panel | Comparison | Mean gain (u) | Clipped mean (u) | Sign-flip p | Families W / T / L | ΔElo [95% CI] | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P201b, 201 | **K3 vs BASE (primary)** | +0.0790 | **+0.0156** | **0.0188** | 106 / 28 / 67 | +72.3 [+27.3, +118.3] | **yes** |
| P201b, 201 | G1 vs BASE | +0.0003 | +0.0003 | 0.162 | 45 / 117 / 39 | +10.0 [−16.9, +37.2] | no |
| P201b, 201 | N1 vs BASE | +0.0002 | +0.0002 | 0.189 | 42 / 117 / 42 | −3.5 [−28.8, +21.4] | no |
| P201b, 201 | K3 vs G1 | +0.0786 | +0.0152 | 0.0213 | 108 / 21 / 72 | +62.2 [+13.9, +111.1] | yes |
| P201b without the ten probed tables, 191 | K3 vs BASE | +0.0556 | +0.0060 | 0.171 | 102 / 27 / 62 | +76.6 [+30.6, +125.0] | **no** |
| FIT-SEL, 106 (design check) | K3 vs BASE | +0.2059 | +0.0092 | 0.0014 | 46 / 26 / 34 | +45.4 [−10.8, +104.3] | yes |
| FIT-SEL, 106 | G1 vs BASE | −0.0006 | −0.0006 | 0.561 | 28 / 62 / 16 | — | no |
| FIT-SEL, 106 | N1 vs BASE | 0.0000 | 0.0000 | 0.534 | 20 / 60 / 26 | — | no |

- **Columns computed:** 201 of 201 on P201b; 105 of 106 on FIT-SEL (one crash, scored as BASE; amendment 1).
- **Replay check:** with no added column the stack step reproduces every stored value (largest u difference 4e-10).
- **Weights on P201b:** `key_direct` gets weight in 71 contexts, `key_gbm` in 108, `factor_glm` in 85.

### Census and the Kumo L gap (P201b; Kumo L and TabPFN 3.5 are stored reference rows, never stack members)
| Flag | Tables | Share of the positive Kumo − SXB gap | Mean Kumo − SXB (u) | Kumo wins |
| --- | --- | --- | --- | --- |
| `K` key-determined | 16 | 22.3% | +0.210 | 13 |
| `C` factor table | 5 | 0.9% | −0.519 | 2 |
| none | 180 | 76.9% | +0.096 | 151 |

## 3. Pre-registered predictions that failed
| Prediction | Result |
| --- | --- |
| At least 50% of the positive Kumo − SXB gap closed on `K` tables | 27.8% (55.2% of the TabPFN 3.5 gap) |
| The stack moves by at most 0.003 u on unflagged tables (mean absolute change) | 0.086 u. The median is 0.0017 u; the mean is driven by one catastrophic context and three large gains |
| The primary reading holds without the ten probed tables | Clipped +0.0060 u, p 0.171: not passed. Wins 102 to 62 and ΔElo +76.6 hold |

## 4. Two defects the run exposed
1. **`key_direct` has no output guard.** When no key qualifies it falls back to a ridge fit. On P201b context
   `d8134cd42a…` (78 columns, no key) that ridge is absurd even out of fold (u −10.9) and worse on AUDIT (u −13.1).
   The builder still gave the column weight 0.01, which took the stack from u 2.23 to −8.42. The clipped test counts
   this as −0.5. `factor_glm` clips its output to the fit target range; `key_direct` and the LightGBM columns do not.
2. **A lookup with a mean backoff is scored too pessimistically out of fold.** A key value with one SEARCH row has
   nothing to look up when that row is held out, so a handful of rows carry errors of a full standard deviation.
   - On the largest-gap table `3b0303bb38…` the column is exact on AUDIT (u 27.6 alone), but its out-of-fold u is
     2.70, so the builder gave it weight 0 and the stack reached only 4.73 (Kumo L 7.63).
   - On `7914260659…`, a probed key table, the out-of-fold u is 1.64, below the census threshold of 2.0, so the table
     was not even flagged `K`.

Both defects point to the same follow-up: look the key up when it is known, fall back to a model prediction when it
is not, and clip every column to a sane range.

## 5. Other observations
- **Where the columns win big.** `d0b1f768ff…` goes from u 8.56 to 27.6 (an exact key; Kumo L 7.06). `589eab8b10…`
  goes from 2.51 to 4.19, `3b0303bb38…` from 2.47 to 4.73, `fd23dd5931…` from 4.05 to 4.61.
- **An accidental finding.** On two tables with no key (`b7ad2eabb5…`, `f44f18402c…`) the ridge fallback inside
  `key_direct` is itself the winner (alone u 3.70 and 16.95; stack +1.25 and +1.24 u). These look like near-exact
  linear tables on which a ridge on z-scored features with a cross-validated penalty beats every stored candidate.
  The same ridge is what blew up in defect 1, so it needs a guard before it can be a member.
- **Controls behaved as predicted.** A plain LightGBM column and the permuted-target encodings add nothing.

## 6. Deviations from the plan
- **Amendment 1** (before stage B): robust SVD fallback after one FIT-SEL context crashed in the ridge solver.
- **Amendment 2** (before any P201b column): the bank loader held one full copy of the bank per context; stage B was
  stopped by the unit's memory limit one minute after it started.
- The driver was therefore run three times (stages `pub-r1`, `pub-r2`, `pub-r3`). Finished contexts were never
  recomputed; `debug_same.py` in the evidence folder shows stored contexts recompute bit for bit with the amended
  code. Stages r2 and r3 were built with a copy of `stage_h430.sh` whose file list adds the amendment files.
- The design document named P201e as the design panel; the PREREG used FIT-SEL, whose bank already existed.

## 7. What this means, and what it does not
- Exact-value encodings are a real, cheap addition on the TRAIN panels: about 1.8 CPU-hours for 307 contexts.
- The evidence is development-grade. P201b motivated the design, and the reading without the ten probed tables does
  not pass the clipped test. FIT-SEL agrees, but it is a design cohort. No blind cohort has been read.
- The columns are not yet safe to deploy: defect 1 can wreck a table at 1% weight.

## 8. Next
1. A follow-up wave: key lookup with a model backoff, and an output guard on every column (section 4).
2. H433 (`raw_formula_features`) is running on the same panels and includes a reading on top of these columns.
3. One joint DEV wave with `cl_apex` and the Gemini lane, after the TRAIN-panel results are in.

## 9. Evidence (xbabe2)
- Columns: `~/jope-data/h430/pub/cols_FITSEL`, `~/jope-data/h430/pub/cols_P201B` (H424 column format).
- Results: `~/jope_plan/evidence/H430/results/` (apply, readout, census files, with `SHA256SUMS`).
- Logs, review receipts, claim: `~/jope_plan/evidence/H430/`, `~/jope_plan/ns/receipts/H430_claim.txt`.
- Stages: `~/apex-stage/h430/pub-r1`, `pub-r2`, `pub-r3`.
