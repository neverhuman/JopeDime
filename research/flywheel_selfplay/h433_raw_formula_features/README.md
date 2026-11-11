# H433 raw_formula_features: large gains on a few tables, the pre-registered primary just misses (closeout)

**In one line.** Formula features discovered on raw columns lift a handful of TRAIN P201b tables by a lot (the mean
u of the stack rises from 2.055 to 2.223), but the pre-registered clipped test narrowly fails: clipped family-mean
gain +0.0129 u, one-sided sign-flip p 0.0261 against a 0.025 threshold, 66 families won / 83 tied / 52 lost. On top
of H430's key columns the same four columns pass (clipped +0.0144 u, p 0.011). The capacity-only control adds
nothing. DEV was not read.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). `u` is −ln(RMSE / RMSE of the mean
predictor); +0.01 u is about 1% lower RMSE. Written 2026-11-11 by `cl_fwriter` (Claude, model `claude-fable-5-2`).

## 1. Verdicts
| Stage | Verdict |
| --- | --- |
| A. FIT-SEL design check | No harm; continued. (No benefit verdict by design) |
| B1. P201b replication | `REP_NULL`: every condition of the primary test holds except the p-value (0.0261, needs < 0.025) |
| B2. Structure or capacity | `UNDECIDED` by the pre-registered rule, because the formula arm is not confirmed. Descriptively the formula arm's clipped gain is +0.0129 u and the capacity control's is +0.0008 u |
| B3. Increment over H430's key columns | `INCREMENT_CONFIRMED` |
| DEV | Not read (left to one later joint pre-registered wave) |

## 2. Results
Arms: BASE = the stored stack; F4 = stored 64 candidates + `form_gbm`, `form_gam`, `form_sr`, `form_interp`; G1 = +
a plain LightGBM column (control); C1 = + a capacity-only column (3,000-round LightGBM averaged with full-row kernel
ridge; control). "On keys" rows add H430's `key_direct`, `key_gbm`, `factor_glm` to every arm, BASE included.

| Panel | Comparison | Mean gain (u) | Clipped mean (u) | Sign-flip p | Families W / T / L | ΔElo [95% CI] | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P201b, 201 | **F4 vs BASE (primary)** | +0.1687 | **+0.0129** | **0.0261** | 66 / 83 / 52 | +24.7 [−10.4, +60.1] | **no** |
| P201b, 201 | C1 vs BASE | +0.0008 | +0.0008 | 0.094 | 49 / 112 / 40 | +20.8 [−8.3, +50.4] | no |
| P201b, 201 | G1 vs BASE | +0.0003 | +0.0003 | 0.231 | 55 / 111 / 35 | +29.0 [−0.4, +58.7] | no |
| P201b, 201 | F4 vs C1 | +0.1678 | +0.0121 | 0.041 | 72 / 60 / 69 | +3.9 [−38.1, +45.6] | no |
| P201b on keys, 201 | **F4 vs BASE (increment)** | +0.1712 | **+0.0144** | **0.0108** | 58 / 94 / 49 | +16.0 [−18.2, +51.2] | **yes** |
| P201b on keys, 201 | F4 vs C1 | +0.1706 | +0.0138 | 0.013 | 71 / 63 / 67 | +6.5 [−34.7, +47.4] | yes |
| P201b without the ten probed tables, 191 | F4 vs BASE | +0.0750 | +0.0085 | 0.076 | 61 / 82 / 48 | +23.7 [−11.8, +59.6] | no |
| FIT-SEL, 106 (design check) | F4 vs BASE | +0.0021 | +0.0021 | 0.0045 | 38 / 45 / 23 | +59.4 [+9.0, +109.6] | yes |
| FIT-SEL, 106 | C1 vs BASE | +0.0018 | +0.0018 | 0.385 | 25 / 61 / 20 | +11.5 [−28.3, +52.0] | no |
| FIT-SEL on keys, 106 | F4 vs BASE | +0.0024 | +0.0024 | 0.0030 | 37 / 46 / 23 | +44.5 [−4.1, +93.2] | yes |

- **Columns computed:** 201 of 201 on P201b and 106 of 106 on FIT-SEL; no crash, no timeout, no amendment.
- **Replay check:** with no added column the stack step reproduces every stored value.
- **Why the mean and the clipped mean differ so much.** The test clips each family's gain at ±0.5 u. Seven P201b
  tables gain more than 0.1 u and four gain more than 2 u; each counts as at most 0.5.

### The tables that move most (P201b)
| Context | Stored stack | With formula columns | Kumo L | Feature found | Column used |
| --- | --- | --- | --- | --- | --- |
| `26abcd69a5…` | 1.85 | 14.91 | 4.21 | `x7 / x6` | `form_interp` |
| `f44f18402c…` | 15.72 | 27.63 | 6.10 | `x3 − x5` | `form_sr` |
| `1d1fcc386c…` | 2.40 | 8.94 | 5.10 | `x13 / x11` | `form_interp` |
| `b684716022…` | 3.21 | 5.59 | 3.59 | `x12 / x2` | `form_interp`, `form_sr` |
| `3902f96821…` | 5.82 | 7.03 | 6.35 | `x17 / x16` | `form_interp` |
| `7d3e30f33f…` | 17.05 | 15.09 | 6.09 | `1 / x22` | `form_interp` at weight 0.03 (the one loss above 0.1 u) |

`26abcd69a5…` and `1d1fcc386c…` are two of the ten probed tables. The second was the one the probe could not explain.

### Census (P201b)
- 36 of 201 tables are flagged `F`: a feature survives the screen in at least 4 of 5 folds with gain ≥ 0.3 u and
  score ≥ 1.0 u. Best features: 10 ratios, 10 products, 10 sums, 3 differences, 3 log-ratios or logs.
- Those 36 tables carry 27.8% of the positive Kumo L − SXB gap. The formula arm closes 16.0% of that part.
- The stack moves on unflagged tables too: mean absolute change 0.043 u (predicted ≤ 0.003).

## 3. Against Kumo L on P201b (stored reference rows; descriptive, not a pre-registered test)
| Our arm | Mean of ours − Kumo (u) | Clipped mean (u) | Families we win | Families Kumo wins |
| --- | --- | --- | --- | --- |
| Stored stack | −0.089 | −0.146 | 35 | 166 |
| + H430 key columns | −0.011 | −0.144 | 38 | 163 |
| + key columns + formula columns | +0.161 | −0.122 | 42 | 159 |

The two structure waves turn the mean gap around, because a few tables go from far behind to far ahead. They barely
change who wins the typical table: Kumo L still wins 159 of 201 families. Elo counts family wins, so the Elo gap to
Kumo L stays almost where it was. The broad gap is not a structure problem.

## 4. What this means, and what it does not
- **Formula discovery on raw columns works where the structure exists**, and nothing in the stored pool or in a
  much larger tree and kernel model finds it. That is evidence for structure over capacity, though the pre-registered
  rule for that verdict was not met.
- **The pre-registered primary failed, narrowly.** The effect is concentrated in few tables, which the clipped test
  discounts by design. I am reporting `REP_NULL`, not rounding 0.0261 down.
- **The evidence is development-grade.** P201b motivated the design; without the ten probed tables the clipped gain
  is +0.0085 u with p 0.076. FIT-SEL agrees in sign with a small effect (+0.0021 u, p 0.0045).
- **The two structure waves add up.** With the key columns already in the stack, the formula columns pass.

## 5. Deviations from the design document
Stated in PREREG section 4 before any real table was read: forward selection replaced thresholded least squares, and
interpolation on raw feature values replaced 128 bin means. FIT-SEL replaced P201e as the design panel.

## 6. Next
1. A follow-up to H430: key lookup with a model backoff and guarded columns.
2. One joint DEV wave with `cl_apex` and the Gemini lane for the surviving columns of H430, H433 and the Gemini
   column wave.
3. A Rust follow-up specification: a stage mode that runs before scaling, so ratios of raw columns are reachable.

## 7. Evidence (xbabe2)
- Columns: `~/jope-data/h433/pub/cols_FITSEL`, `~/jope-data/h433/pub/cols_P201B` (H424 column format; per-context
  JSON files list the surviving features and the sparse formula of each fold).
- Results: `~/jope_plan/evidence/H433/results/` (apply, readout, census files, with `SHA256SUMS`).
- Logs, review receipts, claim: `~/jope_plan/evidence/H433/`, `~/jope_plan/ns/receipts/H433_claim.txt`.
- Stage: `~/apex-stage/h433/pub-r1`. Compute: about 4.4 CPU-hours of column fitting on 8 processes.
