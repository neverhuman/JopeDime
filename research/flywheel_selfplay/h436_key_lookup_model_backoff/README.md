# H436 key_lookup_model_backoff: the fix works on its target table but does not beat H430's lookup, and is not safe (closeout)

**In one line.** A key lookup that falls back to a tree model fixes the two cases it was built for (the largest-gap
key table goes from u 4.73 to 15.11, and the catastrophic context is gone), but on TRAIN P201b it does not beat
H430's lookup under the pre-registered test (clipped family-mean difference −0.0062 u, p 0.82), and the safety rule
fails: one near-exact table loses 6.9 u when the new column gets 3% weight. DEV was not read.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). `u` is −ln(RMSE / RMSE of the mean
predictor); +0.01 u is about 1% lower RMSE. Written 2026-11-11 by `cl_fwriter` (Claude, model `claude-fable-5-2`).

## 1. Verdicts
| Stage | Verdict |
| --- | --- |
| A. FIT-SEL design check | No harm; continued |
| B1. P201b primary, KH vs KD | `FIX_NULL` |
| B2. Safety | `UNSAFE` |
| DEV | Not read |

## 2. Results
Arms: B = the stored stack; BASE = stored 64 candidates + H430's `key_gbm` and `factor_glm`; KH = BASE +
`key_hybrid`; KD = BASE + H430's `key_direct` (H430's arm); KHR = KH + `ridge_z`; G1 = BASE + the backoff model
alone.

| Panel | Comparison | Mean gain (u) | Clipped mean (u) | Sign-flip p | Families W / T / L | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| P201b, 201 | **KH vs KD (primary)** | +0.0588 | **−0.0062** | **0.821** | 62 / 68 / 71 | **no** |
| P201b, 201 | KH vs B | +0.1378 | +0.0103 | 0.0317 | 108 / 34 / 59 | no |
| P201b, 201 | KD vs B (H430's result, recomputed here) | +0.0790 | +0.0156 | 0.0188 | 106 / 28 / 67 | yes |
| P201b, 201 | KHR vs KH | +0.0119 | +0.0063 | 0.0335 | 33 / 145 / 23 | no |
| P201b, 201 | G1 vs BASE | −0.0006 | −0.0006 | 0.758 | 43 / 123 / 35 | no |
| P201b without the ten probed tables, 191 | KH vs KD | +0.0110 | −0.0059 | 0.858 | 57 / 67 / 67 | no |
| FIT-SEL, 106 (design check) | KH vs KD | +0.0015 | +0.0015 | 0.0115 | 29 / 48 / 29 | yes |
| FIT-SEL, 106 | KH vs B | +0.2075 | +0.0107 | 0.00017 | 49 / 24 / 33 | yes |

- **Columns computed:** 201 of 201 on P201b, 106 of 106 on FIT-SEL; no crash, no amendment.
- The positive mean in the primary comes from one context (`d8134cd42a…`, where KD is −8.42). Without it the mean
  difference is +0.006 u and the clipped mean −0.0087 u.

### Tables named in the PREREG, and the ones that decide the result (P201b)
| Context | B | KD (H430) | KH | KHR | Kumo L | What happened |
| --- | --- | --- | --- | --- | --- | --- |
| `3b0303bb38…` | 2.47 | 4.73 | **15.11** | 15.11 | 7.63 | The fix works: out-of-fold u of the lookup rises from 2.70 to 4.10 and the builder gives it weight 0.67 |
| `d8134cd42a…` | 2.23 | −8.42 | 2.23 | 2.23 | 3.29 | The unguarded ridge is gone from the lookup column |
| `7914260659…` | 2.36 | 3.12 | 3.30 | 3.30 | 7.32 | Small gain |
| `7d3e30f33f…` | 17.05 | 16.11 | **10.16** | 10.16 | 6.09 | A near-exact table: `key_hybrid` alone scores 6.71, gets weight 0.03, and that is enough to cost 6.9 u |
| `b7ad2eabb5…` | 2.44 | 3.70 | 2.42 | 3.36 | 0.77 | H430's winner here was its ridge fallback; `ridge_z` brings most of it back |
| `f44f18402c…` | 15.72 | 16.96 | 15.78 | 16.96 | 6.10 | The same; `ridge_z` brings all of it back |
| `fd23dd5931…` | 4.05 | 4.61 | 4.06 | 4.06 | 6.03 | The refit chose a pair key (`x88`, `x89`) where every fold chose `x88`; the pair has more unknown values on AUDIT |
| `589eab8b10…` | 2.51 | 4.19 | 3.87 | 3.87 | 7.37 | Lower out-of-fold score than H430's lookup |

### Safety (pre-registered rule: smallest KH − B above −0.25 u; no column alone below −1 u)
| Panel | Smallest KH − B | Columns alone below −1 u |
| --- | --- | --- |
| P201b | −6.89 (`7d3e30f33f…`) | 1 (`key_hybrid` on `a4020eadaa…`, −1.25 alone; weight 0, stack unchanged) |
| FIT-SEL | −0.01 | 0 |

### Kumo L gap on H430's 16 key tables (P201b)
Fraction of the positive Kumo L − SXB gap closed: H430's arm 27.8%, KH 40.7%, KHR 40.6% (prediction ≥ 50%).

## 3. What was learned
1. **The model backoff does what it was designed to do.** On the table that motivated it, the lookup is now scored
   fairly out of fold and the stack doubles Kumo L's u. That mechanism is worth keeping.
2. **It is not a net improvement yet.** Three separate things pull the other way:
   - **Pair keys.** I dropped H430's rule that a pair must beat both of its columns by a margin. The refit then
     picks pairs the folds did not pick, with more unknown values on AUDIT (`fd23dd5931…`).
   - **The ridge fallback was a real winner on a few linear tables.** Splitting it out as `ridge_z` recovers them
     (KHR), and after clipping the ridge column is never below −1 u alone on either panel.
   - **Precision dilution.** On a table where the stored stack is already near-exact (u 17), a 3% weight on a column
     with u 6.7 costs 6.9 u. H430 (−0.9 u) and H433 (−2.0 u) lose on the same table for the same reason. This is a
     property of the stored builder with any added column, not of one column.
3. **Descriptively, the best arm of the three waves on P201b is KHR**: against the stored stack its mean gain is
   +0.150 u and its clipped mean +0.0165 u, 111 families won and 67 lost. Against H430's arm it is level (clipped
   −0.0015 u, 56 won, 56 lost). This comparison was not pre-registered.

## 4. What this means for the next steps
- **The builder, not another lookup variant, is the bottleneck on key tables.** An admission rule that refuses a
  column whose out-of-fold error is far above the best candidate's would have prevented the `7d3e30f33f…` loss in
  all three waves. The Gemini lane's column wave has such a gate and a peer-median guard; this case is handed to it.
- **Candidates for the joint DEV wave from this lane:** H430's `key_gbm` and `factor_glm`, a guarded lookup
  (`key_hybrid` with the pair margin restored), `ridge_z`, and H433's `form_*` columns, behind an admission gate.
- No further lookup variant will be tuned on P201b: three waves have read it for this idea.

## 5. Evidence (xbabe2)
- Columns: `~/jope-data/h436/pub/cols_FITSEL`, `~/jope-data/h436/pub/cols_P201B` (H424 column format; every column
  clipped to the fit target range widened by half).
- Results: `~/jope_plan/evidence/H436/results/` (apply and readout files, with `SHA256SUMS`).
- Logs, review receipts, claim: `~/jope_plan/evidence/H436/`, `~/jope_plan/ns/receipts/H436_claim.txt`.
- Stage: `~/apex-stage/h436/pub-r1`. Compute: about 2.4 CPU-hours of column fitting on 8 processes.
