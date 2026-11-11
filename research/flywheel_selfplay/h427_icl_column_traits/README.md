# H427 — where the in-context column earns its weight (analysis only; no verdict; closeout)

**In one line.** The stack uses the model's column where the column's own out-of-fold fit is close to the best GP
candidate, and that is on tables with **at most 30 features**; on the 35 `DEV` tables with more than 30 features the
column gets almost no weight and adds nothing. Most of the rest is noise at this sample size.
Stored H424 / H425 files only; 115 `DEV` families once; no model, GPU or table; `CONFIRM` and dev91 untouched.
**These are leads for the next training run, not findings.**

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json).

## How much is noise (ridge on eight pre-deployment traits, cluster-grouped cross-validation, 2,000 shuffles)
| Outcome | Out-of-fold R² | Shuffle 95th percentile | Permutation p | By the PREREG rule |
| --- | --- | --- | --- | --- |
| Column weight | −0.03 | +0.03 | 0.61 | **noise** |
| Gain | +0.05 | +0.02 | 0.005 | signal, small |
| Column weight, with the hindsight trait added | −0.01 | +0.03 | 0.36 | noise |
| Gain, with the hindsight trait added | −0.57 | +0.02 | 1.00 | noise (unstable) |

- **Plainly:** the traits predict about 5% of the variance of the gain out of fold and none of the weight in a
  linear model. About 95% of what differs between families is not explained by these traits.
- The weight result looks odd next to the next table: weight follows one trait strongly by rank (ρ +0.70) but not
  linearly; that trait has a long lower tail, and the pre-declared linear model fails on it.

## Per trait (Spearman ρ with cluster-bootstrap 95% interval; about twenty looks, uncorrected)
| Trait | ρ with weight | ρ with gain | Reading |
| --- | --- | --- | --- |
| **Column alone vs best GP candidate, out-of-fold (before deployment)** | **+0.70 [+0.58, +0.79]** | **+0.42 [+0.25, +0.57]** | the main trait |
| Column alone vs stack on AUDIT rows (hindsight) | +0.58 [+0.43, +0.70] | +0.55 [+0.41, +0.67] | same thing seen afterwards |
| **log features** | **−0.23 [−0.39, −0.06]** | **−0.23 [−0.38, −0.05]** | wider tables: less weight, less gain |
| Low-cardinality share | −0.18 [−0.37, −0.01] | −0.14 [−0.33, +0.02] | weak; interval for weight barely clears 0 |
| Target skew | −0.01 [−0.21, +0.18] | +0.15 [−0.04, +0.34] | not separable from noise |
| How well the GP candidates fit | −0.15 [−0.34, +0.04] | +0.08 [−0.10, +0.26] | not separable from noise |
| Copy flag (H425) | −0.06 [−0.25, +0.15] | −0.08 [−0.20, +0.19] | not separable from noise |
| log rows | +0.05 [−0.17, +0.18] | −0.01 [−0.17, +0.18] | almost no variation (800 to 1,653 rows) |
| Missingness | — | — | **no variation: no `DEV` table has a missing cell** |

## Where the column is weak
| Feature columns | Families | Mean weight | Mean gain (u) | Wins / losses | Column alone vs best GP candidate (out-of-fold u) |
| --- | --- | --- | --- | --- | --- |
| ≤ 10 | 39 | 0.111 | +0.0041 | 22 / 5 | −0.55 |
| 11–30 | 41 | 0.171 | +0.0221 | 26 / 6 | −0.34 |
| **31–100** | **26** | **0.029** | **0.0000** | **7 / 8** | **−1.74** |
| **> 100** | **9** | **0.032** | **−0.0008** | **4 / 3** | **−0.71** |

| Column alone vs best GP candidate | Families | Mean weight | Mean gain (u) | Wins / losses |
| --- | --- | --- | --- | --- |
| lowest third (about 2.2 u behind) | 39 | 0.020 | +0.0031 | 8 / 10 |
| middle third (about 0.09 u behind) | 38 | 0.084 | +0.0029 | 22 / 8 |
| highest third (level or ahead) | 38 | 0.222 | +0.0218 | 29 / 4 |

- **Why wide tables are the weak bin (from the training code and the stored monitor, not from this analysis):**
  training draws 3–12 features half the time, 13–32 for 30%, 33–100 for 20%, and never more than 100; a wide
  batch holds 5 to 8 tasks against about 39 for a narrow one, so wide tables are roughly 6% of the tasks seen.
  On the `FIT-SEL` monitor at hour 18 the model's margin over ridge is +0.52 u at ≤ 30 features, +0.13 at 31–100
  and +0.01 above 100.
- **A deployment lead:** the main trait is known before deployment. In the lowest third the column wins 8 and loses
  10; skipping inference there would save a third of the CPU time at no visible cost. Not tested.

## Negatives
- The pre-declared linear model does not predict the column's weight out of fold (noise by its own rule).
- Adding the hindsight trait made the gain model much worse out of fold: gains are heavy-tailed (five families
  above 0.05 u) and a regression on them is fragile.
- Missingness and row count cannot be analysed on `DEV`: they do not vary.
- Target skew, GP fit quality and the copy flag show nothing separable from noise.
- "Categorical share" is a stand-in (columns with at most 10 distinct values); no column type is stored.

## Prompts for the next H419 work (pre-registerable as they stand; none started)
**A. Wide tables, inference only (cheapest; do first).**
- *Mechanism:* the model is strong up to about 30 features and was never trained above 100; a wide table can be
  shown to it as several narrow ones.
- *Prompt:* "On the stored `FIT-SEL` monitor bank, with one pinned 24M checkpoint: for tables with more than 30
  features, average the residual-of-ridge readout over 8 seeded random subsets of 30 features; tables with at
  most 30 features keep the monitored readout. Compare with the monitored readout on the same checkpoint."
- *Baseline:* the monitored readout. *Budget:* inference only, at most 1 GPU-hour on xbabe4, no training run
  touched.
- *Primary:* paired clipped u difference on the 33 `FIT-SEL` contexts with more than 30 features, one-sided
  sign-flip p < 0.025. *Success:* gain above 0 at that level; the bagged readout is then used for wide tables in
  the next stack-column test. *Failure:* anything else; recorded; the number of subsets and their size are not
  tuned afterwards.

**B. Wide tables, training mix.**
- *Mechanism:* as above; the model sees too few wide tasks.
- *Prompt:* "Two 24M arms from the final `ICL_M_r3` checkpoint, 8 training hours each on one GPU. Control: the
  current feature-count draw (50% 3–12, 30% 13–32, 20% 33–100). Treatment: 30% 3–12, 30% 13–32, 30% 33–100, 10%
  101–200; everything else equal."
- *Baseline:* the control arm, not the starting checkpoint, so that extra training time is not credited to the mix.
  *Budget:* 16 GPU-hours (two xbabe4 arms under the memory supervisor, or the 4090 after the 46-hour run ends);
  stop an arm that writes no checkpoint for 40 minutes.
- *Primary:* `FIT-SEL` monitor, residual-of-ridge u on the 33 contexts with more than 30 features, treatment minus
  control, one-sided sign-flip p < 0.025. *Guard:* on the 73 narrower contexts the treatment is at most 0.02 u
  below the control on average. *Success:* primary and guard hold; the mix becomes the default for the next long
  run and the `DEV` column test is repeated once on its 35 wide tables. *Failure:* otherwise; the current mix stays.

**C. Categorical-like columns in the synthetic prior (rests on a weak lead).**
- *Honest status:* for weight the interval barely clears 0 (ρ −0.18 [−0.37, −0.01]); for gain it does not. This
  analysis could not separate the lead from noise.
- *Mechanism:* the synthetic prior quantises features but keeps them ordered; real category codes are unordered.
- *Prompt:* "Add a synthetic task kind in which 30–100% of the features are integer codes with 2 to 10 levels whose
  effect on the target is a random per-level value, with pairwise level interactions; give it 15% of the
  synthetic tasks. Third arm beside prompt B's two (8 GPU-hours)."
- *Baseline:* prompt B's control arm. *Primary:* `FIT-SEL` monitor contexts in the top third of low-cardinality
  share (computed from the monitor bank before the run), treatment minus control, one-sided p < 0.025.
  *Success:* primary holds and the other contexts are at most 0.02 u worse. *Failure:* otherwise; dropped.

## Checks (5 of 5 pass)
PREREG (`3c66b97670`) older than the output; inputs are the stored H424 value file and bank and the published H425
list; the sandbox view still holds no `CONFIRM` table; 86 seconds on 8 threads; published stage.

Evidence (xbabe2): `~/jope_plan/evidence/H427/results/H427_DEV.analysis.{json,txt}`, `H427.gates.json`.
