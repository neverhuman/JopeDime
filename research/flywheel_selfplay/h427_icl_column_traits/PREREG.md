# H427 PREREG — where does the in-context column earn its weight? (analysis only; no new verdict)

**Status.** Published on `origin/v1/next` before any trait is related to an outcome. Claim:
`~/jope_plan/ns/receipts/H427_claim.txt` (2026-11-11T10:49Z). **This wave gives no verdict and changes none.** Its
numbers describe 115 `DEV` families once; they are leads for the next training run, not findings.

**Authority.** Relayed by ENG:watcher on 2026-11-11 (stamped 3:50 AM MST): an analysis-only wave from stored H424 /
H425 values; regress per-family gain and column weight on table traits with family-grouped cross-validation, say
how much is noise, and write two or three pre-registerable prompts for the next H419 run; CPU only, at most 8
threads, at most 1 hour; no model forward pass, no GPU, no `CONFIRM`, never dev91; training runs and the hour-24
driver are left alone.

## 1. Inputs (stored files only)
H424's `DEV` inference bank (SEARCH rows and AUDIT features of the 115 contexts; no AUDIT target) and its value file
(first read, hour-20 checkpoint); H425's frozen list and screen output; H426's `feature` clustering. No table is
opened, no model runs.

## 2. Outcomes and traits (per `DEV` context)
- **Outcomes:** `weight` = the stack weight of the model's column; `gain` = clipped gain of the stack with the
  column over the stored stack (u).
- **Traits, from SEARCH rows only:** log10 rows; log10 features; share of feature columns with at most 10
  distinct values (a stand-in for "categorical": the tables store no column type); share of missing cells;
  |skewness| of the target (capped at 10); the column alone against the stack before deployment (out-of-fold u
  of the column minus the best stored candidate's); H425 copy flag (tier `copy` or `possible`).
- **Added trait:** the best stored candidate's out-of-fold u (how well the GP candidates fit the table).
- **Hindsight trait, kept apart:** the column's AUDIT u alone minus the stored stack's. It uses AUDIT outcomes, so
  it explains after the fact and cannot guide a deployment.

## 3. Analysis (fixed here)
- **Regression:** ridge on standardised traits; 5-fold cross-validation grouped by cluster; the penalty of each
  fold from an inner grouped cross-validation over {0.1, 1, 10, 100}. Two models per outcome: `traits` (the eight
  pre-deployment traits) and `hindsight` (plus the hindsight trait).
- **How much is noise:** out-of-fold R² against the fold-training mean, with a permutation null (2,000 shuffles of
  the outcome, same folds, same procedure). A model is called **signal** only if its permutation p is below 0.05;
  otherwise its R² is reported as noise.
- **Per trait:** Spearman correlation with each outcome and a cluster-bootstrap 95% interval (2,000 draws);
  outcome means, share of contexts where the column is used, and wins / losses by tercile.
- Two outcomes, two models, nine traits: about twenty looks. Intervals are not corrected for that; a single trait
  with an interval just clear of 0 is expected by chance.

## 4. What comes out
- A table of the above, a plain statement of how much of the weight and of the gain the traits predict out of
  fold, and the trait bins where the column is weak.
- **Two or three prompts for the next H419 training run** in the queue file and the README, each with mechanism,
  baseline, budget and success / failure criteria, written so they can be pre-registered as they stand. A prompt
  may be motivated by a trait bin only if that trait's interval excludes 0 or its model is signal; otherwise the
  prompt must say it rests on a lead that this analysis could not separate from noise.

## 5. What was already seen (H424, H425)
The column is used in 84 of 115 families; where it beats the stack alone (23 families) the gain is +0.036 u,
elsewhere +0.002 u; the gain is +0.002 u on the 21 copy-flagged contexts and +0.011 u on the others. These are the
hindsight trait and the copy flag; no other trait has been related to an outcome.

## 6. Limits
- 115 contexts (110 clusters), one checkpoint, one readout: weak power for anything but large effects.
- Gains are small and heavy-tailed (five contexts above 0.05 u); a regression on them is fragile.
- "Low-cardinality share" is not a type; integer-coded numeric columns count as categorical-like.
- `DEV` has been read by H420–H426; this is a further look at the same contexts.

## 7. Files
`analyze_h427.py`, `gates_h427.sh`, `stage_h427.sh`.
