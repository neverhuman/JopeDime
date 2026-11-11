# H437 PREREG — how many folds of rows the weight screen needs (FIT-SEL, stored values, nothing fitted)

**Status.** Published on `origin/v1/next` before any output of this wave exists. Claim:
`~/jope_plan/ns/receipts/H437_claim.txt` (2026-11-11T17:20Z). H436 belongs to another session and is not touched.

**Authority.** User instruction of 2026-11-11 after the H435 closeout: prompt A of H435 as H437; two fixed fold
counts plus the one-fold baseline (H435 primary: accuracy 0.783, false negatives per decision 0.090, saved 0.665);
stored hour-12 `FIT-SEL` values only, no new fits; TRAIN only, no `CONFIRM`, no AUDIT, no dev91; success = a fold
count with false negatives per decision ≤ 0.05 (upper interval end ≤ 0.09) that keeps saved ≥ 0.60; failure = no
count beats one fold; CPU at most 4 threads, at most 30 minutes; negative findings recorded; prompts B and C stay
queued; the read-only apply to H433's columns is left for Jepson.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not staged, opened or
read. `DEV` is not read. No AUDIT array is loaded. No table is opened, no model pass is made, **no weight is
fitted** (the weight builder is not imported).

## 1. Question
H435: on one fold of rows the weight rule misses too many real-weight columns (9.0% of decisions); on all rows it
is accurate (3.3%). How many folds does it take to get under 5%, and what is left of the saving?

## 2. Decisions and inputs (pinned by sha256; the gates check them)
- **The same 212 decisions as H435's primary:** the two hour-12 in-context columns (residual and plain readout) on
  the 106 `FIT-SEL` tables. Columns: H419's `oof_M_r3_h012.npz` (`199754684f58…`), rows of the monitor bank
  (`8e517921b7d5…`). Stack: the stored `FIT-SEL` matrices of H420 `TBOTH`, SEARCH-side arrays only, with the stored
  weights of each record.
- **Realised weight and cost per table: H435's stored per-table values** (`024cdf09aa08…`); "real" = weight ≥ 0.10.
- Clusters: the stored H425 rehearsal screen of `FIT-SEL` (`d1c91046c567…`), 97 clusters.

## 3. The rule and the fold counts
- The rule is H434's, as H435 applied it: overlap and RMSE ratio of the column against the stored blend on a set
  of rows; keep the table when the predicted weight is ≥ 0.10. Nothing is tuned.
- **Rows used: the stored folds 0 … k − 1.** Baseline k = 1 (H435's primary, reproduced exactly as a check).
  **Tested: k = 2 and k = 3.** k = 4 and k = 5 (all rows) are reported without a verdict.
- **Cost, as in H435:** k screening passes over SEARCH rows on every table, counted as extra work, plus the full
  six passes on the tables kept. Saved = 1 − that ÷ the cost of always running. One screening pass costs 11.7% of
  always-run inference on this cohort.

## 4. Test
For each tested k: false negatives ÷ decisions with its 95% cluster-bootstrap interval (10,000 resamples, H435's
seed), accuracy, and saved inference.

| Verdict | Condition |
| --- | --- |
| `FOLDS_PASS` | for k = 2 or k = 3: false negatives ÷ decisions ≤ 0.05, **and** the upper end of its interval ≤ 0.09, **and** saved ≥ 0.60. The smaller passing k is recommended |
| `FOLDS_FAIL` | neither k has fewer false negatives than one fold (19) |
| `FOLDS_BETTER_NOT_ENOUGH` | some k has fewer false negatives than one fold, but none meets all three conditions |

- A pass would make the k-fold screen the proposed first step before a new column's full inference; it would
  still need one prospective use on a new column. Anything else: the screen is not used; recorded.

## 5. Reported beside the verdict (no verdict of their own)
- k = 1 … 5: the four counts, accuracy, false negatives, saved inference, with intervals; the paired difference in
  false negatives against one fold with its interval.
- The saving if the screening passes are reused on kept tables (H435's second accounting).
- Each column apart for k = 1, 2, 3.
- Fold-choice spread: the same figures for every choice of k folds out of five (minimum, median, maximum).
- Replay: k = 1 must reproduce H435's primary counts, saving and intervals; k = 5 must reproduce H435's all-row
  counts.

## 6. Budget
One thread at the lowest priority (nice 19); seconds for the sums, about a minute for the bootstrap; stopped at
25 minutes. No GPU; no model library; no weight builder.

## 7. Gates (`gates_h437.sh`, after the per-fold sums, before the readout)
This PREREG is older than the sums file and no readout exists; the five pinned inputs; 106 tables, with the fold-0
and all-row estimates equal to H435's stored ones (difference ≤ 1e-9); one thread, no model library and no weight
builder loaded; the 30-minute cap; the sandbox view still holds no `CONFIRM` table; published stage.

## 8. Expected, stated in advance
- **The saving condition is nearly out of reach by arithmetic.** Each added screening pass costs 11.7% of
  always-run inference. Even if the k-fold screen kept exactly what the all-row rule keeps (14.9% of the cost),
  saved would be 0.62 at k = 2 and 0.50 at k = 3. **So k = 3 cannot pass, whatever its false negatives;** k = 2
  passes on saving only if it keeps almost as few tables as the all-row rule (one fold keeps 21.9% of the cost).
- False negatives: I expect about 6–7% at k = 2 and about 5% at k = 3 (all rows: 3.3%).
- **Expected verdict: `FOLDS_BETTER_NOT_ENOUGH`.**
- Under the reuse accounting (reported), k = 2 would save about 0.63.

## 9. What was looked at before this PREREG
H435's readout (one fold, half a fold, all rows; its per-table costs, from which the 11.7% and the two ceilings
above are computed). A self-test of the code on **random numbers only**. **No estimate on two, three or four folds
has been computed for any table.**

## 10. Weaknesses
- The same 212 decisions that H435 used: this is a second look at one data set, not a new test. A passing k would
  have been chosen among two on these data.
- The saving is H435's per-pass time model, not a measurement, and the verdict depends on the accounting
  (extra passes against reused passes); both are reported, the first decides.
- The first k folds are one choice of rows; the fold-choice spread shows how much that choice matters.
- Hour-12 columns of one model family on its monitor cohort; weight, not gain.

## 11. Files
`folds_h437.py`, `gates_h437.sh`, `stage_h437.sh`. Reused at published bytes: `restate_h426.py` (clusters).
