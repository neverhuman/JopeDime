# H435 PREREG — a prospective test of the two-number weight rule as a screen before column inference (FIT-SEL only)

**Status.** Published on `origin/v1/next` before any output of this wave exists. Claim:
`~/jope_plan/ns/receipts/H435_claim.txt` (2026-11-11T16:57Z). H430 to H434 are taken; other sessions' folders are
read only.

**Authority.** Relayed by ENG:watcher on 2026-11-11, after the H434 closeout: prompt C of H434 as H435; freeze the
rule and its threshold from H434, apply it on TRAIN-side data to columns it was not fitted on (stored `FIT-SEL`
columns of earlier waves, and finished stored columns of other sessions by read only), with overlap and RMSE ratio
estimated from a small sample of rows per table, and compare predicted weight ≥ 0.10 with the realised stack weight;
baseline = always run full inference; report accuracy, false-negative rate and the CPU saved; success = at most
1 false negative per 20 tables with at least 40% of inference saved. CPU only, at most 4 threads at low priority,
at most 45 minutes, no new inference, no GPU. No gain on AUDIT rows in this wave.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not staged, opened or
read. **`DEV` is not read at all.** No table is opened and no model pass is made. Of the stored `FIT-SEL` stack
matrices only SEARCH-side arrays are loaded (candidates' out-of-fold predictions, SEARCH targets, fold labels);
no AUDIT array is loaded and no gain is computed.

## 1. Question
Before spending full inference on a new stack column, one screening pass per table could give the column's
predictions on one fold of rows. Does H434's rule, applied to those rows only, say correctly which tables will give
the column at least 10% stack weight, and how much inference would skipping the others save?

## 2. The rule (frozen in H434; nothing is fitted or tuned here)
For a column and the stored blend (stored candidates combined with the stored weights of the record), on a given
set of rows: r = RMSE of the column ÷ RMSE of the blend; ρ = uncentred correlation of their out-of-fold errors;
predicted weight w = (1 − ρr) / (1 + r² − 2ρr), clipped to [0, 1]. **The screen keeps a table when w ≥ 0.10.**

## 3. Columns (all computed in earlier waves; pinned by sha256; the gates check them)
| Column | Tables | Source |
| --- | --- | --- |
| **In-context, hour 12, residual-of-ridge readout** | 106 `FIT-SEL` | H419 stack test, `oof_M_r3_h012.npz` (`199754684f58…`), rows of the H419 monitor bank (`8e517921b7d5…`) |
| **In-context, hour 12, plain readout** | 106 `FIT-SEL` | same file |
| In-context, hour 20, residual readout | 8 `FIT-SEL` | H424's code smoke (`apply_smoke1.json`, `8e916f300687…`) |
| Five columns of H430 (`key_direct`, `key_gbm`, `factor_glm`, `gbm_plain`, `key_noise`) | 105 `FIT-SEL` | another session's closed wave, **read only** (`APPLY_FITSEL.json`, `155eb59d2157…`) |

- Stack: the stored `FIT-SEL` matrices of H420 `TBOTH` (the recipe H434 used on `DEV`), each checked against its
  record. An hour-12 column is used on a table only if the bank's SEARCH targets and fold labels equal the stored
  matrix's; tables that do not align are left out and counted.
- H428's `FIT-SEL` outputs cannot be used: they hold scores per table, no per-row predictions.
- None of these columns was used to derive or check the rule (H434 checked it on H424's hour-20 column on `DEV`).

## 4. The screen, emulated from stored values
- **Sample: the rows with stored fold label 0** (about 20% of a table's SEARCH rows). A 5-fold out-of-fold column
  predicts these rows from the other four folds, which is what one screening pass would produce. The stored
  candidates' predictions for the same rows already exist at no cost.
- **Prediction:** the rule of section 2 on those rows → keep or skip.
- **Realised weight:** the stored weight builder (meta_h416 BCAR, the call the stack makes) fitted on the 64 stored
  candidates plus the full column; the added column's weight. "Real weight" means ≥ 0.10.

## 5. Inference cost (a model, not a measurement)
One pass of the in-context model takes about (rows ÷ 1,600) × (35 + 0.72 × features) seconds on one thread (H424).
- Always run: 6 passes over SEARCH and AUDIT rows per table.
- With the screen: one pass over SEARCH rows on every table, plus the 6 passes on the tables kept. The screening
  pass is counted as extra work (not reused).
- Saved = 1 − cost with the screen ÷ cost of always running. H430's columns are not in-context columns; for them
  only the share of tables skipped is reported.

## 6. Test, success and failure
Primary set: **the two hour-12 in-context columns on all aligned `FIT-SEL` tables, pooled** (up to 212 decisions).
- False negative: real weight, yet the screen would have skipped the table.

| Verdict | Condition |
| --- | --- |
| `SCREEN_PASSES` | false negatives ÷ decisions ≤ 0.05 (at most 1 per 20 tables) **and** saved inference ≥ 40% |
| `SCREEN_FAILS` | otherwise; the failing part is named |

- **Consequence of a pass:** the screen becomes the default first step before any new column's full inference in
  this campaign. **Of a fail:** it is not used as a gate; recorded.

## 7. Reported beside the verdict (no verdict of their own)
- Accuracy, the four counts, the miss rate among real-weight columns, and cluster-bootstrap 95% intervals (clusters
  of related `FIT-SEL` tables from the stored H425 rehearsal screen; the two columns of a table stay together).
- Each column on its own, including the 8-table hour-20 column and H430's five (pooled and apart).
- Three pre-declared variants on the primary set: keep when the predicted weight is ≥ 0.05; estimates from all
  rows; estimates from half of the fold. Wide and narrow tables apart.
- **The plain readout against the residual readout at hour 12** (overlap, distance to the blend, real weight; wide
  and narrow tables). **Flag for H434:** if the plain column has real weight on at least one third of the wide
  `FIT-SEL` clusters, H434's no-go is marked `REVISIT_H434`; otherwise `H434_NO_GO_STANDS`.
- Replay: realised weights against the values stored by H424's smoke and by H430 for its single-column arms.

## 8. Budget and CPU
- 4 single-thread processes at the lowest priority (nice 19), on the 4 least busy cpus of a 10-second sample taken
  at launch (xbabe2 had no idle cpu set today; the sample is recorded). About 750 weight fits, 10 to 15 CPU-minutes
  expected; stopped at 40 minutes. No GPU; no model library is loaded.

## 9. Gates (`gates_h435.sh`, after the fits, before the readout)
This PREREG is older than the first output and no readout exists; the five pinned inputs; all 106 `FIT-SEL` tables
with 33 wide; at most 4 single-thread processes and no model library; the 45-minute cap; the sandbox view still
holds no `CONFIRM` table; the other session's folders unchanged; published stage.

## 10. Expected, stated in advance
- Accuracy about 90%. Saved inference above 45%.
- **False negatives close to the limit; a narrow fail is slightly more likely than a pass.** On `DEV` with all rows
  the rule missed 4 of 40 real-weight tables and raised no false alarm: it is conservative. With a fifth of the
  rows, more tables near the threshold will flip. If about 35–40% of decisions have real weight and 15% of those
  are missed, false negatives are 5–6% of decisions.
- Plain readout on wide tables: real weight on fewer than one third of clusters (`H434_NO_GO_STANDS`).

## 11. What was looked at before this PREREG
The file layouts and array shapes of every input; the summary of H419's hour-12 stack test (with H417's matrices
and another weight builder: column used in 49% of tables, mean weight 0.113); H424's smoke summary; the head of
H430's `FIT-SEL` readout (its arms' gains). A self-test of the code on **random numbers only**. **No overlap, RMSE
ratio, predicted weight or stored-builder weight has been computed for any of these columns on any `FIT-SEL` table.**

## 12. Weaknesses
- The saving comes from a per-pass time model measured on `DEV` tables, not from measured `FIT-SEL` runs.
- Same model family as the column the rule was checked on; `FIT-SEL` is that model's monitor cohort. The plain
  readout and H430's columns are the parts that differ in kind.
- False negatives are counted per decision, as ordered; the miss rate among real-weight columns will be higher
  and is reported.
- The screen targets weight, not held-out gain; a skipped table with real weight is a lost chance, not a measured
  loss.
- The two hour-12 columns share tables and are dependent; the bootstrap resamples clusters of tables.
- H430's realised weights are single additions fitted here, not that wave's own three-column arm.

## 13. Files
`screen_h435.py`, `gates_h435.sh`, `stage_h435.sh`. Reused at published bytes: `cache_h422.py` (record reader and
weight builder call), `meta_h416.py`, `restate_h426.py`.
