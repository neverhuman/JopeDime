# H435 — the two-number weight rule as a screen before column inference: `SCREEN_FAILS` on one fold of rows (closeout)

**Verdict: `SCREEN_FAILS`, a negative result.** Applied to one fold of rows per table, H434's rule would have
skipped 19 real-weight columns in 212 decisions: **9.0% false negatives** (95% interval 5.0% to 13.6%) against a
limit of 5%. That is 42% of the 45 columns that really earn 10% stack weight. The saving part passes (66% of
inference under the cost model, 40% needed), but a screen that drops two in five of the columns worth having is
not usable as a gate.
**The rule itself transfers; the row sample is what breaks it.** With all rows the same rule is right on 93% of
the same decisions (3.3% false negatives), on tables, a checkpoint and a readout it was never checked on.
**Side reading, measured for the first time:** on the 33 wide `FIT-SEL` tables the plain-readout column earns 10%
weight on none. H434's no-go for that column stands.
`FIT-SEL` only: `DEV` not read, no AUDIT array, no gain; no inference, no GPU; `CONFIRM` and dev91 untouched.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h435.sh`](gates_h435.sh).

## Result (106 `FIT-SEL` tables in 97 clusters; realised weight from the stored builder; "real" = weight ≥ 0.10)
| Columns screened (estimate from one fold, 224 of 1,120 rows on the median table) | Decisions | Real | Kept | Hit | False alarm | **False negative** | Accuracy | **False negatives ÷ decisions** | Inference saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary: the two hour-12 in-context columns, pooled** | 212 | 45 | 53 | 26 | 27 | **19** | 78.3% | **9.0% [5.0, 13.6]** | 66.5% [58.8, 73.4] |
| Hour 12, residual readout | 106 | 25 | 28 | 13 | 15 | 12 | 74.5% | 11.3% [5.7, 17.8] | 60.5% |
| Hour 12, plain readout | 106 | 20 | 25 | 13 | 12 | 7 | 82.1% | 6.6% [2.6, 11.8] | 72.4% |
| Hour 20, residual readout (H424's smoke) | 8 | 1 | 1 | 1 | 0 | 0 | 8 of 8 | 0 | — |
| H430's five columns, pooled (another session's, read only) | 525 | 26 | 80 | 13 | 67 | 13 | 84.8% | 2.5% [1.2, 4.0] | 85% of tables skipped |

Rule: false negatives ÷ decisions ≤ 5% **and** saved inference ≥ 40% → pass. **First part fails, second passes.**

**Pre-declared variants on the primary set:**
| Variant | Hit | False alarm | False negative | Accuracy | False negatives ÷ decisions | Inference saved |
| --- | --- | --- | --- | --- | --- | --- |
| Keep when the predicted weight is ≥ 0.05 | 31 | 35 | 14 | 76.9% | 6.6% [2.9, 11.0] | 59.6% |
| **Estimates from all rows** (needs the whole out-of-fold column) | 38 | 7 | 7 | **93.4%** | **3.3% [1.3, 5.9]** | not a saving: 5 of the 6 passes are already spent |
| Estimates from half of the fold | 20 | 33 | 25 | 72.6% | 11.8% [7.1, 17.1] | 68.3% |
| Primary, 33 wide tables only | 1 | 9 | 3 | 81.8% | 4.5% [0, 10.7] | 71.1% |
| Primary, 73 narrow tables only | 25 | 18 | 16 | 76.7% | 11.0% [5.7, 16.9] | 60.8% |

## What the result does and does not say
- **One fold of rows is too little.** Against the all-row estimate, the one-fold estimate moves the overlap by 0.03
  on the median decision and by 0.18 on one in ten; the RMSE ratio by 0.045 u and 0.30 u. The keep / skip decision
  flips on 42 of 212. Out-of-fold errors are dominated by a few rows; a fifth of the rows often misses them.
  (Added after the readout, descriptive.)
- **Most misses are near the threshold:** 15 of the 19 are columns whose real weight is between 0.10 and 0.20;
  4 are columns with 0.20 or more. Six of the 19 would be missed with all rows as well.
- **A lower bar does not rescue it:** keeping at a predicted 0.05 still misses 14 (6.6%).
- **The rule transfers.** With all rows it is 93.4% accurate here against 96.5% on `DEV` in H434, on a different
  cohort, an earlier checkpoint and a readout (plain) it had never seen. It remains a valid description of when
  the stack gives weight; it is the cheap version that fails.
- **H430's columns look better only because real weight is rare there** (26 of 525). The screen still misses half
  of them (13 of 26) and raises 67 false alarms.
- **The stored builder gave 10% weight to a noise column on 2 of 105 tables** (H430's `key_noise` control). Weight
  is not value; this wave computes no gain.
- **Not shown:** anything about held-out gain, about `DEV`, or about screens that use more rows.
- **The saving is a model, not a measurement** (per-pass time formula of H424; the screening pass counted as extra
  work).

## Side reading: the plain readout against the residual readout at hour 12 (cluster medians with 95% intervals)
| | 33 wide tables: residual | 33 wide tables: plain | 73 narrow tables: residual | 73 narrow tables: plain |
| --- | --- | --- | --- | --- |
| Overlap with the stored blend | 0.837 [0.761, 0.882] | 0.631 [0.501, 0.786] | 0.827 [0.747, 0.887] | 0.815 [0.726, 0.864] |
| Distance behind the blend (u) | 0.170 [0.124, 0.329] | 0.551 [0.373, 0.898] | 0.143 [0.096, 0.272] | 0.179 [0.096, 0.314] |
| Share of clusters with real weight | 0.15 [0.04, 0.31] | **0.00 [0.00, 0.00]** | 0.28 [0.18, 0.38] | 0.27 [0.17, 0.37] |
| Mean real weight | 0.048 | 0.005 | 0.083 | 0.072 |

- **Wide tables:** the plain column is less like the stack (overlap lower by 0.135 [0.075, 0.292]) and much less
  accurate (0.279 u weaker [0.216, 0.629]). It earns 10% weight on none of the 33.
  **Flag: `H434_NO_GO_STANDS`.** H434 assumed a handicap of 0.17 u; on wide tables it is larger.
- **Narrow tables:** the two readouts are the same on the median table (distance difference 0.000, overlap +0.001)
  and earn weight equally often.
- Limits: hour-12 checkpoint, the model's monitor cohort, weight not gain.

## Integrity gates (`gates_h435.sh`, after the fits, before the readout; 8 of 8 pass)
PREREG (`e14bcb9216`) older than the first output and no readout existed; the five pinned inputs; all 106 `FIT-SEL`
tables (33 wide), hour-12 columns aligned on all 106, 745 columns scored; 4 single-thread processes, no model
library; 171 seconds of the 45-minute cap; the sandbox view still holds no `CONFIRM` table; the other session's
folders unchanged (modification times before and after); published stage.
Replay: the realised weights equal the stored ones exactly on H424's 8 smoke tables and on H430's two
single-column arms (105 tables each; difference 0).

## Deviations, negatives and corrections
1. **`SCREEN_FAILS`.** The screen is not adopted as a gate. Prompt C of H434 is closed as written.
2. **My expectation was too mild.** I wrote "a narrow fail is slightly more likely than a pass", with accuracy
   about 90% and false negatives 5–6%. Found: 78% and 9.0%. The saving (66%) was above my guess.
3. **H428's `FIT-SEL` columns could not be used** (named in the order): they are stored as scores per table, with
   no per-row predictions.
4. A descriptive file was added after the readout (`H435_FITSEL.sample_noise.posthoc.json`). Its first write failed
   on a syntax error and left an empty file, which was then overwritten; nothing was read from it in between.
5. CPU: 4 single-thread processes at the lowest priority on cpus 2, 4, 8 and 9, which were idle at launch (0.3% to
   4.5% busy in the 10-second sample); 651 CPU-seconds, 171 seconds.
6. The 5% limit is counted per decision, as ordered. Per real-weight column the miss rate is 42%.

## Queued prompts (none started)
- **A. How many rows does the rule need? Stored values only, minutes of CPU.** "On the same 212 `FIT-SEL` decisions
  and stored columns, apply the frozen rule to the rows of the first k folds, k = 1 to 5, and to a sequential
  version that stops a table as soon as the predicted weight is below 0.05 after k folds. Cost: k screening passes
  that are reused if the table is kept. Success: some pre-declared k ≤ 3 (fixed before the read: k = 2) has false
  negatives ≤ 5% of decisions with ≥ 25% of inference saved. Failure: otherwise; the screen idea is dropped."
- **B. Training mix for wide tables** (prompt B of H427; held until a GPU frees up). Targets from H434: 0.06 u more
  accuracy or 0.07 less overlap on the blocked wide tables.
- **C. Does the rule predict the gain?** (prompt A of H434; needs AUDIT values, not run in this wave by order):
  promised against realised gain for H424's column on `DEV`, from stored values.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 16:57 | H435 claimed; inputs located (layouts and shapes only) |
| 17:09 | PREREG and code published (`e14bcb9216`); fits started (4 processes, lowest priority) |
| 17:12 | fits ended (106 of 106 tables); gates 8 of 8; readout: `SCREEN_FAILS` |
| 17:14 | descriptive addition from the stored per-table outputs |

Evidence (xbabe2): `~/jope_plan/evidence/H435/results/H435_FITSEL.readout.{json,txt}`, `H435.gates.json`,
`H435_FITSEL.sample_noise.posthoc.json`, `logs/`; per-table outputs in `~/jope-data/h435/pub/tables/`.
