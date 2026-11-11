# H437 — how many folds of rows the weight screen needs: `FOLDS_BETTER_NOT_ENOUGH`; the row-sample screen is dropped (closeout)

**Verdict: `FOLDS_BETTER_NOT_ENOUGH`, a negative result.** Neither tested fold count makes the weight rule a
usable screen. With two folds of rows it still misses 17 real-weight columns in 212 decisions (8.0%, interval 4.2%
to 12.5%; limit 5% with an upper end of 9%), and that is not distinguishable from one fold (19). With three folds
it misses 14 (6.6%) and saves only 50% of inference (60% needed).
**False negatives fall below 5% only at four of the five folds (4.2%), where 36% of inference is left to save.**
Accuracy rises slowly with rows while every fold costs 11.7% of the inference: there is no fold count where the
screen is both safe and cheap. **The row-sample screen is dropped.**
Stored hour-12 `FIT-SEL` values only; nothing fitted; no `DEV`, no AUDIT array, no inference; `CONFIRM` and dev91
untouched.

Plan: [`PREREG.md`](PREREG.md), [addendum 1](PREREG_ADDENDUM_1.md). Numbers: [`metrics.json`](metrics.json).
Gates: [`gates_h437.sh`](gates_h437.sh), [`gates2_h437.sh`](gates2_h437.sh).

## Result (the 212 decisions of H435: two hour-12 in-context columns × 106 `FIT-SEL` tables; 45 have real weight)
| Rows used (stored folds 0 … k − 1) | Kept | Hit | False alarm | **False negative** | Accuracy | **False negatives ÷ decisions [95%]** | **Saved inference [95%]** | Saved if passes are reused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 fold, 20% (H435's baseline, reproduced exactly) | 53 | 26 | 27 | 19 | 78.3% | 9.0% [5.0, 13.6] | 66.5% [58.8, 73.4] | 69.0% |
| **2 folds, 40% (tested)** | 48 | 28 | 20 | **17** | 82.5% | **8.0% [4.2, 12.5]** | **60.3% [53.2, 66.2]** | 64.2% |
| **3 folds, 60% (tested)** | 50 | 31 | 19 | **14** | 84.4% | **6.6% [3.2, 10.7]** | **49.7% [42.8, 55.2]** | 55.1% |
| 4 folds, 80% (reported) | 51 | 36 | 15 | 9 | 88.7% | 4.2% [1.5, 7.5] | 36.3% [28.9, 42.4] | 44.3% |
| 5 folds, all rows (reported; H435's all-row counts reproduced) | 45 | 38 | 7 | 7 | 93.4% | 3.3% [1.3, 5.9] | 26.8% [20.0, 32.2] | 35.5% |

| Rule (PREREG) | 2 folds | 3 folds |
| --- | --- | --- |
| False negatives ÷ decisions ≤ 5% | no (8.0%) | no (6.6%) |
| Upper end of the interval ≤ 9% | no (12.5%) | no (10.7%) |
| Saved inference ≥ 60% | yes (60.3%) | no (49.7%) |
| Fewer false negatives than one fold | yes (17 against 19) | yes (14 against 19) |

Paired difference in false negatives ÷ decisions against one fold: two folds −0.9 points, interval [−3.8, +1.9];
three folds −2.4 points [−5.2, +0.5]; four folds −4.7 points [−8.5, −1.4].

## What the result does and does not say
- **Two folds buy almost nothing:** two fewer misses for 6 points of saving, inside the noise.
- **The safe fold count is not cheap.** Under 5% needs four folds; by then the screen has spent most of what it
  was meant to save. Counting the screening passes as reused on kept tables does not change the picture (44%).
- **Which fold matters as much as how many.** Over the five possible single folds, false negatives run from 4.7%
  to 9.0% (median 6.6%); over the ten pairs from 2.8% to 8.0% (median 5.9%); over the ten triples from 2.8% to
  6.6% (median 5.7%). **The stored folds 0, 0–1 and 0–2 are the least favourable choice at every count.** Even the
  median choice stays above 5% up to three folds, and the median pair saves 58%.
- **Correction of emphasis for H435:** its 9.0% (42% of real-weight columns missed) came from fold 0, the worst of
  the five folds. The median single fold misses 6.6% (31% of real-weight columns). H435's verdict stands either
  way; its size was a bad draw.
- **The two readouts behave differently.** Residual readout: 12, 9, 7 misses at one, two, three folds. Plain
  readout: 7, 8, 7. More rows do not help the plain column's misses.
- **Not shown:** anything on new columns or tables (the same 212 decisions as H435); anything about gain.
- **What remains true:** with all rows the rule is 93.4% accurate. For a column that is already computed it costs
  nothing and says where the stack will give weight. It does not save inference.

## Integrity gates
- **First run (`gates_h437.sh`): 6 of 7; gate 3 failed.** The rebuilt fold-0 and all-row estimates differ from
  H435's stored ones by up to 7.5e-9 in absolute terms, on RMSE ratios of 2e7 to 5e7 where the plain column fails
  (relative difference 1.4e-16, one rounding unit). The absolute tolerance was my mistake. No readout was run.
- **Addendum 1** (`ace7234c58`, before the readout): the tolerance is relative. **Second run (`gates2_h437.sh`):
  9 of 9.** PREREG (`5b67f4b6c5`) older than the sums file and no readout existed; the five pinned inputs; 106
  tables, 212 columns, largest relative replay difference 7.8e-16; one thread, no model library, no weight builder;
  the 30-minute cap; the sandbox view still holds no `CONFIRM` table; published stage; addendum before the
  readout; readout code byte-identical to the copy staged before the sums.
- Replay in the readout: the one-fold row equals H435's primary exactly (counts, saving and both intervals); the
  five-fold row equals H435's all-row counts.

## Deviations, negatives and corrections
1. **`FOLDS_BETTER_NOT_ENOUGH`:** no pre-registered fold count passes. The screen idea is dropped, as its prompt
   said it would be on a failure.
2. **The gate failure and addendum above.** Nothing that depends on the fold count was seen in between.
3. **My expectations were too optimistic on false negatives:** about 6–7% at two folds (found 8.0%) and about 5%
   at three (found 6.6%). The arithmetic on saving held: three folds could not pass (49.7%); two folds passed
   narrowly (60.3%).
4. **The prefix folds are an unlucky choice** (see above). The verdict uses them, as pre-registered; the spread
   is reported and would not change the verdict at the median.
5. A second look at H435's data, not a new test; the saving is a cost model.
6. Cost: one thread at the lowest priority; 3.5 CPU-seconds for the sums, 30 seconds for the readout.

## Queue after this wave (none started)
- **B. Training mix for wide tables** (prompt B of H427; held until a GPU frees up). Targets from H434: 0.06 u more
  accuracy or 0.07 less overlap on the blocked wide tables.
- **C. Does the rule predict the gain?** (prompt A of H434; needs AUDIT values on `DEV`; not run without an order.)
- **Waiting for Jepson:** the read-only apply of the all-row rule to H433's finished `FIT-SEL` columns (minutes);
  the `CONFIRM` decision; follow-up 1 of H422.
- No further screen prompt is queued.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 17:20 | H437 claimed |
| 17:22 | PREREG and code published (`5b67f4b6c5`); per-fold sums (4 s); first gate run: gate 3 fails |
| 17:24 | addendum 1 and second gate script published (`ace7234c58`); gates 9 of 9; readout: `FOLDS_BETTER_NOT_ENOUGH` |

Evidence (xbabe2): `~/jope_plan/evidence/H437/results/` (`H437_FITSEL.fold_sums.json`, `H437_FITSEL.readout.{json,txt}`,
`H437.gates.json`, `H437.gates2.json`), `logs/`.
