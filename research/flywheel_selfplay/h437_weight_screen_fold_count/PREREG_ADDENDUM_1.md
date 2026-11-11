# H437 PREREG addendum 1 — the replay gate's tolerance is relative, not absolute

**Status.** Published on `origin/v1/next` before the readout. The per-fold sums exist
(`H437_FITSEL.fold_sums.json`, 17:22:59Z); **no estimate on two, three or four folds, no count and no verdict has been
computed.** The first gate run is kept as it is (`H437.gates.json`: 6 of 7 pass, gate 3 fails).

## 1. What failed
Gate 3 of the PREREG required the fold-0 and all-row estimates rebuilt from the per-fold sums to equal H435's
stored ones "within 1e-9", as an absolute difference. The largest difference is 7.5e-9.

## 2. Why
The tolerance was mis-specified by me: it is absolute, and one of the compared quantities is an RMSE ratio that is
very large where the plain-readout column fails badly.
| Table (prefix), column | H435's stored RMSE ratio | Absolute difference | Relative difference |
| --- | --- | --- | --- |
| `246e643fc2`, plain | 50,387,134 | 7.5e-9 | 1.4e-16 |
| `4286cf7f94`, plain | 20,421,270 | 3.7e-9 | 1.8e-16 |
| the other 210 columns | up to 252,827 | at most 2.9e-11 | at most 1.4e-16 |

Every difference is one unit of floating-point rounding. The stored values are reproduced.

## 3. The change (nothing else changes)
- **Gate 3, replaced:** each rebuilt estimate (overlap and RMSE ratio, on fold 0 and on all rows) must equal H435's
  stored value within **1e-9 × max(1, |stored value|)**. The check is recomputed from the per-fold sums by
  `gates2_h437.sh`, which also repeats the other six gates and adds two: this addendum is published and no readout
  exists; the readout code is byte-identical to the copy staged before the sums were made.
- Rule, fold counts, cost accounting, test, verdicts and reported items are those of the PREREG. The sums file is
  not recomputed.

## 4. What was looked at between the PREREG and this addendum
The sums step's status line; the first gate output; per column, the replay difference beside H435's stored ratio
(the table above). Nothing that depends on the fold count.
