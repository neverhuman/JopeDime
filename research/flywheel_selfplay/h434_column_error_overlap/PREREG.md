# H434 PREREG — error overlap of the in-context column with the stored candidates on wide tables; a go / no-go for a plain-readout column

**Status.** Published on `origin/v1/next` before any output of this wave exists. Claim:
`~/jope_plan/ns/receipts/H434_claim.txt` (2026-11-11T16:21Z). H430 to H433 belong to other sessions and are not touched.

**Authority.** Relayed by ENG:watcher on 2026-11-11, after the H432 closeout: prompt A of H432 as H434, from stored
values only; on the 35 wide `DEV` tables the correlation and overlap of the column's out-of-fold errors with each
stored candidate's errors and with the best candidate, at cluster level; state before the readout the overlap level
at which prompt B (a plain-readout wide column, about 6.5 core-hours) would be worth running; success = a
pre-declared go / no-go for B with the measured overlap and its interval, recorded either way; also whether the
gain concentration holds with its largest table left out. CPU only, at most 4 threads at low priority, at most
30 minutes, no new inference, no GPU; development evidence only.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not staged, opened or
read. No table is opened, no model pass is made, no stack weight is refitted.

## 1. Questions
1. On the wide `DEV` tables, how much of the in-context column's out-of-fold error is the same error the stored
   candidates and the stored blend already make?
2. Given that, is a second column from the model's plain readout (prompt B of H432) worth its 6.5 core-hours?
3. Does H424's gain stay concentrated in the 11–20-feature tables when its largest table is left out?

## 2. Inputs (pinned by sha256; the gates check them)
H424's value file (`ce0eb88749eb…`) and its 115 stored column files; the stored `DEV` stack matrices (H420 `TBOTH`,
each checked against its record) with the stored stack weights of each record; the frozen H425 screen
(`fa8edd8a08a9…`) for H426's `feature` clusters; H424's bank for table shapes only (`dccf9beb3312…`); H432's stored
per-table weight curves (`69081a5bee17…`).

## 3. Measures (per table; SEARCH rows, out of fold)
- Errors: the column's, each of the 64 stored candidates', the best single candidate's (lowest RMSE), and the stored
  blend's (the candidates combined with the stored weights of the record).
- **Overlap** of two error vectors = their uncentred correlation. Reported for the column against: the blend, the
  best candidate, and all 64 candidates (median, quartiles, nearest). Pearson correlation with the blend beside it.
- Cluster level: tables averaged within a cluster, then the median and mean over the 30 wide clusters with a
  cluster-bootstrap 95% interval (10,000 resamples). The 80 narrow tables are reported the same way as a reference.

## 4. The go / no-go for prompt B (the only verdict)
**The bound.** For two predictors, blend and column, with RMSE ratio r and error overlap ρ, the error-minimising
weight of the column is w = (1 − ρr) / (1 + r² − 2ρr). It reaches 0.10 exactly when ρ ≤ ρmax(r) =
(0.9 − 0.1r²) / (0.8r). With ρ = 0 this gives r ≤ 3, a distance of ln 3 = 1.10 u: H432's independent-error result
(1.09 u). H432's same-pattern result (0.18 u) corresponds to ρ ≈ 0.79.

**The level, stated before the read.**
- On the median wide table the column is 0.27 u behind the best candidate. There, 10% weight needs an overlap of
  **at most about 0.70**.
- A plain-readout column is expected to be about 0.17 u weaker than today's (H419 monitor at hour 24.3: residual
  readout median 1.047 u, plain readout 0.877 u). At 0.44 u behind, 10% weight needs an overlap of **at most
  about 0.53**.
- **B is worth running only if both hold:** (1) today's overlap is above its own limit on most wide tables, so
  that overlap and not accuracy is what denies the column weight; and (2) an overlap low enough for a column of
  the plain readout's expected accuracy has been seen on those tables at all.

**The rule, per table with its own distances, clusters weighted equally:**
- `blocked`: the column is within ln 3 of the blend and its overlap with the blend exceeds ρmax at its distance.
- `floor`: the lowest overlap with the blend (own weight removed, the rest renormalised) among stored candidates
  at least as accurate as a column 0.17 u weaker than today's; or today's column's own overlap if lower. This is
  the most different predictor of that accuracy on record for the table.
- `reachable`: floor ≤ ρmax at a distance 0.17 u beyond today's.

| Verdict | Condition | Consequence |
| --- | --- | --- |
| `B_UNDECIDED` | the bound classifies "real stack weight ≥ 0.10" correctly on fewer than 75% of the 115 tables | the bound does not describe the stack; no recommendation from this wave |
| `B_GO` | bound holds, blocked share > 1/2 and reachable share ≥ 1/3 (30 wide clusters) | prompt B is recommended to the watcher as written |
| `B_NO_GO` | bound holds, otherwise | prompt B is not recommended; the reason (accuracy or no attainable overlap) is stated |

- Reported with the verdict: the measured overlap and the two shares with their intervals; the reachable share
  with a handicap of 0 and of 0.34 u; the bound's agreement with H432's stored need-distances.
- The verdict is a recommendation about spending 6.5 core-hours. It is not a test of any model.

## 5. Gain concentration (reported, no verdict)
From H424's stored values: the gain by table width with and without the largest table; that table's share; the
five largest; and the cluster-level tests of `I` vs `B` and of H432's narrow-only arm vs `B` with that table's
cluster left out.

## 6. Budget, CPU and gates
- One thread at the lowest priority (nice 19); expected under 2 minutes; stopped at 25 minutes. No GPU, no model
  library. (xbabe2 had no idle cpu set at 15:53Z; one low-priority thread is not pinned.)
- Gates (`gates_h434.sh`), run by the driver **before** the read because the whole wave is one read: this PREREG
  published and equal to the staged copy, no result file exists; the pinned inputs and 115 column files; the
  sandbox view still holds no `CONFIRM` table; published stage.

## 7. Expected, stated in advance
- Overlap of the column with the blend on wide tables: about 0.8 (0.75 to 0.9).
- **`B_NO_GO`.** I expect the column to be blocked on roughly 60% of the wide clusters (9 of 35 tables are too far
  behind for any error pattern, 4 already have 10%), and the floor to sit near 0.7 on the typical table, above the
  0.53 a plain column would need. My confidence is low; I could not predict H432's two numbers.
- Concentration: by arithmetic from H432's table, without the largest table the 11–20-feature bin holds 74% of
  the summed gain (0.542 of 0.730 u). The tests without that cluster should still pass (leave-one-unit-out minimum
  +0.0067 u).

## 8. What was looked at before this PREREG
H424's stored values and weights (H427, H429, H432), H432's weight curves, the per-table distances to the best
candidate; a self-test of the code on **random numbers only**. **No error correlation has been computed on any
stored `DEV` matrix.**

## 9. Weaknesses
- **The plain readout's own overlap is not measured;** measuring it is prompt B. The verdict rests on a proxy: the
  most different predictor on record. GP candidates share preprocessing and rows; a neural predictor could differ
  more than any of them (then `B_NO_GO` is too strict) or less (then `B_GO` is too lenient).
- The 0.17 u handicap is the monitor's figure over all widths on `FIT-SEL`; on wide tables it may be larger.
- The bound treats the stored blend as fixed; the real builder refits all weights with bagging and pruning. The
  75% rule guards against a bound that does not fit.
- The blend's weights were fitted on these same out-of-fold rows, so its error is a little optimistic and the
  column looks a little further behind it.
- The thresholds (1/2, 1/3, 75%) are my choices, fixed here. 35 tables in 30 clusters give wide intervals.
- `DEV` is a reused development cohort.

## 10. Files
`overlap_h434.py`, `gates_h434.sh`, `stage_h434.sh`. Reused at published bytes: `cache_h422.py`, `need_h432.py`
(the need-distance reader), `restate_h426.py`, `train_h416.py` and the frozen H335 statistics.
