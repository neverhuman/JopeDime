# H432 PREREG — the in-context column on narrow tables only, restated from stored values; and how good a wide-table column must be

**Status.** Published on `origin/v1/next` before any output of this wave exists. Claim:
`~/jope_plan/ns/receipts/H432_claim.txt` (2026-11-11T15:47Z). H430 and H431 belong to other sessions and are not touched.

**Authority.** Relayed by ENG:watcher on 2026-11-11, after the H429 closeout: prompt A of H429 as H432, from stored
values only; baseline = H424's full column arm, ablation = no column; clipped gain, one-sided p and win / tie / loss
at cluster level, plus the inference-CPU saving; success = the narrow-only arm keeps at least 90% of the H424 gain
with p < 0.05 and lower CPU, failure = it loses more than 10%; prompt C folded in if cheap; CPU only, at most
8 threads on an idle cpu set (to be checked first), at most 1 hour, no GPU, no new inference; development evidence
only; a PREREG before the readout.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not staged, opened or
read. No table is opened and no model pass is made: the inputs are H424's stored values, H424's stored column files
and the stored `DEV` stack matrices.

## 1. Questions
1. If the in-context column is offered only on tables with at most 30 features, does the stack keep H424's gain on
   `DEV`, and how much inference CPU does that save?
2. How close to the best stored candidate must an added column be, out of fold, before the stored weight builder
   gives it 10% weight on a wide table? (H429: the wide-table column is 1.48 u behind and gets 3%.)

## 2. Inputs (pinned by sha256; gates check them)
| Input | sha256 |
| --- | --- |
| H424 value file, hour-20 read (`H424_DEV.apply.json`) | `ce0eb88749eb…` |
| H424 value file, hour-24 read (`H424_DEV_h024.apply.json`) | `aacbcd6894de…` |
| H425 frozen screen (`SCREEN_DEV.json`), for H426's `feature` clusters | `fa8edd8a08a9…` |
| H424 inference bank (table shapes only are read) | `dccf9beb3312…` |
| Stored `DEV` stack matrices (H420 `TBOTH`), each checked against its record; H424's stored column files | per record |

## 3. Part 1 — narrow-only column (restatement; the only verdict of this wave)
Arms per `DEV` context, all stored AUDIT u values, nothing refitted:
- `B` the stored stack, no column (ablation);
- `I` H424's arm, the column offered on all 115 tables (baseline);
- `N` narrow-only: `I` on the 80 tables with at most 30 feature columns, `B` on the 35 wider ones.

Units: H426's `feature` clusters (110). Statistic: `h335_stats.primary_test` through `train_h416.pair_stats`.
- **Retention** R = clipped unit-mean gain of `N` over `B` ÷ clipped unit-mean gain of `I` over `B`.

| Verdict | Condition |
| --- | --- |
| `NARROW_KEEPS` | R ≥ 0.90, and `N` vs `B` one-sided sign-flip p < 0.05, and the narrow-only recipe needs less inference CPU than H424's |
| `NARROW_LOSES` | R < 0.90 |
| `NARROW_UNCLEAR` | R ≥ 0.90 but p ≥ 0.05 |

- Reported beside it: whether `N` vs `B` also passes the campaign's usual rule (p < 0.025, leave-one-unit-out
  minimum above 0, median not negative); `N` vs `I`; a cluster-bootstrap interval for R (10,000 resamples);
  the same arms with the hour-24 column; inference CPU-seconds of both recipes from the stored per-table times.
- **Threshold table, reported only:** the same restatement with the column kept up to 10, 20, 30, 50, 100 features
  and on all tables. The verdict uses 30 and nothing else.
- **Consequence of `NARROW_KEEPS`:** "column only up to 30 features" becomes the development default of the column
  recipe. It is a cost rule, not a claim that wide tables cannot gain. No champion consequence; `CONFIRM` stays closed.

## 4. Part 2 — the column quality a wide table needs (descriptive, no verdict; prompt C of H429)
Per `DEV` context: the stored out-of-fold matrix (64 candidates), the SEARCH targets, and one synthetic 65th
column placed at a fixed out-of-fold distance d behind the best stored candidate (its RMSE = best stored RMSE ×
exp(d)); the stored weight builder (the call H424 made) gives the weights; the added column's weight is recorded.
- Distances: −0.2, −0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5 u (negative = better than the best candidate).
- **S, same error pattern (main):** H424's own column with its out-of-fold error rescaled to the target RMSE. It
  asks: what if the model were uniformly better, with the errors it has now? All 115 tables.
- **N, independent noise (optimistic bound):** seeded Gaussian error at the target RMSE, 3 seeds, 35 wide tables.
  No real model has errors unrelated to every other candidate's, so this is the most weight a column of that
  accuracy could get.
- Reported: by distance, mean weight and the share of tables with weight ≥ 0.10; per table, the distance at which
  the weight first falls below 0.10 (quartiles over tables); wide and narrow tables apart; and, from stored values,
  the weight the real column gets against its real distance.
- Replay check: the unscaled column must get exactly the weight stored by H424 (gate).
- Only SEARCH-side arrays are used. No AUDIT value is computed in part 2; nothing in it is a score.

## 5. Budget and the CPU check
- **Check made before this PREREG (15:46Z):** xbabe2 has no idle cpu set. Load average 91 on 128 cpus, from other
  sessions' jobs; in a 2-second sample the least busy cpu was 9% busy and only 10 cpus were under 30%.
- **Rule fixed here:** the driver samples all cpus for 10 seconds at launch, pins the work to the 8 least busy and
  records the sample. 8 single-thread processes at the lowest priority (nice 19), so other jobs keep their cpus.
- Part 1: seconds of one thread. Part 2: about 2,500 weight fits, 0.5 to 1.2 CPU-hours by the times of H424's
  stack step. Part 2 is stopped after 45 minutes; if it is incomplete then, it is reported as not done and part 1
  stands alone. One hour for the whole wave. No GPU; no model library is loaded.

## 6. Gates (`gates_h432.sh`, after part 2's fits, before the two readouts)
This PREREG is older than the first output and no readout exists; the four pinned inputs; 115 contexts (35 wide)
with an exact replay of H424's stored column weight; at most 8 single-thread processes and no model library
loaded; the one-hour cap; the sandbox view still holds no `CONFIRM` table; published stage.

## 7. Expected, stated in advance
- **Part 1: `NARROW_KEEPS`, with R a little above 1 (about 1.02). This outcome is already implied by published
  numbers:** H429 reported H424's column at −0.0002 u against the stack on the 35 wide tables, and 53% of the
  column's inference time on those tables. This PREREG fixes the rule and the wording; it cannot make the read blind.
- **Part 2 (not known):** my guess is that under S a column must be within about 0.3 u of the best stored candidate
  to earn 10% weight on half of the wide tables, and within about 0.6 u under N. A self-test on random numbers
  (not data) gave 0.6 to 0.7.

## 8. What was looked at before this PREREG
All stored H424 values by table width (H427, H429), including the comparison that decides part 1; the stored
per-table inference times; code self-tests on **random numbers only** (`narrow_h432.py --selftest` replaces the
stored values by random ones; `need_h432.py selftest` uses a random matrix). **No synthetic-column weight has been
computed on any stored `DEV` matrix.**

## 9. Weaknesses
- **The 30-feature threshold was chosen from these same `DEV` values (H427).** Dropping the column exactly where it
  was seen to earn nothing must look harmless on the same tables. Part 1 measures the cost saving honestly; it
  does not show that 30 is the right threshold on new families. The threshold table shows whether 30 sits on a
  plateau or an edge, on the same cohort.
- `DEV` is a reused development cohort (read by H420, H421, H422, H424, H429).
- R is a ratio of two small means; its bootstrap interval is reported and may be wide.
- Part 2 answers for two error models only. A real, better model would not keep today's error pattern exactly (S)
  and would not have independent errors (N); the truth lies between.
- Part 2 reads weights, not held-out gain: 10% weight does not guarantee a stack gain.

## 10. Files
`narrow_h432.py` (part 1), `need_h432.py` (part 2), `gates_h432.sh`, `stage_h432.sh`. Reused at published bytes:
`cache_h422.py`, `meta_h416.py`, `train_h416.py`, `restate_h426.py` and the frozen H335 statistics.
