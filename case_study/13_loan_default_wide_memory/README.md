# Loan default wide memory

> With 735 features, which program is most accurate within a training-memory budget?

| Dataset | |
| --- | --- |
| Catalog id | `053a91a333d4f3a7` |
| Name | LoanDefaultPrediction |
| Source | openml `44067` — <https://www.openml.org/d/44067> |
| Shape | 1999 rows × 735 features (0 categorical); train 1000, test 999 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 053a91a333d4f3a7` → CLEAR |

## Why this objective

735 features on a memory-limited training worker: a hard cap on peak training heap, best RMSE inside it.

Business constraint: **fit_peak_heap_bytes max**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/13_loan_default_wide_memory` sets the limits/prices from a declared calibration run before the main runs.

`baseline.json` ranks by the same accuracy metric alone, with non-binding limits, so the same
resource readings are recorded for every candidate in every arm.

## Arms and budget

| Arm | Generation 0 | Ranked by |
| --- | --- | --- |
| A | bootstrap + 11 random + DL seeds (≤ 11) | `objective.json` |
| B | bootstrap + 11 random | `objective.json` |
| C | bootstrap + 11 random + DL seeds (≤ 11) | `baseline.json` |
| D | bootstrap + 11 random | `baseline.json` |

Budget 96 evaluations (≈ generation 0 + 8–10 generations), 3600 s per run; RNG seeds 7, 11, 13; `--threads 1`.

## Run

```sh
case_study/bin/run_case.sh case_study/13_loan_default_wide_memory --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/13_loan_default_wide_memory            # all arms and seeds
```

Results land in `~/jope-data/case_study/13_loan_default_wide_memory/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Corpus copy is 1,999 rows (1,000 train).
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
