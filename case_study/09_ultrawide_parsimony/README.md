# Ultrawide parsimony

> With 1,142 features and 228 rows, which program balances accuracy against auditability (program size)? (structural, reproducible)

| Dataset | |
| --- | --- |
| Catalog id | `9ae61bc7da3ba01c` |
| Name | mtp2 |
| Source | openml `430` — <https://www.openml.org/d/430> |
| Shape | 274 rows × 1142 features (0 categorical); train 228, test 46 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 9ae61bc7da3ba01c` → CLEAR |

## Why this objective

With 1,142 features and only 228 training rows, overfitting and opacity are the risks. Pricing program nodes keeps the champion small enough to audit. Structural and reproducible.

Business constraint: **price per program node**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/09_ultrawide_parsimony` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/09_ultrawide_parsimony --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/09_ultrawide_parsimony            # all arms and seeds
```

Results land in `~/jope-data/case_study/09_ultrawide_parsimony/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Very small sample; variance across seeds is expected to be large.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
