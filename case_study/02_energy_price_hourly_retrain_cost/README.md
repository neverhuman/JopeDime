# Energy price hourly retrain cost

> When every hourly refit costs compute, which program minimizes forecast error plus the CPU cost of refitting?

| Dataset | |
| --- | --- |
| Catalog id | `14bc1b3a41c041b5` |
| Name | Hourly Day-Ahead Electricity Prices with Exogenous Variables |
| Source | github — nixtla `Nixtla/transfer-learning-time-series:datasets/electricity-short-with-ex-vars.csv` — <https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/electricity-short-with-ex-vars.csv> |
| Shape | 6720 rows × 11 features (1 categorical); train 5376, test 1344 |
| Licence | unknown |
| Time column | ds |
| Sealed check | `case_study/bin/check_sealed.sh 14bc1b3a41c041b5` → CLEAR |

## Why this objective

An hourly refit is paid in compute every hour, so the cost is a price per CPU second, not a cliff. A composite adds that price to scaled RMSE, and the search finds the cheapest accurate program.

Business constraint: **price per CPU second**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/02_energy_price_hourly_retrain_cost` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/02_energy_price_hourly_retrain_cost --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/02_energy_price_hourly_retrain_cost            # all arms and seeds
```

Results land in `~/jope-data/case_study/02_energy_price_hourly_retrain_cost/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Time-ordered data, but jope pipeline run validates on an unordered fold partition of train.csv, so this case does not exercise chronological (regime) validation
- CPU price is illustrative; set your own cost per CPU second.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
