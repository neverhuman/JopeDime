# Bike rebalancing latency

> Which station-demand model minimizes newsvendor cost (a missing bike costs 3, an idle one 1) within an online latency budget?

| Dataset | |
| --- | --- |
| Catalog id | `bbc0bb2e8fcf2ba6` |
| Name | Bike Sharing Hour |
| Source | uci `bike-sharing-hour` — <https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip> |
| Shape | 1999 rows × 16 features (0 categorical); train 1000, test 999 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh bbc0bb2e8fcf2ba6` → CLEAR |

## Why this objective

Station rebalancing trucks need a demand number per station on request. A missing bike costs 3 and an idle one 1 (newsvendor), within an online latency budget.

Business constraint: **row_latency_p99_us max**. Accuracy metric: `{"metric":"asymmetric_absolute","under_cost":3.0,"over_cost":1.0}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/11_bike_rebalancing_latency` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/11_bike_rebalancing_latency --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/11_bike_rebalancing_latency            # all arms and seeds
```

Results land in `~/jope-data/case_study/11_bike_rebalancing_latency/<UTC>/`: `SUMMARY.md` (champion per run,
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
