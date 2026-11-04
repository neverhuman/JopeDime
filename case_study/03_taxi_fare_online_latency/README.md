# Taxi fare online latency

> Which fare model is most accurate while quoting a single trip within a p99 latency budget?

| Dataset | |
| --- | --- |
| Catalog id | `7394ac9e4f167075` |
| Name | NYC Green Taxi Trip Sample with Fare and Distance |
| Source | socrata `sp2f-wtwu` — <https://data.cityofnewyork.us/api/v3/views/sp2f-wtwu/export.csv?accessType=DOWNLOAD> |
| Shape | 10000 rows × 11 features (0 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | lpep_pickup_datetime |
| Sealed check | `case_study/bin/check_sealed.sh 7394ac9e4f167075` → CLEAR |

## Why this objective

A fare quote is served one trip at a time inside a booking request, so what matters is one-row p99 latency, not batch throughput. The search may use any program whose p99 stays within the budget.

Business constraint: **row_latency_p99_us max**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/03_taxi_fare_online_latency` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/03_taxi_fare_online_latency --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/03_taxi_fare_online_latency            # all arms and seeds
```

Results land in `~/jope-data/case_study/03_taxi_fare_online_latency/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Single-row latency is measured on up to 256 held-out rows, nearest rank
- Trip timestamps are numeric-encoded in the corpus copy.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
