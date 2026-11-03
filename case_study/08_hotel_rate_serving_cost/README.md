# Hotel rate serving cost

> Which room-rate model minimizes absolute error plus the serving cost of each prediction?

| Dataset | |
| --- | --- |
| Catalog id | `78dc8d077285ce09` |
| Name | modeldata hotel_rates, avg_price_per_room |
| Source | rdatasets `modeldata/hotel_rates` — <https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hotel_rates.csv> |
| Shape | 15402 rows × 26 features (9 categorical); train 12321, test 3081 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 78dc8d077285ce09` → CLEAR |

## Why this objective

A revenue-management API is billed per call and per compute, so each prediction's cost is its absolute error plus a price per microsecond of serving time, written directly as a custom row expression.

Business constraint: **serving cost per row**. Accuracy metric: `{"metric":"mae"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/08_hotel_rate_serving_cost` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/08_hotel_rate_serving_cost --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/08_hotel_rate_serving_cost            # all arms and seeds
```

Results land in `~/jope-data/case_study/08_hotel_rate_serving_cost/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Price per microsecond is illustrative.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
