# City power realtime dispatch

> Which city power-demand model is most accurate while meeting both a real-time scoring budget and a retraining budget?

| Dataset | |
| --- | --- |
| Catalog id | `ca8b2f30cf07dd87` |
| Name | Power Consumption of Tetouan City with Weather |
| Source | uci machine learning repository `UCI-849` — <https://archive.ics.uci.edu/static/public/849/data.csv> |
| Shape | 10000 rows × 7 features (0 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh ca8b2f30cf07dd87` → CLEAR |

## Why this objective

Grid dispatch scores each 10-minute reading in real time and refits often, so the program must meet both a median one-row latency budget and a training-time budget.

Business constraint: **row_latency_p50_us max + fit_seconds max**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/10_city_power_realtime_dispatch` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/10_city_power_realtime_dispatch --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/10_city_power_realtime_dispatch            # all arms and seeds
```

Results land in `~/jope-data/case_study/10_city_power_realtime_dispatch/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Time-ordered data, but jope pipeline run validates on an unordered fold partition of train.csv, so this case does not exercise chronological (regime) validation
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
