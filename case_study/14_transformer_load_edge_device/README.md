# Transformer load edge device

> Which transformer load/temperature model is most accurate on a substation edge device with a model-size and latency budget?

| Dataset | |
| --- | --- |
| Catalog id | `6ce1759b1a18e332` |
| Name | Electricity Transformer Temperature and Load Measurements (ETTm1) |
| Source | github `skforecast-datasets:ETTm1` — <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv> |
| Shape | 10000 rows × 6 features (0 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 6ce1759b1a18e332` → CLEAR |

## Why this objective

A substation edge device forecasts transformer oil temperature from load readings. The model must fit the device's storage and answer each reading within a latency budget.

Business constraint: **model_bytes max + row_latency_p99_us max**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/14_transformer_load_edge_device` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/14_transformer_load_edge_device --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/14_transformer_load_edge_device            # all arms and seeds
```

Results land in `~/jope-data/case_study/14_transformer_load_edge_device/<UTC>/`: `SUMMARY.md` (champion per run,
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
