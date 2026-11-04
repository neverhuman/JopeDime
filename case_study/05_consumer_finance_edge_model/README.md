# Consumer finance edge model

> Which program is most accurate while fitting in an on-device model budget? (structural, reproducible)

| Dataset | |
| --- | --- |
| Catalog id | `9721aa73ef1df237` |
| Name | 2022 Survey of Consumer Finances — Summary Extract |
| Source | federal reserve board `SCFP2022` — <https://www.federalreserve.gov/econres/files/scfp2022excel.zip> |
| Shape | 10000 rows × 346 features (0 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 9721aa73ef1df237` → CLEAR |

## Why this objective

An on-device financial-advice app ships the model inside the app bundle, so the model's serialized size is a hard limit. model_bytes is structural, so this case reproduces bit-for-bit on any host.

Business constraint: **model_bytes max**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/05_consumer_finance_edge_model` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/05_consumer_finance_edge_model --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/05_consumer_finance_edge_model            # all arms and seeds
```

Results land in `~/jope-data/case_study/05_consumer_finance_edge_model/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- model_bytes is the canonical JSON serialization of the fitted model, a size proxy rather than the exact size of a native artifact.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
