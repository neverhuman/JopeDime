# Energy price fast retrain

> Which program gives the best P90 day-ahead price forecast that can be retrained within the window before gate closure?

| Dataset | |
| --- | --- |
| Catalog id | `127a1b5eccc09a84` |
| Name | Polish Day-Ahead Electricity Market, Demand, Renewables and Weather |
| Source | zenodo `10.5281/zenodo.20818296` — <https://zenodo.org/records/20818296/files/biogas_energy_dataset.csv?download=1> |
| Shape | 9901 rows × 8 features (0 categorical); train 7921, test 1980 |
| Licence | unknown |
| Time column | ts |
| Sealed check | `case_study/bin/check_sealed.sh 127a1b5eccc09a84` → CLEAR |

## Why this objective

Day-ahead bids are due before gate closure, and the model is refitted on the latest prices every day. Under-forecasting a price spike is the expensive mistake, hence P90 pinball. A program that is accurate but cannot be retrained inside the window is useless, hence a hard limit on fit_seconds rather than a soft price.

Business constraint: **fit_seconds max**. Accuracy metric: `{"metric":"pinball","quantile":0.9}`.

## Objective (`objective.json`)

```json
(not yet calibrated: run case_study/bin/calibrate.sh on this case)
```

Calibration pending: `case_study/bin/calibrate.sh case_study/01_energy_price_fast_retrain` sets the limits/prices from a declared calibration run before the main runs.

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
case_study/bin/run_case.sh case_study/01_energy_price_fast_retrain --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/01_energy_price_fast_retrain            # all arms and seeds
```

Results land in `~/jope-data/case_study/01_energy_price_fast_retrain/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Time-ordered data, but jope pipeline run validates on an unordered fold partition of train.csv, so this case does not exercise chronological (regime) validation
- fit_seconds is a reading of this host under its load; the limit is calibrated on this host and must be recalibrated elsewhere
- Corpus copy is 9,901 rows (Jul 2024-Sep 2025).
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->
