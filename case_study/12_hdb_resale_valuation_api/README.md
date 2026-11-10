# Hdb resale valuation api

> Which resale-price model has the lowest percentage error while meeting an API latency budget and a model-size budget?

| Dataset | |
| --- | --- |
| Catalog id | `f2d74feab0307e24` |
| Name | Singapore HDB Resale Flat Prices from January 2017 |
| Source | data.gov.sg `resale-flat-prices` — <https://storage.data.gov.sg/resale-flat-prices/resources/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards-2023-04-17T03-28-08Z.csv> |
| Shape | 10000 rows × 7 features (4 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh f2d74feab0307e24` → CLEAR |

## Why this objective

A public resale-valuation API answers one query at a time and is deployed in a small container, so latency and model size are both hard limits, and percentage error is what users judge.

Business constraint: **row_latency_p99_us max + model_bytes max**. Accuracy metric: `{"metric":"mape","epsilon":1.0}`.

## Objective (`objective.json`)

```json
{
  "metric": "constrained",
  "fitness": {
    "metric": "mape",
    "epsilon": 1.0
  },
  "limits": [
    {
      "resource": "row_latency_p99_us",
      "max": 17
    },
    {
      "resource": "model_bytes",
      "max": 600000
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.130219.

`baseline.json` ranks by the same accuracy metric alone, with non-binding limits, so the same
resource readings are recorded for every candidate in every arm.

## Arms and budget

| Arm | Generation 0 | Ranked by |
| --- | --- | --- |
| A | bootstrap + 11 random + DL seeds (≤ 11) | `objective.json` |
| B | bootstrap + 11 random | `objective.json` |
| C | bootstrap + 11 random + DL seeds (≤ 11) | `baseline.json` |
| D | bootstrap + 11 random | `baseline.json` |

Budget 96 evaluations (≈ generation 0 + 8–10 generations), 600 s per run; RNG seeds 7, 11, 13; `--threads 1`.

## Run

```sh
case_study/bin/run_case.sh case_study/12_hdb_resale_valuation_api --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/12_hdb_resale_valuation_api            # all arms and seeds
```

Results land in `~/jope-data/case_study/12_hdb_resale_valuation_api/<UTC>/`: `SUMMARY.md` (champion per run,
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

## Business context

- **Dataset provenance**: Singapore HDB Resale Flat Prices from January 2017 (`f2d74feab0307e24`, 10,000 rows × 7 features; train 8,000, test 2,000) from [data.gov.sg resale-flat-prices](https://storage.data.gov.sg/resale-flat-prices/resources/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards-2023-04-17T03-28-08Z.csv) ([Singapore HDB Resale Statistics](https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics)).
- **Domain & SLA context**: Automated property valuation APIs commonly evaluate relative percentage error (`{"metric":"mape","epsilon":1.0}`) while enforcing joint deployment constraints on single-request tail latency (`row_latency_p99_us` max) and serialized artifact footprint (`model_bytes` max).

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
