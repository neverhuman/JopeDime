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
{
  "metric": "custom",
  "name": "error_plus_serving_cost",
  "direction": "minimize",
  "expression": {
    "op": "add",
    "left": {
      "op": "abs",
      "arg": {
        "op": "sub",
        "left": {
          "op": "y"
        },
        "right": {
          "op": "pred"
        }
      }
    },
    "right": {
      "op": "mul",
      "left": {
        "op": "constant",
        "value": 0.00067
      },
      "right": {
        "op": "resource",
        "name": "predict_us_per_row"
      }
    }
  },
  "aggregation": {
    "kind": "mean"
  }
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.173006.

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

## Business context

- **Dataset provenance**: `modeldata/hotel_rates` (`78dc8d077285ce09`, 15,402 rows × 26 features; train 12,321, test 3,081) from [Rdatasets modeldata/hotel_rates.csv](https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hotel_rates.csv) ([tidymodels modeldata documentation](https://modeldata.tidymodels.org/reference/hotel_rates.html), derived from [Antonio, de Almeida & Nunes, Data in Brief 2019](https://doi.org/10.1016/j.dib.2018.11.126)).
- **Domain & SLA context**: Travel and hospitality search engines score room rates across many candidate properties and date combinations per user search, so pricing per-row batch prediction latency (`predict_us_per_row`) against `{"metric":"mae"}` trades off serving CPU cost against rate error.

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
