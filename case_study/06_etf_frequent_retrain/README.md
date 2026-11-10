# Etf frequent retrain

> For a price model retrained every session under drift, which program minimizes error plus training time?

| Dataset | |
| --- | --- |
| Catalog id | `f285fd52a7c14111` |
| Name | Taiwan 0050 ETF Historical OHLCV |
| Source | github `tw-stocker-0050` — <https://raw.githubusercontent.com/voidful/tw_stocker/main/data/0050.csv> |
| Shape | 10000 rows × 3 features (0 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | Datetime |
| Sealed check | `case_study/bin/check_sealed.sh f285fd52a7c14111` → CLEAR |

## Why this objective

Under drift a price model is refitted every session. Training time is a recurring cost, priced per second in a composite with scaled RMSE.

Business constraint: **price per training second**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
{
  "metric": "composite",
  "terms": [
    {
      "weight": 1.0,
      "scale": 0.37,
      "fitness": {
        "metric": "rmse"
      }
    },
    {
      "weight": 0.1,
      "scale": 0.14,
      "fitness": {
        "metric": "resource",
        "name": "fit_seconds"
      }
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.369024.

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
case_study/bin/run_case.sh case_study/06_etf_frequent_retrain --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/06_etf_frequent_retrain            # all arms and seeds
```

Results land in `~/jope-data/case_study/06_etf_frequent_retrain/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Time-ordered data, but jope pipeline run validates on an unordered fold partition of train.csv, so this case does not exercise chronological (regime) validation
- Corpus copy keeps 3 features (Datetime, Volume, Dividends); OHLC were dropped at ingest.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->

## Business context

- **Dataset provenance**: Taiwan 0050 ETF Historical OHLCV (`f285fd52a7c14111`, 10,000 rows × 3 features; train 8,000, test 2,000) from [GitHub voidful/tw_stocker 0050.csv](https://raw.githubusercontent.com/voidful/tw_stocker/main/data/0050.csv), tracking the Yuanta/P-shares Taiwan Top 50 ETF ([Taiwan Stock Exchange 0050](https://www.twse.com.tw/en/products/securities/etf/products/domestic.html)).
- **Domain & SLA context**: Walk-forward financial regime adaptation refits the pipeline across rolling market windows ([Lo, "Adaptive Markets Hypothesis", JPM 2004](https://doi.org/10.3905/jpm.2004.442611)). Pricing `fit_seconds` directly against `{"metric":"rmse"}` favors fast-refitting pipelines over slow nonlinear solvers when accuracy gains are marginal.

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
