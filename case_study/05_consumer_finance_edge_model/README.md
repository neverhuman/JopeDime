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
{
  "metric": "constrained",
  "fitness": {
    "metric": "rmse"
  },
  "limits": [
    {
      "resource": "model_bytes",
      "max": 310000
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.0271489.

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

## Business context

- **Dataset provenance**: 2022 Survey of Consumer Finances — Summary Extract (`9721aa73ef1df237`, 10,000 rows × 346 features; train 8,000, test 2,000) from the [Federal Reserve Board 2022 Survey of Consumer Finances (SCFP2022)](https://www.federalreserve.gov/econres/files/scfp2022excel.zip) ([Federal Reserve SCF Overview](https://www.federalreserve.gov/econres/scfindex.htm)).
- **Domain & SLA context**: Deploying models over 346 household balance-sheet attributes into edge Workers or client bundles requires bounding serialized artifact size (`model_bytes` max; see e.g. [Cloudflare Workers Script Size Limits](https://developers.cloudflare.com/workers/platform/limits/)) while minimizing `{"metric":"rmse"}`.

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
