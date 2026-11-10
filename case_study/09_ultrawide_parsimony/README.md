# Ultrawide parsimony

> With 1,142 features and 228 rows, which program balances accuracy against auditability (program size)? (structural, reproducible)

| Dataset | |
| --- | --- |
| Catalog id | `9ae61bc7da3ba01c` |
| Name | mtp2 |
| Source | openml `430` — <https://www.openml.org/d/430> |
| Shape | 274 rows × 1142 features (0 categorical); train 228, test 46 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 9ae61bc7da3ba01c` → CLEAR |

## Why this objective

With 1,142 features and only 228 training rows, overfitting and opacity are the risks. Pricing program nodes keeps the champion small enough to audit. Structural and reproducible.

Business constraint: **price per program node**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
{
  "metric": "composite",
  "terms": [
    {
      "weight": 1.0,
      "scale": 0.18,
      "fitness": {
        "metric": "rmse"
      }
    },
    {
      "weight": 0.1,
      "scale": 4,
      "fitness": {
        "metric": "resource",
        "name": "program_nodes"
      }
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.177955.

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
case_study/bin/run_case.sh case_study/09_ultrawide_parsimony --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/09_ultrawide_parsimony            # all arms and seeds
```

Results land in `~/jope-data/case_study/09_ultrawide_parsimony/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Very small sample; variance across seeds is expected to be large.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->

## Business context

- **Dataset provenance**: OpenML `mtp2` (`9ae61bc7da3ba01c`, 274 rows × 1,142 features; train 228, test 46) from [OpenML dataset 430](https://www.openml.org/d/430).
- **Domain & SLA context**: In ultrawide $p \gg n$ molecular/tabular settings (1,142 features vs. 228 training rows), unpenalized symbolic/pipeline search can bloat expression trees and overfit ([La Cava et al., "Contemporary Symbolic Regression Methods and their Relative Performance", NeurIPS Datasets & Benchmarks 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c0c7c76d30bd3dcaefc96f40275bdc0a-Abstract-round1.html)). Pricing structural size (`program_nodes`) alongside `{"metric":"rmse"}` enforces parsimony.

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
