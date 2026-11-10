# Ct slice wide training memory

> On 384 wide features and 40k rows, which program trades accuracy against peak training memory best?

| Dataset | |
| --- | --- |
| Catalog id | `2056a3aad9468d87` |
| Name | Relative Location of CT Slices on Axial Images (UCI 206 slice_localization) |
| Source | uci `206` — <https://archive.ics.uci.edu/dataset/206> |
| Shape | 50000 rows × 384 features (0 categorical); train 39877, test 10123 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 2056a3aad9468d87` → CLEAR |

## Why this objective

With 384 wide features and about 40k training rows, kernel and expansion stages can multiply the working set. On a shared training host memory is a cost, so a composite prices peak training heap next to scaled RMSE.

Business constraint: **price per heap byte**. Accuracy metric: `{"metric":"rmse"}`.

## Objective (`objective.json`)

```json
{
  "metric": "composite",
  "terms": [
    {
      "weight": 1.0,
      "scale": 0.23,
      "fitness": {
        "metric": "rmse"
      }
    },
    {
      "weight": 0.1,
      "scale": 460000000,
      "fitness": {
        "metric": "resource",
        "name": "fit_peak_heap_bytes"
      }
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (20 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.23405.

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
case_study/bin/run_case.sh case_study/04_ct_slice_wide_training_memory --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/04_ct_slice_wide_training_memory            # all arms and seeds
```

Results land in `~/jope-data/case_study/04_ct_slice_wide_training_memory/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Peak heap is process heap growth from the jope binary's counting allocator, valid because candidates run one at a time
- The largest dataset here; expect the slowest runs.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->

## Business context

- **Dataset provenance**: Relative Location of CT Slices on Axial Images (`2056a3aad9468d87`, 50,000 rows × 384 features; train 39,877, test 10,123) from [UCI Machine Learning Repository dataset 206](https://archive.ics.uci.edu/dataset/206) ([Graf et al., 2011](https://lms.uni-passau.de/)).
- **Domain & SLA context**: Each row encodes 384 spatial and density histogram descriptors across 39,877 training slices (~12.2 MB raw `f64` feature matrix). Multi-stage basis expansions and kernel learners allocate large intermediate matrices during training, so pricing `fit_peak_heap_bytes` alongside `{"metric":"rmse"}` penalizes peak heap growth under memory-capped container limits ([Kubernetes Memory Resource Limits](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)).

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
