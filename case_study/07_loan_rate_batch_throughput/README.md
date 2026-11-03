# Loan rate batch throughput

> Which interest-rate model is most accurate while repricing the book at a minimum batch throughput?

| Dataset | |
| --- | --- |
| Catalog id | `18f905fa597f92d3` |
| Name | OpenIntro loans_full_schema (Lending Club), interest_rate |
| Source | rdatasets `openintro/loans_full_schema` — <https://vincentarelbundock.github.io/Rdatasets/csv/openintro/loans_full_schema.csv> |
| Shape | 10000 rows × 41 features (10 categorical); train 8000, test 2000 |
| Licence | unknown |
| Time column | none |
| Sealed check | `case_study/bin/check_sealed.sh 18f905fa597f92d3` → CLEAR |

## Why this objective

Nightly repricing scores the whole loan book in one batch window, so the constraint is a minimum batch throughput in rows per second. Among programs that meet it, MAE in rate points decides.

Business constraint: **predict_rows_per_second min**. Accuracy metric: `{"metric":"mae"}`.

## Objective (`objective.json`)

```json
{
  "metric": "constrained",
  "fitness": {
    "metric": "mae"
  },
  "limits": [
    {
      "resource": "predict_rows_per_second",
      "min": 180000
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.194525.

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
case_study/bin/run_case.sh case_study/07_loan_rate_batch_throughput --smoke    # one seed, tiny budget
case_study/bin/run_case.sh case_study/07_loan_rate_batch_throughput            # all arms and seeds
```

Results land in `~/jope-data/case_study/07_loan_rate_batch_throughput/<UTC>/`: `SUMMARY.md` (champion per run,
best-so-far objective by generation) and `generations.csv`.

## How to read the result

- **A vs C**: the objective's effect. Does the champion become cheaper (faster, smaller,
  lower-latency) at a stated accuracy cost?
- **A vs B**: the generation-0 seeds' effect under the business objective.
- A run marked `resource_infeasible` in `observations.json` was refused by a limit, never
  ranked.

## Caveats

- Throughput is measured on the held-out batch, median of three passes.
- Measured readings are host- and load-dependent; `run.json` records the host, load and limits of each run.

<!-- notes -->

## Business context

- **Dataset provenance**: OpenIntro `loans_full_schema` (`18f905fa597f92d3`, 10,000 rows × 41 features; train 8,000, test 2,000) from [Rdatasets openintro/loans_full_schema.csv](https://vincentarelbundock.github.io/Rdatasets/csv/openintro/loans_full_schema.csv) ([OpenIntro LendingClub dataset documentation](https://www.openintro.org/data/index.php?data=loans_full_schema)).
- **Domain & SLA context**: Nightly batch credit repricing across large consumer loan portfolios must complete within fixed batch processing windows, motivating a hard minimum throughput floor (`predict_rows_per_second` min) while minimizing `{"metric":"mae"}`.

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
