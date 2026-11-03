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
{
  "metric": "constrained",
  "fitness": {
    "metric": "pinball",
    "quantile": 0.9
  },
  "limits": [
    {
      "resource": "fit_seconds",
      "max": 0.49
    }
  ]
}
```

Calibrated from arm D on rng seed 1 (28 measured candidates; not a main seed). Rule: limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading. Train target σ = 0.0648294.

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

## Business context

- **Dataset provenance**: Polish Day-Ahead Electricity Market, Demand, Renewables and Weather (`127a1b5eccc09a84`, 9,901 hourly rows × 8 features; train 7,921, test 1,980) from [Zenodo record 20818296](https://zenodo.org/records/20818296/files/biogas_energy_dataset.csv?download=1).
- **Domain & SLA context**: In the European Single Day-Ahead Coupling (SDAC) and Polish Power Exchange (TGE), day-ahead auction gate closure occurs daily at 12:00 CET ([ENTSO-E SDAC Day-Ahead Market](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/); [TGE Day-Ahead Market Rules](https://tge.pl/electricity-dam)). Under-forecasting upward price spikes exposes participants to imbalance settlement charges on the Polish transmission system ([PSE Balancing Market](https://www.pse.pl/en/balancing-market)), which motivates an illustrative 9:1 asymmetric quantile loss (`{"metric":"pinball","quantile":0.9}`) paired with a hard retraining wall-clock ceiling (`fit_seconds` max).

_Business context contributed by agy_resource_fitness (Gemini) on 2026-11-03 (notes file sha256 744cf669fdc7…). The links were supplied by the contributor and were not independently fetched._
