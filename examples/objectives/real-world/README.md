# Real-world objectives on public data

Squared error is the right loss only when a miss in either direction costs the same and big misses
cost quadratically. The datasets below are public tables whose owners care about something else.
`benchmark.sh` fits each one three ways on the same training rows and scores the same held-out
rows with the dataset's objective:

| Arm | Command | What it shows |
| --- | --- | --- |
| RMSE-selected | `jopedime fit` | The default: methods chosen by out-of-fold RMSE. |
| objective-selected | `--fitness F --calibration none` | Selection by the objective alone. |
| objective + calibration | `--fitness F` | Selection plus the cross-fitted output map (the default with an objective). |

```sh
bash examples/objectives/real-world/fetch.sh          # data stays outside the checkout
bash examples/objectives/real-world/benchmark.sh      # or: benchmark.sh wind sp500
FETCH_FREMTPL=1 FREMTPL_SHA256=<digest you verified> bash examples/objectives/real-world/fetch.sh
```

`fetch.sh` writes to `$JOPE_FITNESS_DATA` (default `~/jope-data/fitness-real-world`), refuses a
directory inside the checkout, pins every download by sha256 and converts it to a header CSV.
`benchmark.sh` writes `target/fitness-real-world/results.{json,md}` and every intermediate file
(splits, models, receipts, predictions and evaluation requests) so each number can be re-derived.

## Datasets

| Name | Source and licence | Rows | Target | Objective | Why not RMSE |
| --- | --- | ---: | --- | --- | --- |
| `wind` | PMLB `503_wind` (MIT; StatLib, Haslett and Raftery 1989) | 6,574 | Daily mean wind speed at Malin Head, from 11 other Irish stations and the date | `pinball:quantile=0.1` | A firm commitment must be met 9 days in 10: the 10th percentile, not the mean. |
| `houses` | PMLB `537_houses` (MIT; Pace and Barry 1997, 1990 California census) | 20,640 | Median house value per block group | `mape:epsilon=1` | Valuation error is judged relative to price. |
| `pm10` | PMLB `522_pm10` (MIT; Norwegian Public Roads Administration via StatLib) | 500 | Hourly log PM10 concentration near a road | `pinball:quantile=0.9` | An air-quality warning needs an upper forecast. |
| `no2` | PMLB `547_no2` (same source) | 500 | Hourly log NO2 concentration | `asymmetric_absolute:under_cost=4,over_cost=1` | Missing a peak costs more than a false alarm. |
| `sp500` | Shiller monthly S&P 500 (ODC-PDDL-1.0, `datasets/s-and-p-500`) | 1,470 | Next month's total return, from valuation, rates, inflation and momentum | `sortino:cost=0.001,periods_per_year=12` (forward validation, gap 1) | Allocation is judged on downside risk after costs, not on return MSE. |
| `fremtpl` | freMTPL2freq (OpenML 41214, recorded CC0; CASdatasets upstream GPL≥2; optional, every 20th policy) | 33,900 | Claims per policy-year (`ClaimNb` capped at 4 over `Exposure` capped at 1) | `claims_frequency_exposure.json` | Frequency is priced per unit of exposure: exposure-weighted Poisson deviance. |

The S&P features are dividend, earnings and CAPE yields, the long rate, 12-month inflation and
12-month momentum, all known at the month in question. Shiller's prices are monthly averages,
which makes adjacent returns positively correlated, so last month's return is deliberately not a
feature and absolute ratios still overstate what a trader could realize. Compare the arms, not the
level. Each split is one deterministic split (chronological for `wind` and `sp500`, a fixed hash
of the row number otherwise), so small differences are within noise.

## Results

Each table is fitted three ways on the same training rows, with the same validation scheme, and
the same held-out rows are scored with that table's objective. The arms are: `--fitness rmse`
with no calibration, the objective with `--calibration none` (selection only), and the objective
with the default cross-fitted output map.

| Dataset | Objective | Train/test rows | RMSE-selected | Objective-selected | + calibration | Change vs RMSE-selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wind` | `pinball:quantile=0.1` (minimize) | 5,259 / 1,315 | 1.4101 | 1.4101 | 0.5169 | 63% lower loss |
| `houses` | `mape:epsilon=1` (minimize) | 16,513 / 4,127 | 0.2096 | 0.2096 | 0.1946 | 7% lower loss |
| `pm10` | `pinball:quantile=0.9` (minimize) | 400 / 100 | 0.2405 | 0.2405 | 0.1220 | 49% lower loss |
| `no2` | `asymmetric_absolute:under_cost=4,over_cost=1` (minimize) | 400 / 100 | 0.9748 | 0.9748 | 0.6289 | 35% lower loss |
| `sp500` | `sortino:cost=0.001,periods_per_year=12` (maximize) | 1,176 / 294 | 0.8938 | 0.8938 | 0.9332 | 4% higher |
| `fremtpl` | exposure-weighted Poisson deviance (minimize) | 27,120 / 6,780 | undefined | 0.4923 | 0.4754 | RMSE arm cannot be scored |

What the numbers say, and what they do not:

- **Selection alone changed the learner once in six.** On five tables the objective picked the
  same learner as RMSE, so the objective-selected predictions are identical. The exception is
  `fremtpl`: the RMSE choice (`rust.gam_boost`) predicts a negative claim rate for 68 of 6,780
  held-out policies, where a Poisson deviance is undefined, and the objective chose
  `rust.extra_trees`, whose rates are all positive.
- **The output map does the work.** `max(a + b·pred, floor)` moves a mean forecast to the act the
  objective rewards: a 10th or 90th percentile, a 4:1 cost-weighted level, a relative-error
  optimum or a position size. With the map, `sp500` and `fremtpl` also switched learners
  (`rust.gam_boost` and `rust.elasticnet` with a floor of 0.001), because selection compares
  calibrated validation losses.
- **These are single deterministic splits** with one seed. `pm10` and `no2` have 100 test rows,
  and the S&P figures use Shiller's monthly averages, so compare the arms, not the levels.
- **The `fremtpl` row was measured on the CASdatasets copy** (677,991 policies, the `freMTPL2freq.rda` in
  `dutangc/CASdatasets`, sha256 `82c8598513d9fa78226b6c7d271a9940eca4b8083e9e32320a75d377bdfe15a3`)
  converted locally, because OpenML was not reachable from the machine
  that produced this table. `fetch.sh` uses OpenML's 678,013-policy copy, so the every-20th sample
  differs slightly; expect nearby, not identical, numbers.

## Program search ranked by the objective

`search-benchmark.sh` repeats the comparison for the program search on the ten committed
regression datasets: `jope pipeline run` once ranking candidate programs by held-out RMSE and once
with `--fitness pinball:quantile=0.9`, same seeds and budget (64 evaluations, 2 cuts), each
selected program scored once on the sealed test split with `jope pipeline score --fitness`.

| Dataset | Test pinball (0.9), RMSE-ranked | Objective-ranked | Change |
| --- | ---: | ---: | ---: |
| `openml-23-cmc` | 0.07355 | 0.07541 | +2.5% |
| `openml-29-credit-approval` | 0.004132 | 0.004058 | −1.8% |
| `pmlb-229-pwlinear` | 0.02062 | 0.02384 | +15.6% |
| `pmlb-695-chatfield-4` | 0.02167 | 0.02002 | −7.6% |
| `pmlb-first-principles-supernovae-zg` | 0.001470 | 0.001545 | +5.1% |
| `rdatasets-agridat-bancroft-peanut-uniformity` | 0.04099 | 0.04941 | +20.5% |
| `rdatasets-stevedata-chile88` | 0.07281 | 0.07603 | +4.4% |
| `rdatasets-wooldridge-barium` | 0.04040 | 0.03558 | −11.9% |
| `socrata-cdc-food-safety-culture` | 0.008641 | 0.008509 | −1.5% |
| `uci-parkinsons-telemonitoring` | 0.007144 | 0.008988 | +25.8% |

The objective changed the selected program on all ten datasets but lowered the test loss on only
four (geometric mean +4.5%). Every candidate program is still fitted by its heads' own
squared-error estimators, so ranking mean predictors by a quantile loss mostly adds selection
noise. This is the same lesson as the selection arm above, at the level of written programs: the
objective has to change what is fitted or emitted (as the output map does), not only what is
ranked. Run with a debug binary through `JOPE_BIN=target/debug/jope`; one seed, so treat it as a
diagnostic rather than an effect size.

## More open datasets where the objective matters

These were checked against their publishers' pages; Kaggle-licensed tables are named for their
metrics but must not be redistributed.

| Dataset | Domain | Licence | Objective in practice |
| --- | --- | --- | --- |
| freMTPL2freq / freMTPL2sev (OpenML 41214 / 41215) | Motor insurance | OpenML: CC0; CASdatasets: GPL≥2 (conflicting; fetch, never vendor) | Exposure-weighted Poisson deviance, Gamma deviance, normalized Gini (Noll, Salzmann and Wüthrich 2018) |
| Bike Sharing (UCI 275) | Demand | CC BY 4.0 | Pinball / newsvendor; Kaggle used RMSLE |
| Individual household electric power (UCI 235) | Load | CC BY 4.0 | Pinball over quantiles (GEFCom2014 style) |
| Appliances energy (UCI 374) | Load | CC BY 4.0 | Pinball, MAE; chronological |
| Concrete compressive strength (UCI 165) | Safety | CC BY 4.0 | 5th percentile: Eurocode characteristic strength is a 5% fractile |
| Dow Jones Index (UCI 312) | Finance | CC BY 4.0 | Rank IC, Sharpe/Sortino of a weekly strategy |
| medical_charges (OpenML 42720) | Healthcare cost | CC0 | Gamma deviance weighted by discharges |
| house_sales, King County (OpenML 42092) | Real estate | CC0 on OpenML | RMSLE, MAPE |
| delays_zurich_transport (OpenML 42495) | Transport | CC0 | Asymmetric lateness cost, pinball |
| CAMELS (Zenodo, DOI 10.5065/D6MW2F4D) | Hydrology | CC BY 4.0 | NSE (R²), KGE |
| M5, Rossmann, Favorita, Allstate (Kaggle) | Retail, insurance | Competition rules; not redistributable | WRMSSE, RMSPE, NWRMSLE, MAE |
| GEFCom2014 | Energy | No licence stated; citation required | Pinball over 99 quantiles (Hong et al. 2016) |

Avoid FRED and the Ken French Data Library for this purpose: their terms restrict storage,
redistribution or use for training software.
