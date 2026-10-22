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
bash examples/objectives/real-world/fetch.sh                  # data stays outside the checkout
FETCH_FREMTPL=1 bash examples/objectives/real-world/fetch.sh   # optional: also freMTPL2freq
bash examples/objectives/real-world/benchmark.sh              # or: benchmark.sh wind sp500
```

`fetch.sh` writes to `$JOPE_FITNESS_DATA` (default `~/jope-data/fitness-real-world`), refuses a
directory inside the checkout, pins every download by sha256 and converts it to a header CSV.
`benchmark.sh` writes the summary to `target/fitness-real-world/results.{json,md}` and every
intermediate file (splits, models, predictions and evaluation requests) to
`$JOPE_FITNESS_DATA/work/`, next to the data and outside the checkout, so each number can be
re-derived.

## Datasets

| Name | Source and licence | Rows | Target | Objective | Why not RMSE |
| --- | --- | ---: | --- | --- | --- |
| `wind` | PMLB `503_wind` (MIT; StatLib, Haslett and Raftery 1989) | 6,574 | Daily mean wind speed at Malin Head, from 11 other Irish stations and the date | `pinball:quantile=0.1` | A firm commitment must be met 9 days in 10: the 10th percentile, not the mean. |
| `houses` | PMLB `537_houses` (MIT; Pace and Barry 1997, 1990 California census) | 20,640 | Median house value per block group | `mape:epsilon=1` | Valuation error is judged relative to price. |
| `pm10` | PMLB `522_pm10` (MIT; Norwegian Public Roads Administration via StatLib) | 500 | Hourly log PM10 concentration near a road | `pinball:quantile=0.9` | An air-quality warning needs an upper forecast. |
| `no2` | PMLB `547_no2` (same source) | 500 | Hourly log NO2 concentration | `asymmetric_absolute:under_cost=4,over_cost=1` | Missing a peak costs more than a false alarm. |
| `sp500` | Shiller monthly S&P 500 (ODC-PDDL-1.0, `datasets/s-and-p-500`) | 1,469 | Next month's total return, from valuation, rates, inflation and momentum | `sortino:cost=0.001,periods_per_year=12` (forward validation, gap 1) | Allocation is judged on downside risk after costs, not on return MSE. |
| `fremtpl` | freMTPL2freq (OpenML 41214, recorded CC0; CASdatasets upstream GPL≥2; optional, every 20th policy) | 33,901 | Claims per policy-year (`ClaimNb` capped at 4 over `Exposure` capped at 1) | `claims_frequency_exposure.json` | Frequency is priced per unit of exposure: exposure-weighted Poisson deviance. |

The S&P features are the dividend and earnings yields (dividends and earnings lagged three months,
as they are reported after the quarter), Shiller's CAPE yield, the long rate, 12-month inflation
lagged one month (CPI is published about two weeks after its month) and 12-1 momentum. Shiller's
prices are monthly averages, which makes adjacent returns positively correlated, so no feature
contains the latest month's return, and absolute ratios still overstate what a trader could
realize. Compare the arms, not the level. Each split is one deterministic split (chronological for `wind` and `sp500`, a fixed hash
of the row number otherwise), so small differences are within noise.

## Results

Each table is fitted three ways on the same training rows, with the same validation scheme, and
the same held-out rows are scored with that table's objective. The arms are: `--fitness rmse`
with no calibration, the objective with `--calibration none` (selection only), and the objective
with the default cross-fitted output map. Measured 2026-10-22 with a release `jopedime` on
x86_64 Linux, default candidates and five folds.

| Dataset | Objective | Train/test rows | RMSE-selected | Objective-selected | + calibration | Change vs RMSE-selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wind` | `pinball:quantile=0.1` (minimize) | 5,259 / 1,315 | 1.4101 | 1.4101 | 0.5169 | 63% lower loss |
| `houses` | `mape:epsilon=1` (minimize) | 16,513 / 4,127 | 0.2096 | 0.2096 | 0.1946 | 7% lower loss |
| `pm10` | `pinball:quantile=0.9` (minimize) | 400 / 100 | 0.2405 | 0.2405 | 0.1220 | 49% lower loss |
| `no2` | `asymmetric_absolute:under_cost=4,over_cost=1` (minimize) | 400 / 100 | 0.9748 | 0.9748 | 0.6289 | 35% lower loss |
| `sp500` | `sortino:cost=0.001,periods_per_year=12` (maximize) | 1,175 / 294 | 0.7831 | 1.1553 | 0.8392 | 7% higher (objective-selected: 48% higher) |
| `fremtpl` | exposure-weighted Poisson deviance (minimize) | 27,120 / 6,781 | undefined | 0.7892 | 0.6365 | RMSE arm cannot be scored; 19% below objective-selected |

What the numbers say, and what they do not:

- **Selection alone changed the learner twice in six.** On four tables the objective picked the
  same learner as RMSE, so the objective-selected predictions are identical. On `fremtpl` the RMSE
  choice (`rust.gam_boost`) predicts a non-positive claim rate for 80 of 6,781 held-out policies,
  where a Poisson deviance is undefined, and the objective chose `rust.extra_trees`, whose rates
  are all positive. On `sp500` the objective chose `rust.elasticnet` over `rust.extra_trees`,
  raising the test Sortino ratio from 0.783 to 1.155.
- **The output map does the work.** `max(a + b·pred, floor)` moves a mean forecast to the act the
  objective rewards: a 10th or 90th percentile, a 4:1 cost-weighted level or a relative-error
  optimum. With the map, `fremtpl` switched back to `rust.gam_boost`, shrunk (slope 0.14) and
  floored at 0.001, because selection compares calibrated validation losses.
- **The map is not always the best act.** On `sp500` the calibrated arm kept `rust.extra_trees`
  with a map that is nearly a constant full position (slope 0.001, intercept 1.01): buy-and-hold,
  which scored best on the forward validation folds (whose calibration rows are embargoed by the
  gap). On the test months it reached 0.839, above RMSE selection's 0.783 but below the
  uncalibrated elastic net's 1.155.
- **These are single deterministic splits** with one seed. `pm10` and `no2` have 100 test rows,
  and the S&P figures use Shiller's monthly averages, so compare the arms, not the levels.
- **`fremtpl` is OpenML's 678,013-policy file**, pinned by sha256 in `fetch.sh` (its md5 matches
  the checksum OpenML publishes, `f8875568bf0ca622929105197e2db613`); every 20th policy gives
  33,901 rows.

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
