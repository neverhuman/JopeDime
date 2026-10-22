# Fitness recipe library

Every JSON recipe is accepted unchanged by CLI `--fitness @path`, Rust `Fitness::parse`,
API `options.fitness`, and MCP `regression_fit_predict` arguments. Replace illustrative units
and prices with your training-only, predeclared business assumptions.

| Recipe | Real-world decision | What it rewards |
| --- | --- | --- |
| `sensor_robust.json` | Industrial sensors with occasional spikes | Huber delta 0.5 limits outlier influence. |
| `inventory_p90.json` | Stock replenishment with expensive stockouts | 90th-quantile pinball selection. |
| `grid_underforecast.json` | Electricity load scheduling | Underforecast cost 5, overforecast cost 1. |
| `fab_tail_error.json` | Semiconductor metrology excursions | 99th percentile of absolute error. |
| `process_deadband.json` | Process control with a measurement tolerance | Squared error only beyond a 0.005 deadband. |
| `property_relative_error.json` | Valuation across price scales | Relative error with denominator floor 1. |
| `revenue_wape.json` | Aggregate revenue forecast accuracy | Total absolute error relative to absolute target volume. |
| `sales_log_error.json` | Nonnegative sales over a large dynamic range | Root mean squared log1p error. |
| `claims_count.json` | Insurance claim-frequency forecasts | Poisson deviance; nonnegative targets, positive forecasts. |
| `positive_severity.json` | Strictly positive claim-size forecasts | Gamma deviance. |
| `aggregate_claims.json` | Zero-inclusive aggregate insurance costs | Tweedie deviance at power 1.5. |
| `portfolio_sortino.json` | Downside-sensitive strategy selection | Sortino on realized strategy returns after turnover cost. |
| `portfolio_sharpe.json` | Volatility-sensitive strategy selection | Annualized Sharpe after turnover cost. |
| `portfolio_net_return.json` | Strategy profitability before risk adjustment | Mean net return after turnover cost. |
| `quality_composite.json` | Balance average and worst metrology error | 80% scaled MAE and 20% scaled max error. |
| `inventory_profit.json` | Order quantity selection | Revenue 10 per fulfilled unit minus order cost 4 and shortage cost 2. Domain assumes nonnegative quantities. |
| `large_target_weight.json` | Prioritize high-volume observations | Mean `(1+abs(y))*abs(y-pred)`. This is a weighted cost, not normalized weighted MAE. |
| `demand_expectile.json` | Penalize large shortages especially strongly | Asymmetric squared error at expectile 0.9. |

`evaluation_request.json` exercises an asymmetric cost with hand-checkable output 17/3.
`api_request.jsonl` contains a complete inline CSV fit/predict request.
`train.csv` and `predict.csv` are synthetic CLI smoke fixtures.

Run `bash examples/fitness/run.sh` from the repository root. It creates its outputs under
`target/fitness-example/`, verifies a custom CLI fit/predict, evaluates the cost recipe, and
executes the equivalent API request. For return recipes, supply chronological return data and
`--validation forward --gap N`; see [the full contract](../../docs/regression-fitness.md).
