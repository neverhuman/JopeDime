# Objective recipe library

Every JSON recipe here is accepted unchanged by the CLI (`--fitness @path`), Rust
(`Fitness::parse`), JSON-RPC (`options.fitness`) and MCP (`regression_fit_predict`). The program
search (`jope pipeline run --fitness @path`) accepts the recipes that neither read a column nor
depend on row order. Costs and prices are illustrative: replace them with your own predeclared
business values. Recipes that read a column (`exposure`, `unit_margin`, …) need that numeric
column in the training CSV; it is never used as a feature.

## Retail and operations

| Recipe | Decision | What it rewards |
| --- | --- | --- |
| `inventory_p90.json` | Replenishment with expensive stock-outs | 90th-quantile pinball loss. |
| `bike_rebalancing.json` | Bikes to stage per station | Newsvendor: a missing bike costs 3, an idle one 1. |
| `newsvendor_margin_columns.json` | Order quantity per SKU | Shortage costs each row's `unit_margin`, excess its `holding_cost`. |
| `inventory_profit.json` | Order quantity selection | Revenue 10 per fulfilled unit minus order cost 4 and shortage cost 2 (maximized). |
| `stockout_fixed_penalty.json` | Service-level contracts | A fixed penalty of 25 per stock-out plus the absolute error. |
| `demand_expectile.json` | Penalize large shortages strongly | Asymmetric squared error at expectile 0.9. |
| `retail_rmspe.json` | Store sales across sizes | Root mean squared percentage error (Rossmann's metric), floor 1. |
| `weekly_mase.json` | Daily demand with weekly seasonality | MASE against the 7-day seasonal naive; chronological rows. |
| `revenue_wape.json` | Aggregate revenue accuracy | Total absolute error over total volume. |
| `sales_log_error.json` | Nonnegative sales over a large range | Root mean squared log1p error. |
| `large_target_weight.json` | Prioritize high-volume rows | Mean `(1+abs(y))*abs(y-pred)`. |

## Insurance, healthcare and pricing

| Recipe | Decision | What it rewards |
| --- | --- | --- |
| `claims_frequency_exposure.json` | Claim frequency per policy-year | Poisson deviance weighted by the `exposure` column. |
| `claims_count.json` | Claim counts | Poisson deviance. |
| `positive_severity.json` | Claim size | Gamma deviance. |
| `aggregate_claims.json` | Pure premium with many zeros | Tweedie deviance, power 1.5. |
| `pure_premium_rank_and_fit.json` | Pricing that must fit and rank | Exposure-weighted Tweedie deviance plus exposure-weighted normalized Gini. |
| `hospital_cost_by_volume.json` | Average cost per hospital and diagnosis | Gamma deviance weighted by `discharges`. |
| `property_relative_error.json` | Valuation across price levels | Relative error with denominator floor 1. |
| `fraud_loss_ranking.json` | Investigation queue | Normalized Gini of losses ranked by prediction. |
| `marketing_top_decile.json` | Campaign targeting | Share of total value captured by the top 10% of rows. |

## Energy, environment and safety

| Recipe | Decision | What it rewards |
| --- | --- | --- |
| `grid_underforecast.json` | Electricity load scheduling | Under-forecast cost 5, over-forecast cost 1. |
| `grid_linex.json` | Reserve planning | LinEx with a = 0.5: shortfalls grow exponentially costly. |
| `imbalance_price_cost.json` | Balancing-market exposure | Absolute error times each row's `imbalance_price`. |
| `streamflow_kge.json` | River-flow forecasting | Kling–Gupta efficiency. |
| `pollution_exceedance.json` | Air-quality alerts | 90th-quantile loss plus missed exceedances of 50. |
| `concrete_safe_strength.json` | Structural safety | 5th-quantile loss: over-predicting strength is the dangerous error. |
| `eta_lateness_cvar.json` | Delivery promises | Mean lateness in the worst 5% of deliveries (CVaR). |
| `sensor_robust.json` | Sensors with spikes | Huber loss, delta 0.5. |
| `fab_tail_error.json` | Semiconductor metrology | 99th percentile of absolute error. |
| `process_deadband.json` | Process control tolerance | Squared error beyond a 0.005 deadband. |
| `quality_composite.json` | Average and worst metrology error | 80% scaled MAE plus 20% scaled max error. |

## Finance (chronological rows, `--validation forward`)

| Recipe | Decision | What it rewards |
| --- | --- | --- |
| `portfolio_sortino.json` | Downside-sensitive allocation | Annualized Sortino after turnover cost. |
| `portfolio_sharpe.json` | Volatility-sensitive allocation | Annualized Sharpe after turnover cost. |
| `portfolio_net_return.json` | Profitability | Mean net return after cost. |
| `portfolio_calmar.json` | Growth versus drawdown | Compound annual growth over maximum drawdown. |
| `portfolio_cvar.json` | Tail risk | Expected shortfall of the worst 5% of periods. |
| `portfolio_omega.json` | Gain/loss asymmetry | Omega ratio at a zero threshold. |
| `portfolio_sortino_drawdown.json` | Risk budget | Sortino plus a drawdown penalty (composite). |
| `alpha_signal_ic.json` | Signal research | Rank information coefficient (Spearman); no order needed. |
| `direction_hit_rate.json` | Direction calls | Share of signs predicted correctly. |
| `notional_weighted_error.json` | Risk by position size | MAE weighted by the `notional` column. |

`evaluation_request.json` scores an asymmetric cost with hand-checkable output 17/3.
`api_request.jsonl` is a complete inline CSV fit/predict request. `train.csv` and `predict.csv`
are synthetic smoke fixtures.

Run `bash examples/objectives/run.sh` from the repository root: it writes under
`target/fitness-example/`, fits and reloads a model for a custom cost, evaluates a recipe and
checks the CLI, model file and API agree. [`real-world/`](real-world/README.md) measures recipes on
public datasets and compares RMSE-ranked with objective-ranked program search.
The full contract is [docs/regression-objectives.md](../../docs/regression-objectives.md).
