# Regression objectives: optimize the cost that matters

Root mean squared error is a statistician's default, not a business decision. A stock-out costs
more than an unsold unit, an underestimated pollution peak is worse than a false alarm, an insurer
prices claim *rates* per unit of exposure, and a portfolio manager is judged on downside risk after
costs. JopeDime lets you state that cost once, as a portable JSON specification, and uses it
everywhere a model is chosen, tuned or written:

| Where | What the objective changes |
| --- | --- |
| `jopedime fit --fitness …` | **Selection**: every native method is scored by your objective on out-of-fold predictions. |
| same, `--calibration` (on by default) | **Calibration**: each method's outputs get a monotone map `max(a + b·pred, floor)` learned to minimize your objective, cross-fitted so the score stays honest. A squared-error model becomes a 90th-percentile forecast, an order quantity or a position size. |
| `jope pipeline run --fitness …`, `jope pipeline invent --fitness …` | **Program search**: every candidate program is ranked by your objective on held-out rows; the receipt records the objective and its identity. |
| Rust, JSON-RPC (`jopedime api`) and MCP (`jopedime mcp`) | The same specification, the same scorer, the same identity. |

Without an objective nothing changes: selection is RMSE (`jope.runtime-select/v1`) and the program
search ranks by held-out RMSE.

## Quick start

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo build --locked --release -p jope-appliance
J=target/release/jopedime

$J fitness list                                  # 38 built-ins, defaults, directions
$J fitness validate pinball:quantile=0.9         # resolved spec + sha256 identity
$J fit --train examples/objectives/train.csv --target y --task regression \
  --fitness pinball:quantile=0.9 --output p90.jope --json
$J predict --model p90.jope --input examples/objectives/predict.csv --output p90.csv
```

`--fitness` accepts a catalog name (`mae`), a name with numeric overrides
(`sortino:cost=0.0005,periods_per_year=52`), inline JSON, or `@path.json`. The receipt's
`selection` block records the exact specification, its `fitness_sha256`, each candidate's raw and
calibrated held-out loss, and the map the winner keeps. `examples/objectives/` holds 41 ready recipes.

## Built-in objectives

Let `e = y − pred`. Ratios are fractions, not percentages. Maximized objectives are negated into a
loss for selection; both values are recorded.

| Name | Direction | Definition and typical use |
| --- | --- | --- |
| `rmse`, `mse` | min | Squared error; emphasizes large misses. |
| `mae` | min | Absolute error; linear cost per unit. |
| `median_absolute_error` | min | Median absolute error. |
| `max_error` | min | Worst absolute error on scored rows (a measurement, not a guarantee). |
| `mape`, `smape`, `rmspe` | min | Relative errors with a denominator floor `epsilon`; RMSPE was the Rossmann sales metric. |
| `wape` | min | Total absolute error over total absolute target volume. |
| `msle`, `rmsle` | min | Squared `log1p` error; nonnegative targets and predictions. |
| `r2`, `explained_variance` | max | Variance explained; R² is the hydrologists' NSE. |
| `pearson`, `spearman` | max | Linear and rank correlation (Spearman is the information coefficient of a signal). |
| `huber`, `pseudo_huber`, `log_cosh` | min | Robust losses for spiky sensors. |
| `pinball` | min | Quantile loss; `quantile=0.9` makes a shortage 9× as costly as an excess. |
| `expectile` | min | Asymmetric squared error. |
| `linex` | min | `exp(a·e) − a·e − 1`: a > 0 punishes under-prediction exponentially (Varian, 1975). |
| `asymmetric_absolute`, `asymmetric_squared` | min | Newsvendor cost: `under_cost` per unit short, `over_cost` per unit over. |
| `poisson_deviance`, `gamma_deviance`, `tweedie_deviance` | min | Counts, positive severities and zero-inflated costs (insurance, healthcare). |
| `mase` | min | MAE over the seasonal naive MAE (`season`) on the same chronological rows. |
| `mean_return`, `sharpe`, `sortino` | max | Strategy return and risk ratios after turnover cost (see below). |
| `max_drawdown` | min | Largest peak-to-trough loss of compounded equity. |
| `calmar`, `omega` | max | Growth over drawdown; gains above a threshold over shortfalls below it. |
| `expected_shortfall` | min | Mean loss of the worst `alpha` fraction of net returns (CVaR). |
| `hit_rate` | max | Share of nonzero targets whose sign is predicted. |
| `normalized_gini` | max | Ranking quality: Lorenz-curve Gini over the perfect ranking's (actuarial lift; Kaggle). Tied predictions share their mean target, so row order earns nothing. |
| `kge` | max | Kling–Gupta efficiency for streamflow (Gupta et al., 2009). |
| `capture_at` | max | Share of the target total captured by the top `fraction` of rows (targeting, fraud); ties share their mean target. |

Undefined values — a constant target for R², a zero-risk Sharpe, a negative prediction under a
deviance, an overflow — fail the candidate with its reason. They never become artificial perfect
scores, and a failing objective never silently falls back to RMSE.

## Your own objective

A custom objective is a row expression followed by an aggregation. It can read `y`, `pred`, and
**numeric columns of the training table** (`{"op":"column","name":"unit_margin"}`). Columns an
objective reads are taken out of the features, required in every training row, and ignored when
present in prediction input. Example: a newsvendor whose shortage cost is each row's own margin.

```json
{
  "metric": "custom", "name": "newsvendor_cost", "direction": "minimize",
  "expression": {"op": "where",
    "condition": {"op": "sub", "left": {"op": "y"}, "right": {"op": "pred"}},
    "then": {"op": "mul", "left": {"op": "column", "name": "unit_margin"},
             "right": {"op": "sub", "left": {"op": "y"}, "right": {"op": "pred"}}},
    "else": {"op": "mul", "left": {"op": "column", "name": "holding_cost"},
             "right": {"op": "sub", "left": {"op": "pred"}, "right": {"op": "y"}}}},
  "aggregation": {"kind": "mean"}
}
```

| Expression `op` | Fields |
| --- | --- |
| `y`, `pred` | none |
| `constant` | `value` |
| `column` | `name`: a numeric training column, never a feature |
| `add`, `sub`, `mul`, `div`, `min`, `max`, `pow` | `left`, `right` |
| `abs`, `neg`, `square`, `sqrt`, `ln`, `log1p`, `exp` | `arg` |
| `step` | `arg`: 1 when `arg > 0`, else 0 (fixed penalties, exceedances) |
| `clip` | `arg`, `min`, `max` |
| `where` | `condition`, `then`, `else`: `then` when `condition > 0` |

Aggregations: `mean`, `sum`, `max`, `{"kind":"quantile","quantile":q}`,
`{"kind":"tail_mean","fraction":f}` (the mean of the worst `f` share of rows: CVaR of a row loss)
and `root_mean`.

Two combinators complete the grammar:

- **Weighted**: `{"metric":"weighted","weights":<expression>,"fitness":<objective>}` gives rows
  nonnegative weights — an exposure column for claim frequency, a notional, `abs(y)`. Weighted R²,
  Gini, WAPE and quantiles use their weighted definitions (equal weights reproduce the unweighted
  values exactly, quantiles included); objectives whose meaning weights would
  change (`max_error`, `spearman`, `kge`, strategy metrics) refuse them.
- **Composite**: `{"metric":"composite","terms":[{"weight":w,"scale":s,"fitness":<objective>}]}`
  sums `w·loss/s` over terms, each converted to a minimization loss first. Choose scales from
  business units before selection.

Limits: expressions have depth ≤ 16 and at most 256 nodes in one whole specification,
compositions depth ≤ 8 and ≤ 64 objectives,
at most 16 columns and 32 KiB per specification. JSON objects reject unknown fields. There is no
code execution, file, network or callback access from an objective. The grammar is
`schemas/jope.regression-fitness.v1.schema.json`; `jopedime fitness schema` prints it.

## Calibration: why choosing among mean models is not enough

Every native method fits its parameters with its own loss, almost always squared error. Choosing
among such models by a pinball or newsvendor objective can only find the least-bad *mean*
forecast. Calibration learns the decision layer that turns a mean forecast into the action your
objective rewards:

```text
pred' = max(intercept + slope · pred, lower)      slope > 0
```

- **Learned on held-out predictions.** For each candidate, the map applied to fold *k* is learned
  on the out-of-fold prediction/label pairs of the other folds, never on fold *k*'s own pairs.
  With forward validation it uses only the rows fold *k*'s model could train on: earlier rows,
  minus the `--gap` rows just before the fold. As in stacking, with IID folds fold *k*'s labels
  still trained the models that produced the other folds' predictions; with two map parameters
  this second-order effect is small, and forward validation removes it. A candidate keeps its
  raw predictions unless calibration improves this cross-fitted score; taking the better of the
  two on the same rows makes the winner's recorded validation loss slightly optimistic, which is
  why a sealed test look remains the measure. Cross-fitting stops at the selection's time limit
  like the folds do (the candidate is then skipped), and so does the winner's final map (the
  selection then fails with a budget error).
- **Stored with the model.** The winner's final map is learned on all of its out-of-fold
  predictions, written to `model.jope` (`output_transform`), checked against the selection record
  when the file loads, and applied after the native model at prediction time. A reader that does
  not know the field refuses the file instead of silently dropping the map.
- **Domain-safe.** For Poisson, Gamma and Tweedie deviances the floor keeps predictions positive
  (one thousandth of the smallest positive training target); for log errors it is zero. When the
  raw predictions win, the stored map is that floor alone, so the refitted model cannot leave the
  domain on new rows. `--calibration none` stores no map at all, floor included.
- **Families.** `--calibration auto` (default: `affine` with an objective, `none` without),
  `none`, `shift` (intercept only) or `affine`. The search is a deterministic, bounded
  Nelder–Mead from fixed starts.

Calibration cannot add information the features lack, cannot reorder predictions, and is not a
replacement for a model trained with the objective. It is the cheapest, model-agnostic step that
makes a selection objective change what you actually predict.

## Order-dependent objectives

`mean_return`, `sharpe`, `sortino`, `max_drawdown`, `calmar`, `omega`, `expected_shortfall` and
`mase` score a *sequence*. They require `--validation forward` and chronological rows; IID folds are
refused.

- `y[t]` is the simple return realized during interval *t*;
  `position[t] = clamp(pred[t], −max_position, max_position)`.
- `net[t] = position[t]·y[t] − cost·|position[t] − position[t−1]|`, starting flat.
- Sortino divides by `sqrt(mean(min(net − target, 0)²))` over **all** scored intervals; Sharpe uses
  the standard deviation with divisor N. Ratios are multiplied by `sqrt(periods_per_year)`.
- Drawdown and Calmar compound `1 + net[t]`; a period losing everything ends the curve at zero.
  Calmar divides the compound annual growth rate by the maximum drawdown.
- With *k* folds, the first `floor(n/(k+1))` rows are training-only warm-up; each later block is
  predicted from earlier rows minus `--gap` rows; the objective runs once on the pooled suffix.

```sh
$J fit --train returns.csv --target next_return --task regression \
  --validation forward --gap 1 --fitness @examples/objectives/portfolio_sortino.json \
  --output allocation.jope --json
```

A risk ratio on a backtest is not a valid trading claim by itself: survivorship, look-ahead in
features, overlapping labels and capacity are the caller's responsibility.

## Program search under your objective

The program search ranks every candidate program by the objective on held-out rows instead of
RMSE, so the program it keeps (transforms, features and model head) is the one that scores best
on your cost:

```sh
cargo run --locked --release --bin jope -- pipeline run \
  --dataset datasets/test/pmlb-229-pwlinear --fitness pinball:quantile=0.9 \
  --budget evaluations=64,seconds=300 --emit rust,python --output target/p90-run
cargo run --locked --release --bin jope -- pipeline invent \
  --train train.csv --target y --work-dir target/p90-invent --fitness pinball:quantile=0.9
```

Every measured observation carries `objective_loss`; the run writes
`jope.pipeline-objective-receipt/v1`, which is `jope.pipeline-receipt/v3` plus `search.objective`
(specification, name, direction and identity) and `validation_objective`, and selection uses
`all_cuts_then_min_mean_validation_objective_then_genome`. `pipeline verify-receipt --evidence`
re-derives the champion from the published observations under the same rule, and
`pipeline score --split test` reports the objective on the sealed test rows (`--fitness` reports
another objective on the same single look, for example to compare an RMSE-ranked run).

Programs still fit their own parameters with their heads' estimators; the objective decides which
program wins, not what each program predicts. Measured on the ten committed regression datasets
(`examples/objectives/real-world/search-benchmark.sh`), ranking by a 90th-percentile pinball loss
changed the selected program every time but lowered the sealed test loss on only four of ten.
Ranking mean predictors by a quantile loss is a weak lever on its own; for quantile and asymmetric
costs today, `jopedime` with calibration is the stronger tool.
The search refuses order-dependent objectives (its cuts are a fold partition) and objectives that
read columns (prepared cuts carry features only); use `jopedime` for those. Multi-round
`pipeline evolve` lineages and the kernel2 profiles keep their own ranking and refuse `--fitness`.

## API and MCP

Rust: set `jope_runtime::api::FitOptions { fitness, validation, calibration, .. }` and call
`api::fit`. Build the table with `table::training_with_auxiliary(…, &spec.columns())` when the
objective reads columns. Score directly with `spec.evaluate_with(&y, &pred, &columns)`.

JSON-RPC over JSON lines (`jopedime api`) and MCP over stdio (`jopedime mcp`) expose four tools:

| Tool | Purpose |
| --- | --- |
| `fitness_list` | Built-ins with defaults, directions, descriptions and the custom grammar summary. |
| `fitness_validate` | Resolve a name, shorthand or JSON spec; return its identity, columns and requirements. |
| `fitness_evaluate` | Score targets and predictions (plus `columns` when the objective reads them). |
| `regression_fit_predict` | Select, calibrate and refit native models for an objective on inline CSV; predict. |

Register the server with an MCP client:

```sh
claude mcp add jopedime -- /absolute/path/to/jopedime mcp
```

```json
{"mcpServers": {"jopedime": {"command": "/absolute/path/to/jopedime", "args": ["mcp"]}}}
```

The server speaks MCP revision 2025-11-25 and echoes 2025-06-18 or 2024-11-05 when a client asks
for one of them; 2025-03-26 is answered with 2025-11-25, because that revision requires JSON-RPC
batches, which the server does not accept. It sends `instructions` describing the grammar to the client. Unknown
tools are protocol errors; a failing tool call is an `isError` result with the reason. Requests are
limited to 8 MiB, 100,000 rows, 1,000 columns, 32 candidate selectors and 2–20 folds, with one
numerical worker and a 30-second cooperative selection budget. The service has no network listener
or file access; a hosting wrapper owns transport security.

## Real-world datasets

`examples/objectives/real-world/` fetches public datasets where the right objective is not RMSE
(outside the checkout, digest-pinned), and benchmarks three arms on the same held-out rows:
RMSE selection, objective selection, and objective selection with calibration. Its README lists
the datasets, licences, objectives and measured results, and a wider shortlist of open datasets
(insurance, retail, energy, hydrology, finance) with their domain-standard metrics.

On the six tables measured there, objective selection alone chose the same learner as RMSE four
times. The exceptions were French motor claims, where the RMSE choice predicted non-positive rates
and could not be scored, and the S&P 500 series, where selection by the after-cost Sortino ratio
raised it on the test months from 0.783 to 1.155. Calibration reduced held-out loss by 7% to 63% on
the four loss objectives and the claims deviance by 19% (0.789 to 0.637); on the S&P series its map,
nearly a constant full position, scored 0.839.
These are single splits with one seed: they show the mechanism, not a general effect size.

## Evidence and identity

The objective identity is the sha256 of the JSON serialization of
`["jope.regression-fitness/v1", specification]`; every parameter, including defaults resolved from
short names, is part of it. A fitted model stores the selection record, the objective's columns
and the output map; loading checks that they agree. Prediction never re-evaluates the objective or
refits.

```sh
cargo test --locked -p jope-runtime -p jope-appliance
cargo test --locked -p jope-pipeline --lib -- objective
bash examples/objectives/run.sh
```

## References

- Granger, C. W. J. (1969). Prediction with a generalized cost of error function. *Journal of the
  Operational Research Society* 20(2), 199–207.
- Varian, H. R. (1975). A Bayesian approach to real estate assessment. In *Studies in Bayesian
  Econometrics and Statistics*, 195–208 (LinEx loss).
- Christoffersen, P. F., Diebold, F. X. (1997). Optimal prediction under asymmetric loss.
  *Econometric Theory* 13(6), 808–817.
- Gneiting, T. (2011). Making and evaluating point forecasts. *JASA* 106(494), 746–762.
- Sortino, F. A., Price, L. N. (1994). Performance measurement in a downside risk framework.
  *Journal of Investing* 3(3), 59–64.
- Hong, T., et al. (2016). Probabilistic energy forecasting: GEFCom2014 and beyond. *International
  Journal of Forecasting* 32(3), 896–913.
- Frees, E. W., Meyers, G., Cummings, A. D. (2011). Summarizing insurance scores using a Gini index.
  *JASA* 106(495), 1085–1098.
- Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F. (2009). Decomposition of the mean squared
  error and NSE performance criteria. *Journal of Hydrology* 377(1–2), 80–91.
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: results,
  findings, and conclusions. *International Journal of Forecasting* 38(4), 1346–1364.
