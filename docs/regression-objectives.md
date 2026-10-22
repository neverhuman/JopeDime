# Regression objective: optimize the cost that matters

`jopedime fit --method auto` can select native regression models using 25 built-in objectives,
your own row loss, or a scaled combination of objectives. The Rust API, JSON-lines API and MCP
server use the same specification and scorer. The objective changes **candidate selection on
out-of-fold predictions**; it does not change an individual estimator's training loss. The
winner is refitted on all training rows. Keep a separate final test set for an unbiased report.

This interface belongs to the deployable `jopedime` runtime. The research `jope pipeline`
algorithm-synthesis search, frozen benchmark/Elo protocols, and policy-training rewards retain
their existing objectives and evidence identities. They do not silently consume these settings.

## Start with a built-in

```sh
cargo build --locked -p jope-appliance
./target/debug/jopedime fitness list
./target/debug/jopedime fit \
  --train examples/fitness/train.csv --target y --task regression \
  --method auto --candidates ridge,histgbm --folds 3 --fitness mae \
  --output target/fitness-model.jope --receipt target/fitness-receipt.json --json
./target/debug/jopedime predict --model target/fitness-model.jope \
  --input examples/fitness/predict.csv --output target/fitness-predictions.csv
```

`--fitness` accepts a catalog name, an inline JSON object, or `@path.json`. The examples use
synthetic fixtures to exercise the interface; their numbers are illustrative business costs,
not calibrated industry recommendations. Select `--task regression` when the target's meaning
is continuous but a small integer CSV could otherwise be inferred as classification.

```sh
./target/debug/jopedime fit \
  --train examples/fitness/train.csv --target y --task regression \
  --method auto --candidates ridge,histgbm --folds 3 \
  --fitness @examples/fitness/grid_underforecast.json \
  --output target/grid-model.jope --json
./target/debug/jopedime fitness evaluate --input examples/fitness/evaluation_request.json
```

Omitting objective preserves RMSE model selection and the existing `jope.runtime-select/v1`
policy. Explicit objective, or forward validation, uses `jope.runtime-select/v2`. A named method
cannot use a selection objective and is refused rather than accepting an unused flag.
Classification refuses regression objective. The fitted container's shape and runtime ABI remain
v1; its metric field admits custom names, and its selection record carries the exact specification.
Older schema validators with the former three-metric enum need the updated fitted schema.

## Choose an objective

Let `e = y - pred`. Percentage errors below are ratios, not percentages multiplied by 100.

| Name | Natural direction | Definition / useful setting |
| --- | --- | --- |
| `rmse`, `mse` | minimize | Square-root mean squared error / mean squared error; emphasize large misses. |
| `mae` | minimize | Mean absolute error; linear cost per unit missed. |
| `median_absolute_error` | minimize | Median absolute error, linearly interpolated for even row counts. |
| `max_error` | minimize | Worst absolute error; a measured worst-case objective, not a safety guarantee. |
| `mape` | minimize | Mean `abs(e)/max(abs(y),epsilon)`; default epsilon `1e-8`. Near-zero targets need a meaningful floor. |
| `smape` | minimize | Mean `2*abs(e)/max(abs(y)+abs(pred),epsilon)`; default epsilon `1e-8`. |
| `wape` | minimize | `sum(abs(e))/sum(abs(y))`; undefined for all-zero targets. |
| `msle`, `rmsle` | minimize | Mean squared `log1p` error / its root; y and pred must be nonnegative. |
| `r2` | maximize | `1 - mean(e²)/population_variance(y)`. |
| `explained_variance` | maximize | `1 - population_variance(e)/population_variance(y)`; insensitive to constant bias. |
| `pearson` | maximize | Pearson correlation; measures association rather than calibration. |
| `spearman` | maximize | Pearson correlation of average tied ranks; signed zeros tie. |
| `huber` | minimize | `0.5*e²` inside delta; `delta*(abs(e)-0.5*delta)` outside; default delta 1. |
| `pseudo_huber` | minimize | `delta²*(sqrt(1+(e/delta)²)-1)`; default delta 1. |
| `log_cosh` | minimize | Stable `log(cosh(e))`; quadratic near zero, approximately absolute for large errors. |
| `pinball` | minimize | `q*e` when e≥0, `(q-1)*e` otherwise; default q=0.5. Set q=0.9 for costly shortages. |
| `expectile` | minimize | `q*e²` when e≥0, `(1-q)*e²` otherwise; default q=0.5. |
| `poisson_deviance` | minimize | Mean Poisson deviance; y≥0, pred>0. Includes y=0 by its limit. |
| `gamma_deviance` | minimize | Mean Gamma deviance; y>0, pred>0. |
| `tweedie_deviance` | minimize | Mean Tweedie deviance with 1<power<2, default 1.5; y≥0, pred>0. Use named endpoint deviances for power 1 or 2. |
| `mean_return` | maximize | Mean net strategy return under the explicit position/cost convention below. |
| `sharpe` | maximize | Excess mean return / standard deviation with divisor N × sqrt(periods_per_year). |
| `sortino` | maximize | Excess mean return / full-sample downside deviation × sqrt(periods_per_year). |

Constant-target R²/explained variance, constant-vector correlations, and zero-risk ratios are
undefined and refused. They never receive artificial perfect scores. Log/deviance objectives
do not clip predictions to hide domain violations; incompatible candidates fail with a reason.
Every built-in parameter, including all defaults resolved from short names, is retained.

## Build your own

A custom objective is a JSON expression over **held-out** `y` and `pred`, followed by an
aggregation. `examples/fitness/grid_underforecast.json` implements
`mean(5*max(y-pred,0) + max(pred-y,0))`. Underforecasting one unit costs five; overforecasting
one unit costs one. Change those costs to your operational values.

```json
{
  "metric": "custom",
  "name": "absolute_business_error",
  "direction": "minimize",
  "expression": {
    "op": "abs",
    "arg": {"op": "sub", "left": {"op": "y"}, "right": {"op": "pred"}}
  },
  "aggregation": {"kind": "mean"}
}
```

| Expression | Fields |
| --- | --- |
| `y`, `pred` | none |
| `constant` | `value` |
| `add`, `sub`, `mul`, `div`, `min`, `max` | `left`, `right` expressions |
| `abs`, `square`, `sqrt`, `ln`, `exp` | `arg` expression |

Aggregations: `mean`, `sum`, `max`, or `{"kind":"quantile","quantile":0.99}`.
Quantiles use linear interpolation at `q*(n-1)`. A 99th-percentile loss is not CVaR and is not
a guarantee about unobserved failures. Custom names need 1–64 ASCII letters, digits or underscores.

Use `{"metric":"composite","terms":[{"weight":0.8,"scale":0.01,"objective":{"metric":"mae"}},
{"weight":0.2,"scale":0.05,"objective":{"metric":"max_error"}}]}` to combine normalized costs.
Each term is converted to a minimization loss, divided by its positive scale, and multiplied by
its positive weight. Choose scales from business units or training-only baselines before
selection; weights need not sum to one. Maximized terms enter with a negative sign.

The complete machine-readable grammar is
`schemas/jope.regression-fitness.v1.schema.json`.
MCP discovery includes it. JSON objects reject unknown fields, including parameter-free metrics.
There is no Python, shell, dynamic library, network or filesystem access from a objective.
Expressions are bounded to depth 16 and 256 nodes; compositions to depth 8, 64 objectives and
16 terms per composite. Text specs are limited to 32 KiB. Division by zero, domain errors and
overflow fail that candidate. Weighted row costs can be expressed using y/pred; arbitrary
external sample weights, feature columns, protected groups and user callbacks are not part of
this v1 data contract. Such costs need an explicit future input/identity extension.

## Sortino and chronological evaluation

Sortino is a strategy objective, not a generic regression residual metric. Here:

- `y[t]` is the simple realized return during interval t, in decimals.
- `position[t] = clamp(pred[t], -max_position, max_position)`. Predictions are positions directly;
  forecasting a return of 0.01 gives 0.01 units of exposure under this rule. There is no implicit
  sign strategy, threshold, leverage scaling or allocation optimization.
- `net[t] = position[t]*y[t] - cost*abs(position[t]-position[t-1])`, starting from position 0.
  Cost is per unit turnover, charged on entry and position changes. No final liquidation cost,
  compounding, financing, borrow fees or execution-price model is assumed.
- `target` is the minimum acceptable return **per interval**.
- Sortino's denominator is `sqrt(mean(min(net-target,0)^2))`, counting **all scored intervals**.
  Sharpe uses standard deviation with divisor N of net returns.
- Ratios multiply by `sqrt(periods_per_year)`; mean return remains per interval. The scaling is
  an explicit conventional assumption, not a correction for serial dependence or overlapping returns.

```sh
./target/debug/jopedime fit --train chronological_returns.csv --target realized_return \
  --task regression --method auto --candidates ridge,histgbm --folds 3 \
  --validation forward --gap 1 --fitness @examples/fitness/portfolio_sortino.json \
  --output target/returns-model.jope --json
```

Return-based metrics, including nested composite terms, require forward validation. They are
rejected under IID folds. Sort input rows chronologically before calling: v1 does not infer
chronology from a timestamp column. Choose a gap at least as large as the label-overlap horizon
for your labeling scheme. The caller owns that temporal contract and past-only feature design.

For k folds and n rows, block size is floor(n/(k+1)). The first block is training-only warmup.
Each later block is predicted using only earlier rows, excluding the configured gap; the final
validation block includes the remainder. At least two training rows must remain after the gap.
Predictions are pooled in original row order and the objective runs **once on the scored suffix**;
warmup never contributes a made-up prediction or return. Turnover carries across scored fold
boundaries, because the evaluation models one sequential strategy. The record names the fold
policy, gap, warmup and scored row counts. Return metrics alone do not establish a valid backtest.

## API and MCP

Rust callers set `jope_runtime::api::FitOptions.fitness: Option<Fitness>` and
`validation: automl::Validation`, then call the existing `api::fit`. Direct scoring is
`fitness.evaluate(&targets, &predictions)`. `Score.value` is natural; `Score.loss` is negated for
maximization. Candidate outcomes record both. Exact ties retain roster order.

For local JSON-RPC over JSON lines:

```sh
./target/debug/jopedime api < examples/fitness/api_request.jsonl
```

Methods: `fitness_list`, `fitness_evaluate` and `regression_fit_predict`. The latter accepts
inline `train_csv`, `input_csv`, `target` and `options` (method, candidates, folds, objective,
validation). It returns predictions CSV, model header, selection evidence and budget. It does
not save a fitted model: use the CLI or Rust API when you need persistence.

For MCP, configure a stdio server with `command` pointing to the absolute `jopedime` executable
and `args: ["mcp"]`. This server implements the **2025-11-25 handshake protocol**; configure a
compatible client. Initialize, send `notifications/initialized`, then discover/call the same
three tools. It negotiates this supported revision even if another revision is requested.
Example tool arguments for scoring:

```json
{"name":"fitness_evaluate","arguments":{"fitness":{"metric":"pinball","quantile":0.9},"targets":[10,20,30],"predictions":[9,22,28]}}
```

Only protocol responses go to stdout. MCP tool failures use `isError: true`; bad JSON produces
a parse error. Notifications do not produce responses. API/MCP requests are limited to 8 MiB,
100,000 rows, 1,000 columns, 32 candidate selectors and 2–20 folds; one numerical worker and a
30-second **cooperative selection** budget apply. Fits and final refits are not preempted.
The service has no network listener or remote authentication layer; a hosting wrapper must own
those concerns. Packaged executables authenticate their package before serving.

## Evidence and examples

The selection record retains the exact resolved spec, `fitness_sha256`, both natural and
minimization scores for each successful candidate, failed/skipped reasons, validation plan
settings, and the chosen model. The digest hashes JSON serialization of `["jope.regression-fitness/v1", spec]`.
Changing a parameter or expression changes it. Model reload checks the objective/name/identity
against its selection metadata; prediction does not reevaluate objective or refit.

See `examples/fitness/README.md` for 18 business examples.

Validation commands:

```sh
cargo test --locked -p jope-runtime
cargo test --locked -p jope-appliance --test single_file
cargo clippy --locked -p jope-runtime -p jope-appliance --no-deps -- -D warnings
bash examples/fitness/run.sh
```

References for definitions: [scikit-learn regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics),
[CFA Institute on the Sortino ratio](https://rpc.cfainstitute.org/-/media/documents/code/gips/the-sortino-ratio.pdf),
and [MCP 2025-11-25 schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-11-25/schema.ts).
These inform the definitions; JopeDime's explicit edge-case, cost, pooling and validation conventions above control this implementation.
