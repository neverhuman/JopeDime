# Native AutoML regressors

`GPT6p1_Sol_Max` is an original safe Rust regressor with no dependencies beyond the
standard library. Its entire learning procedure lives in
[`src/GPT6p1_Sol_Max.rs`](src/GPT6p1_Sol_Max.rs). The source was frozen before
repository-code inspection or numerical execution and compiles independently:

```sh
rustc --edition=2021 --crate-name gpt6p1_sol_max --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/GPT6p1_Sol_Max.rs -o libgpt6p1_sol_max.rlib
```

AutoML callers can enumerate `jope_automl::REGRESSORS`, resolve `definition(id)`,
and call `fit_rows(id, training_features, training_targets, seed)`. Each returned
`FittedRegressor` owns independent state and accepts features only through
`predict_rows`. The adapter appends training labels as the final column and
preserves NaN missing features; it adds no numerical preprocessing. The exact
challenge API is also available through `Gpt6p1SolMax` and `RegressorApi`.

The existing same-cut benchmark accepts this candidate alongside its existing
baseline, program, and external-adapter methods:

```sh
cargo run --release -p jope-bench -- run \
  --datasets /path/to/catalog --out /path/to/reports \
  --native GPT6p1_Sol_Max --seed 20260907 --folds 5 --top 20 --workers 1

cargo run --release -p jope-bench -- ladder \
  --reports /path/to/reports --track-root /path/to/arena \
  --track native-regressors
```

`--native` is explicit and repeatable; unknown names fail before fitting. The
method is recorded as `native.GPT6p1_Sol_Max` with source SHA-256 and seed in its
report. It is fitted anew on each cut's training rows, then scored and pooled by
the existing benchmark. Non-regression datasets are explicitly skipped when a
native regressor is requested. Historical default rosters, frozen panel plans,
GP grammars, and Elo calculations retain their existing definitions.

The same-cut integration is trusted in-process product code. It does not implement
the isolated, CPU/memory/time-enforced challenge supervisor described by the master
specification. A competition organizer must provide that supervisor separately.

The [submission manifest](submissions/GPT6p1_Sol_Max/submission.json),
[design](submissions/GPT6p1_Sol_Max/design.md), and
[authorship/session receipt](submissions/GPT6p1_Sol_Max/authorship.json) distinguish
source authoring from later correctness checks. No comparative predictive result
or Elo rating is claimed. Prior project context existed in this conversation, so
this artifact is not represented as a pristine fresh-session primary entrant.

Verification:

```sh
cargo test -p jope-automl --locked
cargo test -p jope-bench --lib native::tests --locked
```

These checks cover target placement, lifecycle, deterministic refits, prediction
batching, missing/degenerate/extreme inputs, source identity, registration,
train/test separation, and same-cut routing. They impose no accuracy threshold and
were run only after the source freeze. The frozen source is excluded from automatic
formatting and Clippy style rewrites; adapters follow the workspace lint policy.
