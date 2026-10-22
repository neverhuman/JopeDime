# Native AutoML regressors

Two original safe Rust regressors, with no dependencies beyond the standard library, are
registered in `jope_automl::REGRESSORS`:

| Id | Source | SHA-256 |
| --- | --- | --- |
| `GPT6p1_Sol_Max` | [`src/GPT6p1_Sol_Max.rs`](src/GPT6p1_Sol_Max.rs) | `fe5bb21e…f75a8` |
| `Gemini4_argon_high` | [`src/Gemini4_argon_high.rs`](src/Gemini4_argon_high.rs) | `d5cd85fc…053d3` |

Each regressor's entire learning procedure lives in its own file. That file is included
byte-for-byte and compiles independently:

```sh
rustc --edition=2021 --crate-name gpt6p1_sol_max --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/GPT6p1_Sol_Max.rs -o libgpt6p1_sol_max.rlib
rustc --edition=2021 --crate-name gemini4_argon_high --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/Gemini4_argon_high.rs -o libgemini4_argon_high.rlib
```

`GPT6p1_Sol_Max` was frozen before repository-code inspection or numerical execution.
`Gemini4_argon_high` was supplied by the owner on 2026-10-22. It replaces the earlier
`gemini-argon-4` candidate as the Gemini entrant, and its authoring conditions are recorded as
unknown.

## Using a regressor

AutoML callers can enumerate `jope_automl::REGRESSORS`, resolve `definition(id)` and call
`fit_rows(id, training_features, training_targets, seed)`. Each returned `FittedRegressor` owns
independent state and accepts features only through `predict_rows`.

- **Transport.** The adapter appends training labels as the final column, preserves NaN missing
  features and adds no numerical preprocessing.
- **Non-finite output.** It refuses a non-finite prediction with an error, because a frozen source
  is never patched.
- **Exact challenge API.** Also available directly: `Gpt6p1SolMax` with `RegressorApi`, and
  `Gemini4ArgonHigh` with `Gemini4ArgonHighApi`.

The existing same-cut benchmark accepts either candidate alongside its existing baseline, program
and external-adapter methods:

```sh
cargo run --release -p jope-bench -- run \
  --datasets /path/to/catalog --out /path/to/reports \
  --native GPT6p1_Sol_Max --native Gemini4_argon_high \
  --seed 20260907 --folds 5 --top 20 --workers 1

cargo run --release -p jope-bench -- ladder \
  --reports /path/to/reports --track-root /path/to/arena \
  --track native-regressors
```

- `--native` is explicit and repeatable; unknown names fail before fitting.
- A method is recorded as `native.<id>`, with its source SHA-256 and seed in the report.
- It is fitted anew on each cut's training rows, then scored and pooled by the existing benchmark.
- Non-regression datasets are skipped when a native regressor is requested.
- Historical default rosters, frozen panel plans, GP grammars and same-cut Elo calculations keep
  their existing definitions.

The same-cut integration is trusted in-process product code. It does not implement the isolated,
CPU-, memory- and time-enforced challenge supervisor described by the master specification. A
competition organizer must provide that supervisor separately.

## AutoML Elo

[`assets/automl-elo/v1`](../../assets/automl-elo/v1/SOURCES.md) records the four-entrant AutoML
Elo: these two native regressors plus `Opus5p5_Extra` and `GPT6_Astra_Max`.

- **Panel.** 64 dataset families × 3 folds × 3 seeds, with one CPU and 8 GiB per fit.
- **Ratings.** Opus5p5_Extra 1641, **Gemini4_argon_high 1561**, GPT6p1_Sol_Max 1456,
  GPT6_Astra_Max 1342, each with a 95% family-bootstrap interval.
- **Caveat.** Gemini4_argon_high's RMSE differences from the other entrants are not significant
  after Holm adjustment.
- **Figures.** Ratings, pairwise win shares and source sizes are in
  [`figures/`](../../assets/automl-elo/v1/figures).

## Submission records

| | `GPT6p1_Sol_Max` | `Gemini4_argon_high` |
| --- | --- | --- |
| Manifest | [submission.json](submissions/GPT6p1_Sol_Max/submission.json) | [submission.json](submissions/Gemini4_argon_high/submission.json) |
| Design | [design.md](submissions/GPT6p1_Sol_Max/design.md) | [design.md](submissions/Gemini4_argon_high/design.md) |
| Authorship and session receipt | [authorship.json](submissions/GPT6p1_Sol_Max/authorship.json) | [authorship.json](submissions/Gemini4_argon_high/authorship.json) |
| Verification | [verification.json](submissions/GPT6p1_Sol_Max/verification.json) | [verification.json](submissions/Gemini4_argon_high/verification.json) |

They distinguish source authoring from later correctness checks. Prior project context existed
when `GPT6p1_Sol_Max` was written, so it is not represented as a pristine fresh-session primary
entrant.

## Verification

```sh
cargo test -p jope-automl --locked
cargo test -p jope-bench --lib native::tests --locked
```

These checks cover target placement, lifecycle, deterministic refits, prediction batching and row
locality, missing, degenerate and extreme inputs, source identity, registration, train/test
separation and same-cut routing. They impose no accuracy threshold.

The frozen sources are excluded from automatic formatting and Clippy style rewrites; adapters
follow the workspace lint policy.
