# Native AutoML regressors

`gemini-argon-4` is also registered. Its implementation is
[`src/gemini_argon_4.rs`](src/gemini_argon_4.rs); the
[original submission](submissions/gemini-argon-4/original.rs), supplied design,
telemetry and correction record remain under
[`submissions/gemini-argon-4/`](submissions/gemini-argon-4/).
The original fails compilation. Its successor changes exactly four type
annotations, with SHA-256
`b0aa42d24d0b067e0597f6f538b14cab75dca8b976731668c894d80684d4992d`.
Neither its numerical procedure nor the other entrants' source bytes are changed.

Both benchmark selectors use that same implementation:

```sh
jope-bench automl
jope-bench run --datasets /path/to/catalog --out /path/to/reports --automl gemini-argon-4
jope-bench run --datasets /path/to/catalog --out /path/to/reports --native gemini-argon-4
```

Gemini classification requests fail validation before fitting. Its adapters reject
malformed dimensions, infinite features and nonfinite targets; NaN features remain
missing values. `fit_flat` accepts row-major training data with the target last,
and `predict_flat` accepts features only. `GeminiArgon4` and `GeminiRegressorApi`
expose the original challenge interface. Each fitted instance owns its state.

Reproduce the original's four required annotation errors and compile the corrected
source with the repository-pinned compiler using the offline verification harness:

```sh
python3 tools/qualify-gemini.py --output /owned/fresh/compiler-evidence
cargo test --locked -p jope-bench gemini
```

The harness verifies all frozen entrant hashes and retains commands, full compiler
identity, diagnostics and outcomes. It does not run a scientific fit. Module-level
formatting and style-lint exclusions preserve the frozen implementation; adapter
code follows normal workspace rules. The
[entrant review](submissions/gemini-argon-4/REVIEW.md) retains its internal-validation
limitations. The [delivery checklist](../../docs/coordination/four-model-benchmark-20261022.md)
tracks the separate benchmark and code-production work; registration implies no
comparative performance result.

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
`GPT6p1_Sol_Max` is requested. Historical default rosters, frozen panel plans,
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
