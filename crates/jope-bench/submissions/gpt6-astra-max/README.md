# GPT6_Astra_Max: original Rust AutoML submission

The complete learner is [`../GPT6_Astra_Max.rs`](../GPT6_Astra_Max.rs). It is one safe Rust source file with no external dependencies. Training receives a row-major table with the **target in its final column**; prediction receives **features only**. The same immutable implementation fits independently on every task.

The implementation combines a regularized linear expansion and histogram residual trees, choosing their mixture inside the current training partition. See [design.md](design.md) for the procedure, complexity, numerical behavior, and limitations.

## Use in JopeDime's AutoML comparison and Elo path

```sh
cargo build --release --locked -p jope-bench
target/release/jope-bench automl
target/release/jope-bench run \
  --datasets /path/to/regression-catalog \
  --out /path/to/reports \
  --heads ridge \
  --automl gpt6-astra-max \
  --seed 20260907 \
  --workers 1 \
  --adapter-timeout 600
target/release/jope-bench ladder \
  --reports /path/to/reports \
  --track-root /path/to/arena \
  --track original-rust-comparison
```

These are usage commands, **not experiments executed during authoring**. The catalog directory must follow JopeDime's existing dataset manifest/input contract; `run` is not an arbitrary-CSV loader. Omit `--heads ridge` to include the existing default baseline set. This is an opt-in addition: historical method catalogs, frozen panel recipes, GP syntax, and the existing Elo calculation keep their identities.

The registered method is `adapter.GPT6_Astra_Max`. The adapter runs inside the same Rust executable as a child process. It receives only its assigned training rows and held-out features. The parent scores its returned predictions, and the existing screen/deepen selector can select it for full-fold evaluation. Its version records the frozen source hash and explicit seed. Passing a classification catalog with `--automl` is rejected; the child independently rejects non-regression tasks.

For programmatic fitting, import `jope_bench::automl::submission::{Regressor, RegressorApi}`. The model stays in memory; this contribution does not add a serialized fitted-model format or a new `jope optimize` export recipe.

The adapter supervisor supplies the existing per-invocation timeout. The learner is single-threaded, while the controller's `--workers` controls concurrent jobs. This integration does not claim the isolated 8 GiB memory limit or separate fit/predict time accounting required for an official challenge campaign. That campaign still needs its own frozen resource/scoring cards and runner.

## Independent source build and verification

```sh
rustc --edition=2021 --crate-name submission --crate-type rlib \
  -C opt-level=3 -D unsafe-code \
  crates/jope-bench/submissions/GPT6_Astra_Max.rs \
  -o target/libsubmission.rlib
cargo test --locked -p jope-bench --lib automl -- --test-threads=1
cargo test --locked -p jope-bench --bin jope-bench original_automl
cargo test --locked -p jope-bench --test adapter_contract
```

Existing Rust CI lanes already compile and test `jope-bench`; no additional workflow or new test target is needed. Conformance tests and adapter tests are housed in existing targets. They check shapes, final-column handling, lifecycle, invalid values, missingness, determinism, batch invariance, numeric extremes, source identity, seed transport, process errors, and same-cut selection. They impose no predictive accuracy threshold. Source hash checks fail if the frozen file changes, and its module is excluded from automatic formatting so formatter changes cannot rewrite the entrant.

## Provenance and effort

- [submission.json](submission.json): mathematical design declaration and complexity.
- [provenance.json](provenance.json): source freeze, model identifier as reported by the session, and context limitations.
- [source.sha256](source.sha256): SHA-256 of the source, relative to the parent `submissions` directory.
- [effort.json](effort.json): token estimates and timestamp-based milestones, with measurement limitations.
- [verification.json](verification.json): post-freeze compilation and contract/integration results.

The source was frozen before repository inspection or numerical execution. No datasets or predictive experiments informed its design. This was an existing-context integration request, not a neutral fresh-session primary challenge attempt; eligibility is not asserted. Registration and passing contract tests establish executable integration, not competitive Elo. Any future algorithm correction must preserve this entrant and receive a distinct source identity.
