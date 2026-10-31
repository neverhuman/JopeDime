# `feat-math`

`feat-math` provides the numerical and program-synthesis engine used by JopeDime. It contains
the legacy pipeline DSL and reference interpreter, the versioned `kernel` program APIs, fitted
table models, search and evolution, and exporters for standalone runtimes. The crate is not a
standalone release or a claim that every backend has been qualified for production.

## Choosing an API

- `advanced_v3` contains the schema-aware table fitting and prediction API, including
  `RawTable`, `TaskSpec`, `fit_table`, and the `BoundModel` runtime.
- `kernel` contains versioned program, capability, admission, serialization, and export APIs.
  Check a program's declared capability and target support before treating an export as runnable.
- `model`, `GenomeIr`, and the root-level `fit` / `cv_r2` functions are the historical pipeline
  interface. Compatibility behavior is guarded by `tests/compat_freeze.rs` and its committed
  fixtures.
- `evolve` and `advanced_v3::evolution` provide candidate generation and search helpers. A
  search result is not a qualification or promotion decision.

Generated runtimes and supported operation sets vary by API and target. Use the relevant
versioned export or capability report to determine what a particular program can execute; a
successful build alone does not establish cross-language parity or target qualification.

## Features and checks

The default `fitted-cpp` feature builds the embedded fitted C++ runtime. `go-parity-tests` and
`cuda-parity-tests` enable additional target-specific tests; CUDA remains an explicit local
qualification path and is not selected by the standard test command. The `gpr-scratch-reuse`
feature is opt-in and requires its own parity and performance qualification before use.

Run the crate suite and strict Clippy from the workspace root:

```sh
cargo test --locked -p feat-math
cargo clippy --locked --all-targets -p feat-math -- -D warnings
```

Some integration tests execute generated artifacts and need their corresponding host toolchains.
If a required compiler or interpreter is absent, treat that result as an environment limitation;
do not count it as passing coverage.
