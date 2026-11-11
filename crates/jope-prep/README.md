# `jope-prep`

Pure-Rust, WebAssembly-compatible client preparation library for the Carbon execution plane ([ADR-0007](../../docs/adr/ADR-0007-carbon-lambda-cells.md)):

- **Deterministic FIT / HOLDOUT partitioning** ([`SplitPlan`](src/split.rs)): partitions `0..row_count` into non-empty, disjoint `fit_rows` and `holdout_rows` from an explicit `u64` seed, derives a bounded `<= 1,024`-row FIT conditioning sample strictly from `fit_rows`, and binds SHA-256 index digests for both partitions.
- **FIT-only high-resolution landmark extraction** ([`landmarks`](src/landmarks.rs), [`conditioning`](src/conditioning.rs)): hosts the canonical numeric-regression `high` landmark recipe (`256`-float table summary vector + `[64, 17]` ranker column descriptors) and serializes authenticated `6,368`-byte `FitLandmarkBundle` envelopes (`JLMB` v1) without touching a single HOLDOUT row.
- **Standalone JMAT v1 matrix encoding** ([`jmat_lite`](src/jmat_lite.rs)): encodes and validates column-major JMAT v1 binary tables (`fit.jmat` and `holdout.jmat`) with 64-byte alignment and trailing SHA-256 integrity, bit-compatible with `jope-service`'s reader and free of OS memory-map or heavy runtime dependencies so `cargo check -p jope-prep --target wasm32-unknown-unknown` stays green.
- **Typed failure and repair guidance** ([`PrepError`](src/error.rs)): every refusal returns a panic-free `PrepError` value with `PrepError::guidance()`.

## Verification

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test -p jope-prep
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo check -p jope-prep --target wasm32-unknown-unknown
```
