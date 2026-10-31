# hyperion

Hyperion v5/v7 candle inference, ported from jain-core `feat-core` (WP3 of
`FABLE_GITHUB_PORT_PLAN.md`): the matrix-sweep encoder (`fit_v7`, `orig_v2`), its configuration
and runtime, the `.hvc2` cache, the router, and jain-battle-gpu's std-only shared-GPU lease queue.
It is a port, not a rewrite: every module below is byte-identical to its source except for the
edits listed under "Edits".

## Features and binaries

| Feature | Meaning |
|---|---|
| `hyperion-cpu` (default) | candle 0.11.0 CPU forward (`dep:candle-core`, `dep:candle-nn`) — feat-core's feature of the same name |
| `hyperion-cuda` | `hyperion-cpu` + `candle-core/cuda`, `candle-nn/cuda` (candle selects cudarc's dynamic-linking flavour, as feat-core did) |
| `batch-worker` (default) | pulls `jope-parallel` (crates/jope-engine) + rayon for the `hyperion-batch-worker` binary only; the library never needs it |

Without `hyperion-cpu` the crate still builds: every candle path has feat-core's `bail!`
fallback (`cargo check -p hyperion --no-default-features --all-targets`).

| Binary | Source | Notes |
|---|---|---|
| `hyperion-v5-reference` | `src/bin/hyperion-v5-reference.rs` | the preserved `hyperion_vs` reference encoder CLI (needs `hyperion-cpu`, exits 2 otherwise) |
| `hyperion-batch-worker` | `src/bin/hyperion-batch-worker.rs` | the persistent batched GPU encoding worker from `parallel/native/hyperion-host`; `required-features = ["hyperion-cpu", "batch-worker"]` |

## Weights and environment

The served bundle is a directory holding `model.safetensors` + `manifest.json`.

| Variable | Effect |
|---|---|
| `HYPERION_V7_WEIGHTS` | bundle directory (replaces feat-core's hard-coded `artifacts/model-bundle/hyperion_v5_arm_b.*` and the `JAIN_MODEL_BUNDLE_ROOT` packaging root) |
| `JOPE_HOME` | when `HYPERION_V7_WEIGHTS` is unset the bundle is `$JOPE_HOME/models/bundles/hyperion-v7` (relative `models/bundles/hyperion-v7` when both are unset) |
| `JAIN_HYPERION_WEIGHTS`, `JAIN_HYPERION_MANIFEST` | per-file overrides, kept verbatim from feat-core (`HyperionConfig::from_env`) |
| `JAIN_HYPERION`, `JAIN_HYPERION_SEED`, `JAIN_HYPERION_SWEEPS`, `JAIN_HYPERION_ENGINE`, `JAIN_HYPERION_SCALES`, `JAIN_HYPERION_BATCH`, `JAIN_HYPERION_DUMP`, `JAIN_HYPERION_CELLMAP_DUMP`, `JAIN_HYPERION_NO_NORMALIZE`, `JAIN_HYPERION_FULL_CANVAS`, `JAIN_HYPERION_CUDA_MAX_LOAD_RATIO` | feat-core's runtime knobs, unchanged |
| `JAIN_GPU_QUEUE_*` (see `src/battle_gpu/queue/storage.rs`) | jain-battle-gpu lease-queue tuning, unchanged |

### Test gates

`cargo test -p hyperion` runs 55 unit tests (encoder, sweeps, hvc2, router, lease queue) plus the
two reference-CLI tests without any weights. One test is weight-bound and explicitly ignored by
the source-only suite:

- `tests/weights_smoke.rs::served_bundle_encodes_a_small_table` — needs `HYPERION_V7_WEIGHTS`.
  Run it with `--include-ignored`; it fails when the variable or bundle is absent. With the
  variable set it encodes a 24×3 table through `fit_transform_pair` (mode `On`, 2 sweeps) and
  checks the 128-column block.

## Provenance (source → destination)

Sources: jain-core `3692748` (`jain-core-v10.0.0-split.4-57-g3692748`, 2026-09-18) `crates/feat-core/src/`;
jain-battle-gpu `5b9eb0e` (2026-09-04) `crates/battle-gpu/src/`; the jope-dime `parallel/assets/source/hyperion-jain`
family snapshot and `parallel/native/hyperion-host` (the preserved host build).

| Source | Destination | State |
|---|---|---|
| feat-core `hyperion.rs` | `src/hyperion.rs` | edited (weights resolution, `use crate::battle_gpu`) |
| feat-core `hyperion/{config,encoder,hvc2,orig_v2,runtime,tests}.rs` (6) | `src/hyperion/` | `config.rs`, `runtime.rs`, `tests.rs` edited; others verbatim |
| feat-core `hyperion_v7/{backend,config,frames,grid,mha,mod,model,ops,pooling,weights}.rs` + `sweep/{driver,flush,mod,orig_v2,patch,tests,types,validate}.rs` (18) | `src/hyperion_v7/` | `backend.rs` edited (one import); others verbatim |
| feat-core `hyperion_router.rs` | `src/hyperion_router.rs` | verbatim |
| feat-core `progress.rs` (schema v13) | `src/progress.rs` | one field type edited (see shims) |
| feat-core `rng.rs`, `hashing.rs` | `src/rng.rs`, `src/hashing.rs` | verbatim |
| hyperion-host `src/lib.rs` `data::Matrix` (12 lines) | `src/data.rs` | verbatim (rustfmt) |
| jain-battle-gpu `queue.rs`, `queue/storage.rs`, `queue_tests.rs`, `control.rs` | `src/battle_gpu/` | `queue.rs` edited (`crate::control` → `super::control`); others verbatim |
| hyperion-jain `crates/feat-cli/src/bin/hyperion_vs.rs` | `src/bin/hyperion-v5-reference.rs` | edited (crate paths, default bundle paths, usage message) |
| hyperion-host `src/worker.rs` | `src/bin/hyperion-batch-worker.rs` | edited (`feat_core::` → `hyperion::`; rustfmt) |

Where jain-core and the hyperion-jain snapshot differ, jain-core's file was taken (plan risk #5):
`hyperion.rs` (the snapshot's monolith is now `hyperion.rs` + `hyperion/`), `hyperion_v7/backend.rs`,
`hyperion_v7/model.rs`, `hyperion_v7/sweep/{driver,mod,tests,types}.rs`, `progress.rs` (v13 vs v9);
jain-core-only: `hyperion/*`, `hyperion_v7/sweep/orig_v2.rs`, `hyperion_router.rs`. `rng.rs` and
`hashing.rs` are identical in both.

`crates/hyperion/host/` (the `#[path]`/`build.rs` host build) is deleted and removed from the
workspace `exclude` list; this crate replaces it.

## Shims (new files)

| File | Purpose |
|---|---|
| `src/lib.rs` | module tree + `RETRAIN_WITH_JAIN_10` (feat-core `lib.rs` constant that `progress.rs` formats into messages) |
| `src/data.rs` | the host build's column-major `Matrix {n, c, data}` with `new`/`col`/`at` |
| `src/diagnostics.rs` | opaque `DiagnosticBundle(serde_json::Value)` with `validate()` standing in for jain-math's `feat_math::DiagnosticBundle`, which the reconciled feat-math does not have |
| `src/battle_gpu/mod.rs` | `pub mod control; pub mod queue;` |
| `tests/weights_smoke.rs` | the env-gated bundle test above |

## Edits to copied code

- `src/hyperion.rs`: `DEFAULT_WEIGHTS`/`DEFAULT_MANIFEST` consts replaced by `WEIGHTS_DIR_ENV`,
  `JOPE_HOME_ENV`, `DEFAULT_BUNDLE_SUBDIR`, `DEFAULT_WEIGHTS_FILE`, `DEFAULT_MANIFEST_FILE` and
  `default_weights_dir()/default_weights_path()/default_manifest_path()`; added
  `#[cfg(feature = "hyperion-cuda")] use crate::battle_gpu;` (the queue is a module now, not a crate).
- `src/hyperion/config.rs`: `HyperionConfig::default()` uses the resolver above; the
  `JAIN_MODEL_BUNDLE_ROOT` block in `from_env` is gone (the resolver covers it).
- `src/hyperion/tests.rs`: `model_bundle_root_supplies_packaged_hyperion_defaults` became
  `hyperion_v7_weights_env_supplies_bundle_defaults` (checks `HYPERION_V7_WEIGHTS`, the
  `JOPE_HOME` fallback and the bare default); helper renamed `with_weights_dir_env`.
- `src/hyperion/runtime.rs`: `load_settings` and `fit_quantile_bounds` are `pub` (the host build
  made the same change in `build.rs`; the batch worker calls them).
- `src/hyperion_v7/backend.rs`: `#[cfg(feature = "hyperion-cuda")] use crate::battle_gpu;`.
- `src/progress.rs`: `Event.diagnostics: Option<feat_math::DiagnosticBundle>` →
  `Option<crate::diagnostics::DiagnosticBundle>`.
- `src/battle_gpu/queue.rs`: two `use crate::control::…` → `use super::control::…`.
- `src/bin/hyperion-v5-reference.rs`: `feat_core::` → `hyperion::`; `PathBuf::from(DEFAULT_WEIGHTS)`
  → `default_weights_path()` (same for the manifest); the no-feature usage message names this crate.
- `src/bin/hyperion-batch-worker.rs`: `feat_core::` → `hyperion::` (2 sites).
- `Cargo.toml` `[lints.rust] unexpected_cfgs`: declares the jain-battle-gpu loader feature names
  `queue.rs` still mentions (`gpu`, `gpu-dynamic-loading`, `gpu-dynamic-linking`) so the verbatim
  cfgs do not warn.
- `cargo fmt -p hyperion` (import order in `backend.rs`, line breaks in the worker; no semantic change).

## Commands

```
cargo check -p hyperion --all-targets --locked
cargo check -p hyperion --no-default-features --all-targets --locked
cargo test  -p hyperion --locked
HYPERION_V7_WEIGHTS=/path/to/bundle cargo test -p hyperion --locked --test weights_smoke -- --include-ignored --nocapture
```
