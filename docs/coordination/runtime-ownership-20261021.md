# Runtime ownership map — 2026-10-21

This is a dated map of the current source and its boundaries. The
[capability matrix](../parity/capability-matrix.md), [workspace membership](../../Cargo.toml)
and linked implementations are the inventory. Bare Cargo commands use the
workspace default members, not every crate. A source implementation, a passing
fixture, a qualified live run and a published release are separate states. The
[architecture page](../architecture.md) also retains the earlier target design.

## Executables and authoritative state

| Surface | Owning source and boundary | Current limits |
| --- | --- | --- |
| CPU appliance, `jopedime` | [CLI](../../crates/jope-appliance/src/main.rs), [runtime](../../crates/jope-runtime/src/lib.rs), [package authentication](../../crates/jope-assets/src/lib.rs), [build-time packer](../../crates/jope-dist/src/main.rs). Native CSV fit/predict, `model.jope`, selection and receipts share the runtime with benchmark/pipeline consumers. | CPU implementation and fixture/native-only profiles exist. `cuda:N` is an explicit backend failure; `auto` explains CPU selection ([backend](../../crates/jope-runtime/src/backend.rs)). Deadlines are cooperative. Neural methods, GPU execution and published `jopedime` releases remain pending. The full profile is blocked; model distribution rights are separate from successful packing. See [appliance status](../../deploy/appliance/README.md) and [model lock](../../deploy/appliance/models.lock.json). |
| Product CLI and data/model stores | [`jope` entry point](../../crates/jope-cli/src/main.rs), [datasets](../../crates/jope-datasets/src/lib.rs), [model bundles](../../crates/jope-models/src/lib.rs), [runtime roots](../../crates/jope-paths/src/lib.rs). Rust owns catalog/bundle validation, content-addressed fetch/verification and explicit publication operations. | Remote writes depend on the configured credentials and requested operation. A cache hit, checksum or faster hash is not distribution permission, admission or an end-to-end learning speedup. See [datasets](../datasets.md), [models](../models.md) and [runtime-root policy](../adr/ADR-0003-runtime-roots.md). |
| Search, learning records and benchmark scoring | [Program/evaluator engine](../../crates/jope-engine/src/lib.rs), [learning store](../../crates/jope-learning/src/lib.rs), [benchmark](../../crates/jope-bench/src/lib.rs) and [pipeline](../../crates/jope-pipeline/src/lib.rs). The separate `jope-learning` binary owns its ledger/snapshot commands; the umbrella CLI does not acquire every historical command. | Compare scores only within their bound task/dataset/cut/evaluator/vocabulary/protocol. A recorded checkpoint or source port does not qualify trained-model utility or serving promotion. See [learning records](../learning.md) and [canonical learning-loop ADR](../adr/ADR-0005-canonical-learning-loop.md). |

## Python and historical research boundaries

The [external estimator adapters](../../crates/jope-bench/adapters/README.md) bridge
training-only rows to XGBoost, LightGBM, CatBoost and scikit-learn. Their
[panel contract](../../crates/jope-bench/adapters/_panel.py) owns preprocessing and
private fitted artifacts, not product catalogs, hidden evaluation labels or
admission. Pickle is executable trusted code: SHA-256 checks integrity, not source
authenticity. The CLI reloads its own freshly fitted bytes; arbitrary artifact
imports are not safe. Input and dense-matrix limits do not bound estimator-internal
memory or fit time. The four real engines require all twelve task/reload cases.

The remaining [Python package](../../python/pyproject.toml) and
[flywheel tree](../../deploy/dl-flywheel/README.md) include research/training code,
source-pinned contracts and versioned runtime snapshots. Their
[source manifest](../../deploy/dl-flywheel/SOURCE_MANIFEST.json) and
[role/hash records](../../deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json)
record provenance; they do not authorize launching a controller, adopting a model
or reusing a past campaign deadline. Historical copies are preserved with their
consumers and pins. Refactoring them requires an explicit source/custody migration;
a duplication finding alone does not establish that deletion is safe.

## Observation and proof

The [Rust observatory](../../crates/jope-observatory/src/main.rs) exposes read-only
loopback snapshot/program APIs and serves the [React frontend](../../web/training-dashboard/src/App.tsx).
Its bounded readers display evidence from configured sources. Resource activity,
producer-reported updates, stale observations, qualified numeric results and
admission are distinct facts. The observer does not start training or grant
production authority. [Browser checks](../../web/training-dashboard/README.md) use
synthetic API responses and prove UI behavior, not real optimizer work.

[CI](../../.github/workflows/ci.yml) binds [lane receipts](../../deploy/ci/receipt.py)
and [aggregate checks](../../deploy/ci/aggregate.py) to source/run/attempt.
Full manual lanes, the actual PR suite and post-merge guards have different roles.
Jankurai findings are an input to review; a score, mock control or configured lane
is not evidence that every security/release/runtime requirement passed.
[Published-release verification](../release/README.md) remains a separate process.

## Find and validate the boundary you changed

These are verification recipes, not PASS claims. Use the existing qualified lane,
pinned toolchain and its reused target; record the exact SHA and resulting receipts.
For Cargo, keep `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`; keep `CARGO_TARGET_DIR` unset globally.

```sh
rtk proxy git rev-parse HEAD
rtk proxy cargo run --locked -p jope-cli -- --help
rtk proxy cargo run --locked -p jope-dist -- lock --check
rtk proxy cargo test --locked -p jope-appliance --test single_file
rtk proxy cargo test --locked -p jope-datasets -p jope-models
rtk proxy bash deploy/ci/parity-matrix.sh
```

In the qualified benchmark Python environment, run
`rtk proxy python -m pytest crates/jope-bench/adapters/test_panel.py -q`; omitting the external
engine case is only a boundary subset. For frontend changes, run the typecheck,
Vitest and rendered commands in its README, preserving screenshots/geometry/axe
receipts. Do not use historical research launch scripts as validation shortcuts.
