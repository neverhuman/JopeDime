# Start here: JopeDime agents

Read `../AGENTS.md` first; it controls ownership, verification and custody. This guide is an entry map, not execution admission.

1. Read [`MASTER_PLAN.md`](../MASTER_PLAN.md), then the [47-task tracker](plans/major-release-20261018/MAJOR_RELEASE_TRACKER.md), [GP workplan](plans/major-release-20261018/GP_UPGRADE_WORKPLAN.md) and [release closure workplan](plans/release-closure-20261019/WORKPLAN.md). The old66-item ledger is a separate milestone; preserve both.
2. Read the current `AGENT_CHAT.md` and claim under its lock before writing. Reuse an explicitly owned registered lane. Read latest fleet rulings/claims and obtain a stopped handoff before overlapping edits. An idle agent is not a released checkout.
3. Verify GitHub remote main, open PR queue, branch/base/fullhead, dirty/untracked state and ownership. Use one active integration PR; keep shared history and independent exact-head review. Do not apply another project's CI exceptions to JopeDime.

## Product and implementation map

- [`pipeline.md`](pipeline.md), [`architecture.md`](architecture.md), [`emitters.md`](emitters.md): actual consumer, budgets, fitted programs and supported export subsets.
- `crates/jope-cli`: umbrella `jope` CLI; `jope-pipeline`: dataset/fold/search/selection/emit and model-build orchestration; `feat-math`: typed GP, fitting and emitters. Directory `jope-engine` is Cargo package `jope-parallel`.
- [`models.md`](models.md), `crates/jope-models`, `models/manifest.json`, `models/PINNED`, `models/bundles/`: self-hosted immutable weights and cache contract.
- `deploy/dl-flywheel`: imported native/continuous training, source-custody inventories, admission and research tests. Read its README and source classification before using a producer. Python training is a separate runtime from the default Rust product build.
- [`learning.md`](learning.md), [`datasets.md`](datasets.md), `jope-learning` and `jope-datasets`: HWM/champions, data/split/license and snapshot contracts.
- [`dashboard/README.md`](dashboard/README.md), `crates/jope-observatory`, `web/training-dashboard`: Rust API and React/Vite/TypeScript observatory. It reads evidence; observer freshness is not training admission or autonomous execution.
- [`research/waves/INDEX.jsonl`](research/waves/INDEX.jsonl), [`runbooks/training-artifact-locations.md`](runbooks/training-artifact-locations.md) and [`catalogs/training-artifacts.json`](../catalogs/training-artifacts.json): compact history, durable data/model/archive locators and restoration proof.
- `.github/workflows/ci.yml`, `deploy/ci/check.sh`, `deploy/ci/coverage.json`, [`release/README.md`](release/README.md): required validation and release process.

## Get existing encoder/policy bundles

From a built checkout with admitted S3 read access:

```sh
cargo run --locked --bin jope -- models verify --manifests-only
cargo run --locked --bin jope -- models fetch hyperion-v7
cargo run --locked --bin jope -- models verify hyperion-v7
cargo run --locked --bin jope -- models path hyperion-v7
cargo run --locked --bin jope -- models fetch policy-step-60000
cargo run --locked --bin jope -- models verify policy-step-60000
```

Default store: `s3://veox-jopedime-models-use2`. Each immutable bundle has its own `$JOPE_HOME/models/bundles/<id>/` directory; `$JOPE_HOME` defaults to the XDG cache's `jope` folder. `models path` prints the actual fetched directory. `HYPERION_V7_WEIGHTS` expects the Hyperion bundle directory, not a single file. `--offline` serves the cache; absence/corruption must refuse. Do not add `--accept-restricted` merely to silence a license refusal. Do not print credentials or give host writer credentials to PR runners.

The native Kumo282-token PolicyLM research model and legacy Hyperion/step60000 policy are distinct families. A research checkpoint is not a qualified release. P02 must supply a verified versioned release set binding encoder/tokenizer/transforms/policy/runtime/source before the new neural pipeline can execute. The release workplan tracks this gap; do not claim a mutable `latest` pointer is a promoted model.

## Modeling contract

Required default: encoding → deep learning → GP → standalone predictor. Thinking independently extends the neural GP seed with bounded evolution. AutoML independently admits a broader CV roster; ensemble independently tries eligible winners. See README for the four-flag matrix and implementation/qualification state. Train-fold transforms, group-disjoint five-fold selection, real fitted artifacts and untouched outer-test history are required. A mocked fitter, planning enum or refusal test does not qualify a working route.

## Verification and results

Before Cargo set `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`, retain the lane's reused target and existing fleet caps; do not set `CARGO_TARGET_DIR` globally. New source/tests and fixture evidence differ from live qualification and serving promotion. Record exact SHA/command/exit/nonzero census and immutable log hashes. Required CI and independent native review bind the final head, followed by receiving-main validation.

Every wave keeps small manifest/card/metrics/context in Git, with bulk compressed immutable artifacts outside the tree and exact VersionIds/hashes/restoration receipts. Preserve failure/unknown outcomes and original source identities. Count proposals, actual measurements, published labels, optimizer updates and fresh-feedback updates separately. No source inventory percentage is proof of self-play gains.

Never reset/clean valuable work, remove symlinks, touch another writer's files, repeat a benchmark to fill GPUs, or infer rights/admission from object existence. See the closure workplan for exact host lanes, current owners, stops, cleanup eligibility and handoff fields. Use the Admiral's existing dispatch path; no new worker/screen/worktree or competing platform deploy.


## Combined source integration — 2026-10-20T03:55Z

Root01a13a98 is consolidating stopped W2 `e66442c8847fe3155a08f1fadb6a899d95111349`
(tree `3cea9f8cbf7d8c244c2b45c8e4a54549405cc3b9`) in the existing main-based
`codex/major-release-closure-20261019` branch, preserving both histories. No new lane or
competing PR is created. C14 receipts are under
`/home/ubuntu/jope-evidence/release-closure-20261019/monitor-0340-20261020/c14-integration/`.

The stopped component has 626 passed / 0 failed / 3 ignored, including 33 native-study and 8 compile-fail
doctests (both included in 626), plus noncached clippy exit0. Independent driver-delta review
`review_w2_e66442c88_w1_native_driver_20261020.json` SHA256
`acefed409a34af1924f96de4e284dd0a9c6c9df9f397cc3f018f456ecc2890dd` has no blocking findings.
The original producer receipt has a future timestamp and omitted executed doctests; addendum
`ccb6ec4e21f4152d5af2d5bc49435779f312dfb9c16ea39669011835b15780a7` preserves it and corrects
those metadata errors. Observed full-log SHA256:
`cd3dc2c2473efbaaf36ad521f5295a844a383882b2d2cf3f4eed203919548880`.

Component results do not substitute for combined-head qualification or final native approval.
Combined tests, generated inventory, README consistency and exact-head review remain pending
until their actual receipts exist. The library validates the whole plan before any fit, uses
one ledger/counter, freezes all arms before any O and refuses retry/reassessment of an existing
study. It preserves the legacy finite budget API, which has no production caller.

Real admission root/provider/loader, a production study entry point and physical E9 linkage
remain open. Separate Qwen9B recovery-r3 performs inference and GP evaluation, not native
optimizer continuation; native training remains at 429 updates. No MR47 or older I66 task is
accepted by this integration note.


Count audit: [validation count notes](plans/release-closure-20261019/VALIDATION_COUNT_NOTES.json) separates 625 component top-level passes from its successful fresh-process replay, and 903 combined top-level passes from its replay. The earlier 626/904 aggregate summaries remain preserved. Combined 5c software tests passed; scoped clippy and CI hygiene/e2e still required repairs. No full release or scientific acceptance follows.
