# Architecture

> **Status on `main`:** this page describes the target design from the port plan. Run `jope --help` on `main` for the verbs that exist today. There is no `jope learn`, `jope hwm` or `jope scores`; the learning store is the separate `jope-learning` binary, with `hwm` and `scores` subcommands. See the README's "What this SHA actually does".

JopeDime is one repository that holds the whole pipeline
**dataset → encoders → policy → program → emitted source code**. Rust owns search, evaluation,
the emitters, the CLI and every contract validator. Owner decision 2 of the plan gave encoder
inference/training and the deep-learning models to a `uv`-managed Python package;
[ADR-0006](adr/ADR-0006-rust-only-stack.md) (owner ruling 2026-09-26) supersedes it. The stack
is Rust-only apart from the Python the emitters generate and its harness, and the Python that
still holds the training stack and some operational tools stays only until phases P2/P3 port or
retire it. This page is the map; the numbers were verified on the tree at the commit that
introduced it (2026-09-18).

## Data flow

```mermaid
flowchart LR
  DS[(datasets<br/>S3 blobs + 10 test sets)] --> ENC[jope-encoders<br/>foundation / target / chimera]
  DS --> HYP[hyperion<br/>v5 arm B, v7 signatures]
  ENC --> Z[z ∈ R^1536<br/>hyperion-512 ‖ foundation-1024<br/>target-512]
  HYP --> Z
  Z --> POL[jope-policy<br/>step-60000 transformer, grammar-masked decode]
  POL --> SEEDS[parallel.seed_population.v1<br/>80 candidate programs]
  SEEDS --> ENG[jope-engine<br/>program search, cluster CV eval, eval registry]
  ENG --> WIN[cut_champion.v1<br/>high-water marks]
  WIN --> FM[feat-math<br/>DSL interpreter, heads]
  FM --> EMIT[transpile::emit<br/>Python · Rust · C++17 · C++20 · JavaScript]
  WIN --> LEARN[jope-learning<br/>corpus freeze, train, qualify, promote]
  LEARN --> POL
  ENG --> ARENA[jope-arena / jope-campaigns<br/>Golden42/46, SOTArena scores, Elo]
```

Every arrow is a file with a named schema (see [Contracts](#contracts-and-schemas)); nothing is
passed by shared memory between stages, which is what makes each stage replayable from S3.

## Crate graph

```mermaid
flowchart TB
  subgraph product["product crates (workspace version)"]
    numpy[numpy-pcg64<br/>bit-exact numpy 1.26 RNG, 0 deps]
    fm[feat-math<br/>DSL · search · heads · registry · emitters · fitted C++ runtime]
    bg[battle-gpu<br/>raw dlopen CUDA screening, feature gpu]
    enc[jope-encoders<br/>lib name starforge: foundation, target, chimera, policy decoder]
    hyp[hyperion<br/>candle v5/v7 inference, router, GPU lease]
    pol[jope-policy<br/>step-60000 policy port]
    eng[jope-engine<br/>package jope-parallel: 88 subcommands, --root, feature neural]
    camp[jope-campaigns<br/>package jope-dime-regressor: 55 programs]
    self[jope-selfplay<br/>flywheel control plane]
    paths[jope-paths<br/>ADR-0003 roots, 0 deps]
    data[jope-datasets<br/>catalog · ingest · fetch/verify/link]
    models[jope-models<br/>bundle manifests · fetch/verify/publish]
    learn[jope-learning<br/>HWM ledger · snapshots · pointers]
    arena[jope-arena<br/>SOTArena fetch · Elo · scores]
    cli[jope-cli<br/>the jope umbrella binary]
  end
  subgraph vendored["vendored (upstream versions)"]
    sc[crates/smartcluster/*<br/>jain-smartcluster @ 715128f8, 7 crates]
  end
  fm --> camp
  fm --> eng
  bg -.gpu.-> camp
  bg -.gpu.-> eng
  numpy --> enc
  enc --> pol
  hyp --> pol
  fm --> pol
  paths --> eng
  paths --> camp
  paths --> self
  sc -.cluster.-> eng
  eng --> self
  eng --> learn
  data --> cli
  models --> cli
  learn --> cli
  arena --> cli
  eng --> cli
  camp --> cli
  self --> cli
```

| Crate (directory) | Package name | Status on `main` today | What it owns |
|---|---|---|---|
| `crates/feat-math` | `feat-math` | landed (WP2) | Program DSL (`dsl.rs`, `genome.rs`, `spec.rs`), interpreter (`model.rs`, `heads.rs`, `gpr.rs`, `krr.rs`, `boost2.rs`, `portfolio.rs`), in-process search (`evolve.rs`), the op authority `registry.rs`, all emitters (`emit_py.rs`, `emit_rs.rs`, `emit_cpp.rs`, `emit_js.rs`, `transpile.rs`), the fitted C++ runtime (`advanced_v3/`, `cpp/fitted_runtime.cpp`), the one compiler helper `cxx.rs`. Crate-local `assets/champions/8944cbdf.dsl` (`include_str!`). |
| `crates/numpy-pcg64` | `numpy-pcg64` | landed (WP3) | Bit-exact numpy 1.26.4 `SeedSequence`/PCG64/`Generator` integer sampling and the frozen sampling recipes. |
| `crates/jope-encoders` | `starforge` (lib name kept so no source edit was needed) | landed (WP3) | Foundation encoder (Starlight / TabPFN-v3-reg, TabICL), target encoder, chimera fusion, hash-verified bundle loader. Weight-bound tests are env-gated. |
| `crates/battle-gpu` | `battle-gpu` | landed | Raw `dlopen("libcuda.so.1")` GPU screening for rung-0 ridge; feature `gpu`. No cudarc, so it never conflicts with candle. |
| `crates/jope-engine` | `jope-parallel` | landed (WP1) | The engine: program search (`gp.rs`, `grammar.rs`, `proposals*.rs`, `champions.rs`, `corpus*.rs`, `tournament.rs`), eval registry authority (`eval_registry/`), the `jope-parallel` binary with its **88 clap subcommands and global `--root`** (the jain-core consumer contract), 11 more binaries incl. `jope-build-receipt`; crate-local `assets/grammar/` (4 JSON, the committed `featmath-jain-v1.json` bakes the literal key `parallel/assets/grammar/featmath.json` — plan blocker B4) and `manifests/`. Feature `neural` = tch/LibTorch (self-hosted only); feature `cluster` (WP11) gates the smartcluster-bound modules. |
| `crates/jope-campaigns` | `jope-dime-regressor` (rename lands with WP5) | landed (WP1); tests compile since B6 | Benchmark campaigns: Golden42/46 panels, exact-CV, test-adaptive, DL best5, generalized search, search v3, hyperion gate/router, plus today the 55 `src/bin` programs. Crate-local `manifests/`, `seeds/`, `fixtures/` (`include_str!`/`include_bytes!` inputs, B1/B7/B8), `known-evaluator-identities.json` (ADR-0004). |
| `crates/jope-selfplay` | `jope-selfplay` | in flight (WP5, `wp5/cli-split`) | The flywheel control plane split out of the campaigns crate: ratchet, queue/supply/wave/promotion, watchdogs, TUI — one binary with argv[0] dispatch so unit files keep their `jope_selfplay_*` names. |
| `crates/hyperion` | `hyperion` | in PR [#6](https://github.com/veox-ai/JopeDime/pull/6) (WP3) | Hyperion v5 arm B / v7 inference (candle 0.11), router, `.hvc2` cache, GPU lease; bins `hyperion-v5-reference`, `hyperion-batch-worker`. `HYPERION_V7_WEIGHTS` replaces the historical absolute path. |
| `crates/jope-policy` | `jope-policy` | in PR [#6](https://github.com/veox-ai/JopeDime/pull/6) (WP3) | The step-60000 policy port: grammar/automaton, model, decode, ranking, seeds. |
| `crates/jope-paths` | `jope-paths` | in PR [#13](https://github.com/veox-ai/JopeDime/pull/13) (WP4) | The runtime roots of [ADR-0003](adr/ADR-0003-runtime-roots.md); zero dependencies; re-exported as `jope_parallel::paths`. |
| `crates/jope-datasets` | `jope-datasets` | in PR [#11](https://github.com/veox-ai/JopeDime/pull/11) (WP-D1/D5) | Catalog and collection types, `ingest` (de-dup), `fetch`/`verify`/`link` over `object_store`; the `datasets/test` verifier in hygiene. [docs/datasets.md](datasets.md). |
| `crates/jope-models` | `jope-models` | in flight (WP7, `wp7/models-cli`; manifests in PR [#9](https://github.com/veox-ai/JopeDime/pull/9)) | `jope models fetch / verify / publish / path` over the content-addressed models bucket. `docs/models.md` lands with WP7. |
| `crates/jope-learning` | `jope-learning` | in flight (WP11, `wp11/learning-store`) | The S3-facing half of the learning loop of [ADR-0005](adr/ADR-0005-canonical-learning-loop.md): HWM ledger, corpus snapshots, checkpoint lineage, wave pointers. `docs/learning.md` lands with WP11. |
| `crates/jope-arena` | `jope-arena` | in flight (WP12, `wp12/arena`) | SOTArena fetch, score emission, family Elo. `docs/arena.md` lands with WP12. |
| `crates/jope-cli` | `jope-cli` (binary `jope`) | on `main` since #17–#22: `verify-inputs`, `generate-fixtures`, `encode`, `fit`, `emit`, `search`, `champions`, `train-step`, `route`; the `models`/`datasets` umbrella verbs arrive with `wp7/models-cli` | The umbrella `jope search | emit | encode | propose | learn | hwm | scores | arena | campaign | selfplay | models | datasets | doctor`; subcommands attach as their crates land. |
| `crates/smartcluster/*` | `jain-smartcluster{,-core,-fabric,-daemon}`, `jain-worker`, `jainnode`, `scq` | landed (vendored at 715128f8, versions 8.0.0) | The scheduler the engine's native loop, smartcluster worker and native graph bind to; crates.io dependencies only. Not covered by `deploy/ci/version-consistency.sh`. |

Excluded from the workspace on purpose: `crates/hyperion/host` (superseded by `crates/hyperion`),
`deploy/kernel-v1/source` (the frozen kernel-v1 evaluator lineage that
`deploy/ci/kernel-v1-product-build.sh` builds inside a glibc-2.35 container; B11) and
`xtools/ot-export` (its own tch workspace).

## The Python package (`python/`)

**Status: being retired ([ADR-0006](adr/ADR-0006-rust-only-stack.md)).** Phase P1 removed the
Python with no live consumer (694 files, plus 9 unit templates that only ran it), among them all
of `zyal`, `fitness_corrector` and
`genome_perf`, all but one file of `chimera`, most of `tools` and `research`, and the kernel-v1
Python. [`docs/archive/python-archive-manifest.json`](archive/python-archive-manifest.json) lists
each with its restore command. What remains is the training stack (`policy`, `encoders`,
`target_maker`, `data`, `dime_score`, `broad_shadow` and the `jope-py` console script), the
research drivers and helpers that kept tests still use, the sha-pinned grammar sources and the
files that records digest-check; phases P2 and P3 port or retire them. New Python is not accepted
except the emitted model code and its harness.

One `uv` project, package `jope`, src layout `python/src/jope/` so the grammar pins keyed by
`jope/grammars/…` stay valid. Sub-packages by function as ported (WP6, PR
[#8](https://github.com/veox-ai/JopeDime/pull/8)): `encoders`, `policy`, `chimera`,
`target_maker`, `zyal`, `dime_score`, `fitness_corrector`, `grammars`, `tools`, `research` (the
former `scripts/` drivers, import-checked only), `tests/` with a `conftest.py` replacing the
historical `sys.path` hacks. Extras: `dev`, `cpu`, `cu130` (torch index per extra), `foundry`,
`research`. Tests that need the xbabe2 host carry the `xbabe2` marker and never run on hosted
runners.

## Contracts and schemas

Every record that crosses a stage boundary has a schema id in its `schema` field. The single
source per contract is `schemas/<name>.schema.json` (landing with WP10, crate `jope-contracts`);
Rust types round-trip through `schemars`; the remaining Python validates with `jsonschema` until
ADR-0006 retires it. Today's ids, by stage:

| Stage | Contract | Producer → consumer |
|---|---|---|
| datasets | `jope.dataset-catalog/v1`, `jope.dataset-collection/v1` | `jope datasets ingest` → `fetch`/`verify`, `deploy/ci/check.sh` |
| models | `jope.model-bundle/v2` (+ `jain.model-license/v1` blocks), `jope.model-index/v1` | `jope models publish` → `fetch`/`verify`, `deploy/ci/manifests.sh` |
| policy | `parallel.seed_population.v1`, `jope.checkpoint_export.v1` | `jope-policy` (export receipts are written by jain-datasets `tools/ot-export`) → `jope-engine search` |
| engine | `jain.jope-search-receipt/v1` (the jain-core consumer contract, `tests/cli_contract.rs`), `parallel.cut_champion.v1`, `parallel.teaching_shards.v1`, `parallel.checkpoint.v1`, `parallel.cycle.v1`, `parallel.gp_jobs.v1`, `parallel.clean_completion.v3` | `jope-parallel` subcommands → each other, `jope-learning` |
| evaluation | `JopeDimeEvaluatorIdentity.v1` + `known-evaluator-identities.json` (ADR-0004), `JopeDimeSemanticEvaluationReceipt.v2` | `jope-campaigns` `evaluation_v2.rs` |
| learning | `jope.hwm/v1`, `jope.score/v1`, `parallel.promotion_evidence.v2`, `parallel.promotion_decision.v2`, `RatchetArtifactPointerV1`, `LearningWaveManifest.v1`, `jope.learn-pull/v1` | `jope-learning hwm` / `jope-learning scores` |
| builds | `jope.product_evaluator_build.v1` (`jope-build-receipt`), `jope.sdk-receipt/v1` (`deploy/receipts/libtorch.json`, landed by PR [#25](https://github.com/veox-ai/JopeDime/pull/25)), `jope.cuda-qualification/v1`, `jope.release-manifest/v1` (`release.yml`) | CI lanes → artefacts / S3 `ci-receipts/` |
| parity | `jope.test-timings/v1` (`docs/parity/test-timings.json`, landed by PR [#29](https://github.com/veox-ai/JopeDime/pull/29)), the capability matrix and test inventory (landed by PR [#28](https://github.com/veox-ai/JopeDime/pull/28)) | WP1/WP10 → `deploy/ci/{parity-matrix,test-inventory}.sh` |

## Where state lives

- **Git:** source, grammars, manifests, seeds, fixtures, the 10 test datasets, model *manifests*,
  parity records. Nothing over 5 MiB (hygiene size cap), no weights, no evidence.
- **S3 (us-east-2, account-private):** `s3://veox-jopedime-models-use2` (`blobs/sha256/<sha>` +
  `bundles/<id>/manifest.json`; bundles `policy-step-60000`, `hyperion-v7`,
  `jope-prime-gen31-v2`), `s3://veox-jopedime-datasets-use2` (blobs, catalog, collections),
  `s3://veox-jopedime-learning-use2` (the `jope-learning` layout — HWM ledger, scores — once a
  `--store` mirror is configured; on 2026-09-23 it held only `staging/` second copies, see
  [docs/learning.md](learning.md)), `s3://veox-jopedime-archive-use2` (git bundles
  `git/20260916-…`, `git/20260918-jope-dime-all-refs.bundle` and `git-bundles/20260921/`, plus
  the `staging/node-high-value/` second copy; no `evidence/` batches in the 2026-09-23
  current-object inventory). Prefix status, sizes
  and replicas: [catalogs/s3.json](../catalogs/s3.json). CI reaches them only through the
  OIDC roles `JopeDimeCiRead`, `JopeDimeTrainer`, `JopeDimeRelease`.
- **Local:** the five roots of [ADR-0003](adr/ADR-0003-runtime-roots.md) — `JOPE_HOME` (the
  checkout), `JOPE_STATE` (mutable state, the restored `parallel/` tree), `JOPE_REGISTRY`
  (dataset cache by id), `SOTARENA_ROOT`, `SOTARENA_REPORTER_ROOT`.

## Determinism invariants

1. Emitted sources are byte-identical across runs and hosts (`deploy/ci/emitters.sh` pins their
   sha256).
2. Predictions of the interpreter and of every emitted program agree bit for bit
   (IEEE-754 pattern compare), across `g++`/`clang++` × C++17/C++20 and across `-march` values
   because every C++ build carries `-ffp-contract=off -fno-fast-math` ([docs/emitters.md](emitters.md)).
3. Numerical libraries run single-threaded in tests (`OMP/RAYON/OPENBLAS/MKL_NUM_THREADS=1`).
4. Every random choice takes an explicit seed; numpy streams are reproduced bit-exactly by
   `numpy-pcg64`.
5. Receipts bind outputs to inputs by sha256 (evaluator identity, bundle id, git sha), so a score
   can be replayed under the evaluator that produced it.

## Related

[docs/emitters.md](emitters.md) · [docs/datasets.md](datasets.md) ·
[docs/campaigns.md](campaigns.md) · [docs/adr/](adr/) ·
[docs/models.md](models.md) · [docs/runbooks/runner.md](runbooks/runner.md) ·
`docs/learning.md`, `docs/arena.md` (landing with WP11/WP12) ·
[docs/legacy/root-docs/](legacy/root-docs/) (the pre-port documents, kept verbatim for provenance).
