# JopeDime

[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![Main CPU CI](https://github.com/neverhuman/JopeDime/actions/workflows/full-main.yml/badge.svg?branch=main)](https://github.com/neverhuman/JopeDime/actions/workflows/full-main.yml)

Standalone JOPE-Dime: a typed program-synthesis engine for tabular models,
encoders (Hyperion and companions), and emitters that lower a program to Python,
Rust, C++17, and JavaScript. C++20 CI compiles the C++17 artifact; it is not a
second numeric dialect.

This is the product tree for [`neverhuman/JopeDime`](https://github.com/neverhuman/JopeDime),
the `repository` named in `Cargo.toml`. It is **not** the 8×A100 node workspace and **not** a
`jain-split` sibling. Training corpora and model weights live in S3. Git holds source, catalogs,
forty generated synthetic fixtures and ten tiny test datasets.

SOTArena **ASPFM** / `jope-dime` campaign wins are the feat-math program track. The
encodings→program neural policy will ship because the product needs a router; it is
**not** a held-out SOTA claim.

The 1.0.0 draft in [`CHANGELOG.md`](CHANGELOG.md) includes the GP language and its opt-in
LM refinement, Pareto archive, expression rewrite and minimax subtree routes. It also records
train-only raw-context admission and publication bindings, sealed-IPC descriptor ownership,
compiler caching and independently collected teacher-handoff checks. Source and fixture
receipts retain their scope; these changes do not establish a live learning or transfer result.

The [pending release queue](CHANGELOG.md#pending-release-queue) lists candidates awaiting
serialized merge and receiving-main qualification. This draft is local and unpublished; final
release notes must match the qualified merged source. It is prepared on main `df760a3ed`, whose
full-main CPU qualification is still failing. The 1.0.0 version and local metadata checks do not
establish release readiness. The current direction and landing record
are in [`MASTER_PLAN.md`](MASTER_PLAN.md), and release verification is in
[`docs/release/README.md`](docs/release/README.md).

Contributors and agents should read [AGENTS.md](AGENTS.md) before changing this repository.

## 1.0.0 — GP language

Versioned GP trees describe scalar expressions and admitted row and ordered-axis kernels.
Internal emitters generate Python, Rust and C++17 source directly from those trees, without
an LLM. The `cpp20` target compiles the same C++17 source as C++20. Export support and search
admission depend on the selected capability manifest and profile; experimental kernels retain
their documented bounds.

The GP emit targets are `python`, `rust`, `cpp17` and `cpp20`.

The [GP language coverage record](docs/parity/gp-language-tricks.md) describes tree representation
and differential compilation. The [emitter determinism record](docs/parity/gp-emitter-determinism.md)
describes repeated and cross-process emission, compiler checks, protected-domain behavior and
the documented numerical comparison contract.

The [capability matrix](docs/parity/capability-matrix.md) and operator receipts distinguish
implemented kernels, admitted exports and search profiles. The linked GP records state the
corpus and platform limits of the differential checks.

After publication, check out the release tag with `git checkout v1.0.0` and install the CLI with
`cargo install --locked --path crates/jope-cli --bin jope`. Release assets have signed checksums;
the Git tag is annotated. Publication follows the reviewed release and its qualification gates.

## v0.2.0 reference inventory

The following inventory records v0.2.0. Its counts describe that release; the linked receipts
and commands identify their sources.

| What | 0.2.0 |
| --- | --- |
| `jope` with no arguments | prints `jope 0.2.0` <sup>1</sup> |
| Workspace | 26 members: 19 crates at 0.2.0 and 7 vendored SmartCluster crates at 8.0.0; 8 are `default-members` <sup>2</sup> |
| `jope` subcommands | 17 in `jope --help` on Linux, plus the hidden `motifs` <sup>3</sup> |
| Emit targets | 5: `python`, `rust`, `cpp17`, `cpp20` (the C++17 source compiled as C++20) and `javascript` <sup>4</sup> |
| Rust tests | 3,796 listed in 26 packages, 62 of them ignored; a listing, not a run <sup>5</sup> |
| Python tests | 172, found by a static scan, not collected or run <sup>5</sup> |
| CI | 9 jobs per pull request in `ci.yml`; 11 `full-main-cpu` families on every push to `main` <sup>6</sup> |
| Coverage map | 425 items over 23 lanes <sup>7</sup> |
| Capability matrix | 818 rows: 353 moved, 55 new, 1 pending, 402 dropped, 7 lost <sup>8</sup> |
| Datasets in Git | 40 synthetic fixtures (10 each of binary, multiclass, regression and time series) and 10 test datasets <sup>9</sup> |
| Python still tracked | 216 `.py` files, 157 of them under `python/` <sup>10</sup> |
| Architecture decision records | 6 <sup>10</sup> |

1. `cargo run --locked --bin jope`
2. `cargo metadata --locked --no-deps --format-version 1 | jq '(.workspace_members, .workspace_default_members | length), ([.packages[].version] | group_by(.) | map({(.[0]): length}) | add)'`
3. `cargo run --locked --bin jope -- --help`; the hidden verb is the `#[command(hide = true)]` variant of `enum Command` in `crates/jope-cli/src/main.rs`
4. the single-language arms of `fn languages` in `crates/jope-cli/src/kernel.rs`
5. `jq '.rust.count, .rust.ignored, (.rust.packages | length), .python.count' docs/parity/test-inventory.json`
6. the top-level `jobs:` of `.github/workflows/ci.yml`; `jq '[.lanes | keys[] | select(startswith("full-"))] | length' deploy/ci/coverage.json` (the families `deploy/ci/full-main.py` runs)
7. `jq '.items, .lanes | length' deploy/ci/coverage.json`
8. the **total** row of the Counts table in `docs/parity/capability-matrix.md`
9. `jq '.datasets | length' datasets/synthetic/catalog.json`; `jq '.members | length' datasets/manifest.json`
10. `git ls-files '*.py' | wc -l`; `git ls-files 'python/*.py' | wc -l`; `git ls-files 'docs/adr/ADR-*.md' | wc -l`

## What this SHA actually does

| Surface | Status on this SHA |
| --- | --- |
| `jope` CLI | Fixtures (`verify-inputs`, `generate-fixtures`), CPU `encode` (fails closed without the Hyperion weights), `fit` / `search` / `emit`, `champions --local` (searches the supported regression fixtures and reports the rest as unsupported), `train-step --local` (fits a 128-d centroid router), `route`, `models`, `datasets`, `pipeline`, and the private ASM verbs `doctor`, `demo`, `benchmark`, `optimize` and `train` ([docs/asm.md](docs/asm.md)). `champions --pull/--push` and `train-step --env dev` or `--env prod` **fail closed**. |
| `jope pipeline run` | CPU spine only: dataset -> folds -> program search -> select -> emit -> `jope.pipeline-receipt/v3`, optionally seeded from a `jope.pipeline-seeds/v1` file (`--seed-genomes`) or a verified parent receipt (`--incumbent`); seeds are re-measured, never trusted. Every candidate is fitted with `feat_math::fit` on a cut's training rows and scored on that cut's held-out rows, and selection requires measurements on every requested cut before comparing mean held-out RMSE. Exploration and cross-cut comparison share the [total per-cut budget](docs/pipeline.md). `--threads` and `--islands` (parallel search groups over the same cuts that pass their best program to each other at a fixed interval) set the workers, which must fit the CPUs the process may use and `--cpu-ceiling`. It does **not** use the engine's own search, which cannot measure in a shipped binary (`eval_registry/authority.rs` blocks every capability outside `cfg(test)` and `fit_boundary.rs` fences fitting behind a service socket). The encode/propose stage is **skipped** and recorded (`--policy require` fails closed); nothing in it publishes to S3 or trains the router. |
| Targeted single-dataset search | `jope pipeline evolve` runs rounds of `run` on one dataset's fixed cuts, each seeded from the previous round's champion and runners-up, chained in a `lineage.json` that `jope pipeline lineage verify` re-derives; `--control` adds one unseeded run. `jope pipeline score --split test` scores a selected program on the sealed test split at most once per dataset and program in its registry (a local record, not access control), and lists every earlier look. High-water-mark ingest (`jope-learning hwm ingest --source pipeline`) and the append-only local champion bank (`jope-learning bank add`, `list`, `export`; an export seeds the next round) are separate steps. The rounds select on reused folds, so only the test split speaks to generalisation. |
| W7 native loop | `jope-native-loop`, a `jope-engine` binary: host census, run plans, signed participant grants, authenticated cross-host transport with pinned certificates and an absolute deadline on every read and write, controller recovery, and SmartCluster dispatch of bounded ranks that prove their job before any work. It is qualified on source and fixtures only: a same-host fixture with hermetic host identities, and a local one-rank dry run. The three-host proof ([runbook](docs/runbooks/w7-three-host-cpu-proof.md)) has **not** run and needs its own admission; live cross-host checkpoint exchange and continuation routing are deferred. |
| S3 | `jope datasets` and `jope models` read and write `s3://` stores through `object_store`, resolving `credential_process` profiles themselves; `jope-learning` mirrors its store and publishes W8 progress when `--store` or `JOPE_LEARNING_URL` names one. `champions --pull/--push` has no live route (issue #62). No live S3 training run has been made, and the bound `jope-policy` checkpoint weights are **not** updated. |
| Encoders and policy | CPU Hyperion on Candle (`hyperion-cpu`) is the default; a CUDA feature exists but is not the CI default. `jope-policy` needs the bound step-60000 checkpoint to propose. |
| Git datasets | Forty synthetic fixtures in `datasets/synthetic/`, which CI regenerates and compares (`jope generate-fixtures --check`) and verifies (`jope verify-inputs`), and the ten licensed test datasets (`test-10`) in `datasets/test/`. Real tables stay in S3. |
| Release | A signed host release from `deploy/release/release.sh`, with no Python wheel; see [Releases](#releases). |

## Measured results

Generated by `jope-records readme-benchmarks` from committed evidence. Do not edit between the markers by hand.

<!-- arena:begin -->
### Golden45 — regression, 45 datasets

`tabular-regression-v2` · primary metric **RMSE** (minimize) · Elo by `regularized-bradley-terry-elo-scale` · test labels are **not public**; 3 datasets are restricted; excluded datasets: ZTS.

#### Benchmark-valid — the ladder this repository claims

**304 methods** · 8,718 rankable scores · 45 published rows excluded as test-selected (see the next block). Bold rows are this repository's entries. *Benchmark-valid* here means that no artefact in this repository contradicts the row's published flag; it is not a claim that every comparator's protocol was verified.

| rank | method | Elo | avg rank | dataset wins | coverage | protocol |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | TabFM Ensemble | 1934.9 | 21.51 | 7 | 45/45 | `public-development` |
| 2 | TabPFN v3 | 1909.5 | 24.16 | 1 | 45/45 | `public-development` |
| 3 | TabPFN v3 Default | 1906.3 | 24.51 | 0 | 45/45 | `public-development` |
| 4 | Tabpfn V25 Regressor Default | 1893.3 | 25.98 | 5 | 45/45 | `public-development` |
| 5 | Tabicl | 1887.6 | 26.64 | 4 | 45/45 | `public-development` |
| … | | | | | | |
| **31** | **Jope Dime Regressor** | **1687.0** | 60.59 | 1 | 42/45 | `public-development` |

**Read the protocol column before the Elo column.** `leaderboard_policy.protocols_ranked_together` is `true`, so entries produced under different protocols share one ladder. `jope-dime-regressor` is one fixed-seed reproducible run of this repository's engine under `public-development`, the same protocol as every other comparator. The same-cut benchmark is what turns this into a like-for-like comparison.

#### Best known program per dataset — as published; test-selected, not a held-out result

The benchmark published `jope-dime` at rank 1, Elo 2259.8, 11 dataset wins, 45/45 coverage, every row `benchmark_valid: True`. This repository's own code says otherwise about every one of those rows, so the benchmark-valid ladder above carries none of them:

- **42 Golden42 rows** are each the per-dataset minimum RMSE over the ledger's test-scored candidate evaluations (63,108 across the 42 datasets), not the score of one program frozen before the test targets were opened. The red-team lane in this tree names that quantity the *best-test ASPFM envelope*, role `oracle_reference_only`, and keeps it out of its fair ladder (`python/src/jope/research/index_combined_regression_evolution.py:183`, `crates/jope-campaigns/src/commands/train_only_redteam.rs:402-407`).
- **3 Golden46 rows** come from run `golden46-new3-20260723-v4`, whose artefacts this repository's campaign code writes with `benchmark_valid: false, test_adaptive: true` (`crates/jope-campaigns/src/test_adaptive/aggressive.rs:732-748`); the acceptance manifest relabelled the replayed winners `true`.

Both are the best program found per dataset when candidates were scored on the test targets: reported here as best-known results, never as a held-out claim. The reproducible fixed-seed run of the same engine, `jope-dime-regressor`, is what stands on the ladder above. The same-cut benchmark replaces every row on this page with numbers this tree can reproduce.

| ladder | `jope-dime` | Elo | dataset wins | coverage |
| --- | ---: | ---: | ---: | --- |
| as published | rank 1 | 2259.8 | 11 | 45/45 |
| benchmark-valid | not ranked | — | — | 0/45 (every row test-selected) |

Sources: published `assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv` (sha256 `2d28ffbd8b5830c6…`), recomputed byte-for-byte from `scores.csv`; benchmark-valid `assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv` (sha256 `4c39f51c2e269114…`), the same `scores.csv` with `assets/regression/protocol-corrections-tabular-regression-v2.json` (sha256 `6a831c4a97c2fc29…`) applied. Both are asserted by `crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:

```sh
cargo run -p jope-arena --bin jope-arena -- elo \
  --track tabular-regression-v2 --root assets/regression/sotarena-tabular-regression-v2 --check
cargo run -p jope-arena --bin jope-arena -- elo \
  --track tabular-regression-v2 --root assets/regression/sotarena-tabular-regression-v2 \
  --ladder benchmark-valid --corrections assets/regression/protocol-corrections-tabular-regression-v2.json \
  --check --committed assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv
```
<!-- arena:end -->

<!-- samecut:begin -->
### Same-cut — every method on identical held-out rows, reproducible from this tree

`jope-same-cut-v1-deep` · 10 regression datasets from `datasets/test` · primary metric **RMSE** (minimize) · fold plan k=5, seed 20260907, screen fold 0, top 5 deepened · search budget 32 programs × 8 rounds, re-run on every cut · recorded at `8c90a3279` with 16 workers on 128 cores.

Every row here is a number this tree produced and can reproduce: the ten single-head baselines, four external engines (XGBoost, LightGBM, CatBoost and a scikit-learn random forest, each run as a subprocess with its library defaults, a fixed seed and one thread, handed a cut's training rows and its held-out features only, and scored by this benchmark from the predictions it returns) and the program the search wrote for each dataset, all fitted on the same training rows and scored on the same held-out rows by one evaluator (`docs/same-cut-bench.md`). The search sees only a cut's training rows, and its deep row is the *procedure* re-run on every cut, not one program chosen with knowledge of the other cuts' held-out rows. Bold is the searched program.

**The datasets are a favourable selection.** They are the `test-10` collection, chosen from the published ladder where the historical engine ranked first (`docs/datasets.md`), and they are used here because they are the only committed regression data. Identical cuts make the comparison between methods fair on these ten; they say nothing about a broader panel. The frozen 30-dataset PMLB panel of the experiment contract (`docs/asm.md`) excludes them on purpose; a same-cut run over it is the next evidence step.

| rank | method | Elo | avg rank | dataset wins | coverage |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | baseline.gpr | 1711.6 | 1.00 | 4 | 4/10 |
| 2 | adapter.rf (scikit-learn 1.7.2 RandomForest) | 1560.0 | 2.00 | 0 | 1/10 |
| **3** | **afm.lite-race.v1.resampled** | **1556.6** | 2.60 | 2 | 5/10 |
| 4 | baseline.ridge | 1546.4 | 2.00 | 0 | 1/10 |
| 5 | baseline.stack | 1539.4 | 2.71 | 0 | 7/10 |
| 6 | baseline.ensemble | 1538.1 | 2.50 | 2 | 4/10 |
| 7 | baseline.krr | 1488.7 | 3.00 | 1 | 6/10 |
| 8 | adapter.xgboost (xgboost 2.1.4) | 1486.6 | 3.00 | 1 | 2/10 |
| 9 | adapter.catboost (catboost 1.2.10) | 1470.5 | 3.33 | 0 | 3/10 |
| 10 | baseline.histgbm | 1470.1 | 4.00 | 0 | 2/10 |
| 11 | baseline.ridge_gcv | 1465.9 | 3.33 | 0 | 3/10 |
| 12 | adapter.lightgbm (lightgbm 4.6.0) | 1461.9 | 3.33 | 0 | 3/10 |
| 13 | baseline.ksmooth | 1446.1 | 4.00 | 0 | 1/10 |
| 14 | baseline.elasticnet | 1424.1 | 4.00 | 0 | 2/10 |
| 15 | baseline.histgbm2 | 1333.7 | 4.33 | 0 | 6/10 |

**Where the search stands today:** rank 3 of 15 on the deep ladder, Elo 1556.6, lowest pooled RMSE on 2 of 10 datasets (the ladder's own rule), a finalist on 5; on the other 5 it did not make the top-5 screen. Against the four external engines it ranks above adapter.catboost (rank 9), above adapter.lightgbm (rank 12), below adapter.rf (rank 2), above adapter.xgboost (rank 8). Coverage counts datasets where a method reached the deep stage, so the ladder ranks finalists only. The screen ladder (every method, one cut) is committed beside it.

**Seconds per program.** Writing the program took the search 2.2 s per dataset on average (median 1.0 s, longest 8.3 s) on the screen cut's training rows, wall time under whatever else the host was running. The owner's stated reference point for a large language model writing a program for the same task, not measured here, is about 1 hour and 10,000,000 tokens; that comparison is one of cost, not of quality, and the quality standing is the ladder above.

| dataset | rows | features | lowest pooled RMSE | RMSE | search rank (penalised score) | search |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `047e62b52a3f4ede` | 195 | 12 | **afm.lite-race.v1.resampled** | 0.0712 | 1 | 0.4 s |
| `1cb97b1d7314d43d` | 166 | 10 | baseline.gpr | 0.06635 | 2 | 8.3 s |
| `21fbfe8b53dbcd60` | 1000 | 21 | adapter.xgboost | 0.02428 | screen 10 | 1.4 s |
| `30ef3eaa6d2971a9` | 109 | 30 | **afm.lite-race.v1.resampled** | 0.071 | 1 | 1.3 s |
| `43b4cc3ab0ebbd97` | 482 | 39 | baseline.krr | 0.09057 | screen 6 | 1.3 s |
| `693890aae9dc860b` | 202 | 1 | baseline.gpr | 0.004291 | 4 | 6.6 s |
| `79e600bc5fcd5da7` | 180 | 3 | baseline.gpr | 0.1491 | screen 8 | 0.4 s |
| `b5beff4a53fa12c0` | 575 | 15 | baseline.ensemble | 0.05628 | screen 13 | 0.7 s |
| `c32fa09b48fb2f27` | 1000 | 9 | baseline.ensemble | 0.185 | screen 9 | 0.7 s |
| `c7d815714ca52bce` | 1000 | 7 | baseline.gpr | 0.1808 | 4 | 0.7 s |

Sources: `assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv` (sha256 `bf2c6dc0e8267492…`) recomputed byte-for-byte from `assets/same-cut/v2/track/results/jope-same-cut-v1-deep/scores.csv`, itself re-derived from the 10 reports under `assets/same-cut/v2/reports` by `crates/jope-bench/tests/committed_evidence.rs`, which also re-evaluates one dataset's baselines on the same cuts. Provenance and the exact commands: `assets/same-cut/v2/SOURCES.md`. Reproduce:

```sh
cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \
  --adapter "catboost=python3 crates/jope-bench/adapters/catboost_adapter.py" \
  --adapter "lightgbm=python3 crates/jope-bench/adapters/lightgbm_adapter.py" \
  --adapter "rf=python3 crates/jope-bench/adapters/sklearn_rf_adapter.py" \
  --adapter "xgboost=python3 crates/jope-bench/adapters/xgboost_adapter.py" \
  --top 5 --search --search-pop 32 --search-gens 8 --resample-search --workers 16
cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \
  --track-root target/same-cut/track --track jope-same-cut-v1
cargo run -p jope-arena -- elo --track jope-same-cut-v1-deep --root target/same-cut/track \
  --check --committed assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv
```
<!-- samecut:end -->

<!-- bench:begin -->
### Emit-path latency

`emit-champion-8944cbdf` — 200 runs of the release `emit` example on `crates/feat-math/assets/champions/8944cbdf.dsl`.

| min | median | mean | p95 |
| ---: | ---: | ---: | ---: |
| 3.61 ms | 20.59 ms | 48.40 ms | 222.33 ms |

**This is not an engine-speed number.** Its own `what` field says it measures "wall time of one `feat-math` release `emit` example run (process spawn included) on the committed champion; lower is better". The min-to-p95 spread is dominated by process spawn and scheduling, not by emission. Treat it as a regression tripwire (alert at 10% drift), not as a performance claim.

Recorded 2026-09-19T00:34:16Z on `xbabe2` (x86_64, 128 cores, kernel 6.8.0-139-generic) at `a53a1499d` with rustc 1.97.1 (8bab26f4f 2026-07-14). Baselines are host-pinned; `bash deploy/ci/regression-suite.sh bench` compares against this file.
<!-- bench:end -->

## Install and check

Linux x86_64. Rust from `rust-toolchain.toml`. `cargo build --locked` builds the
workspace `default-members`: the `jope` binary and the crates it depends on, CPU only
(no LibTorch, CUDA, or cluster runtime). `cargo build --workspace --locked` builds
every member.

```sh
cargo build --locked
cargo run --locked --bin jope
# jope 1.0.0
cargo run --locked --bin jope -- verify-inputs --offline --root datasets/synthetic
# Verified 40 synthetic datasets, … bytes, offline.
# Encode needs the Hyperion bundle (not in git):
# HYPERION_V7_WEIGHTS=… cargo run --locked --bin jope -- encode --input datasets/synthetic/binary/linear.json
cargo run --locked --bin jope -- fit --input datasets/synthetic/regression/linear.json --json
cargo run --locked --bin jope -- emit --lang all --output target/jope-emitted --json
cargo run --locked --bin jope -- search --input datasets/synthetic/regression/linear.json --json
cargo run --locked --bin jope -- champions --local --root datasets/synthetic --json
cargo run --locked --bin jope -- train-step --local --root datasets/synthetic --output target/jope-router --json
cargo run --locked --bin jope -- route --input datasets/synthetic/regression/linear.json --router target/jope-router --json
cargo run --locked --bin jope -- pipeline run --dataset datasets/test/pmlb-229-pwlinear \
  --budget evaluations=8,seconds=120 --emit rust,python,cpp17,cpp20,js --policy skip --output target/e2e
cargo run --locked --bin jope -- pipeline verify-receipt --receipt target/e2e/receipt.json
```

Prove emitters on this SHA:

```sh
JOPE_PARITY_STRICT=1 cargo test --locked -p feat-math
```

Pull-request CI is `.github/workflows/ci.yml`: nine jobs on the self-hosted `jope-ci` runners.
Eight each run one lane of `deploy/ci/fast-lane.sh`, and `aggregate` validates their receipts
(`deploy/ci/aggregate.py`). The hygiene lane includes `deploy/ci/check.sh`
(formatting, the coverage map, the prose rule, the repository layout and the README's generated
blocks) and builds and tests the standalone AWS Lambda workspaces under `deploy/lambda`
(`deploy/ci/lambda.sh`). Every push to `main` runs `main-guard` and `full-main-cpu`. Locally,
`just ci-pr` (alias `just required`) runs `check.sh`, `cargo fmt --check`, clippy with
`-D warnings`, the pull-request nextest tier, and ruff and pytest over the remaining Python; each
tool must be installed.
`just lint-paths` runs only the absolute-home-path check.

## Releases

A release is cut on xbabe2 by [`deploy/release/release.sh`](deploy/release/release.sh) from a merge
commit on `main`: a dry run first, then the same command with `--publish`, which creates the
annotated tag `vX.Y.Z` and the GitHub release and reads both back. A release holds the `jope` binary
(built in the digest-pinned `rust:1.97.1-bookworm` image, glibc 2.36 at most), a build receipt,
SPDX and CycloneDX SBOMs, the release notes, a release manifest and `SHA256SUMS`, signed with cosign
by a release key held on xbabe2. There is no Python wheel and no provenance attestation.
[`release.yml`](.github/workflows/release.yml) is dispatch-only and fails closed. This tree is the
source of v1.0.0; the release exists only once `release.sh --publish` has run.

To verify one, take `signing.pub` from the tag and check its fingerprint as
[docs/release/README.md](docs/release/README.md) describes, then check the signature and the sums:

```sh
TUF_ROOT="$D/.tuf" cosign verify-blob --key signing.pub --insecure-ignore-tlog=true \
  --signature "SHA256SUMS-$TAG.txt.sig" "SHA256SUMS-$TAG.txt"
sha256sum -c --strict "SHA256SUMS-$TAG.txt"
```

## Intended pipeline

```text
dataset cut
  → encoders (Hyperion + foundation + target)
  → encodings context
  → policy that proposes / routes programs
  → typed program kernel (feat-math)
  → emitters → working Python / Rust / C++17 / JS
  → S3: best programs per dataset
  → train the policy on those champions
```

Do not open a stack of PRs. Merge or close the current one before starting the next.

## Layout (this SHA)

The 26 workspace members, by directory under `crates/`; the package name follows where it differs.
[docs/architecture.md](docs/architecture.md#crate-graph) describes each crate.

| Group | Crates | Role |
| --- | --- | --- |
| Default build | `jope-cli` (binary `jope`), `feat-math`, `jope-pipeline`, `jope-bench`, `jope-datasets`, `jope-models`, `jope-motifs`, `jope-policy` | The workspace `default-members`, what a bare `cargo build` selects: the `jope` binary and its direct workspace dependencies except `hyperion` and `jope-learning`, which it still builds as dependencies (`Cargo.toml` says why). Programs, fit and emitters; the pipeline; the same-cut benchmark; dataset and model stores; program fragments; the proposal policy. |
| Encoders | `jope-encoders` (`starforge`), `hyperion`, `numpy-pcg64` | Encoders, Hyperion v5/v7 on Candle, and a bit-exact numpy RNG. |
| Engine and campaigns | `jope-engine` (`jope-parallel`), `jope-campaigns` (`jope-dime-regressor`), `battle-gpu`, `jope-arena` | The engine and its binaries, `jope-native-loop` among them; benchmark campaigns; raw-CUDA screening (feature `gpu`); SOTArena scores and Elo. |
| Learning, records and CI | `jope-learning`, `jope-records`, `jope-ci`, `jope-paths` | The HWM ledger, champion bank and W8 progress; record verifiers and the hygiene guards; CI tooling; the ADR-0003 runtime roots. |
| Vendored | `smartcluster/*` (7 crates) | The jain-smartcluster scheduler, at its own version 8.0.0. |

| Path | Role |
| --- | --- |
| `datasets/synthetic` | The forty generated fixtures and `catalog.json` |
| `datasets/test` | The ten licensed test datasets (`test-10`) |
| `python/` | The Python that remains after [ADR-0006](docs/adr/ADR-0006-rust-only-stack.md) phase P1 (#117): the training stack (`policy`, `encoders`, `target_maker`, `data`, `dime_score`, `broad_shadow`), the research helpers that kept tests use, and the sha-pinned grammar sources; package `jope` 0.1.0, never released. Phases P2 and P3 port or retire it, with the CI helpers in `deploy/ci/*.py`. No new Python except the emitted model code and its harness. |
| `deploy/ci`, `deploy/release` | CI lane scripts and the coverage map (`check.sh` is the hygiene lane); the release script and its tests |
| `deploy/lambda` | Standalone AWS Lambda workers, an offline assessment tool for the regression worker's stored models, the host-side search engine of the teacher campaign and a terminal monitor, each its own Cargo workspace with its own lock; not part of the `jope` binary or its build. `SNAPSHOT.json` records every imported file's hash ([README](deploy/lambda/README.md)) |

## Further reading

- [`CHANGELOG.md`](CHANGELOG.md) — what changed in 0.2.0
- [`MASTER_PLAN.md`](MASTER_PLAN.md) — current direction and the landing record
- [`docs/release/README.md`](docs/release/README.md) — how a release is cut, signed and verified
- [`docs/pipeline.md`](docs/pipeline.md) — the pipeline's budget, selection, seeding, parallelism and test-split contracts
- [`docs/architecture.md`](docs/architecture.md) — crates, contracts and the intended pipeline
- [`docs/emitters.md`](docs/emitters.md) — per-target coverage of the emitters
- [`docs/datasets.md`](docs/datasets.md) — dataset store layout, licence rule and `datasets/test`
- [`docs/campaigns.md`](docs/campaigns.md) — benchmark campaigns
- [`docs/runbooks/`](docs/runbooks/) — operator runbooks, the W7 three-host proof among them
- [Private maintainer manual](docs/asm-maintainer-manual.md) — detailed PDF, section Markdown, current evidence, dataset inventories and acceptance checklist
- [`docs/quickstart.md`](docs/quickstart.md) — the ASM build and the feat-math quickstart (timings measured at `4235db6`, not this SHA)
- [`docs/adr/`](docs/adr/) — architecture decision records 0001–0006
- [`docs/archive/`](docs/archive/) — retired plans and the restore manifests of removed files
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
