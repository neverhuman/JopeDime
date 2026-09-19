# JopeDime

**Dataset → encoders → policy → GP genome → emitted Python / Rust / C++17 / C++20 / JavaScript.**

JopeDime turns a tabular dataset into a standalone, dependency-free model program. Frozen
foundation encoders (Starlight / TabPFN-v3-reg, TabICL) and the Hyperion signature encoder embed
the dataset; a policy transformer (step-60000) proposes candidate genomes in a small S-expression
DSL; a genetic-programming engine searches and evaluates them under exact cluster cross-validation;
the winning genome is lowered by `feat_math::transpile` to source code in five targets whose
predictions are **bit-identical** to the interpreter and to each other. The learned state (GP
high-water marks, corpora, checkpoints, serving pointers) lives in S3 and is the training signal
for the next policy — the learning loop of [ADR-0005](docs/adr/ADR-0005-canonical-learning-loop.md).

This is the standalone product repository (`veox-ai/JopeDime`, private now, public later —
owner decision 1). Source, grammars, manifests, seeds, small fixtures and ten tiny test datasets
are in git; datasets, model weights, the learning store and the unfiltered history are in S3 and
addressed by hash. Licence: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

## Quickstart (laptop, CPU only)

Requirements: Rust 1.97.1 via `rustup` (`rust-toolchain.toml` selects it; MSRV 1.96), `g++`
(feat-math's `build.rs` compiles the fitted C++ runtime), and for the emitter matrix `clang++`,
`node` and `python3`. No CUDA, LibTorch or cluster runtime is needed for anything below. Every
command was run on this tree on 2026-09-18/19 (128-core host, `-j 16`; a laptop will be slower):

```bash
git clone https://github.com/veox-ai/JopeDime && cd JopeDime

# 1. the whole workspace, CPU features only                          (1 m 57 s)
cargo build --workspace --locked

# 2. the GP engine and emitters: 408 tests incl. the four-language   (13 m 01 s, includes compiling
#    parity proofs (g++/clang++/node/python3 are exercised)           the test binaries)
cargo test -p feat-math --locked

# 3. the emitted-code matrix: two genomes on datasets/test/pmlb-229-pwlinear —
#    Python, Rust, g++/clang++ × C++17/C++20 and node produce bit-identical predictions
#    and byte-identical sources pinned in crates/feat-math/tests/fixtures/emitted/  (20 s)
bash deploy/ci/emitters.sh

# 4. emit one genome yourself: model.py, model.rs, model.cpp, model.js + manifest.json (0.2 s)
cargo run -p feat-math --example emit -- crates/feat-math/tests/fixtures/emitted/portable-4lang.dsl /tmp/out
```

Known state of step 2 on `main` today: `legacy_v2_freeze::legacy_v2_reference_files_match_the_takeover_snapshot`
fails because `tests/golden.rs` was rustfmt-ed in `4235db6` and its frozen sha256 is re-pinned in
PR [#10](https://github.com/veox-ai/JopeDime/pull/10) (whitespace-only change, `git diff -w`
empty); the other 407 tests pass. The full PR tier is `just ci-pr` (hygiene, fmt, clippy,
`cargo nextest -P pr`, Python checks); `just docs` runs rustdoc and the markdown link check.

To emit a genome of your own, write its DSL on one line — the champion in
`crates/feat-math/assets/champions/8944cbdf.dsl` is
`(pipe X (winsor_clip append=true q=0.1) (cheb append=true degree=6) (robust_scale) (robust_scale) (poly_interact append=true) (quantile_bins bins=16) (histgbm depth=5 lr=0.15 rounds=150))`
— and pass it to the `emit` example; `--all` makes an uncovered target an error instead of a note
(the `histgbm` head is Python/Rust only today; see [docs/emitters.md](docs/emitters.md)). Every
emitted program takes `train.csv test_features.csv predictions` (headerless CSV, last train column
is the target); the C++ and JavaScript artefacts are libraries, and
`deploy/ci/emitters/driver.{cpp,js}` are the CSV drivers the matrix appends to them.

## Repository map

| Path | What is there |
|---|---|
| `crates/feat-math` | The GP engine: DSL, interpreter, heads, evolution, op registry, **all emitters**, fitted C++ runtime, the compiler helper `cxx.rs` |
| `crates/jope-engine` | The search/eval engine (`jope-parallel`: 88 subcommands, global `--root`), eval registry, grammar assets, `jope-build-receipt` |
| `crates/jope-campaigns` | Benchmark campaigns (Golden42/46, exact-CV, test-adaptive, DL-GP, search v3) and, until WP5 lands, the self-play control plane |
| `crates/jope-encoders`, `crates/numpy-pcg64` | Foundation/target/chimera encoder inference (lib `starforge`); bit-exact numpy RNG |
| `crates/hyperion`, `crates/jope-policy` | Hyperion v5/v7 signatures and the step-60000 policy (candle) — PR [#6](https://github.com/veox-ai/JopeDime/pull/6) |
| `crates/jope-cli` | The `jope` umbrella binary (a version stub today; subcommands attach as crates land) |
| `crates/smartcluster/*` | The vendored jain-smartcluster workspace (7 crates @ `715128f8`) |
| `crates/battle-gpu` | Raw CUDA-driver screening, feature `gpu` |
| `python/` | The `jope` uv package: encoders, policy trainer, chimera, target_maker, research drivers, tests |
| `datasets/` | `manifest.json` (collection `test-10`) + `test/<slug>/{train.csv,test.csv,metadata.json,LICENSE}` — [docs/datasets.md](docs/datasets.md) |
| `models/` | Bundle manifests, index and `PINNED` only — never weights (PR [#9](https://github.com/veox-ai/JopeDime/pull/9)) |
| `deploy/` | `ci/*.sh` (the lane scripts), `systemd/{selfplay,engine,pending}` unit templates, `kernel-v1/source` (frozen evaluator lineage), `receipts/`, `mcp/` |
| `docs/` | [architecture](docs/architecture.md), [emitters](docs/emitters.md), [datasets](docs/datasets.md), [campaigns](docs/campaigns.md), [ADRs](docs/adr/), [runbooks](docs/runbooks/), [public-readiness checklist](docs/public-readiness-checklist.md), `parity/`, `legacy/` |
| `xtools/` | Repository maintenance tooling (`ops-sh/`, `registry-dump-legacy/`; history and S3 scripts) |
| `.github/` | `workflows/{ci,release}.yml` (+ `main-guard.yml` with PR #7), `CODEOWNERS`, `dependabot.yml` |

## Models and datasets

Nothing large is in git. Both stores are content-addressed and manifest-pinned:

- **Models** — `s3://veox-jopedime-models-use2`: `blobs/sha256/<sha>` + `bundles/<id>/manifest.json`.
  Bundles today: `policy-step-60000`, `hyperion-v7`, `jope-prime-gen31-v2`; the repo commits their
  manifests under `models/` and the lanes that need weights read `models/PINNED`.
  `jope models fetch <bundle-id>` (crate `jope-models`, landing with WP7) downloads by hash into
  `$JOPE_HOME/models/`, prints each file's licence block and refuses `redistribution: prohibited`
  without `--accept-restricted`; `jope models verify` re-hashes a cache. Pull-request lanes never
  download weights.
- **Datasets** — `s3://veox-jopedime-datasets-use2`: `blobs/sha256/<sha>`, `datasets/<hash>/manifest.json`,
  `collections/<name>/v<N>.json`, `catalog/`. The ten committed test datasets under `datasets/test/`
  are the public collection `test-10`; `jope datasets fetch <collection|id>` (crate
  `jope-datasets`, PR [#11](https://github.com/veox-ai/JopeDime/pull/11)) populates
  `$JOPE_HOME/datasets/{blobs,by-id}` with `.partial` + rename on hash match, and
  `JOPE_OFFLINE=1` serves the cache only. `JOPE_REGISTRY` (default `$JOPE_HOME/datasets/by-id`) is
  the directory of `<hash>/{train.csv,test.csv,meta.json}` the engine reads.
- **Learning store** — `s3://veox-jopedime-learning-use2` (HWM ledger, scores, corpora,
  checkpoints, wave pointers; `jope learn` / `jope hwm` / `jope scores`, WP11).
- **Archive** — `s3://veox-jopedime-archive-use2`: the unfiltered git bundles
  (`git/20260916-…`, `git/20260918-jope-dime-all-refs.bundle`, with the filter-repo commit-map and
  ref-map beside them) and evidence batches.

Runtime roots (`JOPE_HOME`, `JOPE_STATE`, `JOPE_REGISTRY`, `SOTARENA_ROOT`,
`SOTARENA_REPORTER_ROOT`) are resolved once by `jope-paths`
([ADR-0003](docs/adr/ADR-0003-runtime-roots.md), PR [#13](https://github.com/veox-ai/JopeDime/pull/13)).

## CI

Pull requests run on hosted `ubuntu-24.04` runners with no credentials and no self-hosted label
(`.github/workflows/ci.yml`): `hygiene` (`deploy/ci/check.sh`: `cargo metadata --locked`,
shellcheck, actionlint, gitleaks, no-family-leftovers, no-absolute-paths, size caps; then
cargo-deny) on every PR, and with PR [#7](https://github.com/veox-ai/JopeDime/pull/7) the
path-gated `rust-check` and `emitters` lanes plus `main-guard.yml`. The Python lane, the
`rust-engine` nextest lane (`.config/nextest.toml` profile `pr`, slow tests named from the
committed timing run in PR #10) and the heavy/GPU lanes on the xbabe2 self-hosted runner
(`main.yml`, `nightly.yml`: full nextest, `neural` with the LibTorch receipt of PR
[#14](https://github.com/veox-ai/JopeDime/pull/14), encoder parity, e2e smoke, kernel-v1 product
build) land with WP8 — see `docs/ci.md` when it arrives. Releases:
[`.github/workflows/release.yml`](.github/workflows/release.yml) on `v*` tags builds `jope` in
`rust:1.97.1-bookworm` (glibc 2.36), receipts it, builds the wheel, produces SBOMs and a
provenance attestation, and publishes a GitHub release with git-cliff notes. Locally: `just ci-pr`
mirrors the hosted lanes, `just ci-main` adds the full tier.

## Contributing

Every change is a pull request against `main`; the owner merges; commit subjects are
`<area>: what changed`. Read [CONTRIBUTING.md](CONTRIBUTING.md) (landing policy, what reviewers
check, fixture and dependency rules), [SECURITY.md](SECURITY.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Decisions are recorded as ADRs in
[docs/adr/](docs/adr/): [0001 history migration](docs/adr/ADR-0001-history-migration.md),
[0002 feat-math reconciliation](docs/adr/ADR-0002-feat-math-reconciliation.md),
[0003 runtime roots](docs/adr/ADR-0003-runtime-roots.md),
[0004 evaluator identity](docs/adr/ADR-0004-evaluator-identity-known-table.md),
[0005 canonical learning loop](docs/adr/ADR-0005-canonical-learning-loop.md).

## Status of the port (work packages of the plan)

| WP | Scope | State |
|---|---|---|
| WP0 | Safety and second copy: credentials out of the tree, all-refs bundles + first-day weights to S3, buckets, OIDC roles, `LICENSE`/`NOTICE` | done (rotation of the old `gh.env` PAT and `aws.env` key remains an owner action — [checklist](docs/public-readiness-checklist.md)) |
| WP1 | filter-repo history, one Cargo workspace, vendored smartcluster, MSRV 1.96, first push | done (`16b388e`…); committed timing run + `pr` profile in PR [#10](https://github.com/veox-ai/JopeDime/pull/10) |
| WP2 | feat-math reconciliation ([ADR-0002](docs/adr/ADR-0002-feat-math-reconciliation.md)), strict four-language parity + C++20, `cxx.rs`, `deploy/ci/emitters.sh` | done (`106cb36`…`4235db6`); [ADR-0004](docs/adr/ADR-0004-evaluator-identity-known-table.md) in `5473e69` |
| WP3 | `numpy-pcg64`, `jope-encoders`, `hyperion`, `jope-policy` | first two done (`c67dad6`); hyperion + policy in PR [#6](https://github.com/veox-ai/JopeDime/pull/6) |
| WP4 | Path relocation ([ADR-0003](docs/adr/ADR-0003-runtime-roots.md)), systemd templates, `lint-paths` | PR [#13](https://github.com/veox-ai/JopeDime/pull/13) |
| WP5 | `jope-selfplay` / `jope-campaigns` split, argv[0] dispatch, `jope engine` | in flight (`wp5/cli-split`) |
| WP6 | Python package (uv, import-scan dependencies, conftest, xbabe2 triage) | PR [#8](https://github.com/veox-ai/JopeDime/pull/8) |
| WP7 | Model bundle manifests, `jope models`, datasets crate | manifests PR [#9](https://github.com/veox-ai/JopeDime/pull/9); `jope-models` in flight (`wp7/models-cli`); datasets PR [#11](https://github.com/veox-ai/JopeDime/pull/11) |
| WP8 | Complete CI: hosted lanes, main-guard, LibTorch receipt, runner, main/nightly | PR [#7](https://github.com/veox-ai/JopeDime/pull/7) (lanes + main-guard), PR [#14](https://github.com/veox-ai/JopeDime/pull/14) (neural receipt); runner lanes in flight (`wp8/runner-lanes`) |
| WP9 | This: docs, ADRs, release engineering, public-readiness checklist | this pull request |
| WP10 | Capability-parity matrix, test-inventory guard, `jope-contracts` + `schemas/` | census half in PR [#12](https://github.com/veox-ai/JopeDime/pull/12); contracts in flight |
| WP11 | Learning store and loop (`jope learn` / `hwm` / `scores`) | in flight (`wp11/learning-store`) |
| WP12 | `jope-arena`, weekly regression suite, restore drill | in flight (`wp12/arena`) |
| WP-D0…D8 | Buckets/IAM, ingest, second copies, T1/T3 uploads, fetch, `datasets/test`, licence backfill | D0 done; D1/D5 in PR #11 (ingest dry-run reproduces the plan's duplicate counts); the rest follow |
