# Contributing to JopeDime

Thank you for working on this. The rules below exist so that a change to a numeric kernel, an
emitter or a learning-loop pointer can be reviewed and reproduced by someone who was not there.

## Before you start

- Read [README.md](README.md) (what the pipeline is, the quickstart) and
  [docs/architecture.md](docs/architecture.md) (crate graph, contracts).
- Decisions with history are in [docs/adr/](docs/adr/). If your change contradicts one, write a new
  ADR (next number, same four sections) instead of editing the old one.
- Toolchain: `rust-toolchain.toml` pins Rust 1.97.1 with rustfmt and clippy; the workspace MSRV is
  1.96 (decision 8 of the plan) and CI checks it. `g++` is a hard build requirement (feat-math's
  `build.rs` compiles the fitted C++ runtime); `clang++`, `node` and `python3` are needed for the
  emitter matrix; `uv` for the Python package; `just` runs the recipes.

## The landing policy

Every change lands as a pull request against `main`; there are no direct pushes, the owner merges.
`main-guard.yml` (landing with PR #7) fails and files an issue when a commit reaches `main`
without a merged pull request, until branch protection is enforced on the Team plan. Squash or
rebase merges only (linear history); the branch is deleted on merge.

1. Branch from `main`: `<area>/<topic>` (`feat-math/…`, `engine/…`, `docs/…`, `wp<N>/…` for plan
   work packages).
2. Commit subjects are `<area>: what changed` — that prefix is what
   [cliff.toml](cliff.toml) groups the release notes by. Explain the *why* in the body; cite the
   plan section, ADR or blocker number when there is one.
3. Run `just ci-pr` before opening the PR (the exact scripts the hosted lanes call: hygiene, fmt,
   clippy `-D warnings`, `cargo nextest -P pr`, Python checks). A PR that touches only docs runs
   `hygiene` alone in CI; still run `just docs` locally.
4. Open the PR with a description that a reviewer can verify: what was run, on which host, with
   which output. Tests you had to change are listed explicitly (the test-inventory guard landing
   with WP10 fails a PR that silently removes tests).
5. Review by someone other than the author; every conversation resolved; the required checks
   (`hygiene`, `rust-check`, `emitters`, `python` once each lane exists) green.

## What reviewers look for

- **Bit-exactness is a feature.** Anything touching `crates/feat-math` (interpreter, heads,
  emitters, fitted runtime, `cxx.rs`) must keep `cargo test -p feat-math` green with
  `JOPE_PARITY_STRICT=1` and `bash deploy/ci/emitters.sh` byte-identical. If an expectation must
  change, regenerate it with `UPDATE=1`, commit the new `.sha256` files in the same PR, and say
  why the old bits were wrong. See [docs/emitters.md](docs/emitters.md).
- **No absolute home paths.** `just lint-paths` must be empty; runtime roots come from the
  resolvers in ADR-0003 ([docs/adr/ADR-0003-runtime-roots.md](docs/adr/ADR-0003-runtime-roots.md)).
- **Contracts once.** JSON contracts have one schema (`schemas/`, landing with WP10) and one Rust
  type; do not add a second serialisation of an existing record.
- **No weights, no datasets, no evidence in git.** Fixture policy: ≤ 2 MB per file, ≤ 10 test
  datasets, ≤ 12 MB total, plain files (no LFS). Models are manifests only (`models/`), datasets are
  the ten committed test sets plus a catalog ([docs/datasets.md](docs/datasets.md)).
- **No secrets.** `gitleaks` runs in hygiene; `*.env` is ignored; if you pasted a token into a
  commit, stop and follow [docs/runbooks/rotate-credentials.md](docs/runbooks/rotate-credentials.md)
  — the token is burned even if the commit never leaves your machine.
- **Dependencies.** `cargo deny check bans licenses sources` must pass (allow-list in
  [deny.toml](deny.toml); no git dependencies). Python dependencies are derived from an import
  scan (see `python/pyproject.toml`); `research` extras never run in CI.
- **Rust is authoritative.** Python owns encoder inference/training and the deep-learning models;
  search, evaluation, emitters, the CLI and every contract validator are Rust. A Python
  reimplementation of a Rust path is a parity oracle, not a second source of truth.

## Adding things

- A dataset: [docs/runbooks/add-a-dataset.md](docs/runbooks/add-a-dataset.md).
- An emitter target or a new op in an existing target:
  [docs/runbooks/add-an-emitter-target.md](docs/runbooks/add-an-emitter-target.md).
- A model bundle or a learning-loop change: the models and learning documents landing with
  WP7/WP11 (`docs/models.md`, `docs/learning.md`, `docs/runbooks/ingest-winners.md`).
- A CI lane: keep pull-request lanes hosted, credential-free and under 35 billed minutes; heavy
  or GPU work belongs to main/nightly on the self-hosted runner (`docs/ci.md`, landing with WP8).

## Releases

`just bump X.Y.Z` sets the version everywhere `deploy/ci/version-consistency.sh` checks, then a
PR, then the owner pushes the tag `vX.Y.Z`, which runs
[release.yml](.github/workflows/release.yml): glibc-2.36 build of `jope`, build receipt, wheel,
SBOMs, provenance attestation, git-cliff notes, GitHub release.

## Licence

By contributing you agree that your contribution is licensed under Apache-2.0
([LICENSE](LICENSE)); third-party code you bring in must be compatible with the allow-list in
`deny.toml` and be attributed in [NOTICE](NOTICE).
