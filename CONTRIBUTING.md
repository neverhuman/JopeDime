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
  emitter matrix (`python3` runs the emitted Python); `uv` for the remaining Python package,
  which [ADR-0006](docs/adr/ADR-0006-rust-only-stack.md) is retiring; `just` runs the recipes.

## The landing policy

Every change lands as a pull request against `main`; there are no direct pushes. The same
agent may implement, review and merge after substantive review at the exact head and passing
CI. Follow [AGENTS.md](AGENTS.md#pull-requests) for the current procedure. `main-guard.yml`
checks that commits reaching `main` belong to merged pull requests; branch protection enforces
the required checks. Squash or rebase merges only (linear history); preserve source branches
and evidence until cleanup satisfies the custody rules in `AGENTS.md`.

1. Branch from `main`: `<area>/<topic>` (`feat-math/…`, `engine/…`, `docs/…`, `wp<N>/…` for plan
   work packages).
2. Commit subjects are `<area>: what changed` — that prefix is what
   [cliff.toml](cliff.toml) groups the release notes by. Explain the *why* in the body; cite the
   plan section, ADR or blocker number when there is one.
3. Run `just ci-pr` before opening the PR. It is a local subset of the pull-request lanes (the
   lanes themselves are `deploy/ci/fast-lane.sh`) plus clippy: `deploy/ci/check.sh`, the core of
   the `hygiene` lane, `cargo fmt --check`, clippy `-D warnings` (which no CI lane runs),
   `cargo nextest -P pr` over the product crates, and ruff and pytest over the Python that
   remains until ADR-0006's phase P3. Path-gated PR runs may skip declared lanes; the full
   exact-head manual CI requirement in `AGENTS.md` still applies. For documentation, include
   diff and link checks; `just docs` also runs rustdoc and the README/`docs/` link check
   (rustdoc warnings remain advisory until promoted to blocking).
4. Open the PR with a description that a reviewer can verify: what was run, on which host, with
   which output. Tests you had to change are listed explicitly (the test-inventory guard landing
   with WP10 fails a PR that silently removes tests).
5. Record substantive review against the exact full head SHA; the implementing agent may
   perform it. Outside review and native GitHub approval are optional. Resolve conversations,
   pass the current full manual CI procedure in `AGENTS.md`, and retain green protected checks
   (`aggregate`, `changes`, `hygiene`, `observatory-ui`) on an up-to-date branch. Any changed head
   needs fresh checks and review; use a head-bound `COMMENT` or durable receipt for same-agent
   review, without manufacturing a native self-approval.

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
  the ten committed test sets with their collection manifest and the forty generated synthetic
  fixtures with their catalog ([docs/datasets.md](docs/datasets.md)).
- **No secrets.** `gitleaks` runs in hygiene; `*.env` is ignored; if you pasted a token into a
  commit, stop and follow [docs/runbooks/rotate-credentials.md](docs/runbooks/rotate-credentials.md)
  — the token is burned even if the commit never leaves your machine.
- **Dependencies.** `cargo deny check bans licenses sources` must pass (allow-list in
  [deny.toml](deny.toml); no git dependencies). The remaining Python dependencies are derived from
  an import scan (see `python/pyproject.toml`) and may shrink but not grow; `research` extras
  never run in CI.
- **Rust only ([ADR-0006](docs/adr/ADR-0006-rust-only-stack.md), owner ruling 2026-09-26).** The
  stack is Rust, Vite, TypeScript and React. New Python is not accepted, with one exception:
  the Python the emitters generate and the harness that runs and checks it (the emitter matrix,
  `compare-bits.py`, the portable references). The Python still in the tree (the training stack,
  CI receipts, runner admission, private install, dataset preparation, documentation tooling and
  the files kept tests or records depend on) stays only until phase P2 or P3 ports or retires it:
  fix it when it breaks, do not extend it. Removed Python is listed with its restore command in
  [docs/archive/python-archive-manifest.json](docs/archive/python-archive-manifest.json). A Python
  reimplementation of a Rust path is never a second source of truth.

## Adding things

- A dataset: [docs/runbooks/add-a-dataset.md](docs/runbooks/add-a-dataset.md).
- An emitter target or a new op in an existing target:
  [docs/runbooks/add-an-emitter-target.md](docs/runbooks/add-an-emitter-target.md).
- A model bundle or a learning-loop change: the models and learning documents landing with
  WP7/WP11 (`docs/models.md`, `docs/learning.md`, `docs/runbooks/ingest-winners.md`).
- A CI lane: pull-request lanes run on the self-hosted `jope-ci` runners, one isolated container
  per job with no GPU devices ([docs/runbooks/runner.md](docs/runbooks/runner.md)), and use no
  secrets and no cloud credentials. The heavier test families run in `full-main-cpu` on every
  push to `main`; GPU work runs in no pull-request lane.

## Tests

A test that needs files on disk takes a scratch directory from `tempfile` (removed when the handle
drops, also when the test panics), not a fixed name under `std::env::temp_dir()`, which is left
behind on every run. Local runs set `TMPDIR` to a directory outside the checkout and outside
`/tmp`, for example `~/.cache/jope-test-tmp/<lane>`. Inside the checkout, scratch directories sit
under the tracked `.jope-home` marker and break `jope-paths`' home-resolution test; `/tmp` on the
build hosts is shared by every tenant and emptied at boot.

## Releases

Releases come from [`deploy/release/release.sh`](deploy/release/release.sh), run on xbabe2 after
the release PR (version bump and git-cliff `CHANGELOG.md`) has merged: first a dry run, then the
same command with `--publish`, which creates the annotated tag `vX.Y.Z` and the GitHub release. It
builds `jope` in the digest-pinned `rust:1.97.1-bookworm` image (glibc 2.36) and writes a build
receipt, SBOMs, the release notes, a release manifest and `SHA256SUMS`. The trust statement is a
cosign signature over `SHA256SUMS`, made with the root-held release key; there is no provenance
attestation and no Python wheel. [docs/release/README.md](docs/release/README.md) explains how to
verify a release and how the key is held. [release.yml](.github/workflows/release.yml) is
dispatch-only and is not the release path.

## Licence

By contributing you agree that your contribution is licensed under Apache-2.0
([LICENSE](LICENSE)); third-party code you bring in must be compatible with the allow-list in
`deny.toml` and be attributed in [NOTICE](NOTICE).
