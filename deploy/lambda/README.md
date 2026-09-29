# AWS Lambda workers and terminal monitor

This directory holds the frozen AWS Lambda work of 2026-09-28: Rust workers that ran model fits on
AWS Lambda, first one fit per call and then a whole dataset corpus, and a terminal monitor for
those runs. It was imported unchanged from the ignored `.agent/experiments/` staging tree after
each piece was frozen.

None of it is part of the `jope` binary, the root Cargo workspace or the release build. Each Rust
body is its own Cargo workspace with its own frozen `Cargo.lock`, and the root `Cargo.toml`,
`Cargo.lock` and `crates/` do not refer to it.

## Contents

| Path | What it is |
| --- | --- |
| `csv/rust/` | Package `jope-lambda-probe`, the Lambda `bootstrap` binaries: `bootstrap` (one fit on one CSV), `bootstrap-batch` (several fits per call, for the cost study) and the three corpus workers `bootstrap-gp`, `bootstrap-gp-extra` and `bootstrap-gp-diagnose`. The `*_v1*.rs` and `*_pre_archive.rs` files are earlier editions, kept for reference and not compiled. |
| `csv/cost/` | The cost study: its protocol, results, shell controller and the cross-compile wrappers. |
| `csv/gp/` | The corpus run's plan, review, storage and logging notes, and its two frozen job manifests: `corpus/train-jobs.json.gz` and `corpus/extra-train-jobs.json`. |
| `csv/versions/` | Earlier editions: the first Rust attempt (`rust-v1`) and the single-fit edition (`rust-single-v2`, whose manifest is tracked as `Cargo.toml.v2`). |
| `csv/*.md`, `csv/*.sh` | The feasibility results, the test and sizing plan, the deploy script, the env loader and its test. |
| `watch/` | Package `jope-lambda-watch`, a Ratatui terminal monitor for the runs, with its telemetry contract and performance record. |
| `SNAPSHOT.json` | Every file of the frozen work: its sha256, its source anchor and whether it is tracked, gzipped, renamed, in custody or excluded. |

The runs themselves, their costs and their archive hashes are summarised in
[`docs/internal/lambda/README.md`](../../docs/internal/lambda/README.md).

## Build and test

```sh
bash deploy/ci/lambda.sh
```

The hygiene lane runs the same script. It checks the tracked files against `SNAPSHOT.json`,
inflates `csv/gp/corpus/train-jobs.json` from its `.gz` (the raw file is over the repository's
5 MiB size cap; the inflated file is gitignored) and checks its sha256, then runs
`cargo metadata --locked`, `cargo test --locked` and `cargo deny check bans licenses sources` for
each workspace. By hand, after that inflate step:

```sh
JOPE_SOURCE_HEAD=$(git rev-parse HEAD) \
JOPE_GP_MANIFEST_PATH=$PWD/deploy/lambda/csv/gp/corpus/extra-train-jobs.json \
  cargo test --locked --manifest-path deploy/lambda/csv/rust/Cargo.toml
cargo test --locked --manifest-path deploy/lambda/watch/Cargo.toml
```

`JOPE_SOURCE_HEAD` is stamped into the workers' result records. `JOPE_GP_MANIFEST_PATH` names the
job manifest that `bootstrap-gp-extra` embeds; the frozen review records `extra-train-jobs.json`
(sha256 `3a838ea7…`) as the manifest it was built with.

## What is not here

- **Python.** The deploy, invoke, budget, archive and close controllers, the corpus inventory,
  the analysis and plots, the monitor's feed adapter and launcher, and the planning tools are
  Python. Under the owner's no-Python rule they are preserved byte for byte in custody, and
  [`docs/archive/lambda-python-manifest.json`](../../docs/archive/lambda-python-manifest.json)
  lists each file with its hash and restore route. A Rust port is a follow-up.
- **`.env` files.** They were never tracked. The `.env.example` templates hold identifiers only:
  an AWS profile name, the region, the account ID, function names and host paths.
- **Evidence.** Receipts, logs, packages and results stay in the runs' custody on xbabe2 and in
  their S3 archives.
- **The regression workers.** They were still running when this was imported; they follow once
  their owner freezes them.

## Notes

- The shell scripts, plans and `watch/build` are a historical record. They name the xbabe2 paths,
  the AWS account and the function names of the 2026-09-28 runs. The hygiene lane only
  syntax-checks the scripts (`bash -n`); running one deploys or invokes AWS resources.
- A rebuild is not claimed to reproduce the deployed ZIP hashes: `feat-math` is a path dependency
  outside the nested workspace, so its absolute path is embedded. Identity is by source hash
  (`SNAPSHOT.json`).
- The sources are not reformatted; the root `cargo fmt --all` does not reach nested workspaces.
- Three files differ from their frozen bytes, each recorded in `SNAPSHOT.json` with `tree_change`:
  a `license` line in both manifests, `publish = false` in the monitor's, and the `feat-math`
  entry of the workers' lock at this release's 0.2.0.
