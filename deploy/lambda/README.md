# AWS Lambda workers and terminal monitor

This directory holds the frozen AWS Lambda work of 2026-09-28: Rust workers that ran model fits on
AWS Lambda, first one fit per call and then a whole dataset corpus, and a terminal monitor for
those runs. Beside it are the regression workers, frozen on 2026-09-29: a Lambda worker that fits
candidate programs on three inner folds of a regression view, and an assessment tool that replays
a stored fold model. Each piece was imported from the ignored `.agent/experiments/` staging tree
after it was frozen.

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
| `regression/rust/` | Package `jope-regression-lambda`, the Lambda binary `regression-bootstrap`: it checks and downloads each job's input, fits every requested candidate program on the three inner folds in a child process with a timeout, replays each serialized model, and writes one gzipped record of the call to a presigned, create-only S3 URL. Its local modes (`--propose`, `--canonical-program`, `--canonical-programs`) served the controller. |
| `regression/schemas/` | The five record schemas the worker and its controller introduced, their registry (`REGISTRY.json`) and a compatibility fixture for the existing score record. The two existing schemas the registry lists as unchanged are the repository's own `schemas/jope.score.v1.schema.json` and `schemas/jope.hwm.v1.schema.json`, so they are not copied here. |
| `regression/*.md` | The worker's execution and recovery plan and its schema notes. |
| `regression-ops/rust/` | Package `jope-regression-assessment`, the binary `regression-assess`: it restores one stored fold model, checks its hash and its predictions against the archived ones, and measures how much its validation error grows when one feature column is shuffled, on a sample of validation rows, with no new fit. |
| `regression-ops/*.md` | The run's status at handoff, the S3 audit and the schema addendum. |
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
`cargo metadata --locked`, `cargo test --locked` and, against the root `deny.toml`,
`cargo deny --manifest-path <manifest> --all-features --locked --config deny.toml check bans licenses sources`
for each workspace. That is cargo-deny 0.20.2's command line (the runner image's, which the lane
asserts): `--config` is a top-level option there, while 0.19 takes it only after `check`. Outside
CI, any other cargo-deny version is skipped with a note. By hand, after that inflate step:

```sh
JOPE_SOURCE_HEAD=$(git rev-parse HEAD) \
JOPE_GP_MANIFEST_PATH=$PWD/deploy/lambda/csv/gp/corpus/extra-train-jobs.json \
  cargo test --locked --manifest-path deploy/lambda/csv/rust/Cargo.toml
cargo test --locked --manifest-path deploy/lambda/regression/rust/Cargo.toml
cargo test --locked --manifest-path deploy/lambda/regression-ops/rust/Cargo.toml
cargo test --locked --manifest-path deploy/lambda/watch/Cargo.toml
```

`JOPE_SOURCE_HEAD` is stamped into the workers' result records. `JOPE_GP_MANIFEST_PATH` names the
job manifest that `bootstrap-gp-extra` embeds; the frozen review records `extra-train-jobs.json`
(sha256 `3a838ea7…`) as the manifest it was built with.

## What is not here

- **Python.** The deploy, invoke, budget, archive and close controllers, the corpus inventory,
  the analysis and plots, the monitor's feed adapter and launcher, the planning tools, and the
  regression run's audit, history index, input preparation, campaign, accounting, query-table,
  reconcile and closeout tools are Python. Under the owner's no-Python rule they are preserved
  byte for byte in custody, and
  [`docs/archive/lambda-python-manifest.json`](../../docs/archive/lambda-python-manifest.json)
  lists each file with its hash and restore route. A Rust port is a follow-up.
- **`.env` files.** They were never tracked. The `.env.example` templates hold identifiers only:
  an AWS profile name, the region, the account ID, function names and host paths. The regression
  worker's `.env.example` is byte-identical to its private `.env`, so it is not tracked either;
  `SNAPSHOT.json` records the existence and sha256 of both, and of the monitor feed's `watch.env`.
- **Evidence.** Receipts, logs, packages and results stay in the runs' custody on xbabe2 and in
  their S3 archives.

## Notes

- The shell scripts, plans and `watch/build` are a historical record. They name the xbabe2 paths,
  the AWS account and the function names of the 2026-09-28 runs. The hygiene lane only
  syntax-checks the scripts (`bash -n`); running one deploys or invokes AWS resources.
- A rebuild is not claimed to reproduce the deployed ZIP hashes: `feat-math` is a path dependency
  outside the nested workspace, so its absolute path is embedded. Identity is by source hash
  (`SNAPSHOT.json`).
- The sources are not reformatted; the root `cargo fmt --all` does not reach nested workspaces.
- Twelve files differ from their frozen bytes, each recorded in `SNAPSHOT.json` with
  `tree_change`:
  - a `license` line in all four manifests, plus `publish = false` in the monitor's and the
    assessment tool's;
  - the `feat-math` entry of the three locks that take it by path, which reads the root workspace
    version;
  - four regression notes whose absolute xbabe2 paths are written home-relative (`~/`);
  - the regression schema registry `regression/schemas/REGISTRY.json`, whose schema paths name
    the tracked copies under `deploy/lambda/regression/schemas/` instead of the xbabe2 staging
    directory.
- `just bump X.Y.Z` keeps the `feat-math` lock entries true. A workspace's lock pins each root
  crate it takes by path at the version it was locked with, so after the root version changes
  `cargo metadata --locked` fails for that workspace until its lock is updated. The bump runs
  `bash deploy/ci/lambda-workspaces.sh sync-locks`: `cargo update -p <crate> --offline` for each
  such lock, a refusal unless only those version lines changed, and the lock's new sha256 in
  `SNAPSHOT.json`. It then runs the same checks as the hygiene lane, and fails unless they pass.
- After any other deliberate change to a tracked file here, record its new bytes with
  `bash deploy/ci/lambda-workspaces.sh snapshot-refresh <path> '<why>'`. It writes only that
  file's `tree_sha256` and `tree_change`, never its frozen `sha256`, and refuses a file that has
  no frozen record of its own bytes. `bash deploy/ci/lambda-workspaces.sh verify` runs the
  `cargo metadata --locked --offline` and `SNAPSHOT.json` checks.
- The regression worker writes its per-call scratch under `/tmp/jope-<attempt id>`
  (`regression/rust/src/main.rs`): `/tmp` is the only writable path in the AWS Lambda runtime.
  It removes that directory only on success, after the archive upload. An error after the
  directory is created returns without removing it: an input download or identity failure, a
  candidate run error, an oversized archive, or a failed upload. Because the directory is created
  with `create_dir`, a warm retry with the same attempt id then fails at that step until the
  execution environment is recycled. The frozen source is kept as it ran; a fix belongs in a
  successor worker. That is the repository's one exception to its no-`/tmp` rule, and it holds only
  inside Lambda. The worker's tests, `deploy/ci/lambda.sh` and every other tool write nothing under
  `/tmp`. No guard needed an exception for it: the repo-layout temp-directory rule covers test code
  under `crates/`.
