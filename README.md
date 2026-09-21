# JopeDime

Standalone JOPE-Dime: a typed genetic-programming engine for tabular models,
encoders (Hyperion and companions), and emitters that lower a genome to Python,
Rust, C++17, and JavaScript. C++20 CI compiles the C++17 artifact; it is not a
second numeric dialect.

This is the product tree for [`veox-ai/JopeDime`](https://github.com/veox-ai/JopeDime).
It is **not** the 8×A100 node workspace and **not** a `jain-split` sibling.
Training corpora and model weights live in S3. Git holds source, catalogs, and
at most ten tiny test datasets.

SOTArena **ASPFM** / `jope-dime` campaign wins are the GP/feat-math track. The
encodings→GP neural policy will ship because the product needs a router; it is
**not** a held-out SOTA claim.

## What this SHA actually does

| Surface | Status on this SHA |
| --- | --- |
| `jope` CLI | No subcommand still prints `jope 0.1.0`. Fixtures, CPU `encode` (fail-closed without weights), `fit` / `search` / `emit`, `champions --local`, `train-step --local` (fits a 128-d centroid router), `route`. `champions --pull/--push` and `train-step --env dev` **fail closed**. Live S3 is **not** wired. |
| `crates/feat-math` | Typed GP IR, fit, and portable emitters (Rust / Python / JS / C++17; C++20 compiles C++17). |
| `crates/jope-encoders` | Encoder implementations (package name `starforge`). |
| `crates/hyperion` | Workspace member. CPU/Candle default (`hyperion-cpu`). CUDA feature exists; not the CI default. |
| `crates/jope-policy` | Workspace member. Step-60000 proposal policy on Candle. Needs the bound checkpoint to propose. |
| `crates/jope-engine`, `crates/jope-campaigns` | Engine and campaign crates in the workspace. |
| Git datasets | Ten synthetic fixtures in `datasets/synthetic/` plus `datasets/test/pmlb-229-pwlinear/`. Real tables stay in S3. |
| S3 fetch / champion training | Live S3 fail-closed. Local champion bank + 128-d centroid router train-step exist. Bound `jope-policy` checkpoint weights are **not** updated. |

The engineering spec remains [`CURSOR_GITHUB_PORT_PLAN.md`](CURSOR_GITHUB_PORT_PLAN.md).
Serial recovery of withdrawn PRs #5–#16 (unmerged, branches retained) is
[`MASTER_PLAN.md`](MASTER_PLAN.md).

## Install and check

Linux x86_64. Rust from `rust-toolchain.toml`. CPU-only default build (no
LibTorch, CUDA, or cluster runtime).

```sh
cargo build --locked
cargo run --locked --bin jope
# jope 0.1.0
cargo run --locked --bin jope -- verify-inputs --offline --root datasets/synthetic
# Verified 10 synthetic datasets, … bytes, offline.
# Encode needs the Hyperion bundle (not in git):
# HYPERION_V7_WEIGHTS=… cargo run --locked --bin jope -- encode --input datasets/synthetic/binary/linear.json
cargo run --locked --bin jope -- fit --input datasets/synthetic/regression/linear.json --json
cargo run --locked --bin jope -- emit --lang all --output /tmp/jope-emitted --json
cargo run --locked --bin jope -- search --input datasets/synthetic/regression/linear.json --json
cargo run --locked --bin jope -- champions --local --root datasets/synthetic --json
cargo run --locked --bin jope -- train-step --local --root datasets/synthetic --output /tmp/jope-router --json
cargo run --locked --bin jope -- route --input datasets/synthetic/regression/linear.json --router /tmp/jope-router --json
```

GitHub PR CI is `.github/workflows/ci.yml` (hygiene + cargo-deny).
`ops/ci/required.sh` is the fuller local gate (path audit, `rustfmt`, locked
workspace tests, `jope 0.` version line). Replacing the hosted workflow with
`ops/ci/github-ci.yml` needs a token with `workflow` scope.

Prove emitters on this SHA:

```sh
JOPE_PARITY_STRICT=1 cargo test --locked -p feat-math
```

`just required` / `just ci-pr` also run clippy, nextest, and Python ruff when
those tools are installed.

## Intended pipeline (later PRs, one at a time)

```text
dataset cut
  → encoders (Hyperion + foundation + target)
  → encodings context
  → policy that proposes / routes GP genomes
  → typed GP kernel (feat-math)
  → emitters → working Python / Rust / C++17 / JS
  → S3: best GP trees per dataset
  → train the policy on those champions
```

Do not open a stack of PRs. Merge or close the current one before starting the next.

## Layout (this SHA)

| Path | Role |
| --- | --- |
| `crates/jope-cli` | `jope` binary: fixtures, encode, fit/search/emit, local champions, centroid router |
| `crates/feat-math` | GP genomes, fit, emitters |
| `crates/jope-encoders` | Encoders (`starforge`) |
| `crates/hyperion` | Hyperion v5/v7 CPU/Candle encoder |
| `crates/jope-policy` | Bound step-60000 GP router (needs checkpoint) |
| `crates/jope-engine` | Storage / training / execution contracts |
| `crates/jope-campaigns` | Campaign and qualification tooling |
| `datasets/synthetic` | Ten integer-only Git fixtures + `catalog.json` |
| `datasets/test` | Tiny CI table(s) |
| `python/` | Bounded compatibility / training surface |
| `ops/ci` | Required local/hosted checks |

## Further reading

- [`docs/quickstart.md`](docs/quickstart.md) — the measured feat-math quickstart (timings from `4235db6`, before PRs #17–#26; not this SHA)
- [`docs/architecture.md`](docs/architecture.md) — crates, contracts and the intended pipeline
- [`docs/emitters.md`](docs/emitters.md) — per-target coverage of the Python / Rust / C++17 / JavaScript emitters
- [`docs/datasets.md`](docs/datasets.md) — dataset store layout, licence rule and `datasets/test`
- [`docs/campaigns.md`](docs/campaigns.md) — benchmark campaigns
- [`docs/adr/`](docs/adr/) — architecture decision records 0001–0005
- [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
