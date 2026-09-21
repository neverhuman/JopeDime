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

| Surface | Status on `main` |
| --- | --- |
| `jope` CLI | Prints `jope 0.1.0`. Encode / search / emit / S3 commands are **not** wired yet. |
| `crates/feat-math` | Typed GP IR, fit, and portable emitters (Rust / Python / JS / C++17; C++20 compiles C++17). |
| `crates/jope-encoders` | Encoder implementations (package name `starforge`). |
| `crates/hyperion` | Host/runtime sources present; **not** a default workspace member. |
| `crates/jope-engine`, `crates/jope-campaigns` | Engine and campaign crates in the workspace. |
| Git datasets | One tiny table: `datasets/test/pmlb-229-pwlinear/`. Ten synthetic fixtures land next. |
| S3 fetch / champion training | Not on this SHA. |

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
bash ops/ci/required.sh
```

`ops/ci/required.sh` is path audit, `rustfmt`, locked workspace tests, and the
`jope 0.` version line. GitHub Actions YAML is `ops/ci/github-ci.yml` until a
token with `workflow` scope can copy it to `.github/workflows/ci.yml`.

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
| `crates/jope-cli` | `jope` binary (version only) |
| `crates/feat-math` | GP genomes, fit, emitters |
| `crates/jope-encoders` | Encoders (`starforge`) |
| `crates/hyperion` | Hyperion host sources (not default members) |
| `crates/jope-engine` | Storage / training / execution contracts |
| `crates/jope-campaigns` | Campaign and qualification tooling |
| `datasets/test` | Tiny CI table(s) |
| `python/` | Bounded compatibility / training surface |
| `ops/ci` | Required local/hosted checks |

Apache-2.0. See [LICENSE](LICENSE).
