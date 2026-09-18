# JopeDime

Standalone JOPE-Dime: genetic programming for tabular models, with Hyperion and
companion encoders, a neural proposer, and emitters that lower a genome to
Python, Rust, C++17, and JavaScript. C++20 CI compiles the C++17 artifact; it
is not a second numeric dialect.

This repository is the product tree for [`veox-ai/JopeDime`](https://github.com/veox-ai/JopeDime).
It is **not** a mirror of the 8×A100 node workspace and it is **not** a
`jain-split` sibling. Training corpora and model weights live in S3. Git holds
source, catalogs, and at most ten tiny test datasets.

## Status

Bootstrap (P1). The `jope` binary currently prints its version. Encoders,
search, export, S3 fetch, and the ten fixtures land in later PRs. The full
engineering spec is [`CURSOR_GITHUB_PORT_PLAN.md`](CURSOR_GITHUB_PORT_PLAN.md).

SOTArena **ASPFM** / `jope-dime` campaign wins are the GP/feat-math track.
The encodings→GP neural policy ships because the product needs it; it is **not**
a held-out SOTA claim.

## Build

Rust 1.97.1 (workspace toolchain). CPU-only; no LibTorch, CUDA, or cluster
runtime in the default build.

```bash
cargo build --locked
cargo run --locked --bin jope
just required
```

The required lane is `ops/ci/required.sh` (path audit, `rustfmt`, locked tests,
`jope` version). GitHub Actions YAML lives at `ops/ci/github-ci.yml` until a
token with the `workflow` scope can copy it to `.github/workflows/ci.yml`.

## Layout (target)

| Path | Role |
|---|---|
| `crates/jope-cli` | `jope` binary (stub in P1) |
| `crates/feat-math` | GP genomes, fit, emitters (later) |
| `crates/jope-data` | Catalog + S3 + prepare (later) |
| `crates/jope-encode` | Hyperion + foundation + target (later) |
| `crates/jope-policy` | encodings → GP (later) |
| `crates/jope-search` | GP search (later) |
| `crates/jope-export` | genome → Python/Rust/C++17/JS (later) |
| `tests/fixtures/datasets` | ten CI datasets (later) |
| `catalogs/` | dataset and model hashes (later) |

## License

Apache-2.0. See [LICENSE](LICENSE).
