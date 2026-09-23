**For the currently published benchmark numbers, read [Measured results](../README.md#measured-results)** —
that section is regenerated from committed evidence by `deploy/ci/readme-benchmarks.py` and names the
SHA it was generated at. The timings on *this* page are historical: measured 2026-09-18 at `4235db6`,
never re-measured at the head of `main`.

The `wp7/models-cli` and `wp11/learning-store` work this page described as unlanded has since merged
(#33, #34); run `jope --help` on current `main` for the verbs that exist today.

# Quickstart (laptop, CPU only)

For what the head of `main` actually does — the `jope` subcommands that exist today
(`verify-inputs`, `generate-fixtures`, `encode`, `fit`, `emit`, `search`, `champions`,
`train-step`, `route`) and the install/check commands for them — read the
[README](../README.md). This page keeps the feat-math quickstart from the WP9 docs pass; every
timing below was measured at `4235db6`, not at the head of `main`.

Requirements: Rust 1.97.1 via `rustup` (`rust-toolchain.toml` selects it; MSRV 1.96), `g++`
(feat-math's `build.rs` compiles the fitted C++ runtime), and for the emitter matrix `clang++`,
`node` and `python3`. No CUDA, LibTorch or cluster runtime is needed for anything below. Every
command was run at `4235db6` on 2026-09-18/19 (128-core host, `-j 16`; a laptop will be slower):

```bash
git clone https://github.com/veox-ai/JopeDime && cd JopeDime

# 1. the whole workspace, CPU features only                          (1 m 57 s at 4235db6)
cargo build --workspace --locked

# 2. the program engine and emitters: 408 tests incl. the four-language (13 m 01 s at 4235db6, includes
#    parity proofs (g++/clang++/node/python3 are exercised)           compiling the test binaries)
cargo test -p feat-math --locked

# 3. the emitted-code matrix: two programs on datasets/test/pmlb-229-pwlinear —
#    Python, Rust, g++/clang++ × C++17/C++20 and node produce bit-identical predictions
#    and byte-identical sources pinned in crates/feat-math/tests/fixtures/emitted/  (20 s at 4235db6)
bash deploy/ci/emitters.sh

# 4. emit one program yourself: model.py, model.rs, model.cpp, model.js + manifest.json (0.2 s at 4235db6)
cargo run -p feat-math --example emit -- crates/feat-math/tests/fixtures/emitted/portable-4lang.dsl /tmp/out
```

Known state of step 2 at `4235db6`:
`legacy_v2_freeze::legacy_v2_reference_files_match_the_takeover_snapshot` failed because
`tests/golden.rs` was rustfmt-ed in `4235db6` and its frozen sha256 was re-pinned in PR
[#10](https://github.com/veox-ai/JopeDime/pull/10) (whitespace-only change, `git diff -w` empty;
#10 was later withdrawn with #5–#16, see `MASTER_PLAN.md`); the other 407 tests passed. The full
PR tier is `just ci-pr` (hygiene, fmt, clippy, `cargo nextest -P pr`, Python checks); `just docs`
runs rustdoc and the markdown link check.

To emit a program of your own, write its DSL on one line — the champion in
`crates/feat-math/assets/champions/8944cbdf.dsl` is
`(pipe X (winsor_clip append=true q=0.1) (cheb append=true degree=6) (robust_scale) (robust_scale) (poly_interact append=true) (quantile_bins bins=16) (histgbm depth=5 lr=0.15 rounds=150))`
— and pass it to the `emit` example; `--all` makes an uncovered target an error instead of a note
(the `histgbm` head is Python/Rust only today; see [emitters.md](emitters.md)). Every emitted
program takes `train.csv test_features.csv predictions` (headerless CSV, last train column is the
target); the C++ and JavaScript artefacts are libraries, and `deploy/ci/emitters/driver.{cpp,js}`
are the CSV drivers the matrix appends to them.

## Models and datasets (as planned at `4235db6`)

The `jope models`, `jope datasets`, `jope learn`, `jope hwm` and `jope scores` verbs named here
are **not** on `main`; the bucket layout is the plan they implement.

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
([ADR-0003](adr/ADR-0003-runtime-roots.md), PR [#13](https://github.com/veox-ai/JopeDime/pull/13)).
