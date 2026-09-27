# ADR-0008 — Rust-only stack: Python remains only for emitted code and its harness

**Status:** accepted 2026-09-26 (owner ruling); phase P1 implemented by the `python-p1` pull
request · **Supersedes:** owner decision 2 of the port plan ("a `uv`-managed Python package owns
encoder inference/training and the deep-learning models") and, for future work, the
Python-training parts of [ADR-0005](ADR-0005-canonical-learning-loop.md) · **Scope:** every
tracked `.py` file, `python/`, `deploy/ci`, `deploy/maintenance`, `deploy/systemd`,
`crates/jope-engine/policy`, `benchmarks/`, `docs/internal/*/tools`

## Context

The port plan split the stack in two: Rust owned search, evaluation, the emitters, the CLI and
every contract validator; a `uv`-managed Python package owned encoder inference/training and the
deep-learning models (owner decision 2). ADR-0005 built on that split: the canonical loop is
driven by `continuous_learning.py`, promotion is verified by `promotion_tournament.py`, and the
lineage-A `v3-jsonl` trainer is Python.

By 2026-09-26 the tree at `8a55c4fef` held 921 `.py` files plus 5 packaging files. A
classification of every one (consumers, callers, CI lanes, installed units, provenance
citations) found: 151 live (L), 70 tests of live code (T), 8 files that are GP-emitted code or
its harness (G), 649 with no live consumer (A) and 48 needing a decision (?). A read-only host
check the same day found the xbabe2 flywheel units inactive since 2026-09-16 (they run Rust
binaries from bundles, not repository Python), the system `jope-workspace-sync` timer dead, and
no live Python on xbabe1/xbabe3 beyond a failed legacy unit. So the workspace sync (U12), the
engine services (U13) and the continuous-learning controller (U14) had no installed consumer.
The Rust side already carries native trainers (`native_loop`, `training.rs`, `runtime.cpp`) and
the policy loader. On 2026-09-26 the owner ruled that the stack is Rust, Vite, TypeScript and
React, with live tools ported under differential tests and unused code archived.

## Decision

1. **The stack is Rust, Vite, TypeScript and React.** New Python is not accepted. The one
   exception is GP-emitted Python and the harness that runs and checks it: the emitters target
   Python among other languages, and that output must be executed to prove parity.
2. **Always keep** (not candidates for removal):
   - the eight G files: `crates/feat-math/src/advanced_v3/fitted_portable.py`,
     `crates/feat-math/tests/portable/full_ref.py` (byte-pinned),
     `crates/jope-bench/src/panel/inference_boundary.py`,
     `crates/jope-bench/src/panel/inference_cpp_boundary.py`,
     `deploy/ci/emitters/compare-bits.py`, `docs/internal/asm-manual/examples/validate-outputs.py`,
     `docs/internal/core-ip/examples/model.py` and `docs/internal/core-ip/tools/worked_examples.py`;
   - `python3` in the runner image, for them;
   - the grammar files `featmath.py`, `tokens.py` and `automaton.py`, which are sha-pinned in
     assets, until phase P3 decides whether they are G or are ported.
3. **Remove the rest in three phases**, each keeping every guard
   honest (coverage map, test inventory and its static-Python floor, capability matrix,
   `spec_drift` registry, ruff/pytest configuration, asset provenance) and never weakens one:
   - **P1 (this ADR's first PR).** Archive every A file that has no live consumer, plus U12–U14
     with their tests and the systemd templates that only ran them, and the Python of the frozen
     kernel-v1 lineage (`deploy/kernel-v1`, 103 files) as one unit. Port four live tools with
     differential proof against the Python on the same real inputs, keeping the captured Python
     outputs as committed fixtures: U8 `family_growth_policy.py` → `jope_engine::family_growth`; U2
     `cargo-artifact.py` → `jope-ci cargo-artifact`; U3 `coverage.py`,
     `rust-test-scope-check.py`, `parity-census.py` (check mode), `readme-benchmarks.py` →
     `jope-records` subcommands; U5 `jope_report.py` → `jope-ci build-report`.
   - **P2.** U1 CI receipts and orchestration (`receipt`, `aggregate`, `ci-plan`, `results`,
     `status`, `training-proof`, `full-main`; needs a bootstrap order: a prebuilt binary or a
     `cargo run` pre-step); U4 runner admission (`job-policy.py` is a root-installed hook with a
     provenance digest; the old hook stays installed until the new one is qualified); U6 private
     install and qualification; U7 `benchmarks/prepare.py` → `jope-datasets prepare` (under a
     new dataset release identity, never an edit of `regression-10-v1`); U10 the experimental
     proposal producer; U16 documentation tooling → an xtask; and the ? units (the `jope-py`
     console script and its modules; the benchmark adapters, kept as external-reference adapters
     or ported through the C API).
   - **P3 — training stack** (U9 policy model and training, U11 the lineage-A research
     trainers, U15 the encoder loaders and target maker; about 33k lines, torch). Decide per path
     whether the native trainer already supersedes it; if so route `jope train` to it and retire
     the Python, otherwise port to `tch`. This needs GPU numerical qualification. The release
     wheel and the `python` CI lane are removed at the end of P3.
4. **Archived means recoverable, not rewritten.** Removal is `git rm`; history keeps every blob.
   [`docs/archive/python-archive-manifest.json`](../archive/python-archive-manifest.json) lists
   each removed path with its blob, mode, category, unit, disposition (archived or ported, with
   the Rust replacement) and restore command (`git checkout 8a55c4fef -- <path>`). A custody
   tarball with `SHA256SUMS` and an extraction verification sits outside the tree. Provenance
   citations to archived files are rewritten to `8a55c4fef:<path>`, never deleted.

## Consequences

- For future work this supersedes owner decision 2 and ADR-0005's Python-training parts: the
  canonical loop, the F10 tournament and the `v3-jsonl` trainer are to be native. ADR-0005's
  record of what ran, and its data model, are unchanged. Until P3 lands, the Python trainers
  that remain are kept running as they are; they are not extended.
- Until P2 and P3 land, Python stays only on the live paths the classification names (CI
  receipts, runner admission, private install, dataset preparation, the experimental producer,
  documentation tooling, the training stack, and their tests). `CONTRIBUTING.md`,
  `docs/architecture.md`, `README.md` and `docs/asm.md` describe that state.
- `python/pyproject.toml`, `uv.lock`, the `python` CI lane and its training-proof cases stay
  until P3, covering only what remains.
- A file with any live consumer stays, even without a product consumer: a module a kept test
  imports or launches by path stays until that test goes, and a file a record digest-checks stays
  while the record does. P1 kept 24 A-classified files for these reasons (21 used by kept tests,
  3 `retained` inputs of `docs/internal/core-ip/elo-history/histories.json`) and 3 U13 files that
  `crates/jope-engine/tests/sealed_storage.rs` runs; the P1 evidence record lists them.

## Verification

- P1: every removed path appears in the archive manifest with a blob equal to the base tree;
  the custody tarball extracts to exactly those paths with matching blobs and modes; coverage
  validation reports zero unassigned items; `verify-test-inventory`, the static-Python floor
  (lowered to the new exact count, citing the manifest), the capability-matrix check, the
  `spec_drift` tests, ruff, pytest and the training-proof's required cases all pass. The ports'
  differential runs and post-change gates are recorded in
  `docs/internal/core-ip/work-orders/evidence/PY-P1-python-removal.md`.
- P2 and P3 each add their own differential evidence and remove their rows from the kept set; at
  the end of P3, `git ls-files '*.py'` lists only the always-keep set and GP-emitted fixtures.
