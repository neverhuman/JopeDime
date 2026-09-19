# ADR-0002 — One reconciled feat-math: registry authority, portable build flags, strict parity

**Status:** accepted 2026-09-18 · **Plan section:** "feat-math reconciliation (ADR-0002)",
"Code emission", blockers B2, B3, B9, B28, B29 · **Scope:** `crates/feat-math`, `deploy/ci/emitters.sh`

## Context

Three copies of feat-math existed: `vendor/feat-math` (the search engine's, whose bytes produced
`parity.json`'s `vocab_sha256 8d55fee4…`, the golden tests, the kernel codec and all 11,903
winners), the self-copy under `parallel/assets/source/jope-dime/vendor/feat-math` (commit
`2c0c2b25`, newer: its `gpr.rs` implements `gpr-scratch-reuse` with
`fit_impl<const REUSE_SCRATCH: bool>` and `chol_inverse_into` where `vendor/`'s declares the
feature as a no-op; the other nine differing files were rustfmt-only), and the `jain-math` fork
(modularised emitters, a 1,379-line `registry.rs` that is the product's op executability authority
— `OperatorLifecycle`: parseable/executable/generatable/quarantined — plus `plan/prime/sandbox`
dialects and no `advanced_v3/`, `kernel/`, `portable_math.rs`, `emit_cpp.rs`). The engine's
`eval_registry/query.rs` `include_bytes!`-hashed the self-copy's `model.rs` (B3); the campaigns
crate hashed ~30 `vendor/feat-math/src/*.rs` into its evaluator identity (B2, solved by
[ADR-0004](ADR-0004-evaluator-identity-known-table.md) first). `build.rs` hard-coded
`-march=znver2`; eight `Command::new("g++")` sites carried their own flags (B28); the four-language
parity test silently skipped a backend whose tool was missing.

## Decision

1. **Base = the jope-dime lineage** (`vendor/feat-math`), the bytes every winner and receipt was
   produced with. First change: the self-copy's `gpr.rs` (feature still default-off) and
   `cargo fmt` over the crate (`106cb36`, `4235db6`). The self-copy is deleted; `eval_registry`
   includes point at the single crate.
2. **`src/registry.rs` verbatim from jain-math**, `pub mod registry;` — and nothing else from that
   fork (no `plan/prime/sandbox/engine/diagnostics`, no refactored emitter trees, no
   `tests/customer_manifest.rs`). `parallel/tools/jain-math-registry-dump` becomes
   `crates/feat-math/examples/registry_dump.rs` (reads `CARGO_MANIFEST_DIR`, no git pin);
   `tests/registry_vocabulary.rs` asserts every op in
   `crates/jope-engine/assets/grammar/featmath-jain-v1.json` is `executable && !quarantined` and
   that the snapshot's executable set and `registry_hash` equal the registry (`cdb1a01`).
3. **One compiler helper, `src/cxx.rs`** (`4235db6`): `cxx::command(Option<CxxStd>)`,
   `compiler()`, `default_standard()`, `march_flag()`, `fitted_flags()`, `version()` — honoured at
   every call site (`fitted_cpp_runtime.rs`, the six emitted-C++ gates in `transpile.rs`,
   `tests/advanced_v3_fitted_cpp.rs`, `tests/portable_parity.rs`), reading `CXX`, `JOPE_CPP_STD`
   (`17`/`20`; anything else fails closed) and `JOPE_CXX_MARCH`. Emitted programs get no `-march`
   unless it is set. `build.rs` mirrors the same variables with `rerun-if-env-changed`.
4. **Strict five-backend parity** (`53ba1f8`): `tests/portable_parity.rs` compiles and runs
   `tests/portable/full_ref.{rs,cpp,py,js}` and adds a **C++20** backend on the same C++ source;
   under `JOPE_PARITY_STRICT=1` (set by CI and `just test-pr`) a `NotFound` tool is a failure.
   C++20 is a compile target, not a second emitter: the emitted `model.cpp` is one source compiled
   under both standards (a `--cxx-standard` emit option may later change only the header comment
   and a `static_assert(__cplusplus >= …)`).
5. **Fitted-runtime default ISA `x86-64-v2`** (`f469fac`, `cxx::FITTED_DEFAULT_MARCH`), was
   `znver2`. `-O3 -ffp-contract=off -fno-fast-math` stay, so the produced bits do not depend on
   the ISA. **Codec-identity consequence:** the fitted codec identity hashes the joined flag string
   (`runtime_flags_sha256`), so a codec encoded under one environment decodes only under the same
   one; historical node codecs were produced with
   `-std=c++17 -O3 -march=znver2 -ffp-contract=off -fno-fast-math`, and `JOPE_CXX_MARCH=znver2`
   reproduces that identity. No committed receipt pins the fitted `.so` hash, so nothing persisted
   breaks; the nightly `-march` matrix proves identical predictions.
6. **The emitted-code matrix** `deploy/ci/emitters.sh` with pins in `tests/fixtures/emitted/`
   ([docs/emitters.md](../emitters.md)): champion `8944cbdf` (histgbm head → Python/Rust pair),
   `portable-4lang` (ridge → seven programs across python, rustc, g++/clang++ × c++17/c++20, node),
   sources byte-identical and predictions bit-identical on `datasets/test/pmlb-229-pwlinear`.
7. **The `golden.rs` re-pin.** `tests/legacy_v2_freeze.rs` freezes the sha256 of `tests/golden.rs`
   against the G-00 takeover snapshot. The rustfmt pass in `4235db6` changed that file's bytes
   (whitespace only; `git diff -w` is empty), so on `main` today
   `legacy_v2_reference_files_match_the_takeover_snapshot` fails
   (`91f759…` vs the frozen `cc9600…`). The re-pin — new value in the assertion, old value kept in
   the comment — is in PR [#10](https://github.com/veox-ai/JopeDime/pull/10) together with the
   committed timing run; it is a freeze update, not a semantic change, and the golden genomes
   themselves are unchanged (all 83 `golden.rs` tests pass).

## Consequences

- One crate is the authority for the DSL, the interpreter, the emitters and the op lifecycle; the
  searchable vocabulary (`featmath.json`, 401 tokens) and the product vocabulary
  (`featmath-jain-v1.json`) are checked against the registry in CI instead of by convention.
- `gpr-scratch-reuse` is now real code behind a default-off feature; enabling it requires the
  documented exact-parity and performance qualification (Risks table: bit-exactness drift).
- Hosted runners and the RTX box produce the same bits because the ISA is portable by default and
  FMA contraction is off everywhere; a developer who needs the node's codec identity sets one
  variable.
- Silent skips are gone from the parity proofs; a hosted runner missing `clang++` or `node` fails
  the `emitters` lane instead of passing vacuously.
- Every reformat of feat-math changes the campaigns crate's *current* evaluator identity; persisted
  receipts keep validating under their recorded identity through ADR-0004's known table.

## Verification

- `cargo test -p feat-math --locked`: 407 of 408 tests pass on `main` (13 m 01 s on the 128-core
  host, 2026-09-18); the one failure is item 7 and is closed by PR #10.
- `cargo run -p feat-math --example registry_dump | cmp - crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json`
  is byte-identical; `cargo run -p jope-parallel --example derive_jain_vocabulary -- --check` holds
  with the crate-local grammar and the preserved literal key (`vocab_sha256 8d55fee4…`).
- `bash deploy/ci/emitters.sh` green (20 s), also under `JOPE_CXX_MARCH=znver2`, `JOPE_CPP_STD=20`
  and `CXX=clang++`; `JOPE_CPP_STD=23` fails closed (verified on xbabe2 in `4235db6`).
- `JOPE_PARITY_STRICT=1 cargo test -p feat-math --test portable_parity` proves Rust ≡ C++17 ≡
  C++20 ≡ Python ≡ JavaScript.
