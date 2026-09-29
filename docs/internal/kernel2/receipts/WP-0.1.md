# WP-0.1: baseline record for kernel2 PR 01

Base: `629e69d865a674222e00224cebf45c1ffddc8dea`, `origin/main` when the lane `kernel2/m0-freeze`
was created (the v0.2.0 release commit). Measured on 2026-09-29 in that lane before any change,
on xbabe2: x86_64-unknown-linux-gnu, glibc 2.39, rustc 1.97.1 (8bab26f4f 2026-07-14),
cargo-nextest 0.9.133, g++ 13.3.0, clang++ 18.1.3, node v26.1.0. Environment:
`CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `CARGO_BUILD_JOBS=16`,
`nice -n 19 ionice -c3`; test runs add the rust-test lane's `JOPE_PARITY_STRICT=1` and one
OpenMP, rayon, OpenBLAS and MKL thread. This is local evidence. It is not hosted CI, and the
wall times are for this host only.

## Drift from the workplan's base

The workplan's anchors were taken at `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`. The base is 42
commits later, with 707 paths changed. Nothing changed under `crates/feat-math`,
`crates/jope-engine/src`, `crates/jope-engine/assets/grammar` or `crates/jope-policy`, so the
workplan's file and line anchors hold. `crates/jope-campaigns` changed in five files (12
insertions, 10 deletions). One of them, `Cargo.toml` (version 0.1.0 to 0.2.0), is an input of the
ADR-0004 evaluator identity, as is the root `Cargo.lock` (79 lines changed), so the evaluator
identity at the base differs from the one at `a8e4ce8`.

## Commands and results

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format (`just fmt`) | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| Lint (`just clippy`) | `cargo clippy --workspace --all-targets --locked -- -D warnings` | 101 | fails at the base; see below | 38.0 s |
| feat-math tests | `cargo nextest run -P pr --locked -p feat-math --test-threads 24` | 0 | 431 passed, 0 failed, 8 skipped | 73.0 s (59.3 s of tests) |
| Grammar parity | `cargo nextest run -P pr --locked -p jope-parallel --test grammar` | 0 | 3 passed | 41.7 s with the build |
| Registry snapshot (ADR-0002) | `cargo run -q --locked -p feat-math --example registry_dump \| cmp - crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json` | 0 | byte-identical | 0.2 s |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 4 workspaces built and tested, 61 tests passed | 50.0 s |

The workplan names `-p jope-engine` for the grammar test; the package in `crates/jope-engine` is
`jope-parallel`. The clippy and Lambda rows were measured at a lane commit whose sources equal the
base (the PR adds tests, fixtures and records only).

Workspace clippy is red at the base, and no CI lane runs it. Rerun with `--keep-going` (exit 101,
25.4 s), it reports `-D warnings` errors in five crates: jope-datasets (1, test
`staging_disposition`), battle-gpu (4 in the library, 4 in its tests), starforge (1 and 3),
jope-parallel (88 and 101) and jope-dime-regressor (30 and 38). Crates that depend on a failing
library are not reached. feat-math has no findings: `cargo clippy -p feat-math --tests --locked
-- -D warnings` passes, with the new test targets included.

The Lambda run: `csv/rust` 40 tests, `regression-ops/rust` 2, `regression/rust` 5 and `watch` 14.
Its cargo-deny step was skipped, because cargo-deny 0.19.8 is installed here and the script's
command line is 0.20.2's (the hygiene lane asserts 0.20.2); that step is not a pass.

### feat-math tests per binary (pr profile)

| Binary | Listed | Run | `#[ignore]` | Excluded by the pr profile filter |
|---|---|---|---|---|
| `feat-math` (lib) | 295 | 292 | 2 | 1 |
| `advanced_v3_conformance` | 7 | 7 | 0 | 0 |
| `advanced_v3_fitted_cpp` | 13 | 13 | 0 | 0 |
| `advanced_v3_multiclass` | 6 | 6 | 0 | 0 |
| `advanced_v3_parity` | 1 | 1 | 0 | 0 |
| `advanced_v3_property` | 2 | 2 | 0 | 0 |
| `bench` | 1 | 0 | 1 | 0 |
| `champion_parity` | 2 | 2 | 0 | 0 |
| `golden` | 83 | 80 | 0 | 3 |
| `ip_scrub` | 6 | 6 | 0 | 0 |
| `kernel_codec` | 4 | 4 | 0 | 0 |
| `kernel_emission_safety` | 1 | 1 | 0 | 0 |
| `kernel_expression` | 4 | 4 | 0 | 0 |
| `kernel_pipeline` | 3 | 3 | 0 | 0 |
| `legacy_v2_freeze` | 4 | 3 | 1 | 0 |
| `portable_parity` | 1 | 1 | 0 | 0 |
| `registry_vocabulary` | 2 | 2 | 0 | 0 |
| `tokens` | 4 | 4 | 0 | 0 |
| **Total** | **439** | **431** | **4** | **4** |

Ignored: `heads::tests::ridge_gcv_wide_perf`, `portfolio::tests::portfolio_timing_700x18`,
`bench::emitter_speed`, `legacy_v2_freeze::regenerate_legacy_v2_freeze`. Excluded by the pr
profile's default filter (`.config/nextest.toml`): `model::tests::portfolio_noise_target_no_leak`,
`golden::fast_profile_portfolio_head`, `golden::golden_portfolio_core`,
`golden::golden_ridge_gcv_screen_wide`.

### Known-broken tests

`deploy/ci/known-broken-tests.txt` lists one test, in a crate the PR lane defers:
`jope-parallel failed_exit_signal_and_timeout_cannot_be_cleared_by_result_files`, a race-sensitive
library test that passes in isolation.

## Digests at the base

| Item | sha256 or value |
|---|---|
| `feat_math::registry::registry_hash()` | `8173b07f37fde39bdc8bbf2dfe2e496aa5ccaeb4e0274843c235ec1fc6972c62` |
| `crates/jope-engine/assets/grammar/featmath.json` | `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787` |
| `crates/jope-engine/assets/grammar/featmath-jain-v1.json` | `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` |
| `crates/jope-engine/assets/grammar/parity.json` | `9dc5efc698204ab3354ccb34659fee8ccdbf1d34a7a62da27882f0d9930c1c1a` |
| `crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json` | `21c194e2d2c162e29d3893877bc2a2944ce33815db09f21626b5a99371d398aa` |
| `crates/jope-engine/src/grammar.rs` | `6a8e6aa43efb8af6aa4806fd5d403ca34c06563436d933266a7377d815625e6a` |
| `Cargo.lock` | `8ecdd1fc461f5700897e2e65dac13e3a05200bfcd2226e3b62d39bbe772c1fee` |
| ADR-0004 `implementation_sha256` | `9bee59f35055d574a73c6b32b6ee9945c01b531c968bb07075d1cc771f0294e9` |
| ADR-0004 evaluator identity digest | `430a0119915927af71d5ef535955cf03e1186199d0e94c7f76370dcc46c3075b` |
| `crates/jope-campaigns/known-evaluator-identities.json` | `ac0193aecf265f6d022b2b4b1c03a26af2459e7341760291e30d094c0b83bd6e` (no rows) |

## What PR 01 adds on top of the base

- `crates/feat-math/tests/fixtures/math_boundary/v1.json` (WP-0.3): 66 boundary inputs in nine
  groups, checked against their definition in `tests/compat_freeze.rs`.
- `crates/feat-math/tests/compat_freeze.rs` (WP-0.2): 9 tests over the fixtures in
  `tests/fixtures/compat_freeze/`, generated from the base's sources. The binary runs in about
  8.3 s on this host (the slowest test is the generator streams). Repeated regenerations produced
  byte-identical fixtures, and compare mode passes with one thread and with the default threads.
- `crates/feat-math/tests/mathexpr_current_semantics.rs` (WP-0.5): 18 tests.
- Guard-constant perturbation caught: with the division guard moved from `1e-12` to `1e-11`,
  `division_guard_is_strictly_below_1e_minus_12`, `expression_values_match_the_frozen_fixture`
  (96 differing values) and `evaluator_identity_inputs_match_the_recorded_identity` fail with
  readable messages; after `git checkout` of the source the tree is clean and all 27 tests pass.

## Findings recorded while freezing

1. **A stored `Model` does not reload bit-exactly in a feat-math-only build.** serde_json parses
   floats exactly only with its `float_roundtrip` feature. feat-math does not enable it, while
   jope-engine, jope-pipeline, jope-policy, jope-bench and others do, and Cargo unifies features
   in multi-package builds such as the rust-test lane's. In a `-p feat-math` build, reloading the
   stored `golden_map_derive_ridge` model moved its predictions by up to 28 ulps. The freeze
   therefore pins serialization bytes exactly and reloaded predictions to a relative 1e-9.
   Anything that replays predictions from stored model JSON depends on which build parses it.
2. **Empty-row sums are -0.0.** `row_mean`, `row_sum`, `row_l1`, `row_l2` and `row_std` of an
   empty row return -0.0, because an empty `f64` sum is -0.0 with this toolchain. The WP-0.5
   table expected 0.0; `row_min` and `row_max` of an empty row are +0.0. This is the Rust side of
   the plan's "sign of zero in row reductions" divergence candidate; the emitted side is not
   checked here.
3. **`round` of values above 1e12 is the clamp.** The half-away-from-zero formula would move
   2^52 + 1 to 2^52 + 2, but every result above 1e12 leaves the evaluator as 1e12.
4. **The compiled form agrees with the interpreter** on all 82 expressions and all 266 corpus rows,
   in row context and, for the 20 map-context expressions, per element (an element error is 0.0
   on the compiled path).
