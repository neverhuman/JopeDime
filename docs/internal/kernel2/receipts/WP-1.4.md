# WP-1.4 and the rest of WP-1.8: stage-level fault telemetry (kernel2 PR 04)

Base: `6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, `origin/main` after the islands-v2 source import
(#129) merged. The series was written on `79a88767f21dccf7505987f9b0ac7366d71f2c0b`, the reviewed
head of PR 03 (#128), which merged by rebase at that head as `main`
`3a32d780814d5536489e9378335fb81e2808f9b2`, whose tree is that head's, and it was replayed onto
`6e0ae2491` before its first push (see "Replay"). The series is on the branch
`kernel2/claude-m1-telemetry` in the lane `.agent/worktrees/kernel2`. Before review, the feat-math
sources were final at `1cd0f4e50`, and the commits after it changed the parity test inventory and
these records only. The review follow-ups (see "Review follow-ups") change two of the new tests,
the new module's documentation comments, the domain table and these records; no value, count or
identity moves. Measured on
2026-09-29 on xbabe2: x86_64-unknown-linux-gnu, glibc 2.39, AMD Ryzen Threadripper PRO 3995WX,
rustc 1.97.1 (8bab26f4f 2026-07-14), cargo-nextest 0.9.133, g++ 13.3.0, clang++ 18.1.3, node
v26.1.0, Python 3.12.3 (for the emitted Python program only). Environment: `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `TMPDIR=~/.cache/jope-test-tmp/kernel2-pr04` and
`nice -n 19 ionice -c3`; test runs add the rust-test lane's `JOPE_PARITY_STRICT=1` and one
OpenMP, rayon, OpenBLAS and MKL thread. This is local evidence, not hosted CI, and the wall times
are for this host only. The logs, the perturbation diffs and the listings are in
`~/jope-evidence/kernel2/pr04/prepare-20260929T162357Z/`, those of the review follow-ups in
`~/jope-evidence/kernel2/pr04/fixer-20260929T165709Z/`, and those of the replay in
`~/jope-evidence/kernel2/pr04/rebase-20260929T171848Z/`, each with `SHA256SUMS`.

## What changed

| Commit | Change |
|---|---|
| `304f7c702` | `crates/feat-math/src/kernel/telemetry.rs`, registered by one line in `kernel/mod.rs`: `StageFaultCounts`, `PipelineTelemetry`, `ZeroWidthInput`, `apply_op_traced`, `fit_traced` and `predict_traced` (WP-1.4, and the zero-width refusal of WP-1.8). |
| `f8bb2dfac` | `crates/feat-math/tests/stage_telemetry.rs`, 18 tests. The target is assigned in `deploy/ci/coverage.json` (rust-check, rust-test, full-default) and classified in the regression test inventory with `fault_trace`'s subsystems. |
| `f08034c43` | `docs/parity/test-inventory.json`: the feat-math section re-listed. |
| `75360933c` | `docs/internal/kernel2/math-domain-table.md`: four sections for the stage level, and one line of its introduction. |
| `1cd0f4e50` | A 19th test, which fits every corpus expression as a raw and an affine symbolic program. Perturbation 5 below showed that the corpus alone did not hold the mirrored scaling to `fit`. |
| `b0d226f49` | `docs/parity/test-inventory.json` re-listed for the 19th test. |
| `3935d86e7` | Review follow-up (F1, F2): two of the new tests pin what the corrected documents say, the portfolio test on non-finite rows and the zero-width test on the legacy `fit`'s outcomes. No test is added or renamed. |
| `c38f65ad0` | Review follow-up (F1, F2, F4): the documentation comments of `kernel/telemetry.rs`. No code changes. |
| `cad9b120c` | Review follow-up (F1 to F4, NR-1, R1): the domain table's stage level. |
| last | These records: `CHANGELOG.md`, `MASTER_PLAN.md` and this receipt, regenerated after the replay from the two pre-replay records commits, `af2c7bfb9` (the records) and `13b5b6c2a` (review follow-up F4 and F5, and the record of the review). |

Every commit has Jepson Taylor `<130782313+jepsontaylor@users.noreply.github.com>` as author and
committer.

## Replay

#129, the islands-v2 source import, merged as `6e0ae2491` while this series was in review. Before
its first push, the series was replayed onto it. The pre-replay head
`13b5b6c2a0815c09270f0b5d671c197dab6764ba` is kept locally as
`refs/preserve/kernel2/pr04-prerebase-20260929T171848Z`, so every pre-replay commit named in this
receipt stays reachable.

- `git range-diff 79a88767f..13b5b6c2a 6e0ae2491..HEAD` shows the nine replayed commits unchanged
  (`=`), with no conflict: `00065af25` → `304f7c702`, `8e63e7b3b` → `f8bb2dfac`, `769aca8cc` →
  `f08034c43`, `97bd34e00` → `75360933c`, `b13dd4eeb` → `1cd0f4e50`, `afa696b9c` → `b0d226f49`,
  `4a5ca77d7` → `3935d86e7`, `7dfbc96ee` → `c38f65ad0` and `b9d52cf5e` → `cad9b120c`. The two
  records commits, `af2c7bfb9` and `13b5b6c2a`, are regenerated as the last commit.
- #129 changes none of the series' paths, and the only input of the series' checks it changes is
  `deploy/lambda`: `git diff --quiet 79a88767f 6e0ae2491 --` over `crates`, the root Cargo files,
  `rust-toolchain.toml`, `.config`, `deploy/ci`, `docs/parity`, `docs/internal/core-ip` and
  `docs/internal/kernel2` exits 0. It adds two frozen Lambda workspaces, and the teacher-v2 worker
  builds `feat-math` by path, so `deploy/ci/lambda.sh` now builds and tests six workspaces instead
  of four.
- The coverage map and both test inventories need no regeneration. This series' two items,
  re-applied to the new main's `deploy/ci/coverage.json` and findings test inventory with the
  prepare step's `jq`, give the replayed files byte for byte, and the parity inventory's feat-math
  section equals a live listing at `cad9b120c`.
- The pull request number, #130, replaces the number placeholder in `CHANGELOG.md`,
  `MASTER_PLAN.md` and this receipt.

Commands at `cad9b120c`, the tree of the last commit apart from these records, with a clean tree
(`a` and `a2` in the replay's evidence). The cargo and script steps ran under
`nice -n 19 ionice -c3`, the test runs with one OpenMP, rayon, OpenBLAS and MKL thread, and
`JOPE_PARITY_STRICT=1` where the command shows it. Where a step ran twice, both exit codes and wall
times are given. The same commands ran again at the head, with these records committed (`d` and
`d2`), with the same results.

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| feat-math PR tier | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 500 passed; 10 skipped | 69.5 s with the build, 61.3 s |
| Named targets, compare mode | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test stage_telemetry --test fault_trace --test mathexpr_current_semantics --test ip_scrub --test legacy_v2_freeze --test portable_parity` | 0, 0 | 72 passed (12, 19, 13, 18, 6, 3, 1); 3 skipped | 12.1 s, 12.3 s |
| Parity with the emitted programs | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0, 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale | 3.5 s, 3.4 s |
| The new tests, uncaptured | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test stage_telemetry --no-capture` | 0 | 19 passed; `CORPUS`, `BOUNDARY` and `SYMBOLIC` print the counts under "Tests"; the two expected panics print, at `ops.rs:339` and `heads.rs:512` | 5.8 s |
| Legacy value digest | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fault_trace --run-ignored ignored-only --no-capture -E 'test(=legacy_value_digest)'` | 0 | `b64cf538…`, 191,716 results, as PR 03 recorded | 0.7 s |
| Evaluator identity tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0 | 15 passed: 13 in `evaluation_v2` and 2 in `commands::exact_cv_v2`, so the identity table is unaffected | 56.1 s with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (csv 40, regression-ops 2, regression 5, the teacher-v2 search engine 2, the teacher-v2 worker 3, monitor 14); cargo-deny skipped (0.19.8 is installed, and the lane asserts 0.20.2), so that step is not a pass | 50.6 s |
| Emitters | `EMITTERS_OUT=<scratch> bash deploy/ci/emitters.sh` | 0 | both programs: sources byte-identical to the pins, predictions bit-identical across the 2 and 7 programs | 16.4 s |
| Identity inputs | `git diff --quiet 6e0ae2491 HEAD --` over the paths under "Values unchanged" | 0 | unchanged, with and without `rust-toolchain.toml` | 0.0 s |
| Fixtures | `git diff --stat 6e0ae2491 HEAD -- crates/feat-math/tests/fixtures/` | 0 | empty | 0.0 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.9 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 8.7 s |
| Workspace check | `cargo check --workspace --all-targets --locked` | 0 | warnings only in `battle-gpu`, `jope-dime-regressor` and `jope-parallel`, as before, and none in feat-math | 43.3 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.4 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 2.5 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python`, then with `--update` | 0, 0 | consistent: Rust 3,873 kept, Python 172; the update changes no byte | 0.1 s, 0.1 s |
| The feat-math section against a live listing | `cargo nextest list --locked -P main -p feat-math --message-format json`, then the re-list script with `--check` | 0 | equal | 0.7 s |
| The two record items against the new main | the prepare step's `jq` on `6e0ae2491`'s `deploy/ci/coverage.json` and findings inventory, then `cmp` | 0 | both equal the replayed files | 0.1 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok | 19.5 s |
| Whitespace | `git diff --check 6e0ae2491 HEAD` | 0 | clean | 0.0 s |
| Replay | `git range-diff 79a88767f..13b5b6c2a 6e0ae2491..HEAD` | 0 | nine `=`, and the two records commits dropped (`<`); at the head, `af2c7bfb9` pairs with the regenerated records commit (`!`) and `13b5b6c2a` is dropped | 0.1 s |
| Subjects | `git log --format=%s 6e0ae2491..HEAD`, searched with the ip-scrub pattern and the plan's additions | 0 | no subject matches | 0.0 s |
| Authors and trailers | `git log --format='%an <%ae>\|%cn <%ce>' 6e0ae2491..HEAD`, and the trailer on every commit | 0 | one identity, author and committer; every commit ends with the trailer | 0.0 s |

With these records in the working tree, the checks that read them ran again, each with exit 0:
`cargo fmt --all -- --check`, `coverage validate` (288 targets), `verify-test-inventory` (324
items), `test-inventory.sh --no-rust --static-python`, `ip-scrub.sh` (47 files), `check.sh`,
`git diff --check 6e0ae2491` and the placeholder check: `git grep -nI` finds no number
placeholder, and the three records files hold no template braces. Not rerun after the replay:
`just test-main` (3,808 of 3,808 at the pre-replay `afa696b9c`, "Commands and results") and the
perturbations. The library code both exercised is this head's: the later commits change a
feat-math test file, documentation comments and documents only, and #129 changes no crate of the
root workspace.

## Review follow-ups

One review of the pre-replay `af2c7bfb9` returned findings F1 to F5, NR-1 and R1 to R3. Three
commits and these records answer them. No value, count or identity moves: the code of
`kernel/telemetry.rs` is unchanged (only its documentation comments change), and two existing
tests gain assertions, so there are still 19 tests and both test inventories are unchanged.

| Finding | Severity | Resolution |
|---|---|---|
| F1 | should-fix | Fixed in `c38f65ad0` and `cad9b120c`, pinned in `3935d86e7`. The documentation of `input_sanitized` and the domain table said that an explicit portfolio runs no input sanitize. It does: `fit_portfolio` and `PortfolioModel::predict` replace the non-finite cells they read with 0.0 (`portfolio.rs` lines 334 and 760), in code that is not traced, so the count is 0 whatever the input holds. That is the one path through the traced entry points on which a non-finite input cell becomes a 0.0 that no count records, and the documents now say so. `an_explicit_portfolio_is_fitted_by_fit_and_reports_no_stage` now fits and predicts on rows with a NaN, an infinity and a negative infinity: the telemetry stays empty, zeros in their place give the same prediction bits, and another value does not. The finding's alternative, counting the portfolio's replacements, would give portfolios a partial count the brief does not ask for; it is left to PR 08 ("Still open"). |
| F2 | nit | Fixed in `c38f65ad0` and `cad9b120c`, pinned in `3935d86e7`. The legacy `fit`'s check refuses only an empty first row. With only a later row empty, `standardize` (`ops.rs` line 339) indexes the empty row and panics, and so does the `ridge` head (`std_fit`, `heads.rs` line 512), which the fixer found: "a pipeline without transforms accepts it" also held only for an empty first row. The domain table, the documentation of `fit_traced`, decision 5 and the zero-width test's comment now give each case, and the test checks both panics and a symbolic program's acceptance. |
| F3, NR-1, R1 | nit | Fixed in `cad9b120c`: the domain table's introduction says "the last four sections". |
| F4 | nit | Fixed in `c38f65ad0` and `cad9b120c`, in the API list below and in `MASTER_PLAN.md`. A model with a portfolio in it, an explicit portfolio or a compact portfolio head, records wall-clock fit times (`fit_ms`) in its member diagnostics, so its bytes differ between two calls of `fit` itself. The byte identity holds for every other model; the tests compare such a model by its predictions. |
| F5 | nit | Fixed in this receipt: `STAGES` is checked by `compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix`, not by `the_program_corpus_is_compat_freezes`. |
| R2 | nit | Fixed after the replay: the pull request number, #130, replaces the number placeholder in `CHANGELOG.md`, `MASTER_PLAN.md` and this receipt. The check before the first push fails on the number placeholder as well as on template braces, and finds neither (see "Replay"). Until the number was known the placeholder stayed, as PR 03's did until `79a88767f`. |
| R3 | nit | No change to the series. It is advice for probing the canonical checkout (`git --no-optional-locks status`); the fixer ran no command in that checkout. |

Perturbation 8 (see "Perturbations caught") checks the strengthened portfolio test: before the
follow-ups, a portfolio that counted its non-finite input passed all 19 tests.

Commands at the pre-replay `b9d52cf5e` (`cad9b120c` after the replay), the tree of `13b5b6c2a`
apart from its records, with a clean tree (`g1` and `g2` in the fixer's evidence):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.9 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 0.6 s (8.4 s when it rebuilt after the documentation change) |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 500 passed; 10 skipped | 72.6 s with the build, 61.4 s |
| Named targets | the eight `--test` targets of the table below | 0, 0 | 78 passed (12, 19, 13, 6, 18, 6, 3, 1); 3 skipped | 12.7 s, 12.4 s |
| The new tests, uncaptured | `cargo nextest run --locked -P pr -p feat-math --test stage_telemetry --no-capture` | 0 | 19 passed; the two expected panics print, at `ops.rs:339` and `heads.rs:512`; `CORPUS`, `BOUNDARY` and `SYMBOLIC` print the counts below, unchanged | 5.8 s |
| Rustdoc | `cargo doc -p feat-math --no-deps --locked` | 0 | 7 warnings, the same as at `af2c7bfb9`, none in `kernel/telemetry.rs` | 0.2 s (2.4 s for the rebuild after `7dfbc96ee`) |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent: Rust 3,873 kept, Python 172 | 0.1 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items | 0.2 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.4 s |
| The feat-math section against a live listing | `cargo nextest list --locked -P main -p feat-math --message-format json`, then the re-list script with `--check` | 0 | equal | 0.7 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.6 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok | 20.4 s |
| Workspace check | `cargo check --workspace --all-targets --locked` | 0 | warnings only in `battle-gpu`, `jope-dime-regressor` and `jope-parallel`, as before | 37.1 s |
| Whitespace | `git diff --check 79a88767f HEAD` | 0 | clean | 0.0 s |
| Identity inputs | `git diff --quiet 79a88767f HEAD --` over the paths under "Values unchanged" | 0 | unchanged | 0.0 s |

With the records of `13b5b6c2a` in the working tree, the checks that read them ran again, each with
exit 0: `cargo fmt --all -- --check`, `test-inventory.sh --no-rust --static-python`,
`verify-test-inventory` (324 items), `coverage validate` (288 targets), `ip-scrub.sh` (47 files),
`check.sh` and `git diff --check 79a88767f`. The placeholder listing found no template braces, and
the number placeholder only where those records kept it for the pull request (R2); the replay
writes the number. Not rerun then: `just test-main`, `lambda.sh`, `emitters.sh`, the legacy value
digest and the evaluator-identity tests. The follow-ups change code only in a feat-math test file
(the library change is documentation comments), change no emitted program and no identity input,
and the workspace check compiles every consumer. All of them but `just test-main` ran again after
the replay (see "Replay").

## The API

Everything is new, in `feat_math::kernel::telemetry`:

- `apply_op_traced(op: &FittedOp, x: &Rows) -> (Rows, StageFaultCounts)`: `apply_op`'s rows, bit
  for bit, and the stage's counts.
- `fit_traced(program: &GenomeIr, x: &Rows, y: &[f64]) -> Result<(Model, PipelineTelemetry)>`:
  a model whose `serde_json` bytes are `fit`'s (apart from the wall-clock `fit_ms` that a model with
  a portfolio in it records), and the counts of the training pass.
- `predict_traced(model: &Model, x: &Rows) -> Result<(Vec<f64>, PipelineTelemetry)>`:
  `Model::predict`'s bits and the counts of the inference pass.
- `StageFaultCounts { evaluated_cells, faulted_cells: [u64; FAULT_SLOTS], hard_rows,
  compile_failed, constant_output, epilogue_sanitized }`, with `faulted(FaultClass)`;
  `PipelineTelemetry { input_sanitized, stages }`; `FAULT_SLOTS` = 32. Both structs are
  `#[non_exhaustive]` and derive `Clone`, `Debug`, `Default`, `PartialEq`, `Eq`, `Serialize` and
  `Deserialize`.
- `ZeroWidthInput { row }`, an error type, with `REASON` ("zero-width input") and `find(&Rows)`.

The definition of every count is in the domain table's "Stage telemetry" section. No existing item
changes: no new dependency, no changed signature, no new enum variant and no removed item. The one
existing file edited under `crates/` is `kernel/mod.rs`, by one added line.

## Decisions taken while implementing

1. **The code is in `kernel/telemetry.rs`, so no identity moves.** The brief asks for the route that
   mints none. `crates/feat-math/src/kernel/` is outside the ADR-0004 hash list
   (`crates/jope-campaigns/src/evaluation_v2.rs`, `compute_implementation_sha256`), outside every
   jope-engine lineage (`lineage_sources.v1.json` lists `model.rs` as the only feat-math source),
   and outside compat_freeze's pinned files. The plan already puts PR 05's `kernel/scope.rs` and PR
   06's `kernel/width.rs` there. A top-level `telemetry.rs` would have needed a line in `lib.rs`,
   which is hashed.
2. **`finish` and `op_append` are mirrored, not made `pub(crate)`.** Both are private in `ops.rs`,
   which is hashed, so a visibility change there would mint an identity. The brief's premise that
   they are already reachable does not hold, and its rule for that case applies: a mirror outside
   `ops.rs`, tested exhaustively against `apply_op`. The append mirror lists all 36 `FittedOp`
   variants without a wildcard, so a new op does not compile until it is added there. There is no
   separate fallback helper to reuse: the `MapExpr` and `DeriveExpr` fallbacks are inline in
   `apply_op_generated`, and `apply_op_traced` repeats their branches with the traced evaluators.
   Every other op runs `apply_op_generated` itself, which is `pub(crate)`.
3. **`fit_traced` is one pass that mirrors `fit`, not `fit` followed by a replay.** Search will call
   it for every candidate program (plan E4, WP-1.5), and a replay would apply every transform
   twice. The mirror calls `split_target`, `split_stages`, `fit_target`, `fit_op` and `fit_head` in
   `fit`'s order and copies the symbolic scaling verbatim, keeping the order of every
   floating-point operation; the tests hold it to `fit`'s bytes.
4. **`predict_traced` returns a `Result`.** The brief gives it the type `(Vec<f64>,
   PipelineTelemetry)`, which cannot carry the zero-width refusal its item 4 asks for at the traced
   entry points. `fit_traced` returns a `Result` already, so now both do.
5. **The zero-width refusal.** A zero-width input is one with a row that has no columns; a set with
   no rows is not one. The error is a type, `ZeroWidthInput { row }`, so a caller matches it with
   `downcast_ref` rather than by its text, which starts with WP-1.8's reason, "zero-width input".
   `fit_traced` runs `fit`'s own check first, so an empty or short input keeps `fit`'s message. This
   is the one behaviour in which the traced entry points are stricter than the legacy ones. The
   legacy `fit` takes the width from the first row: with an empty first row, a pipeline without
   transforms accepts the input and a transform refuses it; with only a later row empty,
   `standardize` and the `ridge` head index the empty row and panic; a symbolic program accepts
   both (review follow-up F2). `predict` does not check. The identities hold on every input the
   traced entry points accept.
6. **What the counts mean.** The expression counts cover the cells an expression generated, before
   the append merge and the width cap. `hard_rows` counts rows with a Hard cell, so a `map` row
   without cells never is one. `constant_output` needs at least one cell and compares bits: one
   cell is trivially constant, and +0.0 and -0.0 differ. A compile failure counts each fallback
   cell as evaluated and as `COMPILE_FAILED`, so rates stay defined.
7. **`faulted_cells` is `[u64; 32]`**, one slot per bit of `FaultSet`, rather than the workplan's
   `[u64; N_CLASSES]`: `FaultClass::ALL` is a slice precisely so that an appended class changes no
   type (PR 03's `d208f4fcf`).
8. **`input_sanitized` is added.** A pipeline replaces non-finite inputs with 0.0 on entry, so its
   stages never see them. Without this count a non-finite input would become a fault-free zero,
   which WP-1.8's goal rules out. An explicit portfolio replaces them with 0.0 too, inside
   `portfolio.rs`, which is not traced, so its count is 0 whatever its input holds. That is the one
   path on which a non-finite input cell still becomes an uncounted zero (review follow-up F1).
9. **Symbolic programs and portfolios.** A symbolic program is a model, not an op, so
   `fit_traced` and `predict_traced` trace it, as one entry. It is never compiled (`fit` and
   `predict` call the interpreter), so a wrong arity there is the interpreter's `COMPILE_FAILED` on
   every row, and `compile_failed` stays false. An explicit portfolio is handed to `fit` and
   reports no stage, because its members are fitted inside `portfolio.rs`; the members of a compact
   portfolio head are not traced either, and neither is an explicit portfolio's own input sanitize
   (decision 8).
10. **The struct shape.** `#[non_exhaustive]` lets a later change add a field: PR 08 needs the
    cells with at least one Domain class, which per-class counts cannot give when a cell has two.
    `Serialize` and `Deserialize` come from the existing `serde` dependency, for receipts.

## Values unchanged

- **No evaluator identity, lineage source or fixture moves.** `git diff --quiet 6e0ae2491
  b0d226f49 --` over `crates/feat-math/tests/fixtures`, `crates/jope-campaigns`,
  `crates/jope-engine`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, and feat-math's
  `Cargo.toml`, `lib.rs`, `ops.rs`, `model.rs` and `math_expr.rs` exits 0. Apart from the records,
  the series changes seven files (below, at `b0d226f49`). The review follow-ups change three of
  them again, and the same check at `cad9b120c` exits 0 too.
- **compat_freeze** passes 12 of 12 in compare mode, twice, so every frozen surface is unchanged,
  `evaluator_identity.v1.json` (`implementation_sha256` `3ec654cb…`, recorded by PR 03) and
  `lineage_sources.v1.json` among them.
- **Models and predictions.** For the 54 fitted programs of PR 01's corpus, `fit_traced`'s model has
  `fit`'s `serde_json` bytes, and `predict_traced`'s predictions have `Model::predict`'s bits on the
  held-out and the training rows, for the model of either fit; the portfolio program is skipped, as
  compat_freeze skips it. The same holds for the 82 corpus expressions as raw and affine symbolic
  programs (164 fits; 57 affine fits take the slope branch) and for the five stored models.
- **Stage outputs.** `apply_op_traced` returns `apply_op`'s and `apply_op_train`'s rows bit for bit
  at every op of those 54 programs, on the training pass and on the held-out rows; for the 82 corpus
  expressions as `map` and `derive` stages, append off and on, on the 266 corpus rows (328 stages);
  for compat_freeze's 16 stages on its raw stage matrix; for one stage of each of the 36 ops, append
  off and on, on the training rows and on raw boundary rows; and past the 512-column cap.
- **Emitted programs and the evaluator.** `parity_bits` finds the divergences #127 recorded (rust
  6, python 6 and cpp17-fitted 10, all allowlisted, 0 unexpected, 0 stale); `deploy/ci/emitters.sh`
  writes sources byte-identical to their pins, with predictions bit-identical across programs; the
  legacy value digest is `b64cf538…`, over 191,716 results, as PR 03 recorded.

```text
$ git diff --stat 6e0ae2491 b0d226f49
 crates/feat-math/src/kernel/mod.rs                 |    1 +
 crates/feat-math/src/kernel/telemetry.rs           |  521 +++++++++
 crates/feat-math/tests/stage_telemetry.rs          | 1145 ++++++++++++++++++++
 deploy/ci/coverage.json                            |   16 +
 docs/internal/core-ip/findings/test-inventory.json |    7 +
 docs/internal/kernel2/math-domain-table.md         |  114 +-
 docs/parity/test-inventory.json                    |   32 +-
 7 files changed, 1832 insertions(+), 4 deletions(-)
```

## Tests (`crates/feat-math/tests/stage_telemetry.rs`: 19)

Each expression stage the tests meet is recounted independently: cell by cell through the
interpreter's traced calls (PR 03 proved the compiled form's values and faults equal theirs), with
the epilogue applied to the generated values by hand (`recount`). Every other op must report
nothing but `epilogue_sanitized`. Every comparison of values is by `to_bits()`.

- **PR 01's corpus.** `the_program_corpus_is_compat_freezes`: the copied `NAMED_PROGRAMS`,
  `IP_SCRUB_PIPELINES` and `IP_SCRUB_SYMBOLIC` equal compat_freeze.rs's lines, entry for entry; the
  55 names are `model_serde.v1.json`'s, in order; and the one program skipped here is the one the
  fixture records as skipped. `fit_traced_serializes_to_fits_bytes_on_every_corpus_program`:
  on the 54 other programs, `fit_traced`'s model has `fit`'s `serde_json` bytes;
  `predict_traced`'s predictions have `Model::predict`'s bits on the held-out and the training rows,
  for the model of either fit; the training pass's telemetry equals `predict_traced`'s on the
  training rows and a replay of every op; and every op, on the training pass and on the held-out
  rows, returns `apply_op`'s and `apply_op_train`'s rows, with recounted counts (28 expression
  stages). `two_runs_give_identical_telemetry`: two fits of each give equal telemetry, model bytes,
  predictions and inference telemetry, and the telemetry round-trips through `serde_json`.
  `predict_traced_equals_predict_on_the_stored_models`: the five stored models, loaded once.
- **Boundary rows.** `expression_stages_equal_apply_op_on_every_corpus_expression_and_row`: the 82
  corpus expressions as `map` and `derive` stages, append off and on, on all 266 corpus rows (NaN,
  ±inf, ±0, the guards' neighbours and the empty row among them): 328 stages, of which 320 report a fault and
  198 a hard row; with append, `epilogue_sanitized` is exactly the corpus's non-finite cells.
  `every_corpus_expression_as_a_symbolic_program_fits_to_fits_bytes`: the same expressions as raw
  and affine symbolic programs on eight-column rows (164 fits; 57 affine fits take the slope
  branch), with model bytes, predictions on held-out and on the 265 non-empty corpus rows, and the
  recounted counts of every pass. `compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix`:
  the copied `STAGES` equal compat_freeze.rs's lines, entry for entry, and its 16 stages run on its
  raw 5×4 stage matrix, and as programs predicting it (the entry sanitize replaces its NaN, inf and
  -inf: `input_sanitized` 3).
  `every_op_equals_apply_op_with_append_on_and_off`: one stage per `FittedOp` variant (36, each
  identified by its serialized tag, with its fitted append flag), append off and on, on the training
  rows and on raw boundary rows. `the_width_cap_comes_before_the_epilogue_count`: 300 columns
  appended to 300 generated ones; `polar`'s overflowing radius is counted at merged column 300 and
  not at 598, which the cap cuts; a `map` counts its 300 generated cells.
- **The workplan's cases.** `log_of_a_negative_column_counts_log_domain_and_of_a_positive_one_nothing`:
  `(pipe X (derive expr="log(col(0))") (ridge))` reports `LOG_DOMAIN` on exactly the training rows
  whose `col(0)` is at most 1e-300, and no class on `|v| + 1`.
  `a_column_past_the_width_is_a_hard_row_and_the_legacy_zeros`: `(map expr="col(9)")` at width 4 on
  40 rows: `hard_rows` 40, `COLUMN_OOB` on 160 cells and nothing else, and the rows are
  `apply_op`'s, all +0.0. `a_step_above_the_data_is_a_constant_output`: `step(col(0), 100)` is
  constant, `step(col(0), 0)` is not, and neither is `c0` on +0.0 and -0.0.
- **Also.** `a_stage_that_emits_a_non_finite_value_counts_the_epilogue` (`symlog c=1` on two cells of
  ±1e200: 2, in fit and predict; a merged NaN input: 1);
  `a_structurally_invalid_expression_is_a_compile_failure` (a deserialized wrong-arity AST in `map`
  and `derive`, append off and on; and in a symbolic program, where the interpreter reports it on
  every row and `compile_failed` stays false); `the_symbolic_path_reads_raw_inputs_and_a_pipeline_sanitizes_them`;
  `an_explicit_portfolio_is_fitted_by_fit_and_reports_no_stage` (on rows with a NaN, an infinity
  and a negative infinity, which `portfolio.rs` replaces with 0.0 itself: the telemetry stays empty
  and the model is not the mean-only fallback; zeros in their place fit and predict the same bits,
  and another value in place of one of them does not; prediction bits only, because a portfolio's
  bytes carry wall-clock fit times); `stage_level_empty_row_identities`;
  `the_traced_entry_points_refuse_a_zero_width_input` (ragged and all-empty rows, four program
  kinds; `fit`'s own message first; no rows accepted by `predict_traced`; and the legacy `fit`:
  all-empty rows accepted by a head-only pipeline and refused by a transform, a later empty row a
  panic (index out of bounds) under `standardize` and under the `ridge` head, and a symbolic program
  accepting both); `the_telemetry_types_stay_shareable_across_threads`.

The test binary runs in about 2.5 s in a debug build here; its slowest test takes 2.4 s. The
`CORPUS`, `BOUNDARY` and `SYMBOLIC` lines of `--no-capture` print the counts above. Under
`--no-capture` the default panic hook also prints the zero-width test's two expected panics.

## Perturbations caught

Each perturbation was applied to `crates/feat-math/src/kernel/telemetry.rs` before the replay (at
`97bd34e00` for 1 to 5, at `b13dd4eeb` for 6 and 7, and at `4a5ca77d7` for 8, a review follow-up;
they are `75360933c`, `1cd0f4e50` and `3935d86e7` after it, and this section keeps the pre-replay
ids), the `stage_telemetry` tests were run, and the file was restored with `git checkout`; no
tracked file was left modified. The diffs and logs are in the evidence.

| # | Perturbation | Failing tests |
|---|---|---|
| 1 | The epilogue counts non-finite values before the width cap, not after | `the_width_cap_comes_before_the_epilogue_count` (1) |
| 2 | The append mirror reads `false` for `cheb` | `every_op_equals_apply_op_with_append_on_and_off`, `fit_traced_serializes_to_fits_bytes_on_every_corpus_program` (2) |
| 3 | `hard_rows` counts Hard cells, not rows | `a_column_past_the_width_is_a_hard_row_and_the_legacy_zeros`, `a_structurally_invalid_expression_is_a_compile_failure`, `compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix`, `expression_stages_equal_apply_op_on_every_corpus_expression_and_row` (4) |
| 4 | `constant_output` compares values with `==`, so +0.0 equals -0.0 | `a_step_above_the_data_is_a_constant_output` (1) |
| 5 | The mirrored symbolic slope is `cov * (1.0 / var)`, not `cov / var` | **none of the 18 tests at `97bd34e00`**; after `b13dd4eeb`, `every_corpus_expression_as_a_symbolic_program_fits_to_fits_bytes` (1) |
| 6 | A `map` stage's compile failure leaves `compile_failed` false | `a_structurally_invalid_expression_is_a_compile_failure` (1) |
| 7 | `predict_traced` reports 0 for `input_sanitized` | `compat_freeze_stages_equal_apply_op_on_the_raw_stage_matrix` (1) |
| 8 | Both portfolio branches report their non-finite input as `input_sanitized` | **none of the 19 tests of `af2c7bfb9`** (exit 0); with the tests of `4a5ca77d7`, `an_explicit_portfolio_is_fitted_by_fit_and_reports_no_stage` (1, exit 100) |

Perturbation 8 is the review's F1 seen from the tests: before the follow-ups, the portfolio test
ran on finite rows only, so nothing held a portfolio's count to 0 whatever the input holds. It was
run twice on the perturbed module: with the test file of `af2c7bfb9` checked out in its place, then
with the file of `4a5ca77d7`.

Perturbation 5 is why `b13dd4eeb` exists: the corpus holds four affine symbolic programs, and for
each of them the two slopes happen to round to the same bits, so the serialized models agreed. A
change that is not an equivalence can therefore pass a byte comparison over a small set of
programs; the new test fits 82 affine programs, 57 of which take the slope branch.

One change is not caught, by design: taking the interpreter where `apply_op_generated` takes the
compiled form (or the reverse) gives the same values and the same faults, which PR 03 proved, so
no test can tell the two apart.

## Evaluator identity (ADR-0004)

None moves, so there is no known-table row, no receipt run and no rebaseline. The
`implementation_sha256` stays `3ec654cb6a44f26d5770bcab589273879605831570025cb42475fa1aad1e1f63`,
the identity digest `891dae89…` and `cargo_lock_sha256` `8ecdd1fc…`, as PR 03 recorded them.
compat_freeze's `evaluator_identity_inputs_match_the_recorded_identity` and
`lineage_sources_match_the_recorded_sources` pass in compare mode, and `git diff --stat 6e0ae2491
HEAD -- crates/feat-math/tests/fixtures/` is empty.

Two things still change with this series, as with any feat-math change. jope-engine's identities
hash its own executable (`/proc/self/exe`) besides their listed sources, so a jope-engine binary
that links the new module mints new teaching, corpus and query identities (`lineage_sources.v1.json`
says so). And `kernel/telemetry.rs` belongs in the ADR-0004 amendment (plan WP-8.3): it is outside
the hash list today, which is right while nothing scores with it, and becomes part of scoring when
PR 08 wires it into search.

## Commands and results

Before review and before the replay, at `afa696b9c` (`b0d226f49` after the replay) with a clean
tree: the tree of `af2c7bfb9` apart from the records. Where a step ran twice, both exit codes and
wall times are given. The commands of the review follow-ups are in "Review follow-ups", and those
after the replay in "Replay".

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 7.0 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 8.3 s |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 500 passed; 10 skipped (6 ignored, 4 by the `pr` filter) | 60.5 s, 60.9 s |
| Named targets | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test stage_telemetry --test fault_trace --test parity_bits --test mathexpr_current_semantics --test ip_scrub --test legacy_v2_freeze --test portable_parity` | 0, 0 | 78 passed (12, 19, 13, 6, 18, 6, 3, 1); 3 skipped | 12.4 s, 12.3 s |
| Parity with the emitted programs | `cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0, 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale | 3.5 s, 3.5 s |
| Legacy value digest | `cargo nextest run --locked -P pr -p feat-math --test fault_trace --run-ignored ignored-only --no-capture -E 'test(=legacy_value_digest)'` | 0 | `b64cf538…`, 191,716 results, as PR 03 recorded | 0.7 s |
| Evaluator identity tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0 | 15 passed: 13 in `evaluation_v2` and 2 in `commands::exact_cv_v2` | 54.2 s with the build |
| Main tier, whole workspace | `just test-main` (`cargo nextest run -P main --workspace --locked -j 32`, one thread each for OpenMP, rayon, OpenBLAS and MKL, `JOPE_PARITY_STRICT=1`) | 0 | 3,808 run, 3,808 passed, 65 skipped. The consumers of feat-math all pass: `jope-dime-regressor` 997 and `jope-parallel` 775; feat-math 504, among them the 19 new tests and the four the `pr` filter skips. `fast_profile_portfolio_head` ran longer than 600 s (reported slow) and passed | 687.0 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.5 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 0.3 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent: Rust 3,873 kept, Python 172 | 0.1 s |
| The feat-math section against a live listing | `cargo nextest list --locked -P main -p feat-math --message-format json`, then the re-list script with `--check` | 0 | equal | 0.7 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok; gitleaks found no leaks; coverage 288 targets; repository layout 0 findings | 20.2 s |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 4 built and tested with `--locked`, 61 tests passed; cargo-deny skipped (0.19.8 is installed, and the lane asserts 0.20.2), so that step is not a pass | 34.9 s |
| Emitters | `EMITTERS_OUT=<scratch> bash deploy/ci/emitters.sh` | 0 | both programs: sources byte-identical to the pins, predictions bit-identical across the 2 and 7 programs | 17.5 s |
| Workspace check | `cargo check --workspace --all-targets --locked` (the rust-check lane's command) | 0 | the workspace checks; warnings only in `battle-gpu`, `jope-dime-regressor` and `jope-parallel`, which this series does not touch, and none in feat-math (the lane does not deny warnings) | 38.2 s |
| Whitespace | `git diff --check 79a88767f HEAD` | 0 | clean | 0.0 s |
| Identity inputs | `git diff --quiet 79a88767f HEAD --` over the paths under "Values unchanged" | 0 | unchanged | 0.0 s |
| Subjects | `git log --format=%s 79a88767f..HEAD`, searched with the ip-scrub pattern and the plan's additions | 0 | no subject matches | 0.0 s |
| Authors and trailers | `git log --format='%an <%ae>\|%cn <%ce>' 79a88767f..HEAD`, and the trailer on every commit | 0 | one identity, author and committer; every commit ends with the trailer | 0.0 s |

The second runs of the PR tier, the named targets and `parity_bits` were made with the records of
`af2c7bfb9` already in the working tree, beside the unchanged sources. That records commit changes
documentation only: `git diff --stat afa696b9c af2c7bfb9` lists `CHANGELOG.md`, `MASTER_PLAN.md`
and this receipt. With the records in the working tree, the checks that read them ran again and exit 0:
`cargo fmt --all -- --check`, `coverage validate` (288 targets), `verify-test-inventory` (324
items), `test-inventory.sh --no-rust --static-python`, `ip-scrub.sh` (47 files), `check.sh` and
`git diff --check 79a88767f`.

The brief's identity-test command names `-p jope-campaigns`, which is not a package: the package is
`jope-dime-regressor`, its tests are modules of its binary, and the `pr` profile's default filter
excludes it, so the command above, PR 03's, is the one that runs them. `just test-main` was run as
its recipe's `cargo nextest run` line, with the recipe's environment.

## Still open

- **Hosted CI.** Every lane runs for this change: `crates/feat-math/**` and `deploy/ci/coverage.json`
  select all nine jobs. The first hosted run is also the cross-host check of the libm-dependent
  values the new tests compare (always two evaluations by one build, never a recorded value).
- **An explicit portfolio's own input sanitize** is not counted (review finding F1). If search
  scores portfolios by their telemetry, PR 08 can count it: the non-finite cells within the fitted
  width that `fit_portfolio` and `PortfolioModel::predict` replace.
- **Cells per severity for PR 08.** WP-1.5 scores `domain_rate` as the Domain-faulted cells over the
  evaluated cells. `faulted_cells` counts a cell once under each of its classes, and one cell can
  carry two Domain classes (`col(0)` on a NaN raises `NONFINITE_INPUT` and `NONFINITE_SANITIZED`),
  so the sum over Domain classes can exceed the number of Domain-faulted cells. PR 08 should add a
  per-severity cell count; `StageFaultCounts` is `#[non_exhaustive]` so that it can.
- **The stage cache.** `fit_traced` mirrors `fit`, not `fit_with_stage_cache`; a search that uses
  the cache and wants telemetry has to choose, or a traced cache path is added (PR 08).
- **The ADR-0004 amendment** (plan WP-8.3) should list `kernel/telemetry.rs` once PR 08 scores with
  it, beside `math_expr/compiled.rs` and `math_expr/fault.rs`. It needs Codex's agreement.
- **Local gaps.** cargo-deny was skipped by `lambda.sh` (0.19.8 is installed, 0.20.2 is asserted);
  the hosted hygiene lane runs it. Clippy `-D warnings` on `jope-dime-regressor`, which fails at
  PR 03's base in files this series does not touch, was not rerun: the series changes no file of
  that package.
- **The feat-math merge announcement** in `AGENT_CHAT.md` belongs to the orchestrator.
