# WP-1.7: fitted-width contract (kernel2 PR 06)

Base: `dd5ccf3f19c3314f5ef9bf34ccf362422650cce7`, `origin/main` after PR 05 (#131) merged; PR 01
to PR 03 are #126 to #128, #129 is the islands-v2 source import, and PR 04 is #130. The series was
written on `6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, `origin/main` after #129 merged, and it was
replayed onto `dd5ccf3f1` before its first push (see "Replay"). This pull request is #132.
Branch `kernel2/claude-m1-width`, in the lane `.agent/worktrees/kernel2`. The implementation was
complete at the pre-replay `925ec2a32` (`75f08888a` after the replay), which changes only two
documentation comments after the pre-replay `135cd5b4c` (`6037be4f2`), where its full gate ran,
and `ebf647557` recorded it. The review round then fixed four findings in the pre-replay
`a9c0c4766`, `9105c6ff5`, `c9983d2e1` and `c8e36e260` (`cb27d2151`, `bd9945d00`, `c4d4756cd` and
`524c79701` after the replay), where the feat-math sources and tests are final and the gate ran
again ("The review round", below); the last commit regenerates these records (see "Replay").
Measured on 2026-09-29 on xbabe2: x86_64-unknown-linux-gnu,
glibc 2.39, AMD Ryzen Threadripper PRO 3995WX, rustc 1.97.1 (8bab26f4f
2026-07-14), cargo-nextest 0.9.133, g++ 13.3.0, clang 18.1.3, jq 1.7. Environment: `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `TMPDIR` under `~/.cache/jope-test-tmp/`,
`nice -n 19 ionice -c3`, with `JOPE_COMPAT_FREEZE_REGENERATE` and `JOPE_COMPAT_FREEZE_SHA` unset;
test runs add the rust-test lane's `JOPE_PARITY_STRICT=1` and one OpenMP, rayon, OpenBLAS and MKL
thread. This is local evidence, not hosted CI, and the wall times are for this host only (other
lanes' builds and tests ran on it at the same time).

This is the Rust half of plan row 06. The fitted-C++ half (the plan's X3, runtime revision 3) is
PR 07, because it moves the pinned fitted-C++ bytes and needs its own recording procedure.

## What changed

| Commit | Change |
|---|---|
| `5b75f3f9f` | `crates/feat-math/src/kernel/width.rs` (new): `WidthReq`, `required_input_width`, `check_fitted_state`, `output_width` and `required_head_width`. `kernel/mod.rs` gains `pub mod width;`, after #131's `pub mod scope;` and #130's `pub mod telemetry;`. |
| `65d90c92e` | `kernel/width.rs`: `Model::try_predict` (an inherent method declared in this module), `check_input_width`, `WidthMismatch` and `MismatchKind`. |
| `827fda6cf` | `advanced_v3/transforms/mod.rs`: three guards, each a new private function called from one new line, in `legacy_to_block` and in the `corr_screen` and `pls_project` arms of `apply_block`. No other line changes. The replay reworded its subject to name those three stages (see "Replay"). |
| `6037be4f2` | `crates/feat-math/tests/fitted_width.rs` (new): 16 tests. |
| `75f08888a` | `kernel/width.rs`, documentation only: `try_predict` rules out the panics of a fitted stage, and a head's own tables stay trusted; `OutputOverflow` names the arithmetic overflow too. Every changed line is a `///` comment, and every line keeps its number. |
| `cb27d2151` | Review round. `advanced_v3/transforms/mod.rs`: with no rows, the legacy check compares no width and refuses only a fitted column index at which finishing the stage looks the block up, and the `pls_project` check checks nothing (decision 9). `fitted_width.rs`: a test of tables with no rows. |
| `bd9945d00` | Review round. `advanced_v3/transforms/mod.rs`: with rows, the legacy check also refuses an output width that overflows (decision 9). `kernel/width.rs`: `check_fitted_state`'s documentation names that exception, which `output_width` refuses. `fitted_width.rs`: three overflowing stages in the schema-bound test. |
| `c4d4756cd` | Review round. `fitted_width.rs`: the head test requires every refusal to be one the requirement makes. |
| `524c79701` | Review round. `fitted_width.rs`: a test that pins the unguarded `piecewise_linear` arm (decision 10). |
| last | These records: `deploy/ci/coverage.json`, both test inventories, `CHANGELOG.md`, `MASTER_PLAN.md` and this receipt, regenerated after the replay from the two pre-replay records commits, `ebf647557` (the first records) and `dbd10526d` (the records of the review round). |

Every commit has Jepson Taylor `<130782313+jepsontaylor@users.noreply.github.com>` as author and
committer.

The public API changes by additions only: `feat_math::kernel::width::{WidthReq,
required_input_width, check_fitted_state, output_width, required_head_width, check_input_width,
WidthMismatch, MismatchKind}` and `Model::try_predict`. `WidthReq` has `admits(width)` and a
`Display` ("exactly 8", "at least 8"). `WidthMismatch` has the public fields `stage`, `expected`,
`got` and `kind`, and implements `Display`, `std::error::Error`, `Clone`, `Copy`, `Debug`,
`PartialEq` and `Eq`. There is no new dependency, no new variant of an existing enum, no changed
public signature and no removed item, and the crate root re-exports nothing new. Nothing that ships
calls `try_predict`. The only behaviour that changes is at the schema-bound boundary, where fitted
state that no fit produces, and a table with no rows at a stage whose finishing panicked, now give
errors instead of panics or misreads (decision 9).

## Replay

#130 (PR 04, stage telemetry) merged while this series was being prepared, and #131 (PR 05, the
static scope check) while it was in review, as `dd5ccf3f1`. Before its first push, the series was
replayed onto `dd5ccf3f1`. The
pre-replay head `dbd10526de4a2eced15fe399c273f3763e7ce653` is kept locally as
`refs/preserve/kernel2/pr06-prerebase-20260929T202502Z`, so every pre-replay commit named in this
receipt stays reachable.

- `git range-diff 6e0ae2491..dbd10526d dd5ccf3f1..HEAD` shows the nine replayed commits:
  `f34ff4656` → `5b75f3f9f`, changed only by the resolution below (`!`); `5b6a089ec` →
  `827fda6cf`, changed only in its subject (`!`, below); and `fd4ffe5f3` → `65d90c92e`,
  `135cd5b4c` → `6037be4f2`, `925ec2a32` → `75f08888a`, `a9c0c4766` → `cb27d2151`, `9105c6ff5` →
  `bd9945d00`, `c9983d2e1` → `c4d4756cd` and `c8e36e260` → `524c79701`, unchanged (`=`). Each
  keeps its author and author date, and each but `827fda6cf` its message. The two records commits,
  `ebf647557` and `dbd10526d`, are regenerated as the last commit: range-diff pairs `ebf647557`
  with it (`!`) and lists `dbd10526d` as dropped (`<`).
- **One conflict, the expected one.** #130 and #131 added `pub mod telemetry;` and
  `pub mod scope;` to `kernel/mod.rs` at the line where `f34ff4656` adds `pub mod width;`. All
  three lines are kept, in rustfmt's order: the file is the new main's with `pub mod width;` added
  after `pub mod telemetry;`, and it differs from `f34ff4656`'s by #130's and #131's lines alone.
  range-diff shows them as context lines of the hunk.
- **One subject reworded (R1).** The subject of `5b6a089ec`, "feat-math: the schema-bound table
  boundary returns width errors instead of panicking", claimed more than the commit does, because
  the `piecewise_linear` arm of `apply_block` still panics (F1, "Still open"). Right after its
  pick, `git commit --amend` gave it the subject "feat-math: width errors instead of panics for
  the legacy, corr_screen and pls_project stages of a schema-bound pipeline", which names the
  stages the CHANGELOG line names. Its body, author and author date are unchanged, and range-diff
  shows the one changed line of the commit message and nothing else.
- **The sources are the reviewed ones.** `kernel/width.rs`, `advanced_v3/transforms/mod.rs` and
  `tests/fitted_width.rs` are byte-identical to `c8e36e260`, and every other difference under
  `crates/` between `c8e36e260` and `524c79701` is #130's and #131's, byte for byte:
  `kernel/telemetry.rs`, `kernel/scope.rs`, `kernel/pipeline.rs`, `math_expr/evolution.rs`,
  `tests/stage_telemetry.rs`, `tests/scope_validation.rs` and their two lines in `kernel/mod.rs`.
  Before the reset, `git diff --quiet 6e0ae2491 dd5ccf3f1 --` over the series' own files and this
  receipt exited 0: #130 and #131 changed none of them.
- **The records are regenerated, not taken whole.** #130 and #131 added their own coverage items,
  findings roles and 33 feat-math tests (`stage_telemetry` 19, `scope_validation` 14) beside the
  places this series writes. This series' coverage item and findings role are re-applied to the new
  main's `deploy/ci/coverage.json` and findings test inventory with `jq`, and each adds exactly the
  lines `ebf647557` added: coverage 289 to 290 targets, findings 325 to 326 items. The parity
  inventory's feat-math section is re-listed from `cargo nextest list --locked -P main -p feat-math
  --message-format json` at `524c79701`, in the shape `test-inventory.sh --update` writes:
  feat-math 524 to 542 tests (6 ignored) in 25 binaries, Rust 3,887 to 3,905 (65 ignored), and
  every other package unchanged; the `fitted_width` entry is `dbd10526d`'s, byte for byte.
  `test-inventory.sh --no-rust --static-python --update` then changes no byte.
- The pull request number, #132, replaces the number placeholder in `CHANGELOG.md` and
  `MASTER_PLAN.md`, whose record now carries #131's readback instead of #129's.
- **One correction besides the replay.** The reference check below finds that a captured file
  list in a jope-records replay fixture,
  `crates/jope-records/tests/fixtures/python-parity/coverage/files.txt`, names
  `advanced_v3/transforms/mod.rs` and `kernel/mod.rs` as paths, so "Identity unchanged" no longer
  says that the benchmark receipt is the one other place naming the first. The fixture includes and
  hashes nothing (#131's receipt found the same list), and no `include_str!` or `include_bytes!`
  reads a source under `src/kernel/` or `src/advanced_v3/`.

Commands at `524c79701`, the tree of the last commit apart from these records, with a clean tree
(`a` and `a2` in the replay's evidence). The cargo and script steps ran under
`nice -n 19 ionice -c3`, the test runs with one OpenMP, rayon, OpenBLAS and MKL thread, and
`JOPE_PARITY_STRICT=1` where the command shows it. Where a step ran twice, both exit codes and wall
times are given. The same commands ran again at the head, with these records committed (`d` and
`d2`), with the same results.

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| feat-math PR tier | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 532 passed in 25 binaries: #131's 514 and the 18 of `fitted_width`; 10 skipped (6 ignored, 4 by the pr filter) | 69.2 s with the build, 59.9 s |
| Eight named targets, compare mode | `cargo nextest run --locked -P pr -p feat-math --test fitted_width --test compat_freeze --test scope_validation --test stage_telemetry --test advanced_v3_conformance --test advanced_v3_parity --test advanced_v3_property --test advanced_v3_fitted_cpp` | 0, 0 | 86 passed (18, 12, 14, 19, 7, 1, 2, 13); 0 skipped | 45.8 s, 46.0 s |
| The review round's eleven targets, with `scope_validation` and `stage_telemetry` | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fitted_width --test compat_freeze --test ip_scrub --test legacy_v2_freeze --test portable_parity --test advanced_v3_conformance --test advanced_v3_parity --test advanced_v3_property --test advanced_v3_fitted_cpp --test advanced_v3_multiclass --test kernel_pipeline --test scope_validation --test stage_telemetry` | 0 | 105 passed (18, 12, 6, 3, 1, 7, 1, 2, 13, 6, 3, 14, 19); 1 skipped | 45.4 s |
| The new tests' output | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fitted_width --no-capture` | 0, 0 | 18 passed; the `STAGES` and `CORPUS` lines byte-identical across the runs and to the implementation's (sorted, sha256 `24a74b77dcc26027e94ab12da9aa61b3c08555d493126a1634c68643c12b018e`) | 2.0 s, 1.9 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed, compare mode; the fixtures unchanged | 11.9 s, 11.5 s |
| Parity with the emitted programs | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences over 19 programs, all allowlisted, 0 unexpected, 0 stale | 3.5 s |
| Legacy value digest | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fault_trace --run-ignored ignored-only --no-capture -E 'test(=legacy_value_digest)'` | 0 | `b64cf538…`, 191,716 results, as PR 03 recorded | 0.7 s |
| Evaluator identity tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0 | 15 passed, so the identity table is unaffected | 53.5 s with the build |
| Consumers of `predict_table` | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P main -p jope-parallel -p jope-dime-regressor --test-threads 32` | 0 | 1,772 passed in 141 binaries; 25 skipped | 286.3 s with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (csv 40, regression-ops 2, regression 5, the teacher-v2 search engine 2, the teacher-v2 worker 3, monitor 14); cargo-deny skipped in all 6 (0.19.8 installed, 0.20.2 asserted), so that step is not a pass | 47.5 s |
| Emitters | `EMITTERS_OUT=<a new directory> bash deploy/ci/emitters.sh` | 0 | the champion's Python and Rust programs, and the portable program's seven builds (Python, Rust, g++ and clang++ at C++17 and C++20, JavaScript), agree bit for bit on 34 rows; every expectation digest matches | 17.0 s |
| Workspace check | `cargo check --workspace --all-targets --locked` | 0 | warnings only in `battle-gpu`, `jope-parallel` and `jope-dime-regressor`, as before, and none in feat-math | 39.2 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.8 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 9.0 s |
| Identity inputs | `git diff --quiet dd5ccf3f1 HEAD --` over `crates/feat-math/tests/fixtures`, `crates/jope-campaigns`, `crates/jope-engine`, `Cargo.toml`, `Cargo.lock`, and feat-math's `Cargo.toml`, `lib.rs`, `ops.rs`, `model.rs`, `math_expr.rs` and `evolve.rs` | 0 | unchanged, and again with `rust-toolchain.toml`, `.config`, `deploy/lambda`, `advanced_v3.rs` and `heads.rs` | 0.0 s |
| The 70 identity paths | `git diff --quiet` over the 70 paths of "Identity unchanged", from `dd5ccf3f1` and from `6e0ae2491` to HEAD, and from HEAD to the working tree | 0 | none differs, and none is a path this pull request changes (`model.rs` and `grammar.rs` are among the 70) | 0.1 s |
| Fixtures | `git diff --stat dd5ccf3f1 HEAD -- crates/feat-math/tests/fixtures/`, and `git status --porcelain` of it | 0 | both empty | 0.0 s |
| `kernel/mod.rs` | `git diff -U0 dd5ccf3f1 HEAD -- crates/feat-math/src/kernel/mod.rs` | 0 | one added line, `pub mod width;`, after `pub mod telemetry;` | 0.0 s |
| The series' files | `git diff --quiet c8e36e260 HEAD --` over `kernel/width.rs`, `advanced_v3/transforms/mod.rs` and `tests/fitted_width.rs` | 0 | byte-identical to the reviewed sources | 0.0 s |
| References | `git grep` under `crates/`, `deploy/ci` and `tools/` for the changed paths, and for `include_str!` or `include_bytes!` of anything under `kernel/` or `advanced_v3/` | 0 | the paths: the captured file list above, and `aggressive.rs` and the `advanced-v3` manifests under a `vendor/` path; the includes: only `advanced_v3_conformance.rs` reading its test fixture `fixtures/advanced_v3/csv-runtime-v1.json`, no source | 0.2 s |
| Subjects | `git log --format=%s dd5ccf3f1..HEAD`, searched with the ip-scrub pattern and the plan's additions | 0 | no subject matches | 0.0 s |
| Authors and trailers | `git log --format='%an <%ae>\|%cn <%ce>' dd5ccf3f1..HEAD`, and the last line of every message | 0 | one identity, author and committer; every message ends with the trailer | 0.1 s |
| Whitespace | `git diff --check dd5ccf3f1 HEAD` | 0 | clean | 0.0 s |

With these records in the working tree (`b` in the replay's evidence, and `b2` after their last
edit), the checks that read them ran, each with exit 0:

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 290 targets, 0 unassigned or invalid | 0.9 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 326 items, 197 of them regression, 33 of those in `fitting` | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 0.3 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped; its `records` tests read the live coverage map and inventories | 14.7 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python`, then with `--update` | 0, 0 | consistent: Rust 3,905 kept, Python 172; the update changes no byte | 0.2 s, 0.1 s |
| The feat-math section against a live listing | `cargo nextest list --locked -P main -p feat-math --message-format json`, then the re-list script with `--check` | 0 | equal | 0.7 s |
| The two record items against the new main | the re-apply `jq` on `dd5ccf3f1`'s `deploy/ci/coverage.json` and findings inventory, then `cmp` | 0 | both equal the files here, and each adds exactly the lines `ebf647557` added (16 and 6) | 0.2 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok: coverage 290 targets; gitleaks found no leaks (the 731 commits reachable from `524c79701`, 728 scanned); the repository layout with 0 findings | 32.5 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.9 s |
| Whitespace | `git diff --check dd5ccf3f1` | 0 | clean | 0.0 s |
| Placeholders | `git grep -nI` for the number placeholder, and `grep` for template braces and draft markers in the three records files | 0 | none | 0.0 s |

## Decisions taken while implementing

1. **Where the code lives.** `kernel/width.rs` is outside the ADR-0004 hash list, the lineage
   sources and the frozen pins, so no identity moves. `try_predict` is an inherent method of
   `Model` declared in that module. `ops.rs` keeps `op_append` and `finish` private, so the module
   has its own exhaustive `append` extractor and restates `finish`'s arithmetic: the generated
   columns, after the input's when the stage appends, capped at 512, and 1 for an empty row.
2. **`try_predict` is `predict` behind a static walk.** `check_input_width(model, width)` walks the
   stages with the width each receives, from the fitted state alone, and never evaluates a stage;
   if it accepts, `try_predict` returns `self.predict(x)` itself. The bits on an accepted input are
   therefore `predict`'s by construction, not by reimplementation. The walk's widths are checked
   against a stage-by-stage replay on every corpus program and every variant (below). With no rows
   there is no width, and `try_predict` returns `predict(x)`.
3. **The error.** The brief's `WidthMismatch { stage, expected, got }` is kept, with a fourth field,
   `kind`, because three failures that are not a stage width need their own report:
   `RaggedInput { row }` (stage 0, `expected` is row 0's width), `StageInput`, `HeadInput`
   (`stage` is the number of fitted stages), `InconsistentState { reason }` and `OutputOverflow`.
   `Display` gives, for example, `stage 0 needs at least 8 input columns, got 6`.
4. **Fitted state that disagrees with itself.** A width class alone cannot keep `apply_op` from
   panicking on state that no fit produces: a `winsor_clip` bound above its upper bound makes
   `f64::clamp` panic, a `bin_onehot` with no bins or an empty `target_bin_encode` table
   underflows, and an `impute` indicator can lie outside the fitted columns. `check_fitted_state`
   is one exhaustive match with the rules that make every admitted width safe (table below). It
   generalises the brief's `pair_ratio` rule (`eps.len() == pairs.len()`), which by itself never
   panics in Rust (`zip` truncates) but is undefined behaviour in the fitted C++. State produced
   by `fit_op` always passes: every variant, fitted, is checked in the tests.
5. **Heads.** No head panics on width: they read columns with `take`, `min` or bounds-checked
   routing. `required_head_width` states the width the head's state reveals: `Exactly` for the
   heads that keep statistics of every training column (ridge, elastic net, krr, both gpr heads,
   ksmooth, the ensemble's ridge, the hurdle gate, the stack's members, a portfolio's padded
   width); a lower bound for the tree heads, whose `n_feat` is capped (by `max_features`, 256 by
   default), and for `ridge_gcv`, which may screen columns; nothing for a constant fallback. The
   corpus champion's `histgbm` sees 512 columns with `n_feat` 256, which an `Exactly(n_feat)` rule
   would have refused. Heads' own tables are trusted.
6. **`roll` and the dot-product stages.** `roll` is `AtLeast(1)`, which says in the type what the
   brief's "Any, with a caveat" says in prose: `k % d` divides by the width. The dot-product
   stages (`rff_cos`, `rff_sin`, `rand_proj`, `oblique_hinge`, `landmark_dist`, `nystrom`,
   `svd_project`) are `Exactly` by meaning, as the brief says, although `predict` truncates them
   silently.
7. **Closed-form and portfolio models.** A closed-form model accepts any width: where its
   expression reads a column beyond the row it evaluates to 0.0, and the corpus relies on that
   (`symbolic_x`). `validate_columns` is therefore not used. An explicit portfolio pads or truncates
   rows to its fitted width, which `try_predict` requires as it does of a portfolio head; its
   members are not checked. A ragged input is refused for every model.
8. **`output_width`.** It returns `kernel::Result`, like `next_width`: `Shape` when the state is
   inconsistent or the width is not admitted, `Resource` when the width overflows `usize` or is
   wider than any `Vec<f64>` can be (a tampered `degree`, `bins` or `k`). A size that fits a `Vec`
   but not memory still aborts, in `predict` as in `try_predict`; that is outside the contract.
9. **The schema-bound boundary.** `legacy_to_block` checks `check_fitted_state`,
   `required_input_width` against the block's width, `roles.len()`, which `FeatureBlock::validate`
   makes every row's width, and `output_width`, which refuses a fitted size whose output row no
   `Vec` can hold (a tampered `degree`, `bins` or `k`, where `apply_op_generated` panicked with a
   capacity overflow; the review found that the implementation left this check out). The message
   names the stage and the widths: `legacy stage select_corr needs at least 100 input columns, but
   the feature block has 2`. It has no stage index, because `legacy_to_block` has none and the
   brief does not authorise context at its callers. The `corr_screen` and `pls_project` arms of
   `apply_block` check exactly what they index: every fitted column index below the block's width,
   and every weight no longer than the block and the fitted means. A block wider than 512 columns
   keeps working (`standardize` over 1,024 columns requires `Exactly(1024)`; nothing is capped
   here). The guard also runs at fit time, through `fit_v3_op`'s default arm, where the state was
   just fitted on the block.

   **A table with no rows has no width to compare.** Nothing is read, and the generated-width arm
   of `legacy_generated_to_block`, which finishes `rank`, `cheb`, `hinge`, `pair_ratio` and most
   other stages, takes its width from the first row, so after such a stage the block has no
   column, or only the appended input. The implementation compared widths without rows as well,
   and the review found that this turned `predict_table`'s empty result into an error, for
   example for `(rank) (standardize) (ridge)` and in `predict_multiclass_table`. Without rows the
   legacy check now checks the fitted state and, beyond it, only the fitted column indices at
   which finishing the stage looks the block up (`select_corr`'s indices, and `impute`'s
   indicators when it does not append), where finishing panicked. The `pls_project` check checks
   nothing without rows, since nothing is read. The `corr_screen` check is unchanged, because it
   refuses exactly where finishing its selection panicked, with rows or without. A probe of 23
   programs (20 regression, 3 multiclass), each fitted with `fit_table` and predicted on a table
   with no rows, was built against the base, the reviewed head `ebf647557` and the fix. The
   reviewed head had turned 10 of the base's 14 empty results into errors. The fix gives the
   base's outcome for all 23, messages included, except the 6 that panicked at the base, which
   are errors now. On one row, all 23 agree across the three builds.
10. **Left as they are.** `V3Op::apply` is infallible (`Rows`, not `Result`), so its
    `corr_screen` (`:114`) and `pls_project` (`:131`) reads are recorded as a gap rather than
    changed; production reaches `apply` only through the pls head, with state fitted there. The
    `piecewise_linear` arm of `apply_block` still panics on edges fitted for more columns than the
    block has: `finish_generated_dense` gathers the missing mask at every fitted edge column. The
    brief does not name it, because its facts said the stage uses `zip` and never panics, which
    is true of `piecewise_row` but not of that gather. The review found it; guarding it would
    change an evaluator arm that the brief does not authorise, so it is recorded as a gap, and a
    test pins the panic. The fold-local target encoder at `:418` fits and applies on rows of one
    block. `BoundPredictor::Legacy` still calls `Model::predict`.

## The width classes

| Class | Stages | Read from | `predict` on a mismatch |
|---|---|---|---|
| `Exactly(n)` | `standardize`, `robust_scale`, `rank`, `blom`, `quantile_bins`, `bin_onehot`, `target_bin_encode`, `hinge`, `spline`, `winsor_clip`, `impute` | the per-column state's length (`means`, `medians`, `sorted_cols`, `edges`, `knots`, `lo`, `fill`) | wider panics; narrower is silently misread, except `impute` with indicators, which panics (`impute` panics wider only on a zero cell) |
| `Exactly(n)`, by meaning | `rff_cos`, `rff_sin`, `rand_proj`, `oblique_hinge`, `landmark_dist`, `nystrom`, `svd_project` | the first weight or landmark row, `means` for `svd_project`; `Any` when there is none | the dot product is truncated, no panic |
| `AtLeast(max + 1)` | `select_corr`, `poly_interact`, `pair_prod`, `pair_ratio` | the fitted column indices; `Any` when there is none | narrower panics; wider is read at the same indices |
| `AtLeast(1)` | `roll` | | an empty row divides by zero |
| `Any` | `cheb`, `row_stats`, `sigmoid`, `step`, `norm_row`, `polar`, `lag`, `fft_mag`, `row_spectral`, `row_diff_stats`, `symlog`, `map`, `derive` | | no panic |

`check_fitted_state`'s rules, each a panic or a misread it rules out:

| Stage | Rule |
|---|---|
| `standardize`, `robust_scale` | `scales` as long as `means` (`medians`) |
| `winsor_clip` | `hi` as long as `lo`, and every `lo[j] <= hi[j]` (so neither is NaN) |
| `bin_onehot` | `bins > 0` |
| `hinge` | at least one knot column |
| `spline` | at least two knots per column |
| `target_bin_encode` | `enc` as long as `edges`, and no empty encoding |
| `rff_cos`, `rff_sin`, `oblique_hinge` | one offset or threshold per weight row, and equal weight rows |
| `rand_proj`, `landmark_dist`, `nystrom` | equal rows |
| `svd_project` | every component as long as `means` |
| `impute` | every indicator column below `fill.len()` |
| `pair_ratio` | one guard per pair |

## Tests (`crates/feat-math/tests/fitted_width.rs`: 18 run, 0 ignored)

The implementation added 16 tests; the review round added two (tables with no rows, and the
recorded `piecewise_linear` gap) and tightened two (the heads, and the schema-bound boundary).

**Every stage.** The 36 `FittedOp` variants, each fitted on 8 columns with and without `append`
(72 fitted stages): each passes `check_fitted_state` and has the class tabled above; at each of the
widths 0, 1, 2, 3, 5, 7, 8, 9, 16 and 600 that its class admits, every row `apply_op` returns has
`output_width`'s width (348 cases), and at the others `output_width` is a `Shape` error. Twelve more
cases pin the cap (900, 800 and 640 columns become 512) and the padding (an empty output row becomes
one column, and appending nothing keeps the input), and a spline with a constant column has
1 + 4 + 4 columns.

**The workplan's cases**, with their exact results:

| Case | `try_predict` | `predict` |
|---|---|---|
| `select_corr k=3`, fitted on 8 columns, keeps `[1, 5, 7]`; 6 columns | `stage 0 needs at least 8 input columns, got 6` | panics |
| the same at 8 columns, and at 9 | `predict`'s bits | |
| `impute`, fitted on 4 columns; 5 columns with a zero in column 4 | `stage 0 needs exactly 4 input columns, got 5` | panics |
| the same without a zero | the same refusal | returns, column 4 passed through unfitted |

**Their siblings**, each refused at stage 0 one column off its requirement, and bit-identical to
`predict` at the fitted width:

| Stage | Fitted on | Refused at | `predict` there |
|---|---|---|---|
| `pair_ratio topk=4`, `pair_prod topk=4` | 6 | one below the largest fitted index + 1 | panics |
| `poly_interact` | 4 | 3 | panics |
| `select_corr k=3` | 8 | one below the largest fitted index + 1 | panics |
| `spline knots=4`, `standardize`, `winsor_clip`, `rank`, `bin_onehot bins=3`, `target_bin_encode`, `hinge` | 4 | 5 | panics |
| `impute` (no indicators) | 4 | 5 | panics (a zero in column 4) |
| `impute add_indicator=true` (every column has a zero, so every column has an indicator) | 5 | 4 | panics |
| `standardize` | 4 | 3 | misreads, no panic |
| `rff_cos n=6`, `svd_project k=2` | 4 | 5 | misreads, no panic |
| `nystrom k=6` | 4 | 3 | misreads, no panic |
| `roll k=2` | 3 | 0 (empty rows) | panics (divides by zero) |

**Heads.** Fifteen heads fitted on 3 columns. `predict` never panics at 2, 3 or 4 columns, and
`try_predict` either returns its bits or refuses at the head, as its class says: `Exactly(3)` for
`ridge`, `elasticnet`, `ensemble`, `krr`, `gpr`, `gpr kernel=rq+linear`, `ksmooth`, `hurdle2` and
`stack`; `AtLeast(3)` for `ridge_gcv`, `histgbm` and `histgbm2`; `Any` for the constant fallbacks
of `histgbm2` on 8 rows, `histgbm` on 3 rows and `hurdle2` on 8 rows. Every refusal must be one
the class makes: a refused width is one the class does not admit (the review added this, since
`hurdle2`, which the compatibility corpus does not cover, could otherwise have been refused at its
own fitted width unnoticed). A `histgbm2` fitted on 300 columns requires `AtLeast(256)`: 300
columns give `predict`'s bits, 255 are refused.

**Portfolios, closed-form models, ragged rows and empty inputs.** A mean-only portfolio accepts
any width. One padded to 3 columns, alone and as a head, accepts 3 and refuses 5 at the head, where
`predict` pads without panicking. A closed-form model accepts 0, 1, 3 and 9 columns with
`predict`'s bits. A row of the wrong width is refused as `ragged input: row 2 has 2 columns, every
row needs exactly 3`, for a pipeline and a closed-form model. No rows give `predict`'s empty result.

**Inconsistent state.** Twenty hand edits of fitted stages, at least one per rule of
`check_fitted_state`. Each fails that check with its reason, `output_width` is a `Shape` error, and
`try_predict` refuses with `InconsistentState` at the width the edited stage claims. At that width
`predict` panics on 15 of them; on the other 5 (unequal weight or landmark rows, a missing
`pair_ratio` guard) it misreads without panicking.

**Overflow.** `cheb` with `degree = usize::MAX` has no output width at 2 columns (`Resource`), 1 at
0 columns, and `try_predict` refuses with `OutputOverflow`; `bin_onehot` with `bins = usize::MAX /
2` has none at 3 columns. `fft_mag` with `k = 2^40` or `k` = the longest `Vec<f64>` is capped at
512; one more, or appending the longest to one input column, or `usize::MAX`, is refused. None of
these stages is applied.

**The compatibility corpus.** The 55 programs of `emitted_bytes.v1.json`, read at run time; 54
fit on the first 64 rows of `toy_data(80, 5, 42)`, and the portfolio is skipped, as compat_freeze
skips it. On the 16 held-out rows `try_predict` equals `predict` bit for bit, and both equal the
predictions recorded in `model_serde.v1.json`. The static widths match a stage-by-stage replay:
the champion's are `[5, 10, 70, 70, 70, 512, 512]` (`poly_interact` appends 512 pairs to 70
columns, capped at 512) and `empty_stage_output`'s `[5, 1, 1]` (one column leaves `pair_ratio` no
pair). At 0, 1, 4, 6 and 9 columns, 270 calls never panic: 65 return `predict`'s bits and 205 are
refused. The 5 stored models, deserialized, give `try_predict` = `predict` too.

**The schema-bound boundary.** Eight models fitted with `fit_table` on a two-column table and
edited through `serde_json` (`validate_binding` does not hash the stages, so each edit loads). Each
`predict_table` returns the exact error, and none panics:

| Edit | Error |
|---|---|
| `select_corr` indices `[0, 99]` | `legacy stage select_corr needs at least 100 input columns, but the feature block has 2` |
| an `impute` indicator 5 | `legacy stage impute has inconsistent fitted state: an indicator column is not a fitted column` |
| `standardize` scales `[1.0]` | `legacy stage standardize has inconsistent fitted state: means and scales differ in length` |
| `corr_screen` indices `[0, 7]` | `corr_screen reads column 7, but the feature block has 2 columns` |
| a `pls_project` weight of 4 entries | `pls_project weight spans 4 columns, but the feature block has 2 and the fitted means 2` |
| `cheb` `degree` 2^61 | `legacy stage cheb output width overflows at 2 input columns` |
| `bin_onehot` `bins` 2^61 | `legacy stage bin_onehot output width overflows at 2 input columns` |
| `fft_mag` `k` 2^61 | `legacy stage fft_mag output width overflows at 2 input columns` |

The last three were added in the review round. Without the overflow check, the `cheb` case panicked
at `alloc/src/raw_vec/mod.rs:28:5` (a capacity overflow).

**Tables with no rows.** On a table with the two columns and no rows, eight regression pipelines
return the empty result, as at the base: `(rank) (standardize)`, `(quantile_bins) (winsor_clip)`,
`(rank append=true) (standardize)`, `(cheb degree=3) (standardize)`, `(pair_ratio topk=4)
(standardize)`, `(hinge knots=3) (impute)` and `(rank append=true) (pls_project k=2)`, each before
`(ridge)`, and `(select_corr k=2) (impute add_indicator=true) (ridge)`. So do two multiclass
pipelines through `predict_multiclass_table`, `(rank) (standardize)` and `(rank append=true)
(standardize)` before `(multiclass_ridge)`. Three pipelines that panicked at the base return an
error: `(rank) (select_corr k=2)` and `(hinge knots=3) (impute add_indicator=true)` give `legacy
stage … reads column N, but the feature block has 0 columns`, and `(hinge) (corr_screen k=32)`
gives `corr_screen reads column 0, but the feature block has 0 columns`.

**The recorded gap.** A `piecewise_linear` model fitted on the two columns, given one more fitted
edge column through `serde_json`, still panics in `predict_table` (decision 10).

**Messages.** The `Display` of each of the five kinds.

## Exhaustiveness

At the pre-replay `135cd5b4c` (`6037be4f2` after the replay; this section keeps the pre-replay
ids), a scratch variant `ScratchWidthProbe { append: bool }` was added at the end of
`FittedOp` in `ops.rs` and `cargo check -p feat-math --locked` run. It failed with 11 E0004 errors,
five of them in `kernel/width.rs`: `required_input_width` (`:160`), `check_fitted_state` (`:211`),
the stage name (`:427`), the append extractor (`:469`) and the generated width (`:513`). The other
six are the existing exhaustive matches: `ops.rs:903` and `:1255`, `transforms/mod.rs:815`, and
`fitted_cpp.rs:111`, `:635` and `:918`. `git checkout -- crates/feat-math/src/ops.rs` restored the
file: sha256 `9a5dcb94…` before and after, `git diff --quiet` against the commit and the base both
exit 0, and the tree was clean. The scratch variant was never committed. The same check, made
earlier while implementing, gave the same eleven errors; `925ec2a32` moves no line.

The review round changes no match arm in `kernel/width.rs`. `9105c6ff5` adds three documentation
lines above `check_fitted_state`, so at `c8e36e260` the five matches are at `:160`, `:214`,
`:430`, `:472` and `:516`. The round's one new match over `FittedOp`, the fitted indices that
finishing a stage looks up without rows (`transforms/mod.rs:928`), has a wildcard arm on purpose:
it mirrors the arms of `legacy_generated_to_block`, where only `select_corr` and `impute` look the
block up by a fitted index, and a new stage that did would need its own arm there as well.

The check was not repeated after the replay: it edits `ops.rs`, an ADR-0004 identity input. The
replay leaves `kernel/width.rs` and `transforms/mod.rs` byte-identical to `c8e36e260`, so the five
matches and their line numbers hold at `524c79701`. #130 added one more exhaustive match over
`FittedOp` with no wildcard arm, `appends` in `kernel/telemetry.rs` (`:485`), which the same
scratch variant would stop at too; #131's `kernel/scope.rs` matches on no `FittedOp`.

## Perturbations caught

Each was applied with `sed` to the sources as they are at the pre-replay `135cd5b4c` (then
uncommitted, and different only in two documentation comments of `width.rs`), `fitted_width` ran,
and the file was restored from a copy; the two files were byte-identical to their copies
afterwards. The review round's own perturbations are in its section below. Neither set was re-run
after the replay, which leaves the perturbed sources and `fitted_width.rs` byte-identical.

| Perturbation | Failing tests |
|---|---|
| `select_corr` classified `Any` | 5: the corpus, every stage, the siblings, the schema-bound boundary, the workplan's `select_corr` case |
| no 512-column cap in the width | 4: the corpus (the champion's widths), every stage, the cap and padding, overflow (`fft_mag` capped at 512) |
| no padding of an empty row | 4: the corpus (`empty_stage_output`'s widths), every stage, the cap and padding, overflow (`cheb` at 0 columns) |
| `check_fitted_state` always passes | 2: inconsistent state, the schema-bound boundary |
| the ridge head classified `Any` | 1: heads |
| `legacy_to_block` unguarded | 1: the schema-bound boundary |
| the spline's constant-column test never true | 1: the cap and padding |
| ragged rows not checked | 1: ragged rows and empty inputs |
| `roll` classified `Any` | 2: every stage, the siblings |
| the `impute` indicator rule off by one | 1: inconsistent state |

## Identity unchanged

- **compat_freeze** passes 12 of 12 in compare mode, twice in each round, and no fixture changes:
  `git diff --stat 6e0ae2491 HEAD -- crates/feat-math/tests/fixtures` was empty before the replay
  and `git diff --stat dd5ccf3f1 HEAD -- crates/feat-math/tests/fixtures` is empty after it, and
  `git status --porcelain -- crates/feat-math/tests/fixtures/compat_freeze` was empty after the
  runs. No regeneration mode was used, and `known-evaluator-identities.json` gains no row.
- **The ADR-0004 evaluator identity does not move.** None of its 38 inputs changes
  (`evaluator_identity.v1.json` `inputs`: the 31 feat-math sources that
  `crates/jope-campaigns/src/evaluation_v2.rs` hashes, 5 jope-campaigns files,
  `crates/feat-math/Cargo.toml` and `Cargo.lock`). Neither do the 14 lineage sources
  (`lineage_sources.v1.json`, among them `crates/feat-math/src/model.rs` and
  `crates/jope-engine/src/grammar.rs`) or the 20 frozen pins (`frozen_pins.v1.json`, among them
  `registry.rs` and `cpp/fitted_runtime.cpp`). `git diff --quiet 6e0ae2491 HEAD --` over the 70
  distinct paths of those lists, read from the three fixtures with `jq`, exits 0, and after the
  replay so does `git diff --quiet dd5ccf3f1 HEAD --` over the same paths (see "Replay").
- **The changed sources** are `kernel/width.rs` (new), `kernel/mod.rs` and
  `advanced_v3/transforms/mod.rs`. None is in any of those lists, and nothing in `crates/` or
  `tools/` includes or hashes them. Three other places name `advanced_v3/transforms/mod.rs`: a
  benchmark receipt in `jope-campaigns/src/aggressive.rs`, which reads it under a `vendor/`
  directory that does not exist in this tree; the historical manifests under
  `crates/jope-campaigns/manifests/advanced-v3/`, which name the same old `vendor/` path; and a
  captured file list in a jope-records replay fixture (`python-parity/coverage/files.txt`), which
  names it and `kernel/mod.rs` as paths (see "Replay").
  `Cargo.toml`, `Cargo.lock`, `crates/feat-math/Cargo.toml`, `rust-toolchain.toml` and everything
  under `deploy/lambda` are unchanged. The review round changes only `kernel/width.rs`,
  `advanced_v3/transforms/mod.rs` and the test file, and its gate at `c8e36e260` repeats every
  check of this section, as the replay's gate does at `524c79701` and at the head.
- **Serialized shapes do not change.** No field, variant or derive is added to `FittedOp`,
  `V3Op`, `BoundModel` or any head, so `model_serde.v1.json` and the Lambda workers'
  `serde_json(BoundModel)` hashes are unaffected.
- One thing does change with any rebuild: jope-engine's teaching, corpus and query identities also
  hash the running executable (`/proc/self/exe`; see the note in `lineage_sources.v1.json`), so a
  new binary has new runtime identities. That is true of every feat-math change.

## Commands and results

Before the replay, at `135cd5b4c` (`6037be4f2` after the replay), with a clean tree. Where a step
ran twice, both exit codes and wall times are given. The review round's commands follow in "The
review round", and those after the replay in "Replay".

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 7.1 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 8.5 s |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 497 passed in 23 binaries; 10 skipped (6 ignored, 4 by the pr filter) | 72.0 s, 66.1 s |
| Named targets | `cargo nextest run --locked -P pr -p feat-math --test fitted_width --test compat_freeze --test ip_scrub --test legacy_v2_freeze --test portable_parity --test advanced_v3_conformance --test advanced_v3_parity --test advanced_v3_property --test advanced_v3_fitted_cpp` | 0, 0 | 61 passed (16, 12, 6, 3, 1, 7, 1, 2, 13); 1 skipped | 46.3 s, 46.1 s |
| The new tests' output | `cargo nextest run --locked -P pr -p feat-math --test fitted_width --no-capture` | 0, 0 | 16 passed; the `STAGES` and `CORPUS` lines byte-identical across the runs (sorted, sha256 `24a74b77dcc26027e94ab12da9aa61b3c08555d493126a1634c68643c12b018e`) | 2.0 s, 1.9 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed, compare mode | 11.5 s, 11.5 s |
| Fixtures | `git status --porcelain -- crates/feat-math/tests/fixtures/compat_freeze`; `git diff --stat 6e0ae2491 HEAD -- crates/feat-math/tests/fixtures` | 0 | both empty after the runs | 0.0 s |
| Identity inputs | `git diff --quiet 6e0ae2491 HEAD --` over the 70 paths above | 0 | none differs | 0.0 s |
| Frozen files | `git diff --quiet 6e0ae2491 HEAD --` over `Cargo.toml`, `Cargo.lock`, `crates/feat-math/Cargo.toml`, `rust-toolchain.toml`, `deploy/lambda`, feat-math's `ops.rs`, `model.rs`, `lib.rs`, `advanced_v3.rs` and `heads.rs`, and `crates/jope-engine/src/grammar.rs` | 0 | none differs; the changed files are the four sources above | 0.0 s |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (csv 40, regression-ops 2, regression 5, search engine 2, worker 3, monitor 14); cargo-deny skipped (0.19.8 is installed, and the lane asserts 0.20.2), so that step is not a pass | 51.6 s |
| Emitters | `EMITTERS_OUT=<a new directory> bash deploy/ci/emitters.sh` | 0 | the champion's Python and Rust programs, and the portable program's seven builds (Python, Rust, g++ and clang++ at C++17 and C++20, JavaScript), agree bit for bit on 34 rows; every expectation digest matches | 16.7 s |
| Consumers of `predict_table` | `cargo nextest run --locked -P main -p jope-parallel -p jope-dime-regressor --test-threads 32` | 0 | 1,772 passed in 141 binaries; 25 skipped | 239.7 s with the build |
| Whitespace | `git diff --check 6e0ae2491 HEAD` | 0 | clean | 0.0 s |

`925ec2a32` (`75f08888a` after the replay) changes two documentation comments of
`kernel/width.rs` and nothing else: every changed line is a `///` comment, and no other file
differs from `135cd5b4c`. At it, with a clean tree, the steps its change could reach ran again:

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 7.9 s |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0 | 497 passed; 10 skipped | 69.7 s |
| The new tests' output | `cargo nextest run --locked -P pr -p feat-math --test fitted_width --no-capture` | 0 | 16 passed; the same `STAGES` and `CORPUS` lines | 2.0 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0 | 12 passed, compare mode; the fixtures unchanged | 11.6 s |
| Whitespace | `git diff --check 6e0ae2491 HEAD` | 0 | clean | 0.0 s |

With the first records, those of `ebf647557`, in the working tree (the sources as at `925ec2a32`):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Coverage discovery | `cargo run -q --locked -p jope-records -- coverage discover` | 0 | lists `target:feat-math/test/fitted_width` with exactly the committed discovery | 0.3 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.3 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items, 195 of them regression, 32 of those in `fitting` | 0.2 s |
| jope-records unit tests | `cargo test -q --locked -p jope-records --lib -- coverage:: scope_check:: parity:: readme::` | 0 | 24 passed | 0.3 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped; its `records` tests read the live coverage map and inventories | 13.9 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent; the Rust half kept at 3,870 tests, Python 172 | 0.0 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.4 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok: coverage 288 targets, gitleaks over the 711 commits reachable from `925ec2a32`, the repository layout with 0 findings | 19.6 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| Whitespace | `git diff --check`, and from `6e0ae2491` to `HEAD` | 0 | clean | 0.0 s |

The feat-math part of the parity inventory is `cargo nextest list --locked -P main -p feat-math
--message-format json` at `925ec2a32`, written with `jq` into the feat-math section and the totals
only: with those removed, the file is unchanged, and every binary but the new one is listed as
before. feat-math goes from 491 to 507 tests (6 ignored) in 23 binaries instead of 22, and the Rust
total from 3,854 to 3,870 (65 ignored). The review round re-lists it the same way at `c8e36e260`:
`fitted_width` goes from 16 to 18 tests, feat-math to 509 and the Rust total to 3,872, and nothing
else in the file changes. These are the counts before the replay; after it, the same re-listing
on #131's inventory gives feat-math 524 to 542 and the Rust total 3,887 to 3,905 (see "Replay").

## The review round

Reviewers read the pre-replay `ebf647557` for correctness, for regressions and for the
repository's rules. Their findings, and what this branch did with each (this section keeps the
pre-replay ids; "Replay" maps them to the replayed commits):

| Finding | Disposition |
|---|---|
| NR-1, blocking: a table with no rows turned `predict_table`'s empty result into a width error after a stage that takes its generated width from the first row | Fixed in `a9c0c4766` (decision 9). The same regression hit `pls_project` after an appending stage, which the finding did not list; that check skips tables with no rows too. |
| F2: the legacy check left out the output-width overflow that `try_predict` refuses, so `predict_table` still panicked with a capacity overflow | Fixed in `9105c6ff5`; `check_fitted_state`'s documentation and the pointer for PR 07 corrected. |
| F1: the `piecewise_linear` arm of `apply_block` panics on edges fitted for more columns than the block has | Recorded as a gap (decision 10), with a pinning test in `c8e36e260`. Guarding it would change an evaluator arm that the brief does not authorise. It stays open ("Still open"). |
| R1: the CHANGELOG line promised errors instead of panics at the whole schema-bound boundary | Narrowed to the legacy, `corr_screen` and `pls_project` stages. The subject of `5b6a089ec` said the same; the replay rewords it to name those stages (`827fda6cf`, see "Replay"). |
| Nit: the head test did not require a refusal to be one the requirement makes | Fixed in `c9983d2e1`. |
| Nits: the number placeholders, and the readback of #129 now that #130 has merged | Done in the replay, which regenerates the records commits onto #131, the last merged pull request: `MASTER_PLAN.md` carries #131's readback, and #132 stands wherever the placeholder stood. |

The gate, at the pre-replay `c8e36e260` (`524c79701` after the replay) with a clean tree:

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 7.1 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 10.6 s |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 499 passed in 23 binaries; 10 skipped (6 ignored, 4 by the pr filter) | 73.2 s, 61.6 s |
| Named targets | the implementation's nine, with `advanced_v3_multiclass` and `kernel_pipeline` | 0, 0 | 72 passed (`fitted_width` 18, `compat_freeze` 12, `ip_scrub` 6, `legacy_v2_freeze` 3, `portable_parity` 1, `advanced_v3_conformance` 7, `advanced_v3_parity` 1, `advanced_v3_property` 2, `advanced_v3_fitted_cpp` 13, `advanced_v3_multiclass` 6, `kernel_pipeline` 3); 1 skipped | 45.6 s, 45.0 s |
| The new tests' output | `cargo nextest run --locked -P pr -p feat-math --test fitted_width --no-capture` | 0, 0 | 18 passed; the `STAGES` and `CORPUS` lines byte-identical across the runs and to the implementation's (sorted, sha256 `24a74b77dcc26027e94ab12da9aa61b3c08555d493126a1634c68643c12b018e`) | 2.0 s, 1.9 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed, compare mode | 12.0 s, 13.4 s |
| Fixtures | `git status --porcelain -- crates/feat-math/tests/fixtures/compat_freeze`; `git diff --stat 6e0ae2491 HEAD -- crates/feat-math/tests/fixtures` | 0 | both empty after the runs | 0.0 s |
| Identity inputs | `git diff --quiet 6e0ae2491 HEAD --` over the 70 paths | 0 | none differs | 0.0 s |
| Frozen files | as for the implementation | 0 | none differs | 0.0 s |
| Lambda workspaces | `bash deploy/ci/lambda.sh`, twice | 0, 0 | 6 built and tested with `--locked`, 66 tests passed each time; cargo-deny skipped (0.19.8 installed, 0.20.2 asserted), not a pass | 54.0 s, 7.8 s |
| Emitters | `EMITTERS_OUT=<a new directory> bash deploy/ci/emitters.sh` | 0 | the champion's two programs and the portable program's seven builds agree bit for bit on 34 rows; every expectation digest matches | 17.5 s |
| Consumers of `predict_table` | `cargo nextest run --locked -P main -p jope-parallel -p jope-dime-regressor --test-threads 32`, twice | 0, 0 | 1,772 passed in 141 binaries, 25 skipped, each time | 225.7 s with the build, 195.7 s |
| Whitespace | `git diff --check 6e0ae2491 HEAD` | 0 | clean | 0.0 s |
| Clean tree | `git status --porcelain` after the gate | 0 | empty | 0.0 s |

Perturbations of the lines the round added, each applied with `sed` at `c8e36e260`, run against
`fitted_width`, and undone from a copy (`transforms/mod.rs` byte-identical afterwards):

| Perturbation | Failing tests |
|---|---|
| without rows, the legacy check compares widths again | 1: tables with no rows |
| without rows, the `pls_project` check compares widths again | 1: tables with no rows |
| no output-width overflow check | 1: the schema-bound boundary |
| without rows, `select_corr`'s indices are not checked | 1: tables with no rows (the pipeline panics) |

The zero-row probe (decision 9) is `zero_rows_fixer.rs` in the evidence: 23 programs, each fitted
with `fit_table` on the two-column table of the tests and predicted on one row and on a table with
no rows, compiled with rustc 1.97.1 against three feat-math builds: the base (built from
`6e0ae2491` by the no-regression reviewer), the reviewed head (`ebf647557`) and the fix
(`c8e36e260`).

| Outcome on a table with no rows | Base | Reviewed head | Fix |
|---|---|---|---|
| empty result | 14 | 4 | 14 |
| error | 3 | 19 | 9 |
| panic | 6 | 0 | 0 |

The fix's 9 errors are the base's 3, with the base's messages, and the 6 base panics. On one row,
all 23 programs give the same outcome in the three builds.

With the records of `dbd10526d` in the working tree (the sources as at `c8e36e260`):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| JSON form | `jq` round trips of the coverage map and both inventories | 0 | byte-identical | 0.0 s |
| Coverage discovery | `cargo run -q --locked -p jope-records -- coverage discover` | 0 | lists `target:feat-math/test/fitted_width` with exactly the committed discovery | 0.4 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.3 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items, 195 of them regression, 32 of those in `fitting` | 0.2 s |
| jope-records unit tests | `cargo test -q --locked -p jope-records --lib -- coverage:: scope_check:: parity:: readme::` | 0 | 24 passed | 0.3 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped | 14.1 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent; the Rust half kept at 3,872 tests, Python 172 | 0.0 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.4 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok: coverage 288 targets, gitleaks over 713 commits with no leak, the repository layout with 0 findings | 20.3 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| Whitespace | `git diff --check`, and from `6e0ae2491` to `HEAD` | 0 | clean | 0.0 s |
| Unchanged maps | `git diff --quiet ebf647557 -- deploy/ci/coverage.json docs/internal/core-ip/findings/test-inventory.json` | 0 | both as the first records left them | 0.0 s |

Evidence: `~/jope-evidence/kernel2/pr06/review-20260929T195726Z/` (the probe with its three logs
and comparison, the gate, the perturbations, the records checks, the feat-math listing, the commit
messages and the scripts, with `SHA256SUMS`). Every command of this gate but the perturbations and
the probe ran again after the replay (see "Replay").

## Still open

- **Hosted CI.** `crates/feat-math/**` and `deploy/ci/coverage.json` select every lane.
- **PR 07**, the fitted-C++ runtime revision 3: the C++ half of WP-1.7, which bounds-checks every
  fitted index in the emitted C++. Its load-time check can reuse `output_width`, which applies
  `check_fitted_state`, the width requirement and the overflow check together; the two separate
  functions leave the overflow out. A table with no rows has no width to compare (decision 9).
- **The kept gap F1: the `piecewise_linear` arm.** The V3 `piecewise_linear` arm of `apply_block`
  still panics on inconsistent state, edges fitted for more columns than the feature block has
  (decision 10): `finish_generated_dense` gathers the missing mask at every fitted edge column.
  `schema_bound_piecewise_linear_with_extra_edge_columns_still_panics` pins the panic, so the
  change that closes the gap fails it and updates it. The fix is scheduled for a follow-up: item 7
  of the PR 07 brief guards the arm as this pull request guards the legacy, `corr_screen` and
  `pls_project` stages, keeps the no-rows rule, and turns that test into one that asserts the
  error.
- **Other gaps kept on purpose.** `V3Op::apply` is infallible, so its `corr_screen` and
  `pls_project` reads (`transforms/mod.rs:114` and `:131`) are unchanged; production reaches it
  only through the pls head, with state that head fitted. `BoundPredictor::Legacy` (the models of
  `fit_legacy_numeric_bound`, and closed-form or portfolio programs fitted with `fit_table`) still
  calls `Model::predict`, which panics on inconsistent stage state. A head's own tables, a
  portfolio's members and a target transform's tables are trusted. A fitted size that fits a `Vec`
  but not memory aborts. WP-1.7's acceptance, "no panic reachable through `try_predict` or V3",
  therefore holds for `try_predict` with those tables trusted, and for the legacy, `corr_screen`
  and `pls_project` stages of a schema-bound pipeline; it does not hold for F1 or these gaps.
- **Nothing that ships calls `try_predict` yet.** Callers such as jope-engine's width guard before
  `predict` (`gp.rs`) keep their own checks.
- **cargo-deny** is 0.19.8 here and 0.20.2 in CI, so the Lambda script's deny step is skipped
  locally, not passed.

Evidence: `~/jope-evidence/kernel2/pr06/prepare-20260929T185801Z/` for the implementation (the
logs of its commands, the exhaustiveness check, the perturbation diffs and logs, and the feat-math
listing, with `SHA256SUMS`), `~/jope-evidence/kernel2/pr06/review-20260929T195726Z/` for the
review round (above), and `~/jope-evidence/kernel2/pr06/rebase-20260929T202502Z/` for the replay
(the logs of its commands, the conflict and its resolution, the reworded message, the feat-math
listings, the re-applied record items, the scripts and the drafts, with `SHA256SUMS`).
