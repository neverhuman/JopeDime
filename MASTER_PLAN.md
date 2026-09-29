# JopeDime current direction

**2026-09-29 kernel2 pull request 06 (#132): the fitted-width contract, with no identity change.**
Branch `kernel2/claude-m1-width`, prepared in the lane `.agent/worktrees/kernel2` on `origin/main`
`6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, after #129 merged, and replayed onto `origin/main`
`dd5ccf3f19c3314f5ef9bf34ccf362422650cce7` after #131 merged, before its first push. It carries the
Rust half of the kernel2 plan's WP-1.7 as the PR 06 brief cuts it; the fitted-C++ half (runtime
revision 3) is PR 07.
- **Readback of #131.** kernel2 PR 05 (#131, branch `kernel2/claude-m1-scope`) merged by rebase at
  2026-09-29T19:47:50Z at the exact reviewed head `5996071572709cc6e7616467ae8699b142d4d24f`.
  Approval 5357426190 (neverhumanbot) was submitted at 19:33:32Z, after the head push (no later
  than 19:22:57Z). The `pull_request` run 36618838019 and the `workflow_dispatch` run 36618836795
  were green on attempt 1, all nine jobs of each. main = `dd5ccf3f19c3314f5ef9bf34ccf362422650cce7`,
  and its tree equals the reviewed head's tree (`a0fd9b16…`). As for #128 to #130, gh-role warned
  that the account that opened the pull request also merged it, where `AGENTS.md` asks for a merger
  other than the author. Evidence: `~/jope-evidence/kernel2/pr131-merge-20260929T194531Z/`
  (`merge.log`, `SHA256SUMS`). Before it, #130 merged at 18:31:14Z (main `9146cbcc2`); #131's
  record carries that readback.
- **Contents.** `crates/feat-math/src/kernel/width.rs`: `required_input_width` gives the input
  width each fitted stage accepts, read from its fitted state (`WidthReq::{Any, Exactly,
  AtLeast}`); `check_fitted_state` checks that the state agrees with itself; `output_width` gives
  the width `apply_op` produces, its 512-column cap and empty-row padding included; and
  `required_head_width` gives the width a head's state reveals. `Model::try_predict` walks a model
  with them and returns a `WidthMismatch` (stage, expected, got, kind) before any stage runs, or
  else `Model::predict`'s own result, so an accepted input keeps its bits; `predict` is unchanged.
  In `advanced_v3/transforms/mod.rs`, `legacy_to_block` and the `corr_screen` and `pls_project`
  arms of `apply_block` return an error where they would panic on fitted state that does not fit
  the feature block, including a legacy stage whose output row no `Vec` can hold. A table with no
  rows compares no width, because a stage that generates columns takes their number from the first
  row: it keeps its result, and only a fitted column index at which finishing a stage looks the
  block up, where that panicked, is refused. Every match over the fitted stages and heads names
  each variant: a scratch variant of `FittedOp` stopped the crate compiling at the five new
  matches, and was removed.
- **Identity.** None moves. The changed sources (`kernel/width.rs`, new; `kernel/mod.rs`;
  `advanced_v3/transforms/mod.rs`) are outside the ADR-0004 hash list, the lineage sources and the
  frozen pins; no fixture changes, and `compat_freeze` passes in compare mode. After the replay,
  `git diff --quiet dd5ccf3f1 HEAD --` over the fixtures, `crates/jope-campaigns`,
  `crates/jope-engine`, the root Cargo files and feat-math's `Cargo.toml`, `lib.rs`, `ops.rs`,
  `model.rs`, `math_expr.rs` and `evolve.rs` exits 0, and so does the same check over the 70
  ADR-0004, lineage and pin paths. As compat_freeze's lineage note says, jope-engine's lineage
  identities also hash the running executable, so a rebuilt binary has new lineage identities;
  that is not an evaluator-identity change.
- **Replay.** The nine source commits `f34ff4656` to `c8e36e260` were cherry-picked onto
  `dd5ccf3f1`. The one conflict was the expected one: #130 and #131 added `pub mod telemetry;` and
  `pub mod scope;` to `kernel/mod.rs` at the line where `f34ff4656` adds `pub mod width;`, and all
  three lines are kept, in rustfmt's order. Right after its pick, `5b6a089ec` was reworded with
  `git commit --amend`, because its subject claimed more than the commit does (review R1: the
  `piecewise_linear` arm still panics): "the schema-bound table boundary returns width errors
  instead of panicking" became "width errors instead of panics for the legacy, corr_screen and
  pls_project stages of a schema-bound pipeline", and the body is unchanged. `git range-diff
  6e0ae2491..dbd10526d dd5ccf3f1..HEAD` shows `f34ff4656` → `5b75f3f9f` with those context lines
  only (`!`), `5b6a089ec` → `827fda6cf` with its subject line only (`!`), and the other seven
  unchanged (`=`): `fd4ffe5f3` → `65d90c92e`, `135cd5b4c` → `6037be4f2`, `925ec2a32` →
  `75f08888a`, `a9c0c4766` → `cb27d2151`, `9105c6ff5` → `bd9945d00`, `c9983d2e1` → `c4d4756cd`
  and `c8e36e260` → `524c79701`. Every replayed commit keeps its author and author date, and every
  one but `827fda6cf` its message. The series' three source and test files are byte-identical to
  the reviewed `c8e36e260`, and every other difference under `crates/` is #130's and #131's, byte
  for byte. The two old records commits, `ebf647557` and `dbd10526d`, are regenerated as this
  commit: the coverage item and the findings role re-applied to the new main's files, each adding
  exactly the lines `ebf647557` added; the parity inventory's feat-math section re-listed from a
  live listing; the receipt as it was at `dbd10526d`, with the base, the commit ids, the pull
  request number, the reworded subject and the regenerated counts changed, a "Replay" section
  added and the kept gap F1 named under "Still open"; the CHANGELOG line after #131's; this record
  with #131's readback; and "Next allowed action" item 2. The pre-replay head is kept locally as
  `refs/preserve/kernel2/pr06-prerebase-20260929T202502Z`.
- **One correction besides the replay.** The receipt said that a benchmark receipt in
  `jope-campaigns/src/aggressive.rs` is the one other place that names
  `advanced_v3/transforms/mod.rs`. The captured file list in a jope-records replay fixture
  (`python-parity/coverage/files.txt`), which #131's record found, names it and `kernel/mod.rs` as
  paths too; it includes and hashes nothing. The receipt now says so.
- **Records.** Coverage map +1 item (`target:feat-math/test/fitted_width`: rust-check, rust-test,
  full-default; 290 targets); findings inventory +1 role (regression, fitting; 326 items); test
  inventory +18 Rust tests (3,887 to 3,905, 65 ignored as before; feat-math 524 to 542), with
  feat-math re-listed and every other package unchanged; the CHANGELOG `[Unreleased]` line; the
  receipt `docs/internal/kernel2/receipts/WP-1.7.md`.
- **Commands, first round,** at the pre-replay `135cd5b4c`, the sources and tests of this pull
  request apart from two documentation comments (exit code, seconds), with `JOPE_PARITY_STRICT=1`
  for the tests: `cargo fmt --all --check` (0, 7); feat-math clippy `-D warnings` (0, 9); the
  feat-math PR tier twice, 497 passed and 10 skipped each time (0, 72; 0, 66); `fitted_width`,
  `compat_freeze`, `ip_scrub`, `legacy_v2_freeze`, `portable_parity` and the four `advanced_v3`
  suites twice, 61 passed each time (0, 46; 0, 46); `fitted_width` with `--no-capture` twice, 16
  passed, its output lines byte-identical (0, 2; 0, 2); `compat_freeze` twice in compare mode, 12
  passed each time (0, 12; 0, 12); the fixtures and the 70 identity paths unchanged (0);
  `deploy/ci/lambda.sh` (0, 52: 6 workspaces, 66 tests; its cargo-deny step skipped, 0.19.8
  installed where 0.20.2 is pinned); `deploy/ci/emitters.sh` into a new directory (0, 17);
  `jope-parallel` and `jope-dime-regressor` under `-P main` (0, 240: 1,772 passed, 25 skipped);
  `git diff --check` (0). At the pre-replay `925ec2a32`, which changes only those comments: fmt
  (0, 7), clippy (0, 8), the PR tier (0, 70: 497 passed), `fitted_width` (0, 2) and
  `compat_freeze` (0, 12). The exhaustiveness check and ten perturbations of the new code, each
  caught by one to five tests, are in the receipt. With the first records (`ebf647557`) in the
  working tree: `coverage validate` (0: 288 targets, 0 unassigned); `verify-test-inventory` (0:
  324 items); the jope-records PR tier (0, 14: 166 passed); `test-inventory.sh --no-rust
  --static-python` (0); `deploy/ci/check.sh` (0, 20); `deploy/ci/ip-scrub.sh` (0: 47 files); `cargo
  fmt --all --check` (0, 7); `git diff --check 6e0ae2491` (0).
- **Review round** (the pre-replay `a9c0c4766`, `9105c6ff5`, `c9983d2e1` and `c8e36e260`, then the
  records of `dbd10526d`):
  - NR-1: a table with no rows had its empty result turned into a width error after any stage that
    takes its generated width from the first row, for example `(rank) (standardize) (ridge)`, also
    through `predict_multiclass_table` and, after an appending stage, `pls_project`. Without rows
    the checks now compare no width. A probe of 23 programs built against the base, the reviewed
    head and the fix gives the base's outcome, messages included, for every program except the 6
    that panicked at the base, which are errors now.
  - F2: the legacy check also refuses a fitted size whose output row no `Vec` can hold, where
    `predict_table` panicked with a capacity overflow.
  - The head test requires every refusal to be one the requirement makes; the unguarded
    `piecewise_linear` arm (F1) is pinned by a test and listed under Open; the CHANGELOG line names
    the stages that are guarded (R1). The number placeholders and the #129 readback were left for
    the replay, which fills them in (above).
  - Commands at the pre-replay `c8e36e260` with a clean tree (exit code, seconds): fmt (0, 7);
    clippy (0, 11); the feat-math PR tier twice, 499 passed and 10 skipped each time (0, 73; 0,
    62); the named targets with `advanced_v3_multiclass` and `kernel_pipeline` twice, 72 passed
    each time (0, 46; 0, 45); `fitted_width --no-capture` twice, 18 passed, its output lines
    byte-identical to the implementation's (0, 2; 0, 2); `compat_freeze` twice, 12 passed (0, 12;
    0, 13); the fixtures and the 70 identity paths unchanged (0); `deploy/ci/lambda.sh` twice (0,
    54; 0, 8: 66 tests; cargo-deny skipped as before); `deploy/ci/emitters.sh` (0, 18);
    `jope-parallel` and `jope-dime-regressor` under `-P main` twice, 1,772 passed and 25 skipped
    each time (0, 226; 0, 196); `git diff --check` (0). Four perturbations of the added lines are
    each caught by one test. With the records of `dbd10526d` in the working tree: `coverage
    validate` (0: 288 targets, 0 unassigned); `verify-test-inventory` (0: 324 items); the
    jope-records PR tier (0, 14: 166 passed); `test-inventory.sh --no-rust --static-python` (0:
    3,872 Rust tests kept); `deploy/ci/check.sh` (0, 20); `deploy/ci/ip-scrub.sh` (0: 47 files);
    `cargo fmt --all --check` (0, 7); `git diff --check` (0); the coverage map and the findings
    inventory unchanged since `ebf647557` (0).
- **Commands after the replay** at `524c79701`, the tree of this commit apart from these records,
  with a clean tree (exit code, seconds): the feat-math PR tier with `JOPE_PARITY_STRICT=1` twice,
  532 passed and 10 skipped each time, in 25 binaries (0, 69 with the build; 0, 60): #131's 514
  and this pull request's 18 `fitted_width` tests; `fitted_width`, `compat_freeze`,
  `scope_validation`, `stage_telemetry` and the four `advanced_v3` suites in compare mode twice, 86
  passed each time, `compat_freeze` 12 of 12 and `fitted_width` 18 of 18 (0, 46; 0, 46); the review
  round's eleven targets with `scope_validation` and `stage_telemetry`, 105 passed and 1 skipped (0,
  45); `fitted_width` with `--no-capture` twice, 18 passed, its `STAGES` and `CORPUS` lines
  byte-identical to the implementation's (sorted sha256 `24a74b77…`; 0, 2; 0, 2); `compat_freeze`
  twice more, 12 passed (0, 12; 0, 12); `parity_bits` with `JOPE_PARITY_STRICT=1`, 6 passed (rust
  6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale; 0, 4); the
  legacy value digest `b64cf538…`, 191,716 results, as PR 03 recorded (0, 1); the
  `jope-dime-regressor` evaluator-identity tests, 15 passed, so the identity table is unaffected
  (0, 54 with the build); `jope-parallel` and `jope-dime-regressor` under `-P main`, the consumers
  of `predict_table`, 1,772 passed and 25 skipped (0, 286 with the build); `deploy/ci/lambda.sh`
  (0, 48: 6 workspaces, 66 tests, all `--locked`; its cargo-deny step skipped, 0.19.8 installed
  where 0.20.2 is pinned); `deploy/ci/emitters.sh` (0, 17); `cargo check --workspace --all-targets`
  (0, 39); `cargo fmt --all --check` (0, 7); feat-math clippy `-D warnings` (0, 9); the identity
  checks above, the fixtures and the replay checks (0). With these records in the working tree:
  `coverage validate` (0: 290 targets, 0 unassigned); `verify-test-inventory` (0: 326 items, 197
  regression); the jope-records PR tier (0, 15: 166 passed);
  `test-inventory.sh --no-rust --static-python`, and again with `--update` (0: no file changed);
  the feat-math section equal to a live listing (0); the two record items equal to the new main's
  files with this pull request's items re-applied (0); `deploy/ci/ip-scrub.sh` (0: 47 files);
  `deploy/ci/check.sh` (0, 33); `git diff --check dd5ccf3f1` (0); and the placeholder check
  (`git grep -nI` finds no number placeholder) (0). With these records committed, every command
  above ran again at the head with the same results.
- **Open.** Hosted CI. PR 07, the fitted-C++ runtime revision 3. **The kept gap F1:** the V3
  `piecewise_linear` arm of `apply_block` still panics on inconsistent state, edges fitted for
  more columns than the feature block has, and
  `schema_bound_piecewise_linear_with_extra_edge_columns_still_panics` pins that panic; the fix is
  scheduled for a follow-up (item 7 of the PR 07 brief). So WP-1.7's "no panic reachable through
  V3" holds for the legacy, `corr_screen` and `pls_project` stages only. The infallible
  `V3Op::apply` (reached in production only by the pls head, with state it fitted) and
  `BoundPredictor::Legacy`, which calls `Model::predict`, keep their panics on inconsistent state;
  heads' own tables are trusted. Nothing that ships calls `try_predict` yet. cargo-deny was skipped
  locally (0.19.8 installed, 0.20.2 asserted). The workspace check warns only in `battle-gpu`,
  `jope-dime-regressor` and `jope-parallel`, as before, and not in feat-math.
- **Evidence.** `~/jope-evidence/kernel2/pr06/prepare-20260929T185801Z/` for the first round,
  `~/jope-evidence/kernel2/pr06/review-20260929T195726Z/` for the review round and
  `~/jope-evidence/kernel2/pr06/rebase-20260929T202502Z/` for the replay: the logs of these
  commands, the exhaustiveness check, the perturbation diffs, the zero-row probe, the feat-math
  listings, the conflict and its resolution, the reworded message and the re-applied record items,
  each with `SHA256SUMS`.

**2026-09-29 kernel2 pull request 05 (#131): a static scope and width check for programs, library only.**
Branch `kernel2/claude-m1-scope`, prepared in the lane `.agent/worktrees/kernel2-next` on
`origin/main` `6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, after #129 merged, and replayed onto
`origin/main` `9146cbcc26b609fb8f59788088f6c87cc60bade2` after #130 merged, before its first push.
It carries the kernel2 plan's WP-1.6 as the PR 05 brief cuts it: the check and the search
profile's expression caps, with nothing calling either yet.
- **Readback of #130.** kernel2 PR 04 (#130, branch `kernel2/claude-m1-telemetry`) merged by rebase
  at 2026-09-29T18:31:14Z at the exact reviewed head `2204096ea9721d5104d108f9561e4c47d43de812`.
  Approval 5356592621 (neverhumanbot) was submitted at 18:14:04Z, after the head push (no later
  than 18:05:16Z). The `pull_request` run 36609540695 and the `workflow_dispatch` run 36609543324
  were green on attempt 1, all nine jobs of each. main = `9146cbcc26b609fb8f59788088f6c87cc60bade2`,
  and its tree equals the reviewed head's tree (`fcfce3f3…`). As for #128 and #129, gh-role warned
  that the account that opened the pull request also merged it, where `AGENTS.md` asks for a merger
  other than the author. Evidence: `~/jope-evidence/kernel2/pr130-merge-20260929T182855Z/`
  (`merge.log`, `SHA256SUMS`). Before it, #129 merged at 16:49:20Z (main `6e0ae2491`); #130's
  record carries that readback.
- **Contents.** `crates/feat-math/src/kernel/scope.rs`: `validate_scope(program, input_width)`
  walks a program's stages with the pipeline slots' width rules (`kernel::pipeline::next_width`,
  now `pub(crate)`, its body unchanged) and checks every `map` and `derive` expression against the
  lower bound of the width that reaches its stage. `ScopeError` names the stage, the portfolio
  member, the class (`ILLEGAL_SCOPE` or `COLUMN_OOB`) and the exact reason; a program that cannot
  be checked fails with its own reason and no class. `ExpressionScope::search_profile(columns,
  elementwise)` caps expressions at 15 nodes and depth 4. The check differs from
  `validate_legacy_widths` on purpose in two ways: it has no row count, so it takes `nystrom`'s
  width for one training row, and it accepts inputs wider than 512 columns (up to 2^24), because
  the runtime truncates a row only after a stage. An explicit portfolio's members are walked from
  one column, because the portfolio drops the columns that are constant in training before its
  members see the rows.
- **Identity.** None moves. The four changed sources are outside the ADR-0004 hash list, the
  lineage sources and the frozen pins; no fixture changes, and `compat_freeze` passes in compare
  mode. After the replay, `git diff --quiet 9146cbcc2 HEAD --` over the fixtures,
  `crates/jope-campaigns`, `crates/jope-engine`, the root Cargo files and feat-math's `Cargo.toml`,
  `lib.rs`, `ops.rs`, `model.rs`, `math_expr.rs` and `evolve.rs` exits 0, and so does the same
  check over the 70 ADR-0004, lineage and pin paths. In `kernel/pipeline.rs` only the signature of
  `next_width` changes (`pub(crate)`, which rustfmt wraps: five lines for one).
- **Replay.** The five source commits `38fd1caae` to `f7e5bcb5d` were cherry-picked onto
  `9146cbcc2`. The one conflict was the expected one: #130 added `pub mod telemetry;` to
  `kernel/mod.rs` at the line where `38fd1caae` adds `pub mod scope;`, and both lines are kept, in
  rustfmt's order. `git range-diff 6e0ae2491..153e688e6 9146cbcc2..HEAD` shows `38fd1caae` →
  `9bf5d0d86` with that context line only (`!`), and the other four unchanged (`=`): `3965228b3` →
  `89a7ca8ed`, `e1aba6d0b` → `ba623e0fe`, `c47f9906a` → `cd86c9152` and `f7e5bcb5d` →
  `67b5a5f7b`. Every replayed commit keeps its message, author and author date. The series' four
  source and test files are byte-identical to the reviewed `f7e5bcb5d`, and every other difference
  under `crates/` is #130's. The two old records commits, `e4bedb78c` and `153e688e6`, are
  regenerated as this commit: the coverage item and the findings role re-applied to the new main's
  files, each adding exactly the lines `e4bedb78c` added; the parity inventory's feat-math section
  re-listed from a live listing; the receipt as it was at `153e688e6`, with the base, the commit
  ids and the pull request number changed and a "Replay" section added; the CHANGELOG line after
  #130's; this record with #130's readback; and "Next allowed action". The pre-replay head is kept
  locally as `refs/preserve/kernel2/pr05-prerebase-20260929T183131Z`.
- **One correction besides the replay.** The receipt said that nothing in `crates/`, `deploy/ci` or
  `tools/` names the changed sources. A captured file list in a jope-records replay fixture
  (`python-parity/coverage/files.txt`) names three of them as paths; it includes and hashes
  nothing. The receipt now says so.
- **Records.** Coverage map +1 item (`target:feat-math/test/scope_validation`: rust-check,
  rust-test, full-default; 289 targets); findings inventory +1 role (regression; 325 items); test
  inventory +14 Rust tests (3,873 to 3,887, 65 ignored as before; feat-math 510 to 524), with
  feat-math re-listed and every other package unchanged; the CHANGELOG `[Unreleased]` line; the
  receipt `docs/internal/kernel2/receipts/WP-1.6.md`.
- **Commands, first round,** at the pre-replay `e1aba6d0b`, the sources the reviews read (exit
  code, seconds): with `JOPE_PARITY_STRICT=1`, the feat-math PR tier twice, 494 passed and 10
  skipped each time (0, 70; 0, 60); `compat_freeze` twice in compare mode, 12 passed each time (0,
  11; 0, 12); `compat_freeze`, `mathexpr_current_semantics`, `fault_trace`, `ip_scrub`,
  `legacy_v2_freeze`, `portable_parity`, `kernel_pipeline`, `kernel_expression` and
  `scope_validation` twice, 73 passed each time (0, 19; 0, 20); `scope_validation` with
  `--no-capture` twice, 13 passed, its output lines byte-identical (0, 6; 0, 6); four
  perturbations of `scope.rs`, each caught (1 to 6 failing tests) and restored;
  `deploy/ci/lambda.sh` (0, 48: 6 workspaces, 66 tests; its cargo-deny step skipped, 0.19.8
  installed where 0.20.2 is pinned). With the first records in the working tree: `cargo fmt --all
  --check` (0, 7); feat-math clippy `-D warnings` (0, 9); `coverage validate` (0: 288 targets, 0
  unassigned); `verify-test-inventory` (0: 324 items); the jope-records PR tier (0, 15: 166
  passed); `test-inventory.sh --no-rust --static-python` (0); `deploy/ci/check.sh` (0, 20);
  `deploy/ci/ip-scrub.sh` (0: 47 files); `git diff --check 6e0ae2491` (0); and `git diff --quiet`
  from the base over the ADR-0004 inputs, the lineage sources and the frozen pins (0).
- **Review fixes.** Three independent reviews of the pre-replay `e4bedb78c` (correctness, no
  regression, rules) found one should-fix and three nits; all are fixed, none rejected. An explicit
  portfolio's members see only the columns that vary in training, but the first round checked
  them at the full input width, so a member reading `c3` passed at width 4 and read out of range at
  runtime once a column was constant. `cd86c9152` walks members from one column; a new test pins
  it, including a fit with a constant column, and one perturbation shows that it fails on the old
  behaviour. `67b5a5f7b` makes the determinism test assert its digest, which the fix leaves
  unchanged. This record no longer names the findings role's subsystem. Commands at the
  pre-replay `f7e5bcb5d` (exit code, seconds): the feat-math PR tier twice, 495 passed and 10
  skipped each time (0, 70; 0, 60); `compat_freeze` twice in compare mode, 12 passed each time (0,
  12; 0, 12); the nine named targets twice, 74 passed each time (0, 19; 0, 19);
  `scope_validation` with `--no-capture` twice, 14 passed, its output lines byte-identical (0, 6;
  0, 6); `deploy/ci/lambda.sh` (0, 40: 6 workspaces, 66 tests; cargo-deny skipped as before). With
  the records of `153e688e6` in the working tree: `cargo fmt --all --check` (0, 7); feat-math
  clippy `-D warnings` (0); `coverage validate` (0: 288 targets, 0 unassigned);
  `verify-test-inventory` (0: 324 items); the jope-records PR tier (0, 14: 166 passed);
  `test-inventory.sh --no-rust --static-python` (0: the Rust half at 3,868); `deploy/ci/check.sh`
  (0, 20); `deploy/ci/ip-scrub.sh` (0: 47 files); `git diff --check 6e0ae2491` (0); and the
  identity `git diff --quiet` over the same 70 paths (0).
- **Commands after the replay** at `67b5a5f7b`, the tree of this commit apart from these records,
  with a clean tree (exit code, seconds): the feat-math PR tier with `JOPE_PARITY_STRICT=1` twice,
  514 passed and 10 skipped each time (0, 70 with the build; 0, 62); `compat_freeze`,
  `scope_validation`, `stage_telemetry`, `fault_trace`, `kernel_pipeline` and `kernel_expression`
  in compare mode twice, 65 passed each time, `compat_freeze` 12 of 12 and `scope_validation` 14
  of 14 (0, 20; 0, 20); the nine named targets of the earlier rounds with `stage_telemetry`, 93
  passed (0, 20); `scope_validation` with `--no-capture` twice, 14 passed, its six output lines
  byte-identical to the review fixes' (sorted sha256 `cfded17e…`; 0, 7; 0, 6), so the asserted
  determinism digest `4579e907…` holds; `parity_bits` with `JOPE_PARITY_STRICT=1` twice, 6 passed
  (rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale; 0, 4;
  0, 4); the legacy value digest `b64cf538…`, 191,716 results, as PR 03 recorded (0, 1); the
  `jope-dime-regressor` evaluator-identity tests, 15 passed, so the identity table is unaffected
  (0, 56 with the build); `deploy/ci/lambda.sh` (0, 43: 6 workspaces, 66 tests, all `--locked`;
  its cargo-deny step skipped, 0.19.8 installed where 0.20.2 is pinned); `deploy/ci/emitters.sh`
  (0, 17); `cargo check --workspace --all-targets` (0, 38); `cargo fmt --all --check` (0, 7);
  feat-math clippy `-D warnings` (0, 9); the identity checks above (0). With these records in the
  working tree: `coverage validate` (0: 289 targets, 0 unassigned); `verify-test-inventory` (0: 325
  items); the jope-records PR tier (0, 14: 166 passed);
  `test-inventory.sh --no-rust --static-python`, and again with `--update` (0: no file changed);
  the feat-math section equal to a live listing (0); the two record items equal to the new main's
  files with this pull request's items re-applied (0); `deploy/ci/ip-scrub.sh` (0: 47 files);
  `deploy/ci/check.sh` (0, 20); `git diff --check 9146cbcc2` (0); and the placeholder check
  (`git grep -nI` finds no number placeholder) (0). With these records committed, every command
  above ran again at the head with the same results.
- **Open.** Hosted CI. The wiring belongs to the admission policy (plan PR 08; the brief says
  PR 07). The expression parser's recursion, shared with the runtime and the legacy check, is not
  changed. cargo-deny was skipped locally (0.19.8 installed, 0.20.2 asserted). The workspace check
  warns only in `battle-gpu`, `jope-dime-regressor` and `jope-parallel`, as before, and not in
  feat-math.
- **Evidence.** `~/jope-evidence/kernel2/pr05/prepare-20260929T173902Z/` for the first round,
  `~/jope-evidence/kernel2/pr05/fix-20260929T181730Z/` for the review fixes and
  `~/jope-evidence/kernel2/pr05/rebase-20260929T183131Z/` for the replay: the logs of these
  commands, the feat-math listings, the perturbation diffs, the conflict and its resolution and
  the re-applied record items, each with `SHA256SUMS`.

**2026-09-29 kernel2 pull request 04 (#130): per-stage fault counts beside unchanged fits and predictions.**
Branch `kernel2/claude-m1-telemetry`, prepared in the lane `.agent/worktrees/kernel2` on
`79a88767f21dccf7505987f9b0ac7366d71f2c0b`, the reviewed head of #128, and replayed onto
`origin/main` `6e0ae2491cd4e7918c4d8b07aa211c83376c6764` after #129 merged, before its first push.
It carries the kernel2 plan's WP-1.4 and the rest of WP-1.8: every `map` and `derive` stage, and
every fit and prediction, can report its fault counts, and every stage output, model and
prediction stays bit-identical.
- **Readback of #129.** The islands-v2 source import (#129, branch `lambda/teacher-v2-import`)
  merged by rebase at 2026-09-29T16:49:20Z at its exact round-2 reviewed head
  `b754c1391babc41beaba7718b20552f762dc8a85`. Round 1 (review 5355567473, changes requested at
  16:34:25Z) had asked for the evaluator statement to be qualified, and round 2 did that. Approval
  5355667067 (neverhumanbot) was submitted at 16:43:30Z, after the head push (no later than
  16:37:03Z). The `pull_request` run 36599024961 and the `workflow_dispatch` run 36599023262 were
  green on attempt 1. main = `6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, and its tree equals the
  reviewed head's tree (`5108f2d4…`). As for #128, gh-role warned that the account that opened the
  pull request also merged it, where `AGENTS.md` asks for a merger other than the author.
  Evidence: `~/jope-evidence/kernel2/pr129-merge-20260929T164902Z/` (`merge.log`, `SHA256SUMS`).
  Before it, #128 (PR 03) merged at 15:43:17Z (main `3a32d780`); #129's record carries that
  readback.
- **Contents.** A new module, `crates/feat-math/src/kernel/telemetry.rs`, registered by one line in
  `kernel/mod.rs`. `apply_op_traced`, `fit_traced` and `predict_traced` return `apply_op`'s rows,
  `fit`'s model (the same `serde_json` bytes, apart from the wall-clock fit times a portfolio
  records) and `Model::predict`'s bits, with `StageFaultCounts`
  per stage (cells evaluated, cells per fault class, rows with a Hard cell, compile failure,
  constant output, epilogue replacements) and `PipelineTelemetry` per pass, which also counts the
  non-finite inputs a pipeline's entry sanitize replaced. Both traced entry points refuse a
  zero-width input with `ZeroWidthInput`, so `predict_traced` returns a `Result`. The stage
  epilogue and the append flag are private to `ops.rs`, and `fit` and `predict` live in `model.rs`,
  so that code is mirrored, and 19 tests hold every mirror to its original. The domain table gains
  the stage level.
- **Identity.** None moves. `kernel/` is outside the ADR-0004 hash list and every jope-engine
  lineage; `ops.rs`, `lib.rs`, `model.rs`, `math_expr.rs`, the Cargo files and every compat_freeze
  fixture are byte-identical to the base, and compat_freeze passes 12 of 12 in compare mode. No
  known-table row and no rebaseline.
- **Replay.** The nine commits `00065af25` to `b9d52cf5e` were cherry-picked onto `6e0ae2491`
  without a conflict, and `git range-diff 79a88767f..13b5b6c2a 6e0ae2491..HEAD` shows each one
  unchanged, as `304f7c702` to `cad9b120c`. #129 changed none of their paths, no crate of the root
  workspace and no root Cargo file. It added two frozen Lambda workspaces, and the teacher-v2
  worker builds `feat-math` by path, so `deploy/ci/lambda.sh` now builds and tests six workspaces
  with `--locked` (below). The two old records commits, `af2c7bfb9` and `13b5b6c2a`, are
  regenerated as this commit: the receipt as it was at `13b5b6c2a`, with the base, the commit ids
  and the pull request number changed and a "Replay" section added; the CHANGELOG line after
  #129's; this record with #129's readback; and "Next allowed action". The coverage map and both
  test inventories need no change: this pull request's two items, re-applied to the new main's
  files, give the replayed files byte for byte, and the parity inventory's feat-math section
  equals a live listing. The pre-replay head is kept locally as
  `refs/preserve/kernel2/pr04-prerebase-20260929T171848Z`.
- **Found before review.** Perturbation 5 of the receipt: the corpus's four affine symbolic
  programs did not hold the mirrored scaling to `fit` (`cov * (1.0 / var)` for `cov / var` left
  every test green), so `1cd0f4e50` adds a test that fits every corpus expression as a symbolic
  program. Six other perturbations each fail at least one test.
- **Review follow-ups.** One review of the pre-replay `af2c7bfb9` (F1 to F5, NR-1, R1 to R3) is
  answered by `3935d86e7`, `c38f65ad0`, `cad9b120c` and these records. No value, count or identity
  moves, and the test count and both inventories stay as they were. F1 (should-fix): an explicit
  portfolio does sanitize its input, inside the untraced `portfolio.rs`, so its `input_sanitized`
  is 0 whatever the input holds. The documentation said it ran no sanitize, and the portfolio test
  now runs on non-finite rows and pins the behaviour. F2: the legacy `fit` refuses only an empty
  first row, and `standardize` and the `ridge` head panic on a later one. F3, NR-1, R1: "the last
  four sections". F4: a portfolio's bytes carry wall-clock fit times. F5: the receipt names the
  test that checks `STAGES`. R2: the pull request number, #130, is written wherever the
  placeholder stood, and the check before the first push finds no placeholder. R3: a plain
  `git status` ran in the canonical checkout during implementation; no harm was found, and the
  process rule is tightened: no git command at all runs in the canonical checkout. Perturbation 8, a
  portfolio that counts its non-finite input, passed every test before the follow-ups and fails the
  portfolio test after them. Commands at the pre-replay `b9d52cf5e` with a clean tree (exit code,
  seconds): the feat-math PR tier twice, 500 passed each time (0, 73 with the build; 0, 61); the
  eight named targets twice, 78 passed each time (0, 13; 0, 12); `stage_telemetry` uncaptured (0,
  6: 19 passed, with the zero-width test's two expected panics printed); fmt (0, 7); feat-math
  clippy `-D warnings` (0, 1); feat-math rustdoc (0: 7 warnings as before, none in the new module);
  `cargo check --workspace --all-targets` (0, 37); `coverage validate` (0: 288 targets),
  `verify-test-inventory` (0: 324 items), `test-inventory.sh` (0) and the feat-math section equal
  to a live listing (0); ip-scrub (0: 47 files); `check.sh` (0, 20); `git diff --check 79a88767f
  HEAD` and the identity-input check (0).
- **Records.** Coverage map +1 item (`target:feat-math/test/stage_telemetry`: rust-check,
  rust-test, full-default; 288 targets); findings inventory +1 role (regression; 324 items); test
  inventory +19 Rust tests (3,854 to 3,873, 65 ignored as before; feat-math 491 to 510), with
  feat-math re-listed and every other package unchanged; the CHANGELOG `[Unreleased]` line; the
  receipt `docs/internal/kernel2/receipts/WP-1.4.md`.
- **Commands** at `cad9b120c`, the tree of this commit apart from these records, with a clean tree
  (exit code, seconds): the feat-math PR tier with `JOPE_PARITY_STRICT=1` twice, 500 passed and 10
  skipped each time (0, 70 with the build; 0, 61); `compat_freeze`, `stage_telemetry`,
  `fault_trace`, `mathexpr_current_semantics`, `ip_scrub`, `legacy_v2_freeze` and
  `portable_parity` in compare mode twice, 72 passed each time, `compat_freeze` 12 of 12 and
  `stage_telemetry` 19 of 19 (0, 12; 0, 12); `parity_bits` with `JOPE_PARITY_STRICT=1` twice, 6
  passed each time (rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0
  unexpected, 0 stale; 0, 4; 0, 3); `stage_telemetry` uncaptured (0, 6: 19 passed; the `CORPUS`,
  `BOUNDARY` and `SYMBOLIC` counts as before); the legacy value digest `b64cf538…`, 191,716
  results, as PR 03 recorded (0, 1); the `jope-dime-regressor` evaluator-identity tests, 15
  passed, so the identity table is unaffected (0, 56 with the build); `deploy/ci/lambda.sh` (0,
  51: 6 workspaces, 66 tests, all `--locked`; its cargo-deny step skipped, 0.19.8 installed where
  0.20.2 is pinned); `git diff --quiet 6e0ae2491 HEAD --` over the fixtures,
  `crates/jope-campaigns`, `crates/jope-engine`, the root Cargo files and feat-math's
  `Cargo.toml`, `lib.rs`, `ops.rs`, `model.rs` and `math_expr.rs` (0). Also
  `cargo fmt --all --check` (0, 7); feat-math clippy `-D warnings` (0, 9);
  `cargo check --workspace --all-targets` (0, 43); `coverage validate` (0: 288 targets, 0
  unassigned); `verify-test-inventory` (0: 324 items);
  `test-inventory.sh --no-rust --static-python`, and again with `--update` (0: no file changed);
  the feat-math section equal to a live listing (0); `deploy/ci/ip-scrub.sh` (0: 47 files);
  `deploy/ci/check.sh` (0, 20); `deploy/ci/emitters.sh` (0, 16); `git diff --check 6e0ae2491 HEAD`
  (0); the subject scrub and the author and trailer checks (0). With these records
  in the working tree, fmt, `coverage validate`, `verify-test-inventory`, `test-inventory.sh`,
  ip-scrub, `check.sh`, `git diff --check` and the placeholder check (`git grep -nI` finds no
  number placeholder) ran again (0 each), and with these records committed, every command above
  ran again at the head with the same results. Before the replay, at `afa696b9c`,
  `just test-main` passed 3,808 of 3,808 (0, 687; 65 skipped; `jope-dime-regressor` 997,
  `jope-parallel` 775 and feat-math 504 passed). It was not rerun: the library code it tested is
  this head's, since the later commits change a feat-math test file, documentation comments and
  documents only, and #129 changes no crate of the root workspace.
- **Commands** before review, at `afa696b9c`, the tree of `af2c7bfb9` apart from the records: as
  recorded in the receipt's "Commands and results", including seven perturbations of the new
  module, each run against the tests and reverted.
- **Open.** Hosted CI. PR 08 needs a per-severity cell count for WP-1.5's `domain_rate`, because
  a cell can carry two Domain classes, and has to choose between the stage cache and `fit_traced`;
  it can also count an explicit portfolio's own input sanitize (review F1). The ADR-0004 amendment
  should list `kernel/telemetry.rs` once PR 08 scores with it, which needs Codex. cargo-deny was
  skipped locally (0.19.8 installed, 0.20.2 asserted). Clippy `-D warnings` on
  `jope-dime-regressor` failed at #128's base in files this series does not touch (#128's record)
  and was not rerun; the series changes no file of that package, and feat-math is clean.
- **Evidence.** `~/jope-evidence/kernel2/pr04/prepare-20260929T162357Z/`: the logs of every
  command before review, the perturbation diffs and logs, the nextest listings, the scripts and
  the commit messages, with `SHA256SUMS`. The review follow-ups:
  `~/jope-evidence/kernel2/pr04/fixer-20260929T165709Z/`, laid out the same way. The replay and
  the commands at `cad9b120c` and at the head: `~/jope-evidence/kernel2/pr04/rebase-20260929T171848Z/`,
  with `SHA256SUMS`.

**2026-09-29 the teacher campaign's second stage is imported (#129).**
Branch `lambda/teacher-v2-import`, prepared in the lane `.agent/worktrees/kernel2-next` on
`origin/main` `f65026dd54ba29ef3e7e990b51b15eb42652c089` while kernel2 PR 03 (#128) held the next
slot, and replayed onto `origin/main` `3a32d780814d5536489e9378335fb81e2808f9b2` after #128 merged,
before its first push. It is not a kernel2 pull request: like #124 and #125, it imports frozen
Lambda work, and it changes no product source, root Cargo file or CI script.
- **Readback of #128.** kernel2 PR 03 (#128, branch `kernel2/claude-m1-faults`) merged by rebase
  at 2026-09-29T15:43:17Z at the exact reviewed head `79a88767f21dccf7505987f9b0ac7366d71f2c0b`.
  Approval 5354941699 (neverhumanbot) was submitted at 15:41:02Z, after the head push. The
  `pull_request` run 36589436324 and the `workflow_dispatch` run 36589437281 were green on
  attempt 1. main = `3a32d780814d5536489e9378335fb81e2808f9b2`, and its tree equals the reviewed
  head's tree (`8964c7b0…`). Evidence: `~/jope-evidence/kernel2/pr128-merge-20260929T154259Z/`
  (`merge.log`, `SHA256SUMS`).
- **Owner ruling (2026-09-29, about 08:20Z), paraphrased.** The work in
  `.agent/experiments/lambda-training/test-adaptive-gp-islands-v2/` is mission critical: it must
  not be deleted, and it should be rolled into a proper location in the main repository.
  `deploy/lambda/SNAPSHOT.json` (`owner_decisions`) keeps the ruling verbatim, except that the
  home-directory prefix is written `~/`. This pull request
  copies the work from custody; the source directory is neither moved nor deleted.
- **Source and freeze evidence.** Codex posted no stopped-state handoff: its last ledger entry is
  its 06:55:00Z claim of this work. After Claude's notice of 12:46:26Z, the orchestrator decided
  to import from the verified custody snapshot `~/jope-evidence/islands-v2-custody/20260929T124431Z/`
  (6,299 files, 846,984,567 bytes, tree list sha256 `28826359…`). Step 20 re-verified it
  locally and on its archive mirror: 6,299 of 6,299 files on each. The fallback freeze evidence:
  - the three snapshots' fresh hashes of the live source are equal, the 08:21Z one once its
    `.env` line is dropped, and their six listings are byte-identical;
  - no source changed after 07:31:56Z, and no build output after 07:33:36Z;
  - no process had its cwd or exe in the directory at 12:44Z;
  - a fourth snapshot, taken at 14:44Z after the import, recorded the same hash of the live
    source, the same listings and an empty process list.

  Nothing under `.agent/experiments` (which holds the live directory) or `~/scratch` was read,
  listed or hashed, and nothing touched AWS; the only access to those two roots was a test,
  while probing bwrap (which cannot run on xbabe2), that they exist. Nothing scanned `/proc` for
  the campaign's processes. Apart from the import's own processes and kernel settings, only
  `pgrep` read it, to check for a running `test-release.sh`, since such runs share a scratch
  directory; `pgrep` reads every process's command line. `33-review-fixes` lists these reads.
  Steps 21, 30 and 31 ran under a filtered `strace`, with 0 violations.
- **What lands.** `deploy/lambda/training/teacher-v2/`, at the source's own depth, so that the
  worker's `feat-math` path resolves unchanged: 13 tracked files and 1 renamed manifest.
  `SNAPSHOT.json` records all 24 source files and 14 excluded entries.
  - Two new standalone workspaces: the host-side search engine (`rust/island-engine`, 2 tests)
    and the Lambda worker (`rust/lambda-worker`, bin `bootstrap`, 3 tests).
  - `rust/proposal-engine` is the regression worker's source. Its manifest is tracked as
    `Cargo.toml.v2` and is not built.
  - Four files differ from their frozen bytes, each with a `tree_change`: `license` in both
    manifests and `publish = false` in the engine's, the worker lock's `feat-math` line at 0.2.0,
    and home-relative paths in the campaign README. That makes sixteen across `deploy/lambda`.
  - No CI script, root Cargo file, `deny.toml`, `feat-math` source or allowlist changes;
    `lambda-workspaces.sh` finds the new workspaces by itself.
  - The worker keeps the regression worker's Lambda-only `/tmp` scratch exception
    (`/tmp/jope-teacher-<attempt id>`). The two frozen workers are the only exceptions, and their
    tests write nothing there.
- **Custody.** The 10 untracked sources (8 Python files, the env template and a C pilot) are in
  `lambda-python-teacher-v2.tar` (sha256 `c353271e…`), with `FILES-teacher-v2.sha256` and
  `SHA256SUMS-teacher-v2` (`d69bc3ef…`), in `~/jope-evidence/lambda-20260929/python/` and its
  archive copy. `docs/archive/lambda-python-manifest.json` now lists 83 files in six tarballs. The
  private `.env` is recorded by existence, size and sha256 only.
- **Replay.** The four import commits `a03c19268` to `be43b6cc2` were cherry-picked onto
  `3a32d780` without a conflict, and `git range-diff` shows each one unchanged. #128 changed none
  of their paths. It did change `feat-math`'s source, which four of the six Lambda workspaces
  (csv, regression-ops, regression and the teacher-v2 worker) build by path: they still build with
  `--locked` and pass, and `compat_freeze` stays green (below). The two old records commits,
  `f40beb09e` and `40ce78d83`, are regenerated as this commit: `README.md` and
  `docs/internal/lambda/README.md` as they were at `40ce78d83` (#128 changed neither), the
  CHANGELOG lines after #128's, this record with #128's readback, and "Next allowed action". The
  coverage map and both test inventories stay as they are, because no root-workspace crate or
  test changes: `test-inventory.sh --no-rust --static-python --update` leaves
  `docs/parity/test-inventory.json` byte-identical, and the validators pass (below). The replay
  read nothing under either off-limits root; its only process reads were `pgrep` checks for
  concurrent test and build commands. The pre-replay head is kept locally as
  `refs/preserve/lambda/teacher-v2-import-prerebase-20260929T154400Z`.
- **Commands** at `b075d1292`, the tree of this commit apart from these records (exit code,
  seconds): `lambda-workspaces.sh snapshot-check` (0, 0: 87 tracked files) and `verify` (0, 1: 6
  workspaces); `deploy/ci/lambda.sh` with the lane's own target directory (0, 37: 6 workspaces
  built and tested, 66 tests: csv 40, regression-ops 2, regression 5, search engine 2, worker 3,
  monitor 14; its cargo-deny step skipped, 0.19.8 installed where 0.20.2 is pinned), and again
  (0, 8); `deploy/ci/test-bump.sh` (0, 28: 38 passed); the import gate's V1 byte-equality check
  (0, 1: `bad=0`; the 24 rows equal the frozen list and the snapshot, the four recorded changes
  are exact, and only `zig-arm-link.sh` is 100755); `compat_freeze` in compare mode (0, 19: 12
  passed); `coverage validate` (0, 1: 287 targets, 0 unassigned);
  `test-inventory.sh --no-rust --static-python --update` (0, 0: no file changed);
  `verify-test-inventory` (0, 1: 323 items); `deploy/ci/check.sh` (0, 20);
  `deploy/ci/ip-scrub.sh` (0, 2: 47 files); `deploy/release/test-release.sh` (0, 10: 195 of
  195); `cargo fmt --all --check` (0, 7); `git diff --check 3a32d780 HEAD` (0, 0). They run again
  at this pull request's head, with the logs in
  `~/jope-evidence/kernel2/import/rebase-20260929T154400Z/`.
- **Earlier, at the pre-replay head `40ce78d83`**, on inputs the replay does not change (the
  Lambda manifests and locks, `feat-math`'s manifest, `deny.toml` and the toolchain):
  cargo-deny 0.20.2 in `jope-runner:current` passed bans, licenses and sources for all six
  workspaces, and a release rebuild of the search engine reproduced its binary byte for byte
  (`77a5c2bd…`). Logs: `~/jope-evidence/teacher-v2-import/32-gates/` and `33-review-fixes/`.
- **Open.**
  - Codex's stopped-state handoff has not been received.
  - The run's results have no verified custody: they live only in Codex's runtime directory
    under `~/scratch/` and on S3. The owner and Codex were asked about them.
  - The campaign's first stage, `.agent/experiments/test-adaptive-gp-v1`, has no custody copy;
    `SNAPSHOT.json` lists it under `not_imported`.
  - Clippy `-D warnings`, which no CI lane runs, passes for the search engine and fails for the
    worker on one lint in its frozen source: `possible_missing_else` at
    `rust/lambda-worker/src/main.rs:119:89`. The source is imported verbatim and is not edited.
    The frozen workspaces already on main show the same class of finding: the regression worker
    4 lints and the monitor 3.
  - Codex has not published the run's source manifest. Until it does, the values recomputed from
    custody (`6044521e…`, published body `c867e95a…`) and the three binaries' provenance are
    unconfirmed.
  - The NTFS archive mirror of the 08:21Z snapshot may still hold the deleted `.env` bytes on the
    device. The owner decides whether that needs action.
- **Evidence.** `~/jope-evidence/teacher-v2-import/` (`20-freeze`, `21-frozen`, `30-import`,
  `31-python-custody`, `32-gates` and `33-review-fixes`), sealed and mirrored under
  `/mnt/jope-archive/jopedime-teacher-v2-import/`. `~/jope-evidence/INDEX.md` lists them and the
  four custody snapshots. The replay and its commands:
  `~/jope-evidence/kernel2/import/rebase-20260929T154400Z/`, with `SHA256SUMS`.

**2026-09-29 kernel2 pull request 03 (#128): fault reporting in the scalar evaluator, values unchanged.**
Branch `kernel2/claude-m1-faults`, prepared in the lane `.agent/worktrees/kernel2` on
`ab48e389c9951f87e158b5f52930f4d2c9b6d2f4` and replayed onto `origin/main`
`f65026dd54ba29ef3e7e990b51b15eb42652c089` after #127 merged, before its first push. It carries
the kernel2 plan's WP-1.1, WP-1.2 and WP-1.3, with the evaluator half of WP-1.8: the scalar
evaluator also reports which of its guards fired, and every value stays bit-identical.
- **Readback of #127.** kernel2 PR 02 (#127, branch `kernel2/claude-m0-harness`) merged by rebase
  at 2026-09-29T13:09:12Z at the exact reviewed head `11c5dc1b17d2c7d4135dd27c7b9a45b69f4b197d`.
  Approval 5352906262 (neverhumanbot) was submitted at 12:57:34Z, after the head push. The
  `pull_request` run 36570144285 and the `workflow_dispatch` run 36570145957 were green on
  attempt 1. main = `f65026dd54ba29ef3e7e990b51b15eb42652c089`, and its tree equals the reviewed
  head's tree (`cefd065e…`). The first merge attempt, at 13:07Z, got a GitHub 502 (its retries
  were refused with "Merge already in progress") and did not merge: the pull request stayed open
  and main unchanged. The guarded retry at 13:09Z merged. Evidence:
  `~/jope-evidence/kernel2/pr127-merge-20260929T130853Z/` (`merge.log`, `SHA256SUMS`); the failed
  attempt is in `pr127-merge-20260929T130352Z/`.
- **Contents.** `FaultSet`, `FaultClass` and `FaultSeverity` in
  `crates/feat-math/src/math_expr/fault.rs`: 16 append-only classes, bits 0 to 15 of a `u32`, each
  Hard, Domain or Saturation. Traced entry points: `MathExpr::eval_row_traced`,
  `eval_element_traced` and `eval_row_traced_strict`, and `CompiledMathExpr::eval_row_traced`,
  `eval_element_traced` and `map_row_traced`. The legacy entry points are unchanged; the old
  evaluator functions are one-line wrappers over sink-generic twins, with the arithmetic untouched.
  The domain table is `docs/internal/kernel2/math-domain-table.md`.
- **Identity.** `math_expr.rs` is in the ADR-0004 hash list, so the evaluator identity moves:
  `implementation_sha256` `9bee59f3…` → `3ec654cb…`, digest `430a0119…` → `891dae89…`;
  `Cargo.lock` is unchanged. The v0.2.0 identity is appended to `known-evaluator-identities.json`,
  with a receipt the base build produced, because the learning store's receipts are out of reach;
  the build of `f65026dd5` writes the same receipt byte for byte. `math_expr/compiled.rs` and
  `math_expr/fault.rs` are outside the hash list. Only `compat_freeze/evaluator_identity.v1.json`
  was rebaselined, with `JOPE_COMPAT_FREEZE_REGENERATE=identity`; every other fixture is
  byte-identical.
- **Replay.** The nine commits `07384a8d0` to `256711ce2` were replayed onto `f65026dd5`. Five are
  unchanged cherry-picks. The known-table row's `builds` field now ends at `f65026dd5`, the
  identity fixture was regenerated at `600333e80` (only `generated_at_sha` and the known-table
  digest differ), the parity inventory was re-listed on top of #127's, and the first records
  commit was rewritten. #127 changes no input of `compat_freeze`. The pre-replay head is kept
  locally as `refs/preserve/kernel2/pr03-prerebase-20260929T134256Z`.
- **Review follow-ups.** `d208f4fcf` makes `FaultClass::ALL` a slice, so an appended class keeps
  its type. `a7a541678` corrects the domain table's `ILLEGAL_SCOPE` line. `319683f64` adds the NaN
  and the infinity that `^` and `/` absorb without a class to the table's "Not reported" list and
  pins them in three test cases. The receipt now reports the compiled `map_row` path as a
  reproducible slowdown of 1.3% to 2.3% (five measurements, one of them at this head; only map
  stages that read a row summary take it), and records the consumer test tiers below.
- **Records.** Coverage map +1 item (`target:feat-math/test/fault_trace`: rust-check, rust-test,
  full-default); findings inventory +1 role (regression); test inventory +16 Rust tests (3,838 to
  3,854, ignored 63 to 65), feat-math re-listed and every other package unchanged; the CHANGELOG
  `[Unreleased]` line; the receipt `docs/internal/kernel2/receipts/WP-1.1-1.3.md`. A second
  records commit, made before the first push, writes the pull request number, gives the time and
  cause of #127's failed first merge attempt, and rewrites "Next allowed action" items 2 and 3.
- **Commands** at `319683f64`, the tree of this commit apart from these records (exit code,
  seconds): with `JOPE_PARITY_STRICT=1`, the feat-math PR tier twice, 481 passed each time (0, 60;
  0, 60); `parity_bits` twice, 6 passed each time (rust 6, python 6 and cpp17-fitted 10
  divergences, all listed, 0 unexpected, 0 stale); `compat_freeze`, `mathexpr_current_semantics`,
  `fault_trace`, `ip_scrub`, `legacy_v2_freeze` and `portable_parity` twice, 53 passed each time;
  the `jope-dime-regressor` evaluator-identity tests twice, 15 passed each time; `just test-main`
  (100, 683): 3,789 run, 3,788 passed, 1 failed, 65 skipped, with `jope-parallel` 775 and
  `jope-dime-regressor` 997 passed. The one failure, jainnode's
  `concurrent_runtime_accepts_durable_cancellation_while_execution_runs`, is in a crate that uses
  no feat-math or campaigns code; it failed under the run's load and passed 5 of 5 alone. Also
  `cargo fmt --all --check` (0, 7); feat-math clippy `-D warnings` (0); `coverage validate` (0: 287
  targets, 0 unassigned); `verify-test-inventory` (0: 323 items); `test-inventory.sh --no-rust
  --static-python` (0); `deploy/ci/check.sh` (0, 20); `deploy/ci/ip-scrub.sh` (0: 47 files);
  `deploy/ci/lambda.sh` (0, 31: 4 workspaces, 61 tests; its cargo-deny step skipped, 0.19.8
  installed where 0.20.2 is pinned); `deploy/ci/emitters.sh` (0, 16); `git diff --check
  f65026dd5 HEAD` (0); the legacy value digest `b64cf538…`, in dev and in release, as at the
  base; the ADR-0004 receipt from the builds of `f65026dd5` (byte-identical to the committed one)
  and `319683f64` (only the identity-derived digests differ). With these records in the working
  tree, fmt, `coverage validate`, `verify-test-inventory`, `test-inventory.sh`, ip-scrub,
  `check.sh` and `git diff --check` ran again (0 each), and `compat_freeze` and `fault_trace`
  passed again.
- **Open.** Hosted CI; the ADR-0004 amendment for `compiled.rs` and `fault.rs`, which needs
  Codex; an executed test that the new known-table row is accepted, which needs a
  `jope-campaigns` change. Clippy `-D warnings` on `jope-dime-regressor` fails at the base, in 16
  `crates/jope-campaigns/src` files this change does not touch (30 library and 38 test-build
  errors, the same at this head); this change adds no finding, and feat-math is clean. No CI lane
  runs clippy, but `just ci-pr` does. The jainnode test above is a race under load, like the one
  jope-parallel test that `deploy/ci/known-broken-tests.txt` lists as cause C; it is not listed
  there.
- **Evidence.** `~/jope-evidence/kernel2/pr03/replay-20260929T134256Z/`: the replay, the review
  follow-ups and these commands (logs, the throughput data, the ADR-0004 reproductions), with
  `SHA256SUMS`.

**2026-09-29 kernel2 pull request 02 (#127): a search ablation harness and a bit-exact parity test.**
Branch `kernel2/claude-m0-harness`, prepared in the lane `.agent/worktrees/kernel2-next` on
`629e69d865a674222e00224cebf45c1ffddc8dea` and replayed onto `origin/main`
`ab48e389c9951f87e158b5f52930f4d2c9b6d2f4` after #126 merged. It carries the approved kernel2
plan's WP-0.4 and X0: tests, a harness and records only. No existing code path changes:
jope-bench's `main()` gains one match arm for the appended `Ablation` command and `lib.rs` one
module declaration, and no Cargo manifest, lock file or ADR-0004 evaluator-identity input changes.
- **Readback of #126.** kernel2 PR 01 (#126, branch `kernel2/claude-m0-freeze`) merged by rebase
  at 2026-09-29T11:55:17Z at the exact reviewed head `1dc59fbd4e6d1b45c57dc17aa3782f9f724f2917`.
  Approval 5351955355 (neverhumanbot) was submitted at 11:42:28Z, after the head push. The
  `pull_request` run 36562783349 and the `workflow_dispatch` run 36562783666 were green on
  attempt 1. main = `ab48e389c9951f87e158b5f52930f4d2c9b6d2f4`, and its tree equals the reviewed
  head's tree. Evidence: `~/jope-evidence/kernel2/pr126-merge-20260929T115500Z/` (`merge.log`,
  `SHA256SUMS`).
- **Harness.** `jope-bench ablation` (`crates/jope-bench/src/ablation.rs`; protocol in
  `docs/internal/kernel2/ablation-protocol.md`) runs two profiles of the shipping search on the
  same datasets, cuts, seeds and evaluation budget and writes one `jope.bench-ablation/v1`
  document: metrics M1 to M11, a per-profile aggregate, gates G-A to G-E and a verdict, with wall
  time outside the digested body. The only profile is `legacy`. A=A over four D5 datasets and
  eight seeds at 16 evaluations reported `no difference` with body sha256 `f0bf07de…`, the digest
  recorded before the replay. An M8 clause is not evaluated unless every run of both profiles has
  a value. Scratch goes under `--scratch` or `TMPDIR`; with neither, the harness refuses to run.
- **Parity.** `crates/feat-math/tests/parity_bits.rs` compares every prediction of 19 programs
  (4,788 per lane) bit for bit between the runtime and the emitted Rust, Python and fitted C++17
  programs; two divide by an input column, and the test checks that the division guard is
  reached. The 22 divergences present at the base are listed with reasons in
  `crates/feat-math/tests/fixtures/parity_allowlist.v1.json`, all fix item X6: the sign of a zero
  row sum (6), fitted C++ minimum and maximum on mixed-sign zeros (4), and non-finite inputs to
  closed-form programs (12; the input contract, X4, settles which side X6 changes). An edited
  entry fails the lane as unexpected plus stale; a missing `rustc`, `python3` or C++ compiler
  skips its lane loudly, and fails it under `JOPE_PARITY_STRICT=1`. The fitted C++ lane treats a
  compiler other than g++ 13.3, the one its entries were recorded with, the same way. The test
  reads the shared boundary corpus `tests/fixtures/math_boundary/v1.json` and pins its sha256, so
  a corpus change fails it as well as `compat_freeze`.
- **Replay.** After #126 merged, the nine commits `71502f3f4` to `cf654f56d` were cherry-picked
  onto `ab48e389c` without a conflict; `git range-diff` shows each one unchanged. `6d14d91c9`
  switches `parity_bits` from its checked copy of the boundary corpus to the shared file, with the
  same bytes (sha256 `ed1dfe9c…`). The two old records commits, `2d48ac975` and `12b24f3fe`, are
  regenerated as this commit. The pre-replay head is kept locally as
  `refs/preserve/kernel2/pr02-prerebase-20260929T115536Z`.
- **Records.** Coverage map +1 item (`target:feat-math/test/parity_bits`: rust-check, rust-test,
  full-default); findings inventory +1 role (regression, export); test inventory +12 Rust tests
  (3,826 to 3,838, ignored 62 to 63), feat-math and jope-bench re-listed with
  `cargo nextest list --locked -P main -p feat-math -p jope-bench` and every other package
  unchanged; the CHANGELOG `[Unreleased]` line.
- **Commands** at `6d14d91c9`, with this commit's record files in the working tree (exit code,
  seconds): `JOPE_PARITY_STRICT=1` `parity_bits` twice, 6 passed each time (0, 4; 0, 3; rust 6,
  python 6 and cpp17-fitted 10 divergences, all listed, 0 unexpected, 0 stale); the A=A test
  twice (0, 13; 0, 13); `compat_freeze`, `mathexpr_current_semantics`, `ip_scrub`,
  `legacy_v2_freeze` and `portable_parity` 40 passed (0, 11); `JOPE_PARITY_STRICT=1`
  `nextest -P pr -p feat-math -p jope-bench` 578 passed (0, 61); `nextest -P pr -p jope-bench
  --features asm` 177 passed (0, 27); `cargo fmt --all --check` (0, 6);
  `cargo clippy -p feat-math --all-targets --locked -- -D warnings` (0, 9); `coverage validate` (0, 1:
  286 targets, 0 unassigned); `verify-test-inventory` (0, 0); `deploy/ci/check.sh` (0, 20);
  `deploy/ci/ip-scrub.sh` (0, 1: 47 files); `deploy/ci/lambda.sh` (0, 9: 4 workspaces, 61 tests;
  its cargo-deny step skipped, 0.19.8 installed where 0.20.2 is pinned);
  `git diff --check origin/main...HEAD` (0, 0); the A=A run (0, 152), body sha256 `f0bf07de…`.
- **Open.** Clippy `-D warnings` on jope-bench fails at the base, in files this change does not
  touch: linted with its dependencies it stops at jope-pipeline's `dead_code` (`refit_request`,
  `search_plan.rs:200`); with `--no-deps` it fails on six `needless_borrow` in
  `panel/inference.rs` (lines 334, 347, 348, 389, 420 and 465); `--features asm` adds four more in
  `panel/` (`demo_asm.rs:38`, `experimental_proposal.rs:355`, `fitted_export_run.rs:431`,
  `optimize.rs:1081`). No CI lane runs clippy, but `just ci-pr` does. Clearing them edits existing
  code, so it needs a separate lint pull request or an owner waiver. This change adds no finding,
  and feat-math is clean.
- **Evidence.** `~/jope-evidence/kernel2/pr02/`: the review fixes under `fixer/`, the replay and
  these commands under `rebase-20260929T115536Z/`.

**2026-09-29 v0.2.0 is published; the kernel2 train begins with #126.**

- **v0.2.0.** The owner authorized publication at about 08:15Z.
  - `deploy/release/release.sh v0.2.0 629e69d865a674222e00224cebf45c1ffddc8dea --publish` exited 0 at 08:24:30Z.
  - The annotated tag `ed5ddd5190bea066f82ad4394cb3a25051af86c8` peels to M2 `629e69d86`.
  - The script's own readback passed 10/10. An independent readback passed 22/22: all eight downloaded assets byte-identical, `sha256sum -c`, cosign "Verified OK" against `docs/release/signing.pub`, a tampered copy rejected, the binary printing `jope 0.2.0`, glibc at most 2.34, and no GPU `NEEDED` entry.
  - Evidence: `~/jope-evidence/release-v0.2.0/publish-20260929T081827Z/` and `~/jope-evidence/closeout-20260929/40-release/READBACK-VERIFY.json`, sha256 `d44c0b4e…`, mirrored.
  - The main merge freeze was lifted at 08:37Z.
- **Owner decisions of 2026-09-29.** The owner approved the kernel2 plan: the major release of the algorithm kernel, targeting 1.0.0. The plan and its PR train are recorded in `~/xbabe2_kicker/docs/JOPEDIME_GP_MATH_SYNTAX_WORKPLAN_v2.md` (sha256 `d3040dd1…`), which amends the math-syntax workplan. The decisions:
  - The programmable learner core ships in the next major.
  - C++17 gets a fitted predictor early and a train-and-predict recipe for every baseline element by 1.0.
  - Only the fitted-C++ undefined behaviour is fixed. The round, asinh and erf values are kept.
  - The agent promotes low-risk rungs; the owner signs off exp/log, pow, new primitives and stages, quarantine admissions and the 1.0 default flip.
  - The registry is not edited this release.
  - kernel2 runs ahead of regression trials v2.

  Separately, the owner ruled that the islands-v2 experiment directory is mission critical: it must never be deleted and should be imported into a proper tracked location. A read-only custody snapshot exists, and the import waits for Codex's frozen handoff.
- **#126 (merged; main `ab48e389c`): the compatibility freeze** (kernel2 PR 01, workplan WP-0.1, WP-0.2, WP-0.3 and WP-0.5). It contains tests, fixtures and records only; no product source changes.
  - **The freeze covers:**
    - 82 scalar expressions × 266 boundary rows, through the interpreter, the compiled form and element and map paths, with error kinds;
    - map and derive stage outputs;
    - emitted bytes of 55 programs (`emitted_bytes.v1.json`):
      - Rust, Python and the manifest: all 55, under the default options and under the options `jope emit` uses;
      - C++17 recipe and JavaScript, under the default options only: bytes for 7 programs each, and the emitter's refusal for the other 48 each. The 25 ops each of these emitters accepts are all used by those 7 programs;
      - fitted C++17 source and codec: 54 programs. The portfolio program is recorded as skipped;
    - serialized models (`model_serde.v1.json`): the serde_json bytes and 16 held-out predictions of 54 fitted models, five of them stored whole and reloaded. The same portfolio program is recorded as skipped;
    - the grammar, registry, policy-vocabulary, lineage-source and ADR-0004 evaluator-identity inputs;
    - the seeded generator streams of every legacy search generator (seeds 0..64);
    - whole seeded search runs.
  - **Rebaselining.** `JOPE_COMPAT_FREEZE_REGENERATE=identity` rewrites only the identity and lineage fixtures. `JOPE_COMPAT_FREEZE_REGENERATE=gained-support` only replaces a value recorded as a refusal with the bytes its emitter now writes, and changes no recorded bytes; compare mode fails on such bytes until it has run. The other surfaces never rebaseline without an owner decision.
  - **Proof that it catches changes.** Moving the division guard from 1e-12 to 1e-11 fails three tests with readable diffs. Restoring the old rule that let gained bytes pass, or letting `gained-support` record changed bytes, fails the new gained-support test.
  - **Review 5351319682** requested changes at `ec59a27d4`. These records overstated the byte freeze, and bytes for a value recorded as a refusal passed without being pinned, so they could change again in a later PR. `f8854527a` makes compare mode fail on such bytes, naming the program, the language and the `gained-support` mode, which records them and nothing else. `be8ad9565` proves the rules on an in-memory copy of the fixture. The counts above come from the fixtures, with jq.
  - **Verification before that review:**
    - feat-math PR tier 460/460;
    - a replica of the CI rust-test lane 1750/1750;
    - `deploy/ci/check.sh`, ip-scrub, the coverage validator and `verify-test-inventory`;
    - the Lambda workspaces 61/61, with cargo-deny skipped locally for a version mismatch.
  - **Verification after it**, at `be8ad9565` on xbabe2:
    - compat_freeze 12/12, mathexpr_current_semantics 18/18 and ip_scrub 6/6, twice, in compare mode;
    - `cargo fmt --all --check`, feat-math clippy with `-D warnings`, the coverage validator, `verify-test-inventory` and `git diff --check`;
    - the feat-math part of the parity test inventory equals a live nextest listing (469 tests).
    - The PR tier, the rust-test replica and the Lambda workspaces were not rerun.
  - Record: `docs/internal/kernel2/receipts/WP-0.1.md`.
  - The freeze fixtures come from xbabe2. The first hosted rust-test run on xbabe1 is the cross-host libm check.

Next allowed action:
1. The close-out is complete. Its independent end-state verification (`~/jope-evidence/closeout-20260929/70-endstate/VERIFY.json`, 12:09:24Z) passed checks 1 to 5, 7 to 9 and 12. Its three blocking findings were bookkeeping gaps, not losses, and were remediated at 12:26Z (`~/jope-evidence/closeout-20260929/80-remediation-20260929T122626Z/`, sealed and mirrored):
   - Check 6: the eight local `rr2/*` branches of the merged PRs #117 to #122, #124 and #125 (161 commits; their tips are also on GitHub as `refs/pull/<n>/head` and in the final bundle) moved create-only to `refs/preserve/20260929/heads/rr2/`. The recount is 11, exactly this pull request's commits, unpushed at the time.
   - Check 10, ruled not blocking by the orchestrator: the 2,644 `/tmp` entries newer than 04:00Z belong to other projects on the host (RedlineDB tests, a pytest project, `dope`, the Codex CLI runtime, screen), and none traces to a close-out command. Nothing was deleted; the owner may rule otherwise.
   - Check 11: the two seals broken by a `run.log` line appended after sealing (`34-pr125-r1-fix`, `60-refs/apply-20260929T113950Z`) were resealed, and each old seal is kept as `SHA256SUMS.superseded-appended-runlog`. `40-release` is now sealed. The missing mirrors (`10-pr124`, `34-pr125-r1-fix`, `40-release`, `60-refs` and every step's `VERIFY.json`) were copied to the archive and verified there. `~/jope-evidence/INDEX.md` now lists every step.
   - Done before the verification: the demo lane is gone; the refs are pinned (250 pins, 691 to 941 refs), and the final all-refs bundle (sha256 `592950b4…`, 941 refs) was restored and checked in an independent bare repository (`60-refs/`, independently verified at 11:21:00Z). The remote branch renames and deletions ran from 11:39:50Z to 11:45:44Z, after the AGENT_CHAT notice of 11:23:58Z and its 15-minute objection window (`60-refs/apply-20260929T113950Z/run.log`: local 5 of 5 and remote 24 of 24 rows acted, none refused; the stash was cleared; 140 remote-tracking refs were pruned).
   - A completeness critic restored the final bundle again, independently, and found every T0 census commit reachable (`70-endstate/critic-endstate-critic-20260929T123116Z/`). Its one blocking item, a ledger end snapshot, was taken after the close-out handoff was posted in AGENT_CHAT at 12:35:15Z: `01-agent-chat/snapshots/AGENT_CHAT-20260929T123515Z-end.md`, 509,995 bytes, with the 06:47Z snapshot as its byte prefix. It is sealed and mirrored. `81-remediation-20260929T123356Z/` holds the custody copies made the same afternoon: the kernel2 plan documents and the `release-v0.2.0` run directories.
2. The kernel2 train, one implementation PR at a time, each merged by `merge-guarded.sh` at its exact reviewed head:
   - PR 01 to PR 03 (#126 to #128), the islands-v2 source import (#129), PR 04 (#130) and PR 05 (#131) are merged; their readbacks are in the records of #127 to #131 and this pull request;
   - PR 06 is #132, this pull request: the Rust fitted-width contract (branch `kernel2/claude-m1-width`, prepared in the lane `.agent/worktrees/kernel2` on main after #129 and replayed onto main after #131 before its first push);
   - after it merges, PR 07, fitted-C++ runtime revision 3. It stacks on this pull request's width contract;
   - PR 08, the library search admission policy, is prepared in the lane `.agent/worktrees/kernel2-next` and needs only PR 03 to PR 05. It lands in whichever order the queue allows, with one pull request open at a time;
   - then M2: PR 09, the capability manifest and search profiles; PR 10, the closure gate and the export report;
   - then the rest of the train in plan order.
3. The islands-v2 import (owner ruling of 2026-09-29): the source import merged as #129. It imported the frozen source into `deploy/lambda/training/teacher-v2`, from the verified custody snapshot of 2026-09-29T12:44:31Z (`~/jope-evidence/islands-v2-custody/20260929T124431Z/`). Still open:
   - Codex's stopped-state handoff;
   - custody of the run's results (its run directory under `~/scratch/`, and S3);
   - custody of the parent study `test-adaptive-gp-v1`.

   Codex was asked about all three at 14:02:12Z.
4. Regression trials v2 after the kernel2 train, unless the owner reorders.

**2026-09-28 wrap-up train and v0.2.0.** Eight pull requests, #115 to #122, landed on main between
2026-09-27T09:56:32Z and 2026-09-28T19:29:10Z. Clean-base part B (#123) followed at
2026-09-28T21:39:07Z, and #124, which carries the v0.2.0 release tooling, at 2026-09-29T05:12:34Z.
This pull request, #125, imports the frozen regression workers and completes the records, and
v0.2.0 is cut from its merge commit (owner decision O1, below). That closes the wrap-up train the
owner approved at 2026-09-27T22:20Z, release first. Next comes regression trials v2, before the
live three-host S3 demonstration (owner decision O2).

The same rules held for all eight, for #123 (whose one failed CI attempt was at an earlier head,
below) and for #124:
- **Merge.** Each one merged by GitHub rebase at the exact reviewed head, and each merged tree is
  byte-identical to the reviewed head's tree. `jepsontaylor` performed every merge.
- **Review.** Every approval and every changes-requested review came from `neverhumanbot`
  (219957145). Every commit that landed on main has Jepson Taylor
  `<130782313+jepsontaylor@users.noreply.github.com>` as both author and committer. Claude
  appears only as a Co-Authored-By trailer.
- **CI.** A `pull_request` run and a `workflow_dispatch` run of `ci.yml` passed on attempt 1 at
  each final head. A full run is nine jobs: `changes`, `hygiene`, `rust-check`, `rust-test`,
  `emitters`, `inputs`, `python`, `e2e-smoke` and `aggregate`. Lanes skipped by the path matrix
  are named below.
- **Issues.** None of the eight closes an issue.
- **Custody.** `…/claude-w0-w12/` stands for
  `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/`.

Merge order, main SHA after each merge, and final main:
`cee1fc8bd242688972c352133847793c6927279f` (#115) →
`f46a9048fbb2697d6d1f4463a688083ea9dfcce1` (#116) →
`dd49b23bc94678106d33544e138f22e8979e8344` (#118) →
`7aec174e055af6ffe41f71446876e0b0e6f632ce` (#117) →
`4ff4e7b697dcd5b7097dacbae9e99c07ffef8e28` (#119) →
`01ef9ddcb31133bea27188e29af9978c2d36b8c4` (#121) →
`4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6` (#120) →
`a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5` (#122) →
`0737da51102ad502676c1b1c3c06549a3ad0442e` (#123) →
`5d29d86e4dd80dd87d074f3d02d82d51923e4c73` (#124) → this pull request's merge commit.

A rebase merge writes new commits, so no commit of this pull request can name that merge commit.
It is `origin/main` immediately after the merge, and the v0.2.0 release readback under
`~/jope-evidence/release-v0.2.0/` records its SHA.

- **[#115](https://github.com/neverhuman/JopeDime/pull/115): the record of #114's landing.**
  It added the MASTER_PLAN record that follows this one.
  - Merge: one commit, main `cee1fc8bd242688972c352133847793c6927279f`.
  - Reviewed head: `a2e45bca35a31db33bbd4b7a24d1038be8142c1a`. The tree is
    `070bf1c6969c01963fcd8f86991e758ba16e5c10` on both sides.
  - Review: approval 5329823074.
  - PR run 36310255122: `changes`, `hygiene` and `aggregate` passed; the six other lanes were
    skipped by path.
  - Manual run 36310333178: all nine jobs passed.
  - Manual dispatches 36310263359 and 36310263463 were cancelled. They are not evidence.
  - After merge: main-guard 36310954778 passed. full-main-cpu 36310954813 failed. No run of
    that workflow succeeded until #116.
  - Custody: `…/claude-w0-w12/pr115-merge-20260927T095621Z/` and
    `…/claude-w0-w12/independent-review-pr115/`.
- **[#116](https://github.com/neverhuman/JopeDime/pull/116): the known full-main-cpu failures
  are routed to the lanes that can prove them.**
  - It carries Codex's `health/full-main-green`. The 8 host-data tests go to
    `deploy/ci/real-corpus-lane.sh`. The CPU fit-service contracts are ignored only under
    `neural`; that list grew from 74 to 80 with #114's six new contracts.
    `deploy/ci/check-test-routing.sh` guards both lists.
  - No test was deleted, and no fence was weakened.
  - Merge: four commits, main `f46a9048fbb2697d6d1f4463a688083ea9dfcce1`.
  - Reviewed head: `05a5cf84c90bca27f00c0d67dda624e5d63634fd`. The tree is
    `3cb9737c1b6f6f10bd412815c386a78359b85db0`.
  - Review: approval 5331659297.
  - CI: PR run 36340840488 and manual run 36340841630 passed all nine jobs.
  - After merge: main-guard 36343656758 passed, and full-main-cpu 36343656796 passed. That was
    the workflow's first successful run: the 53 runs created before it ended in 48 failures and
    5 cancellations.
  - Custody: `…/claude-w0-w12/pr116-merge-20260927T191447Z/` and
    `…/claude-w0-w12/independent-review-pr116-05a5cf84c/`.
- **[#118](https://github.com/neverhuman/JopeDime/pull/118): targeted single-dataset search in
  `jope pipeline`.**
  - What it adds:
    - seeded generation 0 and an incumbent;
    - deterministic threads and islands (parallel search groups over the same cuts);
    - `evolve` rounds chained in a verified `lineage.json`, with an unseeded control;
    - a sealed test split scored at most once per dataset and program in its registry (a local
      record, not access control);
    - `hwm ingest --source pipeline`;
    - an append-only champion bank whose export feeds the next `evolve --seeds`.
  - The local demonstration improved on validation folds that every round selected on. On the
    sealed test split the control's champion scored better. The PR says so.
  - Merge: 32 commits, main `dd49b23bc94678106d33544e138f22e8979e8344`.
  - Reviewed head: `9fa240906991bd10f5538c29edc66c7b99d28b29`. The tree is
    `d5ddc3358218de0711d39e98c71fdd3284816771`.
  - Review:
    - 5334007364 requested changes at `d7e89f21197f446618bd1c80e03e90890f185010`. Refused
      arguments had to leave nothing written, and the determinism claim had to be limited to
      phases that the budget, not the deadline, ends.
    - 5334625316 requested changes at `053c7862f755a17ae7434561e626e4d4c096b686`. Two
      concurrent test-split scores both recorded no earlier looks. The fix is a per-dataset lock.
    - 5334865651 approved.
  - CI: PR run 36387309556 and manual run 36387306913 passed all nine jobs.
  - After merge: main-guard 36388986311 and full-main-cpu 36388986268 passed.
  - Custody: `…/claude-w0-w12/pr118-merge-20260928T065616Z/`,
    `…/claude-w0-w12/independent-review-pr118-*/` and `…/claude-w0-w12/pr118-review-*`.
- **[#117](https://github.com/neverhuman/JopeDime/pull/117): Python removal P1.**
  - The Python that CI and the engine depend on is ported to Rust:
    - family growth, in `jope-engine`;
    - `jope-ci cargo-artifact`;
    - the four records guards, as `jope-records` verbs;
    - `jope-ci build-report`.
  - 703 files with no live consumer are archived, with a manifest and a custody tarball.
    ADR-0006 records the Rust-only stack.
  - Merge: 50 commits, main `7aec174e055af6ffe41f71446876e0b0e6f632ce`.
  - Reviewed head: `083485a76c1ab13e7ed73ac20f608e6332f6e12f`. The tree is
    `d2e989ce2d00eeb4f061896728e622369d4f94db`.
  - Review: four requests for changes, then approval. Each finding was fixed across its whole
    class, and each fix is backed by differential fixtures captured from the original tools. In
    every harness there are 0 cases where the original refuses and the port accepts.
    - 5332299199 at `a997ab82500ce9708fe88054c8147a38ae4d9bd7`: the coverage validator
      accepted malformed lanes and dispositions.
    - 5333747872 at `e8640b7b07ab7977d20b99b450bd99456e7fe4ca`: POSIX `//` roots and
      Python's 4300-digit integer limit.
    - 5334300806 at `bc22bc6ca229c6f95153e0a8b2372506a0191de1`: two cases the original
      rejects and the port accepted.
    - 5337854418 at `bcdadfc77caf34ef32fd8fe279427ee4b617c909`: inexact arithmetic on
      input-derived values.
    - 5338691773 approved.
  - CI: PR run 36421749004 and manual run 36421748661 passed all nine jobs.
  - After merge: main-guard 36424942633 and full-main-cpu 36424942246 passed.
  - Custody: `…/claude-w0-w12/pr117-merge-20260928T125447Z/`,
    `…/claude-w0-w12/independent-review-pr117-*/` and `…/claude-w0-w12/pr117-review-*`.
- **[#119](https://github.com/neverhuman/JopeDime/pull/119): clean-base part A.**
  - Changes:
    - three unreferenced leftovers and the dead `ops/ci` directory are removed;
    - `actions/checkout` is pinned to v7.0.1 in `ci.yml`, `full-main.yml` and
      `main-guard.yml`;
    - the stash-only ledger block and plan are archived under `docs/`;
    - all 33 feat-math temp-dir test callers now use `tempfile`.
  - `AGENT_CHAT.md` is unchanged.
  - Merge: 10 commits, main `4ff4e7b697dcd5b7097dacbae9e99c07ffef8e28`.
  - Reviewed head: `d582c30bfab7a14dad44cc05764bda680c76e28d`. The tree is
    `2876e389b09e849e0b1246fe80b424f6a418acd4`.
  - Review, two approvals:
    - 5338771284, submitted at 12:42:58Z. It was given at
      `01f3b6f1872eca8ca4281d6b090251090eb2eec9`, before the rebase onto #117; its body names
      that head. The API now reports its `commit_id` as the rebased head.
    - 5339999982, submitted at 14:19:47Z, after the rebased head was pushed (its PR run was
      created at 13:32:47Z). The merge relied on this one.
  - CI: PR run 36429372310 and manual run 36429371243 passed all nine jobs.
  - After merge: main-guard 36435189783 and full-main-cpu 36435189776 passed.
  - Custody: `…/claude-w0-w12/pr119-merge-20260928T141950Z/`,
    `…/claude-w0-w12/independent-review-pr119-01f3b6f18/` and
    `…/claude-w0-w12/independent-review-pr119-d582c30bf/`.
- **[#121](https://github.com/neverhuman/JopeDime/pull/121): D5, all forty synthetic
  fixtures.**
  - The fixtures were regenerated by this tree's generator, not copied. On main, the
    `datasets/synthetic` tree is `43e962350afe29fd1c93c0ba3128bd8bcfc8e58e`, which is the tree of
    Codex's `15a1d3102c7c45e79396749c8ba1fd516cfc34e9`. It holds 40 fixtures plus
    `catalog.json`, whose sha256 is
    `651d28c8456bc160ed33960110142cdb8e9941d2a727348416db999f99d22e93`.
  - The ten real `test-10` datasets are kept, and `MAX_DATASETS` stays 10.
  - `champions --local` and `train-step --local` use one verified snapshot. They search the 8
    supported regression fixtures and report the other 32 as unsupported.
  - No Python was added or edited.
  - Merge: four commits, main `01ef9ddcb31133bea27188e29af9978c2d36b8c4`.
  - Reviewed head: `e0b906b239b107701364adb41d522f7a7c1ff051`. The tree is
    `65d8c3f2ae2e59dbccfa322ebc36d42f558fe947`.
  - Review: approval 5340876669.
  - PR run 36442700858: seven jobs passed; `python` and `emitters` were skipped by path.
  - Manual run 36442700618: all nine jobs passed.
  - After merge: main-guard 36445370527 and full-main-cpu 36445370746 passed.
  - Custody: `…/claude-w0-w12/pr121-merge-20260928T154011Z/` and
    `…/claude-w0-w12/independent-review-pr121-e0b906b23/`.
- **[#120](https://github.com/neverhuman/JopeDime/pull/120): W7, the bounded multi-host
  native training loop.**
  - This lands Codex's `rr2/w7-live` through cherry-picks with `-x`: W7-T01, T03 to T07 and
    W7-D1 to D6.
  - #114's read-once API is kept, and `full-lane.sh` calls the Rust `cargo-artifact` port.
  - No live three-host run is claimed.
  - Review 5339604155 requested changes at `3d12729a2181bcbe5c153770c60095be9c4414fe`: the
    TLS handshake and framed-I/O deadlines were checked only between `complete_io` calls.
    `transport::DeadlineSocket` now applies the absolute deadline to every read and write, and
    the courtesy close is capped at 1 s.
  - The branch was updated without rewriting history. Merge commit
    `1c19341fcd5e387863e0b71aa1aae5814d63dd98` brought in main
    `01ef9ddcb31133bea27188e29af9978c2d36b8c4`, and the corrections were cherry-picked on top.
    The previously pushed head stayed an ancestor.
  - GitHub's rebase merge replayed the 25 non-merge commits of the 26 and dropped the merge
    commit. Main became `4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6`.
  - Reviewed head: `58a919e214b4c46ac6614156d7bc498084cb17b3`. The tree is
    `7c544624efdb3af542a33365a54401ec974bb984` on both sides.
  - Approval: 5341334345, at 16:00:21Z, after the 15:56Z push.
  - CI: PR run 36447376345 and manual run 36447429049 passed all nine jobs.
  - Codex ran the review, CI and merge (AGENT_CHAT 2026-09-28T15:42Z).
  - After merge: main-guard 36449887715 passed. full-main-cpu 36449887665 **failed**. The failure
    reproduced in #122 is 11 tests: 6 in `jope-parallel::bounded_census` and 5 in
    `bounded_smartcluster`. They read the runner container's empty `/etc/machine-id`, and PR
    CI's `rust-test` lane does not run those suites.
  - Custody:
    - `/home/ubuntu/jope-rescue-20260921/pr120-codex-resume-20260928/`;
    - `/mnt/jope-archive/jopedime-pr120-custody-20260928/`, the honesty lane archive;
    - `…/claude-w0-w12/independent-review-pr120-3d12729a2/`;
    - Claude's evidence, in `…/claude-w0-w12/runs/w7-*`.
- **[#122](https://github.com/neverhuman/JopeDime/pull/122): W7's host-identity tests become
  hermetic.**
  - `collect::HostSources` has no `Default`. The product binary hard-wires
    `HostSources::system()`, which reads `/etc/machine-id` with the same strict refusal.
  - The tests go through the example `jope-native-loop-fixture` and checked-in fixture host
    files. No test was weakened or skipped.
  - Test support gained three things: a stale-fixture guard (mtime and dep-info only), coordinator
    diagnostics, and a port reservation outside the ephemeral range.
  - Authorship: Claude wrote eight commits, through `4af8afe7547f7fbdb9b0881742b96778965e04cc`,
    and handed off. Codex added the ninth, which only corrects the freshness guard's scope wording
    in comments and docs. That commit is main `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5` itself.
  - Merge: nine commits, main `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`.
  - Reviewed head: `00541d319493c772c053821ae6b3f8b0462c5423`. The tree is
    `37ce3d5cbfe638bc22bfd6a3eeb0fd1a71ec328c`.
  - Review: approval 5343417155.
  - CI: PR run 36469868551 and manual run 36469978500 passed all nine jobs.
  - Codex ran the review, CI and merge.
  - After merge: main-guard 36472565789 passed, and full-main-cpu 36472565658 passed on attempt 1
    (completed 2026-09-28T20:11:15Z), so the #120 failure is fixed on main.
  - Custody:
    - `/home/ubuntu/jope-rescue-20260921/fix-census-codex-lifecycle-20260928/`;
    - `/mnt/jope-archive/jopedime-pr122-custody-20260928/`;
    - Claude's evidence, in `…/claude-w0-w12/runs/fix-machine-id-*`.
- **[#123](https://github.com/neverhuman/JopeDime/pull/123): clean-base part B.** It completes
  the clean-base plan (D-A) on a main that holds #117 and #119 to #122. Main has 536 fewer
  tracked files: 2,880 at `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`, 2,344 at the merge (557
  deleted, 21 added).
  - Changes:
    - it retires `deploy/kernel-v1` (304 files, among them second copies of `jope-parallel`,
      `feat-math`, `feat-cli` and `feat-core`) and `docs/legacy` (250 files), after custody in
      `/home/ubuntu/jope-evidence/clean-base-20260927/`: a tarball, a bundle restored
      independently with `git clone --no-local --bare` by two verifiers, `SHA256SUMS` and
      `VERIFICATION.txt`;
    - `docs/archive/clean-base-manifest.json` is generated and gives each of its 564 entries a
      `git checkout <sha> -- <path>` restore;
    - `repository = "https://github.com/neverhuman/JopeDime"` in `[workspace.package]`, and the
      `jope` binary with its direct crates as the workspace `default-members`;
    - `actions/checkout` is pinned to v7.0.1 in `weekly.yml`;
    - the `jope-records repo-layout` guard joins the hygiene lane, with a shrink-only allowlist;
    - `check-test-routing.sh` checks both directions.
  - Review 5344298377 requested changes at `fc39cfb59118d8622eccbf58fea45d3c3c249254`: the
    duplicate-package rule read `[package]` names with a regex, so a copy whose name was a
    single-quoted TOML string passed at exit 0, and manifests the regex could not read were
    skipped silently. The fix, two commits added without a rewrite
    (`7982c479fd673084ed1490b3edcd4d72f315544e` and `dd4675c097901ad323323adc7361ef67a6b0ab0e`),
    reads every tracked Cargo manifest as TOML 1.0, adds the rule `bad-manifest` for a manifest
    the guard cannot read, and reports every tracked path that is not UTF-8 (`bad-path`) instead
    of skipping it. Neither rule takes an exception. It added 7 jope-records tests (3,789 to
    3,796 in the inventory).
  - Merge: 12 commits, main `0737da51102ad502676c1b1c3c06549a3ad0442e`, at
    2026-09-28T21:39:07Z.
  - Reviewed head: `dd4675c097901ad323323adc7361ef67a6b0ab0e`. The tree is
    `67677e988df377cfd157219b047875b1d8d508bc` on both sides.
  - Review: 5344298377 requested changes at `fc39cfb59118d8622eccbf58fea45d3c3c249254`, as
    above. 5344815205 approved at `dd4675c097901ad323323adc7361ef67a6b0ab0e`.
  - CI at `fc39cfb59118d8622eccbf58fea45d3c3c249254`:
    - PR run 36479466799 failed on attempt 1. `hygiene` and `changes` stopped at the
      runner-provenance step on the xbabe1 runners, because the job's cgroup limits did not equal
      the approved CPU budget, and `aggregate` failed with them. The PR attributes this to a host
      `systemctl daemon-reload` at 20:03Z that reset the idle containers' swap limit. Attempt 2
      passed all nine jobs.
    - Manual run 36479467996 passed all nine jobs.
  - CI at `dd4675c097901ad323323adc7361ef67a6b0ab0e`: PR run 36485227333 and manual run
    36485226033 passed all nine jobs on attempt 1.
  - After merge: main-guard 36487524917 passed, and full-main-cpu 36487524712 passed on attempt 1
    (completed 2026-09-28T22:24:59Z).
  - Custody: `…/claude-w0-w12/pr123-merge-20260928T213854Z/`,
    `…/claude-w0-w12/independent-review-pr123-fc39cfb59/` and
    `…/claude-w0-w12/independent-review-pr123-dd4675c09/`.
- **[#124](https://github.com/neverhuman/JopeDime/pull/124): the v0.2.0 release tooling,
  CHANGELOG, README and the first Lambda import.** Rebased on #123's merge,
  `0737da51102ad502676c1b1c3c06549a3ad0442e`.
  - Changes:
    - the lock-only dependency bump that was PR #5's intent, on current main: `cc` 1.4.5 to
      1.5.1, `clap` 4.6.6 to 4.6.7 and `crc32fast` 1.5.1 to 1.5.2;
    - `deploy/release/release.sh` (bash and jq) and its refusal tests,
      `deploy/release/test-release.sh`, which runs in the hygiene lane (owner decision D-D);
    - `docs/release/README.md` and the public key `docs/release/signing.pub`, sha256
      `de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e`;
    - `release.yml` made dispatch-only; it fails closed at its first step;
    - version 0.2.0 for the workspace and every product crate;
    - `CHANGELOG.md` for 0.2.0, generated with git-cliff and curated by pull request;
    - the README's 0.2.0 status, stats, releases section and layout, and the lines that #117 to
      #123 and the release made false in `SECURITY.md`, `CONTRIBUTING.md` and `docs/`;
    - the frozen AWS Lambda work of 2026-09-28 under `deploy/lambda/` (the Lambda import below);
    - its MASTER_PLAN record.
  - Review: the first independent review (5345652450, CHANGES_REQUESTED at
    `97d5ffa8336c31468262cf3755bbe53905e39139`) found an unpinned Syft and a missing
    temporary-directory policy across the builder boundary. The commits that follow it pin Syft by
    digest and reported version, add a reported-version check to cosign, whose digest was already
    pinned, require root-owned host tools, push only the release tag, carry `TMPDIR` across
    docker, sudo and `env -i`, and isolate jq, git and the other host tools from per-user
    configuration. Two more requests for changes followed, 5347008178 at
    `e439ae6683d27cbf14f083eb375a0032c364652a` and 5347271804 at
    `23516d0ce4f904987a483415282c5e1576a43721`. Approval 5347539592 was submitted at 03:58:50Z,
    after the last push (03:49:02Z at the latest), at the reviewed head.
  - Lambda import, under the owner's decisions of 2026-09-29 at about 00:15Z: all Lambda Rust
    goes into that pull request and is built and tested by CI; the Lambda Python and other
    untracked orchestration is preserved byte-exact in custody with a tracked hash manifest, and a
    Rust port is a follow-up; v0.2.0 waits for the regression-Lambda v3 freeze.
    - What: `deploy/lambda/csv` (package `jope-lambda-probe`: the single-fit, batch and corpus
      workers) and `deploy/lambda/watch` (package `jope-lambda-watch`, the terminal monitor),
      imported byte for byte from the frozen `.agent/experiments/lambda-csv` and `lambda-watch`
      with the same layout. Each is its own Cargo workspace; the root `Cargo.toml`, `Cargo.lock`
      and `crates/` are unchanged, so the `jope` build inputs are too.
    - Anchors: the monitor's source tarball `d2fda4e7d582a633d4b364ceb96801112e89bbf25139b2328a57826dd9115c86`
      (17 members); the csv survey manifest
      `33d3c18d0b3cca1cf246620946cf818d7466c68c8746ad2119857662ad686413` (68 files; the 37 that
      are also in the corpus run's S3 source bundle `d359ff3e…` equal the restored bundle; COMMIT
      `68ad89ba…`, CLOSE `a9d9bbd7…`); the planning survey manifest
      `f85ebf46bc2189b0f411ecba3839ec6556dc498f31fa454f38807f090cf4ec3f` (14 files; CLOSE
      `83a85ba8…`). `deploy/lambda/SNAPSHOT.json` records all 99 files (52 tracked, 1 gzipped,
      1 renamed, 45 in custody) and 33 excluded entries.
    - Build changes, each recorded in `SNAPSHOT.json`: the nested lock's `feat-math` entry at
      0.2.0 and `license`/`publish` lines in the two nested manifests. `train-jobs.json`
      (8,166,343 bytes) is tracked gzipped and inflated by CI; the historical
      `rust-single-v2` manifest is tracked as `Cargo.toml.v2` (a duplicate package name).
    - CI: `deploy/ci/lambda.sh` in the hygiene lane checks `SNAPSHOT.json`, then runs
      `cargo metadata --locked`, `cargo test --locked` (54 tests: 40 and 14) and `cargo deny` per
      workspace; 42.0 s cold and 7.7 s warm on xbabe2.
    - Custody: 45 Python and companion files in three deterministic tarballs under
      `~/jope-evidence/lambda-20260929/python/` on xbabe2 (`SHA256SUMS`
      `6cb7def8390edb0c31d7bf77d1b1d92692fc83402a9d6f5f6b6e23033418ab58`), re-extracted and
      re-hashed 45 of 45; `docs/archive/lambda-python-manifest.json` lists each one.
    - Records: `deploy/lambda/README.md`, `docs/internal/lambda/README.md`, and
      `docs/coordination/codex-training-support-ledger-20260925.md`, two ledger entries from the
      `codex-training-support` lane's uncommitted `AGENT_CHAT.md`. Their exact heading lines are
      only in that lane; the canonical `AGENT_CHAT.md`, also uncommitted, has the same bodies under
      headings that differ only in the microseconds. Neither ledger is in git.
    - Not imported by #124: the regression-Lambda v3 workers (`regression-lambda-v3` and its
      `-ops` tools), which were still running. #125 imports them.
  - Merge: 23 commits, main `5d29d86e4dd80dd87d074f3d02d82d51923e4c73`, at 2026-09-29T05:12:34Z,
    by the guarded merge script (approval after the last push, both runs green at the head,
    reviewer eligibility, then a rebase merge pinned to the head).
  - Reviewed head: `34c17d031c21486051c3c4f78430d49f019179df`. The tree is
    `b3ddbee3fdf3410b6068f1a782eef7eb24746b9f` on both sides.
  - CI at the reviewed head: PR run 36518812720 and manual run 36518812212 passed all nine jobs on
    attempt 1.
  - After merge: main-guard 36525095424 and full-main-cpu 36525095499 both passed on attempt 1
    (full-main-cpu completed 2026-09-29T05:55:02Z).
  - Custody: `…/claude-w0-w12/pr124-merge-20260929T051219Z/`, copied with the pre-merge checks to
    `~/jope-evidence/closeout-20260929/10-pr124/`.
  - Its record said the regression workers would be imported before it merged. Owner decision O1
    (below) changed that order: #124 merged at its approved head without them, and v0.2.0 is cut
    from #125's merge commit instead.
  - Rehearsal, not the release: a dry run of `release.sh` at
    `5d29d86e4dd80dd87d074f3d02d82d51923e4c73` (nothing tagged or published) passed, and an
    independent verifier passed it. Its manifest binds that commit, the binary prints
    `jope 0.2.0`, its highest glibc symbol is 2.34, cosign verified the signature against the
    committed key, and two tampered copies failed. The evidence is
    `~/jope-evidence/closeout-20260929/12-rehearsal-m1/`.
- **This pull request, #125: the frozen regression workers and the v0.2.0 completion records.**
  On #124's merge, `5d29d86e4dd80dd87d074f3d02d82d51923e4c73`.
  - What: `deploy/lambda/regression` (package `jope-regression-lambda`, the Lambda binary
    `regression-bootstrap`, its five record schemas, their registry and a compatibility fixture)
    and `deploy/lambda/regression-ops` (package `jope-regression-assessment`, the offline
    `regression-assess`), each its own Cargo workspace at the depth of the csv workers. The root
    `Cargo.toml`, `Cargo.lock` and `crates/` are unchanged.
  - Source: read only from the custody copy of the proven freeze,
    `~/jope-evidence/closeout-20260929/21-v3-frozen/tree/`. The freeze proof is
    `~/jope-evidence/closeout-20260929/20-v3-freeze-proof/FREEZE-PROOF.json`, sha256
    `c4a740be1544e7a92758c32fbe9c04c74f244400cf9d382b2379d223c879e63b`: no process referenced
    the two directories in three full `/proc` scans, and two sha256 passes 620 s apart equalled
    each other and the pre-freeze snapshot (52 files, list sha256
    `d2fe056bf3712d3118cd1d3b7d6f24065f320ed24af587a11eb247d34a3093f1`). Codex's stopped-state
    handoff of 2026-09-29T05:04:31Z released the sources for this import.
  - `deploy/lambda/SNAPSHOT.json` now records 148 files (71 tracked, 1 gzipped, 1 renamed, 73 in
    custody, and 2 legacy schemas that equal `schemas/jope.score.v1.schema.json` and
    `schemas/jope.hwm.v1.schema.json`) and 52 excluded entries. The regression bodies' private
    environment files are recorded by existence and sha256 only.
  - Build changes, each recorded in `SNAPSHOT.json`: `license` in both new manifests and
    `publish = false` in the assessment tool's (cargo-deny policy), both new locks' `feat-math`
    entry at 0.2.0 (`lambda-workspaces.sh sync-locks`; no source change was needed), and
    home-relative host paths in four notes.
  - The worker's per-call scratch under `/tmp` is the one exception to the no-`/tmp` rule: it is
    the only writable path in the AWS Lambda runtime. Its tests write nothing there.
  - CI: `deploy/ci/lambda.sh` builds and tests four workspaces, 61 tests (40, 2, 5 and 14).
  - Custody: the 28 Python files in two deterministic tarballs beside #124's, with
    `SHA256SUMS-regression` sha256
    `9e90c7406e02d6b60538b3f2d7e0ea3985c766df462542a5d543d06ab825d819`, re-extracted and
    re-hashed 28 of 28; `docs/archive/lambda-python-manifest.json` gains a row for each.
  - Records: `CHANGELOG.md` (0.2.0 is dated 2026-09-29 and covers #91 to #125), this record, the
    owner's decisions of 2026-09-29, the regression run's closure facts below, and
    `docs/internal/lambda/README.md`.
  - Merge: this pull request's merge commit, `origin/main` immediately after the merge; its SHA
    is recorded in the release readback, not here. No tag exists yet: `release.sh --publish`
    creates `v0.2.0` on that commit. The Python package stays 0.1.0 and unreleased, and no wheel
    ships.

**Owner decisions, 2026-09-29 (answered at about 04:20Z).** Verbatim:

| # | Decision |
|---|---|
| O1 | Merge #124 now at its approved green head. Import the frozen v3 workers in a small #125. Tag v0.2.0 from #125's merge commit, so the release still contains everything. |
| O2 | After v0.2.0, regression v2 comes **before** the three-host demo. Record this in MASTER_PLAN. |
| O3 | Trim v2 to the first live qualification (Q5). Defer P11 (teacher corpus), P12 (operations), the novelty arm and multi-fidelity. |
| O4 | Claude may stop the three idle Codex processes in `codex-training-support`, after re-verifying custody and that they are idle, then remove the lane. |

Standing decisions: Rust in, Python preserved in custody with a tracked manifest; the fast signed
host release; no wheel.

**Priority change (O2 and O3).** After v0.2.0 the next milestone is regression trials v2 (the
regression v2 workplan under an IP-neutral name), trimmed to its first live qualification, Q5
(five stratified views). The teacher corpus (P11), operations (P12), the novelty arm and
multi-fidelity are deferred. The live three-host S3 demonstration comes after it. This replaces
the order #124 recorded: release, then the three-host demonstration.

**Regression trials v2 workplan.** Codex's draft package (28 files) is preserved in
`~/jope-evidence/closeout-20260929/03-workplan/`, mirrored under
`/mnt/jope-archive/jopedime-closeout-20260929/03-workplan/`. Its `MANIFEST.sha256` has sha256
`1d7927263f77a9b0259ac88949521c2fbc35570911000f040fa1576dd91ba1e4`, and its
`PLAN-MANIFEST.sha256` has sha256
`89d789fe6da6d284c1c4b5c62987685b56819b8886fee1e4757384e53fdb8e71`. Its critique and the v2.1
plan are recorded in regression trials v2 PR-1.

**Regression-Lambda v3 run: closure.** From Codex's stopped-state handoff in `AGENT_CHAT.md`
(2026-09-29T05:04:31Z); nothing here was re-measured.
- Terminal state: published. The closeout supervisor wrote `closeout-complete.json` at
  2026-09-29T05:03:06Z (sha256 `9ed8cdf377536c56867467f9f228cbd9072357a8fdb3ed15320fd14cf63a5002`).
  No run process remains, and the reserved concurrency of the four owned functions read back zero.
- Terminal S3 authority:
  `s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/campaigns/regression-20260928-v3/COMMIT.json`,
  VersionId `u6VROvAgGDHlJE220X6Nl._UutWYKA_o`, sha256
  `69643bbf38d9b3becf14df8e52ad100f57651c22a6897ac331f817c2ae1657a3`. The manifest has sha256
  `4a022fccc11f9c7c53bec7966b07a837bc571f63a23d667ca0ed9a0caa9205f4` (VersionId
  `0UxtrH95iVNNfR3CIiAX2hy4pM3W2f5o`, 112,341,749 bytes), and the restore proof sha256
  `42ded421b44e91fd8c28ff6c632ade34779775ad98b43fba5550e086759a650a` (VersionId
  `5LF1jUSqdIcjHPevOEJ3Kr9KlQUhJ6pA`).
- Coverage: 3,533 source views, of which 3,502 were eligible and 31 excluded; 3,476 views have a
  committed round. Zero views reached round 100: the summary records
  `all_views_reached100=false`, with `outer_labels_generated=false` and
  `canonical_teacher_promotion=false`.
- Accounting: 55,109 admitted attempts and 54,852 verified invocation archives (the query closure
  proves 54,852 accepted and 54,852 indexed, as equal sets); 58 attempts remain unknown and
  reserved. The observed Lambda list-price estimate is about $5.96, and the conservative
  reservation, earlier runs included, is $63.37, under the $1,000 controlled-Lambda ceiling. S3
  and logging costs are separate. Accounted experiment storage is 105.8 GB (105,833,397,478
  bytes).
- Its source is imported by #125, above.

Deferred to the next milestones. These are recorded, not dropped:
1. **Regression trials v2** (owner decisions O2 and O3): restarted as small, repo-grounded,
   owner-gated pull requests, trimmed to the Q5 qualification. PR-1 records the plan, its review
   and the owner decisions. Every live stage needs its own written owner admission.
2. **The live three-host S3 training demonstration**, after regression trials v2.
   - First, P6-0 search throughput. #118's run measured about 3 of 8 cores busy; the work is
     parallel folds and steady-state generations.
   - Then the P6-1 to P6-7 host setup.
   - Then windows A to E: 80% CPU search islands on three hosts, 3-GPU training, and S3 progress
     and restore.
   - The live W7 three-host run follows its runbooks under its own admission window.
3. **A self-hosted `release.yml`:** a release lane, the Rust admission hook (ADR-0006 U4) and the
   `jope-ci release` commands. Until then, releases come from `deploy/release/release.sh`.
4. **Python P2 and P3:** the remaining CI tools move to Rust, and then the torch lineage, the
   `python` lane and `pyproject` are retired.
5. **The remaining W-series gaps and open issues:**
   - #48, the C++/JS emit binding;
   - #62, live S3 from the command line. Its last status (2026-09-27) lists the live champion
     pull/push route, the unused `s3` feature of `jope-arena`, a receipted live `jope.hwm/v1`
     round-trip through the `jope-learning` mirror and sync, and where `JOPE_DATASETS_CACHE`
     actually puts fetched files;
   - #47, the gitleaks history scan, whose findings depend on the clone's refs. #114 made the
     hygiene scan cover `HEAD` and all of its history; still open are an audit of the 243
     findings on custody refs that are not ancestors of the source head (225 vendored
     `generic-api-key`, 18 `jope-env-files`) and a scheduled scan of every retained ref;
   - W7 cross-host exchange and continuation;
   - the other open W1–W12 items.
6. **Two W7 test-support items:**
   - the transient `bounded_smartcluster` failure (post2) is not root-caused;
   - the port-reservation window left open by #122 needs a production change.
7. **Smaller items:**
   - the jankurai measurement and its caps;
   - the WP5 cli-split;
   - the 35 `std::env::temp_dir` calls in the test code of 25 files outside feat-math, held on
     the repo-layout R5 allowlist (3 of them in the pinned fit-authority source, which stays as
     it is);
   - the README's generated arena block cites
     `python/src/jope/research/index_combined_regression_evolution.py:183` and
     `docs/learning.md:46` cites `promotion_gate.py:22-27`, both archived by #117. The arena text
     comes from the sha256-pinned `assets/regression/protocol-corrections-tabular-regression-v2.json`
     through the `jope-records readme-benchmarks` generator, so the fix goes through it (PY-P2);
     the generated block is not edited by hand;
   - owner: `docs/runbooks/rotate-credentials.md` gives the OIDC trust policies' `sub` claims
     as `repo:veox-ai/JopeDime:…`, with a `refs/tags/v*` binding for `release.yml`, which no
     longer requests `id-token: write`. Re-check the policies' repository and ref bindings
     against `neverhuman/JopeDime`; they cannot be read from here;
   - untracking `AGENT_CHAT.md`, which waits for the owner and Codex;
   - the Rust port of the Lambda Python held in custody (`docs/archive/lambda-python-manifest.json`),
     and parameterising the xbabe2 paths that the imported Lambda scripts name.

Next allowed action:
1. Cut v0.2.0 with `deploy/release/release.sh` on xbabe2, from a clean registered worktree at
   this pull request's merge commit (`origin/main` immediately after the merge, after a fresh
   fetch): run `deploy/release/release.sh v0.2.0 "$(git rev-parse origin/main)"` (the dry run),
   inspect its evidence, then run the same command with `--publish`. `release.sh` refuses unless
   `HEAD` and a freshly fetched `origin/main` both equal that SHA. Read back the receipts under
   `~/jope-evidence/release-v0.2.0/`: `readback.json` must show cosign `Verified OK` against the
   committed key, `sha256sum -c` passing, the manifest `git_sha` equal to
   `git rev-parse v0.2.0^{commit}`, the binary printing `jope 0.2.0`, glibc at most 2.36 and a
   failing tampered copy. Record them with the main-guard and full-main-cpu runs on that commit.
   The main merge freeze holds until the tag is read back.
2. Close the `demo` and `codex-training-support` lanes under custody: bundle, restore
   independently, then `git worktree remove` without `--force`. For `codex-training-support`,
   stop its three idle Codex processes first, only after re-verifying custody and that they are
   idle (owner decision O4). Then fast-forward the canonical checkout.
3. Clean up the refs: pin the unreachable and dropped-stash commits under
   `refs/preserve/20260929/`, bundle every ref and restore the bundle independently, then delete
   the landed branches and rename remote branches that hold unique commits to `archive/<name>`;
   prune only after that. Confirm that `gh pr list --state open` is empty and that
   `git log --all --not --remotes` shows only custody refs.
4. Regression trials v2 PR-1: record the plan, its review and the owner decisions, after Codex
   hands off the workplan package or the owner assigns it. Each later PR, and each live stage
   with its own written owner admission, follows one at a time. This record does not start them.

**2026-09-27 W1–W12 source gap closure.** [PR #114](https://github.com/neverhuman/JopeDime/pull/114)
landed the remaining source- and fixture-scope W1–W12 work on top of #111 and #113:
- W1 input boundaries;
- W2 semantic manifest, typed context and authenticated bytes;
- W3 revocation consumers, the authenticated W2 handoff with its producer side, requalification
  refusal, fitted-artifact binding and HWM recovery;
- W4 per-draw outcomes;
- W5 frozen inputs and evaluator process groups;
- W8 publication, reconcile and S3 preflight;
- W9 package seal and install;
- W12 claim inventory, conformance pack and duplicate-cell refusal;
- the never-loop Hyperion fix;
- the owner's conservative policy defaults, `OWNER-DECISIONS.md` D1–D20.

It merged by rebase, 45 commits, at 2026-09-27T09:39:08Z as main
`1b301a9c52b8ef409a85e76d878a0bbcb6bb0f40`. The merged tree is byte-identical to the reviewed head
`9dc747155a34f1f5194ded2a5abb9c11899d06a4`.

Review was by independent Codex processes as `neverhumanbot` (219957145); every commit author and
committer is `jepsontaylor`:
- Changes were requested at `2160a28c9`: context substitution, a later context window, S3 URL
  normalization, and a commit whose email mapped to the reviewer, re-authored with its tree
  unchanged.
- Changes were requested at `09b9676b3`: HTTPS S3 routes.
- All findings were reproduced first and then fixed.
- Approvals followed at `94b1616fb`, `4d475b320`, `31383aa48` and `93bfcb956`, and finally review
  5329763785 at `9dc747155`.

PR run 36309043047 and manual run 36309054082 passed all eight jobs and aggregate on attempt 1 at
that head. It closes #57 and #58. #47, #48 and #62 stay open with the reasons in
`docs/internal/core-ip/findings/pr114-issue-followups-20260927.md`.

Earlier exact-head attempts failed only on GitHub network steps, for two causes:
- Runner containers resolved through the LAN router, which failed 18 of 90 lookups from xbabe1.
  This is fixed on the xbabe1 and xbabe2 hosts: resolved now serves `docker0`, and receipts are in
  `~/jope-evidence/runner-dns-fix-20260927/`.
- Intermittent SYN loss on the LAN/NAT path. The final commit gives `changes`, `hygiene` and
  `aggregate` room for one stalled connection and retries the aggregate download at most three
  times. No gate or pass criterion changed.

Task status: 92 tasks, of which 70 are implemented, 67 fixture-proven, 35 reproduced and 22 blocked.
None is CI-verified or live-qualified, and all 195 findings remain open. The blocked tasks are the
live gates W7-T07/T08/T09, W8-T07/T08, W9-T07, W10 (except T09), W11-T02/T05–T08 and W12-T06/T08.

Custody:
- `/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/pr114-merge-20260927T093857Z/`
- `independent-review-pr114*/`
- `runs/`

Owner decisions of 2026-09-27:
- **D-A:** Codex lands the queued source branches.
- **D-B:** the Python policy-v3 training lineage is retired in favour of the native trainer.
- **D-C:** the following infrastructure actions are authorized, each logged with receipts and
  reversible:
  - the scoped S3 role `JopeDimeW8Publisher`, which now exists and is verified;
  - root host setup on xbabe1–3;
  - retiring xbabe2's out-of-repo qualification daemons, which is done;
  - pausing competing workloads in admitted windows.
- **D-D:** the next release is v0.2.0, from the self-hosted runner with a repo-held signature.

Next allowed action: land the queued branches in order, each rebased on `1b301a9c5` with records
regenerated:
1. `health/full-main-green`;
2. `rr2/w7-live`;
3. `rr2/python-p1`.

Under D-A the owner changed the PR-first order in writing: Claude proceeds in parallel, on paths
disjoint from those branches, with the following work, each as its own reviewed PR:
- clean the base;
- run and fix jankurai;
- finish removing Python;
- close the remaining source gaps;
- the targeted GP evolution demo;
- the live three-host S3 training demonstration;
- the v0.2.0 release.

This record does not start any of them.

**2026-09-26 core-IP regression remediation.** [PR #111](https://github.com/neverhuman/JopeDime/pull/111)
delivered W0 and the recorded source- and fixture-scope slices across W1–W12 of
`docs/internal/core-ip/work-orders`. Source integration items and live gates remain open;
`STATUS.md` records them per task.
It merged as squash `7b86622ea694bc069338c268f23e17c29f76e8eb` at 2026-09-26T15:12:59Z. The tree is
byte-identical to the reviewed head `bddb8fed16e4c73cb0277998f37ab4a8871e6a3e`, and main-guard
36251156798 passed.

It was squashed, not rebased, because GitHub refuses rebase merges of more than 100 commits (121
here, `canBeRebased=false`). The squash body lists every commit. The remote head branch was
deleted on merge (`delete_branch_on_merge`), but the full history stays reachable at GitHub's
`refs/pull/111/head` (= `bddb8fed1`) and in the local branch `feat/regression-remediation`.

Review was by independent Codex processes acting as `neverhumanbot` (219957145), against `jepsontaylor` as author
and committer of all 121 commits:
- Review 5326209256 requested changes at `0b303b293` with two P1 defects: grant expiry bypassed
  during coordinator recovery, and an outer-score grant that re-read its preparation after
  verifying it.
- Both were reproduced and fixed, together with sibling gaps, in `4ac185bb7`, `5ad932450` and
  `bddb8fed1`.
- Review 5326304313 approved `bddb8fed1`.

PR run 36250125057 and manual run 36250695256 passed all eight jobs and aggregate at that head.

Task status is 70 implemented, 66 fixture-proven, 22 blocked and 0 planned. None is CI-verified or
live-qualified, and all 195 findings remain open. The blocked tasks are live or dependent release
gates, for example three-host W7, live S3 W8, W9-T07, loop/transfer W10, GPU timing W11 and
W12-T06/T08. Per-task state and open items are in
`docs/internal/core-ip/work-orders/STATUS.md`.

The PR added no Python and deleted seven files. The core-IP verifiers now live in the Rust crate
`jope-records`. The owner's repo-wide Python migration go-ahead is pending (owner decision 10 on the
PR).

Custody for the landing:
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/`, specifically
`pr111-lifecycle/`, `independent-review-pr111*/` and `runs/`.

Next allowed action: close the lanes that served this PR (`bench`, `demo`, `honesty`) under the
custody rules. Then take the recorded owner decisions and the live-gate prerequisites; this record
does not start them.

**2026-09-22 CPU foundation delivery.** Publication repo: private `veox-ai/JopeDime`.
Issue #52 is closed by [PR #55](https://github.com/veox-ai/JopeDime/pull/55).
Its source head `c5a5b1138b4e878327eb2f429ac75be9b8a9374c` received independent
Claude review and native `neverhuman` approval after verifying `jepsontaylor` was
both PR author and every GitHub-associated commit author. All eight manual jobs
passed in [run 35756345708](https://github.com/veox-ai/JopeDime/actions/runs/35756345708).
Rust executed 751 passing tests with 13 nextest skips; Python passed 207 with 15
deselected. JUnit omits those Rust skips. These are limited gate results, not full
CPU qualification. PR #54 was closed as superseded; its discussion is retained.
PR synthetic merge refs and source-head dispatch evidence are recorded separately.

The guarded rebase merge produced remote main
`8eaafa3915c937c49581191fd00680c9d3f01cc0`; main-guard
[35757914440](https://github.com/veox-ai/JopeDime/actions/runs/35757914440) passed.
Canonical returned clean to that main before claiming `fix/cargo-artifacts-46`.
Lifecycle evidence is outside the checkout under
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/foundation/issue52`.
Writer access is recovered. Protection API still returns 403; enforcement is procedural.

Issue #46 is closed by [PR #56](https://github.com/veox-ai/JopeDime/pull/56). Local proof at
`96ec9a996` passes both the canonical reused `target/` and the registered
`artifact-proof` lane's stable external target directory with spaces. Each layout
passed nine artifact protocol tests, twelve root tests plus one doctest, emitter
parity and pipeline smoke, and a cached Cargo build reported `fresh: true`.
The old external-root test failed at main before the change. Logs, input hashes,
commands, statuses, environment and 16 GiB/16 CPU budgets are retained under
`foundation/issue46/{normal,external}` in the same custody tree. An initial scratch
fixture placed under the repository marker failed its no-marker precondition;
that failure is preserved separately. Final source `161ddeb58b2fcf1768651f8dc5b7aff6ebdc61ac`
passed all eight manual jobs in run35760152193 and all PR jobs35760158987. Independent
Claude/native neverhuman approval5281400221 binds that head. Main is now
`c7ae554a85f0e2d35d282aa87ee72a539d6cb4c2`; main-guard35761248043 passed.
Nextest normal/external proofs passed at the final head after correcting its
workspace-relative default report store and rejecting stale reports. The proof lane
and external build were archived, independently restored/verified, then removed
with fresh dependency checks. See `foundation/issue46/lifecycle-receipt.json` and
`proof-closure.json`. Canonical returned clean to verified main before this phase.
GPU/S3 regression and release/container wrappers have static validation only here;
their required capabilities are not claimed as exercised by CPU smoke proofs.

Coverage foundation [PR #59](https://github.com/veox-ai/JopeDime/pull/59) merged from
`a13f8cee4fa10cbde4b56305a2e6d842f089d749` after all eight manual35763473661 and
PR35763657978 jobs passed and independent Claude/native neverhuman approval5281705335.
Main `aa1e5986824b41632fce482bb06a64c02a21d65d`, main-guard35764421202 passed;
canonical returned clean before the next claim. Custody: `foundation/coverage/lifecycle-receipt.json`.
The review corrected a feature-gated target wrongly assigned to default commands.

Aggregation [PR #60](https://github.com/veox-ai/JopeDime/pull/60) merged from
`f08bcb6486fff1c6472a5a07224d110444853b86` after both nine-job manual35767660672 and
PR35767666966 runs passed, including aggregate, and independent Claude/native
neverhuman approval5282087456. Main is `8ea1289fec019caa21facda4dea26bfb328bbbe8`;
main-guard passed (exact receipt: `foundation/aggregation/lifecycle-receipt.json`).
Canonical returned clean before claiming `ci/full-main-cpu`. The first aggregate
correctly stayed red when nested product receipts were misclassified; that defect
was fixed and the real failed-run artifacts replayed before final hosted acceptance.

[PR #61](https://github.com/veox-ai/JopeDime/pull/61) delivered the CPU foundation from
exact source `ac26f6596e24a2af6e627b87e87d9bd18a90514e`, independently reviewed by
Claude with native neverhuman approval 5282585692. Manual 35773238753/1 and
PR 35772555348/2 passed all nine jobs. Guarded rebase merge produced
`fdb2e8e3e3d152590d9246b5226eca417f5c6cbc` with the identical reviewed tree.
The preliminary genuine docs-only run 35769025859 passed changes, hygiene and
aggregate with six declared path exclusions. Failed pre-deployment provenance runs
and the first aggregate failure remain evidence, not superseded success claims.

The reviewed runner source was deployed after draining owned listeners. Actual readback
verified two fast slots on xbabe1 at 16 GiB/16 CPUs each and one exclusive full slot on
xbabe2 at 48 GiB/16 CPUs, no swap, separate PR/trusted-main work and cache volumes,
read-only root/policy/hooks, no GPU/Docker/PAT/models mounts or training/publishing/S3
credentials. Other JopeDime roles are disabled; the held 24 GiB/28 CPU proposal was not
used. Provenance remains mandatory. Full jobs have 120 minutes and at least 50 GB disk
reserve. Main-guard shares the full slot serially, so it can queue behind the baseline.
See the [runner runbook](docs/runbooks/runner.md) and custody `foundation/full-main`.

Current static discovery at that source assigns 22 packages, 18 features, 214 Cargo targets,
218 Python test-named files, 6 workflows/16 jobs and 32 benchmark sources. These are discovery
counts, not tests executed. Fast CI executed 54 protocol tests, 751 nextest passes with 13 skips
from 764 listed, 207 Python passes with 15 deselected from 222, four systemd tests and four
emitter tests with 278 filtered. Counts overlap across invocations. Non-test commands
have unknown/not-applicable counts, never invented zeros. Collection, execution and
explicit deferred/research dispositions remain distinct in
[coverage/receipt semantics](deploy/ci/coverage.md).

The first push-triggered full-main run,
[35775436225](https://github.com/veox-ai/JopeDime/actions/runs/35775436225), attempt 1,
completed at `fdb2e8e3e3d152590d9246b5226eca417f5c6cbc`. All11families produced fresh,
verified receipts and a retained inventory: eight passed, two failed, one unavailable.
The overall job correctly failed. Downloaded artifacts replayed through the verifier
with matching evidence digests; no missing/invalid family results were accepted.

| Family | Observed outcome at that main |
| --- | --- |
| Default workspace | 2,523 passed, 29 failed, 37 skipped; 2,589 listed; exit 100 |
| Maintained Python | 207 passed, 15 deselected; 222 collected |
| Doctests | 8 passed |
| Scratch feature variants | 1,879 passed, 30 failed, 16 skipped; 1,925 listed; ten-minute family limit, exit 124 |
| Policy/testing | 23 passed, 5 skipped; 28 listed |
| S3 feature, offline only | 34 passed; no live S3 credentials or qualification |
| CPU-neural | Unavailable, exit 78: configured `/usr/include/cuda_runtime.h` missing; counts unknown |
| Emitters | 4 passed, 278 filtered plus emitted-code parity |
| Input verification and pipeline smoke | Both passed; non-test counts not applicable |
| Build artifact measurement | Passed; initial artifact `fresh:false`; subsequent cached reuse is a post-merge acceptance check |

The scratch timeout interrupted `feat-math::golden fast_profile_portfolio_head` after
466.505 seconds of that test; compilation and the rest of the family used the remaining
budget. Counts are per invocation and overlap. Ignored/deselected tests, research scripts,
GPU and live-service dependencies remain explicit non-execution in the mapping/inventory.
Main-guard35775436296 passed at that exact main. Negative admission run35775538345
proved rejection in both domains: actual Worker hook invocation followed by exit137
in0.467seconds (fast) and0.674seconds (main), no bypass-step execution, and no OOM,
service stop or timeout explaining the exits. Installed hashes/policies and deterministic
authorization replay bind the inference; denial stderr itself was lost with the Workers.
The expected-rejection run was cancelled after preserving both proofs; it is not passing CI.
The workflow runs on every main push and manual dispatch of main, and remains diagnostic.

Proposed next repair PRs, derived from the 29 default failures and the other family outcomes:

- **Repair campaign spec and fixture discovery**: four spec-drift failures and nine missing
  or host-dependent fixture/module/catalog failures. Preserve dataset identities and sealed
  input boundaries; do not copy research data into the learning worker's input closure.
- **Repair campaign status and UI assertions**: three failures in island counts, selfplay
  alarms and requeued-attempt history.
- **Repair engine cache, history and queue contracts**: five failures in native cache
  accounting/invalidation, displayed learning metrics and unsupported-task queue handling.
- **Make fit-authority test setup explicit**: eight failing engine fit/view contracts;
  preserve the production fit fence and require valid test authority rather than bypass it.
- **Bound the scratch-feature portfolio proof**: investigate the measured timeout and
  preserve the approved 120 minute full-job/resource limits when revising family budgets.
- **Repair CPU-neural SDK discovery**: reconcile the declared CPU feature's header/runtime
  prerequisites; an unavailable SDK is not a passing test result or GPU qualification.

These are measured failure groupings, not yet established root causes. The held lint patch
belongs to the later lint repair PR; reproduce its warning census before strict enforcement.
Custody includes `push-35775436225-verified.json`, `default-repair-observations.json`,
exact source reviews/CI/deployment fingerprints, and the original failed logs.

Full Rust/Python repairs, strict Clippy enforcement, persistent managed runners and
mandatory full-CPU gates remain the following phase. GPU/S3 qualification and learning
campaigns remain outstanding. Qualification and soak clocks are unstarted. The approval
procedure and verified credential paths are now explicit in
[AGENTS.md](AGENTS.md#approval-procedure-on-this-host).

Post-merge completion is recorded outside the checkout at
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/foundation/completion.json`,
including the final main SHA, documentation review/CI, a subsequent job's fresh cache-reuse
receipt, and the clean canonical handoff. This source record does not preclaim those outcomes.

Stopped-head handoffs preserve `runner-prov` at
`b8acc6202824698e6480ae2fa92f1a89a6e04626` and `lint` at
`187c2f14dfca1c10d6670b81e6db7cebbe7fcc6a`, with existing authorship.
The 24 GiB/28 CPU runner proposal is withdrawn as written and remains undeployed.
The lint patch stays deferred; its reported 388 warnings require reproduction.
The deployed launcher now rejects unrecorded effective resource overrides; actual
container resources are verified by provenance. Handoffs: `foundation/held-lanes-handoff.json`.

PR49's actual scheduled report succeeded at **04:00–04:02 UTC**, with32 roots,
six unchanged installed fingerprints and report-only behavior. Independent Claude
review approved exact head `252d466a3d3c48bce2ee2432fb198eeda9f249ca`; all eight
manual CI jobs passed, and main-guard passed at the merge commit. The final immutable
receipt is `/home/ubuntu/jope-rescue-20260921/completion-20260921t201512z/final-merge-readback.json`,
linked in [the PR closeout](https://github.com/veox-ai/JopeDime/pull/49#issuecomment-5771273200).
Historical preservation limitations remain explicit in the cleanup record.

Other sessions merged #44/#50/#51/#53. This session does not claim their complete
independent-review or lane-evidence preservation lifecycle. #44's observed
cross-cut budget defect is tracked by [issue52](https://github.com/veox-ai/JopeDime/issues/52):
a four-attempt budget produced ten attempts on each of three cuts. The current
focused correction reserves comparison work within the total allowance, records
phase costs and requires complete cut coverage before selecting a program.
Implementation and qualification are separate; see [the contract](docs/pipeline.md).
Evidence lives under `/home/ubuntu/jope-rescue-20260921/implementation-20260922/pipeline-budget`.

No learning qualification or soak has started. The remaining delivery sequence is
CI coverage/aggregation/full-main, deterministic security/#47, emitter binding/#48, dataset/fold
contracts, S3 conditional recovery, benchmark/task adapters, isolated GPU execution,
canonical training/resume, successor proposals/promotion and the bounded controller,
followed by the one-hour qualification and24-hour soak. Ownership is recorded in
[AGENT_CHAT.md](AGENT_CHAT.md). The Bulletfarm-inspired contract improvements are
merged in49; its private tools and conflicting clone/build policies were not imported.

## Queue

PRs **#5–#16 were closed unmerged** on 2026-09-20 (administrative withdrawal,
not a merge or supersession). Source branches are retained. The full-history
bundle and per-PR repair checklists live in the builder's
`jope-pr-resolution/20260920` evidence tree (bundle SHA-256
`3bafaa52b9a2c6a6996a2dacc22b9c1bff9248b7833bf07d7cd9882fbcba978e`). See that
directory's `REPORT.md`.

PRs **A–F** ([#17](https://github.com/veox-ai/JopeDime/pull/17)–[#22](https://github.com/veox-ai/JopeDime/pull/22))
are merged, and so is everything through #39, including #34 — see the landing board. **Zero open PRs is not “done.”** Recover selected work **one PR at
a time**. Do not reopen the twelve-PR pile. Live S3 is not claimed. Bound
`jope-policy` checkpoint training is not claimed.

## ASM demonstration delivery — 2026-09-22

The latest explicit user direction is the private [ASM demonstration](docs/asm.md):
a fixed ten-regression/ten-binary/ten-multiclass panel, fold-zero screening and top-five
recipe confirmation over five development folds. Preserve labels, private discovery
and untouched final evaluation. The full acceptance list and unchanged runtime limits
are recorded there; wins are an empirical target, not an implementation claim.

Prerequisites PR63/65 are merged with exact-head full CI, independent native review
and successful main-guard. Canonical was verified clean at remote main
95e92af34fc95854f102965b3f4fbf01d08beeb7; four extra lanes were preserved and removed,
leaving one registered checkout. Lifecycle/custody receipts live under
`/home/ubuntu/jope-rescue-20260921/asm-demonstration-20260922`.

Current focused branch: `feat/asm-experiment-contract`. Work in progress: task refusal,
capability inventory, experiment/evaluation contracts and frozen input panel. Execution,
training and fitted export qualification remain subsequent serial PRs. The panel and
training clocks have not started; a source implementation is not a measured win.

## Product target

```text
verified training inputs → encoders → neural GP proposals + GP exploration →
  measured programs and bound emitted code → durable S3 targets and comparisons → immutable corpus →
  policy training → export and qualification → next bounded GP generation
```

Git holds tiny fixtures only (D5: the ten licensed test datasets from `wpd1`,
≈1 MB, and the forty synthetic fixtures from Codex `15a1d31`, ≈0.8 MB, are both
admitted). Every other dataset and all weights stay in S3.
This is the requested product goal, not a claim that the loop is qualified.
Preserve dataset labels and private admission; keep outer test folds and sealed
meta-test families outside learning. Accepted serving state changes only through
its bound promotion protocol. The existing centroid router is a separate baseline.
The Sep22 user-approved plan freezes the encoder for the first regression experiment.
The policy proposes GP programs; a held-out SOTA claim requires separate frozen evaluation.

## Serial recovery (Grok's lane, one open PR at a time; now merged through F)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B | Ten Git fixtures + offline verify | Merged #18 `049410f` |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Merged #19 `089c127` |
| D | Search/fit/emit CLI on feat-math | Merged #20 `8326c36` |
| E | Local champion bank; live S3 fail-closed | Merged #21 `86c88bd` |
| F | Local 128-d centroid router train-step + `jope route` | Merged #22 `1f0fc5a` |
| G | Live S3 champion pull/push | #34 merged `6a85fa398504f99e7787a58d7d3efd623bb6100d`; pull/recovery and server-enforced conditional replacement remain follow-ups. [The mirror implementation](crates/jope-learning/src/mirror.rs) records its compare-then-put limitation. The merge alone does not prove live S3 restoration. |
| H | Docs/ADRs after the above are true | Not started |

Grok's lane deferred: Dependabot #5, workflow copy (#7), 365-file Python dump
(#8), timings (#10), census regen (#12), path mega-move (#13), LibTorch (#14).
Those items are queued on the integrator's landing board below (it holds the
`workflow`-scope PAT and the AWS credentials that blocked them here); none is a
wholesale import — each lands as its own reviewed PR. The deferred forty-fixture
import is on that board as `rr2/d5-forty-fixtures`. It regenerates the thirty
missing fixtures instead of importing them wholesale, and it keeps the ten
`test-10` datasets (D5).

## Blockers (do not fake)

- `full-main-cpu` runs on every push to `main`; its first success came after #116 (see the
  2026-09-28 record). The `xbabe2-gpu` runner that `weekly.yml` targets is not provisioned.
  Keep runner credentials separate and preserve independent reviewer eligibility for each
  subsequent PR.
- Private-repo rulesets API returns HTTP 403 until the org is upgraded (D3,
  owner action in progress). Until then the merge rule below is process-only.
- No `--admin` merge. Worktrees only in `.agent/worktrees/`, at most four,
  with the current serial queue policy in [AGENTS.md](AGENTS.md). No credentials in git.

Coordinate in [`AGENT_CHAT.md`](AGENT_CHAT.md).

## Owner decisions — 2026-09-21

D2–D5 remain in effect. The Sep22 user-directed continuation clarifies D1 to
check both PR and commit authors and uses one implementation PR at a time;
the current operating rules are in [AGENTS.md](AGENTS.md).

| # | Decision |
| --- | --- |
| D1 | Agents merge after exact-head CI and independent review by an identity different from both PR and GitHub-associated commit authors (`jepsontaylor` / `neverhuman`, when eligible), with guarded merge and remote `main` readback. |
| D2 | PR gate at `70506066` = the seven jobs `ci.yml` defines: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python` (#38), `rust-test` (#39) — all path-gated. **`rust-check` runs `cargo check --workspace --all-targets --locked` plus a systemd test and an MSRV check; it runs neither clippy nor rustfmt** — those exist only in the local `justfile`, which is why #37 had to land rustfmt drift that no lane would have caught. Full nextest, LibTorch `features`, sanitizers, security, artifacts, Jankurai = visible non-blocking main/nightly lanes, each promoted to blocking once it passes on main. Jankurai is a no-regression ratchet, never a merge gate. |
| D3 | Owner upgrades `veox-ai` to Team; branch protection (required checks + 1 approval + no force-push) is enabled the moment the API stops returning 403. |
| D4 | `veox-jopedime-{datasets,models,learning,archive}-use2` (+usw2 replicas) = product buckets the crates target. `jopedime-714044927654-us-east-2-{dev,prod,sealed}` = preservation custody. Nothing deleted; `catalogs/s3.json` records both. |
| D5 | Both the ten real test datasets (`wpd1`) and the forty synthetic fixtures (Codex `15a1d31`) are admitted to Git. |

## Merge rule (per PR)

1. Required contexts = every job that `.github/workflows/ci.yml` defines **at the reviewed head** (on `main` at `3e061605`: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python`, `rust-test`, `e2e-smoke`). A lane legitimately excluded by its declared path matrix is recorded as `skipped-by-path`, not green. Every required execution must pass at the exact head; absent, cancelled or failed is not green. The configured jobs are not proof of execution. Heavy functional lanes become required after their successful main qualification; Jankurai remains informational under D2.
2. Review must come from a session that did not write the PR and a GitHub identity distinct from both the PR author and every GitHub-associated commit author. Follow the PR-scoped identity checks in `AGENTS.md`; check `gh api user --jq .login` in the posting environment. #40 and #41 were opened under the wrong account and replaced by #42 and #43. An agent-operated approval proves an independent read at the exact head, not human sign-off. Trust-surface PRs (workflows, lockfiles, credentials, protection manifests) also receive a security review.
3. `gh pr merge <n> --rebase --match-head-commit <reviewed-full-sha>`, then verify fresh remote `main` and `main-guard`. Preserve shared source-branch history when updating its base.
4. Resolve the existing queue, then one implementation PR at a time. Fix review findings on its existing branch; record explicit replacement/disposition for superseded work. The four managed-lane limit does not authorize four concurrent implementation PRs.

## Landing board (Claude integrator; old PR → new PR → merged SHA)

Open rows are pinned to the head and review they describe, exactly as merged
rows are pinned to a merge SHA. A row about a moving PR is unfalsifiable and
goes stale within the hour; live state belongs on the PR, where GitHub is the
source of truth.

A merge and its landing row land in the same push, and any change to this file
re-reads `git ls-remote origin main` and `gh pr list --state merged` at **push**
time rather than draft time. This revision exists because #42 merged two
minutes before the previous one was pushed, so a board opened to fix a stale
readback shipped one of its own. Neither command takes a second, and neither
can be satisfied by remembering.

| Old PR | Branch | Head | New PR | Merged |
| --- | --- | --- | --- | --- |
| — | `coord/integrator-lane-20260921` | | #23 | `4753a9a` 04:06:47Z |
| #7 | `ci/lanes` | 38aeaa1 | #24 | `932f7b7` 04:06:52Z (rebase; + inputs lane; review found the strict-parity step ran 0 tests — fixed) |
| #10 | `wp1/test-timings` | 5f66e3f | #29 | `0c14405` 05:08:29Z (rebase; 3 nextest exclusions module-qualified after review) |
| #14 | `wp8/neural-receipt` | 44501a9 | #25 | `dc5ccf9` 2026-09-21T03:58:58Z (rebase; reviewer `neverhuman`) |
| #12 | `wp10/parity-census` | c483544 | #28 | `d1317a1` 04:16:19Z (rebase; + rows for the #19 Hyperion bins) |
| #9 | `wp7/model-manifests` | f62095c | #26 | `44443ab` 04:08:07Z (rebase; manifests byte-identical to S3) |
| #6 | `wp3/hyperion-policy` | a043f8b | — | recovered by Grok's #19 (`089c127`): `main` is a strict superset; no PR needed |
| #11 | `wpd1/datasets` | d672e76 | #27 | `a6df267` 04:44:49Z (rebase; object_store 0.14 for RUSTSEC-2026-0194/0195; scoped webpki-root-certs exception; lock additive) |
| #15 | `wp7/models-cli` | b497325 | #33 | `5c66edb` 04:54:50Z (rebase; `jope models`/`datasets` umbrella on Grok's verbs; blob-lock deadlock fixed) |
| #8 | `wp6/python` | d49662f | #31 | `99981e6` 04:42:33Z (rebase; ruff format + pidfd fallback) |
| #13 | `wp4/paths` | 96a61c8 | #32 | `ff5da5b` 05:04:22Z (rebase; abs-path hygiene step now blocking for crates/deploy/.github) |
| #16 | `wp9/docs-release` | 9bec536 | #30 | `aa57388` 04:33:58Z (rebase; README kept as Grok's honest front door, quickstart → docs/quickstart.md, status banners) |
| — | `wp12/arena` | 53f5ec4 | #35 | `53ea45d` 08:21:44Z (rebase; rustls provider guard fails closed, ureq 3 proxy/redirect deltas disclosed) |
| — | board | | #36 | `ebf055d` 06:36:46Z (landing rows through #33) |
| — | rustfmt drift | | #37 | `dc46e86` 06:47:00Z (two drifted hunks in `jope-engine eval_registry/query`) |
| — | python lane | | #38 | `d43bc5f` 07:49:28Z (fixes the systemd test #32 broke; **adds the hosted `python` lane that would have caught it**) |
| — | `wp11/learning-store` | 44aa4b69 | #34 | `6a85fa398504f99e7787a58d7d3efd623bb6100d` 2026-09-21T17:30:58Z; remaining live-recovery limits are recorded above |
| — | rust-test lane | | #39 | `7050606` 11:11:35Z (4 rounds; the hosted gate ran no Rust tests beyond two `feat-math` targets, and the scope guard itself failed open twice before the reviewer's negative probes closed it) |
| — | ADR-0003 test paths | | #42 | `5feb2fa0` 11:50:13Z (three tests red since the ADR-0003 relocation, fixed; quarantine 15 → 12 and the misfiled `queue.rs` panic given its own cause. #40 was the same branch opened from the wrong identity and is closed) |
| — | `wp13/pipeline-run` | 16ed3505 | #44 | `584112293221563aa1acfe1b9a06bf3c987f1e6f` 2026-09-22T03:11:35Z; CPU GP pipeline. Codex owns budget/selection correction [#52](https://github.com/veox-ai/JopeDime/issues/52) and independent validation of the pipeline/CI assertions after #49. Artifact binding #48 remains open. The final review disclosed author involvement in two CI commits; this readback does not certify full review independence. |
| — | `wp5/cli-split` | d0b4819 | | needs a mechanical re-do on current main, not a replay (#32 rewrote 67 of the files it moves) |
| — | `wp8/runner-lanes` | 51857841 | #45 | `c572f1778562768ebc7c29fbfeff047f871cba6c` 2026-09-21T16:55:46Z |
| — | strict Python parity prerequisites | d76cdf1e | #50 | `baa4c1f24e7b2ab7f38af5061feea6029d27cc94` 2026-09-22T03:10:24Z |
| — | generated README benchmark results | 9e44a206 | #51 | `25242513173e1a8b01f7d1e9f73775b7df14110e` 2026-09-22T03:20:41Z |
| — | feat-math Clippy cleanup | c0e8d473 | #53 | `3e0616057bc3234d568deceb884ea1c69ea8451b` 2026-09-22T04:09:11Z; external merge, remaining workspace lint work still requires its own measured evidence |
| #5 | dependabot cargo group | 1151493 | the v0.2.0 release PR | its merge commit, whose SHA the v0.2.0 release readback records: #5's intent re-cut on current main as a lock-only bump (`cc` 1.4.5 to 1.5.1, `clap` 4.6.6 to 4.6.7, `crc32fast` 1.5.1 to 1.5.2), one commit of the release PR |
| — | `rr2/d5-forty-fixtures` | e0b906b239b107701364adb41d522f7a7c1ff051 | #121 | `01ef9ddcb31133bea27188e29af9978c2d36b8c4` 2026-09-28T15:40:24Z (rebase; four commits; reviewed tree `65d8c3f2ae2e59dbccfa322ebc36d42f558fe947` equals the merged tree; approval 5340876669; PR run 36442700858 and manual run 36442700618 green; the forty fixtures, 8 supported and 32 reported unsupported; the `test-10` datasets are kept) |

Recovery of #5–#16: bundle `pr-5-through-16.bundle` (sha256 3bafaa52…) is also
at `s3://veox-jopedime-archive-use2/git-bundles/20260921/`.

The #44 review disclosure and checks are retained in
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/queue/external-merges-20260922.json`;
the #52 acceptance is linked in its row. Corrective validation cannot retroactively
make the earlier merge independent.

## What the gate actually proves (2026-09-21, measured)

Read this before trusting a green check. Every number here names where it was
measured; a figure without a source is a defect in this section.

- Until #38 and #39 the hosted gate ran **almost** no tests: `hygiene`,
  `rust-check`, `emitters`, `inputs`. The one exception is real and worth
  naming rather than rounding away — `emitters` ran `cargo test -p feat-math
  --lib -- cxx::` and `--test portable_parity`. Nothing else. No other crate's
  tests and no Python ran at all, so a green PR meant "it compiles and the
  emitters agree". #38 added the `python` lane, and exists *because* #32 broke
  an existing Python test that nothing ran. #39 added the Rust half.
- **`rust-check` is `cargo check`, not a lint gate.** It runs `cargo check
  --workspace --all-targets --locked`, the `systemd_unit_paths` test and an
  MSRV check. Clippy and rustfmt live only in the local `justfile`, so
  formatting drift reaches `main` unchallenged — #37 is exactly that, landed
  after the fact. Adding `cargo fmt --all --check` to the gate is cheap and
  owed, and is already written on `ci/rust-test-followup-20260921`
  (`f1cbf820`) awaiting a slot. Clippy cannot join it: `cargo clippy
  --workspace --all-targets --locked -- -D warnings` at `70506066` on this
  host fails in `feat-math` — **31 errors in the lib, 37 in the lib test** —
  and exits `RC=101`. So the `~31 inside` half of the figure this board quoted
  at `9628a652` reproduces almost exactly, and the `143 outside feat-math` half
  is **unmeasured rather than refuted**: every diagnostic in that run names a
  `crates/feat-math/` path, and once the build failed cargo stopped scheduling
  the rest (`warning: build failed, waiting for other jobs to finish...`),
  having already checked `jope-paths`, `numpy-pcg64`, `jope-arena` and
  `jope-datasets` clean. So no workspace figure is quoted here. A count for the
  rest is one `--exclude feat-math` away; nobody has run it.
- **The hosted `rust-test` lane, measured on the first green run that retained
  an artifact** (`35586778652`, artifact `rust-test-evidence` = `10633028231`,
  independently audited by the qualification reviewer; the lane was already
  green at `35580809723` 09:00Z and `35583896833` 09:32Z, neither of which kept
  anything — which is the whole reason retention landed): **651 tests executed and passed**, target **8.98 GB**, free disk
  11.5 GB before and 3.9 GB after, nextest **626 s** inside an **822 s** job.
  Full product inventory 664 = 651 executed + 9 ignored + 4 filtered by the
  `pr` profile. The tree's own comments said 9.4 GB / 617 tests / "about a
  minute" — local numbers from before the lane ever ran hosted, corrected on
  `ci/rust-test-followup-20260921`.
- A full-workspace `cargo test --no-run` links ~72 GB against a runner's ~14 GB
  (documented identically in `deploy/ci/rust-test-scope.sh` and `ci.yml`) — it
  SIGBUSes rather than failing a test. #39 therefore gates a **checked
  partition**: 11 product crates on pull requests, 9 deferred to self-hosted,
  with `deploy/ci/rust-test-scope-check.py` asserting against `cargo metadata`
  that the two agree with the script's own output and leave nothing unplaced.
  A crate in neither list would be gated by nothing; `jope-arena` did that
  once, and the guard caught `jope-pipeline` doing it next — on the unmerged
  `wp13/pipeline-run` (#44, `8b4cf466`), so do not look for that crate on
  `main`.
- **A full-workspace run is red on `main` for reasons unrelated to any branch.**
  18 failures across `jope-parallel`, `jainnode` and `jain-smartcluster-daemon`,
  measured by the integrator session on this host with `cargo nextest run -P pr
  --locked --workspace` on 2026-09-21 and reproduced on `main` itself. **This
  has not been reproduced in CI and has no receipt in the tree** — treat it as
  an integrator measurement, not a verified fact, and re-measure before relying
  on it. Note the `pr` profile *as gated* is green: 651/651. Note also that
  `deploy/ci/known-broken-tests.txt` contains **zero** `jainnode` and
  `jain-smartcluster-daemon` entries: every entry in it is a `jope-parallel`
  binary, so the file is scoped to that crate and is not a census of what fails
  on `main`. It should say so.
- **The quarantine, counted rather than estimated** (15 entries at
  `70506066`): the `cause A` block holds 9, but the fit-authority fence
  explains **5** of them (2 "namespace capability inactive", 3 "fit authority
  fenced"). The other 4 are 3 × the ADR-0003 relocation (cause A0, fixed in the
  #42, merged) and 1 × a `queue.rs:133` panic that was misfiled under the fence.
  `cause B` (behavioural assertions) holds 5 and `cause C` (racy, passes
  serially) holds 1. #42 took the file to 12 and unfiled the panic.

## Correction to the product plan: the engine cannot measure

`crates/jope-engine/src/fit_boundary.rs` requires `JOPE_FIT_SERVICE_SOCKET`
unconditionally — no socket, no fitting — and
`eval_registry/authority.rs::activation_block` returns a block for every
non-test scope ("production capabilities have no verifier yet"), so **no
shipped binary can measure through the engine**. That is not a bug to route
around quietly: it is a deliberate fence, and it explains **5** of the 15
quarantined tests — not the 9 the `cause A` block appears to hold, which is
why that block was unfiled in #42.

Consequence for the pipeline: **WP-P1 (`jope pipeline run`) builds on
`feat_math::evolve` + `feat_math::fit` + `feat_math::transpile`, not on
`jope-engine`'s held entry points.** `crates/jope-cli/src/kernel.rs` is the
unfenced path today, but its `SEARCH_CANDIDATES` is a three-candidate loop, not
a genetic program — the GP is `feat_math::evolve`
(`random_canonical_pipeline` / `mutate_canonical_pipeline` /
`crossover_canonical_pipelines`) and it is complete. Lifting the fence is a
separate, reviewed change with its own verifier, never a side effect of a
pipeline PR.

## Recorded conflict — owner to settle (2026-09-21)

Two agents hold different owner directions dated 2026-09-21:

- **Integrator (Claude, this board's D1–D5):** rapid PRs; agents merge after CI +
  cross-identity review; fast PR gate; both fixture sets admitted; org upgrade
  for protection. Owner-approved plan copy:
  `/home/ubuntu/jope-migration-evidence/claude/jopedime-unstall-20260921/PLAN.md`.
- **Coordinator (`jdime_merger_cursor`, review on #23 at 03:50Z):** a
  "stopped-writer plan" requiring explicit owner releases from every writer,
  stopped preservation workers/supervisors and restoration first, one
  implementation/ledger owner, exactly ten synthetic scenarios, and
  protection/qualification before any merge. It reports that Grok has released
  its JopeDime lane and that the preservation owner has not.

**Owner ruling (2026-09-21 ≈03:58Z, given directly to the integrator session
with all three directions quoted verbatim): D1–D5 control.** Codex's foundation
work lands through the same reviewed-PR gate one small PR at a time with the
≥95 audit as a ratchet; Cursor keeps the reviewer/coordinator role; the
preservation lane and its live workers (PIDs 115122, 3692153, 3692154, 3777416
at 03:20Z) are not touched by anyone. The stopped-writer and audit-before-merge
holds do not apply to this repository from that time. Posted on the canonical
board, the canonical chat and the cross-tree wire at 03:58:38Z.

Merge history for the record: #1–#16 were closed unmerged (2026-09-18/20);
#17–#22 (Grok's A–F) are the first merges in this repository, 2026-09-21
01:24–03:16Z, `main` `4235db6` → `1f0fc5a`.

## Known stale — do not cite

Claims that were true once and are not now. They are listed rather than
silently deleted, because they have been quoted at each other across three
sessions.

| Claim, and where it still appears | Why it is wrong |
| --- | --- |
| "617 tests", "9.4 GB", "about a minute" — `deploy/ci/rust-test-scope.sh` and `.github/workflows/ci.yml` at `70506066`; still in `deploy/ci/rust-test-scope.sh` and the `test-pr` comment of `justfile` | Local numbers from before the lane ever ran hosted. Measured: 651 tests, 8.98 GB, 626 s. Corrected on `ci/rust-test-followup-20260921`; the tree carries the old ones until that lands. |
| "the hosted gate ran no Rust tests at all" — this board's earlier draft and the #39 discussion | `emitters` ran `cargo test -p feat-math --lib -- cxx::` and `--test portable_parity`. The rest of the claim holds. |
| "the fence explains 9 of the quarantined tests" — this board's earlier draft | It explains 5. The `cause A` block held 9 because 3 relocation failures and 1 panic were misfiled under it. |
| "`rust-check` runs clippy and fmt" — the D2 row before this revision | It runs `cargo check`, a systemd test and an MSRV check. `cargo fmt --all --check` now runs in the hosted `hygiene` lane (`deploy/ci/check.sh`); clippy is still in no hosted lane. |
| "Full-main CPU workflow and runner delivery remain unimplemented and unqualified" — this file's Blockers before the 2026-09-28 record | `full-main-cpu` runs on every push to `main`. Its first success, 36343656796, came after #116; 36472565658 passed on `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`. |
| "Live S3 is **not** wired" — the README before the v0.2.0 release PR | The dataset and model stores and the learning mirror reach `s3://` through `object_store`. What fails closed is the champion route (#62) and `train-step --env`; no live S3 training run has been made. |
| "A `v0.1.0` tag through `release.yml` with attested `jope` binary, wheel, SBOMs and receipt" — `docs/public-readiness-checklist.md` before the v0.2.0 release PR | Releases come from `deploy/release/release.sh` (owner decision D-D), with a cosign signature over `SHA256SUMS`; `release.yml` is dispatch-only; there is no wheel and no attestation. The first release is v0.2.0. |
| In the stale `jain-split` mirror: "Eight-A100 recovery" in the header of `/home/ubuntu/jain-split/jope-dime/AGENT_CHAT.md:12` (not this repository's file, whose header says the opposite); and `main` at `458ef58` **with** "zero Actions workflows / runs" in that tree's `FINAL_GITHUB_PORT_PLAN.md:116` and `CODEX_GITHUB_PORT_PLAN.md:27-28`. "13 open PRs" appears in neither and was this appendix's own misattribution | Pre-2026-09-21 state of the stale `jain-split` mirror, which is now a pointer. |
| PR #40, PR #41 | Opened from the reviewer's identity and closed unmerged; the same branches are #42 and #43. Not withdrawals of the work. |

### CPU foundation record (PR61, 2026-09-22)

Source, reviewed deployment and live run evidence are recorded at the top of this document.
They remain separate from full CPU qualification, which requires the subsequent repair phase.
