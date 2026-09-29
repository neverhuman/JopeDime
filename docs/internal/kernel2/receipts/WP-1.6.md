# WP-1.6: static scope and width check (kernel2 PR 05)

Base: `9146cbcc26b609fb8f59788088f6c87cc60bade2`, `origin/main` after PR 04 (#130) merged; PR 01
to PR 03 are #126 to #128, and #129 is the islands-v2 source import. The series was written on
`6e0ae2491cd4e7918c4d8b07aa211c83376c6764`, `origin/main` after #129 merged, and it was replayed
onto `9146cbcc2` before its first push (see "Replay"). This pull request is #131.
Branch `kernel2/claude-m1-scope`, in the lane `.agent/worktrees/kernel2-next`. The first round's
feat-math sources and tests are at the pre-replay `e1aba6d0b` (`ba623e0fe` after the replay) and
its records at `e4bedb78c`. Three reviews of `e4bedb78c` followed; their fixes are the pre-replay
`c47f9906a` and `f7e5bcb5d` (`cd86c9152` and `67b5a5f7b` after the replay), and the last commit
regenerates these records (see "Review fixes" and "Replay").
Measured on 2026-09-29 on xbabe2: x86_64-unknown-linux-gnu,
glibc 2.39, AMD Ryzen Threadripper PRO 3995WX, rustc 1.97.1 (8bab26f4f 2026-07-14),
cargo-nextest 0.9.133, jq 1.7. Environment: `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `TMPDIR` under
`~/.cache/jope-test-tmp/`, `nice -n 19 ionice -c3`, with `JOPE_COMPAT_FREEZE_REGENERATE` and
`JOPE_COMPAT_FREEZE_SHA` unset; test runs add the rust-test lane's `JOPE_PARITY_STRICT=1` and one
OpenMP, rayon, OpenBLAS and MKL thread. This is local evidence, not hosted CI, and the wall times
are for this host only.

## What changed

| Commit | Change |
|---|---|
| `9bf5d0d86` | `crates/feat-math/src/kernel/scope.rs` (new): `validate_scope`, `ScopeError` and `MAX_INPUT_WIDTH`. `kernel/mod.rs` gains `pub mod scope;`, beside #130's `pub mod telemetry;`. In `kernel/pipeline.rs`, `next_width` becomes `pub(crate)`; its body is unchanged, and rustfmt wraps the longer signature. |
| `89a7ca8ed` | `math_expr/evolution.rs`: `ExpressionScope::search_profile(columns, elementwise)`, caps 15 nodes and depth 4. |
| `ba623e0fe` | `crates/feat-math/tests/scope_validation.rs` (new): 13 tests. |
| `cd86c9152` | Review fix: `validate_scope` walks each member of an explicit portfolio from one column. `scope_validation.rs` gains the portfolio members test (14 tests), and its corpus agreement compares a portfolio member at one column. |
| `67b5a5f7b` | Review fix: the determinism test asserts its digest. |
| last | These records: `deploy/ci/coverage.json`, both test inventories, `CHANGELOG.md`, `MASTER_PLAN.md` and this receipt, regenerated after the replay from the two pre-replay records commits, `e4bedb78c` (the first records) and `153e688e6` (the records of the review fixes). |

Every commit has Jepson Taylor `<130782313+jepsontaylor@users.noreply.github.com>` as author and
committer.

The public API changes by additions only: `feat_math::kernel::scope::{validate_scope, ScopeError,
MAX_INPUT_WIDTH}` and `ExpressionScope::search_profile`. `ScopeError` is `#[non_exhaustive]`, has
the public fields `member`, `stage`, `class` and `reason`, and implements `Display`,
`std::error::Error`, `Clone`, `Debug`, `PartialEq` and `Eq`. There is no new dependency, no new
enum variant, no changed public signature and no removed item, and the crate root re-exports
nothing new. Nothing that ships calls the new code.

## Replay

#130, kernel2 PR 04 (stage telemetry), merged as `9146cbcc2` while this series was in review.
Before its first push, the series was replayed onto it. The pre-replay head
`153e688e604fa6e113e947bf5bcb0fe50bcff99b` is kept locally as
`refs/preserve/kernel2/pr05-prerebase-20260929T183131Z`, so every pre-replay commit named in this
receipt stays reachable.

- `git range-diff 6e0ae2491..153e688e6 9146cbcc2..HEAD` shows the five replayed commits:
  `38fd1caae` → `9bf5d0d86`, changed only by the resolution below (`!`), and `3965228b3` →
  `89a7ca8ed`, `e1aba6d0b` → `ba623e0fe`, `c47f9906a` → `cd86c9152` and `f7e5bcb5d` →
  `67b5a5f7b`, unchanged (`=`). Each keeps its message, author and author date. The two records
  commits, `e4bedb78c` and `153e688e6`, are regenerated as the last commit: range-diff pairs
  `e4bedb78c` with it (`!`) and lists `153e688e6` as dropped (`<`).
- **One conflict, the expected one.** #130 added `pub mod telemetry;` to `kernel/mod.rs` at the
  line where `38fd1caae` adds `pub mod scope;`. Both lines are kept, in rustfmt's order: the file
  is the new main's with `pub mod scope;` added before `pub mod telemetry;`, and it differs from
  `38fd1caae`'s by #130's line alone. range-diff shows it as a context line of the hunk.
- **The sources are the reviewed ones.** `kernel/scope.rs`, `kernel/pipeline.rs`,
  `math_expr/evolution.rs` and `tests/scope_validation.rs` are byte-identical to `f7e5bcb5d`, and
  every other difference under `crates/` between `f7e5bcb5d` and `67b5a5f7b` is #130's, byte for
  byte: `kernel/telemetry.rs`, `tests/stage_telemetry.rs` and its line in `kernel/mod.rs`.
- **The records are regenerated, not taken whole.** #130 added its own coverage item, findings role
  and 19 feat-math tests (`stage_telemetry`) beside the places this series writes. This series'
  coverage item and findings role are re-applied to the new main's `deploy/ci/coverage.json` and
  findings test inventory with `jq`, and each adds exactly the lines `e4bedb78c` added: coverage
  288 to 289 targets, findings 324 to 325 items. The parity inventory's feat-math section is
  re-listed from `cargo nextest list --locked -P main -p feat-math --message-format json` at
  `67b5a5f7b`, in the shape `test-inventory.sh --update` writes: feat-math 510 to 524 tests (6
  ignored), Rust 3,873 to 3,887 (65 ignored), and every other package unchanged.
  `test-inventory.sh --no-rust --static-python --update` then changes no byte.
- The pull request number, #131, replaces the number placeholder in `CHANGELOG.md` and
  `MASTER_PLAN.md`.
- **One correction besides the replay.** The reference check below finds a captured file list in a
  jope-records replay fixture,
  `crates/jope-records/tests/fixtures/python-parity/coverage/files.txt`, that names
  `kernel/mod.rs`, `kernel/pipeline.rs` and `math_expr/evolution.rs` as paths, so "Identity
  unchanged" no longer says that nothing names them. The fixture includes and hashes nothing, and
  no `include_str!` or `include_bytes!` reads anything under `kernel/` or `math_expr/`.

Commands at `67b5a5f7b`, the tree of the last commit apart from these records, with a clean tree
(`a` and `a2` in the replay's evidence). The cargo and script steps ran under
`nice -n 19 ionice -c3`, the test runs with one OpenMP, rayon, OpenBLAS and MKL thread, and
`JOPE_PARITY_STRICT=1` where the command shows it. Where a step ran twice, both exit codes and wall
times are given. The same commands ran again at the head, with these records committed (`d` and
`d2`), with the same results.

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| feat-math PR tier | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 514 passed; 10 skipped (6 ignored, 4 by the pr filter) | 69.9 s with the build, 61.8 s |
| The six named targets, compare mode | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test scope_validation --test stage_telemetry --test fault_trace --test kernel_pipeline --test kernel_expression` | 0, 0 | 65 passed (12, 14, 19, 13, 3, 4); 2 skipped | 19.7 s, 19.8 s |
| The earlier rounds' nine targets and `stage_telemetry` | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test mathexpr_current_semantics --test fault_trace --test ip_scrub --test legacy_v2_freeze --test portable_parity --test kernel_pipeline --test kernel_expression --test scope_validation --test stage_telemetry` | 0 | 93 passed (12, 18, 13, 6, 3, 1, 3, 4, 14, 19); 3 skipped | 19.6 s |
| The new tests' output | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test scope_validation --no-capture` | 0, 0 | 14 passed; the 6 output lines byte-identical across the runs and to the review fixes' (sorted sha256 `cfded17e…`), the asserted determinism digest `4579e907…` among them | 6.5 s, 6.3 s |
| Parity with the emitted programs | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0, 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale | 3.5 s, 3.6 s |
| Legacy value digest | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fault_trace --run-ignored ignored-only --no-capture -E 'test(=legacy_value_digest)'` | 0 | `b64cf538…`, 191,716 results, as PR 03 recorded | 0.7 s |
| Evaluator identity tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0 | 15 passed, so the identity table is unaffected | 55.9 s with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (csv 40, regression-ops 2, regression 5, the teacher-v2 search engine 2, the teacher-v2 worker 3, monitor 14); cargo-deny skipped in all 6 (0.19.8 installed, 0.20.2 asserted), so that step is not a pass | 43.2 s |
| Emitters | `EMITTERS_OUT=<scratch> bash deploy/ci/emitters.sh` | 0 | both programs: sources byte-identical to the pins, predictions bit-identical across the 2 and 7 programs | 16.6 s |
| Workspace check | `cargo check --workspace --all-targets --locked` | 0 | warnings only in `battle-gpu`, `jope-dime-regressor` and `jope-parallel`, as before, and none in feat-math | 38.2 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 8.6 s |
| Identity inputs | `git diff --quiet 9146cbcc2 HEAD --` over `crates/feat-math/tests/fixtures`, `crates/jope-campaigns`, `crates/jope-engine`, `Cargo.toml`, `Cargo.lock`, and feat-math's `Cargo.toml`, `lib.rs`, `ops.rs`, `model.rs`, `math_expr.rs` and `evolve.rs` | 0 | unchanged, and again with `rust-toolchain.toml` and `.config` | 0.0 s |
| The 70 identity paths | `git diff --quiet` over the 70 paths of "Identity unchanged", from `9146cbcc2` and from `6e0ae2491` to HEAD, and from HEAD to the working tree | 0 | none differs, and none is a path this pull request changes | 0.1 s |
| Fixtures | `git diff --stat 9146cbcc2 HEAD -- crates/feat-math/tests/fixtures/` | 0 | empty | 0.0 s |
| `next_width` | `git diff -U0 9146cbcc2 HEAD -- crates/feat-math/src/kernel/pipeline.rs` | 0 | one hunk, the signature: `pub(crate)`, wrapped by rustfmt (5 lines for 1); the body unchanged | 0.0 s |
| References | `git grep` under `crates/`, `deploy/ci` and `tools/` for the changed paths, and for `include_str!` or `include_bytes!` of anything under `kernel/` or `math_expr/` | 0 | the captured file list above, and `evaluation_v2.rs` hashing `math_expr.rs`, which this series does not change | 0.2 s |
| Subjects | `git log --format=%s 9146cbcc2..HEAD`, searched with the ip-scrub pattern and the plan's additions | 0 | no subject matches | 0.0 s |
| Authors and trailers | `git log --format='%an <%ae>\|%cn <%ce>' 9146cbcc2..HEAD`, and the last line of every message | 0 | one identity, author and committer; every message ends with the trailer | 0.0 s |
| Whitespace | `git diff --check 9146cbcc2 HEAD` | 0 | clean | 0.0 s |

With these records in the working tree (`b` in the replay's evidence, and `b2` after their last
edit), the checks that read them ran, each with exit 0:

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 289 targets, 0 unassigned or invalid | 0.9 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 325 items, 196 of them regression | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 2.6 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped; its `records` tests read the live coverage map and inventory | 13.9 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python`, then with `--update` | 0, 0 | consistent: Rust 3,887 kept, Python 172; the update changes no byte | 0.1 s, 0.1 s |
| The feat-math section against a live listing | `cargo nextest list --locked -P main -p feat-math --message-format json`, then the re-list script with `--check` | 0 | equal | 0.7 s |
| The two record items against the new main | the re-apply `jq` on `9146cbcc2`'s `deploy/ci/coverage.json` and findings inventory, then `cmp` | 0 | both equal the files here, and each adds exactly the lines `e4bedb78c` added | 0.2 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok; gitleaks found no leaks (the 721 commits reachable from `67b5a5f7b`, 718 scanned) | 20.0 s |
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.8 s |
| Whitespace | `git diff --check 9146cbcc2` | 0 | clean | 0.0 s |
| Placeholders | `git grep -nI` for the number placeholder, and `grep` for template braces in the three records files | 0 | none | 0.0 s |

## Decisions taken while implementing

1. **Where the code lives.** `kernel/scope.rs` and `math_expr/evolution.rs` are outside the ADR-0004
   hash list, as the brief requires. `validate_scope` reuses `PipelineSlots::from_pipeline` for the
   stage layout and `next_width` for the widths, so it walks a pipeline in the order of
   `validate_legacy_widths`, with the same reasons.
2. **The error.** `stage` is the index in the pipeline's `stages`, a target stage included;
   `member` is an explicit portfolio's member; `class` is `Some(ILLEGAL_SCOPE)` or
   `Some(COLUMN_OOB)` for a scope fault, and `None` when the program cannot be checked, so a later
   fault count sees only real scope faults. `Display` is `[member M, ][stage S: ][CLASS: ]reason`,
   for example `stage 1: COLUMN_OOB: column index 3 out of bounds for width 2`.
3. **The class comes from the tree, not from the message.** `validate_columns` checks `x` before
   any column, so an expression outside a `map` that reads `x` is `ILLEGAL_SCOPE`, and every other
   failure of `validate_columns` is `COLUMN_OOB`. The reason is `validate_columns`' message,
   verbatim. An expression with both faults is `ILLEGAL_SCOPE` here; the runtime reports whichever
   it evaluates first.
4. **Training rows.** `next_width` takes a row count, and `validate_scope` has none. Only `nystrom`
   depends on it (its width is `min(k, 128, rows)`). The walk assumes one row, the fewest a fit can
   have, so its lower bound holds for any training set: a column read after `nystrom` must exist at
   width 1, plus any appended input columns. With more rows, `validate_legacy_widths` accepts some
   of these; the tests list every such case.
5. **Input width.** `validate_legacy_widths` refuses inputs wider than `MAX_WIDTH` (512), but the
   runtime truncates a row to 512 only after a stage, so the first stage reads every input column,
   and the plan's E3 searches at width 600. `validate_scope` accepts 1 to `MAX_INPUT_WIDTH` (2^24)
   columns and refuses 0 and anything wider with its own reason. The bound also keeps `next_width`'s
   unchecked `lo + 1` for `poly_interact` from overflowing: without it, `usize::MAX` would panic in
   a debug build.
6. **Closed-form and portfolio programs.** A closed-form program's expression is evaluated per row
   (`model.rs`), so it is checked in row context at the input width: `(sym raw "x")` is
   `ILLEGAL_SCOPE`. The members of an explicit portfolio do not see the input as it is.
   `fit_portfolio` drops every column that is constant in training before its members see the
   rows, and prediction applies the same mask (`portfolio.rs`, step 1 of spec §4). So a member's rows
   hold between one column and the input width, and its column indices count the kept columns: on
   training data whose column 0 is constant, a member's `c2` is input column 3. Members are fitted
   only when at least one column is kept, so each member is walked as a pipeline from one column,
   by the lower-bound rule `nystrom`'s rows follow. The first round walked members from the input
   width; the correctness review found it (see "Review fixes").
7. **Layout and width errors.** A stage layout that the slots refuse is an error of the whole
   program (`stage` is `None`): an unknown op gives `unknown stage <op>`, and a misplaced stage
   keeps the slots' reason, which names its index. An input encoding stage, an `expr` that is not a
   string, an expression that does not parse and a stage whose width only the table-v3 path
   computes are errors of their stage, with the reasons of `validate_legacy_widths`.
8. **`search_profile` is a constructor only.** No existing scope, stream or default changes, and
   the kernel pipeline operators keep 64 nodes and depth 16.
9. **No wiring.** The brief names PR 07 for it. The plan's execution notes (the re-cut M1) put the
   admission policy, which also calls `validate_scope` from `evolve()` and `evolve_lite_race`, in
   PR 08.

## Tests (`crates/feat-math/tests/scope_validation.rs`: 14 run, 0 ignored)

**The workplan's cases**, each with its exact result:

| Case | Result |
|---|---|
| `(pipe X (map expr="x") (ridge))` at width 3 | accepted |
| `(pipe X (derive expr="x") (ridge))` at widths 1, 3, 8 and 512 | `stage 0: ILLEGAL_SCOPE: variable x is only valid in map expressions` |
| `col(0)` at width 1, in a `derive` and in a `map` | accepted |
| `(pipe X (derive expr="col(7)") (ridge))` at width 1 | `stage 0: COLUMN_OOB: column index 7 out of bounds for width 1` |
| `(pipe X (select_corr k=2) (derive expr="col(3)") (ridge))` at width 8 | `stage 1: COLUMN_OOB: column index 3 out of bounds for width 2`; `col(1)` is accepted |
| `(pipe X (row_stats) (derive expr="col(9)") (ridge))` at widths 1, 3 and 20 | accepted; `col(10)` gives `stage 1: COLUMN_OOB: column index 10 out of bounds for width 10` |

The facts behind them are checked too. `row_stats` is `ShapeRule::Fixed(10)` in the registry
(`registry.rs:755`) and emits ten values at runtime, and `select_corr k=2` keeps two of eight
columns. The runtime meets `col(7)` on a one-column row, and `x` in row context, only when it
evaluates, with the same classes: `eval_row_traced` returns `(0.0, {COLUMN_OOB})` and
`(0.0, {ILLEGAL_SCOPE})`. Its message, `column index 7 out of bounds`, has no width; the static
reason is `validate_columns`' message.

**Location and class.** `x` is reported before a column; the first column out of range in reading
order is reported, not the largest; a target stage counts in the index; an appending stage keeps
its input columns; a closed-form program has no stage; a portfolio error names the member and the
stage inside it.

**Portfolio members** (added by the review fixes). A member is checked from one column:

- At input widths 1, 4 and 600, a member `(derive expr="c3")` gives `member 0, stage 0:
  COLUMN_OOB: column index 3 out of bounds for width 1`, and `c0` is accepted; the same pipeline on
  its own reads `c3` at width 4.
- A stage of fixed width (`row_stats`) or an appended column sets what later stages may read, and
  the error names the failing member (`member 1, stage 0: …`).
- Every member of the width-rule grid (the 40 stages with and without `append`, × 35
  expressions) is judged exactly as its pipeline is at one column (member, stage, class and
  reason), at input widths 1 to 5, 8, 16, 600 and 2^24: 25,200 comparisons, 21,654 of them
  rejections.
- At runtime, on 60 rows of 4 columns whose column 0 is constant, the fitted portfolio keeps
  columns 1 to 3. The member's `c3` evaluates to `(0.0, {COLUMN_OOB})` on a kept row, and the
  portfolio predicts one value everywhere. Its `c2` (input column 3) and the pipeline's `c3` both
  reach a training R² above 0.99, and so does the member's `c3` when no column is constant.

**Programs that cannot be checked**, each with its own reason and no class: a numeric `expr` (the
printer leaves a numeric-looking expression unquoted, and it reads back as a number), a missing
`expr`, seven expressions that do not parse (the parser's own message, in a `map` and a `derive`),
an unknown op, two misplaced stages, an input encoding stage, `pls_project`, and the input widths
0, 2^24 + 1 and `usize::MAX` for all three program forms. Width 2^24 itself is accepted.

**No panic.** All 43 transform ops, alone and twice before a column read, with and without
`append`, at widths 0, 1, 2, 511, 512, 513, 600, 2^24 − 1, 2^24, 2^24 + 1 and `usize::MAX`.

**Agreement with `validate_legacy_widths`**, called with any stage count. Each comparison checks
the verdict and, when both reject, the exact reason; for a scope fault, the class must match the
reason.

| Set | Programs | Input widths | Training rows | Comparisons | Both accept | Both reject (scope faults) | `nystrom` cases |
|---|---|---|---|---|---|---|---|
| The compatibility corpus (`emitted_bytes.v1.json`, read only) | the 48 pipelines of its 55 programs, all accepted by the slots | 1 to 8, 10, 16, 20, 64, 511, 512 | 1, 32, 4096 | 2,016 | 1,920 | 96 (96) | 0 |
| Every width rule, then an expression stage | 40 stages, with and without `append`, × 35 expressions | 1 to 5, 8, 16 | 1, 4096 | 39,200 | 15,940 | 22,890 (21,910) | 370 |
| Random pipelines (`full_rust::random_pipeline`, seeds 0 to 299) | 296 (the generator refuses 4 seeds itself) | 1, 2, 3, 5, 8, 16 | 1, 4096 | 3,552 | 3,462 | 84 (84) | 6 |

- **The corpus agrees everywhere.** Only `derive_x_body` is rejected at every width
  (`ILLEGAL_SCOPE`). The 7 closed-form programs match, at every width, an oracle computed from a
  walk of their trees. The corpus has no explicit portfolio; a member would be compared at one
  column.
- **The only differences are `nystrom`'s, and only with more than one training row**: 376
  comparisons, each a column read after `nystrom` that the legacy check accepts, or rejects against
  a wider width. With one row the two agree exactly, and the test asserts that for each case. The
  corpus never reads a column after `nystrom`.
- The grid's 980 rejections without a class are `pls_project`'s `needs table-v3 width binding`,
  reported alike by both checks.

**The two intended differences**, on their own:

- `(pipe X (nystrom k=4) (derive expr="c2") (ridge))` at width 8 is `stage 1: COLUMN_OOB: column
  index 2 out of bounds for width 1`. The legacy check agrees with one training row, rejects against
  width 2 with two rows, and accepts with three or 4,096. At runtime, `nystrom k=4` emits 2, 3 and 4
  columns for 2, 3 and 8 training rows.
- `(pipe X (derive expr="c599") (ridge))` is accepted at width 600, which the legacy check refuses;
  after `standardize`, `c599` is rejected against width 512. At runtime the derived column is input
  column 599, and `standardize` truncates 600 columns to 512.

**The search profile.** `search_profile(c, e)` is `{c, e, 15, 4}`. Depth 4 fits and depth 5 does
not; 15 nodes fit and 16 do not, both at depth 4; the 64-node, depth-16 scope admits both.

**Property.** 5,000 chained edits by the v1 edit operator (`MathExpr::mutate_v1`) under
`search_profile`. The scope changes every 50 edits (map or row context) and every 250 (0, 1, 2, 3
or 8 columns), and a parent outside the new scope restarts from `0.1` (24 restarts).

- Every child passes the scope's own check.
- A separate walk of its tree finds at most 15 nodes and depth 4, no `x` outside a map and no
  column past the scope.
- For every scope with at least one column (4,000 of the 5,000 edits), `validate_scope` accepts it
  in a stage at that width. The 1,000 zero-column edits skip that call, because `validate_scope`
  refuses width 0 by design.
- 4,881 edits changed the tree and 4,398 trees are distinct. The maxima are exactly 15 nodes and
  depth 4, so the caps bind.

The operator keeps only candidates that pass `ExpressionScope::validate`, so the property holds by
construction; the test keeps it as a regression test.

**Determinism.** 780 cases: the 23 hand-written programs and the 55 corpus programs, at 10 widths
including the out-of-bounds ones. Three more calls, and the program read back from its canonical
text, give equal results, reason bytes included. The test asserts the case count and the digest
of every case's displayed result, `4579e907329496ced7ac5bfe8cabb519096eaa08b40a6eaba116c65a7317c63f`
(since the pre-replay `f7e5bcb5d`; the first round only printed it). A change to any of those
bytes, including the parser's messages, which no exact-string case covers, fails the test and has
to be made on purpose, in the test and here. The portfolio member fix leaves the digest unchanged:
no hand-written case or corpus program reads a column in a member before a stage of fixed width.

**Shipped text.** Every distinct display and reason from those cases passes
`transpile::lint_shipped_text`.

The test output's `AGREEMENT`, `PROPERTY` and `DETERMINISM` lines (`--no-capture`) are
byte-identical in both runs of each round. Sorted, their sha256 is
`d9ad24838c71303168b3499be2a0051af2fddd4d70d2f1c05e430a4c158223f4` for the first round's five
lines, and `cfded17e33d0126dcc45276b68de788224819f66c001d94184a70434ef9d3949` for the six at
`f7e5bcb5d`: the five are unchanged, and `AGREEMENT members` is new. After the replay the six lines
are the same bytes again, at `67b5a5f7b` and at the head (see "Replay").

**Perturbations caught.** The first four were applied to `scope.rs` before the replay, at
`e1aba6d0b`, and the fifth at `c47f9906a` (they are `ba623e0fe` and `cd86c9152` after it, and this
section keeps the pre-replay ids); each time `scope_validation` ran, and the file was restored with
`git checkout`. The tree was clean after each.

| Perturbation | Failing tests |
|---|---|
| Expressions are checked against the upper end of the width interval | the grid agreement (1) |
| The walk assumes `usize::MAX` training rows | the grid and random agreements, the intended differences (3) |
| The class is inverted (`x` read outside a map reported as `COLUMN_OOB`, and the reverse) | the three agreements, location and class, the intended differences, the workplan's cases (6) |
| A target stage is not counted in the index | programs that cannot be checked, the random agreement, location and class (3) |
| Portfolio members are walked from the input width again, as in the first round | the portfolio members test (1): `a member reading c3 is accepted at width 4` |

## Identity unchanged

- **compat_freeze** passes 12 of 12 in compare mode, twice in each round, and no fixture changes:
  `git diff --stat origin/main -- crates/feat-math/tests/fixtures` is empty, and
  `git status --porcelain -- crates/feat-math/tests/fixtures` was empty after every run.
- **The ADR-0004 evaluator identity does not move.** None of its 38 inputs changes
  (`evaluator_identity.v1.json` `inputs`: the 31 feat-math sources that
  `crates/jope-campaigns/src/evaluation_v2.rs` `compute_implementation_sha256` hashes, 5
  jope-campaigns files, `crates/feat-math/Cargo.toml` and `Cargo.lock`). Neither do the 14 lineage
  sources (`lineage_sources.v1.json`, among them `crates/feat-math/src/model.rs` and
  `crates/jope-engine/src/grammar.rs`) or the 20 frozen pins (`frozen_pins.v1.json`, among them
  `registry.rs` and `cpp/fitted_runtime.cpp`). `git diff --quiet 6e0ae2491 --` over the 70
  distinct paths of those lists, read from the three fixtures with `jq`, exits 0, and after the
  replay so does `git diff --quiet 9146cbcc2 HEAD --` over the same paths (see "Replay").
- **The changed sources** are `kernel/scope.rs` (new), `kernel/mod.rs`, `kernel/pipeline.rs` and
  `math_expr/evolution.rs`. None is in any of those lists, and nothing in `crates/`, `deploy/ci` or
  `tools/` includes or hashes them. The only mention of any of them is a captured file list in a
  jope-records replay fixture (`python-parity/coverage/files.txt`), which names three of them as
  paths (see "Replay"). `Cargo.toml`, `Cargo.lock`, `crates/feat-math/Cargo.toml`
  and `rust-toolchain.toml` do not change.
- **The legacy streams do not move.** `next_width` changes visibility only, and nothing that ships
  calls `validate_scope` or `search_profile`. compat_freeze's generator streams, including
  `PipelineSlots::mutate_v1` and `crossover_v1`, which call `next_width`, compare equal.
- One thing does change with any rebuild: jope-engine's teaching, corpus and query identities also
  hash the running executable (`/proc/self/exe`; see the note in `lineage_sources.v1.json`), so a
  new binary has new runtime identities. That is true of every feat-math change.

## Commands and results

First round, before the replay, at `e1aba6d0b` (`ba623e0fe` after the replay), the sources the
reviews read, with a clean tree. Where a step ran twice, both exit codes and wall times are given.
The review fixes' commands follow in "Review fixes", and those after the replay in "Replay".

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 494 passed; 10 skipped (6 ignored, 4 by the pr filter) | 70.4 s with the build, 60.0 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed, compare mode; the fixtures unchanged after each run | 11.4 s, 11.5 s |
| Named targets | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test mathexpr_current_semantics --test fault_trace --test ip_scrub --test legacy_v2_freeze --test portable_parity --test kernel_pipeline --test kernel_expression --test scope_validation` | 0, 0 | 73 passed (12, 18, 13, 6, 3, 1, 3, 4, 13); 3 skipped | 19.2 s, 19.6 s |
| The new tests' output | `cargo nextest run --locked -P pr -p feat-math --test scope_validation --no-capture` | 0, 0 | 13 passed; the 5 `AGREEMENT`, `PROPERTY` and `DETERMINISM` lines byte-identical across the runs | 6.3 s, 6.2 s |
| Perturbations | the four above, then `git checkout` | 100 × 4 | each caught by 1 to 6 tests; the tree clean after each | 9 to 13 s each with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (csv 40, regression-ops 2, regression 5, search engine 2, worker 3, monitor 14); cargo-deny skipped (0.19.8 is installed, and the lane asserts 0.20.2), so that step is not a pass | 48.4 s |

With the first records in the working tree (the sources as at `e1aba6d0b`):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.8 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 8.6 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.4 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items, 195 of them regression | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 2.8 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped; its `records` tests read the live coverage map and inventory | 15.1 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent; the Rust half kept at 3,867 tests | 0.1 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok: coverage 288 targets, gitleaks over the 709 commits reachable from `e1aba6d0b`, the repository layout | 19.6 s |
| Whitespace | `git diff --check 6e0ae2491` | 0 | clean | 0.0 s |
| Identity inputs | `git diff --quiet 6e0ae2491 --` over the 70 paths above | 0 | none differs; the fixtures and the Cargo files do not change | 0.0 s |

The feat-math part of the parity inventory is `cargo nextest list --locked -P main -p feat-math
--message-format json` at `e1aba6d0b`, written with `jq` into the feat-math section and the totals
only: with those removed, the file is unchanged, and every binary but the new one is listed as
before.

## Review fixes

Three reviews of the pre-replay `e4bedb78c` ran independently: correctness, no regression, and the
host and owner rules. They found one should-fix and three nits. All four are fixed; none is
rejected.

| Finding | Severity | What was done |
|---|---|---|
| F1: explicit portfolio members were checked at the full input width, but a member sees only the columns that vary in training. A member reading `c3` passed at width 4 and, with one constant column, read out of range on every row (the reviewer's probe against the first round's library) | should-fix | `cd86c9152`: members are walked from one column (decision 6); the doc comments describe the column drop; the portfolio members test pins it, including a fit with a constant column |
| F2 and N1: the determinism test printed its digest without asserting it | nit | `67b5a5f7b`: the test asserts the case count and the digest |
| R1: this pull request's `MASTER_PLAN.md` record named the findings role's subsystem in bare prose | nit | the last commit: the record says "(regression)", as #128's does |

F1 was confirmed in the source before it was fixed: `model.rs` hands an explicit portfolio to
`fit_portfolio`, which keeps the columns whose training standard deviation exceeds 1e-12 and fits
every member on those columns only (`View::Raw` for explicit members), and `PortfolioModel::predict`
applies the same mask. The perturbation above shows that the new test fails on the first round's
behaviour.

At the pre-replay `f7e5bcb5d` (`67b5a5f7b` after the replay), with a clean tree at the start (the
records of `153e688e6` were edited in the working tree while the tests ran; no test reads them):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 495 passed; 10 skipped (6 ignored, 4 by the pr filter) | 69.8 s with the build, 60.4 s |
| compat_freeze | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed, compare mode; the fixtures unchanged after each run | 11.5 s, 11.5 s |
| Named targets | the nine targets above | 0, 0 | 74 passed (12, 18, 13, 6, 3, 1, 3, 4, 14); 3 skipped | 19.4 s, 19.2 s |
| The new tests' output | `cargo nextest run --locked -P pr -p feat-math --test scope_validation --no-capture` | 0, 0 | 14 passed; the 6 output lines byte-identical across the runs | 6.3 s, 6.3 s |
| Perturbation | members walked from the input width, then `git checkout` (at `c47f9906a`) | 100 | caught by the portfolio members test alone; the tree clean after it | 13.0 s with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (40, 2, 5, 2, 3, 14 as above); cargo-deny skipped in all 6 (0.19.8 installed, 0.20.2 asserted), so that step is not a pass | 39.9 s |

With the records of `153e688e6` in the working tree (the sources as at `f7e5bcb5d`):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.7 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 0.3 s (the same sources as an 8.3 s run just before) |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 288 targets, 0 unassigned or invalid | 0.4 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 324 items, 195 of them regression | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 0.3 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped | 13.7 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent; the Rust half kept at 3,868 tests | 0.1 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok; gitleaks found no leaks (scope: `f7e5bcb5d` and its 712 reachable commits, 709 scanned) | 19.5 s |
| Whitespace | `git diff --check 6e0ae2491` | 0 | clean | 0.0 s |
| Identity inputs | `git diff --quiet 6e0ae2491 --` over the 70 paths above | 0 | none differs; the fixtures and the Cargo files do not change | 0.0 s |

The parity inventory's feat-math section, Rust count and Rust total were written again the same
way at `f7e5bcb5d`: feat-math 504 to 505 tests (6 ignored), Rust 3,867 to 3,868 (65 ignored), and
`scope_validation` 13 to 14. With the feat-math section and those counts removed, the file is
unchanged, and every feat-math binary but `scope_validation` is listed as before. The coverage map
and the findings inventory do not change: the test target is the same. All of these commands but
the perturbation ran again after the replay (see "Replay").

## Still open

- **Hosted CI.** `crates/feat-math/**` and `deploy/ci/coverage.json` select every lane.
- **The wiring**: the admission policy (plan PR 08) calls `validate_scope` for the non-legacy
  policies; E0's `admit_child` and E3's `ExprGrow` use it too.
- **The expression parser's recursion.** `MathExpr::parse` descends recursively and checks depth
  only after parsing, so a source with thousands of nested parentheses can exhaust the stack before
  any check. `validate_scope`, `validate_legacy_widths` and the runtime share this; bounded search
  children cannot reach it. It is not changed here.
- **cargo-deny** is 0.19.8 here and 0.20.2 in CI, so the Lambda script's deny step is skipped
  locally, not passed.

Evidence: `~/jope-evidence/kernel2/pr05/prepare-20260929T173902Z/` for the first round (the logs of
its commands, the feat-math listing and the perturbation diffs, with `SHA256SUMS`),
`~/jope-evidence/kernel2/pr05/fix-20260929T181730Z/` for the review fixes (the logs of their
commands, the feat-math listing, the inventory before and after, the perturbation diff and the
scripts, with `SHA256SUMS`), and `~/jope-evidence/kernel2/pr05/rebase-20260929T183131Z/` for the
replay (the logs of its commands, the conflict and its resolution, the feat-math listings, the
re-applied record items, the scripts and the drafts, with `SHA256SUMS`).
