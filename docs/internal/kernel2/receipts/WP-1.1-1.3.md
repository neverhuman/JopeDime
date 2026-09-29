# WP-1.1 to WP-1.3: fault reporting in the scalar evaluator (kernel2 PR 03)

Base: `f65026dd54ba29ef3e7e990b51b15eb42652c089`, `origin/main` after PR 02 (#127) merged. The
series was written on `ab48e389c9951f87e158b5f52930f4d2c9b6d2f4`, after PR 01 (#126) merged, and
replayed onto `f65026dd5` before its first push, together with three review follow-ups (see
"Replay and review follow-ups"). The feat-math sources are final at `d208f4fcf`, and the identity
is recorded at `bfbd1a8b3`; the commits after it change the parity test inventory, the domain
table, three test cases and these records only. Measured on 2026-09-29 on xbabe2:
x86_64-unknown-linux-gnu, glibc 2.39, AMD Ryzen Threadripper PRO 3995WX, rustc 1.97.1 (8bab26f4f
2026-07-14), cargo-nextest 0.9.133, g++ 13.3.0, clang++ 18.1.3, node v26.1.0, Python 3.12.3 (for
the emitted Python program only). Environment: `CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `nice -n 19 ionice -c3`; test runs add the rust-test
lane's `JOPE_PARITY_STRICT=1` and one OpenMP, rayon, OpenBLAS and MKL thread. This is local
evidence, not hosted CI, and the wall times are for this host only.

## What changed

| Commit | Change |
|---|---|
| `eef3b5eb9` | `crates/feat-math/tests/fault_trace.rs` with two ignored probes that call only the legacy API: a digest of every legacy result on the compatibility corpus, and legacy throughput. It came before any source change, so a base build of the probes exists. The target is assigned in `deploy/ci/coverage.json` and classified in the regression test inventory. |
| `eaf2b68cb` | `math_expr/fault.rs`: `FaultSet` (`#[repr(transparent)]` over `u32`), `FaultClass` and `FaultSeverity`, re-exported from `math_expr` (WP-1.1). |
| `930ef9808` | The interpreter reports its guards (WP-1.2, and the evaluator half of WP-1.8). `compute_expr`, `compute_binary`, `compute_func`, `eval_var` and `safe` become one-line wrappers over `*_with` twins that are generic over a crate-private `FaultSink`. The arithmetic and the errors are unchanged; only `sink.raise` calls are added. New: `MathExpr::eval_row_traced`, `eval_element_traced` and `eval_row_traced_strict`. |
| `843701c82` | The compiled form reports the same faults (WP-1.3): `evaluate_with`, a summary cache that keeps each summary's faults, and `CompiledMathExpr::eval_row_traced`, `eval_element_traced` and `map_row_traced`. The legacy compiled path is unchanged. |
| `d208f4fcf` | `FaultClass::ALL` is a `&'static [FaultClass]`, not a `[FaultClass; 16]`, so appending a class does not change its type (review follow-up). |
| `d32069f7d` | `docs/internal/kernel2/math-domain-table.md`: one row per guarded branch. |
| `600333e80` | `crates/jope-campaigns/known-evaluator-identities.json`: the v0.2.0 evaluator identity, with its receipt in `receipts/evaluator-identity-v0.2.0/`. |
| `bfbd1a8b3` | `compat_freeze/evaluator_identity.v1.json`, rebaselined with `JOPE_COMPAT_FREEZE_REGENERATE=identity`. |
| `5279f1d45` | `docs/parity/test-inventory.json`: the feat-math section. |
| `a7a541678` | The domain table's `ILLEGAL_SCOPE` row cites its one raise line, 918 (review follow-up). |
| `319683f64` | The NaN and the infinity that `^` and `/` absorb without a class: the domain table's "Not reported" list, and three cases that pin it (review follow-up). |
| last | These records, `CHANGELOG.md` and `MASTER_PLAN.md`, and one wording fix in the domain table. |

Every commit has Jepson Taylor `<130782313+jepsontaylor@users.noreply.github.com>` as author and
committer.

The public API changes by additions only: `feat_math::math_expr::{FaultSet, FaultClass,
FaultSeverity}`, `MathExpr::{eval_row_traced, eval_element_traced, eval_row_traced_strict}` and
`CompiledMathExpr::{eval_row_traced, eval_element_traced, map_row_traced}`. `FaultClass` and
`FaultSeverity` are `#[non_exhaustive]`, `FaultClass::ALL` is a slice, and `FaultSet`'s field is
private. There is no new dependency, no new variant in an existing enum, no changed signature and
no removed item. `FaultSink`, `NoSink` and `Collect` are `pub(crate)`.

## Replay and review follow-ups

#127 merged as `f65026dd5` while this series was in review. Before its first push, the series was
replayed onto it. The pre-replay head `256711ce2b05d8b0fd1d05aca94f0b6694365f99` is kept locally
as `refs/preserve/kernel2/pr03-prerebase-20260929T134256Z`, so every pre-replay commit named
below stays reachable.

- #127 changes no input of this series' checks: `git diff --quiet ab48e389c f65026dd5 --` over
  compat_freeze's `SOURCE_PATHSPECS` and `crates/feat-math/tests/fixtures/compat_freeze` exits 0.
  It adds `parity_bits`, whose six tests now also run against the new evaluator (below).
- `git range-diff ab48e389c..256711ce2 f65026dd5..319683f64`:
  - unchanged (`=`): `07384a8d0` → `eef3b5eb9`, `167b05450` → `eaf2b68cb`, `04865e314` →
    `930ef9808`, `c649e6c3a` → `843701c82` and `c87fd2843` → `d32069f7d`;
  - changed (`!`): `f0f173dee` → `600333e80`, whose row's `builds` field now ends at `f65026dd5`
    and whose message names the reproduction at `f65026dd5`; `18a7fcd71` → `bfbd1a8b3`,
    regenerated at `600333e80`, where only `generated_at_sha` and `known_identities_sha256` differ
    from `18a7fcd71`'s fixture; `659f2c62c` → `5279f1d45`, the feat-math section re-listed on top
    of #127's inventory (feat-math 475 to 491 tests with 4 to 6 ignored, Rust 3,838 to 3,854 with
    63 to 65 ignored);
  - `256711ce2` is replaced by the last commit, these records, rewritten on top of #127's.
- The review follow-ups are three new commits: `d208f4fcf`, `a7a541678` and `319683f64`. The
  others are in these records: the throughput reading, the consumer test tiers, the lint state of
  `jope-dime-regressor`, and how the known-table row is proved.

## Decisions taken while implementing

1. **The legacy compiled `evaluate` is not rewritten.** The brief says to thread the sink through
   it. Doing that in place would leave `compute_binary`, `compute_func` and `eval_var` without
   callers, which `clippy -D warnings` refuses as dead code, while the brief also says to keep
   every old signature as a wrapper. So `evaluate_with` sits beside `evaluate`, the legacy path
   keeps calling the wrappers, and the shipping compiled path is unchanged. The differential
   proves the two agree.
2. **Faults accumulate until a Hard error**, which ends the evaluation. So every legacy error
   carries exactly one Hard class, and no Hard class comes without an error. The workplan's row
   `clip(col(0), col(1), col(1))` on `[1, NaN]` is recorded as
   `{NONFINITE_INPUT, INVALID_CLIP_BOUNDS}`: by the brief's own rule, reading the NaN bound
   raises `NONFINITE_INPUT` first.
3. **The compat shape returns `(f64, FaultSet)`**, not a `Result`: an error becomes
   `(0.0, faults)`, as the stage fallback's `unwrap_or(0.0)` does. The strict call returns the
   legacy error itself, so its message and type (`MathEvalError::InvalidClipBounds`) are the
   legacy ones.
4. **The interpreter's evaluation-time arity check also raises `COMPILE_FAILED`.** Only a
   deserialized AST can fail it, because `parse` and `from_ast` check arity, and `compile()`
   refuses the same AST; `COMPILE_FAILED` is the class the stage will report for that from WP-1.4
   on. Without this, that error would carry no Hard class. A stage never reaches the check,
   because it compiles first.
5. **`SQRT_PROJECTED` also covers a NaN argument**, which `max(0.0)` replaces as it does a
   negative one; otherwise a generated NaN would become a fault-free 0.0 there. `-0.0` is not a
   fault.
6. **`POW_NONFINITE`'s "finite inputs"** are the base and the exponent actually used, which is the
   clamped one.
7. **`map_row_traced` follows each cell's own evaluation.** Adding the row pass's faults to every
   cell would differ from the interpreter wherever the row pass and an element's evaluation stop
   at different points. `map_cells_report_exactly_what_their_own_evaluation_reaches` pins three
   such cases; in one of them even the Hard class would differ.

The domain table also lists what is not reported: unguarded `+`, `-` and `*`; `asinh` overflow;
a NaN absorbed by `min`, `max`, `step`, `sign`, a single NaN clip bound, or `^` with a zero
exponent or a base of 1; an infinity absorbed where a later operation returns a finite value for
it, as `/` does for an infinite divisor; and the functions without a guard.

## Values unchanged

- **compat_freeze** passes 12 of 12 in compare mode at `319683f64`, twice, so every frozen surface
  is unchanged: MathExpr values and error kinds, stage values, emitted bytes, models, pins,
  streams and search runs.
- **The legacy value digest** covers 191,716 results: 82 expressions × 266 rows, through the
  interpreter and the compiled form, in row, element and map context. It is
  `b64cf538773f70f0f587a7fbbe97de65af1ff7f36c4127eeecbbcdf369d81228` at the base and at the head,
  in the dev profile and in release (`lto = true`, `codegen-units = 1`). It was first taken at the
  pre-replay `07384a8d0` and `c87fd2843`, and is the same at `319683f64` in both profiles. Release
  matters because the dev profile does not inline, and inlining is where a refactor could change,
  for example, the sign of a `min` of two zeros.
- **The evaluation receipt** of the ADR-0004 manifest below, produced with the head build, has the
  same prediction, truth and fitness bits as the base receipt. Of its 33 scalar fields, the 7 that
  differ are the digests downstream of the evaluator identity: `run_contract_sha256`,
  `evaluation_key` (twice), `fit_binding_sha256`, `tape_sha256`, `score_key` and
  `semantic_receipt_sha256`. The head receipt (sha256 `438b8600…`) is the same at the pre-replay
  head and at `319683f64`, and is not committed.
- **The emitted programs**: #127's `parity_bits` finds the same divergences as at #127's head
  (below), and `deploy/ci/emitters.sh` writes sources byte-identical to its pins, with predictions
  bit-identical across programs.

## Tests (`crates/feat-math/tests/fault_trace.rs`: 13 run, 2 ignored)

- **The classes:** bits, names and severities; the severity masks (Hard `0x9c00`, Domain
  `0x611f`, Saturation `0x02e0`, a partition of `0xffff`); set operations and display; every
  name passes `transpile::lint_shipped_text`; `FaultClass::ALL` is bound to
  `&'static [FaultClass]`, so a return to an array does not compile.
- **The fault tables:** the WP-1.2 table and the WP-1.8 list row by row, plus a case for every
  class, each guard's boundary, NaN and infinity absorption and the accumulation order: 62 cases
  (59, and three added in review), through the interpreter and the compiled form. Values are
  compared with the legacy entry points, and with recorded bits where they are free of libm.
  `sqrt(-0.0)` is recorded as +0.0 (bits `0x0000000000000000`).
- **Property, on the compatibility corpus:** 82 expressions × 266 rows, giving 21,812 row
  evaluations and 49,364 element evaluations, through every traced form (the interpreter's row,
  strict and element calls; the compiled row and element calls; the compiled map).
  - Every traced value equals the legacy value bit for bit.
  - A Hard class is raised exactly when the legacy call errs, with that error's kind. The
    row-context errors are COLUMN_OOB 3,153, ILLEGAL_SCOPE 4,523 and INVALID_CLIP_BOUNDS 544,
    8,220 in all: the ERR counts in PR 01's fixture.
  - 11,859 row evaluations and 17,514 element evaluations raise at least one fault.
- **Differential:** 10,000 random expressions from chained edits (the v1 edit operator, with a
  recombination at every third seed from a donor that holds every row summary, a negation and
  `x`), in scopes of 0, 1, 2, 3 and 8 columns, element and row context.
  - Each is checked on all 266 corpus rows through every traced form: 2,660,000 row and 6,020,000
    element evaluations, with 0 differences.
  - 9,216 of the expressions are distinct within their shard (2,296, 2,309, 2,299 and 2,312).
  - Four shards of 2,500 seeds take about 11 s each in a debug build here.
- A deserialized AST with a wrong arity; the three map cases and an empty row; `MathExpr`,
  `CompiledMathExpr` and `FaultSet` stay `Send + Sync`.
- One unit test in `math_expr::fault` for the crate-private sinks.

The per-class counts on the corpus and in the differential are in the test output
(`--no-capture`, the `PROPERTY` and `DIFFERENTIAL` lines). Those lines are byte-identical at the
pre-replay head and in both runs at `319683f64`, sha256
`7550ea66ffd831af28700b13bc4281330a5d34e86f828ec40cefdab0eb593988`.

**Parity with the emitted programs.** #127's `parity_bits` (`JOPE_PARITY_STRICT=1`) passes 6 of 6
at `319683f64`, twice, with the divergences #127 recorded: rust 6, python 6 and cpp17-fitted 10,
all allowlisted, 0 unexpected and 0 stale. The runtime's predictions against the emitted Rust,
Python and fitted C++17 programs are therefore unchanged by the new evaluator code.

**Perturbations caught.** Each was applied to the committed source at the pre-replay head and the
fault_trace tests were run; then the file was restored with `git checkout`, and the tree was clean
afterwards.

| Perturbation | Failing tests |
|---|---|
| A map cell keeps walking past the element-independent instruction that failed | the corpus property and the 4 differential shards (5) |
| A map cell does not raise the element-independent faults again | the corpus property, the map cases, the compiled table and the 4 differential shards (7) |
| `sqrt(NaN)` no longer raises `SQRT_PROJECTED` | the interpreter and compiled tables (2) |
| The empty sum raises `EMPTY_ROW_REDUCTION` | the interpreter and compiled tables (2) |

## Throughput (informational)

The legacy entry points on the corpus in release (`lto = true`, `codegen-units = 1`). The base
binary was built at the pre-replay `07384a8d0`, before any source change; its feat-math sources
are those of `f65026dd5`. Each value is nanoseconds per evaluation, the minimum over 201 timed
passes; the tables give the minimum, median and maximum of that value over the rounds.

First measurement, base against the pre-replay head binary (`c87fd2843`, whose `math_expr.rs` and
`compiled.rs` are those of this head), interleaved over 7 rounds:

| Path | Evaluations per pass | Base min / median / max | Head min / median / max | Median change |
|---|---|---|---|---|
| interpreter `eval_row`, pairs that return a value | 13,592 | 21.400 / 21.517 / 21.603 | 21.232 / 21.262 / 21.341 | −1.19% |
| interpreter `eval_row`, all pairs | 21,812 | 38.038 / 38.118 / 38.253 | 37.185 / 37.360 / 37.507 | −1.99% |
| interpreter `eval_element`, all cells | 49,364 | 40.340 / 40.680 / 40.801 | 39.699 / 39.919 / 40.087 | −1.87% |
| compiled `eval_row`, pairs that return a value | 13,592 | 33.690 / 33.914 / 33.967 | 33.934 / 33.995 / 34.022 | +0.24% |
| compiled `map_row`, all cells | 49,364 | 38.385 / 38.663 / 38.871 | 39.121 / 39.281 / 40.204 | +1.60% |

Second measurement, with the release binary of `319683f64` added: 6 rounds, the order of the three
binaries rotated each round, pinned to CPU 60 (median change against the base in parentheses):

| Path | Base min / median / max | Pre-replay head min / median / max | `319683f64` min / median / max |
|---|---|---|---|
| interpreter `eval_row`, pairs that return a value | 21.354 / 21.418 / 21.592 | 21.143 / 21.200 / 21.301 (−1.02%) | 21.113 / 21.154 / 21.326 (−1.23%) |
| interpreter `eval_row`, all pairs | 37.956 / 38.169 / 38.330 | 37.265 / 37.480 / 37.558 (−1.81%) | 37.742 / 37.791 / 38.141 (−0.99%) |
| interpreter `eval_element`, all cells | 40.594 / 40.607 / 40.818 | 39.631 / 39.897 / 40.119 (−1.75%) | 40.088 / 40.139 / 40.390 (−1.15%) |
| compiled `eval_row`, pairs that return a value | 33.675 / 33.755 / 33.914 | 33.729 / 33.797 / 33.921 (+0.12%) | 33.668 / 33.731 / 33.790 (−0.07%) |
| compiled `map_row`, all cells | 38.271 / 38.410 / 38.746 | 38.899 / 39.151 / 39.687 (+1.93%) | 38.413 / 38.904 / 39.699 (+1.29%) |

**Reading.** The interpreter paths are about 1 to 2% faster at the head, and compiled `eval_row`
is within 0.3%. **Compiled `map_row` is a reproducible slowdown of 1.3% to 2.3%, not noise.** It is
+1.60% above, where the base maximum (38.871) is below the head minimum (39.121), and the review
measured it again three times with the same binaries: +2.3% (CPU 100, 5 rounds), +2.2% (CPU 60, 6
ABBA rounds) and +1.6% (CPU 20, 6 ABBA rounds); in three of the four sets the base and head ranges
do not overlap. At `319683f64` it is +1.29% (the ranges overlap this time), against +1.93% for the
pre-replay head binary in the same rounds: the size of the effect moves with the build's code
layout, but not its sign. The legacy compiled source is byte-unchanged, so the slowdown
comes from how its callees, now one-line wrappers over the sink-generic functions, are compiled
and laid out. It sits at the edge of the ±2% expectation. Among the shipped paths only a map stage
whose expression reads a row summary takes it (`crates/feat-math/src/ops.rs`, the
`FittedOp::MapExpr` arm). A derive stage with a row summary and constant fitting
(`math_expr/evolution.rs`, `optimize_constants_v1`) take compiled `eval_row`; every other map and
derive stage takes the interpreter paths. Changing the inlining would edit `math_expr.rs` and mint
another evaluator identity, so the source is left as it is. The host is shared, and every run was
at nice 19.

## Evaluator identity (ADR-0004)

`math_expr.rs` is in the ADR-0004 hash list (`crates/jope-campaigns/src/evaluation_v2.rs`,
`compute_implementation_sha256`); `math_expr/fault.rs` and `math_expr/compiled.rs` are not. The
series therefore mints one new evaluator identity:

| | Base (v0.2.0) | Head |
|---|---|---|
| `crates/feat-math/src/math_expr.rs` sha256 | `ea00845fa1521c22393775b573e7def76998711d8340d74fd33c5dcb8eca393f` | `47bc9ba5a7e7627d998bca5429f1c593fe364a132e84375bc20840e66baf1e57` |
| `implementation_sha256` | `9bee59f35055d574a73c6b32b6ee9945c01b531c968bb07075d1cc771f0294e9` | `3ec654cb6a44f26d5770bcab589273879605831570025cb42475fa1aad1e1f63` |
| identity digest | `430a0119915927af71d5ef535955cf03e1186199d0e94c7f76370dcc46c3075b` | `891dae89b1b56ef4b95c83ef1a1c5e4501f1da8c9c03c8aea1eaef3bb01efd6b` |
| `cargo_lock_sha256` | `8ecdd1fc461f5700897e2e65dac13e3a05200bfcd2226e3b62d39bbe772c1fee` | unchanged |
| known identity table | `ac0193ae…`, 0 rows | `5a2d18c0f0b8afa119bbb7afa2ae0a361da533d00628232de9a4913ca89dcb6c`, 1 row |

**The known-table row.** The base identity is appended as ADR-0004 prescribes, so receipts made by
v0.2.0 still validate. The ADR asks for the path of a receipt that proves the identity existed.
Its historical receipts are in the learning store's `runs/` prefix, which this lane may not read,
so the receipt was produced by the base build itself, at `ab48e389c` with a clean tree, before
any source change:

- `docs/internal/kernel2/receipts/evaluator-identity-v0.2.0/receipt.json` (sha256
  `8ae5933e700316f56383a419e6ae32da862a0915e23bd28ca23c4f8d394fa2a0`) is a
  `JopeDimeSemanticEvaluationReceipt.v2`. It was written by
  `jope-dime-regressor --registry <dir> --dataset fixture verify-evaluation-v2 --manifest manifest.json`.
  - `<dir>/fixture/train.csv` is the `train.csv` beside it (sha256
    `6fa03c67c3890f73a59f0f3b78cb2cac88dc411e280586d2ccd17efed9adfc09`).
  - `manifest.json` has sha256 `fc636ece2a8cbd2ed22ab7f5ec5fb85cd07da4375c8de3d8c23c521201b512b4`.
- The build of `f65026dd5`, this series' base, with a clean tree, writes the same receipt byte
  for byte from the committed `manifest.json` and `train.csv`, so anyone can reproduce it.
- `git diff --quiet 629e69d86 f65026dd5 -- crates/feat-math/src crates/feat-math/Cargo.toml
  crates/jope-campaigns/src crates/jope-campaigns/Cargo.toml Cargo.lock rust-toolchain.toml`
  exits 0, so v0.2.0 and the base hash the same identity inputs. The row's `builds` field names
  that range.

What ties the receipt to the pair, and what does not:

- The receipt does not record `implementation_sha256` or `cargo_lock_sha256`. Its
  `run_contract_sha256` commits to the run contract, whose prediction context includes the
  evaluator identity digest (`evaluation_v2.rs`, `RunContractV2::digest`), and the run contract
  itself is not written anywhere by the command. The link is therefore shown by reproduction: the
  base build writes this receipt, and the head build (`319683f64`) writes, from the same inputs,
  a receipt (sha256 `438b8600a8fba7c67057b618f942bcef3c7940cb40588c0d0623e000dbc98a6e`, the same
  as at the pre-replay head) whose `run_contract_sha256` (`39ddb22e…` against `aca76fa9…`) and
  the six digests derived from it differ, while every prediction, truth and fitness bit is equal.
  Between the two builds only the evaluator identity in the run contract differs (the fit spec,
  conditioning, math policy, objective and training view are the same), so `aca76fa9…` is the
  run contract of the base identity.
- The pair itself is the one compat_freeze's `evaluator_identity.v1.json` records at `629e69d86`
  and at `f65026dd5`, computed there from the sources: `implementation_sha256` `9bee59f3…`,
  `cargo_lock_sha256` `8ecdd1fc…`, digest `430a0119…`.
- No executed check accepts a receipt through the new row. `known_evaluator_identity` is
  consulted only by `EvaluatorIdentityV1::validate`, for an identity that is not the current
  build's, and no command loads a persisted run contract. The two `evaluation_v2` tests that
  consult the table assert that a fabricated identity (`"4".repeat(64)`) is still rejected. A test
  of acceptance would edit `evaluation_v2.rs`, an identity input, or another `jope-campaigns`
  source, which this change's brief does not authorize.

**The rebaseline.**

- The command: `JOPE_COMPAT_FREEZE_REGENERATE=identity
  JOPE_COMPAT_FREEZE_SHA=600333e80b08938b92f35049c76bdf43b1e3063f cargo nextest run --locked -P
  pr -p feat-math --test compat_freeze`, at `600333e80` (every source change plus the known-table
  row) with a clean tree.
- It passed 12 of 12, so every other surface compared equal while the identity was rewritten.
- The mode also rewrote `lineage_sources.v1.json`, but only its `generated_at_sha` (`629e69d86`
  to `600333e80`). With that field removed, the files are equal under `jq -S`, and no lineage
  source changed. That file was restored with `git checkout`.
- Compare mode afterwards: 12 of 12.
- Against the pre-replay fixture (`18a7fcd71`), only `generated_at_sha` (`f0f173dee` to
  `600333e80`) and `known_identities_sha256` (`70ded9e5…` to `5a2d18c0…`, the row's `builds`
  field) differ.
- `evaluator_identity.v1.json` names `600333e80`, a commit of this branch. A merge that rewrites
  commits will not carry that id; the content equality can be rechecked with
  `git diff --quiet 600333e80 <merged commit> --` followed by compat_freeze's `SOURCE_PATHSPECS`.

The fixtures against the base (`git diff --stat f65026dd5 HEAD -- crates/feat-math/tests/fixtures/`):

```text
 .../tests/fixtures/compat_freeze/evaluator_identity.v1.json  | 12 ++++++------
 1 file changed, 6 insertions(+), 6 deletions(-)
```

## Commands and results

At `319683f64`, the tree of the last commit apart from these records, with a clean tree. Where a
step ran twice, both exit codes and wall times are given.

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.8 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings | 0.3 s (the lint build was current) |
| jope-dime-regressor clippy | `cargo clippy -p jope-dime-regressor --all-targets --locked -- -D warnings` | 101 | 30 library and 38 test-build errors, at exactly the locations the same command reports at the base `f65026dd5`, all in files this change does not touch (see "Still open") | 31.1 s |
| feat-math PR tier | `cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 481 passed; 10 skipped (6 ignored, 4 by the pr filter) | 59.8 s, 59.6 s |
| Parity with the emitted programs | `cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0, 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale | 3.6 s, 3.5 s |
| Named targets | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test mathexpr_current_semantics --test fault_trace --test ip_scrub --test legacy_v2_freeze --test portable_parity` | 0, 0 | 53 passed (12, 18, 13, 6, 3, 1); 3 skipped | 12.1 s, 12.1 s |
| Property and differential output | the corpus property and the four differential shards, `--no-capture` | 0, 0 | 5 passed; the `PROPERTY` and `DIFFERENTIAL` lines are byte-identical in both runs and to the pre-replay head's (sha256 `7550ea66…`) | 44.0 s, 44.0 s |
| Evaluator identity tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0, 0 | 15 passed: 13 in `evaluation_v2` and 2 in `commands::exact_cv_v2` | 50.6 s with the build, 0.8 s |
| Main tier, whole workspace | `just test-main` (`cargo nextest run -P main --workspace --locked -j 32`, one thread each for OpenMP, rayon, OpenBLAS and MKL, `JOPE_PARITY_STRICT=1`) | 100 | 3,789 run: 3,788 passed, 1 failed, 65 skipped. The consumers of feat-math and of the known-identity table all pass: `jope-parallel` 775 (with `bounded_smartcluster`, whose `scqd` this workspace build provides, and `resident_protocol`) and `jope-dime-regressor` 997; feat-math 485, including the four tests the pr filter skips. The one failure is below. | 682.6 s |
| The main-tier failure, alone | `cargo nextest run --locked -P main -p jainnode -E 'binary_id(jainnode) & test(=executor::tests::dispatch_tests::concurrent_runtime_accepts_durable_cancellation_while_execution_runs)'`, five times | 0 × 5 | 1 passed each time (0.66 to 0.87 s). In the main tier it failed after 8.5 s, at load average 40, because no `Cancelled` terminal frame arrived (`dispatch_tests.rs:328`). jainnode depends on no crate this change touches. | 15.4 s |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 287 targets, 0 unassigned or invalid | 0.5 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 323 items | 0.2 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | 10 passed | 0.3 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python` | 0 | consistent | 0.1 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.4 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok; gitleaks found no leaks; repository layout 0 findings | 19.7 s |
| Whitespace | `git diff --check f65026dd5 HEAD` | 0 | clean | 0.0 s |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 4 built and tested with `--locked`, 61 tests passed; cargo-deny skipped (0.19.8 is installed, and the lane asserts 0.20.2), so that step is not a pass | 31.3 s |
| Emitters | `EMITTERS_OUT=<scratch> bash deploy/ci/emitters.sh` | 0 | both programs: sources byte-identical to the pins, predictions bit-identical across the 2 and 7 programs | 16.4 s |
| Legacy value digest, dev | the `legacy_value_digest` probe, `--run-ignored ignored-only --no-capture` | 0 | `b64cf538…`, 191,716 results | 0.8 s |
| Legacy value digest, release | the release `fault_trace` binary of `319683f64`, `--ignored --exact legacy_value_digest`, and the base release binary (`07384a8d0`) again | 0 | `b64cf538…` from both | 0.03 s each |
| Release tests | the release `fault_trace` binary of `319683f64`, every test | 0 | 13 passed, 2 ignored | 1.7 s |
| ADR-0004 receipt, base | `verify-evaluation-v2` with the build of `f65026dd5` on the committed manifest and CSV | 0 | byte-identical to the committed `receipt.json` | 56.1 s with the build |
| ADR-0004 receipt, head | the same with the build of `319683f64` | 0 | sha256 `438b8600…`; only the 7 identity-derived digests differ from the base receipt | 53.2 s with the build |

This commit changes documentation only: `git diff --stat 319683f64 HEAD` lists `CHANGELOG.md`,
`MASTER_PLAN.md`, the domain table and this receipt. With these records in the working tree, the
checks that read them ran again and exit 0: `cargo fmt --all -- --check`, `coverage validate`
(287 targets), `verify-test-inventory` (323 items), `test-inventory.sh --no-rust --static-python`,
`ip-scrub.sh` (47 files), `check.sh` and `git diff --check f65026dd5`; `compat_freeze` (12) and
`fault_trace` (13) pass again.

The brief's identity-test command, `-P pr -p jope-campaigns --test …`, cannot select these tests:
the package is `jope-dime-regressor`; the `pr` profile's default filter excludes that package;
and `evaluation_v2` and `commands` are modules of the `jope-dime-regressor` binary, not of its
library or of a test target. The command above is the one that runs them.

## Still open

- **Hosted CI.** Every lane runs for this change: `crates/feat-math/**`,
  `known-evaluator-identities.json` and `deploy/ci/coverage.json` select all nine jobs. The first
  hosted run is also the cross-host check of the libm-dependent values.
- **The PR number** is written as `#NNN` in `CHANGELOG.md` and `MASTER_PLAN.md`; it exists only
  once the pull request is opened, and replacing it changes only those two files.
- **The ADR-0004 amendment** (plan WP-8.3): `math_expr/compiled.rs` and `math_expr/fault.rs`
  change evaluation behaviour or its reporting but are outside the hash list. The amendment needs
  Codex's agreement.
- **Clippy `-D warnings` on `jope-dime-regressor`** fails at the base `f65026dd5` and at this head
  alike: 30 errors in the library and 38 in its test build, all in 16 files under
  `crates/jope-campaigns/src` that this change does not touch (`hyperion_gate.rs`,
  `selfplay_shadow/v3.rs`, `selfplay_data.rs`, `training_history.rs`, `spec_drift.rs` and
  others). The only `jope-campaigns` file this change edits is the known-identity table, which
  `evaluation_v2.rs` reads with `include_str!`, and it adds no finding. Clearing them edits
  existing code, so it needs a separate lint pull request or an owner waiver. feat-math's clippy is
  clean.
- **An executed check that the known-table row is accepted** needs a `jope-campaigns` change (see
  "The known-table row").
- **A main-tier race outside this change.** jainnode's
  `concurrent_runtime_accepts_durable_cancellation_while_execution_runs` failed once in
  `just test-main`, under load, and passed 5 of 5 alone. jainnode uses no crate this change
  touches, and the test is not in `deploy/ci/known-broken-tests.txt`.
- **The feat-math merge announcement** in `AGENT_CHAT.md` belongs to the orchestrator.
