# Scalar evaluator: domain table and fault classes

kernel2 WP-1.1 to WP-1.3, with the evaluator half of WP-1.8; the stage level (WP-1.4 and the rest
of WP-1.8) is in the last three sections. The code is in
`crates/feat-math/src/math_expr.rs` (the guards and every raise site),
`crates/feat-math/src/math_expr/fault.rs` (the classes, the sinks and the interpreter's traced
entry points) and `crates/feat-math/src/math_expr/compiled.rs` (the compiled form's traced entry
points). The tests are in `crates/feat-math/tests/fault_trace.rs`.

Nothing here changes a value. Every guard returns what it returned before this change; a fault
only records that a guard fired. The legacy entry points (`eval_row`, `eval_element`, the compiled
`eval_row`, `eval_element` and `map_row`) are unchanged and report nothing.

## Classes

Bits are append-only: a class keeps its bit, name and severity forever, and a new class takes the
next free bit. `FaultSet` is a `u32`, so bits 16 to 31 are free.

| Bit | Class | Severity | Raised in |
|---|---|---|---|
| 0 | `DIV_NEAR_ZERO` | Domain | `compute_binary_with` |
| 1 | `LOG_DOMAIN` | Domain | `compute_func_with` |
| 2 | `LOG1P_DOMAIN` | Domain | `compute_func_with` |
| 3 | `SQRT_PROJECTED` | Domain | `compute_func_with` |
| 4 | `POW_NONFINITE` | Domain | `compute_binary_with` |
| 5 | `POW_EXP_CLAMPED` | Saturation | `compute_binary_with` |
| 6 | `EXP_ARG_CLAMPED` | Saturation | `compute_func_with` |
| 7 | `SIGMOID_ARG_CLAMPED` | Saturation | `compute_func_with` |
| 8 | `NONFINITE_SANITIZED` | Domain | `safe_with` |
| 9 | `MAGNITUDE_CLAMPED` | Saturation | `safe_with` |
| 10 | `INVALID_CLIP_BOUNDS` | Hard | `compute_func_with` |
| 11 | `COLUMN_OOB` | Hard | `eval_var_with` (through `raise_read`) |
| 12 | `ILLEGAL_SCOPE` | Hard | `eval_var_with` (through `raise_read`) |
| 13 | `NONFINITE_INPUT` | Domain | `eval_var_with` (through `raise_read` and `raise_reduction`) |
| 14 | `EMPTY_ROW_REDUCTION` | Domain | `eval_var_with` (through `raise_reduction`) |
| 15 | `COMPILE_FAILED` | Hard | the stage when `compile()` fails (WP-1.4); `compute_expr_with` for a wrong arity |

Severity masks: Hard `0x9c00`, Domain `0x611f`, Saturation `0x02e0`. They are disjoint and
together make `0xffff`.

- **Hard**: the expression cannot be evaluated on this input, and the legacy entry point returns
  an error. The stage fallback's value is 0.0.
- **Domain**: an input was outside the operation's mathematical domain, or a value was not
  finite, and a guard substituted its fallback.
- **Saturation**: a value was clamped to a bound; the result is still a number.

## The guarded branches

One row per guarded branch in `compute_binary_with`, `compute_func_with`, `eval_var_with` and
`safe_with`, and the one in `compute_expr_with`. "Line" is the `sink.raise` line in
`crates/feat-math/src/math_expr.rs` at this change's head. Each guard test in the "Fires when"
column is the code's own expression; where the raise needs a test of its own, that test is
written so that NaN never fires a clamp class (`v.abs() > bound` is false for NaN).

| Line | Operation | Mathematical domain | Fires when | Fallback (unchanged) | Class | Severity |
|---|---|---|---|---|---|---|
| 719 | `a / b` | `b != 0` | `b.abs() < 1e-12` (strict: 1/1e-12 is exactly 1e12) | 0.0 | `DIV_NEAR_ZERO` | Domain |
| 727 | `a ^ b`, exponent | the evaluator clamps the exponent to `[-8, 8]` | `b.abs() > 8.0` (±inf too; NaN is not clamped and does not fire) | the exponent is `b.clamp(-8.0, 8.0)` | `POW_EXP_CLAMPED` | Saturation |
| 732 | `a ^ b`, result | a finite power: `a > 0`, or `a < 0` with an integer exponent, or `a = 0` with an exponent of at least 0; and no overflow | the power is not finite while `a` is finite and `b` is not NaN, so the clamped exponent is finite | none at the site: the value propagates, to 0.0 at the output filter unless a parent absorbs it | `POW_NONFINITE` | Domain |
| 749 | `sqrt(v)` | `v >= 0` | `v < 0.0 \|\| v.is_nan()` (`-0.0` does not fire) | `v.max(0.0)` first: `sqrt(0.0)` = +0.0; `sqrt(-0.0)` is +0.0 at the base | `SQRT_PROJECTED` | Domain |
| 755 | `log1p(v)` | `v > -1` | `v <= -1.0` (NaN does not fire; `ln_1p(NaN)` is NaN) | 0.0 | `LOG1P_DOMAIN` | Domain |
| 766 | `sigmoid(v)` | all reals; the clamp keeps `exp` finite in every emitted language | `v.abs() > 30.0` | the argument is `v.clamp(-30.0, 30.0)` | `SIGMOID_ARG_CLAMPED` | Saturation |
| 776 | `clip(v, b1, b2)` | at least one bound is a number | `lo.is_nan() \|\| hi.is_nan()`, where `lo = b1.min(b2)`, `hi = b1.max(b2)`: both bounds NaN | the legacy error `MathEvalError::InvalidClipBounds`; 0.0 in the compat shape and the stage | `INVALID_CLIP_BOUNDS` | Hard |
| 794 | `exp(v)` | all reals; the clamp keeps `exp` finite in every emitted language | `v.abs() > 30.0` | the argument is `v.clamp(-30.0, 30.0)` | `EXP_ARG_CLAMPED` | Saturation |
| 802 | `log(v)` | `v > 0` | `v <= 1e-300`: tiny positives in `(0, 1e-300]` fire too; NaN does not (`ln(NaN)` is NaN) | 0.0 | `LOG_DOMAIN` | Domain |
| 918 (from 858) | read `x` | map (element) context only | row context: `ctx.x` is `None` | the legacy error "variable x is only valid in map expressions"; 0.0 in the compat shape and the stage | `ILLEGAL_SCOPE` | Hard |
| 916 (from 858) | read `x` | a finite element | the element read is not finite | none: the value is returned as read | `NONFINITE_INPUT` | Domain |
| 918 (from 863) | read `col(i)` | `i < row.len()` | `row.get(i)` is `None` | the legacy error "column index i out of bounds"; 0.0 in the compat shape and the stage | `COLUMN_OOB` | Hard |
| 916 (from 863) | read `col(i)` | a finite cell | the cell read is not finite | none: the value is returned as read | `NONFINITE_INPUT` | Domain |
| 928 (from 872-896) | `row_mean`, `row_sum`, `row_l1`, `row_l2`, `row_min`, `row_max`, `row_std` | finite cells | any cell of the row is not finite | none: the reduction is computed as before | `NONFINITE_INPUT` | Domain |
| 931 (from 872) | `row_mean` | a non-empty row | the row is empty | the empty sum over `max(len, 1)`: -0.0 | `EMPTY_ROW_REDUCTION` | Domain |
| 931 (from 896) | `row_std` | a non-empty row | the row is empty | the root of the empty sum over `max(len, 1)`: -0.0 | `EMPTY_ROW_REDUCTION` | Domain |
| 931 (from 888) | `row_min` | a non-empty row | the row is empty | the fold's start, +inf, which the output filter maps to 0.0 | `EMPTY_ROW_REDUCTION` | Domain |
| 931 (from 892) | `row_max` | a non-empty row | the row is empty | the fold's start, -inf, which the output filter maps to 0.0 | `EMPTY_ROW_REDUCTION` | Domain |
| 950 | the output filter | a finite result | `!value.is_finite()`, whether generated or read from the input | 0.0 | `NONFINITE_SANITIZED` | Domain |
| 946 | the output filter | `\|value\| <= 1e12` | `value.abs() > MAX_ABS` for a finite value (exactly ±1e12 does not fire) | `value.clamp(-1e12, 1e12)` | `MAGNITUDE_CLAMPED` | Saturation |
| 696 | a function call | the function's arity | `validate_arity` fails: only a deserialized AST can reach this, because `parse` and `from_ast` check arity | the legacy error "`F` received `n` arguments"; 0.0 in the compat shape. `compile()` refuses the same AST, and the stage reports that as `COMPILE_FAILED` from WP-1.4 on | `COMPILE_FAILED` | Hard |

The empty sum, `row_l1` and `row_l2` of an empty row are true identities (-0.0 with this
toolchain) and raise nothing. `row_std` of a single element is +0.0 and raises nothing.

## Order, and the Hard rule

- Faults accumulate in evaluation order until the evaluation ends: depth first, left operand
  before right, arguments in order, then the output filter.
- A Hard class is raised immediately before the legacy error is returned, and nothing after it
  runs. So every legacy error carries exactly one Hard class, together with every fault raised
  before it, and no Hard class is raised without an error. `clip(col(0), col(1), col(1))` on
  `[1, NaN]` reports `{NONFINITE_INPUT, INVALID_CLIP_BOUNDS}`; `col(5) + 1/col(1)` on `[1, 0]`
  reports `{COLUMN_OOB}` and `1/col(1) + col(5)` reports `{DIV_NEAR_ZERO, COLUMN_OOB}`.
- The compat shape (`eval_row_traced`, `eval_element_traced`) returns `(0.0, faults)` for a
  legacy error, the value the stage fallback (`unwrap_or(0.0)`) computes. The strict form
  (`eval_row_traced_strict`) returns the legacy error itself, unchanged, exactly where `eval_row`
  does.

## The compiled form

`CompiledMathExpr::eval_row_traced`, `eval_element_traced` and `map_row_traced` report exactly
the interpreter's faults:

- `evaluate_with` is `evaluate` with the sink threaded through. The postorder instructions run in
  the interpreter's evaluation order and stop at the same first error. The legacy `evaluate`,
  `eval_row`, `eval_element` and `map_row` are unchanged.
- A row summary is computed once per call and cached with the faults its computation raised.
  Every read of the cached summary raises them again, as the interpreter does by computing the
  summary at each read.
- `map_row_traced` returns `map_row`'s values bit for bit, and for each cell the faults of the
  interpreter's `eval_element_traced` on that element. As in `map_row`, the element-independent
  instructions run once per row, each keeping its own faults. A cell re-raises those of every
  such instruction its own evaluation reaches, in instruction order, and stops where that
  evaluation stops. A failed instruction therefore never reports the faults of a later one, and
  a cell whose own evaluation fails first reports its own Hard class, not the row pass's.
- A stage's faults over a row are its cells' faults. An empty row has no cell, so it has no
  fault, whatever its element-independent instructions raised.

`compiled.rs` and `fault.rs` are outside the ADR-0004 evaluator identity's hash list
(`crates/jope-campaigns/src/evaluation_v2.rs`, `compute_implementation_sha256`); `math_expr.rs`,
which holds every guard and raise site, is in it.

## Not reported

These produce or absorb a non-finite value without a class of their own. Values are unchanged;
the list is what a later change could add.

- **`+`, `-` and `*` have no guard.** An overflow to ±inf, or `inf - inf` = NaN, is reported
  only if the non-finite value reaches the output filter (`NONFINITE_SANITIZED`).
- **`asinh` overflows** (`crate::ops::asinh_stable`: `v * v` overflows for `|v|` above about
  1.34e154, and `ln(inf)` is inf). There is no site class; the output filter reports
  `NONFINITE_SANITIZED` if the value reaches it (OQ-7 keeps the value).
- **Branches that absorb a NaN without a class:** `min` and `max` (IEEE minNum and maxNum return
  the other operand), `step` (a comparison with NaN is false, so 0.0), `sign` (NaN is 0.0), a
  `clip` with one NaN bound (minNum and maxNum pick the other bound), and `^` with a zero
  exponent or a base of 1 (`pow(NaN, ±0)` and `pow(1, NaN)` are 1.0, which is finite, so neither
  `^` test fires). A NaN read from the input is still reported at the read (`NONFINITE_INPUT`),
  and one generated by `^` at the `^` (`POW_NONFINITE`); a NaN from an unguarded `+`, `-` or
  `*`, absorbed this way, is not reported at all: `min(col(0)*col(0) - col(0)*col(0), 1)` and
  `(col(0)*col(0) - col(0)*col(0))^0` on `[1e200]` are 1.0 with no fault.
- **An infinity absorbed without a class:** an overflow of an unguarded `+`, `-` or `*` to ±inf
  is likewise lost wherever a later operation returns a finite value for it. A finite value
  divided by it is ±0.0, because the divisor guard tests only `|b| < 1e-12`:
  `1/(col(0)*col(0))` on `[1e200]` is +0.0 with no fault. `^` with a zero exponent gives 1.0;
  `tanh`, `atan` and `erf` give ±1, ±π/2 and ±1; and `min`, `max`, `clip`, `step` and `sign` can
  also return a finite value for an infinite operand.
- **Functions without a guard:** `abs`, `sin`, `cos`, `tanh`, `atan`, `erf`, `floor`, `round`.
  `sin` and `cos` of ±inf are NaN, reported only at the output filter.

The `min`, `^` and `/` examples above are cases in `class_cases` of `tests/fault_trace.rs`,
checked through the interpreter and the compiled form.

## Where this differs from the workplan

The workplan's tables (§7, WP-1.1, WP-1.2 and WP-1.8) are kept, with these readings, each
recorded at the base or chosen here and pinned in `tests/fault_trace.rs`:

- `sqrt(col(0))` on `[-0.0]` is +0.0 (bits `0x0000000000000000`) at the base, not -0.0.
- `clip(col(0), col(1), col(1))` on `[1, NaN]` reports `NONFINITE_INPUT` as well as
  `INVALID_CLIP_BOUNDS`: reading the NaN bound fires first (WP-1.2 step 5), and faults raised
  before a Hard error are kept.
- An empty `row_mean`, `row_sum`, `row_l1`, `row_l2` and `row_std` is -0.0, not +0.0 (PR 01's
  finding); the tables' "0.0" is the value.
- `SQRT_PROJECTED` also fires for a NaN argument, which `max(0.0)` replaces as it does a negative
  one. Otherwise a generated NaN would become a fault-free 0.0 here.
- `POW_NONFINITE`'s "finite inputs" are the base and the exponent actually used, which is the
  clamped one: an infinite exponent is clamped to ±8 (`POW_EXP_CLAMPED`), and a finite base
  raised to it that overflows is `POW_NONFINITE`.
- `COMPILE_FAILED` is also raised by the interpreter's evaluation-time arity check, which is the
  structural check `compile()` makes, so that every evaluation error carries a Hard class. Inside a
  stage that check is never reached: the stage compiles first.

## Stage telemetry (WP-1.4)

`crates/feat-math/src/kernel/telemetry.rs` counts, stage by stage, what the evaluator reports cell
by cell. `apply_op_traced`, `fit_traced` and `predict_traced` return what `apply_op`, `fit` and
`Model::predict` return: the same rows bit for bit, a model with the same `serde_json` bytes, the
same prediction bits. The tests are in `crates/feat-math/tests/stage_telemetry.rs`.

The module is under `kernel/`, which neither the ADR-0004 identity list nor any jope-engine lineage
hashes, so it mints no identity. The code it needs is private to `ops.rs` (the stage epilogue
`finish` and its append flag `op_append`) or lives in the identity-bearing `model.rs` (`fit` and
`predict`), so the module mirrors it line for line rather than editing it, and the tests hold each
mirror to its original.

**Where each value and its faults come from.** A stage takes `apply_op_generated`'s branch
(`crates/feat-math/src/ops.rs`, the `MapExpr` and `DeriveExpr` arms, lines 1159 to 1186 at this
change's base):

| Stage | Value (unchanged) | Faults of each generated cell |
|---|---|---|
| `map` whose expression reads a row summary | compiled `map_row` | compiled `map_row_traced` |
| `map`, otherwise | `eval_element(x, row).unwrap_or(0.0)` | `eval_element_traced` |
| `derive` whose expression reads a row summary | compiled `eval_row(row).unwrap_or(0.0)` | compiled `eval_row_traced` |
| `derive`, otherwise | `eval_row(row).unwrap_or(0.0)` | `eval_row_traced` |
| `map` or `derive` when `compile()` fails | 0.0 for every cell of the row (`map`); one 0.0 per row (`derive`) | `COMPILE_FAILED` |
| a symbolic program, in `fit_traced` and `predict_traced` | `eval_row(row).unwrap_or(0.0)`, then 0.0 for a non-finite value (`model.rs` lines 84 to 88 and 198 to 199) | `eval_row_traced` |
| every other op | `apply_op_generated` itself | none |

A symbolic program is a model, not an op, so `apply_op_traced` has no case for it. It is never
compiled: `fit` and `predict` call the interpreter directly, so a wrong arity from a deserialized
AST is the interpreter's `COMPILE_FAILED` on every row there, and the stage's `compile_failed`
flag stays false. That is the one place where the interpreter's arity check is reached.

**The counts of one stage** (`StageFaultCounts`):

| Field | What it counts |
|---|---|
| `evaluated_cells` | Cells the expression generated: every cell of a `map` row (an empty row generates none) and one cell per `derive` row. Counted before the append merge and the width cap, so a cell the cap cuts still counts. 0 for every other op. |
| `faulted_cells[i]` | Generated cells whose faults include the class whose bit is `i`. A cell with two classes counts once under each, so the classes can add up to more than `evaluated_cells`. There are 32 slots, one per bit of `FaultSet`, so an appended class keeps the field's type. |
| `hard_rows` | Rows with at least one generated cell whose faults include a Hard class. Each such cell is the fallback 0.0 of a legacy error (`unwrap_or(0.0)`) or of a compile failure. A `map` row with no cells is never a hard row. |
| `compile_failed` | `compile()` refused the stage's expression. Every generated cell is then 0.0 and carries `COMPILE_FAILED`. |
| `constant_output` | At least one cell was generated, and every generated cell has the first one's bits. One cell is trivially constant; +0.0 and -0.0 differ. The zeros of a compile failure, or of a Hard fault on every cell, are constant. |
| `epilogue_sanitized` | Non-finite values that the stage epilogue (`finish`, `ops.rs` lines 1295 to 1317) replaced with 0.0, for every op. The epilogue runs after the append merge and the width cap, so a merged input cell counts and a cell the cap cuts does not. The 0.0 that pads an empty row is not a replacement. The expression's own output filter is `faulted_cells[NONFINITE_SANITIZED]`, never this counter, so the two never count the same value. |

**The counts of one pass** (`PipelineTelemetry`):

- `stages`: one entry per fitted transform of a pipeline, in order, so entry `i` is the stage of
  `ops[i]`; the target stage and the head have none. One entry for a symbolic program's
  expression, with one cell per row. None for an explicit portfolio, which `fit_traced` hands to
  `fit` and whose members are fitted and applied inside `portfolio.rs`. The members of a compact
  portfolio head are not traced either; the transforms before the head are.
- `input_sanitized`: the non-finite input cells that a pipeline's entry sanitize replaced with 0.0
  before its first stage. It is 0 for symbolic and portfolio models, which run no such sanitize.

**What a stage in a pipeline can report.**

- The pipeline sanitizes its input on entry and every stage's output in its epilogue, so every
  value a stage reads is finite, and `NONFINITE_INPUT` never fires in a pipeline stage. The
  entry's replacements are `input_sanitized` instead, so a non-finite input never becomes a
  fault-free 0.0. Only the symbolic path, which reads raw inputs, and direct calls on raw rows
  report `NONFINITE_INPUT`.
- An expression's output is always finite (the output filter), so an expression stage's
  `epilogue_sanitized` counts only merged input cells, which in a pipeline are finite too. Other
  ops can overflow: `symlog`'s asinh above about 1.34e154, or `polar`'s radius.
- `fit` applies `apply_op_train`, which forwards every op to `apply_op`, so the training pass of
  `fit_traced` reports what `predict_traced` reports on the training rows. The tests check it.

## Zero-width input (WP-1.8)

- `fit_traced` and `predict_traced` refuse a zero-width input, one with a row that has no columns,
  with the error `ZeroWidthInput { row }`: "zero-width input: row `i` has no columns", where `i` is
  the first such row. `ZeroWidthInput::REASON` is "zero-width input", and `ZeroWidthInput::find` is
  the check. `fit_traced` runs `fit`'s own check first, with its message ("training data is empty
  or target is short"). A set with no rows is not zero-width: `predict_traced` returns no
  predictions for it, as `predict` does.
- This is the one place where the traced entry points differ from the legacy ones, and the model
  and prediction identities hold on every input they accept. The legacy `fit` accepts a zero-width
  input for a pipeline without transforms and for a symbolic program, and a transform stage
  refuses it itself ("empty input to stage `op`", `ops.rs` lines 326 to 328). `predict` does not
  check.
- No stage produces a zero-width row: the epilogue pads an empty row with one 0.0.

## The empty row at stage level (WP-1.8)

The empty-row identities of the table above, seen from a stage. A `derive` stage on an empty row
generates one cell, whose value and faults are the evaluator's:

| Expression on `[]` | Value | Faults |
|---|---|---|
| `row_sum`, `row_l1`, `row_l2` | -0.0 | none: these are true identities |
| `row_mean`, `row_std` | -0.0 | `EMPTY_ROW_REDUCTION` |
| `row_min`, `row_max` | +0.0: the fold's start, +inf or -inf, through the output filter | `EMPTY_ROW_REDUCTION`, `NONFINITE_SANITIZED` |

A `map` stage on an empty row generates no cell, and its output row is the epilogue's single 0.0.
`stage_level_empty_row_identities` in `tests/stage_telemetry.rs` checks both.

## Where the stage level differs from the workplan

- **The files.** WP-1.4 puts `apply_op_traced` in `ops.rs` and `fit_traced` in `model.rs`. Here
  both are in the new `kernel/telemetry.rs`: `model.rs` is a jope-engine lineage source, `ops.rs`
  and `lib.rs` are in the ADR-0004 list, and `finish` and `op_append` are private. No evaluator
  identity moves, and the mirrored code is held to its original by test instead.
- **`faulted_cells` is `[u64; 32]`**, not `[u64; N_CLASSES]`, as `FaultClass::ALL` is a slice:
  an appended class keeps the type.
- **Two sanitize counters.** `epilogue_sanitized` is separate from the expression's classes, as
  WP-1.4 step 3 asks. `input_sanitized` is new: without it a non-finite pipeline input would be a
  fault-free zero, which WP-1.8 rules out.
- **`predict_traced` returns a `Result`**, so that it can refuse a zero-width input.
- **The symbolic model** is traced in `fit_traced` and `predict_traced`; `apply_op_traced` covers
  ops only.
- **Portfolios.** An explicit portfolio is fitted by `fit` and reports no stage; see above.
