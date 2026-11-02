use super::*;

pub(crate) struct EvalContext<'a> {
    pub(crate) x: Option<f64>,
    pub(crate) row: &'a [f64],
}

pub(crate) fn compute_expr(expr: &Expr, ctx: &EvalContext<'_>) -> Result<f64> {
    compute_expr_with(expr, ctx, &mut NoSink)
}

/// The evaluator, reporting every guard it takes to `sink` (kernel2 WP-1.2). The arithmetic is
/// the legacy evaluator's, unchanged, and so is every error; only the `sink.raise` calls are new.
/// Each raise site is a row of docs/internal/kernel2/math-domain-table.md.
fn compute_expr_with<S: FaultSink>(
    expr: &Expr,
    ctx: &EvalContext<'_>,
    sink: &mut S,
) -> Result<f64> {
    match expr {
        Expr::Const(v) => Ok(*v),
        Expr::Var(var) => eval_var_with(*var, ctx, sink),
        Expr::Unary(UnOp::Neg, child) => Ok(-compute_expr_with(child, ctx, sink)?),
        Expr::Binary(op, left, right) => {
            let a = compute_expr_with(left, ctx, sink)?;
            let b = compute_expr_with(right, ctx, sink)?;
            Ok(compute_binary_with(*op, a, b, sink))
        }
        Expr::Call(func, args) => {
            // Only a deserialized AST reaches this with a wrong arity: parse and from_ast check
            // it, and compile() refuses such an AST, which the stage reports as COMPILE_FAILED.
            let arity = validate_arity(*func, args.len());
            if arity.is_err() {
                sink.raise(FaultSet::COMPILE_FAILED);
            }
            arity?;
            let mut values = [0.0; 3];
            for (slot, arg) in values.iter_mut().zip(args) {
                *slot = compute_expr_with(arg, ctx, sink)?;
            }
            compute_func_with(*func, values, sink)
        }
    }
}

fn compute_binary(op: BinOp, a: f64, b: f64) -> f64 {
    compute_binary_with(op, a, b, &mut NoSink)
}

fn compute_binary_with<S: FaultSink>(op: BinOp, a: f64, b: f64, sink: &mut S) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            if b.abs() < 1e-12 {
                sink.raise(FaultSet::DIV_NEAR_ZERO);
                0.0
            } else {
                a / b
            }
        }
        BinOp::Pow => {
            if b.abs() > 8.0 {
                sink.raise(FaultSet::POW_EXP_CLAMPED);
            }
            let value = a.powf(b.clamp(-8.0, 8.0));
            // Non-finite from a finite base and a finite (clamped) exponent: b is not NaN.
            if !value.is_finite() && a.is_finite() && !b.is_nan() {
                sink.raise(FaultSet::POW_NONFINITE);
            }
            value
        }
    }
}

fn compute_func(func: Func, values: [f64; 3]) -> Result<f64> {
    compute_func_with(func, values, &mut NoSink)
}

fn compute_func_with<S: FaultSink>(func: Func, values: [f64; 3], sink: &mut S) -> Result<f64> {
    Ok(match func {
        Func::Abs => values[0].abs(),
        Func::Sqrt => {
            // max(0.0) replaces a negative argument and NaN; -0.0 is not a fault.
            if values[0] < 0.0 || values[0].is_nan() {
                sink.raise(FaultSet::SQRT_PROJECTED);
            }
            values[0].max(0.0).sqrt()
        }
        Func::Log1p => {
            if values[0] <= -1.0 {
                sink.raise(FaultSet::LOG1P_DOMAIN);
                0.0
            } else {
                values[0].ln_1p()
            }
        }
        Func::Sin => values[0].sin(),
        Func::Cos => values[0].cos(),
        Func::Tanh => values[0].tanh(),
        Func::Sigmoid => {
            if values[0].abs() > 30.0 {
                sink.raise(FaultSet::SIGMOID_ARG_CLAMPED);
            }
            1.0 / (1.0 + (-values[0].clamp(-30.0, 30.0)).exp())
        }
        Func::Min => values[0].min(values[1]),
        Func::Max => values[0].max(values[1]),
        Func::Clip => {
            let lo = values[1].min(values[2]);
            let hi = values[1].max(values[2]);
            if lo.is_nan() || hi.is_nan() {
                sink.raise(FaultSet::INVALID_CLIP_BOUNDS);
                return Err(MathEvalError::InvalidClipBounds.into());
            }
            values[0].clamp(lo, hi)
        }
        // 1.0 exactly when a > c, else 0.0; a NaN operand makes the
        // strict comparison false, so the indicator is 0.0.
        Func::Step => {
            if values[0] > values[1] {
                1.0
            } else {
                0.0
            }
        }
        // Argument clipped to [-30, 30] so exp stays finite in both
        // artifact languages (mirrors sigmoid's guard).
        Func::Exp => {
            if values[0].abs() > 30.0 {
                sink.raise(FaultSet::EXP_ARG_CLAMPED);
            }
            values[0].clamp(-30.0, 30.0).exp()
        }
        // Natural log, 0.0 at or below 1e-300 (mirrors log1p's
        // invalid-argument policy).
        Func::Log => {
            if values[0] <= 1e-300 {
                sink.raise(FaultSet::LOG_DOMAIN);
                0.0
            } else {
                values[0].ln()
            }
        }
        Func::Asinh => crate::ops::asinh_stable(values[0]),
        Func::Atan => values[0].atan(),
        // Abramowitz-Stegun 7.1.26 rational approximation, identical
        // arithmetic in the emitted twins (Rust std has no erf).
        Func::Erf => {
            let v = values[0];
            let a = v.abs();
            let s = if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            };
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let poly = t
                * (0.254829592
                    + t * (-0.284496736
                        + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            s * (1.0 - poly * (-(a * a)).exp())
        }
        // NaN compares false on both branches, so sign(NaN) = 0.0.
        Func::Sign => {
            if values[0] > 0.0 {
                1.0
            } else if values[0] < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Func::Floor => values[0].floor(),
        // Half-away-from-zero via the explicit branch formula shared
        // by all three implementations (Python's builtin is banker's).
        Func::Round => {
            if values[0] >= 0.0 {
                (values[0] + 0.5).floor()
            } else {
                (values[0] - 0.5).ceil()
            }
        }
    })
}

fn eval_var(var: Var, ctx: &EvalContext<'_>) -> Result<f64> {
    eval_var_with(var, ctx, &mut NoSink)
}

fn eval_var_with<S: FaultSink>(var: Var, ctx: &EvalContext<'_>, sink: &mut S) -> Result<f64> {
    match var {
        Var::X => raise_read(
            ctx.x.context("variable x is only valid in map expressions"),
            FaultSet::ILLEGAL_SCOPE,
            sink,
        ),
        Var::Col(idx) => raise_read(
            ctx.row
                .get(idx)
                .copied()
                .with_context(|| format!("column index {idx} out of bounds")),
            FaultSet::COLUMN_OOB,
            sink,
        ),
        Var::RowMean => {
            raise_reduction(ctx.row, true, sink);
            Ok(mean(ctx.row))
        }
        Var::RowSum => {
            raise_reduction(ctx.row, false, sink);
            Ok(ctx.row.iter().copied().sum())
        }
        Var::RowL1 => {
            raise_reduction(ctx.row, false, sink);
            Ok(ctx.row.iter().map(|v| v.abs()).sum())
        }
        Var::RowL2 => {
            raise_reduction(ctx.row, false, sink);
            Ok(ctx.row.iter().map(|v| v * v).sum::<f64>().sqrt())
        }
        Var::RowMin => {
            raise_reduction(ctx.row, true, sink);
            Ok(ctx.row.iter().copied().fold(f64::INFINITY, f64::min))
        }
        Var::RowMax => {
            raise_reduction(ctx.row, true, sink);
            Ok(ctx.row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        Var::RowStd => {
            raise_reduction(ctx.row, true, sink);
            let mu = mean(ctx.row);
            Ok((ctx
                .row
                .iter()
                .map(|v| {
                    let d = *v - mu;
                    d * d
                })
                .sum::<f64>()
                / ctx.row.len().max(1) as f64)
                .sqrt())
        }
    }
}

/// An input read (`x` or `col(i)`), returned unchanged: `missing` when it cannot be read (the
/// read's error is the legacy error), NONFINITE_INPUT when the value read is not finite.
fn raise_read<S: FaultSink>(value: Result<f64>, missing: FaultSet, sink: &mut S) -> Result<f64> {
    match &value {
        Ok(v) if !v.is_finite() => sink.raise(FaultSet::NONFINITE_INPUT),
        Ok(_) => {}
        Err(_) => sink.raise(missing),
    }
    value
}

/// The cells a row reduction reads: NONFINITE_INPUT when any is not finite, and
/// EMPTY_ROW_REDUCTION for an empty row when the reduction is undefined on it (mean, std, min and
/// max; the empty sum, l1 and l2 are true identities).
fn raise_reduction<S: FaultSink>(row: &[f64], undefined_when_empty: bool, sink: &mut S) {
    if S::ENABLED && row.iter().any(|v| !v.is_finite()) {
        sink.raise(FaultSet::NONFINITE_INPUT);
    }
    if undefined_when_empty && row.is_empty() {
        sink.raise(FaultSet::EMPTY_ROW_REDUCTION);
    }
}

fn mean(row: &[f64]) -> f64 {
    row.iter().copied().sum::<f64>() / row.len().max(1) as f64
}

pub(crate) fn safe(value: f64) -> f64 {
    safe_with(value, &mut NoSink)
}

fn safe_with<S: FaultSink>(value: f64, sink: &mut S) -> f64 {
    if value.is_finite() {
        if value.abs() > MAX_ABS {
            sink.raise(FaultSet::MAGNITUDE_CLAMPED);
        }
        value.clamp(-MAX_ABS, MAX_ABS)
    } else {
        sink.raise(FaultSet::NONFINITE_SANITIZED);
        0.0
    }
}

pub(crate) fn count_nodes(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Unary(_, child) => 1 + count_nodes(child),
        Expr::Binary(_, left, right) => 1 + count_nodes(left) + count_nodes(right),
        Expr::Call(_, args) => 1 + args.iter().map(count_nodes).sum::<usize>(),
    }
}

pub(crate) fn depth(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Unary(_, child) => 1 + depth(child),
        Expr::Binary(_, left, right) => 1 + depth(left).max(depth(right)),
        Expr::Call(_, args) => 1 + args.iter().map(depth).max().unwrap_or(0),
    }
}
