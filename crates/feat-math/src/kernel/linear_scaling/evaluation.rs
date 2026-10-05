//! Stable zero ties for this new scalar profile; legacy evaluation is untouched.
use super::{Error, MathExpr, Result};
use crate::math_expr::{CompiledMathExpr, Expr, Var};
use std::borrow::Cow;

pub(super) struct Evaluator<'a> {
    expression: &'a MathExpr,
    compiled: CompiledMathExpr,
    row_extrema: bool,
}

impl<'a> Evaluator<'a> {
    pub(super) fn new(expression: &'a MathExpr) -> Result<Self> {
        let mut pending = vec![expression.ast()];
        let mut row_extrema = false;
        while let Some(node) = pending.pop() {
            match node {
                Expr::Var(Var::RowMin | Var::RowMax) => row_extrema = true,
                Expr::Unary(_, child) => pending.push(child),
                Expr::Binary(_, left, right) => pending.extend([left.as_ref(), right.as_ref()]),
                Expr::Call(_, args) => pending.extend(args),
                _ => {}
            }
        }
        Ok(Self {
            expression,
            compiled: expression
                .compile()
                .map_err(|e| Error::Domain(e.to_string()))?,
            row_extrema,
        })
    }

    // Call only after the public entry point's finite/nonempty/width validation.
    // The original tree and serialized winner stay unchanged. Only tied signed
    // zeros need specialization: the frozen fold has optimizer-dependent ties.
    pub(super) fn for_row(&self, row: &[f64]) -> Result<Cow<'_, CompiledMathExpr>> {
        if self.row_extrema {
            let (mut lower, mut upper) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut negative_zero, mut positive_zero) = (false, false);
            for &value in row {
                lower = if lower < value { lower } else { value };
                upper = if upper > value { upper } else { value };
                if value == 0.0 {
                    negative_zero |= value.is_sign_negative();
                    positive_zero |= !value.is_sign_negative();
                }
            }
            if negative_zero && positive_zero && (lower == 0.0 || upper == 0.0) {
                let expression =
                    MathExpr::from_ast(bind_extrema(self.expression.ast(), lower, upper))
                        .map_err(|e| Error::Domain(e.to_string()))?;
                return expression
                    .compile()
                    .map(Cow::Owned)
                    .map_err(|e| Error::Domain(e.to_string()));
            }
        }
        Ok(Cow::Borrowed(&self.compiled))
    }
}

fn bind_extrema(node: &Expr, lower: f64, upper: f64) -> Expr {
    match node {
        Expr::Var(Var::RowMin) => Expr::Const(lower),
        Expr::Var(Var::RowMax) => Expr::Const(upper),
        Expr::Unary(op, child) => Expr::Unary(*op, Box::new(bind_extrema(child, lower, upper))),
        Expr::Binary(op, left, right) => Expr::Binary(
            *op,
            Box::new(bind_extrema(left, lower, upper)),
            Box::new(bind_extrema(right, lower, upper)),
        ),
        Expr::Call(func, args) => Expr::Call(
            *func,
            args.iter()
                .map(|child| bind_extrema(child, lower, upper))
                .collect(),
        ),
        _ => node.clone(),
    }
}
