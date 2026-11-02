//! Forward-mode dual numbers over the protected scalar evaluator.
//!
//! [`Program::lower`] lowers an expression to postorder, as `CompiledMathExpr` lowers it, with
//! the literal sites (the refinement's `literal_paths`) replaced by parameters. [`Program::row`]
//! evaluates one row. Every value comes from the evaluator's own scalar functions, in the
//! evaluator's order, so the value and the faults are `eval_row_traced`'s bit for bit. Each
//! parameter lane carries the derivative of the value by that parameter. No AST node is added:
//! the dual numbers live only here.
//!
//! Tangent rules (`tie-mean-v1`):
//! - **Branches.** Away from ties, the branch the evaluator's own predicate selects is
//!   differentiated. A protected or saturated constant branch contributes an exact zero.
//! - **Ties.** At an exact tie of selectable operands (`min`, `max`, `clip`, and the clamps inside
//!   `exp`, `sigmoid`, the `^` exponent and the output clamp), the tangent is the mean of the tied
//!   operands' tangents.
//! - **Closed forms.** `asinh`, `erf`, `tanh` and `sigmoid` take their slopes from the argument,
//!   not from the rounded value. `asinh` and `erf` use closed forms, which hold at 0 too, where
//!   differentiating their sign-times-magnitude code would give 0. `tanh` uses `4 e / (1 + e)²`
//!   with `e = e^(-2|v|)` and `sigmoid` uses `e / (1 + e)²` with `e = e^(-|v|)`; unlike
//!   `1 - tanh²` and `σ (1 - σ)`, they do not cancel once the value rounds to ±1.
//! - **Range.** No rule loses a slope that is in range at its node to an intermediate that is not.
//!   - `/` takes `a'/b - a b'/b²`. Its second term is formed on the operands' mantissas and
//!     exponents separately and joined once, so neither `q = a / b`, `q b'` nor `b²` can under- or
//!     overflow on the way.
//!   - Where a `^` coefficient (`e a^(e-1)` or `v ln a`) or the `erf` or `tanh` coefficient leaves
//!     the normal range, each lane is formed in the log domain, accurate to about 1e-12 relative.
//!     The `^` base term first tries the ratio form `(e v) (a' / a)`, while `v` is normal; where
//!     `a' / a` or the ratio leaves the range, that product is formed on mantissas and exponents.
//!   - `exp` applies its clamp factor to `v` before the lane, so the tie's half slope is not lost
//!     to `v a'` overflowing first.
//!   - Where `v²` overflows (`|v|` above about 1.34e154), `atan` divides by `v` twice.
//!   - Every other rule's lane is one product or quotient of the incoming lane with one factor in
//!     range, or a sum in which a site's lane has one nonzero term (a site lies in one operand).
//!
//!   A slope that is itself outside the normal range at its node is lost: below it, the lane keeps
//!   only a subnormal's bits, or is 0; above it, see the last paragraph.
//! - **Infinite one-sided slopes.** `sqrt` at 0 and `0 ^ e` with `0 < e < 1` have tangent 0.
//! - **Steps.** `step`, `sign`, `floor` and `round` have tangent 0.
//! - **Non-finite operands.** Any node other than a selector with a non-finite operand has tangent
//!   0.
//!
//! A row reports `nonsmooth` when a tie, an infinite one-sided slope, a non-finite operand or an
//! overflowing `^` slope was differentiated. A term whose coefficient is an exact zero is skipped,
//! so a zero coefficient never meets an infinite lane. In `^`, a saturated exponent's term is an
//! exact zero even where `v ln a` overflows. Every lane of a `^` tangent that is still not finite
//! (a base or exponent term, including one whose incoming lane was already not finite) is 0 and
//! marks the row nonsmooth. Other rules pass an overflowing lane on, and the normal equations then
//! refuse the Jacobian.

// B17 core.

use super::*;
use crate::math_expr::evaluator::{
    compute_binary_with, compute_func_with, eval_var_with, safe_with,
};
use crate::math_expr::fault::Collect;

mod program;
mod range;
mod tangent;

#[cfg(test)]
mod tests;

pub(super) use program::{Program, RowDual};
pub(super) use range::{Lanes, MAX_PARAMS};

use program::{Dual, Op, DUAL_ZERO};
use range::{join, log_lane, product, split, SCALE_DOWN, SCALE_UP, ZERO};
use tangent::*;
