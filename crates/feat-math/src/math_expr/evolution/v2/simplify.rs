//! The discrete minimax polynomial of a one-variable subtree on its observed training range,
//! lowered to ordinary AST nodes, and the bounded minimax linearization built on it.
//!
//! The one caller outside the tests is [`MathExpr::linearize_v2`]: a bounded minimax
//! linearization, degree 1 only, by the owner's ruling. Under the kernel2 caps (15 nodes, depth
//! 4) a dense Horner tree of degree `n` (every coefficient nonzero) has depth at least `2n + 1`
//! (a negative leading literal adds a unary minus), so no dense higher degree can be lowered into
//! an admissible expression; [`lower`] skips zero coefficients, so a sparse one can be shallower
//! (`[0, 0, 2]` lowers to `c0*(c0*2)`, depth 3), but none is built. This is not a general
//! polynomial or rational compression. A line is only a
//! proposal: it is offered when its fit's bracket closed, it is smaller than the subtree, its own
//! error on the training samples meets the minimax bound ([`assess`]), and the profile admits the
//! whole expression with it in place. The native protected re-score of the whole program then
//! decides, never the approximation error. The bound is the fit's own largest training error,
//! with room for rounding: it checks that lowering the fit to ordinary nodes lost no precision,
//! not that the line approximates the subtree well, which only the native score judges. The fit
//! of higher degrees, and the pole-free certificate a rational denominator would need
//! (`certify`, test-only), have no caller outside the tests.
//!
//! Algorithm: the second Remez exchange on the sample abscissae (Pachón and Trefethen, BIT 49,
//! 2009), in the Chebyshev basis on `t = (2v - (a + b)) / (b - a)`.
//! - It starts from the Chebyshev extrema.
//! - Each step solves `Σ c_j T_j(t_r) + (-1)^r h = f_r` on the reference by Gaussian elimination
//!   with partial pivoting.
//! - The new reference is the alternating run maxima with `|e| ≥ |h|`: same-sign neighbours are
//!   merged, and the `n + 2` consecutive points containing the global maximum are kept.
//! - The bracket is `[levelled, max|e|]`, with `levelled` the smaller of `|h|` and the smallest
//!   `|e|` on the reference, so it never inverts. Both ends are residuals of this module's own
//!   floating-point evaluation, not certified bounds: de la Vallée Poussin bounds the discrete
//!   minimax error E* by the exact errors, so E* lies within that evaluation's rounding of the
//!   bracket, not necessarily inside it. Rounding can also put `|h|` above `max|e|`; both stop
//!   tests bound that excess too.
//! - It stops when the bracket is within `1e-12 max|e|` (`Converged`). It stops `NoiseLimited`
//!   only when the bracket is within the rounding level `64 ε max|f|`, `|h|` is at least half of
//!   `max|e|`, the fit is the best seen, and the exchange is spent: it finds no new reference, or
//!   returns this one or one run before. The next reference depends on this one alone, so every
//!   later fit would repeat one already seen. A step whose `|h|` did not rise is not enough.
//!   Otherwise a run ends `Degenerate`, `Stalled` or at the iteration cap, with the best fit seen.
//! - Every tie goes to the lowest index and every sum runs in index order, so a fit is
//!   bit-reproducible and independent of sample order and duplicates.

// B19 core; the linearization and its kernel2 caller are laneD:B19-minimax-linearize.

use super::*;
use crate::kernel::expression_search::{
    LinearFit, LinearizationEvidence, MinimaxStop, SubtreeLinearization, LINEARIZATION_DEGREE,
    LINEARIZE_FIT_REFUSED, LINEARIZE_INPUT_FAULTS, LINEARIZE_MISSES_BOUND, LINEARIZE_NON_FINITE,
    LINEARIZE_NOT_SMALLER, LINEARIZE_OPEN_BRACKET, LINEARIZE_OUT_OF_RANGE, LINEARIZE_OVER_BUDGET,
    LINEARIZE_OVER_LIMITS, LINEARIZE_PROFILE_REFUSED, LINEARIZE_UNSAMPLED,
};

#[cfg(test)]
#[path = "simplify/certificate.rs"]
mod certificate;
#[path = "simplify/expression.rs"]
mod expression;
#[path = "simplify/fit.rs"]
mod fit;
#[path = "simplify/linearize.rs"]
mod linearize;
#[path = "simplify/lowering.rs"]
mod lowering;

#[cfg(test)]
pub(super) use certificate::*;
pub(super) use expression::*;
pub(super) use fit::*;
pub(super) use lowering::*;

#[cfg(test)]
mod linearization_tests;

#[cfg(test)]
#[path = "simplify/tests.rs"]
mod tests;
