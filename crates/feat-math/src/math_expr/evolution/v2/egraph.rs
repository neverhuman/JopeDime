//! Protected-semantics e-graph simplification.
//!
//! Opt-in: [`MathExpr::simplify_v2`] is the only entry. The kernel2 route calls it only from its
//! exact rewrite pass (`--rewrite-simplify`) on the ranked parents' expressions after each search
//! round; with the pass off nothing calls it.
//!
//! A small in-crate e-graph: union-find, a hashcons keyed by literal bits (so `0.0` and `-0.0`
//! are different nodes), and a rebuild back to congruence after every write phase (Willsey et
//! al., POPL 2021). It saturates under a fixed rewrite set in which every rule is exact under
//! the engine's protected semantics: on every input row whose cells, and every literal, are not
//! signaling NaNs, the rewritten tree gives the same error status, the same root value bits
//! (signed zero included) and the same fault set.
//! - Signaling NaNs are outside that domain. R1, R2, R3 and R10 delete the IEEE arithmetic step
//!   that quiets one, while `-` and `abs` pass it through. The evaluator's `powf` gives 1.0 for
//!   `pow(qNaN, ±0)` and `pow(1, qNaN)` but NaN for a signaling operand, which `safe` maps to 0.0
//!   with NONFINITE_SANITIZED. Text parsing never produces a signaling NaN; raw-bit rows and the
//!   binary codec can, and the search re-evaluates every proposed child natively.
//! - Inside the domain, NaN bits cannot reach an output, because every operation's non-NaN
//!   result ignores them and `safe` maps a non-finite root to 0.0.
//!
//! Extraction gives every class a tree with the fewest nodes the class represents and, among
//! those, the least depth. It picks in increasing order of that node count, so every child class
//! is settled before its parent, and a remaining tie goes to the smallest `Debug` form (then
//! literal bits) among the class's own nodes, each built over its children's picks. That last
//! tie-break is local: it need not find the smallest `Debug` form of all equally small trees.
//! Nothing in extraction depends on e-class numbering. Limits are counts only (e-nodes and
//! iterations), checked before every application, so the bound is hard. When saturation
//! completes the result does not depend on rule order either; when a limit stops it, the partial
//! write phase can. A tree is replaced only when the result is strictly smaller, or as small and
//! shallower. Every input first passes the structure check `from_ast` runs (the node and depth
//! limits, and every call's arity, none of which serde decoding checks) on the borrowed tree,
//! before anything counts, clones or recurses into it. That check is iterative and stops at the
//! first violation, so its cost is bounded by the limits, not by the input. An input that fails
//! it is returned unchanged: the result borrows the caller's tree and neither copies nor walks it.
//!
//! The rules (proof notes per rule in the tests):
//! - R1 `x*1`, `1*x` → `x`; R2 `x/1` → `x`;
//! - R3 `x + (-0)`, `(-0) + x`, `x - 0` → `x`;
//! - R4 `-(-x)` → `x`; R5 `|-x|` → `|x|`, `||x||` → `|x|`;
//! - R7 `x - y` ↔ `x + (-y)`, `x - (-y)` → `x + y`;
//! - R8 `(-a)*(-b)` → `a*b`, `(-a)*b` → `-(a*b)`, `a*(-b)` → `-(a*b)`;
//! - R9 `x*2` → `x + x`; R10 `x*(-1)` → `-x`, `x/(-1)` → `-x`;
//! - R11 `x/c` → `x*(1/c)` for `c = ±2^k` with `-39 ≤ k ≤ 1023` (the protected division's
//!   threshold is 1e-12, and 2^-40 is below it);
//! - constant folding of `+ - *` and of `/` with a divisor of magnitude at least 1e-12, to finite
//!   results only, with the evaluator's own arithmetic; and of `abs` of any literal, which is
//!   exact because it only clears the sign bit.
//!
//! Pinned as outside the exact domain: R1, R2, R3 and R10 on signaling-NaN rows under `pow`.
//! Excluded, each pinned by a counterexample test: `x + 0.0`, `x - (-0.0)`, `x*0`, `x - x`, `x/x`,
//! reassociation, distribution, `a - b` → `-(b - a)`, operand commutation (on an error row it
//! changes which operand's faults are raised before the Hard one), `(-a)/b` ↔ `-(a/b)` (the
//! protected branch returns +0.0 for both), `x^2` ↔ `x*x` (at overflow `pow` raises
//! POW_NONFINITE and `*` does not; `powf` precision is unspecified too), `exp(log(x))` → `x` (the
//! protected `log` of a negative raises LOG_DOMAIN), and `x/c` for `|c| = 2^-40`. Folding a
//! libm-backed function is excluded too, because libm results are not portable; that one is
//! pinned only as never applied.

// B20 core; the kernel2 caller is laneD:B20-rewrite-simplify.

use super::*;
use crate::kernel::expression_search::{ExpressionSimplification, SaturationStop};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

include!("egraph/graph.rs");
include!("egraph/rewrite.rs");
include!("egraph/simplify.rs");
include!("egraph/extract.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pcg32;

    include!("egraph/tests/support.rs");
    include!("egraph/tests/rules.rs");
    include!("egraph/tests/saturation.rs");
    include!("egraph/tests/exactness.rs");
    include!("egraph/tests/structure.rs");
}
