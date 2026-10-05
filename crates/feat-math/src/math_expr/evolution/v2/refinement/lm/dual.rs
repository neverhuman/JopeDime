//! Forward-mode dual numbers over the protected scalar evaluator (B17).
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
//!     The `^` base term first tries the ratio form `(e v) (a' / a)`, while `v` is normal.
//!   - Where `v²` overflows (`|v|` above about 1.34e154), `atan` divides by `v` twice.
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

use super::*;
use crate::math_expr::fault::Collect;

/// The most parameters one pass differentiates.
pub(super) const MAX_PARAMS: usize = 16;

/// One tangent lane per parameter. Lanes at and above the parameter count stay zero.
pub(super) type Lanes = [f64; MAX_PARAMS];

const ZERO: Lanes = [0.0; MAX_PARAMS];

/// 2^600 and 2^-600. Multiplying by either is exact while the product stays a normal number.
const SCALE_UP: f64 = f64::from_bits((1023 + 600) << 52);
const SCALE_DOWN: f64 = f64::from_bits((1023 - 600) << 52);

/// `x = m · 2^e` with `1 ≤ |m| < 2` for a finite nonzero `x`, subnormals included. Zero and
/// non-finite `x` come back as `(x, 0)`.
fn split(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    // A subnormal is first scaled into the normal range, exactly.
    let (x, shift) = if x.abs() < f64::MIN_POSITIVE {
        (x * SCALE_UP, -600)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & !(0x7ff_u64 << 52)) | (1023_u64 << 52));
    (mantissa, exponent + shift)
}

/// `m · 2^e`, rounded once: the steps before the last are exact power-of-two scalings that keep
/// the value normal, so only the last can round (to a subnormal) or overflow.
fn join(m: f64, e: i32) -> f64 {
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let (mut x, mut e) = (m, e);
    while e > 600 {
        x *= SCALE_UP;
        e -= 600;
    }
    while e < -600 {
        x *= SCALE_DOWN;
        e += 600;
    }
    x * f64::from_bits(((e + 1023) as u64) << 52)
}

/// `±exp(log_c) · d` for a slope coefficient `±exp(log_c)` outside the normal range, formed in the
/// log domain so a large or small lane still gets its product. An underflow is an exact +0.
fn log_lane(negative: bool, log_c: f64, d: f64) -> f64 {
    if d == 0.0 || log_c == f64::NEG_INFINITY {
        return 0.0;
    }
    let r = (log_c + d.abs().ln()).exp();
    if r == 0.0 {
        0.0
    } else if negative != (d < 0.0) {
        -r
    } else {
        r
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Constant(f64),
    Param(u8),
    Variable(Var),
    Neg(u8),
    Binary(BinOp, u8, u8),
    Call(Func, [u8; 3], u8),
}

#[derive(Clone, Copy, Debug)]
struct Dual {
    v: f64,
    d: Lanes,
}

const DUAL_ZERO: Dual = Dual { v: 0.0, d: ZERO };

/// One row's evaluation: `eval_row_traced`'s value and faults, and the tangent of the value.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowDual {
    pub value: f64,
    pub faults: FaultSet,
    pub tangent: Lanes,
    pub nonsmooth: bool,
}

/// A row expression in postorder with parameter sites.
#[derive(Clone, Debug)]
pub(super) struct Program {
    code: Vec<Op>,
    params: usize,
}

impl Program {
    /// `root` with the literal node at `sites[k]` as parameter `k`. Every site must be a literal
    /// (`Const` or `Neg(Const)`), and none may lie inside another.
    pub(super) fn lower(root: &Expr, sites: &[Vec<usize>]) -> Result<Self> {
        if sites.len() > MAX_PARAMS {
            bail!("{} parameter sites, at most {MAX_PARAMS}", sites.len());
        }
        let mut program = Self {
            code: Vec::new(),
            params: sites.len(),
        };
        let mut path = Vec::new();
        program.lower_node(root, sites, &mut path)?;
        let used = program
            .code
            .iter()
            .filter(|op| matches!(op, Op::Param(_)))
            .count();
        if used != sites.len() {
            bail!("a parameter site is not a node of the expression");
        }
        Ok(program)
    }

    pub(super) fn params(&self) -> usize {
        self.params
    }

    fn lower_node(
        &mut self,
        expr: &Expr,
        sites: &[Vec<usize>],
        path: &mut Vec<usize>,
    ) -> Result<u8> {
        let op = if let Some(k) = sites.iter().position(|site| site == path) {
            if literal_value(expr).is_none() {
                bail!("parameter site is not a literal");
            }
            Op::Param(k as u8)
        } else {
            match expr {
                Expr::Const(value) => Op::Constant(*value),
                Expr::Var(var) => Op::Variable(*var),
                Expr::Unary(UnOp::Neg, child) => Op::Neg(self.lower_child(child, 0, sites, path)?),
                Expr::Binary(op, a, b) => {
                    let a = self.lower_child(a, 0, sites, path)?;
                    let b = self.lower_child(b, 1, sites, path)?;
                    Op::Binary(*op, a, b)
                }
                Expr::Call(func, args) => {
                    validate_arity(*func, args.len())?;
                    let mut registers = [0; 3];
                    for (i, (slot, arg)) in registers.iter_mut().zip(args).enumerate() {
                        *slot = self.lower_child(arg, i, sites, path)?;
                    }
                    Op::Call(*func, registers, args.len() as u8)
                }
            }
        };
        if self.code.len() >= MAX_NODES {
            bail!("dual expression node limit");
        }
        self.code.push(op);
        Ok((self.code.len() - 1) as u8)
    }

    fn lower_child(
        &mut self,
        child: &Expr,
        index: usize,
        sites: &[Vec<usize>],
        path: &mut Vec<usize>,
    ) -> Result<u8> {
        path.push(index);
        let register = self.lower_node(child, sites, path);
        path.pop();
        register
    }

    /// The row's value, faults and tangent at parameters `theta` (one value per site). As in
    /// `eval_row_traced`, an evaluation error gives the value 0.0 with the faults raised so far,
    /// and then the tangent is zero.
    pub(super) fn row(&self, theta: &[f64], row: &[f64]) -> RowDual {
        debug_assert_eq!(theta.len(), self.params);
        let mut registers = [DUAL_ZERO; MAX_NODES];
        let mut sink = Collect::default();
        let mut nonsmooth = false;
        for (i, op) in self.code.iter().enumerate() {
            match self.step(*op, &registers, theta, row, &mut sink, &mut nonsmooth) {
                Ok(dual) => registers[i] = dual,
                Err(_) => {
                    return RowDual {
                        value: 0.0,
                        faults: sink.0,
                        tangent: ZERO,
                        nonsmooth,
                    }
                }
            }
        }
        let top = registers[self.code.len() - 1];
        let value = safe_with(top.v, &mut sink);
        let tangent = if top.v.is_finite() {
            clamped(
                self.params,
                top.v,
                -MAX_ABS,
                MAX_ABS,
                &top.d,
                &mut nonsmooth,
            )
        } else {
            nonsmooth = true;
            ZERO
        };
        RowDual {
            value,
            faults: sink.0,
            tangent,
            nonsmooth,
        }
    }

    fn step(
        &self,
        op: Op,
        registers: &[Dual; MAX_NODES],
        theta: &[f64],
        row: &[f64],
        sink: &mut Collect,
        nonsmooth: &mut bool,
    ) -> Result<Dual> {
        let p = self.params;
        Ok(match op {
            Op::Constant(v) => Dual { v, d: ZERO },
            Op::Param(k) => {
                let mut d = ZERO;
                d[k as usize] = 1.0;
                Dual {
                    v: theta[k as usize],
                    d,
                }
            }
            Op::Variable(var) => Dual {
                v: eval_var_with(var, &EvalContext { x: None, row }, sink)?,
                d: ZERO,
            },
            Op::Neg(a) => {
                let a = &registers[a as usize];
                let d = if a.v.is_finite() {
                    scale(p, -1.0, &a.d)
                } else {
                    *nonsmooth = true;
                    ZERO
                };
                Dual { v: -a.v, d }
            }
            Op::Binary(op, a, b) => {
                let (a, b) = (&registers[a as usize], &registers[b as usize]);
                let v = compute_binary_with(op, a.v, b.v, sink);
                Dual {
                    v,
                    d: binary_tangent(p, op, a, b, v, nonsmooth),
                }
            }
            Op::Call(func, args, n) => {
                let n = n as usize;
                let mut values = [0.0; 3];
                let mut operands = [&DUAL_ZERO; 3];
                for i in 0..n {
                    operands[i] = &registers[args[i] as usize];
                    values[i] = operands[i].v;
                }
                let v = compute_func_with(func, values, sink)?;
                Dual {
                    v,
                    d: call_tangent(p, func, &operands[..n], v, nonsmooth),
                }
            }
        })
    }
}

/// `ka * a + kb * b` lane by lane, skipping a term whose coefficient is an exact zero.
fn comb(p: usize, ka: f64, a: &Lanes, kb: f64, b: &Lanes) -> Lanes {
    let mut out = ZERO;
    if ka != 0.0 {
        for k in 0..p {
            out[k] = ka * a[k];
        }
    }
    if kb != 0.0 {
        for k in 0..p {
            out[k] += kb * b[k];
        }
    }
    out
}

fn scale(p: usize, k: f64, a: &Lanes) -> Lanes {
    comb(p, k, a, 0.0, &ZERO)
}

/// `a / denominator` lane by lane.
fn divide(p: usize, a: &Lanes, denominator: f64) -> Lanes {
    let mut out = ZERO;
    for k in 0..p {
        out[k] = a[k] / denominator;
    }
    out
}

/// The slope factor of `value.clamp(lo, hi)`: 1 inside, 0 outside or NaN, and the tie mean 1/2
/// on a bound.
fn clamp_factor(value: f64, lo: f64, hi: f64) -> f64 {
    if value > lo && value < hi {
        1.0
    } else if value == lo || value == hi {
        0.5
    } else {
        0.0
    }
}

/// The tangent of `value.clamp(lo, hi)` with constant bounds.
fn clamped(p: usize, value: f64, lo: f64, hi: f64, d: &Lanes, nonsmooth: &mut bool) -> Lanes {
    let factor = clamp_factor(value, lo, hi);
    if factor == 0.5 {
        *nonsmooth = true;
    }
    scale(p, factor, d)
}

/// The mean tangent of the operands whose value equals the selected `result` (Rust `==`, so -0.0
/// ties 0.0 and NaN never matches). Zero when nothing matches.
fn tie_mean(p: usize, result: f64, operands: &[&Dual], nonsmooth: &mut bool) -> Lanes {
    let mut out = ZERO;
    let mut tied = 0usize;
    for operand in operands {
        if operand.v == result {
            for (lane, d) in out.iter_mut().zip(&operand.d).take(p) {
                *lane += d;
            }
            tied += 1;
        }
    }
    if tied > 1 {
        *nonsmooth = true;
        let share = 1.0 / tied as f64;
        for lane in out.iter_mut().take(p) {
            *lane *= share;
        }
    }
    out
}

fn binary_tangent(p: usize, op: BinOp, a: &Dual, b: &Dual, v: f64, nonsmooth: &mut bool) -> Lanes {
    if !a.v.is_finite() || !b.v.is_finite() {
        *nonsmooth = true;
        return ZERO;
    }
    match op {
        BinOp::Add => comb(p, 1.0, &a.d, 1.0, &b.d),
        BinOp::Sub => comb(p, 1.0, &a.d, -1.0, &b.d),
        BinOp::Mul => comb(p, b.v, &a.d, a.v, &b.d),
        // The protected branch is a constant 0. Otherwise the slope is a'/b - a b'/b². Its second
        // term is formed on the mantissas and exponents of a, b' and b separately and joined
        // once, so no intermediate (q = a/b, q b', b²) under- or overflows: a lane is lost only
        // where it is itself out of range. A site lies in a or in b, so one term is zero; a term
        // with an exact zero factor is skipped, and an incoming lane that is not finite is passed
        // on.
        BinOp::Div if b.v.abs() < 1e-12 => ZERO,
        BinOp::Div => {
            let mut out = ZERO;
            let ((ma, ea), (mb, eb)) = (split(a.v), split(b.v));
            for (k, lane) in out.iter_mut().enumerate().take(p) {
                let direct = if a.d[k] == 0.0 { 0.0 } else { a.d[k] / b.v };
                let through_b = if b.d[k] == 0.0 || a.v == 0.0 {
                    0.0
                } else {
                    let (md, ed) = split(b.d[k]);
                    join(ma * md / (mb * mb), ea + ed - 2 * eb)
                };
                *lane = direct - through_b;
            }
            out
        }
        BinOp::Pow => pow_tangent(p, a, b, v, nonsmooth),
    }
}

/// `v = a.powf(e)` with `e = b.clamp(-8, 8)`, both operands finite.
fn pow_tangent(p: usize, a: &Dual, b: &Dual, v: f64, nonsmooth: &mut bool) -> Lanes {
    let e = b.v.clamp(-8.0, 8.0);
    let exponent_factor = clamp_factor(b.v, -8.0, 8.0);
    if exponent_factor == 0.5 {
        *nonsmooth = true;
    }
    // e * a^(e - 1) keeps a tiny base's slope where v / a would lose it to underflow (e > 1). Where
    // it leaves the normal range anyway, `base_lanes` takes another form.
    let base_slope = |a: f64| if e == 0.0 { 0.0 } else { e * a.powf(e - 1.0) };
    let (by_base, by_exponent) = if a.v > 0.0 {
        (base_slope(a.v), v * a.v.ln())
    } else if a.v == 0.0 {
        if e == 1.0 {
            (1.0, 0.0)
        } else if e > 1.0 {
            (0.0, 0.0)
        } else if e > 0.0 {
            // The right slope is +inf.
            *nonsmooth = true;
            (0.0, 0.0)
        } else if e == 0.0 {
            // 0^e jumps among 0, 1 and inf at e = 0.
            *nonsmooth = true;
            (0.0, 0.0)
        } else {
            // inf, with POW_NONFINITE.
            (0.0, 0.0)
        }
    } else if e == e.trunc() {
        // A negative base has a value only at integral exponents.
        *nonsmooth = true;
        (base_slope(a.v), 0.0)
    } else {
        // NaN, with POW_NONFINITE.
        (0.0, 0.0)
    };
    // The base slope was computed, not set: a nonzero base (a negative one only at an integral
    // exponent) and e != 0. Only a computed slope can leave the normal range by rounding.
    let computed = e != 0.0 && (a.v > 0.0 || (a.v < 0.0 && e == e.trunc()));
    // Each term is checked: a finite slope times a large lane can still overflow. A site lies in
    // the base or the exponent, never both, so a lane's sum has one nonzero term and is checked
    // only defensively.
    let mut out = base_lanes(p, by_base, e, v, a, computed, nonsmooth);
    // A saturated exponent is a constant, so its term is an exact zero even where v ln a
    // overflows; so is a base of exactly 1, where ln a = 0.
    if exponent_factor != 0.0 && a.v > 0.0 && a.v != 1.0 {
        let by_exponent = exponent_factor * by_exponent;
        // v ln a is used directly while it and v are normal numbers. Otherwise (v under- or
        // overflowed, or is subnormal) each lane is formed in the log domain.
        let direct = by_exponent.is_finite()
            && by_exponent.abs() >= f64::MIN_POSITIVE
            && v >= f64::MIN_POSITIVE;
        let ln_a = a.v.ln();
        let log_c = exponent_factor.ln() + e * ln_a + ln_a.abs().ln();
        for (lane, d) in out.iter_mut().zip(&b.d).take(p) {
            if *d == 0.0 {
                continue;
            }
            let term = if direct {
                by_exponent * d
            } else {
                log_lane(ln_a < 0.0, log_c, *d)
            };
            let exponent_term = finite_or_zero(term, nonsmooth);
            *lane = finite_or_zero(*lane + exponent_term, nonsmooth);
        }
    }
    out
}

/// `value`, or 0 with the row marked nonsmooth when it is not finite.
fn finite_or_zero(value: f64, nonsmooth: &mut bool) -> f64 {
    if value.is_finite() {
        value
    } else {
        *nonsmooth = true;
        0.0
    }
}

/// `e a^(e-1) a'` lane by lane. An explicit slope (`computed` false: a zero base, or e = 0), or a
/// computed `by_base` that is a finite normal number, is used directly. A computed one outside
/// that range is taken in the ratio form `(e v) (a'_k / a)` while `v` is normal and `e v` finite
/// (a large base with e < 0 underflows `e a^(e-1)` while `v` is normal), and in the log domain
/// otherwise (both underflow for a tiny base and e > 1). A lane whose `a'_k` is 0 stays 0. A lane
/// that is not finite is 0 and marks the row nonsmooth.
fn base_lanes(
    p: usize,
    by_base: f64,
    e: f64,
    v: f64,
    a: &Dual,
    computed: bool,
    nonsmooth: &mut bool,
) -> Lanes {
    let mut out = ZERO;
    if !computed || (by_base.is_finite() && by_base.abs() >= f64::MIN_POSITIVE) {
        if by_base != 0.0 {
            for (lane, d) in out.iter_mut().zip(&a.d).take(p) {
                *lane = finite_or_zero(by_base * d, nonsmooth);
            }
        }
        return out;
    }
    let ev = e * v;
    let ratio = v.abs() >= f64::MIN_POSITIVE && ev.is_finite();
    // e a^(e-1) is negative when e is, flipped for a negative base at an odd integral e - 1.
    let negative = (e < 0.0) != (a.v < 0.0 && (e - 1.0) % 2.0 != 0.0);
    let log_c = e.abs().ln() + (e - 1.0) * a.v.abs().ln();
    for (lane, d) in out.iter_mut().zip(&a.d).take(p) {
        if *d != 0.0 {
            let term = if ratio {
                ev * (d / a.v)
            } else {
                log_lane(negative, log_c, *d)
            };
            *lane = finite_or_zero(term, nonsmooth);
        }
    }
    out
}

fn call_tangent(p: usize, func: Func, operands: &[&Dual], v: f64, nonsmooth: &mut bool) -> Lanes {
    if matches!(func, Func::Min | Func::Max | Func::Clip) {
        return tie_mean(p, v, operands, nonsmooth);
    }
    if operands.iter().any(|operand| !operand.v.is_finite()) {
        *nonsmooth = true;
        return ZERO;
    }
    let a = operands[0];
    match func {
        Func::Abs if a.v > 0.0 => a.d,
        Func::Abs if a.v < 0.0 => scale(p, -1.0, &a.d),
        Func::Abs => {
            // The mean of the one-sided tangents a' and -a'.
            *nonsmooth = true;
            ZERO
        }
        Func::Sqrt if a.v > 0.0 => scale(p, 0.5 / v, &a.d),
        Func::Sqrt => {
            // At 0 the right slope is +inf; below 0 the value is projected (SQRT_PROJECTED).
            if a.v == 0.0 {
                *nonsmooth = true;
            }
            ZERO
        }
        Func::Log1p if a.v > -1.0 => divide(p, &a.d, 1.0 + a.v),
        Func::Log1p => ZERO,
        Func::Sin => scale(p, a.v.cos(), &a.d),
        Func::Cos => scale(p, -a.v.sin(), &a.d),
        Func::Tanh => sech_squared_lanes(p, a.v, &a.d),
        Func::Sigmoid => {
            let factor = clamp_factor(a.v, -30.0, 30.0);
            if factor == 0.5 {
                *nonsmooth = true;
            }
            // σ (1 - σ) from the argument: within the clamp it is at least e^-30 / 4.
            let e = (-a.v.abs()).exp();
            scale(p, factor * (e / ((1.0 + e) * (1.0 + e))), &a.d)
        }
        Func::Exp => clamped(p, a.v, -30.0, 30.0, &scale(p, v, &a.d), nonsmooth),
        Func::Log if a.v > 1e-300 => divide(p, &a.d, a.v),
        Func::Log => ZERO,
        // The evaluator's asinh squares its argument too, so where v² overflows its value is not
        // finite and this slope is never used.
        Func::Asinh => divide(p, &a.d, (a.v * a.v + 1.0).sqrt()),
        // Past |v| ≈ 1.34e154 v² overflows while atan stays finite: a' / (1 + v²) is then taken
        // as (a' / v) / v, so the slope does not silently become 0.
        Func::Atan if (a.v * a.v).is_finite() => divide(p, &a.d, 1.0 + a.v * a.v),
        Func::Atan => divide(p, &divide(p, &a.d, a.v), a.v),
        Func::Erf => erf_lanes(p, a.v, &a.d),
        Func::Step | Func::Sign | Func::Floor | Func::Round => ZERO,
        Func::Min | Func::Max | Func::Clip => unreachable!("selectors return above"),
    }
}

/// `c a'` lane by lane for a positive slope coefficient `c = exp(log_c)`: directly while `c` is a
/// normal number, and in the log domain otherwise.
fn tail_lanes(p: usize, c: f64, log_c: f64, d: &Lanes) -> Lanes {
    if c >= f64::MIN_POSITIVE {
        return scale(p, c, d);
    }
    let mut out = ZERO;
    for (lane, d) in out.iter_mut().zip(d).take(p) {
        *lane = log_lane(false, log_c, *d);
    }
    out
}

/// `sech²(x) a'` lane by lane, from the argument: `4 e / (1 + e)²` with `e = e^(-2|x|)`. Unlike
/// `1 - tanh²`, it does not cancel to 0 once `tanh` rounds to ±1 (`|x|` above about 19.1).
fn sech_squared_lanes(p: usize, x: f64, d: &Lanes) -> Lanes {
    let e = (-2.0 * x.abs()).exp();
    let c = 4.0 * e / ((1.0 + e) * (1.0 + e));
    tail_lanes(p, c, 4f64.ln() - 2.0 * x.abs() - 2.0 * e.ln_1p(), d)
}

/// The derivative of the evaluator's `erf` (Abramowitz and Stegun 7.1.26, the same constants),
/// valid at 0: with `u = |v|`, `t = 1 / (1 + p u)`, `P(t) = a1 t + … + a5 t⁵` and `E = e^(-u²)`,
/// the value is `sign(v) (1 - P(t) E)`, whose slope is `E (p t² P'(t) + 2 u P(t))` on both sides.
/// The coefficients sum to `1 - 1e-9`, so the value itself steps by about `2e-9` across 0; this
/// slope ignores that step and does not mark the row nonsmooth for it. Where `E` leaves the normal
/// range (`u` above about 26.5) the lanes are formed in the log domain; where `u²` overflows the
/// slope is an exact 0.
fn erf_lanes(p: usize, v: f64, d: &Lanes) -> Lanes {
    let u = v.abs();
    if !(u * u).is_finite() {
        return ZERO;
    }
    let bracket = erf_bracket(u);
    let tail = (-(u * u)).exp();
    tail_lanes(p, tail * bracket, bracket.ln() - u * u, d)
}

/// `p t² P'(t) + 2 u P(t)` of [`erf_lanes`], positive for `u ≥ 0`.
fn erf_bracket(u: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * u);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let slope = 0.254829592
        + t * (2.0 * -0.284496736
            + t * (3.0 * 1.421413741 + t * (4.0 * -1.453152027 + t * (5.0 * 1.061405429))));
    0.3275911 * t * t * slope + 2.0 * u * poly
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pcg32;

    fn sites_of(source: &str) -> (MathExpr, Vec<Vec<usize>>) {
        let expression = MathExpr::parse(source).unwrap();
        let sites = literal_paths(&expression.root);
        (expression, sites)
    }

    fn theta_of(expression: &MathExpr, sites: &[Vec<usize>]) -> Vec<f64> {
        sites
            .iter()
            .map(|site| literal_value(at(&expression.root, site)).unwrap())
            .collect()
    }

    fn native(expression: &MathExpr, row: &[f64]) -> (f64, FaultSet) {
        expression.compile().unwrap().eval_row_traced(row)
    }

    /// Seeded random row expressions over every function, operator and variable.
    fn random_expr(rng: &mut Pcg32, depth: usize, columns: usize) -> Expr {
        let leaf = |rng: &mut Pcg32| -> Expr {
            match rng.next_u32() % 4 {
                0 => Expr::Var(Var::Col(rng.next_u32() as usize % columns)),
                1 => Expr::Var(
                    [
                        Var::RowMean,
                        Var::RowSum,
                        Var::RowL1,
                        Var::RowL2,
                        Var::RowMin,
                        Var::RowMax,
                        Var::RowStd,
                    ][rng.next_u32() as usize % 7],
                ),
                2 => Expr::Const(
                    [0.0, -0.0, 0.5, 1.0, 2.0, 3.0, 8.0, 30.0, 1e-12, 1e12]
                        [rng.next_u32() as usize % 10],
                ),
                _ => Expr::Const(rng.range_f64(-3.0, 3.0)),
            }
        };
        if depth == 0 || rng.next_u32().is_multiple_of(4) {
            return leaf(rng);
        }
        let funcs = [
            Func::Abs,
            Func::Sqrt,
            Func::Log1p,
            Func::Sin,
            Func::Cos,
            Func::Tanh,
            Func::Sigmoid,
            Func::Min,
            Func::Max,
            Func::Clip,
            Func::Step,
            Func::Exp,
            Func::Log,
            Func::Asinh,
            Func::Atan,
            Func::Erf,
            Func::Sign,
            Func::Floor,
            Func::Round,
        ];
        match rng.next_u32() % 3 {
            0 => {
                let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow]
                    [rng.next_u32() as usize % 5];
                Expr::Binary(
                    op,
                    Box::new(random_expr(rng, depth - 1, columns)),
                    Box::new(random_expr(rng, depth - 1, columns)),
                )
            }
            1 => Expr::Unary(UnOp::Neg, Box::new(random_expr(rng, depth - 1, columns))),
            _ => {
                let func = funcs[rng.next_u32() as usize % funcs.len()];
                let arity = (1..=3).find(|n| validate_arity(func, *n).is_ok()).unwrap();
                Expr::Call(
                    func,
                    (0..arity)
                        .map(|_| random_expr(rng, depth - 1, columns))
                        .collect(),
                )
            }
        }
    }

    fn edge_rows() -> Vec<Vec<f64>> {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            -2.5,
            30.0,
            -30.0,
            8.0,
            1e-300,
            1e300,
            f64::MIN_POSITIVE,
            5e-324,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut rows = Vec::new();
        for (i, a) in specials.iter().enumerate() {
            rows.push(vec![*a, specials[(i + 3) % specials.len()], 1.5]);
        }
        rows.push(vec![0.25, -1.75, 3.0]);
        rows.push(vec![]);
        rows
    }

    #[test]
    fn values_and_faults_are_the_native_evaluators_bit_for_bit() {
        let rows = edge_rows();
        let mut rng = Pcg32::new(0xB17, 1);
        let mut checked = 0usize;
        for _ in 0..600 {
            let root = random_expr(&mut rng, 4, 3);
            let Ok(expression) = MathExpr::from_ast(root) else {
                continue;
            };
            let sites = literal_paths(&expression.root);
            if sites.len() > MAX_PARAMS {
                continue;
            }
            let program = Program::lower(&expression.root, &sites).unwrap();
            let theta = theta_of(&expression, &sites);
            for row in &rows {
                let (value, faults) = native(&expression, row);
                let dual = program.row(&theta, row);
                assert_eq!(
                    dual.value.to_bits(),
                    value.to_bits(),
                    "{:?} on {row:?}",
                    expression.root
                );
                assert_eq!(dual.faults, faults, "{:?} on {row:?}", expression.root);
                checked += 1;
            }
        }
        assert!(checked > 5_000, "{checked}");
    }

    #[test]
    fn tangents_match_central_differences_on_smooth_rows() {
        let cases = [
            "1.5*exp(0.4*c0)+0.25",
            "sin(1.3*c0+0.2)*(0.7-c1)",
            "tanh(0.9*c0)/(1.4+(c1*c1))",
            "sigmoid(2.0*c0-0.5)*log(2.5+c1)",
            "atan(0.6*c0)+asinh(1.7*c1)",
            "erf(0.8*c0+0.1)*sqrt(3.0+c1)",
            "(1.2*c0+3.0)^1.7",
            "2.0^(0.3*c0)",
            "log1p(0.5*abs(c0))*cos(0.9*c1)",
            "min(0.5*c0,1.5*c1)+max(2.0*c0,0.3)",
            "clip(1.1*c0,-2.0,2.5)",
        ];
        let rows: Vec<Vec<f64>> = (0..9)
            .map(|i| vec![-1.2 + 0.31 * f64::from(i), 0.4 + 0.17 * f64::from(i)])
            .collect();
        let mut compared = 0usize;
        for source in cases {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let theta = theta_of(&expression, &sites);
            for row in &rows {
                let dual = program.row(&theta, row);
                if dual.nonsmooth || !dual.faults.is_empty() {
                    continue;
                }
                for k in 0..sites.len() {
                    let h = 1e-6 * theta[k].abs().max(1.0);
                    let mut up = theta.clone();
                    up[k] += h;
                    let mut down = theta.clone();
                    down[k] -= h;
                    let plus = program.row(&up, row);
                    let minus = program.row(&down, row);
                    if plus.nonsmooth || minus.nonsmooth {
                        continue;
                    }
                    let central = (plus.value - minus.value) / (up[k] - down[k]);
                    let exact = dual.tangent[k];
                    assert!(
                        (central - exact).abs() <= 1e-6 * exact.abs().max(1.0),
                        "{source} lane {k} on {row:?}: dual {exact}, central {central}"
                    );
                    compared += 1;
                }
            }
        }
        assert!(compared > 150, "{compared}");
    }

    #[test]
    fn spot_values_follow_the_tie_mean_rules() {
        let at_zero = |source: &str| {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            program.row(&theta_of(&expression, &sites), &[0.0, 3.0])
        };
        // erf'(0) = p P'(1), 7.19e-6 above 2/sqrt(pi); asinh'(0) = 1.
        let erf = at_zero("erf(0.0*c1+0.0)");
        assert!(
            (erf.tangent[1] - 1.128_386_358).abs() < 1e-9,
            "{:?}",
            erf.tangent
        );
        assert_eq!(at_zero("asinh(0.0)").tangent[0], 1.0);
        // abs at 0: the mean of +1 and -1.
        let abs = at_zero("abs(0.0)");
        assert_eq!(abs.tangent[0], 0.0);
        assert!(abs.nonsmooth);
        // sqrt at 0: the infinite right slope is taken as 0.
        let sqrt = at_zero("sqrt(0.0)");
        assert_eq!(sqrt.tangent[0], 0.0);
        assert!(sqrt.nonsmooth);
        // A min tie, including +0 against -0.
        let min = at_zero("min(0.0,-0.0*c1)");
        assert_eq!(min.tangent[0], 0.5);
        assert_eq!(min.tangent[1], 0.5 * 3.0);
        assert!(min.nonsmooth);
        // clip contact: x at the upper bound.
        let clip = at_zero("clip(2.0,-1.0,2.0)");
        assert_eq!(&clip.tangent[..3], &[0.5, 0.0, 0.5]);
        // exp exactly at its argument clamp: half of e^30 (scaled under the 1e12 output clamp).
        let exp = at_zero("exp(30.0)*0.01");
        assert_eq!(exp.tangent[0], 0.01 * (0.5 * 30f64.exp()));
        assert_eq!(exp.tangent[1], 30f64.exp());
        assert!(exp.nonsmooth);
        // Above the output clamp the value saturates and the tangent is zero.
        let saturated = at_zero("exp(30.0)");
        assert_eq!(saturated.value, 1e12);
        assert_eq!(saturated.tangent[0], 0.0);
        assert!(saturated.faults.contains(FaultSet::MAGNITUDE_CLAMPED));
        // ^ exactly at the exponent clamp.
        let pow = at_zero("2.0^8.0");
        assert_eq!(pow.tangent[0], 8.0 * 2f64.powf(7.0));
        assert_eq!(pow.tangent[1], 0.5 * 256.0 * 2f64.ln());
    }

    #[test]
    fn masked_overflow_leaks_no_nan_into_a_lane() {
        for source in [
            "step(1e300*1e300*(1.0*c0),0.0)",
            "tanh(1e300*1e300*(1.0*c0))",
            "min(1e300*1e300*(1.0*c0),3.0*c1)",
            "sign(1e300*1e300*(1.0*c0))",
        ] {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let theta = theta_of(&expression, &sites);
            let row = [1.0, 2.0];
            let dual = program.row(&theta, &row);
            let (value, _) = native(&expression, &row);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
            assert!(dual.value.is_finite(), "{source}");
            assert!(
                dual.tangent.iter().all(|lane| lane.is_finite()),
                "{source}: {:?}",
                dual.tangent
            );
        }
    }

    #[test]
    fn overflowing_pow_slopes_leave_finite_lanes() {
        let at = |source: &str, row: &[f64]| {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let dual = program.row(&theta_of(&expression, &sites), row);
            assert_eq!(
                dual.value.to_bits(),
                native(&expression, row).0.to_bits(),
                "{source}"
            );
            dual
        };
        // A saturated exponent (9 > 8) is a constant, so its lane is exactly 0, although
        // v ln a = 6.6e307 * 88.6 overflows; 0 * inf made a NaN lane before.
        let saturated = at("log(c0^9.0)", &[3e38]);
        assert_eq!(saturated.tangent[0], 0.0);
        assert!(!saturated.nonsmooth);
        // Inside the clamp v ln a overflows (v = 1.3e308), although log(c0^e) has the finite slope
        // ln c0 = 89.8 that forward mode cannot carry through v: the lane is 0 and the row is
        // nonsmooth, never inf.
        let overflowed = at("log(c0^7.9)", &[1e39]);
        assert_eq!(overflowed.tangent[0], 0.0);
        assert!(overflowed.nonsmooth);
        // A tiny base with e < 1: e a^(e-1) overflows to -inf, and the ratio form gives the exact
        // slope e v (a'/a); the base's zero lane for the exponent no longer turns NaN.
        let tiny = at("(1.0*c0)^(-0.03)", &[1e-300]);
        let v = tiny.value;
        assert!(v > 1e8 && v < 1e10, "{v}");
        assert_eq!(tiny.tangent[0], -0.03 * v);
        assert_eq!(tiny.tangent[1], v * 1e-300f64.ln());
        assert!(!tiny.nonsmooth);
    }

    /// Independent review of #195, R3: a finite base slope times a large lane, and an exponent
    /// term times a large lane, overflowed past the `^` rule into the normal equations. Both rows
    /// have finite native values and no faults.
    #[test]
    fn power_terms_that_overflow_after_their_slopes_are_zero() {
        // Each case with the lane its overflow zeroes.
        for (source, row, zeroed) in [
            ("log((1.0*c0-c1+1e-300)^(-0.02))", vec![1e308, 1e308], 0),
            ("0.001*log(c2^(1.0*c0-c1))", vec![1e308, 1e308, 1e200], 1),
        ] {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let dual = program.row(&theta_of(&expression, &sites), &row);
            let (value, faults) = native(&expression, &row);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
            assert!(
                value.is_finite() && faults.is_empty(),
                "{source}: {value} {faults:?}"
            );
            assert!(
                dual.tangent.iter().all(|lane| lane.is_finite()),
                "{source}: {:?}",
                dual.tangent
            );
            assert_eq!(dual.tangent[zeroed].to_bits(), 0.0f64.to_bits(), "{source}");
            assert!(dual.nonsmooth, "{source}");
        }
    }

    /// The mirror of R3: for a large base and e < 0, e a^(e-1) underflows to 0 while v = a^e is a
    /// normal number, which left a silent zero lane on a smooth row. The ratio form gives the slope.
    #[test]
    fn power_slopes_that_underflow_take_the_ratio_form() {
        // d/dθ log((θ c0)^e) = e / θ = e at θ = 1.
        for (source, row, slope) in [
            ("log((1.0*c0)^(-1.0))", 1e170, -1.0),
            ("log((1.0*c0)^(-7.9))", 1e37, -7.9),
        ] {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let dual = program.row(&theta_of(&expression, &sites), &[row]);
            let (value, faults) = native(&expression, &[row]);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
            assert!(
                value.is_finite() && faults.is_empty(),
                "{source}: {value} {faults:?}"
            );
            assert_eq!(dual.tangent[0], slope, "{source}: {:?}", dual.tangent);
            assert!(!dual.nonsmooth, "{source}");
        }
    }

    /// Independent review of #195, R4: past `|v|` of about 27.3 `e^(-v²)` is 0 while `2 v P(t)`
    /// can overflow. The slope is an exact 0, not NaN, and the row stays smooth.
    #[test]
    fn erf_tails_have_an_exact_zero_slope() {
        let (expression, sites) = sites_of("erf(1.0*c0)");
        let program = Program::lower(&expression.root, &sites).unwrap();
        for x in [1e308, -1e308, 1e200, 40.0, -40.0] {
            let dual = program.row(&[1.0], &[x]);
            let (value, faults) = native(&expression, &[x]);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{x}");
            assert_eq!(value.abs(), 1.0, "{x}");
            assert!(faults.is_empty(), "{x}: {faults:?}");
            assert_eq!(dual.tangent[0].to_bits(), 0.0f64.to_bits(), "{x}");
            assert!(!dual.nonsmooth, "{x}");
        }
        // Before the tail the closed form still gives a positive, finite slope.
        let near = program.row(&[1.0], &[5.0]);
        assert!(near.tangent[0] > 0.0 && near.tangent[0].is_finite());
    }

    /// The same class as R3/R4 in other rules: an intermediate that overflows while the native
    /// value and the slope are finite. `atan` squares its argument, and `/` forms q b'. Each
    /// case's slope is checked against its closed form.
    #[test]
    fn large_arguments_keep_finite_slopes() {
        for (source, row, expected) in [
            // d/dθ atan(θ c0) = c0 / (1 + (θ c0)²) = 1 / 2e154 at θ c0 = 2e154.
            ("atan(1.0*c0)", vec![2e154], 1.0 / 2e154),
            // d/dθ c0 / (θ c1) = -c0 / (θ² c1) = -1e20 at θ = 1e-10, c0 = c1 = 1e300.
            ("c0/(1e-10*c1)", vec![1e300, 1e300], -1e20),
        ] {
            let (expression, sites) = sites_of(source);
            let program = Program::lower(&expression.root, &sites).unwrap();
            let dual = program.row(&theta_of(&expression, &sites), &row);
            let (value, faults) = native(&expression, &row);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
            assert!(
                value.is_finite() && faults.is_empty(),
                "{source}: {value} {faults:?}"
            );
            let slope = dual.tangent[0];
            assert!(
                (slope - expected).abs() <= 1e-15 * expected.abs(),
                "{source}: {slope} against {expected}"
            );
            assert!(!dual.nonsmooth, "{source}");
        }
    }

    /// Independent review of #195, R5: in `(c0/(θ c1)) c2` at θ = 1e13 with c1 = 1e-24, b = 1e-11
    /// and b' = 1e-24, so q b' fell below the smallest normal number before the division by b
    /// scaled it back up. On the reviewer's row it rounded to 0, a silent zero lane on a smooth
    /// row; on the second it kept about 18 bits (a relative error of 1.3e-6). The slope is
    /// -value/θ.
    #[test]
    fn division_numerators_that_underflow_keep_their_slope() {
        assert_eq!(SCALE_UP, 2f64.powi(600));
        assert_eq!(SCALE_UP * SCALE_DOWN, 1.0);
        let (expression, sites) = sites_of("(c0/(1e13*c1))*c2");
        let program = Program::lower(&expression.root, &sites).unwrap();
        let theta = theta_of(&expression, &sites);
        // Each row with the relative tolerance its intermediate slope -q/θ allows: on the first
        // row that slope is about -1e-313, a subnormal with about 34 bits.
        for (row, tolerance) in [
            ([1e-311, 1e-24, 1e308], 1e-9),
            ([1e-305, 1e-24, 1e300], 1e-14),
        ] {
            let dual = program.row(&theta, &row);
            let (value, faults) = native(&expression, &row);
            assert_eq!(dual.value.to_bits(), value.to_bits(), "{row:?}");
            assert!(
                value.is_finite() && faults.is_empty(),
                "{row:?}: {value} {faults:?}"
            );
            let expected = -value / theta[0];
            let slope = dual.tangent[0];
            assert!(
                (slope - expected).abs() <= tolerance * expected.abs(),
                "{row:?}: {slope} against {expected}"
            );
            assert!(!dual.nonsmooth, "{row:?}");
        }
    }

    /// The slope of a one-site row, checked against the native value bit for bit and for no
    /// fault, and against `expected` to `tolerance` relative, on a smooth row.
    fn check_slope(source: &str, row: &[f64], expected: f64, tolerance: f64) {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let dual = program.row(&theta_of(&expression, &sites), row);
        let (value, faults) = native(&expression, row);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
        assert!(
            value.is_finite() && faults.is_empty(),
            "{source}: {value} {faults:?}"
        );
        let slope = dual.tangent[0];
        assert!(
            (slope - expected).abs() <= tolerance * expected.abs(),
            "{source}: {slope} against {expected}"
        );
        assert!(!dual.nonsmooth, "{source}");
    }

    /// Independent review of #195, R7: `q = a / b` can itself underflow to 0 while the slope at
    /// the division, `-a b' / b²`, is normal. Each expected slope is formed in an order whose
    /// intermediates are normal.
    #[test]
    fn division_quotients_that_underflow_keep_their_slope() {
        let a = f64::from_bits(1);
        let b = 1e-300 * 1e308;
        let slope = -(a * 1e308) / (b * b);
        check_slope("c0/(1e-300*c1)", &[a, 1e308], slope, 1e-14);
        check_slope(
            "(c0/(1e-300*c1))*c2",
            &[a, 1e308, 1e30],
            slope * 1e30,
            1e-14,
        );
    }

    /// Independent review of #195, R8: for a tiny base and e > 1 both `e a^(e-1)` and `v = a^e`
    /// underflow, while the slope at the power, `e a^(e-1) a'`, is normal; and for a tiny base and
    /// a large exponent lane, `v ln a` is subnormal while `v ln a b'` is normal.
    #[test]
    fn power_coefficients_that_underflow_keep_their_slope() {
        // a = 1e-200 and a' = 1e110: 3 a² a' = 3e-290.
        let a = 1e-310 * 1e110;
        let slope = 3.0 * (a * 1e100) * (a * 1e100) * 1e110 * 1e-200;
        check_slope("(1e-310*c0)^3.0", &[1e110], slope, 1e-12);
        // A non-integral exponent: a = 1e-150, 3.5 a^2.5 a' = 3.5e-265.
        let a = 1e-260f64 * 1e110;
        let slope = 3.5 * (a * 1e100).powf(2.5) * 1e110 * 1e-250;
        check_slope("(1e-260*c0)^3.5", &[1e110], slope, 1e-12);
        // The exponent term: a = 1e-40 and e = 7.9, so v = 1e-316 is subnormal, and
        // d/dθ a^(θ c1) = v ln(a) c1 = -7.3e-15.
        let (a, e) = (1e-40f64, 1e-299 * 7.9e299);
        let half = a.powf(e / 2.0);
        let slope = half * (half * (a.ln() * 7.9e299));
        check_slope("c0^(1e-299*c1)", &[a, 7.9e299], slope, 1e-12);
    }

    /// Independent review of #195, R6: `1 - tanh²` cancels to 0 once `tanh` rounds to 1, and the
    /// coefficients of `tanh`, `erf` and `sigmoid` can leave the normal range while a large lane
    /// brings their product back into it.
    #[test]
    fn saturating_functions_keep_their_slope() {
        // 20 sech²(20), to 100 digits 3.398683404233271e-16.
        check_slope("tanh(1.0*c0)", &[20.0], 3.398683404233271e-16, 1e-14);
        // sech²(x) = 4 e^-2x / (1 + e^-2x)² at x = 400 is about 1e-347; times a' = 1e301.
        let x = 4e-299f64 * 1e301;
        let slope = 4.0 * ((-x).exp() * 1e301) * (-x).exp();
        check_slope("tanh(4e-299*c0)", &[1e301], slope, 1e-12);
        // erf'(30) is about 1e-391: E = e^-900 is split in two halves.
        let u = 3e-300f64 * 1e301;
        let half = (-(u * u) / 2.0).exp();
        let slope = erf_bracket(u) * (half * 1e301) * half;
        check_slope("erf(3e-300*c0)", &[1e301], slope, 1e-12);
        // σ (1 - σ) at 29 from the argument; 1 - σ would keep only about 12 bits here.
        let e = (-29f64).exp();
        check_slope(
            "sigmoid(1.0*c0)",
            &[29.0],
            29.0 * e / ((1.0 + e) * (1.0 + e)),
            1e-14,
        );
    }

    #[test]
    fn lanes_are_independent_of_one_another() {
        let rows: Vec<Vec<f64>> = (0..7)
            .map(|i| vec![0.3 * f64::from(i) - 0.9, 1.0])
            .collect();
        for source in [
            "1.5*exp(0.4*c0)+0.25*sin(2.0*c1)",
            "clip(0.5*c0,-0.25,0.75)*min(1.0,2.0*c0)",
            "(1.2*c0+3.0)^1.7-erf(0.3*c0)",
        ] {
            let (expression, sites) = sites_of(source);
            let all = Program::lower(&expression.root, &sites).unwrap();
            let theta = theta_of(&expression, &sites);
            for (k, site) in sites.iter().enumerate() {
                let one = Program::lower(&expression.root, std::slice::from_ref(site)).unwrap();
                for row in &rows {
                    let full = all.row(&theta, row).tangent[k];
                    let alone = one.row(&[theta[k]], row).tangent[0];
                    assert_eq!(
                        full.to_bits(),
                        alone.to_bits(),
                        "{source} lane {k} on {row:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn lowering_refuses_sites_that_are_not_literal_nodes() {
        let (expression, _) = sites_of("1.5*c0");
        assert!(Program::lower(&expression.root, &[vec![1]]).is_err());
        assert!(Program::lower(&expression.root, &[vec![0, 0]]).is_err());
        let too_many: Vec<Vec<usize>> = (0..=MAX_PARAMS).map(|_| vec![0]).collect();
        assert!(Program::lower(&expression.root, &too_many).is_err());
    }
}
