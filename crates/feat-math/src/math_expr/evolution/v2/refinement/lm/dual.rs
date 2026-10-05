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
//! - **Closed forms.** `asinh` and `erf` use closed-form slopes, which hold at 0 too, where
//!   differentiating their sign-times-magnitude code would give 0.
//! - **Infinite one-sided slopes.** `sqrt` at 0 and `0 ^ e` with `0 < e < 1` have tangent 0.
//! - **Steps.** `step`, `sign`, `floor` and `round` have tangent 0.
//! - **Non-finite operands.** Any node other than a selector with a non-finite operand has tangent
//!   0.
//!
//! A row reports `nonsmooth` when a tie, an infinite one-sided slope or a non-finite operand was
//! differentiated. A term whose coefficient is an exact zero is skipped, so `0 * inf` never makes a
//! NaN lane.

use super::*;
use crate::math_expr::fault::Collect;

/// The most parameters one pass differentiates.
pub(super) const MAX_PARAMS: usize = 16;

/// One tangent lane per parameter. Lanes at and above the parameter count stay zero.
pub(super) type Lanes = [f64; MAX_PARAMS];

const ZERO: Lanes = [0.0; MAX_PARAMS];

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
        // The protected branch is a constant 0. Otherwise (a' - q b') / b with q = a / b, which,
        // unlike (a' b - a b') / b², cannot overflow through b².
        BinOp::Div if b.v.abs() < 1e-12 => ZERO,
        BinOp::Div => divide(p, &comb(p, 1.0, &a.d, -v, &b.d), b.v),
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
    // e * a^(e - 1), never e * v / a, which underflows for tiny a.
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
    comb(p, by_base, &a.d, exponent_factor * by_exponent, &b.d)
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
        Func::Tanh => scale(p, 1.0 - v * v, &a.d),
        Func::Sigmoid => {
            let factor = clamp_factor(a.v, -30.0, 30.0);
            if factor == 0.5 {
                *nonsmooth = true;
            }
            scale(p, factor * v * (1.0 - v), &a.d)
        }
        Func::Exp => clamped(p, a.v, -30.0, 30.0, &scale(p, v, &a.d), nonsmooth),
        Func::Log if a.v > 1e-300 => divide(p, &a.d, a.v),
        Func::Log => ZERO,
        Func::Asinh => divide(p, &a.d, (a.v * a.v + 1.0).sqrt()),
        Func::Atan => divide(p, &a.d, 1.0 + a.v * a.v),
        Func::Erf => scale(p, erf_slope(a.v), &a.d),
        Func::Step | Func::Sign | Func::Floor | Func::Round => ZERO,
        Func::Min | Func::Max | Func::Clip => unreachable!("selectors return above"),
    }
}

/// The derivative of the evaluator's `erf` (Abramowitz and Stegun 7.1.26, the same constants),
/// valid at 0: with `u = |v|`, `t = 1 / (1 + p u)`, `P(t) = a1 t + … + a5 t⁵` and `E = e^(-u²)`,
/// the value is `sign(v) (1 - P(t) E)`, whose slope is `E (p t² P'(t) + 2 u P(t))` on both sides.
fn erf_slope(v: f64) -> f64 {
    let u = v.abs();
    let t = 1.0 / (1.0 + 0.3275911 * u);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let slope = 0.254829592
        + t * (2.0 * -0.284496736
            + t * (3.0 * 1.421413741 + t * (4.0 * -1.453152027 + t * (5.0 * 1.061405429))));
    (-(u * u)).exp() * (0.3275911 * t * t * slope + 2.0 * u * poly)
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
        // erf'(0) = p P'(1), 6.4e-6 from 2/sqrt(pi); asinh'(0) = 1.
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
