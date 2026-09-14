//! Parser + code generators for the scalar math-expression grammar.
//!
//! A compatibility adapter over `crate::math_expr`'s single AST and parser.
//! Both code generators mirror
//! `MathExpr::eval` exactly, including every guard:
//!   - division: |denominator| < 1e-12  => 0.0
//!   - pow: exponent clamped to [-8, 8], C `pow` semantics (NaN/inf preserved)
//!   - sqrt: sqrt(max(v, 0.0))
//!   - log1p: v <= -1  => 0.0
//!   - sigmoid: logistic with argument clamped to [-30, 30]
//!   - min/max: IEEE-754 minNum/maxNum (NaN yields the other operand)
//!   - clip(v, a, b): clamp to [min(a,b), max(a,b)]
//!   - step(a, c): hard indicator, 1.0 when a > c else 0.0 (NaN => 0.0). This is
//!     the honest primitive for thresholds; it emits an explicit conditional in
//!     both languages, never the `clip((a - c) * 1e15, 0, 1)` idiom that older
//!     symbolic genomes used. `detect_hard_step_idiom` recognizes that legacy
//!     idiom so tooling can flag genomes for re-synthesis with `step`; the
//!     emitter itself keeps the literal clip form for such genomes because a
//!     conditional rewrite is NOT bit-identical to the runtime (a true step and
//!     the clip-scaled form disagree within a ~1e-15 window), which would break
//!     the py == rs == runtime parity invariant.
//!
//! The final value passes through `safe`: non-finite => 0.0, else clamped to
//! +/-1e12. A column index outside the row (or the element variable `x` in a
//! whole-row context) aborts the whole expression to 0.0, matching
//! `eval_row(...).unwrap_or(0.0)` upstream.

use anyhow::Result;
use std::collections::BTreeSet;
pub use crate::math_expr::{Expr as MathAst, Var as MathVar, BinOp as MathBinOp, Func as MathFunc};
use crate::math_expr::{MathExpr, UnOp};

/// Features of a parsed expression that drive helper selection and guards.
#[derive(Clone, Debug, Default)]
pub struct ExprFeatures {
    pub uses_x: bool,
    pub max_col: Option<usize>,
    pub funcs: BTreeSet<MathFunc>,
    pub vars: BTreeSet<MathVar>,
    pub uses_div: bool,
    pub uses_pow: bool,
}

impl MathAst {
    pub fn parse(src: &str) -> Result<MathAst> {
        Ok(MathExpr::parse(src)?.ast().clone())
    }

    pub fn features(&self) -> ExprFeatures {
        let mut f = ExprFeatures::default();
        collect(self, &mut f);
        f
    }

    /// Python expression over a local `row` list (and `x` in element context).
    pub fn to_python(&self) -> String {
        match self {
            MathAst::Const(v) => fmt_f64(*v),
            MathAst::Var(var) => match var {
                MathVar::X => "x".to_string(),
                MathVar::Col(i) => format!("row[{i}]"),
                MathVar::RowMean => "_row_mean(row)".to_string(),
                MathVar::RowSum => "_row_sum(row)".to_string(),
                MathVar::RowL1 => "_row_l1(row)".to_string(),
                MathVar::RowL2 => "_row_l2(row)".to_string(),
                MathVar::RowMin => "_row_min(row)".to_string(),
                MathVar::RowMax => "_row_max(row)".to_string(),
                MathVar::RowStd => "_row_std(row)".to_string(),
            },
            MathAst::Unary(UnOp::Neg, a) => format!("(-{})", a.to_python()),
            MathAst::Binary(op, a, b) => {
                let (a, b) = (a.to_python(), b.to_python());
                match op {
                    MathBinOp::Add => format!("({a} + {b})"),
                    MathBinOp::Sub => format!("({a} - {b})"),
                    MathBinOp::Mul => format!("({a} * {b})"),
                    MathBinOp::Div => format!("_div({a}, {b})"),
                    MathBinOp::Pow => format!("_pow({a}, _clamp({b}, -8.0, 8.0))"),
                }
            }
            MathAst::Call(func, args) => {
                let a: Vec<String> = args.iter().map(|e| e.to_python()).collect();
                match func {
                    MathFunc::Abs => format!("abs({})", a[0]),
                    MathFunc::Sqrt => format!("math.sqrt(_fmax({}, 0.0))", a[0]),
                    MathFunc::Log1p => format!("_log1p({})", a[0]),
                    MathFunc::Sin => format!("_safe_sin({})", a[0]),
                    MathFunc::Cos => format!("_safe_cos({})", a[0]),
                    MathFunc::Tanh => format!("math.tanh({})", a[0]),
                    MathFunc::Sigmoid => format!("_sigmoid({})", a[0]),
                    MathFunc::Min => format!("_fmin({}, {})", a[0], a[1]),
                    MathFunc::Max => format!("_fmax({}, {})", a[0], a[1]),
                    MathFunc::Clip => format!("_clip({}, {}, {})", a[0], a[1], a[2]),
                    // Explicit hard indicator — no `*1e15` clip trick.
                    MathFunc::Step => format!("(1.0 if {} > {} else 0.0)", a[0], a[1]),
                    MathFunc::Exp => format!("_exp({})", a[0]),
                    MathFunc::Log => format!("_log({})", a[0]),
                    MathFunc::Asinh => format!("_asinh({})", a[0]),
                    MathFunc::Atan => format!("math.atan({})", a[0]),
                    MathFunc::Erf => format!("_erf({})", a[0]),
                    MathFunc::Sign => format!("_sign({})", a[0]),
                    MathFunc::Floor => format!("_floor({})", a[0]),
                    MathFunc::Round => format!("_round_half_away({})", a[0]),
                }
            }
        }
    }

    /// Rust expression over a local `row: &[f64]` (and `x: f64` in element context).
    pub fn to_rust(&self) -> String {
        match self {
            // The f64 suffix keeps literals unambiguous under method calls.
            MathAst::Const(v) => format!("{}f64", fmt_f64(*v)),
            MathAst::Var(var) => match var {
                MathVar::X => "x".to_string(),
                MathVar::Col(i) => format!("row[{i}]"),
                MathVar::RowMean => "row_mean(row)".to_string(),
                MathVar::RowSum => "row_sum(row)".to_string(),
                MathVar::RowL1 => "row_l1(row)".to_string(),
                MathVar::RowL2 => "row_l2(row)".to_string(),
                MathVar::RowMin => "row_min(row)".to_string(),
                MathVar::RowMax => "row_max(row)".to_string(),
                MathVar::RowStd => "row_std(row)".to_string(),
            },
            MathAst::Unary(UnOp::Neg, a) => format!("(-{})", a.to_rust()),
            MathAst::Binary(op, a, b) => {
                let (a, b) = (a.to_rust(), b.to_rust());
                match op {
                    MathBinOp::Add => format!("({a} + {b})"),
                    MathBinOp::Sub => format!("({a} - {b})"),
                    MathBinOp::Mul => format!("({a} * {b})"),
                    MathBinOp::Div => format!("m_div({a}, {b})"),
                    MathBinOp::Pow => format!("({a}).powf(({b}).clamp(-8.0, 8.0))"),
                }
            }
            MathAst::Call(func, args) => {
                let a: Vec<String> = args.iter().map(|e| e.to_rust()).collect();
                match func {
                    MathFunc::Abs => format!("({}).abs()", a[0]),
                    MathFunc::Sqrt => format!("({}).max(0.0).sqrt()", a[0]),
                    MathFunc::Log1p => format!("m_log1p({})", a[0]),
                    MathFunc::Sin => format!("({}).sin()", a[0]),
                    MathFunc::Cos => format!("({}).cos()", a[0]),
                    MathFunc::Tanh => format!("({}).tanh()", a[0]),
                    MathFunc::Sigmoid => format!("m_sigmoid({})", a[0]),
                    MathFunc::Min => format!("({}).min({})", a[0], a[1]),
                    MathFunc::Max => format!("({}).max({})", a[0], a[1]),
                    MathFunc::Clip => format!("m_clip({}, {}, {}, &mut math_failed)", a[0], a[1], a[2]),
                    // Explicit hard indicator — no `*1e15` clip trick.
                    MathFunc::Step => format!("(if {} > {} {{ 1.0 }} else {{ 0.0 }})", a[0], a[1]),
                    MathFunc::Exp => format!("m_exp({})", a[0]),
                    MathFunc::Log => format!("m_log({})", a[0]),
                    MathFunc::Asinh => format!("m_asinh({})", a[0]),
                    MathFunc::Atan => format!("({}).atan()", a[0]),
                    MathFunc::Erf => format!("m_erf({})", a[0]),
                    MathFunc::Sign => format!("m_sign({})", a[0]),
                    MathFunc::Floor => format!("({}).floor()", a[0]),
                    MathFunc::Round => format!("m_round({})", a[0]),
                }
            }
        }
    }

    /// C++17 expression over `row: const double*` and `row_len`.
    ///
    /// Transcendentals lower through `m_*` wrappers supplied by the C++
    /// emitter. That single routing layer selects PlatformLibm or the
    /// verified portable kernel without duplicating AST semantics here.
    #[allow(dead_code)] // Shared P-1C/P-1D lowering boundary; C++ symbolic emission fails closed today.
    pub fn to_cpp(&self, portable: bool) -> String {
        match self {
            MathAst::Const(v) => {
                if v.is_nan() {
                    "std::numeric_limits<double>::quiet_NaN()".to_string()
                } else if *v == f64::INFINITY {
                    "std::numeric_limits<double>::infinity()".to_string()
                } else if *v == f64::NEG_INFINITY {
                    "(-std::numeric_limits<double>::infinity())".to_string()
                } else {
                    fmt_f64(*v)
                }
            }
            MathAst::Var(var) => match var {
                MathVar::X => "x".to_string(),
                MathVar::Col(i) => format!("row[{i}]"),
                MathVar::RowMean => "row_mean(row, row_len)".to_string(),
                MathVar::RowSum => "row_sum(row, row_len)".to_string(),
                MathVar::RowL1 => "row_l1(row, row_len)".to_string(),
                MathVar::RowL2 => "row_l2(row, row_len)".to_string(),
                MathVar::RowMin => "row_min(row, row_len)".to_string(),
                MathVar::RowMax => "row_max(row, row_len)".to_string(),
                MathVar::RowStd => "row_std(row, row_len)".to_string(),
            },
            MathAst::Unary(UnOp::Neg, a) => format!("(-{})", a.to_cpp(portable)),
            MathAst::Binary(op, a, b) => {
                let (a, b) = (a.to_cpp(portable), b.to_cpp(portable));
                match op {
                    MathBinOp::Add => format!("({a} + {b})"),
                    MathBinOp::Sub => format!("({a} - {b})"),
                    MathBinOp::Mul => format!("({a} * {b})"),
                    MathBinOp::Div => format!("m_div({a}, {b})"),
                    MathBinOp::Pow => format!("m_pow({a}, m_clamp({b}, -8.0, 8.0))"),
                }
            }
            MathAst::Call(func, args) => {
                let a: Vec<String> = args.iter().map(|e| e.to_cpp(portable)).collect();
                match func {
                    MathFunc::Abs => format!("std::fabs({})", a[0]),
                    MathFunc::Sqrt => format!("m_sqrt({})", a[0]),
                    MathFunc::Log1p => format!("m_log1p({})", a[0]),
                    MathFunc::Sin => format!("m_sin({})", a[0]),
                    MathFunc::Cos => format!("m_cos({})", a[0]),
                    MathFunc::Tanh => format!("m_tanh({})", a[0]),
                    MathFunc::Sigmoid => format!("m_sigmoid({})", a[0]),
                    MathFunc::Min => format!("std::fmin({}, {})", a[0], a[1]),
                    MathFunc::Max => format!("std::fmax({}, {})", a[0], a[1]),
                    MathFunc::Clip => format!("m_clip({}, {}, {})", a[0], a[1], a[2]),
                    MathFunc::Step => format!("(({}) > ({}) ? 1.0 : 0.0)", a[0], a[1]),
                    MathFunc::Exp => format!("m_exp({})", a[0]),
                    MathFunc::Log => format!("m_log({})", a[0]),
                    MathFunc::Asinh => format!("m_asinh({})", a[0]),
                    MathFunc::Atan => format!("m_atan({})", a[0]),
                    MathFunc::Erf => format!("m_erf({})", a[0]),
                    MathFunc::Sign => format!("m_sign({})", a[0]),
                    MathFunc::Floor => format!("std::floor({})", a[0]),
                    MathFunc::Round => format!("m_round({})", a[0]),
                }
            }
        }
    }
}

/// Shortest round-trip decimal (Rust `{:?}`) — a valid literal in both
/// languages that parses back to the identical f64 bits.
pub fn fmt_f64(v: f64) -> String {
    format!("{v:?}")
}

impl MathAst {
    /// Recognize the legacy hard-threshold idiom `clip((A - B) * K, 0, 1)` with
    /// a huge scale `K` (>= 1e12) — the `*1e15` step-function that reviewers flag
    /// as machine-looking. Returns the threshold operands `(A, B)` of the
    /// equivalent `step(A, B)` when the shape matches.
    ///
    /// This is a *detector*, not a rewrite: the emitter still renders the literal
    /// clip so it stays bit-identical to the runtime (which evaluates the DSL as
    /// written). Callers use it to steer such genomes toward the real `step`
    /// primitive, whose emitted code carries no `1e15`. Wired for search/judge
    /// tooling; exercised by unit tests here.
    #[allow(dead_code)]
    pub fn detect_hard_step_idiom(&self) -> Option<(&MathAst, &MathAst)> {
        const HUGE_SCALE: f64 = 1e12;
        let MathAst::Call(MathFunc::Clip, args) = self else {
            return None;
        };
        let [inner, lo, hi] = args.as_slice() else {
            return None;
        };
        if !matches!((lo, hi), (MathAst::Const(l), MathAst::Const(h)) if *l == 0.0 && *h == 1.0) {
            return None;
        }
        let MathAst::Binary(MathBinOp::Mul, a, b) = inner else {
            return None;
        };
        // The scale can sit on either side of the product.
        let diff = match (a.as_ref(), b.as_ref()) {
            (MathAst::Const(k), other) | (other, MathAst::Const(k)) if k.abs() >= HUGE_SCALE => {
                other
            }
            _ => return None,
        };
        if let MathAst::Binary(MathBinOp::Sub, lhs, rhs) = diff {
            Some((lhs, rhs))
        } else {
            None
        }
    }
}

fn collect(e: &MathAst, f: &mut ExprFeatures) {
    match e {
        MathAst::Const(_) => {}
        MathAst::Var(v) => {
            f.vars.insert(*v);
            match v {
                MathVar::X => f.uses_x = true,
                MathVar::Col(i) => {
                    f.max_col = Some(f.max_col.map_or(*i, |m| m.max(*i)));
                }
                _ => {}
            }
        }
        MathAst::Unary(UnOp::Neg, a) => collect(a, f),
        MathAst::Binary(op, a, b) => {
            match op {
                MathBinOp::Div => f.uses_div = true,
                MathBinOp::Pow => f.uses_pow = true,
                _ => {}
            }
            collect(a, f);
            collect(b, f);
        }
        MathAst::Call(func, args) => {
            f.funcs.insert(*func);
            for a in args {
                collect(a, f);
            }
        }
    }
}

// ------------------------------- lexer/parser -------------------------------
// A line-by-line port of newmath-core math_expr.rs so acceptance matches.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_both_languages() {
        let e = MathAst::parse("sigmoid(col(0)) + 0.5*c1 - log1p(abs(col2))").unwrap();
        let py = e.to_python();
        assert!(py.contains("_sigmoid(row[0])"));
        assert!(py.contains("row[1]"));
        assert!(py.contains("_log1p(abs(row[2]))"));
        let rs = e.to_rust();
        assert!(rs.contains("m_sigmoid(row[0])"));
        let f = e.features();
        assert_eq!(f.max_col, Some(2));
        assert!(!f.uses_x);
    }

    #[test]
    fn step_emits_explicit_conditional_not_scale_trick() {
        let e = MathAst::parse("step(col(0), 0.5)").unwrap();
        let py = e.to_python();
        let rs = e.to_rust();
        assert_eq!(py, "(1.0 if row[0] > 0.5 else 0.0)");
        assert_eq!(rs, "(if row[0] > 0.5f64 { 1.0 } else { 0.0 })");
        assert!(!py.contains("1e15") && !rs.contains("1e15"));
        // Runtime and emitter must agree on acceptance.
        assert!(crate::math_expr::MathExpr::parse("step(col(0), 0.5)").is_ok());
        assert!(MathAst::parse("step(c0)").is_err());
    }

    #[test]
    fn detects_legacy_hard_step_idiom() {
        let e = MathAst::parse("clip((col(0) - 0.5) * 1e15, 0, 1)").unwrap();
        let (a, b) = e
            .detect_hard_step_idiom()
            .expect("idiom should be detected");
        assert_eq!(*a, MathAst::Var(MathVar::Col(0)));
        assert_eq!(*b, MathAst::Const(0.5));
        // Scale on the left is detected too.
        assert!(MathAst::parse("clip(1000000000000000 * (c1 - c2), 0, 1)")
            .unwrap()
            .detect_hard_step_idiom()
            .is_some());
        // A genuine clip is not a false positive.
        assert!(MathAst::parse("clip(c0 + c1, 0, 1)")
            .unwrap()
            .detect_hard_step_idiom()
            .is_none());
        // Parity is preserved: the emitter still renders the literal clip form.
        assert!(e.to_python().contains("1000000000000000.0"));
    }

    #[test]
    fn mirrors_reference_acceptance() {
        for src in [
            "x^2 + 1",
            "clip(row_l2 + row_mean, 0, 10)",
            "min(c0, max(c1, 2))",
        ] {
            assert!(MathAst::parse(src).is_ok(), "{src}");
            assert!(crate::math_expr::MathExpr::parse(src).is_ok(), "{src}");
        }
        for src in ["made_up(x)", "col(-1)", "1 +", "row_q"] {
            assert!(MathAst::parse(src).is_err(), "{src}");
            assert!(crate::math_expr::MathExpr::parse(src).is_err(), "{src}");
        }
    }
}
