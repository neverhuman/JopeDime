//! The expression DSL for synthesized features: the `Expr` tree, its evaluator (over a
//! column-major `Matrix`), and a formula formatter. Ported from tools/gpfeat/src/main.rs and kept
//! numerically identical (same ops, same NaN rules, same erf/rank/zscore).
use crate::data::Matrix;
use serde::{Deserialize, Serialize};

pub mod ops;
pub use ops::erf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    Const(f32),
    Col(u16),
    Miss(u16),
    Un(Un, Box<Expr>),
    Par(Par, Box<Expr>, f32, f32),
    Bin(Bin, Box<Expr>, Box<Expr>),
    Where(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Un {
    Neg,
    Abs,
    Square,
    Cube,
    SqrtAbs,
    Recip,
    Log1p,
    Log,
    Exp,
    Sigmoid,
    Tanh,
    Arcsinh,
    Erf,
    GaussCdf,
    Sin,
    Cos,
    Softplus,
    Relu,
    Logit,
    Zscore,
    Rank,
    Sign,
}
pub const UNS: [Un; 22] = [
    Un::Neg,
    Un::Abs,
    Un::Square,
    Un::Cube,
    Un::SqrtAbs,
    Un::Recip,
    Un::Log1p,
    Un::Log,
    Un::Exp,
    Un::Sigmoid,
    Un::Tanh,
    Un::Arcsinh,
    Un::Erf,
    Un::GaussCdf,
    Un::Sin,
    Un::Cos,
    Un::Softplus,
    Un::Relu,
    Un::Logit,
    Un::Zscore,
    Un::Rank,
    Un::Sign,
];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Par {
    Hill,
    Mm,
    Rbf,
    Clip,
    Powr,
}
pub const PARS: [Par; 5] = [Par::Hill, Par::Mm, Par::Rbf, Par::Clip, Par::Powr];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    SafeDiv,
    Min,
    Max,
    LogRatio,
    Hinge,
    Gt,
    Lt,
    Ge,
    Le,
    EqA,
}
pub const BINS: [Bin; 13] = [
    Bin::Add,
    Bin::Sub,
    Bin::Mul,
    Bin::SafeDiv,
    Bin::Min,
    Bin::Max,
    Bin::LogRatio,
    Bin::Hinge,
    Bin::Gt,
    Bin::Lt,
    Bin::Ge,
    Bin::Le,
    Bin::EqA,
];

impl Expr {
    pub fn depth(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Col(_) | Expr::Miss(_) => 1,
            Expr::Un(_, a) | Expr::Par(_, a, _, _) => 1 + a.depth(),
            Expr::Bin(_, a, b) => 1 + a.depth().max(b.depth()),
            Expr::Where(c, a, b) => 1 + c.depth().max(a.depth()).max(b.depth()),
        }
    }
    pub fn nodes(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Col(_) | Expr::Miss(_) => 1,
            Expr::Un(_, a) | Expr::Par(_, a, _, _) => 1 + a.nodes(),
            Expr::Bin(_, a, b) => 1 + a.nodes() + b.nodes(),
            Expr::Where(c, a, b) => 1 + c.nodes() + a.nodes() + b.nodes(),
        }
    }

    /// Collect every column index this expression references (via `Col(i)`/`Miss(i)`), into `out`.
    /// Used to detect whether a GP feature references a Hyperion embedding column.
    pub fn col_refs(&self, out: &mut Vec<usize>) {
        match self {
            Expr::Const(_) => {}
            Expr::Col(i) | Expr::Miss(i) => out.push(*i as usize),
            Expr::Un(_, a) | Expr::Par(_, a, _, _) => a.col_refs(out),
            Expr::Bin(_, a, b) => {
                a.col_refs(out);
                b.col_refs(out);
            }
            Expr::Where(c, a, b) => {
                c.col_refs(out);
                a.col_refs(out);
                b.col_refs(out);
            }
        }
    }

    /// Evaluate on a column-major matrix. NaN is preserved (missing is a first-class value).
    pub fn eval(&self, m: &Matrix) -> Vec<f32> {
        match self {
            Expr::Const(k) => vec![*k; m.n],
            // A column index beyond the matrix width = a feature absent from THIS matrix (e.g. an
            // expression built against a wider matrix, or replayed on a narrower scoring/holdout frame).
            // Treat it as fully missing (NaN) rather than panicking; downstream median-fill handles NaN.
            Expr::Col(i) => {
                let i = *i as usize;
                if i < m.c {
                    m.col(i).to_vec()
                } else {
                    vec![f32::NAN; m.n]
                }
            }
            Expr::Miss(i) => {
                let i = *i as usize;
                if i < m.c {
                    m.col(i)
                        .iter()
                        .map(|v| if v.is_finite() { 0.0 } else { 1.0 })
                        .collect()
                } else {
                    vec![1.0; m.n] // absent column is missing everywhere
                }
            }
            Expr::Un(op, a) => {
                let mut v = a.eval(m);
                ops::apply_un(*op, &mut v);
                v
            }
            Expr::Par(op, a, p, q) => {
                let mut v = a.eval(m);
                ops::apply_par(*op, &mut v, *p, *q);
                v
            }
            Expr::Bin(op, a, b) => {
                let va = a.eval(m);
                let vb = b.eval(m);
                ops::apply_bin(*op, &va, &vb)
            }
            Expr::Where(c, a, b) => {
                let vc = c.eval(m);
                let va = a.eval(m);
                let vb = b.eval(m);
                (0..m.n)
                    .map(|i| {
                        if vc[i].is_finite() && vc[i] > 0.5 {
                            va[i]
                        } else {
                            vb[i]
                        }
                    })
                    .collect()
            }
        }
    }

    /// Human-readable formula (matches the original gpfeat DSL formatting).
    pub fn format(&self, names: &[String]) -> String {
        match self {
            Expr::Const(k) => fmtnum(*k),
            Expr::Col(i) => names
                .get(*i as usize)
                .cloned()
                .unwrap_or_else(|| format!("f{i}")),
            Expr::Miss(i) => format!(
                "is_missing({})",
                names.get(*i as usize).map(String::as_str).unwrap_or("?")
            ),
            Expr::Un(op, a) => {
                let s = a.format(names);
                match op {
                    Un::Neg => format!("(-{s})"),
                    Un::Abs => format!("abs({s})"),
                    Un::Square => format!("({s} * {s})"),
                    Un::Cube => format!("powr({s}, 3)"),
                    Un::SqrtAbs => format!("sqrt(abs({s}))"),
                    Un::Recip => format!("safe_divide(1, {s})"),
                    Un::Log1p => format!("log1p({s})"),
                    Un::Log => format!("log({s})"),
                    Un::Exp => format!("exp({s})"),
                    Un::Sigmoid => format!("sigmoid({s})"),
                    Un::Tanh => format!("tanh({s})"),
                    Un::Arcsinh => format!("arcsinh({s})"),
                    Un::Erf => format!("erf({s})"),
                    Un::GaussCdf => format!("gauss_cdf({s})"),
                    Un::Sin => format!("sin({s})"),
                    Un::Cos => format!("cos({s})"),
                    Un::Softplus => format!("softplus({s})"),
                    Un::Relu => format!("relu({s})"),
                    Un::Logit => format!("logit({s})"),
                    Un::Zscore => format!("zscore({s})"),
                    Un::Rank => format!("rank({s})"),
                    Un::Sign => format!("sign({s})"),
                }
            }
            Expr::Par(op, a, p, q) => {
                let s = a.format(names);
                match op {
                    Par::Hill => format!("hill({s}, {}, {})", fmtnum(*p), fmtnum(*q)),
                    Par::Mm => format!("mm({s}, {}, {})", fmtnum(*p), fmtnum(*q)),
                    Par::Rbf => format!("rbf({s}, {}, {})", fmtnum(*p), fmtnum(*q)),
                    Par::Clip => format!("clip({s}, {}, {})", fmtnum(*p), fmtnum(*q)),
                    Par::Powr => format!("powr({s}, {})", fmtnum(*p)),
                }
            }
            Expr::Bin(op, a, b) => {
                let (x, y) = (a.format(names), b.format(names));
                match op {
                    Bin::Add => format!("({x} + {y})"),
                    Bin::Sub => format!("({x} - {y})"),
                    Bin::Mul => format!("({x} * {y})"),
                    Bin::SafeDiv => format!("safe_divide({x}, {y})"),
                    Bin::Min => format!("min({x}, {y})"),
                    Bin::Max => format!("max({x}, {y})"),
                    Bin::LogRatio => format!("logratio({x}, {y})"),
                    Bin::Hinge => format!("hinge({x}, {y})"),
                    Bin::Gt => format!("gt({x}, {y})"),
                    Bin::Lt => format!("lt({x}, {y})"),
                    Bin::Ge => format!("ge({x}, {y})"),
                    Bin::Le => format!("le({x}, {y})"),
                    Bin::EqA => format!("eq({x}, {y})"),
                }
            }
            Expr::Where(c, a, b) => {
                format!(
                    "where({}, {}, {})",
                    c.format(names),
                    a.format(names),
                    b.format(names)
                )
            }
        }
    }
}

pub fn fmtnum(k: f32) -> String {
    if (k - k.round()).abs() < 1e-6 && k.abs() < 1e6 {
        format!("{}", k.round() as i64)
    } else {
        format!("{:.4}", k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Matrix;

    // Regression: a GP expression referencing a column beyond the matrix width (e.g. replayed on a
    // narrower scoring/holdout frame) must yield NaN "missing", never panic with an out-of-range slice.
    #[test]
    fn col_beyond_width_is_missing_not_panic() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // 3 rows x 2 cols
        let out = Expr::Col(5).eval(&m);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_nan()));
        assert_eq!(Expr::Miss(5).eval(&m), vec![1.0, 1.0, 1.0]);
        assert_eq!(Expr::Col(1).eval(&m), vec![4.0, 5.0, 6.0]); // in-range unchanged
        assert_eq!(Expr::Col(5).format(&["a".into(), "b".into()]), "f5"); // format no panic
    }

    // col_refs collects every referenced column so we can flag a GP feature that "captures" a Hyperion col.
    #[test]
    fn col_refs_collects_all_column_leaves() {
        let e = Expr::Bin(Bin::Add, Box::new(Expr::Col(3)), Box::new(Expr::Miss(130)));
        let mut refs = Vec::new();
        e.col_refs(&mut refs);
        refs.sort_unstable();
        assert_eq!(refs, vec![3, 130]);
        // if hyperion columns start at index 128, this GP feature references one of them
        assert!(refs.iter().any(|&i| i >= 128));
    }
}
