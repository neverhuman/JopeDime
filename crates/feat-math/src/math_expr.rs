use anyhow::{bail, Context, Result};
mod compiled;
mod evolution;
pub use evolution::numeric_v2 as numeric_v2_scope;
mod exact;
mod fault;
pub use compiled::CompiledMathExpr;
pub use evolution::{ExpressionEdit, ExpressionScope};
pub use fault::{FaultClass, FaultSet, FaultSeverity};
use fault::{FaultSink, NoSink};

const MAX_NODES: usize = 64;
const MAX_DEPTH: usize = 16;
const MAX_ABS: f64 = 1.0e12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathEvalError {
    InvalidClipBounds,
}

impl std::fmt::Display for MathEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidClipBounds => f.write_str("clip: both bounds evaluated to NaN"),
        }
    }
}

impl std::error::Error for MathEvalError {}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MathExpr {
    root: Expr,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Expr {
    Const(f64),
    Var(Var),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Var {
    X,
    Col(usize),
    RowMean,
    RowSum,
    RowL1,
    RowL2,
    RowMin,
    RowMax,
    RowStd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnOp {
    Neg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Func {
    Abs,
    Sqrt,
    Log1p,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Min,
    Max,
    Clip,
    Step,
    Exp,
    Log,
    Asinh,
    Atan,
    Erf,
    Sign,
    Floor,
    Round,
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    End,
}

impl MathExpr {
    pub fn ast(&self) -> &Expr {
        &self.root
    }

    /// Readable source preserving finite literal values and operator order.
    /// Use the program binary codec for structural identity and nonfinite bits.
    pub fn to_source(&self) -> crate::kernel::Result<String> {
        self.check_structure(MAX_NODES, MAX_DEPTH)
            .map_err(|e| crate::kernel::Error::Resource(e.to_string()))?;
        fn render(expr: &Expr) -> crate::kernel::Result<String> {
            Ok(match expr {
                Expr::Const(v) if v.is_finite() => format!("{v:?}"),
                Expr::Const(_) => {
                    return Err(crate::kernel::Error::Domain(
                        "nonfinite literal requires the typed codec".into(),
                    ))
                }
                Expr::Var(v) => match v {
                    Var::X => "x".into(),
                    Var::Col(i) => format!("c{i}"),
                    Var::RowMean => "row_mean".into(),
                    Var::RowSum => "row_sum".into(),
                    Var::RowL1 => "row_l1".into(),
                    Var::RowL2 => "row_l2".into(),
                    Var::RowMin => "row_min".into(),
                    Var::RowMax => "row_max".into(),
                    Var::RowStd => "row_std".into(),
                },
                Expr::Unary(UnOp::Neg, a) => format!("(-{})", render(a)?),
                Expr::Binary(op, a, b) => {
                    let symbol = match op {
                        BinOp::Add => '+',
                        BinOp::Sub => '-',
                        BinOp::Mul => '*',
                        BinOp::Div => '/',
                        BinOp::Pow => '^',
                    };
                    format!("({}{}{})", render(a)?, symbol, render(b)?)
                }
                Expr::Call(func, args) => format!(
                    "{}({})",
                    format!("{func:?}").to_lowercase(),
                    args.iter()
                        .map(render)
                        .collect::<crate::kernel::Result<Vec<_>>>()?
                        .join(",")
                ),
            })
        }
        render(&self.root)
    }

    pub fn from_ast(root: Expr) -> Result<Self> {
        let expression = Self { root };
        expression.check_structure(MAX_NODES, MAX_DEPTH)?;
        Ok(expression)
    }

    fn check_structure(&self, max_nodes: usize, max_depth: usize) -> Result<()> {
        let mut pending = vec![(&self.root, 1)];
        let mut nodes = 0;
        while let Some((node, depth)) = pending.pop() {
            nodes += 1;
            if nodes > max_nodes || depth > max_depth {
                bail!("math expression exceeds node/depth limits");
            }
            match node {
                Expr::Unary(_, child) => pending.push((child, depth + 1)),
                Expr::Binary(_, a, b) => {
                    pending.push((b, depth + 1));
                    pending.push((a, depth + 1));
                }
                Expr::Call(func, args) => {
                    validate_arity(*func, args.len())?;
                    pending.extend(args.iter().map(|arg| (arg, depth + 1)));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn eval_element(&self, x: f64, row: &[f64]) -> Result<f64> {
        let ctx = EvalContext { x: Some(x), row };
        Ok(safe(compute_expr(&self.root, &ctx)?))
    }

    pub fn eval_row(&self, row: &[f64]) -> Result<f64> {
        let ctx = EvalContext { x: None, row };
        Ok(safe(compute_expr(&self.root, &ctx)?))
    }

    /// C++17 expression used by the fitted-model backend. The caller supplies
    /// `row`, `row_len`, and (for elementwise maps) `x`, plus the `m_*`
    /// arithmetic helpers that mirror this evaluator's guards.
    pub(crate) fn to_cpp17(&self, elementwise: bool) -> Result<String> {
        let mut columns = Vec::new();
        let mut uses_x = false;
        references(&self.root, &mut columns, &mut uses_x);
        if uses_x && !elementwise {
            return Ok("0.0".to_string());
        }
        let body = expr_to_cpp17(&self.root, elementwise)?;
        // Failure belongs to the complete expression, including when a parent
        // such as min/sign would otherwise mask a NaN from a failed child.
        let body = format!(
            "([&](){{try{{return {body};}}catch(const MathDomainError&){{return 0.0;}}}}())"
        );
        Ok(match columns.iter().max() {
            Some(index) => format!("(row_len>{index}?{body}:0.0)"),
            None => body,
        })
    }

    /// Width requirements are checked by new admission; legacy callers retain
    /// the historical whole-expression zero fallback at runtime.
    pub fn validate_columns(&self, width: usize, elementwise: bool) -> Result<()> {
        let mut columns = Vec::new();
        let mut uses_x = false;
        references(&self.root, &mut columns, &mut uses_x);
        if uses_x && !elementwise {
            bail!("variable x is only valid in map expressions");
        }
        if let Some(index) = columns.into_iter().find(|index| *index >= width) {
            bail!("column index {index} out of bounds for width {width}");
        }
        Ok(())
    }
}

fn references(expr: &Expr, columns: &mut Vec<usize>, uses_x: &mut bool) {
    match expr {
        Expr::Var(Var::Col(index)) => columns.push(*index),
        Expr::Var(Var::X) => *uses_x = true,
        Expr::Unary(_, child) => references(child, columns, uses_x),
        Expr::Binary(_, a, b) => {
            references(a, columns, uses_x);
            references(b, columns, uses_x);
        }
        Expr::Call(_, args) => args.iter().for_each(|arg| references(arg, columns, uses_x)),
        _ => {}
    }
}

fn expr_to_cpp17(expr: &Expr, elementwise: bool) -> Result<String> {
    let rendered = match expr {
        Expr::Const(value) => format!("f64_from_bits(0x{:016x}ULL)", value.to_bits()),
        Expr::Var(Var::X) => {
            if !elementwise {
                bail!("variable x is only valid in map expressions");
            }
            "x".to_string()
        }
        Expr::Var(Var::Col(index)) => format!("row[{index}]"),
        Expr::Var(Var::RowMean) => "row_mean(row,row_len)".to_string(),
        Expr::Var(Var::RowSum) => "row_sum(row,row_len)".to_string(),
        Expr::Var(Var::RowL1) => "row_l1(row,row_len)".to_string(),
        Expr::Var(Var::RowL2) => "row_l2(row,row_len)".to_string(),
        Expr::Var(Var::RowMin) => "row_min(row,row_len)".to_string(),
        Expr::Var(Var::RowMax) => "row_max(row,row_len)".to_string(),
        Expr::Var(Var::RowStd) => "row_std(row,row_len)".to_string(),
        Expr::Unary(UnOp::Neg, child) => {
            format!("(-{})", expr_to_cpp17(child, elementwise)?)
        }
        Expr::Binary(op, left, right) => {
            let left = expr_to_cpp17(left, elementwise)?;
            let right = expr_to_cpp17(right, elementwise)?;
            match op {
                BinOp::Add => format!("({left}+{right})"),
                BinOp::Sub => format!("({left}-{right})"),
                BinOp::Mul => format!("({left}*{right})"),
                BinOp::Div => format!("m_div({left},{right})"),
                BinOp::Pow => format!("m_pow({left},m_clamp({right},-8.0,8.0))"),
            }
        }
        Expr::Call(function, args) => {
            let args = args
                .iter()
                .map(|arg| expr_to_cpp17(arg, elementwise))
                .collect::<Result<Vec<_>>>()?;
            match function {
                Func::Abs => format!("std::abs({})", args[0]),
                Func::Sqrt => format!("m_sqrt({})", args[0]),
                Func::Log1p => format!("m_log1p({})", args[0]),
                Func::Sin => format!("m_sin({})", args[0]),
                Func::Cos => format!("m_cos({})", args[0]),
                Func::Tanh => format!("m_tanh({})", args[0]),
                Func::Sigmoid => format!("m_sigmoid({})", args[0]),
                Func::Min => format!("std::fmin({},{})", args[0], args[1]),
                Func::Max => format!("std::fmax({},{})", args[0], args[1]),
                Func::Clip => format!("m_clip({},{},{})", args[0], args[1], args[2]),
                Func::Step => format!("(({})>({})?1.0:0.0)", args[0], args[1]),
                Func::Exp => format!("m_exp({})", args[0]),
                Func::Log => format!("m_log({})", args[0]),
                Func::Asinh => format!("m_asinh({})", args[0]),
                Func::Atan => format!("m_atan({})", args[0]),
                Func::Erf => format!("m_erf({})", args[0]),
                Func::Sign => format!("m_sign({})", args[0]),
                Func::Floor => format!("std::floor({})", args[0]),
                Func::Round => format!("m_round({})", args[0]),
            }
        }
    };
    Ok(rendered)
}

mod parser;
use parser::validate_arity;
mod evaluator;
use evaluator::{compute_expr, count_nodes, depth, safe, EvalContext};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_clip_bounds_are_a_structured_whole_expression_failure() {
        for source in [
            "clip(1,(-1)^0.5,(-1)^0.5)",
            "min(clip(1,(-1)^0.5,(-1)^0.5),7)+1",
        ] {
            let expr = MathExpr::parse(source).unwrap();
            let error = expr.eval_row(&[]).unwrap_err();
            assert_eq!(
                error.downcast_ref::<MathEvalError>(),
                Some(&MathEvalError::InvalidClipBounds)
            );
            assert_eq!(expr.eval_row(&[]).unwrap_or(0.0), 0.0);
        }
        // A single NaN bound retains Rust minNum/maxNum legacy semantics.
        assert_eq!(
            MathExpr::parse("clip(1,(-1)^0.5,3)")
                .unwrap()
                .eval_row(&[])
                .unwrap(),
            3.0
        );
    }

    #[test]
    fn evaluates_elementwise_expression() {
        let expr = MathExpr::parse("sigmoid(x) + c0").unwrap();
        let y = expr.eval_element(0.0, &[2.0, 3.0]).unwrap();
        assert!((y - 2.5).abs() < 1e-12);
    }

    #[test]
    fn evaluates_row_expression() {
        let expr = MathExpr::parse("clip(row_l2 + row_mean, 0, 10)").unwrap();
        let y = expr.eval_row(&[3.0, 4.0]).unwrap();
        assert!((y - 8.5).abs() < 1e-12);
    }

    #[test]
    fn step_is_a_hard_indicator() {
        let expr = MathExpr::parse("step(c0, c1)").unwrap();
        assert_eq!(expr.eval_row(&[2.0, 1.0]).unwrap(), 1.0);
        assert_eq!(expr.eval_row(&[1.0, 1.0]).unwrap(), 0.0);
        assert_eq!(expr.eval_row(&[0.0, 1.0]).unwrap(), 0.0);
        // NaN operand -> strict comparison false -> 0.0.
        assert_eq!(expr.eval_row(&[f64::NAN, 1.0]).unwrap(), 0.0);
        // Arity is enforced.
        assert!(MathExpr::parse("step(c0)").is_err());
    }

    #[test]
    fn rejects_unknown_and_deep_expressions() {
        assert!(MathExpr::parse("made_up(x)").is_err());
        let deep = "sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x)))))))))))))))))";
        assert!(MathExpr::parse(deep).is_err());
    }
}
