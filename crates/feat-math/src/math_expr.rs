use anyhow::{bail, Context, Result};
mod compiled;
mod evolution;
mod exact;
mod fault;
pub use compiled::CompiledMathExpr;
pub use evolution::{ExpressionEdit, ExpressionScope};
pub use fault::{FaultClass, FaultSet, FaultSeverity};

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

    pub fn parse(src: &str) -> Result<Self> {
        let toks = lex(src)?;
        let mut parser = Parser {
            toks: &toks,
            pos: 0,
        };
        let root = parser.parse_expr()?;
        parser.expect_end()?;
        let nodes = count_nodes(&root);
        let depth = depth(&root);
        if nodes > MAX_NODES {
            bail!("math expression has {nodes} nodes, max is {MAX_NODES}");
        }
        if depth > MAX_DEPTH {
            bail!("math expression depth {depth} exceeds max {MAX_DEPTH}");
        }
        Ok(MathExpr { root })
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

fn lex(src: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let mut chars = src.char_indices().peekable();
    while let Some((idx, ch)) = chars.peek().copied() {
        match ch {
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' => {
                toks.push(Tok::Plus);
                chars.next();
            }
            '-' => {
                toks.push(Tok::Minus);
                chars.next();
            }
            '*' => {
                toks.push(Tok::Star);
                chars.next();
            }
            '/' => {
                toks.push(Tok::Slash);
                chars.next();
            }
            '^' => {
                toks.push(Tok::Caret);
                chars.next();
            }
            '(' => {
                toks.push(Tok::LParen);
                chars.next();
            }
            ')' => {
                toks.push(Tok::RParen);
                chars.next();
            }
            ',' => {
                toks.push(Tok::Comma);
                chars.next();
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = idx;
                let mut end = idx + ch.len_utf8();
                chars.next();
                while let Some((j, c)) = chars.peek().copied() {
                    if c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || ((c == '+' || c == '-') && j > start && exponent_sign_allowed(src, j))
                    {
                        end = j + c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                let raw = &src[start..end];
                let value = raw
                    .parse::<f64>()
                    .with_context(|| format!("invalid numeric literal '{raw}'"))?;
                if !value.is_finite() {
                    bail!("non-finite numeric literal '{raw}'");
                }
                toks.push(Tok::Num(value));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = idx;
                let mut end = idx + ch.len_utf8();
                chars.next();
                while let Some((j, c)) = chars.peek().copied() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        end = j + c.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Ident(src[start..end].to_ascii_lowercase()));
            }
            _ => bail!("unexpected character '{ch}' in math expression"),
        }
    }
    toks.push(Tok::End);
    Ok(toks)
}

fn exponent_sign_allowed(src: &str, idx: usize) -> bool {
    src[..idx]
        .chars()
        .last()
        .map(|c| c == 'e' || c == 'E')
        .unwrap_or(false)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<Expr> {
        let mut expr = self.parse_mul()?;
        loop {
            match self.peek() {
                Tok::Plus => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Add, Box::new(expr), Box::new(self.parse_mul()?));
                }
                Tok::Minus => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Sub, Box::new(expr), Box::new(self.parse_mul()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_mul(&mut self) -> Result<Expr> {
        let mut expr = self.parse_pow()?;
        loop {
            match self.peek() {
                Tok::Star => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Mul, Box::new(expr), Box::new(self.parse_pow()?));
                }
                Tok::Slash => {
                    self.pos += 1;
                    expr = Expr::Binary(BinOp::Div, Box::new(expr), Box::new(self.parse_pow()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_pow(&mut self) -> Result<Expr> {
        let expr = self.parse_unary()?;
        if matches!(self.peek(), Tok::Caret) {
            self.pos += 1;
            Ok(Expr::Binary(
                BinOp::Pow,
                Box::new(expr),
                Box::new(self.parse_pow()?),
            ))
        } else {
            Ok(expr)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Tok::Plus => {
                self.pos += 1;
                self.parse_unary()
            }
            Tok::Minus => {
                self.pos += 1;
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Tok::Num(v) => {
                self.pos += 1;
                Ok(Expr::Const(v))
            }
            Tok::Ident(name) => {
                self.pos += 1;
                if matches!(self.peek(), Tok::LParen) {
                    self.pos += 1;
                    let args = self.parse_args()?;
                    if name == "col" {
                        return parse_col_call(&args);
                    }
                    let func = parse_func(&name)?;
                    validate_arity(func, args.len())?;
                    Ok(Expr::Call(func, args))
                } else {
                    Ok(Expr::Var(parse_var(&name)?))
                }
            }
            Tok::LParen => {
                self.pos += 1;
                let expr = self.parse_expr()?;
                self.expect_rparen()?;
                Ok(expr)
            }
            other => bail!("expected expression, got {:?}", other),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        if matches!(self.peek(), Tok::RParen) {
            self.pos += 1;
            return Ok(args);
        }
        loop {
            args.push(self.parse_expr()?);
            match self.peek() {
                Tok::Comma => self.pos += 1,
                Tok::RParen => {
                    self.pos += 1;
                    return Ok(args);
                }
                other => bail!("expected ',' or ')', got {:?}", other),
            }
        }
    }

    fn expect_rparen(&mut self) -> Result<()> {
        match self.peek() {
            Tok::RParen => {
                self.pos += 1;
                Ok(())
            }
            other => bail!("expected ')', got {:?}", other),
        }
    }

    fn expect_end(&self) -> Result<()> {
        if matches!(self.peek(), Tok::End) {
            Ok(())
        } else {
            bail!("trailing tokens in math expression")
        }
    }

    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::End)
    }
}

fn parse_col_call(args: &[Expr]) -> Result<Expr> {
    if args.len() != 1 {
        bail!("col() expects exactly one argument");
    }
    match args[0] {
        Expr::Const(v) if v >= 0.0 && (v - v.round()).abs() < 1e-12 => {
            Ok(Expr::Var(Var::Col(v as usize)))
        }
        _ => bail!("col() expects a non-negative integer literal"),
    }
}

fn parse_var(name: &str) -> Result<Var> {
    match name {
        "x" => Ok(Var::X),
        "row_mean" | "mean" => Ok(Var::RowMean),
        "row_sum" => Ok(Var::RowSum),
        "row_l1" => Ok(Var::RowL1),
        "row_l2" | "row_norm" => Ok(Var::RowL2),
        "row_min" => Ok(Var::RowMin),
        "row_max" => Ok(Var::RowMax),
        "row_std" => Ok(Var::RowStd),
        _ if name.starts_with('c') && name[1..].chars().all(|c| c.is_ascii_digit()) => {
            Ok(Var::Col(name[1..].parse()?))
        }
        _ if name.starts_with("col") && name[3..].chars().all(|c| c.is_ascii_digit()) => {
            Ok(Var::Col(name[3..].parse()?))
        }
        _ => bail!("unknown math variable '{name}'"),
    }
}

fn parse_func(name: &str) -> Result<Func> {
    match name {
        "abs" => Ok(Func::Abs),
        "sqrt" => Ok(Func::Sqrt),
        "log1p" => Ok(Func::Log1p),
        "sin" => Ok(Func::Sin),
        "cos" => Ok(Func::Cos),
        "tanh" => Ok(Func::Tanh),
        "sigmoid" => Ok(Func::Sigmoid),
        "min" => Ok(Func::Min),
        "max" => Ok(Func::Max),
        "clip" => Ok(Func::Clip),
        // Hard indicator: step(a, c) = 1 when a > c else 0 (an honest,
        // finite-valued threshold, replacing the clip(...*1e15,...) idiom).
        "step" => Ok(Func::Step),
        "exp" => Ok(Func::Exp),
        "log" => Ok(Func::Log),
        "asinh" => Ok(Func::Asinh),
        "atan" => Ok(Func::Atan),
        "erf" => Ok(Func::Erf),
        "sign" => Ok(Func::Sign),
        "floor" => Ok(Func::Floor),
        "round" => Ok(Func::Round),
        _ => bail!("unsupported math function '{name}'"),
    }
}

fn validate_arity(func: Func, len: usize) -> Result<()> {
    let ok = match func {
        Func::Abs
        | Func::Sqrt
        | Func::Log1p
        | Func::Sin
        | Func::Cos
        | Func::Tanh
        | Func::Sigmoid
        | Func::Exp
        | Func::Log
        | Func::Asinh
        | Func::Atan
        | Func::Erf
        | Func::Sign
        | Func::Floor
        | Func::Round => len == 1,
        Func::Min | Func::Max | Func::Step => len == 2,
        Func::Clip => len == 3,
    };
    if ok {
        Ok(())
    } else {
        bail!("{func:?} received {len} arguments")
    }
}

struct EvalContext<'a> {
    x: Option<f64>,
    row: &'a [f64],
}

fn compute_expr(expr: &Expr, ctx: &EvalContext<'_>) -> Result<f64> {
    match expr {
        Expr::Const(v) => Ok(*v),
        Expr::Var(var) => eval_var(*var, ctx),
        Expr::Unary(UnOp::Neg, child) => Ok(-compute_expr(child, ctx)?),
        Expr::Binary(op, left, right) => {
            let a = compute_expr(left, ctx)?;
            let b = compute_expr(right, ctx)?;
            Ok(compute_binary(*op, a, b))
        }
        Expr::Call(func, args) => {
            validate_arity(*func, args.len())?;
            let mut values = [0.0; 3];
            for (slot, arg) in values.iter_mut().zip(args) {
                *slot = compute_expr(arg, ctx)?;
            }
            compute_func(*func, values)
        }
    }
}

fn compute_binary(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            if b.abs() < 1e-12 {
                0.0
            } else {
                a / b
            }
        }
        BinOp::Pow => a.powf(b.clamp(-8.0, 8.0)),
    }
}

fn compute_func(func: Func, values: [f64; 3]) -> Result<f64> {
    Ok(match func {
        Func::Abs => values[0].abs(),
        Func::Sqrt => values[0].max(0.0).sqrt(),
        Func::Log1p => {
            if values[0] <= -1.0 {
                0.0
            } else {
                values[0].ln_1p()
            }
        }
        Func::Sin => values[0].sin(),
        Func::Cos => values[0].cos(),
        Func::Tanh => values[0].tanh(),
        Func::Sigmoid => 1.0 / (1.0 + (-values[0].clamp(-30.0, 30.0)).exp()),
        Func::Min => values[0].min(values[1]),
        Func::Max => values[0].max(values[1]),
        Func::Clip => {
            let lo = values[1].min(values[2]);
            let hi = values[1].max(values[2]);
            if lo.is_nan() || hi.is_nan() {
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
        Func::Exp => values[0].clamp(-30.0, 30.0).exp(),
        // Natural log, 0.0 at or below 1e-300 (mirrors log1p's
        // invalid-argument policy).
        Func::Log => {
            if values[0] <= 1e-300 {
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
    match var {
        Var::X => ctx.x.context("variable x is only valid in map expressions"),
        Var::Col(idx) => ctx
            .row
            .get(idx)
            .copied()
            .with_context(|| format!("column index {idx} out of bounds")),
        Var::RowMean => Ok(mean(ctx.row)),
        Var::RowSum => Ok(ctx.row.iter().copied().sum()),
        Var::RowL1 => Ok(ctx.row.iter().map(|v| v.abs()).sum()),
        Var::RowL2 => Ok(ctx.row.iter().map(|v| v * v).sum::<f64>().sqrt()),
        Var::RowMin => Ok(ctx.row.iter().copied().fold(f64::INFINITY, f64::min)),
        Var::RowMax => Ok(ctx.row.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        Var::RowStd => {
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

fn mean(row: &[f64]) -> f64 {
    row.iter().copied().sum::<f64>() / row.len().max(1) as f64
}

fn safe(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-MAX_ABS, MAX_ABS)
    } else {
        0.0
    }
}

fn count_nodes(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Unary(_, child) => 1 + count_nodes(child),
        Expr::Binary(_, left, right) => 1 + count_nodes(left) + count_nodes(right),
        Expr::Call(_, args) => 1 + args.iter().map(count_nodes).sum::<usize>(),
    }
}

fn depth(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) | Expr::Var(_) => 1,
        Expr::Unary(_, child) => 1 + depth(child),
        Expr::Binary(_, left, right) => 1 + depth(left).max(depth(right)),
        Expr::Call(_, args) => 1 + args.iter().map(depth).max().unwrap_or(0),
    }
}

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
