//! Parser + code generators for the scalar math-expression grammar.
//!
//! This is a faithful re-implementation of the grammar accepted by
//! `crate::math_expr::MathExpr::parse` (that type keeps its AST
//! private, so we re-parse the source text). Both code generators mirror
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

use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;

const MAX_NODES: usize = 64;
const MAX_DEPTH: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub enum MathAst {
    Const(f64),
    Var(MathVar),
    Neg(Box<MathAst>),
    Bin(MathBinOp, Box<MathAst>, Box<MathAst>),
    Call(MathFunc, Vec<MathAst>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MathVar {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MathFunc {
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
        let toks = lex(src)?;
        let mut parser = Parser {
            toks: &toks,
            pos: 0,
        };
        let root = parser.parse_expr()?;
        parser.expect_end()?;
        let nodes = count_nodes(&root);
        if nodes > MAX_NODES {
            bail!("math expression has {nodes} nodes, max is {MAX_NODES}");
        }
        let d = depth(&root);
        if d > MAX_DEPTH {
            bail!("math expression depth {d} exceeds max {MAX_DEPTH}");
        }
        Ok(root)
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
            MathAst::Neg(a) => format!("(-{})", a.to_python()),
            MathAst::Bin(op, a, b) => {
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
            MathAst::Neg(a) => format!("(-{})", a.to_rust()),
            MathAst::Bin(op, a, b) => {
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
                    MathFunc::Clip => format!("m_clip({}, {}, {})", a[0], a[1], a[2]),
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
            MathAst::Neg(a) => format!("(-{})", a.to_cpp(portable)),
            MathAst::Bin(op, a, b) => {
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
        let MathAst::Bin(MathBinOp::Mul, a, b) = inner else {
            return None;
        };
        // The scale can sit on either side of the product.
        let diff = match (a.as_ref(), b.as_ref()) {
            (MathAst::Const(k), other) | (other, MathAst::Const(k)) if k.abs() >= HUGE_SCALE => {
                other
            }
            _ => return None,
        };
        if let MathAst::Bin(MathBinOp::Sub, lhs, rhs) = diff {
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
        MathAst::Neg(a) => collect(a, f),
        MathAst::Bin(op, a, b) => {
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
    fn parse_expr(&mut self) -> Result<MathAst> {
        self.parse_add()
    }

    fn parse_add(&mut self) -> Result<MathAst> {
        let mut expr = self.parse_mul()?;
        loop {
            match self.peek() {
                Tok::Plus => {
                    self.pos += 1;
                    expr =
                        MathAst::Bin(MathBinOp::Add, Box::new(expr), Box::new(self.parse_mul()?));
                }
                Tok::Minus => {
                    self.pos += 1;
                    expr =
                        MathAst::Bin(MathBinOp::Sub, Box::new(expr), Box::new(self.parse_mul()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_mul(&mut self) -> Result<MathAst> {
        let mut expr = self.parse_pow()?;
        loop {
            match self.peek() {
                Tok::Star => {
                    self.pos += 1;
                    expr =
                        MathAst::Bin(MathBinOp::Mul, Box::new(expr), Box::new(self.parse_pow()?));
                }
                Tok::Slash => {
                    self.pos += 1;
                    expr =
                        MathAst::Bin(MathBinOp::Div, Box::new(expr), Box::new(self.parse_pow()?));
                }
                _ => return Ok(expr),
            }
        }
    }

    fn parse_pow(&mut self) -> Result<MathAst> {
        let expr = self.parse_unary()?;
        if matches!(self.peek(), Tok::Caret) {
            self.pos += 1;
            Ok(MathAst::Bin(
                MathBinOp::Pow,
                Box::new(expr),
                Box::new(self.parse_pow()?),
            ))
        } else {
            Ok(expr)
        }
    }

    fn parse_unary(&mut self) -> Result<MathAst> {
        match self.peek() {
            Tok::Plus => {
                self.pos += 1;
                self.parse_unary()
            }
            Tok::Minus => {
                self.pos += 1;
                Ok(MathAst::Neg(Box::new(self.parse_unary()?)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<MathAst> {
        match self.peek().clone() {
            Tok::Num(v) => {
                self.pos += 1;
                Ok(MathAst::Const(v))
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
                    Ok(MathAst::Call(func, args))
                } else {
                    Ok(MathAst::Var(parse_var(&name)?))
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

    fn parse_args(&mut self) -> Result<Vec<MathAst>> {
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

fn parse_col_call(args: &[MathAst]) -> Result<MathAst> {
    if args.len() != 1 {
        bail!("col() expects exactly one argument");
    }
    match args[0] {
        MathAst::Const(v) if v >= 0.0 && (v - v.round()).abs() < 1e-12 => {
            Ok(MathAst::Var(MathVar::Col(v as usize)))
        }
        _ => bail!("col() expects a non-negative integer literal"),
    }
}

fn parse_var(name: &str) -> Result<MathVar> {
    match name {
        "x" => Ok(MathVar::X),
        "row_mean" | "mean" => Ok(MathVar::RowMean),
        "row_sum" => Ok(MathVar::RowSum),
        "row_l1" => Ok(MathVar::RowL1),
        "row_l2" | "row_norm" => Ok(MathVar::RowL2),
        "row_min" => Ok(MathVar::RowMin),
        "row_max" => Ok(MathVar::RowMax),
        "row_std" => Ok(MathVar::RowStd),
        _ if name.starts_with('c')
            && name.len() > 1
            && name[1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            Ok(MathVar::Col(name[1..].parse()?))
        }
        _ if name.starts_with("col")
            && name.len() > 3
            && name[3..].chars().all(|c| c.is_ascii_digit()) =>
        {
            Ok(MathVar::Col(name[3..].parse()?))
        }
        _ => bail!("unknown math variable '{name}'"),
    }
}

fn parse_func(name: &str) -> Result<MathFunc> {
    match name {
        "abs" => Ok(MathFunc::Abs),
        "sqrt" => Ok(MathFunc::Sqrt),
        "log1p" => Ok(MathFunc::Log1p),
        "sin" => Ok(MathFunc::Sin),
        "cos" => Ok(MathFunc::Cos),
        "tanh" => Ok(MathFunc::Tanh),
        "sigmoid" => Ok(MathFunc::Sigmoid),
        "min" => Ok(MathFunc::Min),
        "max" => Ok(MathFunc::Max),
        "clip" => Ok(MathFunc::Clip),
        "step" => Ok(MathFunc::Step),
        "exp" => Ok(MathFunc::Exp),
        "log" => Ok(MathFunc::Log),
        "asinh" => Ok(MathFunc::Asinh),
        "atan" => Ok(MathFunc::Atan),
        "erf" => Ok(MathFunc::Erf),
        "sign" => Ok(MathFunc::Sign),
        "floor" => Ok(MathFunc::Floor),
        "round" => Ok(MathFunc::Round),
        _ => bail!("unsupported math function '{name}'"),
    }
}

fn validate_arity(func: MathFunc, len: usize) -> Result<()> {
    let ok = match func {
        MathFunc::Abs
        | MathFunc::Sqrt
        | MathFunc::Log1p
        | MathFunc::Sin
        | MathFunc::Cos
        | MathFunc::Tanh
        | MathFunc::Sigmoid
        | MathFunc::Exp
        | MathFunc::Log
        | MathFunc::Asinh
        | MathFunc::Atan
        | MathFunc::Erf
        | MathFunc::Sign
        | MathFunc::Floor
        | MathFunc::Round => len == 1,
        MathFunc::Min | MathFunc::Max | MathFunc::Step => len == 2,
        MathFunc::Clip => len == 3,
    };
    if ok {
        Ok(())
    } else {
        bail!("{func:?} received {len} arguments")
    }
}

fn count_nodes(expr: &MathAst) -> usize {
    match expr {
        MathAst::Const(_) | MathAst::Var(_) => 1,
        MathAst::Neg(child) => 1 + count_nodes(child),
        MathAst::Bin(_, left, right) => 1 + count_nodes(left) + count_nodes(right),
        MathAst::Call(_, args) => 1 + args.iter().map(count_nodes).sum::<usize>(),
    }
}

fn depth(expr: &MathAst) -> usize {
    match expr {
        MathAst::Const(_) | MathAst::Var(_) => 1,
        MathAst::Neg(child) => 1 + depth(child),
        MathAst::Bin(_, left, right) => 1 + depth(left).max(depth(right)),
        MathAst::Call(_, args) => 1 + args.iter().map(depth).max().unwrap_or(0),
    }
}

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
