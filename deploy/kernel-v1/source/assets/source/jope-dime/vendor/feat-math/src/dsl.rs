use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{self, Display};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DslExpr {
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Stage {
    pub op: String,
    pub params: BTreeMap<String, ParamValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

impl ParamValue {
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            ParamValue::Int(v) if *v >= 0 => Some(*v as usize),
            ParamValue::Float(v) if *v >= 0.0 => Some(*v as usize),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            ParamValue::Int(v) if *v >= 0 => Some(*v as u64),
            ParamValue::Float(v) if *v >= 0.0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParamValue::Int(v) => Some(*v as f64),
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }
}

impl Stage {
    pub fn new(op: impl Into<String>) -> Self {
        Stage {
            op: op.into(),
            params: BTreeMap::new(),
        }
    }

    pub fn get_usize(&self, key: &str, default: usize) -> usize {
        self.params
            .get(key)
            .and_then(ParamValue::as_usize)
            .unwrap_or(default)
    }

    pub fn get_u64(&self, key: &str, default: u64) -> u64 {
        self.params
            .get(key)
            .and_then(ParamValue::as_u64)
            .unwrap_or(default)
    }

    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.params
            .get(key)
            .and_then(ParamValue::as_f64)
            .unwrap_or(default)
    }
}

impl Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{v}"),
            ParamValue::Float(v) => {
                if (v - v.round()).abs() < 1e-12 {
                    write!(f, "{:.1}", v)
                } else {
                    write!(f, "{v}")
                }
            }
            ParamValue::Bool(v) => write!(f, "{v}"),
            ParamValue::Str(v) => {
                if needs_quotes(v) {
                    write!(f, "\"{}\"", escape_quoted(v))
                } else {
                    write!(f, "{v}")
                }
            }
        }
    }
}

impl Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", self.op)?;
        for (key, value) in &self.params {
            write!(f, " {key}={value}")?;
        }
        write!(f, ")")
    }
}

impl Display for DslExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(pipe X")?;
        for stage in &self.stages {
            write!(f, " {stage}")?;
        }
        write!(f, ")")
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    L,
    R,
    Atom(String),
}

pub fn parse_dsl(src: &str) -> Result<DslExpr> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks: &toks,
        pos: 0,
    };
    let expr = p.parse_pipe()?;
    if p.pos != toks.len() {
        bail!("trailing tokens after DSL expression");
    }
    Ok(expr)
}

/// Parse the explicit nested portfolio form:
/// `(portfolio (member (pipe X ...))+ (blend kind=nnls folds=5 repeats=auto guard=clip))`.
/// Returns the member pipelines (order-preserving) and the blend stage
/// (defaulted to `(blend)` when absent). Structural caps (member count 1..12,
/// no nested portfolios) are enforced here so no caller can bypass them.
pub fn parse_portfolio(src: &str) -> Result<(Vec<DslExpr>, Stage)> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks: &toks,
        pos: 0,
    };
    p.expect_l()?;
    p.expect_atom("portfolio")?;
    let mut members: Vec<DslExpr> = Vec::new();
    let mut blend: Option<Stage> = None;
    while !p.peek_r() {
        p.expect_l()?;
        let tag = p
            .next_atom()
            .context("portfolio: expected member or blend")?;
        match tag.as_str() {
            "member" => {
                if blend.is_some() {
                    bail!("portfolio: members must precede the blend stage");
                }
                let pipe = p.parse_pipe()?;
                p.expect_r()?;
                members.push(pipe);
            }
            "blend" => {
                if blend.is_some() {
                    bail!("portfolio: duplicate blend stage");
                }
                let mut stage = Stage::new("blend");
                while !p.peek_r() {
                    let atom = p.next_atom().context("blend parameter must be an atom")?;
                    let (key, value) = atom
                        .split_once('=')
                        .with_context(|| format!("parameter '{atom}' must be key=value"))?;
                    stage
                        .params
                        .insert(key.to_string(), parse_param_value(value));
                }
                p.expect_r()?;
                blend = Some(stage);
            }
            other => bail!("portfolio: unexpected element '{other}' (expected member or blend)"),
        }
    }
    p.expect_r()?;
    if p.pos != toks.len() {
        bail!("trailing tokens after portfolio expression");
    }
    if members.is_empty() || members.len() > 12 {
        bail!(
            "portfolio: member count must be 1..12, got {}",
            members.len()
        );
    }
    for m in &members {
        if m.stages.iter().any(|s| s.op == "portfolio") {
            bail!("portfolio: members must not nest portfolios");
        }
    }
    Ok((members, blend.unwrap_or_else(|| Stage::new("blend"))))
}

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '(' => toks.push(Tok::L),
            ')' => toks.push(Tok::R),
            c if c.is_whitespace() => {}
            '"' => {
                let mut atom = String::new();
                let mut closed = false;
                while let Some(c) = chars.next() {
                    match c {
                        '\\' => {
                            let next =
                                chars.next().context("unterminated escape in quoted atom")?;
                            atom.push(next);
                        }
                        '"' => {
                            closed = true;
                            break;
                        }
                        _ => atom.push(c),
                    }
                }
                if !closed {
                    bail!("unterminated quoted atom");
                }
                toks.push(Tok::Atom(atom));
            }
            _ => {
                let mut atom = String::from(ch);
                let mut in_quote = false;
                while let Some(&c) = chars.peek() {
                    if !in_quote && (c.is_whitespace() || c == '(' || c == ')') {
                        break;
                    }
                    if c == '"' {
                        in_quote = !in_quote;
                    }
                    if c == '\\' && in_quote {
                        chars.next();
                        let next = chars.next().context("unterminated escape in quoted atom")?;
                        atom.push(next);
                        continue;
                    }
                    atom.push(c);
                    chars.next();
                }
                if in_quote {
                    bail!("unterminated quoted parameter");
                }
                toks.push(Tok::Atom(atom));
            }
        }
    }
    if toks.is_empty() {
        bail!("empty DSL expression");
    }
    Ok(toks)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn parse_pipe(&mut self) -> Result<DslExpr> {
        self.expect_l()?;
        self.expect_atom("pipe")?;
        self.expect_atom("X")?;
        let mut stages = Vec::new();
        while !self.peek_r() {
            stages.push(self.parse_stage()?);
        }
        self.expect_r()?;
        if stages.is_empty() {
            bail!("pipe must contain at least one stage");
        }
        Ok(DslExpr { stages })
    }

    fn parse_stage(&mut self) -> Result<Stage> {
        self.expect_l()?;
        let op = self.next_atom().context("stage missing operator")?;
        let mut stage = Stage::new(op);
        while !self.peek_r() {
            let atom = self.next_atom().context("stage parameter must be atom")?;
            let (key, value) = atom
                .split_once('=')
                .with_context(|| format!("parameter '{atom}' must be key=value"))?;
            stage
                .params
                .insert(key.to_string(), parse_param_value(value));
        }
        self.expect_r()?;
        Ok(stage)
    }

    fn expect_l(&mut self) -> Result<()> {
        match self.toks.get(self.pos) {
            Some(Tok::L) => {
                self.pos += 1;
                Ok(())
            }
            other => bail!("expected '(' at token {}, got {:?}", self.pos, other),
        }
    }

    fn expect_r(&mut self) -> Result<()> {
        match self.toks.get(self.pos) {
            Some(Tok::R) => {
                self.pos += 1;
                Ok(())
            }
            other => bail!("expected ')' at token {}, got {:?}", self.pos, other),
        }
    }

    fn expect_atom(&mut self, want: &str) -> Result<()> {
        let got = self.next_atom().context("expected atom")?;
        if got != want {
            bail!("expected atom '{want}', got '{got}'");
        }
        Ok(())
    }

    fn next_atom(&mut self) -> Option<String> {
        match self.toks.get(self.pos) {
            Some(Tok::Atom(atom)) => {
                self.pos += 1;
                Some(atom.clone())
            }
            _ => None,
        }
    }

    fn peek_r(&self) -> bool {
        matches!(self.toks.get(self.pos), Some(Tok::R))
    }
}

pub fn parse_param_value(value: &str) -> ParamValue {
    if let Some(unquoted) = unquote(value) {
        return ParamValue::Str(unquoted);
    }
    if value.eq_ignore_ascii_case("true") {
        return ParamValue::Bool(true);
    }
    if value.eq_ignore_ascii_case("false") {
        return ParamValue::Bool(false);
    }
    if let Ok(v) = value.parse::<i64>() {
        return ParamValue::Int(v);
    }
    if let Ok(v) = value.parse::<f64>() {
        return ParamValue::Float(v);
    }
    ParamValue::Str(value.to_string())
}

fn needs_quotes(value: &str) -> bool {
    value.is_empty()
        || value
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '(' | ')' | '"' | '='))
}

fn escape_quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out
}

fn unquote(value: &str) -> Option<String> {
    if !(value.starts_with('"') && value.ends_with('"') && value.len() >= 2) {
        return None;
    }
    let mut out = String::new();
    let mut chars = value[1..value.len() - 1].chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_round_trips_canonical_pipeline() {
        let src = "(pipe X (robust_scale) (fft_mag axis=features) (polar) (rff_cos n=32 seed=7) (cheb degree=4) (ridge alpha=1.0))";
        let expr = parse_dsl(src).unwrap();
        assert_eq!(expr.stages.len(), 6);
        assert_eq!(expr.stages[3].get_usize("n", 0), 32);
        assert_eq!(expr.to_string(), src);
    }

    #[test]
    fn parser_rejects_unsupported_shape() {
        assert!(parse_dsl("(robust_scale X)").is_err());
        assert!(parse_dsl("(pipe X)").is_err());
    }

    #[test]
    fn parser_round_trips_quoted_param_values() {
        let src = "(pipe X (map expr=\"sigmoid(col(0))\") (derive expr=\"row_mean + c0\"))";
        let expr = parse_dsl(src).unwrap();
        assert_eq!(
            expr.stages[0].params.get("expr"),
            Some(&ParamValue::Str("sigmoid(col(0))".to_string()))
        );
        assert_eq!(expr.to_string(), src);
    }
}
