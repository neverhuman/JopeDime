//! Port of the reference policy action vocabulary and masking automaton.
use anyhow::{ensure, Context, Result};
use feat_math::dsl::{parse_dsl, parse_param_value, DslExpr, ParamValue, Stage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Slot {
    pub param: String,
    pub kind: String,
    pub val_ids: Vec<usize>,
    pub bin_texts: Vec<String>,
    pub absent_id: Option<usize>,
    pub is_expr: bool,
    pub required: bool,
    pub default: serde_json::Value,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Op {
    pub token_id: usize,
    pub engine: String,
    pub name: String,
    pub role: String,
    pub slots: Vec<Slot>,
    pub allow_append: bool,
    pub takes_expr: bool,
    pub generate: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Expr {
    pub var_map_ids: Vec<usize>,
    pub var_col_ids: Vec<usize>,
    pub var_common_ids: Vec<usize>,
    pub const_ids: Vec<usize>,
    pub const_values: Vec<f64>,
    pub func_arity: BTreeMap<usize, usize>,
    pub end_id: usize,
    pub symbol: BTreeMap<usize, String>,
    pub is_binop: BTreeMap<usize, bool>,
    pub max_depth: usize,
    pub max_nodes: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vocab {
    pub tokens: Vec<String>,
    pub engines: Vec<String>,
    pub engine_ids: BTreeMap<String, usize>,
    pub ops: BTreeMap<usize, Op>,
    pub engine_transforms: BTreeMap<String, Vec<usize>>,
    pub engine_heads: BTreeMap<String, Vec<usize>>,
    pub append_ids: [usize; 3],
    pub expr: Expr,
    pub max_transforms: BTreeMap<String, usize>,
    pub max_expr_stages: BTreeMap<String, usize>,
    pub tasks: Vec<String>,
    pub task_ids: BTreeMap<String, usize>,
    pub head_tasks: BTreeMap<usize, Vec<String>>,
}
#[derive(Clone, Debug, PartialEq)]
enum Phase {
    Engine,
    Task,
    Stage,
    Val,
    Append,
    Expr,
    ExprEnd,
    Eos,
    Done,
}
#[derive(Clone, Debug)]
pub struct State {
    phase: Phase,
    engine: String,
    task: String,
    stages: usize,
    expr_stages: usize,
    op: Option<usize>,
    slot: usize,
    stack: Vec<(usize, usize)>,
    nodes: usize,
    pub sampling: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            phase: Phase::Engine,
            engine: String::new(),
            task: String::new(),
            stages: 0,
            expr_stages: 0,
            op: None,
            slot: 0,
            stack: vec![],
            nodes: 0,
            sampling: true,
        }
    }
}
impl State {
    pub fn done(&self) -> bool {
        self.phase == Phase::Done
    }
    fn expr_legal(&self, v: &Vocab, stack: &[(usize, usize)], nodes: usize) -> Vec<usize> {
        let table = &v.expr;
        let mut out = table.var_common_ids.clone();
        out.extend(&table.const_ids);
        if self.engine != "newmath" || !self.sampling {
            out.extend(&table.var_col_ids);
        }
        if self.op.is_some_and(|id| v.ops[&id].name == "map") {
            out.extend(&table.var_map_ids);
        }
        let pending = stack.iter().map(|x| x.0).sum::<usize>();
        let depth = stack.last().map(|x| x.1).unwrap_or(1);
        out.extend(
            table
                .func_arity
                .iter()
                .filter(|(_, arity)| {
                    depth + 1 <= table.max_depth
                        && nodes + 1 + (pending - 1 + **arity) <= table.max_nodes
                })
                .map(|(&id, _)| id),
        );
        out
    }
    pub fn legal(&self, v: &Vocab) -> Vec<usize> {
        let mut out = match self.phase {
            Phase::Engine => v.engines.iter().map(|e| v.engine_ids[e]).collect(),
            Phase::Task => v.tasks.iter().map(|t| v.task_ids[t]).collect(),
            Phase::Stage => {
                let mut ids = Vec::new();
                if self.stages < v.max_transforms[&self.engine] {
                    ids.extend(
                        v.engine_transforms[&self.engine]
                            .iter()
                            .copied()
                            .filter(|i| {
                                let o = &v.ops[i];
                                (!o.takes_expr
                                    || self.expr_stages < v.max_expr_stages[&self.engine])
                                    && (!self.sampling || o.generate)
                            }),
                    );
                }
                ids.extend(v.engine_heads[&self.engine].iter().copied().filter(|i| {
                    v.head_tasks
                        .get(i)
                        .is_none_or(|tasks| tasks.contains(&self.task))
                        && (!self.sampling || v.ops[i].generate)
                }));
                ids
            }
            Phase::Val => {
                let slot = &v.ops[&self.op.unwrap()].slots[self.slot];
                let mut ids = slot.val_ids.clone();
                if slot.is_expr {
                    ids.extend(
                        self.expr_legal(v, &[(1, 1)], 0)
                            .into_iter()
                            .filter(|id| !v.expr.const_ids.contains(id)),
                    );
                } else if let Some(id) = slot.absent_id {
                    ids.push(id);
                }
                ids
            }
            Phase::Append => v.append_ids.to_vec(),
            Phase::Expr => self.expr_legal(v, &self.stack, self.nodes),
            Phase::ExprEnd => vec![v.expr.end_id],
            Phase::Eos => vec![2],
            Phase::Done => vec![],
        };
        out.sort_unstable();
        out.dedup();
        out
    }
    pub fn advance(&mut self, v: &Vocab, token: usize) -> Result<()> {
        ensure!(
            self.legal(v).contains(&token),
            "illegal token {token} in {:?}",
            self.phase
        );
        match self.phase {
            Phase::Engine => {
                self.engine = v
                    .engine_ids
                    .iter()
                    .find(|(_, id)| **id == token)
                    .unwrap()
                    .0
                    .clone();
                self.phase = Phase::Task;
            }
            Phase::Task => {
                self.task = v
                    .task_ids
                    .iter()
                    .find(|(_, id)| **id == token)
                    .unwrap()
                    .0
                    .clone();
                self.phase = Phase::Stage;
            }
            Phase::Stage => {
                self.op = Some(token);
                self.slot = 0;
                if v.ops[&token].takes_expr {
                    self.expr_stages += 1;
                }
                self.enter(v);
            }
            Phase::Val => {
                let slot = &v.ops[&self.op.unwrap()].slots[self.slot];
                if slot.is_expr && !slot.val_ids.contains(&token) {
                    self.stack = vec![(1, 1)];
                    self.nodes = 0;
                    self.consume_expr(v, token);
                } else {
                    self.slot += 1;
                    self.enter(v);
                }
            }
            Phase::Append => self.finish(v),
            Phase::Expr => self.consume_expr(v, token),
            Phase::ExprEnd => {
                self.slot += 1;
                self.enter(v);
            }
            Phase::Eos => self.phase = Phase::Done,
            Phase::Done => unreachable!(),
        };
        Ok(())
    }
    fn consume_expr(&mut self, v: &Vocab, token: usize) {
        let (children, depth) = self.stack.pop().unwrap();
        if children > 1 {
            self.stack.push((children - 1, depth));
        }
        self.nodes += 1;
        if let Some(&arity) = v.expr.func_arity.get(&token) {
            self.stack.push((arity, depth + 1));
        }
        self.phase = if self.stack.is_empty() {
            Phase::ExprEnd
        } else {
            Phase::Expr
        };
    }
    fn enter(&mut self, v: &Vocab) {
        let o = &v.ops[&self.op.unwrap()];
        if self.slot < o.slots.len() {
            self.phase = Phase::Val;
        } else if o.allow_append {
            self.phase = Phase::Append;
        } else {
            self.finish(v);
        }
    }
    fn finish(&mut self, v: &Vocab) {
        if v.ops[&self.op.unwrap()].role == "head" {
            self.phase = Phase::Eos;
        } else {
            self.stages += 1;
            self.phase = Phase::Stage;
        }
        self.op = None;
        self.slot = 0;
    }
}
impl Vocab {
    pub fn replay(&self, tokens: &[usize]) -> Result<Vec<Vec<usize>>> {
        ensure!(tokens.first() == Some(&1), "BOS missing");
        let mut state = State {
            sampling: false,
            ..Default::default()
        };
        let mut out = Vec::new();
        for &token in &tokens[1..] {
            out.push(state.legal(self));
            state.advance(self, token)?;
        }
        ensure!(state.done(), "unterminated program");
        Ok(out)
    }
    pub fn decode(&self, tokens: &[usize]) -> Result<String> {
        self.replay(tokens)?;
        let mut i = 3;
        let mut stages = Vec::new();
        while tokens.get(i) != Some(&2) {
            let op = self
                .ops
                .get(tokens.get(i).context("operator absent")?)
                .context("unknown operator")?;
            i += 1;
            let mut stage = Stage::new(&op.name);
            for slot in &op.slots {
                let id = *tokens.get(i).context("parameter absent")?;
                i += 1;
                if slot.is_expr && !slot.val_ids.contains(&id) {
                    let start = i - 1;
                    while tokens.get(i) != Some(&self.expr.end_id) {
                        ensure!(i < tokens.len(), "unterminated expression");
                        i += 1;
                    }
                    let mut pos = 0;
                    let text = render_expr(&tokens[start..i], &self.expr, &mut pos)?;
                    ensure!(pos == i - start, "trailing expression tokens");
                    i += 1;
                    stage
                        .params
                        .insert(slot.param.clone(), ParamValue::Str(text));
                } else if slot.absent_id != Some(id) {
                    let position = slot
                        .val_ids
                        .iter()
                        .position(|&x| x == id)
                        .context("unknown value")?;
                    stage.params.insert(
                        slot.param.clone(),
                        if slot.is_expr {
                            ParamValue::Str(slot.bin_texts[position].clone())
                        } else {
                            parse_param_value(&slot.bin_texts[position])
                        },
                    );
                }
            }
            if op.allow_append {
                let id = tokens[i];
                i += 1;
                if id != self.append_ids[0] {
                    stage
                        .params
                        .insert("append".into(), ParamValue::Bool(id == self.append_ids[2]));
                }
            }
            stages.push(stage);
            if op.role == "head" {
                break;
            }
        }
        Ok(DslExpr { stages }.to_string())
    }
    /// Project a proposal BEFORE evaluation. The resulting program needs its own
    /// score; encode() deliberately never makes this lossy conversion.
    pub fn project_proposal(&self, genome: &str) -> Result<String> {
        let mut pipeline = parse_dsl(genome)?;
        for stage in &mut pipeline.stages {
            let op = self
                .ops
                .values()
                .find(|o| o.engine == "featmath" && o.name == stage.op)
                .context("operator outside vocabulary")?;
            stage.params.retain(|k, _| {
                (k == "append" && op.allow_append) || op.slots.iter().any(|s| s.param == *k)
            });
            for slot in &op.slots {
                let Some(value) = stage.params.get(&slot.param) else {
                    continue;
                };
                if slot.is_expr {
                    if let ParamValue::Str(s) = value {
                        if slot.bin_texts.contains(s) || parse_expr_tokens(s, &self.expr).is_ok() {
                            continue;
                        }
                    }
                }
                let text = match value {
                    ParamValue::Float(v) => v.to_string(),
                    ParamValue::Int(v) => v.to_string(),
                    ParamValue::Bool(v) => v.to_string(),
                    ParamValue::Str(s) => s.clone(),
                };
                if slot.bin_texts.iter().any(|s| {
                    s == &text
                        || s.parse::<f64>()
                            .ok()
                            .zip(text.parse::<f64>().ok())
                            .is_some_and(|(a, b)| a == b)
                }) {
                    continue;
                }
                let chosen = if let Ok(number) = text.parse::<f64>() {
                    slot.bin_texts
                        .iter()
                        .filter_map(|s| s.parse::<f64>().ok().map(|n| ((n - number).abs(), s)))
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|(_, s)| s)
                } else {
                    None
                }
                .or_else(|| slot.bin_texts.first())
                .context("parameter has no proposal bins")?;
                stage.params.insert(
                    slot.param.clone(),
                    if slot.is_expr {
                        ParamValue::Str(chosen.clone())
                    } else {
                        parse_param_value(chosen)
                    },
                );
            }
        }
        let tokens = self.encode(&pipeline.to_string())?;
        self.decode(&tokens)
    }
    pub fn task_id(&self, task: &str) -> Result<usize> {
        crate::task_contract::require_task(task)?;
        Ok(*self
            .task_ids
            .get(task)
            .with_context(|| format!("{task} absent"))?)
    }
    pub fn encode(&self, genome: &str) -> Result<Vec<usize>> {
        self.encode_for_task(genome, "regression")
    }
    pub fn encode_for_task(&self, genome: &str, task: &str) -> Result<Vec<usize>> {
        let pipeline = parse_dsl(genome)?;
        let mut tokens = vec![
            1,
            *self.engine_ids.get("featmath").context("featmath absent")?,
            self.task_id(task)?,
        ];
        for stage in &pipeline.stages {
            let op = self
                .ops
                .values()
                .find(|o| o.engine == "featmath" && o.name == stage.op)
                .context("operator outside frozen vocabulary")?;
            ensure!(
                stage
                    .params
                    .keys()
                    .all(|k| k == "append" || op.slots.iter().any(|s| &s.param == k)),
                "parameter outside frozen vocabulary"
            );
            tokens.push(op.token_id);
            for slot in &op.slots {
                let value = stage.params.get(&slot.param);
                if slot.is_expr {
                    let text = match value {
                        Some(ParamValue::Str(s)) => s.clone(),
                        None => slot
                            .default
                            .as_str()
                            .context("expression default missing")?
                            .into(),
                        _ => anyhow::bail!("expression must be text"),
                    };
                    if let Some(i) = slot.bin_texts.iter().position(|s| s == &text) {
                        tokens.push(slot.val_ids[i]);
                    } else {
                        tokens.extend(parse_expr_tokens(&text, &self.expr)?);
                        tokens.push(self.expr.end_id);
                    }
                    continue;
                }
                if let Some(value) = value {
                    let position=slot.bin_texts.iter().position(|s|{let bin=parse_param_value(s);if let(Some(a),Some(b))=(value.as_f64(),bin.as_f64()){a==b}else{&bin==value}}).context("exact parameter has no frozen token; scores cannot be attached to a snapped program")?;
                    tokens.push(slot.val_ids[position]);
                } else {
                    tokens.push(slot.absent_id.context("required parameter absent")?);
                }
            }
            if op.allow_append {
                let append = match stage.params.get("append") {
                    None => 0,
                    Some(ParamValue::Bool(false)) => 1,
                    Some(ParamValue::Bool(true)) => 2,
                    _ => anyhow::bail!("invalid append flag"),
                };
                tokens.push(self.append_ids[append]);
            }
        }
        tokens.push(2);
        ensure!(
            tokens.len() <= 128,
            "program exceeds policy sequence length"
        );
        self.replay(&tokens)?;
        let replayed = parse_dsl(&self.decode(&tokens)?)?;
        ensure!(
            replayed.stages.len() == pipeline.stages.len(),
            "round trip stage count changed"
        );
        for (a, b) in pipeline.stages.iter().zip(&replayed.stages) {
            ensure!(
                a.op == b.op && a.params.keys().eq(b.params.keys()),
                "round trip changed operator/parameters"
            );
            for (k, av) in &a.params {
                let bv = &b.params[k];
                let same = if let (Some(x), Some(y)) = (av.as_f64(), bv.as_f64()) {
                    x == y
                } else if k == "expr" {
                    match (av, bv) {
                        (ParamValue::Str(x), ParamValue::Str(y)) => {
                            parse_expr_tokens(x, &self.expr)? == parse_expr_tokens(y, &self.expr)?
                        }
                        _ => false,
                    }
                } else {
                    av == bv
                };
                ensure!(same, "round trip changed parameter value");
            }
        }
        Ok(tokens)
    }
}
fn render_expr(tokens: &[usize], table: &Expr, pos: &mut usize) -> Result<String> {
    let id = *tokens.get(*pos).context("expression token absent")?;
    *pos += 1;
    let symbol = table.symbol.get(&id).context("unknown expression symbol")?;
    if let Some(&arity) = table.func_arity.get(&id) {
        let args = (0..arity)
            .map(|_| render_expr(tokens, table, pos))
            .collect::<Result<Vec<_>>>()?;
        Ok(if table.is_binop.get(&id) == Some(&true) {
            format!("({} {symbol} {})", args[0], args[1])
        } else {
            format!("{symbol}({})", args.join(", "))
        })
    } else {
        Ok(symbol.clone())
    }
}
fn parse_expr_tokens(text: &str, table: &Expr) -> Result<Vec<usize>> {
    #[derive(Clone)]
    enum Token {
        Name(String),
        Number(f64),
        Symbol(char),
    }
    let chars = text.chars().collect::<Vec<_>>();
    let mut lexical = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == '.' {
            let start = i;
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || matches!(chars[i], '.' | 'e' | 'E')
                    || matches!(chars[i], '+' | '-') && matches!(chars[i - 1], 'e' | 'E'))
            {
                i += 1;
            }
            lexical.push(Token::Number(
                chars[start..i].iter().collect::<String>().parse()?,
            ));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            lexical.push(Token::Name(chars[start..i].iter().collect()));
        } else {
            ensure!("()+-*/,".contains(c), "unsupported expression character");
            lexical.push(Token::Symbol(c));
            i += 1;
        }
    }
    struct Parser<'a> {
        tokens: &'a [Token],
        at: usize,
        table: &'a Expr,
    }
    impl Parser<'_> {
        fn symbol(&self, s: char) -> bool {
            matches!(self.tokens.get(self.at),Some(Token::Symbol(c))if *c==s)
        }
        fn need(&mut self, s: char) -> Result<()> {
            ensure!(self.symbol(s), "expected {s} in expression");
            self.at += 1;
            Ok(())
        }
        fn number(&self, n: f64) -> Result<usize> {
            self.table
                .const_values
                .iter()
                .position(|v| (*v - n).abs() < 1e-12)
                .map(|i| self.table.const_ids[i])
                .context("expression constant outside frozen vocabulary")
        }
        fn id(&self, s: &str, arity: Option<usize>) -> Result<usize> {
            self.table
                .symbol
                .iter()
                .find(|(id, sym)| {
                    sym.as_str() == s && self.table.func_arity.get(id).copied() == arity
                })
                .map(|(&id, _)| id)
                .context("expression symbol outside frozen vocabulary")
        }
        fn read(&mut self, min: u8) -> Result<Vec<usize>> {
            let token = self
                .tokens
                .get(self.at)
                .cloned()
                .context("expression ended early")?;
            self.at += 1;
            let mut left = match token {
                Token::Number(n) => vec![self.number(n)?],
                Token::Symbol('(') => {
                    let x = self.read(0)?;
                    self.need(')')?;
                    x
                }
                Token::Symbol('-') => {
                    if let Some(Token::Number(n)) = self.tokens.get(self.at) {
                        let n = *n;
                        self.at += 1;
                        vec![self.number(-n)?]
                    } else {
                        let mut out = vec![self.id("-", Some(2))?, self.number(0.)?];
                        out.extend(self.read(3)?);
                        out
                    }
                }
                Token::Name(name) => {
                    if self.symbol('(') {
                        self.at += 1;
                        if name == "col" {
                            let n = if let Some(Token::Number(n)) = self.tokens.get(self.at) {
                                *n
                            } else {
                                anyhow::bail!("column index missing")
                            };
                            self.at += 1;
                            self.need(')')?;
                            vec![self.id(&format!("col({n})"), None)?]
                        } else {
                            let mut args = vec![self.read(0)?];
                            while self.symbol(',') {
                                self.at += 1;
                                args.push(self.read(0)?);
                            }
                            self.need(')')?;
                            let mut out = vec![self.id(&name, Some(args.len()))?];
                            for arg in args {
                                out.extend(arg);
                            }
                            out
                        }
                    } else {
                        vec![self.id(&name, None)?]
                    }
                }
                _ => anyhow::bail!("invalid expression prefix"),
            };
            loop {
                let op = match self.tokens.get(self.at) {
                    Some(Token::Symbol(op @ ('+' | '-' | '*' | '/'))) => *op,
                    _ => break,
                };
                let prec = if matches!(op, '+' | '-') { 1 } else { 2 };
                if prec < min {
                    break;
                }
                self.at += 1;
                let right = self.read(prec + 1)?;
                let mut out = vec![self.id(&op.to_string(), Some(2))?];
                out.extend(left);
                out.extend(right);
                left = out;
            }
            Ok(left)
        }
    }
    let mut parser = Parser {
        tokens: &lexical,
        at: 0,
        table,
    };
    let ids = parser.read(0)?;
    ensure!(parser.at == lexical.len(), "trailing expression text");
    Ok(ids)
}

#[cfg(test)]
mod task_encode_tests {
    use super::*;
    use crate::manifest::read_json;
    use std::path::Path;

    #[test]
    fn encode_for_task_emits_regression_and_rejects_unqualified_tasks() {
        let vocab: Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let genome = "(pipe X (ridge))";
        let regression = vocab.encode(genome).unwrap();
        assert_eq!(regression[2], vocab.task_id("regression").unwrap());
        assert_eq!(
            regression,
            vocab.encode_for_task(genome, "regression").unwrap()
        );
        assert!(vocab
            .encode_for_task(genome, "binary")
            .unwrap_err()
            .to_string()
            .contains("regression only"));
    }
}
