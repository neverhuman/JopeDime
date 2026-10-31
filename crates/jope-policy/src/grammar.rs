//! Port of jope-dime `grammar.rs` (07304d6f): the policy action vocabulary, the masking
//! automaton, and exact program encode/decode.
//!
//! The vocabulary file format is jope-dime's `parallel.action_vocab.v1` JSON (`featmath.json`):
//! `ops` is keyed by token id, `expr` holds the expression tables, and extra keys (`index`,
//! `counters`, `source_sha256`) are ignored exactly as jope-dime's serde model ignores them.
//! DSL parsing and printing use the pinned jain-math `feat_math::dsl`, whose `parse_dsl`,
//! `parse_param_value` and `Display` are the same as jope-dime's vendored copy (the vendored
//! copy only adds `parse_portfolio`, which this grammar never emits).
use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::{parse_dsl, parse_param_value, DslExpr, ParamValue, Stage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{sha256_hex, PRODUCT_RESTRICTIONS};

/// Token id reserved for padding in policy input sequences.
pub const PAD: usize = 0;
/// Token id that starts a policy sequence.
pub const BOS: usize = 1;
/// Token id that terminates a complete program.
pub const EOS: usize = 2;
/// Longest token sequence (BOS..EOS) the policy accepts.
pub const MAX_PROGRAM_TOKENS: usize = 128;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Slot {
    pub param: String,
    pub kind: String,
    pub val_ids: Vec<usize>,
    pub bin_texts: Vec<String>,
    pub absent_id: Option<usize>,
    pub is_expr: bool,
    pub required: bool,
    pub default: Value,
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
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

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
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
    /// Present only on a product restriction (`parallel.vocab_restriction.v1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restriction: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Automaton state; `sampling` hides `generate=false` operators (jope-dime default).
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

    /// The automaton phase, for decode outcome records: `done` only after EOS was accepted, `eos`
    /// when EOS is the one legal token.
    pub fn phase_name(&self) -> &'static str {
        match self.phase {
            Phase::Engine => "engine",
            Phase::Task => "task",
            Phase::Stage => "stage",
            Phase::Val => "value",
            Phase::Append => "append",
            Phase::Expr => "expression",
            Phase::ExprEnd => "expression_end",
            Phase::Eos => "eos",
            Phase::Done => "done",
        }
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
                    depth < table.max_depth
                        && nodes + 1 + (pending - 1 + **arity) <= table.max_nodes
                })
                .map(|(&id, _)| id),
        );
        out
    }

    /// Sorted, deduplicated legal next tokens.
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
                let slot = &v.ops[&self.op.expect("value phase has an operator")].slots[self.slot];
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
            Phase::Eos => vec![EOS],
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
                    .context("engine token")?
                    .0
                    .clone();
                self.phase = Phase::Task;
            }
            Phase::Task => {
                self.task = v
                    .task_ids
                    .iter()
                    .find(|(_, id)| **id == token)
                    .context("task token")?
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
                let slot =
                    &v.ops[&self.op.context("value phase without operator")?].slots[self.slot];
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
            Phase::Done => bail!("automaton already finished"),
        };
        Ok(())
    }

    fn consume_expr(&mut self, v: &Vocab, token: usize) {
        let (children, depth) = self.stack.pop().expect("expression stack is non-empty");
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
        let o = &v.ops[&self.op.expect("entered stage has an operator")];
        if self.slot < o.slots.len() {
            self.phase = Phase::Val;
        } else if o.allow_append {
            self.phase = Phase::Append;
        } else {
            self.finish(v);
        }
    }

    fn finish(&mut self, v: &Vocab) {
        if v.ops[&self.op.expect("finished stage has an operator")].role == "head" {
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
    /// Parse a vocabulary file (no identity check; see [`bind_vocabulary`]).
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let vocab: Self = serde_json::from_slice(bytes).context("policy vocabulary")?;
        vocab.check_tables()?;
        Ok(vocab)
    }

    /// Internal consistency the automaton relies on (it indexes these maps directly).
    fn check_tables(&self) -> Result<()> {
        let n = self.tokens.len();
        ensure!(
            n > 3 && self.tokens.get(PAD).map(String::as_str) == Some("PAD"),
            "vocabulary token table is malformed"
        );
        ensure!(
            self.tokens[BOS] == "BOS" && self.tokens[EOS] == "EOS",
            "vocabulary reserved tokens moved"
        );
        for engine in &self.engines {
            ensure!(
                self.engine_ids.get(engine).is_some_and(|&id| id < n)
                    && self.engine_transforms.contains_key(engine)
                    && self.engine_heads.contains_key(engine)
                    && self.max_transforms.contains_key(engine)
                    && self.max_expr_stages.contains_key(engine),
                "vocabulary engine {engine} is incomplete"
            );
            for id in self.engine_transforms[engine]
                .iter()
                .chain(&self.engine_heads[engine])
            {
                ensure!(
                    self.ops.contains_key(id),
                    "vocabulary engine {engine} lists unknown operator {id}"
                );
            }
        }
        for task in &self.tasks {
            ensure!(
                self.task_ids.get(task).is_some_and(|&id| id < n),
                "vocabulary task {task} has no token"
            );
        }
        for (id, op) in &self.ops {
            ensure!(
                *id == op.token_id && *id < n,
                "vocabulary operator {id} is misindexed"
            );
            for slot in &op.slots {
                ensure!(
                    slot.val_ids.len() == slot.bin_texts.len()
                        && slot.val_ids.iter().all(|&v| v < n)
                        && slot.absent_id.is_none_or(|v| v < n),
                    "vocabulary operator {} slot {} is malformed",
                    op.name,
                    slot.param
                );
            }
        }
        ensure!(
            self.expr.const_ids.len() == self.expr.const_values.len()
                && self.append_ids.iter().all(|&v| v < n)
                && self.expr.end_id < n,
            "vocabulary expression tables are malformed"
        );
        Ok(())
    }

    pub fn engine_id(&self, engine: &str) -> Result<usize> {
        self.engine_ids
            .get(engine)
            .copied()
            .with_context(|| format!("{engine} absent"))
    }

    /// Task token; like jope-dime `task_contract::require_task`, only regression is qualified.
    pub fn task_id(&self, task: &str) -> Result<usize> {
        ensure!(
            task == "regression",
            "task {task} is not qualified for jope-dime proposals; regression only"
        );
        self.task_ids
            .get(task)
            .copied()
            .with_context(|| format!("{task} absent"))
    }

    /// `[BOS, featmath, task]`.
    pub fn proposal_prefix(&self, task: &str) -> Result<Vec<usize>> {
        Ok(vec![BOS, self.engine_id("featmath")?, self.task_id(task)?])
    }

    /// Automaton advanced past the proposal prefix.
    pub fn prefix_state(&self, task: &str) -> Result<State> {
        let mut state = State::default();
        state.advance(self, self.engine_id("featmath")?)?;
        state.advance(self, self.task_id(task)?)?;
        Ok(state)
    }

    /// Legal sets before each token after BOS; fails unless the program terminates.
    pub fn replay(&self, tokens: &[usize]) -> Result<Vec<Vec<usize>>> {
        ensure!(tokens.first() == Some(&BOS), "BOS missing");
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
        while tokens.get(i) != Some(&EOS) {
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
                let id = *tokens.get(i).context("append flag absent")?;
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
        let program = DslExpr { stages }.to_string();
        feat_math::capture_gp_source!("policy-grammar", "decode", program.as_str());
        Ok(program)
    }

    /// Project a proposal before evaluation (lossy parameter snapping); the projected program
    /// needs its own score. [`Vocab::encode`] never performs this conversion.
    pub fn project_proposal(&self, genome: &str) -> Result<String> {
        let mut pipeline = parse_dsl(genome)?;
        for stage in &mut pipeline.stages {
            let op = self
                .featmath_op(&stage.op)
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
        let program = self.decode(&tokens)?;
        feat_math::capture_gp_source!("policy-grammar", "projection", program.as_str());
        Ok(program)
    }

    fn featmath_op(&self, name: &str) -> Option<&Op> {
        self.ops
            .values()
            .find(|o| o.engine == "featmath" && o.name == name)
    }

    pub fn encode(&self, genome: &str) -> Result<Vec<usize>> {
        self.encode_for_task(genome, "regression")
    }

    /// Exact encoding; refuses any parameter value without a frozen token.
    pub fn encode_for_task(&self, genome: &str, task: &str) -> Result<Vec<usize>> {
        let pipeline = parse_dsl(genome)?;
        let mut tokens = self.proposal_prefix(task)?;
        for stage in &pipeline.stages {
            let op = self
                .featmath_op(&stage.op)
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
                        _ => bail!("expression must be text"),
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
                    let position = slot
                        .bin_texts
                        .iter()
                        .position(|s| {
                            let bin = parse_param_value(s);
                            if let (Some(a), Some(b)) = (value.as_f64(), bin.as_f64()) {
                                a == b
                            } else {
                                &bin == value
                            }
                        })
                        .context(
                            "exact parameter has no frozen token; scores cannot be attached to a snapped program",
                        )?;
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
                    _ => bail!("invalid append flag"),
                };
                tokens.push(self.append_ids[append]);
            }
        }
        tokens.push(EOS);
        ensure!(
            tokens.len() <= MAX_PROGRAM_TOKENS,
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

#[derive(Clone)]
enum Lexeme {
    Name(String),
    Number(f64),
    Symbol(char),
}

fn lex_expr(text: &str) -> Result<Vec<Lexeme>> {
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
            lexical.push(Lexeme::Number(
                chars[start..i].iter().collect::<String>().parse()?,
            ));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            lexical.push(Lexeme::Name(chars[start..i].iter().collect()));
        } else {
            ensure!("()+-*/,".contains(c), "unsupported expression character");
            lexical.push(Lexeme::Symbol(c));
            i += 1;
        }
    }
    Ok(lexical)
}

struct ExprParser<'a> {
    tokens: &'a [Lexeme],
    at: usize,
    table: &'a Expr,
}

impl ExprParser<'_> {
    fn symbol(&self, s: char) -> bool {
        matches!(self.tokens.get(self.at), Some(Lexeme::Symbol(c)) if *c == s)
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
            .find(|(id, sym)| sym.as_str() == s && self.table.func_arity.get(id).copied() == arity)
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
            Lexeme::Number(n) => vec![self.number(n)?],
            Lexeme::Symbol('(') => {
                let x = self.read(0)?;
                self.need(')')?;
                x
            }
            Lexeme::Symbol('-') => {
                if let Some(Lexeme::Number(n)) = self.tokens.get(self.at) {
                    let n = *n;
                    self.at += 1;
                    vec![self.number(-n)?]
                } else {
                    let mut out = vec![self.id("-", Some(2))?, self.number(0.)?];
                    out.extend(self.read(3)?);
                    out
                }
            }
            Lexeme::Name(name) => {
                if self.symbol('(') {
                    self.at += 1;
                    if name == "col" {
                        let n = if let Some(Lexeme::Number(n)) = self.tokens.get(self.at) {
                            *n
                        } else {
                            bail!("column index missing")
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
            _ => bail!("invalid expression prefix"),
        };
        while let Some(Lexeme::Symbol(op @ ('+' | '-' | '*' | '/'))) = self.tokens.get(self.at) {
            let op = *op;
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

fn parse_expr_tokens(text: &str, table: &Expr) -> Result<Vec<usize>> {
    let lexical = lex_expr(text)?;
    let mut parser = ExprParser {
        tokens: &lexical,
        at: 0,
        table,
    };
    let ids = parser.read(0)?;
    ensure!(parser.at == lexical.len(), "trailing expression text");
    Ok(ids)
}

/// Which accepted identity a vocabulary file has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VocabularyIdentity {
    /// Byte-identical to the checkpoint's bound vocabulary.
    Checkpoint,
    /// A documented, token-compatible product restriction of the bound vocabulary.
    ProductRestriction { base_sha256: String },
}

/// A parsed vocabulary with its checked identity.
///
/// The fields are private and [`bind_vocabulary`] is the only constructor, so
/// [`BoundVocabulary::sha256`] is always the sha256 of the exact bytes the vocabulary was parsed
/// from. A caller cannot wrap an arbitrary [`Vocab`] under a checkpoint vocabulary label:
///
/// ```compile_fail,E0451
/// use jope_policy::grammar::{BoundVocabulary, Vocab, VocabularyIdentity};
/// fn forge(vocab: Vocab) -> BoundVocabulary {
///     BoundVocabulary {
///         vocab: vocab.into(),
///         sha256: jope_policy::CHECKPOINT_VOCABULARY_SHA256.into(),
///         identity: VocabularyIdentity::Checkpoint,
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct BoundVocabulary {
    vocab: Arc<Vocab>,
    sha256: String,
    identity: VocabularyIdentity,
}

impl BoundVocabulary {
    /// The parsed vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// sha256 of the bytes the vocabulary was parsed from, computed by [`bind_vocabulary`].
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The identity [`bind_vocabulary`] established against its checkpoint label.
    pub fn identity(&self) -> &VocabularyIdentity {
        &self.identity
    }

    /// Encode a canonical program only when it round-trips through this byte-bound native
    /// vocabulary and stays inside the native policy sequence cap.
    pub fn representable_tokens(&self, program: &str) -> Result<Vec<usize>> {
        let tokens = self.vocab.encode(program)?;
        ensure!(
            tokens.len() <= MAX_PROGRAM_TOKENS
                && tokens.first() == Some(&BOS)
                && tokens.last() == Some(&EOS),
            "program exceeds native cap or lacks BOS/EOS"
        );
        self.vocab.replay(&tokens)?;
        ensure!(
            self.vocab.decode(&tokens)? == program,
            "native program round-trip mismatch"
        );
        Ok(tokens)
    }

    pub(crate) fn shared_vocab(&self) -> Arc<Vocab> {
        Arc::clone(&self.vocab)
    }

    /// Unit tests only: an arbitrary (for example adversarial) vocabulary under a label digest,
    /// which [`bind_vocabulary`] and every product path refuse.
    #[cfg(test)]
    pub(crate) fn for_unit_test(vocab: Vocab) -> Self {
        Self {
            vocab: Arc::new(vocab),
            sha256: sha256_hex(b"jope-policy unit-test vocabulary"),
            identity: VocabularyIdentity::Checkpoint,
        }
    }
}

/// Bind vocabulary bytes to a checkpoint. Accepts the checkpoint's own vocabulary, or a
/// documented product restriction whose base is that vocabulary and whose restriction block
/// declares an unchanged token table. Everything else is refused.
pub fn bind_vocabulary(
    bytes: &[u8],
    checkpoint_vocabulary_sha256: &str,
) -> Result<BoundVocabulary> {
    let sha256 = sha256_hex(bytes);
    let bound = checkpoint_vocabulary_sha256.to_ascii_lowercase();
    let identity = if sha256 == bound {
        VocabularyIdentity::Checkpoint
    } else if let Some((_, base)) = PRODUCT_RESTRICTIONS
        .iter()
        .find(|(restricted, base)| *restricted == sha256 && *base == bound)
    {
        VocabularyIdentity::ProductRestriction {
            base_sha256: (*base).to_string(),
        }
    } else {
        bail!(
            "refused: vocabulary {sha256} is neither the checkpoint vocabulary {bound} nor a documented product restriction of it"
        );
    };
    let vocab = Vocab::from_json(bytes)?;
    match &identity {
        VocabularyIdentity::Checkpoint => ensure!(
            vocab.restriction.is_none(),
            "checkpoint vocabulary must not carry a restriction block"
        ),
        VocabularyIdentity::ProductRestriction { base_sha256 } => {
            let r = vocab
                .restriction
                .as_ref()
                .context("product vocabulary lacks its restriction block")?;
            ensure!(
                r.get("schema").and_then(Value::as_str) == Some("parallel.vocab_restriction.v1")
                    && r.get("base_vocabulary_sha256").and_then(Value::as_str)
                        == Some(base_sha256.as_str())
                    && r.get("token_table").and_then(Value::as_str) == Some("unchanged"),
                "product vocabulary restriction block does not declare a token-compatible restriction of {base_sha256}"
            );
        }
    }
    Ok(BoundVocabulary {
        vocab: Arc::new(vocab),
        sha256,
        identity,
    })
}

/// Full token-compatibility proof of a restriction against its base: identical token table,
/// expression tables, engines, tasks and append ids; every kept operator identical to the
/// base operator; every removed operator listed in the restriction block.
pub fn verify_restriction(base: &Vocab, restricted: &Vocab) -> Result<()> {
    ensure!(
        base.tokens == restricted.tokens
            && base.expr == restricted.expr
            && base.engines == restricted.engines
            && base.engine_ids == restricted.engine_ids
            && base.tasks == restricted.tasks
            && base.task_ids == restricted.task_ids
            && base.append_ids == restricted.append_ids
            && base.max_transforms == restricted.max_transforms
            && base.max_expr_stages == restricted.max_expr_stages,
        "restriction changed the token table or grammar constants"
    );
    ensure!(
        base.engine_transforms
            .keys()
            .eq(restricted.engine_transforms.keys())
            && base.engine_heads.keys().eq(restricted.engine_heads.keys()),
        "restriction changed the engine operator tables"
    );
    for (id, op) in &restricted.ops {
        ensure!(
            base.ops.get(id) == Some(op),
            "restriction changed operator {id}"
        );
    }
    let removed = restricted
        .restriction
        .as_ref()
        .and_then(|r| r.get("removed_ops"))
        .and_then(Value::as_array)
        .context("restriction lists no removed_ops")?
        .iter()
        .map(|r| {
            r.get("token_id")
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .context("removed op without token_id")
        })
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    let actual = base
        .ops
        .keys()
        .filter(|id| !restricted.ops.contains_key(id))
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        removed == actual,
        "restriction removed_ops disagrees with the operator tables"
    );
    for (engine, ids) in &restricted.engine_transforms {
        let expected = base
            .engine_transforms
            .get(engine)
            .with_context(|| format!("restriction lists transforms for unknown engine {engine}"))?
            .iter()
            .filter(|id| restricted.ops.contains_key(id))
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            *ids == expected,
            "restriction reordered {engine} transforms"
        );
    }
    for (engine, ids) in &restricted.engine_heads {
        let expected = base
            .engine_heads
            .get(engine)
            .with_context(|| format!("restriction lists heads for unknown engine {engine}"))?
            .iter()
            .filter(|id| restricted.ops.contains_key(id))
            .copied()
            .collect::<Vec<_>>();
        ensure!(*ids == expected, "restriction reordered {engine} heads");
    }
    Ok(())
}
