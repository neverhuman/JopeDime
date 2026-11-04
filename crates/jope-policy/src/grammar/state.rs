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
