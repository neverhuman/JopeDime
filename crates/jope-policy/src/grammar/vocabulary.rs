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
