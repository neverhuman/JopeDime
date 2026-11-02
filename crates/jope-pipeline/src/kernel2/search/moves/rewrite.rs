use super::*;

/// Whether the search runs [`finish_round`]: some move is selected.
pub(crate) fn selected(options: &Options) -> bool {
    let engine = options.evaluator.engine();
    // laneD:B19-minimax-linearize: or the linearization pass.
    engine.simplify || engine.linearize
}

/// Runs the selected moves after a round's measurements and refinements, in a fixed order.
pub(crate) fn finish_round(
    state: &mut State,
    cut: &Cut,
    options: &Options,
    started: Stamp,
    deadline: Duration,
) -> Result<()> {
    let mut may_start = || !options.search.cancelled() && started.elapsed() < deadline;
    if options.evaluator.engine().simplify {
        simplify_round(state, cut, options, &mut may_start)?;
    }
    // laneD:B19-minimax-linearize begin
    if options.evaluator.engine().linearize {
        linearize_round(state, cut, options, &mut may_start)?;
    }
    // laneD:B19-minimax-linearize end
    Ok(())
}

/// The rewrite pass over the top `refinement.top_k` ranked unique parents with an expression
/// stage. `may_start` is checked before each proposal and again before each measurement; a
/// refusal ends the pass.
fn simplify_round(
    state: &mut State,
    cut: &Cut,
    options: &Options,
    may_start: &mut dyn FnMut() -> bool,
) -> Result<()> {
    let width = cut.train_x.first().map_or(0, Vec::len);
    let parents = owned(ranked_parents(
        &options.evaluator,
        &state.observations,
        REWRITE_SATURATE,
        options.refinement.top_k,
    )?);
    for (parent, parent_sha256) in parents {
        if state.budget.remaining() == 0 || !may_start() {
            break;
        }
        let proposal = propose(&options.evaluator, &parent, width)?;
        let admitted = proposal.admitted;
        let mut record = MoveRecord {
            id: REWRITE_SATURATE.into(),
            round: state.generation,
            parent_sha256,
            outcome: proposal.outcome().into(),
            program: proposal.program,
            reason: proposal.reason,
            measurement: None,
            rewrites: proposal.rewrites,
            linearization: None,
        };
        if admitted {
            let child = record.program.clone().context("admitted proposal")?;
            if state.seen.contains(&child) {
                record.outcome = DUPLICATE.into();
            } else if may_start() {
                let measured = options
                    .evaluator
                    .measure(
                        &child,
                        cut,
                        state.generation,
                        REWRITE,
                        options.search.clock,
                        &mut state.budget,
                    )
                    .expect("checked allowance");
                state.seen.insert(child);
                record.measurement = Some(state.observations.len());
                record.outcome = MEASURED.into();
                state.observations.push(measured);
            } else {
                record.outcome = NOT_STARTED.into();
                state.moves.push(record);
                break;
            }
        }
        state.moves.push(record);
    }
    Ok(())
}

// laneD:B19-minimax-linearize begin
/// The linearization pass over the top `refinement.top_k` ranked unique parents with a subtree it
/// can consider. Every considered subtree of a parent is recorded, in order; each offered program
/// not yet measured is measured once. `may_start` is checked before each parent and again before
/// each measurement. Once it refuses, or the allowance is spent, the parent's remaining offers are
/// recorded unmeasured and the pass ends.
fn linearize_round(
    state: &mut State,
    cut: &Cut,
    options: &Options,
    may_start: &mut dyn FnMut() -> bool,
) -> Result<()> {
    let width = cut.train_x.first().map_or(0, Vec::len);
    let parents = owned(ranked_parents(
        &options.evaluator,
        &state.observations,
        MINIMAX_LINEARIZE,
        options.refinement.top_k,
    )?);
    let mut ended: Option<&'static str> = None;
    for (parent, parent_sha256) in parents {
        if state.budget.remaining() == 0 || !may_start() {
            break;
        }
        for proposal in linearize::propose(&options.evaluator, &parent, cut, width)? {
            let mut record = MoveRecord {
                id: MINIMAX_LINEARIZE.into(),
                round: state.generation,
                parent_sha256: parent_sha256.clone(),
                program: proposal.program,
                outcome: REFUSED.into(),
                reason: proposal.reason,
                measurement: None,
                rewrites: Vec::new(),
                linearization: Some(proposal.linearization),
            };
            if proposal.admitted {
                let child = record.program.clone().context("admitted proposal")?;
                let outcome = if state.seen.contains(&child) {
                    DUPLICATE
                } else if let Some(stop) = ended {
                    stop
                } else if state.budget.remaining() == 0 {
                    ended = Some(NO_ALLOWANCE);
                    NO_ALLOWANCE
                } else if !may_start() {
                    ended = Some(NOT_STARTED);
                    NOT_STARTED
                } else {
                    let measured = options
                        .evaluator
                        .measure(
                            &child,
                            cut,
                            state.generation,
                            LINEARIZE,
                            options.search.clock,
                            &mut state.budget,
                        )
                        .expect("checked allowance");
                    state.seen.insert(child);
                    record.measurement = Some(state.observations.len());
                    state.observations.push(measured);
                    MEASURED
                };
                record.outcome = outcome.into();
            }
            state.moves.push(record);
        }
        if ended.is_some() {
            break;
        }
    }
    Ok(())
}
// laneD:B19-minimax-linearize end

/// Whether `text` is a pipeline program with a `map` or `derive` stage. [`check`] reads the
/// parents back from evidence, which is untrusted, so the text is parsed within
/// [`parse_untrusted_pipeline`]'s bounds, and a program over them is not readable.
fn has_expression_stage(text: &str) -> bool {
    parse_untrusted_pipeline(text).is_ok_and(|pipe| {
        pipe.stages
            .iter()
            .any(|s| matches!(s.op.as_str(), "map" | "derive"))
    })
}

/// The parents the pass `id` reads from `observations`, the measurements made before it: the
/// ranked programs, each at its first place, that the pass can read (an expression stage for the
/// rewrite pass, a subtree it would consider for the linearization pass), at most `top_k`, in
/// rank order. The search and [`check`] both call it.
pub(crate) fn ranked_parents<'a>(
    evaluator: &Kernel2Evaluator,
    observations: &'a [Kernel2Observation],
    id: &str,
    top_k: usize,
) -> Result<Vec<&'a Kernel2Observation>> {
    let mut unique = BTreeSet::new();
    let mut parents = Vec::new();
    for o in evaluator.rank(observations) {
        if parents.len() == top_k {
            break;
        }
        let genome = &o.observation.genome;
        if !unique.insert(genome.clone()) {
            continue;
        }
        let readable = if id == REWRITE_SATURATE {
            has_expression_stage(genome)
        } else {
            linearize::offers(evaluator, genome)?
        };
        if readable {
            parents.push(o);
        }
    }
    Ok(parents)
}

/// Each parent's program and sha256, so the search state can change while the pass reads them.
pub(crate) fn owned(parents: Vec<&Kernel2Observation>) -> Vec<(String, String)> {
    parents
        .into_iter()
        .map(|o| {
            (
                o.observation.genome.clone(),
                o.observation.program_sha256.clone(),
            )
        })
        .collect()
}

/// Each expression stage of `parent` simplified on its own, from its text alone, and the program
/// assembled from the stages the profile admitted, checked by the pipeline admission every
/// measured program passes. It reads no rows: `width` is the input's column count. A stage whose
/// expression cannot be rewritten keeps it and records why ([`NOT_RUN`]), so a rewrite or
/// admission failure refuses that stage instead of ending the run. [`check`] calls it on parents
/// read back from evidence, so `parent` is parsed within [`parse_untrusted_pipeline`]'s bounds
/// before any expression is.
pub(crate) fn propose(
    evaluator: &Kernel2Evaluator,
    parent: &str,
    width: usize,
) -> Result<Proposal> {
    let mut pipe =
        parse_untrusted_pipeline(parent).context("the rewrite pass needs a pipeline program")?;
    let profile = evaluator.profile();
    let pool = profile.expr.as_ref().context("expression pool required")?;
    let mut rewrites = Vec::new();
    for (index, stage) in pipe.stages.iter_mut().enumerate() {
        if !matches!(stage.op.as_str(), "map" | "derive") {
            continue;
        }
        let scope = ExpressionScope {
            columns: pool.column_upper_exclusive,
            elementwise: stage.op == "map",
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        };
        let rewrite = rewrite_stage(index, stage.params.get("expr"), evaluator, scope)
            .unwrap_or_else(|error| StageRewrite {
                stage: index,
                stop: NOT_RUN.into(),
                iterations: 0,
                enodes: 0,
                applications: 0,
                peak: 0,
                expression: None,
                refusal: Some(format!("{REWRITE_FAILED}: {error:#}")),
            });
        if let Some(text) = &rewrite.expression {
            stage
                .params
                .insert("expr".into(), ParamValue::Str(text.clone()));
        }
        rewrites.push(rewrite);
    }
    if rewrites.iter().all(|r| r.expression.is_none()) {
        return Ok(Proposal {
            rewrites,
            program: None,
            reason: None,
            admitted: false,
        });
    }
    let program = GenomeIr::Pipeline(pipe).canonical();
    feat_math::capture_gp_source!("kernel2-move", "rewrite", program.as_str());
    let reason = admit_child(&program, profile, width)
        .err()
        .map(|error| format!("{error:#}"));
    Ok(Proposal {
        rewrites,
        admitted: reason.is_none(),
        program: Some(program),
        reason,
    })
}

/// The rewrite of stage `index`'s expression `source` under `scope`, or why it could not run: the
/// expression is missing or does not parse, `simplify_v2` refuses it (the profile does not admit
/// it), or the rewrite panics. Each is the same for the same text, so `check` re-derives it.
fn rewrite_stage(
    index: usize,
    source: Option<&ParamValue>,
    evaluator: &Kernel2Evaluator,
    scope: ExpressionScope,
) -> Result<StageRewrite> {
    let Some(ParamValue::Str(source)) = source else {
        bail!("expression stage without an expression");
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        MathExpr::parse(source)?.simplify_v2(evaluator.profile(), scope)
    }))
    .map_err(|payload| {
        // The panic's own message, when it carries one, so the record says what failed.
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned());
        match message {
            Some(message) => anyhow!("the rewrite stopped on an internal error: {message}"),
            None => anyhow!("the rewrite stopped on an internal error"),
        }
    })??;
    let expression = outcome
        .simplified
        .as_ref()
        .map(MathExpr::to_source)
        .transpose()?;
    Ok(StageRewrite {
        stage: index,
        stop: outcome.stop.as_str().into(),
        iterations: outcome.iterations,
        enodes: outcome.enodes,
        applications: outcome.applications,
        peak: outcome.peak,
        expression,
        refusal: outcome.refusal,
    })
}
