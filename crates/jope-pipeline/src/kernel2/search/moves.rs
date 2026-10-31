//! Deterministic moves at the end of each search round.
//!
//! A move proposes children from the round's ranked parents without drawing from any random
//! stream. Each new child the profile admits is measured like every other program: one
//! evaluation from the shared allowance, under the start guard refinement uses, with its pass's
//! origin. Proposal work is not charged. Every proposal is recorded in [`PATH`]. With no move
//! selected the search never calls this module and nothing is written.
//!
//! The first pass is the exact rewrite pass ([`REWRITE_SATURATE`], origin [`REWRITE`]): every
//! `map` or `derive` stage's expression of each of the round's top `refinement.top_k` ranked
//! parents with such a stage is simplified by `MathExpr::simplify_v2`, from the expression text
//! alone. A smaller tree the profile refuses is discarded and its reason recorded, and so is a
//! stage whose expression cannot be rewritten at all ([`NOT_RUN`]); the pass goes on either way.
//!
//! The second pass, run after the first when both are selected, is the bounded minimax
//! linearization pass ([`MINIMAX_LINEARIZE`], origin [`LINEARIZE`]). For each of the round's top
//! `refinement.top_k` ranked parents with a `derive` stage that has a subtree it can consider,
//! `MathExpr::linearize_v2` fits each one-variable subtree near the root of each `derive` stage's
//! expression by its degree-1 discrete minimax line, on that stage's training input alone, and
//! offers the line in its place when the line is smaller, meets the fit's minimax bound on those
//! rows, and the profile admits the result. Each offer is one child program, measured once; the
//! measured score ranks it, never the line's error. Every considered subtree is recorded, offered
//! or refused, with its fit, bound and reason.
//!
//! `check` re-derives the records against the published measurements. For each round and pass
//! it ranks the measurements made before the pass, as the pass ranked them, and requires the
//! records to follow that list of parents in order: all of it, or a prefix only in a run that
//! ended in that round on its evaluations or seconds bound (exactly where the allowance ran out
//! under an evaluations stop; under a seconds stop, where the start guard refused, which is not
//! published). A rewrite record re-derives from its parent's text alone, and a measured rewritten
//! program must score bit for bit as its parent. A linearization record re-derives from its
//! parent's text and the fit it records: the training rows are not published, so the fit, its
//! range and the line's measured error are checked for consistency with each other and with the
//! other records of the same stage expression, never measured again.

// laneD:B20-rewrite-simplify (the module); laneD:B19-minimax-linearize (the second pass).

use anyhow::{anyhow, bail, ensure, Context, Result};
use feat_math::{
    dsl::ParamValue,
    kernel::search::admit_child,
    math_expr::{ExpressionScope, MathExpr},
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Duration};

use super::{generation::BARREN_ROUND_LIMIT, CutSearch, Options, State};
use crate::{
    clock::Stamp,
    kernel2::{
        receipt::Search,
        records::{Faults, Measurement},
        refinement::parse_untrusted_pipeline,
        EngineRecord, Kernel2Evaluator, Kernel2Observation, LINEARIZE, MINIMAX_LINEARIZE, REWRITE,
        REWRITE_SATURATE,
    },
    prepare::Cut,
    search::{STOPPED_BY_EVALUATIONS, STOPPED_BY_SECONDS},
    sha256_hex,
};

// laneD:B19-minimax-linearize begin
mod linearize;
pub use linearize::{FitRecord, LinearizationRecord, PREFIX_FAULTS};
// laneD:B19-minimax-linearize end

/// Where a run publishes its move records, relative to its output directory.
pub const PATH: &str = "search/cut-0/moves.json";

/// Every stage kept its expression, and the profile refused nothing.
pub const UNCHANGED: &str = "unchanged";
/// No stage change survived the profile, or the assembled program failed pipeline admission.
pub const REFUSED: &str = "refused";
/// The admitted program was already measured.
pub const DUPLICATE: &str = "duplicate";
/// The start guard refused the measurement; the run stopped on its seconds bound.
pub const NOT_STARTED: &str = "not_started";
/// The admitted program was measured at `measurement`, whatever that measurement's status.
pub const MEASURED: &str = "measured";
// laneD:B19-minimax-linearize begin
/// The shared allowance was spent before the admitted program could be measured; the pass ended
/// and the run stopped on its evaluations bound. Only the linearization pass, which can offer
/// several programs per parent, records it.
pub const NO_ALLOWANCE: &str = "no_allowance";
// laneD:B19-minimax-linearize end

/// The stop of a stage whose expression could not be rewritten at all, because it does not parse
/// or the profile refuses it: saturation never ran, and the stage's refusal says why.
pub const NOT_RUN: &str = "not_run";
/// The start of the refusal of a stage whose expression could not be rewritten at all.
pub const REWRITE_FAILED: &str = "the stage's expression could not be rewritten";

/// One proposal of one move for one parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveRecord {
    /// The move's method id.
    pub id: String,
    /// The search round after which the move ran: the measured child's round index.
    pub round: usize,
    pub parent_sha256: String,
    /// The assembled program, when a stage changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// One of [`UNCHANGED`], [`REFUSED`], [`DUPLICATE`], [`NOT_STARTED`], [`NO_ALLOWANCE`] or
    /// [`MEASURED`].
    pub outcome: String,
    /// Why pipeline admission refused the assembled program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The child's index in `observations.json`, for [`MEASURED`] only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<usize>,
    /// The rewrite pass's record of each expression stage, in stage order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rewrites: Vec<StageRewrite>,
    // laneD:B19-minimax-linearize begin
    /// The linearization pass's record of its one subtree; absent from the rewrite pass's
    /// records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linearization: Option<LinearizationRecord>,
    // laneD:B19-minimax-linearize end
}

/// The rewrite pass on one expression stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRewrite {
    /// The stage's index in the parent.
    pub stage: usize,
    /// Why saturation stopped: `saturated`, `iteration_limit`, `node_limit` or `malformed`; or
    /// [`NOT_RUN`] when it never ran, with every counter 0 and the reason in `refusal`.
    pub stop: String,
    pub iterations: usize,
    pub enodes: usize,
    pub applications: usize,
    pub peak: usize,
    /// The smaller expression the profile admitted, which replaced the stage's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    /// Why the profile refused the smaller expression, which was discarded; or, with the stop
    /// [`NOT_RUN`], why the stage's own expression could not be rewritten.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl StageRewrite {
    /// Whether the profile's admission (the pool, the caps or the negation rule) refused a smaller
    /// expression that saturation found, so the stage kept its own.
    pub fn refused_by_pool(&self) -> bool {
        self.refusal.is_some() && self.stop != NOT_RUN
    }

    /// Whether the stage's own expression could not be rewritten at all ([`NOT_RUN`]).
    pub fn not_run(&self) -> bool {
        self.stop == NOT_RUN
    }
}

/// How many times the rewrite pass dropped a smaller expression because the profile's admission
/// refused it, over every record of `moves`: the count `MovesRecord::refused_by_pool` publishes,
/// refusal events per round a parent is read; a parent ranked top for N rounds adds N. It does
/// not count distinct expressions.
pub fn refused_by_pool(moves: &[MoveRecord]) -> usize {
    rewrite_stages(moves)
        .filter(|r| r.refused_by_pool())
        .count()
}

/// How many stage entries of the rewrite pass's records have the stop [`NOT_RUN`]: the count
/// `MovesRecord::not_run` publishes. A genuine run is expected to have none.
pub fn not_run(moves: &[MoveRecord]) -> usize {
    rewrite_stages(moves).filter(|r| r.not_run()).count()
}

fn rewrite_stages(moves: &[MoveRecord]) -> impl Iterator<Item = &StageRewrite> {
    moves
        .iter()
        .filter(|m| m.id == REWRITE_SATURATE)
        .flat_map(|m| &m.rewrites)
}

/// Whether the search runs [`finish_round`]: some move is selected.
pub(super) fn selected(options: &Options) -> bool {
    let engine = options.evaluator.engine();
    // laneD:B19-minimax-linearize: or the linearization pass.
    engine.simplify || engine.linearize
}

/// Runs the selected moves after a round's measurements and refinements, in a fixed order.
pub(super) fn finish_round(
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
fn ranked_parents<'a>(
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
fn owned(parents: Vec<&Kernel2Observation>) -> Vec<(String, String)> {
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

/// The rewrite pass's proposal for one parent, before any search state is read.
#[derive(Debug)]
pub(crate) struct Proposal {
    rewrites: Vec<StageRewrite>,
    program: Option<String>,
    reason: Option<String>,
    /// A stage changed and the assembled program passed pipeline admission.
    admitted: bool,
}

impl Proposal {
    /// The record's outcome when the proposal was not admitted.
    fn outcome(&self) -> &'static str {
        if self.program.is_some() || self.rewrites.iter().any(|r| r.refusal.is_some()) {
            REFUSED
        } else {
            UNCHANGED
        }
    }
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

/// Writes the move records under `root` when a move was selected, and returns `engine` with
/// their digest and, with the rewrite pass selected, its pool refusal and not-run counts. Without
/// a selected move nothing is written and `engine` is returned as is.
pub(crate) fn publish(
    searched: &CutSearch,
    root: &Path,
    mut engine: Option<EngineRecord>,
) -> Result<Option<EngineRecord>> {
    if let Some(record) = engine.as_mut().and_then(|e| e.moves.as_mut()) {
        let bytes = serde_json::to_vec_pretty(&searched.moves)?;
        std::fs::write(root.join(PATH), &bytes)?;
        record.sha256 = Some(sha256_hex(&bytes));
        record.refused_by_pool = pool_count(&record.ids, &searched.moves);
        record.not_run = not_run_count(&record.ids, &searched.moves);
    }
    Ok(engine)
}

/// The pool refusal count a moves record publishes: present exactly when the rewrite pass ran.
fn pool_count(ids: &[String], moves: &[MoveRecord]) -> Option<usize> {
    ids.iter()
        .any(|id| id == REWRITE_SATURATE)
        .then(|| refused_by_pool(moves))
}

/// The not-run count a moves record publishes: present exactly when the rewrite pass ran.
fn not_run_count(ids: &[String], moves: &[MoveRecord]) -> Option<usize> {
    ids.iter()
        .any(|id| id == REWRITE_SATURATE)
        .then(|| not_run(moves))
}

/// Re-derives published move records against the published measurements, which `observations`
/// holds re-derived, in the same order.
///
/// For each round and each selected pass, in the order the search runs them, it ranks the
/// measurements made before the pass as the pass ranked them and takes the parents the pass read
/// (`ranked_parents`). The records must follow those parents in order, one record (rewrite) or
/// one record per considered subtree (linearization) each: all of them, or a prefix when the pass
/// stopped on the allowance or the start guard, which only the run's last round can do, in a run
/// that stopped on its evaluations or seconds bound. Under an evaluations stop that prefix is
/// exact: in the last round a parent's records start only while a measurement of the run is still
/// to come, and a pass stops short only once every measurement is made. Under a seconds stop it
/// is not: where the start guard refused is not published, so the last round's trailing records
/// that measure nothing could be removed, or such records appended for the next ranked parents,
/// without this check seeing it. Each record's proposal re-derives from its parent alone (and,
/// for a linearization record, the fit it records), and its outcome from the measurements: a
/// measured child is the next measurement, of its pass's origin and round, and one whose program
/// no earlier measurement has; a measured rewritten program scores bit for bit as its parent.
/// Every [`REWRITE`] and [`LINEARIZE`] measurement is accounted for by exactly one record, and
/// the selection record's pool refusal and not-run counts are counted again from the records.
pub(crate) fn check(
    moves: &[MoveRecord],
    s: &Search,
    measurements: &[Measurement],
    observations: &[Kernel2Observation],
    evaluator: &Kernel2Evaluator,
    width: usize,
) -> Result<()> {
    let selection = s
        .engine_selection
        .as_ref()
        .and_then(|e| e.moves.as_ref())
        .context("move records without a selected move")?;
    ensure!(
        measurements.len() == observations.len(),
        "move measurements do not re-derive"
    );
    ensure!(
        selection.refused_by_pool == pool_count(&selection.ids, moves),
        "the pool refusal count does not re-derive from the move records"
    );
    ensure!(
        selection.not_run == not_run_count(&selection.ids, moves),
        "the not-run count does not re-derive from the move records"
    );
    let mut accounted = Accounted {
        s,
        measurements,
        observations,
        measured: BTreeSet::new(),
        known: 0,
        ended: false,
    };
    // The round count comes from the receipt and the loop below replays every round, so it is
    // bounded first: a run has at most `BARREN_ROUND_LIMIT` rounds per completed evaluation, plus
    // that many (the search's stop rule).
    ensure!(
        s.generations
            <= BARREN_ROUND_LIMIT.saturating_mul(s.completed_evaluations.saturating_add(1)),
        "the run's round count exceeds what its evaluations allow"
    );
    let mut next = 0usize;
    // The round in which a pass stopped short of its parents or ended.
    let mut stopped: Option<usize> = None;
    for round in 0..s.generations {
        for id in &selection.ids {
            let (origin, later): (&str, &[&str]) = match id.as_str() {
                REWRITE_SATURATE => (REWRITE, &[REWRITE, LINEARIZE]),
                MINIMAX_LINEARIZE => (LINEARIZE, &[LINEARIZE]),
                _ => bail!("a move record names an unselected method"),
            };
            // The measurements made before the pass: every earlier round's, and this round's but
            // for the children of this pass and of the passes after it.
            let before = |o: &Kernel2Observation| {
                o.observation.generation < round
                    || (o.observation.generation == round && !later.contains(&o.observation.origin))
            };
            let start = observations.iter().take_while(|o| before(o)).count();
            ensure!(
                observations[start..].iter().all(|o| !before(o)),
                "move measurements are out of round order"
            );
            accounted.known = start;
            let parents =
                ranked_parents(evaluator, &observations[..start], id, s.refinement.top_k)?;
            // In the last round of a run that stopped on its evaluations bound nothing is charged
            // after the passes, so the allowance left when a pass reaches a parent is the number
            // of measurements still to come: the pass reads a parent only while one is.
            let spent = round + 1 == s.generations && s.stopped_by == STOPPED_BY_EVALUATIONS;
            let mut read = 0usize;
            while next < moves.len() && moves[next].round == round && moves[next].id == *id {
                ensure!(
                    stopped.is_none() && !accounted.ended,
                    "move records continue after their pass ended"
                );
                ensure!(
                    !spent || accounted.known < observations.len(),
                    "a move record starts after the allowance was spent"
                );
                let m = &moves[next];
                let parent = parents
                    .get(read)
                    .filter(|p| p.observation.program_sha256 == m.parent_sha256)
                    .copied()
                    .context("move records do not follow the ranked parents")?;
                // The records this parent accounts for, each with whether its program was
                // admitted and the outcome it records when not.
                let (group, expected): (&[MoveRecord], Vec<(bool, &str)>) = if m.id
                    == REWRITE_SATURATE
                {
                    let proposal = propose(evaluator, &parent.observation.genome, width)?;
                    ensure!(
                        proposal.rewrites == m.rewrites
                            && proposal.program == m.program
                            && proposal.reason == m.reason
                            && m.linearization.is_none(),
                        "a move record does not re-derive from its parent"
                    );
                    (
                        std::slice::from_ref(m),
                        vec![(proposal.admitted, proposal.outcome())],
                    )
                } else {
                    // laneD:B19-minimax-linearize begin
                    let size = moves[next..]
                        .iter()
                        .take_while(|n| {
                            n.id == m.id && n.round == m.round && n.parent_sha256 == m.parent_sha256
                        })
                        .count();
                    let group = &moves[next..next + size];
                    let recorded = group
                        .iter()
                        .map(|r| r.linearization.clone())
                        .collect::<Option<Vec<_>>>()
                        .context("a linearization move has no subtree record")?;
                    let proposals = linearize::rederive(
                        evaluator,
                        &parent.observation.genome,
                        width,
                        &recorded,
                    )?;
                    ensure!(
                        group.iter().zip(&proposals).all(|(r, p)| {
                            r.linearization.as_ref() == Some(&p.linearization)
                                && r.program == p.program
                                && r.reason == p.reason
                                && r.rewrites.is_empty()
                        }),
                        "a move record does not re-derive from its parent"
                    );
                    (
                        group,
                        proposals.iter().map(|p| (p.admitted, REFUSED)).collect(),
                    )
                    // laneD:B19-minimax-linearize end
                };
                for (m, (admitted, refused)) in group.iter().zip(expected) {
                    accounted.outcome(m, admitted, refused, origin, parent)?;
                }
                next += group.len();
                read += 1;
            }
            if accounted.ended || read < parents.len() {
                // There, a pass stops short or ends only where the allowance ran out: with every
                // measurement of the run made.
                ensure!(
                    !spent || accounted.known == observations.len(),
                    "a move pass stopped short of its ranked parents with allowance left"
                );
                stopped.get_or_insert(round);
            }
        }
    }
    ensure!(
        next == moves.len(),
        "move records do not follow the ranked parents"
    );
    if let Some(round) = stopped {
        ensure!(
            round + 1 == s.generations
                && (s.stopped_by == STOPPED_BY_EVALUATIONS || s.stopped_by == STOPPED_BY_SECONDS),
            "a move pass stopped short of its ranked parents before the run's last round"
        );
    }
    ensure!(
        accounted.measured
            == measurements
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.observation.origin == REWRITE || m.observation.origin == LINEARIZE
                })
                .map(|(i, _)| i)
                .collect(),
        "move measurements do not re-derive"
    );
    Ok(())
}

/// What a measurement scores, as bits: its status, training and held-out R², RMSE, selection
/// score, and fault counts.
type Scored = (
    &'static str,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Faults,
);

fn scored(o: &Kernel2Observation) -> Scored {
    (
        o.observation.status,
        o.observation.train_r2.map(f64::to_bits),
        o.observation.hold_r2.map(f64::to_bits),
        o.observation.rmse.map(f64::to_bits),
        o.selection_score.map(f64::to_bits),
        Faults::from(o.faults),
    )
}

/// What the outcome checks have established so far, against the published measurements.
struct Accounted<'a> {
    s: &'a Search,
    measurements: &'a [Measurement],
    observations: &'a [Kernel2Observation],
    /// The measurement indices the records claim.
    measured: BTreeSet<usize>,
    /// How many measurements had been made when the current record's pass reached it.
    known: usize,
    /// A pass ended on its start guard or its allowance; nothing is measured after it.
    ended: bool,
}

impl Accounted<'_> {
    /// Re-derives one record's outcome from the measurements: `admitted` is whether its
    /// re-derived program was admitted, `refused` the outcome it records when not, `origin` its
    /// pass's measurement origin, and `parent` the ranked measurement its pass read.
    fn outcome(
        &mut self,
        m: &MoveRecord,
        admitted: bool,
        refused: &str,
        origin: &str,
        parent: &Kernel2Observation,
    ) -> Result<()> {
        let s = self.s;
        let program = m.program.as_deref();
        let known = self.measurements[..self.known]
            .iter()
            .any(|o| Some(o.observation.genome.as_str()) == program);
        match m.outcome.as_str() {
            UNCHANGED | REFUSED => ensure!(
                !admitted && refused == m.outcome && m.measurement.is_none(),
                "a move outcome does not re-derive"
            ),
            DUPLICATE => ensure!(
                admitted && m.measurement.is_none() && known,
                "a duplicate move names no earlier measurement"
            ),
            NOT_STARTED => {
                ensure!(
                    admitted
                        && m.measurement.is_none()
                        && !known
                        && s.stopped_by == STOPPED_BY_SECONDS,
                    "an unstarted move lacks a seconds stop"
                );
                self.ended = true;
            }
            // laneD:B19-minimax-linearize
            NO_ALLOWANCE if origin == LINEARIZE => {
                ensure!(
                    admitted
                        && m.measurement.is_none()
                        && !known
                        && s.stopped_by == STOPPED_BY_EVALUATIONS,
                    "a move with no allowance lacks an evaluations stop"
                );
                self.ended = true;
            }
            MEASURED => {
                let index = m.measurement.context("a measured move has no index")?;
                let child = self
                    .measurements
                    .get(index)
                    .context("a move measurement index is out of range")?;
                ensure!(
                    !self.ended
                        && admitted
                        && !known
                        && index == self.known
                        && self.measured.insert(index)
                        && child.observation.origin == origin
                        && Some(child.observation.genome.as_str()) == program
                        && child.observation.generation == m.round,
                    "a measured move does not re-derive"
                );
                // The rewrite rules keep every value bit for bit, so the child scores exactly as
                // its parent: the same status, R², RMSE, selection score and fault counts.
                ensure!(
                    origin != REWRITE || scored(&self.observations[index]) == scored(parent),
                    "a rewritten program does not score bit for bit as its parent"
                );
                self.known += 1;
            }
            _ => bail!("unknown move outcome"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

// laneD:B19-minimax-linearize
#[cfg(test)]
mod linearize_tests;

#[cfg(test)]
mod gp_source_move_capture_tests {
    #[test]
    fn complete_rewrite_and_linearization_proposals_are_captured() {
        if std::env::var_os("JOPE_GP_MOVE_CAPTURE_CHILD").is_some() {
            let profile = feat_math::kernel::profile::shipped_kernel2_profiles()
                .into_iter()
                .find(|p| p.id == "kernel2-baseline")
                .unwrap();
            let evaluator = crate::kernel2::Kernel2Evaluator::new(profile).unwrap();
            let program = "(pipe X (derive expr=\"c0*1.0\" append=false) (ridge))";
            assert!(super::propose(&evaluator, program, 1)
                .unwrap()
                .program
                .is_some());
            let x = (0..16).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
            let cut = crate::prepare::Cut {
                index: 0,
                train_y: x.iter().map(|r| r[0] * 2.0 + 1.0).collect(),
                train_x: x,
                hold_x: vec![],
                hold_y: vec![],
                hold_indices: vec![],
            };
            assert!(super::linearize::propose(
                &evaluator,
                "(pipe X (derive expr=\"sin((0.1*c0))*3.0\" append=false) (ridge))",
                &cut,
                1
            )
            .unwrap()
            .iter()
            .any(|p| p.program.is_some()));
            return;
        }
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "kernel2::search::moves::gp_source_move_capture_tests::complete_rewrite_and_linearization_proposals_are_captured", "--nocapture"])
            .env("JOPE_GP_MOVE_CAPTURE_CHILD", "1").env("JOPE_GP_SOURCE_DIR", directory.path()).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let mut rewrite = 0;
        let mut linearize = 0;
        for entry in std::fs::read_dir(directory.path()).unwrap() {
            for line in std::fs::read_to_string(entry.unwrap().path())
                .unwrap()
                .lines()
            {
                let row: serde_json::Value = serde_json::from_str(line).unwrap();
                feat_math::GenomeIr::parse(row["program"].as_str().unwrap()).unwrap();
                if row["engine"] == "kernel2-move" {
                    let coordinate = row["coordinate"].as_str().unwrap();
                    rewrite += usize::from(coordinate == "rewrite");
                    linearize += usize::from(coordinate.starts_with("linearize/stage/"));
                }
            }
        }
        assert_eq!(rewrite, 1);
        assert!(linearize > 0);
    }
}
