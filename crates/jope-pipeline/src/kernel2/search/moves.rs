//! Deterministic moves at the end of each search round (laneD:B20-rewrite-simplify).
//!
//! A move proposes children from the round's ranked parents without drawing from any random
//! stream. Each new child the profile admits is measured like every other program: one
//! evaluation from the shared allowance, origin [`SIMPLIFY`], under the start guard refinement
//! uses. Proposal work is not charged. Every proposal is recorded in [`PATH`], which
//! `publication::verify_run` re-derives from the parents alone. With no move selected the search
//! never calls this module and nothing is written.
//!
//! The one move is the exact rewrite pass ([`REWRITE_SATURATE`]): every `map` or `derive` stage's
//! expression of each of the round's top `refinement.top_k` ranked parents with such a stage is
//! simplified by `MathExpr::simplify_v2`, from the expression text alone. A smaller tree the
//! profile refuses is discarded and its reason recorded. A later move adds its pass to
//! [`finish_round`] and its id to the selection record.

use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    dsl::ParamValue,
    kernel::search::admit_child,
    math_expr::{ExpressionScope, MathExpr},
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Duration};

use super::{CutSearch, Options, State};
use crate::{
    clock::Stamp,
    kernel2::{
        receipt::Search, records::Measurement, EngineRecord, Kernel2Evaluator, REWRITE_SATURATE,
        SIMPLIFY,
    },
    prepare::Cut,
    sha256_hex,
};

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
    /// One of [`UNCHANGED`], [`REFUSED`], [`DUPLICATE`], [`NOT_STARTED`] or [`MEASURED`].
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
}

/// The rewrite pass on one expression stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRewrite {
    /// The stage's index in the parent.
    pub stage: usize,
    /// Why saturation stopped: `saturated`, `iteration_limit`, `node_limit` or `malformed`.
    pub stop: String,
    pub iterations: usize,
    pub enodes: usize,
    pub applications: usize,
    pub peak: usize,
    /// The smaller expression the profile admitted, which replaced the stage's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
    /// Why the profile refused the smaller expression, which was discarded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// Whether the search runs [`finish_round`]: some move is selected.
pub(super) fn selected(options: &Options) -> bool {
    options.evaluator.engine().simplify
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
    let mut unique = BTreeSet::new();
    let parents: Vec<(String, String)> = options
        .evaluator
        .rank(&state.observations)
        .into_iter()
        .filter(|o| unique.insert(o.observation.genome.clone()))
        .filter(|o| has_expression_stage(&o.observation.genome))
        .take(options.refinement.top_k)
        .map(|o| {
            (
                o.observation.genome.clone(),
                o.observation.program_sha256.clone(),
            )
        })
        .collect();
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
                        SIMPLIFY,
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

fn has_expression_stage(text: &str) -> bool {
    matches!(GenomeIr::parse(text), Ok(GenomeIr::Pipeline(pipe))
        if pipe.stages.iter().any(|s| matches!(s.op.as_str(), "map" | "derive")))
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
/// measured program passes. It reads no rows: `width` is the input's column count.
pub(crate) fn propose(
    evaluator: &Kernel2Evaluator,
    parent: &str,
    width: usize,
) -> Result<Proposal> {
    let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse(parent)? else {
        bail!("the rewrite pass needs a pipeline program");
    };
    let profile = evaluator.profile();
    let pool = profile.expr.as_ref().context("expression pool required")?;
    let mut rewrites = Vec::new();
    for (index, stage) in pipe.stages.iter_mut().enumerate() {
        if !matches!(stage.op.as_str(), "map" | "derive") {
            continue;
        }
        let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
            bail!("expression stage without an expression");
        };
        let scope = ExpressionScope {
            columns: pool.column_upper_exclusive,
            elementwise: stage.op == "map",
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        };
        let outcome = MathExpr::parse(source)?
            .simplify_v2(profile, scope)
            .context("the rewrite pass refused an admitted parent")?;
        let expression = outcome
            .simplified
            .as_ref()
            .map(MathExpr::to_source)
            .transpose()?;
        if let Some(text) = &expression {
            stage
                .params
                .insert("expr".into(), ParamValue::Str(text.clone()));
        }
        rewrites.push(StageRewrite {
            stage: index,
            stop: outcome.stop.as_str().into(),
            iterations: outcome.iterations,
            enodes: outcome.enodes,
            applications: outcome.applications,
            peak: outcome.peak,
            expression,
            refusal: outcome.refusal,
        });
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

/// Writes the move records under `root` when a move was selected, and returns `engine` with
/// their digest. Without a selected move nothing is written and `engine` is returned as is.
pub(crate) fn publish(
    searched: &CutSearch,
    root: &Path,
    mut engine: Option<EngineRecord>,
) -> Result<Option<EngineRecord>> {
    if let Some(record) = engine.as_mut().and_then(|e| e.moves.as_mut()) {
        let bytes = serde_json::to_vec_pretty(&searched.moves)?;
        std::fs::write(root.join(PATH), &bytes)?;
        record.sha256 = Some(sha256_hex(&bytes));
    }
    Ok(engine)
}

/// Re-derives published move records against the published measurements: each record's proposal
/// from its parent alone, its outcome from the measurements, and every [`SIMPLIFY`] measurement
/// from exactly one record.
pub(crate) fn check(
    moves: &[MoveRecord],
    s: &Search,
    measurements: &[Measurement],
    evaluator: &Kernel2Evaluator,
    width: usize,
) -> Result<()> {
    let ids = s
        .engine_selection
        .as_ref()
        .and_then(|e| e.moves.as_ref())
        .map(|m| m.ids.as_slice())
        .unwrap_or_default();
    let mut measured = BTreeSet::new();
    // The current round and its records so far.
    let mut round = (0usize, 0usize);
    for m in moves {
        ensure!(
            ids.contains(&m.id) && m.id == REWRITE_SATURATE,
            "a move record names an unselected method"
        );
        ensure!(m.round >= round.0, "move records are out of round order");
        round = (m.round, if m.round == round.0 { round.1 + 1 } else { 1 });
        ensure!(
            m.round < s.generations && round.1 <= s.refinement.top_k,
            "move records exceed their rounds"
        );
        let parent = measurements
            .iter()
            .find(|p| {
                p.observation.program_sha256 == m.parent_sha256
                    && p.observation.status == crate::search::MEASURED
                    && p.observation.generation <= m.round
            })
            .context("a move parent was not measured")?;
        let proposal = propose(evaluator, &parent.observation.genome, width)?;
        ensure!(
            proposal.rewrites == m.rewrites
                && proposal.program == m.program
                && proposal.reason == m.reason,
            "a move record does not re-derive from its parent"
        );
        let program = m.program.as_deref();
        let earlier = |end: usize| {
            measurements[..end]
                .iter()
                .any(|o| Some(o.observation.genome.as_str()) == program)
        };
        match m.outcome.as_str() {
            UNCHANGED | REFUSED => ensure!(
                !proposal.admitted && proposal.outcome() == m.outcome && m.measurement.is_none(),
                "a move outcome does not re-derive"
            ),
            DUPLICATE => ensure!(
                proposal.admitted
                    && m.measurement.is_none()
                    && measurements.iter().any(|o| {
                        Some(o.observation.genome.as_str()) == program
                            && o.observation.generation <= m.round
                    }),
                "a duplicate move names no earlier measurement"
            ),
            NOT_STARTED => ensure!(
                proposal.admitted
                    && m.measurement.is_none()
                    && s.stopped_by == crate::search::STOPPED_BY_SECONDS,
                "an unstarted move lacks a seconds stop"
            ),
            MEASURED => {
                let index = m.measurement.context("a measured move has no index")?;
                let child = measurements
                    .get(index)
                    .context("a move measurement index is out of range")?;
                ensure!(
                    proposal.admitted
                        && measured.insert(index)
                        && child.observation.origin == SIMPLIFY
                        && Some(child.observation.genome.as_str()) == program
                        && child.observation.generation == m.round
                        && !earlier(index)
                        && measurements[..index].iter().any(|p| {
                            p.observation.program_sha256 == m.parent_sha256
                                && p.observation.status == crate::search::MEASURED
                        }),
                    "a measured move does not re-derive"
                );
            }
            _ => bail!("unknown move outcome"),
        }
    }
    ensure!(
        measured
            == measurements
                .iter()
                .enumerate()
                .filter(|(_, m)| m.observation.origin == SIMPLIFY)
                .map(|(i, _)| i)
                .collect(),
        "rewrite measurements do not re-derive"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
