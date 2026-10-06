//! Opt-in damped least-squares refinement of a stage's literals (`--refine-method lm`).
//!
//! A `derive` stage is refined in three steps. The start's whole-pipeline pass comes first: it is
//! the coordinate search's own first evaluation of the stage (training prefix, admission, guard
//! question, objective, charge and failure), so a start that fails the pipeline objective ends the
//! stage exactly as in the coordinate route. Then the surrogate: the stage's expression fitted by
//! damped least squares on the stage's training-prefix rows, with the coordinate search's
//! linear-scaled residuals. The dual Jacobian exists for the expression, not for the whole
//! pipeline. When the surrogate changes the expression, the winner's whole-pipeline pass follows,
//! and the winner replaces the start only on a strictly lower training loss.
//!
//! `map` stages, stages whose shape the surrogate does not support, and stages whose remaining
//! allowance is under the surrogate minimum use the coordinate search unchanged, and the trace
//! names the reason. The minimum ([`stage_minimum`]) is the fewest evaluations with which the
//! surrogate can change a stage: the two pipeline passes, then the surrogate's initial value, one
//! Jacobian, one trial and the winner's re-score. Under it the surrogate could only spend passes.
//!
//! Budget: each pipeline pass is an objective trial, and so is each surrogate pass (the initial
//! value, each trial, each Jacobian at one evaluation per literal site under the default charge,
//! and the re-score). A stage spends at most its allowance: the start's pass, the surrogate within
//! the allowance less the two pipeline passes, and the winner's pass.
// laneD:B17-lm-refinement begin
use super::*;
use feat_math::kernel::expression_search::{
    GuardedConstantRefinement, GuardedRefinementBudget, LmJacobianCharge, LmOptions, LmStatus,
    LmTrace, LM_MAX_SITES,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Evaluations of a stage's allowance kept for the start's and the winner's pipeline passes.
pub(super) const PIPELINE_PASSES: usize = 2;

/// The surrogate passes a change needs besides one Jacobian: its initial value, one trial and the
/// winner's re-score (`optimize_constants_lm_guarded` keeps one evaluation for the re-score).
const SURROGATE_PASSES: usize = 3;

/// The fewest evaluations of a stage's allowance with which the surrogate can change the stage:
/// the two pipeline passes, and the surrogate's initial value, one Jacobian at `jacobian_cost`,
/// one trial and the re-score. A stage with a smaller allowance uses the coordinate search.
pub(crate) const fn stage_minimum(jacobian_cost: usize) -> usize {
    PIPELINE_PASSES + SURROGATE_PASSES + jacobian_cost
}

/// What one Jacobian of an expression with `sites` literal sites is charged.
fn jacobian_cost(charge: LmJacobianCharge, sites: usize) -> usize {
    match charge {
        LmJacobianCharge::Sites => sites,
        LmJacobianCharge::One => 1,
    }
}

/// Why a stage used the coordinate search.
pub(crate) const ELEMENTWISE: &str = "elementwise stage";
pub(crate) const SMALL_ALLOWANCE: &str = "allowance under the surrogate minimum";
pub(crate) const UNSUPPORTED: &str = "unsupported expression shape";
pub(crate) const FALLBACK_REASONS: [&str; 3] = [ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED];

/// The most bytes among `( + - * / ^` an expression source may hold in a program read back from
/// evidence: a refinement parent the trace check binds, or an observation the archive check
/// rebuilds. Each of these bytes opens at most one level of the expression parser's recursion (a
/// fixed chain of calls) and of the tree it builds, and no other byte opens one. The parser checks
/// its node and depth caps only after it has built the tree, and unary plus and redundant
/// parentheses never reach the tree. Such a program is untrusted, so its sources are counted
/// before they are parsed, and a source over this count is refused unparsed. A profile expression
/// (at most 15 nodes) holds far fewer.
pub(crate) const MAX_PARENT_EXPRESSION_NESTING: usize = 256;

/// The bytes of `source` that can open a level of the expression parser's recursion.
fn expression_nesting(source: &str) -> usize {
    source
        .bytes()
        .filter(|b| matches!(b, b'(' | b'+' | b'-' | b'*' | b'/' | b'^'))
        .count()
}

/// `text`, a pipeline program read back from evidence, parsed within the bounds the expression
/// parser needs. Only the pipeline form, whose own parser does not recurse, is parsed: a
/// closed-form program would reach the expression parser inside `GenomeIr::parse`. Each stage's
/// `expr` is then refused when it holds more than [`MAX_PARENT_EXPRESSION_NESTING`] nesting bytes,
/// before any caller parses it.
pub(crate) fn parse_untrusted_pipeline(text: &str) -> Result<DslExpr> {
    ensure!(
        text.trim().starts_with("(pipe"),
        "program is not a pipeline"
    );
    let GenomeIr::Pipeline(pipe) = GenomeIr::parse(text)? else {
        anyhow::bail!("program is not a pipeline");
    };
    for stage in &pipe.stages {
        if let Some(ParamValue::Str(source)) = stage.params.get("expr") {
            ensure!(
                expression_nesting(source) <= MAX_PARENT_EXPRESSION_NESTING,
                "program expression nests too deeply to parse"
            );
        }
    }
    Ok(pipe)
}

/// The training rows and targets a refinement receives; nothing else reaches it.
#[derive(Clone, Copy)]
pub(super) struct Training<'a> {
    pub(super) rows: &'a Rows,
    pub(super) targets: &'a [f64],
}

/// The per-stage record of a damped least-squares refinement, kept with its training report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementTrace {
    /// The method id.
    pub method: String,
    /// One entry per stage refined, in stage order.
    pub stages: Vec<StageTrace>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StageTrace {
    /// The coordinate search refined the stage, for the reason named.
    Coordinate {
        stage: usize,
        reason: String,
        evaluations: usize,
        rejected_trials: usize,
    },
    /// The start's whole-pipeline pass, or the training prefix before it, was refused: one refused
    /// evaluation, as the coordinate route charges, and the refinement failed.
    Failed { stage: usize },
    /// The guard refused the start's whole-pipeline pass: nothing was spent, and the refinement
    /// stopped.
    Stopped { stage: usize },
    /// The start's whole-pipeline pass, damped least squares on the training-prefix surrogate,
    /// then the whole-pipeline pass of a changed winner.
    Surrogate {
        stage: usize,
        surrogate: SurrogateTrace,
        /// The pipeline passes; a surrogate stage always records them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pipeline: Option<PipelineCheck>,
    },
}

/// The surrogate fit's counts and iterations. Its charge is
/// `initial + trials + jacobian_cost * jacobians + rescore`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurrogateTrace {
    pub status: String,
    pub changed: bool,
    pub sites: usize,
    pub jacobian_cost: usize,
    pub initial: usize,
    pub trials: usize,
    pub jacobians: usize,
    pub rescore: usize,
    pub rejected_trials: usize,
    pub solve_failures: usize,
    pub accepted: usize,
    /// The winner's surrogate re-score, as f64 bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rescored: Option<u64>,
    pub nonsmooth_rows: usize,
    pub saturated_rows: usize,
    pub iterations: Vec<IterationBits>,
}

/// One surrogate iteration with its values as f64 bits, so a replay compares them exactly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IterationBits {
    pub mu: u64,
    pub objective: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial: Option<u64>,
    pub accepted: bool,
    pub solved: bool,
}

/// The whole-pipeline training passes of a surrogate stage: the start's, which runs before the
/// surrogate, and a changed winner's, which runs after it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineCheck {
    /// 1, or 2 when the winner's pass ran.
    pub evaluations: usize,
    /// 1 when the winner's pass was refused.
    pub rejected_trials: usize,
    /// The start's training loss as f64 bits; a surrogate stage's start passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u64>,
    /// The winner's training loss as f64 bits; absent when its pass did not run or was refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<u64>,
    /// The winner replaced the start: its training loss is strictly lower.
    pub accepted: bool,
}

impl SurrogateTrace {
    fn record(trace: &LmTrace) -> Self {
        Self {
            status: trace.status.as_str().into(),
            changed: trace.changed,
            sites: trace.sites,
            jacobian_cost: trace.jacobian_cost,
            initial: trace.initial,
            trials: trace.trials,
            jacobians: trace.jacobians,
            rescore: trace.rescore,
            rejected_trials: trace.rejected_trials,
            solve_failures: trace.solve_failures,
            accepted: trace.accepted,
            rescored: trace.rescored.map(f64::to_bits),
            nonsmooth_rows: trace.nonsmooth_rows,
            saturated_rows: trace.saturated_rows,
            iterations: trace
                .iterations
                .iter()
                .map(|i| IterationBits {
                    mu: i.mu.to_bits(),
                    objective: i.objective.to_bits(),
                    trial: i.trial.map(f64::to_bits),
                    accepted: i.accepted,
                    solved: i.solved,
                })
                .collect(),
        }
    }

    /// The evaluations and refusals charged, re-derived from the counts, after checking the counts
    /// against the iterations. `jacobian` is the selection's charge, `sites` or `one`.
    fn charged(&self, jacobian: &str) -> Result<(usize, usize)> {
        let status = LmStatus::ALL
            .into_iter()
            .find(|s| s.as_str() == self.status)
            .filter(|s| *s != LmStatus::Unsupported)
            .context("unknown surrogate status")?;
        let cost = match jacobian {
            "sites" => self.sites,
            "one" => 1,
            _ => anyhow::bail!("unknown refinement jacobian charge"),
        };
        ensure!(
            self.sites > 0 && self.jacobian_cost == cost,
            "surrogate jacobian charge disagrees"
        );
        ensure!(
            self.initial <= 1
                && self.rescore <= self.initial
                && (self.rescored.is_none() || self.rescore == 1)
                && (!self.changed || self.rescored.is_some()),
            "surrogate start or re-score disagrees"
        );
        let started = self.trials > 0 || self.jacobians > 0 || self.solve_failures > 0;
        ensure!(
            !started || self.initial == 1,
            "surrogate passes without an initial value"
        );
        if status == LmStatus::InitialRefused {
            ensure!(
                self.initial == 1 && !started && self.rescore == 0,
                "a refused start has passes after it"
            );
        }
        // The start's Jacobian comes before any iteration, and every accepted step is followed by
        // one, except a last step whose Jacobian the budget, the guard or an exact fit stopped; so
        // an accepted step with an iteration after it had its Jacobian. The re-score runs only
        // after an accepted step.
        let followed = self
            .iterations
            .iter()
            .rev()
            .skip(1)
            .filter(|i| i.accepted)
            .count();
        // The counts are untrusted here, so `jacobians - accepted` is taken checked: it is 0 or 1
        // exactly when `accepted <= jacobians <= accepted + 1`, with no overflow at any value.
        ensure!(
            self.jacobians
                .checked_sub(self.accepted)
                .is_some_and(|extra| extra <= 1)
                && (self.iterations.is_empty() || self.jacobians > followed)
                && (self.rescore == 0 || self.accepted > 0),
            "surrogate Jacobians or re-score disagree with its accepted steps"
        );
        ensure!(
            Some(self.iterations.len()) == self.trials.checked_add(self.solve_failures),
            "surrogate iterations disagree with trials and failed solves"
        );
        ensure!(
            self.iterations
                .iter()
                .all(|i| (i.solved || (i.trial.is_none() && !i.accepted))
                    && (!i.accepted || i.trial.is_some()))
                && self.iterations.iter().filter(|i| !i.solved).count() == self.solve_failures
                && self.iterations.iter().filter(|i| i.accepted).count() == self.accepted,
            "surrogate iterations disagree with their counts"
        );
        let refused = self
            .iterations
            .iter()
            .filter(|i| i.solved && i.trial.is_none())
            .count();
        ensure!(
            self.rejected_trials
                == refused
                    + usize::from(status == LmStatus::InitialRefused)
                    + usize::from(self.rescore == 1 && self.rescored.is_none()),
            "surrogate refusals disagree with its iterations"
        );
        let evaluations = cost
            .checked_mul(self.jacobians)
            .and_then(|j| j.checked_add(self.trials))
            .and_then(|e| e.checked_add(self.initial + self.rescore))
            .context("surrogate count overflow")?;
        Ok((evaluations, self.rejected_trials))
    }
}

impl PipelineCheck {
    fn charged(&self) -> Result<(usize, usize)> {
        let finite = |bits: Option<u64>| bits.is_none_or(|b| f64::from_bits(b).is_finite());
        ensure!(
            self.start.is_some()
                && finite(self.start)
                && finite(self.winner)
                && (1..=PIPELINE_PASSES).contains(&self.evaluations)
                && (self.winner.is_none() || self.evaluations == PIPELINE_PASSES),
            "pipeline passes disagree"
        );
        let refused = usize::from(self.evaluations == PIPELINE_PASSES && self.winner.is_none());
        let accepted = matches!((self.start, self.winner),
            (Some(start), Some(winner)) if f64::from_bits(winner) < f64::from_bits(start));
        ensure!(
            self.rejected_trials == refused && self.accepted == accepted,
            "pipeline decision does not re-derive"
        );
        Ok((self.evaluations, self.rejected_trials))
    }
}

impl StageTrace {
    fn charged(&self, jacobian: &str) -> Result<(usize, usize, usize)> {
        match self {
            Self::Coordinate {
                stage,
                reason,
                evaluations,
                rejected_trials,
            } => {
                ensure!(
                    FALLBACK_REASONS.contains(&reason.as_str()) && rejected_trials <= evaluations,
                    "invalid coordinate stage record"
                );
                Ok((*stage, *evaluations, *rejected_trials))
            }
            Self::Failed { stage } => Ok((*stage, 1, 1)),
            Self::Stopped { stage } => Ok((*stage, 0, 0)),
            Self::Surrogate {
                stage,
                surrogate,
                pipeline,
            } => {
                let pipeline = pipeline
                    .as_ref()
                    .context("surrogate stage without its pipeline passes")?;
                ensure!(
                    pipeline.evaluations < PIPELINE_PASSES || surrogate.changed,
                    "a winner's pipeline pass without a changed surrogate"
                );
                let (evaluations, rejected) = surrogate.charged(jacobian)?;
                let (more, refused) = pipeline.charged()?;
                Ok((
                    *stage,
                    evaluations
                        .checked_add(more)
                        .context("trace count overflow")?,
                    rejected
                        .checked_add(refused)
                        .context("trace count overflow")?,
                ))
            }
        }
    }
}

impl RefinementTrace {
    /// The evaluations and refusals the stages were charged, after checking every stage record.
    /// `jacobian` is the selection's charge, `sites` or `one`.
    pub(crate) fn charged(&self, jacobian: &str) -> Result<(usize, usize)> {
        ensure!(self.method == DAMPED_LS, "unknown refinement trace method");
        let (mut evaluations, mut rejected) = (0usize, 0usize);
        let mut last: Option<usize> = None;
        for stage in &self.stages {
            let (index, more, refused) = stage.charged(jacobian)?;
            ensure!(
                last.is_none_or(|previous| index > previous),
                "refinement trace stages out of order"
            );
            last = Some(index);
            evaluations = evaluations
                .checked_add(more)
                .context("trace count overflow")?;
            rejected = rejected
                .checked_add(refused)
                .context("trace count overflow")?;
        }
        Ok((evaluations, rejected))
    }

    /// [`Self::charged`], with every stage record bound to the `parent` program it refined under at
    /// most `trials_per_program` trials. The records are the parent's constant-bearing expression
    /// stages in order, as the route visits them. A `map` stage, and only a `map` stage, is an
    /// elementwise fallback. A `derive` stage is a fallback for an unsupported shape only outside
    /// the surrogate's site range, and one for a small allowance only within it and with fewer
    /// evaluations than its minimum. A surrogate stage, or a failed or stopped start, is a
    /// supported `derive` stage whose allowance could hold the minimum, and a surrogate stage
    /// reports the stage's literal sites and spent at most that allowance less the pipeline passes.
    /// A record that ends the refinement (a failed or stopped start, a stopped surrogate, or a
    /// changed winner whose pass did not run) is the last.
    pub(crate) fn check(
        &self,
        parent: &str,
        jacobian: &str,
        trials_per_program: usize,
    ) -> Result<(usize, usize)> {
        let charged = self.charged(jacobian)?;
        let charge = match jacobian {
            "sites" => LmJacobianCharge::Sites,
            "one" => LmJacobianCharge::One,
            _ => anyhow::bail!("unknown refinement jacobian charge"),
        };
        let pipe = parse_untrusted_pipeline(parent).context("refinement parent")?;
        let mut refined = Vec::new();
        for index in 0..pipe.stages.len() {
            if let Some(expression) = stage_expression(&pipe, index)? {
                if has_constant(expression.ast()) {
                    refined.push((index, expression.literal_sites()));
                }
            }
        }
        ensure!(
            self.stages.len() <= refined.len(),
            "refinement trace has more stages than the parent refines"
        );
        let mut spent = 0usize;
        for (position, (record, &(index, sites))) in self.stages.iter().zip(&refined).enumerate() {
            let (stage, evaluations, _) = record.charged(jacobian)?;
            ensure!(
                stage == index,
                "refinement trace stage is not the parent's next expression stage"
            );
            let op = pipe.stages[index].op.as_str();
            let last = position + 1 == self.stages.len();
            let supported = (1..=LM_MAX_SITES).contains(&sites);
            let minimum = stage_minimum(jacobian_cost(charge, sites));
            // The largest allowance the stage can have had.
            let room = trials_per_program
                .checked_sub(spent)
                .context("refinement trace overspends its allowance")?;
            let fitted = op == "derive" && supported && room >= minimum;
            let valid = match record {
                StageTrace::Coordinate { reason, .. } => match reason.as_str() {
                    ELEMENTWISE => op == "map",
                    UNSUPPORTED => op == "derive" && !supported,
                    SMALL_ALLOWANCE => op == "derive" && supported && evaluations < minimum,
                    _ => false,
                },
                StageTrace::Failed { .. } | StageTrace::Stopped { .. } => fitted && last,
                StageTrace::Surrogate {
                    surrogate,
                    pipeline,
                    ..
                } => {
                    let passes = pipeline.as_ref().map_or(0, |p| p.evaluations);
                    let ends = surrogate.status == LmStatus::Stopped.as_str()
                        || (surrogate.changed && passes < PIPELINE_PASSES);
                    fitted
                        && surrogate.sites == sites
                        && (evaluations - passes)
                            .checked_add(PIPELINE_PASSES)
                            .is_some_and(|needed| needed <= room)
                        && (last || !ends)
                }
            };
            ensure!(
                valid,
                "refinement trace stage {index} does not fit the parent program"
            );
            spent = spent
                .checked_add(evaluations)
                .context("trace count overflow")?;
        }
        Ok(charged)
    }
}

/// One expression stage's refinement request.
#[derive(Clone, Copy)]
struct Stage<'a> {
    index: usize,
    expression: &'a MathExpr,
    training: Training<'a>,
    allowance: usize,
}

struct StageOutcome {
    evaluations: usize,
    rejected_trials: usize,
    failure: Option<String>,
    stopped: bool,
    trace: StageTrace,
}

impl Kernel2Evaluator {
    /// `refine_training_guarded` with the damped least-squares method: the same admission,
    /// allowance, guard and stage order, with each expression stage refined as the module
    /// documentation describes. The trace is attached once a trial was charged.
    pub(super) fn refine_training_lm_guarded(
        &self,
        options: LmOptions,
        text: &str,
        training: Training<'_>,
        trial_limit: usize,
        budget: &mut EvaluationBudget,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<GuardedTrainingRefinement> {
        let Training { rows, targets } = training;
        ensure!(
            !rows.is_empty()
                && rows.len() == targets.len()
                && targets.iter().all(|y| y.is_finite()),
            "refinement needs finite aligned training data"
        );
        let width = rectangular_width(rows)?;
        let GenomeIr::Pipeline(mut pipe) = admit_child(text, self.profile(), width)? else {
            anyhow::bail!("pipeline refinement required");
        };
        let initial = GenomeIr::Pipeline(pipe.clone()).canonical();
        let mut result = GuardedTrainingRefinement {
            refinement: TrainingRefinement {
                program: initial.clone(),
                evaluations: 0,
                rejected_trials: 0,
                changed: false,
                failure: None,
                trace: None,
            },
            stopped: false,
        };
        let mut stages = Vec::new();
        let allowance = budget.remaining().min(trial_limit);
        for index in 0..pipe.stages.len() {
            let remaining = allowance - result.refinement.evaluations;
            if remaining == 0 || result.stopped || result.refinement.failure.is_some() {
                break;
            }
            let Some(expression) = stage_expression(&pipe, index)? else {
                continue;
            };
            if !has_constant(expression.ast()) {
                continue;
            }
            if !may_start() {
                result.stopped = true;
                break;
            }
            let stage = Stage {
                index,
                expression: &expression,
                training,
                allowance: remaining,
            };
            let outcome = self.refine_stage_lm(options, &mut pipe, stage, may_start)?;
            assert!(
                budget.charge(outcome.evaluations),
                "optimizer exceeded its supplied allowance"
            );
            result.refinement.evaluations += outcome.evaluations;
            result.refinement.rejected_trials += outcome.rejected_trials;
            result.refinement.failure = outcome.failure;
            result.stopped = outcome.stopped;
            stages.push(outcome.trace);
        }
        result.refinement.program = GenomeIr::Pipeline(pipe).canonical();
        result.refinement.changed = result.refinement.program != initial;
        if result.refinement.evaluations > 0 {
            result.refinement.trace = Some(RefinementTrace {
                method: DAMPED_LS.into(),
                stages,
            });
        }
        Ok(result)
    }

    fn refine_stage_lm(
        &self,
        options: LmOptions,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<StageOutcome> {
        if pipe.stages[stage.index].op == "map" {
            return Ok(self.coordinate_stage(pipe, stage, ELEMENTWISE, may_start));
        }
        let sites = stage.expression.literal_sites();
        if !(1..=LM_MAX_SITES).contains(&sites) {
            return Ok(self.coordinate_stage(pipe, stage, UNSUPPORTED, may_start));
        }
        if stage.allowance < stage_minimum(jacobian_cost(options.jacobian, sites)) {
            return Ok(self.coordinate_stage(pipe, stage, SMALL_ALLOWANCE, may_start));
        }
        self.surrogate_stage(options, pipe, stage, may_start)
    }

    /// The coordinate route's own stage refinement with `max_evaluations`, and its charge: an error
    /// before a result is a refused initial objective.
    fn coordinate_trial(
        &self,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        max_evaluations: usize,
        may_start: &mut dyn FnMut() -> bool,
    ) -> GuardedConstantRefinement {
        self.refine_stage_guarded(
            pipe,
            stage.index,
            stage.expression,
            stage.training.rows,
            stage.training.targets,
            GuardedRefinementBudget {
                max_evaluations,
                may_start,
            },
        )
        .unwrap_or_else(|error| GuardedConstantRefinement {
            refinement: None,
            stopped: false,
            failure: Some(format!("{error:#}")),
        })
    }

    /// The coordinate route's own stage refinement and charge, unchanged.
    fn coordinate_stage(
        &self,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        reason: &str,
        may_start: &mut dyn FnMut() -> bool,
    ) -> StageOutcome {
        let trial = self.coordinate_trial(pipe, stage, stage.allowance, may_start);
        let (evaluations, rejected_trials) = (trial.evaluations(), trial.rejected_trials());
        StageOutcome {
            evaluations,
            rejected_trials,
            failure: trial.failure,
            stopped: trial.stopped,
            trace: StageTrace::Coordinate {
                stage: stage.index,
                reason: reason.into(),
                evaluations,
                rejected_trials,
            },
        }
    }

    /// The start's pipeline pass, the surrogate, and a changed winner's pipeline pass. An error is
    /// an internal fault: the start's pass already prepared the same prefix and admitted the same
    /// start under the same profile and scope, so the surrogate cannot refuse its configuration.
    fn surrogate_stage(
        &self,
        options: LmOptions,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<StageOutcome> {
        let index = stage.index;
        // The coordinate route's first evaluation, which writes the admitted start back.
        let start = self.coordinate_trial(pipe, stage, 1, may_start);
        let Some(started) = &start.refinement else {
            let trace = if start.failure.is_some() {
                StageTrace::Failed { stage: index }
            } else {
                StageTrace::Stopped { stage: index }
            };
            return Ok(StageOutcome {
                evaluations: start.evaluations(),
                rejected_trials: start.rejected_trials(),
                failure: start.failure,
                stopped: start.stopped,
                trace,
            });
        };
        let start_loss = started.score;
        let Training { rows, targets } = stage.training;
        let prefix = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            training_prefix(pipe, index, rows, targets)
        }))
        .map_err(|_| anyhow::anyhow!("training prefix panicked"))??;
        let pool = self
            .profile()
            .expr
            .as_ref()
            .context("expression pool required")?;
        let scope = ExpressionScope {
            columns: prefix.iter().map(Vec::len).min().unwrap_or(0),
            elementwise: false,
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        };
        let fitted = stage.expression.optimize_constants_lm_guarded(
            self.profile(),
            scope,
            &prefix,
            targets,
            options,
            GuardedRefinementBudget {
                max_evaluations: stage.allowance - PIPELINE_PASSES,
                may_start,
            },
        )?;
        ensure!(
            fitted.trace.status != LmStatus::Unsupported,
            "the surrogate refused a supported shape"
        );
        let outcome = &fitted.outcome;
        let mut pipeline = PipelineCheck {
            evaluations: 1,
            rejected_trials: 0,
            start: Some(start_loss.to_bits()),
            winner: None,
            accepted: false,
        };
        // A surrogate that refuses its own start (`initial_refused`) changes nothing and is no
        // failure: the start passed the pipeline objective, and the stage keeps it.
        let (mut stopped, mut failure) = (outcome.stopped, None);
        if fitted.trace.changed {
            let winner = &outcome
                .refinement
                .as_ref()
                .context("a changed surrogate without its winner")?
                .expression;
            if may_start() {
                pipeline.evaluations += 1;
                match self.pipeline_loss(pipe, index, winner, stage.training) {
                    Ok(loss) => {
                        pipeline.winner = Some(loss.to_bits());
                        // Only a strictly lower loss replaces the start; a tie keeps it.
                        pipeline.accepted = loss < start_loss;
                    }
                    Err(_) => pipeline.rejected_trials += 1,
                }
            } else {
                stopped = true;
            }
            if pipeline.accepted {
                if let Err(error) = set_expression(pipe, index, winner) {
                    failure = Some(format!("{error:#}"));
                }
            }
        }
        Ok(StageOutcome {
            evaluations: outcome.evaluations() + pipeline.evaluations,
            rejected_trials: outcome.rejected_trials() + pipeline.rejected_trials,
            failure,
            stopped,
            trace: StageTrace::Surrogate {
                stage: index,
                surrogate: SurrogateTrace::record(&fitted.trace),
                pipeline: Some(pipeline),
            },
        })
    }

    /// The coordinate route's whole-pipeline training objective with `expression` at `index`.
    fn pipeline_loss(
        &self,
        pipe: &DslExpr,
        index: usize,
        expression: &MathExpr,
        training: Training<'_>,
    ) -> Result<f64> {
        let Training { rows, targets } = training;
        let width = rows.first().map_or(0, Vec::len);
        let mut candidate = pipe.clone();
        set_expression(&mut candidate, index, expression)?;
        let genome = admit_child(
            &GenomeIr::Pipeline(candidate).canonical(),
            self.profile(),
            width,
        )?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            training_loss(&genome, rows, targets)
        }))
        .map_err(|_| anyhow::anyhow!("training trial panicked"))?
    }
}
// laneD:B17-lm-refinement end
