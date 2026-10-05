//! Opt-in damped least-squares refinement of a stage's literals (`--refine-method lm`).
//!
//! A `derive` stage's expression is fitted by damped least squares on the stage's training-prefix
//! rows, with the coordinate search's linear-scaled residuals. That fit is a surrogate: the dual
//! Jacobian exists for the expression, not for the whole pipeline. When the surrogate changes the
//! expression, the coordinate search's own whole-pipeline training callback scores the start and
//! then the winner, and the winner replaces the start only on a strict decrease. `map` stages,
//! stages whose remaining allowance cannot hold one surrogate pass beside the two pipeline passes,
//! and shapes the surrogate does not support use the coordinate search unchanged, and the trace
//! names the reason.
//!
//! Budget: each surrogate pass is an objective trial (the initial value, each trial, each Jacobian
//! at one evaluation per literal site, and the re-score), and so is each pipeline pass. A stage
//! reserves two evaluations of its allowance for the pipeline passes. As in the coordinate route,
//! a stage refused before any pass is charged one refused evaluation.
// laneD:B17-lm-refinement begin
use super::*;
use feat_math::kernel::expression_search::{
    GuardedConstantRefinement, GuardedRefinementBudget, LmOptions, LmStatus, LmTrace,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Evaluations of a stage's allowance reserved for the start's and the winner's pipeline passes.
pub(super) const PIPELINE_PASSES: usize = 2;
const _: () = assert!(DAMPED_LS_MIN_TRIALS == PIPELINE_PASSES + 1);

/// Why a stage used the coordinate search.
pub(crate) const ELEMENTWISE: &str = "elementwise stage";
pub(crate) const SMALL_ALLOWANCE: &str = "allowance under three passes";
pub(crate) const UNSUPPORTED: &str = "unsupported expression shape";
pub(crate) const FALLBACK_REASONS: [&str; 3] = [ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED];

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
    /// The stage was refused before any pass: one refused evaluation, as the coordinate route
    /// charges.
    Failed { stage: usize },
    /// Damped least squares on the training-prefix surrogate, then the whole-pipeline check of a
    /// changed winner.
    Surrogate {
        stage: usize,
        surrogate: SurrogateTrace,
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

/// The whole-pipeline training check of a changed surrogate winner: the start's pass, then the
/// winner's.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineCheck {
    pub evaluations: usize,
    pub rejected_trials: usize,
    /// The start's training loss as f64 bits; absent when its pass did not run or was refused.
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
        ensure!(
            self.evaluations <= PIPELINE_PASSES
                && (self.start.is_none() || self.evaluations >= 1)
                && (self.evaluations < 2 || self.start.is_some())
                && (self.winner.is_none() || self.evaluations == 2),
            "pipeline check passes disagree"
        );
        let refused = usize::from(self.evaluations >= 1 && self.start.is_none())
            + usize::from(self.evaluations == 2 && self.winner.is_none());
        let accepted = matches!((self.start, self.winner),
            (Some(start), Some(winner)) if f64::from_bits(winner) < f64::from_bits(start));
        ensure!(
            self.rejected_trials == refused && self.accepted == accepted,
            "pipeline check decision does not re-derive"
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
            Self::Surrogate {
                stage,
                surrogate,
                pipeline,
            } => {
                ensure!(
                    pipeline.is_some() == surrogate.changed,
                    "pipeline check recorded without a changed winner, or missing"
                );
                let (evaluations, rejected) = surrogate.charged(jacobian)?;
                let (more, refused) = pipeline
                    .as_ref()
                    .map_or(Ok((0, 0)), PipelineCheck::charged)?;
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

/// The pipeline check's record and what it means for the stage.
struct Checked {
    record: PipelineCheck,
    stopped: bool,
    failure: Option<String>,
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
            let outcome = self.refine_stage_lm(options, &mut pipe, stage, may_start);
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
    ) -> StageOutcome {
        if pipe.stages[stage.index].op == "map" {
            return self.coordinate_stage(pipe, stage, ELEMENTWISE, may_start);
        }
        if stage.allowance <= PIPELINE_PASSES {
            return self.coordinate_stage(pipe, stage, SMALL_ALLOWANCE, may_start);
        }
        match self.surrogate_stage(options, pipe, stage, may_start) {
            Ok(Some(outcome)) => outcome,
            Ok(None) => self.coordinate_stage(pipe, stage, UNSUPPORTED, may_start),
            Err(error) => StageOutcome {
                evaluations: 1,
                rejected_trials: 1,
                failure: Some(format!("{error:#}")),
                stopped: false,
                trace: StageTrace::Failed { stage: stage.index },
            },
        }
    }

    /// The coordinate route's own stage refinement and charge, unchanged.
    fn coordinate_stage(
        &self,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        reason: &str,
        may_start: &mut dyn FnMut() -> bool,
    ) -> StageOutcome {
        let trial = self
            .refine_stage_guarded(
                pipe,
                stage.index,
                stage.expression,
                stage.training.rows,
                stage.training.targets,
                GuardedRefinementBudget {
                    max_evaluations: stage.allowance,
                    may_start,
                },
            )
            .unwrap_or_else(|error| GuardedConstantRefinement {
                refinement: None,
                stopped: false,
                failure: Some(format!("{error:#}")),
            });
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

    /// `Ok(None)` when the surrogate does not support the shape; it then spent nothing.
    fn surrogate_stage(
        &self,
        options: LmOptions,
        pipe: &mut DslExpr,
        stage: Stage<'_>,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<Option<StageOutcome>> {
        let Training { rows, targets } = stage.training;
        let index = stage.index;
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
        if fitted.trace.status == LmStatus::Unsupported {
            return Ok(None);
        }
        let outcome = fitted.outcome;
        let mut result = StageOutcome {
            evaluations: outcome.evaluations(),
            rejected_trials: outcome.rejected_trials(),
            failure: outcome.failure.clone(),
            stopped: outcome.stopped,
            trace: StageTrace::Failed { stage: index },
        };
        let mut pipeline = None;
        if let Some(refined) = &outcome.refinement {
            // As the coordinate route writes its best measured expression back, this writes the
            // admitted start, or the winner the pipeline check accepted.
            let mut keep = Some(&refined.expression);
            if fitted.trace.changed {
                let checked = self.pipeline_check(
                    pipe,
                    index,
                    &fitted.start,
                    &refined.expression,
                    stage.training,
                    may_start,
                );
                result.evaluations += checked.record.evaluations;
                result.rejected_trials += checked.record.rejected_trials;
                result.stopped |= checked.stopped;
                keep = if checked.failure.is_some() {
                    None
                } else if checked.record.accepted {
                    Some(&refined.expression)
                } else {
                    Some(&fitted.start)
                };
                result.failure = checked.failure;
                pipeline = Some(checked.record);
            }
            if let Some(expression) = keep {
                if let Err(error) = set_expression(pipe, index, expression) {
                    result.failure = Some(format!("{error:#}"));
                }
            }
        }
        result.trace = StageTrace::Surrogate {
            stage: index,
            surrogate: SurrogateTrace::record(&fitted.trace),
            pipeline,
        };
        Ok(Some(result))
    }

    /// The coordinate route's whole-pipeline training callback on the start, then on the winner,
    /// each after its own guard question. A refused start is a failure, as the coordinate route's
    /// refused initial objective is; a refused winner is a refused trial.
    fn pipeline_check(
        &self,
        pipe: &DslExpr,
        index: usize,
        start: &MathExpr,
        winner: &MathExpr,
        training: Training<'_>,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Checked {
        let Training { rows, targets } = training;
        let width = rows.first().map_or(0, Vec::len);
        let loss = |expression: &MathExpr| -> Result<f64> {
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
        };
        let mut checked = Checked {
            record: PipelineCheck {
                evaluations: 0,
                rejected_trials: 0,
                start: None,
                winner: None,
                accepted: false,
            },
            stopped: false,
            failure: None,
        };
        if !may_start() {
            checked.stopped = true;
            return checked;
        }
        checked.record.evaluations += 1;
        let start_loss = match loss(start) {
            Ok(value) => value,
            Err(error) => {
                checked.record.rejected_trials += 1;
                checked.failure = Some(format!("{error:#}"));
                return checked;
            }
        };
        checked.record.start = Some(start_loss.to_bits());
        if !may_start() {
            checked.stopped = true;
            return checked;
        }
        checked.record.evaluations += 1;
        match loss(winner) {
            Ok(value) => {
                checked.record.winner = Some(value.to_bits());
                checked.record.accepted = value < start_loss;
            }
            Err(_) => checked.record.rejected_trials += 1,
        }
        checked
    }
}
// laneD:B17-lm-refinement end
