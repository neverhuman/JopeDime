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

/// Evaluations of a stage's allowance kept for the start's and the winner's pipeline passes.
pub(crate) const PIPELINE_PASSES: usize = 2;

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
/// closed-form program would reach the expression parser inside `GenomeIr::parse`, so it is
/// refused unparsed, by its own message. Each stage's `expr` is then refused when it holds more
/// than [`MAX_PARENT_EXPRESSION_NESTING`] nesting bytes, before any caller parses it.
pub(crate) fn parse_untrusted_pipeline(text: &str) -> Result<DslExpr> {
    ensure!(
        text.trim().starts_with("(pipe"),
        "program is not a pipeline: not in the pipeline form, refused unparsed"
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
pub(crate) struct Training<'a> {
    pub(crate) rows: &'a Rows,
    pub(crate) targets: &'a [f64],
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
    pub(crate) fn refine_training_lm_guarded(
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
        let program = GenomeIr::Pipeline(candidate).canonical();
        feat_math::capture_gp_source!(
            "kernel2-refinement",
            &format!("lm-whole-pipeline/stage/{index}"),
            &program
        );
        let genome = admit_child(&program, self.profile(), width)?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            training_loss(&genome, rows, targets)
        }))
        .map_err(|_| anyhow::anyhow!("training trial panicked"))?
    }
}
// laneD:B17-lm-refinement end

mod trace;
pub use trace::{IterationBits, PipelineCheck, RefinementTrace, StageTrace, SurrogateTrace};
