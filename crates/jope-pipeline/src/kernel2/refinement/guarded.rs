//! Caller admission for E6 training work and generation child measurement.
use super::*;
use feat_math::kernel::expression_search::{GuardedConstantRefinement, GuardedRefinementBudget};

#[derive(Clone, Debug, Serialize)]
pub struct GuardedTrainingRefinement {
    pub refinement: TrainingRefinement,
    /// The caller refused new work; completed trials remain in the report.
    pub stopped: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct GuardedGeneration {
    pub children: Vec<RefinedChild>,
    pub stopped: bool,
}

impl Kernel2Evaluator {
    /// Check admission before preparing each training prefix and before every
    /// numerical trial. The caller supplies its cancellation/deadline predicate.
    /// A started operation completes; a false predicate starts no further work.
    #[allow(clippy::too_many_arguments)]
    pub fn refine_training_guarded(
        &self,
        text: &str,
        rows: &Rows,
        targets: &[f64],
        trial_limit: usize,
        budget: &mut EvaluationBudget,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<GuardedTrainingRefinement> {
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
            },
            stopped: false,
        };
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
            let trial = self.refine_stage_guarded(
                &mut pipe,
                index,
                &expression,
                rows,
                targets,
                GuardedRefinementBudget {
                    max_evaluations: remaining,
                    may_start,
                },
            );
            let trial = trial.unwrap_or_else(|error| GuardedConstantRefinement {
                refinement: None,
                stopped: false,
                failure: Some(format!("{error:#}")),
            });
            assert!(
                budget.charge(trial.evaluations()),
                "optimizer exceeded its supplied allowance"
            );
            result.refinement.evaluations += trial.evaluations();
            result.refinement.rejected_trials += trial.rejected_trials();
            result.refinement.failure = trial.failure;
            result.stopped = trial.stopped;
        }
        result.refinement.program = GenomeIr::Pipeline(pipe).canonical();
        result.refinement.changed = result.refinement.program != initial;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_stage_guarded(
        &self,
        pipe: &mut DslExpr,
        index: usize,
        expression: &MathExpr,
        rows: &Rows,
        targets: &[f64],
        budget: GuardedRefinementBudget<'_>,
    ) -> Result<GuardedConstantRefinement> {
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
            elementwise: pipe.stages[index].op == "map",
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        };
        let width = rows.first().map_or(0, Vec::len);
        let mut objective = |expr: &MathExpr, _: &[Vec<f64>], _: &[f64]| {
            let mut candidate = pipe.clone();
            set_expression(&mut candidate, index, expr)?;
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
        let trial = expression.optimize_constants_v2_guarded(
            self.profile(),
            scope,
            &prefix,
            targets,
            ConstantObjective::Callback(&mut objective),
            budget,
        )?;
        if let Some(best) = &trial.refinement {
            set_expression(pipe, index, &best.expression)?;
        }
        Ok(trial)
    }

    /// Reserve child measurement within the shared allowance. A stopped
    /// refinement retains its training report without reading held-out data.
    /// Recheck admission before remeasurement and never resume after a stop.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_generation_guarded(
        &self,
        observations: &[Kernel2Observation],
        cut: &Cut,
        generation: usize,
        top_k: usize,
        trials_per_program: usize,
        clock: Clock,
        budget: &mut EvaluationBudget,
        may_start: &mut dyn FnMut() -> bool,
    ) -> Result<GuardedGeneration> {
        let mut result = GuardedGeneration {
            children: Vec::new(),
            stopped: false,
        };
        let parents = self
            .rank(observations)
            .into_iter()
            .filter(|parent| contains_constants(&parent.observation.genome))
            .take(top_k);
        for parent in parents {
            if result.stopped || trials_per_program == 0 || budget.remaining() < 2 {
                break;
            }
            ensure!(
                parent.observation.cut == cut.index,
                "refinement parent belongs to another cut"
            );
            let trained = self.refine_training_guarded(
                &parent.observation.genome,
                &cut.train_x,
                &cut.train_y,
                trials_per_program.min(budget.remaining() - 1),
                budget,
                may_start,
            )?;
            result.stopped = trained.stopped;
            let mut measurement = None;
            if !result.stopped
                && trained.refinement.failure.is_none()
                && trained.refinement.evaluations > 0
            {
                if may_start() {
                    measurement = self.measure(
                        &trained.refinement.program,
                        cut,
                        generation,
                        REFINE,
                        clock,
                        budget,
                    );
                } else {
                    result.stopped = true;
                }
            }
            result.children.push(RefinedChild {
                parent_sha256: parent.observation.program_sha256.clone(),
                refinement: trained.refinement,
                measurement,
            });
        }
        Ok(result)
    }
}
