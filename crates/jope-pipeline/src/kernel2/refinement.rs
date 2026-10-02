use super::*;
use anyhow::Context;
use feat_math::{
    dsl::{DslExpr, ParamValue},
    kernel::{
        expression_search::ConstantObjective,
        search::admit_child,
        telemetry::{apply_op_traced, fit_traced, predict_traced},
    },
    math_expr::{Expr, ExpressionScope, MathExpr},
};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingRefinement {
    pub program: String,
    pub evaluations: usize,
    pub rejected_trials: usize,
    pub changed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RefinedChild {
    pub parent_sha256: String,
    pub refinement: TrainingRefinement,
    /// Failed training refinement has no child measurement. Trials remain charged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurement: Option<Kernel2Observation>,
}

impl Kernel2Evaluator {
    /// Refine expression literals in deterministic stage order. This interface
    /// cannot receive held-out rows or labels. Prefixes and whole-pipeline trial
    /// fits use only the supplied training data, with no cache across cuts.
    pub fn refine_training(
        &self,
        text: &str,
        rows: &Rows,
        targets: &[f64],
        trial_limit: usize,
        budget: &mut EvaluationBudget,
    ) -> Result<TrainingRefinement> {
        ensure!(
            !rows.is_empty()
                && rows.len() == targets.len()
                && targets.iter().all(|v| v.is_finite()),
            "refinement needs finite aligned training data"
        );
        let width = rectangular_width(rows)?;
        let GenomeIr::Pipeline(mut pipe) = admit_child(text, &self.profile, width)? else {
            anyhow::bail!("pipeline refinement required");
        };
        let initial = GenomeIr::Pipeline(pipe.clone()).canonical();
        let mut report = TrainingRefinement {
            program: initial.clone(),
            evaluations: 0,
            rejected_trials: 0,
            changed: false,
            failure: None,
        };
        let allowance = trial_limit.min(budget.remaining());
        for index in 0..pipe.stages.len() {
            if report.evaluations == allowance {
                break;
            }
            let Some(expression) = stage_expression(&pipe, index)? else {
                continue;
            };
            if !has_constant(expression.ast()) {
                continue;
            }
            let result = self.refine_stage(
                &mut pipe,
                index,
                &expression,
                rows,
                targets,
                allowance - report.evaluations,
            );
            let (evaluations, rejected) = match result {
                Ok((evaluations, rejected)) => (evaluations, rejected),
                Err(error) => {
                    // E2 errors before a successful result reject the initial
                    // objective; configuration/data/admission were checked above.
                    report.failure = Some(format!("{error:#}"));
                    (1, 1)
                }
            };
            assert!(
                budget.charge(evaluations),
                "optimizer exceeded its supplied allowance"
            );
            report.evaluations += evaluations;
            report.rejected_trials += rejected;
            if report.failure.is_some() {
                break;
            }
        }
        report.program = GenomeIr::Pipeline(pipe).canonical();
        report.changed = report.program != initial;
        Ok(report)
    }

    fn refine_stage(
        &self,
        pipe: &mut DslExpr,
        index: usize,
        expression: &MathExpr,
        rows: &Rows,
        targets: &[f64],
        limit: usize,
    ) -> Result<(usize, usize)> {
        let prefix = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            training_prefix(pipe, index, rows, targets)
        }))
        .map_err(|_| anyhow::anyhow!("training prefix panicked"))??;
        let pool = self
            .profile
            .expr
            .as_ref()
            .context("expression pool required")?;
        let scope = ExpressionScope {
            columns: prefix.iter().map(Vec::len).min().unwrap_or(0),
            elementwise: pipe.stages[index].op == "map",
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        };
        let width = rows.iter().map(Vec::len).min().unwrap_or(0);
        let mut objective = |expr: &MathExpr, _: &[Vec<f64>], _: &[f64]| {
            let mut candidate = pipe.clone();
            set_expression(&mut candidate, index, expr)?;
            let text = GenomeIr::Pipeline(candidate).canonical();
            let genome = admit_child(&text, &self.profile, width)?;
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                training_loss(&genome, rows, targets)
            }))
            .map_err(|_| anyhow::anyhow!("training trial panicked"))?
        };
        let refined = expression.optimize_constants_v2(
            &self.profile,
            scope,
            &prefix,
            targets,
            ConstantObjective::Callback(&mut objective),
            limit,
        )?;
        set_expression(pipe, index, &refined.expression)?;
        // Each trial passed full admission in the callback, including the winner.
        Ok((refined.evaluations, refined.rejected_trials))
    }

    /// Generation-end hook for E6: select the top-k constant-bearing measured
    /// programs, optimize on training data, then re-measure each successful
    /// result like any child. A slot is reserved for that child measurement.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_generation(
        &self,
        observations: &[Kernel2Observation],
        cut: &Cut,
        generation: usize,
        top_k: usize,
        trials_per_program: usize,
        clock: Clock,
        budget: &mut EvaluationBudget,
    ) -> Result<Vec<RefinedChild>> {
        let mut children = Vec::new();
        for parent in self
            .rank(observations)
            .into_iter()
            .filter(|o| contains_constants(&o.observation.genome))
            .take(top_k)
        {
            if budget.remaining() < 2 || trials_per_program == 0 {
                break;
            }
            ensure!(
                parent.observation.cut == cut.index,
                "refinement parent belongs to another cut"
            );
            let limit = trials_per_program.min(budget.remaining() - 1);
            let refinement = self.refine_training(
                &parent.observation.genome,
                &cut.train_x,
                &cut.train_y,
                limit,
                budget,
            )?;
            let measurement = if refinement.failure.is_none() && refinement.evaluations > 0 {
                self.measure(&refinement.program, cut, generation, REFINE, clock, budget)
            } else {
                None
            };
            children.push(RefinedChild {
                parent_sha256: parent.observation.program_sha256.clone(),
                refinement,
                measurement,
            });
        }
        Ok(children)
    }
}

fn training_prefix(pipe: &DslExpr, index: usize, rows: &Rows, targets: &[f64]) -> Result<Rows> {
    let mut current = rows.clone();
    feat_math::sanitize(&mut current);
    let mut faults = FaultTotals::default();
    for stage in &pipe.stages[..index] {
        let op = feat_math::ops::fit_op(stage, &current, targets)?;
        let (next, telemetry) = apply_op_traced(&op, &current);
        faults.add_stage(&telemetry);
        ensure!(
            !faults.is_hard() && faults.domain_cells == 0,
            "faulting training prefix"
        );
        current = next;
    }
    Ok(current)
}

fn training_loss(genome: &GenomeIr, rows: &Rows, targets: &[f64]) -> Result<f64> {
    let (model, fitted) = fit_traced(genome, rows, targets)?;
    let (predictions, predicted) = predict_traced(&model, rows)?;
    let mut faults = FaultTotals::default();
    faults.add_pass(&fitted);
    faults.add_pass(&predicted);
    ensure!(
        !faults.is_hard() && faults.domain_cells == 0,
        "faulting training objective"
    );
    ensure!(
        predictions.len() == targets.len(),
        "unaligned training predictions"
    );
    let loss = predictions
        .iter()
        .zip(targets)
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / targets.len() as f64;
    ensure!(loss.is_finite(), "nonfinite training objective");
    Ok(loss)
}

fn stage_expression(pipe: &DslExpr, index: usize) -> Result<Option<MathExpr>> {
    let stage = &pipe.stages[index];
    if !matches!(stage.op.as_str(), "map" | "derive") {
        return Ok(None);
    }
    let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
        anyhow::bail!("expression carrier needs expr");
    };
    Ok(Some(MathExpr::parse(source)?))
}

fn set_expression(pipe: &mut DslExpr, index: usize, expression: &MathExpr) -> Result<()> {
    pipe.stages[index]
        .params
        .insert("expr".into(), ParamValue::Str(expression.to_source()?));
    Ok(())
}

fn contains_constants(text: &str) -> bool {
    let Ok(GenomeIr::Pipeline(pipe)) = GenomeIr::parse(text) else {
        return false;
    };
    (0..pipe.stages.len()).any(|i| {
        stage_expression(&pipe, i)
            .ok()
            .flatten()
            .is_some_and(|e| has_constant(e.ast()))
    })
}

fn has_constant(root: &Expr) -> bool {
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Const(_) => return true,
            Expr::Var(_) => {}
            Expr::Unary(_, child) => pending.push(child),
            Expr::Binary(_, a, b) => pending.extend([a.as_ref(), b.as_ref()]),
            Expr::Call(_, args) => pending.extend(args),
        }
    }
    false
}
