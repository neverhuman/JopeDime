use super::*;

const HEAD_OPS: [&str; 14] = [
    "ridge",
    "logistic",
    "ridge_gcv",
    "elasticnet",
    "histgbm",
    "gbm",
    "ensemble",
    "ensemble_ridge_gbm",
    "histgbm2",
    "krr",
    "gpr",
    "ksmooth",
    "stack",
    "portfolio",
];

fn get_append(stage: &Stage) -> bool {
    matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
}

pub(crate) fn finite_f64(
    stage: &Stage,
    key: &str,
    default: f64,
    problems: &mut Vec<String>,
) -> f64 {
    let v = stage.get_f64(key, default);
    if !v.is_finite() {
        problems.push(format!("{}: parameter {key} is not finite", stage.op));
    }
    v
}

fn gbm_plan(stage: &Stage, problems: &mut Vec<String>) -> GbmPlan {
    GbmPlan {
        rounds: stage.get_usize("rounds", 150).clamp(1, 500),
        depth: stage.get_usize("depth", 3).clamp(1, 8),
        lr: finite_f64(stage, "lr", 0.1, problems),
        lambda: finite_f64(stage, "lambda", 1.0, problems),
        bins: stage.get_usize("bins", 64).clamp(2, 255),
        min_leaf: stage.get_usize("min_leaf", 10).max(1),
        max_features: stage.get_usize("max_features", 256).max(1),
    }
}

pub fn build_plan(genome: &GenomeIr) -> Result<Plan, Vec<String>> {
    match genome {
        // Explicit nested portfolios are engine-executable but not yet
        // emittable (G5); the compact head form `(pipe X ... (portfolio
        // preset=core ...))` is the supported emission path.
        GenomeIr::Portfolio(_) => Err(vec![
            "portfolio: explicit nested member emission is not supported yet; \
             use the compact head form (pipe X ... (portfolio preset=core ...))"
                .to_string(),
        ]),
        GenomeIr::Symbolic { source, affine, .. } => match MathAst::parse(source) {
            Ok(expr) => Ok(Plan::Symbolic {
                source: source.clone(),
                expr,
                affine: *affine,
            }),
            Err(e) => Err(vec![format!("symbolic expression: {e}")]),
        },
        GenomeIr::Pipeline(dsl) => {
            let mut problems: Vec<String> = Vec::new();
            // Same split semantics as model.rs::split_target: an optional
            // leading target pseudo-stage; anywhere else it is an error.
            let (target_stage, rest) = match dsl.stages.first() {
                Some(first) if crate::target_transform::is_target_op(&first.op) => {
                    (Some(first), &dsl.stages[1..])
                }
                _ => (None, &dsl.stages[..]),
            };
            if rest
                .iter()
                .any(|s| crate::target_transform::is_target_op(&s.op))
            {
                problems.push("target: transform must be the first stage".to_string());
            }
            let target = target_stage.and_then(|stage| plan_target(stage, &mut problems));
            let (transforms, head_stage) = match rest.last() {
                Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
                    (&rest[..rest.len() - 1], Some(last.clone()))
                }
                _ => (rest, None),
            };
            let mut stages = Vec::with_capacity(transforms.len());
            for stage in transforms {
                if HEAD_OPS.contains(&stage.op.as_str()) {
                    problems.push(format!(
                        "{}: model heads are only valid as the final stage",
                        stage.op
                    ));
                    continue;
                }
                match plan_op(stage, &mut problems) {
                    Some(op) => stages.push(PlannedStage {
                        op,
                        append: get_append(stage),
                    }),
                    None => problems.push(stage.op.clone()),
                }
            }
            let head = match &head_stage {
                None => PlannedHead::Ridge { lambda_frac: 0.02 },
                Some(stage) => match stage.op.as_str() {
                    "portfolio" => plan_portfolio_head(stage, &mut problems),
                    // Defaults and clamps mirror battle-runtime heads.rs
                    // (shared with portfolio member planning).
                    _ => match plan_head_stage(stage, &mut problems) {
                        Some(h) => h,
                        None => {
                            problems.push(stage.op.clone());
                            PlannedHead::Ridge { lambda_frac: 0.02 }
                        }
                    },
                },
            };
            if problems.is_empty() {
                Ok(Plan::Pipeline {
                    target,
                    stages,
                    head,
                })
            } else {
                Err(problems)
            }
        }
    }
}
