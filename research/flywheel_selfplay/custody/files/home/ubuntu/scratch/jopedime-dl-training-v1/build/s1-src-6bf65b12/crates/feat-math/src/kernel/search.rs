//! Strict Track E0 admission for kernel2 profile children, before fitting or scoring.
//! Legacy wrappers and their deliberately lossy printer remain unchanged. This
//! seam alone does not qualify shipping generation or emitter closure.

use super::capabilities::{self, Kernel2Use, Tier, BINOPS, FUNCTIONS, VARIABLES};
use super::pipeline::{stage_class, StageClass};
use super::profile::{ExprPool, Limit, Negation, SearchProfile};
use super::{scope, Error, Program, Result};
use crate::dsl::{DslExpr, ParamValue};
use crate::math_expr::{Expr, ExpressionScope, MathExpr, UnOp, Var};
use crate::GenomeIr;

mod wrappers;
pub use wrappers::{crossover_programs, mutate_program, random_program, Offspring};

/// Admit a child only when printing and parsing preserve its exact typed codec,
/// every read is in scope, and its elements and budgets belong to the profile.
/// Birth parameter domains are draw distributions, not post-edit numeric limits.
/// Revision 1's legacy declarations retain their separate historical APIs.
pub fn admit_child(text: &str, profile: &SearchProfile, input_width: usize) -> Result<GenomeIr> {
    if text.len() > super::codec::MAX_BYTES {
        return Err(Error::Resource("child exceeds program byte budget".into()));
    }
    let child = GenomeIr::parse(text).map_err(|e| Error::Syntax(e.to_string()))?;
    let printed = child.canonical();
    let reparsed = GenomeIr::parse(&printed)
        .map_err(|e| Error::Syntax(format!("child is not a canonical fixed point: {e}")))?;
    if Program::import_legacy(&child).to_bytes()? != Program::import_legacy(&reparsed).to_bytes()? {
        return Err(Error::Syntax("child is not a canonical fixed point".into()));
    }
    scope::validate_scope(&child, input_width).map_err(|e| Error::Shape(e.to_string()))?;
    let tier = match profile.id {
        "kernel2-baseline" => Tier::Baseline,
        "kernel2-experimental" => Tier::Experimental,
        _ => return Err(Error::Unsupported("kernel2 search profile required".into())),
    };
    let GenomeIr::Pipeline(pipe) = &child else {
        return Err(Error::Unsupported("profile requires a pipeline".into()));
    };
    check_pipeline(pipe, profile, tier)?;
    Ok(child)
}

fn check_pipeline(pipe: &DslExpr, profile: &SearchProfile, tier: Tier) -> Result<()> {
    let Some((head, stages)) = pipe.stages.split_last() else {
        return Err(Error::Unsupported("terminal profile head required".into()));
    };
    if stage_class(&head.op) != Some(StageClass::Learner)
        || !profile.heads.contains(&head.op.as_str())
    {
        return Err(Error::Unsupported("terminal profile head required".into()));
    }
    check_operator(&head.op, tier)?;
    let Limit::Fixed(max_transforms) = profile.max_transforms.limit else {
        return Err(Error::Bounds(
            "profile requires a fixed transform cap".into(),
        ));
    };
    if stages.len() > max_transforms {
        return Err(Error::Bounds("profile transform cap exceeded".into()));
    }
    let mut expression_stages = 0;
    for stage in stages {
        check_operator(&stage.op, tier)?;
        // Runtime carrier roles are independent of a caller's profile lists.
        if matches!(stage.op.as_str(), "map" | "derive") {
            let carrier = profile
                .expr_stages
                .as_ref()
                .filter(|e| e.carriers.contains(&stage.op.as_str()))
                .ok_or_else(|| Error::Unsupported("profile expression carrier required".into()))?;
            expression_stages += 1;
            if expression_stages > carrier.max_expr_stages {
                return Err(Error::Bounds(
                    "profile expression stage cap exceeded".into(),
                ));
            }
            let pool = profile
                .expr
                .as_ref()
                .ok_or_else(|| Error::Unsupported("profile expression pool required".into()))?;
            let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
                return Err(Error::Syntax("expression carrier needs string expr".into()));
            };
            let expression = MathExpr::parse(source).map_err(|e| Error::Syntax(e.to_string()))?;
            check_expression(&expression, pool, stage.op == "map", tier)?;
        } else if !profile.stages.contains(&stage.op.as_str())
            && !profile
                .expansion
                .as_ref()
                .is_some_and(|e| e.contains(&stage.op.as_str()))
        {
            return Err(Error::Unsupported(format!("profile stage: {}", stage.op)));
        }
    }
    Ok(())
}

fn check_operator(name: &str, tier: Tier) -> Result<()> {
    let allowed = capabilities::op_capability(name).is_some_and(|cap| {
        matches!(cap.kernel2, Kernel2Use::Baseline)
            || matches!(
                (cap.kernel2, tier),
                (Kernel2Use::Experimental, Tier::Experimental)
            )
    });
    if !allowed {
        return Err(Error::Unsupported(format!(
            "stage outside manifest search admission: {name}"
        )));
    }
    Ok(())
}

pub(crate) fn check_expression(
    expr: &MathExpr,
    pool: &ExprPool,
    elementwise: bool,
    tier: Tier,
) -> Result<()> {
    if pool.max_nodes == 0 || pool.max_nodes > 15 || pool.max_depth == 0 || pool.max_depth > 4 {
        return Err(Error::Bounds(
            "profile exceeds expression search caps".into(),
        ));
    }
    ExpressionScope {
        columns: pool.column_upper_exclusive,
        elementwise,
        max_nodes: pool.max_nodes,
        max_depth: pool.max_depth,
    }
    .validate(expr)
    .map_err(|e| Error::Bounds(format!("profile expression caps: {e}")))?;
    let mut variable_leaf = false;
    let mut pending = vec![expr.ast()];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Const(value) => check_constant(*value, pool)?,
            Expr::Var(variable) => {
                check_variable(*variable, pool, tier)?;
                variable_leaf = true;
            }
            Expr::Unary(UnOp::Neg, value) => match (pool.negation, value.as_ref()) {
                (Negation::ConstantsOnly, Expr::Const(value)) => check_constant(-value, pool)?,
                _ => {
                    return Err(Error::Unsupported(
                        "profile permits a negative constant only".into(),
                    ))
                }
            },
            Expr::Binary(op, a, b) => {
                let allowed = BINOPS.iter().any(|(variant, name, cap)| {
                    variant == op && pool.binops.contains(name) && cap.class.admits(tier)
                });
                if !allowed {
                    return Err(Error::Unsupported("profile binary operator".into()));
                }
                pending.extend([a.as_ref(), b.as_ref()]);
            }
            Expr::Call(func, args) => {
                let allowed = FUNCTIONS.iter().any(|(variant, name, cap)| {
                    variant == func && pool.functions.contains(name) && cap.class.admits(tier)
                });
                if !allowed {
                    return Err(Error::Unsupported("profile function".into()));
                }
                pending.extend(args);
            }
        }
    }
    if !variable_leaf {
        return Err(Error::Unsupported(
            "expression requires a variable leaf".into(),
        ));
    }
    Ok(())
}

fn check_constant(value: f64, pool: &ExprPool) -> Result<()> {
    if !value.is_finite() {
        return Err(Error::NonFinite("profile constant must be finite".into()));
    }
    if !pool.continuous_after_birth
        && !pool
            .birth_constants
            .iter()
            .any(|v| v.to_bits() == value.to_bits())
    {
        return Err(Error::Unsupported(
            "constant outside profile birth bins".into(),
        ));
    }
    Ok(())
}

fn check_variable(variable: Var, pool: &ExprPool, tier: Tier) -> Result<()> {
    let member = match variable {
        Var::Col(i) => i < pool.column_upper_exclusive,
        other => VARIABLES
            .iter()
            .any(|(variant, name, _)| *variant == other && pool.variables.contains(name)),
    };
    if !member || !capabilities::var_capability(variable).class.admits(tier) {
        return Err(Error::Unsupported("profile variable".into()));
    }
    Ok(())
}
