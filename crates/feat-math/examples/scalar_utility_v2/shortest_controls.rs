//! Frozen-primitive formula references for the TRAIN-only utility comparison.
//! These supplement the unchanged equal-budget A/B searches. The logistic
//! reference is an approximation to probit, not an inverse-CDF equivalence.
use feat_math::kernel::scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode};
use feat_math::math_expr::{BinOp, Func, Var};

pub struct FormulaReference {
    pub name: String,
    pub compared_functions: Vec<u16>,
    pub note: &'static str,
    pub program: ScalarExpressionProgram,
}

fn call(function: Func, child: ScalarNode) -> ScalarNode {
    ScalarNode::Call(ScalarFunction::V1(function), vec![child])
}

fn sub(left: ScalarNode, right: ScalarNode) -> ScalarNode {
    ScalarNode::Binary(BinOp::Sub, Box::new(left), Box::new(right))
}

fn neg(child: ScalarNode) -> ScalarNode {
    // Express negation using the old subtraction primitive. These controls
    // execute as explicit legacy formula oracles, outside producer offspring.
    sub(ScalarNode::Const(0.0), child)
}

fn reference(
    name: String,
    compared_functions: Vec<u16>,
    note: &'static str,
    root: ScalarNode,
) -> Result<FormulaReference, String> {
    let program = ScalarExpressionProgram::new(root).map_err(|error| error.to_string())?;
    Ok(FormulaReference {
        name,
        compared_functions,
        note,
        program,
    })
}

/// Formula minimality refers to the stated expression form and frozen
/// primitives, not an exhaustive optimality claim over all possible old trees.
/// Every reference uses LegacyV1 semantics, including its guards and clamps.
pub fn references(card: usize, width: usize) -> Result<Vec<FormulaReference>, String> {
    if card > 2 || !(1..=512).contains(&width) {
        return Err("formula-reference card or width".into());
    }
    let mut out = vec![];
    // The shipped base profile admits at most eight explicit column leaves.
    // Row reductions below still consume the complete declared input width.
    for column in 0..width.min(8) {
        let x = || ScalarNode::Var(Var::Col(column));
        match card {
            0 => {
                out.push(reference(
                    format!("logit-log1p-col-{column}"),
                    vec![19, 20],
                    "Seven-node log(p)-log1p(0-p): the logit formula on 0<p<1 before legacy guards. Zero-minus uses old subtraction; this manual formula is outside the frozen generator catalogue. For probit it is a logistic surrogate; fitted standardization absorbs its scale. It does not reproduce the inverse-CDF kernel or its tail/fault semantics.",
                    sub(call(Func::Log, x()), call(Func::Log1p, neg(x()))),
                )?);
                out.push(reference(
                    format!("logit-ratio-col-{column}"),
                    vec![19, 20],
                    "Six-node log(p/(1-p)): an alternative old logit form on 0<p<1 away from legacy division/log guards; a logistic surrogate for probit. The guarded operator and IEEE rounding are compared, not assumed equivalent.",
                    call(
                        Func::Log,
                        ScalarNode::Binary(
                            BinOp::Div,
                            Box::new(x()),
                            Box::new(sub(ScalarNode::Const(1.0), x())),
                        ),
                    ),
                )?);
            }
            1 => {
                out.push(reference(
                    format!("expm1-exp-sub-col-{column}"),
                    vec![21],
                    "Four-node exp(x)-1: the shortest direct old exponential-minus-one formula. Retains cancellation and legacy exp argument clamping.",
                    sub(call(Func::Exp, x()), ScalarNode::Const(1.0)),
                )?);
                out.push(reference(
                    format!("log1mexp-log1p-col-{column}"),
                    vec![22],
                    "Five-node log1p(0-exp(x)): an old log1mexp formula for x<0 before guards, with zero-minus expressed by old subtraction; this manual formula is outside the frozen generator catalogue. Retains old exp clamping and log1p domain projection; it has no new branch kernel.",
                    call(Func::Log1p, neg(call(Func::Exp, x()))),
                )?);
                out.push(reference(
                    format!("log1mexp-log-sub-col-{column}"),
                    vec![22],
                    "Five-node log(1-exp(x)): the usual cancellation-prone alternative with frozen old exp/log guards. Both old forms are measured separately.",
                    call(Func::Log, sub(ScalarNode::Const(1.0), call(Func::Exp, x()))),
                )?);
            }
            _ => out.push(reference(
                format!("asinh-legacy-col-{column}"),
                vec![13],
                "Two-node legacy asinh: the existing primitive is already the minimal direct tree. Comparison isolates its overflowing formula from the opt-in numerical policy.",
                call(Func::Asinh, x()),
            )?),
        }
    }
    if card == 2 {
        for (name, var) in [
            ("sum", Var::RowSum),
            ("mean", Var::RowMean),
            ("l1", Var::RowL1),
            ("l2", Var::RowL2),
        ] {
            out.push(reference(
                format!("row-{name}-legacy"),
                vec![],
                "One-node frozen legacy row reduction: the minimal existing primitive. It evaluates the full input row and retains old overflow/cancellation/fault behavior.",
                ScalarNode::Var(var),
            )?);
        }
    }
    Ok(out)
}
