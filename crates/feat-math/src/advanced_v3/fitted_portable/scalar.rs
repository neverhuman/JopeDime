//! Allowlisted scalar fitted state; no training statistics or recipe metadata.
use super::{floats, linear, Linear};
use crate::advanced_v3::{
    heads::{ExtraNode, V3Head},
    transforms::V3Op,
    types::TaskKind,
};
use anyhow::{bail, ensure, Result};
use serde::Serialize;

#[derive(Serialize)]
pub(super) enum Branch {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        absent_left: bool,
        left: Box<Branch>,
        right: Box<Branch>,
    },
}
#[derive(Serialize)]
pub(super) struct Rule {
    feature: usize,
    threshold: f64,
    absent_left: bool,
    left: f64,
    right: f64,
}
#[derive(Serialize)]
pub(super) struct Bins {
    feature: usize,
    edges: Vec<f64>,
    values: Vec<f64>,
}
#[derive(Serialize)]
pub(super) enum Scalar {
    Probability(Linear),
    Rules {
        base: f64,
        rate: f64,
        binary: bool,
        rules: Vec<Rule>,
    },
    Average {
        binary: bool,
        branches: Vec<Branch>,
    },
    Bins {
        base: f64,
        rate: f64,
        binary: bool,
        terms: Vec<Bins>,
    },
    Projection {
        means: Vec<f64>,
        weights: Vec<Vec<f64>>,
        output: Linear,
    },
    Combination {
        weights: Vec<f64>,
        members: Vec<Scalar>,
    },
}
pub(super) fn supports(head: &V3Head, task: TaskKind) -> bool {
    match head {
        V3Head::LogitRidge(_) | V3Head::ClassificationPortfolio(_) => task == TaskKind::Binary,
        V3Head::TabBoost(model) => {
            model.binary == (task == TaskKind::Binary)
                && matches!(task, TaskKind::Regression | TaskKind::Binary)
        }
        V3Head::ExtraTrees(model) => {
            model.binary == (task == TaskKind::Binary)
                && matches!(task, TaskKind::Regression | TaskKind::Binary)
        }
        V3Head::GamBoost(model) => {
            model.binary == (task == TaskKind::Binary)
                && matches!(task, TaskKind::Regression | TaskKind::Binary)
        }
        V3Head::Pls(_) => task == TaskKind::Regression,
        _ => false,
    }
}
fn branch(node: &ExtraNode) -> Branch {
    match node {
        ExtraNode::Leaf(value) => Branch::Leaf(*value),
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => Branch::Split {
            feature: *feature,
            threshold: *threshold,
            absent_left: *missing_left,
            left: Box::new(branch(left)),
            right: Box::new(branch(right)),
        },
    }
}
pub(super) fn lower(head: &V3Head, task: TaskKind) -> Result<Scalar> {
    ensure!(supports(head, task), "scalar fitted head/task differs");
    Ok(match head {
        V3Head::LogitRidge(model) => Scalar::Probability(linear(
            &model.means,
            &model.scales,
            &model.weights,
            30.,
            false,
        )),
        V3Head::TabBoost(model) => Scalar::Rules {
            base: model.base,
            rate: model.learning_rate,
            binary: model.binary,
            rules: model
                .stumps
                .iter()
                .map(|r| Rule {
                    feature: r.feature,
                    threshold: r.threshold,
                    absent_left: r.missing_left,
                    left: r.left,
                    right: r.right,
                })
                .collect(),
        },
        V3Head::ExtraTrees(model) => Scalar::Average {
            binary: model.binary,
            branches: model.trees.iter().map(branch).collect(),
        },
        V3Head::GamBoost(model) => Scalar::Bins {
            base: model.base,
            rate: model.learning_rate,
            binary: model.binary,
            terms: model
                .terms
                .iter()
                .map(|t| Bins {
                    feature: t.feature,
                    edges: t.edges.clone(),
                    values: t.values.clone(),
                })
                .collect(),
        },
        V3Head::Pls(model) => {
            let V3Op::PlsProject {
                means,
                weights,
                append,
            } = &model.projection
            else {
                bail!("fitted projection variant differs")
            };
            ensure!(!append, "fitted projection unexpectedly appends inputs");
            Scalar::Projection {
                means: means.clone(),
                weights: weights.clone(),
                output: linear(
                    &model.ridge.means,
                    &model.ridge.scales,
                    &model.ridge.w,
                    20.,
                    false,
                ),
            }
        }
        V3Head::ClassificationPortfolio(model) => Scalar::Combination {
            weights: model.weights.clone(),
            members: model
                .members
                .iter()
                .map(|m| lower(m, TaskKind::Binary))
                .collect::<Result<_>>()?,
        },
        _ => bail!("unimplemented scalar fitted state"),
    })
}
fn check_linear(model: &Linear, width: usize) -> Result<()> {
    ensure!(
        model.means.len() == width
            && model.scales.len() == width
            && model.weights.len() == width + 1,
        "scalar fitted linear dimensions differ"
    );
    ensure!(
        model
            .means
            .iter()
            .chain(&model.weights)
            .all(|v| v.is_finite())
            && model.scales.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid scalar fitted linear coefficients"
    );
    Ok(())
}
fn check_branch(node: &Branch, width: usize) -> Result<()> {
    match node {
        Branch::Leaf(value) => ensure!(value.is_finite(), "nonfinite fitted branch value"),
        Branch::Split {
            feature,
            threshold,
            left,
            right,
            ..
        } => {
            ensure!(
                *feature < width && threshold.is_finite(),
                "invalid fitted branch predicate"
            );
            check_branch(left, width)?;
            check_branch(right, width)?;
        }
    }
    Ok(())
}
pub(super) fn validate(model: &Scalar, width: usize) -> Result<()> {
    match model {
        Scalar::Probability(value) => check_linear(value, width)?,
        Scalar::Rules {
            base, rate, rules, ..
        } => {
            ensure!(
                base.is_finite() && rate.is_finite(),
                "invalid fitted sum coefficients"
            );
            ensure!(
                rules.iter().all(|r| r.feature < width
                    && r.threshold.is_finite()
                    && r.left.is_finite()
                    && r.right.is_finite()),
                "invalid fitted rule"
            );
        }
        Scalar::Average { branches, .. } => {
            for b in branches {
                check_branch(b, width)?;
            }
        }
        Scalar::Bins {
            base, rate, terms, ..
        } => {
            ensure!(
                base.is_finite() && rate.is_finite(),
                "invalid fitted bin coefficients"
            );
            for t in terms {
                ensure!(
                    t.feature < width
                        && !t.values.is_empty()
                        && t.values.len() == t.edges.len() + 1
                        && t.edges.iter().chain(&t.values).all(|v| v.is_finite())
                        && t.edges.windows(2).all(|p| p[0] <= p[1]),
                    "invalid fitted bin state"
                );
            }
        }
        Scalar::Projection {
            means,
            weights,
            output,
        } => {
            ensure!(
                means.len() == width
                    && means.iter().all(|v| v.is_finite())
                    && weights
                        .iter()
                        .all(|r| r.len() == width && r.iter().all(|v| v.is_finite())),
                "invalid fitted projection dimensions/coefficients"
            );
            check_linear(output, weights.len())?;
        }
        Scalar::Combination { weights, members } => {
            ensure!(
                !members.is_empty()
                    && weights.len() == members.len()
                    && weights.iter().all(|v| v.is_finite() && *v >= 0.),
                "invalid fitted combination weights"
            );
            for m in members {
                validate(m, width)?;
            }
        }
    }
    Ok(())
}
fn number(value: f64) -> String {
    format!("f64::from_bits(0x{:016x})", value.to_bits())
}
fn linear_expr(model: &Linear, x: &str) -> String {
    format!(
        "linear({x},{},{},{},{},{})",
        floats(&model.means),
        floats(&model.scales),
        floats(&model.weights),
        number(model.bound),
        model.finite_input
    )
}
fn branch_expr(node: &Branch, x: &str) -> String {
    match node{
        Branch::Leaf(value)=>number(*value),
        Branch::Split{feature,threshold,absent_left,left,right}=>format!("if if {x}[{feature}].is_finite() {{ {x}[{feature}] <= {} }} else {{ {absent_left} }} {{ {} }} else {{ {} }}",number(*threshold),branch_expr(left,x),branch_expr(right,x)),
    }
}
fn finish(binary: bool) -> &'static str {
    if binary {
        "sigmoid(value)"
    } else {
        "finite(value)"
    }
}
pub(super) fn rust(model: &Scalar, x: &str) -> String {
    match model {
        Scalar::Probability(value) => format!("sigmoid({})", linear_expr(value, x)),
        Scalar::Rules {
            base,
            rate,
            binary,
            rules,
        } => {
            let mut body = format!("{{ let mut value={};", number(*base));
            for r in rules {
                body.push_str(&format!("value += {} * if if {x}[{}].is_finite() {{ {x}[{}] <= {} }} else {{ {} }} {{ {} }} else {{ {} }};",number(*rate),r.feature,r.feature,number(r.threshold),r.absent_left,number(r.left),number(r.right)));
            }
            body.push_str(finish(*binary));
            body.push('}');
            body
        }
        Scalar::Average { binary, branches } => {
            let mut body = "{ let mut value=-0.0_f64;".to_owned();
            for b in branches {
                body.push_str(&format!("value += ({});", branch_expr(b, x)));
            }
            body.push_str(&format!(
                "value /= {}.0_f64; {} }}",
                branches.len().max(1),
                if *binary {
                    "value.clamp(0.0,1.0)"
                } else {
                    "finite(value)"
                }
            ));
            body
        }
        Scalar::Bins {
            base,
            rate,
            binary,
            terms,
        } => {
            let mut body = format!("{{ let mut value={};", number(*base));
            for t in terms {
                body.push_str(&format!("{{let edges:&[f64]={};let values:&[f64]={};let bin=edges.partition_point(|edge|*edge<{x}[{}]).min(values.len()-1);value += {}*values[bin];}}",floats(&t.edges),floats(&t.values),t.feature,number(*rate)));
            }
            body.push_str(finish(*binary));
            body.push('}');
            body
        }
        Scalar::Projection {
            means,
            weights,
            output,
        } => {
            let rows=weights.iter().map(|w|format!("{{let means:&[f64]={};let weights:&[f64]={};let mut value=-0.0_f64;for j in 0..weights.len(){{value += weights[j]*({x}[j]-means[j]);}}finite(value)}}",floats(means),floats(w))).collect::<Vec<_>>().join(",");
            format!(
                "{{let projected:Vec<f64>=vec![{rows}]; {}.clamp(-60.0,60.0)}}",
                linear_expr(output, "&projected")
            )
        }
        Scalar::Combination { weights, members } => {
            let mut body = "{let mut value=-0.0_f64;".to_owned();
            for (w, m) in weights.iter().zip(members) {
                body.push_str(&format!("value += {}*({});", number(*w), rust(m, x)));
            }
            body.push_str("value.clamp(0.0,1.0)}");
            body
        }
    }
}
