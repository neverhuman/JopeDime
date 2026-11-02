use super::rust_input::*;
use super::*;

pub(super) fn rust_linear_predictor(
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    probability: bool,
    input: &str,
) -> String {
    format!(
        "{{ let means: &[f64] = &{means}; let scales: &[f64] = &{scales}; let weights: &[f64] = &{weights}; let mut value = weights[0]; for j in 0..means.len() {{ value += weights[j+1] * (({input}[j]-means[j])/scales[j]).clamp({lo}, {hi}); }} {finish} }}",
        means = rust_f64_slice(means),
        scales = rust_f64_slice(scales),
        weights = rust_f64_slice(weights),
        lo = if probability { "-30.0" } else { "-20.0" },
        hi = if probability { "30.0" } else { "20.0" },
        finish = if probability { "sigmoid(value)" } else { "value.clamp(-60.0, 60.0)" },
    )
}

pub(super) fn rust_tree_expr(node: &ExtraNode, input: &str) -> String {
    match node {
        ExtraNode::Leaf(value) => rust_f64(*value),
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => format!(
            "if if {input}[{feature}].is_finite() {{ {input}[{feature}] <= {threshold} }} else {{ {missing_left} }} {{ {left} }} else {{ {right} }}",
            threshold = rust_f64(*threshold),
            left = rust_tree_expr(left, input),
            right = rust_tree_expr(right, input),
        ),
    }
}

pub(super) fn rust_head_body(head: &V3Head) -> Result<String> {
    let source = match head {
        V3Head::LogitRidge(model) => {
            rust_linear_predictor(&model.means, &model.scales, &model.weights, true, "x")
        }
        V3Head::Legacy(FittedHead::Ridge(model)) => {
            rust_linear_predictor(&model.means, &model.scales, &model.w, false, "x")
        }
        V3Head::TabBoost(model) => {
            let mut body = format!("{{ let mut value = {};", rust_f64(model.base));
            for stump in &model.stumps {
                body.push_str(&format!(
                    " value += {lr} * if if x[{feature}].is_finite() {{ x[{feature}] <= {threshold} }} else {{ {missing_left} }} {{ {left} }} else {{ {right} }};",
                    lr = rust_f64(model.learning_rate),
                    feature = stump.feature,
                    threshold = rust_f64(stump.threshold),
                    missing_left = stump.missing_left,
                    left = rust_f64(stump.left),
                    right = rust_f64(stump.right),
                ));
            }
            body.push_str(if model.binary {
                " sigmoid(value) }"
            } else {
                " finite(value) }"
            });
            body
        }
        V3Head::ExtraTrees(model) => {
            let trees = model
                .trees
                .iter()
                .map(|tree| format!("({})", rust_tree_expr(tree, "x")))
                .collect::<Vec<_>>();
            let sum = if trees.is_empty() {
                "0.0".to_string()
            } else {
                trees.join(" + ")
            };
            if model.binary {
                format!(
                    "(({sum}) / {}.0_f64).clamp(0.0_f64, 1.0_f64)",
                    trees.len().max(1)
                )
            } else {
                format!("finite(({sum}) / {}.0)", trees.len().max(1))
            }
        }
        V3Head::GamBoost(model) => {
            let mut body = format!("{{ let mut value = {};", rust_f64(model.base));
            for (index, term) in model.terms.iter().enumerate() {
                body.push_str(&format!(
                    " let edges_{index}: &[f64] = &{edges}; let values_{index}: &[f64] = &{values}; let bin_{index} = edges_{index}.partition_point(|edge| *edge < x[{feature}]).min(values_{index}.len()-1); value += {lr} * values_{index}[bin_{index}];",
                    edges = rust_f64_slice(&term.edges),
                    values = rust_f64_slice(&term.values),
                    feature = term.feature,
                    lr = rust_f64(model.learning_rate),
                ));
            }
            body.push_str(if model.binary {
                " sigmoid(value) }"
            } else {
                " finite(value) }"
            });
            body
        }
        V3Head::Pls(model) => {
            let V3Op::PlsProject { means, weights, .. } = &model.projection else {
                unreachable!()
            };
            let mut body = String::from("{ let projected: Vec<f64> = vec![");
            body.push_str(
                &weights
                    .iter()
                    .map(|weight| {
                        format!(
                            "dot_centered(x, &{}, &{})",
                            rust_f64_slice(means),
                            rust_f64_slice(weight)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            );
            body.push_str("]; ");
            body.push_str(&rust_linear_predictor(
                &model.ridge.means,
                &model.ridge.scales,
                &model.ridge.w,
                false,
                "projected",
            ));
            body.push_str(" }");
            body
        }
        V3Head::ClassificationPortfolio(model) => {
            let mut terms = Vec::with_capacity(model.members.len());
            for (weight, member) in model.weights.iter().zip(&model.members) {
                terms.push(format!(
                    "{} * ({})",
                    rust_f64(*weight),
                    rust_head_body(member)?
                ));
            }
            format!("({}).clamp(0.0, 1.0)", terms.join(" + "))
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by scalar bound emission")
        }
        V3Head::Legacy(_) => bail!("this fitted legacy head is not supported by bound emission"),
    };
    Ok(source)
}

pub(super) fn rust_calibration_body(calibration: &CalibrationModel, binary: bool) -> String {
    let calibrated = match calibration {
        CalibrationModel::None => "value".to_string(),
        CalibrationModel::Platt { slope, intercept } => format!(
            "{{ let p = value.clamp(1e-12, 1.0-1e-12); sigmoid({} * (p/(1.0-p)).ln() + {}) }}",
            rust_f64(*slope),
            rust_f64(*intercept)
        ),
        CalibrationModel::Isotonic { upper, values } => format!(
            "{{ let upper: &[f64] = &{}; let values: &[f64] = &{}; let index = upper.partition_point(|bound| *bound < value).min(values.len()-1); values[index] }}",
            rust_f64_slice(upper),
            rust_f64_slice(values)
        ),
    };
    if binary {
        format!("({calibrated}).clamp(0.0, 1.0)")
    } else {
        format!("finite({calibrated})")
    }
}

pub(super) fn rust_target_body(target: &Option<FittedTarget>, input: &str) -> String {
    match target {
        None => input.to_string(),
        Some(FittedTarget::Log1p) => {
            format!("({input}.clamp(-745.0, 700.0).exp() - 1.0).clamp(-60.0, 60.0)")
        }
        Some(FittedTarget::Winsor { .. }) => format!("{input}.clamp(-60.0, 60.0)"),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "({input} * {} + {}).clamp({}, {})",
            rust_f64(*sigma),
            rust_f64(*mu),
            rust_f64(*clamp_lo),
            rust_f64(*clamp_hi)
        ),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "rank_inverse({input}, &{}, &{}).clamp(-60.0, 60.0)",
            rust_f64_slice(table_scores),
            rust_f64_slice(table_values)
        ),
    }
}
