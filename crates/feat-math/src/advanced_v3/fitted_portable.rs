//! Explicitly bounded fitted inference lowering. It carries only inference state.
use super::calibration::CalibrationModel;
use super::heads::V3Head;
use super::input::{CategoricalMode, FittedColumn, FittedRowOrder};
use super::pipeline::{BoundModel, BoundPredictor};
use super::schema::{ColumnKind, RowSemantics};
use super::types::TaskKind;
use crate::heads::FittedHead;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
mod scalar;

mod admission;
mod render;

pub use admission::*;
use render::*;

#[derive(Clone, Debug)]
pub struct FittedPortableSources {
    pub python: String,
    pub rust: String,
    pub javascript: Option<String>,
    /// Matrix columns for multiclass, or [0, 1] for binary; empty for regression.
    pub class_codes: Vec<i64>,
}

#[derive(Serialize)]
enum Input {
    Numeric {
        fill: f64,
        indicator: bool,
        restore_missing: bool,
    },
    Onehot {
        vocabulary: Vec<String>,
        indices: Vec<usize>,
    },
}

#[derive(Serialize)]
struct Linear {
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    bound: f64,
    finite_input: bool,
}

#[derive(Serialize)]
enum Head {
    Scalar {
        binary: bool,
        state: scalar::Scalar,
    },
    Regression(Linear),
    Binary(Linear),
    Softmax(Vec<Linear>),
    Ovr(Vec<Linear>),
    Lda {
        centers: Vec<Vec<f64>>,
        scales: Vec<f64>,
        priors: Vec<f64>,
    },
    Centroid {
        centers: Vec<Vec<f64>>,
        scales: Vec<f64>,
    },
}

#[derive(Serialize)]
struct State {
    input: Vec<Input>,
    head: Head,
}

fn linear(
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    bound: f64,
    finite_input: bool,
) -> Linear {
    Linear {
        means: means.to_vec(),
        scales: scales.to_vec(),
        weights: weights.to_vec(),
        bound,
        finite_input,
    }
}

/// A narrow, fail-closed emitter for a fitted IID input followed directly by a linear
/// regression, logistic, bounded scalar, or one of the four multiclass heads. No operations, calibration,
/// target mapping, historical state, or missing-aware routing are silently dropped.
/// The two admitted branch-based heads preserve native numeric missing masks.
/// Each function takes the existing encoded RawTable rows and returns a matrix: one
/// value per regression row, [1-p,p] for binary, and all fitted multiclass columns.
/// Python/Rust use their platform libm. JavaScript uses ECMAScript math; callers must
/// qualify it with an explicit numerical tolerance, never infer bit parity.
pub fn emit_fitted_portable(model: &BoundModel) -> Result<FittedPortableSources> {
    ensure!(
        fitted_portable_unsupported(model)?.is_none(),
        "outside portable fitted subset"
    );
    ensure!(
        matches!(model.schema.row_semantics, RowSemantics::Iid),
        "portable fitted input requires IID rows"
    );
    ensure!(
        matches!(model.input.row_order, FittedRowOrder::Iid)
            && model.input.historical.iter().all(Option::is_none),
        "portable fitted input does not support historical state"
    );
    let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    else {
        bail!("portable fitted input requires a schema-bound pipeline");
    };
    ensure!(
        target.is_none() && ops.is_empty() && matches!(calibration, CalibrationModel::None),
        "portable fitted input does not support operations, target mapping, or calibration"
    );
    let input = model
        .input
        .columns
        .iter()
        .map(|column| match column {
            FittedColumn::Numeric { fill, indicator } => {
                ensure!(fill.is_finite(), "nonfinite fitted input fill");
                Ok(Input::Numeric {
                    fill: *fill,
                    indicator: *indicator,
                    restore_missing: *missing_aware,
                })
            }
            FittedColumn::Categorical {
                mode: CategoricalMode::Onehot,
                vocabulary,
                onehot,
                ..
            } => {
                ensure!(
                    onehot.iter().all(|index| *index < vocabulary.len()),
                    "invalid fitted onehot index"
                );
                Ok(Input::Onehot {
                    vocabulary: onehot
                        .iter()
                        .map(|index| vocabulary[*index].clone())
                        .collect(),
                    indices: (0..onehot.len()).collect(),
                })
            }
            FittedColumn::Categorical { .. } => {
                bail!("portable fitted input supports only onehot categories")
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let (head, class_codes) = match head {
        V3Head::Legacy(FittedHead::Ridge(value)) if model.task.kind == TaskKind::Regression => (
            Head::Regression(linear(&value.means, &value.scales, &value.w, 20.0, false)),
            vec![],
        ),
        V3Head::LogitRidge(value) if model.task.kind == TaskKind::Binary => (
            Head::Binary(linear(
                &value.means,
                &value.scales,
                &value.weights,
                30.0,
                false,
            )),
            vec![0, 1],
        ),
        V3Head::MulticlassRidge(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.means.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Softmax(
                    value
                        .weights
                        .iter()
                        .map(|weights| linear(&value.means, &value.scales, weights, 30.0, true))
                        .collect(),
                ),
                value.classes.clone(),
            )
        }
        V3Head::MulticlassOvrLogistic(value) if model.task.kind == TaskKind::Multiclass => (
            Head::Ovr(
                value
                    .models
                    .iter()
                    .map(|value| linear(&value.means, &value.scales, &value.weights, 30.0, false))
                    .collect(),
            ),
            value.classes.clone(),
        ),
        V3Head::MulticlassDiagonalLda(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.variances.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Lda {
                    centers: value.means.clone(),
                    scales: value.variances.clone(),
                    priors: value.log_priors.clone(),
                },
                value.classes.clone(),
            )
        }
        V3Head::MulticlassNearestCentroid(value) if model.task.kind == TaskKind::Multiclass => {
            ensure!(
                value.feature_count == value.scales.len(),
                "multiclass fitted width differs"
            );
            (
                Head::Centroid {
                    centers: value.centroids.clone(),
                    scales: value.scales.clone(),
                },
                value.classes.clone(),
            )
        }
        head if scalar::supports(head, model.task.kind) => (
            Head::Scalar {
                binary: model.task.kind == TaskKind::Binary,
                state: scalar::lower(head, model.task.kind)?,
            },
            if model.task.kind == TaskKind::Binary {
                vec![0, 1]
            } else {
                vec![]
            },
        ),
        _ => bail!("this fitted head/task is outside the portable fitted subset"),
    };
    let state = State { input, head };
    validate_state(&state)?;
    let serialized = serde_json::to_string(&state)?;
    let python = render(
        include_str!("../../contracts/fitted_portable_py.template"),
        &[("__STATE_JSON__", serde_json::to_string(&serialized)?)],
    );
    let javascript = if fitted_javascript_unsupported(model)?.is_none() {
        Some(render(
            include_str!("../../contracts/fitted_portable.template"),
            &[("__STATE_JSON__", serialized)],
        ))
    } else {
        None
    };
    let rust = rust_source(&state);
    for source in [&python, &rust].into_iter().chain(javascript.iter()) {
        let hits = lint_fitted_public_text(model, &[], source);
        ensure!(
            hits.is_empty(),
            "fitted source contains banned vocabulary: {}",
            hits.join(", ")
        );
    }
    Ok(FittedPortableSources {
        python,
        rust,
        javascript,
        class_codes,
    })
}

fn validate_state(state: &State) -> Result<()> {
    let width: usize = state
        .input
        .iter()
        .map(|input| match input {
            Input::Numeric { indicator, .. } => 1 + usize::from(*indicator),
            Input::Onehot { indices, .. } => indices.len(),
        })
        .sum();
    let check = |value: &Linear| -> Result<()> {
        ensure!(
            !value.means.is_empty()
                && value.means.len() <= width
                && value.scales.len() == value.means.len()
                && value.weights.len() == value.means.len() + 1,
            "fitted linear dimensions differ"
        );
        ensure!(
            value
                .means
                .iter()
                .chain(&value.weights)
                .all(|v| v.is_finite())
                && value.scales.iter().all(|v| v.is_finite() && *v > 0.0),
            "invalid fitted linear coefficients"
        );
        Ok(())
    };
    match &state.head {
        Head::Scalar { state, .. } => scalar::validate(state, width),
        Head::Regression(value) | Head::Binary(value) => check(value),
        Head::Softmax(values) | Head::Ovr(values) => {
            ensure!(
                values.len() >= 3,
                "multiclass fitted state has fewer than three classes"
            );
            values.iter().try_for_each(check)
        }
        Head::Lda {
            centers, scales, ..
        }
        | Head::Centroid { centers, scales } => {
            // The separate prior check below avoids treating absent priors as zeros.
            ensure!(
                centers.len() >= 3 && !scales.is_empty() && scales.len() <= width,
                "multiclass fitted dimensions differ"
            );
            ensure!(
                scales.iter().all(|v| v.is_finite() && *v > 0.0)
                    && centers
                        .iter()
                        .all(|row| row.len() == scales.len() && row.iter().all(|v| v.is_finite())),
                "invalid fitted distance coefficients"
            );
            if let Head::Lda { priors, .. } = &state.head {
                ensure!(
                    priors.len() == centers.len() && priors.iter().all(|v| v.is_finite()),
                    "invalid fitted priors"
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
