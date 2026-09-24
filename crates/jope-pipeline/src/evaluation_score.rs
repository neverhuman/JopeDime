//! Frozen-panel metrics consume complete probability matrices with explicit class order.
use crate::experiment::Metric;
use anyhow::{bail, ensure, Context, Result};
use feat_math::advanced_v3::{BoundModel, RawTable, TaskKind};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PROBABILITY_EPSILON: f64 = 1e-15;
/// Controller arithmetic is f64; external engines may return f32 probabilities.
/// Admit their bounded row-sum roundoff without renormalizing any prediction.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
pub const NUMERICAL_POLICY: &str =
    "asm-panel-f64-v1:platform-libm-v1:probability-epsilon=1e-15:row-sum-tolerance=1e-6";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Predictions {
    Values(Vec<f64>),
    Probabilities {
        classes: Vec<String>,
        values: Vec<Vec<f64>>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelScore {
    pub metric: Metric,
    pub value: f64,
    /// Smaller is better for all three tasks; AUROC uses 1 - AUROC.
    pub loss: f64,
}

fn validate_class_order(task: Task, classes: &[String]) -> Result<()> {
    let unique: BTreeSet<_> = classes.iter().collect();
    ensure!(
        unique.len() == classes.len() && classes.iter().all(|label| !label.is_empty()),
        "empty or duplicate class label"
    );
    ensure!(
        match task {
            Task::Regression => classes.is_empty(),
            Task::Binary => classes.len() == 2,
            Task::Multiclass => classes.len() >= 3,
            _ => false,
        },
        "class order incompatible with task"
    );
    Ok(())
}

/// Score original labels. Classification columns must already follow the frozen
/// manifest order; accepting a silently permuted matrix would score another task.
/// Probabilities are never renormalized. Only log loss clips an observed-class
/// probability below the pinned epsilon, after the entire matrix validates.
pub fn score_predictions(
    task: Task,
    classes: &[String],
    labels: &[String],
    predictions: &Predictions,
) -> Result<PanelScore> {
    let metric = Metric::for_task(task)?;
    validate_class_order(task, classes)?;
    ensure!(!labels.is_empty(), "cannot score an empty partition");
    let value = match (task, predictions) {
        (Task::Regression, Predictions::Values(values)) => {
            ensure!(
                values.len() == labels.len(),
                "prediction row count mismatch"
            );
            ensure!(
                values.iter().all(|value| value.is_finite()),
                "nonfinite regression prediction"
            );
            let targets = labels
                .iter()
                .map(|label| {
                    let target: f64 = label.trim().parse().context("invalid regression target")?;
                    ensure!(target.is_finite(), "nonfinite regression target");
                    Ok(target)
                })
                .collect::<Result<Vec<_>>>()?;
            feat_math::metrics::rmse(&targets, values)
        }
        (
            Task::Binary | Task::Multiclass,
            Predictions::Probabilities {
                classes: predicted_classes,
                values,
            },
        ) => {
            ensure!(
                predicted_classes == classes,
                "prediction class order differs from frozen order"
            );
            ensure!(
                values.len() == labels.len(),
                "prediction row count mismatch"
            );
            validate_probabilities(values, classes.len())?;
            let targets = labels
                .iter()
                .map(|label| {
                    classes
                        .iter()
                        .position(|class| class == label)
                        .with_context(|| {
                            format!("target {label:?} absent from declared class order")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if task == Task::Binary {
                let positives = targets.iter().filter(|target| **target == 1).count();
                ensure!(
                    positives > 0 && positives < targets.len(),
                    "AUROC requires both declared classes"
                );
                let targets: Vec<_> = targets.into_iter().map(|target| target as f64).collect();
                // IEEE -0 and +0 are the same probability and must tie in AUROC.
                let positive: Vec<_> = values
                    .iter()
                    .map(|row| if row[1] == 0.0 { 0.0 } else { row[1] })
                    .collect();
                feat_math::metrics::auc(&targets, &positive)
            } else {
                let sum = targets
                    .iter()
                    .zip(values)
                    .map(|(target, row)| -row[*target].max(PROBABILITY_EPSILON).ln())
                    .sum::<f64>();
                sum / targets.len() as f64
            }
        }
        (Task::Regression, _) => bail!("regression requires scalar predictions"),
        (Task::Binary | Task::Multiclass, _) => {
            bail!("classification requires a probability matrix, not hard labels")
        }
        _ => bail!("unsupported panel task"),
    };
    let loss = metric.loss(value)?;
    Ok(PanelScore {
        metric,
        value,
        loss,
    })
}

#[doc(hidden)]
pub fn validate_probabilities(values: &[Vec<f64>], classes: usize) -> Result<()> {
    for row in values {
        ensure!(row.len() == classes, "probability column count mismatch");
        ensure!(
            row.iter()
                .all(|probability| probability.is_finite() && (0.0..=1.0).contains(probability)),
            "nonfinite or out-of-range probability"
        );
        ensure!(
            (row.iter().sum::<f64>() - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE,
            "probability row does not sum to one"
        );
    }
    Ok(())
}

/// Predict from complete fitted state, with no fitting or access to holdout labels.
/// Native class IDs are the indices established by the training label mapping.
pub fn predict_bound(
    model: &BoundModel,
    table: &RawTable,
    classes: &[String],
) -> Result<Predictions> {
    let task = match model.task.kind {
        TaskKind::Regression => Task::Regression,
        TaskKind::Binary => Task::Binary,
        TaskKind::Multiclass => Task::Multiclass,
        TaskKind::Timeseries => bail!("timeseries model is outside the frozen panel"),
    };
    validate_class_order(task, classes)?;
    match task {
        Task::Regression => {
            let values = model.predict_table(table)?;
            ensure!(
                values.len() == table.rows.len() && values.iter().all(|value| value.is_finite()),
                "invalid bound regression predictions"
            );
            Ok(Predictions::Values(values))
        }
        Task::Binary => {
            let probabilities = model.predict_table(table)?;
            ensure!(
                probabilities.len() == table.rows.len(),
                "bound prediction row count mismatch"
            );
            let values: Vec<_> = probabilities
                .iter()
                .map(|probability| vec![1.0 - probability, *probability])
                .collect();
            validate_probabilities(&values, classes.len())?;
            Ok(Predictions::Probabilities {
                classes: classes.to_vec(),
                values,
            })
        }
        Task::Multiclass => {
            let prediction = model.predict_multiclass_table(table)?;
            ensure!(
                prediction.probabilities.len() == table.rows.len(),
                "bound prediction row count mismatch"
            );
            ensure!(
                prediction.classes.len() == classes.len(),
                "bound class count mismatch"
            );
            validate_probabilities(&prediction.probabilities, prediction.classes.len())?;
            let order: Vec<_> = (0..classes.len())
                .map(|index| {
                    prediction
                        .classes
                        .iter()
                        .position(|class| *class == index as i64)
                        .context("bound class IDs do not match the frozen label mapping")
                })
                .collect::<Result<_>>()?;
            let values: Vec<Vec<f64>> = prediction
                .probabilities
                .iter()
                .map(|row| order.iter().map(|column| row[*column]).collect())
                .collect();
            validate_probabilities(&values, classes.len())?;
            Ok(Predictions::Probabilities {
                classes: classes.to_vec(),
                values,
            })
        }
        _ => unreachable!(),
    }
}
