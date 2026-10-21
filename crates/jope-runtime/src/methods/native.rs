//! Native fit and prediction through `feat_math::advanced_v3` (SPECIFICATION.md §6.1).
//!
//! [`predict_bound`], [`Predictions`] and [`validate_probabilities`] moved here unchanged from
//! `jope_pipeline::evaluation_score`, which re-exports them, so the frozen panel's fit workers and
//! the appliance turn a bound model into predictions with one implementation.
use super::registry::{rust_methods, task_spec, validate_classes, RustMethod};
use crate::task::TaskKind;
use anyhow::{bail, ensure, Context, Result};
use feat_math::advanced_v3::{
    fit_table, validate_v3_genome, BoundModel, DataSchema, RawTable, TaskKind as Kind,
};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Controller arithmetic is f64; external engines may return f32 probabilities.
/// Admit their bounded row-sum roundoff without renormalizing any prediction.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;

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

impl Predictions {
    pub fn rows(&self) -> usize {
        match self {
            Predictions::Values(values) => values.len(),
            Predictions::Probabilities { values, .. } => values.len(),
        }
    }
}

/// The panel's class-order rule: unique non-empty labels whose count matches the task.
pub fn validate_class_order(kind: TaskKind, classes: &[String]) -> Result<()> {
    let unique: BTreeSet<_> = classes.iter().collect();
    ensure!(
        unique.len() == classes.len() && classes.iter().all(|label| !label.is_empty()),
        "empty or duplicate class label"
    );
    ensure!(
        match kind {
            TaskKind::Regression => classes.is_empty(),
            TaskKind::Binary => classes.len() == 2,
            TaskKind::Multiclass => classes.len() >= 3,
            TaskKind::Timeseries => false,
        },
        "class order incompatible with task"
    );
    Ok(())
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
    let kind = model.task.kind;
    if kind == Kind::Timeseries {
        bail!("timeseries model is outside the frozen panel");
    }
    validate_class_order(kind, classes)?;
    match kind {
        Kind::Regression => {
            let values = model.predict_table(table)?;
            ensure!(
                values.len() == table.rows.len() && values.iter().all(|value| value.is_finite()),
                "invalid bound regression predictions"
            );
            Ok(Predictions::Values(values))
        }
        Kind::Binary => {
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
        Kind::Multiclass => {
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
        Kind::Timeseries => unreachable!("refused above"),
    }
}

/// Validate a registered recipe against the task and schema, then fit it on the training table.
/// `targets` are values for regression and class indices (into `classes`) for classification;
/// every declared class must occur.
pub fn fit_native(
    method: &RustMethod,
    kind: TaskKind,
    schema: &DataSchema,
    classes: &[String],
    train: &RawTable,
    targets: &[f64],
) -> Result<BoundModel> {
    validate_classes(kind, classes)?;
    train.validate_against(schema)?;
    ensure!(
        !train.rows.is_empty() && train.rows.len() == targets.len(),
        "fit targets differ from nonempty training rows"
    );
    ensure!(targets.iter().all(|v| v.is_finite()), "nonfinite target");
    if kind != TaskKind::Regression {
        let mut seen = BTreeSet::new();
        for value in targets {
            ensure!(
                *value >= 0.0 && value.fract() == 0.0 && *value < classes.len() as f64,
                "target is outside the declared class vocabulary"
            );
            seen.insert(*value as usize);
        }
        ensure!(
            seen.len() == classes.len(),
            "a declared class is absent from training"
        );
    }
    ensure!(
        rust_methods(kind)?.contains(method),
        "unregistered or altered Rust method"
    );
    let genome = GenomeIr::parse(&method.dsl)?;
    let spec = task_spec(kind)?;
    validate_v3_genome(&genome, &spec, schema)?;
    fit_table(&genome, &spec, schema, train, targets)
}
