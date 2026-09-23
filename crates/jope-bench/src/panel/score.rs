//! Frozen-panel metrics consume complete probability matrices with explicit class order.
use anyhow::{bail, ensure, Context, Result};
use feat_math::advanced_v3::{BoundModel, RawTable, TaskKind};
use jope_datasets::catalog::Task;
use jope_pipeline::experiment::Metric;
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

fn validate_probabilities(values: &[Vec<f64>], classes: usize) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::data::raw_table;
    use feat_math::advanced_v3::{
        fit_table, ColumnSpec, DataSchema, Objective, RowSemantics, TaskSpec,
    };
    use feat_math::GenomeIr;
    use jope_pipeline::experiment::{Column, ColumnKind};

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn matrix(classes: &[String], values: Vec<Vec<f64>>) -> Predictions {
        Predictions::Probabilities {
            classes: classes.to_vec(),
            values,
        }
    }

    #[test]
    fn regression_uses_rmse_and_refuses_truncated_or_nonfinite_data() {
        let labels = strings(&["0", "2"]);
        let result = score_predictions(
            Task::Regression,
            &[],
            &labels,
            &Predictions::Values(vec![2., 0.]),
        )
        .unwrap();
        assert_eq!(result.metric, Metric::Rmse);
        assert_eq!(result.value, 2.);
        assert_eq!(result.loss, 2.);
        for values in [vec![0.], vec![f64::NAN, 0.], vec![f64::INFINITY, 0.]] {
            assert!(score_predictions(
                Task::Regression,
                &[],
                &labels,
                &Predictions::Values(values)
            )
            .is_err());
        }
        assert!(
            score_predictions(Task::Regression, &[], &[], &Predictions::Values(vec![])).is_err()
        );
    }

    #[test]
    fn binary_auc_uses_manifest_positive_class_and_midrank_ties() {
        let classes = strings(&["2", "10"]);
        let labels = strings(&["2", "10", "2", "10"]);
        let tied = matrix(&classes, vec![vec![0.5, 0.5]; 4]);
        let result = score_predictions(Task::Binary, &classes, &labels, &tied).unwrap();
        assert_eq!(result.value, 0.5);
        assert_eq!(result.loss, 0.5);
        let perfect = matrix(
            &classes,
            vec![
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.8, 0.2],
                vec![0.2, 0.8],
            ],
        );
        assert_eq!(
            score_predictions(Task::Binary, &classes, &labels, &perfect)
                .unwrap()
                .value,
            1.
        );
        assert_eq!(
            score_predictions(
                Task::Binary,
                &classes,
                &strings(&["10", "2", "10", "2"]),
                &perfect
            )
            .unwrap()
            .value,
            0.
        );
        let zeros = matrix(&classes, vec![vec![1., -0.], vec![1., 0.]]);
        assert_eq!(
            score_predictions(Task::Binary, &classes, &strings(&["10", "2"]), &zeros)
                .unwrap()
                .value,
            0.5
        );
        assert!(score_predictions(Task::Binary, &classes, &strings(&["10"; 4]), &perfect).is_err());
    }

    #[test]
    fn multiclass_requires_a_matrix_and_scores_the_declared_column() {
        let classes = strings(&["β", "00", "1"]);
        let labels = strings(&["β", "00", "1"]);
        let predictions = matrix(
            &classes,
            vec![
                vec![0.5, 0.3, 0.2],
                vec![0.1, 0.8, 0.1],
                vec![0.2, 0.1, 0.7],
            ],
        );
        let score = score_predictions(Task::Multiclass, &classes, &labels, &predictions).unwrap();
        let expected = -(0.5f64.ln() + 0.8f64.ln() + 0.7f64.ln()) / 3.;
        assert_eq!(score.metric, Metric::LogLoss);
        assert!((score.value - expected).abs() < 1e-14);
        assert_eq!(score.value, score.loss);
        assert!(score_predictions(
            Task::Multiclass,
            &classes,
            &labels,
            &Predictions::Values(vec![0., 1., 2.])
        )
        .is_err());
        let mut reordered = classes.clone();
        reordered.swap(0, 1);
        let wrong_order = matrix(&reordered, vec![vec![0.3, 0.5, 0.2]; 3]);
        assert!(score_predictions(Task::Multiclass, &classes, &labels, &wrong_order).is_err());
        let wrong = matrix(
            &classes,
            vec![vec![0., 1., 0.], vec![0., 0., 1.], vec![1., 0., 0.]],
        );
        let score = score_predictions(Task::Multiclass, &classes, &labels, &wrong).unwrap();
        assert!((score.value + PROBABILITY_EPSILON.ln()).abs() < 1e-13);
    }

    #[test]
    fn f32_probability_roundoff_is_bounded_without_renormalization() {
        let classes = strings(&["a", "b", "c"]);
        let row = vec![0.1f32 as f64, 0.2f32 as f64, 0.7f32 as f64];
        let error = (row.iter().sum::<f64>() - 1.0).abs();
        assert!(error > 1e-9 && error <= PROBABILITY_SUM_TOLERANCE);
        let score = score_predictions(
            Task::Multiclass,
            &classes,
            &strings(&["a"]),
            &matrix(&classes, vec![row.clone()]),
        )
        .unwrap();
        assert_eq!(score.value.to_bits(), (-row[0].ln()).to_bits());
        let mut invalid = row;
        invalid[0] += 2.0 * PROBABILITY_SUM_TOLERANCE;
        assert!(score_predictions(
            Task::Multiclass,
            &classes,
            &strings(&["a"]),
            &matrix(&classes, vec![invalid]),
        )
        .is_err());
    }

    #[test]
    fn classification_refuses_missing_classes_invalid_shapes_and_bad_probabilities() {
        let classes = strings(&["a", "b", "c"]);
        let labels = strings(&["a", "b", "c"]);
        for values in [
            vec![vec![0.5, 0.5]; 3],
            vec![vec![0.2, 0.2, 0.2]; 3],
            vec![vec![1.1, -0.1, 0.]; 3],
            vec![vec![f64::NAN, 0., 1.]; 3],
            vec![vec![0., f64::INFINITY, 0.]; 3],
            vec![vec![0., 0., 1.]; 2],
        ] {
            assert!(score_predictions(
                Task::Multiclass,
                &classes,
                &labels,
                &matrix(&classes, values)
            )
            .is_err());
        }
        let missing = matrix(&classes[..2], vec![vec![0.5, 0.5]; 3]);
        assert!(score_predictions(Task::Multiclass, &classes, &labels, &missing).is_err());
        let valid = matrix(&classes, vec![vec![0.2, 0.3, 0.5]; 3]);
        assert!(score_predictions(
            Task::Multiclass,
            &classes,
            &strings(&["a", "b", "unknown"]),
            &valid
        )
        .is_err());
        let renamed = strings(&["first", "second", "third"]);
        let renamed_score = score_predictions(
            Task::Multiclass,
            &renamed,
            &renamed,
            &matrix(&renamed, vec![vec![0.2, 0.3, 0.5]; 3]),
        )
        .unwrap();
        assert_eq!(
            renamed_score.value,
            score_predictions(Task::Multiclass, &classes, &labels, &valid)
                .unwrap()
                .value
        );
    }

    fn prediction_bits(predictions: &Predictions) -> Vec<u64> {
        match predictions {
            Predictions::Values(values) => values.iter().map(|value| value.to_bits()).collect(),
            Predictions::Probabilities { values, .. } => values
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect(),
        }
    }

    #[test]
    fn fitted_reload_preserves_null_unseen_category_and_probability_columns() {
        let features = vec![
            Column {
                name: "number".into(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "category".into(),
                kind: ColumnKind::Categorical,
            },
        ];
        let schema = DataSchema {
            columns: vec![
                ColumnSpec::numeric("number"),
                ColumnSpec::categorical("category"),
            ],
            row_semantics: RowSemantics::Iid,
        };
        let values = (0..18)
            .map(|index| {
                vec![
                    Some(index.to_string()),
                    if index % 3 == 0 {
                        None
                    } else if index % 3 == 1 {
                        Some("NA".into())
                    } else {
                        Some("null".into())
                    },
                ]
            })
            .collect::<Vec<_>>();
        let train = raw_table(&features, &values).unwrap();
        let hold = raw_table(
            &features,
            &[
                vec![None, None],
                vec![Some("3.5".into()), Some("unseen".into())],
                vec![Some("8".into()), Some("NA".into())],
                vec![Some("12".into()), Some("null".into())],
            ],
        )
        .unwrap();
        for (task, spec, head, classes) in [
            (
                Task::Regression,
                TaskSpec::regression(Objective::Rmse).unwrap(),
                "ridge",
                vec![],
            ),
            (
                Task::Binary,
                TaskSpec::binary(Objective::RocAuc).unwrap(),
                "logit_ridge",
                strings(&["2", "10"]),
            ),
            // The fitter's existing objective is explicit; evaluation uses matrix log loss.
            (
                Task::Multiclass,
                TaskSpec::multiclass(Objective::Accuracy).unwrap(),
                "multiclass_nearest_centroid",
                strings(&["β", "00", "1"]),
            ),
        ] {
            let targets: Vec<_> = (0..18)
                .map(|index| {
                    if task == Task::Regression {
                        index as f64 / 10.
                    } else {
                        (index % classes.len()) as f64
                    }
                })
                .collect();
            let genome = GenomeIr::parse(&format!("(pipe X (input numeric=median missing=indicator categorical=onehot min_count=1) ({head}))")).unwrap();
            let model = fit_table(&genome, &spec, &schema, &train, &targets).unwrap();
            let expected = predict_bound(&model, &hold, &classes).unwrap();
            let bytes = serde_json::to_vec(&model).unwrap();
            let reloaded: BoundModel = serde_json::from_slice(&bytes).unwrap();
            reloaded.validate_binding().unwrap();
            let actual = predict_bound(&reloaded, &hold, &classes).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(prediction_bits(&actual), prediction_bits(&expected));
            let transported: Predictions =
                serde_json::from_slice(&serde_json::to_vec(&expected).unwrap()).unwrap();
            assert_eq!(transported, expected);
            assert_eq!(prediction_bits(&transported), prediction_bits(&expected));
            let state = String::from_utf8(bytes).unwrap();
            assert!(
                !state.contains("unseen"),
                "holdout category must not enter fitted vocabulary"
            );
            if let Predictions::Probabilities {
                classes: returned,
                values,
            } = expected
            {
                assert_eq!(returned, classes);
                assert_eq!(values.len(), hold.rows.len());
                validate_probabilities(&values, classes.len()).unwrap();
            }
        }
    }
}
