//! Shared scorer API; implementation lives in jope-pipeline.
pub use jope_pipeline::evaluation_score::*;

#[cfg(test)]
use feat_math::advanced_v3::BoundModel;
#[cfg(test)]
use jope_datasets::catalog::Task;
#[cfg(test)]
use jope_pipeline::experiment::Metric;

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
