use super::*;

// ---------------------------------------------------------------------------
// per-candidate evaluation
// ---------------------------------------------------------------------------

// Preserve the structured parse refusal returned to existing receipt callers.
#[allow(clippy::result_large_err)]
pub(super) fn parse_canonical(
    candidate: &ShardCandidate,
) -> std::result::Result<GenomeIr, Outcome> {
    let genome = match GenomeIr::parse(&candidate.canonical_genome) {
        Ok(genome) => genome,
        Err(error) => {
            return Err(Outcome::Fail {
                class: "parse",
                message: format!("{error:#}"),
            })
        }
    };
    if genome.canonical() != candidate.canonical_genome {
        return Err(Outcome::Fail {
            class: "parse",
            message: "genome is not canonical".to_string(),
        });
    }
    Ok(genome)
}

pub(super) fn evaluate_regression(
    candidate: &ShardCandidate,
    train: &Data,
    test: &Data,
) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    let predictions = match fit_predict(&genome, &train.x, &train.y, &test.x) {
        Ok(predictions) => predictions,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    if predictions.len() != test.y.len() || predictions.iter().any(|p| !p.is_finite()) {
        return Outcome::Fail {
            class: "nonfinite",
            message: "predictions are missing or non-finite".to_string(),
        };
    }
    let (score, secondary) = (r2(&test.y, &predictions), rmse(&test.y, &predictions));
    if !score.is_finite() || !secondary.is_finite() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "computed metric is non-finite".into(),
        };
    }
    Outcome::Score {
        score,
        secondary,
        balanced_accuracy: None,
        timeseries_metrics: None,
        predictions,
        multiclass: None,
    }
}

pub(super) fn evaluate_binary(
    candidate: &ShardCandidate,
    train_table: &RawTable,
    train_y: &[f64],
    test_table: &RawTable,
    test_y: &[f64],
    features: usize,
) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    let task = TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective");
    let schema = binary_schema(features);
    let model = match fit_table(&genome, &task, &schema, train_table, train_y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let predictions = match model.predict_table(test_table) {
        Ok(predictions) => predictions,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    if predictions.len() != test_y.len()
        || predictions
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
    {
        return Outcome::Fail {
            class: "nonfinite",
            message: "probabilities are missing or outside [0, 1]".to_string(),
        };
    }
    let (score, secondary) = (auc(test_y, &predictions), logloss(test_y, &predictions));
    if !score.is_finite() || !secondary.is_finite() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "computed metric is non-finite".into(),
        };
    }
    Outcome::Score {
        score,
        secondary,
        balanced_accuracy: Some(balanced_accuracy(test_y, &predictions)),
        timeseries_metrics: None,
        predictions,
        multiclass: None,
    }
}

pub(super) fn evaluate_multiclass(
    candidate: &ShardCandidate,
    train_table: &RawTable,
    train_y: &[f64],
    test_table: &RawTable,
    test_y: &[f64],
    features: usize,
    context: &MulticlassContext,
) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    let task =
        TaskSpec::multiclass(Objective::MacroF1).expect("macro_f1 is a multiclass objective");
    let schema = binary_schema(features);
    let model = match fit_table(&genome, &task, &schema, train_table, train_y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let output = match model.predict_multiclass_table(test_table) {
        Ok(output) => output,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    if output.labels.len() != test_y.len() || output.probabilities.len() != test_y.len() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "multiclass labels or aligned probabilities are invalid".to_string(),
        };
    }
    let mut fitted_columns = std::collections::BTreeMap::new();
    for (column, fitted_class) in output.classes.iter().copied().enumerate() {
        let Ok(fitted_index) = usize::try_from(fitted_class) else {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass model emitted a negative fitted class".to_string(),
            };
        };
        if fitted_index >= context.observed_declared.len()
            || fitted_columns.insert(fitted_index, column).is_some()
        {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass model emitted an unknown or duplicate fitted class"
                    .to_string(),
            };
        }
    }
    if fitted_columns.len() != context.observed_declared.len() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "multiclass probability output omitted an observed training class".to_string(),
        };
    }
    let mut predictions = Vec::with_capacity(output.labels.len());
    for fitted_class in &output.labels {
        let Ok(fitted_index) = usize::try_from(*fitted_class) else {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass model predicted a negative fitted class".to_string(),
            };
        };
        let Some(declared_index) = context.observed_declared.get(fitted_index) else {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass model predicted an unknown fitted class".to_string(),
            };
        };
        predictions.push(*declared_index as f64);
    }
    let mut aligned_probabilities = Vec::with_capacity(output.probabilities.len());
    for row in &output.probabilities {
        if row.len() != output.classes.len()
            || row.iter().any(|value| !value.is_finite() || *value < 0.0)
        {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass probability row is invalid".to_string(),
            };
        }
        let mut aligned = vec![0.0; context.declared_classes];
        for (fitted_index, column) in &fitted_columns {
            let declared_index = context.observed_declared[*fitted_index] as usize;
            aligned[declared_index] = row[*column];
        }
        let sum: f64 = aligned.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Outcome::Fail {
                class: "nonfinite",
                message: "multiclass probability row has a nonpositive sum".to_string(),
            };
        }
        aligned.iter_mut().for_each(|value| *value /= sum);
        aligned_probabilities.push(aligned);
    }
    let score = declared_macro_f1(test_y, &predictions, context.declared_classes);
    let secondary = multiclass_accuracy(test_y, &predictions);
    if !score.is_finite() || !secondary.is_finite() {
        return Outcome::Fail {
            class: "nonfinite",
            message: "computed multiclass metric is non-finite".to_string(),
        };
    }
    Outcome::Score {
        score,
        secondary,
        balanced_accuracy: None,
        timeseries_metrics: None,
        predictions,
        multiclass: Some(MulticlassEvidence {
            classes: (0..context.declared_classes as i64).collect(),
            probabilities: aligned_probabilities,
        }),
    }
}

pub(super) fn terminal_head(genome: &GenomeIr) -> Option<&str> {
    match genome {
        GenomeIr::Pipeline(dsl) => dsl
            .stages
            .last()
            .filter(|stage| is_head_like(&stage.op))
            .map(|stage| stage.op.as_str()),
        _ => None,
    }
}

pub(super) fn evaluate_timeseries(candidate: &ShardCandidate, data: &TimeSeriesData) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
        return Outcome::Fail {
            class: "task",
            message: "binary classification heads are not valid for timeseries".to_string(),
        };
    }
    let reduction = match make_supervised_reduction(data) {
        Ok(reduction) => reduction,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let schema = DataSchema {
        columns: reduction
            .feature_names
            .iter()
            .map(|name| ColumnSpec::numeric(name.clone()))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_table = match raw_table_named(&reduction.feature_names, &reduction.x) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let task = TaskSpec::regression(Objective::Rmse).expect("rmse is a regression objective");
    let model = match fit_table(&genome, &task, &schema, &train_table, &reduction.y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let mut history = data.train.y.clone();
    let mut predictions = Vec::with_capacity(data.horizon);
    for horizon_index in 0..data.horizon {
        let row = match make_forecast_features(data, &history, horizon_index) {
            Ok(row) => row,
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        let table = match raw_table_named(&reduction.feature_names, &[row]) {
            Ok(table) => table,
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        let prediction = match model.predict_table(&table) {
            Ok(mut values) if values.len() == 1 => values.remove(0),
            Ok(values) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("recursive prediction emitted {} rows", values.len()),
                }
            }
            Err(error) => {
                return Outcome::Fail {
                    class: "fit",
                    message: format!("{error:#}"),
                }
            }
        };
        if !prediction.is_finite() {
            return Outcome::Fail {
                class: "nonfinite",
                message: "recursive prediction is non-finite".to_string(),
            };
        }
        predictions.push(prediction);
        history.push(prediction);
    }
    let metrics = match score_timeseries(data, &predictions) {
        Ok(metrics) => metrics,
        Err(error) => {
            return Outcome::Fail {
                class: "nonfinite",
                message: format!("{error:#}"),
            }
        }
    };
    Outcome::Score {
        score: -metrics.mase,
        secondary: metrics.mase,
        balanced_accuracy: None,
        timeseries_metrics: Some(metrics),
        predictions,
        multiclass: None,
    }
}

/// v2 evaluation: direct multi-horizon reduction fit in MASE units, one
/// batched predict call over all H forecast rows (no prediction feedback),
/// inverse-scaled before scoring. Same fitness authority as v1 (-MASE).
pub(super) fn evaluate_timeseries_v2(candidate: &ShardCandidate, data: &TimeSeriesData) -> Outcome {
    let genome = match parse_canonical(candidate) {
        Ok(genome) => genome,
        Err(outcome) => return outcome,
    };
    if matches!(terminal_head(&genome), Some("logit_ridge" | "logistic")) {
        return Outcome::Fail {
            class: "task",
            message: "binary classification heads are not valid for timeseries".to_string(),
        };
    }
    let reduction = match make_direct_reduction(data) {
        Ok(reduction) => reduction,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let schema = DataSchema {
        columns: reduction
            .feature_names
            .iter()
            .map(|name| ColumnSpec::numeric(name.clone()))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train_table = match raw_table_named(&reduction.feature_names, &reduction.x) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let task = TaskSpec::regression(Objective::Rmse).expect("rmse is a regression objective");
    let model = match fit_table(&genome, &task, &schema, &train_table, &reduction.y) {
        Ok(model) => model,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let forecast_rows = match make_direct_forecast_rows(data) {
        Ok(rows) => rows,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let forecast_table = match raw_table_named(&reduction.feature_names, &forecast_rows) {
        Ok(table) => table,
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let scaled = match model.predict_table(&forecast_table) {
        Ok(values) if values.len() == data.horizon => values,
        Ok(values) => {
            return Outcome::Fail {
                class: "fit",
                message: format!(
                    "direct prediction emitted {} rows for horizon {}",
                    values.len(),
                    data.horizon
                ),
            }
        }
        Err(error) => {
            return Outcome::Fail {
                class: "fit",
                message: format!("{error:#}"),
            }
        }
    };
    let predictions: Vec<f64> = scaled
        .iter()
        .map(|value| value * reduction.target_scale)
        .collect();
    if predictions.iter().any(|prediction| !prediction.is_finite()) {
        return Outcome::Fail {
            class: "nonfinite",
            message: "direct prediction is non-finite".to_string(),
        };
    }
    let metrics = match score_timeseries(data, &predictions) {
        Ok(metrics) => metrics,
        Err(error) => {
            return Outcome::Fail {
                class: "nonfinite",
                message: format!("{error:#}"),
            }
        }
    };
    Outcome::Score {
        score: -metrics.mase,
        secondary: metrics.mase,
        balanced_accuracy: None,
        timeseries_metrics: Some(metrics),
        predictions,
        multiclass: None,
    }
}
