use super::*;

pub(super) fn compute_landmarks(
    table: &RawTable,
    schema: &DataSchema,
    task: &TaskSpec,
    targets: &[f64],
    selected_rows: &[usize],
    selected_columns: &[usize],
) -> Result<CheapLandmarksV1> {
    let landmark_table = subset_table(table, selected_rows, selected_columns)?;
    let landmark_schema = DataSchema {
        columns: selected_columns
            .iter()
            .map(|column| schema.columns[*column].clone())
            .collect(),
        // Landmarks contain no row operation.  Treating the deterministic row
        // sample as IID prevents a sampled-out group/order column from
        // changing this cheap baseline contract.
        row_semantics: RowSemantics::Iid,
    };
    landmark_schema.validate()?;
    let landmark_targets: Vec<f64> = selected_rows.iter().map(|row| targets[*row]).collect();
    let encoded_width = landmark_schema
        .columns
        .iter()
        .map(|column| match column.kind {
            ColumnKind::Numeric => 2usize,
            ColumnKind::Categorical => 1usize,
        })
        .sum::<usize>();
    let include_pls = matches!(task.kind, TaskKind::Regression | TaskKind::Timeseries)
        && encoded_width > landmark_targets.len();
    let kinds: Vec<CheapLandmarkKind> = match task.kind {
        TaskKind::Regression | TaskKind::Timeseries => {
            let mut kinds = vec![
                CheapLandmarkKind::RegressionMean,
                CheapLandmarkKind::RegressionRidge,
                CheapLandmarkKind::RegressionShallowTree,
                CheapLandmarkKind::RegressionShallowGam,
            ];
            if include_pls {
                kinds.push(CheapLandmarkKind::RegressionPls);
            }
            kinds
        }
        TaskKind::Binary => vec![
            CheapLandmarkKind::BinaryPrior,
            CheapLandmarkKind::BinaryLogitRidge,
            CheapLandmarkKind::BinaryShallowTree,
            CheapLandmarkKind::BinaryShallowGam,
        ],
        TaskKind::Multiclass => Vec::new(),
    };
    let assignments = stratified_fold_assignments(&landmark_targets, LANDMARK_FOLDS);
    let mut predictions = vec![vec![0.0; landmark_targets.len()]; kinds.len()];

    for fold in 0..LANDMARK_FOLDS {
        let fit_rows: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(row, assigned)| (*assigned != fold).then_some(row))
            .collect();
        let valid_rows: Vec<usize> = assignments
            .iter()
            .enumerate()
            .filter_map(|(row, assigned)| (*assigned == fold).then_some(row))
            .collect();
        if fit_rows.is_empty() || valid_rows.is_empty() {
            continue;
        }
        let fit_table = subset_table(
            &landmark_table,
            &fit_rows,
            &(0..landmark_schema.columns.len()).collect::<Vec<_>>(),
        )?;
        let valid_table = subset_table(
            &landmark_table,
            &valid_rows,
            &(0..landmark_schema.columns.len()).collect::<Vec<_>>(),
        )?;
        let fit_targets: Vec<f64> = fit_rows.iter().map(|row| landmark_targets[*row]).collect();
        let fallback = fold_baseline(task.kind, &fit_targets);
        let input_stage = landmark_input_stage();
        let input_config = InputConfig::from_stage(Some(&input_stage))?;
        let encoded = FittedInput::fit_transform(
            input_config,
            &landmark_schema,
            &fit_table,
            &fit_targets,
            task.kind,
        );
        let (fitted_input, fit_block) = match encoded {
            Ok(encoded) => encoded,
            Err(_) => {
                for candidate in &mut predictions {
                    for &row in &valid_rows {
                        candidate[row] = fallback;
                    }
                }
                continue;
            }
        };
        let valid_block = match fitted_input.transform(&valid_table) {
            Ok(block) => block,
            Err(_) => {
                for candidate in &mut predictions {
                    for &row in &valid_rows {
                        candidate[row] = fallback;
                    }
                }
                continue;
            }
        };

        for (candidate, kind) in kinds.iter().enumerate() {
            let fold_predictions = match kind {
                CheapLandmarkKind::RegressionMean | CheapLandmarkKind::BinaryPrior => {
                    vec![fallback; valid_rows.len()]
                }
                _ => landmark_head_stage(*kind)
                    .and_then(|stage| {
                        fit_v3_head(&stage, &fit_block.values, &fit_targets, task.kind)
                    })
                    .and_then(|head| head.predict(&valid_block.values))
                    .unwrap_or_else(|_| vec![fallback; valid_rows.len()]),
            };
            for (&row, prediction) in valid_rows.iter().zip(fold_predictions) {
                predictions[candidate][row] = finite_prediction(task.kind, prediction, fallback);
            }
        }
    }

    let scores = kinds
        .into_iter()
        .zip(predictions)
        .map(|(kind, predictions)| CheapLandmarkScore {
            kind,
            score: finite_stat(task.score(&landmark_targets, &predictions, 0.5)),
        })
        .collect();
    Ok(CheapLandmarksV1 {
        rows: selected_rows.len(),
        folds: LANDMARK_FOLDS,
        scores,
    })
}

fn landmark_input_stage() -> Stage {
    let mut stage = Stage::new("input");
    set_string(&mut stage, "numeric", "median");
    set_string(&mut stage, "missing", "indicator");
    set_string(&mut stage, "categorical", "frequency");
    stage
}

fn landmark_head_stage(kind: CheapLandmarkKind) -> Result<Stage> {
    let mut stage = match kind {
        CheapLandmarkKind::RegressionRidge => Stage::new("ridge"),
        CheapLandmarkKind::RegressionShallowTree | CheapLandmarkKind::BinaryShallowTree => {
            let mut stage = Stage::new("extra_trees");
            set_integer(&mut stage, "trees", 16);
            set_integer(&mut stage, "depth", 2);
            set_integer(&mut stage, "min_leaf", 2);
            set_integer(&mut stage, "seed", 7);
            stage
        }
        CheapLandmarkKind::RegressionShallowGam | CheapLandmarkKind::BinaryShallowGam => {
            let mut stage = Stage::new("gam_boost");
            set_integer(&mut stage, "rounds", 16);
            set_integer(&mut stage, "bins", 8);
            set_float(&mut stage, "lr", 0.05);
            stage
        }
        CheapLandmarkKind::RegressionPls => {
            let mut stage = Stage::new("pls");
            set_integer(&mut stage, "k", 8);
            stage
        }
        CheapLandmarkKind::BinaryLogitRidge => {
            let mut stage = Stage::new("logit_ridge");
            set_float(&mut stage, "l2", 0.02);
            stage
        }
        CheapLandmarkKind::RegressionMean | CheapLandmarkKind::BinaryPrior => {
            bail!("baseline landmarks do not have fitted head stages")
        }
    };
    if matches!(
        kind,
        CheapLandmarkKind::BinaryLogitRidge
            | CheapLandmarkKind::BinaryShallowTree
            | CheapLandmarkKind::BinaryShallowGam
    ) {
        set_string(&mut stage, "class_weight", "none");
    }
    Ok(stage)
}

fn fold_baseline(task: TaskKind, targets: &[f64]) -> f64 {
    let mean = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
    match task {
        TaskKind::Regression | TaskKind::Timeseries => finite_stat(mean),
        TaskKind::Binary => mean.clamp(1e-6, 1.0 - 1e-6),
        TaskKind::Multiclass => targets.first().copied().unwrap_or(0.0),
    }
}

fn finite_prediction(task: TaskKind, prediction: f64, fallback: f64) -> f64 {
    if !prediction.is_finite() {
        return fallback;
    }
    match task {
        TaskKind::Regression | TaskKind::Timeseries => prediction,
        TaskKind::Binary => prediction.clamp(0.0, 1.0),
        TaskKind::Multiclass => prediction.round().max(0.0),
    }
}

fn set_integer(stage: &mut Stage, name: &str, value: i64) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Int(value));
}

fn set_float(stage: &mut Stage, name: &str, value: f64) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Float(value));
}

fn set_string(stage: &mut Stage, name: &str, value: &str) {
    stage
        .params
        .insert(name.to_string(), ParamValue::Str(value.to_string()));
}
