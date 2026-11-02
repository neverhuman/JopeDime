use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn score_view(
    view: FeatureViewKindV1,
    dataset: &TrainOnlyDatasetV1,
    source: &RegressionTable,
    fold: &GroupedFoldV1,
    cell_fit: &[Vec<f64>],
    cell_hold: &[Vec<f64>],
    row_fit: &[Vec<f64>],
    row_hold: &[Vec<f64>],
    predictor: &FrozenPredictorV1,
    adapter: &CorpusEvalAdapter,
) -> Result<Vec<f64>, String> {
    let fit_raw = select_rows(&source.features, &fold.fit_rows);
    let hold_raw = select_rows(&source.features, &fold.hold_rows);
    let fit_features = if view == FeatureViewKindV1::Raw {
        fit_raw
    } else {
        assemble_feature_view(view, &fit_raw, cell_fit, row_fit)?
    };
    let hold_features = if view == FeatureViewKindV1::Raw {
        hold_raw
    } else {
        assemble_feature_view(view, &hold_raw, cell_hold, row_hold)?
    };
    let fit_position = fold
        .fit_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let hold_position = fold
        .hold_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let rows = (0..source.targets.len())
        .map(|row| {
            let (features, target) = if let Some(position) = fit_position.get(&row) {
                (&fit_features[*position], source.targets[row])
            } else {
                (&hold_features[hold_position[&row]], 0.0)
            };
            let mut fields = features
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>();
            fields.push(target.to_string());
            DatasetRowV1 {
                split: SplitRowV1 {
                    stable_row_id: dataset.stable_row_ids[row].clone(),
                    atomic_group: dataset.atomic_groups[row].clone(),
                    regression_target: Some(source.targets[row]),
                    binary_target: None,
                },
                csv_line: fields.join(","),
            }
        })
        .collect::<Vec<_>>();
    let csv_bytes = rows
        .iter()
        .flat_map(|row| [row.csv_line.as_bytes(), b"\n"].concat())
        .collect::<Vec<_>>();
    let table = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: dataset.dataset_sha256.clone(),
        training_sha256: raw_sha256(&csv_bytes),
        rows,
    };
    table.validate()?;
    let mut fit_ids = fold
        .fit_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    let mut hold_ids = fold
        .hold_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    fit_ids.sort();
    hold_ids.sort();
    let split = RowSplitV1 {
        split_index: fold.fold_index,
        seed: 1,
        train_row_ids: fit_ids,
        validation_row_ids: hold_ids.clone(),
        validation_groups: fold.hold_groups.clone(),
        balance_counts: BTreeMap::new(),
    };
    let candidates = predictor
        .pipelines()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let outcomes = adapter.evaluate_search_split(&table, &split, &candidates)?;
    if outcomes.len() != candidates.len()
        || outcomes.iter().any(|outcome| !outcome.prediction.valid)
    {
        return Err("AFPM evaluator returned incomplete or invalid fold predictions".into());
    }
    let components = outcomes
        .iter()
        .map(|outcome| {
            if outcome.prediction.validation_row_ids != hold_ids {
                return Err("AFPM predictions changed hold-row ordering".into());
            }
            Ok(outcome
                .prediction
                .prediction_f64_bits
                .iter()
                .map(|bits| f64::from_bits(*bits))
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let sorted_prediction = predictor.reconstruct(&components)?;
    let by_id = hold_ids
        .into_iter()
        .zip(sorted_prediction)
        .collect::<BTreeMap<_, _>>();
    fold.hold_rows
        .iter()
        .map(|row| {
            by_id
                .get(&dataset.stable_row_ids[*row])
                .copied()
                .ok_or_else(|| "AFPM prediction omitted a hold row".to_string())
        })
        .collect()
}
