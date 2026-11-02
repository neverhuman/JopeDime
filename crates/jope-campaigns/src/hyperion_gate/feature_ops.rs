use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct FittedFeatureBlocksV1 {
    cell_sorted: Vec<Vec<f64>>,
    row_means: Vec<f64>,
    row_scales: Vec<f64>,
}

impl FittedFeatureBlocksV1 {
    pub fn fit(cell: &[Vec<f64>], row: &[Vec<f64>], raw_width: usize) -> Result<Self, String> {
        Self::fit_available(cell, Some(row), raw_width)
    }

    pub fn fit_available(
        cell: &[Vec<f64>],
        row: Option<&[Vec<f64>]>,
        raw_width: usize,
    ) -> Result<Self, String> {
        validate_matrix(cell, raw_width, "cell fit")?;
        if let Some(row) = row {
            validate_matrix(row, ROW_EMBEDDING_WIDTH, "row fit")?;
            if cell.len() != row.len() {
                return Err("cell and row fit sidecars are row-misaligned".into());
            }
        }
        let mut cell_sorted = vec![Vec::with_capacity(cell.len()); raw_width];
        for values in cell {
            for (column, value) in values.iter().enumerate() {
                cell_sorted[column].push(*value);
            }
        }
        for values in &mut cell_sorted {
            values.sort_by(f64::total_cmp);
        }
        let mut row_means = Vec::new();
        let mut row_scales = Vec::new();
        if let Some(row) = row {
            row_means = vec![0.0; ROW_EMBEDDING_WIDTH];
            for values in row {
                for (column, value) in values.iter().enumerate() {
                    row_means[column] += value / row.len() as f64;
                }
            }
            row_scales = vec![0.0; ROW_EMBEDDING_WIDTH];
            for values in row {
                for (column, value) in values.iter().enumerate() {
                    row_scales[column] += (value - row_means[column]).powi(2) / row.len() as f64;
                }
            }
            row_scales
                .iter_mut()
                .for_each(|value| *value = value.sqrt().max(1e-12));
        }
        Ok(Self {
            cell_sorted,
            row_means,
            row_scales,
        })
    }

    pub fn transform_cell(&self, values: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        validate_matrix(values, self.cell_sorted.len(), "cell transform")?;
        Ok(values
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        let sorted = &self.cell_sorted[column];
                        let lower = sorted.partition_point(|candidate| candidate < value);
                        let upper = sorted.partition_point(|candidate| candidate <= value);
                        (lower + upper) as f64 / (2.0 * sorted.len() as f64)
                    })
                    .collect()
            })
            .collect())
    }

    pub fn transform_row(&self, values: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.row_means.len() != ROW_EMBEDDING_WIDTH
            || self.row_scales.len() != ROW_EMBEDDING_WIDTH
        {
            return Err("row128 preprocessing was not fitted".into());
        }
        validate_matrix(values, ROW_EMBEDDING_WIDTH, "row transform")?;
        Ok(values
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        (value - self.row_means[column]) / self.row_scales[column]
                    })
                    .collect()
            })
            .collect())
    }
}

pub fn assemble_feature_view(
    view: FeatureViewKindV1,
    raw: &[Vec<f64>],
    cell: &[Vec<f64>],
    row: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let raw_width = raw.first().map(Vec::len).unwrap_or(0);
    validate_matrix(raw, raw_width, "raw")?;
    if matches!(
        view,
        FeatureViewKindV1::HyperionCell | FeatureViewKindV1::HyperionBoth
    ) {
        if cell.len() != raw.len() {
            return Err("cell feature block is row-misaligned".into());
        }
        validate_matrix(cell, raw_width, "cell")?;
    }
    if matches!(
        view,
        FeatureViewKindV1::HyperionRow128 | FeatureViewKindV1::HyperionBoth
    ) {
        if row.len() != raw.len() {
            return Err("row128 feature block is row-misaligned".into());
        }
        validate_matrix(row, ROW_EMBEDDING_WIDTH, "row")?;
    }
    Ok((0..raw.len())
        .map(|index| {
            let mut values = raw[index].clone();
            match view {
                FeatureViewKindV1::Raw => {}
                FeatureViewKindV1::HyperionCell => values.extend_from_slice(&cell[index]),
                FeatureViewKindV1::HyperionRow128 => values.extend_from_slice(&row[index]),
                FeatureViewKindV1::HyperionBoth => {
                    values.extend_from_slice(&cell[index]);
                    values.extend_from_slice(&row[index]);
                }
            }
            values
        })
        .collect())
}

fn validate_matrix(values: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    if values.is_empty()
        || width == 0
        || values
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
    {
        return Err(format!("{name} matrix has an invalid shape or value"));
    }
    Ok(())
}

pub(crate) fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    (actual
        .iter()
        .zip(predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt()
}

pub(crate) fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

fn bootstrap_lower_bound(
    folds: &[PairedFoldEvidenceV1],
    arm: FeatureViewKindV1,
    confidence: f64,
    seed: u64,
) -> Result<f64, String> {
    let mut groups = BTreeMap::<&str, (Vec<f64>, Vec<f64>)>::new();
    for fold in folds {
        let predictions = &fold.arm_predictions[&arm];
        for (index, prediction) in predictions.iter().enumerate() {
            let entry = groups
                .entry(fold.atomic_groups[index].as_str())
                .or_default();
            entry
                .0
                .push((fold.actual[index] - fold.raw_predictions[index]).powi(2));
            entry.1.push((fold.actual[index] - *prediction).powi(2));
        }
    }
    if groups.len() < GateThresholdsV1::default().minimum_atomic_groups as usize {
        return Err("insufficient atomic groups for paired bootstrap".into());
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut state = seed.max(1);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut raw_sum = 0.0;
        let mut arm_sum = 0.0;
        let mut rows = 0usize;
        for _ in 0..groups.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let group = &groups[state as usize % groups.len()];
            raw_sum += group.0.iter().sum::<f64>();
            arm_sum += group.1.iter().sum::<f64>();
            rows += group.0.len();
        }
        let raw_value = (raw_sum / rows as f64).sqrt();
        let arm_value = (arm_sum / rows as f64).sqrt();
        let improvement = ((raw_value + 1e-300) / (arm_value + 1e-300)).ln();
        samples.push(improvement);
    }
    samples.sort_by(f64::total_cmp);
    let quantile = (1.0 - confidence).clamp(0.0, 1.0);
    let index = (quantile * (samples.len() - 1) as f64).floor() as usize;
    Ok(samples[index])
}

fn evaluate_arm(
    stage: GateStageV1,
    folds: &[PairedFoldEvidenceV1],
    arm: FeatureViewKindV1,
    thresholds: &GateThresholdsV1,
) -> Result<StageGateResultV1, String> {
    let mut actual = Vec::new();
    let mut raw_predictions = Vec::new();
    let mut arm_predictions = Vec::new();
    let mut fold_wins = 0u32;
    let mut fold_best_arm_wins = 0u32;
    let mut worst_fold_regression = f64::NEG_INFINITY;
    for fold in folds {
        let arm_values = &fold.arm_predictions[&arm];
        let raw_fold = rmse(&fold.actual, &fold.raw_predictions);
        let arm_fold = rmse(&fold.actual, arm_values);
        if arm_fold < raw_fold {
            fold_wins += 1;
        }
        let relative_regression = if raw_fold <= 1e-300 {
            if arm_fold <= 1e-300 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            arm_fold / raw_fold - 1.0
        };
        worst_fold_regression = worst_fold_regression.max(relative_regression);
        let best = fold
            .arm_predictions
            .iter()
            .min_by(|(left_arm, left), (right_arm, right)| {
                rmse(&fold.actual, left)
                    .total_cmp(&rmse(&fold.actual, right))
                    .then_with(|| left_arm.cmp(right_arm))
            })
            .map(|(value, _)| *value)
            .unwrap();
        if best == arm {
            fold_best_arm_wins += 1;
        }
        actual.extend_from_slice(&fold.actual);
        raw_predictions.extend_from_slice(&fold.raw_predictions);
        arm_predictions.extend_from_slice(arm_values);
    }
    let pooled_raw_rmse = rmse(&actual, &raw_predictions);
    let pooled_arm_rmse = rmse(&actual, &arm_predictions);
    let relative_improvement = if pooled_raw_rmse <= 1e-300 {
        0.0
    } else {
        1.0 - pooled_arm_rmse / pooled_raw_rmse
    };
    let confidence = match stage {
        GateStageV1::Cv5Screen => thresholds.cv5_bootstrap_one_sided_confidence,
        GateStageV1::Cv7Confirmation => thresholds.cv7_bootstrap_one_sided_confidence,
    };
    let bootstrap_lower_bound_log_improvement = bootstrap_lower_bound(
        folds,
        arm,
        confidence,
        seed64(&[&format!("{stage:?}"), &format!("{arm:?}")]),
    )?;
    let wins_pass = match stage {
        GateStageV1::Cv5Screen => {
            fold_wins >= thresholds.cv5_minimum_fold_wins
                && fold_best_arm_wins >= thresholds.cv5_minimum_best_arm_wins
        }
        GateStageV1::Cv7Confirmation => fold_wins >= thresholds.cv7_minimum_fold_wins,
    };
    let passed = wins_pass
        && relative_improvement >= thresholds.minimum_relative_improvement
        && bootstrap_lower_bound_log_improvement > 0.0
        && worst_fold_regression <= thresholds.maximum_fold_regression;
    Ok(StageGateResultV1 {
        stage,
        arm,
        folds: folds.len() as u32,
        fold_wins,
        fold_best_arm_wins,
        pooled_raw_rmse,
        pooled_arm_rmse,
        relative_improvement,
        bootstrap_lower_bound_log_improvement,
        worst_fold_regression,
        passed,
    })
}

pub fn screen_cv5(
    folds: &[PairedFoldEvidenceV1],
    thresholds: &GateThresholdsV1,
) -> Result<(Vec<StageGateResultV1>, Option<FeatureViewKindV1>), String> {
    screen_cv5_arms(folds, &FeatureViewKindV1::EXPANDED, thresholds)
}

/// Screen exactly the independently available arms.  Missing or structurally
/// ineligible row/both representations must never suppress a valid cell arm.
pub fn screen_cv5_arms(
    folds: &[PairedFoldEvidenceV1],
    arms: &[FeatureViewKindV1],
    thresholds: &GateThresholdsV1,
) -> Result<(Vec<StageGateResultV1>, Option<FeatureViewKindV1>), String> {
    if arms.is_empty()
        || arms
            .iter()
            .any(|arm| *arm == FeatureViewKindV1::Raw || !FeatureViewKindV1::EXPANDED.contains(arm))
        || arms.iter().collect::<BTreeSet<_>>().len() != arms.len()
    {
        return Err("CV5 requires a nonempty unique expanded-arm set".into());
    }
    validate_stage_folds(folds, 5, arms)?;
    let mut results = arms
        .iter()
        .map(|arm| evaluate_arm(GateStageV1::Cv5Screen, folds, *arm, thresholds))
        .collect::<Result<Vec<_>, _>>()?;
    results.sort_by_key(|value| value.arm);
    let advanced = results
        .iter()
        .filter(|result| result.passed)
        .min_by(|left, right| {
            right
                .relative_improvement
                .total_cmp(&left.relative_improvement)
                .then_with(|| left.arm.cmp(&right.arm))
        })
        .map(|result| result.arm);
    Ok((results, advanced))
}

pub fn confirm_cv7(
    folds: &[PairedFoldEvidenceV1],
    frozen_arm: FeatureViewKindV1,
    thresholds: &GateThresholdsV1,
) -> Result<StageGateResultV1, String> {
    if frozen_arm == FeatureViewKindV1::Raw {
        return Err("CV7 cannot confirm the raw control as an expansion".into());
    }
    validate_stage_folds(folds, 7, &[frozen_arm])?;
    evaluate_arm(GateStageV1::Cv7Confirmation, folds, frozen_arm, thresholds)
}
