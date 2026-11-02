use super::routes::PredictorRouteV2;
use super::{validate_sha256, ROUTER_FOLDS_V2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationFoldV2 {
    pub origin: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub fit_groups: Vec<String>,
    pub hold_groups: Vec<String>,
}

fn fold_rows(groups: &[Vec<usize>], range: std::ops::Range<usize>) -> Vec<usize> {
    groups
        .get(range)
        .unwrap_or(&[])
        .iter()
        .flatten()
        .copied()
        .collect()
}

fn group_names(rows: &[usize], atomic_groups: &[String]) -> Result<Vec<String>, String> {
    let mut names = BTreeSet::new();
    for row in rows {
        names.insert(
            atomic_groups
                .get(*row)
                .ok_or_else(|| "fold row is out of range".to_string())?
                .clone(),
        );
    }
    Ok(names.into_iter().collect())
}

fn timestamp_groups(times: &[i64]) -> Vec<Vec<usize>> {
    let mut ordered = (0..times.len()).collect::<Vec<_>>();
    ordered.sort_by_key(|row| (times[*row], *row));
    let mut result = Vec::new();
    for row in ordered {
        if result
            .last()
            .and_then(|group: &Vec<usize>| group.first())
            .is_some_and(|first| times[*first] == times[row])
        {
            result.last_mut().expect("timestamp group exists").push(row);
        } else {
            result.push(vec![row]);
        }
    }
    result
}

pub fn build_validation_folds_v2(
    times: Option<&[i64]>,
    atomic_groups: &[String],
    scheme: &ValidationSchemeV2,
) -> Result<Vec<ValidationFoldV2>, String> {
    scheme.validate()?;
    if atomic_groups.is_empty() || atomic_groups.iter().any(|value| value.trim().is_empty()) {
        return Err("folds require nonempty atomic groups".into());
    }
    match scheme {
        ValidationSchemeV2::IidGrouped => {
            let mut by_group = BTreeMap::<String, Vec<usize>>::new();
            for (row, group) in atomic_groups.iter().enumerate() {
                by_group.entry(group.clone()).or_default().push(row);
            }
            if by_group.len() < 14 {
                return Err("IidGrouped requires at least fourteen atomic groups".into());
            }
            let mut ordered = by_group.keys().cloned().collect::<Vec<_>>();
            ordered.sort_by_key(|group| {
                let mut digest = Sha256::new();
                digest.update(format!("jain-router-v2|303|{group}").as_bytes());
                format!("{:x}", digest.finalize())
            });
            let mut hold_groups = vec![Vec::new(); ROUTER_FOLDS_V2];
            for (index, group) in ordered.into_iter().enumerate() {
                hold_groups[index % ROUTER_FOLDS_V2].push(group);
            }
            let all_rows = (0..atomic_groups.len()).collect::<BTreeSet<_>>();
            let all_groups = by_group.keys().cloned().collect::<BTreeSet<_>>();
            let mut coverage = vec![0u8; atomic_groups.len()];
            let mut folds = Vec::new();
            for (index, groups) in hold_groups.into_iter().enumerate() {
                let hold_set = groups.into_iter().collect::<BTreeSet<_>>();
                let hold_rows = hold_set
                    .iter()
                    .flat_map(|group| by_group[group].iter().copied())
                    .collect::<Vec<_>>();
                let hold_row_set = hold_rows.iter().copied().collect::<BTreeSet<_>>();
                let fit_rows = all_rows
                    .difference(&hold_row_set)
                    .copied()
                    .collect::<Vec<_>>();
                if fit_rows.len() < 32 || hold_rows.len() < 2 || hold_set.len() < 2 {
                    return Err("IID fold violates the minimum row/group guards".into());
                }
                for row in &hold_rows {
                    coverage[*row] += 1;
                }
                folds.push(ValidationFoldV2 {
                    origin: index as u32,
                    fit_rows,
                    hold_rows,
                    fit_groups: all_groups.difference(&hold_set).cloned().collect(),
                    hold_groups: hold_set.into_iter().collect(),
                });
            }
            if coverage.iter().any(|value| *value != 1) {
                return Err("IID folds do not cover each row exactly once".into());
            }
            Ok(folds)
        }
        ValidationSchemeV2::TimeSeries {
            seasonal_period,
            purge_groups,
            embargo_groups,
            ..
        } => {
            let times = times.ok_or_else(|| "time values are required".to_string())?;
            if times.len() != atomic_groups.len() {
                return Err("time and group lengths differ".into());
            }
            let groups = timestamp_groups(times);
            let mut seen = BTreeMap::<String, usize>::new();
            for (index, rows) in groups.iter().enumerate() {
                for row in rows {
                    if let Some(previous) = seen.insert(atomic_groups[*row].clone(), index) {
                        if previous != index {
                            return Err("an atomic group spans multiple timestamps".into());
                        }
                    }
                }
            }
            let minimum_fit_rows = 32usize.max(2 * seasonal_period);
            let mut fit_end = 0;
            while fit_end < groups.len() && fold_rows(&groups, 0..fit_end).len() < minimum_fit_rows
            {
                fit_end += 1;
            }
            let remaining = groups.len().saturating_sub(fit_end + purge_groups);
            let hold_size = (remaining / 7).max(1);
            let mut folds = Vec::new();
            while folds.len() < 7 {
                let hold_start = fit_end + purge_groups;
                if hold_start >= groups.len() {
                    break;
                }
                let hold_end = (hold_start + hold_size).min(groups.len());
                let fit_rows = fold_rows(&groups, 0..fit_end);
                let hold_rows = fold_rows(&groups, hold_start..hold_end);
                if fit_rows.len() >= minimum_fit_rows && !hold_rows.is_empty() {
                    let fit_groups = group_names(&fit_rows, atomic_groups)?;
                    let hold_groups = group_names(&hold_rows, atomic_groups)?;
                    if fit_groups.iter().any(|group| hold_groups.contains(group)) {
                        return Err("time-series fold leaks an atomic group".into());
                    }
                    folds.push(ValidationFoldV2 {
                        origin: folds.len() as u32,
                        fit_rows,
                        hold_rows,
                        fit_groups,
                        hold_groups,
                    });
                }
                fit_end = hold_end + embargo_groups;
            }
            if folds.len() < 3 {
                return Err(format!("only {} valid time-series origins", folds.len()));
            }
            Ok(folds)
        }
    }
}

pub fn predictor_columns_without_time(
    columns: &[String],
    time_column: &str,
) -> Result<Vec<String>, String> {
    if time_column.trim().is_empty() || !columns.iter().any(|column| column == time_column) {
        return Err("time column is absent from the schema".into());
    }
    Ok(columns
        .iter()
        .filter(|column| column.as_str() != time_column)
        .cloned()
        .collect())
}

pub fn mase_v2(
    fit_targets: &[f64],
    hold_targets: &[f64],
    predictions: &[f64],
    seasonal_period: usize,
) -> Result<f64, String> {
    if seasonal_period == 0
        || fit_targets.len() <= seasonal_period
        || hold_targets.is_empty()
        || hold_targets.len() != predictions.len()
        || fit_targets
            .iter()
            .chain(hold_targets)
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        return Err("invalid MASE inputs".into());
    }
    let denominator = (seasonal_period..fit_targets.len())
        .map(|row| (fit_targets[row] - fit_targets[row - seasonal_period]).abs())
        .sum::<f64>()
        / (fit_targets.len() - seasonal_period) as f64;
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err("MASE denominator is zero or nonfinite".into());
    }
    Ok(hold_targets
        .iter()
        .zip(predictions)
        .map(|(actual, predicted)| (actual - predicted).abs())
        .sum::<f64>()
        / hold_targets.len() as f64
        / denominator)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactV4 {
    pub schema: String,
    pub task: String,
    pub selected_route: PredictorRouteV2,
    pub output_width: usize,
    pub class_label_names: Vec<String>,
    pub predictor_sha256: String,
}

impl ModelArtifactV4 {
    pub const SCHEMA: &'static str = "JainModelArtifact.v4";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.task.trim().is_empty() || self.output_width == 0 {
            return Err("model artifact v4 is invalid".into());
        }
        self.selected_route.validate()?;
        validate_sha256("model predictor", &self.predictor_sha256)?;
        if !self.class_label_names.is_empty() && self.class_label_names.len() != self.output_width {
            return Err("model class order does not match output width".into());
        }
        Ok(())
    }

    pub fn from_legacy_v3(
        task: impl Into<String>,
        output_width: usize,
        class_label_names: Vec<String>,
        predictor_sha256: String,
    ) -> Result<Self, String> {
        let artifact = Self {
            schema: Self::SCHEMA.into(),
            task: task.into(),
            selected_route: PredictorRouteV2::raw(),
            output_width,
            class_label_names,
            predictor_sha256,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV6 {
    pub schema: String,
    pub task: String,
    pub selected_route: PredictorRouteV2,
    pub output_width: usize,
    pub class_label_names: Vec<String>,
    pub model_artifact_sha256: String,
}

impl RunManifestV6 {
    pub const SCHEMA: &'static str = "JainRunManifest.v6";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.task.trim().is_empty() || self.output_width == 0 {
            return Err("run manifest v6 is invalid".into());
        }
        self.selected_route.validate()?;
        validate_sha256("run model artifact", &self.model_artifact_sha256)?;
        if !self.class_label_names.is_empty() && self.class_label_names.len() != self.output_width {
            return Err("run manifest class order does not match output width".into());
        }
        Ok(())
    }

    pub fn from_legacy_v5(
        task: impl Into<String>,
        output_width: usize,
        class_label_names: Vec<String>,
        model_artifact_sha256: String,
    ) -> Result<Self, String> {
        let manifest = Self {
            schema: Self::SCHEMA.into(),
            task: task.into(),
            selected_route: PredictorRouteV2::raw(),
            output_width,
            class_label_names,
            model_artifact_sha256,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}
