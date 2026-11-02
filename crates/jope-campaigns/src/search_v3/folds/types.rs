use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabularRowV2 {
    pub stable_row_id: String,
    pub group_id: Option<String>,
    pub duplicate_signature: String,
    pub regression_target: Option<f64>,
    pub class_label: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabularFoldInputV2 {
    pub task: SearchTaskV1,
    pub training_view_sha256: String,
    pub rows: Vec<TabularRowV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeseriesRowV2 {
    pub stable_row_id: String,
    pub time_key: i64,
    pub feature_max_time_key: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeseriesOriginV2 {
    pub train_end_index: usize,
    pub validation_len: usize,
    pub purge_rows: usize,
    pub embargo_rows: usize,
    pub required_seasonal_history: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeseriesFoldInputV2 {
    pub training_view_sha256: String,
    pub rows: Vec<TimeseriesRowV2>,
    pub origins: Vec<TimeseriesOriginV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldMembershipV2 {
    pub fold_index: u32,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_group_sha256s: Vec<String>,
    pub balance_counts: BTreeMap<String, u64>,
    pub purge_rows: usize,
    pub embargo_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldPlanV2 {
    pub schema: String,
    pub task: SearchTaskV1,
    pub training_view_sha256: String,
    pub stable_row_ids: Vec<String>,
    pub row_group_sha256s: BTreeMap<String, String>,
    pub row_time_keys: BTreeMap<String, i64>,
    pub folds: Vec<FoldMembershipV2>,
}

impl FoldPlanV2 {
    pub const SCHEMA: &'static str = "FoldPlan.v2";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("training_view_sha256", &self.training_view_sha256)?;
        if self.stable_row_ids.is_empty() {
            return Err("fold plan has no stable rows".into());
        }
        let all_rows: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all_rows.len() != self.stable_row_ids.len() {
            return Err("stable row IDs must be unique".into());
        }
        let expected_folds = if self.task == SearchTaskV1::Timeseries {
            if self.folds.len() < 3 {
                return Err("timeseries fold plan requires at least three origins".into());
            }
            self.folds.len()
        } else {
            if self.folds.len() != 5 {
                return Err("tabular fold plan requires exactly five folds".into());
            }
            5
        };
        let mut tabular_validation_counts = BTreeMap::<&str, usize>::new();
        let mut group_fold = BTreeMap::<&str, u32>::new();
        for (expected, fold) in self.folds.iter().enumerate() {
            if fold.fold_index != expected as u32 {
                return Err("folds must be stored in contiguous index order".into());
            }
            let train: BTreeSet<_> = fold.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = fold.validation_row_ids.iter().collect();
            if train.len() != fold.train_row_ids.len()
                || validation.len() != fold.validation_row_ids.len()
                || !train.is_disjoint(&validation)
                || !train
                    .iter()
                    .chain(validation.iter())
                    .all(|row| all_rows.contains(row))
            {
                return Err(format!(
                    "fold {expected} has duplicate, overlapping, or unknown rows"
                ));
            }
            if validation.is_empty() {
                return Err(format!("fold {expected} has no validation rows"));
            }
            if self.task != SearchTaskV1::Timeseries {
                if train.len() + validation.len() != all_rows.len() {
                    return Err(format!(
                        "tabular fold {expected} is not a complete row partition"
                    ));
                }
                for row in &fold.validation_row_ids {
                    *tabular_validation_counts.entry(row).or_default() += 1;
                    let group = self
                        .row_group_sha256s
                        .get(row)
                        .ok_or_else(|| format!("row {row} has no atomic group"))?;
                    match group_fold.insert(group, fold.fold_index) {
                        Some(previous) if previous != fold.fold_index => {
                            return Err(format!("atomic group {group} crosses validation folds"));
                        }
                        _ => {}
                    }
                }
            } else {
                if fold.train_row_ids.is_empty() {
                    return Err(format!(
                        "timeseries origin {expected} has no training prefix"
                    ));
                }
                let positions: BTreeMap<_, _> = self
                    .stable_row_ids
                    .iter()
                    .enumerate()
                    .map(|(index, row)| (row, index))
                    .collect();
                if !fold
                    .train_row_ids
                    .iter()
                    .enumerate()
                    .all(|(index, row)| positions.get(row) == Some(&index))
                {
                    return Err(format!(
                        "timeseries origin {expected} training set is not a strict prefix"
                    ));
                }
                let validation_start =
                    fold.train_row_ids.len() + fold.purge_rows + fold.embargo_rows;
                if !fold
                    .validation_row_ids
                    .iter()
                    .enumerate()
                    .all(|(offset, row)| positions.get(row) == Some(&(validation_start + offset)))
                {
                    return Err(format!(
                        "timeseries origin {expected} validation is not the next contiguous block"
                    ));
                }
            }
        }
        if self.task != SearchTaskV1::Timeseries
            && (tabular_validation_counts.len() != all_rows.len()
                || tabular_validation_counts.values().any(|count| *count != 1))
        {
            return Err("every tabular row must occur in exactly one validation fold".into());
        }
        debug_assert_eq!(expected_folds, self.folds.len());
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
