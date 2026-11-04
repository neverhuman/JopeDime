use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::contracts::SearchTaskV1;
use super::identity::{canonical_sha256, validate_schema, validate_sha256};

mod types;
pub use types::*;

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn root(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            self.parent[index] = self.root(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.root(left);
        let right = self.root(right);
        if left != right {
            let (low, high) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            self.parent[high] = low;
        }
    }
}

#[derive(Debug)]
struct AtomicGroup {
    digest: String,
    rows: Vec<usize>,
    balance: BTreeMap<String, u64>,
}

pub fn build_tabular_fold_plan(input: &TabularFoldInputV2) -> Result<FoldPlanV2, String> {
    validate_sha256("training_view_sha256", &input.training_view_sha256)?;
    if input.task == SearchTaskV1::Timeseries {
        return Err("timeseries must use build_timeseries_fold_plan".into());
    }
    if input.rows.len() < 5 {
        return Err("tabular fold construction requires at least five rows".into());
    }
    let mut row_ids = BTreeSet::new();
    for row in &input.rows {
        if row.stable_row_id.is_empty() || !row_ids.insert(row.stable_row_id.as_str()) {
            return Err("stable row IDs must be non-empty and unique".into());
        }
        if row.duplicate_signature.is_empty() {
            return Err("every tabular row requires a duplicate-row signature".into());
        }
        match input.task {
            SearchTaskV1::Regression => {
                if !row.regression_target.is_some_and(f64::is_finite) || row.class_label.is_some() {
                    return Err("regression rows require only a finite regression target".into());
                }
            }
            SearchTaskV1::Binary => {
                if !matches!(row.class_label, Some(0 | 1)) || row.regression_target.is_some() {
                    return Err("binary rows require only class label 0 or 1".into());
                }
            }
            SearchTaskV1::Multiclass => {
                if row.class_label.is_none() || row.regression_target.is_some() {
                    return Err("multiclass rows require only a class label".into());
                }
            }
            SearchTaskV1::Timeseries => unreachable!(),
        }
    }

    let mut row_balance = vec![String::new(); input.rows.len()];
    if input.task == SearchTaskV1::Regression {
        let mut order: Vec<_> = (0..input.rows.len()).collect();
        order.sort_by(|left, right| {
            input.rows[*left]
                .regression_target
                .unwrap()
                .total_cmp(&input.rows[*right].regression_target.unwrap())
                .then_with(|| {
                    input.rows[*left]
                        .stable_row_id
                        .cmp(&input.rows[*right].stable_row_id)
                })
        });
        let len = order.len();
        for (rank, row_index) in order.into_iter().enumerate() {
            row_balance[row_index] = format!("target_decile_{}", (rank * 10 / len).min(9));
        }
    } else {
        for (index, row) in input.rows.iter().enumerate() {
            row_balance[index] = format!("class_{}", row.class_label.unwrap());
        }
    }

    let mut disjoint = DisjointSet::new(input.rows.len());
    let mut first_group = BTreeMap::<&str, usize>::new();
    let mut first_duplicate = BTreeMap::<&str, usize>::new();
    for (index, row) in input.rows.iter().enumerate() {
        if let Some(group_id) = row.group_id.as_deref() {
            if group_id.is_empty() {
                return Err("explicit group IDs must not be empty".into());
            }
            if let Some(first) = first_group.insert(group_id, index) {
                disjoint.union(first, index);
            }
        }
        if let Some(first) = first_duplicate.insert(row.duplicate_signature.as_str(), index) {
            disjoint.union(first, index);
        }
    }

    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..input.rows.len() {
        let root = disjoint.root(index);
        components.entry(root).or_default().push(index);
    }
    if components.len() < 5 {
        return Err("atomic grouping leaves fewer than five assignable groups".into());
    }
    let mut groups = Vec::with_capacity(components.len());
    for mut rows in components.into_values() {
        rows.sort_by_key(|index| input.rows[*index].stable_row_id.as_str());
        let group_rows: Vec<_> = rows
            .iter()
            .map(|index| input.rows[*index].stable_row_id.as_str())
            .collect();
        let digest = canonical_sha256(&group_rows)?;
        let mut balance = BTreeMap::new();
        for row in &rows {
            *balance.entry(row_balance[*row].clone()).or_default() += 1;
        }
        groups.push(AtomicGroup {
            digest,
            rows,
            balance,
        });
    }
    groups.sort_by(|left, right| left.digest.cmp(&right.digest));

    if input.task != SearchTaskV1::Regression {
        let required: BTreeSet<_> = row_balance.iter().cloned().collect();
        for label in required {
            let supporting_groups = groups
                .iter()
                .filter(|group| group.balance.contains_key(&label))
                .count();
            if supporting_groups < 5 {
                return Err(format!(
                    "{label} occurs in only {supporting_groups} atomic groups; every validation fold requires support"
                ));
            }
        }
    }

    let mut assigned: [Vec<&AtomicGroup>; 5] = std::array::from_fn(|_| Vec::new());
    let mut fold_sizes = [0usize; 5];
    let mut fold_balance: [BTreeMap<String, u64>; 5] = std::array::from_fn(|_| BTreeMap::new());
    for group in &groups {
        let fold = (0..5)
            .min_by_key(|fold| {
                let missing_gain = group
                    .balance
                    .keys()
                    .filter(|key| !fold_balance[*fold].contains_key(*key))
                    .count();
                let balance_load: u64 = group
                    .balance
                    .keys()
                    .map(|key| *fold_balance[*fold].get(key).unwrap_or(&0))
                    .sum();
                (
                    usize::MAX - missing_gain,
                    balance_load,
                    fold_sizes[*fold],
                    *fold,
                )
            })
            .unwrap();
        assigned[fold].push(group);
        fold_sizes[fold] += group.rows.len();
        for (key, count) in &group.balance {
            *fold_balance[fold].entry(key.clone()).or_default() += count;
        }
    }

    if assigned.iter().any(Vec::is_empty) {
        return Err("deterministic group assignment produced an empty fold".into());
    }
    if input.task != SearchTaskV1::Regression {
        let required: BTreeSet<_> = row_balance.iter().cloned().collect();
        if let Some(label) = required.iter().find(|label| {
            fold_balance
                .iter()
                .any(|counts| !counts.contains_key(*label))
        }) {
            return Err(format!(
                "grouping makes complete per-fold support for {label} impossible"
            ));
        }
    }

    let stable_row_ids: Vec<_> = input
        .rows
        .iter()
        .map(|row| row.stable_row_id.clone())
        .collect();
    let mut row_group_sha256s = BTreeMap::new();
    for group in &groups {
        for index in &group.rows {
            row_group_sha256s.insert(
                input.rows[*index].stable_row_id.clone(),
                group.digest.clone(),
            );
        }
    }
    let folds = assigned
        .iter()
        .enumerate()
        .map(|(fold_index, groups)| {
            let validation_indices: BTreeSet<_> = groups
                .iter()
                .flat_map(|group| group.rows.iter().copied())
                .collect();
            FoldMembershipV2 {
                fold_index: fold_index as u32,
                train_row_ids: input
                    .rows
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !validation_indices.contains(index))
                    .map(|(_, row)| row.stable_row_id.clone())
                    .collect(),
                validation_row_ids: input
                    .rows
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| validation_indices.contains(index))
                    .map(|(_, row)| row.stable_row_id.clone())
                    .collect(),
                validation_group_sha256s: groups.iter().map(|group| group.digest.clone()).collect(),
                balance_counts: fold_balance[fold_index].clone(),
                purge_rows: 0,
                embargo_rows: 0,
            }
        })
        .collect();
    let plan = FoldPlanV2 {
        schema: FoldPlanV2::SCHEMA.into(),
        task: input.task,
        training_view_sha256: input.training_view_sha256.clone(),
        stable_row_ids,
        row_group_sha256s,
        row_time_keys: BTreeMap::new(),
        folds,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn build_timeseries_fold_plan(input: &TimeseriesFoldInputV2) -> Result<FoldPlanV2, String> {
    validate_sha256("training_view_sha256", &input.training_view_sha256)?;
    if input.origins.len() < 3 {
        return Err("timeseries requires at least three declared rolling origins".into());
    }
    let mut ids = BTreeSet::new();
    for (index, row) in input.rows.iter().enumerate() {
        if row.stable_row_id.is_empty() || !ids.insert(row.stable_row_id.as_str()) {
            return Err("timeseries stable row IDs must be non-empty and unique".into());
        }
        if index > 0 && input.rows[index - 1].time_key >= row.time_key {
            return Err("timeseries rows must be in strictly increasing time-key order".into());
        }
        if row.feature_max_time_key > row.time_key {
            return Err(format!(
                "row {} contains a future-derived feature",
                row.stable_row_id
            ));
        }
    }
    let mut previous_origin = None;
    let mut folds = Vec::with_capacity(input.origins.len());
    for (fold_index, origin) in input.origins.iter().enumerate() {
        if origin.train_end_index == 0
            || origin.validation_len == 0
            || origin.train_end_index < origin.required_seasonal_history
        {
            return Err(format!(
                "origin {fold_index} has insufficient training or seasonal history"
            ));
        }
        if previous_origin.is_some_and(|previous| origin.train_end_index <= previous) {
            return Err("rolling origins must advance monotonically".into());
        }
        previous_origin = Some(origin.train_end_index);
        let validation_start = origin
            .train_end_index
            .checked_add(origin.purge_rows)
            .and_then(|value| value.checked_add(origin.embargo_rows))
            .ok_or_else(|| "timeseries origin index overflow".to_string())?;
        let validation_end = validation_start
            .checked_add(origin.validation_len)
            .ok_or_else(|| "timeseries validation index overflow".to_string())?;
        if validation_end > input.rows.len() {
            return Err(format!(
                "origin {fold_index} validation exceeds the authorized rows"
            ));
        }
        folds.push(FoldMembershipV2 {
            fold_index: fold_index as u32,
            train_row_ids: input.rows[..origin.train_end_index]
                .iter()
                .map(|row| row.stable_row_id.clone())
                .collect(),
            validation_row_ids: input.rows[validation_start..validation_end]
                .iter()
                .map(|row| row.stable_row_id.clone())
                .collect(),
            validation_group_sha256s: Vec::new(),
            balance_counts: BTreeMap::new(),
            purge_rows: origin.purge_rows,
            embargo_rows: origin.embargo_rows,
        });
    }
    let plan = FoldPlanV2 {
        schema: FoldPlanV2::SCHEMA.into(),
        task: SearchTaskV1::Timeseries,
        training_view_sha256: input.training_view_sha256.clone(),
        stable_row_ids: input
            .rows
            .iter()
            .map(|row| row.stable_row_id.clone())
            .collect(),
        row_group_sha256s: BTreeMap::new(),
        row_time_keys: input
            .rows
            .iter()
            .map(|row| (row.stable_row_id.clone(), row.time_key))
            .collect(),
        folds,
    };
    plan.validate()?;
    Ok(plan)
}
