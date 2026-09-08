use std::collections::{BTreeMap, BTreeSet};

use super::contracts::{
    raw_sha256, Cv7PlanV1, GpTaskV1, RotatingSplitPlanV1, RowSplitV1, SplitRowV1,
};
use crate::search_v3::validate_sha256;

#[derive(Debug)]
struct AtomicGroup {
    id: String,
    row_indices: Vec<usize>,
    balance: BTreeMap<String, u64>,
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

fn prepare_groups(task: GpTaskV1, rows: &[SplitRowV1]) -> Result<Vec<AtomicGroup>, String> {
    if rows.len() < 7 {
        return Err("split construction requires at least seven rows".into());
    }
    let mut row_ids = BTreeSet::new();
    for row in rows {
        row.validate(task)?;
        if !row_ids.insert(row.stable_row_id.as_str()) {
            return Err("split rows require unique stable row IDs".into());
        }
    }
    let mut strata = vec![String::new(); rows.len()];
    match task {
        GpTaskV1::Regression => {
            let mut order: Vec<_> = (0..rows.len()).collect();
            order.sort_by(|left, right| {
                rows[*left]
                    .regression_target
                    .unwrap()
                    .total_cmp(&rows[*right].regression_target.unwrap())
                    .then_with(|| rows[*left].stable_row_id.cmp(&rows[*right].stable_row_id))
            });
            let len = order.len();
            for (rank, index) in order.into_iter().enumerate() {
                strata[index] = format!("target_decile_{}", (rank * 10 / len).min(9));
            }
        }
        GpTaskV1::Binary => {
            for (index, row) in rows.iter().enumerate() {
                strata[index] = format!("class_{}", row.binary_target.unwrap());
            }
        }
    }
    let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        grouped.entry(&row.atomic_group).or_default().push(index);
    }
    let mut result = Vec::with_capacity(grouped.len());
    for (id, mut row_indices) in grouped {
        row_indices.sort_by_key(|index| rows[*index].stable_row_id.as_str());
        let mut balance = BTreeMap::new();
        for index in &row_indices {
            *balance.entry(strata[*index].clone()).or_default() += 1;
        }
        result.push(AtomicGroup {
            id: id.to_string(),
            row_indices,
            balance,
        });
    }
    result.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(result)
}

fn assign_folds(
    task: GpTaskV1,
    groups: &[AtomicGroup],
    folds: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, String> {
    if groups.len() < folds {
        return Err(format!(
            "atomic grouping leaves {} groups for {folds} folds",
            groups.len()
        ));
    }
    if task == GpTaskV1::Binary {
        for label in ["class_0", "class_1"] {
            let support = groups
                .iter()
                .filter(|group| group.balance.contains_key(label))
                .count();
            if support < folds {
                return Err(format!(
                    "{label} occurs in only {support} atomic groups; {folds} are required"
                ));
            }
        }
    }
    let mut order: Vec<_> = (0..groups.len()).collect();
    order.sort_by(|left, right| {
        groups[*right]
            .row_indices
            .len()
            .cmp(&groups[*left].row_indices.len())
            .then_with(|| {
                seed64(&[&seed.to_string(), &groups[*left].id])
                    .cmp(&seed64(&[&seed.to_string(), &groups[*right].id]))
            })
    });
    let mut assigned = vec![Vec::<usize>::new(); folds];
    let mut sizes = vec![0usize; folds];
    let mut balance = vec![BTreeMap::<String, u64>::new(); folds];
    for group_index in order {
        let group = &groups[group_index];
        let fold = (0..folds)
            .min_by_key(|fold| {
                let missing_gain = group
                    .balance
                    .keys()
                    .filter(|label| !balance[*fold].contains_key(*label))
                    .count();
                let balance_load: u64 = group
                    .balance
                    .keys()
                    .map(|label| balance[*fold].get(label).copied().unwrap_or(0))
                    .sum();
                (
                    usize::MAX - missing_gain,
                    balance_load,
                    sizes[*fold],
                    seed64(&[
                        &seed.to_string(),
                        &groups[group_index].id,
                        &fold.to_string(),
                    ]),
                )
            })
            .unwrap();
        assigned[fold].push(group_index);
        sizes[fold] += group.row_indices.len();
        for (label, count) in &group.balance {
            *balance[fold].entry(label.clone()).or_default() += count;
        }
    }
    if assigned.iter().any(Vec::is_empty) {
        return Err("group assignment produced an empty fold".into());
    }
    if task == GpTaskV1::Binary
        && balance.iter().any(|fold| {
            fold.get("class_0").copied().unwrap_or(0) == 0
                || fold.get("class_1").copied().unwrap_or(0) == 0
        })
    {
        return Err("group assignment could not preserve binary class support".into());
    }
    Ok(assigned)
}

fn make_split(
    rows: &[SplitRowV1],
    groups: &[AtomicGroup],
    validation_groups: &[usize],
    split_index: usize,
    seed: u64,
) -> RowSplitV1 {
    let selected: BTreeSet<_> = validation_groups.iter().copied().collect();
    let mut validation_rows = BTreeSet::new();
    let mut group_ids = Vec::new();
    let mut balance = BTreeMap::new();
    for group_index in validation_groups {
        let group = &groups[*group_index];
        group_ids.push(group.id.clone());
        validation_rows.extend(group.row_indices.iter().copied());
        for (label, count) in &group.balance {
            *balance.entry(label.clone()).or_default() += count;
        }
    }
    group_ids.sort();
    let _ = selected;
    RowSplitV1 {
        split_index: split_index as u32,
        seed,
        train_row_ids: rows
            .iter()
            .enumerate()
            .filter(|(index, _)| !validation_rows.contains(index))
            .map(|(_, row)| row.stable_row_id.clone())
            .collect(),
        validation_row_ids: rows
            .iter()
            .enumerate()
            .filter(|(index, _)| validation_rows.contains(index))
            .map(|(_, row)| row.stable_row_id.clone())
            .collect(),
        validation_groups: group_ids,
        balance_counts: balance,
    }
}

fn plan_base(
    schema: &str,
    task: GpTaskV1,
    dataset_sha256: &str,
    training_sha256: &str,
    rows: &[SplitRowV1],
    splits: Vec<RowSplitV1>,
) -> RotatingSplitPlanV1 {
    RotatingSplitPlanV1 {
        schema: schema.into(),
        task,
        dataset_sha256: dataset_sha256.into(),
        training_sha256: training_sha256.into(),
        stable_row_ids: rows.iter().map(|row| row.stable_row_id.clone()).collect(),
        row_groups: rows
            .iter()
            .map(|row| (row.stable_row_id.clone(), row.atomic_group.clone()))
            .collect(),
        splits,
    }
}

pub fn build_rotating_split_plan(
    task: GpTaskV1,
    dataset_sha256: &str,
    training_sha256: &str,
    split_seed_sha256: &str,
    rows: &[SplitRowV1],
) -> Result<RotatingSplitPlanV1, String> {
    for (name, digest) in [
        ("rotating dataset", dataset_sha256),
        ("rotating training", training_sha256),
        ("rotating seed", split_seed_sha256),
    ] {
        validate_sha256(name, digest)?;
    }
    let groups = prepare_groups(task, rows)?;
    let mut splits = Vec::with_capacity(5);
    let mut used = BTreeSet::new();
    for generation in 0..5usize {
        let mut accepted = None;
        for attempt in 0..256usize {
            let seed = seed64(&[
                split_seed_sha256,
                dataset_sha256,
                "search",
                &generation.to_string(),
                &attempt.to_string(),
            ]);
            let folds = assign_folds(task, &groups, 5, seed)?;
            let chosen = generation % 5;
            let split = make_split(rows, &groups, &folds[chosen], generation, seed);
            if used.insert(split.validation_row_ids.clone()) {
                accepted = Some(split);
                break;
            }
        }
        splits.push(accepted.ok_or_else(|| {
            format!("could not derive a distinct rotating split for generation {generation}")
        })?);
    }
    let plan = plan_base(
        RotatingSplitPlanV1::SCHEMA,
        task,
        dataset_sha256,
        training_sha256,
        rows,
        splits,
    );
    plan.validate()?;
    Ok(plan)
}

pub fn build_cv7_plan(
    task: GpTaskV1,
    dataset_sha256: &str,
    training_sha256: &str,
    cv_seed_sha256: &str,
    rows: &[SplitRowV1],
) -> Result<Cv7PlanV1, String> {
    for (name, digest) in [
        ("CV7 dataset", dataset_sha256),
        ("CV7 training", training_sha256),
        ("CV7 seed", cv_seed_sha256),
    ] {
        validate_sha256(name, digest)?;
    }
    let groups = prepare_groups(task, rows)?;
    let seed = seed64(&[cv_seed_sha256, dataset_sha256, "cv7"]);
    let assigned = assign_folds(task, &groups, 7, seed)?;
    let splits = assigned
        .iter()
        .enumerate()
        .map(|(index, validation)| make_split(rows, &groups, validation, index, seed))
        .collect();
    let plan = Cv7PlanV1(plan_base(
        Cv7PlanV1::SCHEMA,
        task,
        dataset_sha256,
        training_sha256,
        rows,
        splits,
    ));
    plan.validate()?;
    Ok(plan)
}
