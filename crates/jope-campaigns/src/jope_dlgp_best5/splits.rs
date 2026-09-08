use std::collections::{BTreeMap, BTreeSet};

use super::{Cv7PlanV1, DataSplitV1, SearchSplitPlanV1, TargetRowV1, SEARCH_SEEDS};
use crate::jope_generalized_gp::raw_sha256;
use crate::search_v3::validate_sha256;

#[derive(Debug)]
struct Group<'a> {
    id: &'a str,
    row_ids: Vec<&'a str>,
    mean_target: f64,
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

fn prepare_groups(rows: &[TargetRowV1]) -> Result<Vec<Group<'_>>, String> {
    let mut stable_rows = BTreeSet::new();
    let mut grouped = BTreeMap::<&str, Vec<&TargetRowV1>>::new();
    for row in rows {
        row.validate()?;
        if !stable_rows.insert(row.stable_row_id.as_str()) {
            return Err("split input contains a duplicate stable row ID".into());
        }
        grouped.entry(&row.atomic_group).or_default().push(row);
    }
    if grouped.len() < 8 {
        return Err("eight atomic-aware splits require at least eight atomic groups".into());
    }
    let mut groups = grouped
        .into_iter()
        .map(|(id, members)| Group {
            id,
            row_ids: members
                .iter()
                .map(|row| row.stable_row_id.as_str())
                .collect(),
            mean_target: members.iter().map(|row| row.target).sum::<f64>() / members.len() as f64,
        })
        .collect::<Vec<_>>();
    groups.sort_by(|left, right| {
        left.mean_target
            .total_cmp(&right.mean_target)
            .then_with(|| left.id.cmp(right.id))
    });
    Ok(groups)
}

fn ranked_groups<'a>(groups: &'a [Group<'a>], seed: u64, purpose: &str) -> Vec<&'a Group<'a>> {
    let width = groups.len().div_ceil(10);
    let mut ranked = Vec::with_capacity(groups.len());
    for (decile, chunk) in groups.chunks(width).enumerate() {
        let mut values = chunk.iter().collect::<Vec<_>>();
        values.sort_by_key(|group| {
            seed64(&[purpose, &seed.to_string(), &decile.to_string(), group.id])
        });
        ranked.extend(values);
    }
    // Interleave target deciles instead of concatenating them, keeping every
    // partial prefix target-representative.
    let bucket_count = groups.len().div_ceil(width);
    let buckets = (0..bucket_count)
        .map(|bucket| ranked[bucket * width..((bucket + 1) * width).min(ranked.len())].to_vec())
        .collect::<Vec<_>>();
    let mut interleaved = Vec::with_capacity(groups.len());
    for offset in 0..width {
        for bucket in &buckets {
            if let Some(group) = bucket.get(offset) {
                interleaved.push(*group);
            }
        }
    }
    interleaved
}

fn search_split(
    rows: &[TargetRowV1],
    groups: &[Group<'_>],
    dataset_sha256: &str,
    master_seed: u64,
    split_index: usize,
    attempt: usize,
) -> DataSplitV1 {
    let seed = seed64(&[
        "jope-dlgp-best5",
        dataset_sha256,
        &master_seed.to_string(),
        "search",
        &split_index.to_string(),
        &attempt.to_string(),
    ]);
    let order = ranked_groups(groups, seed, "search-target-deciles");
    let target_rows = ((rows.len() as f64) * 0.20).round().max(1.0) as usize;
    let mut validation_groups = BTreeSet::new();
    let mut selected_rows = 0usize;
    for group in order {
        if validation_groups.is_empty()
            || selected_rows < target_rows
            || (selected_rows + group.row_ids.len()).abs_diff(target_rows)
                < selected_rows.abs_diff(target_rows)
        {
            validation_groups.insert(group.id);
            selected_rows += group.row_ids.len();
        }
        if selected_rows >= target_rows {
            break;
        }
    }
    if validation_groups.len() == groups.len() {
        validation_groups.remove(groups.last().unwrap().id);
    }
    let mut train_row_ids = Vec::new();
    let mut validation_row_ids = Vec::new();
    for row in rows {
        if validation_groups.contains(row.atomic_group.as_str()) {
            validation_row_ids.push(row.stable_row_id.clone());
        } else {
            train_row_ids.push(row.stable_row_id.clone());
        }
    }
    train_row_ids.sort();
    validation_row_ids.sort();
    DataSplitV1 {
        split_index: split_index as u32,
        split_seed: seed,
        train_row_ids,
        validation_row_ids,
        validation_groups: validation_groups.into_iter().map(str::to_string).collect(),
    }
}

pub fn build_search_split_plan(
    dataset_sha256: &str,
    training_sha256: &str,
    master_seed: u64,
    rows: &[TargetRowV1],
) -> Result<SearchSplitPlanV1, String> {
    validate_sha256("split dataset", dataset_sha256)?;
    validate_sha256("split training", training_sha256)?;
    if !SEARCH_SEEDS.contains(&master_seed) {
        return Err("search split seed must be 101 or 202".into());
    }
    let groups = prepare_groups(rows)?;
    let mut seen = BTreeSet::new();
    let mut splits = Vec::with_capacity(8);
    for split_index in 0..8 {
        let mut accepted = None;
        for attempt in 0..4_096 {
            let candidate = search_split(
                rows,
                &groups,
                dataset_sha256,
                master_seed,
                split_index,
                attempt,
            );
            if seen.insert(candidate.validation_row_ids.clone()) {
                accepted = Some(candidate);
                break;
            }
        }
        splits.push(
            accepted
                .ok_or_else(|| format!("could not derive distinct search split {split_index}"))?,
        );
    }
    let plan = SearchSplitPlanV1 {
        schema: SearchSplitPlanV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        training_sha256: training_sha256.into(),
        master_seed,
        stable_row_ids: rows.iter().map(|row| row.stable_row_id.clone()).collect(),
        row_groups: rows
            .iter()
            .map(|row| (row.stable_row_id.clone(), row.atomic_group.clone()))
            .collect(),
        splits,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn build_cv7_plan(
    dataset_sha256: &str,
    training_sha256: &str,
    master_seed: u64,
    rows: &[TargetRowV1],
) -> Result<Cv7PlanV1, String> {
    validate_sha256("CV7 dataset", dataset_sha256)?;
    validate_sha256("CV7 training", training_sha256)?;
    if !SEARCH_SEEDS.contains(&master_seed) {
        return Err("CV7 seed must derive from search seed 101 or 202".into());
    }
    let groups = prepare_groups(rows)?;
    if groups.len() < 7 {
        return Err("CV7 requires at least seven atomic groups".into());
    }
    let seed = seed64(&[
        "jope-dlgp-best5",
        dataset_sha256,
        &master_seed.to_string(),
        "dataset-only-cv7",
    ]);
    let order = ranked_groups(&groups, seed, "cv7-target-deciles");
    let mut fold_groups = vec![Vec::<&str>::new(); 7];
    let mut fold_rows = [0usize; 7];
    for group in order {
        let fold = (0..7)
            .min_by_key(|index| (fold_rows[*index], *index))
            .unwrap();
        fold_groups[fold].push(group.id);
        fold_rows[fold] += group.row_ids.len();
    }
    let splits = fold_groups
        .iter()
        .enumerate()
        .map(|(index, validation)| {
            let validation = validation.iter().copied().collect::<BTreeSet<_>>();
            let mut train_row_ids = Vec::new();
            let mut validation_row_ids = Vec::new();
            for row in rows {
                if validation.contains(row.atomic_group.as_str()) {
                    validation_row_ids.push(row.stable_row_id.clone());
                } else {
                    train_row_ids.push(row.stable_row_id.clone());
                }
            }
            train_row_ids.sort();
            validation_row_ids.sort();
            DataSplitV1 {
                split_index: index as u32,
                split_seed: seed,
                train_row_ids,
                validation_row_ids,
                validation_groups: validation.into_iter().map(str::to_string).collect(),
            }
        })
        .collect();
    let plan = Cv7PlanV1(SearchSplitPlanV1 {
        schema: Cv7PlanV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        training_sha256: training_sha256.into(),
        master_seed,
        stable_row_ids: rows.iter().map(|row| row.stable_row_id.clone()).collect(),
        row_groups: rows
            .iter()
            .map(|row| (row.stable_row_id.clone(), row.atomic_group.clone()))
            .collect(),
        splits,
    });
    plan.validate()?;
    Ok(plan)
}
