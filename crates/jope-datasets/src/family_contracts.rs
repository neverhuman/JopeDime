//! Rust-owned, content-addressed family split and train-only view contracts.
//!
//! These contracts decide which rows and labels belong to training. Their
//! identities are deterministic and their serialized form matches the existing
//! `SplitViewManifest.v1` and `BinaryTargetSummary.v1` wire contracts.

use crate::catalog::Task;
use crate::hash::sha256_hex;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// A deterministic family-disjoint holdout split with exact requested counts.
pub fn freeze_source_family_split(
    dataset_families: &BTreeMap<String, String>,
    train_count: usize,
    holdout_count: usize,
    seed: u64,
) -> Result<(Vec<String>, Vec<String>)> {
    ensure!(
        train_count.checked_add(holdout_count) == Some(dataset_families.len()),
        "split counts must cover every dataset exactly once"
    );

    let mut families: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (dataset, family) in dataset_families {
        ensure!(
            !dataset.is_empty() && !family.is_empty(),
            "dataset and source-family identities must be non-empty"
        );
        families.entry(family).or_default().push(dataset);
    }
    let mut ordered: Vec<_> = families.keys().copied().collect();
    ordered.sort_by_key(|family| sha256_hex(format!("{seed}|{family}").as_bytes()));

    let mut reachable: BTreeMap<usize, Vec<&str>> = BTreeMap::from([(0, Vec::new())]);
    for family in ordered {
        let size = families[family].len();
        let prior: Vec<_> = reachable
            .iter()
            .rev()
            .map(|(total, chosen)| (*total, chosen.clone()))
            .collect();
        for (total, chosen) in prior {
            let Some(candidate) = total.checked_add(size) else {
                continue;
            };
            if candidate <= holdout_count && !reachable.contains_key(&candidate) {
                let mut next = chosen;
                next.push(family);
                reachable.insert(candidate, next);
            }
        }
    }
    let held = reachable
        .get(&holdout_count)
        .ok_or_else(|| {
            anyhow::anyhow!("source families cannot realize the requested exact holdout count")
        })?
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut holdout = families
        .iter()
        .filter(|(family, _)| held.contains(*family))
        .flat_map(|(_, datasets)| datasets.iter().copied())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    holdout.sort();
    let mut train = dataset_families
        .keys()
        .filter(|dataset| holdout.binary_search(*dataset).is_err())
        .cloned()
        .collect::<Vec<_>>();
    train.sort();
    ensure!(
        train.len() == train_count,
        "source-family split count invariant failed"
    );
    Ok((train, holdout))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OuterSplit {
    Train,
    Holdout,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitViewV1 {
    pub view_index: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    pub row_assignment_hash: String,
    pub fold_hash: String,
    pub train_hash: String,
    pub test_hash: String,
    pub encoder_hash: String,
    #[serde(skip)]
    train_row_ids: Vec<String>,
    #[serde(skip)]
    test_row_ids: Vec<String>,
}

impl SplitViewV1 {
    pub fn train_row_ids(&self) -> &[String] {
        &self.train_row_ids
    }

    pub fn test_row_ids(&self) -> &[String] {
        &self.test_row_ids
    }

    fn to_json(&self) -> Value {
        serde_json::json!({
            "view_index": self.view_index,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "row_assignment_hash": self.row_assignment_hash,
            "fold_hash": self.fold_hash,
            "train_hash": self.train_hash,
            "test_hash": self.test_hash,
            "encoder_hash": self.encoder_hash,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitViewManifestV1 {
    pub base_dataset: String,
    pub source_family: String,
    pub task: Task,
    pub outer_split: OuterSplit,
    pub outer_split_hash: String,
    pub fold_plan_hash: String,
    pub views: Vec<SplitViewV1>,
    pub schema: String,
}

impl SplitViewManifestV1 {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "schema": self.schema,
            "base_dataset": self.base_dataset,
            "source_family": self.source_family,
            "task": self.task,
            "outer_split": self.outer_split,
            "outer_split_hash": self.outer_split_hash,
            "fold_plan_hash": self.fold_plan_hash,
            "views": self.views.iter().map(SplitViewV1::to_json).collect::<Vec<_>>(),
        })
    }

    pub fn sha256(&self) -> Result<String> {
        canonical_hash(&self.to_json())
    }
}

/// Build five complementary, balanced views and train-only encoder bindings.
pub fn build_split_view_manifest(
    dataset: &str,
    source_family: &str,
    task: Task,
    outer_split: OuterSplit,
    row_ids: &[String],
    train_sha256: &str,
    test_sha256: Option<&str>,
) -> Result<SplitViewManifestV1> {
    ensure!(
        !dataset.is_empty() && !source_family.is_empty() && row_ids.len() >= 5,
        "dataset/family and at least five row identities are required"
    );
    let unique = row_ids.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == row_ids.len() && row_ids.iter().all(|id| !id.is_empty()),
        "row identities must be non-empty and unique"
    );
    ensure!(
        is_sha256(train_sha256),
        "train_sha256 must be a lowercase SHA-256 hex digest"
    );
    if let Some(test_sha256) = test_sha256 {
        ensure!(
            is_sha256(test_sha256),
            "test_sha256 must be a lowercase SHA-256 hex digest"
        );
    }

    let row_hashes = row_ids
        .iter()
        .map(|row| {
            (
                row.clone(),
                sha256_hex(format!("{dataset}|{row}").as_bytes()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut ordered = row_ids.to_vec();
    ordered.sort_by_key(|row| row_hashes[row].clone());
    let mut buckets = vec![Vec::new(); 5];
    for (index, row) in ordered.iter().enumerate() {
        buckets[index % 5].push(row.clone());
    }

    let mut views = Vec::with_capacity(5);
    for (view_index, test_row_ids) in buckets.into_iter().enumerate() {
        let test_set = test_row_ids.iter().collect::<BTreeSet<_>>();
        let train_row_ids = ordered
            .iter()
            .filter(|row| !test_set.contains(*row))
            .cloned()
            .collect::<Vec<_>>();
        let train_row_hashes = train_row_ids
            .iter()
            .map(|row| row_hashes[row].clone())
            .collect::<Vec<_>>();
        let test_row_hashes = test_row_ids
            .iter()
            .map(|row| row_hashes[row].clone())
            .collect::<Vec<_>>();
        let train_hash = row_set_hash(&train_row_hashes)?;
        let test_hash = row_set_hash(&test_row_hashes)?;

        let mut assignment = Map::new();
        for row in &ordered {
            assignment.insert(
                row_hashes[row].clone(),
                Value::String(
                    if test_set.contains(row) {
                        "test"
                    } else {
                        "train"
                    }
                    .to_owned(),
                ),
            );
        }
        let row_assignment_hash = canonical_hash(&serde_json::json!({
            "schema": "SplitViewRowAssignment.v1",
            "view_index": view_index,
            "assignment": assignment,
        }))?;
        let fold_hash = canonical_hash(&serde_json::json!({
            "schema": "SplitViewFold.v1",
            "view_index": view_index,
            "row_assignment_hash": row_assignment_hash,
        }))?;
        let mut encoder_kinds = vec!["dataset", "target"];
        if task == Task::Binary {
            encoder_kinds.push("class_stats");
        }
        let encoder_hash = canonical_hash(&serde_json::json!({
            "schema": "TrainOnlyEncoderFits.v1",
            "task": task,
            "train_hash": train_hash,
            "encoders": encoder_kinds,
        }))?;
        views.push(SplitViewV1 {
            view_index,
            train_rows: train_row_ids.len(),
            test_rows: test_row_ids.len(),
            row_assignment_hash,
            fold_hash,
            train_hash,
            test_hash,
            encoder_hash,
            train_row_ids,
            test_row_ids,
        });
    }

    let outer_split_hash = canonical_hash(&serde_json::json!({
        "schema": "OuterSplitBinding.v1",
        "base_dataset": dataset,
        "source_family": source_family,
        "task": task,
        "outer_split": outer_split,
        "train_sha256": train_sha256,
        "test_sha256": test_sha256,
    }))?;
    let fold_plan_hash = canonical_hash(&serde_json::json!({
        "schema": "FiveComplementaryViews.v1",
        "outer_split_hash": outer_split_hash,
        "views": views.iter().map(SplitViewV1::to_json).collect::<Vec<_>>(),
    }))?;
    Ok(SplitViewManifestV1 {
        base_dataset: dataset.to_owned(),
        source_family: source_family.to_owned(),
        task,
        outer_split,
        outer_split_hash,
        fold_plan_hash,
        views,
        schema: "SplitViewManifest.v1".to_owned(),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryTargetSummaryV1 {
    pub split_view_manifest_hash: String,
    pub view_index: usize,
    pub training_rows_hash: String,
    pub training_row_count: usize,
    pub positive_row_count: usize,
    pub values: Vec<f64>,
    pub task: Task,
    pub schema: String,
}

impl BinaryTargetSummaryV1 {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "schema": self.schema,
            "task": self.task,
            "split_view_manifest_hash": self.split_view_manifest_hash,
            "view_index": self.view_index,
            "training_rows_hash": self.training_rows_hash,
            "training_row_count": self.training_row_count,
            "positive_row_count": self.positive_row_count,
            "values": self.values,
        })
    }
}

/// Summarize one selected view's training targets and bind their exact row hash.
pub fn binary_target_summary(
    training_targets: &[f64],
    training_row_ids: &[String],
    split_view_manifest: &SplitViewManifestV1,
    view_index: usize,
) -> Result<BinaryTargetSummaryV1> {
    ensure!(
        split_view_manifest.task == Task::Binary,
        "binary target summary requires a binary split view manifest"
    );
    ensure!(
        training_row_ids.len() == training_targets.len() && !training_targets.is_empty(),
        "binary targets and row identities must be non-empty and aligned"
    );
    ensure!(
        training_row_ids.iter().all(|id| !id.is_empty())
            && training_row_ids.iter().collect::<BTreeSet<_>>().len() == training_row_ids.len(),
        "training row identities must be non-empty and unique"
    );
    ensure!(
        training_targets
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0),
        "binary targets must be a strict 0/1 sequence"
    );
    let view = split_view_manifest
        .views
        .get(view_index)
        .ok_or_else(|| anyhow::anyhow!("binary target summary view_index must be in 0..5"))?;
    ensure!(
        training_row_ids == view.train_row_ids(),
        "binary target rows do not match the selected split view"
    );
    let actual_train_hash = row_set_hash(
        &training_row_ids
            .iter()
            .map(|row| sha256_hex(format!("{}|{row}", split_view_manifest.base_dataset).as_bytes()))
            .collect::<Vec<_>>(),
    )?;
    ensure!(
        actual_train_hash == view.train_hash,
        "binary target rows do not match the selected training hash"
    );

    let n = training_targets.len();
    ensure!(
        n >= 2,
        "binary target summary requires at least two training rows"
    );
    let positive = training_targets
        .iter()
        .filter(|value| **value == 1.0)
        .count();
    ensure!(
        positive > 0 && positive < n,
        "binary target summary requires both classes"
    );
    let rate = positive as f64 / n as f64;
    let defined = [
        n as f64 / (n as f64 + 1.0),
        rate,
        rate.min(1.0 - rate),
        -(rate * rate.ln() + (1.0 - rate) * (1.0 - rate).ln()),
        (rate / (1.0 - rate)).ln().clamp(-12.0, 12.0),
        0.5 / rate,
        0.5 / (1.0 - rate),
    ];
    let values = defined
        .into_iter()
        .map(|value| (value as f32) as f64)
        .chain(std::iter::repeat_n(0.0, 512 - 7))
        .collect::<Vec<_>>();
    Ok(BinaryTargetSummaryV1 {
        split_view_manifest_hash: split_view_manifest.sha256()?,
        view_index,
        training_rows_hash: view.train_hash.clone(),
        training_row_count: n,
        positive_row_count: positive,
        values,
        task: Task::Binary,
        schema: "BinaryTargetSummary.v1".to_owned(),
    })
}

/// Build all five train-only summaries for one binary base dataset.
pub fn build_binary_target_summaries(
    training_targets: &[f64],
    row_ids: &[String],
    split_view_manifest: &SplitViewManifestV1,
) -> Result<Vec<BinaryTargetSummaryV1>> {
    ensure!(
        training_targets.len() == row_ids.len()
            && row_ids.iter().collect::<BTreeSet<_>>().len() == row_ids.len(),
        "binary targets and unique row identities must be aligned"
    );
    let target_by_row = row_ids
        .iter()
        .cloned()
        .zip(training_targets.iter().copied())
        .collect::<BTreeMap<_, _>>();
    split_view_manifest
        .views
        .iter()
        .map(|view| {
            let targets =
                view.train_row_ids()
                    .iter()
                    .map(|row| {
                        target_by_row.get(row).copied().ok_or_else(|| {
                            anyhow::anyhow!("training row is absent from target table")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
            binary_target_summary(
                &targets,
                view.train_row_ids(),
                split_view_manifest,
                view.view_index,
            )
        })
        .collect()
}

fn row_set_hash(row_hashes: &[String]) -> Result<String> {
    let mut sorted = row_hashes.to_vec();
    sorted.sort();
    canonical_hash(&serde_json::json!(sorted))
}

fn canonical_hash(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    write_canonical(value, &mut bytes)?;
    Ok(sha256_hex(&bytes))
}

fn write_canonical(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => {
            output.extend_from_slice(serde_json::to_string(string)?.as_bytes())
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(map) => {
            output.push(b'{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                write_canonical(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<String> {
        (0..10).map(|index| format!("row-{index}")).collect()
    }

    fn manifest() -> SplitViewManifestV1 {
        build_split_view_manifest(
            "dataset-1",
            "family-1",
            Task::Binary,
            OuterSplit::Train,
            &rows(),
            &"a".repeat(64),
            Some(&"b".repeat(64)),
        )
        .expect("valid fixture manifest")
    }

    #[test]
    fn family_split_matches_the_deterministic_python_contract() {
        let families = BTreeMap::from([
            ("d0".to_owned(), "f0".to_owned()),
            ("d1".to_owned(), "f1".to_owned()),
            ("d2".to_owned(), "f1".to_owned()),
            ("d3".to_owned(), "f2".to_owned()),
            ("d4".to_owned(), "f3".to_owned()),
            ("d5".to_owned(), "f3".to_owned()),
            ("d6".to_owned(), "f4".to_owned()),
            ("d7".to_owned(), "f5".to_owned()),
        ]);
        let (train, holdout) = freeze_source_family_split(&families, 4, 4, 17).unwrap();
        assert_eq!(train, ["d3", "d4", "d5", "d6"]);
        assert_eq!(holdout, ["d0", "d1", "d2", "d7"]);
        for family in ["f0", "f1", "f2", "f3", "f4", "f5"] {
            let in_train = families
                .iter()
                .any(|(dataset, candidate)| candidate == family && train.contains(dataset));
            let in_holdout = families
                .iter()
                .any(|(dataset, candidate)| candidate == family && holdout.contains(dataset));
            assert!(
                !(in_train && in_holdout),
                "family {family} crossed the split"
            );
        }
    }

    #[test]
    fn split_view_identities_match_the_python_wire_contract() {
        let manifest = manifest();
        assert_eq!(
            manifest.sha256().unwrap(),
            "36f098be3131051e0ed2ccb7a16512a88417a15c8eb034377af1d68fd1c5be81"
        );
        assert_eq!(
            manifest.outer_split_hash,
            "8d47d7bbd1309c9296de4bc6cfe293e91da8885d9a4c963de3193ae69f2a5ad1"
        );
        assert_eq!(
            manifest.fold_plan_hash,
            "4d5cdc31ca43c6343ee6e73f7d15862f46de7c476ddb4f960863d1ee50547ef2"
        );
        assert_eq!(
            manifest.views[0].row_assignment_hash,
            "29a70d6de809a5109fbabe5d45761c52bfbf2e1aa0ef85dea514cab936a44939"
        );
        assert_eq!(manifest.views[0].train_row_ids().len(), 8);
        assert_eq!(manifest.views[0].test_row_ids().len(), 2);
    }

    #[test]
    fn binary_summaries_are_bound_to_exact_train_rows_and_python_f32_values() {
        let manifest = manifest();
        let summaries = build_binary_target_summaries(
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            &rows(),
            &manifest,
        )
        .unwrap();
        assert_eq!(
            summaries
                .iter()
                .map(|summary| summary.positive_row_count)
                .collect::<Vec<_>>(),
            [5, 4, 4, 3, 4]
        );
        assert_eq!(
            summaries[0].split_view_manifest_hash,
            "36f098be3131051e0ed2ccb7a16512a88417a15c8eb034377af1d68fd1c5be81"
        );
        assert_eq!(summaries[0].values[0], 0.8888888955116272);
        assert_eq!(
            summaries[0].values[1..7],
            [
                0.625,
                0.375,
                0.6615632176399231,
                0.5108256340026855,
                0.800000011920929,
                1.3333333730697632
            ]
        );
        assert!(summaries[0].values[7..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn malformed_or_non_train_only_inputs_are_rejected() {
        assert!(build_split_view_manifest(
            "",
            "f",
            Task::Binary,
            OuterSplit::Train,
            &rows(),
            &"a".repeat(64),
            None
        )
        .is_err());
        assert!(build_split_view_manifest(
            "d",
            "f",
            Task::Binary,
            OuterSplit::Train,
            &rows(),
            "bad",
            None
        )
        .is_err());
        let manifest = manifest();
        assert!(
            binary_target_summary(&[0.0; 8], manifest.views[0].test_row_ids(), &manifest, 0)
                .is_err()
        );
        assert!(binary_target_summary(
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            manifest.views[0].train_row_ids(),
            &manifest,
            0
        )
        .is_err());
    }
}
