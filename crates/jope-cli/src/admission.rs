use crate::{
    generator,
    schema::{
        digest, serialize, Catalog, Dataset, Kind, Provenance, Split, Task, MAX_CORPUS_BYTES,
        MAX_FILE_BYTES, ROWS, SCHEMA,
    },
    storage::{components, Directory},
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Debug, Serialize)]
/// Summary of a verified synthetic input catalog.
pub struct Verification {
    pub schema: &'static str,
    pub ok: bool,
    pub offline: bool,
    pub dataset_count: usize,
    pub task_counts: BTreeMap<String, usize>,
    pub bytes: u64,
    pub catalog_sha256: String,
    pub generator: String,
}

/// Verify exact generator identity, then parse and validate the same bounded bytes that were hashed.
pub fn verify_inputs(root: &Path) -> Result<Verification> {
    verified_datasets(root).map(|(report, _)| report)
}

/// The admitted snapshot: every dataset parsed from the same bounded bytes that were hashed.
/// Consumers use it so that search and training never reopen a path after admission.
pub(crate) fn verified_datasets(root: &Path) -> Result<(Verification, Vec<Dataset>)> {
    let directory = Directory::open(root)?;
    let manifest_bytes = directory.read("catalog.json", MAX_FILE_BYTES)?;
    let manifest: Catalog =
        serde_json::from_slice(&manifest_bytes).context("invalid fixture catalog")?;
    validate_catalog(&manifest)?;
    let (expected, _) = generator::generated()?;
    // Checksums in an editable neighboring manifest are not a trust root. The reviewed generator
    // pins the complete expected catalog, including metadata, paths, split-bearing file digests.
    ensure!(
        manifest == expected,
        "catalog does not match the pinned deterministic generator"
    );
    ensure!(
        manifest_bytes == serialize(&expected)?,
        "catalog serialization differs from deterministic output"
    );
    let mut expected_names: BTreeSet<_> = Task::ALL
        .iter()
        .map(|task| task.name().to_owned())
        .collect();
    expected_names.insert("catalog.json".into());
    ensure!(
        directory.names()? == expected_names,
        "unexpected or missing file at fixture root"
    );
    let mut task_counts = BTreeMap::new();
    let mut total_bytes = manifest_bytes.len() as u64;
    for task in Task::ALL {
        let names = task
            .scenarios()
            .iter()
            .map(|scenario| format!("{scenario}.json"))
            .collect();
        ensure!(
            directory.child_names(task.name())? == names,
            "unexpected or missing file in {}",
            task.name()
        );
        task_counts.insert(task.name().into(), 0);
    }
    let mut datasets = Vec::with_capacity(manifest.datasets.len());
    for entry in &manifest.datasets {
        let bytes = directory.read(&entry.path, MAX_FILE_BYTES)?;
        ensure!(
            bytes.len() as u64 == entry.bytes,
            "size mismatch: {}",
            entry.id
        );
        ensure!(
            digest(&bytes) == entry.sha256,
            "digest mismatch: {}",
            entry.id
        );
        let data: Dataset = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid dataset: {}", entry.id))?;
        ensure!(
            data.id == entry.id
                && data.task == entry.task
                && data.scenario == entry.scenario
                && data.seed == entry.seed,
            "dataset/catalog identity mismatch: {}",
            entry.id
        );
        validate_dataset(&data).with_context(|| format!("dataset {}", entry.id))?;
        datasets.push(data);
        total_bytes += bytes.len() as u64;
        *task_counts.get_mut(entry.task.name()).expect("known task") += 1;
    }
    ensure!(
        total_bytes < MAX_CORPUS_BYTES,
        "fixture corpus exceeds 10 MiB"
    );
    Ok((
        Verification {
            schema: "jope.input-verification.v1",
            ok: true,
            offline: true,
            dataset_count: manifest.datasets.len(),
            task_counts,
            bytes: total_bytes,
            catalog_sha256: digest(&manifest_bytes),
            generator: manifest.provenance.generator,
        },
        datasets,
    ))
}

fn validate_catalog(catalog: &Catalog) -> Result<()> {
    ensure!(catalog.schema == SCHEMA, "unsupported catalog schema");
    ensure!(
        catalog.provenance == Provenance::default(),
        "untrusted synthetic provenance"
    );
    ensure!(catalog.datasets.len() == 40, "expected exactly 40 datasets");
    let mut ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for entry in &catalog.datasets {
        ensure!(
            ids.insert(&entry.id) && paths.insert(&entry.path),
            "duplicate dataset identity or path"
        );
        ensure!(
            components(&entry.path)?.len() == 2,
            "fixture path must have family and filename"
        );
        ensure!(
            entry.rows == ROWS && entry.bytes <= MAX_FILE_BYTES,
            "invalid fixture bounds"
        );
        ensure!(
            entry.sha256.len() == 64
                && entry
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "invalid SHA-256 fingerprint"
        );
        ensure!(
            entry.task.scenarios().contains(&entry.scenario.as_str()),
            "unknown task scenario"
        );
    }
    for task in Task::ALL {
        ensure!(
            catalog
                .datasets
                .iter()
                .filter(|entry| entry.task == task)
                .count()
                == 10,
            "expected ten {} datasets",
            task.name()
        );
    }
    Ok(())
}

pub(crate) fn validate_dataset(data: &Dataset) -> Result<()> {
    ensure!(
        data.schema == SCHEMA && data.task.scenarios().contains(&data.scenario.as_str()),
        "unsupported dataset schema/scenario"
    );
    ensure!(data.rows.len() == ROWS, "unexpected row count");
    ensure!(
        !data.columns.is_empty() && data.columns.len() <= 12,
        "unsupported feature width"
    );
    let mut names = BTreeSet::new();
    for column in &data.columns {
        ensure!(
            !column.name.is_empty() && names.insert(&column.name),
            "empty or duplicate column name"
        );
        match column.kind {
            Kind::Integer => ensure!(
                column.categories.is_empty(),
                "integer column declares categories"
            ),
            Kind::Categorical => ensure!(
                !column.categories.is_empty()
                    && column.categories.iter().collect::<BTreeSet<_>>().len()
                        == column.categories.len()
                    && column
                        .categories
                        .iter()
                        .all(|category| !category.is_empty()),
                "invalid category vocabulary"
            ),
        }
    }
    let vocabulary: BTreeSet<_> = data.classes.iter().copied().collect();
    match data.task {
        Task::Binary => ensure!(data.classes == [0, 1], "binary classes must be [0, 1]"),
        Task::Multiclass => ensure!(
            (3..=7).contains(&data.classes.len())
                && vocabulary.len() == data.classes.len()
                && data
                    .classes
                    .iter()
                    .enumerate()
                    .all(|(index, value)| *value == index as i64),
            "invalid multiclass vocabulary"
        ),
        _ => ensure!(
            data.classes.is_empty(),
            "numerical task must not declare classes"
        ),
    }
    let mut ids = BTreeSet::new();
    let mut groups = BTreeMap::new();
    let mut partitions: BTreeMap<Split, usize> = BTreeMap::new();
    let mut classes = BTreeSet::new();
    let mut chronological: BTreeMap<&str, (i64, Split)> = BTreeMap::new();
    let mut time_bounds: BTreeMap<Split, (i64, i64)> = BTreeMap::new();
    let mut series_partitions: BTreeMap<&str, BTreeSet<Split>> = BTreeMap::new();
    // Group rules apply when the scenario declares grouping or any row carries a group.
    let grouped =
        data.scenario == "grouped_observations" || data.rows.iter().any(|row| row.group.is_some());
    for row in &data.rows {
        ensure!(
            row.id < ROWS && ids.insert(row.id),
            "duplicate or invalid row id"
        );
        ensure!(
            row.features.len() == data.columns.len(),
            "feature width differs from schema"
        );
        for (feature, column) in row.features.iter().zip(&data.columns) {
            if feature.is_null() {
                ensure!(column.nullable, "null in nonnullable feature");
            } else {
                match column.kind {
                    Kind::Integer => ensure!(feature.is_i64(), "expected integer feature"),
                    Kind::Categorical => ensure!(
                        feature.as_str().is_some_and(|value| column
                            .categories
                            .iter()
                            .any(|category| category == value)),
                        "unknown category or invalid feature type"
                    ),
                }
            }
        }
        if !vocabulary.is_empty() {
            ensure!(
                vocabulary.contains(&row.target),
                "target outside class vocabulary"
            );
            classes.insert(row.target);
        }
        *partitions.entry(row.split).or_default() += 1;
        ensure!(
            row.group.is_some() == grouped,
            "unexpected or missing group identity"
        );
        if let Some(group) = &row.group {
            ensure!(!group.is_empty(), "empty group identity");
            if let Some(previous) = groups.insert(group, row.split) {
                ensure!(previous == row.split, "group leaks across partitions");
            }
        }
        if data.task == Task::TimeSeries {
            let timestamp = row.timestamp.context("missing time-series timestamp")?;
            let series = row.series.as_deref().context("missing series identity")?;
            ensure!(!series.is_empty(), "empty series identity");
            if let Some((previous_time, previous_split)) =
                chronological.insert(series, (timestamp, row.split))
            {
                ensure!(
                    timestamp > previous_time && row.split >= previous_split,
                    "series chronology or partition order violated"
                );
            }
            series_partitions
                .entry(series)
                .or_default()
                .insert(row.split);
            let bounds = time_bounds
                .entry(row.split)
                .or_insert((timestamp, timestamp));
            bounds.0 = bounds.0.min(timestamp);
            bounds.1 = bounds.1.max(timestamp);
        } else {
            ensure!(
                row.timestamp.is_none() && row.series.is_none(),
                "temporal metadata in tabular task"
            );
        }
    }
    ensure!(
        partitions.len() == 3,
        "missing train/validation/test partition"
    );
    ensure!(classes == vocabulary, "a declared class is absent");
    if data.task == Task::TimeSeries {
        ensure!(
            series_partitions.values().all(|splits| splits.len() == 3),
            "series missing a partition"
        );
        let bounds: Vec<_> = time_bounds.values().collect();
        ensure!(
            bounds.windows(2).all(|pair| pair[0].1 < pair[1].0),
            "chronological partitions overlap"
        );
    }
    Ok(())
}
