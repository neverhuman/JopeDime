//! Authenticated development inputs. No code in this module opens split or sealed files.
use crate::experiment::{sha256, Column, ColumnKind, Metric, PanelDataset};
use crate::sha256_hex;
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

#[cfg(test)]
use jope_runtime::table::CATEGORY_PREFIX;
/// The raw-cell encoding and fitting-input digest moved to the appliance runtime unchanged;
/// fitted envelopes retain the policy and replay `raw_table` at inference.
pub use jope_runtime::table::{fitting_input_sha256, FITTING_INPUT, RAW_CELL_ENCODING};
const MAX_DEVELOPMENT_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedRow {
    pub id: String,
    pub source_index: usize,
    pub group: String,
    pub fold: u8,
    pub values: Vec<Option<String>>,
    pub target: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTable {
    schema: String,
    dataset: String,
    task: Task,
    source_sha256: String,
    partition: String,
    features: Vec<Column>,
    classes: Vec<String>,
    rows: Vec<PreparedRow>,
}

#[derive(Clone, Debug)]
pub struct DevelopmentData {
    pub dataset: PanelDataset,
    /// Original row order, labels, nulls and group identities, unchanged.
    pub rows: Vec<PreparedRow>,
    pub schema: DataSchema,
    /// W1-T01: the committed ten-regression release's admission of this
    /// partition, present exactly when the release names the dataset. `None`
    /// is a dataset outside the release (another task, or a fixture).
    pub release: Option<ReleaseAdmission>,
}

/// The committed ten-regression release's admission of one development
/// partition (W1-T01): the member identities its receivers bind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReleaseAdmission {
    pub release_id: String,
    /// The member's acquired raw object (`raw.sha256`).
    pub dataset_sha256: String,
    /// The member's development partition bytes (`prepared.development_sha256`).
    pub development_sha256: String,
    /// The member's development row/target alignment digest.
    pub development_alignment_sha256: String,
}

/// W1-T01: a dataset the committed ten-regression release names is admitted
/// only as that member. Its panel entry must be the member's (task, raw
/// source, split, development and sealed digests, row and fold counts, group
/// columns and feature schema), and its development bytes must reproduce the
/// member's exact digest and row/target alignment. A dataset the release does
/// not name carries no release admission.
fn admit_release_member(bytes: &[u8], dataset: &PanelDataset) -> Result<Option<ReleaseAdmission>> {
    use jope_datasets::regression_release::{committed, FeatureKind};
    let release = committed()?;
    let Some(member) = release.members.iter().find(|m| m.dataset == dataset.id) else {
        return Ok(None);
    };
    let features = member.schema.features.len() == dataset.features.len()
        && member
            .schema
            .features
            .iter()
            .zip(&dataset.features)
            .all(|(m, d)| {
                m.name == d.name
                    && matches!(
                        (m.kind, d.kind),
                        (FeatureKind::Numeric, ColumnKind::Numeric)
                            | (FeatureKind::Categorical, ColumnKind::Categorical)
                    )
            });
    ensure!(
        dataset.task == Task::Regression
            && member.raw.sha256 == dataset.source.sha256
            && member.prepared.split_sha256 == dataset.split.sha256
            && member.prepared.development_sha256 == dataset.split.development_sha256
            && member.prepared.sealed_sha256 == dataset.split.sealed_sha256
            && member.rows.total == dataset.rows as u64
            && member.rows.development == dataset.split.development_rows as u64
            && member.rows.sealed == dataset.split.sealed_rows as u64
            && member
                .folds
                .fold_rows
                .iter()
                .map(|&rows| rows as usize)
                .eq(dataset.split.fold_rows.iter().copied())
            && member.groups.columns == dataset.group_columns
            && features,
        "{}: panel entry differs from its member of regression release {}",
        dataset.id,
        release.release_id
    );
    release.verify_development(&dataset.id, bytes)?;
    Ok(Some(ReleaseAdmission {
        release_id: release.release_id.clone(),
        dataset_sha256: member.raw.sha256.clone(),
        development_sha256: member.prepared.development_sha256.clone(),
        development_alignment_sha256: member.rows.development_alignment_sha256.clone(),
    }))
}

/// Parent-side partition. Worker requests must select only fitting targets and
/// prediction features; serializing a holdout partition would expose its labels.
#[derive(Clone, Debug)]
pub struct DataPartition {
    pub table: RawTable,
    pub targets: Vec<f64>,
    pub labels: Vec<String>,
    pub rows: Vec<PreparedRow>,
    pub rows_sha256: String,
}

#[derive(Clone, Debug)]
pub struct DevelopmentCut {
    pub fold: u8,
    pub train: DataPartition,
    pub hold: DataPartition,
}

fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

/// Open exactly `development/<dataset>.json`, never the full preparation tree.
/// Reject symlink entries rather than following an alias into a sealed directory.
pub fn load_development(root: &Path, dataset: &PanelDataset) -> Result<DevelopmentData> {
    ensure!(safe_id(&dataset.id), "unsafe dataset identifier");
    let directory = root.join("development");
    let metadata = std::fs::symlink_metadata(&directory)
        .with_context(|| format!("inspect {}", directory.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "development directory must not be a symlink"
    );
    let path = directory.join(format!("{}.json", dataset.id));
    let metadata =
        std::fs::symlink_metadata(&path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "development input must be a regular non-symlink file"
    );
    ensure!(
        metadata.len() <= MAX_DEVELOPMENT_BYTES,
        "development input exceeds 128 MiB"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .with_context(|| format!("open {}", path.display()))?
        .take(MAX_DEVELOPMENT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_DEVELOPMENT_BYTES,
        "development input exceeds 128 MiB"
    );
    parse_development(&bytes, dataset)
}

/// Check the frozen byte identity before interpreting any row, fold or label.
pub fn parse_development(bytes: &[u8], dataset: &PanelDataset) -> Result<DevelopmentData> {
    ensure!(
        sha256(&dataset.split.development_sha256),
        "invalid development sha256"
    );
    ensure!(
        sha256_hex(bytes) == dataset.split.development_sha256,
        "development sha256 mismatch for {}",
        dataset.id
    );
    ensure!(safe_id(&dataset.id), "unsafe dataset identifier");
    // W1-T01: a release member is admitted only as that member, before any
    // row, fold or label is interpreted.
    let release = admit_release_member(bytes, dataset)?;
    ensure!(sha256(&dataset.source.sha256), "invalid source sha256");
    ensure!(
        Metric::for_task(dataset.task)? == dataset.metric,
        "task/metric mismatch"
    );
    validate_classes(dataset.task, &dataset.classes)?;
    let table: PreparedTable = serde_json::from_slice(bytes).context("parse development input")?;
    ensure!(
        table.schema == "jope.prepared-table/v1" && table.partition == "development",
        "requires prepared development partition"
    );
    ensure!(
        table.dataset == dataset.id
            && table.task == dataset.task
            && table.source_sha256 == dataset.source.sha256,
        "development dataset/task/source mismatch"
    );
    ensure!(
        table.classes == dataset.classes,
        "development class order mismatch"
    );
    ensure!(
        table.features.len() == dataset.features.len()
            && table
                .features
                .iter()
                .zip(&dataset.features)
                .all(|(actual, expected)| actual.name == expected.name
                    && actual.kind == expected.kind),
        "development feature schema mismatch"
    );
    let schema = schema_for(&dataset.features)?;
    ensure!(
        dataset
            .group_columns
            .iter()
            .all(|name| dataset.features.iter().any(|feature| feature.name == *name)),
        "group column absent from feature schema"
    );
    ensure!(
        !table.rows.is_empty() && table.rows.len() == dataset.split.development_rows,
        "development row count mismatch"
    );
    ensure!(
        dataset
            .split
            .development_rows
            .checked_add(dataset.split.sealed_rows)
            == Some(dataset.rows)
            && dataset.split.sealed_rows > 0,
        "partition row count mismatch"
    );
    let mut ids = BTreeSet::new();
    let mut source_indices = BTreeSet::new();
    let mut groups = BTreeMap::new();
    let mut fold_rows = [0usize; 5];
    let mut fold_classes: [BTreeSet<&str>; 5] = std::array::from_fn(|_| BTreeSet::new());
    for row in &table.rows {
        ensure!(
            sha256(&row.id) && ids.insert(&row.id),
            "invalid or duplicate row identity"
        );
        ensure!(
            row.source_index < dataset.rows && source_indices.insert(row.source_index),
            "invalid or duplicate source index"
        );
        ensure!(sha256(&row.group), "invalid group identity");
        ensure!(row.fold < 5, "development fold must be in 0..5");
        if let Some(previous) = groups.insert(&row.group, row.fold) {
            ensure!(previous == row.fold, "group crossed development folds");
        }
        ensure!(
            row.values.len() == dataset.features.len(),
            "ragged development row"
        );
        for (column, value) in dataset.features.iter().zip(&row.values) {
            encode_cell(column.kind, value.as_deref())?;
        }
        encode_target(dataset.task, &dataset.classes, &row.target)?;
        fold_rows[row.fold as usize] += 1;
        fold_classes[row.fold as usize].insert(&row.target);
    }
    ensure!(
        fold_rows == dataset.split.fold_rows && fold_rows.iter().all(|rows| *rows > 0),
        "development fold row count mismatch"
    );
    if dataset.task != Task::Regression {
        let classes: BTreeSet<_> = dataset.classes.iter().map(String::as_str).collect();
        ensure!(
            fold_classes.iter().all(|fold| *fold == classes),
            "a development fold lacks a declared class"
        );
    }
    Ok(DevelopmentData {
        dataset: dataset.clone(),
        rows: table.rows,
        schema,
        release,
    })
}

fn validate_classes(task: Task, classes: &[String]) -> Result<()> {
    let unique: BTreeSet<_> = classes.iter().collect();
    ensure!(
        unique.len() == classes.len() && classes.iter().all(|class| !class.is_empty()),
        "empty or duplicate class label"
    );
    ensure!(
        match task {
            Task::Regression => classes.is_empty(),
            Task::Binary => classes.len() == 2,
            Task::Multiclass => classes.len() >= 3,
            _ => false,
        },
        "class order incompatible with task"
    );
    Ok(())
}

fn schema_for(features: &[Column]) -> Result<DataSchema> {
    let schema = DataSchema {
        columns: features
            .iter()
            .map(|column| match column.kind {
                ColumnKind::Numeric => ColumnSpec::numeric(&column.name),
                ColumnKind::Categorical => ColumnSpec::categorical(&column.name),
            })
            .collect(),
        // External folds carry groups. No time ordering was declared by this panel.
        row_semantics: RowSemantics::Iid,
    };
    schema.validate()?;
    Ok(schema)
}

fn encode_cell(kind: ColumnKind, value: Option<&str>) -> Result<String> {
    jope_runtime::table::encode_cell(engine_kind(kind), value)
}

fn engine_kind(kind: ColumnKind) -> feat_math::advanced_v3::ColumnKind {
    match kind {
        ColumnKind::Numeric => feat_math::advanced_v3::ColumnKind::Numeric,
        ColumnKind::Categorical => feat_math::advanced_v3::ColumnKind::Categorical,
    }
}

/// Reusable label-free preprocessing for fitting, workers and fitted inference.
/// JSON null is the only missing cell; original categorical strings are injective.
pub fn raw_table(features: &[Column], values: &[Vec<Option<String>>]) -> Result<RawTable> {
    jope_runtime::table::raw_table(&schema_for(features)?, values)
}

impl DataPartition {
    /// [`fitting_input_sha256`] of this partition, as a fit worker receives it.
    pub fn fitting_input_sha256(&self) -> Result<String> {
        fitting_input_sha256(&self.table, &self.targets)
    }
}

fn encode_target(task: Task, classes: &[String], label: &str) -> Result<f64> {
    if task == Task::Regression {
        let target: f64 = label.trim().parse().context("invalid regression target")?;
        ensure!(target.is_finite(), "regression target must be finite");
        Ok(target)
    } else {
        classes
            .iter()
            .position(|class| class == label)
            .map(|index| index as f64)
            .with_context(|| format!("target {label:?} absent from declared class order"))
    }
}

impl DevelopmentData {
    /// Use the prepared assignment exactly. No seeded recut or row shuffle occurs.
    pub fn cut(&self, fold: u8) -> Result<DevelopmentCut> {
        ensure!(fold < 5, "development fold must be in 0..5");
        let train = self.partition(
            self.rows
                .iter()
                .filter(|row| row.fold != fold)
                .cloned()
                .collect(),
        )?;
        let hold = self.partition(
            self.rows
                .iter()
                .filter(|row| row.fold == fold)
                .cloned()
                .collect(),
        )?;
        ensure!(
            train.rows_sha256 != hold.rows_sha256,
            "training/scoring rows alias"
        );
        Ok(DevelopmentCut { fold, train, hold })
    }

    fn partition(&self, rows: Vec<PreparedRow>) -> Result<DataPartition> {
        ensure!(!rows.is_empty(), "empty development partition");
        let labels: Vec<_> = rows.iter().map(|row| row.target.clone()).collect();
        let targets = labels
            .iter()
            .map(|label| encode_target(self.dataset.task, &self.dataset.classes, label))
            .collect::<Result<Vec<_>>>()?;
        let ids: Vec<_> = rows.iter().map(|row| &row.id).collect();
        let rows_sha256 = sha256_hex(&serde_json::to_vec(&ids)?);
        let table = raw_table(
            &self.dataset.features,
            &rows
                .iter()
                .map(|row| row.values.clone())
                .collect::<Vec<_>>(),
        )?;
        Ok(DataPartition {
            table,
            targets,
            labels,
            rows,
            rows_sha256,
        })
    }
}

#[cfg(test)]
mod tests;
