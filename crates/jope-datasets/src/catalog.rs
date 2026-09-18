//! Catalog row and collection types (`jope.dataset-catalog/v1`, `jope.dataset-collection/v1`).
//!
//! The committed JSON Schemas under `schemas/` are the public contract; the Rust types derive
//! `JsonSchema` so a test can assert both agree on required fields and enums.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const CATALOG_SCHEMA_ID: &str = "jope.dataset-catalog/v1";
pub const COLLECTION_SCHEMA_ID: &str = "jope.dataset-collection/v1";

/// The committed catalog schema (JSON Schema 2020-12).
pub const CATALOG_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.dataset-catalog.v1.schema.json");
/// The committed collection schema (JSON Schema 2020-12).
pub const COLLECTION_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.dataset-collection.v1.schema.json");

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Regression,
    Binary,
    Multiclass,
    MultiLabel,
    Timeseries,
}

impl Task {
    pub const ALL: [Task; 5] = [
        Task::Regression,
        Task::Binary,
        Task::Multiclass,
        Task::MultiLabel,
        Task::Timeseries,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Task::Regression => "regression",
            Task::Binary => "binary",
            Task::Multiclass => "multiclass",
            Task::MultiLabel => "multi_label",
            Task::Timeseries => "timeseries",
        }
    }

    pub fn parse(s: &str) -> Option<Task> {
        Task::ALL.into_iter().find(|t| t.as_str() == s)
    }

    pub fn target_type(self) -> TargetType {
        match self {
            Task::Regression | Task::Timeseries => TargetType::Numeric,
            Task::Binary => TargetType::Binary,
            Task::Multiclass => TargetType::Multiclass,
            Task::MultiLabel => TargetType::MultiLabel,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TargetType {
    Numeric,
    Binary,
    Multiclass,
    MultiLabel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LicenseStatus {
    Recorded,
    Unknown,
    Restricted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum IngestSource {
    Quant,
    Mirror,
    Hyper,
    Repo,
    Link,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CollectionTask {
    Regression,
    Binary,
    Multiclass,
    MultiLabel,
    Timeseries,
    Mixed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Shape {
    pub rows: u64,
    pub features: u64,
    pub categorical_features: u64,
    pub columns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Target {
    pub name: String,
    /// Always -1: the target is the last column.
    pub index: i64,
    #[serde(rename = "type")]
    pub target_type: TargetType,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FileEntry {
    /// Blob key.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub columns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_header: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Files {
    pub train: FileEntry,
    pub test: FileEntry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<FileEntry>,
}

impl Files {
    /// `(name, entry)` for every file the row references.
    pub fn entries(&self) -> Vec<(&'static str, &FileEntry)> {
        let mut v = vec![("train.csv", &self.train), ("test.csv", &self.test)];
        if let Some(m) = &self.meta {
            v.push(("meta.json", m));
        }
        v
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Split {
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub train_rows: u64,
    pub test_rows: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct License {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spdx: Option<String>,
    pub status: LicenseStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub text_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl License {
    pub fn unknown(note: Option<String>) -> Self {
        License {
            spdx: None,
            status: LicenseStatus::Unknown,
            evidence_url: None,
            text_sha256: None,
            note,
        }
    }

    /// The licence rule: `recorded` needs an SPDX expression and an evidence URL.
    pub fn rule_errors(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.status == LicenseStatus::Recorded {
            if self.spdx.as_deref().is_none_or(str::is_empty) {
                errs.push("license.status is recorded but spdx is missing".into());
            }
            if self.evidence_url.as_deref().is_none_or(str::is_empty) {
                errs.push("license.status is recorded but evidence_url is missing".into());
            }
        }
        errs
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Provenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieved_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dedupe_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citation: Option<String>,
    pub license: License,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Generator {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub recipe_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub script_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Alias {
    pub path: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Ingest {
    pub source: IngestSource,
    pub relpath: String,
    pub ingested_at: String,
    pub tool_version: String,
}

/// One catalog row (`datasets/<dataset_hash>/manifest.json`, a `catalog.jsonl` line, or a
/// committed `datasets/test/<slug>/metadata.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CatalogRow {
    pub schema: String,
    #[schemars(regex(pattern = r"^[A-Za-z0-9._-]+$"))]
    pub id: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{16}$"))]
    pub dataset_hash: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub canonical_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub task: Task,
    pub origin: Origin,
    pub shape: Shape,
    pub target: Target,
    pub files: Files,
    pub split: Split,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<Generator>,
    pub collections: Vec<String>,
    pub aliases: Vec<Alias>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ingest: Option<Ingest>,
}

impl CatalogRow {
    /// Structural invariants the schema cannot express.
    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.schema != CATALOG_SCHEMA_ID {
            errs.push(format!(
                "schema is {:?}, expected {CATALOG_SCHEMA_ID:?}",
                self.schema
            ));
        }
        if !self.canonical_sha256.starts_with(&self.dataset_hash) {
            errs.push(format!(
                "canonical_sha256[:16] {} != dataset_hash {}",
                &self.canonical_sha256[..self.canonical_sha256.len().min(16)],
                self.dataset_hash
            ));
        }
        if crate::dataset_dir_hash(&self.id) != Some(self.dataset_hash.as_str()) {
            errs.push(format!(
                "id {:?} does not resolve to dataset_hash {}",
                self.id, self.dataset_hash
            ));
        }
        if self.target.index != -1 {
            errs.push("target.index must be -1".into());
        }
        if self.shape.columns != self.shape.features + 1 {
            errs.push(format!(
                "shape.columns {} != shape.features {} + 1",
                self.shape.columns, self.shape.features
            ));
        }
        if self.shape.rows != self.split.train_rows + self.split.test_rows {
            errs.push("shape.rows != split.train_rows + split.test_rows".into());
        }
        for (name, f) in [("train", &self.files.train), ("test", &self.files.test)] {
            if f.has_header != Some(false) {
                errs.push(format!("files.{name}.has_header must be false"));
            }
            if f.columns != Some(self.shape.columns) {
                errs.push(format!(
                    "files.{name}.columns {:?} != shape.columns {}",
                    f.columns, self.shape.columns
                ));
            }
        }
        if self.files.train.rows != Some(self.split.train_rows) {
            errs.push("files.train.rows != split.train_rows".into());
        }
        if self.files.test.rows != Some(self.split.test_rows) {
            errs.push("files.test.rows != split.test_rows".into());
        }
        match (self.origin, &self.generator) {
            (Origin::Synthetic, None) => {
                errs.push("origin is synthetic but generator is missing".into())
            }
            (Origin::Real, Some(_)) => errs.push("origin is real but generator is present".into()),
            _ => {}
        }
        errs.extend(self.provenance.license.rule_errors());
        errs
    }

    /// Schema validation (committed schema) plus the structural invariants.
    pub fn validate_value(value: &serde_json::Value) -> Result<CatalogRow, Vec<String>> {
        let mut errs = crate::schema::validate(&catalog_schema(), value);
        let row: Option<CatalogRow> = match serde_json::from_value(value.clone()) {
            Ok(row) => Some(row),
            Err(e) => {
                errs.push(format!("deserialize: {e}"));
                None
            }
        };
        if let Some(row) = &row {
            errs.extend(row.invariant_errors());
        }
        match (errs.is_empty(), row) {
            (true, Some(row)) => Ok(row),
            _ => Err(errs),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Policy {
    pub license_required: bool,
    pub allowed_spdx: Vec<String>,
}

/// A collection document (`collections/<name>/v<N>.json`, `datasets/manifest.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Collection {
    pub schema: String,
    #[schemars(regex(pattern = r"^[A-Za-z0-9._-]+$"))]
    pub name: String,
    pub version: u64,
    pub visibility: Visibility,
    pub task: CollectionTask,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub members: Vec<String>,
    pub policy: Policy,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub members_sha256: String,
}

impl Collection {
    /// sha256 of the members joined by `\n` with a trailing newline.
    pub fn members_digest(members: &[String]) -> String {
        let mut text = members.join("\n");
        text.push('\n');
        crate::hash::sha256_hex(text.as_bytes())
    }

    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.schema != COLLECTION_SCHEMA_ID {
            errs.push(format!(
                "schema is {:?}, expected {COLLECTION_SCHEMA_ID:?}",
                self.schema
            ));
        }
        let want = Self::members_digest(&self.members);
        if self.members_sha256 != want {
            errs.push(format!(
                "members_sha256 {} != computed {want}",
                self.members_sha256
            ));
        }
        errs
    }

    pub fn validate_value(value: &serde_json::Value) -> Result<Collection, Vec<String>> {
        let mut errs = crate::schema::validate(&collection_schema(), value);
        let doc: Option<Collection> = match serde_json::from_value(value.clone()) {
            Ok(c) => Some(c),
            Err(e) => {
                errs.push(format!("deserialize: {e}"));
                None
            }
        };
        if let Some(c) = &doc {
            errs.extend(c.invariant_errors());
        }
        match (errs.is_empty(), doc) {
            (true, Some(c)) => Ok(c),
            _ => Err(errs),
        }
    }

    /// Licence-rule findings for this collection over its member rows:
    /// `(errors, warnings)`. Public collections turn every finding into an error.
    pub fn license_findings(&self, rows: &[&CatalogRow]) -> (Vec<String>, Vec<String>) {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        for row in rows {
            let lic = &row.provenance.license;
            for e in lic.rule_errors() {
                errors.push(format!("{}: {e}", row.id));
            }
            let finding = match lic.status {
                LicenseStatus::Recorded => {
                    let spdx = lic.spdx.clone().unwrap_or_default();
                    if !self.policy.allowed_spdx.is_empty()
                        && !self.policy.allowed_spdx.contains(&spdx)
                    {
                        Some(format!(
                            "{}: spdx {spdx} is not in policy.allowed_spdx",
                            row.id
                        ))
                    } else {
                        None
                    }
                }
                LicenseStatus::Unknown => Some(format!("{}: license status is unknown", row.id)),
                LicenseStatus::Restricted => {
                    Some(format!("{}: license status is restricted", row.id))
                }
            };
            if let Some(f) = finding {
                if self.visibility == Visibility::Public
                    || (self.policy.license_required && lic.status == LicenseStatus::Restricted)
                {
                    errors.push(f);
                } else {
                    warnings.push(f);
                }
            }
        }
        (errors, warnings)
    }
}

pub fn catalog_schema() -> serde_json::Value {
    serde_json::from_str(CATALOG_SCHEMA_JSON).expect("committed catalog schema parses")
}

pub fn collection_schema() -> serde_json::Value {
    serde_json::from_str(COLLECTION_SCHEMA_JSON).expect("committed collection schema parses")
}

/// The schema schemars derives from [`CatalogRow`].
pub fn derived_catalog_schema() -> serde_json::Value {
    schemars::schema_for!(CatalogRow).to_value()
}

/// The schema schemars derives from [`Collection`].
pub fn derived_collection_schema() -> serde_json::Value {
    schemars::schema_for!(Collection).to_value()
}
