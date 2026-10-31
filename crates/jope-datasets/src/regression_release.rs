//! Immutable ten-regression release and family/exposure registry (W1-T02).
//!
//! A release is *derived* from the frozen thirty-dataset panel
//! (`benchmarks/panel.json`, schema `jope.benchmark-panel/v1`); it never rewrites
//! that panel, its prepared partitions or any historical report. For every
//! regression member it binds the acquired raw object, the prepared
//! development/split/sealed partitions, the target identity and original label
//! roles, the ordered row identities, the development row/target alignment, the
//! feature schema, the group and fold assignments and the preparation source and
//! configuration. It also binds the digest of a separately versioned
//! [`FamilyRegistry`] that records aliases, alternate targets, resampling,
//! shared entities, unresolved relationships and per-component exposure.
//!
//! Canonical encoding: JSON with object keys in byte order, no insignificant
//! whitespace, integers and strings only (a float is an encoding error) and one
//! trailing newline. [`RegressionRelease::open`] and [`FamilyRegistry::open`]
//! reject unknown fields, unknown schema versions, non-canonical bytes and (for
//! a release) an identity that does not match its content, so a stored document
//! reopens to exactly its original bytes.
//!
//! Derivation reads only pre-staged local objects named by their exact identity.
//! It never fetches, never consults a cache, `latest` pointer or alternate
//! version, and never substitutes another object: a missing or mismatching
//! object fails. Sealed partitions are hashed as opaque bytes and never parsed.

use crate::hash::sha256_hex;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};

pub const RELEASE_SCHEMA: &str = "jope.regression-release/v1";
pub const REGISTRY_SCHEMA: &str = "jope.regression-family-registry/v1";
pub const PANEL_SCHEMA: &str = "jope.benchmark-panel/v1";
const PREPARED_TABLE_SCHEMA: &str = "jope.prepared-table/v1";
const DEVELOPMENT_PARTITION: &str = "development";
/// Original role of labels in the development partition: adaptive development
/// evidence, visible to optimization. Never an untouched outer score.
pub const DEVELOPMENT_LABEL_ROLE: &str = "development";
/// Original role of labels in the sealed partition: withheld from learning.
pub const SEALED_LABEL_ROLE: &str = "sealed_final";
/// PMLB's single original target column; no alternate-label conversion.
pub const TARGET_COLUMN: &str = "target";
pub const TARGET_KIND: &str = "original_numeric_column";
pub const MEMBERSHIP_RULE: &str = "every frozen panel entry whose original task is regression, \
in frozen panel order; no substitution, addition or outcome-based selection";
pub const PANEL_ENTRIES: usize = 30;
pub const REGRESSION_MEMBERS: usize = 10;
/// Upper bound for any single staged object read during derivation/verification.
const MAX_OBJECT_BYTES: u64 = 256 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Canonical encoding
// ---------------------------------------------------------------------------

/// Canonical bytes: sorted keys, compact separators, integers only, trailing newline.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write_canonical(value, &mut out)?;
    out.push(b'\n');
    Ok(out)
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            ensure!(
                n.is_u64() || n.is_i64(),
                "canonical release encoding admits integers only, found {n}"
            );
            out.extend_from_slice(n.to_string().as_bytes());
        }
        Value::String(s) => out.extend_from_slice(serde_json::to_string(s)?.as_bytes()),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
            out.push(b'{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                out.push(b':');
                write_canonical(&map[key], out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    canonical_json(&serde_json::to_value(value)?)
}

fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_hex(&canonical(value)?))
}

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}

/// Parse `bytes` as `T` only when they are exactly the canonical encoding of the
/// parsed value and name the expected schema. Unknown fields fail through
/// `deny_unknown_fields` on every document type.
fn open_canonical<T: Serialize + serde::de::DeserializeOwned>(
    bytes: &[u8],
    schema: &str,
    what: &str,
) -> Result<T> {
    let value: Value =
        serde_json::from_slice(bytes).with_context(|| format!("parse {what} JSON"))?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some(schema),
        "unsupported {what} schema or version (expected {schema})"
    );
    let parsed: T = serde_json::from_value(value).with_context(|| format!("{what} fields"))?;
    ensure!(
        canonical(&parsed)? == bytes,
        "{what} bytes are not the canonical encoding of their content"
    );
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// Shared value types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Numeric,
    Categorical,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feature {
    pub name: String,
    pub kind: FeatureKind,
}

// ---------------------------------------------------------------------------
// Release document
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionRelease {
    pub schema: String,
    /// SHA-256 of the canonical encoding of every other field.
    pub release_id: String,
    pub derived_from: PanelDerivation,
    pub preparation: PreparationBinding,
    pub members: Vec<ReleaseMember>,
    pub exclusions: Vec<Exclusion>,
    pub family_registry_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelDerivation {
    pub panel_id: String,
    pub panel_schema: String,
    /// SHA-256 of the frozen panel manifest bytes the release was derived from.
    pub panel_sha256: String,
    pub panel_entries: u64,
    pub repository: String,
    pub revision: String,
    pub membership_rule: String,
    /// Original-task census of the frozen panel, including zero counts.
    pub task_counts: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparationBinding {
    pub protocol: String,
    pub seed: u64,
    pub folds: u64,
    /// Repository-relative path of the preparation implementation.
    pub source_path: String,
    pub source_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseMember {
    pub dataset: String,
    /// Position of this entry in the frozen panel; proves membership order.
    pub panel_index: u64,
    pub task: String,
    pub raw: RawBinding,
    pub target: TargetBinding,
    pub schema: SchemaBinding,
    pub rows: RowBinding,
    pub groups: GroupBinding,
    pub folds: FoldBinding,
    pub prepared: PreparedBinding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawBinding {
    /// SHA-256 of the acquired object exactly as staged (`downloads/<sha256>`).
    pub sha256: String,
    pub bytes: u64,
    pub metadata_sha256: String,
    /// Provenance only. Derivation and verification never fetch these.
    pub origin_url: String,
    pub metadata_url: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetBinding {
    pub column: String,
    pub kind: String,
    pub development_label_role: String,
    pub sealed_label_role: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemaBinding {
    pub features: Vec<Feature>,
    /// Canonical digest of `features` (names, kinds and order).
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowBinding {
    pub total: u64,
    pub development: u64,
    pub sealed: u64,
    /// Canonical digest of every row id in original source order (both partitions).
    pub ordered_row_ids_sha256: String,
    /// Canonical digest of `[id, source_index, target]` for development rows, in order.
    pub development_alignment_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupBinding {
    pub columns: Vec<String>,
    /// Canonical digest of `[id, group]` for every row in source order.
    pub assignment_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldBinding {
    pub count: u64,
    pub fold_rows: Vec<u64>,
    /// Canonical digest of `[id, fold|null]` for every row in source order.
    pub assignment_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedBinding {
    pub split_sha256: String,
    pub development_sha256: String,
    /// Hashed as opaque bytes; sealed labels are never parsed by this module.
    pub sealed_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exclusion {
    pub dataset: String,
    pub panel_index: u64,
    pub task: String,
    pub reason: String,
}

impl RegressionRelease {
    /// Strict decode: exact schema, no unknown fields, canonical bytes, valid identity.
    pub fn open(bytes: &[u8]) -> Result<Self> {
        let release: Self = open_canonical(bytes, RELEASE_SCHEMA, "regression release")?;
        release.validate()?;
        Ok(release)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        canonical(self)
    }

    /// Identity over every field except `release_id` itself.
    pub fn compute_id(&self) -> Result<String> {
        let mut body = serde_json::to_value(self)?;
        body.as_object_mut()
            .context("release is not an object")?
            .remove("release_id");
        Ok(sha256_hex(&canonical_json(&body)?))
    }

    pub fn member(&self, dataset: &str) -> Result<&ReleaseMember> {
        self.members
            .iter()
            .find(|m| m.dataset == dataset)
            .with_context(|| format!("{dataset} is not a member of release {}", self.release_id))
    }

    /// Structural validation independent of staged bytes.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == RELEASE_SCHEMA, "unsupported release schema");
        let d = &self.derived_from;
        ensure!(
            d.panel_schema == PANEL_SCHEMA
                && d.panel_entries == PANEL_ENTRIES as u64
                && d.membership_rule == MEMBERSHIP_RULE
                && is_sha256(&d.panel_sha256)
                && !d.panel_id.is_empty(),
            "release panel derivation is not the frozen thirty-entry panel rule"
        );
        ensure!(
            d.task_counts.values().sum::<u64>() == PANEL_ENTRIES as u64
                && d.task_counts.get("regression") == Some(&(REGRESSION_MEMBERS as u64)),
            "release task census does not account for every panel entry"
        );
        let p = &self.preparation;
        ensure!(
            !p.protocol.is_empty()
                && p.folds > 0
                && is_sha256(&p.source_sha256)
                && relative_normal(&p.source_path),
            "release preparation binding incomplete"
        );
        ensure!(
            is_sha256(&self.family_registry_sha256),
            "release family registry digest malformed"
        );
        ensure!(
            self.members.len() == REGRESSION_MEMBERS
                && self.exclusions.len() == PANEL_ENTRIES - REGRESSION_MEMBERS,
            "release must contain exactly ten regression members and twenty exclusions"
        );
        let mut indices = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut previous = None;
        for m in &self.members {
            ensure!(
                previous.is_none_or(|p| p < m.panel_index),
                "members must keep frozen panel order"
            );
            previous = Some(m.panel_index);
            ensure!(
                indices.insert(m.panel_index) && names.insert(m.dataset.as_str()),
                "duplicate release member"
            );
            m.validate(p)?;
        }
        let mut previous = None;
        for e in &self.exclusions {
            ensure!(
                previous.is_none_or(|p| p < e.panel_index),
                "exclusions must keep frozen panel order"
            );
            previous = Some(e.panel_index);
            ensure!(
                safe_id(&e.dataset)
                    && e.task != "regression"
                    && e.reason == exclusion_reason(&e.task)
                    && indices.insert(e.panel_index)
                    && names.insert(e.dataset.as_str()),
                "invalid or duplicate exclusion {}",
                e.dataset
            );
        }
        ensure!(
            indices == (0..PANEL_ENTRIES as u64).collect::<BTreeSet<_>>(),
            "members and exclusions must partition the frozen panel"
        );
        ensure!(
            self.release_id == self.compute_id()?,
            "release identity does not match release content"
        );
        Ok(())
    }

    /// Re-derive `dataset` from staged bytes and require the exact bound member.
    /// This is the receiving-side check: a changed source, target, row order,
    /// schema, group or fold assignment is rejected, never reinterpreted.
    pub fn verify_staged(&self, dataset: &str, staged: &StagedMember) -> Result<()> {
        let member = self.member(dataset)?;
        let spec = MemberSpec::from_member(member);
        let recomputed = bind_member(&spec, &self.preparation, staged)?;
        ensure!(
            &recomputed == member,
            "{dataset}: staged inputs differ from release {}",
            self.release_id
        );
        Ok(())
    }

    /// Receiving-side development check for a worker that is handed only the
    /// development partition: exact bytes plus recomputed row/target alignment.
    pub fn verify_development(&self, dataset: &str, development: &[u8]) -> Result<()> {
        let member = self.member(dataset)?;
        ensure!(
            sha256_hex(development) == member.prepared.development_sha256,
            "{dataset}: development bytes differ from release"
        );
        let table: DevelopmentFile =
            serde_json::from_slice(development).context("parse development partition")?;
        ensure!(
            alignment_digest(&table.rows)? == member.rows.development_alignment_sha256,
            "{dataset}: development row/target alignment differs from release"
        );
        Ok(())
    }
}

/// Identity of the committed ten-regression release (`releases/regression-10-v1/`).
pub const COMMITTED_RELEASE_ID: &str =
    "859ce6895f6ca2b016103465f44e4f84a3dd54dcce8b2c832ba15ec536454d4a";
const COMMITTED_RELEASE: &[u8] = include_bytes!("../releases/regression-10-v1/release.json");
const COMMITTED_REGISTRY: &[u8] =
    include_bytes!("../releases/regression-10-v1/family-registry.json");

/// The committed ten-regression release (W1-T01), embedded at build time and
/// opened with every identity check, together with the family registry it
/// binds. Receivers consume this release; they cannot be pointed at another
/// release document or registry.
pub fn committed() -> Result<RegressionRelease> {
    let release = RegressionRelease::open(COMMITTED_RELEASE)?;
    ensure!(
        release.release_id == COMMITTED_RELEASE_ID,
        "committed regression release identity changed"
    );
    ensure!(
        FamilyRegistry::open(COMMITTED_REGISTRY)?.sha256()? == release.family_registry_sha256,
        "committed family registry is not the one the release binds"
    );
    Ok(release)
}

fn exclusion_reason(task: &str) -> String {
    format!("task_not_regression:{task}")
}

fn relative_normal(path: &str) -> bool {
    let p = Path::new(path);
    !path.is_empty() && p.components().all(|c| matches!(c, Component::Normal(_)))
}

impl ReleaseMember {
    fn validate(&self, preparation: &PreparationBinding) -> Result<()> {
        let id = &self.dataset;
        ensure!(
            safe_id(id) && self.task == "regression",
            "{id}: invalid member identity"
        );
        for (name, value) in [
            ("raw", &self.raw.sha256),
            ("metadata", &self.raw.metadata_sha256),
            ("schema", &self.schema.sha256),
            ("row ids", &self.rows.ordered_row_ids_sha256),
            ("alignment", &self.rows.development_alignment_sha256),
            ("groups", &self.groups.assignment_sha256),
            ("folds", &self.folds.assignment_sha256),
            ("split", &self.prepared.split_sha256),
            ("development", &self.prepared.development_sha256),
            ("sealed", &self.prepared.sealed_sha256),
        ] {
            ensure!(is_sha256(value), "{id}: malformed {name} digest");
        }
        ensure!(
            self.target
                == TargetBinding {
                    column: TARGET_COLUMN.into(),
                    kind: TARGET_KIND.into(),
                    development_label_role: DEVELOPMENT_LABEL_ROLE.into(),
                    sealed_label_role: SEALED_LABEL_ROLE.into(),
                },
            "{id}: target identity or original label roles changed"
        );
        ensure!(
            self.schema.sha256 == digest(&self.schema.features)?,
            "{id}: schema digest does not match features"
        );
        validate_features(&self.schema.features, &self.groups.columns)?;
        ensure!(
            self.rows.total == self.rows.development + self.rows.sealed
                && self.rows.development > 0
                && self.rows.sealed > 0,
            "{id}: row census inconsistent"
        );
        ensure!(
            self.folds.count == preparation.folds
                && self.folds.fold_rows.len() as u64 == self.folds.count
                && self.folds.fold_rows.iter().sum::<u64>() == self.rows.development
                && self.folds.fold_rows.iter().all(|n| *n > 0),
            "{id}: fold census inconsistent"
        );
        Ok(())
    }
}

fn validate_features(features: &[Feature], group_columns: &[String]) -> Result<()> {
    let mut names = BTreeSet::new();
    ensure!(
        !features.is_empty()
            && features
                .iter()
                .all(|f| !f.name.is_empty() && names.insert(f.name.as_str())),
        "feature names empty or duplicate"
    );
    ensure!(
        !names.contains(TARGET_COLUMN),
        "the target column cannot also be a feature"
    );
    let mut groups = BTreeSet::new();
    ensure!(
        group_columns
            .iter()
            .all(|g| names.contains(g.as_str()) && groups.insert(g.as_str())),
        "group columns must be distinct declared features"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Frozen panel and staged prepared inputs (parse-only views)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Panel {
    schema: String,
    panel_id: String,
    repository: String,
    revision: String,
    seed: u64,
    #[allow(dead_code)]
    selection: String,
    folds: u64,
    datasets: Vec<PanelEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PanelEntry {
    id: String,
    task: String,
    metric: String,
    rows: u64,
    features: Vec<Feature>,
    classes: Vec<String>,
    group_columns: Vec<String>,
    source: PanelSource,
    split: PanelSplit,
    #[allow(dead_code)]
    license: Value,
    #[allow(dead_code)]
    checkpoint_exposure: String,
    #[allow(dead_code)]
    description: String,
    #[allow(dead_code)]
    group_limitations: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PanelSource {
    url: String,
    sha256: String,
    bytes: u64,
    metadata_url: String,
    metadata_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PanelSplit {
    protocol: String,
    sha256: String,
    development_sha256: String,
    sealed_sha256: String,
    development_rows: u64,
    sealed_rows: u64,
    fold_rows: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitFile {
    protocol: String,
    seed: u64,
    source_sha256: String,
    rows: Vec<SplitRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitRow {
    id: String,
    source_index: u64,
    group: String,
    fold: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DevelopmentFile {
    schema: String,
    dataset: String,
    task: String,
    source_sha256: String,
    partition: String,
    features: Vec<Feature>,
    classes: Vec<String>,
    rows: Vec<DevelopmentRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DevelopmentRow {
    id: String,
    source_index: u64,
    group: String,
    fold: u64,
    values: Vec<Option<String>>,
    target: String,
}

/// Exact pre-staged objects for one member. `sealed` is hashed, never parsed.
#[derive(Clone, Debug, Default)]
pub struct StagedMember {
    pub raw: Vec<u8>,
    pub metadata: Vec<u8>,
    pub development: Vec<u8>,
    pub split: Vec<u8>,
    pub sealed: Vec<u8>,
}

/// Declared identities a member must reproduce, from the panel or a release.
struct MemberSpec {
    dataset: String,
    panel_index: u64,
    raw: RawBinding,
    features: Vec<Feature>,
    group_columns: Vec<String>,
    total: u64,
    development: u64,
    sealed: u64,
    fold_rows: Vec<u64>,
    prepared: PreparedBinding,
}

impl MemberSpec {
    fn from_panel(index: usize, e: &PanelEntry) -> Self {
        Self {
            dataset: e.id.clone(),
            panel_index: index as u64,
            raw: RawBinding {
                sha256: e.source.sha256.clone(),
                bytes: e.source.bytes,
                metadata_sha256: e.source.metadata_sha256.clone(),
                origin_url: e.source.url.clone(),
                metadata_url: e.source.metadata_url.clone(),
            },
            features: e.features.clone(),
            group_columns: e.group_columns.clone(),
            total: e.rows,
            development: e.split.development_rows,
            sealed: e.split.sealed_rows,
            fold_rows: e.split.fold_rows.clone(),
            prepared: PreparedBinding {
                split_sha256: e.split.sha256.clone(),
                development_sha256: e.split.development_sha256.clone(),
                sealed_sha256: e.split.sealed_sha256.clone(),
            },
        }
    }

    fn from_member(m: &ReleaseMember) -> Self {
        Self {
            dataset: m.dataset.clone(),
            panel_index: m.panel_index,
            raw: m.raw.clone(),
            features: m.schema.features.clone(),
            group_columns: m.groups.columns.clone(),
            total: m.rows.total,
            development: m.rows.development,
            sealed: m.rows.sealed,
            fold_rows: m.folds.fold_rows.clone(),
            prepared: m.prepared.clone(),
        }
    }
}

fn alignment_digest(rows: &[DevelopmentRow]) -> Result<String> {
    let aligned: Vec<(&str, u64, &str)> = rows
        .iter()
        .map(|r| (r.id.as_str(), r.source_index, r.target.as_str()))
        .collect();
    digest(&aligned)
}

fn parse_finite(value: &str, what: &str) -> Result<f64> {
    let parsed: f64 = value
        .trim()
        .parse()
        .with_context(|| format!("{what} is not numeric: {value:?}"))?;
    ensure!(parsed.is_finite(), "{what} is not finite: {value:?}");
    Ok(parsed)
}

/// Recompute a member binding from exact staged bytes against declared identities.
fn bind_member(
    spec: &MemberSpec,
    preparation: &PreparationBinding,
    staged: &StagedMember,
) -> Result<ReleaseMember> {
    let id = spec.dataset.as_str();
    ensure!(safe_id(id), "unsafe dataset identifier {id:?}");
    ensure!(
        sha256_hex(&staged.raw) == spec.raw.sha256 && staged.raw.len() as u64 == spec.raw.bytes,
        "{id}: staged raw object differs from its declared identity"
    );
    ensure!(
        sha256_hex(&staged.metadata) == spec.raw.metadata_sha256,
        "{id}: staged source metadata differs from its declared identity"
    );
    ensure!(
        sha256_hex(&staged.split) == spec.prepared.split_sha256,
        "{id}: staged split differs from its declared identity"
    );
    ensure!(
        sha256_hex(&staged.development) == spec.prepared.development_sha256,
        "{id}: staged development partition differs from its declared identity"
    );
    ensure!(
        sha256_hex(&staged.sealed) == spec.prepared.sealed_sha256,
        "{id}: staged sealed partition differs from its declared identity"
    );
    validate_features(&spec.features, &spec.group_columns)?;

    let split: SplitFile = serde_json::from_slice(&staged.split).context("parse split")?;
    ensure!(
        split.protocol == preparation.protocol
            && split.seed == preparation.seed
            && split.source_sha256 == spec.raw.sha256,
        "{id}: split protocol, seed or source differs from the preparation binding"
    );
    ensure!(
        split.rows.len() as u64 == spec.total,
        "{id}: split row count differs from declared total"
    );
    let folds = preparation.folds;
    let mut ids = BTreeSet::new();
    let mut group_fold: BTreeMap<&str, Option<u64>> = BTreeMap::new();
    let mut fold_rows = vec![0u64; folds as usize];
    let mut sealed = 0u64;
    for (position, row) in split.rows.iter().enumerate() {
        ensure!(
            row.source_index == position as u64,
            "{id}: split rows are not in original source order"
        );
        ensure!(
            is_sha256(&row.id) && ids.insert(row.id.as_str()),
            "{id}: invalid or duplicate row identity"
        );
        ensure!(is_sha256(&row.group), "{id}: invalid group identity");
        match row.fold {
            Some(f) => {
                ensure!(f < folds, "{id}: fold outside the preparation protocol");
                fold_rows[f as usize] += 1;
            }
            None => sealed += 1,
        }
        if let Some(previous) = group_fold.insert(row.group.as_str(), row.fold) {
            ensure!(
                previous == row.fold,
                "{id}: a group crosses folds or the sealed boundary"
            );
        }
    }
    ensure!(
        fold_rows == spec.fold_rows && sealed == spec.sealed,
        "{id}: fold or sealed census differs from declared split"
    );
    ensure!(
        fold_rows.iter().sum::<u64>() == spec.development,
        "{id}: development census differs from declared split"
    );

    let table: DevelopmentFile =
        serde_json::from_slice(&staged.development).context("parse development partition")?;
    ensure!(
        table.schema == PREPARED_TABLE_SCHEMA
            && table.partition == DEVELOPMENT_PARTITION
            && table.dataset == id
            && table.task == "regression"
            && table.source_sha256 == spec.raw.sha256
            && table.classes.is_empty(),
        "{id}: development table identity differs from the release member"
    );
    ensure!(
        table.features == spec.features,
        "{id}: development feature schema differs from the declared schema"
    );
    let expected: Vec<&SplitRow> = split.rows.iter().filter(|r| r.fold.is_some()).collect();
    ensure!(
        table.rows.len() == expected.len(),
        "{id}: development rows differ from split membership"
    );
    for (row, split_row) in table.rows.iter().zip(&expected) {
        ensure!(
            row.id == split_row.id
                && row.source_index == split_row.source_index
                && row.group == split_row.group
                && Some(row.fold) == split_row.fold,
            "{id}: development row order, identity, group or fold differs from split"
        );
        ensure!(
            row.values.len() == spec.features.len(),
            "{id}: ragged development row"
        );
        for (feature, value) in spec.features.iter().zip(&row.values) {
            if let (FeatureKind::Numeric, Some(v)) = (feature.kind, value) {
                parse_finite(v, "numeric feature")?;
            }
        }
        parse_finite(&row.target, "regression target")?;
    }

    let row_ids: Vec<&str> = split.rows.iter().map(|r| r.id.as_str()).collect();
    let groups: Vec<(&str, &str)> = split
        .rows
        .iter()
        .map(|r| (r.id.as_str(), r.group.as_str()))
        .collect();
    let fold_assignment: Vec<(&str, Option<u64>)> =
        split.rows.iter().map(|r| (r.id.as_str(), r.fold)).collect();
    Ok(ReleaseMember {
        dataset: id.into(),
        panel_index: spec.panel_index,
        task: "regression".into(),
        raw: spec.raw.clone(),
        target: TargetBinding {
            column: TARGET_COLUMN.into(),
            kind: TARGET_KIND.into(),
            development_label_role: DEVELOPMENT_LABEL_ROLE.into(),
            sealed_label_role: SEALED_LABEL_ROLE.into(),
        },
        schema: SchemaBinding {
            features: spec.features.clone(),
            sha256: digest(&spec.features)?,
        },
        rows: RowBinding {
            total: spec.total,
            development: spec.development,
            sealed: spec.sealed,
            ordered_row_ids_sha256: digest(&row_ids)?,
            development_alignment_sha256: alignment_digest(&table.rows)?,
        },
        groups: GroupBinding {
            columns: spec.group_columns.clone(),
            assignment_sha256: digest(&groups)?,
        },
        folds: FoldBinding {
            count: folds,
            fold_rows,
            assignment_sha256: digest(&fold_assignment)?,
        },
        prepared: spec.prepared.clone(),
    })
}

/// Inputs to [`derive()`]. Every byte slice is the exact staged object.
pub struct ReleaseSources<'a> {
    pub panel: &'a [u8],
    pub preparation_source_path: &'a str,
    pub preparation_source: &'a [u8],
    pub registry: &'a FamilyRegistry,
    pub staged: &'a BTreeMap<String, StagedMember>,
}

fn parse_panel(bytes: &[u8]) -> Result<Panel> {
    let panel: Panel = serde_json::from_slice(bytes).context("parse frozen panel")?;
    ensure!(panel.schema == PANEL_SCHEMA, "unsupported panel schema");
    ensure!(
        panel.datasets.len() == PANEL_ENTRIES,
        "frozen panel must contain thirty entries"
    );
    let ids: BTreeSet<&str> = panel.datasets.iter().map(|d| d.id.as_str()).collect();
    ensure!(
        ids.len() == PANEL_ENTRIES && ids.iter().all(|id| safe_id(id)),
        "frozen panel identifiers duplicate or unsafe"
    );
    let regression = panel
        .datasets
        .iter()
        .filter(|d| d.task == "regression")
        .count();
    ensure!(
        regression == REGRESSION_MEMBERS,
        "frozen panel must contain exactly ten regression entries, found {regression}"
    );
    for d in panel.datasets.iter().filter(|d| d.task == "regression") {
        ensure!(
            d.metric == "rmse" && d.classes.is_empty(),
            "{}: regression panel entry has classification semantics",
            d.id
        );
    }
    ensure!(
        (1..=64).contains(&panel.folds),
        "panel fold count outside the supported protocol range"
    );
    Ok(panel)
}

/// Regression member ids of a frozen panel, in panel order.
pub fn panel_regression_members(panel: &[u8]) -> Result<Vec<String>> {
    Ok(parse_panel(panel)?
        .datasets
        .iter()
        .filter(|d| d.task == "regression")
        .map(|d| d.id.clone())
        .collect())
}

/// Derive the ten-regression release from the frozen panel and exact staged bytes.
pub fn derive(sources: &ReleaseSources<'_>) -> Result<RegressionRelease> {
    let panel = parse_panel(sources.panel)?;
    ensure!(
        relative_normal(sources.preparation_source_path),
        "preparation source path must be repository-relative"
    );
    sources.registry.validate()?;
    let protocols: BTreeSet<&str> = panel
        .datasets
        .iter()
        .map(|d| d.split.protocol.as_str())
        .collect();
    ensure!(
        protocols.len() == 1,
        "panel entries disagree on the preparation protocol"
    );
    let preparation = PreparationBinding {
        protocol: protocols.into_iter().next().unwrap_or_default().to_owned(),
        seed: panel.seed,
        folds: panel.folds,
        source_path: sources.preparation_source_path.to_owned(),
        source_sha256: sha256_hex(sources.preparation_source),
    };
    let mut members = Vec::new();
    let mut exclusions = Vec::new();
    let mut task_counts: BTreeMap<String, u64> =
        ["regression", "binary", "multiclass", "timeseries"]
            .into_iter()
            .map(|t| (t.to_owned(), 0))
            .collect();
    for (index, entry) in panel.datasets.iter().enumerate() {
        *task_counts.entry(entry.task.clone()).or_default() += 1;
        if entry.task != "regression" {
            exclusions.push(Exclusion {
                dataset: entry.id.clone(),
                panel_index: index as u64,
                task: entry.task.clone(),
                reason: exclusion_reason(&entry.task),
            });
            continue;
        }
        let staged = sources.staged.get(&entry.id).with_context(|| {
            format!(
                "{}: exact staged objects absent; no substitution is permitted",
                entry.id
            )
        })?;
        members.push(bind_member(
            &MemberSpec::from_panel(index, entry),
            &preparation,
            staged,
        )?);
    }
    let extra: Vec<&String> = sources
        .staged
        .keys()
        .filter(|k| !members.iter().any(|m| &m.dataset == *k))
        .collect();
    ensure!(
        extra.is_empty(),
        "staged objects supplied for non-members: {extra:?}"
    );
    let names: BTreeSet<&str> = members.iter().map(|m| m.dataset.as_str()).collect();
    ensure!(
        sources.registry.members() == names,
        "family registry must cover exactly the release members"
    );
    let mut release = RegressionRelease {
        schema: RELEASE_SCHEMA.into(),
        release_id: String::new(),
        derived_from: PanelDerivation {
            panel_id: panel.panel_id,
            panel_schema: panel.schema,
            panel_sha256: sha256_hex(sources.panel),
            panel_entries: PANEL_ENTRIES as u64,
            repository: panel.repository,
            revision: panel.revision,
            membership_rule: MEMBERSHIP_RULE.into(),
            task_counts,
        },
        preparation,
        members,
        exclusions,
        family_registry_sha256: sources.registry.sha256()?,
    };
    release.release_id = release.compute_id()?;
    release.validate()?;
    Ok(release)
}

/// Check a release's declared member identities against frozen panel bytes
/// without opening any staged object (membership proof, not byte verification).
pub fn check_against_panel(release: &RegressionRelease, panel_bytes: &[u8]) -> Result<()> {
    let panel = parse_panel(panel_bytes)?;
    ensure!(
        release.derived_from.panel_sha256 == sha256_hex(panel_bytes)
            && release.derived_from.panel_id == panel.panel_id
            && release.derived_from.repository == panel.repository
            && release.derived_from.revision == panel.revision
            && release.preparation.seed == panel.seed
            && release.preparation.folds == panel.folds,
        "release was not derived from these panel bytes"
    );
    let mut members = release.members.iter();
    let mut exclusions = release.exclusions.iter();
    for (index, entry) in panel.datasets.iter().enumerate() {
        if entry.task == "regression" {
            let m = members.next().context("release lacks a regression entry")?;
            let spec = MemberSpec::from_panel(index, entry);
            ensure!(
                m.dataset == spec.dataset
                    && m.panel_index == spec.panel_index
                    && m.raw == spec.raw
                    && m.schema.features == spec.features
                    && m.groups.columns == spec.group_columns
                    && m.rows.total == spec.total
                    && m.rows.development == spec.development
                    && m.rows.sealed == spec.sealed
                    && m.folds.fold_rows == spec.fold_rows
                    && m.prepared == spec.prepared
                    && entry.split.protocol == release.preparation.protocol,
                "{}: release member differs from the frozen panel entry",
                entry.id
            );
        } else {
            let e = exclusions.next().context("release lacks an exclusion")?;
            ensure!(
                e.dataset == entry.id && e.panel_index == index as u64 && e.task == entry.task,
                "{}: release exclusion differs from the frozen panel entry",
                entry.id
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Staged object loading (no fallback)
// ---------------------------------------------------------------------------

/// Read one exact regular file below `root`. Every component must exist, be a
/// real directory/file (never a symlink) and the opened handle must be the
/// inspected file. There is no alternate location, cache or network fallback.
pub fn read_exact_object(root: &Path, components: &[&str]) -> Result<Vec<u8>> {
    let root_meta = std::fs::symlink_metadata(root)
        .with_context(|| format!("staged root absent: {}", root.display()))?;
    ensure!(
        root_meta.is_dir() && !root_meta.file_type().is_symlink(),
        "staged root must be a real directory: {}",
        root.display()
    );
    ensure!(!components.is_empty(), "empty staged object path");
    let mut path = root.to_path_buf();
    let mut leaf = None;
    for (i, part) in components.iter().enumerate() {
        ensure!(
            relative_normal(part) && Path::new(part).components().count() == 1,
            "unsafe staged path component {part:?}"
        );
        path.push(part);
        let meta = std::fs::symlink_metadata(&path)
            .with_context(|| format!("exact staged object absent: {}", path.display()))?;
        ensure!(
            !meta.file_type().is_symlink(),
            "staged path component is a symlink: {}",
            path.display()
        );
        if i + 1 < components.len() {
            ensure!(meta.is_dir(), "not a directory: {}", path.display());
        } else {
            ensure!(meta.is_file(), "not a regular file: {}", path.display());
            ensure!(meta.len() <= MAX_OBJECT_BYTES, "staged object too large");
            leaf = Some(meta);
        }
    }
    let leaf = leaf.context("staged leaf absent")?;
    let file = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let opened = file.metadata()?;
    ensure!(
        opened.dev() == leaf.dev() && opened.ino() == leaf.ino(),
        "staged object replaced while opening: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(MAX_OBJECT_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 == leaf.len(),
        "staged object changed while reading: {}",
        path.display()
    );
    Ok(bytes)
}

/// Load exactly the regression members' objects of a frozen panel from a
/// preparation root laid out by `benchmarks/prepare.py`. Classification
/// members' objects are never opened.
pub fn load_staged(root: &Path, panel: &[u8]) -> Result<BTreeMap<String, StagedMember>> {
    let panel = parse_panel(panel)?;
    let mut staged = BTreeMap::new();
    for entry in panel.datasets.iter().filter(|d| d.task == "regression") {
        let id = entry.id.as_str();
        ensure!(
            is_sha256(&entry.source.sha256),
            "{id}: malformed raw digest"
        );
        let json = format!("{id}.json");
        staged.insert(
            id.to_owned(),
            StagedMember {
                raw: read_exact_object(root, &["downloads", &entry.source.sha256])?,
                metadata: read_exact_object(root, &["metadata", &format!("{id}.yaml")])?,
                development: read_exact_object(root, &["development", &json])?,
                split: read_exact_object(root, &["splits", &json])?,
                sealed: read_exact_object(root, &["sealed", &json])?,
            },
        );
    }
    Ok(staged)
}

// ---------------------------------------------------------------------------
// Family and exposure registry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// Same dataset under another name or identifier.
    Alias,
    /// Same rows with a different target column.
    AlternateTarget,
    /// Subsample, bootstrap or other resampling of the same source.
    Resampled,
    /// Byte- or row-identical content.
    ContentDuplicate,
    /// Different tables sharing real-world entities (subjects, sites, objects).
    SharedEntity,
    /// Same original collection/source, relationship otherwise unspecified.
    SameSource,
    /// Feature/target derivation of another dataset.
    DerivedVariant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationStatus {
    Resolved,
    Unresolved,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    pub dataset: String,
    pub other: String,
    pub kind: RelationKind,
    pub status: RelationStatus,
    pub evidence: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub family: String,
    pub members: Vec<String>,
    /// External identities inside this family's exclusion unit.
    pub source_keys: Vec<String>,
    pub relations: Vec<Relation>,
}

/// Exposure of one component to a dataset. `Unknown` is a first-class state and
/// is never equivalent to `Unseen`: only `Unseen` with evidence supports a
/// never-seen claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureState {
    Unknown,
    Unseen,
    Exposed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogState {
    Unknown,
    Listed,
    NotListed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    pub description: String,
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exposure {
    pub state: ExposureState,
    pub basis: String,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogAvailability {
    pub state: CatalogState,
    pub basis: String,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureRecord {
    pub dataset: String,
    /// Catalog availability is not exposure; it is recorded separately.
    pub catalog: CatalogAvailability,
    pub encoder: Exposure,
    pub teacher: Exposure,
    pub checkpoint: Exposure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRegistry {
    pub schema: String,
    pub registry_version: u64,
    pub families: Vec<Family>,
    pub exposures: Vec<ExposureRecord>,
}

impl Exposure {
    fn validate(&self, what: &str) -> Result<()> {
        ensure!(
            !self.basis.trim().is_empty(),
            "{what}: exposure basis required"
        );
        for e in &self.evidence {
            ensure!(
                !e.description.trim().is_empty() && e.sha256.as_deref().is_none_or(is_sha256),
                "{what}: malformed exposure evidence"
            );
        }
        if self.state != ExposureState::Unknown {
            ensure!(
                !self.evidence.is_empty(),
                "{what}: {:?} requires evidence; absent evidence is unknown",
                self.state
            );
        }
        Ok(())
    }
}

impl FamilyRegistry {
    pub fn open(bytes: &[u8]) -> Result<Self> {
        let registry: Self = open_canonical(bytes, REGISTRY_SCHEMA, "family registry")?;
        registry.validate()?;
        Ok(registry)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        canonical(self)
    }

    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_hex(&self.encode()?))
    }

    pub fn members(&self) -> BTreeSet<&str> {
        self.families
            .iter()
            .flat_map(|f| f.members.iter().map(String::as_str))
            .collect()
    }

    fn family_of(&self, key: &str) -> Option<&Family> {
        self.families
            .iter()
            .find(|f| f.members.iter().any(|m| m == key) || f.source_keys.iter().any(|k| k == key))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == REGISTRY_SCHEMA && self.registry_version >= 1,
            "unsupported family registry"
        );
        let mut families = BTreeSet::new();
        let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
        let mut members = BTreeSet::new();
        let mut previous: Option<&str> = None;
        for f in &self.families {
            ensure!(
                !f.family.trim().is_empty() && families.insert(f.family.as_str()),
                "family identifier empty or duplicate"
            );
            ensure!(
                previous.is_none_or(|p| p < f.family.as_str()),
                "families must be sorted by identifier"
            );
            previous = Some(&f.family);
            ensure!(!f.members.is_empty(), "{}: family has no members", f.family);
            for key in f.members.iter().chain(&f.source_keys) {
                ensure!(!key.trim().is_empty(), "{}: empty family key", f.family);
                if let Some(other) = owner.insert(key.as_str(), f.family.as_str()) {
                    bail!(
                        "{key} belongs to both {other} and {}: an alias cannot span families",
                        f.family
                    );
                }
            }
            for m in &f.members {
                ensure!(safe_id(m), "{}: unsafe member id {m:?}", f.family);
                members.insert(m.as_str());
            }
        }
        for f in &self.families {
            for r in &f.relations {
                ensure!(
                    f.members.contains(&r.dataset),
                    "{}: relation subject {} is not a family member",
                    f.family,
                    r.dataset
                );
                ensure!(
                    !r.other.trim().is_empty()
                        && r.other != r.dataset
                        && !r.evidence.trim().is_empty(),
                    "{}: relation needs a distinct counterpart and evidence",
                    f.family
                );
                if r.status == RelationStatus::Resolved {
                    ensure!(
                        owner.get(r.other.as_str()) == Some(&f.family.as_str()),
                        "{}: resolved {:?} relation to {} straddles the family exclusion boundary",
                        f.family,
                        r.kind,
                        r.other
                    );
                }
            }
        }
        let mut seen = BTreeSet::new();
        let mut previous: Option<&str> = None;
        for e in &self.exposures {
            ensure!(
                members.contains(e.dataset.as_str()) && seen.insert(e.dataset.as_str()),
                "exposure record {} is not a unique family member",
                e.dataset
            );
            ensure!(
                previous.is_none_or(|p| p < e.dataset.as_str()),
                "exposure records must be sorted by dataset"
            );
            previous = Some(&e.dataset);
            ensure!(
                !e.catalog.basis.trim().is_empty()
                    && (e.catalog.state == CatalogState::Unknown || !e.catalog.evidence.is_empty()),
                "{}: catalog availability needs basis and evidence",
                e.dataset
            );
            e.encoder.validate(&format!("{} encoder", e.dataset))?;
            e.teacher.validate(&format!("{} teacher", e.dataset))?;
            e.checkpoint
                .validate(&format!("{} checkpoint", e.dataset))?;
        }
        ensure!(
            seen == members,
            "every family member needs exactly one exposure record"
        );
        Ok(())
    }

    /// Reasons a "never-seen family" claim about `dataset` is blocked. An empty
    /// result is the only permission; unknown exposure or any unresolved
    /// relationship touching the dataset's family blocks the claim.
    pub fn never_seen_claim_blockers(&self, dataset: &str) -> Result<Vec<String>> {
        let family = self
            .family_of(dataset)
            .with_context(|| format!("{dataset} has no family assignment"))?;
        let record = self
            .exposures
            .iter()
            .find(|e| e.dataset == dataset)
            .with_context(|| format!("{dataset} has no exposure record"))?;
        let mut blockers = Vec::new();
        for (component, exposure) in [
            ("encoder", &record.encoder),
            ("teacher", &record.teacher),
            ("checkpoint", &record.checkpoint),
        ] {
            match exposure.state {
                ExposureState::Unseen => {}
                ExposureState::Unknown => {
                    blockers.push(format!("{component} exposure unknown: {}", exposure.basis))
                }
                ExposureState::Exposed => {
                    blockers.push(format!("{component} exposed: {}", exposure.basis))
                }
            }
        }
        for f in &self.families {
            for r in f
                .relations
                .iter()
                .filter(|r| r.status == RelationStatus::Unresolved)
            {
                let touches = f.family == family.family
                    || r.other == dataset
                    || family.members.contains(&r.other)
                    || family.source_keys.contains(&r.other);
                if touches {
                    blockers.push(format!(
                        "unresolved {:?} relation {} -> {}: {}",
                        r.kind, r.dataset, r.other, r.evidence
                    ));
                }
            }
        }
        Ok(blockers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_encoding_sorts_keys_and_refuses_floats() {
        let value =
            serde_json::json!({"b": 1, "a": [true, null, "é\"x"], "c": {"z": -2, "y": "s"}});
        assert_eq!(
            canonical_json(&value).unwrap(),
            b"{\"a\":[true,null,\"\xc3\xa9\\\"x\"],\"b\":1,\"c\":{\"y\":\"s\",\"z\":-2}}\n"
        );
        assert!(canonical_json(&serde_json::json!({"x": 1.5})).is_err());
    }

    #[test]
    fn unknown_is_a_distinct_state_from_unseen() {
        assert_ne!(ExposureState::Unknown, ExposureState::Unseen);
        for bad in ["never_seen", "none", "", "unseen?", "Unknown"] {
            assert!(serde_json::from_value::<ExposureState>(Value::String(bad.into())).is_err());
        }
        assert_eq!(
            serde_json::from_value::<ExposureState>(Value::String("unknown".into())).unwrap(),
            ExposureState::Unknown
        );
    }
}
