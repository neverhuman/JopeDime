//! The two committed records: `jope.hwm/v1` (one high-water mark per exact cut + evaluator
//! identity) and `jope.score/v1` (one row per evaluation, never edited).

use crate::canon::{canonical_sha256, is_hex64};
use crate::metric::{Direction, Metric};
use anyhow::{ensure, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg_attr(doc, doc = "Schema identifier for HWM values.")]
pub const HWM_SCHEMA: &str = "jope.hwm/v1";
#[cfg_attr(doc, doc = "Schema identifier for score values.")]
pub const SCORE_SCHEMA: &str = "jope.score/v1";
/// The evaluator marker for records whose scoring binary is not bound by hash.
pub const UNBOUND: &str = "unbound";

/// Numerical and class semantics required by new measured teaching admission.
/// Historical identities omit this field and retain their original hashes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComparisonContext {
    #[cfg_attr(doc, doc = "Numerical policy recorded for this comparison context.")]
    pub numerical_policy: String,
    #[cfg_attr(doc, doc = "Stage recorded for this comparison context.")]
    pub stage: String,
    #[cfg_attr(doc, doc = "Aggregation recorded for this comparison context.")]
    pub aggregation: String,
    #[cfg_attr(doc, doc = "Label role recorded for this comparison context.")]
    pub label_role: String,
    #[cfg_attr(doc, doc = "Sealed recorded for this comparison context.")]
    pub sealed: bool,
    #[cfg_attr(doc, doc = "Benchmark valid recorded for this comparison context.")]
    pub benchmark_valid: bool,
    #[cfg_attr(doc, doc = "Recipe encoding recorded for this comparison context.")]
    pub recipe_encoding: String,
    #[cfg_attr(
        doc,
        doc = "Collection of class order values carried by this comparison context."
    )]
    pub class_order: Vec<String>,
    /// Proposal/checkpoint lineage when applicable; None when no proposal checkpoint applies.
    pub checkpoint_sha256: Option<String>,
}

/// What `identity_sha256` hashes: all comparison fields, including optional context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HwmIdentity {
    /// Learning task (`regression`, `binary`, ...).
    pub task: String,
    /// Dataset content hash (16 hex for catalogued datasets; legacy sources keep their own ids).
    pub dataset_hash: String,
    /// Split/fold identity: sha256 of the canonical JSON that names the cut (see docs/learning.md).
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub cut_id: String,
    /// sha256 of the evaluator binary, or `unbound`.
    #[schemars(regex(pattern = r"^(unbound|[0-9a-f]{64})$"))]
    pub evaluator_sha256: String,
    /// sha256 of the grammar vocabulary, or `unbound`.
    #[schemars(regex(pattern = r"^(unbound|[0-9a-f]{64})$"))]
    pub vocabulary_sha256: String,
    /// Scoring protocol the score was measured under.
    pub protocol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        doc,
        doc = "Optional comparison value; `None` means it is not available."
    )]
    pub comparison: Option<ComparisonContext>,
}

impl HwmIdentity {
    #[cfg_attr(doc, doc = "Computes the canonical SHA-256 digest of this value.")]
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    doc,
    doc = "`Genome` is a record for the genome contract in this module: The two committed records: jope.hwm/v1 (one high-water mark per exact cut + evaluator identity) and jope.score/v1 (one row per evaluation, never edited)."
)]
pub struct Genome {
    /// The program in its source DSL.
    pub dsl: String,
    /// sha256 of the DSL bytes.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// Semantic (normalised) hash when the source recorded one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub semantic_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    doc,
    doc = "`Score` is a record for the score contract in this module: The two committed records: jope.hwm/v1 (one high-water mark per exact cut + evaluator identity) and jope.score/v1 (one row per evaluation, never edited)."
)]
pub struct Score {
    #[cfg_attr(doc, doc = "Metric used to evaluate the represented result.")]
    pub metric: Metric,
    #[cfg_attr(doc, doc = "Measured or derived value carried by this result.")]
    pub value: f64,
    /// Redundant with the metric table; recorded so a ledger reader needs no code to rank.
    pub direction: Direction,
    /// `original_target_units` (rmse in the target's units), `dimensionless` (r2, auc, ...).
    pub units: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    doc,
    doc = "`Cost` is a record for the cost contract in this module: The two committed records: jope.hwm/v1 (one high-water mark per exact cut + evaluator identity) and jope.score/v1 (one row per evaluation, never edited)."
)]
pub struct Cost {
    /// Fit/evaluate wall seconds of the winning program.
    #[serde(default)]
    pub runtime_seconds: Option<f64>,
    /// Search wall seconds spent to find it, when recorded.
    #[serde(default)]
    pub search_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    doc,
    doc = "`Evaluator` is a record for the evaluator contract in this module: The two committed records: jope.hwm/v1 (one high-water mark per exact cut + evaluator identity) and jope.score/v1 (one row per evaluation, never edited)."
)]
pub struct Evaluator {
    /// sha256 of the evaluator binary, or `unbound`.
    #[schemars(regex(pattern = r"^(unbound|[0-9a-f]{64})$"))]
    pub sha256: String,
    /// Free-form version label of the evaluator (binary name, schema or legacy tag).
    pub version: String,
}

impl Evaluator {
    #[cfg_attr(doc, doc = "Returns whether bound.")]
    pub fn is_bound(&self) -> bool {
        self.sha256 != UNBOUND
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[cfg_attr(
    doc,
    doc = "`Provenance` is a record for the provenance contract in this module: The two committed records: jope.hwm/v1 (one high-water mark per exact cut + evaluator identity) and jope.score/v1 (one row per evaluation, never edited)."
)]
pub struct Provenance {
    /// `champions`, `champions-sealed`, `genebank`, `snapshot-<watermark>`, `sotarena`, ...
    pub source: String,
    /// Source-specific run identity (observation path, generation id, track/version, ...).
    pub run_id: String,
    /// git sha of the tree that produced the record, or `unbound` for legacy imports.
    pub git_sha: String,
    /// Bundle id the record was migrated from, or `unbound`.
    pub bundle_id: String,
    /// sha256 of the record blob under `blobs/sha256/` (the original bytes for migrated records).
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub record_sha256: String,
}

impl Cost {
    /// Both fields are IEEE doubles that JSON cannot represent when they are not finite: serde
    /// writes NaN and the infinities as `null`, and `Option<f64>` reads `null` back as `None`. So
    /// unlike a non-finite *score*, a non-finite cost does not make the record unreadable — it
    /// makes it quietly *wrong*, with the row hash still validating and nothing to notice. Both
    /// records check it here so that the next source to populate `cost` gets the refusal rather
    /// than the silent loss; `sotarena::parse_line` hardcodes `None` today, which is the only
    /// reason this has never fired.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("runtime_seconds", self.runtime_seconds),
            ("search_seconds", self.search_seconds),
        ] {
            if let Some(v) = v {
                ensure!(v.is_finite(), "cost.{name} is not finite");
            }
        }
        Ok(())
    }
}

/// One high-water mark: the best program known for an exact cut under an evaluator identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HwmRecord {
    /// Always `jope.hwm/v1`.
    pub schema: String,
    #[cfg_attr(doc, doc = "Identity recorded for this HWM record.")]
    pub identity: HwmIdentity,
    /// sha256 of the canonical JSON of `identity`.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub identity_sha256: String,
    /// The source's own identity key (`parallel.cut_champion.v1`: `Identity::key()`), when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub legacy_identity_sha256: Option<String>,
    #[cfg_attr(doc, doc = "Canonical program or genome representation.")]
    pub genome: Genome,
    #[cfg_attr(
        doc,
        doc = "Measured score for this outcome; absence denotes a failed draw."
    )]
    pub score: Score,
    #[cfg_attr(doc, doc = "Cost recorded for this HWM record.")]
    pub cost: Cost,
    #[cfg_attr(doc, doc = "Evaluator recorded for this HWM record.")]
    pub evaluator: Evaluator,
    /// True only for scores measured under a benchmark protocol (never for HWM search scores).
    pub benchmark_valid: bool,
    /// True for sealed (`meta_test`) records: kept, never exported to corpora.
    pub sealed: bool,
    /// True when the bound evaluator can reproduce the score (`hwm replay`); unbound = false.
    pub replayable: bool,
    #[cfg_attr(doc, doc = "Provenance recorded for this HWM record.")]
    pub provenance: Provenance,
    /// sha256 of the event/record this one improved on in its chain, null for the first.
    #[schemars(required, schema_with = "nullable_hex64")]
    pub prev_event_sha256: Option<String>,
    /// 0 for the first record of a chain, +1 per strict improvement.
    pub improvement_number: u64,
    /// Unix seconds the score was achieved.
    pub ts_unix: u64,
}

impl HwmRecord {
    /// Structural self-checks that need no other record.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == HWM_SCHEMA, "schema is not {HWM_SCHEMA}");
        ensure!(
            self.identity_sha256 == self.identity.sha256()?,
            "identity_sha256 does not hash the identity"
        );
        ensure!(is_hex64(&self.identity.cut_id), "cut_id is not 64 hex");
        if let Some(context) = &self.identity.comparison {
            ensure!(
                !context.numerical_policy.trim().is_empty(),
                "empty numerical policy"
            );
            ensure!(
                !context.stage.is_empty()
                    && !context.aggregation.is_empty()
                    && !context.label_role.is_empty()
                    && !context.recipe_encoding.is_empty(),
                "incomplete scoring context"
            );
            ensure!(
                context.sealed == self.sealed && context.benchmark_valid == self.benchmark_valid,
                "record role differs from hashed comparison context"
            );
            ensure!(
                context
                    .checkpoint_sha256
                    .as_ref()
                    .is_none_or(|value| is_hex64(value)),
                "invalid comparison checkpoint hash"
            );
            ensure!(
                context
                    .class_order
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == context.class_order.len(),
                "duplicate comparison classes"
            );
        }
        ensure!(
            self.identity.evaluator_sha256 == self.evaluator.sha256,
            "identity.evaluator_sha256 differs from evaluator.sha256"
        );
        ensure!(
            self.evaluator.sha256 == UNBOUND || is_hex64(&self.evaluator.sha256),
            "evaluator sha256 is neither unbound nor 64 hex"
        );
        ensure!(
            self.score.direction == self.score.metric.direction(),
            "score.direction disagrees with the metric table"
        );
        ensure!(self.score.value.is_finite(), "score is not finite");
        self.cost.validate()?;
        ensure!(
            self.genome.program_sha256 == crate::canon::sha256_hex(self.genome.dsl.as_bytes()),
            "program_sha256 does not hash the DSL"
        );
        ensure!(
            is_hex64(&self.provenance.record_sha256),
            "record_sha256 is not 64 hex"
        );
        ensure!(
            self.prev_event_sha256.is_none() == (self.improvement_number == 0),
            "prev_event_sha256 and improvement_number disagree"
        );
        if let Some(p) = &self.prev_event_sha256 {
            ensure!(is_hex64(p), "prev_event_sha256 is not 64 hex");
        }
        ensure!(
            self.evaluator.is_bound() || !self.replayable,
            "an unbound evaluator cannot be replayable"
        );
        Ok(())
    }
}

/// Where a score row came from (kept open: benchmark trackers add their own keys).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScoreSource {
    /// `sotarena`, `campaign`, `arena`, ...
    pub kind: String,
    /// Benchmark track (e.g. `tabular-regression-v2`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
    /// Published version of the track results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Row status as published (`ok`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        doc,
        doc = "Optional source kind value; `None` means it is not available."
    )]
    pub source_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        doc,
        doc = "Optional source revision value; `None` means it is not available."
    )]
    pub source_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        doc,
        doc = "Optional source URI value; `None` means it is not available."
    )]
    pub source_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(doc, doc = "Optional notes value; `None` means it is not available.")]
    pub notes: Option<String>,
}

/// One evaluation result; appended once, never edited.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScoreRecord {
    /// Always `jope.score/v1`.
    pub schema: String,
    #[cfg_attr(doc, doc = "Hash identifying the dataset used by this record.")]
    pub dataset_hash: String,
    /// sha256 of the canonical JSON naming the evaluation cut/protocol.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub cut_id: String,
    #[cfg_attr(doc, doc = "Identifier for the associated method.")]
    pub method_id: String,
    #[schemars(regex(pattern = r"^(unbound|[0-9a-f]{64})$"))]
    #[cfg_attr(
        doc,
        doc = "SHA-256 digest identifying the evaluator used for this evidence."
    )]
    pub evaluator_sha256: String,
    #[cfg_attr(doc, doc = "Protocol recorded for this score record.")]
    pub protocol: String,
    #[cfg_attr(doc, doc = "Metric used to evaluate the represented result.")]
    pub metric: Metric,
    #[cfg_attr(doc, doc = "Measured or derived value carried by this result.")]
    pub value: f64,
    #[cfg_attr(
        doc,
        doc = "Whether larger or smaller metric values represent improvement."
    )]
    pub direction: Direction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(doc, doc = "Optional cost value; `None` means it is not available.")]
    pub cost: Option<Cost>,
    #[cfg_attr(doc, doc = "Benchmark valid recorded for this score record.")]
    pub benchmark_valid: bool,
    /// sha256 of the receipt/report the value was read from.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub receipt_sha256: String,
    #[cfg_attr(doc, doc = "Identifier for the associated run.")]
    pub run_id: String,
    #[cfg_attr(doc, doc = "Git sha recorded for this score record.")]
    pub git_sha: String,
    #[cfg_attr(doc, doc = "Unix timestamp when this event was recorded.")]
    pub ts_unix: u64,
    #[cfg_attr(doc, doc = "Provenance source for this value.")]
    pub source: ScoreSource,
    /// sha256 of the canonical JSON of this record without `row_sha256`.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub row_sha256: String,
}

impl ScoreRecord {
    /// The row hash: canonical JSON of the record with `row_sha256` blanked.
    pub fn compute_row_sha256(&self) -> Result<String> {
        let mut v = serde_json::to_value(self)?;
        v["row_sha256"] = serde_json::Value::String(String::new());
        canonical_sha256(&v)
    }

    #[cfg_attr(
        doc,
        doc = "Validates this value against its invariants and returns an error when validation fails."
    )]
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCORE_SCHEMA, "schema is not {SCORE_SCHEMA}");
        ensure!(is_hex64(&self.cut_id), "cut_id is not 64 hex");
        ensure!(
            self.evaluator_sha256 == UNBOUND || is_hex64(&self.evaluator_sha256),
            "evaluator sha256 is neither unbound nor 64 hex"
        );
        ensure!(
            self.direction == self.metric.direction(),
            "direction disagrees with the metric table"
        );
        ensure!(self.value.is_finite(), "value is not finite");
        if let Some(cost) = &self.cost {
            cost.validate()?;
        }
        ensure!(
            is_hex64(&self.receipt_sha256),
            "receipt_sha256 is not 64 hex"
        );
        ensure!(
            self.row_sha256 == self.compute_row_sha256()?,
            "row_sha256 does not hash the row"
        );
        Ok(())
    }
}

/// A required-but-nullable 64-hex field (schemars drops the null when `required` is set).
fn nullable_hex64(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["string", "null"],
        "pattern": "^[0-9a-f]{64}$"
    })
}

/// The schemars derivation of [`HwmRecord`] (tests compare it with the committed file).
pub fn derived_hwm_schema() -> serde_json::Value {
    schemars::schema_for!(HwmRecord).to_value()
}

/// The schemars derivation of [`ScoreRecord`].
pub fn derived_score_schema() -> serde_json::Value {
    schemars::schema_for!(ScoreRecord).to_value()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> HwmIdentity {
        HwmIdentity {
            comparison: None,
            task: "regression".into(),
            dataset_hash: "4e9ab20dc7413158".into(),
            cut_id: "0".repeat(64),
            evaluator_sha256: "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393"
                .into(),
            vocabulary_sha256: "8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787"
                .into(),
            protocol: "adaptive_cut_heldout_teacher".into(),
        }
    }

    #[test]
    fn identity_hash_is_sha256_of_sorted_compact_json() {
        // Fixed vector: the six keys in bytewise order, compact, no whitespace.
        let expected_text = format!(
            r#"{{"cut_id":"{}","dataset_hash":"4e9ab20dc7413158","evaluator_sha256":"b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393","protocol":"adaptive_cut_heldout_teacher","task":"regression","vocabulary_sha256":"8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787"}}"#,
            "0".repeat(64)
        );
        assert_eq!(
            String::from_utf8(crate::canon::canonical_json(&identity()).unwrap()).unwrap(),
            expected_text
        );
        assert_eq!(
            identity().sha256().unwrap(),
            crate::canon::sha256_hex(expected_text.as_bytes())
        );
        // Independently computed vector for a fully literal identity.
        let literal = HwmIdentity {
            comparison: None,
            task: "binary".into(),
            dataset_hash: "0259e0b6af043f58".into(),
            cut_id: "a".repeat(64),
            evaluator_sha256: UNBOUND.into(),
            vocabulary_sha256: UNBOUND.into(),
            protocol: "genebank-gen1".into(),
        };
        assert_eq!(
            literal.sha256().unwrap(),
            crate::canon::sha256_hex(
                format!(
                    r#"{{"cut_id":"{}","dataset_hash":"0259e0b6af043f58","evaluator_sha256":"unbound","protocol":"genebank-gen1","task":"binary","vocabulary_sha256":"unbound"}}"#,
                    "a".repeat(64)
                )
                .as_bytes()
            )
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut v = serde_json::to_value(identity()).unwrap();
        v["extra"] = 1.into();
        assert!(serde_json::from_value::<HwmIdentity>(v).is_err());
    }
}
