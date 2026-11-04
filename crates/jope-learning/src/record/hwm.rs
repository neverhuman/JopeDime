use super::*;

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

/// A required-but-nullable 64-hex field (schemars drops the null when `required` is set).
pub(crate) fn nullable_hex64(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({
        "type": ["string", "null"],
        "pattern": "^[0-9a-f]{64}$"
    })
}
