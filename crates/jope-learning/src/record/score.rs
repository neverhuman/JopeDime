use super::*;

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
