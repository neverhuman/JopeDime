use super::*;

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
