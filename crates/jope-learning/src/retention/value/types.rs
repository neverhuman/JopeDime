use super::*;

pub struct ValueWeights {
    /// Influence recorded for this value weights.
    pub influence: f64,
    /// Novelty recorded for this value weights.
    pub novelty: f64,
    /// Uncertainty recorded for this value weights.
    pub uncertainty: f64,
    /// Reproduction cost recorded for this value weights.
    pub reproduction_cost: f64,
    /// Storage cost recorded for this value weights.
    pub storage_cost: f64,
}

impl ValueWeights {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.influence.is_finite() && self.influence >= 0.0,
            "influence weight must be finite and nonnegative"
        );
        for (name, value) in [
            ("novelty", self.novelty),
            ("uncertainty", self.uncertainty),
            ("reproduction_cost", self.reproduction_cost),
            ("storage_cost", self.storage_cost),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} weight must be finite and nonnegative"
            );
        }
        Ok(())
    }
}

/// Versioned view of the validated v1 retention knobs plus explicit value weights.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetentionPolicyV2 {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// K ctx recorded for this retention policy v2.
    pub k_ctx: u64,
    /// K lineage recorded for this retention policy v2.
    pub k_lineage: u64,
    /// Reservoir rate recorded for this retention policy v2.
    pub reservoir_rate: f64,
    /// Margin rel recorded for this retention policy v2.
    pub margin_rel: f64,
    /// Grace days recorded for this retention policy v2.
    pub grace_days: u64,
    /// Weights recorded for this retention policy v2.
    pub weights: ValueWeights,
}

impl RetentionPolicyV2 {
    /// Lift only the validated dormant v1 knobs; weights remain caller-supplied and explicit.
    pub fn from_v1(policy: &Policy, weights: ValueWeights) -> Result<Self> {
        let v2 = Self {
            schema: VALUE_POLICY_V2_SCHEMA.into(),
            k_ctx: policy.k_ctx,
            k_lineage: policy.k_lineage,
            reservoir_rate: policy.reservoir_rate,
            margin_rel: policy.margin_rel,
            grace_days: policy.grace_days,
            weights,
        };
        v2.validate()?;
        Ok(v2)
    }

    /// Parses the supplied representation, returning an error when the input is invalid.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value: Self = serde_json::from_slice(bytes)?;
        value.validate()?;
        Ok(value)
    }

    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == VALUE_POLICY_V2_SCHEMA,
            "unsupported retention value policy schema"
        );
        ensure!(
            self.k_ctx > 0 && self.k_lineage > 0,
            "keeper limits must be positive"
        );
        ensure!(
            self.reservoir_rate.is_finite()
                && self.reservoir_rate > 0.0
                && self.reservoir_rate <= 1.0,
            "reservoir_rate must be in (0, 1]"
        );
        ensure!(
            self.margin_rel.is_finite() && self.margin_rel >= 0.0,
            "margin_rel must be finite and nonnegative"
        );
        ensure!(self.grace_days > 0, "grace_days must be positive");
        self.weights.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
/// `PhysicalObject` is a record for the physical object contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct PhysicalObject {
    /// Bucket recorded for this physical object.
    pub bucket: String,
    /// Key recorded for this physical object.
    pub key: String,
    /// Identifier for the associated version.
    pub version_id: String,
    /// Sha-256 recorded for this physical object.
    pub sha256: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
}

impl PhysicalObject {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.bucket.is_empty() && !self.key.is_empty() && !self.version_id.is_empty(),
            "physical object requires bucket, key, and version"
        );
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "physical object {} has invalid sha256",
            self.key
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `LearningSignals` is a record for the learning signals contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct LearningSignals {
    /// Measured leave-group-out change in the declared value-model metric.
    pub influence: f64,
    /// Novelty recorded for this learning signals.
    pub novelty: f64,
    /// Uncertainty recorded for this learning signals.
    pub uncertainty: f64,
    /// Reproduction cost seconds recorded for this learning signals.
    pub reproduction_cost_seconds: f64,
    /// Reproduction probability recorded for this learning signals.
    pub reproduction_probability: f64,
    /// Storage usd per month recorded for this learning signals.
    pub storage_usd_per_month: f64,
    /// Residual surprise is shown for review ordering only.
    pub surprise: f64,
}

impl LearningSignals {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(self.influence.is_finite(), "influence must be finite");
        for (name, value) in [
            ("novelty", self.novelty),
            ("uncertainty", self.uncertainty),
            ("reproduction_cost_seconds", self.reproduction_cost_seconds),
            ("reproduction_probability", self.reproduction_probability),
            ("storage_usd_per_month", self.storage_usd_per_month),
            ("surprise", self.surprise),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and nonnegative"
            );
        }
        ensure!(self.novelty <= 1.0, "novelty must be in [0, 1]");
        ensure!(
            self.reproduction_probability <= 1.0,
            "reproduction_probability must be in [0, 1]"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `RetentionValue` is a record for the retention value contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct RetentionValue {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Object recorded for this retention value.
    pub object: PhysicalObject,
    /// Pinned recorded for this retention value.
    pub pinned: bool,
    /// Score rows always kept recorded for this retention value.
    pub score_rows_always_kept: bool,
    /// Optional marginal value value; `None` means it is not available.
    pub marginal_value: Option<f64>,
    /// Optional surprise value; `None` means it is not available.
    pub surprise: Option<f64>,
    /// Score basis recorded for this retention value.
    pub score_basis: String,
}

impl RetentionValue {
    /// Performs the classify operation and returns an error if it cannot be completed.
    pub fn classify(
        object: &PhysicalObject,
        pinned: bool,
        score_row: bool,
        signals: Option<&LearningSignals>,
        policy: &RetentionPolicyV2,
    ) -> Result<Self> {
        let (marginal_value, surprise, score_basis) = match signals {
            Some(s) => {
                s.validate()?;
                let w = &policy.weights;
                let value = w.influence * s.influence
                    + w.novelty * s.novelty
                    + w.uncertainty * s.uncertainty
                    + w.reproduction_cost
                        * s.reproduction_cost_seconds
                        * s.reproduction_probability
                    - w.storage_cost * s.storage_usd_per_month;
                ensure!(value.is_finite(), "marginal value is not finite");
                (Some(value), Some(s.surprise), "measured-signals-v1".into())
            }
            None => (None, None, "unknown-no-measured-signals".into()),
        };
        Ok(Self {
            schema: "jope.retention-value/v2".into(),
            object: object.clone(),
            pinned: pinned || score_row,
            score_rows_always_kept: score_row,
            marginal_value,
            surprise,
            score_basis,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `ConsumerClosure` is a record for the consumer closure contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct ConsumerClosure {
    /// Scanned at unix recorded for this consumer closure.
    pub scanned_at_unix: u64,
    /// Observed root generation recorded for this consumer closure.
    pub observed_root_generation: String,
    /// Current root generation recorded for this consumer closure.
    pub current_root_generation: String,
    /// Collection of active consumers values carried by this consumer closure.
    pub active_consumers: Vec<String>,
    /// Collection of unresolved values carried by this consumer closure.
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
/// `ReplacementProof` is an enum for the replacement proof contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub enum ReplacementProof {
    /// The `ExactDuplicate` case of `ReplacementProof`.
    ExactDuplicate {
        /// Replacement recorded for this replacement proof.
        replacement: PhysicalObject,
        /// Content hash verified recorded for this replacement proof.
        content_hash_verified: bool,
        /// Restore tested recorded for this replacement proof.
        restore_tested: bool,
    },
    /// The `Pack` case of `ReplacementProof`.
    Pack {
        /// Replacement recorded for this replacement proof.
        replacement: PhysicalObject,
        /// Collection of members values carried by this replacement proof.
        members: Vec<PhysicalObject>,
        /// Manifest verified recorded for this replacement proof.
        manifest_verified: bool,
        /// Restore tested recorded for this replacement proof.
        restore_tested: bool,
    },
    /// The `Supersession` case of `ReplacementProof`.
    Supersession {
        /// Replacement recorded for this replacement proof.
        replacement: PhysicalObject,
        /// Qualified champion recorded for this replacement proof.
        qualified_champion: bool,
        /// Relative margin recorded for this replacement proof.
        relative_margin: f64,
        /// Days since dominance recorded for this replacement proof.
        days_since_dominance: u64,
        /// Score rows retained recorded for this replacement proof.
        score_rows_retained: bool,
        /// Recovery closure verified recorded for this replacement proof.
        recovery_closure_verified: bool,
        /// Restore tested recorded for this replacement proof.
        restore_tested: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `Candidate` is a record for the candidate contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct Candidate {
    /// Collection of sources values carried by this candidate.
    pub sources: Vec<PhysicalObject>,
    /// Hard t0 recorded for this candidate.
    pub hard_t0: bool,
    /// Active pin recorded for this candidate.
    pub active_pin: bool,
    /// Active promise recorded for this candidate.
    pub active_promise: bool,
    /// Score row recorded for this candidate.
    pub score_row: bool,
    /// Optional inventory at unix value; `None` means it is not available.
    pub inventory_at_unix: Option<u64>,
    /// Optional consumer closure value; `None` means it is not available.
    pub consumer_closure: Option<ConsumerClosure>,
    /// Optional replacement value; `None` means it is not available.
    pub replacement: Option<ReplacementProof>,
    /// Optional signals value; `None` means it is not available.
    pub signals: Option<LearningSignals>,
    /// Optional context rank value; `None` means it is not available.
    pub context_rank: Option<u64>,
    /// Optional lineage rank value; `None` means it is not available.
    pub lineage_rank: Option<u64>,
    /// Optional reservoir selected value; `None` means it is not available.
    pub reservoir_selected: Option<bool>,
    /// Ranking current recorded for this candidate.
    pub ranking_current: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// `Recommendation` is an enum for the recommendation contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub enum Recommendation {
    /// The `KeepT0` case of `Recommendation`.
    KeepT0,
    /// The `KeepScoreRows` case of `Recommendation`.
    KeepScoreRows,
    /// The `Keep` case of `Recommendation`.
    Keep,
    /// The `Hold` case of `Recommendation`.
    Hold,
    /// The `ReviewDuplicate` case of `Recommendation`.
    ReviewDuplicate,
    /// The `ReviewPack` case of `Recommendation`.
    ReviewPack,
    /// The `ReviewSupersession` case of `Recommendation`.
    ReviewSupersession,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `PlanItem` is a record for the plan item contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct PlanItem {
    /// Provenance source for this value.
    pub source: Vec<PhysicalObject>,
    /// Optional replacement value; `None` means it is not available.
    pub replacement: Option<PhysicalObject>,
    /// Recommendation recorded for this plan item.
    pub recommendation: Recommendation,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
    /// Collection of unresolved values carried by this plan item.
    pub unresolved: Vec<String>,
    /// Collection of values values carried by this plan item.
    pub values: Vec<RetentionValue>,
    /// Byte count for the associated current.
    pub current_bytes: u64,
    /// Hypothetical bytes after review recorded for this plan item.
    pub hypothetical_bytes_after_review: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// `RetentionPlanV2` is a record for the retention plan v2 contract in this module: Read-only value and packing review plans. This module returns data in memory and has no storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.
pub struct RetentionPlanV2 {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// As of unix recorded for this retention plan v2.
    pub as_of_unix: u64,
    /// Inventory max age secs recorded for this retention plan v2.
    pub inventory_max_age_secs: u64,
    /// SHA-256 digest identifying the associated source inventory.
    pub source_inventory_sha256: String,
    /// Policy recorded for this retention plan v2.
    pub policy: RetentionPolicyV2,
    /// Collection of items values carried by this retention plan v2.
    pub items: Vec<PlanItem>,
    /// Current objects recorded for this retention plan v2.
    pub current_objects: u64,
    /// Byte count for the associated current.
    pub current_bytes: u64,
    /// Hypothetical objects after review recorded for this retention plan v2.
    pub hypothetical_objects_after_review: u64,
    /// Hypothetical bytes after review recorded for this retention plan v2.
    pub hypothetical_bytes_after_review: u64,
    /// Hypothetical bytes avoided recorded for this retention plan v2.
    pub hypothetical_bytes_avoided: u64,
    /// Net bytes delta recorded for this retention plan v2.
    pub net_bytes_delta: i128,
    /// SHA-256 digest identifying the associated plan.
    pub plan_sha256: String,
}
