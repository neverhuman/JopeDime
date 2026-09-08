use serde::{Deserialize, Serialize};

use super::identity::{canonical_sha256, validate_schema, validate_sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridMatchedRunV1 {
    pub dataset_sha256: String,
    pub context_sha256: String,
    pub seed: u64,
    pub slot: u32,
    pub evaluator_sha256: String,
    pub ancestry_sha256: String,
    pub compute_units: u64,
    pub score: f64,
}

impl HybridMatchedRunV1 {
    fn validate(&self) -> Result<(), String> {
        for (name, digest) in [
            ("dataset_sha256", self.dataset_sha256.as_str()),
            ("context_sha256", self.context_sha256.as_str()),
            ("evaluator_sha256", self.evaluator_sha256.as_str()),
            ("ancestry_sha256", self.ancestry_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.compute_units == 0 || !self.score.is_finite() {
            return Err("hybrid matched run requires positive compute and a finite score".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridContributionReceiptV1 {
    pub schema: String,
    pub gp_or_dl_baseline: HybridMatchedRunV1,
    pub hybrid_candidate: HybridMatchedRunV1,
    pub incremental_gain: f64,
}

impl HybridContributionReceiptV1 {
    pub const SCHEMA: &'static str = "HybridContributionReceipt.v1";

    pub fn new(
        gp_or_dl_baseline: HybridMatchedRunV1,
        hybrid_candidate: HybridMatchedRunV1,
    ) -> Result<Self, String> {
        let receipt = Self {
            schema: Self::SCHEMA.into(),
            incremental_gain: hybrid_candidate.score - gp_or_dl_baseline.score,
            gp_or_dl_baseline,
            hybrid_candidate,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        self.gp_or_dl_baseline.validate()?;
        self.hybrid_candidate.validate()?;
        let left = &self.gp_or_dl_baseline;
        let right = &self.hybrid_candidate;
        if left.dataset_sha256 != right.dataset_sha256
            || left.context_sha256 != right.context_sha256
            || left.seed != right.seed
            || left.slot != right.slot
            || left.evaluator_sha256 != right.evaluator_sha256
            || left.ancestry_sha256 != right.ancestry_sha256
            || left.compute_units != right.compute_units
        {
            return Err(
                "hybrid contribution runs are not matched on every frozen dimension".into(),
            );
        }
        let expected = right.score - left.score;
        if !self.incremental_gain.is_finite()
            || self.incremental_gain.to_bits() != expected.to_bits()
        {
            return Err("hybrid incremental gain does not match the paired scores".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
