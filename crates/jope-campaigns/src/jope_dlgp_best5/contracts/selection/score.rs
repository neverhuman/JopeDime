use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedScoreV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub selection_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub winner_semantic_sha256: String,
    pub predictor: FrozenPredictorV1,
    pub test_access_ordinal: u32,
    pub r2: f64,
    pub rmse: f64,
    pub latency_seconds: f64,
    pub validity: f64,
    pub ancestry: AncestryV1,
}

impl SealedScoreV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedScore.v1";

    pub fn validate(&self, selection: &SelectionFreezeV1) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("sealed campaign", self.campaign_freeze_sha256.as_str()),
            ("sealed selection", self.selection_freeze_sha256.as_str()),
            ("sealed dataset", self.dataset_sha256.as_str()),
            ("sealed test", self.test_sha256.as_str()),
            ("sealed winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.selection_freeze_sha256 != selection.identity()?
            || self.campaign_freeze_sha256 != selection.campaign_freeze_sha256
            || self.arm != selection.arm
            || self.dataset_sha256 != selection.dataset_sha256
            || self.winner_semantic_sha256 != selection.winner_semantic_sha256
            || self.predictor != selection.winner_predictor
        {
            return Err("sealed score does not bind the selection freeze".into());
        }
        if self.test_access_ordinal != 1 {
            return Err("sealed test may be evaluated exactly once".into());
        }
        self.predictor.validate()?;
        if self.predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("sealed predictor semantic binding changed".into());
        }
        for (name, value) in [
            ("sealed R2", self.r2),
            ("sealed RMSE", self.rmse),
            ("sealed latency", self.latency_seconds),
            ("sealed validity", self.validity),
        ] {
            finite(name, value)?;
        }
        if self.rmse < 0.0 || self.latency_seconds < 0.0 || !(0.0..=1.0).contains(&self.validity) {
            return Err("sealed metrics are outside their domains".into());
        }
        self.ancestry.validate()
    }

    pub fn identity(&self, selection: &SelectionFreezeV1) -> Result<String, String> {
        self.validate(selection)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionReceiptV1 {
    pub schema: String,
    pub key: PredictionKeyV1,
    pub prediction_object_sha256: String,
    pub logical_operation_ordinal: u32,
}

impl PredictionReceiptV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5PredictionReceipt.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.key.validate()?;
        validate_sha256("prediction object", &self.prediction_object_sha256)?;
        if self.logical_operation_ordinal == 0 {
            return Err("prediction receipt must charge a logical operation".into());
        }
        Ok(())
    }

    pub fn validate_object(&self, prediction: &PredictionObjectV1) -> Result<(), String> {
        self.validate()?;
        prediction.validate()?;
        if canonical_sha256(prediction)? != self.prediction_object_sha256
            || prediction.training_view_sha256 != self.key.training_sha256
            || prediction.fold_plan_sha256 != self.key.split_sha256
            || prediction.candidate_semantic_sha256 != self.key.semantic_sha256
            || prediction.evaluator_sha256 != self.key.evaluator_sha256
            || prediction.math_policy_sha256 != self.key.math_policy_sha256
            || prediction.backend_policy_sha256 != self.key.backend_policy_sha256
        {
            return Err("prediction receipt does not bind its content-addressed object".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorReceiptSetV1 {
    pub split_index: u32,
    pub prediction_receipt_sha256s: Vec<String>,
    pub median_r2_clipped: f64,
}

impl AnchorReceiptSetV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.prediction_receipt_sha256s.len() != 10
            || self
                .prediction_receipt_sha256s
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != 10
            || !self.median_r2_clipped.is_finite()
            || !(-1.0..=1.0).contains(&self.median_r2_clipped)
        {
            return Err("anchor receipt set is incomplete or invalid".into());
        }
        for receipt in &self.prediction_receipt_sha256s {
            validate_sha256("anchor prediction receipt", receipt)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedClaimV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub selection_freeze_sha256s: Vec<String>,
    pub access_ordinal: u32,
}

impl SealedClaimV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedClaim.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed claim campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("sealed claim dataset", &self.dataset_sha256)?;
        if self.access_ordinal != 1
            || self.selection_freeze_sha256s.len() < 2
            || self.selection_freeze_sha256s.len() > Best5ArmV1::ALL.len()
            || !self
                .selection_freeze_sha256s
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err("sealed claim requires one canonical dataset-level selection set".into());
        }
        for selection in &self.selection_freeze_sha256s {
            validate_sha256("sealed claim selection", selection)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedStagingAttestationV1 {
    pub schema: String,
    pub claim_sha256: String,
    pub training: FrozenFileV1,
    pub test: FrozenFileV1,
}

impl SealedStagingAttestationV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedStagingAttestation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed staged claim", &self.claim_sha256)?;
        self.training.validate("sealed training")?;
        self.test.validate("sealed test")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedBundleV1 {
    pub schema: String,
    pub claim: SealedClaimV1,
    pub staging: SealedStagingAttestationV1,
    pub selections: Vec<SelectionFreezeV1>,
    pub prediction_receipts: Vec<PredictionReceiptV1>,
    pub scores: Vec<SealedScoreV1>,
}

impl SealedBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.claim.validate()?;
        self.staging.validate()?;
        if self.staging.claim_sha256 != self.claim.identity()?
            || self.selections.len() != self.claim.selection_freeze_sha256s.len()
            || self.scores.len() != self.selections.len()
            || self.prediction_receipts.len() != self.selections.len()
        {
            return Err("sealed bundle has incomplete dataset-level coverage".into());
        }
        let mut selection_ids = Vec::new();
        let mut arms = BTreeSet::new();
        for selection in &self.selections {
            selection.validate()?;
            if selection.campaign_freeze_sha256 != self.claim.campaign_freeze_sha256
                || selection.dataset_sha256 != self.claim.dataset_sha256
                || !arms.insert(selection.arm)
            {
                return Err("sealed selection set contains a foreign or duplicate arm".into());
            }
            selection_ids.push(selection.identity()?);
        }
        selection_ids.sort();
        if selection_ids != self.claim.selection_freeze_sha256s {
            return Err("sealed claim does not identify the supplied selection set".into());
        }
        for score in &self.scores {
            let selection = self
                .selections
                .iter()
                .find(|selection| selection.arm == score.arm)
                .ok_or_else(|| "sealed score has no matching selection".to_string())?;
            score.validate(selection)?;
            if score.test_sha256 != self.staging.test.sha256 {
                return Err("sealed score test hash differs from staged bytes".into());
            }
        }
        for receipt in &self.prediction_receipts {
            receipt.validate()?;
            if receipt.key.dataset_sha256 != self.claim.dataset_sha256
                || receipt.key.training_sha256 != self.staging.training.sha256
            {
                return Err("sealed prediction receipt is foreign".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedDatasetReceiptV1 {
    pub schema: String,
    pub sealed_bundle_sha256: String,
    pub claim_sha256: String,
    pub score_sha256s: Vec<String>,
}

impl SealedDatasetReceiptV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedDatasetReceipt.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed bundle", &self.sealed_bundle_sha256)?;
        validate_sha256("sealed dataset claim", &self.claim_sha256)?;
        if self.score_sha256s.len() < 2
            || !self.score_sha256s.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err("sealed dataset receipt requires sorted unique score artifacts".into());
        }
        for score in &self.score_sha256s {
            validate_sha256("sealed score artifact", score)?;
        }
        Ok(())
    }
}
