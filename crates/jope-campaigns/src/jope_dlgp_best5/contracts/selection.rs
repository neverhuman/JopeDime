use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedChoiceV1 {
    pub master_seed: u64,
    pub search_result_sha256: String,
    pub winner_semantic_sha256: String,
    pub winner_predictor: FrozenPredictorV1,
    pub pooled_oof_r2: f64,
    pub fold_r2_sd: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub complexity: u64,
}

impl SeedChoiceV1 {
    pub fn robust_score(&self) -> f64 {
        self.pooled_oof_r2 - 0.25 * self.fold_r2_sd
    }

    pub fn validate(&self) -> Result<(), String> {
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("seed choice must use seed 101 or 202".into());
        }
        validate_sha256("search result", &self.search_result_sha256)?;
        validate_sha256("seed winner", &self.winner_semantic_sha256)?;
        self.winner_predictor.validate()?;
        if self.winner_predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("seed winner semantic does not identify its frozen predictor".into());
        }
        for (name, value) in [
            ("pooled OOF R2", self.pooled_oof_r2),
            ("fold R2 SD", self.fold_r2_sd),
            ("RMSE", self.rmse),
            ("runtime", self.runtime_seconds),
        ] {
            finite(name, value)?;
        }
        if self.fold_r2_sd < 0.0 || self.rmse < 0.0 || self.runtime_seconds < 0.0 {
            return Err("seed selection metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionFreezeV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub choices: Vec<SeedChoiceV1>,
    pub selected_seed: u64,
    pub winner_semantic_sha256: String,
    pub winner_predictor: FrozenPredictorV1,
}

impl SelectionFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SelectionFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("selection campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("selection dataset", &self.dataset_sha256)?;
        validate_sha256("selection winner", &self.winner_semantic_sha256)?;
        self.winner_predictor.validate()?;
        if self.winner_predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("selection winner semantic does not identify its predictor".into());
        }
        if self.choices.len() != 2 {
            return Err("selection freeze requires both search seeds".into());
        }
        for choice in &self.choices {
            choice.validate()?;
        }
        if self
            .choices
            .iter()
            .map(|choice| choice.master_seed)
            .collect::<BTreeSet<_>>()
            != SEARCH_SEEDS.into_iter().collect()
            || self
                .choices
                .iter()
                .map(|choice| choice.master_seed)
                .collect::<Vec<_>>()
                != SEARCH_SEEDS
        {
            return Err("selection choices must cover seeds 101 and 202 in canonical order".into());
        }
        let best = self
            .choices
            .iter()
            .min_by(|left, right| {
                right
                    .robust_score()
                    .total_cmp(&left.robust_score())
                    .then_with(|| left.rmse.total_cmp(&right.rmse))
                    .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                    .then_with(|| left.complexity.cmp(&right.complexity))
                    .then_with(|| {
                        left.winner_semantic_sha256
                            .cmp(&right.winner_semantic_sha256)
                    })
            })
            .unwrap();
        if best.master_seed != self.selected_seed
            || best.winner_semantic_sha256 != self.winner_semantic_sha256
            || best.winner_predictor != self.winner_predictor
        {
            return Err("selection does not follow the frozen seed ordering".into());
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchStagingAttestationV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub proposal_batch_sha256: String,
    pub dataset_sha256: String,
    pub training: FrozenFileV1,
    pub metadata: FrozenFileV1,
    pub test_artifacts_staged: bool,
}

impl SearchStagingAttestationV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchStagingAttestation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("staged campaign", self.campaign_freeze_sha256.as_str()),
            ("staged proposals", self.proposal_batch_sha256.as_str()),
            ("staged dataset", self.dataset_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        self.training.validate("staged training")?;
        self.metadata.validate("staged metadata")?;
        if self.test_artifacts_staged {
            return Err("search staging must not contain test files or hashes".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBundleV1 {
    pub schema: String,
    pub staging: SearchStagingAttestationV1,
    pub target_rows: Vec<TargetRowV1>,
    pub split_plan: SearchSplitPlanV1,
    pub cv7_plan: Cv7PlanV1,
    pub anchor_receipt_sets: Vec<AnchorReceiptSetV1>,
    pub prediction_receipts: Vec<PredictionReceiptV1>,
    pub result: SearchResultV1,
}

impl SearchBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchBundle.v1";

    pub fn normalize_completion_order(&mut self) {
        self.result.normalize_completion_order();
        self.anchor_receipt_sets
            .sort_by_key(|anchors| anchors.split_index);
        self.prediction_receipts.sort_by(|left, right| {
            left.logical_operation_ordinal
                .cmp(&right.logical_operation_ordinal)
                .then_with(|| left.key.identity().ok().cmp(&right.key.identity().ok()))
        });
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.staging.validate()?;
        self.split_plan.validate()?;
        self.cv7_plan.validate()?;
        self.result.validate()?;
        if self.target_rows.is_empty() {
            return Err("search bundle lacks staged training targets".into());
        }
        let mut row_ids = BTreeSet::new();
        for row in &self.target_rows {
            row.validate()?;
            if !row_ids.insert(row.stable_row_id.as_str()) {
                return Err("search bundle contains duplicate training row IDs".into());
            }
        }
        if self.staging.campaign_freeze_sha256 != self.result.campaign_freeze_sha256
            || self.staging.proposal_batch_sha256 != self.result.proposal_freeze_sha256
            || self.staging.dataset_sha256 != self.result.dataset_sha256
            || self.staging.training.sha256 != self.split_plan.training_sha256
            || self.split_plan.dataset_sha256 != self.result.dataset_sha256
            || self.split_plan.master_seed != self.result.master_seed
            || self.split_plan.identity()? != self.result.split_plan_sha256
            || self.cv7_plan.identity()? != self.result.cv7_plan_sha256
        {
            return Err("search bundle bindings disagree".into());
        }
        if self.prediction_receipts.len() != self.result.ledger.logical_operations as usize {
            return Err("search bundle lacks one receipt per charged logical operation".into());
        }
        if self.anchor_receipt_sets.len() != SEARCH_SPLITS
            || self
                .anchor_receipt_sets
                .iter()
                .enumerate()
                .any(|(index, anchors)| {
                    anchors.split_index != index as u32 || anchors.validate().is_err()
                })
        {
            return Err("search bundle requires ten anchor receipts on every search split".into());
        }
        let mut identities = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        for receipt in &self.prediction_receipts {
            receipt.validate()?;
            if receipt.key.dataset_sha256 != self.result.dataset_sha256
                || receipt.key.training_sha256 != self.staging.training.sha256
                || !identities.insert(receipt.identity()?)
                || !ordinals.insert(receipt.logical_operation_ordinal)
            {
                return Err("search prediction receipts are duplicate or foreign".into());
            }
        }
        if ordinals != (1..=self.result.ledger.logical_operations).collect::<BTreeSet<_>>() {
            return Err("search logical-operation receipt ordinals are not contiguous".into());
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonScoreV1 {
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub rmse: f64,
    pub latency_seconds: f64,
    pub finalist_validity: f64,
    pub overall_validity: f64,
    pub strict_benchmark_win: bool,
    pub preserved_existing_win: bool,
    pub predicted_cpu_seconds: f64,
}

impl ComparisonScoreV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("comparison dataset", &self.dataset_sha256)?;
        for (name, value) in [
            ("RMSE", self.rmse),
            ("latency", self.latency_seconds),
            ("finalist validity", self.finalist_validity),
            ("overall validity", self.overall_validity),
            ("predicted CPU", self.predicted_cpu_seconds),
        ] {
            finite(name, value)?;
        }
        if self.rmse <= 0.0
            || self.latency_seconds < 0.0
            || self.predicted_cpu_seconds < 0.0
            || !(0.0..=1.0).contains(&self.finalist_validity)
            || !(0.0..=1.0).contains(&self.overall_validity)
        {
            return Err("comparison metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmComparisonV1 {
    pub arm: Best5ArmV1,
    pub mean_delta: f64,
    pub median_delta: f64,
    pub one_sided_95_lower_bound: f64,
    pub paired_wins: u32,
    pub strict_benchmark_wins: u32,
    pub predicted_cpu_seconds: f64,
    pub advances: bool,
    pub failure_codes: Vec<FailureCodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCodeV1 {
    MalformedProposal,
    UnsupportedHead,
    MissingCheckpoint,
    CheckpointHashMismatch,
    CandidateTimeout,
    StaleEvidence,
    OutputCollision,
    CasCorruption,
    IncompleteSplitCoverage,
    InvalidCv7,
    TestAccessBeforeSelection,
    NonpositiveLowerBound,
    NegativeMedian,
    IncompleteSealedResults,
    FinalistValidity,
    OverallValidity,
    ExistingBenchmarkWinLost,
    LatencyGate,
    ConfirmationPairedWins,
    ConfirmationWinPreservation,
    LaunchGateBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignDecisionV1 {
    PilotWinner { arm: Best5ArmV1 },
    Adopt { arm: Best5ArmV1 },
    NoImprovement { failure_codes: Vec<FailureCodeV1> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub stage: DatasetStageV1,
    pub bootstrap_seed: u64,
    pub bootstrap_repetitions: u32,
    pub dataset_count: u32,
    pub arms: Vec<ArmComparisonV1>,
    pub decision: CampaignDecisionV1,
}

impl ComparisonV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5Comparison.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("comparison campaign", &self.campaign_freeze_sha256)?;
        let expected_count = match self.stage {
            DatasetStageV1::Pilot => PILOT_DATASETS,
            DatasetStageV1::Confirmation => CONFIRMATION_DATASETS,
        };
        if self.bootstrap_seed != BOOTSTRAP_SEED
            || self.bootstrap_repetitions != 10_000
            || self.dataset_count != expected_count as u32
        {
            return Err("comparison bootstrap or dataset cardinality is not frozen".into());
        }
        let expected_arms: BTreeSet<_> = match self.stage {
            DatasetStageV1::Pilot => Best5ArmV1::EXPERIMENTS.into_iter().collect(),
            DatasetStageV1::Confirmation => self.arms.iter().map(|arm| arm.arm).collect(),
        };
        if self.arms.iter().map(|arm| arm.arm).collect::<BTreeSet<_>>() != expected_arms
            || (self.stage == DatasetStageV1::Pilot && self.arms.len() != 5)
            || (self.stage == DatasetStageV1::Confirmation && self.arms.len() != 1)
        {
            return Err("comparison contains the wrong arm set".into());
        }
        for arm in &self.arms {
            for value in [
                arm.mean_delta,
                arm.median_delta,
                arm.one_sided_95_lower_bound,
                arm.predicted_cpu_seconds,
            ] {
                finite("arm comparison", value)?;
            }
            if !arm.failure_codes.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err("comparison failure codes must be sorted and unique".into());
            }
        }
        match self.decision {
            CampaignDecisionV1::PilotWinner { arm } if self.stage == DatasetStageV1::Pilot => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("pilot winner did not clear every gate".into());
                }
            }
            CampaignDecisionV1::Adopt { arm } if self.stage == DatasetStageV1::Confirmation => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("adopted arm did not clear confirmation".into());
                }
            }
            CampaignDecisionV1::NoImprovement { ref failure_codes }
                if !failure_codes.is_empty()
                    && failure_codes.windows(2).all(|pair| pair[0] < pair[1]) => {}
            _ => return Err("comparison decision does not match its stage or gate state".into()),
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
pub struct LaunchEvidenceV1 {
    pub schema: String,
    pub scq_doctor_success: bool,
    pub evaluator_evidence_age_seconds: u64,
    pub supply_evidence_age_seconds: u64,
    pub max_evidence_age_seconds: u64,
    pub reviewed_action_acknowledged: bool,
    pub observed_project_priority_weight: u32,
    pub eligible_worker_disk_free_fractions: Vec<f64>,
}

impl LaunchEvidenceV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5LaunchEvidence.v1";

    pub fn launch_allowed(&self) -> Result<bool, String> {
        self.launch_allowed_for(&ScqPolicyV1::frozen())
    }

    pub fn launch_allowed_for(&self, policy: &ScqPolicyV1) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        policy.validate()?;
        if self.max_evidence_age_seconds == 0 || self.eligible_worker_disk_free_fractions.is_empty()
        {
            return Err("launch evidence lacks freshness or eligible workers".into());
        }
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self.scq_doctor_success
            && self.evaluator_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.supply_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.reviewed_action_acknowledged
            && self.observed_project_priority_weight == policy.priority_weight
            && self
                .eligible_worker_disk_free_fractions
                .iter()
                .all(|fraction| {
                    *fraction >= policy.min_worker_disk_free_fraction_micros as f64 / 1_000_000.0
                }))
    }

    pub fn disk_warning(&self) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| *fraction < 0.05))
    }
}
