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
