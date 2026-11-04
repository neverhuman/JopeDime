use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyScreenPurposeV3 {
    FixedBankDiagnostic,
    NestedGpStream,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyScreenConfigV3 {
    pub schema: String,
    pub campaign_id: String,
    pub purpose: StrategyScreenPurposeV3,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub feature_view: FeatureViewContractV2,
    pub grouped_fold_bank_sha256: String,
    pub candidate_bank_sha256: String,
    pub configuration_sha256: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub seed: u64,
    pub evaluation_budget: u32,
    pub candidate_timeout_seconds: u32,
    pub test_material_present: bool,
}

impl StrategyScreenConfigV3 {
    pub const SCHEMA: &'static str = "JopeStrategyScreenConfig.v3";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.test_material_present
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
        {
            return Err(
                "strategy-screen V3 authority, seed, timeout, or test boundary drifted".into(),
            );
        }
        self.feature_view.validate()?;
        if self.train_sha256 != self.feature_view.train_sha256
            || self.real_cell_sha256 != self.feature_view.real_cell_sha256
        {
            return Err(
                "strategy-screen V3 feature view is not bound to the same train and cell bytes"
                    .into(),
            );
        }
        for (name, value) in [
            ("strategy dataset", self.dataset_sha256.as_str()),
            ("strategy train", self.train_sha256.as_str()),
            ("strategy real cell", self.real_cell_sha256.as_str()),
            (
                "strategy grouped fold bank",
                self.grouped_fold_bank_sha256.as_str(),
            ),
            (
                "strategy candidate bank",
                self.candidate_bank_sha256.as_str(),
            ),
            ("strategy configuration", self.configuration_sha256.as_str()),
            ("strategy source tree", self.source_tree_sha256.as_str()),
            ("strategy binary", self.binary_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        match self.purpose {
            StrategyScreenPurposeV3::FixedBankDiagnostic
                if self.evaluation_budget != (FIXED_BANK_SIZE * OUTER_FOLDS) as u32 =>
            {
                Err("fixed-bank V3 budget must be exactly twelve genomes by seven folds".into())
            }
            StrategyScreenPurposeV3::NestedGpStream
                if self.evaluation_budget != SEARCH_EVALUATIONS_PER_STREAM =>
            {
                Err("nested GP V3 budget must be exactly 2,000 evaluations".into())
            }
            _ => Ok(()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyScreenResultV3 {
    pub schema: String,
    pub config_sha256: String,
    pub campaign_id: String,
    pub purpose: StrategyScreenPurposeV3,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub feature_view_identity_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub candidate_bank_sha256: String,
    pub configuration_sha256: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub seed: u64,
    pub evaluation_budget: u32,
    pub evaluations_completed: u32,
    pub pooled_oof_rmse: f64,
    pub complete: bool,
    pub timed_out_candidates: u32,
    pub test_material_present: bool,
}

impl StrategyScreenResultV3 {
    pub const SCHEMA: &'static str = "JopeStrategyScreenResult.v3";

    pub fn validate(&self, config: &StrategyScreenConfigV3) -> Result<(), String> {
        config.validate()?;
        if self.schema != Self::SCHEMA
            || self.config_sha256 != config.identity()?
            || self.campaign_id != config.campaign_id
            || self.purpose != config.purpose
            || self.dataset_sha256 != config.dataset_sha256
            || self.train_sha256 != config.train_sha256
            || self.real_cell_sha256 != config.real_cell_sha256
            || self.feature_view_identity_sha256 != config.feature_view.identity()?
            || self.grouped_fold_bank_sha256 != config.grouped_fold_bank_sha256
            || self.candidate_bank_sha256 != config.candidate_bank_sha256
            || self.configuration_sha256 != config.configuration_sha256
            || self.source_tree_sha256 != config.source_tree_sha256
            || self.binary_sha256 != config.binary_sha256
            || self.seed != config.seed
            || self.evaluation_budget != config.evaluation_budget
            || self.evaluations_completed != config.evaluation_budget
            || !self.complete
            || self.timed_out_candidates != 0
            || self.test_material_present
            || !self.pooled_oof_rmse.is_finite()
            || self.pooled_oof_rmse < 0.0
        {
            return Err("strategy-screen V3 result is incomplete or not internally bound".into());
        }
        Ok(())
    }

    pub fn identity(&self, config: &StrategyScreenConfigV3) -> Result<String, String> {
        self.validate(config)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldCellEncodingReceiptV2 {
    pub schema: String,
    pub seed: u64,
    pub fold_index: u32,
    pub outer_plan_sha256: String,
    pub train_sha256: String,
    pub encoder_identity_sha256: String,
    pub fit_rows_sha256: String,
    pub hold_rows_sha256: String,
    pub fit_cell_sha256: String,
    pub hold_cell_sha256: String,
    pub fit_row_count: usize,
    pub hold_row_count: usize,
    pub feature_width: usize,
    pub cuda_authenticated: bool,
    pub isolated_gpu: bool,
    pub dummy_hold_targets: bool,
    pub hold_labels_reached_encoder: bool,
    pub test_path_resolved: bool,
    pub finite_output: bool,
}

impl FoldCellEncodingReceiptV2 {
    pub const SCHEMA: &'static str = "HyperionFoldCellEncodingReceipt.v2";

    pub fn validate(
        &self,
        plan: &GroupedOuterPlanV2,
        train_sha256: &str,
        encoder_identity_sha256: &str,
        feature_width: usize,
    ) -> Result<(), String> {
        plan.validate()?;
        let fold = plan
            .folds
            .get(self.fold_index as usize)
            .ok_or_else(|| "fold cell receipt index is outside its outer plan".to_string())?;
        if self.schema != Self::SCHEMA
            || self.seed != plan.seed
            || self.outer_plan_sha256 != plan.identity()?
            || self.train_sha256 != train_sha256
            || self.encoder_identity_sha256 != encoder_identity_sha256
            || self.fit_rows_sha256 != canonical_sha256(&fold.fit_rows)?
            || self.hold_rows_sha256 != canonical_sha256(&fold.hold_rows)?
            || self.fit_row_count != fold.fit_rows.len()
            || self.hold_row_count != fold.hold_rows.len()
            || self.feature_width != feature_width
            || feature_width == 0
            || !self.cuda_authenticated
            || !self.isolated_gpu
            || !self.dummy_hold_targets
            || self.hold_labels_reached_encoder
            || self.test_path_resolved
            || !self.finite_output
        {
            return Err(
                "fold-local cell receipt failed CUDA, isolation, shape, or leakage proof".into(),
            );
        }
        for (name, value) in [
            ("fold train", self.train_sha256.as_str()),
            ("fold encoder", self.encoder_identity_sha256.as_str()),
            ("fold fit rows", self.fit_rows_sha256.as_str()),
            ("fold hold rows", self.hold_rows_sha256.as_str()),
            ("fold fit cell", self.fit_cell_sha256.as_str()),
            ("fold hold cell", self.hold_cell_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldLocalCellCoverageV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub encoder_identity_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub feature_width: usize,
    pub authorized_full_train_cell_sha256: String,
    pub fresh_full_train_cell_sha256: String,
    pub full_train_byte_equal: bool,
    pub receipts: Vec<FoldCellEncodingReceiptV2>,
    pub structurally_eligible: bool,
    pub failure_reason: Option<String>,
}

impl FoldLocalCellCoverageV2 {
    pub const SCHEMA: &'static str = "HyperionFoldLocalCellCoverage.v2";

    pub fn validate(&self, fold_bank: &GroupedFoldBankV2) -> Result<(), String> {
        fold_bank.validate()?;
        if self.schema != Self::SCHEMA
            || self.grouped_fold_bank_sha256 != fold_bank.identity()?
            || self.feature_width == 0
        {
            return Err("fold-local coverage schema, bank, or width drifted".into());
        }
        for (name, value) in [
            ("coverage dataset", self.dataset_sha256.as_str()),
            ("coverage train", self.train_sha256.as_str()),
            ("coverage encoder", self.encoder_identity_sha256.as_str()),
            (
                "coverage authorized full cell",
                self.authorized_full_train_cell_sha256.as_str(),
            ),
            (
                "coverage fresh full cell",
                self.fresh_full_train_cell_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        if !self.structurally_eligible {
            if self
                .failure_reason
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
                || !self.receipts.is_empty()
                || self.full_train_byte_equal
            {
                return Err("ineligible coverage must fail closed with no fold receipts".into());
            }
            return Ok(());
        }
        if self.failure_reason.is_some()
            || !self.full_train_byte_equal
            || self.fresh_full_train_cell_sha256 != self.authorized_full_train_cell_sha256
            || self.receipts.len() != ROUTER_SEEDS.len() * OUTER_FOLDS
        {
            return Err(
                "eligible coverage is incomplete or does not replay train3.vs byte-for-byte".into(),
            );
        }
        let mut keys = BTreeSet::new();
        for receipt in &self.receipts {
            if !keys.insert((receipt.seed, receipt.fold_index)) {
                return Err("fold-local coverage duplicates a seed/fold receipt".into());
            }
            let plan = fold_bank
                .plans
                .iter()
                .find(|plan| plan.seed == receipt.seed)
                .ok_or_else(|| "fold receipt seed is outside the grouped bank".to_string())?;
            receipt.validate(
                plan,
                &self.train_sha256,
                &self.encoder_identity_sha256,
                self.feature_width,
            )?;
        }
        Ok(())
    }

    pub fn identity(&self, fold_bank: &GroupedFoldBankV2) -> Result<String, String> {
        self.validate(fold_bank)?;
        canonical_sha256(self)
    }

    pub fn activation_ready(&self, fold_bank: &GroupedFoldBankV2) -> bool {
        self.validate(fold_bank).is_ok() && self.structurally_eligible
    }
}
