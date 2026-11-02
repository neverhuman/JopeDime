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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedBankSeedResultV2 {
    pub seed: u64,
    pub raw_result_sha256: String,
    pub cell_only_result_sha256: String,
    pub append_result_sha256: String,
    pub raw_pooled_rmse: f64,
    pub cell_only_pooled_rmse: f64,
    pub append_pooled_rmse: f64,
    pub append_fold_relative_regressions: Vec<f64>,
    pub complete: bool,
}

impl FixedBankSeedResultV2 {
    pub fn validate(&self) -> Result<(), String> {
        if !ROUTER_SEEDS.contains(&self.seed)
            || !self.complete
            || self.append_fold_relative_regressions.len() != OUTER_FOLDS
            || ![
                self.raw_pooled_rmse,
                self.cell_only_pooled_rmse,
                self.append_pooled_rmse,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            || self
                .append_fold_relative_regressions
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("fixed-bank seed result is incomplete or nonfinite".into());
        }
        for digest in [
            &self.raw_result_sha256,
            &self.cell_only_result_sha256,
            &self.append_result_sha256,
        ] {
            validate_sha256("fixed-bank strategy result", digest)?;
        }
        Ok(())
    }

    pub fn append_relative_improvement(&self) -> f64 {
        if self.raw_pooled_rmse > 0.0 {
            (self.raw_pooled_rmse - self.append_pooled_rmse) / self.raw_pooled_rmse
        } else if self.append_pooled_rmse == 0.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    pub fn worst_append_fold_regression(&self) -> f64 {
        self.append_fold_relative_regressions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedBankAdmissionV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub candidate_bank_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub seeds: Vec<FixedBankSeedResultV2>,
    pub append_admitted: bool,
    pub activation_authority: bool,
}

impl FixedBankAdmissionV2 {
    pub const SCHEMA: &'static str = "HyperionFixedBankAdmission.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.activation_authority
            || self.seeds.len() != ROUTER_SEEDS.len()
        {
            return Err("fixed-bank screen must remain a two-seed diagnostic".into());
        }
        for (name, value) in [
            ("fixed-bank dataset", self.dataset_sha256.as_str()),
            ("fixed-bank train", self.train_sha256.as_str()),
            ("fixed-bank real cell", self.real_cell_sha256.as_str()),
            (
                "fixed-bank candidate bank",
                self.candidate_bank_sha256.as_str(),
            ),
            (
                "fixed-bank fold bank",
                self.grouped_fold_bank_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        for (index, seed) in self.seeds.iter().enumerate() {
            seed.validate()?;
            if seed.seed != ROUTER_SEEDS[index] {
                return Err("fixed-bank seed order drifted".into());
            }
        }
        let expected_admission = self.seeds.iter().all(|seed| {
            seed.append_relative_improvement() >= 0.0 && seed.worst_append_fold_regression() <= 0.10
        });
        if self.append_admitted != expected_admission {
            return Err(
                "fixed-bank append admission does not match its frozen two-seed gate".into(),
            );
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
pub struct GpStreamReceiptV2 {
    pub schema: String,
    pub seed: u64,
    pub outer_fold: u32,
    pub view: FeatureViewKindV2,
    pub evaluation_budget: u32,
    pub evaluations_completed: u32,
    pub inner_folds: u32,
    pub pooled_oof_rmse: bool,
    pub vocabulary: String,
    pub winner_se_band: f64,
    pub candidate_timeout_seconds: u32,
    pub search_identity_sha256: String,
    pub ranking_sha256: String,
    pub candidate_archive_sha256: String,
    pub complete: bool,
    pub failure_count: u32,
    pub outer_hold_labels_reached_search: bool,
    pub test_material_present: bool,
}

impl GpStreamReceiptV2 {
    pub const SCHEMA: &'static str = "HyperionGpStreamReceipt.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.outer_fold as usize >= OUTER_FOLDS
            || self.evaluation_budget != SEARCH_EVALUATIONS_PER_STREAM
            || self.evaluations_completed != SEARCH_EVALUATIONS_PER_STREAM
            || self.inner_folds != INNER_FOLDS as u32
            || !self.pooled_oof_rmse
            || self.vocabulary != "expanded"
            || (self.winner_se_band - 0.5).abs() > f64::EPSILON
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
            || !self.complete
            || self.failure_count != 0
            || self.outer_hold_labels_reached_search
            || self.test_material_present
        {
            return Err(
                "GP stream drifted from the independent exact-2,000 nested contract".into(),
            );
        }
        for digest in [
            &self.search_identity_sha256,
            &self.ranking_sha256,
            &self.candidate_archive_sha256,
        ] {
            validate_sha256("GP stream", digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedTwoStreamSearchV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub fold_cell_coverage_sha256: String,
    pub fixed_bank_admission_sha256: String,
    pub append_admitted: bool,
    pub streams: Vec<GpStreamReceiptV2>,
    pub finalist_union_sha256: String,
    pub archives_combined_only_after_search: bool,
    pub test_material_present: bool,
}

impl NestedTwoStreamSearchV2 {
    pub const SCHEMA: &'static str = "HyperionNestedTwoStreamSearch.v2";

    pub fn validate(&self, admission: &FixedBankAdmissionV2) -> Result<(), String> {
        admission.validate()?;
        if self.schema != Self::SCHEMA
            || self.dataset_sha256 != admission.dataset_sha256
            || self.grouped_fold_bank_sha256 != admission.grouped_fold_bank_sha256
            || self.fixed_bank_admission_sha256 != admission.identity()?
            || self.append_admitted != admission.append_admitted
            || !self.archives_combined_only_after_search
            || self.test_material_present
        {
            return Err("nested two-stream search is foreign or crossed the test boundary".into());
        }
        validate_sha256("nested search dataset", &self.dataset_sha256)?;
        validate_sha256("nested search coverage", &self.fold_cell_coverage_sha256)?;
        validate_sha256("nested finalist union", &self.finalist_union_sha256)?;
        let expected_views = if self.append_admitted { 3 } else { 2 };
        if self.streams.len() != ROUTER_SEEDS.len() * OUTER_FOLDS * expected_views {
            return Err("nested search stream coverage is incomplete".into());
        }
        let mut keys = BTreeSet::new();
        let mut search_ids = BTreeSet::new();
        let mut ranking_ids = BTreeSet::new();
        let mut archive_ids = BTreeSet::new();
        for stream in &self.streams {
            stream.validate()?;
            if stream.view == FeatureViewKindV2::HyperionCellAppend && !self.append_admitted {
                return Err("append stream ran without diagnostic admission".into());
            }
            if !keys.insert((stream.seed, stream.outer_fold, stream.view))
                || !search_ids.insert(stream.search_identity_sha256.as_str())
                || !ranking_ids.insert(stream.ranking_sha256.as_str())
                || !archive_ids.insert(stream.candidate_archive_sha256.as_str())
            {
                return Err("GP views shared a search, ranking, archive, or seed/fold slot".into());
            }
        }
        for seed in ROUTER_SEEDS {
            for fold in 0..OUTER_FOLDS as u32 {
                for view in [FeatureViewKindV2::Raw, FeatureViewKindV2::HyperionCellOnly] {
                    if !keys.contains(&(seed, fold, view)) {
                        return Err("raw or cell-only nested stream is missing".into());
                    }
                }
                if self.append_admitted
                    && !keys.contains(&(seed, fold, FeatureViewKindV2::HyperionCellAppend))
                {
                    return Err("admitted append nested stream is missing".into());
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self, admission: &FixedBankAdmissionV2) -> Result<String, String> {
        self.validate(admission)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendFitV2 {
    pub alpha: f64,
    pub denominator: f64,
    pub stable: bool,
}

/// Fit the raw-biased blend from inner OOF predictions only.
pub fn fit_raw_biased_blend_v2(
    truth: &[f64],
    raw_oof: &[f64],
    cell_oof: &[f64],
) -> Result<BlendFitV2, String> {
    if truth.is_empty() || truth.len() != raw_oof.len() || truth.len() != cell_oof.len() {
        return Err("blend OOF vectors are empty or have different lengths".into());
    }
    require_finite("blend truth", truth)?;
    require_finite("blend raw predictions", raw_oof)?;
    require_finite("blend cell predictions", cell_oof)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for ((truth, raw), cell) in truth.iter().zip(raw_oof).zip(cell_oof) {
        let direction = cell - raw;
        numerator += direction * (truth - raw);
        denominator += direction * direction;
    }
    let stability_floor = 1e-12 * truth.len() as f64;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= stability_floor {
        return Ok(BlendFitV2 {
            alpha: 0.0,
            denominator,
            stable: false,
        });
    }
    Ok(BlendFitV2 {
        alpha: (numerator / denominator).clamp(0.0, 0.5),
        denominator,
        stable: true,
    })
}

pub fn apply_raw_biased_blend_v2(
    raw_predictions: &[f64],
    cell_predictions: &[f64],
    fit: BlendFitV2,
) -> Result<Vec<f64>, String> {
    if raw_predictions.len() != cell_predictions.len()
        || raw_predictions.is_empty()
        || !fit.alpha.is_finite()
        || !(0.0..=0.5).contains(&fit.alpha)
    {
        return Err("blend prediction inputs or coefficient are invalid".into());
    }
    require_finite("blend raw predictions", raw_predictions)?;
    require_finite("blend cell predictions", cell_predictions)?;
    Ok(raw_predictions
        .iter()
        .zip(cell_predictions)
        .map(|(raw, cell)| raw + fit.alpha * (cell - raw))
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteFamilyV2 {
    Raw,
    CellOnly,
    Append,
    RawCellBlend,
}

impl RouteFamilyV2 {
    fn preference(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::CellOnly => 1,
            Self::Append => 2,
            Self::RawCellBlend => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationThresholdsV2 {
    pub minimum_seed_relative_improvement_micros: u32,
    pub minimum_seed_normalized_improvement_micros: u32,
    pub minimum_fold_wins: u32,
    pub grouped_bootstrap_confidence_micros: u32,
    pub maximum_fold_regression_micros: u32,
    pub minimum_two_seed_median_improvement_micros: u32,
    pub bootstrap_repetitions: u32,
}

impl Default for ActivationThresholdsV2 {
    fn default() -> Self {
        Self {
            minimum_seed_relative_improvement_micros: 5_000,
            minimum_seed_normalized_improvement_micros: 5_000,
            minimum_fold_wins: 5,
            grouped_bootstrap_confidence_micros: 950_000,
            maximum_fold_regression_micros: 20_000,
            minimum_two_seed_median_improvement_micros: 10_000,
            bootstrap_repetitions: GROUPED_BOOTSTRAP_REPETITIONS as u32,
        }
    }
}
