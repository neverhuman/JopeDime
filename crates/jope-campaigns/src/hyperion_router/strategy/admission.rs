use super::*;

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
