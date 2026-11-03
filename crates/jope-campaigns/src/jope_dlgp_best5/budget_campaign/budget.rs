use super::super::{FrozenPredictorComponentV1, LicenseTrackV1, SEARCH_SEEDS};
use super::policy::{
    finite_positive, sorted_unique, BUDGET_RUNGS, STAGE1_DATASET_IDS, STAGE1_MAX_SEARCH_EVALUATIONS,
};
use crate::search_v3::{canonical_sha256, validate_sha256};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetCheckpointV1 {
    pub evaluations: u32,
    pub prefix_prediction_roots: BTreeMap<u32, String>,
    pub population_sha256: String,
    pub selection_sha256: String,
    pub receipt_root_sha256: String,
    pub prediction_root_sha256: String,
    pub cv7_rmse: Vec<f64>,
    pub archive_top20_semantic_sha256s: Vec<String>,
    pub logical_operations: u64,
    pub cache_hits: u64,
    pub cpu_seconds: f64,
    pub gpu_seconds: f64,
}

impl BudgetCheckpointV1 {
    pub fn validate(&self) -> Result<(), String> {
        if !BUDGET_RUNGS.contains(&self.evaluations)
            || self.cv7_rmse.len() != 7
            || !finite_positive(&self.cv7_rmse)
            || self.archive_top20_semantic_sha256s.len() != 20
            || !sorted_unique(&self.archive_top20_semantic_sha256s)
            || self.logical_operations < self.evaluations as u64
            || self.cache_hits > self.logical_operations
            || !self.cpu_seconds.is_finite()
            || self.cpu_seconds < 0.0
            || !self.gpu_seconds.is_finite()
            || self.gpu_seconds < 0.0
        {
            return Err("budget checkpoint metrics, archive, or accounting are invalid".into());
        }
        for digest in [
            &self.population_sha256,
            &self.selection_sha256,
            &self.receipt_root_sha256,
            &self.prediction_root_sha256,
        ] {
            validate_sha256("budget checkpoint", digest)?;
        }
        for digest in &self.archive_top20_semantic_sha256s {
            validate_sha256("archive semantic", digest)?;
        }
        for digest in self.prefix_prediction_roots.values() {
            validate_sha256("checkpoint prefix", digest)?;
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
pub struct BudgetPromotionDecisionV1 {
    pub source_2k_checkpoint_sha256: String,
    pub source_5k_checkpoint_sha256: String,
    pub continue_to_10k: bool,
    pub improved_folds_at_least_point_one_percent: u32,
    pub archive_jaccard_micros: u32,
    pub cv7_mean_ratio_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetLadderBundleV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub master_seed: u64,
    pub license_track: LicenseTrackV1,
    pub metric_sha256: String,
    pub evaluator_sha256: String,
    pub split_sha256: String,
    pub test_csv_read: bool,
    pub checkpoints: Vec<BudgetCheckpointV1>,
    pub promotion: BudgetPromotionDecisionV1,
}

impl BudgetLadderBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetLadderBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_csv_read
            || !SEARCH_SEEDS.contains(&self.master_seed)
            || !(self.checkpoints.len() == 3 || self.checkpoints.len() == 4)
        {
            return Err(
                "budget ladder schema, seed, test-access, or rung cardinality drift".into(),
            );
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.dataset_sha256,
            &self.metric_sha256,
            &self.evaluator_sha256,
            &self.split_sha256,
        ] {
            validate_sha256("budget ladder", digest)?;
        }
        for checkpoint in &self.checkpoints {
            checkpoint.validate()?;
        }
        let actual_rungs = self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.evaluations)
            .collect::<Vec<_>>();
        if actual_rungs != BUDGET_RUNGS[..actual_rungs.len()] {
            return Err("budget ladder checkpoints are not the nested frozen prefix".into());
        }
        for (index, checkpoint) in self.checkpoints.iter().enumerate() {
            let expected = self.checkpoints[..=index]
                .iter()
                .map(|prior| (prior.evaluations, prior.prediction_root_sha256.clone()))
                .collect::<BTreeMap<_, _>>();
            if checkpoint.prefix_prediction_roots != expected {
                return Err(
                    "checkpoint prefix roots do not reproduce the earlier trajectory".into(),
                );
            }
        }
        let two_k = &self.checkpoints[1];
        let five_k = &self.checkpoints[2];
        if self.promotion.source_2k_checkpoint_sha256 != two_k.identity()?
            || self.promotion.source_5k_checkpoint_sha256 != five_k.identity()?
        {
            return Err("10k promotion decision does not bind the 2k and 5k checkpoints".into());
        }
        let improved_folds = two_k
            .cv7_rmse
            .iter()
            .zip(&five_k.cv7_rmse)
            .filter(|(two, five)| **five <= **two * 0.999)
            .count() as u32;
        let left = two_k
            .archive_top20_semantic_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        let right = five_k
            .archive_top20_semantic_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        let intersection = left.intersection(&right).count() as u64;
        let union = left.union(&right).count() as u64;
        let jaccard_micros = (intersection * 1_000_000 / union) as u32;
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let mean_ratio_micros =
            (mean(&five_k.cv7_rmse) / mean(&two_k.cv7_rmse) * 1_000_000.0).round() as u32;
        let should_continue =
            improved_folds >= 4 || (jaccard_micros < 800_000 && mean_ratio_micros <= 1_005_000);
        if self.promotion.improved_folds_at_least_point_one_percent != improved_folds
            || self.promotion.archive_jaccard_micros != jaccard_micros
            || self.promotion.cv7_mean_ratio_micros != mean_ratio_micros
            || self.promotion.continue_to_10k != should_continue
            || should_continue != (self.checkpoints.len() == 4)
        {
            return Err("10k promotion decision differs from the frozen train-only rule".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn verify_prefix_equality(
    left: &BudgetLadderBundleV1,
    right: &BudgetLadderBundleV1,
    evaluations: u32,
) -> Result<(), String> {
    left.validate()?;
    right.validate()?;
    let left_prefix = left
        .checkpoints
        .last()
        .and_then(|checkpoint| checkpoint.prefix_prediction_roots.get(&evaluations));
    let right_prefix = right
        .checkpoints
        .last()
        .and_then(|checkpoint| checkpoint.prefix_prediction_roots.get(&evaluations));
    if left.campaign_freeze_sha256 != right.campaign_freeze_sha256
        || left.dataset_sha256 != right.dataset_sha256
        || left.master_seed != right.master_seed
        || left.license_track != right.license_track
        || left_prefix.is_none()
        || left_prefix != right_prefix
    {
        return Err("resumed/retried budget trajectories differ at their frozen prefix".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetStageInventoryV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub bundles: Vec<BudgetLadderBundleV1>,
}

impl BudgetStageInventoryV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBudgetStageInventory.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.bundles.len() != 20 {
            return Err("stage-1 inventory must contain ten datasets by two seeds".into());
        }
        validate_sha256("stage-1 campaign", &self.campaign_freeze_sha256)?;
        let mut actual = BTreeSet::new();
        let expected = STAGE1_DATASET_IDS
            .iter()
            .flat_map(|dataset| SEARCH_SEEDS.map(|seed| (*dataset, seed)))
            .collect::<BTreeSet<_>>();
        let mut evaluations = 0_u64;
        for bundle in &self.bundles {
            bundle.validate()?;
            let dataset_id = STAGE1_DATASET_IDS
                .iter()
                .copied()
                .find(|dataset| bundle.dataset_sha256.starts_with(dataset))
                .ok_or_else(|| "stage-1 bundle is outside the frozen frontier".to_string())?;
            if bundle.campaign_freeze_sha256 != self.campaign_freeze_sha256
                || !actual.insert((dataset_id, bundle.master_seed))
            {
                return Err("stage-1 inventory contains a foreign or duplicate bundle".into());
            }
            evaluations += bundle.checkpoints.last().unwrap().evaluations as u64;
        }
        if actual != expected || evaluations > STAGE1_MAX_SEARCH_EVALUATIONS {
            return Err("stage-1 dataset grid or 200,000-evaluation cap drifted".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OuterFoldBaseReceiptV1 {
    pub outer_fold: u32,
    pub outer_training_row_ids: Vec<String>,
    pub outer_validation_row_ids: Vec<String>,
    pub row_atomic_groups: BTreeMap<String, String>,
    pub component: FrozenPredictorComponentV1,
    pub inner_oof_prediction_sha256: String,
    pub outer_validation_prediction_sha256: String,
}

impl OuterFoldBaseReceiptV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.outer_fold >= 7
            || self.outer_training_row_ids.is_empty()
            || self.outer_validation_row_ids.is_empty()
            || !sorted_unique(&self.outer_training_row_ids)
            || !sorted_unique(&self.outer_validation_row_ids)
        {
            return Err("outer-fold row inventory is invalid".into());
        }
        self.component.validate()?;
        validate_sha256("inner OOF predictions", &self.inner_oof_prediction_sha256)?;
        validate_sha256(
            "outer validation predictions",
            &self.outer_validation_prediction_sha256,
        )?;
        let training = self.outer_training_row_ids.iter().collect::<BTreeSet<_>>();
        let validation = self
            .outer_validation_row_ids
            .iter()
            .collect::<BTreeSet<_>>();
        if !training.is_disjoint(&validation)
            || self.row_atomic_groups.len() != training.len() + validation.len()
        {
            return Err("nested OOF outer partitions overlap or omit rows".into());
        }
        let training_groups = training
            .iter()
            .map(|row| self.row_atomic_groups.get(*row))
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| "nested OOF training row lacks an atomic group".to_string())?;
        let validation_groups = validation
            .iter()
            .map(|row| self.row_atomic_groups.get(*row))
            .collect::<Option<BTreeSet<_>>>()
            .ok_or_else(|| "nested OOF validation row lacks an atomic group".to_string())?;
        if !training_groups.is_disjoint(&validation_groups) {
            return Err("nested OOF split crosses a duplicate-feature atomic group".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedOofBundleV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub license_track: LicenseTrackV1,
    pub cross_fit_protocol_sha256: String,
    pub folds: Vec<OuterFoldBaseReceiptV1>,
    pub selection_freeze_sha256: Option<String>,
    pub full_train_base_prediction_sha256: Option<String>,
}

impl NestedOofBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpNestedOofBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.folds.len() != 7 {
            return Err("nested OOF bundle must contain seven outer folds".into());
        }
        for digest in [
            &self.campaign_freeze_sha256,
            &self.dataset_sha256,
            &self.cross_fit_protocol_sha256,
        ] {
            validate_sha256("nested OOF bundle", digest)?;
        }
        for (index, fold) in self.folds.iter().enumerate() {
            fold.validate()?;
            if fold.outer_fold != index as u32
                || (self.license_track == LicenseTrackV1::ProductionCompatible
                    && fold.component.license_track() == LicenseTrackV1::ResearchOnly)
            {
                return Err("nested OOF fold order or license track drifted".into());
            }
        }
        match (
            &self.selection_freeze_sha256,
            &self.full_train_base_prediction_sha256,
        ) {
            (None, None) => Ok(()),
            (Some(selection), Some(prediction)) => {
                validate_sha256("selection freeze", selection)?;
                validate_sha256("full-train base prediction", prediction)
            }
            _ => Err("full-train base predictions require a prior selection freeze".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackArtifactNamespacesV1 {
    pub production_compatible_root: String,
    pub research_only_root: String,
}

impl TrackArtifactNamespacesV1 {
    pub fn validate(&self) -> Result<(), String> {
        let production = Path::new(&self.production_compatible_root);
        let research = Path::new(&self.research_only_root);
        if !production.is_absolute()
            || !research.is_absolute()
            || production == research
            || production.starts_with(research)
            || research.starts_with(production)
        {
            return Err(
                "production and research artifact trees must be disjoint absolute roots".into(),
            );
        }
        Ok(())
    }

    pub fn validate_artifact(&self, track: LicenseTrackV1, path: &str) -> Result<(), String> {
        self.validate()?;
        let artifact = Path::new(path);
        let expected = match track {
            LicenseTrackV1::ProductionCompatible => Path::new(&self.production_compatible_root),
            LicenseTrackV1::ResearchOnly => Path::new(&self.research_only_root),
        };
        if !artifact.is_absolute()
            || artifact
                .components()
                .any(|part| part == Component::ParentDir)
            || !artifact.starts_with(expected)
        {
            return Err("artifact path escapes its license-track tree".into());
        }
        Ok(())
    }
}

pub fn validate_search_artifact_json(value: &serde_json::Value) -> Result<(), String> {
    match value {
        serde_json::Value::Object(values) => {
            for (key, nested) in values {
                let normalized = key.to_ascii_lowercase();
                if normalized == "test_csv_read" {
                    if nested != &serde_json::Value::Bool(false) {
                        return Err("search artifact does not attest test_csv_read=false".into());
                    }
                } else if normalized != "test_access_policy"
                    && (normalized.starts_with("test_")
                        || normalized.ends_with("_test")
                        || matches!(
                            normalized.as_str(),
                            "target_rmse" | "comparator_target" | "label_statistics" | "test"
                        ))
                {
                    return Err("search artifact contains sealed test material".into());
                }
                validate_search_artifact_json(nested)?;
            }
        }
        serde_json::Value::Array(values) => {
            for nested in values {
                validate_search_artifact_json(nested)?;
            }
        }
        serde_json::Value::String(value) if value.to_ascii_lowercase().contains("test.csv") => {
            return Err("search artifact contains a test path".into());
        }
        _ => {}
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedDatasetTrackScoreV1 {
    pub dataset_sha256: String,
    pub license_track: LicenseTrackV1,
    pub comparator_rmse: f64,
    pub seed_rmse: BTreeMap<u64, f64>,
}

impl SealedDatasetTrackScoreV1 {
    pub fn robust_win(&self, expected_seeds: [u64; 2]) -> Result<bool, String> {
        validate_sha256("sealed score dataset", &self.dataset_sha256)?;
        if !self.comparator_rmse.is_finite()
            || self.comparator_rmse < 0.0
            || self.seed_rmse.keys().copied().collect::<Vec<_>>() != expected_seeds
            || !finite_positive(&self.seed_rmse.values().copied().collect::<Vec<_>>())
        {
            return Err("sealed paired score is invalid or uses the wrong seeds".into());
        }
        if self.comparator_rmse == 0.0 {
            return Ok(false);
        }
        Ok(self
            .seed_rmse
            .values()
            .all(|rmse| *rmse < self.comparator_rmse))
    }
}
