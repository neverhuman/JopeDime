use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStageV1 {
    Cv5Screen,
    Cv7Confirmation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeHyperionFoldReceiptV1 {
    pub schema: String,
    pub stage: GateStageV1,
    pub fold_index: u32,
    pub fold_plan_sha256: String,
    pub feature_view: JopeFeatureViewV1,
    pub feature_view_sha256: String,
    pub predictor_sha256: String,
    pub encoder_identity_sha256: String,
    pub cache_key_sha256: String,
    pub cache_receipt_sha256: String,
    pub source_training_sha256: String,
    pub fit_row_ids_sha256: String,
    pub hold_row_ids_sha256: String,
    pub fit_atomic_groups_sha256: String,
    pub hold_atomic_groups_sha256: String,
    pub sidecars: Vec<MatrixArtifactV1>,
    pub expanded_fit_sha256: String,
    pub expanded_hold_sha256: String,
    pub raw_predictions_sha256: String,
    pub view_predictions_sha256: String,
    pub raw_rmse: f64,
    pub view_rmse: f64,
    pub encoding_milliseconds: u64,
    pub scoring_milliseconds: u64,
    pub cache_replay: bool,
}

impl JopeHyperionFoldReceiptV1 {
    pub const SCHEMA: &'static str = "JopeHyperionFoldReceipt.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("fold receipt schema drift".into());
        }
        self.feature_view.validate()?;
        for (name, digest) in [
            ("fold plan", &self.fold_plan_sha256),
            ("feature view", &self.feature_view_sha256),
            ("predictor", &self.predictor_sha256),
            ("encoder identity", &self.encoder_identity_sha256),
            ("cache key", &self.cache_key_sha256),
            ("cache receipt", &self.cache_receipt_sha256),
            ("training", &self.source_training_sha256),
            ("fit rows", &self.fit_row_ids_sha256),
            ("hold rows", &self.hold_row_ids_sha256),
            ("fit groups", &self.fit_atomic_groups_sha256),
            ("hold groups", &self.hold_atomic_groups_sha256),
            ("expanded fit", &self.expanded_fit_sha256),
            ("expanded hold", &self.expanded_hold_sha256),
            ("raw predictions", &self.raw_predictions_sha256),
            ("view predictions", &self.view_predictions_sha256),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.feature_view.identity()? != self.feature_view_sha256
            || !self.raw_rmse.is_finite()
            || !self.view_rmse.is_finite()
            || self.raw_rmse < 0.0
            || self.view_rmse < 0.0
        {
            return Err("fold receipt feature binding or metric is invalid".into());
        }
        for sidecar in &self.sidecars {
            sidecar.validate()?;
        }
        let sidecars = self
            .sidecars
            .iter()
            .map(|sidecar| (sidecar.role.as_str(), (sidecar.rows, sidecar.columns)))
            .collect::<BTreeMap<_, _>>();
        let raw = self.feature_view.raw_features;
        let expected = match self.feature_view.view {
            FeatureViewKindV1::Raw => BTreeSet::new(),
            FeatureViewKindV1::HyperionCell => ["cell_fit", "cell_hold"].into_iter().collect(),
            FeatureViewKindV1::HyperionRow128 => ["row_fit", "row_hold"].into_iter().collect(),
            FeatureViewKindV1::HyperionBoth => ["cell_fit", "cell_hold", "row_fit", "row_hold"]
                .into_iter()
                .collect(),
        };
        let actual = sidecars.keys().copied().collect::<BTreeSet<_>>();
        let cell_invalid = expected.contains("cell_fit")
            && (sidecars
                .get("cell_fit")
                .is_none_or(|(_, columns)| *columns != raw)
                || sidecars
                    .get("cell_hold")
                    .is_none_or(|(_, columns)| *columns != raw));
        let row_invalid = expected.contains("row_fit")
            && (sidecars
                .get("row_fit")
                .is_none_or(|(_, columns)| *columns != ROW_EMBEDDING_WIDTH as u32)
                || sidecars
                    .get("row_hold")
                    .is_none_or(|(_, columns)| *columns != ROW_EMBEDDING_WIDTH as u32));
        let both_misaligned = self.feature_view.view == FeatureViewKindV1::HyperionBoth
            && match (
                sidecars.get("cell_fit"),
                sidecars.get("cell_hold"),
                sidecars.get("row_fit"),
                sidecars.get("row_hold"),
            ) {
                (Some(cell_fit), Some(cell_hold), Some(row_fit), Some(row_hold)) => {
                    cell_fit.0 != row_fit.0 || cell_hold.0 != row_hold.0
                }
                _ => true,
            };
        if self.feature_view.view == FeatureViewKindV1::Raw
            || sidecars.len() != self.sidecars.len()
            || actual != expected
            || cell_invalid
            || row_invalid
            || both_misaligned
        {
            return Err("fold receipt sidecar coverage or shape is invalid".into());
        }
        Ok(())
    }
}

pub type JopeHyperionFoldReceiptV2 = JopeHyperionFoldReceiptV1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedFoldEvidenceV1 {
    pub fold_index: u32,
    pub row_ids: Vec<String>,
    pub atomic_groups: Vec<String>,
    pub actual: Vec<f64>,
    pub raw_predictions: Vec<f64>,
    pub arm_predictions: BTreeMap<FeatureViewKindV1, Vec<f64>>,
}

impl PairedFoldEvidenceV1 {
    pub fn validate(&self, expected_arms: &[FeatureViewKindV1]) -> Result<(), String> {
        let rows = self.actual.len();
        if rows == 0
            || self.row_ids.len() != rows
            || self.atomic_groups.len() != rows
            || self.raw_predictions.len() != rows
            || self.row_ids.iter().any(String::is_empty)
            || self.atomic_groups.iter().any(String::is_empty)
            || self.row_ids.iter().collect::<BTreeSet<_>>().len() != rows
            || self.actual.iter().any(|value| !value.is_finite())
            || self.raw_predictions.iter().any(|value| !value.is_finite())
        {
            return Err("paired fold rows or predictions are invalid".into());
        }
        let expected = expected_arms.iter().copied().collect::<BTreeSet<_>>();
        if self
            .arm_predictions
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != expected
        {
            return Err("paired fold has incomplete or unexpected arm coverage".into());
        }
        if self.arm_predictions.iter().any(|(arm, values)| {
            *arm == FeatureViewKindV1::Raw
                || values.len() != rows
                || values.iter().any(|value| !value.is_finite())
        }) {
            return Err("paired fold arm predictions are invalid".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageGateResultV1 {
    pub stage: GateStageV1,
    pub arm: FeatureViewKindV1,
    pub folds: u32,
    pub fold_wins: u32,
    pub fold_best_arm_wins: u32,
    pub pooled_raw_rmse: f64,
    pub pooled_arm_rmse: f64,
    pub relative_improvement: f64,
    pub bootstrap_lower_bound_log_improvement: f64,
    pub worst_fold_regression: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateThresholdsV1 {
    pub minimum_relative_improvement: f64,
    pub cv5_minimum_fold_wins: u32,
    pub cv5_minimum_best_arm_wins: u32,
    pub cv5_bootstrap_one_sided_confidence: f64,
    pub cv7_minimum_fold_wins: u32,
    pub cv7_bootstrap_one_sided_confidence: f64,
    pub maximum_fold_regression: f64,
    pub minimum_atomic_groups: u32,
}

impl Default for GateThresholdsV1 {
    fn default() -> Self {
        Self {
            minimum_relative_improvement: 0.005,
            cv5_minimum_fold_wins: 4,
            cv5_minimum_best_arm_wins: 4,
            cv5_bootstrap_one_sided_confidence: 0.90,
            cv7_minimum_fold_wins: 5,
            cv7_bootstrap_one_sided_confidence: 0.95,
            maximum_fold_regression: 0.02,
            minimum_atomic_groups: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeHyperionGateDecisionV1 {
    pub schema: String,
    pub dataset_id_sha256: String,
    pub source_training_sha256: String,
    pub raw_predictor_sha256: String,
    pub encoder_identity_sha256: String,
    pub cv5_fold_plan_sha256: String,
    pub cv7_fold_plan_sha256: String,
    pub thresholds: GateThresholdsV1,
    pub cv5_candidates: Vec<StageGateResultV1>,
    pub cv5_advanced_arm: Option<FeatureViewKindV1>,
    pub cv7_confirmation: Option<StageGateResultV1>,
    pub selected_view: FeatureViewKindV1,
    pub raw_fallback_reason: Option<String>,
    pub fold_receipt_sha256s: Vec<String>,
    pub encoding_milliseconds: u64,
    pub scoring_milliseconds: u64,
    pub wall_clock_milliseconds: u64,
}

impl JopeHyperionGateDecisionV1 {
    pub const SCHEMA: &'static str = "JopeHyperionGateDecision.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("gate decision schema drift".into());
        }
        for (name, digest) in [
            ("dataset ID", &self.dataset_id_sha256),
            ("training", &self.source_training_sha256),
            ("raw predictor", &self.raw_predictor_sha256),
            ("encoder", &self.encoder_identity_sha256),
            ("CV5 plan", &self.cv5_fold_plan_sha256),
            ("CV7 plan", &self.cv7_fold_plan_sha256),
        ] {
            validate_sha256(name, digest)?;
        }
        for digest in &self.fold_receipt_sha256s {
            validate_sha256("fold receipt", digest)?;
        }
        if self.thresholds != GateThresholdsV1::default() {
            return Err("gate thresholds differ from the frozen policy".into());
        }
        if !self.cv5_candidates.is_empty() {
            let arms = self
                .cv5_candidates
                .iter()
                .map(|value| value.arm)
                .collect::<BTreeSet<_>>();
            if arms.len() != self.cv5_candidates.len()
                || self.cv5_candidates.iter().any(|value| {
                    value.stage != GateStageV1::Cv5Screen
                        || value.folds != 5
                        || !FeatureViewKindV1::EXPANDED.contains(&value.arm)
                })
            {
                return Err("gate decision has malformed CV5 candidate coverage".into());
            }
        }
        if let Some(advanced) = self.cv5_advanced_arm {
            if self
                .cv5_candidates
                .iter()
                .find(|value| value.arm == advanced)
                .is_none_or(|value| !value.passed)
            {
                return Err("CV5 advanced arm did not pass its frozen screen".into());
            }
        }
        if self.cv7_confirmation.as_ref().is_some_and(|value| {
            value.stage != GateStageV1::Cv7Confirmation
                || value.folds != 7
                || Some(value.arm) != self.cv5_advanced_arm
        }) {
            return Err("CV7 evidence does not confirm the frozen CV5 arm".into());
        }
        if self.selected_view == FeatureViewKindV1::Raw {
            if self
                .raw_fallback_reason
                .as_deref()
                .is_none_or(str::is_empty)
            {
                return Err("raw selection requires an explicit fallback reason".into());
            }
        } else if self.raw_fallback_reason.is_some()
            || self.cv5_advanced_arm != Some(self.selected_view)
            || self
                .cv7_confirmation
                .as_ref()
                .is_none_or(|value| value.arm != self.selected_view || !value.passed)
        {
            return Err("expanded selection lacks consistent CV5/CV7 evidence".into());
        }
        if self.wall_clock_milliseconds > WALL_CLOCK_BUDGET.as_millis() as u64
            || self.encoding_milliseconds > ENCODING_BUDGET.as_millis() as u64
            || self.scoring_milliseconds > SCORING_BUDGET.as_millis() as u64
        {
            return Err("gate decision exceeds its frozen wall-clock budget".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
