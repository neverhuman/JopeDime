//! Train-only, fail-closed gate for optional Hyperion feature expansion.
//!
//! The gate deliberately has no sealed/test-table type.  It compares a frozen
//! AFPM predictor on paired, group-disjoint OOF folds and selects `raw` on any
//! incomplete or inconsistent evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::jope_dlgp_best5::FrozenPredictorV1;
use crate::jope_generalized_gp::{
    raw_sha256, CorpusEvalAdapter, DatasetRowV1, DatasetTableV1, FrozenArtifactV1, GpTaskV1,
    RowSplitV1, SplitRowV1,
};
use crate::search_v3::{canonical_json_bytes, canonical_sha256, validate_sha256};

pub const ENCODING_BUDGET: Duration = Duration::from_secs(8 * 60);
pub const SCORING_BUDGET: Duration = Duration::from_secs(2 * 60);
pub const WALL_CLOCK_BUDGET: Duration = Duration::from_secs(10 * 60);
pub const ROW_EMBEDDING_WIDTH: usize = 128;
pub const BOOTSTRAP_REPLICATES: usize = 4_096;
pub const MINIMUM_FOLD_FIT_ROWS: usize = 32;
pub const MINIMUM_FOLD_HOLD_ROWS: usize = 2;
pub const MINIMUM_FOLD_HOLD_GROUPS: usize = 2;
const CV5_SEED: u64 = 0x4a4f_5045_4859_5035;
const CV7_SEED: u64 = 0x4a4f_5045_4859_5037;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewKindV1 {
    Raw,
    HyperionCell,
    HyperionRow128,
    HyperionBoth,
}

impl FeatureViewKindV1 {
    pub const ALL: [Self; 4] = [
        Self::Raw,
        Self::HyperionCell,
        Self::HyperionRow128,
        Self::HyperionBoth,
    ];

    pub const EXPANDED: [Self; 3] = [Self::HyperionCell, Self::HyperionRow128, Self::HyperionBoth];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeFeatureViewV1 {
    pub schema: String,
    pub view: FeatureViewKindV1,
    pub raw_features: u32,
    pub output_features: u32,
    pub cell_preprocessing: Option<String>,
    pub row_preprocessing: Option<String>,
}

impl JopeFeatureViewV1 {
    pub const SCHEMA: &'static str = "JopeFeatureView.v1";
    pub const CELL_PREPROCESSING: &'static str = "fold_train_empirical_percentile_per_column.v1";
    pub const ROW_PREPROCESSING: &'static str = "fold_train_zscore_per_dimension.v1";

    pub fn new(view: FeatureViewKindV1, raw_features: usize) -> Result<Self, String> {
        if raw_features == 0 || raw_features > u32::MAX as usize {
            return Err("feature view requires a positive representable raw width".into());
        }
        let (cell, row, output) = match view {
            FeatureViewKindV1::Raw => (None, None, raw_features),
            FeatureViewKindV1::HyperionCell => (
                Some(Self::CELL_PREPROCESSING.into()),
                None,
                raw_features * 2,
            ),
            FeatureViewKindV1::HyperionRow128 => (
                None,
                Some(Self::ROW_PREPROCESSING.into()),
                raw_features + ROW_EMBEDDING_WIDTH,
            ),
            FeatureViewKindV1::HyperionBoth => (
                Some(Self::CELL_PREPROCESSING.into()),
                Some(Self::ROW_PREPROCESSING.into()),
                raw_features * 2 + ROW_EMBEDDING_WIDTH,
            ),
        };
        let value = Self {
            schema: Self::SCHEMA.into(),
            view,
            raw_features: raw_features as u32,
            output_features: u32::try_from(output)
                .map_err(|_| "expanded feature width is not representable")?,
            cell_preprocessing: cell,
            row_preprocessing: row,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.raw_features == 0 {
            return Err("feature view schema or raw width is invalid".into());
        }
        let expected = Self::new_unchecked(self.view, self.raw_features);
        if self.output_features != expected.output_features
            || self.cell_preprocessing != expected.cell_preprocessing
            || self.row_preprocessing != expected.row_preprocessing
        {
            return Err("feature view layout or preprocessing policy drifted".into());
        }
        Ok(())
    }

    fn new_unchecked(view: FeatureViewKindV1, raw: u32) -> Self {
        let raw_usize = raw as usize;
        let (cell, row, output) = match view {
            FeatureViewKindV1::Raw => (None, None, raw_usize),
            FeatureViewKindV1::HyperionCell => {
                (Some(Self::CELL_PREPROCESSING.into()), None, raw_usize * 2)
            }
            FeatureViewKindV1::HyperionRow128 => (
                None,
                Some(Self::ROW_PREPROCESSING.into()),
                raw_usize + ROW_EMBEDDING_WIDTH,
            ),
            FeatureViewKindV1::HyperionBoth => (
                Some(Self::CELL_PREPROCESSING.into()),
                Some(Self::ROW_PREPROCESSING.into()),
                raw_usize * 2 + ROW_EMBEDDING_WIDTH,
            ),
        };
        Self {
            schema: Self::SCHEMA.into(),
            view,
            raw_features: raw,
            output_features: output as u32,
            cell_preprocessing: cell,
            row_preprocessing: row,
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn validate_feature_view_feasibility(
    view: FeatureViewKindV1,
    raw_features: usize,
    fit_rows: usize,
) -> Result<(), String> {
    if view == FeatureViewKindV1::Raw {
        return Ok(());
    }
    if fit_rows < MINIMUM_FOLD_FIT_ROWS {
        return Err(format!(
            "expanded view has {fit_rows} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}"
        ));
    }
    JopeFeatureViewV1::new(view, raw_features)?;

    // The cell block is one signal per existing raw column.  A dataset that is
    // already wide (p > n) must not lose this candidate merely because adding
    // its aligned cell block keeps it wide.  The frozen AFPM heads already
    // support wide inputs and the paired CV gate, not an arbitrary p <= n
    // rule, decides whether the block is useful.  Row128 is different: it is a
    // dense, unaligned representation, so retain the conservative fit-row
    // guard for any view that includes it.
    if matches!(
        view,
        FeatureViewKindV1::HyperionRow128 | FeatureViewKindV1::HyperionBoth
    ) && ROW_EMBEDDING_WIDTH > fit_rows
    {
        return Err(format!(
            "row128 view adds {ROW_EMBEDDING_WIDTH} dense features but fold has only {fit_rows} fit rows"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixArtifactV1 {
    pub role: String,
    pub rows: u32,
    pub columns: u32,
    pub sha256: String,
}

impl MatrixArtifactV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.role.trim().is_empty() || self.rows == 0 || self.columns == 0 {
            return Err("sidecar matrix identity or shape is incomplete".into());
        }
        validate_sha256("sidecar matrix", &self.sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionEncoderIdentityV1 {
    pub binary: FrozenArtifactV1,
    pub weights: FrozenArtifactV1,
    pub manifest: FrozenArtifactV1,
    pub seed: u64,
    pub sweeps: u32,
    pub scales: Vec<u32>,
    pub batch_size: u32,
    pub device_type: String,
    pub device_ordinal: u32,
    pub runtime_versions: BTreeMap<String, String>,
    pub source_hashes: BTreeMap<String, String>,
}

impl HyperionEncoderIdentityV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.binary.validate()?;
        self.weights.validate()?;
        self.manifest.validate()?;
        if self.seed == 0
            || self.sweeps == 0
            || self.scales.is_empty()
            || self.scales.contains(&0)
            || self.scales.iter().collect::<BTreeSet<_>>().len() != self.scales.len()
            || self.batch_size == 0
            || self.device_type != "cuda"
            || ["rustc", "cuda_driver", "gpu_name", "gpu_uuid"]
                .iter()
                .any(|key| !self.runtime_versions.contains_key(*key))
            || self.source_hashes.is_empty()
            || self
                .runtime_versions
                .iter()
                .any(|(key, value)| key.trim().is_empty() || value.trim().is_empty())
        {
            return Err("encoder configuration is incomplete or is not CUDA-pinned".into());
        }
        for digest in self.source_hashes.values() {
            validate_sha256("encoder source", digest)?;
        }
        Ok(())
    }

    pub fn verify_artifacts(&self) -> Result<(), String> {
        self.validate()?;
        self.binary.verify()?;
        self.weights.verify()?;
        self.manifest.verify()?;
        verify_runtime_identity(self)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    /// Derive the node-oblivious authority used by V2 receipts.  The V1
    /// registry still carries physical placement for admission, but that
    /// proof is intentionally removed from the portable identity before a
    /// cache or policy is frozen.
    pub fn portable_identity_v2(
        &self,
        runtime_class: impl Into<String>,
    ) -> Result<HyperionEncoderIdentityV2, String> {
        self.validate()?;
        let mut runtime_versions = self.runtime_versions.clone();
        runtime_versions.remove("gpu_uuid");
        runtime_versions.remove("gpu_ordinal");
        let identity = HyperionEncoderIdentityV2 {
            schema: HYPERION_ENCODER_IDENTITY_V2_SCHEMA.into(),
            binary: self.binary.clone(),
            weights: self.weights.clone(),
            manifest: self.manifest.clone(),
            seed: self.seed,
            sweeps: self.sweeps,
            scales: self.scales.clone(),
            batch_size: self.batch_size,
            device_type: self.device_type.clone(),
            runtime_class: runtime_class.into(),
            runtime_versions,
            source_hashes: self.source_hashes.clone(),
            argv_schema: hyperion_argv_schema_v2(),
            environment: hyperion_environment_v2(),
        };
        identity.validate()?;
        Ok(identity)
    }
}

pub const HYPERION_ENCODER_IDENTITY_V2_SCHEMA: &str = "HyperionEncoderIdentity.v2";
pub const HYPERION_ENCODER_PROOF_V2_SCHEMA: &str = "HyperionEncoderExecutionProof.v2";

pub fn hyperion_argv_schema_v2() -> Vec<String> {
    [
        "--train",
        "--test",
        "--out-cell-train",
        "--out-cell-test",
        "--out-row-train",
        "--out-row-test",
        "--weights",
        "--manifest",
        "--sweeps",
        "--scales",
        "--seed",
        "--batch",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub fn hyperion_environment_v2() -> BTreeMap<String, String> {
    [
        ("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO", "1000000"),
        ("OMP_NUM_THREADS", "2"),
        ("OPENBLAS_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
        ("BLIS_NUM_THREADS", "2"),
        ("VECLIB_MAXIMUM_THREADS", "2"),
        ("NUMEXPR_NUM_THREADS", "2"),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), value.into()))
    .collect()
}

/// Portable encoder authority. It names software, weights, runtime class,
/// argv, and fixed environment, but deliberately carries no node, GPU UUID,
/// or physical ordinal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionEncoderIdentityV2 {
    pub schema: String,
    pub binary: FrozenArtifactV1,
    pub weights: FrozenArtifactV1,
    pub manifest: FrozenArtifactV1,
    pub seed: u64,
    pub sweeps: u32,
    pub scales: Vec<u32>,
    pub batch_size: u32,
    pub device_type: String,
    pub runtime_class: String,
    pub runtime_versions: BTreeMap<String, String>,
    pub source_hashes: BTreeMap<String, String>,
    pub argv_schema: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl HyperionEncoderIdentityV2 {
    pub fn validate(&self) -> Result<(), String> {
        self.binary.validate()?;
        self.weights.validate()?;
        self.manifest.validate()?;
        if self.schema != HYPERION_ENCODER_IDENTITY_V2_SCHEMA
            || self.seed == 0
            || self.sweeps == 0
            || self.scales.is_empty()
            || self.scales.contains(&0)
            || self.scales.iter().collect::<BTreeSet<_>>().len() != self.scales.len()
            || self.batch_size == 0
            || self.device_type != "cuda"
            || self.runtime_class.trim().is_empty()
            || self.argv_schema != hyperion_argv_schema_v2()
            || self.environment != hyperion_environment_v2()
            || self.runtime_versions.contains_key("gpu_uuid")
            || self.runtime_versions.contains_key("gpu_ordinal")
            || ["rustc", "cuda_driver", "gpu_name"]
                .iter()
                .any(|key| self.runtime_versions.get(*key).is_none_or(String::is_empty))
            || self.source_hashes.is_empty()
        {
            return Err("Hyperion V2 portable identity is incomplete or drifted".into());
        }
        for digest in self.source_hashes.values() {
            validate_sha256("Hyperion V2 source", digest)?;
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
pub struct HyperionEncoderExecutionProofV2 {
    pub schema: String,
    pub encoder_identity_sha256: String,
    pub node: String,
    pub gpu_uuid: String,
    pub gpu_name: String,
    pub cuda_driver: String,
    pub cuda_visible_devices: String,
    pub observed_device: String,
    pub argv_sha256: String,
    pub environment_sha256: String,
    pub stdout: String,
    pub stdout_sha256: String,
    pub stderr: String,
    pub stderr_sha256: String,
}

impl HyperionEncoderExecutionProofV2 {
    pub fn validate(&self, identity: &HyperionEncoderIdentityV2) -> Result<(), String> {
        identity.validate()?;
        if self.schema != HYPERION_ENCODER_PROOF_V2_SCHEMA
            || self.node.trim().is_empty()
            || self.gpu_uuid.trim().is_empty()
            || self.gpu_name != identity.runtime_versions["gpu_name"]
            || self.cuda_driver != identity.runtime_versions["cuda_driver"]
            || self.cuda_visible_devices != self.gpu_uuid
            || self.observed_device != "cuda:0"
            || !encoder_output_proves_device(&self.stdout, &self.stderr, "cuda:0")
            || self.argv_sha256 != canonical_sha256(&identity.argv_schema)?
            || self.environment_sha256 != canonical_sha256(&identity.environment)?
        {
            return Err("Hyperion V2 execution proof does not match portable authority".into());
        }
        if self.encoder_identity_sha256 != identity.identity()? {
            return Err("Hyperion V2 execution proof names a different encoder".into());
        }
        for (name, actual, expected) in [
            (
                "stdout",
                &self.stdout_sha256,
                raw_sha256(self.stdout.as_bytes()),
            ),
            (
                "stderr",
                &self.stderr_sha256,
                raw_sha256(self.stderr.as_bytes()),
            ),
        ] {
            validate_sha256(name, actual)?;
            if actual != &expected {
                return Err(format!("Hyperion V2 {name} hash mismatch"));
            }
        }
        Ok(())
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeFeatureBoundPredictorV1 {
    pub schema: String,
    pub selected_view: JopeFeatureViewV1,
    pub feature_view_sha256: String,
    pub gate_decision_sha256: String,
    pub selected_predictor: FrozenPredictorV1,
    pub selected_predictor_sha256: String,
    pub raw_fallback_predictor: FrozenPredictorV1,
    pub raw_fallback_predictor_sha256: String,
}

impl JopeFeatureBoundPredictorV1 {
    pub const SCHEMA: &'static str = "JopeFeatureBoundPredictor.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("feature-bound predictor schema drift".into());
        }
        self.selected_view.validate()?;
        validate_sha256("feature view", &self.feature_view_sha256)?;
        validate_sha256("gate decision", &self.gate_decision_sha256)?;
        validate_sha256("selected predictor", &self.selected_predictor_sha256)?;
        validate_sha256(
            "raw fallback predictor",
            &self.raw_fallback_predictor_sha256,
        )?;
        self.selected_predictor.validate()?;
        self.raw_fallback_predictor.validate()?;
        if self.selected_view.identity()? != self.feature_view_sha256
            || self.selected_predictor.semantic_sha256()? != self.selected_predictor_sha256
            || self.raw_fallback_predictor.semantic_sha256()? != self.raw_fallback_predictor_sha256
        {
            return Err("feature-bound predictor identity mismatch".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainOnlyDatasetV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub train_csv_path: String,
    pub train_sha256: String,
    pub feature_count: u32,
    pub stable_row_ids: Vec<String>,
    pub atomic_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeHyperionTrainOnlyRegistryV1 {
    pub schema: String,
    pub encoder: HyperionEncoderIdentityV1,
    pub evaluator: crate::jope_generalized_gp::EvaluatorFreezeV1,
    pub datasets: Vec<TrainOnlyDatasetV1>,
}

/// Legacy registry shape retained only so old authorities can be diagnosed.
/// The gate intentionally accepts V2 exclusively because V1 has no bound
/// cache namespace.

impl JopeHyperionTrainOnlyRegistryV1 {
    pub const SCHEMA: &'static str = "JopeHyperionTrainOnlyRegistry.v1";

    pub fn validate(&self, root: &Path) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("train-only registry schema drift".into());
        }
        self.encoder.validate()?;
        self.evaluator.validate()?;
        let mut ids = BTreeSet::new();
        for dataset in &self.datasets {
            validate_opaque_id(&dataset.dataset_id)?;
            validate_sha256("dataset", &dataset.dataset_sha256)?;
            validate_sha256("training", &dataset.train_sha256)?;
            if !ids.insert(dataset.dataset_id.as_str())
                || dataset.feature_count == 0
                || dataset.stable_row_ids.is_empty()
                || dataset.stable_row_ids.len() != dataset.atomic_groups.len()
                || dataset.stable_row_ids.iter().any(String::is_empty)
                || dataset.atomic_groups.iter().any(String::is_empty)
                || dataset.stable_row_ids.iter().collect::<BTreeSet<_>>().len()
                    != dataset.stable_row_ids.len()
            {
                return Err("train-only dataset registry row is incomplete or duplicate".into());
            }
            let train_path = safe_registry_path(root, &dataset.train_csv_path)?;
            if train_path.file_name().and_then(|value| value.to_str()) != Some("train.csv") {
                return Err("train-only registry dataset path must end in train.csv".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeHyperionTrainOnlyRegistryV2 {
    pub schema: String,
    /// Relative storage location.  The effective cache root appends the
    /// registry identity, so two frozen registries can never share entries.
    pub cache_root: String,
    pub encoder: HyperionEncoderIdentityV1,
    pub evaluator: crate::jope_generalized_gp::EvaluatorFreezeV1,
    pub datasets: Vec<TrainOnlyDatasetV1>,
}

impl JopeHyperionTrainOnlyRegistryV2 {
    pub const SCHEMA: &'static str = "JopeHyperionTrainOnlyRegistry.v2";

    pub fn validate(&self, root: &Path) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("train-only registry schema drift".into());
        }
        // Reuse the mature train-only dataset validation without admitting a
        // V1 registry at the gate boundary.
        JopeHyperionTrainOnlyRegistryV1 {
            schema: JopeHyperionTrainOnlyRegistryV1::SCHEMA.into(),
            encoder: self.encoder.clone(),
            evaluator: self.evaluator.clone(),
            datasets: self.datasets.clone(),
        }
        .validate(root)?;
        let cache_root = safe_registry_path(root, &self.cache_root)?;
        if cache_root == root
            || Path::new(&self.cache_root)
                .components()
                .all(|component| component == Component::CurDir)
            || self.cache_root.ends_with("train.csv")
        {
            return Err("registry cache root must be a dedicated relative directory".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate(Path::new("."))?;
        canonical_sha256(self)
    }

    pub fn resolved_cache_root(&self, registry_root: &Path) -> Result<PathBuf, String> {
        self.validate(registry_root)?;
        Ok(safe_registry_path(registry_root, &self.cache_root)?.join(self.identity()?))
    }
}

pub fn validate_opaque_id(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("dataset ID must be an opaque 1..=128 character ASCII identifier".into());
    }
    Ok(())
}

pub fn safe_registry_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.as_os_str().is_empty()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("train-only registry path must be a safe relative path".into());
    }
    Ok(root.join(path))
}

#[derive(Debug, Clone, PartialEq)]
pub struct FittedFeatureBlocksV1 {
    cell_sorted: Vec<Vec<f64>>,
    row_means: Vec<f64>,
    row_scales: Vec<f64>,
}

impl FittedFeatureBlocksV1 {
    pub fn fit(cell: &[Vec<f64>], row: &[Vec<f64>], raw_width: usize) -> Result<Self, String> {
        Self::fit_available(cell, Some(row), raw_width)
    }

    pub fn fit_available(
        cell: &[Vec<f64>],
        row: Option<&[Vec<f64>]>,
        raw_width: usize,
    ) -> Result<Self, String> {
        validate_matrix(cell, raw_width, "cell fit")?;
        if let Some(row) = row {
            validate_matrix(row, ROW_EMBEDDING_WIDTH, "row fit")?;
            if cell.len() != row.len() {
                return Err("cell and row fit sidecars are row-misaligned".into());
            }
        }
        let mut cell_sorted = vec![Vec::with_capacity(cell.len()); raw_width];
        for values in cell {
            for (column, value) in values.iter().enumerate() {
                cell_sorted[column].push(*value);
            }
        }
        for values in &mut cell_sorted {
            values.sort_by(f64::total_cmp);
        }
        let mut row_means = Vec::new();
        let mut row_scales = Vec::new();
        if let Some(row) = row {
            row_means = vec![0.0; ROW_EMBEDDING_WIDTH];
            for values in row {
                for (column, value) in values.iter().enumerate() {
                    row_means[column] += value / row.len() as f64;
                }
            }
            row_scales = vec![0.0; ROW_EMBEDDING_WIDTH];
            for values in row {
                for (column, value) in values.iter().enumerate() {
                    row_scales[column] += (value - row_means[column]).powi(2) / row.len() as f64;
                }
            }
            row_scales
                .iter_mut()
                .for_each(|value| *value = value.sqrt().max(1e-12));
        }
        Ok(Self {
            cell_sorted,
            row_means,
            row_scales,
        })
    }

    pub fn transform_cell(&self, values: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        validate_matrix(values, self.cell_sorted.len(), "cell transform")?;
        Ok(values
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        let sorted = &self.cell_sorted[column];
                        let lower = sorted.partition_point(|candidate| candidate < value);
                        let upper = sorted.partition_point(|candidate| candidate <= value);
                        (lower + upper) as f64 / (2.0 * sorted.len() as f64)
                    })
                    .collect()
            })
            .collect())
    }

    pub fn transform_row(&self, values: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        if self.row_means.len() != ROW_EMBEDDING_WIDTH
            || self.row_scales.len() != ROW_EMBEDDING_WIDTH
        {
            return Err("row128 preprocessing was not fitted".into());
        }
        validate_matrix(values, ROW_EMBEDDING_WIDTH, "row transform")?;
        Ok(values
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        (value - self.row_means[column]) / self.row_scales[column]
                    })
                    .collect()
            })
            .collect())
    }
}

pub fn assemble_feature_view(
    view: FeatureViewKindV1,
    raw: &[Vec<f64>],
    cell: &[Vec<f64>],
    row: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let raw_width = raw.first().map(Vec::len).unwrap_or(0);
    validate_matrix(raw, raw_width, "raw")?;
    if matches!(
        view,
        FeatureViewKindV1::HyperionCell | FeatureViewKindV1::HyperionBoth
    ) {
        if cell.len() != raw.len() {
            return Err("cell feature block is row-misaligned".into());
        }
        validate_matrix(cell, raw_width, "cell")?;
    }
    if matches!(
        view,
        FeatureViewKindV1::HyperionRow128 | FeatureViewKindV1::HyperionBoth
    ) {
        if row.len() != raw.len() {
            return Err("row128 feature block is row-misaligned".into());
        }
        validate_matrix(row, ROW_EMBEDDING_WIDTH, "row")?;
    }
    Ok((0..raw.len())
        .map(|index| {
            let mut values = raw[index].clone();
            match view {
                FeatureViewKindV1::Raw => {}
                FeatureViewKindV1::HyperionCell => values.extend_from_slice(&cell[index]),
                FeatureViewKindV1::HyperionRow128 => values.extend_from_slice(&row[index]),
                FeatureViewKindV1::HyperionBoth => {
                    values.extend_from_slice(&cell[index]);
                    values.extend_from_slice(&row[index]);
                }
            }
            values
        })
        .collect())
}

fn validate_matrix(values: &[Vec<f64>], width: usize, name: &str) -> Result<(), String> {
    if values.is_empty()
        || width == 0
        || values
            .iter()
            .any(|row| row.len() != width || row.iter().any(|value| !value.is_finite()))
    {
        return Err(format!("{name} matrix has an invalid shape or value"));
    }
    Ok(())
}

fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    (actual
        .iter()
        .zip(predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt()
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

fn bootstrap_lower_bound(
    folds: &[PairedFoldEvidenceV1],
    arm: FeatureViewKindV1,
    confidence: f64,
    seed: u64,
) -> Result<f64, String> {
    let mut groups = BTreeMap::<&str, (Vec<f64>, Vec<f64>)>::new();
    for fold in folds {
        let predictions = &fold.arm_predictions[&arm];
        for (index, prediction) in predictions.iter().enumerate() {
            let entry = groups
                .entry(fold.atomic_groups[index].as_str())
                .or_default();
            entry
                .0
                .push((fold.actual[index] - fold.raw_predictions[index]).powi(2));
            entry.1.push((fold.actual[index] - *prediction).powi(2));
        }
    }
    if groups.len() < GateThresholdsV1::default().minimum_atomic_groups as usize {
        return Err("insufficient atomic groups for paired bootstrap".into());
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut state = seed.max(1);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut raw_sum = 0.0;
        let mut arm_sum = 0.0;
        let mut rows = 0usize;
        for _ in 0..groups.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let group = &groups[state as usize % groups.len()];
            raw_sum += group.0.iter().sum::<f64>();
            arm_sum += group.1.iter().sum::<f64>();
            rows += group.0.len();
        }
        let raw_value = (raw_sum / rows as f64).sqrt();
        let arm_value = (arm_sum / rows as f64).sqrt();
        let improvement = ((raw_value + 1e-300) / (arm_value + 1e-300)).ln();
        samples.push(improvement);
    }
    samples.sort_by(f64::total_cmp);
    let quantile = (1.0 - confidence).clamp(0.0, 1.0);
    let index = (quantile * (samples.len() - 1) as f64).floor() as usize;
    Ok(samples[index])
}

fn evaluate_arm(
    stage: GateStageV1,
    folds: &[PairedFoldEvidenceV1],
    arm: FeatureViewKindV1,
    thresholds: &GateThresholdsV1,
) -> Result<StageGateResultV1, String> {
    let mut actual = Vec::new();
    let mut raw_predictions = Vec::new();
    let mut arm_predictions = Vec::new();
    let mut fold_wins = 0u32;
    let mut fold_best_arm_wins = 0u32;
    let mut worst_fold_regression = f64::NEG_INFINITY;
    for fold in folds {
        let arm_values = &fold.arm_predictions[&arm];
        let raw_fold = rmse(&fold.actual, &fold.raw_predictions);
        let arm_fold = rmse(&fold.actual, arm_values);
        if arm_fold < raw_fold {
            fold_wins += 1;
        }
        let relative_regression = if raw_fold <= 1e-300 {
            if arm_fold <= 1e-300 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            arm_fold / raw_fold - 1.0
        };
        worst_fold_regression = worst_fold_regression.max(relative_regression);
        let best = fold
            .arm_predictions
            .iter()
            .min_by(|(left_arm, left), (right_arm, right)| {
                rmse(&fold.actual, left)
                    .total_cmp(&rmse(&fold.actual, right))
                    .then_with(|| left_arm.cmp(right_arm))
            })
            .map(|(value, _)| *value)
            .unwrap();
        if best == arm {
            fold_best_arm_wins += 1;
        }
        actual.extend_from_slice(&fold.actual);
        raw_predictions.extend_from_slice(&fold.raw_predictions);
        arm_predictions.extend_from_slice(arm_values);
    }
    let pooled_raw_rmse = rmse(&actual, &raw_predictions);
    let pooled_arm_rmse = rmse(&actual, &arm_predictions);
    let relative_improvement = if pooled_raw_rmse <= 1e-300 {
        0.0
    } else {
        1.0 - pooled_arm_rmse / pooled_raw_rmse
    };
    let confidence = match stage {
        GateStageV1::Cv5Screen => thresholds.cv5_bootstrap_one_sided_confidence,
        GateStageV1::Cv7Confirmation => thresholds.cv7_bootstrap_one_sided_confidence,
    };
    let bootstrap_lower_bound_log_improvement = bootstrap_lower_bound(
        folds,
        arm,
        confidence,
        seed64(&[&format!("{stage:?}"), &format!("{arm:?}")]),
    )?;
    let wins_pass = match stage {
        GateStageV1::Cv5Screen => {
            fold_wins >= thresholds.cv5_minimum_fold_wins
                && fold_best_arm_wins >= thresholds.cv5_minimum_best_arm_wins
        }
        GateStageV1::Cv7Confirmation => fold_wins >= thresholds.cv7_minimum_fold_wins,
    };
    let passed = wins_pass
        && relative_improvement >= thresholds.minimum_relative_improvement
        && bootstrap_lower_bound_log_improvement > 0.0
        && worst_fold_regression <= thresholds.maximum_fold_regression;
    Ok(StageGateResultV1 {
        stage,
        arm,
        folds: folds.len() as u32,
        fold_wins,
        fold_best_arm_wins,
        pooled_raw_rmse,
        pooled_arm_rmse,
        relative_improvement,
        bootstrap_lower_bound_log_improvement,
        worst_fold_regression,
        passed,
    })
}

pub fn screen_cv5(
    folds: &[PairedFoldEvidenceV1],
    thresholds: &GateThresholdsV1,
) -> Result<(Vec<StageGateResultV1>, Option<FeatureViewKindV1>), String> {
    screen_cv5_arms(folds, &FeatureViewKindV1::EXPANDED, thresholds)
}

/// Screen exactly the independently available arms.  Missing or structurally
/// ineligible row/both representations must never suppress a valid cell arm.
pub fn screen_cv5_arms(
    folds: &[PairedFoldEvidenceV1],
    arms: &[FeatureViewKindV1],
    thresholds: &GateThresholdsV1,
) -> Result<(Vec<StageGateResultV1>, Option<FeatureViewKindV1>), String> {
    if arms.is_empty()
        || arms
            .iter()
            .any(|arm| *arm == FeatureViewKindV1::Raw || !FeatureViewKindV1::EXPANDED.contains(arm))
        || arms.iter().collect::<BTreeSet<_>>().len() != arms.len()
    {
        return Err("CV5 requires a nonempty unique expanded-arm set".into());
    }
    validate_stage_folds(folds, 5, arms)?;
    let mut results = arms
        .iter()
        .map(|arm| evaluate_arm(GateStageV1::Cv5Screen, folds, *arm, thresholds))
        .collect::<Result<Vec<_>, _>>()?;
    results.sort_by_key(|value| value.arm);
    let advanced = results
        .iter()
        .filter(|result| result.passed)
        .min_by(|left, right| {
            right
                .relative_improvement
                .total_cmp(&left.relative_improvement)
                .then_with(|| left.arm.cmp(&right.arm))
        })
        .map(|result| result.arm);
    Ok((results, advanced))
}

pub fn confirm_cv7(
    folds: &[PairedFoldEvidenceV1],
    frozen_arm: FeatureViewKindV1,
    thresholds: &GateThresholdsV1,
) -> Result<StageGateResultV1, String> {
    if frozen_arm == FeatureViewKindV1::Raw {
        return Err("CV7 cannot confirm the raw control as an expansion".into());
    }
    validate_stage_folds(folds, 7, &[frozen_arm])?;
    evaluate_arm(GateStageV1::Cv7Confirmation, folds, frozen_arm, thresholds)
}

fn validate_stage_folds(
    folds: &[PairedFoldEvidenceV1],
    expected_folds: usize,
    arms: &[FeatureViewKindV1],
) -> Result<(), String> {
    if folds.len() != expected_folds {
        return Err(format!(
            "stage requires exactly {expected_folds} paired folds"
        ));
    }
    let mut fold_ids = BTreeSet::new();
    let mut row_ids = BTreeSet::new();
    let mut group_to_fold = BTreeMap::<&str, u32>::new();
    for fold in folds {
        fold.validate(arms)?;
        if fold.row_ids.len() < MINIMUM_FOLD_HOLD_ROWS {
            return Err(format!(
                "fold {} has {} hold rows; minimum is {MINIMUM_FOLD_HOLD_ROWS}",
                fold.fold_index,
                fold.row_ids.len()
            ));
        }
        let hold_group_count = fold.atomic_groups.iter().collect::<BTreeSet<_>>().len();
        if hold_group_count < MINIMUM_FOLD_HOLD_GROUPS {
            return Err(format!(
                "fold {} has {hold_group_count} hold groups; minimum is {MINIMUM_FOLD_HOLD_GROUPS}",
                fold.fold_index
            ));
        }
        if !fold_ids.insert(fold.fold_index) {
            return Err("stage contains duplicate fold identities".into());
        }
        for (row_id, group) in fold.row_ids.iter().zip(&fold.atomic_groups) {
            if !row_ids.insert(row_id.as_str()) {
                return Err("OOF evidence predicts a row more than once".into());
            }
            if group_to_fold
                .insert(group.as_str(), fold.fold_index)
                .is_some_and(|prior| prior != fold.fold_index)
            {
                return Err("atomic group crosses validation folds".into());
            }
        }
    }
    if fold_ids != (0..expected_folds as u32).collect() {
        return Err("stage fold identities are incomplete".into());
    }
    let minimum_groups = (expected_folds * MINIMUM_FOLD_HOLD_GROUPS)
        .max(GateThresholdsV1::default().minimum_atomic_groups as usize);
    if group_to_fold.len() < minimum_groups {
        return Err("stage has insufficient atomic groups".into());
    }
    for fold in folds {
        let fit_rows = row_ids.len().saturating_sub(fold.row_ids.len());
        if fit_rows < MINIMUM_FOLD_FIT_ROWS {
            return Err(format!(
                "fold {} has {fit_rows} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}",
                fold.fold_index
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldPlanV1 {
    pub schema: String,
    pub stage: GateStageV1,
    pub seed: u64,
    pub folds: Vec<GroupedFoldV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldV1 {
    pub fold_index: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub hold_groups: Vec<String>,
}

pub fn build_grouped_fold_plan(
    stage: GateStageV1,
    atomic_groups: &[String],
) -> Result<GroupedFoldPlanV1, String> {
    let fold_count = match stage {
        GateStageV1::Cv5Screen => 5,
        GateStageV1::Cv7Confirmation => 7,
    };
    let seed = match stage {
        GateStageV1::Cv5Screen => CV5_SEED,
        GateStageV1::Cv7Confirmation => CV7_SEED,
    };
    let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
    for (row, group) in atomic_groups.iter().enumerate() {
        if group.is_empty() {
            return Err("atomic group must not be empty".into());
        }
        grouped.entry(group).or_default().push(row);
    }
    let minimum_groups = (fold_count * MINIMUM_FOLD_HOLD_GROUPS)
        .max(GateThresholdsV1::default().minimum_atomic_groups as usize);
    if grouped.len() < minimum_groups {
        return Err("insufficient atomic groups for the Hyperion gate".into());
    }
    let mut groups = grouped.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(group, _)| seed64(&[&seed.to_string(), group]));
    let mut assignments = vec![Vec::<(&str, Vec<usize>)>::new(); fold_count];
    let mut sizes = vec![0usize; fold_count];
    for group in groups {
        let fold = (0..fold_count)
            .min_by_key(|index| (sizes[*index], *index))
            .unwrap();
        sizes[fold] += group.1.len();
        assignments[fold].push(group);
    }
    let all_rows = (0..atomic_groups.len()).collect::<BTreeSet<_>>();
    let folds = assignments
        .into_iter()
        .enumerate()
        .map(|(fold_index, values)| {
            let mut hold_rows = values
                .iter()
                .flat_map(|(_, rows)| rows.iter().copied())
                .collect::<Vec<_>>();
            hold_rows.sort_unstable();
            let hold_set = hold_rows.iter().copied().collect::<BTreeSet<_>>();
            let fit_rows = all_rows.difference(&hold_set).copied().collect();
            let mut hold_groups = values
                .into_iter()
                .map(|(group, _)| group.to_owned())
                .collect::<Vec<_>>();
            hold_groups.sort();
            GroupedFoldV1 {
                fold_index: fold_index as u32,
                fit_rows,
                hold_rows,
                hold_groups,
            }
        })
        .collect::<Vec<_>>();
    for fold in &folds {
        if fold.hold_rows.len() < MINIMUM_FOLD_HOLD_ROWS {
            return Err(format!(
                "fold {} has {} hold rows; minimum is {MINIMUM_FOLD_HOLD_ROWS}",
                fold.fold_index,
                fold.hold_rows.len()
            ));
        }
        if fold.hold_groups.len() < MINIMUM_FOLD_HOLD_GROUPS {
            return Err(format!(
                "fold {} has {} hold groups; minimum is {MINIMUM_FOLD_HOLD_GROUPS}",
                fold.fold_index,
                fold.hold_groups.len()
            ));
        }
        if fold.fit_rows.len() < MINIMUM_FOLD_FIT_ROWS {
            return Err(format!(
                "fold {} has {} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}",
                fold.fold_index,
                fold.fit_rows.len()
            ));
        }
    }
    Ok(GroupedFoldPlanV1 {
        schema: "JopeHyperionGroupedFoldPlan.v1".into(),
        stage,
        seed,
        folds,
    })
}

fn validate_plan_feature_feasibility(
    plan: &GroupedFoldPlanV1,
    arms: &[FeatureViewKindV1],
    raw_features: usize,
) -> Result<(), String> {
    for fold in &plan.folds {
        for arm in arms {
            validate_feature_view_feasibility(*arm, raw_features, fold.fit_rows.len()).map_err(
                |error| {
                    format!(
                        "fold {} is structurally ineligible: {error}",
                        fold.fold_index
                    )
                },
            )?;
        }
    }
    Ok(())
}

/// Return the arms that are feasible on every fold of both frozen stages,
/// together with deterministic reasons for each excluded arm.  Eligibility is
/// deliberately arm-local: an unsafe row128 arm cannot disable cell.
pub fn independently_feasible_feature_views(
    cv5: &GroupedFoldPlanV1,
    cv7: &GroupedFoldPlanV1,
    available: &[FeatureViewKindV1],
    raw_features: usize,
) -> (Vec<FeatureViewKindV1>, BTreeMap<FeatureViewKindV1, String>) {
    let mut eligible = Vec::new();
    let mut excluded = BTreeMap::new();
    for arm in available.iter().copied() {
        if arm == FeatureViewKindV1::Raw
            || !FeatureViewKindV1::EXPANDED.contains(&arm)
            || eligible.contains(&arm)
            || excluded.contains_key(&arm)
        {
            excluded.insert(arm, "arm is raw, unknown, or duplicated".into());
            continue;
        }
        let result = validate_plan_feature_feasibility(cv5, &[arm], raw_features)
            .and_then(|()| validate_plan_feature_feasibility(cv7, &[arm], raw_features));
        match result {
            Ok(()) => eligible.push(arm),
            Err(reason) => {
                excluded.insert(arm, reason);
            }
        }
    }
    eligible.sort_unstable();
    (eligible, excluded)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_fallback_decision(
    dataset_id: &str,
    training_sha256: &str,
    predictor_sha256: &str,
    encoder_sha256: &str,
    cv5_plan_sha256: &str,
    cv7_plan_sha256: &str,
    reason: impl Into<String>,
    started: Instant,
) -> JopeHyperionGateDecisionV1 {
    JopeHyperionGateDecisionV1 {
        schema: JopeHyperionGateDecisionV1::SCHEMA.into(),
        dataset_id_sha256: raw_sha256(dataset_id.as_bytes()),
        source_training_sha256: training_sha256.into(),
        raw_predictor_sha256: predictor_sha256.into(),
        encoder_identity_sha256: encoder_sha256.into(),
        cv5_fold_plan_sha256: cv5_plan_sha256.into(),
        cv7_fold_plan_sha256: cv7_plan_sha256.into(),
        thresholds: GateThresholdsV1::default(),
        cv5_candidates: Vec::new(),
        cv5_advanced_arm: None,
        cv7_confirmation: None,
        selected_view: FeatureViewKindV1::Raw,
        raw_fallback_reason: Some(reason.into()),
        fold_receipt_sha256s: Vec::new(),
        encoding_milliseconds: 0,
        scoring_milliseconds: 0,
        wall_clock_milliseconds: u64::try_from(started.elapsed().as_millis())
            .unwrap_or(u64::MAX)
            .min(WALL_CLOCK_BUDGET.as_millis() as u64),
    }
}

pub fn write_gate_artifacts(
    artifact_root: &Path,
    decision: &JopeHyperionGateDecisionV1,
    predictor: &FrozenPredictorV1,
    raw_features: usize,
) -> Result<(), String> {
    decision.validate()?;
    predictor.validate()?;
    fs::create_dir_all(artifact_root)
        .map_err(|error| format!("create gate artifact root: {error}"))?;
    let decision_bytes = canonical_json_bytes(decision)?;
    let decision_sha256 = raw_sha256(&decision_bytes);
    write_immutable(
        &artifact_root.join("gate-decision.v1.json"),
        &decision_bytes,
    )?;
    let feature_view = JopeFeatureViewV1::new(decision.selected_view, raw_features)?;
    let predictor_sha256 = predictor.semantic_sha256()?;
    let bound = JopeFeatureBoundPredictorV1 {
        schema: JopeFeatureBoundPredictorV1::SCHEMA.into(),
        feature_view_sha256: feature_view.identity()?,
        selected_view: feature_view,
        gate_decision_sha256: decision_sha256,
        selected_predictor: predictor.clone(),
        selected_predictor_sha256: predictor_sha256.clone(),
        raw_fallback_predictor: predictor.clone(),
        raw_fallback_predictor_sha256: predictor_sha256,
    };
    bound.validate()?;
    write_immutable(
        &artifact_root.join("feature-bound-predictor.v1.json"),
        &canonical_json_bytes(&bound)?,
    )
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        let present =
            fs::read(path).map_err(|error| format!("read existing {}: {error}", path.display()))?;
        if present == bytes {
            return Ok(());
        }
        return Err(format!(
            "refusing to replace differing immutable artifact {}",
            path.display()
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create immutable {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write immutable {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync immutable {}: {error}", path.display()))
}

#[derive(Debug, Clone)]
struct RegressionTable {
    features: Vec<Vec<f64>>,
    targets: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FoldEncoding {
    cell_fit: Vec<Vec<f64>>,
    cell_hold: Vec<Vec<f64>>,
    row_fit: Vec<Vec<f64>>,
    row_hold: Vec<Vec<f64>>,
    sidecars: Vec<MatrixArtifactV1>,
    cache_key_sha256: String,
    cache_receipt_sha256: String,
    milliseconds: u64,
    cache_replay: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HyperionCacheKeyV2 {
    pub schema: String,
    pub registry_sha256: String,
    pub dataset_id_sha256: String,
    pub source_training_sha256: String,
    pub stable_row_ids_sha256: String,
    pub atomic_groups_sha256: String,
    pub stage: String,
    pub fold_plan_sha256: String,
    pub fold_index: Option<u32>,
    pub fit_row_ids_sha256: String,
    pub hold_row_ids_sha256: String,
    pub fit_input_sha256: String,
    pub hold_input_sha256: String,
    pub encoder_identity_sha256: String,
    pub encoder_binary_sha256: String,
    pub encoder_weights_sha256: String,
    pub encoder_manifest_sha256: String,
    pub encoder_source_hashes_sha256: String,
    pub seed: u64,
    pub sweeps: u32,
    pub scales: Vec<u32>,
    pub batch_size: u32,
    pub runtime_versions_sha256: String,
    pub observed_device: String,
}

impl HyperionCacheKeyV2 {
    pub const SCHEMA: &'static str = "JopeHyperionCacheKey.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !matches!(self.stage.as_str(), "full_train" | "cv5" | "cv7")
            || (self.stage == "full_train") != self.fold_index.is_none()
            || self.seed == 0
            || self.sweeps == 0
            || self.scales.is_empty()
            || self.scales.contains(&0)
            || self.batch_size == 0
            || self.observed_device != "cuda:0"
        {
            return Err("Hyperion cache key is incomplete".into());
        }
        for (name, digest) in [
            ("registry", &self.registry_sha256),
            ("dataset ID", &self.dataset_id_sha256),
            ("training", &self.source_training_sha256),
            ("row IDs", &self.stable_row_ids_sha256),
            ("atomic groups", &self.atomic_groups_sha256),
            ("fold plan", &self.fold_plan_sha256),
            ("fit row IDs", &self.fit_row_ids_sha256),
            ("hold row IDs", &self.hold_row_ids_sha256),
            ("fit input", &self.fit_input_sha256),
            ("hold input", &self.hold_input_sha256),
            ("encoder identity", &self.encoder_identity_sha256),
            ("encoder binary", &self.encoder_binary_sha256),
            ("encoder weights", &self.encoder_weights_sha256),
            ("encoder manifest", &self.encoder_manifest_sha256),
            ("encoder sources", &self.encoder_source_hashes_sha256),
            ("runtime versions", &self.runtime_versions_sha256),
        ] {
            validate_sha256(name, digest)?;
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
pub struct EncoderExecutionProofV2 {
    pub encoder_identity_sha256: String,
    pub gpu_uuid: String,
    pub cuda_visible_devices: String,
    pub argv_sha256: String,
    pub environment_sha256: String,
    pub observed_device: String,
    pub stdout: String,
    pub stdout_sha256: String,
    pub stderr: String,
    pub stderr_sha256: String,
    pub elapsed_milliseconds: u64,
}

impl EncoderExecutionProofV2 {
    pub fn validate(&self, expected_ordinal: u32) -> Result<(), String> {
        let expected = format!("cuda:{expected_ordinal}");
        self.validate_report(&expected)
    }

    /// Validate the report emitted after CUDA_VISIBLE_DEVICES has masked one
    /// scheduler-assigned GPU.  CUDA numbering is then intentionally local:
    /// the only valid device report is `cuda:0`.
    pub fn validate_namespace(&self) -> Result<(), String> {
        self.validate_report("cuda:0")
    }

    fn validate_report(&self, expected: &str) -> Result<(), String> {
        validate_sha256("encoder identity", &self.encoder_identity_sha256)?;
        validate_sha256("Hyperion argv", &self.argv_sha256)?;
        validate_sha256("Hyperion environment", &self.environment_sha256)?;
        if self.argv_sha256 != canonical_sha256(&hyperion_argv_schema_v2())?
            || self.environment_sha256 != canonical_sha256(&hyperion_environment_v2())?
        {
            return Err("Hyperion execution contract hash drifted".into());
        }
        if self.gpu_uuid.trim().is_empty()
            || self.cuda_visible_devices != self.gpu_uuid
            || self.observed_device != expected
            || raw_sha256(self.stdout.as_bytes()) != self.stdout_sha256
            || raw_sha256(self.stderr.as_bytes()) != self.stderr_sha256
            || !encoder_output_proves_device(&self.stdout, &self.stderr, &expected)
        {
            return Err("captured encoder output does not prove pinned CUDA execution".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldEncodingCacheV2 {
    pub schema: String,
    pub key: HyperionCacheKeyV2,
    pub key_sha256: String,
    pub fit_row_order_sha256: String,
    pub hold_row_order_sha256: String,
    pub hold_target_policy: String,
    pub sidecars: Vec<MatrixArtifactV1>,
    pub proof: EncoderExecutionProofV2,
}

impl FoldEncodingCacheV2 {
    pub const SCHEMA: &'static str = "JopeHyperionFoldEncodingCache.v2";
    pub const DUMMY_HOLD_TARGET_POLICY: &'static str = "constant_zero_no_hold_labels.v1";

    pub fn validate(&self, expected_ordinal: u32) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.key.stage == "full_train"
            || self.key.identity()? != self.key_sha256
            || self.fit_row_order_sha256 != self.key.fit_row_ids_sha256
            || self.hold_row_order_sha256 != self.key.hold_row_ids_sha256
            || self.hold_target_policy != Self::DUMMY_HOLD_TARGET_POLICY
        {
            return Err("fold-local encoding cache identity drifted".into());
        }
        let _ = expected_ordinal;
        self.proof.validate_namespace()?;
        if self.proof.encoder_identity_sha256 != self.key.encoder_identity_sha256 {
            return Err("fold cache execution proof names a different encoder".into());
        }
        validate_sidecar_set(&self.sidecars, true)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullTrainEncodingCacheV2 {
    pub schema: String,
    pub key: HyperionCacheKeyV2,
    pub key_sha256: String,
    pub train_row_order_sha256: String,
    pub sidecars: Vec<MatrixArtifactV1>,
    pub proof: EncoderExecutionProofV2,
}

impl FullTrainEncodingCacheV2 {
    pub const SCHEMA: &'static str = "JopeHyperionFullTrainEncodingCache.v2";

    pub fn validate(&self, expected_ordinal: u32) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.key.stage != "full_train"
            || self.key.identity()? != self.key_sha256
            || self.train_row_order_sha256 != self.key.fit_row_ids_sha256
        {
            return Err("full-train encoding cache identity drifted".into());
        }
        let _ = expected_ordinal;
        self.proof.validate_namespace()?;
        if self.proof.encoder_identity_sha256 != self.key.encoder_identity_sha256 {
            return Err("full-train cache execution proof names a different encoder".into());
        }
        validate_sidecar_set(&self.sidecars, false)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheCoverageEntryV2 {
    pub fold_index: Option<u32>,
    pub cache_key_sha256: String,
    pub cache_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCacheCoverageV2 {
    pub dataset_id: String,
    pub full_train: CacheCoverageEntryV2,
    pub cv5: Vec<CacheCoverageEntryV2>,
    pub cv7: Vec<CacheCoverageEntryV2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryCacheCoverageV2 {
    pub schema: String,
    pub registry_sha256: String,
    pub datasets: Vec<DatasetCacheCoverageV2>,
}

impl RegistryCacheCoverageV2 {
    pub const SCHEMA: &'static str = "JopeHyperionRegistryCacheCoverage.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err("registry cache coverage schema drift".into());
        }
        validate_sha256("registry", &self.registry_sha256)?;
        let mut ids = BTreeSet::new();
        for dataset in &self.datasets {
            validate_opaque_id(&dataset.dataset_id)?;
            if !ids.insert(dataset.dataset_id.as_str())
                || dataset.full_train.fold_index.is_some()
                || dataset.cv5.len() != 5
                || dataset.cv7.len() != 7
                || dataset
                    .cv5
                    .iter()
                    .enumerate()
                    .any(|(fold, entry)| entry.fold_index != Some(fold as u32))
                || dataset
                    .cv7
                    .iter()
                    .enumerate()
                    .any(|(fold, entry)| entry.fold_index != Some(fold as u32))
            {
                return Err("registry cache coverage is incomplete or duplicated".into());
            }
            for entry in std::iter::once(&dataset.full_train)
                .chain(&dataset.cv5)
                .chain(&dataset.cv7)
            {
                validate_sha256("cache key", &entry.cache_key_sha256)?;
                validate_sha256("cache receipt", &entry.cache_receipt_sha256)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalTrain3EntryV1 {
    pub dataset_id: String,
    pub relative_path: String,
    pub state: String,
    pub byte_length: u64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalTrain3InventoryV1 {
    pub schema: String,
    pub registry_sha256: String,
    pub entries: Vec<HistoricalTrain3EntryV1>,
    pub cv_authority: bool,
}

/// Immutable authority for the train-only feature matrices that already live
/// beside each dataset's `train.csv`.  The path grammar cannot represent a
/// test matrix and the receipt records that no test artifact was opened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputedTrainFeatureReceiptV1 {
    pub schema: String,
    pub registry_sha256: String,
    pub dataset_id: String,
    pub dataset_id_sha256: String,
    pub source_training_sha256: String,
    pub stable_row_ids_sha256: String,
    pub encoder_identity_sha256: String,
    pub row_count: u32,
    pub raw_feature_count: u32,
    pub cell_relative_path: String,
    pub cell: MatrixArtifactV1,
    pub row128_relative_path: Option<String>,
    pub row128: Option<MatrixArtifactV1>,
    pub row128_unavailable_reason: Option<String>,
    pub test_artifacts_read: bool,
}

impl PrecomputedTrainFeatureReceiptV1 {
    pub const SCHEMA: &'static str = "JopeHyperionPrecomputedTrainFeatures.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.row_count == 0
            || self.raw_feature_count == 0
            || self.test_artifacts_read
        {
            return Err("precomputed train-feature receipt schema or boundary drifted".into());
        }
        validate_opaque_id(&self.dataset_id)?;
        for (name, digest) in [
            ("registry", &self.registry_sha256),
            ("dataset ID", &self.dataset_id_sha256),
            ("training", &self.source_training_sha256),
            ("stable rows", &self.stable_row_ids_sha256),
            ("encoder", &self.encoder_identity_sha256),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.dataset_id_sha256 != raw_sha256(self.dataset_id.as_bytes())
            || Path::new(&self.cell_relative_path)
                .file_name()
                .and_then(|value| value.to_str())
                != Some("train3.vs")
        {
            return Err("precomputed cell authority is not the train3.vs sibling".into());
        }
        self.cell.validate()?;
        if self.cell.role != "cell_train_precomputed"
            || self.cell.rows != self.row_count
            || self.cell.columns != self.raw_feature_count
        {
            return Err("precomputed cell matrix shape or role drifted".into());
        }
        match (
            &self.row128_relative_path,
            &self.row128,
            &self.row128_unavailable_reason,
        ) {
            (Some(path), Some(row), None) => {
                if Path::new(path).file_name().and_then(|value| value.to_str())
                    != Some("hyperion3train.vs")
                {
                    return Err("precomputed row128 authority names a foreign sidecar".into());
                }
                row.validate()?;
                if row.role != "row128_train_precomputed"
                    || row.rows != self.row_count
                    || row.columns != ROW_EMBEDDING_WIDTH as u32
                {
                    return Err("precomputed row128 matrix shape or role drifted".into());
                }
            }
            (None, None, Some(reason)) if !reason.trim().is_empty() => {}
            _ => {
                return Err(
                    "precomputed row128 availability must be explicit and internally consistent"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn available_views(&self) -> Vec<FeatureViewKindV1> {
        let mut views = vec![FeatureViewKindV1::HyperionCell];
        if self.row128.is_some() {
            views.extend([
                FeatureViewKindV1::HyperionRow128,
                FeatureViewKindV1::HyperionBoth,
            ]);
        }
        views
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputedTrainFeatureCoverageV1 {
    pub schema: String,
    pub registry_sha256: String,
    pub datasets: Vec<PrecomputedTrainFeatureReceiptV1>,
    pub test_artifacts_read: bool,
}

impl PrecomputedTrainFeatureCoverageV1 {
    pub const SCHEMA: &'static str = "JopeHyperionPrecomputedTrainFeatureCoverage.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.datasets.is_empty() || self.test_artifacts_read {
            return Err(
                "precomputed train-feature coverage is empty or crossed the test boundary".into(),
            );
        }
        validate_sha256("registry", &self.registry_sha256)?;
        let mut ids = BTreeSet::new();
        for receipt in &self.datasets {
            receipt.validate()?;
            if receipt.registry_sha256 != self.registry_sha256
                || !ids.insert(receipt.dataset_id.as_str())
                || receipt.test_artifacts_read
            {
                return Err("precomputed train-feature coverage is foreign or duplicated".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone)]
struct PrecomputedTrainFeatureBlocks {
    cell: Vec<Vec<f64>>,
    row128: Option<Vec<Vec<f64>>>,
    receipt: PrecomputedTrainFeatureReceiptV1,
}

impl HistoricalTrain3InventoryV1 {
    pub const SCHEMA: &'static str = "JopeHyperionHistoricalTrain3Inventory.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.cv_authority {
            return Err("historical train3 inventory must be non-authoritative for CV".into());
        }
        validate_sha256("registry", &self.registry_sha256)?;
        let mut ids = BTreeSet::new();
        for entry in &self.entries {
            validate_opaque_id(&entry.dataset_id)?;
            if !ids.insert(entry.dataset_id.as_str())
                || !entry.relative_path.ends_with("train3.vs")
                || !matches!(entry.state.as_str(), "missing" | "zero_byte" | "present")
                || (entry.state == "present") != (entry.byte_length > 0)
                || (entry.state == "present") != entry.sha256.is_some()
            {
                return Err("historical train3 inventory entry is malformed".into());
            }
            if let Some(digest) = &entry.sha256 {
                validate_sha256("historical train3", digest)?;
            }
        }
        Ok(())
    }
}

fn validate_sidecar_set(sidecars: &[MatrixArtifactV1], fold: bool) -> Result<(), String> {
    for sidecar in sidecars {
        sidecar.validate()?;
    }
    let actual = sidecars
        .iter()
        .map(|sidecar| sidecar.role.as_str())
        .collect::<BTreeSet<_>>();
    let expected = if fold {
        ["cell_fit", "cell_hold", "row_fit", "row_hold"]
            .into_iter()
            .collect()
    } else {
        ["cell_train", "row_train"].into_iter().collect()
    };
    if actual != expected || actual.len() != sidecars.len() {
        return Err("Hyperion cache has incomplete or duplicate sidecars".into());
    }
    Ok(())
}

/// Accept only an exact whitespace-delimited report from the encoder.  GPU
/// inventory, environment variables, and lookalike text such as
/// `claimed_device=cuda:0` are deliberately insufficient.
pub fn encoder_output_proves_device(stdout: &str, stderr: &str, expected: &str) -> bool {
    let token = format!("device={expected}");
    let reports = stdout
        .split_ascii_whitespace()
        .chain(stderr.split_ascii_whitespace())
        .filter(|value| value.starts_with("device="))
        .collect::<Vec<_>>();
    reports == [token.as_str()]
}

#[derive(Debug)]
struct StageExecution {
    evidence: Vec<PairedFoldEvidenceV1>,
    receipts: Vec<JopeHyperionFoldReceiptV1>,
    encoding_milliseconds: u64,
    scoring_milliseconds: u64,
}

fn precomputed_fold_encoding(
    blocks: &PrecomputedTrainFeatureBlocks,
    fold: &GroupedFoldV1,
    plan_sha256: &str,
) -> Result<FoldEncoding, String> {
    let cell_fit = select_rows(&blocks.cell, &fold.fit_rows);
    let cell_hold = select_rows(&blocks.cell, &fold.hold_rows);
    let mut sidecars = vec![
        MatrixArtifactV1 {
            role: "cell_fit".into(),
            rows: cell_fit.len() as u32,
            columns: blocks.receipt.raw_feature_count,
            sha256: hash_matrix(&cell_fit),
        },
        MatrixArtifactV1 {
            role: "cell_hold".into(),
            rows: cell_hold.len() as u32,
            columns: blocks.receipt.raw_feature_count,
            sha256: hash_matrix(&cell_hold),
        },
    ];
    let (row_fit, row_hold) = if let Some(row128) = &blocks.row128 {
        let fit = select_rows(row128, &fold.fit_rows);
        let hold = select_rows(row128, &fold.hold_rows);
        sidecars.extend([
            MatrixArtifactV1 {
                role: "row_fit".into(),
                rows: fit.len() as u32,
                columns: ROW_EMBEDDING_WIDTH as u32,
                sha256: hash_matrix(&fit),
            },
            MatrixArtifactV1 {
                role: "row_hold".into(),
                rows: hold.len() as u32,
                columns: ROW_EMBEDDING_WIDTH as u32,
                sha256: hash_matrix(&hold),
            },
        ]);
        (fit, hold)
    } else {
        (Vec::new(), Vec::new())
    };
    let receipt_sha256 = blocks.receipt.identity()?;
    let cache_key_sha256 = canonical_sha256(&(
        "precomputed-train-feature-fold.v1",
        &receipt_sha256,
        plan_sha256,
        fold.fold_index,
        hash_matrix(&cell_fit),
        hash_matrix(&cell_hold),
    ))?;
    Ok(FoldEncoding {
        cell_fit,
        cell_hold,
        row_fit,
        row_hold,
        sidecars,
        cache_key_sha256,
        cache_receipt_sha256: receipt_sha256,
        milliseconds: 0,
        cache_replay: true,
    })
}

fn sidecars_for_view(
    sidecars: &[MatrixArtifactV1],
    view: FeatureViewKindV1,
) -> Vec<MatrixArtifactV1> {
    sidecars
        .iter()
        .filter(|sidecar| match view {
            FeatureViewKindV1::Raw => false,
            FeatureViewKindV1::HyperionCell => sidecar.role.starts_with("cell_"),
            FeatureViewKindV1::HyperionRow128 => sidecar.role.starts_with("row_"),
            FeatureViewKindV1::HyperionBoth => true,
        })
        .cloned()
        .collect()
}

/// Execute the train-only gate from its four authorized inputs.  Operational
/// failures after registry/predictor authorization are materialized as a raw
/// decision; malformed authority inputs are rejected without guessing.
pub fn run_gate_command(
    dataset_id: &str,
    registry_path: &Path,
    predictor_path: &Path,
    artifact_root: &Path,
) -> Result<JopeHyperionGateDecisionV1, String> {
    let started = Instant::now();
    validate_opaque_id(dataset_id)?;
    let registry_bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&registry_bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let dataset = registry
        .datasets
        .iter()
        .find(|value| value.dataset_id == dataset_id)
        .ok_or_else(|| "opaque dataset ID is absent from the train-only registry".to_string())?;
    let predictor_bytes =
        fs::read(predictor_path).map_err(|error| format!("read frozen AFPM predictor: {error}"))?;
    let predictor: FrozenPredictorV1 = serde_json::from_slice(&predictor_bytes)
        .map_err(|error| format!("decode frozen AFPM predictor: {error}"))?;
    predictor.validate()?;
    if predictor.license_track() != crate::jope_dlgp_best5::LicenseTrackV1::ProductionCompatible
        || predictor.pipelines().len() != predictor.component_count()
    {
        return Err("Hyperion gate requires a production-compatible pure-AFPM predictor".into());
    }
    let predictor_sha256 = predictor.semantic_sha256()?;
    let encoder_sha256 = registry.encoder.identity()?;
    let unavailable_cv5_sha256 =
        raw_sha256(format!("unavailable-cv5-plan-v1|{}", dataset.train_sha256).as_bytes());
    let unavailable_cv7_sha256 =
        raw_sha256(format!("unavailable-cv7-plan-v1|{}", dataset.train_sha256).as_bytes());
    let preflight = (|| -> Result<(RegressionTable, PrecomputedTrainFeatureBlocks), String> {
        let table = read_authorized_table(registry_root, dataset)?;
        let precomputed =
            load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)?;
        require_frozen_precomputed_dataset_binding(&registry, registry_root, &precomputed.receipt)?;
        let cache_receipt_root = artifact_root.join("cache-receipts");
        fs::create_dir_all(&cache_receipt_root)
            .map_err(|error| format!("create cache receipt root: {error}"))?;
        write_immutable(
            &cache_receipt_root.join("precomputed-train-features.v1.json"),
            &canonical_json_bytes(&precomputed.receipt)?,
        )?;
        Ok((table, precomputed))
    })();
    let (authorized_table, precomputed) = match preflight {
        Ok(value) => value,
        Err(reason) => {
            let decision = raw_fallback_decision(
                dataset_id,
                &dataset.train_sha256,
                &predictor_sha256,
                &encoder_sha256,
                &unavailable_cv5_sha256,
                &unavailable_cv7_sha256,
                reason,
                started,
            );
            write_gate_artifacts(
                artifact_root,
                &decision,
                &predictor,
                dataset.feature_count as usize,
            )?;
            return Ok(decision);
        }
    };
    let cv5_plan = match build_grouped_fold_plan(GateStageV1::Cv5Screen, &dataset.atomic_groups) {
        Ok(plan) => plan,
        Err(reason) => {
            let decision = raw_fallback_decision(
                dataset_id,
                &dataset.train_sha256,
                &predictor_sha256,
                &encoder_sha256,
                &unavailable_cv5_sha256,
                &unavailable_cv7_sha256,
                reason,
                started,
            );
            write_gate_artifacts(
                artifact_root,
                &decision,
                &predictor,
                dataset.feature_count as usize,
            )?;
            return Ok(decision);
        }
    };
    let cv7_plan =
        match build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &dataset.atomic_groups) {
            Ok(plan) => plan,
            Err(reason) => {
                let decision = raw_fallback_decision(
                    dataset_id,
                    &dataset.train_sha256,
                    &predictor_sha256,
                    &encoder_sha256,
                    &canonical_sha256(&cv5_plan)?,
                    &unavailable_cv7_sha256,
                    reason,
                    started,
                );
                write_gate_artifacts(
                    artifact_root,
                    &decision,
                    &predictor,
                    dataset.feature_count as usize,
                )?;
                return Ok(decision);
            }
        };
    let cv5_plan_sha256 = canonical_sha256(&cv5_plan)?;
    let cv7_plan_sha256 = canonical_sha256(&cv7_plan)?;

    let operational = (|| -> Result<JopeHyperionGateDecisionV1, String> {
        let (eligible_arms, excluded_arms) = independently_feasible_feature_views(
            &cv5_plan,
            &cv7_plan,
            &precomputed.receipt.available_views(),
            dataset.feature_count as usize,
        );
        if eligible_arms.is_empty() {
            return Err(format!(
                "no precomputed Hyperion arm has sufficient CV evidence: {excluded_arms:?}"
            ));
        }
        registry.evaluator.verify_artifact()?;
        let work_root = artifact_root.join("work");
        let receipt_root = artifact_root.join("fold-receipts");
        fs::create_dir_all(&receipt_root)
            .map_err(|error| format!("create fold receipt root: {error}"))?;
        let adapter = CorpusEvalAdapter::new(
            registry.evaluator.clone(),
            work_root.join("scoring"),
            SCORING_BUDGET.as_secs(),
        )?;
        let cv5 = execute_stage(
            &registry.encoder,
            dataset,
            &authorized_table,
            &precomputed,
            &predictor,
            &adapter,
            &cv5_plan,
            &cv5_plan_sha256,
            &eligible_arms,
            &receipt_root,
            started,
            0,
            0,
        )?;
        let thresholds = GateThresholdsV1::default();
        let (cv5_candidates, advanced) =
            screen_cv5_arms(&cv5.evidence, &eligible_arms, &thresholds)?;
        let Some(advanced_arm) = advanced else {
            return build_measured_decision(
                dataset_id,
                dataset,
                &predictor_sha256,
                &encoder_sha256,
                &cv5_plan_sha256,
                &cv7_plan_sha256,
                thresholds,
                cv5_candidates,
                None,
                None,
                Some("no CV5 arm passed every frozen advancement criterion"),
                &cv5.receipts,
                cv5.encoding_milliseconds,
                cv5.scoring_milliseconds,
                started,
            );
        };
        if started.elapsed() >= WALL_CLOCK_BUDGET {
            return Err("wall-clock budget expired before CV7 confirmation".into());
        }
        let cv7 = execute_stage(
            &registry.encoder,
            dataset,
            &authorized_table,
            &precomputed,
            &predictor,
            &adapter,
            &cv7_plan,
            &cv7_plan_sha256,
            &[advanced_arm],
            &receipt_root,
            started,
            cv5.encoding_milliseconds,
            cv5.scoring_milliseconds,
        )?;
        let confirmation = confirm_cv7(&cv7.evidence, advanced_arm, &thresholds)?;
        let fallback = (!confirmation.passed)
            .then_some("frozen CV5 arm failed one or more CV7 confirmation criteria");
        let mut receipts = cv5.receipts;
        receipts.extend(cv7.receipts);
        build_measured_decision(
            dataset_id,
            dataset,
            &predictor_sha256,
            &encoder_sha256,
            &cv5_plan_sha256,
            &cv7_plan_sha256,
            thresholds,
            cv5_candidates,
            Some(advanced_arm),
            Some(confirmation),
            fallback,
            &receipts,
            cv5.encoding_milliseconds + cv7.encoding_milliseconds,
            cv5.scoring_milliseconds + cv7.scoring_milliseconds,
            started,
        )
    })();

    let decision = match operational {
        Ok(decision) => decision,
        Err(reason) => raw_fallback_decision(
            dataset_id,
            &dataset.train_sha256,
            &predictor_sha256,
            &encoder_sha256,
            &cv5_plan_sha256,
            &cv7_plan_sha256,
            reason,
            started,
        ),
    };
    decision.validate()?;
    write_gate_artifacts(
        artifact_root,
        &decision,
        &predictor,
        dataset.feature_count as usize,
    )?;
    Ok(decision)
}

/// Precompute the pinned full-train, CV5, and CV7 encodings for either one
/// registry member or the complete frozen registry.  A complete run publishes
/// a coverage freeze; a targeted run only fills content-addressed entries.
pub fn build_registry_cache(
    registry_path: &Path,
    dataset_id: Option<&str>,
) -> Result<RegistryCacheCoverageV2, String> {
    if let Some(dataset_id) = dataset_id {
        validate_opaque_id(dataset_id)?;
    }
    let registry_bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&registry_bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    registry.encoder.verify_artifacts()?;
    require_cuda_device(registry.encoder.device_ordinal)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let selected = registry
        .datasets
        .iter()
        .filter(|dataset| dataset_id.is_none_or(|wanted| dataset.dataset_id == wanted))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("requested dataset is absent from the train-only registry".into());
    }
    let mut datasets = Vec::with_capacity(selected.len());
    for dataset in selected {
        let source = read_authorized_table(registry_root, dataset)?;
        let full_train = encode_full_train(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
        )?;
        let cv5_plan = build_grouped_fold_plan(GateStageV1::Cv5Screen, &dataset.atomic_groups)?;
        let cv7_plan =
            build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &dataset.atomic_groups)?;
        let mut cv5 = Vec::with_capacity(5);
        let cv5_sha256 = canonical_sha256(&cv5_plan)?;
        for fold in &cv5_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv5_plan,
                &cv5_sha256,
                &cache_root,
                &registry_sha256,
                dataset,
                Instant::now(),
            )?;
            cv5.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        let mut cv7 = Vec::with_capacity(7);
        let cv7_sha256 = canonical_sha256(&cv7_plan)?;
        for fold in &cv7_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv7_plan,
                &cv7_sha256,
                &cache_root,
                &registry_sha256,
                dataset,
                Instant::now(),
            )?;
            cv7.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        datasets.push(DatasetCacheCoverageV2 {
            dataset_id: dataset.dataset_id.clone(),
            full_train,
            cv5,
            cv7,
        });
    }
    let coverage = RegistryCacheCoverageV2 {
        schema: RegistryCacheCoverageV2::SCHEMA.into(),
        registry_sha256,
        datasets,
    };
    coverage.validate()?;
    if dataset_id.is_none() {
        if coverage.datasets.len() != registry.datasets.len() {
            return Err("refusing to freeze incomplete registry cache coverage".into());
        }
        fs::create_dir_all(&cache_root)
            .map_err(|error| format!("create registry cache root: {error}"))?;
        write_immutable(
            &cache_root.join("registry-cache-coverage.v2.json"),
            &canonical_json_bytes(&coverage)?,
        )?;
        let historical = build_historical_train3_inventory(&registry, registry_root)?;
        write_immutable(
            &cache_root.join("historical-train3-inventory.v1.json"),
            &canonical_json_bytes(&historical)?,
        )?;
        verify_complete_registry_cache(registry_path)?;
    }
    Ok(coverage)
}

/// Read-only freeze check.  This never invokes Hyperion: every registry member
/// must already have one validated full-train receipt and all frozen CV5/CV7
/// fold receipts with exact shapes and content hashes.
pub fn verify_complete_registry_cache(
    registry_path: &Path,
) -> Result<RegistryCacheCoverageV2, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let coverage_path = cache_root.join("registry-cache-coverage.v2.json");
    let coverage: RegistryCacheCoverageV2 = serde_json::from_slice(
        &fs::read(&coverage_path)
            .map_err(|error| format!("read registry cache coverage: {error}"))?,
    )
    .map_err(|error| format!("decode registry cache coverage: {error}"))?;
    coverage.validate()?;
    if coverage.registry_sha256 != registry_sha256
        || coverage.datasets.len() != registry.datasets.len()
        || coverage
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
    {
        return Err("registry cache coverage does not exactly match the frozen registry".into());
    }
    for (dataset, covered) in registry.datasets.iter().zip(&coverage.datasets) {
        let source = read_authorized_table(registry_root, dataset)?;
        verify_full_train_entry(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
            &covered.full_train,
        )?;
        for (stage, entries) in [
            (GateStageV1::Cv5Screen, covered.cv5.as_slice()),
            (GateStageV1::Cv7Confirmation, covered.cv7.as_slice()),
        ] {
            let plan = build_grouped_fold_plan(stage, &dataset.atomic_groups)?;
            let plan_sha256 = canonical_sha256(&plan)?;
            for (fold, entry) in plan.folds.iter().zip(entries) {
                verify_fold_entry(
                    &registry.encoder,
                    &source,
                    dataset,
                    &registry_sha256,
                    &cache_root,
                    &plan,
                    &plan_sha256,
                    fold,
                    entry,
                )?;
            }
        }
    }
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
fn verify_fold_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    fold: &GroupedFoldV1,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let stage = match plan.stage {
        GateStageV1::Cv5Screen => "cv5",
        GateStageV1::Cv7Confirmation => "cv7",
    };
    let fit_input = csv_for_rows(source, &fold.fit_rows, false);
    let hold_input = csv_for_rows(source, &fold.hold_rows, true);
    let expected_key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        stage,
        plan_sha256,
        Some(fold.fold_index),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.hold_rows),
        &raw_sha256(&fit_input),
        &raw_sha256(&hold_input),
    )?;
    if entry.fold_index != Some(fold.fold_index)
        || entry.cache_key_sha256 != expected_key.identity()?
    {
        return Err("coverage entry does not match the frozen fold order".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join(stage)
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FoldEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("fold-encoding-cache.v2.json"))
            .map_err(|error| format!("read fold encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode fold encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("fold encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_fit",
            (
                fold.fit_rows.len(),
                dataset.feature_count as usize,
                "cell-fit.vs",
            ),
        ),
        (
            "cell_hold",
            (
                fold.hold_rows.len(),
                dataset.feature_count as usize,
                "cell-hold.vs",
            ),
        ),
        (
            "row_fit",
            (fold.fit_rows.len(), ROW_EMBEDDING_WIDTH, "row-fit.vs"),
        ),
        (
            "row_hold",
            (fold.hold_rows.len(), ROW_EMBEDDING_WIDTH, "row-hold.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_full_train_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let rows = (0..source.targets.len()).collect::<Vec<_>>();
    let train_input = csv_for_rows(source, &rows, false);
    let dummy_input = csv_for_rows(source, &rows, true);
    let row_order_sha256 = hash_selected_strings(&dataset.stable_row_ids, &rows);
    let plan_sha256 =
        raw_sha256(format!("full-train-v2|{}|{row_order_sha256}", dataset.train_sha256).as_bytes());
    let expected_key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        "full_train",
        &plan_sha256,
        None,
        &row_order_sha256,
        &row_order_sha256,
        &raw_sha256(&train_input),
        &raw_sha256(&dummy_input),
    )?;
    if entry.fold_index.is_some() || entry.cache_key_sha256 != expected_key.identity()? {
        return Err("full-train coverage entry identity drifted".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join("full_train")
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FullTrainEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("full-train-encoding-cache.v2.json"))
            .map_err(|error| format!("read full-train encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode full-train encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("full-train encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_train",
            (rows.len(), dataset.feature_count as usize, "cell-train.vs"),
        ),
        (
            "row_train",
            (rows.len(), ROW_EMBEDDING_WIDTH, "row-train.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_receipted_sidecars(
    directory: &Path,
    sidecars: &[MatrixArtifactV1],
    expected: &BTreeMap<&str, (usize, usize, &str)>,
) -> Result<(), String> {
    if sidecars.len() != expected.len() {
        return Err("receipted sidecar coverage is incomplete".into());
    }
    for sidecar in sidecars {
        sidecar.validate()?;
        let (rows, columns, file) = expected
            .get(sidecar.role.as_str())
            .ok_or_else(|| "receipted sidecar role is unexpected".to_string())?;
        if sidecar.rows as usize != *rows || sidecar.columns as usize != *columns {
            return Err("receipted sidecar shape differs from the frozen fold".into());
        }
        let path = directory.join(file);
        let bytes = fs::read(&path)
            .map_err(|error| format!("read receipted sidecar {}: {error}", path.display()))?;
        if bytes.is_empty() || raw_sha256(&bytes) != sidecar.sha256 {
            return Err("receipted sidecar is empty or changed".into());
        }
        read_vs_matrix(&path, *rows, *columns, &sidecar.role)?;
    }
    Ok(())
}

pub fn build_historical_train3_inventory(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
) -> Result<HistoricalTrain3InventoryV1, String> {
    registry.validate(registry_root)?;
    let mut entries = Vec::with_capacity(registry.datasets.len());
    for dataset in &registry.datasets {
        let train = Path::new(&dataset.train_csv_path);
        let relative = train
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("train3.vs");
        let path = safe_registry_path(registry_root, &relative.to_string_lossy())?;
        let (state, byte_length, sha256) = if path.is_file() {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read historical {}: {error}", path.display()))?;
            if bytes.is_empty() {
                ("zero_byte", 0, None)
            } else {
                ("present", bytes.len() as u64, Some(raw_sha256(&bytes)))
            }
        } else {
            ("missing", 0, None)
        };
        entries.push(HistoricalTrain3EntryV1 {
            dataset_id: dataset.dataset_id.clone(),
            relative_path: relative.to_string_lossy().into_owned(),
            state: state.into(),
            byte_length,
            sha256,
        });
    }
    let inventory = HistoricalTrain3InventoryV1 {
        schema: HistoricalTrain3InventoryV1::SCHEMA.into(),
        registry_sha256: registry.identity()?,
        entries,
        cv_authority: false,
    };
    inventory.validate()?;
    Ok(inventory)
}

/// Freeze exact train-side `.vs` coverage for the registry without invoking an
/// encoder or resolving any test path.  The coverage file is immutable and is
/// suitable for the policy's coverage identity.
pub fn freeze_precomputed_train_feature_coverage(
    registry_path: &Path,
) -> Result<PrecomputedTrainFeatureCoverageV1, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let datasets = registry
        .datasets
        .iter()
        .map(|dataset| {
            load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)
                .map(|blocks| blocks.receipt)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let coverage = PrecomputedTrainFeatureCoverageV1 {
        schema: PrecomputedTrainFeatureCoverageV1::SCHEMA.into(),
        registry_sha256,
        datasets,
        test_artifacts_read: false,
    };
    coverage.validate()?;
    let root = registry.resolved_cache_root(registry_root)?;
    fs::create_dir_all(&root)
        .map_err(|error| format!("create precomputed coverage root: {error}"))?;
    write_immutable(
        &root.join("precomputed-train-feature-coverage.v1.json"),
        &canonical_json_bytes(&coverage)?,
    )?;
    verify_precomputed_train_feature_coverage(registry_path)
}

/// Re-read every authorized train-side matrix and require exact equality with
/// the immutable coverage freeze.  This operation cannot invoke Hyperion.
pub fn verify_precomputed_train_feature_coverage(
    registry_path: &Path,
) -> Result<PrecomputedTrainFeatureCoverageV1, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let root = registry.resolved_cache_root(registry_root)?;
    let coverage_path = root.join("precomputed-train-feature-coverage.v1.json");
    let frozen: PrecomputedTrainFeatureCoverageV1 = serde_json::from_slice(
        &fs::read(&coverage_path).map_err(|error| format!("read precomputed coverage: {error}"))?,
    )
    .map_err(|error| format!("decode precomputed coverage: {error}"))?;
    frozen.validate()?;
    let current = PrecomputedTrainFeatureCoverageV1 {
        schema: PrecomputedTrainFeatureCoverageV1::SCHEMA.into(),
        registry_sha256: registry_sha256.clone(),
        datasets: registry
            .datasets
            .iter()
            .map(|dataset| {
                load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)
                    .map(|blocks| blocks.receipt)
            })
            .collect::<Result<Vec<_>, _>>()?,
        test_artifacts_read: false,
    };
    current.validate()?;
    if frozen != current
        || frozen
            .datasets
            .iter()
            .map(|receipt| receipt.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
    {
        return Err("precomputed train-feature coverage changed after freezing".into());
    }
    Ok(frozen)
}

fn require_frozen_precomputed_dataset_binding(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    receipt: &PrecomputedTrainFeatureReceiptV1,
) -> Result<(), String> {
    let root = registry.resolved_cache_root(registry_root)?;
    let path = root.join("precomputed-train-feature-coverage.v1.json");
    let coverage: PrecomputedTrainFeatureCoverageV1 = serde_json::from_slice(
        &fs::read(&path).map_err(|error| format!("read precomputed coverage: {error}"))?,
    )
    .map_err(|error| format!("decode precomputed coverage: {error}"))?;
    coverage.validate()?;
    if coverage.registry_sha256 != registry.identity()?
        || coverage.datasets.len() != registry.datasets.len()
        || coverage
            .datasets
            .iter()
            .map(|entry| entry.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
        || coverage
            .datasets
            .iter()
            .find(|entry| entry.dataset_id == receipt.dataset_id)
            != Some(receipt)
    {
        return Err("dataset cache receipt differs from frozen registry-wide coverage".into());
    }
    Ok(())
}

fn read_authorized_table(
    registry_root: &Path,
    dataset: &TrainOnlyDatasetV1,
) -> Result<RegressionTable, String> {
    let train_path = safe_registry_path(registry_root, &dataset.train_csv_path)?;
    let bytes =
        fs::read(&train_path).map_err(|error| format!("read authorized train CSV: {error}"))?;
    if raw_sha256(&bytes) != dataset.train_sha256 {
        return Err("authorized train CSV differs from its registry hash".into());
    }
    let table = parse_regression_csv(&bytes, dataset.feature_count as usize)?;
    if table.targets.len() != dataset.stable_row_ids.len() {
        return Err("authorized train rows differ from registry row identities".into());
    }
    Ok(table)
}

fn sibling_train_sidecar(
    registry_root: &Path,
    dataset: &TrainOnlyDatasetV1,
    file_name: &str,
) -> Result<(String, PathBuf), String> {
    if !matches!(file_name, "train3.vs" | "hyperion3train.vs") {
        return Err("precomputed feature loader accepts train-side .vs names only".into());
    }
    let relative = Path::new(&dataset.train_csv_path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(file_name);
    let relative = relative.to_string_lossy().into_owned();
    let path = safe_registry_path(registry_root, &relative)?;
    Ok((relative, path))
}

fn load_precomputed_train_features(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    registry_sha256: &str,
    dataset: &TrainOnlyDatasetV1,
) -> Result<PrecomputedTrainFeatureBlocks, String> {
    let rows = dataset.stable_row_ids.len();
    let columns = dataset.feature_count as usize;
    let (cell_relative_path, cell_path) =
        sibling_train_sidecar(registry_root, dataset, "train3.vs")?;
    let cell_bytes = fs::read(&cell_path)
        .map_err(|error| format!("read precomputed cell {}: {error}", cell_path.display()))?;
    if cell_bytes.is_empty() {
        return Err("precomputed train3.vs is empty".into());
    }
    let cell = read_vs_matrix(&cell_path, rows, columns, "precomputed cell train")?;
    let cell_artifact = MatrixArtifactV1 {
        role: "cell_train_precomputed".into(),
        rows: rows as u32,
        columns: columns as u32,
        sha256: raw_sha256(&cell_bytes),
    };

    // Row128 is an optional arm.  Its absence or corruption is recorded and
    // excludes row/both only; the mandatory, valid cell block remains usable.
    let (row_relative, row_path) =
        sibling_train_sidecar(registry_root, dataset, "hyperion3train.vs")?;
    let (row128_relative_path, row128_artifact, row128, row128_unavailable_reason) =
        match fs::read(&row_path) {
            Ok(bytes) if !bytes.is_empty() => {
                match read_vs_matrix(
                    &row_path,
                    rows,
                    ROW_EMBEDDING_WIDTH,
                    "precomputed row128 train",
                ) {
                    Ok(matrix) => (
                        Some(row_relative),
                        Some(MatrixArtifactV1 {
                            role: "row128_train_precomputed".into(),
                            rows: rows as u32,
                            columns: ROW_EMBEDDING_WIDTH as u32,
                            sha256: raw_sha256(&bytes),
                        }),
                        Some(matrix),
                        None,
                    ),
                    Err(reason) => (None, None, None, Some(reason)),
                }
            }
            Ok(_) => (
                None,
                None,
                None,
                Some("precomputed hyperion3train.vs is empty".into()),
            ),
            Err(error) => (
                None,
                None,
                None,
                Some(format!("precomputed row128 unavailable: {error}")),
            ),
        };
    let receipt = PrecomputedTrainFeatureReceiptV1 {
        schema: PrecomputedTrainFeatureReceiptV1::SCHEMA.into(),
        registry_sha256: registry_sha256.into(),
        dataset_id: dataset.dataset_id.clone(),
        dataset_id_sha256: raw_sha256(dataset.dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        stable_row_ids_sha256: hash_strings(&dataset.stable_row_ids),
        encoder_identity_sha256: registry.encoder.identity()?,
        row_count: rows as u32,
        raw_feature_count: dataset.feature_count,
        cell_relative_path,
        cell: cell_artifact,
        row128_relative_path,
        row128: row128_artifact,
        row128_unavailable_reason,
        test_artifacts_read: false,
    };
    receipt.validate()?;
    Ok(PrecomputedTrainFeatureBlocks {
        cell,
        row128,
        receipt,
    })
}

/// Read-only inspection used by registry freezing and tests.  This opens only
/// `train3.vs` and the optional `hyperion3train.vs` sibling for the named
/// train-only registry member.
pub fn inspect_precomputed_train_features(
    registry: &JopeHyperionTrainOnlyRegistryV2,
    registry_root: &Path,
    dataset_id: &str,
) -> Result<PrecomputedTrainFeatureReceiptV1, String> {
    registry.validate(registry_root)?;
    validate_opaque_id(dataset_id)?;
    let dataset = registry
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_id == dataset_id)
        .ok_or_else(|| "opaque dataset ID is absent from the train-only registry".to_string())?;
    Ok(
        load_precomputed_train_features(registry, registry_root, &registry.identity()?, dataset)?
            .receipt,
    )
}

fn encode_full_train(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
) -> Result<CacheCoverageEntryV2, String> {
    let rows = (0..source.targets.len()).collect::<Vec<_>>();
    let train_input = csv_for_rows(source, &rows, false);
    let dummy_input = csv_for_rows(source, &rows, true);
    let row_order_sha256 = hash_selected_strings(&dataset.stable_row_ids, &rows);
    let plan_sha256 =
        raw_sha256(format!("full-train-v2|{}|{row_order_sha256}", dataset.train_sha256).as_bytes());
    let key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        "full_train",
        &plan_sha256,
        None,
        &row_order_sha256,
        &row_order_sha256,
        &raw_sha256(&train_input),
        &raw_sha256(&dummy_input),
    )?;
    let key_sha256 = key.identity()?;
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join("full_train")
        .join(&key_sha256[..2])
        .join(&key_sha256);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create full-train encoding directory: {error}"))?;
    let train_path = directory.join("train.csv");
    let dummy_path = directory.join("dummy-hold.csv");
    write_immutable(&train_path, &train_input)?;
    write_immutable(&dummy_path, &dummy_input)?;
    let outputs = [
        (
            "cell_train",
            directory.join("cell-train.vs"),
            rows.len(),
            dataset.feature_count as usize,
        ),
        (
            "cell_shadow_hold",
            directory.join("cell-shadow-hold.vs"),
            rows.len(),
            dataset.feature_count as usize,
        ),
        (
            "row_train",
            directory.join("row-train.vs"),
            rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
        (
            "row_shadow_hold",
            directory.join("row-shadow-hold.vs"),
            rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
    ];
    let receipt_path = directory.join("full-train-encoding-cache.v2.json");
    if receipt_path.is_file() {
        let receipt: FullTrainEncodingCacheV2 = serde_json::from_slice(
            &fs::read(&receipt_path)
                .map_err(|error| format!("read full-train encoding cache: {error}"))?,
        )
        .map_err(|error| format!("decode full-train encoding cache: {error}"))?;
        receipt.validate(encoder.device_ordinal)?;
        if receipt.key != key || receipt.key_sha256 != key_sha256 {
            return Err("full-train encoding cache identity drifted".into());
        }
        let primary = [outputs[0].clone(), outputs[2].clone()];
        read_output_matrices(&primary)?;
        let actual = output_artifacts(&primary)?;
        if actual != receipt.sidecars {
            return Err("full-train sidecar changed after cache publication".into());
        }
        return Ok(CacheCoverageEntryV2 {
            fold_index: None,
            cache_key_sha256: key_sha256,
            cache_receipt_sha256: canonical_sha256(&receipt)?,
        });
    }
    if outputs.iter().any(|(_, path, _, _)| path.exists()) {
        return Err("partial or unreceipted full-train sidecar output exists".into());
    }
    let execution =
        run_encoder_with_timeout(encoder, &train_path, &dummy_path, &outputs, ENCODING_BUDGET)?;
    if !execution.status.success() {
        return Err(format!(
            "full-train Hyperion encoder exited with {}",
            execution.status
        ));
    }
    execution.proof.validate_namespace()?;
    read_output_matrices(&outputs)?;
    let primary = [outputs[0].clone(), outputs[2].clone()];
    let sidecars = output_artifacts(&primary)?;
    for path in [&outputs[1].1, &outputs[3].1] {
        fs::remove_file(path)
            .map_err(|error| format!("remove unused full-train shadow output: {error}"))?;
    }
    let receipt = FullTrainEncodingCacheV2 {
        schema: FullTrainEncodingCacheV2::SCHEMA.into(),
        key,
        key_sha256,
        train_row_order_sha256: row_order_sha256,
        sidecars,
        proof: execution.proof,
    };
    receipt.validate(encoder.device_ordinal)?;
    write_immutable(&receipt_path, &canonical_json_bytes(&receipt)?)?;
    Ok(CacheCoverageEntryV2 {
        fold_index: None,
        cache_key_sha256: receipt.key_sha256.clone(),
        cache_receipt_sha256: canonical_sha256(&receipt)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_measured_decision(
    dataset_id: &str,
    dataset: &TrainOnlyDatasetV1,
    predictor_sha256: &str,
    encoder_sha256: &str,
    cv5_plan_sha256: &str,
    cv7_plan_sha256: &str,
    thresholds: GateThresholdsV1,
    cv5_candidates: Vec<StageGateResultV1>,
    advanced: Option<FeatureViewKindV1>,
    confirmation: Option<StageGateResultV1>,
    fallback: Option<&str>,
    receipts: &[JopeHyperionFoldReceiptV1],
    encoding_milliseconds: u64,
    scoring_milliseconds: u64,
    started: Instant,
) -> Result<JopeHyperionGateDecisionV1, String> {
    let fallback = fallback.map(str::to_owned);
    let selected_view = if fallback.is_none() {
        confirmation
            .as_ref()
            .filter(|value| value.passed)
            .map(|value| value.arm)
            .unwrap_or(FeatureViewKindV1::Raw)
    } else {
        FeatureViewKindV1::Raw
    };
    let decision = JopeHyperionGateDecisionV1 {
        schema: JopeHyperionGateDecisionV1::SCHEMA.into(),
        dataset_id_sha256: raw_sha256(dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        raw_predictor_sha256: predictor_sha256.into(),
        encoder_identity_sha256: encoder_sha256.into(),
        cv5_fold_plan_sha256: cv5_plan_sha256.into(),
        cv7_fold_plan_sha256: cv7_plan_sha256.into(),
        thresholds,
        cv5_candidates,
        cv5_advanced_arm: advanced,
        cv7_confirmation: confirmation,
        selected_view,
        raw_fallback_reason: fallback,
        fold_receipt_sha256s: receipts
            .iter()
            .map(canonical_sha256)
            .collect::<Result<Vec<_>, _>>()?,
        encoding_milliseconds,
        scoring_milliseconds,
        wall_clock_milliseconds: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    };
    decision.validate()?;
    Ok(decision)
}

#[allow(clippy::too_many_arguments)]
fn execute_stage(
    encoder: &HyperionEncoderIdentityV1,
    dataset: &TrainOnlyDatasetV1,
    source: &RegressionTable,
    precomputed: &PrecomputedTrainFeatureBlocks,
    predictor: &FrozenPredictorV1,
    adapter: &CorpusEvalAdapter,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    arms: &[FeatureViewKindV1],
    receipt_root: &Path,
    started: Instant,
    prior_encoding_milliseconds: u64,
    prior_scoring_milliseconds: u64,
) -> Result<StageExecution, String> {
    let mut evidence = Vec::with_capacity(plan.folds.len());
    let mut receipts = Vec::new();
    let mut encoding_milliseconds = 0u64;
    let mut scoring_milliseconds = 0u64;
    validate_plan_feature_feasibility(plan, arms, dataset.feature_count as usize)?;
    for fold in &plan.folds {
        if started.elapsed() >= WALL_CLOCK_BUDGET {
            return Err("ten-minute Hyperion gate wall-clock budget expired".into());
        }
        let encoding = precomputed_fold_encoding(precomputed, fold, plan_sha256)?;
        encoding_milliseconds += encoding.milliseconds;
        if prior_encoding_milliseconds + encoding_milliseconds > ENCODING_BUDGET.as_millis() as u64
        {
            return Err("eight-minute fold-local encoding budget expired".into());
        }
        let fitted = FittedFeatureBlocksV1::fit_available(
            &encoding.cell_fit,
            (!encoding.row_fit.is_empty()).then_some(encoding.row_fit.as_slice()),
            dataset.feature_count as usize,
        )?;
        let cell_fit = fitted.transform_cell(&encoding.cell_fit)?;
        let cell_hold = fitted.transform_cell(&encoding.cell_hold)?;
        let (row_fit, row_hold) = if encoding.row_fit.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (
                fitted.transform_row(&encoding.row_fit)?,
                fitted.transform_row(&encoding.row_hold)?,
            )
        };
        let raw_scoring_started = Instant::now();
        let raw_prediction = score_view(
            FeatureViewKindV1::Raw,
            dataset,
            source,
            fold,
            &[],
            &[],
            &[],
            &[],
            predictor,
            adapter,
        )?;
        scoring_milliseconds += elapsed_milliseconds(raw_scoring_started);
        let mut arm_predictions = BTreeMap::new();
        for arm in arms {
            let scoring_started = Instant::now();
            let prediction = score_view(
                *arm, dataset, source, fold, &cell_fit, &cell_hold, &row_fit, &row_hold, predictor,
                adapter,
            )?;
            let scoring_elapsed = elapsed_milliseconds(scoring_started);
            scoring_milliseconds += scoring_elapsed;
            if prior_scoring_milliseconds + scoring_milliseconds > SCORING_BUDGET.as_millis() as u64
            {
                return Err("two-minute model scoring and verification budget expired".into());
            }
            let actual = fold
                .hold_rows
                .iter()
                .map(|row| source.targets[*row])
                .collect::<Vec<_>>();
            let feature_view = JopeFeatureViewV1::new(*arm, dataset.feature_count as usize)?;
            let expanded_fit = assemble_feature_view(
                *arm,
                &select_rows(&source.features, &fold.fit_rows),
                &cell_fit,
                &row_fit,
            )?;
            let expanded_hold = assemble_feature_view(
                *arm,
                &select_rows(&source.features, &fold.hold_rows),
                &cell_hold,
                &row_hold,
            )?;
            let receipt = JopeHyperionFoldReceiptV1 {
                schema: JopeHyperionFoldReceiptV1::SCHEMA.into(),
                stage: plan.stage,
                fold_index: fold.fold_index,
                fold_plan_sha256: plan_sha256.into(),
                feature_view_sha256: feature_view.identity()?,
                feature_view,
                predictor_sha256: predictor.semantic_sha256()?,
                encoder_identity_sha256: encoder.identity()?,
                cache_key_sha256: encoding.cache_key_sha256.clone(),
                cache_receipt_sha256: encoding.cache_receipt_sha256.clone(),
                source_training_sha256: dataset.train_sha256.clone(),
                fit_row_ids_sha256: hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows),
                hold_row_ids_sha256: hash_selected_strings(
                    &dataset.stable_row_ids,
                    &fold.hold_rows,
                ),
                fit_atomic_groups_sha256: hash_selected_strings(
                    &dataset.atomic_groups,
                    &fold.fit_rows,
                ),
                hold_atomic_groups_sha256: hash_selected_strings(
                    &dataset.atomic_groups,
                    &fold.hold_rows,
                ),
                sidecars: sidecars_for_view(&encoding.sidecars, *arm),
                expanded_fit_sha256: hash_matrix(&expanded_fit),
                expanded_hold_sha256: hash_matrix(&expanded_hold),
                raw_predictions_sha256: hash_f64(&raw_prediction),
                view_predictions_sha256: hash_f64(&prediction),
                raw_rmse: rmse(&actual, &raw_prediction),
                view_rmse: rmse(&actual, &prediction),
                encoding_milliseconds: encoding.milliseconds,
                scoring_milliseconds: scoring_elapsed,
                cache_replay: encoding.cache_replay,
            };
            receipt.validate()?;
            let receipt_bytes = canonical_json_bytes(&receipt)?;
            let stage_name = match plan.stage {
                GateStageV1::Cv5Screen => "cv5",
                GateStageV1::Cv7Confirmation => "cv7",
            };
            let arm_name = format!("{:?}", arm).to_ascii_lowercase();
            write_immutable(
                &receipt_root.join(format!(
                    "{stage_name}-fold-{}-{arm_name}.v1.json",
                    fold.fold_index
                )),
                &receipt_bytes,
            )?;
            arm_predictions.insert(*arm, prediction);
            receipts.push(receipt);
        }
        evidence.push(PairedFoldEvidenceV1 {
            fold_index: fold.fold_index,
            row_ids: fold
                .hold_rows
                .iter()
                .map(|row| dataset.stable_row_ids[*row].clone())
                .collect(),
            atomic_groups: fold
                .hold_rows
                .iter()
                .map(|row| dataset.atomic_groups[*row].clone())
                .collect(),
            actual: fold
                .hold_rows
                .iter()
                .map(|row| source.targets[*row])
                .collect(),
            raw_predictions: raw_prediction,
            arm_predictions,
        });
    }
    Ok(StageExecution {
        evidence,
        receipts,
        encoding_milliseconds,
        scoring_milliseconds,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_view(
    view: FeatureViewKindV1,
    dataset: &TrainOnlyDatasetV1,
    source: &RegressionTable,
    fold: &GroupedFoldV1,
    cell_fit: &[Vec<f64>],
    cell_hold: &[Vec<f64>],
    row_fit: &[Vec<f64>],
    row_hold: &[Vec<f64>],
    predictor: &FrozenPredictorV1,
    adapter: &CorpusEvalAdapter,
) -> Result<Vec<f64>, String> {
    let fit_raw = select_rows(&source.features, &fold.fit_rows);
    let hold_raw = select_rows(&source.features, &fold.hold_rows);
    let fit_features = if view == FeatureViewKindV1::Raw {
        fit_raw
    } else {
        assemble_feature_view(view, &fit_raw, cell_fit, row_fit)?
    };
    let hold_features = if view == FeatureViewKindV1::Raw {
        hold_raw
    } else {
        assemble_feature_view(view, &hold_raw, cell_hold, row_hold)?
    };
    let fit_position = fold
        .fit_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let hold_position = fold
        .hold_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let rows = (0..source.targets.len())
        .map(|row| {
            let (features, target) = if let Some(position) = fit_position.get(&row) {
                (&fit_features[*position], source.targets[row])
            } else {
                (&hold_features[hold_position[&row]], 0.0)
            };
            let mut fields = features
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>();
            fields.push(target.to_string());
            DatasetRowV1 {
                split: SplitRowV1 {
                    stable_row_id: dataset.stable_row_ids[row].clone(),
                    atomic_group: dataset.atomic_groups[row].clone(),
                    regression_target: Some(source.targets[row]),
                    binary_target: None,
                },
                csv_line: fields.join(","),
            }
        })
        .collect::<Vec<_>>();
    let csv_bytes = rows
        .iter()
        .flat_map(|row| [row.csv_line.as_bytes(), b"\n"].concat())
        .collect::<Vec<_>>();
    let table = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: dataset.dataset_sha256.clone(),
        training_sha256: raw_sha256(&csv_bytes),
        rows,
    };
    table.validate()?;
    let mut fit_ids = fold
        .fit_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    let mut hold_ids = fold
        .hold_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    fit_ids.sort();
    hold_ids.sort();
    let split = RowSplitV1 {
        split_index: fold.fold_index,
        seed: 1,
        train_row_ids: fit_ids,
        validation_row_ids: hold_ids.clone(),
        validation_groups: fold.hold_groups.clone(),
        balance_counts: BTreeMap::new(),
    };
    let candidates = predictor
        .pipelines()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let outcomes = adapter.evaluate_search_split(&table, &split, &candidates)?;
    if outcomes.len() != candidates.len()
        || outcomes.iter().any(|outcome| !outcome.prediction.valid)
    {
        return Err("AFPM evaluator returned incomplete or invalid fold predictions".into());
    }
    let components = outcomes
        .iter()
        .map(|outcome| {
            if outcome.prediction.validation_row_ids != hold_ids {
                return Err("AFPM predictions changed hold-row ordering".into());
            }
            Ok(outcome
                .prediction
                .prediction_f64_bits
                .iter()
                .map(|bits| f64::from_bits(*bits))
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let sorted_prediction = predictor.reconstruct(&components)?;
    let by_id = hold_ids
        .into_iter()
        .zip(sorted_prediction)
        .collect::<BTreeMap<_, _>>();
    fold.hold_rows
        .iter()
        .map(|row| {
            by_id
                .get(&dataset.stable_row_ids[*row])
                .copied()
                .ok_or_else(|| "AFPM prediction omitted a hold row".to_string())
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cache_key(
    registry_sha256: &str,
    dataset: &TrainOnlyDatasetV1,
    encoder: &HyperionEncoderIdentityV1,
    stage: &str,
    fold_plan_sha256: &str,
    fold_index: Option<u32>,
    fit_row_ids_sha256: &str,
    hold_row_ids_sha256: &str,
    fit_input_sha256: &str,
    hold_input_sha256: &str,
) -> Result<HyperionCacheKeyV2, String> {
    let key = HyperionCacheKeyV2 {
        schema: HyperionCacheKeyV2::SCHEMA.into(),
        registry_sha256: registry_sha256.into(),
        dataset_id_sha256: raw_sha256(dataset.dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        stable_row_ids_sha256: hash_strings(&dataset.stable_row_ids),
        atomic_groups_sha256: hash_strings(&dataset.atomic_groups),
        stage: stage.into(),
        fold_plan_sha256: fold_plan_sha256.into(),
        fold_index,
        fit_row_ids_sha256: fit_row_ids_sha256.into(),
        hold_row_ids_sha256: hold_row_ids_sha256.into(),
        fit_input_sha256: fit_input_sha256.into(),
        hold_input_sha256: hold_input_sha256.into(),
        encoder_identity_sha256: encoder.identity()?,
        encoder_binary_sha256: encoder.binary.sha256.clone(),
        encoder_weights_sha256: encoder.weights.sha256.clone(),
        encoder_manifest_sha256: encoder.manifest.sha256.clone(),
        encoder_source_hashes_sha256: canonical_sha256(&encoder.source_hashes)?,
        seed: encoder.seed,
        sweeps: encoder.sweeps,
        scales: encoder.scales.clone(),
        batch_size: encoder.batch_size,
        runtime_versions_sha256: canonical_sha256(&encoder.runtime_versions)?,
        observed_device: "cuda:0".into(),
    };
    key.validate()?;
    Ok(key)
}

fn encode_fold(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    fold: &GroupedFoldV1,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    encoding_root: &Path,
    registry_sha256: &str,
    dataset: &TrainOnlyDatasetV1,
    gate_started: Instant,
) -> Result<FoldEncoding, String> {
    let stage = match plan.stage {
        GateStageV1::Cv5Screen => "cv5",
        GateStageV1::Cv7Confirmation => "cv7",
    };
    let fit_input = csv_for_rows(source, &fold.fit_rows, false);
    let hold_input = csv_for_rows(source, &fold.hold_rows, true);
    let fit_input_sha256 = raw_sha256(&fit_input);
    let hold_input_sha256 = raw_sha256(&hold_input);
    let fit_row_ids_sha256 = hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows);
    let hold_row_ids_sha256 = hash_selected_strings(&dataset.stable_row_ids, &fold.hold_rows);
    let key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        stage,
        plan_sha256,
        Some(fold.fold_index),
        &fit_row_ids_sha256,
        &hold_row_ids_sha256,
        &fit_input_sha256,
        &hold_input_sha256,
    )?;
    let key_sha256 = key.identity()?;
    let directory = encoding_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join(stage)
        .join(&key_sha256[..2])
        .join(&key_sha256);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create fold-local encoding directory: {error}"))?;
    let fit_path = directory.join("fit.csv");
    let hold_path = directory.join("hold.csv");
    write_immutable(&fit_path, &fit_input)?;
    write_immutable(&hold_path, &hold_input)?;
    let outputs = [
        (
            "cell_fit",
            directory.join("cell-fit.vs"),
            fold.fit_rows.len(),
            source.features[0].len(),
        ),
        (
            "cell_hold",
            directory.join("cell-hold.vs"),
            fold.hold_rows.len(),
            source.features[0].len(),
        ),
        (
            "row_fit",
            directory.join("row-fit.vs"),
            fold.fit_rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
        (
            "row_hold",
            directory.join("row-hold.vs"),
            fold.hold_rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
    ];
    let cache_path = directory.join("fold-encoding-cache.v2.json");
    if cache_path.is_file() {
        let cache: FoldEncodingCacheV2 = serde_json::from_slice(
            &fs::read(&cache_path).map_err(|error| format!("read fold encoding cache: {error}"))?,
        )
        .map_err(|error| format!("decode fold encoding cache: {error}"))?;
        cache.validate(encoder.device_ordinal)?;
        if cache.key != key || cache.key_sha256 != key_sha256 {
            return Err("fold-local encoding cache identity drifted".into());
        }
        let matrices = read_output_matrices(&outputs)?;
        let actual = output_artifacts(&outputs)?;
        if actual != cache.sidecars {
            return Err("fold-local sidecar changed after cache publication".into());
        }
        return Ok(FoldEncoding {
            cell_fit: matrices[0].clone(),
            cell_hold: matrices[1].clone(),
            row_fit: matrices[2].clone(),
            row_hold: matrices[3].clone(),
            sidecars: actual,
            cache_key_sha256: key_sha256,
            cache_receipt_sha256: canonical_sha256(&cache)?,
            milliseconds: 0,
            cache_replay: true,
        });
    }
    if outputs.iter().any(|(_, path, _, _)| path.exists()) {
        return Err("partial or unreceipted fold-local sidecar output exists".into());
    }
    let remaining = ENCODING_BUDGET
        .checked_sub(gate_started.elapsed())
        .ok_or_else(|| "eight-minute encoding budget expired".to_string())?;
    let execution = run_encoder_with_timeout(encoder, &fit_path, &hold_path, &outputs, remaining)?;
    if !execution.status.success() {
        return Err(format!(
            "fold-local Hyperion encoder exited with {}",
            execution.status
        ));
    }
    execution.proof.validate_namespace()?;
    let matrices = read_output_matrices(&outputs)?;
    let sidecars = output_artifacts(&outputs)?;
    let cache = FoldEncodingCacheV2 {
        schema: FoldEncodingCacheV2::SCHEMA.into(),
        key,
        key_sha256,
        fit_row_order_sha256: fit_row_ids_sha256,
        hold_row_order_sha256: hold_row_ids_sha256,
        hold_target_policy: FoldEncodingCacheV2::DUMMY_HOLD_TARGET_POLICY.into(),
        sidecars: sidecars.clone(),
        proof: execution.proof,
    };
    cache.validate(encoder.device_ordinal)?;
    write_immutable(&cache_path, &canonical_json_bytes(&cache)?)?;
    Ok(FoldEncoding {
        cell_fit: matrices[0].clone(),
        cell_hold: matrices[1].clone(),
        row_fit: matrices[2].clone(),
        row_hold: matrices[3].clone(),
        sidecars,
        cache_key_sha256: cache.key_sha256.clone(),
        cache_receipt_sha256: canonical_sha256(&cache)?,
        milliseconds: cache.proof.elapsed_milliseconds,
        cache_replay: false,
    })
}

#[derive(Debug)]
struct EncoderRun {
    status: ExitStatus,
    proof: EncoderExecutionProofV2,
}

fn encoder_process_group_exists(pid: libc::pid_t) -> Result<bool, String> {
    #[cfg(unix)]
    unsafe {
        if libc::kill(-pid, 0) == 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        return Err(format!("probe Hyperion process group: {error}"));
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        Ok(false)
    }
}

fn wait_for_encoder_group_exit(pid: libc::pid_t, grace: Duration) -> Result<bool, String> {
    let deadline = Instant::now() + grace;
    loop {
        if !encoder_process_group_exists(pid)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn terminate_encoder_group(child: &mut std::process::Child, grace: Duration) -> Result<(), String> {
    let pid = child.id() as libc::pid_t;
    #[cfg(unix)]
    unsafe {
        if libc::kill(-pid, libc::SIGTERM) != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        {
            return Err(format!(
                "terminate Hyperion process group: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    let deadline = Instant::now() + grace;
    let mut leader_reaped = false;
    while Instant::now() < deadline {
        if !leader_reaped
            && child
                .try_wait()
                .map_err(|error| format!("poll Hyperion process group: {error}"))?
                .is_some()
        {
            leader_reaped = true;
        }
        if leader_reaped && !encoder_process_group_exists(pid)? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    #[cfg(unix)]
    unsafe {
        if libc::kill(-pid, libc::SIGKILL) != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
        {
            return Err(format!(
                "kill Hyperion process group: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    if !leader_reaped {
        child
            .wait()
            .map_err(|error| format!("reap Hyperion process group: {error}"))?;
    }
    if !wait_for_encoder_group_exit(pid, grace)? {
        return Err("Hyperion process group survived SIGKILL grace period".into());
    }
    Ok(())
}

fn run_encoder_with_timeout(
    encoder: &HyperionEncoderIdentityV1,
    fit: &Path,
    hold: &Path,
    outputs: &[(&str, PathBuf, usize, usize); 4],
    timeout: Duration,
) -> Result<EncoderRun, String> {
    let mut header = [0u8; 4];
    fs::File::open(&encoder.binary.path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| format!("read encoder binary header: {error}"))?;
    if header != [0x7f, b'E', b'L', b'F'] {
        return Err("Hyperion encoder must be a pinned native ELF binary".into());
    }
    let scales = encoder
        .scales
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let visible_device = encoder
        .runtime_versions
        .get("gpu_uuid")
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| encoder.device_ordinal.to_string());
    let encoder_identity_sha256 = encoder.identity()?;
    let argv_sha256 = canonical_sha256(&hyperion_argv_schema_v2())?;
    let environment_sha256 = canonical_sha256(&hyperion_environment_v2())?;
    let mut command = Command::new(&encoder.binary.path);
    command
        .arg("--train")
        .arg(fit)
        .arg("--test")
        .arg(hold)
        .arg("--out-cell-train")
        .arg(&outputs[0].1)
        .arg("--out-cell-test")
        .arg(&outputs[1].1)
        .arg("--out-row-train")
        .arg(&outputs[2].1)
        .arg("--out-row-test")
        .arg(&outputs[3].1)
        .arg("--weights")
        .arg(&encoder.weights.path)
        .arg("--manifest")
        .arg(&encoder.manifest.path)
        .arg("--sweeps")
        .arg(encoder.sweeps.to_string())
        .arg("--scales")
        .arg(scales)
        .arg("--seed")
        .arg(encoder.seed.to_string())
        .arg("--batch")
        .arg(encoder.batch_size.to_string())
        .env("CUDA_VISIBLE_DEVICES", &visible_device)
        .env("JAIN_HYPERION_CUDA_MAX_LOAD_RATIO", "1000000")
        .env("OMP_NUM_THREADS", "2")
        .env("OPENBLAS_NUM_THREADS", "2")
        .env("MKL_NUM_THREADS", "2")
        .env("BLIS_NUM_THREADS", "2")
        .env("VECLIB_MAXIMUM_THREADS", "2")
        .env("NUMEXPR_NUM_THREADS", "2")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    unsafe {
        let parent_pid = std::process::id() as libc::pid_t;
        command.pre_exec(move || {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent_pid {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "Hyperion parent exited during encoder spawn",
                ));
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("start fold-local Hyperion encoder: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "capture Hyperion encoder stdout failed".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "capture Hyperion encoder stderr failed".to_string())?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stdout = stdout;
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stderr = stderr;
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let started = Instant::now();
    let status = loop {
        let polled = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                let cleanup = terminate_encoder_group(&mut child, Duration::from_millis(250));
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                if let Err(cleanup) = cleanup {
                    return Err(format!(
                        "poll Hyperion encoder: {error}; cleanup failed: {cleanup}"
                    ));
                }
                return Err(format!("poll Hyperion encoder: {error}"));
            }
        };
        if let Some(status) = polled {
            if !encoder_process_group_exists(child.id() as libc::pid_t)? {
                break status;
            }
            terminate_encoder_group(&mut child, Duration::from_millis(250))?;
            return Err("Hyperion encoder exited with surviving process group".into());
        }
        if started.elapsed() >= timeout {
            let cleanup = terminate_encoder_group(&mut child, Duration::from_millis(250));
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            if let Err(error) = cleanup {
                return Err(error);
            }
            return Err("fold-local Hyperion encoder timed out".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| "Hyperion stdout capture thread panicked".to_string())?
        .map_err(|error| format!("capture Hyperion encoder stdout: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "Hyperion stderr capture thread panicked".to_string())?
        .map_err(|error| format!("capture Hyperion encoder stderr: {error}"))?;
    let stdout = String::from_utf8(stdout)
        .map_err(|error| format!("Hyperion encoder stdout is not UTF-8: {error}"))?;
    let stderr = String::from_utf8(stderr)
        .map_err(|error| format!("Hyperion encoder stderr is not UTF-8: {error}"))?;
    let expected_device = "cuda:0";
    let proof = EncoderExecutionProofV2 {
        encoder_identity_sha256,
        gpu_uuid: visible_device.clone(),
        cuda_visible_devices: visible_device,
        argv_sha256,
        environment_sha256,
        observed_device: expected_device.into(),
        stdout_sha256: raw_sha256(stdout.as_bytes()),
        stderr_sha256: raw_sha256(stderr.as_bytes()),
        stdout,
        stderr,
        elapsed_milliseconds: elapsed_milliseconds(started),
    };
    Ok(EncoderRun { status, proof })
}

fn read_output_matrices(
    outputs: &[(&str, PathBuf, usize, usize)],
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    outputs
        .iter()
        .map(|(role, path, rows, columns)| read_vs_matrix(path, *rows, *columns, role))
        .collect()
}

fn output_artifacts(
    outputs: &[(&str, PathBuf, usize, usize)],
) -> Result<Vec<MatrixArtifactV1>, String> {
    outputs
        .iter()
        .map(|(role, path, rows, columns)| {
            let bytes = fs::read(path)
                .map_err(|error| format!("read {role} sidecar {}: {error}", path.display()))?;
            Ok(MatrixArtifactV1 {
                role: (*role).into(),
                rows: *rows as u32,
                columns: *columns as u32,
                sha256: raw_sha256(&bytes),
            })
        })
        .collect()
}

fn read_vs_matrix(
    path: &Path,
    expected_rows: usize,
    expected_columns: usize,
    role: &str,
) -> Result<Vec<Vec<f64>>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("read {role} sidecar {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("{role} sidecar is not UTF-8: {error}"))?;
    let matrix = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(|character: char| character == ',' || character.is_ascii_whitespace())
                .filter(|field| !field.is_empty())
                .map(|field| {
                    field
                        .parse::<f64>()
                        .map_err(|error| format!("parse {role} sidecar value: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if matrix.len() != expected_rows
        || matrix
            .iter()
            .any(|row| row.len() != expected_columns || row.iter().any(|value| !value.is_finite()))
    {
        return Err(format!(
            "{role} sidecar shape/content is invalid; expected {expected_rows}x{expected_columns}"
        ));
    }
    Ok(matrix)
}

fn require_cuda_device(ordinal: u32) -> Result<(), String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("CUDA GPU unavailable: {error}"))?;
    if !output.status.success()
        || !String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.trim() == ordinal.to_string())
    {
        return Err(format!("pinned CUDA GPU ordinal {ordinal} is unavailable"));
    }
    Ok(())
}

fn verify_runtime_identity(encoder: &HyperionEncoderIdentityV1) -> Result<(), String> {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .map_err(|error| format!("read pinned Rust runtime identity: {error}"))?;
    let observed_rustc = String::from_utf8_lossy(&rustc.stdout).trim().to_owned();
    if !rustc.status.success() || encoder.runtime_versions["rustc"] != observed_rustc {
        return Err("Rust runtime differs from the encoder identity".into());
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("read pinned CUDA runtime identity: {error}"))?;
    if !output.status.success() {
        return Err("CUDA runtime identity query failed".into());
    }
    let expected_ordinal = encoder.device_ordinal.to_string();
    let output_text = String::from_utf8_lossy(&output.stdout);
    let observed = output_text
        .lines()
        .find_map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            (fields.len() == 4 && fields[0] == expected_ordinal).then_some(fields)
        })
        .ok_or_else(|| "pinned CUDA device is absent from the runtime query".to_string())?;
    if encoder.runtime_versions["gpu_name"] != observed[1]
        || encoder.runtime_versions["cuda_driver"] != observed[2]
        || encoder.runtime_versions["gpu_uuid"] != observed[3]
    {
        return Err("CUDA device or driver differs from the encoder identity".into());
    }
    Ok(())
}

fn parse_regression_csv(bytes: &[u8], expected_features: usize) -> Result<RegressionTable, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("authorized train CSV is not UTF-8: {error}"))?;
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field.trim().parse::<f64>().map_err(|error| {
                    format!(
                        "authorized train row {} is nonnumeric: {error}",
                        line_index + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() != expected_features + 1 {
            return Err(format!(
                "authorized train row {} has the wrong feature width",
                line_index + 1
            ));
        }
        let target = values.pop().unwrap();
        if !target.is_finite() {
            return Err(format!(
                "authorized train row {} has a nonfinite target",
                line_index + 1
            ));
        }
        values.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = 0.0;
            }
        });
        features.push(values);
        targets.push(target);
    }
    if features.is_empty() {
        return Err("authorized train CSV is empty".into());
    }
    Ok(RegressionTable { features, targets })
}

fn csv_for_rows(source: &RegressionTable, rows: &[usize], dummy_target: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for row in rows {
        let target = if dummy_target {
            0.0
        } else {
            source.targets[*row]
        };
        let mut fields = source.features[*row]
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        fields.push(target.to_string());
        bytes.extend_from_slice(fields.join(",").as_bytes());
        bytes.push(b'\n');
    }
    bytes
}

fn select_rows(values: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| values[*row].clone()).collect()
}

fn hash_selected_strings(values: &[String], rows: &[usize]) -> String {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend_from_slice(values[*row].as_bytes());
        bytes.push(0);
    }
    raw_sha256(&bytes)
}

fn hash_strings(values: &[String]) -> String {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    raw_sha256(&bytes)
}

fn hash_matrix(values: &[Vec<f64>]) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(values.first().map(Vec::len).unwrap_or(0) as u64).to_le_bytes());
    for row in values {
        for value in row {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    raw_sha256(&bytes)
}

fn hash_f64(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    raw_sha256(&bytes)
}

fn elapsed_milliseconds(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        raw_sha256(label.as_bytes())
    }

    #[test]
    fn hold_encoder_input_masks_targets_but_fit_input_does_not() {
        let first = RegressionTable {
            features: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            targets: vec![10.0, 20.0],
        };
        let mut changed = first.clone();
        changed.targets = vec![-1000.0, 5000.0];
        assert_eq!(
            csv_for_rows(&first, &[0, 1], true),
            csv_for_rows(&changed, &[0, 1], true)
        );
        assert_ne!(
            csv_for_rows(&first, &[0, 1], false),
            csv_for_rows(&changed, &[0, 1], false)
        );
    }

    #[test]
    fn excluded_rows_cannot_change_fit_encoder_input() {
        let first = RegressionTable {
            features: vec![vec![1.0], vec![2.0], vec![3.0]],
            targets: vec![10.0, 20.0, 30.0],
        };
        let mut changed = first.clone();
        changed.features[2][0] = 99_999.0;
        changed.targets[2] = -99_999.0;
        assert_eq!(
            csv_for_rows(&first, &[0, 1], false),
            csv_for_rows(&changed, &[0, 1], false)
        );
        assert_ne!(
            csv_for_rows(&first, &[2], true),
            csv_for_rows(&changed, &[2], true)
        );
    }

    #[test]
    fn unavailable_runtime_materializes_a_valid_raw_decision() {
        let started = Instant::now();
        let decision = raw_fallback_decision(
            "opaque",
            &digest("training"),
            &digest("predictor"),
            &digest("encoder"),
            &digest("cv5"),
            &digest("cv7"),
            "pinned CUDA GPU unavailable",
            started,
        );
        decision.validate().unwrap();
        assert_eq!(decision.selected_view, FeatureViewKindV1::Raw);
        assert!(decision.wall_clock_milliseconds <= WALL_CLOCK_BUDGET.as_millis() as u64);
    }
}
