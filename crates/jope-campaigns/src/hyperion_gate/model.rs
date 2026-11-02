use super::*;

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
