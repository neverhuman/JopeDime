use super::*;

#[derive(Debug, Clone)]
pub(super) struct RegressionTable {
    pub(super) features: Vec<Vec<f64>>,
    pub(super) targets: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(super) struct FoldEncoding {
    pub(super) cell_fit: Vec<Vec<f64>>,
    pub(super) cell_hold: Vec<Vec<f64>>,
    pub(super) row_fit: Vec<Vec<f64>>,
    pub(super) row_hold: Vec<Vec<f64>>,
    pub(super) sidecars: Vec<MatrixArtifactV1>,
    pub(super) cache_key_sha256: String,
    pub(super) cache_receipt_sha256: String,
    pub(super) milliseconds: u64,
    pub(super) cache_replay: bool,
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
            || !encoder_output_proves_device(&self.stdout, &self.stderr, expected)
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
pub(super) struct PrecomputedTrainFeatureBlocks {
    pub(super) cell: Vec<Vec<f64>>,
    pub(super) row128: Option<Vec<Vec<f64>>>,
    pub(super) receipt: PrecomputedTrainFeatureReceiptV1,
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
