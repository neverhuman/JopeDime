use super::*;

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
pub(crate) struct PrecomputedTrainFeatureBlocks {
    pub(crate) cell: Vec<Vec<f64>>,
    pub(crate) row128: Option<Vec<Vec<f64>>>,
    pub(crate) receipt: PrecomputedTrainFeatureReceiptV1,
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
