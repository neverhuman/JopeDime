use super::*;

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
