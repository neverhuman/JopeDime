use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetStageV1 {
    Pilot,
    Confirmation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenDatasetV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub train_relative_path: String,
    pub test_relative_path: String,
    pub metadata_relative_path: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub metadata_sha256: String,
    pub train_rows: u32,
    pub feature_count: u32,
    pub source_family: String,
    pub stage: DatasetStageV1,
    pub current_sotarena_win: bool,
}

impl FrozenDatasetV1 {
    pub fn validate(&self) -> Result<(), String> {
        short_dataset_id(&self.dataset_id)?;
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.train_sha256.as_str()),
            ("test", self.test_sha256.as_str()),
            ("metadata", self.metadata_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if !self.dataset_sha256.starts_with(&self.dataset_id) {
            return Err("full dataset digest must start with its frozen short ID".into());
        }
        if self.train_rows == 0 || self.feature_count == 0 {
            return Err("dataset rows and features must be positive".into());
        }
        for (name, path) in [
            ("training", self.train_relative_path.as_str()),
            ("test", self.test_relative_path.as_str()),
            ("metadata", self.metadata_relative_path.as_str()),
        ] {
            let path = std::path::Path::new(path);
            if path.is_absolute()
                || path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                return Err(format!("{name} dataset path must be a safe relative path"));
            }
        }
        nonempty("source family", &self.source_family)
    }
}
