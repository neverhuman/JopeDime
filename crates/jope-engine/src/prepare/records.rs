use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cut {
    pub index: usize,
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub training_sha256: String,
    pub validation_sha256: String,
    pub preprocessing_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prepared {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<ViewSource>,
    pub schema: String,
    pub dataset: String,
    pub family: String,
    pub task: String,
    pub meta_partition: String,
    pub seed: u64,
    #[serde(default)]
    pub family_partition_frozen: bool,
    pub clean_evaluation: bool,
    pub clean_status: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qualification_only: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluation_limitations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entity_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_registry_binding: Option<crate::families::RegistryBinding>,
    pub original_train_sha256: String,
    pub original_test_sha256: String,
    pub selected_train_indices: Vec<usize>,
    pub selected_test_indices: Vec<usize>,
    pub target_last: bool,
    pub rows_total: usize,
    pub cuts: Vec<Cut>,
}
