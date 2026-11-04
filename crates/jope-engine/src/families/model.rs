use super::*;

/// New four-task campaigns may prepare any supported task. The live
/// regression registry path still requires `source.task == "regression"`.
pub fn campaign_task_admissible(task: &str) -> bool {
    crate::task_contract::supported_task(task)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub path: PathBuf,
    pub sha256: String,
}
impl FileBinding {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            path: path.canonicalize()?,
            sha256: file_hash(path)?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredDataset {
    pub dataset: String,
    pub source_key: String,
    pub audit_family: String,
    pub metadata_sha256: String,
    pub entity_review_sha256: Option<String>,
    pub preparation_eligible: bool,
    pub eligibility_reason: String,
    pub prepared: Vec<FileBinding>,
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyAssignment {
    pub family: String,
    pub partition: String,
    pub assignment_method: String,
    pub assignment_hash: Option<String>,
    pub bucket: Option<u64>,
    pub source_keys: BTreeSet<String>,
    pub datasets: Vec<RegisteredDataset>,
    pub holds: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    pub schema: String,
    pub seed: u64,
    pub registry_id: String,
    pub qualification: bool,
    pub full_inventory_complete: bool,
    pub independent_selection_eligible: bool,
    pub parent: Option<FileBinding>,
    pub audit: FileBinding,
    pub decisions: Vec<FileBinding>,
    pub reviews: Vec<FileBinding>,
    pub prepared_inputs: Vec<FileBinding>,
    pub families: BTreeMap<String, FamilyAssignment>,
    pub held_review_datasets: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub growth: Vec<FileBinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryBinding {
    pub registry_sha256: String,
    pub registry_id: String,
    pub family: String,
    pub partition: String,
    pub metadata_sha256: String,
    pub qualification: bool,
    pub full_inventory_complete: bool,
    pub entity_review_sha256: Option<String>,
}

pub fn stable_assignment(family: &str) -> (String, u64, String) {
    let digest = hash(format!("{SEED}:family:{family}").as_bytes());
    let bucket = u64::from_str_radix(&digest[..16], 16).expect("SHA256 prefix") % 1_000_000;
    let partition = if bucket < 800_000 {
        "meta_train"
    } else if bucket < 900_000 {
        "meta_validation"
    } else {
        "meta_test"
    };
    (digest, bucket, partition.into())
}

pub fn stable_component_assignment(component: &str) -> (String, u64, String) {
    let digest = hash(format!("{SEED}:component:{component}").as_bytes());
    let bucket = u64::from_str_radix(&digest[..16], 16).expect("SHA256 prefix") % 1_000_000;
    let partition = if bucket < 800_000 {
        "meta_train"
    } else {
        "development_validation"
    };
    (digest, bucket, partition.into())
}
