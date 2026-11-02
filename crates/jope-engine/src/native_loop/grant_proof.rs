//! Read-only scientific eligibility proof for an independently owned provisioner.
use super::*;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrantProofInputs {
    pub root: PathBuf,
    pub loop_state: PathBuf,
    pub graph_state: PathBuf,
    pub worker_state: PathBuf,
    pub tenant: String,
    pub project: String,
    pub config: Artifact,
    pub request: Artifact,
    pub admission: Artifact,
    pub task_manifest: Artifact,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeGrantEligibilityProof {
    pub schema: String,
    pub inputs: GrantProofInputs,
    pub run: String,
    pub generation: u64,
    pub role: Role,
    pub scientific_policy: Artifact,
    pub candidate: Artifact,
    pub accepted_parent: Artifact,
    pub original_proposal: Artifact,
    pub evaluations: usize,
    pub seconds: u64,
    pub task_deadline: Timestamp,
    pub task_manifest_sha256: String,
    pub predecessor_receipts: BTreeMap<String, Artifact>,
    pub verifier_executable_sha256: String,
    pub verified_unix_ms: u64,
    pub authorization_issued: bool,
}
mod budget;
mod request;
