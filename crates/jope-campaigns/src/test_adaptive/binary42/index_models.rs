use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvaluationIndexRecord {
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) ancestry_operation: String,
    pub(super) ancestry_parents: Vec<String>,
    pub(super) cache_disposition: String,
    pub(super) assigned_node: String,
    pub(super) evaluation_node: String,
    pub(super) scq_job_id: String,
    pub(super) auc: Option<f64>,
    pub(super) logloss: Option<f64>,
    pub(super) complexity: usize,
    pub(super) receipt_sha256: String,
    pub(super) prediction_sha256: String,
    pub(super) failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenomeIndexRecord {
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
    pub(super) ancestry_operation: String,
    pub(super) ancestry_parents: Vec<String>,
    pub(super) first_generation: usize,
    pub(super) first_slot: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConfirmationRecord {
    pub(super) node: String,
    pub(super) job_id: String,
    pub(super) receipt_sha256: String,
    pub(super) prediction_sha256: String,
    pub(super) auc: f64,
    pub(super) logloss: f64,
    pub(super) auc_bits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConfirmationManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) selected_slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) attempt: usize,
    pub(super) original_prediction_sha256: String,
    pub(super) confirmations: Vec<ConfirmationRecord>,
    pub(super) prediction_hashes_identical: bool,
    pub(super) auc_bits_identical: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LineageRecord {
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
    pub(super) ancestry: BinaryAncestry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WinnerReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
    pub(super) auc: f64,
    pub(super) logloss: f64,
    pub(super) frozen_leader_auc: f64,
    pub(super) generation_zero_best_auc: f64,
    pub(super) margin_vs_frozen_leader: f64,
    pub(super) margin_vs_generation_zero: f64,
    pub(super) prediction_sha256: String,
    pub(super) source_receipt_sha256: String,
    pub(super) four_node_confirmation_sha256: String,
    pub(super) lineage_sha256: String,
    pub(super) release_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WinnerArtifact {
    pub(super) path: String,
    pub(super) sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WinnerArtifactManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) artifacts: Vec<WinnerArtifact>,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct AllEvaluationIndexRecord {
    pub(super) dataset: String,
    pub(super) display_name: String,
    pub(super) frozen_order: usize,
    #[serde(flatten)]
    pub(super) evaluation: EvaluationIndexRecord,
}
