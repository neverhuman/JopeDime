use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CandidateBackendReceipt {
    pub(super) name: String,
    pub(super) language_standard: String,
    pub(super) compiler: String,
    pub(super) compiler_flags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) source_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) codec_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) compile_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerCandidate {
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) receipt_file: String,
    pub(super) receipt_sha256: String,
    pub(super) predictions_file: String,
    pub(super) predictions_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) kind: String,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) shard: usize,
    pub(super) node: String,
    pub(super) hostname: String,
    pub(super) job_id: String,
    pub(super) manifest_sha256: String,
    pub(super) logical_slots: usize,
    pub(super) fresh_candidates: usize,
    pub(super) cpus_reserved: usize,
    pub(super) candidate_threads: usize,
    pub(super) candidates: Vec<WorkerCandidate>,
    pub(super) started_at_unix_ms: u128,
    pub(super) finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct JobReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) job_id: String,
    pub(super) node: String,
    pub(super) attempt: usize,
    pub(super) manifest_sha256: String,
    pub(super) scq_show_sha256: String,
    pub(super) worker_receipt_sha256: String,
    pub(super) state: String,
    pub(super) exit_code: i64,
}

#[derive(Clone, Debug)]
pub(super) struct FreshOutput {
    pub(super) receipt: CandidateReceipt,
    pub(super) receipt_bytes: Vec<u8>,
    pub(super) prediction_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LogicalSlot {
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) assigned_node: String,
    pub(super) member: MemberRecord,
    pub(super) cache_disposition: String,
    pub(super) cache_source_generation: Option<usize>,
    pub(super) cache_source_slot: Option<usize>,
    pub(super) evaluation_node: String,
    pub(super) scq_job_id: String,
    pub(super) receipt_sha256: String,
    pub(super) prediction_sha256: String,
    pub(super) score: Option<CandidateScore>,
    pub(super) failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) logical_slots: usize,
    pub(super) shard_size: usize,
    pub(super) shard_nodes: Vec<String>,
    pub(super) members: Vec<MemberRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) shard_sizes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QualifyingWinner {
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) semantic_sha256: String,
    pub(super) auc: f64,
    pub(super) logloss: f64,
    pub(super) complexity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationCheckpoint {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) previous_checkpoint_sha256: Option<String>,
    pub(super) logical_slots: usize,
    pub(super) fresh_evaluations: usize,
    pub(super) cache_reuses: usize,
    pub(super) valid_scores: usize,
    pub(super) generation_zero_best_auc: f64,
    pub(super) best_slot: usize,
    pub(super) best_auc: f64,
    pub(super) slots: Vec<LogicalSlot>,
    pub(super) ranked_semantics: Vec<String>,
    pub(super) next_population: Vec<MemberRecord>,
    pub(super) qualifying_winner: Option<QualifyingWinner>,
}
