use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct MemberRecord {
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) complexity: usize,
    pub(super) ancestry: BinaryAncestry,
}

impl MemberRecord {
    pub(super) fn from_binary(member: BinaryMember) -> Result<Self> {
        Ok(Self {
            complexity: binary_genome_complexity(&member.canonical_genome)?,
            canonical_genome: member.canonical_genome,
            semantic_sha256: member.semantic_sha256,
            ancestry: member.ancestry,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct WinnerSeedSource {
    pub(super) source_run_id: String,
    pub(super) source_dataset: String,
    pub(super) complete_file: String,
    pub(super) complete_sha256: String,
    pub(super) final_receipt_file: String,
    pub(super) final_receipt_sha256: String,
    pub(super) confirmation_file: String,
    pub(super) confirmation_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) auc_bits: u64,
    pub(super) frozen_leader_auc_bits: u64,
    pub(super) generation_zero_best_auc_bits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SeedManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) seed_file_sha256: String,
    pub(super) population: usize,
    pub(super) members: Vec<MemberRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) winner_seed_sources: Vec<WinnerSeedSource>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct StagedNode {
    pub(super) node: String,
    pub(super) hostname: String,
    pub(super) bundle_root: String,
    pub(super) binary_path: String,
    pub(super) binary_sha256: String,
    pub(super) train_path: String,
    pub(super) train_sha256: String,
    pub(super) test_path: String,
    pub(super) test_sha256: String,
    pub(super) canary_job_id: String,
    pub(super) canary_show_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StagingReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) release_binary_sha256: String,
    pub(super) scq_binary_sha256: String,
    pub(super) doctor_output_sha256: String,
    pub(super) dataset: String,
    pub(super) nodes: Vec<StagedNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cohort_staging_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct CandidateScore {
    pub(super) auc: f64,
    pub(super) logloss: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheEntry {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) semantic_sha256: String,
    pub(super) receipt_sha256: String,
    pub(super) prediction_sha256: String,
    pub(super) score: Option<CandidateScore>,
    pub(super) failure: Option<String>,
    pub(super) evaluation_node: String,
    pub(super) scq_job_id: String,
    pub(super) first_generation: usize,
    pub(super) first_slot: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) runtime_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenomeObject {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShardSlot {
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) member: MemberRecord,
    pub(super) fresh: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ShardManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) kind: String,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) shard: usize,
    pub(super) node: String,
    pub(super) job_id: String,
    pub(super) attempt: usize,
    pub(super) release_binary_sha256: String,
    pub(super) train_path: String,
    pub(super) train_sha256: String,
    pub(super) test_path: String,
    pub(super) test_sha256: String,
    pub(super) train_rows: usize,
    pub(super) test_rows: usize,
    pub(super) features: usize,
    pub(super) cpus_reserved: usize,
    pub(super) gpus_per_task: usize,
    pub(super) candidate_threads: usize,
    pub(super) slots: Vec<ShardSlot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cohort_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) scq_binary_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CandidateReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) kind: String,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) attempt: usize,
    pub(super) semantic_sha256: String,
    pub(super) canonical_genome: String,
    pub(super) complexity: usize,
    pub(super) release_binary_sha256: String,
    pub(super) train_sha256: String,
    pub(super) test_sha256: String,
    pub(super) train_rows: usize,
    pub(super) test_rows: usize,
    pub(super) features: usize,
    pub(super) node: String,
    pub(super) hostname: String,
    pub(super) scq_job_id: String,
    pub(super) candidate_threads: usize,
    pub(super) cpu: usize,
    pub(super) runtime_seconds: f64,
    pub(super) fit_seconds: Option<f64>,
    pub(super) predict_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) backend: Option<CandidateBackendReceipt>,
    pub(super) predictions_sha256: String,
    pub(super) score: Option<CandidateScore>,
    pub(super) failure: Option<String>,
}
