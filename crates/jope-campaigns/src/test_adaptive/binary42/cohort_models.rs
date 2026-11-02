use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AllCohortConfig {
    pub(super) active_datasets: usize,
    pub(super) population: usize,
    pub(super) generation_numbers: Vec<usize>,
    pub(super) shard_sizes: Vec<usize>,
    pub(super) node_cpu_budget_per_active_set: BTreeMap<String, usize>,
    pub(super) candidate_threads: usize,
    pub(super) gpus_per_task: usize,
    pub(super) infrastructure_failover_attempts: usize,
    pub(super) admission_order: String,
    pub(super) completion_policy: String,
    pub(super) execution_backend: String,
    pub(super) compiler_requirement: String,
    pub(super) compiler_flags: Vec<String>,
}

impl AllCohortConfig {
    pub(super) fn fixed() -> Self {
        Self {
            active_datasets: ALL_ACTIVE_DATASETS,
            population: POPULATION,
            generation_numbers: (0..ALL_MAX_GENERATIONS).collect(),
            shard_sizes: ALL_SHARD_SIZES.to_vec(),
            node_cpu_budget_per_active_set: NODES
                .iter()
                .zip(ALL_SHARD_SIZES)
                .map(|(node, shard)| ((*node).to_string(), shard * ALL_ACTIVE_DATASETS))
                .collect(),
            candidate_threads: 1,
            gpus_per_task: 0,
            infrastructure_failover_attempts: 2,
            admission_order:
                "train_rows * features descending, frozen manifest order ascending".to_string(),
            completion_policy:
                "work-conserving admission; each dataset stops after a completed winning barrier or generation 9 exhaustion"
                    .to_string(),
            execution_backend: CPP17_BACKEND.to_string(),
            compiler_requirement: "g++ 13.3".to_string(),
            compiler_flags: CPP17_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AllCohortManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) frozen_cohort_file: String,
    pub(super) frozen_cohort_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) scq_configured_path: String,
    pub(super) scq_resolved_path: String,
    pub(super) scq_binary_sha256: String,
    pub(super) config: AllCohortConfig,
    pub(super) admission_dataset_ids: Vec<String>,
    pub(super) winner_seed_sources: Vec<WinnerSeedSource>,
    pub(super) started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AllStagedDataset {
    pub(super) dataset: String,
    pub(super) train_path: String,
    pub(super) train_sha256: String,
    pub(super) test_path: String,
    pub(super) test_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct AllStagedNode {
    pub(super) node: String,
    pub(super) hostname: String,
    pub(super) logical_cpus: usize,
    pub(super) cpp17_compiler: String,
    pub(super) bundle_root: String,
    pub(super) binary_path: String,
    pub(super) binary_sha256: String,
    pub(super) canary_job_id: String,
    pub(super) canary_show_sha256: String,
    pub(super) canary_cpus_reserved: usize,
    pub(super) datasets: Vec<AllStagedDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AllStagingReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) release_binary_sha256: String,
    pub(super) scq_configured_path: String,
    pub(super) scq_resolved_path: String,
    pub(super) scq_binary_sha256: String,
    pub(super) doctor_output_sha256: String,
    pub(super) nodes: Vec<AllStagedNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExhaustedReceipt {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) status: String,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) completed_generations: usize,
    pub(super) logical_slots: usize,
    pub(super) best_generation: usize,
    pub(super) best_slot: usize,
    pub(super) best_semantic_sha256: String,
    pub(super) best_auc: f64,
    pub(super) best_logloss: f64,
    pub(super) best_prediction_sha256: String,
    pub(super) best_receipt_sha256: String,
    pub(super) lineage_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExhaustedMarker {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) status: String,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) completed_generations: usize,
    pub(super) logical_slots: usize,
    pub(super) best_semantic_sha256: String,
    pub(super) best_auc: f64,
    pub(super) final_receipt_sha256: String,
    pub(super) completed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct DatasetValidationSummary {
    pub(super) dataset: String,
    pub(super) display_name: String,
    pub(super) frozen_order: usize,
    pub(super) status: String,
    pub(super) completed_generations: usize,
    pub(super) logical_slots: usize,
    pub(super) unique_genomes: usize,
    pub(super) best_auc: Option<f64>,
    pub(super) winner_generation: Option<usize>,
    pub(super) winner_semantic_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AllTerminalSummary {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) terminal_datasets: usize,
    pub(super) winners: usize,
    pub(super) exhausted: usize,
    pub(super) logical_evaluations: usize,
    pub(super) completed_at_unix_ms: u128,
}
