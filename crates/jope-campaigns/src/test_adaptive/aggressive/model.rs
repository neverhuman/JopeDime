use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectedGenerationZeroSeed {
    pub(crate) island: usize,
    pub(crate) local_slot: usize,
    pub(crate) seed_class: String,
    pub(crate) provenance: SeedProvenance,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerationZeroSeedSnapshot {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) dataset_hash: String,
    pub(crate) provenance_group: String,
    pub(crate) arm: String,
    pub(crate) strategy: String,
    pub(crate) parent_run_id: String,
    pub(crate) parent_checkpoint_sha256: String,
    pub(crate) parent_seed: SeedProvenance,
    pub(crate) population_sha256: String,
    pub(crate) generation_zero_can_terminate: bool,
    pub(crate) candidates: Vec<SelectedGenerationZeroSeed>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct NodeResources {
    pub(crate) node: String,
    pub(crate) online_cores: usize,
    pub(crate) available_ram_bytes: u64,
    pub(crate) free_disk_bytes: u64,
    pub(crate) swap_used_bytes: u64,
    pub(crate) competing_processes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreflightReceipt {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) required_cores: usize,
    pub(crate) required_available_ram_bytes: u64,
    pub(crate) required_free_disk_bytes: u64,
    pub(crate) maximum_swap_used_bytes_exclusive: u64,
    pub(crate) active_scq_jobs: Vec<String>,
    pub(crate) nodes: Vec<NodeResources>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggressiveManifest {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) selected_backend: String,
    pub(crate) cpp17_admitted: bool,
    pub(crate) cpp17_rejection: String,
    pub(crate) backend_admission_sha256: String,
    pub(crate) release_binary_sha256: String,
    pub(crate) scq_binary_sha256: String,
    pub(crate) preflight_sha256: String,
    pub(crate) staging_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) panel_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) panel_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) leader_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) leader_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) evolution_view_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) arm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) leaderboard_default: Option<bool>,
    pub(crate) registry_root: String,
    pub(crate) population: usize,
    pub(crate) islands: usize,
    pub(crate) population_per_island: usize,
    pub(crate) workers_per_node: usize,
    pub(crate) generation_limit: Option<usize>,
    pub(crate) routing: String,
    pub(crate) retry: String,
    pub(crate) stop_rule: String,
    pub(crate) datasets: Vec<DatasetSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AggressiveCheckpoint {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) dataset_hash: String,
    pub(crate) generation: usize,
    pub(crate) previous_checkpoint_sha256: Option<String>,
    pub(crate) logical_slots: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) cumulative_fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) cumulative_cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) unique_population: usize,
    pub(crate) valid_per_island: Vec<usize>,
    pub(crate) ranked_slots: Vec<usize>,
    pub(crate) best_slot: usize,
    pub(crate) best_rmse: f64,
    pub(crate) winner_slot: Option<usize>,
    pub(crate) slots: Vec<SlotOutcome>,
    pub(crate) next_population: Vec<Member>,
}

pub(crate) fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Confirmation {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) dataset_hash: String,
    pub(crate) generation: usize,
    pub(crate) slot: usize,
    pub(crate) checkpoint_sha256: String,
    pub(crate) prediction_bits_sha256: String,
    pub(crate) rmse_bits: u64,
    pub(crate) nodes: Vec<String>,
    pub(crate) receipt_files: Vec<String>,
    pub(crate) receipt_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatasetResult {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) dataset_hash: String,
    pub(crate) outcome: String,
    pub(crate) stopping_generation: usize,
    pub(crate) leader_rmse: f64,
    pub(crate) best_rmse: f64,
    pub(crate) best_slot: usize,
    pub(crate) canonical_genome: String,
    pub(crate) semantic_sha256: String,
    pub(crate) terminal_checkpoint_sha256: String,
    pub(crate) confirmation_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Summary {
    pub(crate) schema: String,
    pub(crate) benchmark_valid: bool,
    pub(crate) test_adaptive: bool,
    pub(crate) run_id: String,
    pub(crate) selected_backend: String,
    pub(crate) dataset_count: usize,
    pub(crate) confirmed_wins: usize,
    pub(crate) unresolved: usize,
    pub(crate) results: Vec<DatasetResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvolutionViewDataset {
    pub(crate) dataset_hash: String,
    pub(crate) source_train_sha256: String,
    pub(crate) source_test_sha256: String,
    pub(crate) train_sha256: String,
    pub(crate) test_sha256: String,
    pub(crate) removed_train_header: bool,
    pub(crate) removed_test_header: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvolutionViewManifest {
    pub(crate) schema: String,
    pub(crate) campaign_id: String,
    pub(crate) panel_manifest_sha256: String,
    pub(crate) source_registry_root: String,
    pub(crate) view_sha256: String,
    pub(crate) datasets: Vec<EvolutionViewDataset>,
}

#[derive(Clone, Debug)]
pub(crate) struct Golden46Bindings {
    pub(crate) panel_sha256: String,
    pub(crate) leader_sha256: String,
    pub(crate) evolution_view_manifest_sha256: String,
    pub(crate) arm: String,
    pub(crate) leaderboard_default: bool,
}
