use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectedGenerationZeroSeed {
    pub(super) island: usize,
    pub(super) local_slot: usize,
    pub(super) seed_class: String,
    pub(super) provenance: SeedProvenance,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationZeroSeedSnapshot {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) provenance_group: String,
    pub(super) arm: String,
    pub(super) strategy: String,
    pub(super) parent_run_id: String,
    pub(super) parent_checkpoint_sha256: String,
    pub(super) parent_seed: SeedProvenance,
    pub(super) population_sha256: String,
    pub(super) generation_zero_can_terminate: bool,
    pub(super) candidates: Vec<SelectedGenerationZeroSeed>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct NodeResources {
    pub(super) node: String,
    pub(super) online_cores: usize,
    pub(super) available_ram_bytes: u64,
    pub(super) free_disk_bytes: u64,
    pub(super) swap_used_bytes: u64,
    pub(super) competing_processes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct PreflightReceipt {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) required_cores: usize,
    pub(super) required_available_ram_bytes: u64,
    pub(super) required_free_disk_bytes: u64,
    pub(super) maximum_swap_used_bytes_exclusive: u64,
    pub(super) active_scq_jobs: Vec<String>,
    pub(super) nodes: Vec<NodeResources>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct AggressiveManifest {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) selected_backend: String,
    pub(super) cpp17_admitted: bool,
    pub(super) cpp17_rejection: String,
    pub(super) backend_admission_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) scq_binary_sha256: String,
    pub(super) preflight_sha256: String,
    pub(super) staging_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) panel_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) panel_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) leader_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) leader_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) evolution_view_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) arm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) leaderboard_default: Option<bool>,
    pub(super) registry_root: String,
    pub(super) population: usize,
    pub(super) islands: usize,
    pub(super) population_per_island: usize,
    pub(super) workers_per_node: usize,
    pub(super) generation_limit: Option<usize>,
    pub(super) routing: String,
    pub(super) retry: String,
    pub(super) stop_rule: String,
    pub(super) datasets: Vec<DatasetSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct AggressiveCheckpoint {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) generation: usize,
    pub(super) previous_checkpoint_sha256: Option<String>,
    pub(super) logical_slots: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) cumulative_fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) cumulative_cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(super) unique_population: usize,
    pub(super) valid_per_island: Vec<usize>,
    pub(super) ranked_slots: Vec<usize>,
    pub(super) best_slot: usize,
    pub(super) best_rmse: f64,
    pub(super) winner_slot: Option<usize>,
    pub(super) slots: Vec<SlotOutcome>,
    pub(super) next_population: Vec<Member>,
}

pub(super) fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Confirmation {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) generation: usize,
    pub(super) slot: usize,
    pub(super) checkpoint_sha256: String,
    pub(super) prediction_bits_sha256: String,
    pub(super) rmse_bits: u64,
    pub(super) nodes: Vec<String>,
    pub(super) receipt_files: Vec<String>,
    pub(super) receipt_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct DatasetResult {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) dataset_hash: String,
    pub(super) outcome: String,
    pub(super) stopping_generation: usize,
    pub(super) leader_rmse: f64,
    pub(super) best_rmse: f64,
    pub(super) best_slot: usize,
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    pub(super) terminal_checkpoint_sha256: String,
    pub(super) confirmation_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Summary {
    pub(super) schema: String,
    pub(super) benchmark_valid: bool,
    pub(super) test_adaptive: bool,
    pub(super) run_id: String,
    pub(super) selected_backend: String,
    pub(super) dataset_count: usize,
    pub(super) confirmed_wins: usize,
    pub(super) unresolved: usize,
    pub(super) results: Vec<DatasetResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct EvolutionViewDataset {
    pub(super) dataset_hash: String,
    pub(super) source_train_sha256: String,
    pub(super) source_test_sha256: String,
    pub(super) train_sha256: String,
    pub(super) test_sha256: String,
    pub(super) removed_train_header: bool,
    pub(super) removed_test_header: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct EvolutionViewManifest {
    pub(super) schema: String,
    pub(super) campaign_id: String,
    pub(super) panel_manifest_sha256: String,
    pub(super) source_registry_root: String,
    pub(super) view_sha256: String,
    pub(super) datasets: Vec<EvolutionViewDataset>,
}

#[derive(Clone, Debug)]
pub(super) struct Golden46Bindings {
    pub(super) panel_sha256: String,
    pub(super) leader_sha256: String,
    pub(super) evolution_view_manifest_sha256: String,
    pub(super) arm: String,
    pub(super) leaderboard_default: bool,
}
