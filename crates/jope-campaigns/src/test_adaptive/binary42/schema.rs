use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenWinPolicy {
    pub(super) non_saturated: String,
    pub(super) saturated: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenDataset {
    pub(super) order: usize,
    pub(super) id: String,
    pub(super) display_name: String,
    pub(super) category: String,
    pub(super) train_rows: usize,
    pub(super) test_rows: usize,
    pub(super) features: usize,
    pub(super) train_path: String,
    pub(super) train_sha256: String,
    pub(super) test_path: String,
    pub(super) test_sha256: String,
    pub(super) leader_method: String,
    pub(super) leader_auc: f64,
    pub(super) leader_report_path: String,
    pub(super) leader_report_sha256: String,
    pub(super) win_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenCohort {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) source: String,
    pub(super) track_id: String,
    pub(super) cohort_registry_sha256: String,
    pub(super) dataset_registry_sha256: String,
    pub(super) scores_sha256: String,
    pub(super) win_policy: FrozenWinPolicy,
    pub(super) datasets: Vec<FrozenDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct CampaignConfig {
    pub(super) population: usize,
    pub(super) shards: usize,
    pub(super) candidates_per_shard: usize,
    pub(super) elites: usize,
    pub(super) parent_pool: usize,
    pub(super) tournament: usize,
    pub(super) mutation_probability_numerator: usize,
    pub(super) mutation_probability_denominator: usize,
    pub(super) crossover_probability_numerator: usize,
    pub(super) crossover_probability_denominator: usize,
    pub(super) candidate_threads: usize,
    pub(super) cpus_per_shard: usize,
    pub(super) gpus_per_shard: usize,
    pub(super) generation_cap: Option<usize>,
    pub(super) infrastructure_retries: usize,
    pub(super) confirmation_nodes: Vec<String>,
    pub(super) ranking: String,
    pub(super) win_rule: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) shard_sizes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) execution_backend: Option<String>,
}

impl CampaignConfig {
    pub(super) fn fixed() -> Self {
        Self {
            population: POPULATION,
            shards: NODES.len(),
            candidates_per_shard: SHARD_SIZE,
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            mutation_probability_numerator: 3,
            mutation_probability_denominator: 4,
            crossover_probability_numerator: 1,
            crossover_probability_denominator: 4,
            candidate_threads: 1,
            cpus_per_shard: SHARD_SIZE,
            gpus_per_shard: 0,
            generation_cap: None,
            infrastructure_retries: 1,
            confirmation_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            ranking: "test_auc descending, test_logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: "evolved descendant with test_auc strictly greater than both frozen leader and generation-zero best".to_string(),
            shard_sizes: Vec::new(),
            execution_backend: None,
        }
    }

    pub(super) fn all(spec: &FrozenDataset) -> Self {
        Self {
            population: POPULATION,
            shards: NODES.len(),
            candidates_per_shard: ALL_SHARD_SIZES[0],
            elites: ELITES,
            parent_pool: PARENT_POOL,
            tournament: TOURNAMENT,
            mutation_probability_numerator: 3,
            mutation_probability_denominator: 4,
            crossover_probability_numerator: 1,
            crossover_probability_denominator: 4,
            candidate_threads: 1,
            cpus_per_shard: ALL_SHARD_SIZES[0],
            gpus_per_shard: 0,
            generation_cap: Some(ALL_MAX_GENERATIONS),
            infrastructure_retries: 2,
            confirmation_nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            ranking: "test_auc descending, test_logloss ascending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: match spec.win_policy.as_str() {
                "exact_auc_1_tie" => {
                    "non-seed evolved descendant from generation >= 1 with exact test_auc 1.0"
                }
                _ => "non-seed evolved descendant from generation >= 1 with test_auc strictly greater than both frozen leader and generation-zero best",
            }
            .to_string(),
            shard_sizes: ALL_SHARD_SIZES.to_vec(),
            execution_backend: Some(CPP17_BACKEND.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CampaignManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) run_id: String,
    pub(super) dataset: String,
    pub(super) frozen_cohort_file: String,
    pub(super) frozen_cohort_sha256: String,
    pub(super) binary_seed_sha256: String,
    pub(super) release_binary_sha256: String,
    pub(super) scq_binary_sha256: String,
    pub(super) dataset_binding: FrozenDataset,
    pub(super) config: CampaignConfig,
    pub(super) started_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cohort_staging_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) winner_seed_snapshot_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BaselineManifest {
    pub(super) schema: String,
    pub(super) test_adaptive: bool,
    pub(super) benchmark_valid: bool,
    pub(super) dataset: String,
    pub(super) leader_method: String,
    pub(super) leader_auc: f64,
    pub(super) leader_report_path: String,
    pub(super) leader_report_sha256: String,
    pub(super) generation_zero_best_auc: Option<f64>,
}
