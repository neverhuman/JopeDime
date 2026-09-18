//! Four-island generations 20-29 continuation of the immutable Golden42
//! test-adaptive campaign.

use super::*;

const ISLANDS: usize = 4;
const ISLAND_POPULATION: usize = 30;
const TOTAL_POPULATION: usize = ISLANDS * ISLAND_POPULATION;
const ISLAND_ELITES: usize = 4;
const GENERATION_20_PARENT_POOL: usize = 30;
const ISLAND_PARENT_POOL: usize = 15;
const ISLAND_MIN_VALID: usize = 15;
const GLOBAL_MIN_VALID: usize = 60;
const START_GENERATION: usize = 20;
const GENERATIONS: usize = 10;
const END_GENERATION: usize = START_GENERATION + GENERATIONS;
const EXISTING_SURPASSED: usize = 35;
const MIGRATION_GENERATIONS: &[usize] = &[21, 23, 25, 27];
const MANIFEST_SCHEMA: &str = "JopeTestAdaptiveIslandExtensionManifest.v1";
const CAMPAIGN_SCHEMA: &str = "JopeTestAdaptiveIslandDatasetCampaign.v1";
const CHECKPOINT_SCHEMA: &str = "JopeTestAdaptiveIslandGenerationCheckpoint.v1";
const MIGRATION_SCHEMA: &str = "JopeTestAdaptiveIslandMigration.v1";
const RESULT_SCHEMA: &str = "JopeTestAdaptiveIslandDatasetResult.v1";
const SUMMARY_SCHEMA: &str = "JopeTestAdaptiveIslandExtensionSummary.v1";
const MANIFEST_FILE: &str = "island-extension-manifest.v1.json";
const SUMMARY_FILE: &str = "island-extension-summary.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const BEST_SOLUTIONS_FILE: &str = "golden42-best-solutions.v1.json";
const BEST_SOLUTIONS_SCHEMA: &str = "JopeGolden42BestSolutions.v1";
const CHAMPION_START_GENERATION: usize = 30;
const CHAMPION_MANIFEST_SCHEMA: &str = "JopeGolden42ChampionContinuationManifest.v1";
const CHAMPION_CAMPAIGN_SCHEMA: &str = "JopeGolden42ChampionDatasetCampaign.v1";
const CHAMPION_CHECKPOINT_SCHEMA: &str = "JopeGolden42ChampionGenerationCheckpoint.v1";
const CHAMPION_RESULT_SCHEMA: &str = "JopeGolden42ChampionDatasetResult.v1";
const CHAMPION_SUMMARY_SCHEMA: &str = "JopeGolden42ChampionContinuationSummary.v1";
const CHAMPION_MANIFEST_FILE: &str = "champion-continuation-manifest.v1.json";
const CHAMPION_SUMMARY_FILE: &str = "champion-continuation-summary.v1.json";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandConfig {
    islands: usize,
    population_per_island: usize,
    total_population: usize,
    elites_per_island: usize,
    generation_20_parent_pool: usize,
    later_parent_pool_per_island: usize,
    tournament: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    start_generation: usize,
    generations: usize,
    end_generation_exclusive: usize,
    minimum_valid_per_island: usize,
    minimum_valid_global: usize,
    maximum_concurrent_candidates: usize,
    candidate_threads: usize,
    replay_count: usize,
    global_slot_mapping: String,
}

impl IslandConfig {
    fn fixed() -> Self {
        Self {
            islands: ISLANDS,
            population_per_island: ISLAND_POPULATION,
            total_population: TOTAL_POPULATION,
            elites_per_island: ISLAND_ELITES,
            generation_20_parent_pool: GENERATION_20_PARENT_POOL,
            later_parent_pool_per_island: ISLAND_PARENT_POOL,
            tournament: TOURNAMENT,
            crossover_probability: 0.25,
            mutation_probability: 0.75,
            start_generation: START_GENERATION,
            generations: GENERATIONS,
            end_generation_exclusive: END_GENERATION,
            minimum_valid_per_island: ISLAND_MIN_VALID,
            minimum_valid_global: GLOBAL_MIN_VALID,
            maximum_concurrent_candidates: scq_batch::SCQ_CPUS,
            candidate_threads: 1,
            replay_count: 3,
            global_slot_mapping: "island * 30 + local_slot".to_string(),
        }
    }

    fn champion() -> Self {
        Self {
            start_generation: CHAMPION_START_GENERATION,
            generations: 0,
            end_generation_exclusive: 0,
            global_slot_mapping: "island * 30 + local_slot; continue until strict leader crossing"
                .to_string(),
            ..Self::fixed()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MigrationPolicy {
    after_generations: Vec<usize>,
    direction: String,
    migrants_per_island: usize,
    source_ranks: Vec<usize>,
    destination_replaced_ranks: Vec<usize>,
    semantic_duplicates_allowed: bool,
}

impl MigrationPolicy {
    fn fixed() -> Self {
        Self {
            after_generations: MIGRATION_GENERATIONS.to_vec(),
            direction: "island i sends to (i + 1) mod 4".to_string(),
            migrants_per_island: 2,
            source_ranks: vec![0, 1],
            destination_replaced_ranks: vec![28, 29],
            semantic_duplicates_allowed: true,
        }
    }

    fn champion() -> Self {
        Self {
            after_generations: Vec::new(),
            direction: "after every odd generation >=31, island i sends to (i + 1) mod 4"
                .to_string(),
            migrants_per_island: 2,
            source_ranks: vec![0, 1],
            destination_replaced_ranks: vec![28, 29],
            semantic_duplicates_allowed: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RngPolicy {
    seed: u64,
    namespace: String,
    generator: String,
    retry_limit: usize,
}

impl RngPolicy {
    fn fixed() -> Self {
        Self {
            seed: RNG_SEED,
            namespace: "sha256(seed:generation:island:local_slot:retry)".to_string(),
            generator: "Pcg32 with island-specific sequence".to_string(),
            retry_limit: 100,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct IslandPopulationHash {
    island: usize,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandDatasetBinding {
    dataset_hash: String,
    prior_campaign_file: String,
    prior_campaign_sha256: String,
    prior_final_result_file: String,
    prior_final_result_sha256: String,
    prior_terminal_checkpoint_file: String,
    prior_terminal_checkpoint_sha256: String,
    generation_20_island_population_hashes: Vec<IslandPopulationHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandManifest {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    prior_extension_run_id: String,
    island_extension_run_id: String,
    base_panel_manifest_sha256: String,
    base_panel_summary_sha256: String,
    base_winner_map_sha256: String,
    prior_extension_manifest_sha256: String,
    prior_extension_summary_sha256: String,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    scq_staging_receipt_sha256: String,
    scq_policy: scq_batch::ScqPolicy,
    config: IslandConfig,
    migration_policy: MigrationPolicy,
    rng_policy: RngPolicy,
    datasets: Vec<IslandDatasetBinding>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandCampaignIdentity {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset: DatasetSpec,
    release_binary_sha256: String,
    config: IslandConfig,
    migration_policy: MigrationPolicy,
    rng_policy: RngPolicy,
    selection_objective: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct IslandParentRef {
    generation: usize,
    island: Option<usize>,
    local_slot: usize,
    global_slot: usize,
    semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandPopulationMember {
    member: Member,
    parents: Vec<IslandParentRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandSlotOutcome {
    generation: usize,
    island: usize,
    local_slot: usize,
    global_slot: usize,
    member: Member,
    parents: Vec<IslandParentRef>,
    valid: bool,
    metrics: Option<CandidateMetrics>,
    failure: Option<FailureState>,
    cache_source: Option<CacheSource>,
    receipt_file: String,
    receipt_sha256: String,
    node: String,
    scq_job_id: String,
    attempt: usize,
}

impl IslandSlotOutcome {
    fn rmse(&self) -> f64 {
        self.metrics
            .as_ref()
            .map_or(f64::INFINITY, |metrics| metrics.rmse)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandGenerationState {
    island: usize,
    logical_slots: usize,
    fresh_evaluations: usize,
    valid_candidates: usize,
    best_local_slot: usize,
    best_global_slot: usize,
    best_semantic_sha256: String,
    best_rmse: f64,
    ranked_local_slots: Vec<usize>,
    slots: Vec<IslandSlotOutcome>,
    next_population: Vec<IslandPopulationMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct IslandGenomeLocation {
    generation: usize,
    island: usize,
    local_slot: usize,
    global_slot: usize,
    semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandGenerationCheckpoint {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    previous_checkpoint_sha256: String,
    logical_slots: usize,
    fresh_evaluations: usize,
    cumulative_fresh_evaluations: usize,
    valid_candidates: usize,
    fresh_g0_rmse: f64,
    global_best: IslandGenomeLocation,
    global_best_rmse: f64,
    qualifying_descendant_win: bool,
    qualifying_genomes: Vec<IslandGenomeLocation>,
    result_label: Option<String>,
    migration_file: Option<String>,
    migration_sha256: Option<String>,
    islands: Vec<IslandGenerationState>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MigrationGenome {
    generation: usize,
    island: usize,
    local_slot: usize,
    global_slot: usize,
    rank: usize,
    member: Member,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MigrationTransfer {
    source_island: usize,
    destination_island: usize,
    source_rank: usize,
    destination_replaced_rank: usize,
    migrant: MigrationGenome,
    replaced: MigrationGenome,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MigrationRecord {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    policy: MigrationPolicy,
    transfers: Vec<MigrationTransfer>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct IslandLineageRecord {
    generation: usize,
    island: Option<usize>,
    local_slot: usize,
    global_slot: usize,
    semantic_sha256: String,
    operation: String,
    parents: Vec<IslandParentRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandGenomeRecord {
    canonical_genome: String,
    semantic_sha256: String,
    generation: usize,
    island: usize,
    local_slot: usize,
    global_slot: usize,
    rmse: f64,
    mae: f64,
    r2: f64,
    margin_vs_g0: f64,
    margin_vs_leader: f64,
    complexity: usize,
    rust_source_len: usize,
    runtime_seconds: f64,
    lineage: Vec<IslandLineageRecord>,
    prediction_sha256: String,
    receipt_file: String,
    receipt_sha256: String,
    binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandDatasetResult {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    island_extension_run_id: String,
    dataset_hash: String,
    outcome: String,
    start_generation: usize,
    stopping_generation: usize,
    generations_evaluated: usize,
    total_generations_evaluated: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    fresh_g0_rmse: f64,
    first_winning_generation: Option<usize>,
    first_winning_island: Option<usize>,
    generation_29_selected_genome: IslandGenomeRecord,
    qualifying_descendants: Vec<IslandGenomeRecord>,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
    replay: Vec<ReplayRecord>,
    release_binary_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct IslandSummary {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    prior_extension_run_id: String,
    island_extension_run_id: String,
    base_panel_manifest_sha256: String,
    base_panel_summary_sha256: String,
    base_winner_map_sha256: String,
    prior_extension_manifest_sha256: String,
    prior_extension_summary_sha256: String,
    release_binary_sha256: String,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    dataset_count: usize,
    existing_surpassed_count: usize,
    new_leaders: usize,
    still_no_leader: usize,
    cumulative_surpassed_count: usize,
    total_qualifying_genomes: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    best_solutions_file: String,
    best_solutions_sha256: String,
    results: Vec<IslandDatasetResult>,
    artifacts: Vec<ArtifactHash>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Golden42BestSolution {
    dataset_hash: String,
    source_campaign: String,
    source_artifact: String,
    source_artifact_sha256: String,
    benchmark_status: String,
    leader_rmse: f64,
    selected_rmse: f64,
    margin_vs_leader: f64,
    canonical_genome: String,
    semantic_sha256: String,
    generation: usize,
    island: Option<usize>,
    slot: usize,
    binary_sha256: String,
    prediction_sha256: String,
    receipt_sha256: String,
    replay_receipt_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Golden42BestSolutions {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    island_extension_run_id: String,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    dataset_count: usize,
    prior_winner_count: usize,
    island_selected_count: usize,
    solutions: Vec<Golden42BestSolution>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ChampionDatasetBinding {
    dataset_hash: String,
    initial_result_file: String,
    initial_result_sha256: String,
    generation_29_checkpoint_file: String,
    generation_29_checkpoint_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChampionManifest {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    initial_island_run_id: String,
    champion_run_id: String,
    initial_manifest_sha256: String,
    initial_summary_sha256: String,
    initial_best_solutions_sha256: String,
    ordered_dataset_ids: Vec<String>,
    ordered_cohort_sha256: String,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    scq_staging_receipt_sha256: String,
    scq_policy: scq_batch::ScqPolicy,
    config: IslandConfig,
    migration_policy: MigrationPolicy,
    rng_policy: RngPolicy,
    complexity_schedule: String,
    stop_policy: String,
    datasets: Vec<ChampionDatasetBinding>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChampionDatasetResult {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    champion_run_id: String,
    dataset_hash: String,
    outcome: String,
    start_generation: usize,
    stopping_generation: usize,
    generations_evaluated: usize,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    champion: IslandGenomeRecord,
    replay: Vec<ReplayRecord>,
    release_binary_sha256: String,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChampionSummary {
    schema: String,
    benchmark_valid: bool,
    base_run_id: String,
    initial_island_run_id: String,
    champion_run_id: String,
    release_binary_sha256: String,
    ordered_dataset_ids: Vec<String>,
    dataset_count: usize,
    inherited_champion_count: usize,
    continuation_champion_count: usize,
    cumulative_surpassed_count: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    best_solutions_file: String,
    best_solutions_sha256: String,
    results: Vec<ChampionDatasetResult>,
    artifacts: Vec<ArtifactHash>,
}

fn island_run_id(base_run_id: &str) -> String {
    format!("{base_run_id}-extension-g20-g29-islands4x30")
}

fn island_root(base_run_id: &str) -> PathBuf {
    run_root(&island_run_id(base_run_id))
}

fn champion_run_id(base_run_id: &str) -> String {
    format!("{base_run_id}-champions-g30-plus-islands4x30")
}

fn champion_root(base_run_id: &str) -> PathBuf {
    run_root(&champion_run_id(base_run_id))
}

fn global_slot(island: usize, local_slot: usize) -> usize {
    island * ISLAND_POPULATION + local_slot
}

#[cfg(test)]
fn island_slot(global: usize) -> Result<(usize, usize)> {
    if global >= TOTAL_POPULATION {
        bail!("global island slot {global} is out of range");
    }
    Ok((global / ISLAND_POPULATION, global % ISLAND_POPULATION))
}

fn rank_island_outcomes(left: &IslandSlotOutcome, right: &IslandSlotOutcome) -> Ordering {
    left.rmse()
        .total_cmp(&right.rmse())
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .rust_source_len
                .cmp(&right.member.rust_source_len)
        })
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
        .then_with(|| left.global_slot.cmp(&right.global_slot))
}

fn island_offspring_seed(generation: usize, island: usize, local_slot: usize, retry: usize) -> u64 {
    let text = format!("{RNG_SEED}:{generation}:{island}:{local_slot}:{retry}");
    let digest = Sha256::digest(text.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn island_sequence(island: usize) -> u64 {
    let digest = Sha256::digest(format!("{RNG_SEED}:island-stream:{island}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn old_parent_ref(generation: usize, outcome: &SlotOutcome) -> IslandParentRef {
    IslandParentRef {
        generation,
        island: None,
        local_slot: outcome.slot,
        global_slot: outcome.slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

fn island_parent_ref(outcome: &IslandSlotOutcome) -> IslandParentRef {
    IslandParentRef {
        generation: outcome.generation,
        island: Some(outcome.island),
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

fn evolved_member(
    parents: &[Member],
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
    features: usize,
) -> Result<(Member, Vec<usize>)> {
    let seed = island_offspring_seed(generation, island, local_slot, retry);
    let mut rng = Pcg32::new(seed, island_sequence(island));
    let first = tournament_winner(&mut rng, parents.len());
    let first_parent = &parents[first];
    if rng.next_f64() < 0.25 {
        let second = tournament_winner(&mut rng, parents.len());
        let second_parent = &parents[second];
        crossover_pipelines(
            &first_parent.canonical_genome,
            &second_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
        )
        .map(|genome| {
            (
                Member::evolved(
                    genome,
                    Ancestry {
                        operation: "crossover".to_string(),
                        parents: vec![
                            first_parent.semantic_sha256.clone(),
                            second_parent.semantic_sha256.clone(),
                        ],
                    },
                ),
                vec![first, second],
            )
        })
        .map_err(anyhow::Error::msg)
    } else {
        mutate_pipeline(
            &first_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map(|genome| {
            (
                Member::evolved(
                    genome,
                    Ancestry {
                        operation: "mutation".to_string(),
                        parents: vec![first_parent.semantic_sha256.clone()],
                    },
                ),
                vec![first],
            )
        })
        .map_err(anyhow::Error::msg)
    }
}

fn champion_complexity_level(generation: usize) -> usize {
    1 + generation
        .saturating_sub(CHAMPION_START_GENERATION)
        .div_euclid(10)
        .min(3)
}

fn champion_evolved_member(
    parents: &[Member],
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
    features: usize,
) -> Result<(Member, Vec<usize>)> {
    let level = champion_complexity_level(generation);
    let seed = island_offspring_seed(generation, island, local_slot, retry);
    let mut rng = Pcg32::new(seed, island_sequence(island));
    let first = tournament_winner(&mut rng, parents.len());
    let first_parent = &parents[first];
    let crossover_probability = (0.25 + 0.10 * (level.saturating_sub(1) as f64)).min(0.55);
    let (mut genome, parent_indices, operation) = if rng.next_f64() < crossover_probability {
        let second = tournament_winner(&mut rng, parents.len());
        let genome = crossover_pipelines(
            &first_parent.canonical_genome,
            &parents[second].canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
        )
        .map_err(anyhow::Error::msg)?;
        (genome, vec![first, second], "crossover")
    } else {
        let genome = mutate_pipeline(
            &first_parent.canonical_genome,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map_err(anyhow::Error::msg)?;
        (genome, vec![first], "mutation")
    };
    for _ in 1..level {
        genome = mutate_pipeline(
            &genome.canonical,
            rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            features,
        )
        .map_err(anyhow::Error::msg)?;
    }
    let parent_hashes = parent_indices
        .iter()
        .map(|index| parents[*index].semantic_sha256.clone())
        .collect();
    Ok((
        Member::evolved(
            genome,
            Ancestry {
                operation: format!("champion_{operation}_depth_{level}"),
                parents: parent_hashes,
            },
        ),
        parent_indices,
    ))
}

fn generation_20_populations(
    combined: &[(GenerationCheckpoint, String)],
    features: usize,
) -> Result<Vec<Vec<IslandPopulationMember>>> {
    if combined.len() != START_GENERATION {
        bail!("prior history does not contain generations 0-19");
    }
    let terminal = &combined.last().unwrap().0;
    if terminal.generation != START_GENERATION - 1 {
        bail!("prior terminal checkpoint is not generation 19");
    }
    let mut ranked = terminal.slots.clone();
    ranked.sort_by(rank_outcomes);
    let mut seen = HashSet::new();
    let distinct_elites = ranked
        .iter()
        .filter(|outcome| seen.insert(outcome.member.semantic_sha256.clone()))
        .take(ISLANDS * ISLAND_ELITES)
        .cloned()
        .collect::<Vec<_>>();
    if distinct_elites.len() != ISLANDS * ISLAND_ELITES {
        bail!("generation 19 has fewer than 16 distinct valid elites");
    }
    if ranked.len() < GENERATION_20_PARENT_POOL
        || ranked[..GENERATION_20_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("generation 20 parent pool is incomplete or invalid");
    }
    let parents = ranked[..GENERATION_20_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    let mut populations = Vec::with_capacity(ISLANDS);
    for island in 0..ISLANDS {
        let mut population = Vec::with_capacity(ISLAND_POPULATION);
        for rank in (island..ISLANDS * ISLAND_ELITES).step_by(ISLANDS) {
            let source = &distinct_elites[rank];
            let mut member = source.member.clone();
            member.ancestry = Ancestry {
                operation: "island_elite".to_string(),
                parents: vec![source.member.semantic_sha256.clone()],
            };
            population.push(IslandPopulationMember {
                member,
                parents: vec![old_parent_ref(START_GENERATION - 1, source)],
            });
        }
        for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
            let mut produced = None;
            for retry in 0..100 {
                let candidate = evolved_member(
                    &parents,
                    START_GENERATION,
                    island,
                    local_slot,
                    retry,
                    features,
                );
                if let Ok((member, parent_indices)) = candidate {
                    let refs = parent_indices
                        .into_iter()
                        .map(|index| old_parent_ref(START_GENERATION - 1, &ranked[index]))
                        .collect();
                    produced = Some(IslandPopulationMember {
                        member,
                        parents: refs,
                    });
                    break;
                }
            }
            population.push(produced.context("could not create generation 20 island descendant")?);
        }
        populations.push(population);
    }
    Ok(populations)
}

fn next_island_population(
    ranked: &[IslandSlotOutcome],
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<IslandPopulationMember>> {
    if ranked.len() != ISLAND_POPULATION
        || ranked[..ISLAND_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("island {island} parent pool is incomplete or invalid");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for source in ranked.iter().take(ISLAND_ELITES) {
        let mut member = source.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![source.member.semantic_sha256.clone()],
        };
        next.push(IslandPopulationMember {
            member,
            parents: vec![island_parent_ref(source)],
        });
    }
    let parents = ranked[..ISLAND_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100 {
            if let Ok((member, parent_indices)) =
                evolved_member(&parents, generation, island, local_slot, retry, features)
            {
                let refs = parent_indices
                    .into_iter()
                    .map(|index| island_parent_ref(&ranked[index]))
                    .collect();
                produced = Some(IslandPopulationMember {
                    member,
                    parents: refs,
                });
                break;
            }
        }
        next.push(produced.context("could not create valid island descendant")?);
    }
    Ok(next)
}

fn next_champion_population(
    ranked: &[IslandSlotOutcome],
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<IslandPopulationMember>> {
    if ranked.len() != ISLAND_POPULATION
        || ranked[..ISLAND_PARENT_POOL]
            .iter()
            .any(|outcome| !outcome.valid)
    {
        bail!("champion island {island} parent pool is incomplete or invalid");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for source in ranked.iter().take(ISLAND_ELITES) {
        let mut member = source.member.clone();
        member.ancestry = Ancestry {
            operation: "champion_elite".to_string(),
            parents: vec![source.member.semantic_sha256.clone()],
        };
        next.push(IslandPopulationMember {
            member,
            parents: vec![island_parent_ref(source)],
        });
    }
    let parents = ranked[..ISLAND_PARENT_POOL]
        .iter()
        .map(|outcome| outcome.member.clone())
        .collect::<Vec<_>>();
    for local_slot in ISLAND_ELITES..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100 {
            if let Ok((member, parent_indices)) =
                champion_evolved_member(&parents, generation, island, local_slot, retry, features)
            {
                produced = Some(IslandPopulationMember {
                    member,
                    parents: parent_indices
                        .into_iter()
                        .map(|index| island_parent_ref(&ranked[index]))
                        .collect(),
                });
                break;
            }
        }
        next.push(produced.context("could not create valid champion descendant")?);
    }
    Ok(next)
}

fn migration_genome(outcome: &IslandSlotOutcome, rank: usize) -> MigrationGenome {
    MigrationGenome {
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        rank,
        member: outcome.member.clone(),
    }
}

fn migrate_rankings(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if !MIGRATION_GENERATIONS.contains(&generation) {
        bail!("invalid ring migration generation {generation}");
    }
    migrate_rankings_with_policy(
        run_id,
        dataset_hash,
        generation,
        ranked,
        MigrationPolicy::fixed(),
    )
}

fn migrate_rankings_champion(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if generation < CHAMPION_START_GENERATION + 1 || generation % 2 == 0 {
        bail!("invalid champion ring migration generation {generation}");
    }
    migrate_rankings_with_policy(
        run_id,
        dataset_hash,
        generation,
        ranked,
        MigrationPolicy::champion(),
    )
}

fn migrate_rankings_with_policy(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
    policy: MigrationPolicy,
) -> Result<(Vec<Vec<IslandSlotOutcome>>, MigrationRecord)> {
    if ranked.len() != ISLANDS
        || ranked
            .iter()
            .any(|island| island.len() != ISLAND_POPULATION)
    {
        bail!("invalid ring migration input for generation {generation}");
    }
    let original = ranked.to_vec();
    let mut migrated = ranked.to_vec();
    let mut transfers = Vec::with_capacity(ISLANDS * 2);
    for source_island in 0..ISLANDS {
        let destination_island = (source_island + 1) % ISLANDS;
        for source_rank in 0..2 {
            let destination_rank = ISLAND_POPULATION - 2 + source_rank;
            let migrant = &original[source_island][source_rank];
            let replaced = &original[destination_island][destination_rank];
            transfers.push(MigrationTransfer {
                source_island,
                destination_island,
                source_rank,
                destination_replaced_rank: destination_rank,
                migrant: migration_genome(migrant, source_rank),
                replaced: migration_genome(replaced, destination_rank),
            });
            migrated[destination_island][destination_rank] = migrant.clone();
        }
    }
    for island in &mut migrated {
        island.sort_by(rank_island_outcomes);
    }
    Ok((
        migrated,
        MigrationRecord {
            schema: MIGRATION_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: run_id.to_string(),
            dataset_hash: dataset_hash.to_string(),
            generation,
            policy,
            transfers,
        },
    ))
}

fn prior_history(
    base_run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(GenerationCheckpoint, String)>> {
    let base_dataset_root = run_root(base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let base_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(base_dataset_root.join("campaign.v1.json"))?)?;
    let base = load_checkpoints(&base_dataset_root, &base_identity)?;
    let prior_root = extension_root(base_run_id);
    let prior_manifest: ExtensionManifest =
        serde_json::from_slice(&fs::read(prior_root.join(EXTENSION_MANIFEST_FILE))?)?;
    let binding = prior_manifest
        .datasets
        .iter()
        .find(|binding| binding.dataset_hash == spec.dataset_hash)
        .context("dataset is absent from the generations 10-19 manifest")?;
    let dataset_root = prior_root.join("datasets").join(&spec.dataset_hash);
    let prior_identity: CampaignIdentity =
        serde_json::from_slice(&fs::read(dataset_root.join("campaign.v1.json"))?)?;
    let extension = load_extension_checkpoints(
        &dataset_root,
        &prior_identity,
        &binding.base_terminal_checkpoint_sha256,
    )?;
    if extension.len() != EXTENSION_GENERATIONS {
        bail!(
            "dataset {} has incomplete generations 10-19",
            spec.dataset_hash
        );
    }
    combined_checkpoints(&base, &extension)
}

fn prior_no_win_ids(base_run_id: &str) -> Result<Vec<String>> {
    let root = extension_root(base_run_id);
    let manifest: ExtensionManifest =
        serde_json::from_slice(&fs::read(root.join(EXTENSION_MANIFEST_FILE))?)?;
    let summary: ExtensionSummary =
        serde_json::from_slice(&fs::read(root.join(EXTENSION_SUMMARY_FILE))?)?;
    let outcomes = summary
        .results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result.outcome.as_str()))
        .collect::<HashMap<_, _>>();
    let selected = manifest
        .ordered_dataset_ids
        .iter()
        .filter(|dataset| {
            outcomes.get(dataset.as_str()).copied() == Some("no_win_after_20_total_generations")
        })
        .cloned()
        .collect::<Vec<_>>();
    if selected.len() != 7 || summary.new_leaders + summary.still_no_leader != summary.dataset_count
    {
        bail!("generations 10-19 do not reconstruct the ordered seven-dataset no-win cohort");
    }
    let base_summary: PanelSummary =
        serde_json::from_slice(&fs::read(run_root(base_run_id).join(super::SUMMARY_FILE))?)?;
    if base_summary.early_wins + summary.new_leaders != EXISTING_SURPASSED {
        bail!("prior campaigns do not reconstruct 35 surpassed datasets");
    }
    Ok(selected)
}

fn dataset_binding(base_run_id: &str, spec: &DatasetSpec) -> Result<IslandDatasetBinding> {
    let prior_root = extension_root(base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let campaign_file = prior_root.join("campaign.v1.json");
    let result_file = prior_root.join("final-result.v1.json");
    let terminal_file = prior_root
        .join("checkpoints")
        .join("generation-019.v1.json");
    let combined = prior_history(base_run_id, spec)?;
    let populations = generation_20_populations(&combined, spec.features)?;
    Ok(IslandDatasetBinding {
        dataset_hash: spec.dataset_hash.clone(),
        prior_campaign_file: campaign_file.to_string_lossy().into_owned(),
        prior_campaign_sha256: sha256_file(&campaign_file)?,
        prior_final_result_file: result_file.to_string_lossy().into_owned(),
        prior_final_result_sha256: sha256_file(&result_file)?,
        prior_terminal_checkpoint_file: terminal_file.to_string_lossy().into_owned(),
        prior_terminal_checkpoint_sha256: sha256_file(&terminal_file)?,
        generation_20_island_population_hashes: populations
            .iter()
            .enumerate()
            .map(|(island, population)| {
                Ok(IslandPopulationHash {
                    island,
                    sha256: sha256_bytes(&canonical_json(population)?),
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

fn expected_manifest(
    base_run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    scq_staging_receipt_sha256: &str,
    specs: &[DatasetSpec],
    started_at_unix_ms: u128,
) -> Result<IslandManifest> {
    let ordered_dataset_ids = prior_no_win_ids(base_run_id)?;
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let datasets = ordered_dataset_ids
        .iter()
        .map(|dataset| {
            dataset_binding(
                base_run_id,
                by_hash
                    .get(dataset.as_str())
                    .context("island cohort dataset is absent from the Golden42 panel")?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let base_root = run_root(base_run_id);
    let prior_root = extension_root(base_run_id);
    Ok(IslandManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        prior_extension_run_id: extension_run_id(base_run_id),
        island_extension_run_id: island_run_id(base_run_id),
        base_panel_manifest_sha256: sha256_file(&base_root.join("panel-manifest.v1.json"))?,
        base_panel_summary_sha256: sha256_file(&base_root.join(super::SUMMARY_FILE))?,
        base_winner_map_sha256: sha256_file(&base_root.join(WINNER_MAP_FILE))?,
        prior_extension_manifest_sha256: sha256_file(&prior_root.join(EXTENSION_MANIFEST_FILE))?,
        prior_extension_summary_sha256: sha256_file(&prior_root.join(EXTENSION_SUMMARY_FILE))?,
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
        ordered_dataset_ids,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: scq_staging_receipt_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::fixed(),
        config: IslandConfig::fixed(),
        migration_policy: MigrationPolicy::fixed(),
        rng_policy: RngPolicy::fixed(),
        datasets,
        started_at_unix_ms,
    })
}

fn island_identity(manifest: &IslandManifest, spec: &DatasetSpec) -> IslandCampaignIdentity {
    IslandCampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.island_extension_run_id.clone(),
        dataset: spec.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        config: manifest.config.clone(),
        migration_policy: manifest.migration_policy.clone(),
        rng_policy: manifest.rng_policy.clone(),
        selection_objective: "continue immutable generations 0-19 on four isolated islands; rank directly by test RMSE (intentionally test-adaptive)".to_string(),
    }
}

fn worker_identity(identity: &IslandCampaignIdentity) -> CampaignIdentity {
    let mut value = super::identity(
        &identity.run_id,
        identity.release_binary_sha256.clone(),
        &identity.dataset,
    );
    value.population = TOTAL_POPULATION;
    value.elites = ISLAND_ELITES;
    value.parent_pool = ISLAND_PARENT_POOL;
    value.generations = GENERATIONS;
    value.selection_objective = identity.selection_objective.clone();
    value
}

fn island_outcome_from_evaluation(
    generation: usize,
    island: usize,
    local_slot: usize,
    population: IslandPopulationMember,
    evaluation: &CachedEvaluation,
    cached: bool,
) -> IslandSlotOutcome {
    IslandSlotOutcome {
        generation,
        island,
        local_slot,
        global_slot: global_slot(island, local_slot),
        member: population.member,
        parents: population.parents,
        valid: evaluation.receipt.valid(evaluation.receipt.test_rows),
        metrics: evaluation.receipt.metrics.clone(),
        failure: evaluation.receipt.failure.clone(),
        cache_source: cached.then(|| evaluation.source.clone()),
        receipt_file: evaluation.receipt_file.clone(),
        receipt_sha256: evaluation.receipt_sha256.clone(),
        node: evaluation.receipt.node.clone(),
        scq_job_id: evaluation.receipt.scq_job_id.clone(),
        attempt: evaluation.receipt.attempt,
    }
}

fn evaluate_island_generation(
    identity: &CampaignIdentity,
    registry: &Path,
    campaign_root: &Path,
    scq_binary_sha256: &str,
    dataset_index: usize,
    generation: usize,
    populations: Vec<Vec<IslandPopulationMember>>,
    cache: &mut HashMap<String, CachedEvaluation>,
    test: &RegressionData,
) -> Result<(Vec<Vec<IslandSlotOutcome>>, usize)> {
    if populations.len() != ISLANDS
        || populations
            .iter()
            .any(|population| population.len() != ISLAND_POPULATION)
    {
        bail!("island generation does not contain four complete populations");
    }
    let mut outcomes = vec![None; TOTAL_POPULATION];
    let mut pending = BTreeMap::<String, Vec<(usize, usize, IslandPopulationMember)>>::new();
    for (island, population) in populations.into_iter().enumerate() {
        for (local_slot, member) in population.into_iter().enumerate() {
            let slot = global_slot(island, local_slot);
            if let Some(evaluation) = cache.get(&member.member.semantic_sha256) {
                outcomes[slot] = Some(island_outcome_from_evaluation(
                    generation, island, local_slot, member, evaluation, true,
                ));
            } else {
                pending
                    .entry(member.member.semantic_sha256.clone())
                    .or_default()
                    .push((island, local_slot, member));
            }
        }
    }
    let fresh = pending.len();
    let pending = pending.into_values().collect::<Vec<_>>();
    let representatives = pending
        .iter()
        .map(|entries| {
            let (island, local_slot, population) = &entries[0];
            (global_slot(*island, *local_slot), population.member.clone())
        })
        .collect::<Vec<_>>();
    let evaluations = scq_batch::evaluate_batches(
        campaign_root,
        identity,
        registry,
        &identity.binary_sha256,
        scq_binary_sha256,
        dataset_index,
        "candidate",
        generation,
        &representatives,
        test,
    )?;
    if evaluations.len() != pending.len() {
        bail!("SCQ batch result count does not match fresh candidate count");
    }
    for (entries, evaluation) in pending.into_iter().zip(evaluations) {
        cache.insert(
            evaluation.receipt.semantic_sha256.clone(),
            evaluation.clone(),
        );
        for (index, (island, local_slot, population)) in entries.into_iter().enumerate() {
            let slot = global_slot(island, local_slot);
            outcomes[slot] = Some(island_outcome_from_evaluation(
                generation,
                island,
                local_slot,
                population,
                &evaluation,
                index > 0,
            ));
        }
    }
    let flat = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("all island slots evaluated"))
        .collect::<Vec<_>>();
    let islands = (0..ISLANDS)
        .map(|island| flat[global_slot(island, 0)..global_slot(island + 1, 0)].to_vec())
        .collect();
    Ok((islands, fresh))
}

fn checkpoint_path(root: &Path, generation: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

fn migration_relative(generation: usize) -> String {
    format!("migrations/generation-{generation:03}.v1.json")
}

fn load_island_checkpoints(
    root: &Path,
    identity: &IslandCampaignIdentity,
    prior_terminal_sha256: &str,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = prior_terminal_sha256.to_string();
    for generation in START_GENERATION..END_GENERATION {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: IslandGenerationCheckpoint = serde_json::from_slice(&bytes)?;
        validate_checkpoint_envelope(&checkpoint, identity, generation, &previous)
            .with_context(|| format!("invalid island checkpoint {}", path.display()))?;
        if bytes != canonical_json(&checkpoint)? {
            bail!("noncanonical island checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = sha256.clone();
        checkpoints.push((checkpoint, sha256));
    }
    for generation in (START_GENERATION + checkpoints.len())..END_GENERATION {
        if checkpoint_path(root, generation).is_file() {
            bail!("island checkpoint history contains a gap before generation {generation}");
        }
    }
    Ok(checkpoints)
}

fn validate_checkpoint_envelope(
    checkpoint: &IslandGenerationCheckpoint,
    identity: &IslandCampaignIdentity,
    generation: usize,
    previous: &str,
) -> Result<()> {
    if checkpoint.schema != CHECKPOINT_SCHEMA
        || checkpoint.benchmark_valid
        || checkpoint.run_id != identity.run_id
        || checkpoint.dataset_hash != identity.dataset.dataset_hash
        || checkpoint.generation != generation
        || checkpoint.previous_checkpoint_sha256 != previous
        || checkpoint.logical_slots != TOTAL_POPULATION
        || checkpoint.islands.len() != ISLANDS
        || checkpoint.islands.iter().any(|island| {
            island.logical_slots != ISLAND_POPULATION
                || island.slots.len() != ISLAND_POPULATION
                || (generation + 1 < END_GENERATION
                    && island.next_population.len() != ISLAND_POPULATION)
                || (generation + 1 == END_GENERATION && !island.next_population.is_empty())
        })
    {
        bail!("island checkpoint envelope mismatch");
    }
    Ok(())
}

fn validate_island_receipt_binding(
    identity: &IslandCampaignIdentity,
    outcome: &IslandSlotOutcome,
    receipt: &CandidateReceipt,
    request: &CandidateRequest,
) -> Result<()> {
    if receipt.run_id != identity.run_id
        || receipt.dataset_hash != identity.dataset.dataset_hash
        || receipt.binary_sha256 != identity.release_binary_sha256
        || request.binary_sha256 != identity.release_binary_sha256
        || receipt.semantic_sha256 != outcome.member.semantic_sha256
        || outcome.valid != receipt.valid(identity.dataset.test_rows)
        || outcome.metrics != receipt.metrics
        || outcome.failure != receipt.failure
    {
        bail!("island receipt provenance or binary mismatch");
    }
    Ok(())
}

fn write_or_verify(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        Ok(())
    } else {
        immutable_write(path, bytes)
    }
}

fn qualifies(outcome: &IslandSlotOutcome, g0_rmse: f64, leader_rmse: f64) -> bool {
    outcome.member.evolved_descendant
        && outcome.valid
        && outcome.rmse() < g0_rmse
        && outcome.rmse() < leader_rmse
}

fn genome_location(outcome: &IslandSlotOutcome) -> IslandGenomeLocation {
    IslandGenomeLocation {
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        semantic_sha256: outcome.member.semantic_sha256.clone(),
    }
}

fn evolve_islands(
    identity: &IslandCampaignIdentity,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
) -> Result<(
    Vec<Vec<IslandPopulationMember>>,
    Option<(String, String, MigrationRecord)>,
)> {
    if generation + 1 == END_GENERATION {
        return Ok((vec![Vec::new(); ISLANDS], None));
    }
    let (parents, migration) = if MIGRATION_GENERATIONS.contains(&generation) {
        let (parents, record) = migrate_rankings(
            &identity.run_id,
            &identity.dataset.dataset_hash,
            generation,
            ranked,
        )?;
        let relative = migration_relative(generation);
        let bytes = canonical_json(&record)?;
        let sha256 = sha256_bytes(&bytes);
        (parents, Some((relative, sha256, record)))
    } else {
        (ranked.to_vec(), None)
    };
    let next = parents
        .iter()
        .enumerate()
        .map(|(island, ranked)| {
            next_island_population(ranked, generation + 1, island, identity.dataset.features)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((next, migration))
}

fn make_checkpoint(
    identity: &IslandCampaignIdentity,
    root: &Path,
    generation: usize,
    previous_checkpoint_sha256: String,
    outcomes: Vec<Vec<IslandSlotOutcome>>,
    cumulative_fresh_evaluations: usize,
    g0_rmse: f64,
) -> Result<(IslandGenerationCheckpoint, Vec<Vec<IslandPopulationMember>>)> {
    if outcomes.len() != ISLANDS
        || outcomes
            .iter()
            .any(|island| island.len() != ISLAND_POPULATION)
    {
        bail!("cannot checkpoint incomplete island outcomes");
    }
    let mut ranked = outcomes.clone();
    for island in &mut ranked {
        island.sort_by(rank_island_outcomes);
    }
    let valid_candidates = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.valid)
        .count();
    let fresh_evaluations = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.cache_source.is_none())
        .count();
    let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
    all_ranked.sort_by(rank_island_outcomes);
    let qualifying_genomes = outcomes
        .iter()
        .flatten()
        .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
        .map(genome_location)
        .collect::<Vec<_>>();
    let (next, migration) = evolve_islands(identity, generation, &ranked)?;
    if let Some((relative, _, record)) = &migration {
        write_or_verify(&root.join(relative), &canonical_json(record)?)?;
    }
    let islands = outcomes
        .into_iter()
        .enumerate()
        .map(|(island, slots)| {
            let fresh = slots
                .iter()
                .filter(|outcome| outcome.cache_source.is_none())
                .count();
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            Ok(IslandGenerationState {
                island,
                logical_slots: ISLAND_POPULATION,
                fresh_evaluations: fresh,
                valid_candidates: valid,
                best_local_slot: ranked[island][0].local_slot,
                best_global_slot: ranked[island][0].global_slot,
                best_semantic_sha256: ranked[island][0].member.semantic_sha256.clone(),
                best_rmse: ranked[island][0].rmse(),
                ranked_local_slots: ranked[island]
                    .iter()
                    .map(|outcome| outcome.local_slot)
                    .collect(),
                slots,
                next_population: next[island].clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let checkpoint = IslandGenerationCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: identity.run_id.clone(),
        dataset_hash: identity.dataset.dataset_hash.clone(),
        generation,
        previous_checkpoint_sha256,
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations,
        cumulative_fresh_evaluations,
        valid_candidates,
        fresh_g0_rmse: g0_rmse,
        global_best: genome_location(&all_ranked[0]),
        global_best_rmse: all_ranked[0].rmse(),
        qualifying_descendant_win: !qualifying_genomes.is_empty(),
        result_label: (!qualifying_genomes.is_empty()).then(|| "new_leader_found".to_string()),
        qualifying_genomes,
        migration_file: migration.as_ref().map(|value| value.0.clone()),
        migration_sha256: migration.as_ref().map(|value| value.1.clone()),
        islands,
    };
    Ok((checkpoint, next))
}

fn receipt_request_path(root: &Path, receipt: &CandidateReceipt) -> PathBuf {
    root.join("requests")
        .join(&receipt.purpose)
        .join(format!("generation-{:03}", receipt.generation))
        .join(format!(
            "slot-{:03}-attempt-{}.v1.json",
            receipt.slot, receipt.attempt
        ))
}

fn validate_population_parent(
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    generation: usize,
    parent: &IslandParentRef,
) -> Result<()> {
    if parent.generation + 1 != generation {
        bail!("island lineage parent is not from the preceding generation");
    }
    if parent.generation < START_GENERATION {
        if parent.island.is_some()
            || parent.global_slot != parent.local_slot
            || combined
                .get(parent.generation)
                .and_then(|checkpoint| checkpoint.0.slots.get(parent.local_slot))
                .is_none_or(|outcome| outcome.member.semantic_sha256 != parent.semantic_sha256)
        {
            bail!("island lineage does not bind to generations 0-19");
        }
    } else {
        let island = parent.island.context("island parent has no island")?;
        if global_slot(island, parent.local_slot) != parent.global_slot
            || checkpoints
                .get(parent.generation - START_GENERATION)
                .and_then(|checkpoint| checkpoint.0.islands.get(island))
                .and_then(|state| state.slots.get(parent.local_slot))
                .is_none_or(|outcome| {
                    outcome.member.semantic_sha256 != parent.semantic_sha256
                        || outcome.global_slot != parent.global_slot
                })
        {
            bail!("island lineage parent location mismatch");
        }
    }
    Ok(())
}

fn validate_checkpoint_history(
    root: &Path,
    identity: &IslandCampaignIdentity,
    binding: &IslandDatasetBinding,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
) -> Result<usize> {
    let mut expected_population = generation_20_populations(combined, identity.dataset.features)?;
    let expected_hashes = expected_population
        .iter()
        .enumerate()
        .map(|(island, population)| {
            Ok(IslandPopulationHash {
                island,
                sha256: sha256_bytes(&canonical_json(population)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if expected_hashes != binding.generation_20_island_population_hashes {
        bail!("generation 20 island population hashes changed");
    }
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut cumulative_fresh = 0;
    let mut observed = BTreeMap::<(usize, usize), IslandSlotOutcome>::new();
    for (offset, (checkpoint, _)) in checkpoints.iter().enumerate() {
        let generation = START_GENERATION + offset;
        if checkpoint.generation != generation {
            bail!("island checkpoint generation sequence mismatch");
        }
        let mut outcomes = Vec::with_capacity(ISLANDS);
        for island in 0..ISLANDS {
            let state = &checkpoint.islands[island];
            if state.island != island
                || state.logical_slots != ISLAND_POPULATION
                || state.slots.len() != ISLAND_POPULATION
                || state.next_population.len()
                    != if generation + 1 < END_GENERATION {
                        ISLAND_POPULATION
                    } else {
                        0
                    }
            {
                bail!("incomplete 30-slot island {island} in generation {generation}");
            }
            let mut fresh = 0;
            let mut valid = 0;
            for local_slot in 0..ISLAND_POPULATION {
                let outcome = &state.slots[local_slot];
                let expected = &expected_population[island][local_slot];
                if outcome.generation != generation
                    || outcome.island != island
                    || outcome.local_slot != local_slot
                    || outcome.global_slot != global_slot(island, local_slot)
                    || outcome.member != expected.member
                    || outcome.parents != expected.parents
                    || outcome.member.ancestry.parents
                        != outcome
                            .parents
                            .iter()
                            .map(|parent| parent.semantic_sha256.clone())
                            .collect::<Vec<_>>()
                {
                    bail!("island population or global-slot mapping mismatch");
                }
                for parent in &outcome.parents {
                    validate_population_parent(combined, checkpoints, generation, parent)?;
                }
                let canonical = canonicalize_for_features(
                    &outcome.member.canonical_genome,
                    identity.dataset.features,
                )?;
                if canonical.canonical != outcome.member.canonical_genome
                    || canonical.semantic_sha256 != outcome.member.semantic_sha256
                    || canonical.complexity != outcome.member.complexity
                    || canonical.rust_source_len != outcome.member.rust_source_len
                {
                    bail!("island checkpoint contains a noncanonical genome");
                }
                let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                    bail!("island receipt hash mismatch");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                let request: CandidateRequest =
                    serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
                validate_receipt_against_request(&receipt, &request, test)?;
                validate_island_receipt_binding(identity, outcome, &receipt, &request)?;
                if let Some(source) = &outcome.cache_source {
                    if source.generation > generation
                        || (source.generation == generation && source.slot >= outcome.global_slot)
                    {
                        bail!("island cache source is not earlier in current-binary history");
                    }
                    let source_outcome = observed
                        .get(&(source.generation, source.slot))
                        .context("island cache source is absent")?;
                    if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                        || source_outcome.receipt_file != outcome.receipt_file
                        || source_outcome.receipt_sha256 != outcome.receipt_sha256
                    {
                        bail!("island cache source mismatch");
                    }
                } else {
                    if receipt.generation != generation || receipt.slot != outcome.global_slot {
                        bail!("fresh island receipt has the wrong logical slot");
                    }
                    fresh += 1;
                }
                valid += usize::from(outcome.valid);
                observed.insert((generation, outcome.global_slot), outcome.clone());
            }
            let mut ranked = state.slots.clone();
            ranked.sort_by(rank_island_outcomes);
            if valid != state.valid_candidates
                || valid < ISLAND_MIN_VALID
                || fresh != state.fresh_evaluations
                || state.ranked_local_slots
                    != ranked
                        .iter()
                        .map(|outcome| outcome.local_slot)
                        .collect::<Vec<_>>()
                || state.best_local_slot != ranked[0].local_slot
                || state.best_global_slot != ranked[0].global_slot
                || state.best_semantic_sha256 != ranked[0].member.semantic_sha256
                || state.best_rmse.to_bits() != ranked[0].rmse().to_bits()
            {
                bail!("island ranking or validity mismatch");
            }
            outcomes.push(state.slots.clone());
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        let fresh = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.cache_source.is_none())
            .count();
        cumulative_fresh += fresh;
        let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
        all_ranked.sort_by(rank_island_outcomes);
        let qualifying = outcomes
            .iter()
            .flatten()
            .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
            .map(genome_location)
            .collect::<Vec<_>>();
        if checkpoint.valid_candidates != valid
            || valid < GLOBAL_MIN_VALID
            || checkpoint.fresh_evaluations != fresh
            || checkpoint.cumulative_fresh_evaluations != cumulative_fresh
            || checkpoint.fresh_g0_rmse.to_bits() != g0_rmse.to_bits()
            || checkpoint.global_best != genome_location(&all_ranked[0])
            || checkpoint.global_best_rmse.to_bits() != all_ranked[0].rmse().to_bits()
            || checkpoint.qualifying_genomes != qualifying
            || checkpoint.qualifying_descendant_win != !qualifying.is_empty()
            || checkpoint.result_label
                != (!qualifying.is_empty()).then(|| "new_leader_found".to_string())
        {
            bail!("island global accounting or qualifying-genome list mismatch");
        }
        let mut ranked = outcomes;
        for island in &mut ranked {
            island.sort_by(rank_island_outcomes);
        }
        let (expected_next, migration) = evolve_islands(identity, generation, &ranked)?;
        match migration {
            Some((relative, sha256, record)) => {
                let bytes = fs::read(root.join(&relative))?;
                if checkpoint.migration_file.as_deref() != Some(&relative)
                    || checkpoint.migration_sha256.as_deref() != Some(&sha256)
                    || sha256_bytes(&bytes) != sha256
                    || bytes != canonical_json(&record)?
                {
                    bail!("island migration record mismatch");
                }
            }
            None if checkpoint.migration_file.is_none()
                && checkpoint.migration_sha256.is_none() => {}
            None => bail!("unexpected island migration binding"),
        }
        if checkpoint
            .islands
            .iter()
            .enumerate()
            .any(|(island, state)| state.next_population != expected_next[island])
        {
            bail!("island next population mismatch");
        }
        expected_population = expected_next;
    }
    Ok(cumulative_fresh)
}

fn load_island_cache(
    root: &Path,
    checkpoints: &[(IslandGenerationCheckpoint, String)],
) -> Result<HashMap<String, CachedEvaluation>> {
    let mut cache = HashMap::new();
    for (checkpoint, _) in checkpoints {
        for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
            if outcome.cache_source.is_some() {
                continue;
            }
            let bytes = fs::read(root.join(&outcome.receipt_file))?;
            let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
            cache.insert(
                outcome.member.semantic_sha256.clone(),
                CachedEvaluation {
                    source: CacheSource {
                        generation: outcome.generation,
                        slot: outcome.global_slot,
                    },
                    receipt_file: outcome.receipt_file.clone(),
                    receipt_sha256: outcome.receipt_sha256.clone(),
                    receipt,
                },
            );
        }
    }
    Ok(cache)
}

fn reconstruct_island_lineage(
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    selected: &IslandSlotOutcome,
) -> Result<Vec<IslandLineageRecord>> {
    fn visit_old(
        combined: &[(GenerationCheckpoint, String)],
        generation: usize,
        slot: usize,
        seen: &mut BTreeSet<IslandParentRef>,
        output: &mut Vec<IslandLineageRecord>,
    ) -> Result<()> {
        let outcome = combined
            .get(generation)
            .and_then(|checkpoint| checkpoint.0.slots.get(slot))
            .context("old island-lineage location is absent")?;
        let key = IslandParentRef {
            generation,
            island: None,
            local_slot: slot,
            global_slot: slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
        };
        if !seen.insert(key) {
            return Ok(());
        }
        let mut parents = Vec::new();
        for hash in &outcome.member.ancestry.parents {
            let location = (0..generation).rev().find_map(|parent_generation| {
                combined[parent_generation]
                    .0
                    .slots
                    .iter()
                    .find(|candidate| candidate.member.semantic_sha256 == *hash)
                    .map(|candidate| (parent_generation, candidate.slot))
            });
            if let Some((parent_generation, parent_slot)) = location {
                visit_old(combined, parent_generation, parent_slot, seen, output)?;
                parents.push(IslandParentRef {
                    generation: parent_generation,
                    island: None,
                    local_slot: parent_slot,
                    global_slot: parent_slot,
                    semantic_sha256: hash.clone(),
                });
            } else if outcome.member.ancestry.operation != "seed" {
                bail!("old island-lineage parent is absent");
            }
        }
        output.push(IslandLineageRecord {
            generation,
            island: None,
            local_slot: slot,
            global_slot: slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
            operation: outcome.member.ancestry.operation.clone(),
            parents,
        });
        Ok(())
    }

    fn visit_island(
        combined: &[(GenerationCheckpoint, String)],
        checkpoints: &[(IslandGenerationCheckpoint, String)],
        reference: &IslandParentRef,
        seen: &mut BTreeSet<IslandParentRef>,
        output: &mut Vec<IslandLineageRecord>,
    ) -> Result<()> {
        if reference.generation < START_GENERATION {
            return visit_old(
                combined,
                reference.generation,
                reference.local_slot,
                seen,
                output,
            );
        }
        if !seen.insert(reference.clone()) {
            return Ok(());
        }
        let island = reference
            .island
            .context("island lineage reference has no island")?;
        let outcome = checkpoints
            .get(reference.generation - START_GENERATION)
            .and_then(|checkpoint| checkpoint.0.islands.get(island))
            .and_then(|state| state.slots.get(reference.local_slot))
            .context("island lineage reference is absent")?;
        if outcome.global_slot != reference.global_slot
            || outcome.member.semantic_sha256 != reference.semantic_sha256
        {
            bail!("island lineage reference does not match its outcome");
        }
        for parent in &outcome.parents {
            visit_island(combined, checkpoints, parent, seen, output)?;
        }
        output.push(IslandLineageRecord {
            generation: outcome.generation,
            island: Some(outcome.island),
            local_slot: outcome.local_slot,
            global_slot: outcome.global_slot,
            semantic_sha256: outcome.member.semantic_sha256.clone(),
            operation: outcome.member.ancestry.operation.clone(),
            parents: outcome.parents.clone(),
        });
        Ok(())
    }

    let selected_ref = island_parent_ref(selected);
    let mut output = Vec::new();
    visit_island(
        combined,
        checkpoints,
        &selected_ref,
        &mut BTreeSet::new(),
        &mut output,
    )?;
    output.sort_by(|left, right| {
        left.generation
            .cmp(&right.generation)
            .then_with(|| left.island.cmp(&right.island))
            .then_with(|| left.local_slot.cmp(&right.local_slot))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    Ok(output)
}

fn island_genome_record(
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    outcome: &IslandSlotOutcome,
    g0_rmse: f64,
    leader_rmse: f64,
    binary_sha256: &str,
) -> Result<IslandGenomeRecord> {
    let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
    if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
        bail!("island genome receipt hash mismatch");
    }
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let metrics = outcome
        .metrics
        .as_ref()
        .context("selected island genome is invalid")?;
    Ok(IslandGenomeRecord {
        canonical_genome: outcome.member.canonical_genome.clone(),
        semantic_sha256: outcome.member.semantic_sha256.clone(),
        generation: outcome.generation,
        island: outcome.island,
        local_slot: outcome.local_slot,
        global_slot: outcome.global_slot,
        rmse: metrics.rmse,
        mae: metrics.mae,
        r2: metrics.r2,
        margin_vs_g0: g0_rmse - metrics.rmse,
        margin_vs_leader: leader_rmse - metrics.rmse,
        complexity: outcome.member.complexity,
        rust_source_len: outcome.member.rust_source_len,
        runtime_seconds: receipt.runtime_seconds,
        lineage: reconstruct_island_lineage(combined, checkpoints, outcome)?,
        prediction_sha256: receipt.predictions_sha256,
        receipt_file: outcome.receipt_file.clone(),
        receipt_sha256: outcome.receipt_sha256.clone(),
        binary_sha256: binary_sha256.to_string(),
    })
}

fn find_island_outcome<'a>(
    checkpoint: &'a IslandGenerationCheckpoint,
    location: &IslandGenomeLocation,
) -> Result<&'a IslandSlotOutcome> {
    checkpoint
        .islands
        .get(location.island)
        .and_then(|state| state.slots.get(location.local_slot))
        .filter(|outcome| {
            outcome.generation == location.generation
                && outcome.global_slot == location.global_slot
                && outcome.member.semantic_sha256 == location.semantic_sha256
        })
        .context("island genome location is absent")
}

fn collect_qualifying_records(
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    spec: &DatasetSpec,
    binary_sha256: &str,
) -> Result<Vec<IslandGenomeRecord>> {
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut records = Vec::new();
    for (checkpoint, _) in checkpoints {
        for location in &checkpoint.qualifying_genomes {
            records.push(island_genome_record(
                root,
                combined,
                checkpoints,
                find_island_outcome(checkpoint, location)?,
                g0_rmse,
                spec.leader_rmse,
                binary_sha256,
            )?);
        }
    }
    Ok(records)
}

fn replay_island_champion(
    worker: &CampaignIdentity,
    registry: &Path,
    campaign_root: &Path,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<ReplayRecord>> {
    let evaluations = scq_batch::replay_across_nodes(
        campaign_root,
        worker,
        registry,
        &worker.binary_sha256,
        scq_binary_sha256,
        generation,
        champion,
        test,
    )?;
    let baseline = evaluations
        .first()
        .context("cross-node replay produced no receipts")?;
    let baseline_rmse = baseline
        .receipt
        .metrics
        .as_ref()
        .context("cross-node replay failed")?
        .rmse;
    let baseline_predictions = baseline.receipt.predictions.clone();
    evaluations
        .into_iter()
        .map(|evaluation| {
            let rmse = evaluation
                .receipt
                .metrics
                .as_ref()
                .context("cross-node replay failed")?
                .rmse;
            let maximum_prediction_difference = baseline_predictions
                .iter()
                .zip(&evaluation.receipt.predictions)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            if (rmse - baseline_rmse).abs() > 1e-12 || maximum_prediction_difference > 1e-10 {
                bail!("cross-node replay disagreement");
            }
            Ok(ReplayRecord {
                receipt_file: evaluation.receipt_file,
                receipt_sha256: evaluation.receipt_sha256,
                rmse,
                maximum_prediction_difference,
            })
        })
        .collect()
}

fn finalize_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    identity: &IslandCampaignIdentity,
    root: &Path,
    combined: &[(GenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
    started_at_unix_ms: u128,
) -> Result<IslandDatasetResult> {
    if checkpoints.len() != GENERATIONS {
        bail!("cannot finalize an incomplete island dataset");
    }
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let replay = replay_island_champion(
        &worker_identity(identity),
        registry,
        &island_root(&manifest.base_run_id),
        &manifest.scq_binary_sha256,
        END_GENERATION - 1,
        &selected.member,
        test,
    )?;
    let qualifying_descendants = collect_qualifying_records(
        root,
        combined,
        checkpoints,
        &identity.dataset,
        &identity.release_binary_sha256,
    )?;
    let first = qualifying_descendants
        .iter()
        .min_by_key(|record| (record.generation, record.island, record.global_slot));
    let result = IslandDatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: identity.run_id.clone(),
        dataset_hash: identity.dataset.dataset_hash.clone(),
        outcome: if first.is_some() {
            "new_leader_found".to_string()
        } else {
            "no_win_after_30_total_generations".to_string()
        },
        start_generation: START_GENERATION,
        stopping_generation: END_GENERATION - 1,
        generations_evaluated: GENERATIONS,
        total_generations_evaluated: END_GENERATION,
        leader_models: identity.dataset.leader_models.clone(),
        leader_rmse: identity.dataset.leader_rmse,
        fresh_g0_rmse: combined[0].0.g0_best_rmse,
        first_winning_generation: first.map(|record| record.generation),
        first_winning_island: first.map(|record| record.island),
        generation_29_selected_genome: island_genome_record(
            root,
            combined,
            checkpoints,
            selected,
            combined[0].0.g0_best_rmse,
            identity.dataset.leader_rmse,
            &identity.release_binary_sha256,
        )?,
        qualifying_descendants,
        logical_evaluations: GENERATIONS * TOTAL_POPULATION,
        fresh_evaluations: terminal.cumulative_fresh_evaluations,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
        replay,
        release_binary_sha256: identity.release_binary_sha256.clone(),
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

fn generation_progress(
    dataset_hash: &str,
    checkpoint: &IslandGenerationCheckpoint,
    completed_new_leaders: usize,
    already_won: bool,
) {
    let island_text = checkpoint
        .islands
        .iter()
        .map(|island| {
            format!(
                "i{} valid {}/{} best {:.17}",
                island.island, island.valid_candidates, ISLAND_POPULATION, island.best_rmse
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let migration = checkpoint
        .migration_file
        .as_deref()
        .map_or("none", |_| "ring top-2 complete");
    let cumulative = EXISTING_SURPASSED
        + completed_new_leaders
        + usize::from(already_won || checkpoint.qualifying_descendant_win);
    let cached = checkpoint.logical_slots - checkpoint.fresh_evaluations;
    let mut node_counts = BTreeMap::new();
    for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
        if outcome.cache_source.is_none() {
            *node_counts.entry(outcome.node.as_str()).or_insert(0usize) += 1;
        }
    }
    let node_text = node_counts
        .into_iter()
        .map(|(node, count)| format!("{node}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let batches = checkpoint.fresh_evaluations.div_ceil(scq_batch::SCQ_CPUS);
    eprintln!(
        "island extension {dataset_hash} generation {}: {island_text}; global best {:.17}; fresh {} cache {cached}; migration {migration}; SCQ batches {batches} nodes [{node_text}]; cumulative surpassed {cumulative}/42",
        checkpoint.generation,
        checkpoint.global_best_rmse,
        checkpoint.fresh_evaluations,
    );
}

fn run_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    binding: &IslandDatasetBinding,
    spec: &DatasetSpec,
    resume: bool,
    completed_new_leaders: usize,
    dataset_index: usize,
) -> Result<IslandDatasetResult> {
    let root = island_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    if root.join("final-result.v1.json").is_file() {
        return validate_dataset(registry, manifest, binding, spec);
    }
    let identity = island_identity(manifest, spec);
    let campaign_path = root.join("campaign.v1.json");
    if campaign_path.is_file() {
        if !resume {
            bail!("island dataset already exists; pass --resume");
        }
        let bytes = fs::read(&campaign_path)?;
        let recorded: IslandCampaignIdentity = serde_json::from_slice(&bytes)?;
        if recorded != identity || bytes != canonical_json(&identity)? {
            bail!("island dataset identity changed on resume");
        }
        if fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)? {
            bail!("island dataset prior binding changed on resume");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&campaign_path, &canonical_json(&identity)?)?;
        immutable_write(
            &root.join("prior-binding.v1.json"),
            &canonical_json(binding)?,
        )?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, spec)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_island_checkpoints(&root, &identity, &binding.prior_terminal_checkpoint_sha256)?;
    validate_checkpoint_history(&root, &identity, binding, &combined, &checkpoints, &test)?;
    if checkpoints.len() == GENERATIONS {
        return finalize_dataset(
            registry,
            manifest,
            &identity,
            &root,
            &combined,
            &checkpoints,
            &test,
            started_at_unix_ms,
        );
    }
    let mut cache = load_island_cache(&root, &checkpoints)?;
    let mut population = checkpoints.last().map_or_else(
        || generation_20_populations(&combined, spec.features),
        |(checkpoint, _)| {
            Ok(checkpoint
                .islands
                .iter()
                .map(|island| island.next_population.clone())
                .collect())
        },
    )?;
    let mut previous = checkpoints.last().map_or_else(
        || binding.prior_terminal_checkpoint_sha256.clone(),
        |checkpoint| checkpoint.1.clone(),
    );
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    let mut already_won = checkpoints
        .iter()
        .any(|checkpoint| checkpoint.0.qualifying_descendant_win);
    let worker = worker_identity(&identity);
    let mut completed = checkpoints;
    for generation in (START_GENERATION + completed.len())..END_GENERATION {
        let (outcomes, fresh) = evaluate_island_generation(
            &worker,
            registry,
            &island_root(&manifest.base_run_id),
            &manifest.scq_binary_sha256,
            dataset_index,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        for (island, slots) in outcomes.iter().enumerate() {
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            if valid < ISLAND_MIN_VALID {
                bail!("generation {generation} island {island} has only {valid} valid candidates");
            }
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        if valid < GLOBAL_MIN_VALID {
            bail!("generation {generation} has only {valid} globally valid candidates");
        }
        cumulative_fresh += fresh;
        let (checkpoint, next) = make_checkpoint(
            &identity,
            &root,
            generation,
            previous.clone(),
            outcomes,
            cumulative_fresh,
            combined[0].0.g0_best_rmse,
        )?;
        if checkpoint.fresh_evaluations != fresh {
            bail!("fresh island evaluation accounting mismatch");
        }
        let bytes = canonical_json(&checkpoint)?;
        let sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        generation_progress(
            &spec.dataset_hash,
            &checkpoint,
            completed_new_leaders,
            already_won,
        );
        already_won |= checkpoint.qualifying_descendant_win;
        previous = sha256.clone();
        population = next;
        completed.push((checkpoint, sha256));
    }
    finalize_dataset(
        registry,
        manifest,
        &identity,
        &root,
        &combined,
        &completed,
        &test,
        started_at_unix_ms,
    )
}

fn validate_replays(
    root: &Path,
    identity: &IslandCampaignIdentity,
    result: &IslandDatasetResult,
    test: &RegressionData,
) -> Result<()> {
    if result.replay.len() != 3 {
        bail!("island final replay count mismatch");
    }
    let mut receipts = Vec::new();
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!("island replay receipt hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request: CandidateRequest =
            serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
        validate_receipt_against_request(&receipt, &request, test)?;
        if receipt.purpose != "replay-final"
            || receipt.generation != END_GENERATION - 1
            || receipt.slot != slot
            || receipt.run_id != identity.run_id
            || receipt.binary_sha256 != identity.release_binary_sha256
            || receipt.semantic_sha256 != result.generation_29_selected_genome.semantic_sha256
            || receipt.node != scq_batch::NODES[slot]
        {
            bail!("island replay provenance mismatch");
        }
        receipts.push(receipt);
    }
    let base = &receipts[0];
    let base_rmse = base.metrics.as_ref().context("island replay failed")?.rmse;
    for (index, receipt) in receipts.iter().enumerate() {
        let rmse = receipt
            .metrics
            .as_ref()
            .context("island replay failed")?
            .rmse;
        let difference = base
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if (rmse - base_rmse).abs() > 1e-12
            || difference > 1e-10
            || result.replay[index].rmse.to_bits() != rmse.to_bits()
            || result.replay[index].maximum_prediction_difference.to_bits() != difference.to_bits()
        {
            bail!("island replay disagreement");
        }
    }
    Ok(())
}

fn validate_dataset(
    registry: &Path,
    manifest: &IslandManifest,
    binding: &IslandDatasetBinding,
    spec: &DatasetSpec,
) -> Result<IslandDatasetResult> {
    let root = island_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    if sha256_file(Path::new(&binding.prior_campaign_file))? != binding.prior_campaign_sha256
        || sha256_file(Path::new(&binding.prior_final_result_file))?
            != binding.prior_final_result_sha256
        || sha256_file(Path::new(&binding.prior_terminal_checkpoint_file))?
            != binding.prior_terminal_checkpoint_sha256
    {
        bail!("dataset {} prior artifacts changed", spec.dataset_hash);
    }
    let identity = island_identity(manifest, spec);
    let campaign_bytes = fs::read(root.join("campaign.v1.json"))?;
    let recorded_identity: IslandCampaignIdentity = serde_json::from_slice(&campaign_bytes)?;
    if recorded_identity != identity
        || campaign_bytes != canonical_json(&identity)?
        || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
    {
        bail!("dataset {} island identity mismatch", spec.dataset_hash);
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_island_checkpoints(&root, &identity, &binding.prior_terminal_checkpoint_sha256)?;
    if checkpoints.len() != GENERATIONS {
        bail!(
            "dataset {} has incomplete island checkpoints",
            spec.dataset_hash
        );
    }
    let cumulative_fresh =
        validate_checkpoint_history(&root, &identity, binding, &combined, &checkpoints, &test)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let expected_selected = island_genome_record(
        &root,
        &combined,
        &checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let qualifying = collect_qualifying_records(
        &root,
        &combined,
        &checkpoints,
        spec,
        &identity.release_binary_sha256,
    )?;
    let first = qualifying
        .iter()
        .min_by_key(|record| (record.generation, record.island, record.global_slot));
    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: IslandDatasetResult = serde_json::from_slice(&result_bytes)?;
    let expected_outcome = if first.is_some() {
        "new_leader_found"
    } else {
        "no_win_after_30_total_generations"
    };
    if result_bytes != canonical_json(&result)?
        || result.schema != RESULT_SCHEMA
        || result.benchmark_valid
        || result.base_run_id != manifest.base_run_id
        || result.island_extension_run_id != manifest.island_extension_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != expected_outcome
        || result.start_generation != START_GENERATION
        || result.stopping_generation != END_GENERATION - 1
        || result.generations_evaluated != GENERATIONS
        || result.total_generations_evaluated != END_GENERATION
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.fresh_g0_rmse.to_bits() != combined[0].0.g0_best_rmse.to_bits()
        || result.first_winning_generation != first.map(|record| record.generation)
        || result.first_winning_island != first.map(|record| record.island)
        || result.generation_29_selected_genome != expected_selected
        || result.qualifying_descendants != qualifying
        || result.logical_evaluations != GENERATIONS * TOTAL_POPULATION
        || result.fresh_evaluations != cumulative_fresh
        || result.release_binary_sha256 != manifest.release_binary_sha256
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!("dataset {} island final result mismatch", spec.dataset_hash);
    }
    validate_replays(&root, &identity, &result, &test)?;
    Ok(result)
}

fn artifact_name(path: &Path) -> String {
    path.strip_prefix(workspace_root())
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn benchmark_status(rmse: f64, leader_rmse: f64) -> String {
    if rmse < leader_rmse {
        "surpassed_external_benchmark".to_string()
    } else {
        "did_not_surpass_external_benchmark".to_string()
    }
}

fn make_best_solutions(
    manifest: &IslandManifest,
    specs: &[DatasetSpec],
    island_results: &[IslandDatasetResult],
) -> Result<Golden42BestSolutions> {
    let base_root = run_root(&manifest.base_run_id);
    let base_summary: PanelSummary =
        serde_json::from_slice(&fs::read(base_root.join(super::SUMMARY_FILE))?)?;
    let base_map: WinnerMap = serde_json::from_slice(&fs::read(base_root.join(WINNER_MAP_FILE))?)?;
    let prior_root = extension_root(&manifest.base_run_id);
    let prior_summary: ExtensionSummary =
        serde_json::from_slice(&fs::read(prior_root.join(EXTENSION_SUMMARY_FILE))?)?;
    let specs = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let base_genomes = base_map
        .datasets
        .iter()
        .map(|entry| (entry.dataset_hash.as_str(), &entry.final_selected_genome))
        .collect::<HashMap<_, _>>();
    let extension_results = prior_summary
        .results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let island_results = island_results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let mut solutions = Vec::with_capacity(42);
    for dataset in &base_summary.datasets {
        let hash = dataset.dataset_hash.as_str();
        let spec = specs
            .get(hash)
            .context("best-solutions dataset is absent from Golden42")?;
        if dataset.outcome == "early_win" {
            let genome = base_genomes
                .get(hash)
                .context("base winner is absent from winner map")?;
            let source = base_root
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            let result: DatasetResult = serde_json::from_slice(&fs::read(&source)?)?;
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.base_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: None,
                slot: genome.slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        } else if let Some(result) = extension_results
            .get(hash)
            .filter(|result| result.outcome == "new_leader_found")
        {
            let genome = &result.final_selected_genome;
            let source = prior_root
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.prior_extension_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: None,
                slot: genome.slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        } else {
            let result = island_results
                .get(hash)
                .context("unresolved dataset is absent from island results")?;
            let genome = &result.generation_29_selected_genome;
            let source = island_root(&manifest.base_run_id)
                .join("datasets")
                .join(hash)
                .join("final-result.v1.json");
            solutions.push(Golden42BestSolution {
                dataset_hash: hash.to_string(),
                source_campaign: manifest.island_extension_run_id.clone(),
                source_artifact: artifact_name(&source),
                source_artifact_sha256: sha256_file(&source)?,
                benchmark_status: benchmark_status(genome.rmse, spec.leader_rmse),
                leader_rmse: spec.leader_rmse,
                selected_rmse: genome.rmse,
                margin_vs_leader: spec.leader_rmse - genome.rmse,
                canonical_genome: genome.canonical_genome.clone(),
                semantic_sha256: genome.semantic_sha256.clone(),
                generation: genome.generation,
                island: Some(genome.island),
                slot: genome.global_slot,
                binary_sha256: genome.binary_sha256.clone(),
                prediction_sha256: genome.prediction_sha256.clone(),
                receipt_sha256: genome.receipt_sha256.clone(),
                replay_receipt_sha256: result
                    .replay
                    .iter()
                    .map(|replay| replay.receipt_sha256.clone())
                    .collect(),
            });
        }
    }
    if solutions.len() != 42
        || solutions
            .iter()
            .zip(&base_summary.datasets)
            .any(|(solution, summary)| solution.dataset_hash != summary.dataset_hash)
        || solutions
            .iter()
            .take(35)
            .filter(|solution| solution.benchmark_status == "surpassed_external_benchmark")
            .count()
            > 35
    {
        bail!("Golden42 best-solutions map ordering or membership mismatch");
    }
    let prior_winner_count = solutions
        .iter()
        .filter(|solution| solution.source_campaign != manifest.island_extension_run_id)
        .count();
    if prior_winner_count != EXISTING_SURPASSED {
        bail!("Golden42 best-solutions map does not preserve 35 prior winners");
    }
    Ok(Golden42BestSolutions {
        schema: BEST_SOLUTIONS_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: manifest.island_extension_run_id.clone(),
        ordered_dataset_ids: base_summary
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_hash.clone())
            .collect(),
        ordered_cohort_sha256: base_summary.ordered_cohort_sha256,
        dataset_count: solutions.len(),
        prior_winner_count,
        island_selected_count: solutions.len() - prior_winner_count,
        solutions,
    })
}

fn make_summary(
    manifest: &IslandManifest,
    results: Vec<IslandDatasetResult>,
    root: &Path,
    best_solutions_sha256: String,
) -> Result<IslandSummary> {
    let new_leaders = results
        .iter()
        .filter(|result| result.outcome == "new_leader_found")
        .count();
    Ok(IslandSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        prior_extension_run_id: manifest.prior_extension_run_id.clone(),
        island_extension_run_id: manifest.island_extension_run_id.clone(),
        base_panel_manifest_sha256: manifest.base_panel_manifest_sha256.clone(),
        base_panel_summary_sha256: manifest.base_panel_summary_sha256.clone(),
        base_winner_map_sha256: manifest.base_winner_map_sha256.clone(),
        prior_extension_manifest_sha256: manifest.prior_extension_manifest_sha256.clone(),
        prior_extension_summary_sha256: manifest.prior_extension_summary_sha256.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        ordered_dataset_ids: manifest.ordered_dataset_ids.clone(),
        ordered_cohort_sha256: manifest.ordered_cohort_sha256.clone(),
        dataset_count: results.len(),
        existing_surpassed_count: EXISTING_SURPASSED,
        new_leaders,
        still_no_leader: results.len() - new_leaders,
        cumulative_surpassed_count: EXISTING_SURPASSED + new_leaders,
        total_qualifying_genomes: results
            .iter()
            .map(|result| result.qualifying_descendants.len())
            .sum(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        best_solutions_file: BEST_SOLUTIONS_FILE.to_string(),
        best_solutions_sha256,
        results,
        artifacts: collect_artifacts_excluding(root, SUMMARY_FILE)?,
    })
}

fn island_specs<'a>(base_run_id: &str, specs: &'a [DatasetSpec]) -> Result<Vec<&'a DatasetSpec>> {
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    prior_no_win_ids(base_run_id)?
        .iter()
        .map(|dataset| {
            by_hash
                .get(dataset.as_str())
                .copied()
                .context("island cohort dataset is absent from Golden42")
        })
        .collect()
}

pub(crate) fn run_island_extension(registry: &Path, base_run_id: &str, resume: bool) -> Result<()> {
    validate_run_id(base_run_id)?;
    // This audit runs before the immutable manifest is created. It begins by
    // independently validating the base panel, so it preflights both prior
    // campaigns.
    validate_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let release_binary_sha256 = current_binary_sha256()?;
    let root = island_root(base_run_id);
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("island extension already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: IslandManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let cohort = island_specs(base_run_id, &specs)?;
        let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
        scq_batch::validate_staging(&staging, &recorded.release_binary_sha256, &cohort)?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_manifest(
            base_run_id,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            &specs,
            recorded.started_at_unix_ms,
        )?;
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("island manifest, prior artifacts, cohort, populations, or binary changed");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("island extension run directory already exists");
        }
        let cohort = island_specs(base_run_id, &specs)?;
        let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
        let release_binary = std::env::current_exe()?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &cohort,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            base_run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            &specs,
            unix_millis(),
        )?;
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(SUMMARY_FILE).is_file() {
        return validate_island_extension(registry, base_run_id);
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let completed_new_leaders = std::sync::atomic::AtomicUsize::new(0);
    let manifest_ref = &manifest;
    let results = thread::scope(|scope| -> Result<Vec<IslandDatasetResult>> {
        let mut handles = Vec::with_capacity(manifest_ref.datasets.len());
        for (index, binding) in manifest_ref.datasets.iter().enumerate() {
            let spec = *by_hash
                .get(binding.dataset_hash.as_str())
                .context("island manifest dataset is absent from Golden42")?;
            eprintln!(
                "island extension dataset {}/{} {}",
                index + 1,
                manifest_ref.datasets.len(),
                binding.dataset_hash
            );
            let counter = &completed_new_leaders;
            let manifest = manifest_ref;
            handles.push(scope.spawn(move || {
                let result = run_dataset(
                    registry,
                    &manifest,
                    binding,
                    spec,
                    resume,
                    counter.load(std::sync::atomic::Ordering::Relaxed),
                    index,
                )?;
                if result.outcome == "new_leader_found" {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(result)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("island dataset controller panicked"))?
            })
            .collect()
    })?;
    let best_solutions = make_best_solutions(&manifest, &specs, &results)?;
    let best_solutions_bytes = canonical_json(&best_solutions)?;
    immutable_write(&root.join(BEST_SOLUTIONS_FILE), &best_solutions_bytes)?;
    let summary = make_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary.dataset_count != 7
        || summary.total_logical_evaluations != 8_400
        || summary.cumulative_surpassed_count > 42
    {
        bail!("island summary campaign accounting mismatch");
    }
    immutable_write(&root.join(SUMMARY_FILE), &canonical_json(&summary)?)?;
    validate_island_extension(registry, base_run_id)
}

pub(crate) fn validate_island_extension(registry: &Path, base_run_id: &str) -> Result<()> {
    validate_run_id(base_run_id)?;
    // This also validates the complete base panel before opening the prior
    // extension artifacts.
    validate_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let root = island_root(base_run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: IslandManifest = serde_json::from_slice(&manifest_bytes)?;
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let cohort = island_specs(base_run_id, &specs)?;
    let cohort = cohort.into_iter().cloned().collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &cohort)?;
    let expected = expected_manifest(
        base_run_id,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        &specs,
        manifest.started_at_unix_ms,
    )?;
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("island manifest does not reconstruct from immutable prior campaigns");
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let mut results = Vec::with_capacity(manifest.datasets.len());
    let mut expected_fresh = Vec::new();
    for binding in &manifest.datasets {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("island dataset is absent from Golden42")?;
        results.push(validate_dataset(registry, &manifest, binding, spec)?);
        let dataset_root = root.join("datasets").join(&binding.dataset_hash);
        let identity = island_identity(&manifest, spec);
        for (checkpoint, _) in load_island_checkpoints(
            &dataset_root,
            &identity,
            &binding.prior_terminal_checkpoint_sha256,
        )? {
            for outcome in checkpoint
                .islands
                .iter()
                .flat_map(|island| &island.slots)
                .filter(|outcome| outcome.cache_source.is_none())
            {
                expected_fresh.push(scq_batch::ExpectedFresh {
                    dataset_hash: binding.dataset_hash.clone(),
                    generation: outcome.generation,
                    slot: outcome.global_slot,
                    attempt: outcome.attempt,
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    node: outcome.node.clone(),
                    scq_job_id: outcome.scq_job_id.clone(),
                });
            }
        }
    }
    scq_batch::validate_batch_records(
        &root,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &manifest.ordered_dataset_ids,
        &expected_fresh,
    )?;
    let best_solutions_bytes = fs::read(root.join(BEST_SOLUTIONS_FILE))?;
    let recorded_best_solutions: Golden42BestSolutions =
        serde_json::from_slice(&best_solutions_bytes)?;
    let expected_best_solutions = make_best_solutions(&manifest, &specs, &results)?;
    if best_solutions_bytes != canonical_json(&recorded_best_solutions)?
        || recorded_best_solutions != expected_best_solutions
        || recorded_best_solutions.dataset_count != 42
        || recorded_best_solutions.prior_winner_count != 35
        || recorded_best_solutions.island_selected_count != 7
    {
        bail!("Golden42 best-solutions map does not reconstruct");
    }
    let summary_bytes = fs::read(root.join(SUMMARY_FILE))?;
    let recorded: IslandSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary = make_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.dataset_count != 7
        || recorded.total_logical_evaluations != 8_400
        || recorded.cumulative_surpassed_count != EXISTING_SURPASSED + recorded.new_leaders
    {
        bail!("island extension summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated four-island extension for {} datasets: {} new leaders, {} still no leader, {} qualifying genomes, {} logical evaluations, {} fresh evaluations, cumulative surpassed {}/42; summary {}",
        recorded.dataset_count,
        recorded.new_leaders,
        recorded.still_no_leader,
        recorded.total_qualifying_genomes,
        recorded.total_logical_evaluations,
        recorded.total_fresh_evaluations,
        recorded.cumulative_surpassed_count,
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}

fn champion_bindings(base_run_id: &str) -> Result<Vec<ChampionDatasetBinding>> {
    let root = island_root(base_run_id);
    let summary: IslandSummary = serde_json::from_slice(&fs::read(root.join(SUMMARY_FILE))?)?;
    let mut bindings = Vec::new();
    for result in &summary.results {
        if result.outcome == "new_leader_found" {
            continue;
        }
        let result_path = root
            .join("datasets")
            .join(&result.dataset_hash)
            .join("final-result.v1.json");
        let checkpoint_path = root
            .join("datasets")
            .join(&result.dataset_hash)
            .join("checkpoints/generation-029.v1.json");
        bindings.push(ChampionDatasetBinding {
            dataset_hash: result.dataset_hash.clone(),
            initial_result_file: artifact_name(&result_path),
            initial_result_sha256: sha256_file(&result_path)?,
            generation_29_checkpoint_file: artifact_name(&checkpoint_path),
            generation_29_checkpoint_sha256: sha256_file(&checkpoint_path)?,
        });
    }
    if EXISTING_SURPASSED + summary.new_leaders + bindings.len() != 42 {
        bail!("sealed island block does not partition all 42 datasets");
    }
    Ok(bindings)
}

fn expected_champion_manifest(
    base_run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    staging_sha256: &str,
    started_at_unix_ms: u128,
) -> Result<ChampionManifest> {
    let initial_root = island_root(base_run_id);
    let bindings = champion_bindings(base_run_id)?;
    let ordered_dataset_ids = bindings
        .iter()
        .map(|binding| binding.dataset_hash.clone())
        .collect::<Vec<_>>();
    Ok(ChampionManifest {
        schema: CHAMPION_MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: base_run_id.to_string(),
        initial_island_run_id: island_run_id(base_run_id),
        champion_run_id: champion_run_id(base_run_id),
        initial_manifest_sha256: sha256_file(&initial_root.join(MANIFEST_FILE))?,
        initial_summary_sha256: sha256_file(&initial_root.join(SUMMARY_FILE))?,
        initial_best_solutions_sha256: sha256_file(
            &initial_root.join(BEST_SOLUTIONS_FILE),
        )?,
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_dataset_ids),
        ordered_dataset_ids,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: staging_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::fixed(),
        config: IslandConfig::champion(),
        migration_policy: MigrationPolicy::champion(),
        rng_policy: RngPolicy::fixed(),
        complexity_schedule:
            "depth 1 at generations 30-39, depth 2 at 40-49, depth 3 at 50-59, depth 4 thereafter; crossover probability rises from 0.25 to 0.55"
                .to_string(),
        stop_policy:
            "complete each generation; stop a dataset only after an evolved valid genome has RMSE strictly below both generation-0 and the external leader"
                .to_string(),
        datasets: bindings,
        started_at_unix_ms,
    })
}

fn champion_identity(manifest: &ChampionManifest, spec: &DatasetSpec) -> IslandCampaignIdentity {
    IslandCampaignIdentity {
        schema: CHAMPION_CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.champion_run_id.clone(),
        dataset: spec.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        config: manifest.config.clone(),
        migration_policy: manifest.migration_policy.clone(),
        rng_policy: manifest.rng_policy.clone(),
        selection_objective: "test-target RMSE; continue until strict external-leader crossing"
            .to_string(),
    }
}

fn load_initial_island_checkpoints(
    base_run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let root = island_root(base_run_id);
    let manifest: IslandManifest = serde_json::from_slice(&fs::read(root.join(MANIFEST_FILE))?)?;
    let binding = manifest
        .datasets
        .iter()
        .find(|binding| binding.dataset_hash == spec.dataset_hash)
        .context("champion dataset is absent from initial island manifest")?;
    let identity = island_identity(&manifest, spec);
    let checkpoints = load_island_checkpoints(
        &root.join("datasets").join(&spec.dataset_hash),
        &identity,
        &binding.prior_terminal_checkpoint_sha256,
    )?;
    if checkpoints.len() != GENERATIONS {
        bail!("initial island history is incomplete");
    }
    Ok(checkpoints)
}

fn generation_30_populations(
    initial: &[(IslandGenerationCheckpoint, String)],
    features: usize,
) -> Result<Vec<Vec<IslandPopulationMember>>> {
    let terminal = &initial
        .last()
        .context("initial island history has no terminal checkpoint")?
        .0;
    if terminal.generation != CHAMPION_START_GENERATION - 1 {
        bail!("champion continuation source is not generation 29");
    }
    terminal
        .islands
        .iter()
        .enumerate()
        .map(|(island, state)| {
            let mut ranked = state.slots.clone();
            ranked.sort_by(rank_island_outcomes);
            next_champion_population(&ranked, CHAMPION_START_GENERATION, island, features)
        })
        .collect()
}

fn evolve_champion_islands(
    identity: &IslandCampaignIdentity,
    generation: usize,
    ranked: &[Vec<IslandSlotOutcome>],
    terminal: bool,
) -> Result<(
    Vec<Vec<IslandPopulationMember>>,
    Option<(String, String, MigrationRecord)>,
)> {
    if terminal {
        return Ok((vec![Vec::new(); ISLANDS], None));
    }
    let (parents, migration) = if generation >= CHAMPION_START_GENERATION + 1 && generation % 2 == 1
    {
        let (parents, record) = migrate_rankings_champion(
            &identity.run_id,
            &identity.dataset.dataset_hash,
            generation,
            ranked,
        )?;
        let relative = migration_relative(generation);
        let bytes = canonical_json(&record)?;
        let sha256 = sha256_bytes(&bytes);
        (parents, Some((relative, sha256, record)))
    } else {
        (ranked.to_vec(), None)
    };
    let next = parents
        .iter()
        .enumerate()
        .map(|(island, island_ranked)| {
            next_champion_population(
                island_ranked,
                generation + 1,
                island,
                identity.dataset.features,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((next, migration))
}

fn make_champion_checkpoint(
    identity: &IslandCampaignIdentity,
    root: &Path,
    generation: usize,
    previous_checkpoint_sha256: String,
    outcomes: Vec<Vec<IslandSlotOutcome>>,
    cumulative_fresh_evaluations: usize,
    g0_rmse: f64,
) -> Result<(IslandGenerationCheckpoint, Vec<Vec<IslandPopulationMember>>)> {
    if outcomes.len() != ISLANDS
        || outcomes
            .iter()
            .any(|island| island.len() != ISLAND_POPULATION)
    {
        bail!("cannot checkpoint incomplete champion outcomes");
    }
    let mut ranked = outcomes.clone();
    for island in &mut ranked {
        island.sort_by(rank_island_outcomes);
    }
    let valid_candidates = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.valid)
        .count();
    let fresh_evaluations = outcomes
        .iter()
        .flatten()
        .filter(|outcome| outcome.cache_source.is_none())
        .count();
    let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
    all_ranked.sort_by(rank_island_outcomes);
    let qualifying_genomes = outcomes
        .iter()
        .flatten()
        .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
        .map(genome_location)
        .collect::<Vec<_>>();
    let terminal = !qualifying_genomes.is_empty();
    let (next, migration) = evolve_champion_islands(identity, generation, &ranked, terminal)?;
    if let Some((relative, _, record)) = &migration {
        write_or_verify(&root.join(relative), &canonical_json(record)?)?;
    }
    let islands = outcomes
        .into_iter()
        .enumerate()
        .map(|(island, slots)| {
            let fresh = slots
                .iter()
                .filter(|outcome| outcome.cache_source.is_none())
                .count();
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            IslandGenerationState {
                island,
                logical_slots: ISLAND_POPULATION,
                fresh_evaluations: fresh,
                valid_candidates: valid,
                best_local_slot: ranked[island][0].local_slot,
                best_global_slot: ranked[island][0].global_slot,
                best_semantic_sha256: ranked[island][0].member.semantic_sha256.clone(),
                best_rmse: ranked[island][0].rmse(),
                ranked_local_slots: ranked[island]
                    .iter()
                    .map(|outcome| outcome.local_slot)
                    .collect(),
                slots,
                next_population: next[island].clone(),
            }
        })
        .collect();
    Ok((
        IslandGenerationCheckpoint {
            schema: CHAMPION_CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: identity.run_id.clone(),
            dataset_hash: identity.dataset.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256,
            logical_slots: TOTAL_POPULATION,
            fresh_evaluations,
            cumulative_fresh_evaluations,
            valid_candidates,
            fresh_g0_rmse: g0_rmse,
            global_best: genome_location(&all_ranked[0]),
            global_best_rmse: all_ranked[0].rmse(),
            qualifying_descendant_win: terminal,
            qualifying_genomes,
            result_label: terminal.then(|| "new_leader_found".to_string()),
            migration_file: migration.as_ref().map(|value| value.0.clone()),
            migration_sha256: migration.as_ref().map(|value| value.1.clone()),
            islands,
        },
        next,
    ))
}

fn load_champion_checkpoints(
    root: &Path,
    identity: &IslandCampaignIdentity,
    initial_terminal_sha256: &str,
) -> Result<Vec<(IslandGenerationCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    let mut previous = initial_terminal_sha256.to_string();
    let mut generation = CHAMPION_START_GENERATION;
    loop {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: IslandGenerationCheckpoint = serde_json::from_slice(&bytes)?;
        let terminal = checkpoint.qualifying_descendant_win;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHAMPION_CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || checkpoint.run_id != identity.run_id
            || checkpoint.dataset_hash != identity.dataset.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.logical_slots != TOTAL_POPULATION
            || checkpoint.islands.len() != ISLANDS
            || checkpoint.islands.iter().any(|island| {
                island.logical_slots != ISLAND_POPULATION
                    || island.slots.len() != ISLAND_POPULATION
                    || island.next_population.len() != if terminal { 0 } else { ISLAND_POPULATION }
            })
        {
            bail!("invalid champion checkpoint {}", path.display());
        }
        let sha256 = sha256_bytes(&bytes);
        previous = sha256.clone();
        checkpoints.push((checkpoint, sha256));
        generation += 1;
        if terminal {
            if checkpoint_path(root, generation).is_file() {
                bail!("champion history continues after a strict leader crossing");
            }
            break;
        }
    }
    Ok(checkpoints)
}

fn validate_champion_history(
    root: &Path,
    identity: &IslandCampaignIdentity,
    initial: &[(IslandGenerationCheckpoint, String)],
    checkpoints: &[(IslandGenerationCheckpoint, String)],
    test: &RegressionData,
) -> Result<usize> {
    let mut expected_population = generation_30_populations(initial, identity.dataset.features)?;
    let lineage_checkpoints = initial
        .iter()
        .cloned()
        .chain(checkpoints.iter().cloned())
        .collect::<Vec<_>>();
    let combined = prior_history(
        &identity.run_id[..identity
            .run_id
            .find("-champions-g30-plus")
            .context("invalid champion run id")?],
        &identity.dataset,
    )?;
    let g0_rmse = combined[0].0.g0_best_rmse;
    let mut cumulative_fresh = 0usize;
    let mut observed = BTreeMap::<(usize, usize), IslandSlotOutcome>::new();
    for (offset, (checkpoint, _)) in checkpoints.iter().enumerate() {
        let generation = CHAMPION_START_GENERATION + offset;
        if checkpoint.generation != generation {
            bail!("champion checkpoint generation sequence mismatch");
        }
        let mut outcomes = Vec::with_capacity(ISLANDS);
        for island in 0..ISLANDS {
            let state = &checkpoint.islands[island];
            let terminal = checkpoint.qualifying_descendant_win;
            if state.island != island
                || state.logical_slots != ISLAND_POPULATION
                || state.slots.len() != ISLAND_POPULATION
                || state.next_population.len() != if terminal { 0 } else { ISLAND_POPULATION }
            {
                bail!("incomplete champion island {island} in generation {generation}");
            }
            let mut fresh = 0usize;
            let mut valid = 0usize;
            for local_slot in 0..ISLAND_POPULATION {
                let outcome = &state.slots[local_slot];
                let expected = &expected_population[island][local_slot];
                if outcome.generation != generation
                    || outcome.island != island
                    || outcome.local_slot != local_slot
                    || outcome.global_slot != global_slot(island, local_slot)
                    || outcome.member != expected.member
                    || outcome.parents != expected.parents
                    || outcome.member.ancestry.parents
                        != outcome
                            .parents
                            .iter()
                            .map(|parent| parent.semantic_sha256.clone())
                            .collect::<Vec<_>>()
                {
                    bail!("champion population or slot mapping mismatch");
                }
                for parent in &outcome.parents {
                    validate_population_parent(
                        &combined,
                        &lineage_checkpoints,
                        generation,
                        parent,
                    )?;
                }
                let canonical = canonicalize_for_features(
                    &outcome.member.canonical_genome,
                    identity.dataset.features,
                )?;
                if canonical.canonical != outcome.member.canonical_genome
                    || canonical.semantic_sha256 != outcome.member.semantic_sha256
                    || canonical.complexity != outcome.member.complexity
                    || canonical.rust_source_len != outcome.member.rust_source_len
                {
                    bail!("champion checkpoint contains a noncanonical genome");
                }
                let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                    bail!("champion receipt hash mismatch");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                let request: CandidateRequest =
                    serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
                validate_receipt_against_request(&receipt, &request, test)?;
                validate_island_receipt_binding(identity, outcome, &receipt, &request)?;
                if let Some(source) = &outcome.cache_source {
                    if source.generation > generation
                        || (source.generation == generation && source.slot >= outcome.global_slot)
                    {
                        bail!("champion cache source is not earlier");
                    }
                    let source_outcome = observed
                        .get(&(source.generation, source.slot))
                        .context("champion cache source is absent")?;
                    if source_outcome.member.semantic_sha256 != outcome.member.semantic_sha256
                        || source_outcome.receipt_file != outcome.receipt_file
                        || source_outcome.receipt_sha256 != outcome.receipt_sha256
                    {
                        bail!("champion cache source mismatch");
                    }
                } else {
                    if receipt.generation != generation || receipt.slot != outcome.global_slot {
                        bail!("fresh champion receipt has the wrong slot");
                    }
                    fresh += 1;
                }
                valid += usize::from(outcome.valid);
                observed.insert((generation, outcome.global_slot), outcome.clone());
            }
            let mut ranked = state.slots.clone();
            ranked.sort_by(rank_island_outcomes);
            if valid != state.valid_candidates
                || valid < ISLAND_MIN_VALID
                || fresh != state.fresh_evaluations
                || state.ranked_local_slots
                    != ranked
                        .iter()
                        .map(|outcome| outcome.local_slot)
                        .collect::<Vec<_>>()
                || state.best_local_slot != ranked[0].local_slot
                || state.best_global_slot != ranked[0].global_slot
                || state.best_semantic_sha256 != ranked[0].member.semantic_sha256
                || state.best_rmse.to_bits() != ranked[0].rmse().to_bits()
            {
                bail!("champion island ranking or validity mismatch");
            }
            outcomes.push(state.slots.clone());
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        let fresh = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.cache_source.is_none())
            .count();
        cumulative_fresh += fresh;
        let mut all_ranked = outcomes.iter().flatten().cloned().collect::<Vec<_>>();
        all_ranked.sort_by(rank_island_outcomes);
        let qualifying = outcomes
            .iter()
            .flatten()
            .filter(|outcome| qualifies(outcome, g0_rmse, identity.dataset.leader_rmse))
            .map(genome_location)
            .collect::<Vec<_>>();
        if checkpoint.valid_candidates != valid
            || valid < GLOBAL_MIN_VALID
            || checkpoint.fresh_evaluations != fresh
            || checkpoint.cumulative_fresh_evaluations != cumulative_fresh
            || checkpoint.fresh_g0_rmse.to_bits() != g0_rmse.to_bits()
            || checkpoint.global_best != genome_location(&all_ranked[0])
            || checkpoint.global_best_rmse.to_bits() != all_ranked[0].rmse().to_bits()
            || checkpoint.qualifying_genomes != qualifying
            || checkpoint.qualifying_descendant_win != !qualifying.is_empty()
            || checkpoint.result_label
                != (!qualifying.is_empty()).then(|| "new_leader_found".to_string())
        {
            bail!("champion global accounting mismatch");
        }
        let mut ranked = outcomes;
        for island in &mut ranked {
            island.sort_by(rank_island_outcomes);
        }
        let (expected_next, migration) = evolve_champion_islands(
            identity,
            generation,
            &ranked,
            checkpoint.qualifying_descendant_win,
        )?;
        match migration {
            Some((relative, sha256, record)) => {
                let bytes = fs::read(root.join(&relative))?;
                if checkpoint.migration_file.as_deref() != Some(&relative)
                    || checkpoint.migration_sha256.as_deref() != Some(&sha256)
                    || sha256_bytes(&bytes) != sha256
                    || bytes != canonical_json(&record)?
                {
                    bail!("champion migration record mismatch");
                }
            }
            None if checkpoint.migration_file.is_none()
                && checkpoint.migration_sha256.is_none() => {}
            None => bail!("unexpected champion migration binding"),
        }
        if checkpoint
            .islands
            .iter()
            .enumerate()
            .any(|(island, state)| state.next_population != expected_next[island])
        {
            bail!("champion next population mismatch");
        }
        expected_population = expected_next;
    }
    Ok(cumulative_fresh)
}

fn champion_generation_progress(
    dataset_hash: &str,
    checkpoint: &IslandGenerationCheckpoint,
    completed_champions: usize,
) {
    let island_text = checkpoint
        .islands
        .iter()
        .map(|island| {
            format!(
                "i{} valid {}/30 best {:.17}",
                island.island, island.valid_candidates, island.best_rmse
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let cached = TOTAL_POPULATION - checkpoint.fresh_evaluations;
    let migration = if checkpoint.migration_file.is_some() {
        "ring top-2 complete"
    } else {
        "none"
    };
    let mut node_counts = BTreeMap::<&str, usize>::new();
    for outcome in checkpoint.islands.iter().flat_map(|island| &island.slots) {
        if outcome.cache_source.is_none() {
            *node_counts.entry(outcome.node.as_str()).or_default() += 1;
        }
    }
    let nodes = node_counts
        .into_iter()
        .map(|(node, count)| format!("{node}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "champion continuation {dataset_hash} generation {} depth {}: {island_text}; global best {:.17}; fresh {} cache {cached}; migration {migration}; nodes [{nodes}]; cumulative surpassed {}/42",
        checkpoint.generation,
        champion_complexity_level(checkpoint.generation),
        checkpoint.global_best_rmse,
        checkpoint.fresh_evaluations,
        EXISTING_SURPASSED + completed_champions + usize::from(checkpoint.qualifying_descendant_win),
    );
}

fn validate_champion_replays(
    root: &Path,
    identity: &IslandCampaignIdentity,
    result: &ChampionDatasetResult,
    test: &RegressionData,
) -> Result<()> {
    if result.replay.len() != scq_batch::NODES.len() {
        bail!("champion replay count mismatch");
    }
    let original: CandidateReceipt =
        serde_json::from_slice(&fs::read(root.join(&result.champion.receipt_file))?)?;
    let original_rmse = original
        .metrics
        .as_ref()
        .context("champion source receipt failed")?
        .rmse;
    for (slot, replay) in result.replay.iter().enumerate() {
        let bytes = fs::read(root.join(&replay.receipt_file))?;
        if sha256_bytes(&bytes) != replay.receipt_sha256 {
            bail!("champion replay receipt hash mismatch");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
        let request: CandidateRequest =
            serde_json::from_slice(&fs::read(receipt_request_path(root, &receipt))?)?;
        validate_receipt_against_request(&receipt, &request, test)?;
        let rmse = receipt
            .metrics
            .as_ref()
            .context("champion cross-node replay failed")?
            .rmse;
        let difference = original
            .predictions
            .iter()
            .zip(&receipt.predictions)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if receipt.purpose != "replay-final"
            || receipt.generation != result.stopping_generation
            || receipt.slot != slot
            || receipt.run_id != identity.run_id
            || receipt.binary_sha256 != identity.release_binary_sha256
            || receipt.semantic_sha256 != result.champion.semantic_sha256
            || receipt.node != scq_batch::NODES[slot]
            || (rmse - original_rmse).abs() > 1e-12
            || difference > 1e-10
            || replay.rmse.to_bits() != rmse.to_bits()
        {
            bail!("champion replay disagreement");
        }
    }
    Ok(())
}

fn validate_champion_dataset(
    registry: &Path,
    manifest: &ChampionManifest,
    binding: &ChampionDatasetBinding,
    spec: &DatasetSpec,
) -> Result<ChampionDatasetResult> {
    if sha256_file(Path::new(&binding.initial_result_file))? != binding.initial_result_sha256
        || sha256_file(Path::new(&binding.generation_29_checkpoint_file))?
            != binding.generation_29_checkpoint_sha256
    {
        bail!("champion source artifacts changed");
    }
    let root = champion_root(&manifest.base_run_id)
        .join("datasets")
        .join(&spec.dataset_hash);
    let identity = champion_identity(manifest, spec);
    let campaign_bytes = fs::read(root.join("campaign.v1.json"))?;
    let recorded_identity: IslandCampaignIdentity = serde_json::from_slice(&campaign_bytes)?;
    if campaign_bytes != canonical_json(&recorded_identity)?
        || recorded_identity != identity
        || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
    {
        bail!("champion dataset identity mismatch");
    }
    let (_, test) = read_campaign_data(registry, spec)?;
    let initial = load_initial_island_checkpoints(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_champion_checkpoints(&root, &identity, &binding.generation_29_checkpoint_sha256)?;
    if checkpoints
        .last()
        .is_none_or(|checkpoint| !checkpoint.0.qualifying_descendant_win)
    {
        bail!("champion dataset has no terminal leader crossing");
    }
    let cumulative_fresh =
        validate_champion_history(&root, &identity, &initial, &checkpoints, &test)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let all_checkpoints = initial
        .iter()
        .cloned()
        .chain(checkpoints.iter().cloned())
        .collect::<Vec<_>>();
    let champion = island_genome_record(
        &root,
        &combined,
        &all_checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: ChampionDatasetResult = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&result)?
        || result.schema != CHAMPION_RESULT_SCHEMA
        || result.benchmark_valid
        || result.base_run_id != manifest.base_run_id
        || result.champion_run_id != manifest.champion_run_id
        || result.dataset_hash != spec.dataset_hash
        || result.outcome != "new_leader_found"
        || result.start_generation != CHAMPION_START_GENERATION
        || result.stopping_generation != terminal.generation
        || result.generations_evaluated != checkpoints.len()
        || result.logical_evaluations != checkpoints.len() * TOTAL_POPULATION
        || result.fresh_evaluations != cumulative_fresh
        || result.leader_models != spec.leader_models
        || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
        || result.champion != champion
        || result.champion.rmse >= spec.leader_rmse
        || result.release_binary_sha256 != manifest.release_binary_sha256
        || result.finished_at_unix_ms < result.started_at_unix_ms
    {
        bail!("champion final result mismatch");
    }
    validate_champion_replays(&root, &identity, &result, &test)?;
    Ok(result)
}

fn run_champion_dataset(
    registry: &Path,
    manifest: &ChampionManifest,
    binding: &ChampionDatasetBinding,
    spec: &DatasetSpec,
    resume: bool,
    completed_champions: usize,
    dataset_index: usize,
) -> Result<ChampionDatasetResult> {
    let campaign_root = champion_root(&manifest.base_run_id);
    let root = campaign_root.join("datasets").join(&spec.dataset_hash);
    if root.join("final-result.v1.json").is_file() {
        return validate_champion_dataset(registry, manifest, binding, spec);
    }
    let identity = champion_identity(manifest, spec);
    let campaign_path = root.join("campaign.v1.json");
    if campaign_path.is_file() {
        if !resume {
            bail!("champion dataset already exists; pass --resume");
        }
        if fs::read(&campaign_path)? != canonical_json(&identity)?
            || fs::read(root.join("prior-binding.v1.json"))? != canonical_json(binding)?
        {
            bail!("champion dataset identity changed on resume");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&campaign_path, &canonical_json(&identity)?)?;
        immutable_write(
            &root.join("prior-binding.v1.json"),
            &canonical_json(binding)?,
        )?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, spec)?;
    let initial = load_initial_island_checkpoints(&manifest.base_run_id, spec)?;
    let checkpoints =
        load_champion_checkpoints(&root, &identity, &binding.generation_29_checkpoint_sha256)?;
    validate_champion_history(&root, &identity, &initial, &checkpoints, &test)?;
    if checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.qualifying_descendant_win)
    {
        return validate_champion_dataset(registry, manifest, binding, spec);
    }
    let mut cache = load_island_cache(&root, &checkpoints)?;
    let mut population = checkpoints.last().map_or_else(
        || generation_30_populations(&initial, spec.features),
        |checkpoint| {
            Ok(checkpoint
                .0
                .islands
                .iter()
                .map(|island| island.next_population.clone())
                .collect())
        },
    )?;
    let mut previous = checkpoints.last().map_or_else(
        || binding.generation_29_checkpoint_sha256.clone(),
        |checkpoint| checkpoint.1.clone(),
    );
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    let mut completed = checkpoints;
    let worker = worker_identity(&identity);
    loop {
        let generation = CHAMPION_START_GENERATION + completed.len();
        let (outcomes, fresh) = evaluate_island_generation(
            &worker,
            registry,
            &campaign_root,
            &manifest.scq_binary_sha256,
            dataset_index,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        for (island, slots) in outcomes.iter().enumerate() {
            let valid = slots.iter().filter(|outcome| outcome.valid).count();
            if valid < ISLAND_MIN_VALID {
                bail!(
                    "champion generation {generation} island {island} has only {valid} valid candidates"
                );
            }
        }
        let valid = outcomes
            .iter()
            .flatten()
            .filter(|outcome| outcome.valid)
            .count();
        if valid < GLOBAL_MIN_VALID {
            bail!("champion generation {generation} has only {valid} valid candidates");
        }
        cumulative_fresh += fresh;
        let combined = prior_history(&manifest.base_run_id, spec)?;
        let (checkpoint, next) = make_champion_checkpoint(
            &identity,
            &root,
            generation,
            previous.clone(),
            outcomes,
            cumulative_fresh,
            combined[0].0.g0_best_rmse,
        )?;
        if checkpoint.fresh_evaluations != fresh {
            bail!("fresh champion evaluation accounting mismatch");
        }
        let bytes = canonical_json(&checkpoint)?;
        let sha256 = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        champion_generation_progress(&spec.dataset_hash, &checkpoint, completed_champions);
        let won = checkpoint.qualifying_descendant_win;
        previous = sha256.clone();
        population = next;
        completed.push((checkpoint, sha256));
        if won {
            break;
        }
    }
    let terminal = &completed.last().unwrap().0;
    let selected = find_island_outcome(terminal, &terminal.global_best)?;
    let combined = prior_history(&manifest.base_run_id, spec)?;
    let all_checkpoints = initial
        .iter()
        .cloned()
        .chain(completed.iter().cloned())
        .collect::<Vec<_>>();
    let champion = island_genome_record(
        &root,
        &combined,
        &all_checkpoints,
        selected,
        combined[0].0.g0_best_rmse,
        spec.leader_rmse,
        &identity.release_binary_sha256,
    )?;
    let replay = replay_island_champion(
        &worker,
        registry,
        &campaign_root,
        &manifest.scq_binary_sha256,
        terminal.generation,
        &selected.member,
        &test,
    )?;
    let result = ChampionDatasetResult {
        schema: CHAMPION_RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        champion_run_id: manifest.champion_run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        outcome: "new_leader_found".to_string(),
        start_generation: CHAMPION_START_GENERATION,
        stopping_generation: terminal.generation,
        generations_evaluated: completed.len(),
        logical_evaluations: completed.len() * TOTAL_POPULATION,
        fresh_evaluations: cumulative_fresh,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        champion,
        replay,
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    validate_champion_dataset(registry, manifest, binding, spec)
}

fn make_champion_best_solutions(
    manifest: &ChampionManifest,
    results: &[ChampionDatasetResult],
) -> Result<Golden42BestSolutions> {
    let initial_path = island_root(&manifest.base_run_id).join(BEST_SOLUTIONS_FILE);
    let initial: Golden42BestSolutions = serde_json::from_slice(&fs::read(&initial_path)?)?;
    let by_dataset = results
        .iter()
        .map(|result| (result.dataset_hash.as_str(), result))
        .collect::<HashMap<_, _>>();
    let mut solutions = Vec::with_capacity(initial.solutions.len());
    for solution in initial.solutions {
        if solution.benchmark_status == "surpassed_external_benchmark" {
            solutions.push(solution);
            continue;
        }
        let result = by_dataset
            .get(solution.dataset_hash.as_str())
            .context("unresolved Golden42 solution has no champion result")?;
        let source = champion_root(&manifest.base_run_id)
            .join("datasets")
            .join(&result.dataset_hash)
            .join("final-result.v1.json");
        solutions.push(Golden42BestSolution {
            dataset_hash: result.dataset_hash.clone(),
            source_campaign: manifest.champion_run_id.clone(),
            source_artifact: artifact_name(&source),
            source_artifact_sha256: sha256_file(&source)?,
            benchmark_status: benchmark_status(result.champion.rmse, result.leader_rmse),
            leader_rmse: result.leader_rmse,
            selected_rmse: result.champion.rmse,
            margin_vs_leader: result.leader_rmse - result.champion.rmse,
            canonical_genome: result.champion.canonical_genome.clone(),
            semantic_sha256: result.champion.semantic_sha256.clone(),
            generation: result.champion.generation,
            island: Some(result.champion.island),
            slot: result.champion.global_slot,
            binary_sha256: result.champion.binary_sha256.clone(),
            prediction_sha256: result.champion.prediction_sha256.clone(),
            receipt_sha256: result.champion.receipt_sha256.clone(),
            replay_receipt_sha256: result
                .replay
                .iter()
                .map(|replay| replay.receipt_sha256.clone())
                .collect(),
        });
    }
    if solutions.len() != 42
        || solutions
            .iter()
            .any(|solution| solution.benchmark_status != "surpassed_external_benchmark")
        || solutions
            .iter()
            .map(|solution| solution.dataset_hash.as_str())
            .collect::<Vec<_>>()
            != initial
                .ordered_dataset_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
    {
        bail!("champion best-solutions map is not an ordered 42/42 winner map");
    }
    Ok(Golden42BestSolutions {
        schema: BEST_SOLUTIONS_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        island_extension_run_id: manifest.champion_run_id.clone(),
        ordered_dataset_ids: initial.ordered_dataset_ids,
        ordered_cohort_sha256: initial.ordered_cohort_sha256,
        dataset_count: 42,
        prior_winner_count: 42 - results.len(),
        island_selected_count: results.len(),
        solutions,
    })
}

fn make_champion_summary(
    manifest: &ChampionManifest,
    results: Vec<ChampionDatasetResult>,
    root: &Path,
    best_solutions_sha256: String,
) -> Result<ChampionSummary> {
    let initial: IslandSummary = serde_json::from_slice(&fs::read(
        island_root(&manifest.base_run_id).join(SUMMARY_FILE),
    )?)?;
    Ok(ChampionSummary {
        schema: CHAMPION_SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        base_run_id: manifest.base_run_id.clone(),
        initial_island_run_id: manifest.initial_island_run_id.clone(),
        champion_run_id: manifest.champion_run_id.clone(),
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        ordered_dataset_ids: manifest.ordered_dataset_ids.clone(),
        dataset_count: results.len(),
        inherited_champion_count: initial.cumulative_surpassed_count,
        continuation_champion_count: results.len(),
        cumulative_surpassed_count: initial.cumulative_surpassed_count + results.len(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        best_solutions_file: BEST_SOLUTIONS_FILE.to_string(),
        best_solutions_sha256,
        results,
        artifacts: collect_artifacts_excluding(root, CHAMPION_SUMMARY_FILE)?,
    })
}

pub(crate) fn run_champion_continuation(
    registry: &Path,
    base_run_id: &str,
    resume: bool,
) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_island_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let release_binary_sha256 = current_binary_sha256()?;
    let root = champion_root(base_run_id);
    let manifest_path = root.join(CHAMPION_MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("champion continuation already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: ChampionManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let by_hash = specs
            .iter()
            .map(|spec| (spec.dataset_hash.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let cohort = recorded
            .ordered_dataset_ids
            .iter()
            .map(|dataset| {
                by_hash
                    .get(dataset.as_str())
                    .copied()
                    .context("champion cohort dataset is absent from Golden42")
            })
            .collect::<Result<Vec<_>>>()?;
        scq_batch::validate_staging(
            &staging,
            &recorded.release_binary_sha256,
            &cohort.into_iter().cloned().collect::<Vec<_>>(),
        )?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_champion_manifest(
            base_run_id,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            recorded.started_at_unix_ms,
        )?;
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("champion manifest, ancestors, staging, or binary changed");
        }
        recorded
    } else {
        if root.exists() && !resume {
            bail!("champion continuation directory already exists");
        }
        let bindings = champion_bindings(base_run_id)?;
        let by_hash = specs
            .iter()
            .map(|spec| (spec.dataset_hash.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let cohort = bindings
            .iter()
            .map(|binding| {
                by_hash
                    .get(binding.dataset_hash.as_str())
                    .copied()
                    .context("champion cohort dataset is absent from Golden42")
            })
            .collect::<Result<Vec<_>>>()?;
        let release_binary = std::env::current_exe()?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &cohort.into_iter().cloned().collect::<Vec<_>>(),
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_champion_manifest(
            base_run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            unix_millis(),
        )?;
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(CHAMPION_SUMMARY_FILE).is_file() {
        return validate_champion_continuation(registry, base_run_id);
    }
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let completed_champions = std::sync::atomic::AtomicUsize::new(0);
    let results = thread::scope(|scope| -> Result<Vec<ChampionDatasetResult>> {
        let mut handles = Vec::with_capacity(manifest.datasets.len());
        for (index, binding) in manifest.datasets.iter().enumerate() {
            let spec = *by_hash
                .get(binding.dataset_hash.as_str())
                .context("champion manifest dataset is absent from Golden42")?;
            eprintln!(
                "champion continuation dataset {}/{} {}",
                index + 1,
                manifest.datasets.len(),
                binding.dataset_hash
            );
            let counter = &completed_champions;
            let manifest = &manifest;
            handles.push(scope.spawn(move || {
                let result = run_champion_dataset(
                    registry,
                    manifest,
                    binding,
                    spec,
                    resume,
                    counter.load(std::sync::atomic::Ordering::Relaxed),
                    index,
                )?;
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(result)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("champion dataset controller panicked"))?
            })
            .collect()
    })?;
    let best_solutions = make_champion_best_solutions(&manifest, &results)?;
    let best_solutions_bytes = canonical_json(&best_solutions)?;
    immutable_write(&root.join(BEST_SOLUTIONS_FILE), &best_solutions_bytes)?;
    let summary = make_champion_summary(
        &manifest,
        results,
        &root,
        sha256_bytes(&best_solutions_bytes),
    )?;
    if summary.cumulative_surpassed_count != 42 {
        bail!("champion continuation stopped before 42/42");
    }
    immutable_write(
        &root.join(CHAMPION_SUMMARY_FILE),
        &canonical_json(&summary)?,
    )?;
    validate_champion_continuation(registry, base_run_id)
}

pub(crate) fn validate_champion_continuation(registry: &Path, base_run_id: &str) -> Result<()> {
    validate_run_id(base_run_id)?;
    validate_island_extension(registry, base_run_id)?;
    let (_, specs, _, _) = load_panel_specs(registry)?;
    let root = champion_root(base_run_id);
    let manifest_bytes = fs::read(root.join(CHAMPION_MANIFEST_FILE))?;
    let manifest: ChampionManifest = serde_json::from_slice(&manifest_bytes)?;
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let by_hash = specs
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<HashMap<_, _>>();
    let cohort = manifest
        .ordered_dataset_ids
        .iter()
        .map(|dataset| {
            by_hash
                .get(dataset.as_str())
                .copied()
                .context("champion cohort dataset is absent from Golden42")
        })
        .collect::<Result<Vec<_>>>()?;
    scq_batch::validate_staging(
        &staging,
        &manifest.release_binary_sha256,
        &cohort.into_iter().cloned().collect::<Vec<_>>(),
    )?;
    let expected_manifest = expected_champion_manifest(
        base_run_id,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    )?;
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected_manifest {
        bail!("champion continuation manifest does not reconstruct");
    }
    let mut results = Vec::with_capacity(manifest.datasets.len());
    let mut expected_fresh = Vec::new();
    for binding in &manifest.datasets {
        let spec = by_hash
            .get(binding.dataset_hash.as_str())
            .context("champion dataset is absent from Golden42")?;
        results.push(validate_champion_dataset(
            registry, &manifest, binding, spec,
        )?);
        let dataset_root = root.join("datasets").join(&binding.dataset_hash);
        let identity = champion_identity(&manifest, spec);
        for (checkpoint, _) in load_champion_checkpoints(
            &dataset_root,
            &identity,
            &binding.generation_29_checkpoint_sha256,
        )? {
            for outcome in checkpoint
                .islands
                .iter()
                .flat_map(|island| &island.slots)
                .filter(|outcome| outcome.cache_source.is_none())
            {
                expected_fresh.push(scq_batch::ExpectedFresh {
                    dataset_hash: binding.dataset_hash.clone(),
                    generation: outcome.generation,
                    slot: outcome.global_slot,
                    attempt: outcome.attempt,
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    node: outcome.node.clone(),
                    scq_job_id: outcome.scq_job_id.clone(),
                });
            }
        }
    }
    scq_batch::validate_batch_records(
        &root,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &manifest.ordered_dataset_ids,
        &expected_fresh,
    )?;
    let best_bytes = fs::read(root.join(BEST_SOLUTIONS_FILE))?;
    let recorded_best: Golden42BestSolutions = serde_json::from_slice(&best_bytes)?;
    let expected_best = make_champion_best_solutions(&manifest, &results)?;
    if best_bytes != canonical_json(&recorded_best)?
        || recorded_best != expected_best
        || recorded_best.dataset_count != 42
        || recorded_best
            .solutions
            .iter()
            .any(|solution| solution.benchmark_status != "surpassed_external_benchmark")
    {
        bail!("42/42 champion best-solutions map does not reconstruct");
    }
    let summary_bytes = fs::read(root.join(CHAMPION_SUMMARY_FILE))?;
    let recorded: ChampionSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary =
        make_champion_summary(&manifest, results, &root, sha256_bytes(&best_bytes))?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.cumulative_surpassed_count != 42
        || recorded.continuation_champion_count != recorded.dataset_count
    {
        bail!("champion continuation summary does not reconstruct");
    }
    eprintln!(
        "validated Golden42 champion continuation: {} inherited + {} continuation champions = 42/42; {} logical evaluations, {} fresh evaluations; map {}, summary {}",
        recorded.inherited_champion_count,
        recorded.continuation_champion_count,
        recorded.total_logical_evaluations,
        recorded.total_fresh_evaluations,
        sha256_bytes(&best_bytes),
        sha256_bytes(&summary_bytes),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec() -> DatasetSpec {
        DatasetSpec {
            dataset_hash: "dataset".to_string(),
            train_sha256: "train".to_string(),
            test_sha256: "test".to_string(),
            train_rows: 1,
            test_rows: 1,
            features: 20,
            leader_models: vec!["leader".to_string()],
            leader_rmse: 0.5,
        }
    }

    fn test_identity() -> IslandCampaignIdentity {
        IslandCampaignIdentity {
            schema: CAMPAIGN_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "run".to_string(),
            dataset: test_spec(),
            release_binary_sha256: "binary".to_string(),
            config: IslandConfig::fixed(),
            migration_policy: MigrationPolicy::fixed(),
            rng_policy: RngPolicy::fixed(),
            selection_objective: "test".to_string(),
        }
    }

    fn simple_member(hash: &str) -> Member {
        Member {
            canonical_genome: format!("genome-{hash}"),
            semantic_sha256: hash.to_string(),
            complexity: 1,
            rust_source_len: 2,
            evolved_descendant: true,
            ancestry: Ancestry {
                operation: "mutation".to_string(),
                parents: Vec::new(),
            },
        }
    }

    fn outcome(island: usize, local_slot: usize, rmse: f64) -> IslandSlotOutcome {
        IslandSlotOutcome {
            generation: 21,
            island,
            local_slot,
            global_slot: global_slot(island, local_slot),
            member: simple_member(&format!("{island}-{local_slot}")),
            parents: Vec::new(),
            valid: true,
            metrics: Some(CandidateMetrics {
                rmse,
                mae: rmse,
                r2: 0.0,
            }),
            failure: None,
            cache_source: None,
            receipt_file: String::new(),
            receipt_sha256: String::new(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: String::new(),
            attempt: 0,
        }
    }

    fn old_checkpoint(generation: usize, slots: Vec<SlotOutcome>) -> GenerationCheckpoint {
        GenerationCheckpoint {
            schema: super::super::CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "prior".to_string(),
            dataset_hash: "dataset".to_string(),
            generation,
            previous_checkpoint_sha256: None,
            logical_slots: slots.len(),
            fresh_evaluations: slots.len(),
            cumulative_fresh_evaluations: slots.len(),
            valid_candidates: slots.len(),
            g0_best_rmse: 1.0,
            best_slot: 0,
            best_semantic_sha256: slots
                .first()
                .map_or_else(String::new, |slot| slot.member.semantic_sha256.clone()),
            best_rmse: slots.first().map_or(f64::INFINITY, SlotOutcome::rmse),
            qualifying_descendant_win: false,
            result_label: None,
            ranked_slots: slots.iter().map(|slot| slot.slot).collect(),
            slots,
            next_population: Vec::new(),
        }
    }

    fn synthetic_combined() -> Vec<(GenerationCheckpoint, String)> {
        let sources = include_str!("../../seeds/generation-zero.txt")
            .lines()
            .take(POPULATION)
            .enumerate()
            .map(|(slot, source)| {
                let canonical = canonicalize(source).unwrap();
                let metrics = CandidateMetrics {
                    rmse: (slot + 1) as f64 / 100.0,
                    mae: (slot + 1) as f64 / 200.0,
                    r2: 0.0,
                };
                SlotOutcome {
                    generation: START_GENERATION - 1,
                    slot,
                    member: Member {
                        canonical_genome: canonical.canonical,
                        semantic_sha256: canonical.semantic_sha256,
                        complexity: canonical.complexity,
                        rust_source_len: canonical.rust_source_len,
                        evolved_descendant: true,
                        ancestry: Ancestry {
                            operation: "seed".to_string(),
                            parents: Vec::new(),
                        },
                    },
                    valid: true,
                    metrics: Some(metrics),
                    failure: None,
                    cache_source: None,
                    receipt_file: String::new(),
                    receipt_sha256: String::new(),
                    node: EXECUTION_NODE.to_string(),
                    scq_job_id: String::new(),
                    attempt: 0,
                }
            })
            .collect::<Vec<_>>();
        (0..START_GENERATION)
            .map(|generation| {
                let slots = if generation + 1 == START_GENERATION {
                    sources.clone()
                } else {
                    Vec::new()
                };
                (old_checkpoint(generation, slots), format!("g{generation}"))
            })
            .collect()
    }

    fn complete_state(island: usize, generation: usize) -> IslandGenerationState {
        let slots = (0..ISLAND_POPULATION)
            .map(|local| {
                let mut value = outcome(island, local, local as f64);
                value.generation = generation;
                value
            })
            .collect::<Vec<_>>();
        IslandGenerationState {
            island,
            logical_slots: ISLAND_POPULATION,
            fresh_evaluations: ISLAND_POPULATION,
            valid_candidates: ISLAND_POPULATION,
            best_local_slot: 0,
            best_global_slot: global_slot(island, 0),
            best_semantic_sha256: slots[0].member.semantic_sha256.clone(),
            best_rmse: slots[0].rmse(),
            ranked_local_slots: (0..ISLAND_POPULATION).collect(),
            slots,
            next_population: (0..ISLAND_POPULATION)
                .map(|slot| IslandPopulationMember {
                    member: simple_member(&format!("next-{island}-{slot}")),
                    parents: Vec::new(),
                })
                .collect(),
        }
    }

    fn checkpoint_envelope(previous: &str) -> IslandGenerationCheckpoint {
        IslandGenerationCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "run".to_string(),
            dataset_hash: "dataset".to_string(),
            generation: START_GENERATION,
            previous_checkpoint_sha256: previous.to_string(),
            logical_slots: TOTAL_POPULATION,
            fresh_evaluations: TOTAL_POPULATION,
            cumulative_fresh_evaluations: TOTAL_POPULATION,
            valid_candidates: TOTAL_POPULATION,
            fresh_g0_rmse: 1.0,
            global_best: IslandGenomeLocation {
                generation: START_GENERATION,
                island: 0,
                local_slot: 0,
                global_slot: 0,
                semantic_sha256: "0-0".to_string(),
            },
            global_best_rmse: 0.0,
            qualifying_descendant_win: false,
            qualifying_genomes: Vec::new(),
            result_label: None,
            migration_file: None,
            migration_sha256: None,
            islands: (0..ISLANDS)
                .map(|island| complete_state(island, START_GENERATION))
                .collect(),
        }
    }

    #[test]
    fn global_slot_mapping_is_exact_and_reversible() {
        for island in 0..ISLANDS {
            for local in 0..ISLAND_POPULATION {
                let global = global_slot(island, local);
                assert_eq!(island_slot(global).unwrap(), (island, local));
            }
        }
        assert!(island_slot(TOTAL_POPULATION).is_err());
    }

    #[test]
    fn island_rng_namespaces_are_deterministic_and_distinct() {
        let one = island_offspring_seed(20, 0, 4, 0);
        assert_eq!(one, island_offspring_seed(20, 0, 4, 0));
        let values = (0..ISLANDS)
            .map(|island| island_offspring_seed(20, island, 4, 0))
            .collect::<BTreeSet<_>>();
        assert_eq!(values.len(), ISLANDS);
        assert_ne!(island_sequence(0), island_sequence(1));
    }

    #[test]
    fn generation_20_initialization_is_deterministic_and_distributes_distinct_elites() {
        let combined = synthetic_combined();
        let first = generation_20_populations(&combined, 20).unwrap();
        let second = generation_20_populations(&combined, 20).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), ISLANDS);
        assert!(first
            .iter()
            .all(|population| population.len() == ISLAND_POPULATION));
        let elites = first
            .iter()
            .flat_map(|population| &population[..ISLAND_ELITES])
            .map(|member| member.member.semantic_sha256.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(elites.len(), ISLANDS * ISLAND_ELITES);
        for population in &first {
            assert!(population[..ISLAND_ELITES]
                .iter()
                .all(|member| member.parents[0].island.is_none()));
        }
        let offspring = first
            .iter()
            .map(|population| population[ISLAND_ELITES].member.semantic_sha256.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(offspring.len(), ISLANDS);
    }

    #[test]
    fn later_populations_use_only_their_local_parent_pool() {
        let combined = synthetic_combined();
        let terminal = &combined.last().unwrap().0.slots;
        let ranked = terminal[..ISLAND_POPULATION]
            .iter()
            .enumerate()
            .map(|(local_slot, old)| IslandSlotOutcome {
                generation: START_GENERATION,
                island: 2,
                local_slot,
                global_slot: global_slot(2, local_slot),
                member: old.member.clone(),
                parents: Vec::new(),
                valid: true,
                metrics: old.metrics.clone(),
                failure: None,
                cache_source: None,
                receipt_file: String::new(),
                receipt_sha256: String::new(),
                node: EXECUTION_NODE.to_string(),
                scq_job_id: String::new(),
                attempt: 0,
            })
            .collect::<Vec<_>>();
        let next = next_island_population(&ranked, START_GENERATION + 1, 2, 20).unwrap();
        assert!(next
            .iter()
            .all(|member| member.parents.iter().all(|parent| {
                parent.generation == START_GENERATION && parent.island == Some(2)
            })));
    }

    #[test]
    fn ring_migration_replaces_exact_bottom_two() {
        let ranked = (0..ISLANDS)
            .map(|island| {
                (0..ISLAND_POPULATION)
                    .map(|slot| outcome(island, slot, island as f64 + slot as f64 / 100.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (migrated, record) = migrate_rankings("run", "data", 21, &ranked).unwrap();
        assert_eq!(record.transfers.len(), 8);
        for source in 0..ISLANDS {
            let destination = (source + 1) % ISLANDS;
            for rank in 0..2 {
                let transfer = &record.transfers[source * 2 + rank];
                assert_eq!(transfer.source_island, source);
                assert_eq!(transfer.destination_island, destination);
                assert_eq!(transfer.source_rank, rank);
                assert_eq!(
                    transfer.destination_replaced_rank,
                    ISLAND_POPULATION - 2 + rank
                );
                assert_eq!(
                    transfer.migrant.member.semantic_sha256,
                    ranked[source][rank].member.semantic_sha256
                );
                assert_eq!(
                    transfer.replaced.member.semantic_sha256,
                    ranked[destination][ISLAND_POPULATION - 2 + rank]
                        .member
                        .semantic_sha256
                );
                assert!(migrated[destination].iter().any(|outcome| {
                    outcome.member.semantic_sha256 == ranked[source][rank].member.semantic_sha256
                }));
            }
        }
    }

    #[test]
    fn checkpoint_loader_rejects_incomplete_islands_and_broken_chain() {
        let identity = test_identity();
        let mut checkpoint = checkpoint_envelope("prior-terminal");
        validate_checkpoint_envelope(&checkpoint, &identity, START_GENERATION, "prior-terminal")
            .unwrap();
        checkpoint.islands[2].slots.pop();
        assert!(validate_checkpoint_envelope(
            &checkpoint,
            &identity,
            START_GENERATION,
            "prior-terminal"
        )
        .is_err());
        let checkpoint = checkpoint_envelope("wrong-chain");
        assert!(validate_checkpoint_envelope(
            &checkpoint,
            &identity,
            START_GENERATION,
            "prior-terminal"
        )
        .is_err());
    }

    #[test]
    fn migration_policy_is_exact() {
        assert_eq!(
            MigrationPolicy::fixed().after_generations,
            vec![21, 23, 25, 27]
        );
        assert_eq!(
            MigrationPolicy::fixed().destination_replaced_ranks,
            vec![28, 29]
        );
        assert!(MigrationPolicy::fixed().semantic_duplicates_allowed);
    }

    #[test]
    fn duplicate_logical_slots_bind_to_one_cached_receipt() {
        let population = IslandPopulationMember {
            member: simple_member("duplicate"),
            parents: Vec::new(),
        };
        let receipt = CandidateReceipt {
            schema: RECEIPT_SCHEMA.to_string(),
            benchmark_valid: false,
            test_adaptive: None,
            purpose: "candidate".to_string(),
            run_id: "run".to_string(),
            generation: 20,
            slot: 0,
            attempt: 0,
            semantic_sha256: "duplicate".to_string(),
            canonical_genome: population.member.canonical_genome.clone(),
            binary_sha256: "binary".to_string(),
            dataset_hash: "dataset".to_string(),
            train_sha256: "train".to_string(),
            test_sha256: "test".to_string(),
            train_rows: 1,
            test_rows: 1,
            features: 20,
            execution_host: EXECUTION_HOST.to_string(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: "job".to_string(),
            runtime_seconds: 1.0,
            thread_counts: ThreadCounts::single(),
            predictions: vec![0.0],
            predictions_sha256: sha256_bytes(&prediction_bytes(&[0.0])),
            metrics: Some(CandidateMetrics {
                rmse: 0.0,
                mae: 0.0,
                r2: 1.0,
            }),
            failure: None,
        };
        let evaluation = CachedEvaluation {
            source: CacheSource {
                generation: 20,
                slot: 0,
            },
            receipt_file: "receipt.json".to_string(),
            receipt_sha256: "receipt-hash".to_string(),
            receipt,
        };
        let cached = island_outcome_from_evaluation(20, 1, 0, population, &evaluation, true);
        assert_eq!(cached.global_slot, 30);
        assert_eq!(cached.cache_source, Some(evaluation.source));
        assert_eq!(cached.receipt_file, evaluation.receipt_file);
    }

    #[test]
    fn receipt_and_binary_tampering_are_rejected() {
        let identity = test_identity();
        let member = simple_member("candidate");
        let worker = worker_identity(&identity);
        let request = make_request(
            &worker,
            Path::new("registry"),
            "candidate",
            20,
            0,
            0,
            &member,
            EXECUTION_NODE,
        );
        let metrics = CandidateMetrics {
            rmse: 0.0,
            mae: 0.0,
            r2: 1.0,
        };
        let receipt = CandidateReceipt {
            schema: RECEIPT_SCHEMA.to_string(),
            benchmark_valid: false,
            test_adaptive: None,
            purpose: request.purpose.clone(),
            run_id: request.run_id.clone(),
            generation: request.generation,
            slot: request.slot,
            attempt: request.attempt,
            semantic_sha256: request.semantic_sha256.clone(),
            canonical_genome: request.canonical_genome.clone(),
            binary_sha256: request.binary_sha256.clone(),
            dataset_hash: request.dataset_hash.clone(),
            train_sha256: request.train_sha256.clone(),
            test_sha256: request.test_sha256.clone(),
            train_rows: request.train_rows,
            test_rows: request.test_rows,
            features: request.features,
            execution_host: request.execution_host.clone(),
            node: request.node.clone(),
            scq_job_id: request.scq_job_id.clone(),
            runtime_seconds: 1.0,
            thread_counts: request.thread_counts.clone(),
            predictions: vec![0.0],
            predictions_sha256: sha256_bytes(&prediction_bytes(&[0.0])),
            metrics: Some(metrics.clone()),
            failure: None,
        };
        let outcome = IslandSlotOutcome {
            generation: 20,
            island: 0,
            local_slot: 0,
            global_slot: 0,
            member,
            parents: Vec::new(),
            valid: true,
            metrics: Some(metrics),
            failure: None,
            cache_source: None,
            receipt_file: String::new(),
            receipt_sha256: String::new(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: request.scq_job_id.clone(),
            attempt: 0,
        };
        validate_island_receipt_binding(&identity, &outcome, &receipt, &request).unwrap();
        let mut changed_request = request.clone();
        changed_request.binary_sha256 = "altered".to_string();
        assert!(
            validate_island_receipt_binding(&identity, &outcome, &receipt, &changed_request)
                .is_err()
        );
        let mut changed_receipt = receipt;
        changed_receipt.semantic_sha256 = "altered".to_string();
        assert!(
            validate_island_receipt_binding(&identity, &outcome, &changed_receipt, &request)
                .is_err()
        );
    }

    #[test]
    fn island_lineage_reconstructs_through_generation_19() {
        let combined = synthetic_combined();
        let source = &combined.last().unwrap().0.slots[0];
        let selected = IslandSlotOutcome {
            generation: 20,
            island: 0,
            local_slot: 0,
            global_slot: 0,
            member: {
                let mut member = source.member.clone();
                member.ancestry = Ancestry {
                    operation: "island_elite".to_string(),
                    parents: vec![source.member.semantic_sha256.clone()],
                };
                member
            },
            parents: vec![old_parent_ref(19, source)],
            valid: true,
            metrics: source.metrics.clone(),
            failure: None,
            cache_source: None,
            receipt_file: String::new(),
            receipt_sha256: String::new(),
            node: EXECUTION_NODE.to_string(),
            scq_job_id: String::new(),
            attempt: 0,
        };
        let checkpoint = IslandGenerationCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            run_id: "run".to_string(),
            dataset_hash: "dataset".to_string(),
            generation: 20,
            previous_checkpoint_sha256: "g19".to_string(),
            logical_slots: 1,
            fresh_evaluations: 1,
            cumulative_fresh_evaluations: 1,
            valid_candidates: 1,
            fresh_g0_rmse: 1.0,
            global_best: genome_location(&selected),
            global_best_rmse: selected.rmse(),
            qualifying_descendant_win: false,
            qualifying_genomes: Vec::new(),
            result_label: None,
            migration_file: None,
            migration_sha256: None,
            islands: vec![IslandGenerationState {
                island: 0,
                logical_slots: 1,
                fresh_evaluations: 1,
                valid_candidates: 1,
                best_local_slot: 0,
                best_global_slot: 0,
                best_semantic_sha256: selected.member.semantic_sha256.clone(),
                best_rmse: selected.rmse(),
                ranked_local_slots: vec![0],
                slots: vec![selected.clone()],
                next_population: Vec::new(),
            }],
        };
        let lineage =
            reconstruct_island_lineage(&combined, &[(checkpoint, "g20".to_string())], &selected)
                .unwrap();
        assert!(lineage.iter().any(|record| {
            record.generation == 19
                && record.island.is_none()
                && record.semantic_sha256 == source.member.semantic_sha256
        }));
        assert!(lineage
            .iter()
            .any(|record| { record.generation == 20 && record.island == Some(0) }));
    }

    #[test]
    fn altered_population_migration_and_qualifying_lists_change_bindings() {
        let combined = synthetic_combined();
        let populations = generation_20_populations(&combined, 20).unwrap();
        let original_hash = sha256_bytes(&canonical_json(&populations[0]).unwrap());
        let mut altered = populations[0].clone();
        altered[0].parents[0].semantic_sha256 = "altered".to_string();
        assert_ne!(
            original_hash,
            sha256_bytes(&canonical_json(&altered).unwrap())
        );

        let ranked = (0..ISLANDS)
            .map(|island| {
                (0..ISLAND_POPULATION)
                    .map(|slot| outcome(island, slot, slot as f64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (_, migration) = migrate_rankings("run", "dataset", 21, &ranked).unwrap();
        let original_hash = sha256_bytes(&canonical_json(&migration).unwrap());
        let mut altered_migration = migration;
        altered_migration.transfers[0].destination_island = 3;
        assert_ne!(
            original_hash,
            sha256_bytes(&canonical_json(&altered_migration).unwrap())
        );

        let expected = vec![genome_location(&ranked[0][0])];
        let mut altered_qualifying = expected.clone();
        altered_qualifying.push(genome_location(&ranked[0][1]));
        assert_ne!(expected, altered_qualifying);
    }

    #[test]
    fn prior_no_win_cohort_reconstructs_in_order() {
        assert_eq!(
            prior_no_win_ids("test-adaptive-golden42-v1").unwrap(),
            vec![
                "21fbfe8b53dbcd60",
                "3f781a51af45387e",
                "55f06d147bcc2a32",
                "5f91a00bd3b66742",
                "860e5558f732f638",
                "b1f909fcfceeef4b",
                "b5beff4a53fa12c0",
            ]
        );
    }

    #[test]
    fn champion_complexity_schedule_widens_and_caps_deterministically() {
        assert_eq!(champion_complexity_level(30), 1);
        assert_eq!(champion_complexity_level(39), 1);
        assert_eq!(champion_complexity_level(40), 2);
        assert_eq!(champion_complexity_level(50), 3);
        assert_eq!(champion_complexity_level(60), 4);
        assert_eq!(champion_complexity_level(600), 4);
    }

    #[test]
    fn champion_policy_is_unbounded_and_uses_odd_generation_migration() {
        let config = IslandConfig::champion();
        let migration = MigrationPolicy::champion();
        assert_eq!(config.start_generation, 30);
        assert_eq!(config.generations, 0);
        assert_eq!(config.end_generation_exclusive, 0);
        assert!(migration.after_generations.is_empty());
        assert!(migration.direction.contains("every odd generation"));
    }

    #[test]
    fn sealed_generation_29_artifacts_bind_seven_champion_continuations() {
        let bindings = champion_bindings("test-adaptive-golden42-v1").unwrap();
        assert_eq!(bindings.len(), 7);
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            [
                "21fbfe8b53dbcd60",
                "3f781a51af45387e",
                "55f06d147bcc2a32",
                "5f91a00bd3b66742",
                "860e5558f732f638",
                "b1f909fcfceeef4b",
                "b5beff4a53fa12c0",
            ]
        );
    }

    #[test]
    fn generation_30_populations_reconstruct_deterministically() {
        let registry = paths::jope_registry();
        let (_, specs, _, _) = load_panel_specs(&registry).unwrap();
        let spec = specs
            .iter()
            .find(|spec| spec.dataset_hash == "21fbfe8b53dbcd60")
            .unwrap();
        let initial = load_initial_island_checkpoints("test-adaptive-golden42-v1", spec).unwrap();
        let first = generation_30_populations(&initial, spec.features).unwrap();
        let second = generation_30_populations(&initial, spec.features).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert!(first.iter().all(|island| island.len() == 30));
    }
}
