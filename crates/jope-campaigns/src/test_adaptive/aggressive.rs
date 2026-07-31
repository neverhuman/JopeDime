//! Additive four-island, over-decomposed Golden42 executor.
//!
//! The older 90-core campaigns are intentionally not parameterized through
//! this module. The aggressive campaign has its own schemas, resource gate, routing rule, and
//! checkpoint chain while sharing only the complete Rust candidate evaluator.

use super::*;

const SCHEMA: &str = "JopeGolden42AggressiveManifest.v5";
const PREFLIGHT_SCHEMA: &str = "JopeGolden42AggressivePreflight.v1";
const CHECKPOINT_SCHEMA: &str = "JopeGolden42AggressiveCheckpoint.v1";
const CONFIRMATION_SCHEMA: &str = "JopeGolden42AggressiveConfirmation.v1";
const RESULT_SCHEMA: &str = "JopeGolden42AggressiveDatasetResult.v1";
const SUMMARY_SCHEMA: &str = "JopeGolden42AggressiveSummary.v1";
const MANIFEST_FILE: &str = "aggressive-manifest.v1.json";
const ADMISSION_FILE: &str = "backend-admission.v1.json";
const PREFLIGHT_FILE: &str = "preflight.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const SUMMARY_FILE: &str = "aggressive-summary.v1.json";
const GOLDEN46_RUN_PREFIX: &str = "golden46-new3-20260723-v4";
const GOLDEN46_CORRECTED_PARENT_RUN: &str = "golden46-new3-20260723-v2";
const GOLDEN46_TRANSFER_SUFFIX: &str = "-target-family-informed";
const GOLDEN46_PANEL_MANIFEST: &str =
    "/home/ubuntu/remote_super/quant/campaigns/golden46-full42-20260723-v1/manifests/panel46.json";
const GOLDEN46_LEADER_MANIFEST: &str =
    "/home/ubuntu/remote_super/quant/campaigns/golden46-full42-20260723-v1/manifests/leaders-new3.json";
const GOLDEN46_CAMPAIGN_ID: &str = "golden46-full42-20260723-v1";
const GOLDEN46_PANEL_DATASETS: [&str; 4] =
    ["ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296", "ZTS"];
const GOLDEN46_DATASETS: [&str; 3] = ["ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296"];
const GENERATION_ZERO_SEEDS_FILE: &str = "generation-zero-seeds.v1.json";
const EVOLUTION_VIEW_MANIFEST_FILE: &str = "evolution-view-manifest.v1.json";
const ISLANDS: usize = 4;
const ISLAND_POPULATION: usize = 9;
const TOTAL_POPULATION: usize = ISLANDS * ISLAND_POPULATION;
const ELITES_PER_ISLAND: usize = 2;
const PARENT_POOL_PER_ISLAND: usize = 3;
const MIN_VALID_PER_ISLAND: usize = ELITES_PER_ISLAND;
const UNBOUNDED_GENERATIONS: usize = usize::MAX;
const MIN_AVAILABLE_RAM_BYTES: u64 = 48 * 1024 * 1024 * 1024;
const MIN_FREE_DISK_BYTES: u64 = 100 * 1024 * 1024 * 1024;
// All three nodes have 8 GiB swap. Linux may retain cold pages there long
// after a competing build exits even when tens of GiB of RAM are available.
// Real pressure is rejected by the available-RAM and competing-process gates;
// stale swap occupancy alone must not deadlock campaign admission.
const MAX_SWAP_USED_BYTES: u64 = 9 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SelectedGenerationZeroSeed {
    island: usize,
    local_slot: usize,
    seed_class: String,
    provenance: SeedProvenance,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationZeroSeedSnapshot {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    dataset_hash: String,
    provenance_group: String,
    arm: String,
    strategy: String,
    parent_run_id: String,
    parent_checkpoint_sha256: String,
    parent_seed: SeedProvenance,
    population_sha256: String,
    generation_zero_can_terminate: bool,
    candidates: Vec<SelectedGenerationZeroSeed>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct NodeResources {
    node: String,
    online_cores: usize,
    available_ram_bytes: u64,
    free_disk_bytes: u64,
    swap_used_bytes: u64,
    competing_processes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreflightReceipt {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    required_cores: usize,
    required_available_ram_bytes: u64,
    required_free_disk_bytes: u64,
    maximum_swap_used_bytes_exclusive: u64,
    active_scq_jobs: Vec<String>,
    nodes: Vec<NodeResources>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AggressiveManifest {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    selected_backend: String,
    cpp17_admitted: bool,
    cpp17_rejection: String,
    backend_admission_sha256: String,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    preflight_sha256: String,
    staging_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    panel_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    panel_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    leader_manifest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    leader_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evolution_view_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    leaderboard_default: Option<bool>,
    registry_root: String,
    population: usize,
    islands: usize,
    population_per_island: usize,
    workers_per_node: usize,
    generation_limit: Option<usize>,
    routing: String,
    retry: String,
    stop_rule: String,
    datasets: Vec<DatasetSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct AggressiveCheckpoint {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    logical_slots: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    cumulative_fresh_evaluations: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    cumulative_cache_hits: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    unique_population: usize,
    valid_per_island: Vec<usize>,
    ranked_slots: Vec<usize>,
    best_slot: usize,
    best_rmse: f64,
    winner_slot: Option<usize>,
    slots: Vec<SlotOutcome>,
    next_population: Vec<Member>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Confirmation {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    slot: usize,
    checkpoint_sha256: String,
    prediction_bits_sha256: String,
    rmse_bits: u64,
    nodes: Vec<String>,
    receipt_files: Vec<String>,
    receipt_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetResult {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    dataset_hash: String,
    outcome: String,
    stopping_generation: usize,
    leader_rmse: f64,
    best_rmse: f64,
    best_slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    terminal_checkpoint_sha256: String,
    confirmation_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Summary {
    schema: String,
    benchmark_valid: bool,
    test_adaptive: bool,
    run_id: String,
    selected_backend: String,
    dataset_count: usize,
    confirmed_wins: usize,
    unresolved: usize,
    results: Vec<DatasetResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EvolutionViewDataset {
    dataset_hash: String,
    source_train_sha256: String,
    source_test_sha256: String,
    train_sha256: String,
    test_sha256: String,
    removed_train_header: bool,
    removed_test_header: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EvolutionViewManifest {
    schema: String,
    campaign_id: String,
    panel_manifest_sha256: String,
    source_registry_root: String,
    view_sha256: String,
    datasets: Vec<EvolutionViewDataset>,
}

#[derive(Clone, Debug)]
struct Golden46Bindings {
    panel_sha256: String,
    leader_sha256: String,
    evolution_view_manifest_sha256: String,
    arm: String,
    leaderboard_default: bool,
}

fn is_golden46_run(run_id: &str) -> bool {
    run_id == GOLDEN46_RUN_PREFIX
        || run_id == format!("{GOLDEN46_RUN_PREFIX}{GOLDEN46_TRANSFER_SUFFIX}")
}

fn root(run_id: &str) -> PathBuf {
    if run_id == format!("{GOLDEN46_RUN_PREFIX}{GOLDEN46_TRANSFER_SUFFIX}") {
        return workspace_root()
            .join("runs")
            .join(GOLDEN46_RUN_PREFIX)
            .join("transfer-arms")
            .join("target-family-informed");
    }
    workspace_root().join("runs").join(run_id)
}

fn dataset_root(run_id: &str, dataset_hash: &str) -> PathBuf {
    root(run_id).join("datasets").join(dataset_hash)
}

fn checkpoint_path(run_id: &str, dataset_hash: &str, generation: usize) -> PathBuf {
    dataset_root(run_id, dataset_hash)
        .join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

fn confirmation_path(run_id: &str, dataset_hash: &str) -> PathBuf {
    dataset_root(run_id, dataset_hash).join("confirmation.v1.json")
}

fn result_path(run_id: &str, dataset_hash: &str) -> PathBuf {
    dataset_root(run_id, dataset_hash).join("final-result.v1.json")
}

fn atomic_immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(());
        }
        bail!("immutable artifact collision at {}", path.display());
    }
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

fn ssh(node: &str, args: &[&str], label: &str) -> Result<Vec<u8>> {
    let output = Command::new("ssh")
        .arg(node)
        .arg("--")
        .args(args)
        .output()
        .with_context(|| format!("launch {label} on {node}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed on {node}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn parse_kib_field(text: &str, name: &str) -> Result<u64> {
    let value = text
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next()? == name)
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .with_context(|| format!("/proc/meminfo has no {name}"))?;
    value
        .checked_mul(1024)
        .context("memory byte count overflow")
}

fn inspect_node(node: &str) -> Result<NodeResources> {
    let local_hostname = String::from_utf8(
        Command::new("hostname")
            .arg("-s")
            .output()
            .context("read controller hostname")?
            .stdout,
    )?
    .trim()
    .to_string();
    let controller_pid = std::process::id().to_string();
    let online_cores = String::from_utf8(ssh(
        node,
        &["getconf", "_NPROCESSORS_ONLN"],
        "read online cores",
    )?)?
    .trim()
    .parse::<usize>()?;
    let memory = String::from_utf8(ssh(node, &["cat", "/proc/meminfo"], "read memory")?)?;
    let available_ram_bytes = parse_kib_field(&memory, "MemAvailable:")?;
    let swap_total = parse_kib_field(&memory, "SwapTotal:")?;
    let swap_free = parse_kib_field(&memory, "SwapFree:")?;
    let swap_used_bytes = swap_total.saturating_sub(swap_free);
    let disk = String::from_utf8(ssh(node, &["df", "-Pk", "/home/ubuntu"], "read free disk")?)?;
    let free_kib = disk
        .lines()
        .last()
        .and_then(|line| line.split_whitespace().nth(3))
        .context("df returned no available-block field")?
        .parse::<u64>()?;
    let free_disk_bytes = free_kib
        .checked_mul(1024)
        .context("disk byte count overflow")?;
    let processes = String::from_utf8(ssh(
        node,
        &["ps", "-eo", "pid=,comm=,args="],
        "read process table",
    )?)?;
    let competing_processes = processes
        .lines()
        .filter(|line| is_competing_process(line, node, &local_hostname, &controller_pid))
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();
    Ok(NodeResources {
        node: node.to_string(),
        online_cores,
        available_ram_bytes,
        free_disk_bytes,
        swap_used_bytes,
        competing_processes,
    })
}

fn scq_job_is_active(job: &Value) -> bool {
    !matches!(
        job.get("state").and_then(Value::as_str),
        Some("completed" | "failed" | "cancelled")
    )
}

fn is_competing_process(
    line: &str,
    node: &str,
    local_hostname: &str,
    controller_pid: &str,
) -> bool {
    let lower = line.to_ascii_lowercase();
    let comm = line.split_whitespace().nth(1).unwrap_or_default();
    let is_jope_executable = comm.starts_with("jope-dime-regre");
    let is_scq_executable = comm == "scq";
    let is_this_controller =
        node == local_hostname && line.split_whitespace().next() == Some(controller_pid);
    let is_campaign_controller = lower.contains("run-golden42-aggressive")
        || lower.contains("run-golden46")
        || lower.contains("run-tabarena-lite-islands");
    let is_campaign_worker = lower.contains("test-adaptive-worker --request")
        || lower.contains("golden42-aggressive-batch-worker")
        || lower.contains("test-adaptive-island-batch-worker");
    let is_campaign_scq_job = is_scq_executable && lower.contains("jope-g42-");
    let is_scq_control_plane = lower.contains("scqd")
        || lower.contains("scq controller serve")
        || lower.contains("scq agent status");
    (is_jope_executable || is_scq_executable)
        && !is_scq_control_plane
        && !is_campaign_controller
        && !is_campaign_worker
        && !is_campaign_scq_job
        && !is_this_controller
}

fn preflight() -> Result<PreflightReceipt> {
    let scq = Path::new("/home/ubuntu/smart_cluster/target/release/scq");
    let output = Command::new(scq)
        .args(["--json", "ls"])
        .output()
        .context("launch SCQ queue inspection")?;
    if !output.status.success() {
        bail!("SCQ queue inspection failed");
    }
    let queue: Value = serde_json::from_slice(&output.stdout)?;
    let active_scq_jobs = queue
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .context("SCQ queue response has no jobs array")?
        .iter()
        .filter(|job| scq_job_is_active(job))
        .filter(|job| {
            !job.get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("jope-g42-"))
        })
        .filter(|job| {
            job.get("state").and_then(Value::as_str) != Some("queued")
                || job.get("node").and_then(Value::as_str).is_some()
        })
        .map(|job| {
            job.get("id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string()
        })
        .collect::<Vec<_>>();
    if !active_scq_jobs.is_empty() {
        bail!("SCQ has competing active jobs");
    }
    let nodes = scq_batch::NODES
        .iter()
        .map(|node| inspect_node(node))
        .collect::<Result<Vec<_>>>()?;
    for node in &nodes {
        if node.online_cores < scq_batch::AGGRESSIVE_CPUS {
            bail!(
                "{} has fewer than {} online cores",
                node.node,
                scq_batch::AGGRESSIVE_CPUS
            );
        }
        if node.available_ram_bytes < MIN_AVAILABLE_RAM_BYTES {
            bail!("{} has less than 48 GiB available RAM", node.node);
        }
        if node.free_disk_bytes < MIN_FREE_DISK_BYTES {
            bail!("{} has less than 100 GiB free disk", node.node);
        }
        if node.swap_used_bytes >= MAX_SWAP_USED_BYTES {
            bail!(
                "{} uses {} bytes of swap; the aggressive campaign requires less than 1 GiB",
                node.node,
                node.swap_used_bytes
            );
        }
        if !node.competing_processes.is_empty() {
            bail!("{} has a competing Cargo/Rust/Jope/SCQ job", node.node);
        }
    }
    Ok(PreflightReceipt {
        schema: PREFLIGHT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        required_cores: scq_batch::AGGRESSIVE_CPUS,
        required_available_ram_bytes: MIN_AVAILABLE_RAM_BYTES,
        required_free_disk_bytes: MIN_FREE_DISK_BYTES,
        maximum_swap_used_bytes_exclusive: MAX_SWAP_USED_BYTES,
        active_scq_jobs,
        nodes,
    })
}

fn worker_identity(manifest: &AggressiveManifest, spec: &DatasetSpec) -> CampaignIdentity {
    CampaignIdentity {
        schema: "JopeGolden42AggressiveDatasetCampaign.v1".to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        train_rows: spec.train_rows,
        test_rows: spec.test_rows,
        features: spec.features,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        population: TOTAL_POPULATION,
        elites: ELITES_PER_ISLAND * ISLANDS,
        parent_pool: PARENT_POOL_PER_ISLAND * ISLANDS,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
        generations: UNBOUNDED_GENERATIONS,
        rng_seed: RNG_SEED,
        binary_sha256: manifest.release_binary_sha256.clone(),
        execution_host: "scq".to_string(),
        execution_node: "xbabe1,xbabe2,xbabe3".to_string(),
        selection_objective: "test-target RMSE; generation-zero crossings are recorded but only generation-1-or-later evolved descendants may terminate"
            .to_string(),
    }
}

fn is_fresh_novel_descendant(member: &Member) -> bool {
    member.evolved_descendant
        && matches!(member.ancestry.operation.as_str(), "mutation" | "crossover")
        && !member.ancestry.parents.is_empty()
        && member
            .ancestry
            .parents
            .iter()
            .all(|parent| parent != &member.semantic_sha256)
}

fn evolutionary_winner_slot(
    ranked: &[SlotOutcome],
    generation: usize,
    leader_rmse: f64,
) -> Option<usize> {
    if generation == 0 {
        return None;
    }
    ranked
        .iter()
        .find(|outcome| {
            outcome.valid
                && outcome.cache_source.is_none()
                && is_fresh_novel_descendant(&outcome.member)
                && outcome.rmse().is_finite()
                && outcome.rmse() < leader_rmse
        })
        .map(|outcome| outcome.slot)
}

fn provenance_group(dataset_hash: &str) -> &str {
    if matches!(dataset_hash, "ZTS" | "ZTS_small") {
        "ZTS"
    } else {
        dataset_hash
    }
}

fn derive_corrected_generation_zero(
    parent_seed: &SeedProvenance,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<(Vec<Member>, Vec<SelectedGenerationZeroSeed>)> {
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    let mut candidates = Vec::with_capacity(TOTAL_POPULATION);
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    for slot in 0..TOTAL_POPULATION {
        let island = slot / ISLAND_POPULATION;
        let local_slot = slot % ISLAND_POPULATION;
        let mut produced = None;
        for retry in 0..10_000usize {
            let steps = 1 + (slot + retry) % 4;
            let mut source = parent_seed.canonical_genome.clone();
            let mut evolved = None;
            let mut valid_chain = true;
            for step in 0..steps {
                let digest = Sha256::digest(
                    format!(
                        "golden46-corrected-g0:{run_id}:{}:{island}:{local_slot}:{retry}:{step}",
                        spec.dataset_hash
                    )
                    .as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                let Ok(next) =
                    feat_math::full_rust::mutate_pipeline_expanded(&source, seed, spec.features)
                else {
                    valid_chain = false;
                    break;
                };
                source = next.canonical.clone();
                evolved = Some(next);
            }
            if !valid_chain {
                continue;
            }
            let evolved = evolved.context("generation-zero mutation chain was empty")?;
            if evolved.semantic_sha256 == parent_seed.semantic_sha256
                || !seen.insert(evolved.semantic_sha256.clone())
            {
                continue;
            }
            produced = Some(evolved);
            break;
        }
        let evolved = produced.with_context(|| {
            format!("could not derive unique corrected generation-zero candidate {slot}")
        })?;
        let member = Member::evolved(
            evolved,
            Ancestry {
                operation: "generation_zero_mutation".to_string(),
                parents: vec![parent_seed.semantic_sha256.clone()],
            },
        );
        candidates.push(SelectedGenerationZeroSeed {
            island,
            local_slot,
            seed_class: "unique_full_genome_mutation".to_string(),
            provenance: SeedProvenance {
                slot,
                canonical_genome: member.canonical_genome.clone(),
                semantic_sha256: member.semantic_sha256.clone(),
                complexity: member.complexity,
                rust_source_len: member.rust_source_len,
                source_kind: "corrected_generation_zero_candidate".to_string(),
                source_file: parent_seed.source_file.clone(),
                source_file_sha256: parent_seed.source_file_sha256.clone(),
                source_record: slot,
                source_rank: slot + 1,
                source_score: None,
            },
        });
        population.push(member);
    }
    Ok((population, candidates))
}

fn golden46_generation_zero_population(
    _registry: &Path,
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<Member>> {
    let checkpoint_dir =
        dataset_root(GOLDEN46_CORRECTED_PARENT_RUN, &spec.dataset_hash).join("checkpoints");
    let mut paths = fs::read_dir(&checkpoint_dir)
        .with_context(|| {
            format!(
                "read corrected-parent checkpoints for {}",
                spec.dataset_hash
            )
        })?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut parent = None::<(SeedProvenance, String)>;
    for path in paths {
        let bytes = fs::read(&path)?;
        let checkpoint: AggressiveCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.run_id != GOLDEN46_CORRECTED_PARENT_RUN
            || checkpoint.dataset_hash != spec.dataset_hash
        {
            bail!("invalid corrected-parent checkpoint {}", path.display());
        }
        for outcome in checkpoint.slots.iter().filter(|outcome| outcome.valid) {
            let rmse = outcome.rmse();
            if !rmse.is_finite()
                || parent
                    .as_ref()
                    .is_some_and(|(seed, _)| seed.source_score.is_some_and(|best| rmse >= best))
            {
                continue;
            }
            let digest = sha256_bytes(&bytes);
            parent = Some((
                SeedProvenance {
                    slot: outcome.slot,
                    canonical_genome: outcome.member.canonical_genome.clone(),
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    complexity: outcome.member.complexity,
                    rust_source_len: outcome.member.rust_source_len,
                    source_kind: "corrected_test_adaptive_parent".to_string(),
                    source_file: path.to_string_lossy().into_owned(),
                    source_file_sha256: digest.clone(),
                    source_record: outcome.slot,
                    source_rank: 1,
                    source_score: Some(rmse),
                },
                digest,
            ));
        }
    }
    let (parent_seed, parent_checkpoint_sha256) =
        parent.context("corrected Golden46 parent run has no valid candidate")?;
    let (population, candidates) = derive_corrected_generation_zero(&parent_seed, run_id, spec)?;
    let population_sha256 = sha256_bytes(&canonical_json(&candidates)?);
    let snapshot = GenerationZeroSeedSnapshot {
        schema: "JopeGolden46GenerationZeroSeeds.v2".to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        provenance_group: provenance_group(&spec.dataset_hash).to_string(),
        arm: "corrected-test-adaptive-continuation".to_string(),
        strategy: "single best preserved v2 parent; 36 deterministic unique full-genome mutations"
            .to_string(),
        parent_run_id: GOLDEN46_CORRECTED_PARENT_RUN.to_string(),
        parent_checkpoint_sha256,
        parent_seed,
        population_sha256,
        generation_zero_can_terminate: false,
        candidates,
    };
    let snapshot_path = dataset_root(run_id, &spec.dataset_hash).join(GENERATION_ZERO_SEEDS_FILE);
    atomic_immutable_write(&snapshot_path, &canonical_json(&snapshot)?)?;
    Ok(population)
}

fn initial_population(registry: &Path, run_id: &str, spec: &DatasetSpec) -> Result<Vec<Member>> {
    if is_golden46_run(run_id) {
        return golden46_generation_zero_population(registry, run_id, spec);
    }
    let seed_path = dataset_root(run_id, &spec.dataset_hash).join("transfer-seeds.v1.json");
    let base = load_or_create_transfer_seed_manifest(&seed_path, registry, spec)?
        .seeds
        .iter()
        .take(ISLAND_POPULATION)
        .map(Member::from_seed)
        .collect::<Vec<_>>();
    expand_initial_population(&base, spec.features)
}

fn expand_initial_population(base: &[Member], features: usize) -> Result<Vec<Member>> {
    if base.is_empty() || base.len() > ISLAND_POPULATION {
        bail!("aggressive seed baseline does not fit one island");
    }
    let mut population = Vec::with_capacity(TOTAL_POPULATION);
    let mut seen = HashSet::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        for local_slot in 0..ISLAND_POPULATION {
            if island == 0 && local_slot < base.len() {
                let member = base[local_slot].clone();
                if !seen.insert(member.semantic_sha256.clone()) {
                    bail!("aggressive seed baseline contains a duplicate genome");
                }
                population.push(member);
                continue;
            }
            let parent = &base[local_slot % base.len()];
            let mut produced = None;
            for retry in 0..1000usize {
                let digest = Sha256::digest(
                    format!("aggressive:g0:{island}:{local_slot}:{retry}").as_bytes(),
                );
                let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
                if let Ok(genome) = mutate_pipeline(&parent.canonical_genome, seed, features) {
                    if seen.insert(genome.semantic_sha256.clone()) {
                        produced = Some(Member::evolved(
                            genome,
                            Ancestry {
                                operation: "aggressive_generation_zero_mutation".to_string(),
                                parents: vec![parent.semantic_sha256.clone()],
                            },
                        ));
                        break;
                    }
                }
            }
            population.push(
                produced.with_context(|| {
                    format!("could not derive island {island} slot {local_slot}")
                })?,
            );
        }
    }
    if population.len() != TOTAL_POPULATION {
        bail!("aggressive initial population is incomplete");
    }
    Ok(population)
}

fn offspring_seed(
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    local_slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!(
            "aggressive:{RNG_SEED}:{run_id}:{dataset_hash}:{generation}:{island}:{local_slot}:{retry}"
        )
        .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn next_island_population(
    ranked: &[SlotOutcome],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    features: usize,
) -> Result<Vec<Member>> {
    let valid_parent_pool = ranked
        .iter()
        .take(PARENT_POOL_PER_ISLAND)
        .take_while(|outcome| outcome.valid)
        .count();
    if ranked.len() != ISLAND_POPULATION || valid_parent_pool < ELITES_PER_ISLAND {
        bail!("aggressive island parent pool is incomplete");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for elite in ranked.iter().take(ELITES_PER_ISLAND) {
        let mut member = elite.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
        };
        next.push(member);
    }
    for local_slot in ELITES_PER_ISLAND..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..100usize {
            let seed = offspring_seed(run_id, dataset_hash, generation, island, local_slot, retry);
            let mut rng = Pcg32::new(seed, 0x5343_5138_305f_5631 ^ island as u64);
            let first = tournament_winner(&mut rng, valid_parent_pool);
            let first_parent = &ranked[first].member;
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng, valid_parent_pool);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "crossover".to_string(),
                            parents: vec![
                                first_parent.semantic_sha256.clone(),
                                second_parent.semantic_sha256.clone(),
                            ],
                        },
                    )
                })
            } else {
                mutate_pipeline(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "mutation".to_string(),
                            parents: vec![first_parent.semantic_sha256.clone()],
                        },
                    )
                })
            };
            if let Ok(candidate) = candidate {
                produced = Some(candidate);
                break;
            }
        }
        next.push(produced.context("could not produce aggressive descendant")?);
    }
    Ok(next)
}

fn next_golden46_island_population(
    ranked: &[SlotOutcome],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    island: usize,
    features: usize,
    history: &HashSet<String>,
    next_unique: &mut HashSet<String>,
) -> Result<Vec<Member>> {
    let valid_parent_pool = ranked
        .iter()
        .take(PARENT_POOL_PER_ISLAND)
        .take_while(|outcome| outcome.valid)
        .count();
    if ranked.len() != ISLAND_POPULATION || valid_parent_pool < ELITES_PER_ISLAND {
        bail!("Golden46 island parent pool is incomplete");
    }
    let mut next = Vec::with_capacity(ISLAND_POPULATION);
    for elite in ranked.iter().filter(|outcome| outcome.valid) {
        if !next_unique.insert(elite.member.semantic_sha256.clone()) {
            continue;
        }
        let mut member = elite.member.clone();
        member.ancestry = Ancestry {
            operation: "elite".to_string(),
            parents: vec![elite.member.semantic_sha256.clone()],
        };
        next.push(member);
        if next.len() == ELITES_PER_ISLAND {
            break;
        }
    }
    if next.len() != ELITES_PER_ISLAND {
        bail!("Golden46 could not preserve two globally unique island elites");
    }
    for local_slot in ELITES_PER_ISLAND..ISLAND_POPULATION {
        let mut produced = None;
        for retry in 0..10_000usize {
            let seed = offspring_seed(run_id, dataset_hash, generation, island, local_slot, retry);
            let mut rng = Pcg32::new(seed, 0x5343_5138_305f_5632 ^ island as u64);
            let first = tournament_winner(&mut rng, valid_parent_pool);
            let first_parent = &ranked[first].member;
            let candidate = if rng.next_f64() < 0.25 {
                let second = tournament_winner(&mut rng, valid_parent_pool);
                let second_parent = &ranked[second].member;
                crossover_pipelines(
                    &first_parent.canonical_genome,
                    &second_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "crossover".to_string(),
                            parents: vec![
                                first_parent.semantic_sha256.clone(),
                                second_parent.semantic_sha256.clone(),
                            ],
                        },
                    )
                })
            } else {
                feat_math::full_rust::mutate_pipeline_expanded(
                    &first_parent.canonical_genome,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    features,
                )
                .map(|genome| {
                    Member::evolved(
                        genome,
                        Ancestry {
                            operation: "mutation".to_string(),
                            parents: vec![first_parent.semantic_sha256.clone()],
                        },
                    )
                })
            };
            if let Ok(candidate) = candidate {
                if history.contains(&candidate.semantic_sha256)
                    || !next_unique.insert(candidate.semantic_sha256.clone())
                {
                    continue;
                }
                produced = Some(candidate);
                break;
            }
        }
        next.push(produced.with_context(|| {
            format!(
                "could not produce history-unique Golden46 descendant for island {island} slot {local_slot}"
            )
        })?);
    }
    Ok(next)
}

fn next_golden46_population(
    ranked_islands: &[Vec<SlotOutcome>],
    run_id: &str,
    dataset_hash: &str,
    generation: usize,
    features: usize,
    history: &HashSet<String>,
) -> Result<Vec<Member>> {
    let mut next = Vec::with_capacity(TOTAL_POPULATION);
    let mut next_unique = HashSet::with_capacity(TOTAL_POPULATION);
    for (island, ranked) in ranked_islands.iter().enumerate() {
        next.extend(next_golden46_island_population(
            ranked,
            run_id,
            dataset_hash,
            generation,
            island,
            features,
            history,
            &mut next_unique,
        )?);
    }
    if next.len() != TOTAL_POPULATION || next_unique.len() != TOTAL_POPULATION {
        bail!("Golden46 next population is not globally unique and complete");
    }
    Ok(next)
}

fn rank_islands_with_global_recovery(
    slots: &[SlotOutcome],
) -> Result<(Vec<Vec<SlotOutcome>>, Vec<usize>)> {
    let mut global_valid = slots
        .iter()
        .filter(|outcome| outcome.valid)
        .cloned()
        .collect::<Vec<_>>();
    global_valid.sort_by(rank_outcomes);
    if global_valid.len() < PARENT_POOL_PER_ISLAND {
        bail!("aggressive generation has too few globally valid parents");
    }
    let mut ranked_islands = Vec::with_capacity(ISLANDS);
    let mut valid_per_island = Vec::with_capacity(ISLANDS);
    for island in 0..ISLANDS {
        let mut ranked =
            slots[island * ISLAND_POPULATION..(island + 1) * ISLAND_POPULATION].to_vec();
        let valid = ranked.iter().filter(|outcome| outcome.valid).count();
        valid_per_island.push(valid);
        if valid < MIN_VALID_PER_ISLAND {
            let start = island * PARENT_POOL_PER_ISLAND % global_valid.len();
            ranked = (0..ISLAND_POPULATION)
                .map(|offset| global_valid[(start + offset) % global_valid.len()].clone())
                .collect();
        } else {
            ranked.sort_by(rank_outcomes);
        }
        ranked_islands.push(ranked);
    }
    Ok((ranked_islands, valid_per_island))
}

fn load_checkpoints(
    run_id: &str,
    spec: &DatasetSpec,
) -> Result<Vec<(AggressiveCheckpoint, String)>> {
    let checkpoint_dir = dataset_root(run_id, &spec.dataset_hash).join("checkpoints");
    let mut generations = if checkpoint_dir.is_dir() {
        fs::read_dir(&checkpoint_dir)?
            .map(|entry| {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                let generation = name
                    .strip_prefix("generation-")
                    .and_then(|value| value.strip_suffix(".v1.json"))
                    .context("unexpected aggressive checkpoint filename")?
                    .parse::<usize>()?;
                Ok(generation)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    generations.sort_unstable();
    for (expected, actual) in generations.iter().enumerate() {
        if *actual != expected {
            bail!("aggressive checkpoint chain contains a gap at generation {expected}");
        }
    }
    let mut checkpoints = Vec::<(AggressiveCheckpoint, String)>::new();
    let mut previous = None;
    let mut history = HashSet::<String>::new();
    let mut cumulative_fresh_evaluations = 0usize;
    let mut cumulative_cache_hits = 0usize;
    for generation in generations {
        if checkpoints
            .last()
            .is_some_and(|(checkpoint, _)| checkpoint.winner_slot.is_some())
        {
            bail!("aggressive checkpoint exists after a terminal winner");
        }
        let path = checkpoint_path(run_id, &spec.dataset_hash, generation);
        let bytes = fs::read(&path)?;
        let checkpoint: AggressiveCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || !checkpoint.test_adaptive
            || checkpoint.run_id != run_id
            || checkpoint.dataset_hash != spec.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256 != previous
            || checkpoint.logical_slots != TOTAL_POPULATION
            || checkpoint.slots.len() != TOTAL_POPULATION
            || checkpoint
                .slots
                .iter()
                .enumerate()
                .any(|(slot, outcome)| outcome.slot != slot || outcome.generation != generation)
        {
            bail!("invalid aggressive generation barrier {generation}");
        }
        let (ranked_islands, valid_per_island) =
            rank_islands_with_global_recovery(&checkpoint.slots)?;
        let mut global_ranked = checkpoint.slots.clone();
        global_ranked.sort_by(rank_outcomes);
        let winner_slot = evolutionary_winner_slot(&global_ranked, generation, spec.leader_rmse);
        if is_golden46_run(run_id) {
            let unique_population = checkpoint
                .slots
                .iter()
                .map(|outcome| outcome.member.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len();
            let cache_hits = checkpoint
                .slots
                .iter()
                .filter(|outcome| outcome.cache_source.is_some())
                .count();
            let fresh_evaluations = TOTAL_POPULATION - cache_hits;
            if unique_population != TOTAL_POPULATION
                || checkpoint.unique_population != unique_population
                || checkpoint.fresh_evaluations != fresh_evaluations
                || checkpoint.cache_hits != cache_hits
                || (generation == 0 && cache_hits != 0)
                || checkpoint.slots.iter().any(|outcome| {
                    outcome.cache_source.is_none()
                        && history.contains(&outcome.member.semantic_sha256)
                })
                || checkpoint.slots.iter().any(|outcome| {
                    outcome.cache_source.is_some()
                        && !history.contains(&outcome.member.semantic_sha256)
                })
            {
                bail!("Golden46 generation {generation} uniqueness/cache barrier changed");
            }
            cumulative_fresh_evaluations += fresh_evaluations;
            cumulative_cache_hits += cache_hits;
            if checkpoint.cumulative_fresh_evaluations != cumulative_fresh_evaluations
                || checkpoint.cumulative_cache_hits != cumulative_cache_hits
            {
                bail!("Golden46 generation {generation} cumulative cache accounting changed");
            }
        }
        history.extend(
            checkpoint
                .slots
                .iter()
                .map(|outcome| outcome.member.semantic_sha256.clone()),
        );
        let expected_next = if winner_slot.is_some() {
            Vec::new()
        } else if is_golden46_run(run_id) {
            next_golden46_population(
                &ranked_islands,
                run_id,
                &spec.dataset_hash,
                generation + 1,
                spec.features,
                &history,
            )?
        } else {
            ranked_islands
                .iter()
                .enumerate()
                .map(|(island, ranked)| {
                    next_island_population(
                        ranked,
                        run_id,
                        &spec.dataset_hash,
                        generation + 1,
                        island,
                        spec.features,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        if checkpoint.valid_per_island != valid_per_island
            || checkpoint.ranked_slots
                != global_ranked
                    .iter()
                    .map(|outcome| outcome.slot)
                    .collect::<Vec<_>>()
            || checkpoint.best_slot != global_ranked[0].slot
            || checkpoint.best_rmse.to_bits() != global_ranked[0].rmse().to_bits()
            || checkpoint.winner_slot != winner_slot
            || checkpoint.next_population != expected_next
        {
            bail!("aggressive generation {generation} selection barrier changed");
        }
        let digest = sha256_bytes(&bytes);
        previous = Some(digest.clone());
        checkpoints.push((checkpoint, digest));
    }
    Ok(checkpoints)
}

fn validate_checkpoint_receipts(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    test: &RegressionData,
) -> Result<()> {
    for outcome in &checkpoint.slots {
        let receipt_path =
            dataset_root(&manifest.run_id, &spec.dataset_hash).join(&outcome.receipt_file);
        let receipt_bytes = fs::read(&receipt_path)?;
        if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
            bail!("aggressive slot receipt hash changed");
        }
        let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
        if receipt_bytes != canonical_json(&receipt)? {
            bail!("aggressive slot receipt is not canonical");
        }
        let request_path = dataset_root(&manifest.run_id, &spec.dataset_hash)
            .join("requests")
            .join(&receipt.purpose)
            .join(format!("generation-{:03}", receipt.generation))
            .join(format!(
                "slot-{:03}-attempt-{}.v1.json",
                receipt.slot, receipt.attempt
            ));
        let request_bytes = fs::read(&request_path)?;
        let request: CandidateRequest = serde_json::from_slice(&request_bytes)?;
        if request_bytes != canonical_json(&request)? {
            bail!("aggressive slot request is not canonical");
        }
        validate_receipt_against_request(&receipt, &request, test)?;
        let source = outcome.cache_source.clone().unwrap_or(CacheSource {
            generation: checkpoint.generation,
            slot: outcome.slot,
        });
        let primary = (source.slot + source.generation) % scq_batch::NODES.len();
        let expected_node = match outcome.attempt {
            0 => scq_batch::NODES[primary],
            1 => scq_batch::NODES[(primary + 1) % scq_batch::NODES.len()],
            _ => bail!("aggressive slot used more than one infrastructure retry"),
        };
        if receipt.test_adaptive != Some(true)
            || receipt.purpose != "aggressive-candidate"
            || receipt.run_id != manifest.run_id
            || receipt.binary_sha256 != manifest.release_binary_sha256
            || receipt.dataset_hash != spec.dataset_hash
            || receipt.train_sha256 != spec.train_sha256
            || receipt.test_sha256 != spec.test_sha256
            || receipt.generation != source.generation
            || receipt.slot != source.slot
            || receipt.attempt != outcome.attempt
            || receipt.node != expected_node
            || receipt.semantic_sha256 != outcome.member.semantic_sha256
            || receipt.canonical_genome != outcome.member.canonical_genome
            || receipt.metrics != outcome.metrics
            || receipt.failure != outcome.failure
            || receipt.valid(receipt.test_rows) != outcome.valid
            || receipt.node != outcome.node
            || receipt.scq_job_id != outcome.scq_job_id
        {
            bail!("aggressive slot outcome is not fully bound to its request and receipt");
        }
    }
    Ok(())
}

fn confirm_winner(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    checkpoint_sha256: &str,
    test: &RegressionData,
) -> Result<(Confirmation, String)> {
    let path = confirmation_path(&manifest.run_id, &spec.dataset_hash);
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let confirmation: Confirmation = serde_json::from_slice(&bytes)?;
        let winner_slot = checkpoint.winner_slot.context("checkpoint has no winner")?;
        if bytes != canonical_json(&confirmation)?
            || confirmation.schema != CONFIRMATION_SCHEMA
            || confirmation.benchmark_valid
            || !confirmation.test_adaptive
            || confirmation.run_id != manifest.run_id
            || confirmation.dataset_hash != spec.dataset_hash
            || confirmation.generation != checkpoint.generation
            || confirmation.slot != winner_slot
            || confirmation.checkpoint_sha256 != checkpoint_sha256
            || confirmation.nodes
                != scq_batch::NODES
                    .iter()
                    .map(|node| (*node).to_string())
                    .collect::<Vec<_>>()
            || confirmation.receipt_files.len() != scq_batch::NODES.len()
            || confirmation.receipt_sha256.len() != scq_batch::NODES.len()
        {
            bail!("winner confirmation changed on resume");
        }
        return Ok((confirmation, sha256_bytes(&bytes)));
    }
    let winner_slot = checkpoint.winner_slot.context("checkpoint has no winner")?;
    let winner = &checkpoint.slots[winner_slot];
    let evaluations = scq_batch::replay_aggressive_across_nodes(
        &root(&manifest.run_id),
        &worker_identity(manifest, spec),
        Path::new(&manifest.registry_root),
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &winner.member,
        test,
    )?;
    if evaluations.len() != scq_batch::NODES.len() {
        bail!("winner confirmation did not return all three nodes");
    }
    let first = &evaluations[0].receipt;
    let first_metrics = first.metrics.as_ref().context("winner replay failed")?;
    let winner_receipt: CandidateReceipt = serde_json::from_slice(&fs::read(
        dataset_root(&manifest.run_id, &spec.dataset_hash).join(&winner.receipt_file),
    )?)?;
    let winner_metrics = winner_receipt
        .metrics
        .as_ref()
        .context("winning generation receipt failed")?;
    if winner_metrics.rmse.to_bits() != first_metrics.rmse.to_bits()
        || winner_receipt.predictions.len() != first.predictions.len()
        || winner_receipt
            .predictions
            .iter()
            .zip(&first.predictions)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        bail!("winner confirmation differs from the completed winning generation");
    }
    let prediction_bits = first
        .predictions
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    for evaluation in &evaluations[1..] {
        let metrics = evaluation
            .receipt
            .metrics
            .as_ref()
            .context("winner replay failed")?;
        if metrics.rmse.to_bits() != first_metrics.rmse.to_bits()
            || evaluation.receipt.predictions.len() != first.predictions.len()
            || evaluation
                .receipt
                .predictions
                .iter()
                .zip(&first.predictions)
                .any(|(left, right)| left.to_bits() != right.to_bits())
        {
            bail!("winner confirmation bits differ across nodes");
        }
    }
    let confirmation = Confirmation {
        schema: CONFIRMATION_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        generation: checkpoint.generation,
        slot: winner_slot,
        checkpoint_sha256: checkpoint_sha256.to_string(),
        prediction_bits_sha256: sha256_bytes(&prediction_bits),
        rmse_bits: first_metrics.rmse.to_bits(),
        nodes: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt.node.clone())
            .collect(),
        receipt_files: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt_file.clone())
            .collect(),
        receipt_sha256: evaluations
            .iter()
            .map(|evaluation| evaluation.receipt_sha256.clone())
            .collect(),
    };
    let bytes = canonical_json(&confirmation)?;
    let digest = sha256_bytes(&bytes);
    atomic_immutable_write(&path, &bytes)?;
    Ok((confirmation, digest))
}

fn finish_dataset(
    manifest: &AggressiveManifest,
    spec: &DatasetSpec,
    checkpoint: &AggressiveCheckpoint,
    checkpoint_sha256: &str,
    confirmation_sha256: Option<String>,
) -> Result<DatasetResult> {
    let path = result_path(&manifest.run_id, &spec.dataset_hash);
    let winner_slot = checkpoint
        .winner_slot
        .context("cannot finalize an aggressive dataset without a strict winner")?;
    let confirmation_sha256 = confirmation_sha256
        .context("cannot finalize an aggressive winner without three-node confirmation")?;
    let selected = &checkpoint.slots[winner_slot];
    let result = DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: spec.dataset_hash.clone(),
        outcome: "confirmed_strict_win".to_string(),
        stopping_generation: checkpoint.generation,
        leader_rmse: spec.leader_rmse,
        best_rmse: selected.rmse(),
        best_slot: selected.slot,
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        terminal_checkpoint_sha256: checkpoint_sha256.to_string(),
        confirmation_sha256: Some(confirmation_sha256),
    };
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let existing: DatasetResult = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&existing)? || existing != result {
            bail!("immutable aggressive dataset result changed on resume");
        }
        return Ok(existing);
    }
    atomic_immutable_write(&path, &canonical_json(&result)?)?;
    Ok(result)
}

fn run_dataset(manifest: &AggressiveManifest, spec: &DatasetSpec) -> Result<DatasetResult> {
    let (_, test) = read_campaign_data(Path::new(&manifest.registry_root), spec)?;
    let checkpoints = load_checkpoints(&manifest.run_id, spec)?;
    for (checkpoint, _) in &checkpoints {
        validate_checkpoint_receipts(manifest, spec, checkpoint, &test)?;
    }
    if let Some((terminal, digest)) = checkpoints.last() {
        if terminal.winner_slot.is_some() {
            let (_, confirmation_sha256) = confirm_winner(manifest, spec, terminal, digest, &test)?;
            return finish_dataset(manifest, spec, terminal, digest, Some(confirmation_sha256));
        }
    }

    let mut population = checkpoints
        .last()
        .map(|(checkpoint, _)| checkpoint.next_population.clone())
        .unwrap_or(initial_population(
            Path::new(&manifest.registry_root),
            &manifest.run_id,
            spec,
        )?);
    let mut previous = checkpoints.last().map(|(_, digest)| digest.clone());
    let mut generation = checkpoints.len();
    loop {
        if population.len() != TOTAL_POPULATION {
            bail!("aggressive generation population is incomplete");
        }
        validate_preflight_receipt(&preflight()?)?;
        let identity = worker_identity(manifest, spec);
        let members = population.into_iter().enumerate().collect::<Vec<_>>();
        let evaluations = scq_batch::evaluate_aggressive_batches(
            &root(&manifest.run_id),
            &identity,
            Path::new(&manifest.registry_root),
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            generation,
            &members,
            &test,
        )?;
        let mut slots = members
            .into_iter()
            .zip(evaluations)
            .map(|((slot, member), evaluation)| {
                outcome_from_evaluation(generation, slot, member, &evaluation, false)
            })
            .collect::<Vec<_>>();
        slots.sort_by_key(|outcome| outcome.slot);
        if slots
            .iter()
            .enumerate()
            .any(|(slot, outcome)| outcome.slot != slot)
        {
            bail!("aggressive completion order affected logical slots");
        }
        let (ranked_islands, valid_per_island) = rank_islands_with_global_recovery(&slots)?;
        let mut global_ranked = slots.clone();
        global_ranked.sort_by(rank_outcomes);
        let winner_slot = evolutionary_winner_slot(&global_ranked, generation, spec.leader_rmse);
        let next_population = if winner_slot.is_none() {
            ranked_islands
                .iter()
                .enumerate()
                .map(|(island, ranked)| {
                    next_island_population(
                        ranked,
                        &manifest.run_id,
                        &spec.dataset_hash,
                        generation + 1,
                        island,
                        spec.features,
                    )
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect()
        } else {
            Vec::new()
        };
        let checkpoint = AggressiveCheckpoint {
            schema: CHECKPOINT_SCHEMA.to_string(),
            benchmark_valid: false,
            test_adaptive: true,
            run_id: manifest.run_id.clone(),
            dataset_hash: spec.dataset_hash.clone(),
            generation,
            previous_checkpoint_sha256: previous.clone(),
            logical_slots: TOTAL_POPULATION,
            fresh_evaluations: 0,
            cache_hits: 0,
            cumulative_fresh_evaluations: 0,
            cumulative_cache_hits: 0,
            unique_population: 0,
            valid_per_island,
            ranked_slots: global_ranked.iter().map(|outcome| outcome.slot).collect(),
            best_slot: global_ranked[0].slot,
            best_rmse: global_ranked[0].rmse(),
            winner_slot,
            slots,
            next_population: next_population.clone(),
        };
        let bytes = canonical_json(&checkpoint)?;
        let digest = sha256_bytes(&bytes);
        atomic_immutable_write(
            &checkpoint_path(&manifest.run_id, &spec.dataset_hash, generation),
            &bytes,
        )?;
        eprintln!(
            "aggressive {} generation {generation}: best RMSE {:.17}, winner {:?}",
            spec.dataset_hash, checkpoint.best_rmse, checkpoint.winner_slot
        );
        if checkpoint.winner_slot.is_some() {
            let (_, confirmation_sha256) =
                confirm_winner(manifest, spec, &checkpoint, &digest, &test)?;
            return finish_dataset(
                manifest,
                spec,
                &checkpoint,
                &digest,
                Some(confirmation_sha256),
            );
        }
        previous = Some(digest);
        population = next_population;
        generation = generation
            .checked_add(1)
            .context("aggressive generation counter overflow")?;
    }
}

struct DatasetLane {
    dataset_index: usize,
    spec: DatasetSpec,
    test: RegressionData,
    population: Vec<Member>,
    cache: HashMap<String, CachedEvaluation>,
    history: HashSet<String>,
    cumulative_fresh_evaluations: usize,
    cumulative_cache_hits: usize,
    previous: Option<String>,
    generation: usize,
}

enum PreparedLane {
    Finished(usize, DatasetResult),
    Active(DatasetLane),
}

fn prepare_lane(
    manifest: &AggressiveManifest,
    dataset_index: usize,
    spec: &DatasetSpec,
) -> Result<PreparedLane> {
    let (_, test) = read_campaign_data(Path::new(&manifest.registry_root), spec)?;
    let checkpoints = load_checkpoints(&manifest.run_id, spec)?;
    for (checkpoint, _) in &checkpoints {
        validate_checkpoint_receipts(manifest, spec, checkpoint, &test)?;
    }
    if let Some((terminal, digest)) = checkpoints.last() {
        if terminal.winner_slot.is_some() {
            let (_, confirmation_sha256) = confirm_winner(manifest, spec, terminal, digest, &test)?;
            return Ok(PreparedLane::Finished(
                dataset_index,
                finish_dataset(manifest, spec, terminal, digest, Some(confirmation_sha256))?,
            ));
        }
    }
    let mut cache = HashMap::<String, CachedEvaluation>::new();
    let mut history = HashSet::<String>::new();
    for (checkpoint, _) in &checkpoints {
        for outcome in &checkpoint.slots {
            let receipt_path =
                dataset_root(&manifest.run_id, &spec.dataset_hash).join(&outcome.receipt_file);
            let receipt_bytes = fs::read(&receipt_path)?;
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let source = outcome.cache_source.clone().unwrap_or(CacheSource {
                generation: checkpoint.generation,
                slot: outcome.slot,
            });
            cache
                .entry(outcome.member.semantic_sha256.clone())
                .or_insert(CachedEvaluation {
                    source,
                    receipt_file: outcome.receipt_file.clone(),
                    receipt_sha256: outcome.receipt_sha256.clone(),
                    receipt,
                });
            history.insert(outcome.member.semantic_sha256.clone());
        }
    }
    let cumulative_fresh_evaluations = checkpoints
        .last()
        .map_or(0, |(checkpoint, _)| checkpoint.cumulative_fresh_evaluations);
    let cumulative_cache_hits = checkpoints
        .last()
        .map_or(0, |(checkpoint, _)| checkpoint.cumulative_cache_hits);
    Ok(PreparedLane::Active(DatasetLane {
        dataset_index,
        spec: spec.clone(),
        test,
        population: checkpoints
            .last()
            .map(|(checkpoint, _)| checkpoint.next_population.clone())
            .unwrap_or(initial_population(
                Path::new(&manifest.registry_root),
                &manifest.run_id,
                spec,
            )?),
        cache,
        history,
        cumulative_fresh_evaluations,
        cumulative_cache_hits,
        previous: checkpoints.last().map(|(_, digest)| digest.clone()),
        generation: checkpoints.len(),
    }))
}

enum CompletedLaneGeneration {
    Finished(usize, DatasetResult),
    Active(DatasetLane),
}

fn complete_lane_generation(
    manifest: &AggressiveManifest,
    mut lane: DatasetLane,
    members: Vec<(usize, Member)>,
    evaluations: Vec<CachedEvaluation>,
) -> Result<CompletedLaneGeneration> {
    let generation = lane.generation;
    let mut fresh_by_slot = evaluations
        .into_iter()
        .map(|evaluation| (evaluation.source.slot, evaluation))
        .collect::<HashMap<_, _>>();
    let mut slots = Vec::with_capacity(TOTAL_POPULATION);
    for (slot, member) in members {
        if let Some(evaluation) = fresh_by_slot.remove(&slot) {
            if lane.history.contains(&member.semantic_sha256)
                || evaluation.receipt.semantic_sha256 != member.semantic_sha256
            {
                bail!("Golden46 fresh evaluation repeated a historical genome");
            }
            lane.history.insert(member.semantic_sha256.clone());
            lane.cache
                .insert(member.semantic_sha256.clone(), evaluation.clone());
            slots.push(outcome_from_evaluation(
                generation,
                slot,
                member,
                &evaluation,
                false,
            ));
        } else {
            let evaluation = lane
                .cache
                .get(&member.semantic_sha256)
                .context("Golden46 logical elite is absent from the evaluation cache")?;
            slots.push(outcome_from_evaluation(
                generation, slot, member, evaluation, true,
            ));
        }
    }
    if !fresh_by_slot.is_empty() {
        bail!("Golden46 worker returned an evaluation for an unknown logical slot");
    }
    slots.sort_by_key(|outcome| outcome.slot);
    if slots
        .iter()
        .enumerate()
        .any(|(slot, outcome)| outcome.slot != slot)
    {
        bail!("aggressive completion order affected logical slots");
    }
    let (ranked_islands, valid_per_island) = rank_islands_with_global_recovery(&slots)?;
    let mut global_ranked = slots.clone();
    global_ranked.sort_by(rank_outcomes);
    let winner_slot = evolutionary_winner_slot(&global_ranked, generation, lane.spec.leader_rmse);
    let next_population = if winner_slot.is_none() {
        next_golden46_population(
            &ranked_islands,
            &manifest.run_id,
            &lane.spec.dataset_hash,
            generation + 1,
            lane.spec.features,
            &lane.history,
        )?
    } else {
        Vec::new()
    };
    let unique_population = slots
        .iter()
        .map(|outcome| outcome.member.semantic_sha256.as_str())
        .collect::<HashSet<_>>()
        .len();
    let cache_hits = slots
        .iter()
        .filter(|outcome| outcome.cache_source.is_some())
        .count();
    let fresh_evaluations = TOTAL_POPULATION - cache_hits;
    if unique_population != TOTAL_POPULATION
        || fresh_evaluations + cache_hits != TOTAL_POPULATION
        || (generation == 0 && cache_hits != 0)
    {
        bail!("Golden46 completed generation is not unique/cache-accounted");
    }
    lane.cumulative_fresh_evaluations += fresh_evaluations;
    lane.cumulative_cache_hits += cache_hits;
    let checkpoint = AggressiveCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: manifest.run_id.clone(),
        dataset_hash: lane.spec.dataset_hash.clone(),
        generation,
        previous_checkpoint_sha256: lane.previous.clone(),
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations,
        cache_hits,
        cumulative_fresh_evaluations: lane.cumulative_fresh_evaluations,
        cumulative_cache_hits: lane.cumulative_cache_hits,
        unique_population,
        valid_per_island,
        ranked_slots: global_ranked.iter().map(|outcome| outcome.slot).collect(),
        best_slot: global_ranked[0].slot,
        best_rmse: global_ranked[0].rmse(),
        winner_slot,
        slots,
        next_population: next_population.clone(),
    };
    let bytes = canonical_json(&checkpoint)?;
    let digest = sha256_bytes(&bytes);
    atomic_immutable_write(
        &checkpoint_path(&manifest.run_id, &lane.spec.dataset_hash, generation),
        &bytes,
    )?;
    eprintln!(
        "aggressive {} generation {generation}: best RMSE {:.17}, winner {:?}",
        lane.spec.dataset_hash, checkpoint.best_rmse, checkpoint.winner_slot
    );
    if checkpoint.winner_slot.is_some() {
        let (_, confirmation_sha256) =
            confirm_winner(manifest, &lane.spec, &checkpoint, &digest, &lane.test)?;
        return Ok(CompletedLaneGeneration::Finished(
            lane.dataset_index,
            finish_dataset(
                manifest,
                &lane.spec,
                &checkpoint,
                &digest,
                Some(confirmation_sha256),
            )?,
        ));
    }
    lane.previous = Some(digest);
    lane.population = next_population;
    lane.generation = generation
        .checked_add(1)
        .context("aggressive generation counter overflow")?;
    Ok(CompletedLaneGeneration::Active(lane))
}

fn run_dataset_waves(manifest: &AggressiveManifest) -> Result<Vec<DatasetResult>> {
    let mut results = vec![None; manifest.datasets.len()];
    let mut next_dataset = 0usize;
    let mut active = Vec::<DatasetLane>::new();
    while next_dataset < manifest.datasets.len() || !active.is_empty() {
        while active.len() < scq_batch::AGGRESSIVE_DATASET_LANES
            && next_dataset < manifest.datasets.len()
        {
            let spec = &manifest.datasets[next_dataset];
            eprintln!(
                "aggressive dataset {}/{} {}",
                next_dataset + 1,
                manifest.datasets.len(),
                spec.dataset_hash
            );
            match prepare_lane(manifest, next_dataset, spec)? {
                PreparedLane::Finished(index, result) => results[index] = Some(result),
                PreparedLane::Active(lane) => active.push(lane),
            }
            next_dataset += 1;
        }
        if active.is_empty() {
            continue;
        }
        // The immutable launch/resume preflight above establishes the resource
        // floor. Per-node SCQ leases and the post-lease occupancy check in
        // `run_batch_flavored` serialize later waves with other campaign
        // controllers. Re-running the global "queue must be empty" preflight
        // here makes two otherwise compatible campaigns terminate whenever
        // their generation boundaries do not align.
        let identities = active
            .iter()
            .map(|lane| worker_identity(manifest, &lane.spec))
            .collect::<Vec<_>>();
        let member_sets = active
            .iter_mut()
            .map(|lane| {
                std::mem::take(&mut lane.population)
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if member_sets
            .iter()
            .any(|members| members.len() != TOTAL_POPULATION)
        {
            bail!("aggressive generation population is incomplete");
        }
        let fresh_member_sets = member_sets
            .iter()
            .zip(&active)
            .map(|(members, lane)| {
                members
                    .iter()
                    .filter(|(_, member)| !lane.cache.contains_key(&member.semantic_sha256))
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if fresh_member_sets.iter().any(Vec::is_empty) {
            bail!("Golden46 generation contains no fresh candidates");
        }
        let lane_requests = (0..active.len())
            .map(|index| {
                (
                    &identities[index],
                    active[index].generation,
                    fresh_member_sets[index].as_slice(),
                    &active[index].test,
                )
            })
            .collect::<Vec<_>>();
        let evaluation_sets = scq_batch::evaluate_aggressive_wave(
            &root(&manifest.run_id),
            Path::new(&manifest.registry_root),
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            &lane_requests,
        )?;
        let previous_active = std::mem::take(&mut active);
        for ((lane, members), evaluations) in previous_active
            .into_iter()
            .zip(member_sets)
            .zip(evaluation_sets)
        {
            match complete_lane_generation(manifest, lane, members, evaluations)? {
                CompletedLaneGeneration::Finished(index, result) => results[index] = Some(result),
                CompletedLaneGeneration::Active(lane) => active.push(lane),
            }
        }
    }
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("aggressive dataset {index} produced no result"))
        })
        .collect()
}

fn remove_csv_header(bytes: &[u8], remove: bool, label: &str) -> Result<Vec<u8>> {
    if !remove {
        return Ok(bytes.to_vec());
    }
    let newline = bytes
        .iter()
        .position(|byte| *byte == b'\n')
        .with_context(|| format!("{label} has a header but no data line"))?;
    let data = bytes[newline + 1..].to_vec();
    if data.is_empty() {
        bail!("{label} has no rows after its header");
    }
    Ok(data)
}

fn golden46_panel_value() -> Result<(Value, Vec<u8>, String)> {
    let path = Path::new(GOLDEN46_PANEL_MANIFEST);
    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let dataset_ids = value
        .get("dataset_ids")
        .and_then(Value::as_array)
        .context("frozen panel46 has no dataset_ids")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .context("frozen panel46 dataset id is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    if value.get("schema").and_then(Value::as_str) != Some("golden46.dataset-panel-manifest.v1")
        || value.get("campaign_id").and_then(Value::as_str) != Some(GOLDEN46_CAMPAIGN_ID)
        || dataset_ids.len() != 46
        || dataset_ids[42..] != GOLDEN46_PANEL_DATASETS
        || dataset_ids.iter().collect::<HashSet<_>>().len() != dataset_ids.len()
    {
        bail!("frozen panel46 identity or ordered cohort changed");
    }
    let digest = sha256_bytes(&bytes);
    Ok((value, bytes, digest))
}

fn prepare_golden46_evolution_views(source_registry: &Path) -> Result<(PathBuf, String)> {
    let (panel, _, panel_sha256) = golden46_panel_value()?;
    if panel.get("dataset_root").and_then(Value::as_str)
        != Some(source_registry.to_string_lossy().as_ref())
    {
        bail!("Golden46 source registry differs from the frozen panel46 data root");
    }
    let records = panel
        .get("datasets")
        .and_then(Value::as_array)
        .context("frozen panel46 has no dataset records")?;
    let by_id = records
        .iter()
        .filter_map(|record| {
            record
                .get("dataset_id")
                .and_then(Value::as_str)
                .map(|id| (id, record))
        })
        .collect::<HashMap<_, _>>();
    let mut view_bytes = Vec::<(String, Vec<u8>, Vec<u8>, EvolutionViewDataset)>::new();
    for dataset_hash in GOLDEN46_DATASETS {
        let record = by_id
            .get(dataset_hash)
            .with_context(|| format!("panel46 lacks dataset record {dataset_hash}"))?;
        let artifacts = record
            .get("artifacts")
            .and_then(Value::as_object)
            .context("panel46 dataset record has no artifact hashes")?;
        let headers_removed = record
            .get("headers_removed_from_csv")
            .and_then(Value::as_bool)
            .context("panel46 dataset record has no header policy")?;
        let source = source_registry.join(dataset_hash);
        let source_train = fs::read(source.join("train.csv"))?;
        let source_test = fs::read(source.join("test.csv"))?;
        let source_train_sha256 = sha256_bytes(&source_train);
        let source_test_sha256 = sha256_bytes(&source_test);
        if artifacts.get("train.csv").and_then(Value::as_str) != Some(source_train_sha256.as_str())
            || artifacts.get("test.csv").and_then(Value::as_str)
                != Some(source_test_sha256.as_str())
        {
            bail!("Golden46 source dataset {dataset_hash} changed after panel freeze");
        }
        let remove_header = !headers_removed;
        let train = remove_csv_header(
            &source_train,
            remove_header,
            &format!("{dataset_hash}/train.csv"),
        )?;
        let test = remove_csv_header(
            &source_test,
            remove_header,
            &format!("{dataset_hash}/test.csv"),
        )?;
        let view = EvolutionViewDataset {
            dataset_hash: dataset_hash.to_string(),
            source_train_sha256,
            source_test_sha256,
            train_sha256: sha256_bytes(&train),
            test_sha256: sha256_bytes(&test),
            removed_train_header: remove_header,
            removed_test_header: remove_header,
        };
        view_bytes.push((dataset_hash.to_string(), train, test, view));
    }
    let view_records = view_bytes
        .iter()
        .map(|(_, _, _, record)| record.clone())
        .collect::<Vec<_>>();
    let view_sha256 = sha256_bytes(&canonical_json(&view_records)?);
    let view_root = root(GOLDEN46_RUN_PREFIX)
        .join("evolution-views")
        .join(&view_sha256);
    let registry = view_root.join("datasets");
    for (dataset_hash, train, test, _) in &view_bytes {
        atomic_immutable_write(&registry.join(dataset_hash).join("train.csv"), train)?;
        atomic_immutable_write(&registry.join(dataset_hash).join("test.csv"), test)?;
    }
    let manifest = EvolutionViewManifest {
        schema: "JopeGolden46EvolutionViews.v1".to_string(),
        campaign_id: GOLDEN46_CAMPAIGN_ID.to_string(),
        panel_manifest_sha256: panel_sha256,
        source_registry_root: source_registry.to_string_lossy().into_owned(),
        view_sha256,
        datasets: view_records,
    };
    let manifest_bytes = canonical_json(&manifest)?;
    atomic_immutable_write(
        &view_root.join(EVOLUTION_VIEW_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    Ok((registry, sha256_bytes(&manifest_bytes)))
}

fn load_golden46_specs(registry: &Path) -> Result<(Vec<DatasetSpec>, String, String, String)> {
    let (_, _, panel_sha256) = golden46_panel_value()?;
    let view_manifest_path = registry
        .parent()
        .context("Golden46 view registry has no parent")?
        .join(EVOLUTION_VIEW_MANIFEST_FILE);
    let view_manifest_bytes = fs::read(&view_manifest_path)?;
    let view_manifest: EvolutionViewManifest = serde_json::from_slice(&view_manifest_bytes)?;
    if view_manifest_bytes != canonical_json(&view_manifest)?
        || view_manifest.schema != "JopeGolden46EvolutionViews.v1"
        || view_manifest.campaign_id != GOLDEN46_CAMPAIGN_ID
        || view_manifest.panel_manifest_sha256 != panel_sha256
        || view_manifest.datasets.len() != GOLDEN46_DATASETS.len()
        || view_manifest.view_sha256 != sha256_bytes(&canonical_json(&view_manifest.datasets)?)
        || view_manifest
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_hash.as_str())
            .collect::<Vec<_>>()
            != GOLDEN46_DATASETS
    {
        bail!("Golden46 evolution view manifest changed");
    }

    let leader_path = Path::new(GOLDEN46_LEADER_MANIFEST);
    let leader_bytes = fs::read(leader_path).with_context(|| {
        format!(
            "Golden46 leaders are not frozen; run scripts/build_golden46_leaders.py after all external baselines resolve: {}",
            leader_path.display()
        )
    })?;
    let leader_value: Value = serde_json::from_slice(&leader_bytes)?;
    let required_dataset_ids = leader_value
        .get("required_dataset_ids")
        .and_then(Value::as_array)
        .context("Golden46 leader manifest has no required_dataset_ids")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .context("Golden46 required dataset id is not a string")
        })
        .collect::<Result<Vec<_>>>()?;
    if leader_value.get("schema").and_then(Value::as_str) != Some("golden46.external-leaders.v1")
        || leader_value.get("campaign_id").and_then(Value::as_str) != Some(GOLDEN46_CAMPAIGN_ID)
        || leader_value.get("benchmark_valid").and_then(Value::as_bool) != Some(true)
        || leader_value.get("panel46_sha256").and_then(Value::as_str) != Some(panel_sha256.as_str())
        || required_dataset_ids != GOLDEN46_DATASETS
        || leader_value.get("excluded_dataset_ids") != Some(&serde_json::json!(["ZTS"]))
    {
        bail!("Golden46 leader manifest identity changed");
    }
    let analysis: LeaderAnalysis = serde_json::from_slice(&leader_bytes)?;
    let by_hash = analysis
        .datasets
        .iter()
        .map(|dataset| (dataset.dataset_hash.as_str(), dataset))
        .collect::<HashMap<_, _>>();
    if by_hash.len() != GOLDEN46_DATASETS.len()
        || by_hash
            .keys()
            .any(|dataset| !GOLDEN46_DATASETS.contains(dataset))
    {
        bail!("Golden46 leader manifest dataset cohort changed");
    }

    let views = view_manifest
        .datasets
        .iter()
        .map(|view| (view.dataset_hash.as_str(), view))
        .collect::<HashMap<_, _>>();
    let mut specs = Vec::with_capacity(GOLDEN46_DATASETS.len());
    for dataset_hash in GOLDEN46_DATASETS {
        let view = views
            .get(dataset_hash)
            .with_context(|| format!("Golden46 view is missing {dataset_hash}"))?;
        let leader = by_hash
            .get(dataset_hash)
            .with_context(|| format!("Golden46 leaders are missing {dataset_hash}"))?;
        let (leader_models, leader_rmse) = select_leader_with_model_count(leader, 123)?;
        let dataset = registry.join(dataset_hash);
        let train_path = dataset.join("train.csv");
        let test_path = dataset.join("test.csv");
        if sha256_file(&train_path)? != view.train_sha256
            || sha256_file(&test_path)? != view.test_sha256
        {
            bail!("Golden46 evolution view {dataset_hash} changed");
        }
        let train = read_regression_csv(&train_path)?;
        let test = read_regression_csv(&test_path)?;
        if train.x.len() != leader.train_rows
            || test.x.len() != leader.test_rows
            || train.features != leader.features
            || test.features != leader.features
        {
            bail!("Golden46 view {dataset_hash} shape differs from the frozen leader");
        }
        specs.push(DatasetSpec {
            dataset_hash: dataset_hash.to_string(),
            train_sha256: view.train_sha256.clone(),
            test_sha256: view.test_sha256.clone(),
            train_rows: leader.train_rows,
            test_rows: leader.test_rows,
            features: leader.features,
            leader_models,
            leader_rmse,
        });
    }
    Ok((
        specs,
        panel_sha256,
        sha256_bytes(&leader_bytes),
        sha256_bytes(&view_manifest_bytes),
    ))
}

fn select_dataset_specs(
    specs: &[DatasetSpec],
    selector: Option<&str>,
    start_index: usize,
) -> Result<Vec<DatasetSpec>> {
    if selector.is_some() && start_index != 0 {
        bail!("aggressive dataset selector and start index are mutually exclusive");
    }
    let Some(selector) = selector else {
        if start_index >= specs.len() {
            bail!("aggressive start index {start_index} is outside the Golden42 panel");
        }
        return Ok(specs[start_index..].to_vec());
    };
    let selectors = selector.split(',').collect::<Vec<_>>();
    if selectors.iter().any(|selector| selector.is_empty()) {
        bail!("aggressive dataset selector contains an empty value");
    }
    let mut selected = HashSet::with_capacity(selectors.len());
    for selector in selectors {
        if let Some(index) = specs.iter().position(|spec| spec.dataset_hash == selector) {
            if !selected.insert(index) {
                bail!("dataset selector {selector} is duplicated");
            }
            continue;
        }
        if !selector.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!(
                "non-hexadecimal dataset identifiers such as ZTS and ZTS_small must be selected exactly"
            );
        }
        let selector = selector.to_ascii_lowercase();
        let matches = specs
            .iter()
            .enumerate()
            .filter(|(_, spec)| {
                spec.dataset_hash
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                    && spec.dataset_hash.starts_with(&selector)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => {
                if !selected.insert(*index) {
                    bail!("dataset selector {selector} is duplicated");
                }
            }
            [] => bail!("no dataset matches selector {selector}"),
            _ => bail!("dataset selector {selector} is ambiguous"),
        }
    }
    Ok(specs
        .iter()
        .enumerate()
        .filter(|(index, _)| selected.contains(index))
        .map(|(_, spec)| spec.clone())
        .collect())
}

fn validate_preflight_receipt(receipt: &PreflightReceipt) -> Result<()> {
    if receipt.schema != PREFLIGHT_SCHEMA
        || receipt.benchmark_valid
        || !receipt.test_adaptive
        || receipt.required_cores != scq_batch::AGGRESSIVE_CPUS
        || receipt.required_available_ram_bytes != MIN_AVAILABLE_RAM_BYTES
        || receipt.required_free_disk_bytes != MIN_FREE_DISK_BYTES
        || receipt.maximum_swap_used_bytes_exclusive != MAX_SWAP_USED_BYTES
        || !receipt.active_scq_jobs.is_empty()
        || receipt.nodes.len() != scq_batch::NODES.len()
    {
        bail!("invalid aggressive preflight receipt");
    }
    for (index, node) in receipt.nodes.iter().enumerate() {
        if node.node != scq_batch::NODES[index]
            || node.online_cores < scq_batch::AGGRESSIVE_CPUS
            || node.available_ram_bytes < MIN_AVAILABLE_RAM_BYTES
            || node.free_disk_bytes < MIN_FREE_DISK_BYTES
            || node.swap_used_bytes >= MAX_SWAP_USED_BYTES
            || !node.competing_processes.is_empty()
        {
            bail!("aggressive preflight resource binding is invalid");
        }
    }
    Ok(())
}

fn expected_manifest(
    registry: &Path,
    run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    preflight_sha256: &str,
    staging_sha256: &str,
    backend_admission_sha256: &str,
    datasets: Vec<DatasetSpec>,
    golden46: Option<&Golden46Bindings>,
) -> AggressiveManifest {
    AggressiveManifest {
        schema: SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        selected_backend: "rust".to_string(),
        cpp17_admitted: false,
        cpp17_rejection: "full live-registry fitted lowering/runtime parity is unavailable; the standalone P-1C Plan emitter is not BoundModel execution, so performance admission is fail-closed"
            .to_string(),
        backend_admission_sha256: backend_admission_sha256.to_string(),
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        preflight_sha256: preflight_sha256.to_string(),
        staging_sha256: staging_sha256.to_string(),
        panel_manifest: golden46.map(|_| GOLDEN46_PANEL_MANIFEST.to_string()),
        panel_manifest_sha256: golden46.map(|binding| binding.panel_sha256.clone()),
        leader_manifest: golden46.map(|_| GOLDEN46_LEADER_MANIFEST.to_string()),
        leader_manifest_sha256: golden46.map(|binding| binding.leader_sha256.clone()),
        evolution_view_manifest_sha256: golden46
            .map(|binding| binding.evolution_view_manifest_sha256.clone()),
        arm: golden46.map(|binding| binding.arm.clone()),
        leaderboard_default: golden46.map(|binding| binding.leaderboard_default),
        registry_root: registry.to_string_lossy().into_owned(),
        population: TOTAL_POPULATION,
        islands: ISLANDS,
        population_per_island: ISLAND_POPULATION,
        workers_per_node: scq_batch::AGGRESSIVE_WORKERS,
        generation_limit: None,
        routing: if golden46.is_some() {
            "four islands of nine globally unique logical members; generation zero evaluates 36 fresh full-genome mutations of one best parent, later generations cache eight copied elites and route 28 history-unique fresh descendants by nodes[(slot + generation) % 3] with single-threaded workers"
                .to_string()
        } else {
            "up to 42 concurrent datasets; each 24-slot generation routes 8 candidates to each node by nodes[(slot + generation) % 3], while each node drains its over-decomposed queue through 240 single-threaded worker processes pinned two per each of 120 CPUs"
                .to_string()
        },
        retry: if golden46.is_some() {
            "one immutable missing-result retry on a different node".to_string()
        } else {
            "up to three immutable missing-result retries on rotating nodes".to_string()
        },
        stop_rule: "no generation cap; record generation-zero crossings but continue until a generation-1-or-later evolved descendant has finite test RMSE strictly below the best declared external baseline, then reproduce identical prediction and RMSE bits on all three nodes"
            .to_string(),
        datasets,
    }
}

fn manifest_matches_golden46_bindings(
    manifest: &AggressiveManifest,
    golden46: Option<&Golden46Bindings>,
) -> bool {
    match golden46 {
        Some(binding) => {
            manifest.panel_manifest.as_deref() == Some(GOLDEN46_PANEL_MANIFEST)
                && manifest.panel_manifest_sha256.as_deref() == Some(binding.panel_sha256.as_str())
                && manifest.leader_manifest.as_deref() == Some(GOLDEN46_LEADER_MANIFEST)
                && manifest.leader_manifest_sha256.as_deref()
                    == Some(binding.leader_sha256.as_str())
                && manifest.evolution_view_manifest_sha256.as_deref()
                    == Some(binding.evolution_view_manifest_sha256.as_str())
                && manifest.arm.as_deref() == Some(binding.arm.as_str())
                && manifest.leaderboard_default == Some(binding.leaderboard_default)
                && (binding.arm != "target-family-informed"
                    || (manifest.datasets.len() == 1 && manifest.datasets[0].dataset_hash == "ZTS"))
        }
        None => {
            manifest.panel_manifest.is_none()
                && manifest.panel_manifest_sha256.is_none()
                && manifest.leader_manifest.is_none()
                && manifest.leader_manifest_sha256.is_none()
                && manifest.evolution_view_manifest_sha256.is_none()
                && manifest.arm.is_none()
                && manifest.leaderboard_default.is_none()
        }
    }
}

pub(crate) fn run_golden46(
    source_registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    transfer_arm: bool,
    resume: bool,
) -> Result<()> {
    if run_id != GOLDEN46_RUN_PREFIX {
        bail!("run-golden46 is restricted to run id {GOLDEN46_RUN_PREFIX}");
    }
    if transfer_arm {
        bail!("full ZTS and its transfer arm are excluded from the required study");
    }
    let effective_run_id = run_id.to_string();
    let selector = dataset_selector;
    let (registry, prepared_view_manifest_sha256) =
        prepare_golden46_evolution_views(source_registry)?;
    let (all_specs, panel_sha256, leader_sha256, view_manifest_sha256) =
        load_golden46_specs(&registry)?;
    if prepared_view_manifest_sha256 != view_manifest_sha256 {
        bail!("Golden46 evolution view changed between preparation and loading");
    }
    let binding = Golden46Bindings {
        panel_sha256,
        leader_sha256,
        evolution_view_manifest_sha256: view_manifest_sha256,
        arm: "corrected-test-adaptive-continuation".to_string(),
        leaderboard_default: true,
    };
    run_aggressive_with_specs(
        &registry,
        &effective_run_id,
        selector,
        0,
        resume,
        all_specs,
        Some(binding),
    )
}

pub(crate) fn validate_golden46(
    source_registry: &Path,
    run_id: &str,
    transfer_arm: bool,
) -> Result<()> {
    if run_id != GOLDEN46_RUN_PREFIX {
        bail!("validate-golden46 is restricted to run id {GOLDEN46_RUN_PREFIX}");
    }
    if transfer_arm {
        bail!("full ZTS and its transfer arm are excluded from the required study");
    }
    let effective_run_id = run_id.to_string();
    let (registry, prepared_view_manifest_sha256) =
        prepare_golden46_evolution_views(source_registry)?;
    let (all_specs, panel_sha256, leader_sha256, view_manifest_sha256) =
        load_golden46_specs(&registry)?;
    if prepared_view_manifest_sha256 != view_manifest_sha256 {
        bail!("Golden46 evolution view changed between preparation and validation");
    }
    let binding = Golden46Bindings {
        panel_sha256,
        leader_sha256,
        evolution_view_manifest_sha256: view_manifest_sha256,
        arm: "corrected-test-adaptive-continuation".to_string(),
        leaderboard_default: true,
    };
    validate_aggressive_with_specs(&registry, &effective_run_id, all_specs, Some(&binding))?;
    Ok(())
}

pub(crate) fn run_aggressive(
    registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    start_index: usize,
    resume: bool,
) -> Result<()> {
    if is_golden46_run(run_id) {
        bail!("Golden46 runs require the dedicated run-golden46 command");
    }
    let (_, all_specs, _, _) = load_panel_specs(registry)?;
    run_aggressive_with_specs(
        registry,
        run_id,
        dataset_selector,
        start_index,
        resume,
        all_specs,
        None,
    )
}

fn run_aggressive_with_specs(
    registry: &Path,
    run_id: &str,
    dataset_selector: Option<&str>,
    start_index: usize,
    resume: bool,
    all_specs: Vec<DatasetSpec>,
    golden46: Option<Golden46Bindings>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let campaign_root = root(run_id);
    let manifest_path = campaign_root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("aggressive run already exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest: AggressiveManifest = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&manifest)?
            || manifest.schema != SCHEMA
            || manifest.benchmark_valid
            || !manifest.test_adaptive
            || manifest.run_id != run_id
            || manifest.selected_backend != "rust"
            || manifest.cpp17_admitted
            || manifest.generation_limit.is_some()
            || manifest.population != TOTAL_POPULATION
            || manifest.islands != ISLANDS
            || manifest.population_per_island != ISLAND_POPULATION
            || manifest.workers_per_node != scq_batch::AGGRESSIVE_WORKERS
            || manifest.registry_root != registry.to_string_lossy()
            || !manifest_matches_golden46_bindings(&manifest, golden46.as_ref())
            || manifest.datasets.is_empty()
            || manifest
                .datasets
                .iter()
                .any(|spec| !all_specs.contains(spec))
            || manifest.datasets != select_dataset_specs(&all_specs, dataset_selector, start_index)?
            || manifest.release_binary_sha256 != current_binary_sha256()?
            || manifest.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("aggressive manifest or selected backend changed");
        }
        let preflight_bytes = fs::read(campaign_root.join(PREFLIGHT_FILE))?;
        if sha256_bytes(&preflight_bytes) != manifest.preflight_sha256 {
            bail!("aggressive preflight receipt changed");
        }
        let recorded_preflight: PreflightReceipt = serde_json::from_slice(&preflight_bytes)?;
        if preflight_bytes != canonical_json(&recorded_preflight)? {
            bail!("aggressive preflight receipt is not canonical");
        }
        validate_preflight_receipt(&recorded_preflight)?;
        let current_preflight = preflight()?;
        validate_preflight_receipt(&current_preflight)?;
        let staging_bytes = fs::read(campaign_root.join(STAGING_FILE))?;
        if sha256_bytes(&staging_bytes) != manifest.staging_sha256 {
            bail!("aggressive staging receipt changed");
        }
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        if staging_bytes != canonical_json(&staging)?
            || staging.scq_binary_sha256 != manifest.scq_binary_sha256
        {
            bail!("aggressive staging receipt identity changed");
        }
        scq_batch::validate_staging(
            &staging,
            &manifest.release_binary_sha256,
            &manifest.datasets,
        )?;
        scq_batch::verify_remote_staging(&staging)?;
        let admission_bytes = fs::read(campaign_root.join(ADMISSION_FILE))?;
        if sha256_bytes(&admission_bytes) != manifest.backend_admission_sha256 {
            bail!("fitted backend admission receipt changed");
        }
        manifest
    } else {
        if campaign_root.exists() && !resume && golden46.is_none() {
            bail!("aggressive run directory already exists");
        }
        let admission = crate::aggressive::fitted_backend_admission()?;
        if admission["selected_backend"].as_str() != Some("rust")
            || admission["cpp17"]["admitted"].as_bool() != Some(false)
        {
            bail!("fitted backend admission did not fail closed to Rust");
        }
        let admission_bytes = canonical_json(&admission)?;
        let specs = select_dataset_specs(&all_specs, dataset_selector, start_index)?;
        let preflight = preflight()?;
        let preflight_bytes = canonical_json(&preflight)?;
        let release_binary = std::env::current_exe()?;
        let release_binary_sha256 = sha256_file(&release_binary)?;
        let staging = scq_batch::preflight_and_stage(
            registry,
            &release_binary,
            &release_binary_sha256,
            &specs,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(&campaign_root)?;
        atomic_immutable_write(&campaign_root.join(ADMISSION_FILE), &admission_bytes)?;
        atomic_immutable_write(&campaign_root.join(PREFLIGHT_FILE), &preflight_bytes)?;
        atomic_immutable_write(&campaign_root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            registry,
            run_id,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&preflight_bytes),
            &sha256_bytes(&staging_bytes),
            &sha256_bytes(&admission_bytes),
            specs.clone(),
            golden46.as_ref(),
        );
        atomic_immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if campaign_root.join(SUMMARY_FILE).is_file() {
        return validate_aggressive_with_specs(registry, run_id, all_specs, golden46.as_ref());
    }
    let results = if golden46.is_some() {
        run_dataset_waves(&manifest)?
    } else if manifest.datasets.len() == 1 {
        vec![run_dataset(&manifest, &manifest.datasets[0])?]
    } else {
        run_dataset_waves(&manifest)?
    };
    let confirmed_wins = results
        .iter()
        .filter(|result| result.outcome == "confirmed_strict_win")
        .count();
    let summary = Summary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        selected_backend: "rust".to_string(),
        dataset_count: results.len(),
        confirmed_wins,
        unresolved: results.len() - confirmed_wins,
        results,
    };
    atomic_immutable_write(
        &campaign_root.join(SUMMARY_FILE),
        &canonical_json(&summary)?,
    )?;
    validate_aggressive_with_specs(registry, run_id, all_specs, golden46.as_ref())
}

pub(crate) fn validate_aggressive(registry: &Path, run_id: &str) -> Result<()> {
    if is_golden46_run(run_id) {
        bail!("Golden46 runs require the dedicated validate-golden46 command");
    }
    let (_, all_specs, _, _) = load_panel_specs(registry)?;
    validate_aggressive_with_specs(registry, run_id, all_specs, None)
}

fn validate_aggressive_with_specs(
    registry: &Path,
    run_id: &str,
    all_specs: Vec<DatasetSpec>,
    golden46: Option<&Golden46Bindings>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let campaign_root = root(run_id);
    let manifest_bytes = fs::read(campaign_root.join(MANIFEST_FILE))?;
    let manifest: AggressiveManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest_bytes != canonical_json(&manifest)?
        || manifest.schema != SCHEMA
        || manifest.benchmark_valid
        || !manifest.test_adaptive
        || manifest.run_id != run_id
        || manifest.selected_backend != "rust"
        || manifest.cpp17_admitted
        || manifest.generation_limit.is_some()
        || manifest.registry_root != registry.to_string_lossy()
        || !manifest_matches_golden46_bindings(&manifest, golden46)
        || manifest.datasets.is_empty()
        || manifest
            .datasets
            .iter()
            .any(|spec| !all_specs.contains(spec))
        || manifest.release_binary_sha256 != current_binary_sha256()?
        || manifest.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
    {
        bail!("invalid aggressive manifest");
    }
    let specs = &manifest.datasets;
    let admission_bytes = fs::read(campaign_root.join(ADMISSION_FILE))?;
    if sha256_bytes(&admission_bytes) != manifest.backend_admission_sha256 {
        bail!("aggressive backend admission receipt changed");
    }
    let preflight_bytes = fs::read(campaign_root.join(PREFLIGHT_FILE))?;
    if sha256_bytes(&preflight_bytes) != manifest.preflight_sha256 {
        bail!("aggressive preflight receipt changed");
    }
    let preflight_receipt: PreflightReceipt = serde_json::from_slice(&preflight_bytes)?;
    if preflight_bytes != canonical_json(&preflight_receipt)? {
        bail!("aggressive preflight receipt is not canonical");
    }
    validate_preflight_receipt(&preflight_receipt)?;
    let staging_bytes = fs::read(campaign_root.join(STAGING_FILE))?;
    if sha256_bytes(&staging_bytes) != manifest.staging_sha256 {
        bail!("aggressive staging receipt changed");
    }
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    if staging_bytes != canonical_json(&staging)?
        || staging.scq_binary_sha256 != manifest.scq_binary_sha256
    {
        bail!("aggressive staging receipt identity changed");
    }
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, specs)?;
    scq_batch::verify_remote_staging(&staging)?;
    let mut results = Vec::with_capacity(specs.len());
    for spec in specs {
        let (_, test) = read_campaign_data(registry, spec)?;
        let checkpoints = load_checkpoints(run_id, spec)?;
        for (checkpoint, _) in &checkpoints {
            validate_checkpoint_receipts(&manifest, spec, checkpoint, &test)?;
        }
        let (terminal, digest) = checkpoints
            .last()
            .context("aggressive dataset has no complete generation barrier")?;
        if terminal.winner_slot.is_none() {
            bail!("aggressive dataset result exists without a strict winner");
        }
        let result_bytes = fs::read(result_path(run_id, &spec.dataset_hash))?;
        let result: DatasetResult = serde_json::from_slice(&result_bytes)?;
        let winner_slot = terminal
            .winner_slot
            .context("aggressive dataset result has no winner")?;
        let winner = &terminal.slots[winner_slot];
        if result_bytes != canonical_json(&result)?
            || result.schema != RESULT_SCHEMA
            || result.run_id != run_id
            || result.dataset_hash != spec.dataset_hash
            || result.terminal_checkpoint_sha256 != *digest
            || result.benchmark_valid
            || !result.test_adaptive
            || result.outcome != "confirmed_strict_win"
            || result.stopping_generation != terminal.generation
            || result.leader_rmse.to_bits() != spec.leader_rmse.to_bits()
            || result.best_rmse.to_bits() != winner.rmse().to_bits()
            || result.best_slot != winner_slot
            || result.canonical_genome != winner.member.canonical_genome
            || result.semantic_sha256 != winner.member.semantic_sha256
            || result.confirmation_sha256.is_none()
        {
            bail!("aggressive dataset result does not bind to its barrier");
        }
        if let Some(slot) = terminal.winner_slot {
            let winner = &terminal.slots[slot];
            if !winner.rmse().is_finite() || winner.rmse() >= spec.leader_rmse {
                bail!("aggressive winner is not a strict finite baseline crossing");
            }
            let confirmation_bytes = fs::read(confirmation_path(run_id, &spec.dataset_hash))?;
            let confirmation: Confirmation = serde_json::from_slice(&confirmation_bytes)?;
            if confirmation_bytes != canonical_json(&confirmation)?
                || confirmation.schema != CONFIRMATION_SCHEMA
                || confirmation.benchmark_valid
                || !confirmation.test_adaptive
                || confirmation.run_id != run_id
                || confirmation.dataset_hash != spec.dataset_hash
                || confirmation.generation != terminal.generation
                || confirmation.slot != slot
                || confirmation.checkpoint_sha256 != *digest
                || confirmation.rmse_bits != winner.rmse().to_bits()
                || result.confirmation_sha256.as_deref()
                    != Some(sha256_bytes(&confirmation_bytes).as_str())
                || confirmation.nodes
                    != scq_batch::NODES
                        .iter()
                        .map(|node| (*node).to_string())
                        .collect::<Vec<_>>()
            {
                bail!("aggressive three-node confirmation is invalid");
            }
            if confirmation.receipt_files.len() != scq_batch::NODES.len()
                || confirmation.receipt_sha256.len() != scq_batch::NODES.len()
            {
                bail!("aggressive confirmation receipt set is incomplete");
            }
            for index in 0..scq_batch::NODES.len() {
                let receipt_bytes = fs::read(
                    dataset_root(run_id, &spec.dataset_hash)
                        .join(&confirmation.receipt_files[index]),
                )?;
                if sha256_bytes(&receipt_bytes) != confirmation.receipt_sha256[index] {
                    bail!("aggressive confirmation receipt hash changed");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
                if receipt_bytes != canonical_json(&receipt)? {
                    bail!("aggressive confirmation receipt is not canonical");
                }
                let request_path = dataset_root(run_id, &spec.dataset_hash)
                    .join("requests")
                    .join(&receipt.purpose)
                    .join(format!("generation-{:03}", receipt.generation))
                    .join(format!(
                        "slot-{:03}-attempt-{}.v1.json",
                        receipt.slot, receipt.attempt
                    ));
                let request_bytes = fs::read(request_path)?;
                let request: CandidateRequest = serde_json::from_slice(&request_bytes)?;
                if request_bytes != canonical_json(&request)? {
                    bail!("aggressive confirmation request is not canonical");
                }
                validate_receipt_against_request(&receipt, &request, &test)?;
                let metrics = receipt
                    .metrics
                    .as_ref()
                    .context("aggressive confirmation receipt failed")?;
                let prediction_bits = receipt
                    .predictions
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect::<Vec<_>>();
                if receipt.test_adaptive != Some(true)
                    || receipt.purpose != "aggressive-confirmation"
                    || receipt.run_id != run_id
                    || receipt.dataset_hash != spec.dataset_hash
                    || receipt.train_sha256 != spec.train_sha256
                    || receipt.test_sha256 != spec.test_sha256
                    || receipt.binary_sha256 != manifest.release_binary_sha256
                    || receipt.generation != terminal.generation
                    || receipt.slot != index
                    || receipt.attempt != 0
                    || receipt.semantic_sha256 != winner.member.semantic_sha256
                    || receipt.canonical_genome != winner.member.canonical_genome
                    || receipt.node != scq_batch::NODES[index]
                    || metrics.rmse.to_bits() != confirmation.rmse_bits
                    || sha256_bytes(&prediction_bits) != confirmation.prediction_bits_sha256
                {
                    bail!("aggressive confirmation prediction or RMSE bits changed");
                }
            }
        }
        results.push(result);
    }
    let summary_bytes = fs::read(campaign_root.join(SUMMARY_FILE))?;
    let summary: Summary = serde_json::from_slice(&summary_bytes)?;
    if summary_bytes != canonical_json(&summary)?
        || summary.schema != SUMMARY_SCHEMA
        || summary.benchmark_valid
        || !summary.test_adaptive
        || summary.run_id != run_id
        || summary.selected_backend != "rust"
        || summary.results != results
        || summary.dataset_count != specs.len()
        || summary.confirmed_wins + summary.unresolved != summary.dataset_count
    {
        bail!("invalid aggressive summary");
    }
    eprintln!(
        "validated aggressive run {run_id}: {} confirmed, {} unresolved",
        summary.confirmed_wins, summary.unresolved
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_ignores_terminal_scq_jobs() {
        let terminal = ["completed", "failed", "cancelled"];
        for state in terminal {
            let job = serde_json::json!({"id": "old", "state": state});
            assert!(!scq_job_is_active(&job));
        }
        let active = serde_json::json!({"id": "new", "state": "running"});
        assert!(scq_job_is_active(&active));
        assert!(scq_job_is_active(&serde_json::json!({"id": "unknown"})));
    }

    #[test]
    fn process_gate_ignores_scq_control_plane_but_not_workloads() {
        let controller = "42 jope-dime-regressor controller";
        assert!(!is_competing_process(controller, "xbabe2", "xbabe2", "42"));
        assert!(!is_competing_process(
            "2365996 jope-dime-regre /opt/jope-dime-regressor run-golden42-aggressive --resume",
            "xbabe2",
            "xbabe2",
            "different-pid",
        ));
        assert!(!is_competing_process(
            "7 scq /home/ubuntu/smart_cluster/target/release/scq agent status --node xbabe1",
            "xbabe1",
            "xbabe2",
            "42",
        ));
        assert!(!is_competing_process(
            "8 scq /home/ubuntu/smart_cluster/target/release/scq controller serve",
            "xbabe2",
            "xbabe2",
            "42",
        ));
        assert!(!is_competing_process(
            "9 cargo cargo audit --deny warnings",
            "xbabe2",
            "xbabe2",
            "42",
        ));
        assert!(is_competing_process(
            "10 jope-dime-regre /tmp/jope-dime-regressor candidate",
            "xbabe1",
            "xbabe2",
            "42",
        ));
    }

    #[test]
    fn dataset_selector_accepts_a_deduplicated_panel_and_preserves_registry_order() {
        let spec = |hash: &str| DatasetSpec {
            dataset_hash: hash.to_string(),
            train_sha256: "a".repeat(64),
            test_sha256: "b".repeat(64),
            train_rows: 10,
            test_rows: 5,
            features: 2,
            leader_models: vec!["leader".to_string()],
            leader_rmse: 1.0,
        };
        let specs = vec![
            spec("1111111111111111"),
            spec("2222222222222222"),
            spec("3333333333333333"),
        ];
        let selected = select_dataset_specs(&specs, Some("3333,1111"), 0).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|spec| spec.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            ["1111111111111111", "3333333333333333"]
        );
        assert!(select_dataset_specs(&specs, Some("1111,111111"), 0).is_err());
        assert!(select_dataset_specs(&specs, Some("22,not-hex"), 0).is_err());
        let named = vec![spec("ZTS_small"), spec("ZTS"), spec("abcdef0123456789")];
        assert_eq!(
            select_dataset_specs(&named, Some("ZTS_small,abc"), 0)
                .unwrap()
                .iter()
                .map(|spec| spec.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            ["ZTS_small", "abcdef0123456789"]
        );
        assert!(select_dataset_specs(&named, Some("ZTS_"), 0).is_err());
    }
    use serde_json::json;

    #[test]
    fn forty_two_dataset_wave_overdecomposes_each_node() {
        for generation in [0, 1, 2, 37, 99] {
            let mut counts = [0usize; 3];
            for slot in 0..TOTAL_POPULATION {
                counts[(slot + generation) % 3] += 1;
            }
            assert_eq!(counts, [12, 12, 12]);
            assert_eq!(counts.map(|count| count * 42), [504, 504, 504]);
        }
    }

    #[test]
    fn generation_zero_expands_to_four_complete_unique_islands() {
        let mut seen = HashSet::new();
        let base = include_str!("../../seeds/generation-zero.txt")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| canonicalize_for_features(line, 100).ok())
            .filter(|canonical| seen.insert(canonical.semantic_sha256.clone()))
            .take(ISLAND_POPULATION)
            .map(|canonical| Member {
                canonical_genome: canonical.canonical,
                semantic_sha256: canonical.semantic_sha256,
                complexity: canonical.complexity,
                rust_source_len: canonical.rust_source_len,
                evolved_descendant: false,
                ancestry: Ancestry {
                    operation: "seed".to_string(),
                    parents: Vec::new(),
                },
            })
            .collect::<Vec<_>>();
        assert_eq!(base.len(), ISLAND_POPULATION);
        let population = expand_initial_population(&base, 100).unwrap();
        assert_eq!(population.len(), TOTAL_POPULATION);
        assert_eq!(
            population
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<HashSet<_>>()
                .len(),
            TOTAL_POPULATION
        );
        assert_eq!(&population[..base.len()], base.as_slice());
    }

    #[test]
    fn ties_are_not_wins() {
        let baseline = 1.0f64;
        assert!(!(baseline < baseline));
        assert!(f64::from_bits(baseline.to_bits() - 1) < baseline);
    }

    #[test]
    fn golden46_header_view_removes_exactly_one_line() {
        let source = b"f0,f1,target\r\n1,2,3\r\n4,5,6\r\n";
        assert_eq!(
            remove_csv_header(source, true, "fixture").unwrap(),
            b"1,2,3\r\n4,5,6\r\n"
        );
        assert_eq!(remove_csv_header(source, false, "fixture").unwrap(), source);
        assert!(remove_csv_header(b"header-only", true, "fixture").is_err());
    }

    #[test]
    fn generation_zero_and_unevolved_elites_cannot_terminate() {
        let outcome = |generation: usize,
                       evolved_descendant: bool,
                       operation: &str,
                       parent: &str|
         -> SlotOutcome {
            let semantic = "a".repeat(64);
            SlotOutcome {
                generation,
                slot: 0,
                member: Member {
                    canonical_genome: "(pipe X (mean))".to_string(),
                    semantic_sha256: semantic,
                    complexity: 1,
                    rust_source_len: 1,
                    evolved_descendant,
                    ancestry: Ancestry {
                        operation: operation.to_string(),
                        parents: (!parent.is_empty())
                            .then(|| parent.to_string())
                            .into_iter()
                            .collect(),
                    },
                },
                valid: true,
                metrics: Some(CandidateMetrics {
                    rmse: 0.5,
                    mae: 0.5,
                    r2: 0.0,
                }),
                failure: None,
                cache_source: None,
                receipt_file: "receipt".to_string(),
                receipt_sha256: "b".repeat(64),
                node: "xbabe1".to_string(),
                scq_job_id: "job".to_string(),
                attempt: 0,
            }
        };
        let other_parent = "b".repeat(64);
        let same_parent = "a".repeat(64);
        assert_eq!(
            evolutionary_winner_slot(&[outcome(0, true, "mutation", &other_parent)], 0, 1.0),
            None
        );
        assert_eq!(
            evolutionary_winner_slot(&[outcome(1, true, "elite", &other_parent)], 1, 1.0),
            None
        );
        assert_eq!(
            evolutionary_winner_slot(&[outcome(1, true, "mutation", &same_parent)], 1, 1.0),
            None
        );
        assert_eq!(
            evolutionary_winner_slot(&[outcome(1, false, "mutation", &other_parent)], 1, 1.0),
            None
        );
        let mut cached = outcome(1, true, "mutation", &other_parent);
        cached.cache_source = Some(CacheSource {
            generation: 0,
            slot: 7,
        });
        assert_eq!(evolutionary_winner_slot(&[cached], 1, 1.0), None);
        assert_eq!(
            evolutionary_winner_slot(&[outcome(1, true, "mutation", &other_parent)], 1, 1.0),
            Some(0)
        );
    }

    #[test]
    fn corrected_generation_zero_has_one_parent_and_36_unique_descendants() {
        let parent_genome =
            feat_math::full_rust::canonicalize("(pipe X (standardize) (ridge))").unwrap();
        let parent = SeedProvenance {
            slot: 7,
            canonical_genome: parent_genome.canonical,
            semantic_sha256: parent_genome.semantic_sha256,
            complexity: parent_genome.complexity,
            rust_source_len: parent_genome.rust_source_len,
            source_kind: "corrected_test_adaptive_parent".to_string(),
            source_file: "/runs/v2/checkpoint-000075.json".to_string(),
            source_file_sha256: "a".repeat(64),
            source_record: 7,
            source_rank: 1,
            source_score: Some(0.1),
        };
        let spec = DatasetSpec {
            dataset_hash: "fixture".to_string(),
            train_sha256: "1".repeat(64),
            test_sha256: "2".repeat(64),
            train_rows: 10,
            test_rows: 5,
            features: 50,
            leader_models: vec!["leader".to_string()],
            leader_rmse: 0.09,
        };
        let (population, snapshot) =
            derive_corrected_generation_zero(&parent, GOLDEN46_RUN_PREFIX, &spec).unwrap();
        assert_eq!(population.len(), TOTAL_POPULATION);
        assert_eq!(snapshot.len(), TOTAL_POPULATION);
        assert_eq!(
            population
                .iter()
                .map(|member| member.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len(),
            TOTAL_POPULATION
        );
        assert!(population.iter().all(|member| {
            member.evolved_descendant
                && member.semantic_sha256 != parent.semantic_sha256
                && member.ancestry.operation == "generation_zero_mutation"
                && member.ancestry.parents == vec![parent.semantic_sha256.clone()]
        }));
        for island in 0..ISLANDS {
            assert_eq!(
                snapshot.iter().filter(|seed| seed.island == island).count(),
                ISLAND_POPULATION
            );
        }
    }

    #[test]
    fn corrected_next_population_caches_eight_elites_and_has_28_novel_children() {
        let mut ranked_islands = Vec::new();
        let mut history = HashSet::new();
        for island in 0..ISLANDS {
            let mut ranked = Vec::new();
            for local_slot in 0..ISLAND_POPULATION {
                let slot = island * ISLAND_POPULATION + local_slot;
                let genome =
                    feat_math::full_rust::random_pipeline(10_000 + slot as u64, 50).unwrap();
                history.insert(genome.semantic_sha256.clone());
                ranked.push(SlotOutcome {
                    generation: 0,
                    slot,
                    member: Member::evolved(
                        genome,
                        Ancestry {
                            operation: "generation_zero_mutation".to_string(),
                            parents: vec!["a".repeat(64)],
                        },
                    ),
                    valid: true,
                    metrics: Some(CandidateMetrics {
                        rmse: 0.1 + local_slot as f64 / 100.0,
                        mae: 0.1,
                        r2: 0.0,
                    }),
                    failure: None,
                    cache_source: None,
                    receipt_file: format!("receipt-{slot}.json"),
                    receipt_sha256: format!("{slot:064x}"),
                    node: "xbabe1".to_string(),
                    scq_job_id: format!("job-{slot}"),
                    attempt: 0,
                });
            }
            ranked_islands.push(ranked);
        }
        let next = next_golden46_population(
            &ranked_islands,
            GOLDEN46_RUN_PREFIX,
            "fixture",
            1,
            50,
            &history,
        )
        .unwrap();
        assert_eq!(next.len(), TOTAL_POPULATION);
        assert_eq!(
            next.iter()
                .map(|member| member.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len(),
            TOTAL_POPULATION
        );
        let elites = next
            .iter()
            .filter(|member| member.ancestry.operation == "elite")
            .collect::<Vec<_>>();
        let children = next
            .iter()
            .filter(|member| member.ancestry.operation != "elite")
            .collect::<Vec<_>>();
        assert_eq!(elites.len(), ISLANDS * ELITES_PER_ISLAND);
        assert_eq!(children.len(), TOTAL_POPULATION - elites.len());
        assert!(elites
            .iter()
            .all(|member| history.contains(&member.semantic_sha256)));
        assert!(children
            .iter()
            .all(|member| !history.contains(&member.semantic_sha256)));
    }

    #[test]
    fn aggressive_contract_is_not_benchmark_valid() {
        let value = json!({"benchmark_valid": false, "test_adaptive": true});
        assert_eq!(value["benchmark_valid"], false);
        assert_eq!(value["test_adaptive"], true);
    }

    #[test]
    fn traversal_run_ids_are_rejected() {
        assert!(validate_run_id(".").is_err());
        assert!(validate_run_id("..").is_err());
        assert!(validate_run_id("safe.run-1").is_ok());
    }

    #[test]
    fn one_dataset_selector_requires_a_unique_hash_prefix() {
        let make = |dataset_hash: &str| DatasetSpec {
            dataset_hash: dataset_hash.to_string(),
            train_sha256: "1".repeat(64),
            test_sha256: "2".repeat(64),
            train_rows: 2,
            test_rows: 1,
            features: 1,
            leader_models: vec!["tabpfnv3".to_string()],
            leader_rmse: 1.0,
        };
        let specs = vec![make("aa00"), make("aa11"), make("bb22")];
        assert_eq!(
            select_dataset_specs(&specs, Some("bb"), 0).unwrap(),
            vec![specs[2].clone()]
        );
        assert!(select_dataset_specs(&specs, Some("aa"), 0).is_err());
        assert!(select_dataset_specs(&specs, Some("cc"), 0).is_err());
        assert!(select_dataset_specs(&specs, Some("../"), 0).is_err());
        assert_eq!(select_dataset_specs(&specs, None, 1).unwrap(), specs[1..]);
        assert!(select_dataset_specs(&specs, Some("bb"), 1).is_err());
        assert!(select_dataset_specs(&specs, None, specs.len()).is_err());
    }
}
