use super::*;
use feat_math::advanced_v3::RawTable;

const SOURCE_ROOT: &str = "/home/ubuntu/tabArenaLite/data/tabarena-lite-v0.1/datasets/regression";
const CANONICAL_SCORES: &str =
    "/home/ubuntu/tabArenaLite/artifacts/leaderboard-20260723/canonical_scores.csv";
const LOCAL_RESULTS: &str = "/home/ubuntu/tabArenaLite/results";
const MANIFEST_SCHEMA: &str = "JopeTabArenaLiteIslandManifest.v1";
const CHECKPOINT_SCHEMA: &str = "JopeTabArenaLiteIslandCheckpoint.v1";
const RESULT_SCHEMA: &str = "JopeTabArenaLiteIslandResult.v1";
const SUMMARY_SCHEMA: &str = "JopeTabArenaLiteIslandSummary.v1";
const MANIFEST_FILE: &str = "tabarena-island-manifest.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const SUMMARY_FILE: &str = "tabarena-island-summary.v1.json";
const ISLANDS: usize = 4;
const POPULATION: usize = 8;
const TOTAL_POPULATION: usize = ISLANDS * POPULATION;
const ELITES: usize = 2;
const PARENT_POOL: usize = 4;
const MIN_VALID_ISLAND: usize = ELITES;
const MIN_VALID_GLOBAL: usize = ISLANDS * ELITES;
const DATASET_LANES: usize = 13;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaSourceBinding {
    dataset: String,
    source_train_sha256: String,
    source_test_sha256: String,
    metadata_sha256: String,
    encoded_train_sha256: String,
    encoded_test_sha256: String,
    train_rows: usize,
    test_rows: usize,
    features: usize,
    categorical_features: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaManifest {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    source_root: String,
    canonical_scores_sha256: String,
    local_leader_artifacts: Vec<ArtifactHash>,
    release_binary_sha256: String,
    scq_binary_sha256: String,
    scq_staging_receipt_sha256: String,
    scq_policy: scq_batch::ScqPolicy,
    ordered_datasets: Vec<String>,
    ordered_cohort_sha256: String,
    datasets: Vec<TabArenaSourceBinding>,
    started_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaCheckpoint {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    generation: usize,
    previous_checkpoint_sha256: Option<String>,
    logical_slots: usize,
    fresh_evaluations: usize,
    cumulative_fresh_evaluations: usize,
    valid_candidates: usize,
    island_valid_candidates: Vec<usize>,
    island_best_rmse: Vec<f64>,
    best_slot: usize,
    best_rmse: f64,
    leader_rmse: f64,
    leader_beaten: bool,
    migration_complete: bool,
    slots: Vec<SlotOutcome>,
    next_population: Vec<Member>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaResult {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset: String,
    stopping_generation: usize,
    generations_evaluated: usize,
    leader_models: Vec<String>,
    leader_rmse: f64,
    selected_rmse: f64,
    margin_vs_leader: f64,
    canonical_genome: String,
    semantic_sha256: String,
    selected_slot: usize,
    prediction_sha256: String,
    receipt_sha256: String,
    binary_sha256: String,
    logical_evaluations: usize,
    fresh_evaluations: usize,
    replay: Vec<ReplayRecord>,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct TabArenaSummary {
    schema: String,
    benchmark_valid: bool,
    run_id: String,
    dataset_count: usize,
    leaders_beaten: usize,
    total_generations: usize,
    total_logical_evaluations: usize,
    total_fresh_evaluations: usize,
    results: Vec<TabArenaResult>,
    artifacts: Vec<ArtifactHash>,
}

#[derive(Clone, Debug)]
struct EncodedDataset {
    binding: TabArenaSourceBinding,
    spec: DatasetSpec,
}

fn campaign_root(run_id: &str) -> PathBuf {
    run_root(run_id)
}

fn encode_source_dataset(source: &Path, destination: &Path) -> Result<(usize, usize, usize)> {
    let train_text = fs::read_to_string(source.join("train.csv"))?;
    let test_text = fs::read_to_string(source.join("test.csv"))?;
    let train_table = RawTable::from_csv(&train_text)?;
    let test_table = RawTable::from_csv(&test_text)?;
    if train_table.headers != test_table.headers {
        bail!(
            "TabArena train/test headers differ for {}",
            source.display()
        );
    }
    let width = train_table.headers.len();
    if width < 2 {
        bail!("TabArena CSV has no feature columns");
    }
    let train_rows = train_table.rows;
    let test_rows = test_table.rows;
    let features = width - 1;
    let metadata: Value = serde_json::from_slice(&fs::read(source.join("metadata.json"))?)?;
    let categorical = metadata
        .get("categorical_indicator")
        .and_then(Value::as_array)
        .context("TabArena metadata lacks categorical_indicator")?
        .iter()
        .map(|value| {
            value
                .as_bool()
                .context("TabArena categorical_indicator is not boolean")
        })
        .collect::<Result<Vec<_>>>()?;
    if categorical.len() != features {
        bail!("TabArena categorical_indicator width mismatch");
    }
    let dictionaries = (0..features)
        .map(|column| {
            if !categorical[column] {
                return BTreeMap::new();
            }
            let values = train_rows
                .iter()
                .map(|row| row[column].clone())
                .collect::<BTreeSet<_>>();
            values
                .into_iter()
                .enumerate()
                .map(|(index, value)| (value, index as f64))
                .collect()
        })
        .collect::<Vec<BTreeMap<String, f64>>>();
    let numeric_fill = (0..features)
        .map(|column| {
            if categorical[column] {
                return 0.0;
            }
            let mut values = train_rows
                .iter()
                .filter_map(|row| row[column].parse::<f64>().ok())
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            match values.len() {
                0 => 0.0,
                len if len % 2 == 1 => values[len / 2],
                len => (values[len / 2 - 1] + values[len / 2]) / 2.0,
            }
        })
        .collect::<Vec<_>>();
    let encode = |rows: &[Vec<String>]| -> Result<Vec<u8>> {
        let mut output = Vec::new();
        for row in rows {
            let mut values = Vec::with_capacity(width);
            for column in 0..features {
                let value = if categorical[column] {
                    dictionaries[column]
                        .get(&row[column])
                        .copied()
                        .unwrap_or(-1.0)
                } else {
                    row[column]
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite())
                        .unwrap_or(numeric_fill[column])
                };
                values.push(format!("{value:.17}"));
            }
            let target = row[features]
                .parse::<f64>()
                .with_context(|| "TabArena target is not numeric")?;
            if !target.is_finite() {
                bail!("TabArena target is not finite");
            }
            values.push(format!("{target:.17}"));
            output.extend_from_slice(values.join(",").as_bytes());
            output.push(b'\n');
        }
        Ok(output)
    };
    fs::create_dir_all(destination)?;
    let train_bytes = encode(&train_rows)?;
    let test_bytes = encode(&test_rows)?;
    let train_path = destination.join("train.csv");
    let test_path = destination.join("test.csv");
    if train_path.exists() {
        if fs::read(&train_path)? != train_bytes || fs::read(&test_path)? != test_bytes {
            bail!("TabArena encoded dataset changed on resume");
        }
    } else {
        immutable_write(&train_path, &train_bytes)?;
        immutable_write(&test_path, &test_bytes)?;
    }
    Ok((
        train_rows.len(),
        test_rows.len(),
        categorical.iter().filter(|value| **value).count(),
    ))
}

fn collect_canonical_leaders(
    canonical: &str,
    leaders: &mut HashMap<String, (f64, BTreeSet<String>)>,
) -> Result<()> {
    let table = RawTable::from_csv(canonical)?;
    let column = |name: &str| {
        table
            .headers
            .iter()
            .position(|header| header == name)
            .with_context(|| format!("canonical score CSV lacks {name}"))
    };
    let method = column("method")?;
    let dataset = column("dataset")?;
    let fold = column("fold")?;
    let metric = column("metric")?;
    let metric_error = column("metric_error")?;
    for row in &table.rows {
        if row[fold] != "0" || row[metric] != "rmse" {
            continue;
        }
        let Ok(value) = row[metric_error].parse::<f64>() else {
            continue;
        };
        if value.is_finite() {
            update_leader(leaders, &row[dataset], &row[method], value);
        }
    }
    Ok(())
}

fn collect_local_leaders() -> Result<(HashMap<String, (f64, BTreeSet<String>)>, Vec<ArtifactHash>)>
{
    let mut leaders = HashMap::<String, (f64, BTreeSet<String>)>::new();
    let canonical = fs::read_to_string(CANONICAL_SCORES)?;
    collect_canonical_leaders(&canonical, &mut leaders)?;
    let mut artifacts = vec![ArtifactHash {
        file: CANONICAL_SCORES.to_string(),
        sha256: sha256_file(Path::new(CANONICAL_SCORES))?,
    }];
    let results_root = Path::new(LOCAL_RESULTS);
    let mut paths = Vec::new();
    for model in fs::read_dir(results_root)? {
        let model = model?.path();
        if !model.is_dir() {
            continue;
        }
        for entry in fs::read_dir(model)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if path.is_file()
                && (name == "consolidated.json"
                    || (name.starts_with("part-") && name.ends_with(".jsonl")))
            {
                paths.push(path);
            }
        }
    }
    paths.sort();
    for path in paths {
        let bytes = fs::read(&path)?;
        let values = if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
            std::str::from_utf8(&bytes)?
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .collect::<Vec<_>>()
        } else {
            serde_json::from_slice::<Vec<Value>>(&bytes).unwrap_or_default()
        };
        let mut used = false;
        for value in values {
            if value.get("category").and_then(Value::as_str) != Some("regression")
                || value.get("status").and_then(Value::as_str) != Some("ok")
                || value.get("metric").and_then(Value::as_str) != Some("rmse")
                || value.get("benchmark_valid").and_then(Value::as_bool) == Some(false)
                || value
                    .get("fold")
                    .and_then(Value::as_u64)
                    .is_some_and(|fold| fold != 0)
                || value
                    .get("repeat")
                    .and_then(Value::as_u64)
                    .is_some_and(|repeat| repeat != 0)
            {
                continue;
            }
            let (Some(dataset), Some(model), Some(score)) = (
                value.get("dataset").and_then(Value::as_str),
                value.get("model").and_then(Value::as_str),
                value
                    .get("score")
                    .or_else(|| value.get("metric_error"))
                    .and_then(Value::as_f64),
            ) else {
                continue;
            };
            if score.is_finite() {
                update_leader(&mut leaders, dataset, model, score);
                used = true;
            }
        }
        if used {
            artifacts.push(ArtifactHash {
                file: path.to_string_lossy().into_owned(),
                sha256: sha256_bytes(&bytes),
            });
        }
    }
    Ok((leaders, artifacts))
}

fn update_leader(
    leaders: &mut HashMap<String, (f64, BTreeSet<String>)>,
    dataset: &str,
    model: &str,
    score: f64,
) {
    let entry = leaders
        .entry(dataset.to_string())
        .or_insert_with(|| (f64::INFINITY, BTreeSet::new()));
    match score.total_cmp(&entry.0) {
        Ordering::Less => {
            entry.0 = score;
            entry.1.clear();
            entry.1.insert(model.to_string());
        }
        Ordering::Equal => {
            entry.1.insert(model.to_string());
        }
        Ordering::Greater => {}
    }
}

fn prepare_datasets(run_id: &str) -> Result<(Vec<EncodedDataset>, Vec<ArtifactHash>)> {
    let (leaders, artifacts) = collect_local_leaders()?;
    let source_root = Path::new(SOURCE_ROOT);
    let mut sources = fs::read_dir(source_root)?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    sources.retain(|path| path.is_dir());
    sources.sort_by_key(|path| path.file_name().map(|name| name.to_os_string()));
    let encoded_root = campaign_root(run_id).join("encoded");
    let mut datasets = Vec::new();
    for source in sources {
        let dataset = source
            .file_name()
            .and_then(|name| name.to_str())
            .context("non-UTF8 TabArena dataset name")?
            .to_string();
        let (leader_rmse, leader_models) = leaders
            .get(&dataset)
            .with_context(|| format!("TabArena dataset {dataset} has no finite RMSE leader"))?;
        let destination = encoded_root.join(&dataset);
        let (train_rows, test_rows, categorical_features) =
            encode_source_dataset(&source, &destination)?;
        let train = read_regression_csv(&destination.join("train.csv"))?;
        let test = read_regression_csv(&destination.join("test.csv"))?;
        let binding = TabArenaSourceBinding {
            dataset: dataset.clone(),
            source_train_sha256: sha256_file(&source.join("train.csv"))?,
            source_test_sha256: sha256_file(&source.join("test.csv"))?,
            metadata_sha256: sha256_file(&source.join("metadata.json"))?,
            encoded_train_sha256: sha256_file(&destination.join("train.csv"))?,
            encoded_test_sha256: sha256_file(&destination.join("test.csv"))?,
            train_rows,
            test_rows,
            features: train.features,
            categorical_features,
            leader_models: leader_models.iter().cloned().collect(),
            leader_rmse: *leader_rmse,
        };
        if train.x.len() != train_rows
            || test.x.len() != test_rows
            || test.features != train.features
        {
            bail!("encoded TabArena dataset shape mismatch");
        }
        datasets.push(EncodedDataset {
            spec: DatasetSpec {
                dataset_hash: dataset,
                train_sha256: binding.encoded_train_sha256.clone(),
                test_sha256: binding.encoded_test_sha256.clone(),
                train_rows,
                test_rows,
                features: binding.features,
                leader_models: binding.leader_models.clone(),
                leader_rmse: binding.leader_rmse,
            },
            binding,
        });
    }
    if datasets.len() != 13 {
        bail!(
            "TabArena Lite regression cohort contains {} datasets, expected 13",
            datasets.len()
        );
    }
    Ok((datasets, artifacts))
}

fn select_datasets(
    datasets: Vec<EncodedDataset>,
    selector: Option<&str>,
) -> Result<Vec<EncodedDataset>> {
    let Some(selector) = selector else {
        return Ok(datasets);
    };
    let requested = selector.split(',').collect::<Vec<_>>();
    if requested.is_empty() || requested.iter().any(|name| name.is_empty()) {
        bail!("TabArena dataset selector is empty or malformed");
    }
    let requested = requested
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if requested.len() != selector.split(',').count() {
        bail!("TabArena dataset selector contains a duplicate");
    }
    let available = datasets
        .iter()
        .map(|dataset| dataset.spec.dataset_hash.clone())
        .collect::<BTreeSet<_>>();
    let unknown = requested
        .difference(&available)
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        bail!(
            "unknown TabArena dataset selector(s): {}",
            unknown.join(",")
        );
    }
    let selected = datasets
        .into_iter()
        .filter(|dataset| requested.contains(&dataset.spec.dataset_hash))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("TabArena dataset selector selected no datasets");
    }
    Ok(selected)
}

fn identity(run_id: &str, binary_sha256: &str, spec: &DatasetSpec) -> CampaignIdentity {
    CampaignIdentity {
        schema: CAMPAIGN_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        train_rows: spec.train_rows,
        test_rows: spec.test_rows,
        features: spec.features,
        leader_models: spec.leader_models.clone(),
        leader_rmse: spec.leader_rmse,
        population: TOTAL_POPULATION,
        elites: ELITES,
        parent_pool: PARENT_POOL,
        tournament: TOURNAMENT,
        crossover_probability: 0.25,
        mutation_probability: 0.75,
        max_transforms: feat_math::full_rust::MAX_TRANSFORMS,
        generations: usize::MAX,
        rng_seed: RNG_SEED,
        binary_sha256: binary_sha256.to_string(),
        execution_host: "scq".to_string(),
        execution_node: "xbabe1-xbabe2-xbabe3-ring".to_string(),
        selection_objective:
            "complete four-island generations until test RMSE beats the frozen TabArena leader"
                .to_string(),
    }
}

fn initial_population(
    registry: &Path,
    root: &Path,
    spec: &DatasetSpec,
    run_id: &str,
) -> Result<Vec<Member>> {
    let manifest = load_or_create_transfer_seed_manifest(
        &root.join("transfer-seeds.v1.json"),
        registry,
        spec,
    )?;
    initial_population_from_seeds(&manifest.seeds, run_id, &spec.dataset_hash)
}

fn initial_population_from_seeds(
    seeds: &[SeedProvenance],
    run_id: &str,
    dataset_hash: &str,
) -> Result<Vec<Member>> {
    if seeds.len() < TOTAL_POPULATION {
        bail!("TabArena island initialization requires 60 frozen transfer seeds");
    }
    const STRONG_SEEDS: usize = TOTAL_POPULATION / 2;
    let mut diverse = (STRONG_SEEDS..seeds.len()).collect::<Vec<_>>();
    diverse.sort_by_key(|index| {
        sha256_bytes(
            format!("{run_id}:{dataset_hash}:{}", seeds[*index].semantic_sha256).as_bytes(),
        )
    });
    diverse.truncate(TOTAL_POPULATION - STRONG_SEEDS);
    let mut selected = Vec::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        for local in 0..STRONG_SEEDS / ISLANDS {
            selected.push(island + local * ISLANDS);
        }
        for local in 0..diverse.len() / ISLANDS {
            selected.push(diverse[island + local * ISLANDS]);
        }
    }
    if selected.len() != TOTAL_POPULATION {
        bail!("TabArena island initialization produced an incomplete seed selection");
    }
    Ok(selected
        .into_iter()
        .map(|index| {
            let seed = &seeds[index];
            Member {
                canonical_genome: seed.canonical_genome.clone(),
                semantic_sha256: seed.semantic_sha256.clone(),
                complexity: seed.complexity,
                rust_source_len: seed.rust_source_len,
                evolved_descendant: false,
                ancestry: Ancestry {
                    operation: "tabarena_seed".to_string(),
                    parents: Vec::new(),
                },
            }
        })
        .collect())
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

fn offspring_seed(
    run_id: &str,
    dataset: &str,
    generation: usize,
    island: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!("tabarena:{RNG_SEED}:{run_id}:{dataset}:{generation}:{island}:{slot}:{retry}")
            .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn next_population(
    run_id: &str,
    dataset: &str,
    generation: usize,
    features: usize,
    slots: &[SlotOutcome],
) -> Result<Vec<Member>> {
    if slots.len() != TOTAL_POPULATION {
        bail!("TabArena checkpoint population is incomplete");
    }
    let mut ranked = (0..ISLANDS)
        .map(|island| {
            let mut values = slots[island * POPULATION..(island + 1) * POPULATION].to_vec();
            values.sort_by(rank_outcomes);
            values
        })
        .collect::<Vec<_>>();
    if generation % 2 == 1 {
        let original = ranked.clone();
        for source in 0..ISLANDS {
            let destination = (source + 1) % ISLANDS;
            for rank in 0..2 {
                ranked[destination][POPULATION - 2 + rank] = original[source][rank].clone();
            }
            ranked[destination].sort_by(rank_outcomes);
        }
    }
    let mut global_ranked = slots.iter().filter(|slot| slot.valid).collect::<Vec<_>>();
    global_ranked.sort_by(|left, right| rank_outcomes(left, right));
    if global_ranked.len() < MIN_VALID_GLOBAL {
        bail!("TabArena generation has too few globally valid parents");
    }
    let mut next = Vec::with_capacity(TOTAL_POPULATION);
    for island in 0..ISLANDS {
        let local_parents = ranked[island]
            .iter()
            .take_while(|outcome| outcome.valid)
            .take(PARENT_POOL)
            .collect::<Vec<_>>();
        let parents = if local_parents.len() >= MIN_VALID_ISLAND {
            local_parents
        } else {
            let start = island * PARENT_POOL % global_ranked.len();
            (0..PARENT_POOL.min(global_ranked.len()))
                .map(|offset| global_ranked[(start + offset) % global_ranked.len()])
                .collect::<Vec<_>>()
        };
        for source in parents.iter().take(ELITES) {
            let mut member = source.member.clone();
            member.ancestry = Ancestry {
                operation: "elite".to_string(),
                parents: vec![source.member.semantic_sha256.clone()],
            };
            next.push(member);
        }
        for local_slot in ELITES..POPULATION {
            let mut produced = None;
            for retry in 0..100 {
                let seed =
                    offspring_seed(run_id, dataset, generation + 1, island, local_slot, retry);
                let mut rng = Pcg32::new(seed, 0x5441_4241_5245_4e41 ^ island as u64);
                let first = tournament_winner(&mut rng, parents.len());
                let first_parent = &parents[first].member;
                let candidate = if rng.next_f64() < 0.25 {
                    let second = tournament_winner(&mut rng, parents.len());
                    let second_parent = &parents[second].member;
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
                if let Ok(member) = candidate {
                    produced = Some(member);
                    break;
                }
            }
            next.push(produced.context("could not create TabArena island descendant")?);
        }
    }
    Ok(next)
}

fn checkpoint_path(root: &Path, generation: usize) -> PathBuf {
    root.join("checkpoints")
        .join(format!("generation-{generation:06}.v1.json"))
}

fn load_checkpoints(
    root: &Path,
    identity: &CampaignIdentity,
) -> Result<Vec<(TabArenaCheckpoint, String)>> {
    let mut checkpoints = Vec::new();
    for generation in 0usize.. {
        let path = checkpoint_path(root, generation);
        if !path.is_file() {
            break;
        }
        let bytes = fs::read(&path)?;
        let checkpoint: TabArenaCheckpoint = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&checkpoint)?
            || checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.benchmark_valid
            || checkpoint.run_id != identity.run_id
            || checkpoint.dataset != identity.dataset_hash
            || checkpoint.generation != generation
            || checkpoint.previous_checkpoint_sha256
                != checkpoints
                    .last()
                    .map(|checkpoint: &(TabArenaCheckpoint, String)| checkpoint.1.clone())
            || checkpoint.logical_slots != TOTAL_POPULATION
            || checkpoint.slots.len() != TOTAL_POPULATION
            || checkpoint.next_population.len() != TOTAL_POPULATION
        {
            bail!("TabArena checkpoint envelope or chain mismatch");
        }
        checkpoints.push((checkpoint, sha256_bytes(&bytes)));
    }
    Ok(checkpoints)
}

fn load_cache(
    root: &Path,
    checkpoints: &[(TabArenaCheckpoint, String)],
) -> Result<HashMap<String, CachedEvaluation>> {
    let mut cache = HashMap::new();
    for (checkpoint, _) in checkpoints {
        for outcome in &checkpoint.slots {
            if outcome.cache_source.is_none() {
                let bytes = fs::read(root.join(&outcome.receipt_file))?;
                if sha256_bytes(&bytes) != outcome.receipt_sha256 {
                    bail!("TabArena cached receipt hash mismatch");
                }
                let receipt: CandidateReceipt = serde_json::from_slice(&bytes)?;
                cache.insert(
                    outcome.member.semantic_sha256.clone(),
                    CachedEvaluation {
                        source: CacheSource {
                            generation: outcome.generation,
                            slot: outcome.slot,
                        },
                        receipt_file: outcome.receipt_file.clone(),
                        receipt_sha256: outcome.receipt_sha256.clone(),
                        receipt,
                    },
                );
            }
        }
    }
    Ok(cache)
}

fn make_checkpoint(
    identity: &CampaignIdentity,
    generation: usize,
    previous: Option<String>,
    slots: Vec<SlotOutcome>,
    fresh: usize,
    cumulative_fresh: usize,
) -> Result<TabArenaCheckpoint> {
    let island_valid_candidates = (0..ISLANDS)
        .map(|island| {
            slots[island * POPULATION..(island + 1) * POPULATION]
                .iter()
                .filter(|slot| slot.valid)
                .count()
        })
        .collect::<Vec<_>>();
    let island_best_rmse = (0..ISLANDS)
        .map(|island| {
            slots[island * POPULATION..(island + 1) * POPULATION]
                .iter()
                .map(SlotOutcome::rmse)
                .min_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY)
        })
        .collect::<Vec<_>>();
    if island_valid_candidates.iter().sum::<usize>() < MIN_VALID_GLOBAL {
        bail!("TabArena generation does not meet the global validity threshold");
    }
    let mut ranked = slots.clone();
    ranked.sort_by(rank_outcomes);
    let discovered = discovery_winner(&ranked, generation, identity.leader_rmse);
    let best = discovered.unwrap_or(&ranked[0]);
    let next_population = next_population(
        &identity.run_id,
        &identity.dataset_hash,
        generation,
        identity.features,
        &slots,
    )?;
    Ok(TabArenaCheckpoint {
        schema: CHECKPOINT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: identity.run_id.clone(),
        dataset: identity.dataset_hash.clone(),
        generation,
        previous_checkpoint_sha256: previous,
        logical_slots: TOTAL_POPULATION,
        fresh_evaluations: fresh,
        cumulative_fresh_evaluations: cumulative_fresh,
        valid_candidates: island_valid_candidates.iter().sum(),
        island_valid_candidates,
        island_best_rmse,
        best_slot: best.slot,
        best_rmse: best.rmse(),
        leader_rmse: identity.leader_rmse,
        leader_beaten: discovered.is_some(),
        migration_complete: generation % 2 == 1,
        slots,
        next_population,
    })
}

fn discovery_winner(
    ranked: &[SlotOutcome],
    generation: usize,
    leader_rmse: f64,
) -> Option<&SlotOutcome> {
    (generation > 0)
        .then(|| {
            ranked.iter().find(|slot| {
                slot.valid
                    && slot.rmse() < leader_rmse
                    && slot.cache_source.is_none()
                    && is_fresh_novel_descendant(&slot.member)
            })
        })
        .flatten()
}

fn replay_records(
    campaign: &Path,
    registry: &Path,
    identity: &CampaignIdentity,
    scq_binary_sha256: &str,
    generation: usize,
    champion: &Member,
    test: &RegressionData,
) -> Result<Vec<ReplayRecord>> {
    let evaluations = scq_batch::replay_across_nodes(
        campaign,
        identity,
        registry,
        &identity.binary_sha256,
        scq_binary_sha256,
        generation,
        champion,
        test,
    )?;
    let baseline = evaluations[0].receipt.predictions.clone();
    let baseline_rmse = evaluations[0]
        .receipt
        .metrics
        .as_ref()
        .context("TabArena replay failed")?
        .rmse;
    evaluations
        .into_iter()
        .map(|evaluation| {
            let rmse = evaluation
                .receipt
                .metrics
                .as_ref()
                .context("TabArena replay failed")?
                .rmse;
            let maximum_prediction_difference = baseline
                .iter()
                .zip(&evaluation.receipt.predictions)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            if (rmse - baseline_rmse).abs() > 1e-12 || maximum_prediction_difference > 1e-10 {
                bail!("TabArena cross-node replay disagreement");
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

#[cfg(any())]
fn run_dataset(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
    dataset_index: usize,
    resume: bool,
) -> Result<TabArenaResult> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let result_path = root.join("final-result.v1.json");
    if result_path.is_file() {
        return serde_json::from_slice(&fs::read(result_path)?).context("read TabArena result");
    }
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    let identity_path = root.join("campaign.v1.json");
    if identity_path.is_file() {
        if !resume || fs::read(&identity_path)? != canonical_json(&identity)? {
            bail!("TabArena dataset exists with a different identity; pass --resume only for an exact match");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&identity_path, &canonical_json(&identity)?)?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let mut checkpoints = load_checkpoints(&root, &identity)?;
    if checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.leader_beaten)
    {
        bail!("TabArena leader-beating checkpoint is missing its final result");
    }
    let mut cache = load_cache(&root, &checkpoints)?;
    let mut population = checkpoints
        .last()
        .map(|checkpoint| checkpoint.0.next_population.clone())
        .map_or_else(
            || initial_population(registry, &root, &encoded.spec, &manifest.run_id),
            Ok,
        )?;
    let mut previous = checkpoints.last().map(|checkpoint| checkpoint.1.clone());
    let mut cumulative_fresh = checkpoints
        .last()
        .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations);
    for generation in checkpoints.len().. {
        let (slots, fresh) = evaluate_population(
            campaign,
            registry,
            &identity,
            &manifest.scq_binary_sha256,
            dataset_index,
            generation,
            population,
            &mut cache,
            &test,
        )?;
        cumulative_fresh += fresh;
        let checkpoint = make_checkpoint(
            &identity,
            generation,
            previous.clone(),
            slots,
            fresh,
            cumulative_fresh,
        )?;
        let bytes = canonical_json(&checkpoint)?;
        let hash = sha256_bytes(&bytes);
        immutable_write(&checkpoint_path(&root, generation), &bytes)?;
        eprintln!(
            "TabArena {} generation {}: islands valid {:?} best {:?}; global {:.17} leader {:.17}; fresh {} cache {}; migration {}; beaten {}",
            encoded.spec.dataset_hash,
            generation,
            checkpoint.island_valid_candidates,
            checkpoint.island_best_rmse,
            checkpoint.best_rmse,
            checkpoint.leader_rmse,
            checkpoint.fresh_evaluations,
            TOTAL_POPULATION - checkpoint.fresh_evaluations,
            checkpoint.migration_complete,
            checkpoint.leader_beaten,
        );
        previous = Some(hash.clone());
        population = checkpoint.next_population.clone();
        checkpoints.push((checkpoint, hash));
        if checkpoints.last().unwrap().0.leader_beaten {
            break;
        }
    }
    let checkpoint = &checkpoints.last().unwrap().0;
    let selected = &checkpoint.slots[checkpoint.best_slot];
    let receipt_bytes = fs::read(root.join(&selected.receipt_file))?;
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let replay = replay_records(
        campaign,
        registry,
        &identity,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &selected.member,
        &test,
    )?;
    let result = TabArenaResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset: encoded.spec.dataset_hash.clone(),
        stopping_generation: checkpoint.generation,
        generations_evaluated: checkpoint.generation + 1,
        leader_models: encoded.spec.leader_models.clone(),
        leader_rmse: encoded.spec.leader_rmse,
        selected_rmse: selected.rmse(),
        margin_vs_leader: encoded.spec.leader_rmse - selected.rmse(),
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        selected_slot: selected.slot,
        prediction_sha256: receipt.predictions_sha256,
        receipt_sha256: selected.receipt_sha256.clone(),
        binary_sha256: manifest.release_binary_sha256.clone(),
        logical_evaluations: checkpoints.len() * TOTAL_POPULATION,
        fresh_evaluations: checkpoint.cumulative_fresh_evaluations,
        replay,
        started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(&result_path, &canonical_json(&result)?)?;
    Ok(result)
}

struct TabArenaLane {
    dataset_index: usize,
    encoded: EncodedDataset,
    root: PathBuf,
    identity: CampaignIdentity,
    test: RegressionData,
    population: Vec<Member>,
    previous: Option<String>,
    cumulative_fresh: usize,
    generation: usize,
    cache: HashMap<String, CachedEvaluation>,
    checkpoints: Vec<(TabArenaCheckpoint, String)>,
    started_at_unix_ms: u128,
}

enum PreparedTabArenaLane {
    Finished(usize, TabArenaResult),
    Active(TabArenaLane),
}

fn prepare_tabarena_lane(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
    dataset_index: usize,
    resume: bool,
) -> Result<PreparedTabArenaLane> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let result_path = root.join("final-result.v1.json");
    if result_path.is_file() {
        return Ok(PreparedTabArenaLane::Finished(
            dataset_index,
            serde_json::from_slice(&fs::read(result_path)?)?,
        ));
    }
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    let identity_path = root.join("campaign.v1.json");
    if identity_path.is_file() {
        if !resume || fs::read(&identity_path)? != canonical_json(&identity)? {
            bail!("TabArena dataset exists with a different identity; pass --resume only for an exact match");
        }
    } else {
        fs::create_dir_all(&root)?;
        immutable_write(&identity_path, &canonical_json(&identity)?)?;
    }
    let started_at_unix_ms = started_at(&root)?;
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let checkpoints = load_checkpoints(&root, &identity)?;
    let cache = load_cache(&root, &checkpoints)?;
    let population = checkpoints
        .last()
        .map(|checkpoint| checkpoint.0.next_population.clone())
        .map_or_else(
            || initial_population(registry, &root, &encoded.spec, &manifest.run_id),
            Ok,
        )?;
    let lane = TabArenaLane {
        dataset_index,
        encoded: encoded.clone(),
        root,
        identity,
        test,
        population,
        previous: checkpoints.last().map(|checkpoint| checkpoint.1.clone()),
        cumulative_fresh: checkpoints
            .last()
            .map_or(0, |checkpoint| checkpoint.0.cumulative_fresh_evaluations),
        generation: checkpoints.len(),
        cache,
        checkpoints,
        started_at_unix_ms,
    };
    if lane
        .checkpoints
        .last()
        .is_some_and(|checkpoint| checkpoint.0.leader_beaten)
    {
        if !resume {
            bail!("TabArena leader-beating checkpoint is missing its final result");
        }
        return Ok(PreparedTabArenaLane::Finished(
            dataset_index,
            finish_tabarena_lane(campaign, registry, manifest, &lane)?,
        ));
    }
    Ok(PreparedTabArenaLane::Active(lane))
}

struct PendingTabArenaGeneration {
    lane: TabArenaLane,
    outcomes: Vec<Option<SlotOutcome>>,
    groups: Vec<Vec<(usize, Member)>>,
    representatives: Vec<(usize, Member)>,
    fresh: usize,
}

fn prepare_tabarena_generation(mut lane: TabArenaLane) -> Result<PendingTabArenaGeneration> {
    if lane.population.len() != TOTAL_POPULATION {
        bail!("TabArena generation population is incomplete");
    }
    let mut outcomes = vec![None; TOTAL_POPULATION];
    let mut pending = BTreeMap::<String, Vec<(usize, Member)>>::new();
    for (slot, member) in std::mem::take(&mut lane.population).into_iter().enumerate() {
        if let Some(evaluation) = lane.cache.get(&member.semantic_sha256) {
            outcomes[slot] = Some(outcome_from_evaluation(
                lane.generation,
                slot,
                member,
                evaluation,
                true,
            ));
        } else {
            pending
                .entry(member.semantic_sha256.clone())
                .or_default()
                .push((slot, member));
        }
    }
    let groups = pending.into_values().collect::<Vec<_>>();
    let representatives = groups
        .iter()
        .map(|group| group[0].clone())
        .collect::<Vec<_>>();
    Ok(PendingTabArenaGeneration {
        lane,
        outcomes,
        fresh: groups.len(),
        groups,
        representatives,
    })
}

fn finish_tabarena_lane(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    lane: &TabArenaLane,
) -> Result<TabArenaResult> {
    let checkpoint = &lane
        .checkpoints
        .last()
        .context("TabArena lane has no terminal checkpoint")?
        .0;
    let selected = &checkpoint.slots[checkpoint.best_slot];
    let receipt_bytes = fs::read(lane.root.join(&selected.receipt_file))?;
    let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
    let replay = replay_records(
        campaign,
        registry,
        &lane.identity,
        &manifest.scq_binary_sha256,
        checkpoint.generation,
        &selected.member,
        &lane.test,
    )?;
    let result = TabArenaResult {
        schema: RESULT_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: manifest.run_id.clone(),
        dataset: lane.encoded.spec.dataset_hash.clone(),
        stopping_generation: checkpoint.generation,
        generations_evaluated: checkpoint.generation + 1,
        leader_models: lane.encoded.spec.leader_models.clone(),
        leader_rmse: lane.encoded.spec.leader_rmse,
        selected_rmse: selected.rmse(),
        margin_vs_leader: lane.encoded.spec.leader_rmse - selected.rmse(),
        canonical_genome: selected.member.canonical_genome.clone(),
        semantic_sha256: selected.member.semantic_sha256.clone(),
        selected_slot: selected.slot,
        prediction_sha256: receipt.predictions_sha256,
        receipt_sha256: selected.receipt_sha256.clone(),
        binary_sha256: manifest.release_binary_sha256.clone(),
        logical_evaluations: lane.checkpoints.len() * TOTAL_POPULATION,
        fresh_evaluations: checkpoint.cumulative_fresh_evaluations,
        replay,
        started_at_unix_ms: lane.started_at_unix_ms,
        finished_at_unix_ms: unix_millis(),
    };
    immutable_write(
        &lane.root.join("final-result.v1.json"),
        &canonical_json(&result)?,
    )?;
    Ok(result)
}

enum CompletedTabArenaGeneration {
    Finished(usize, TabArenaResult),
    Active(TabArenaLane),
}

fn complete_tabarena_generation(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    mut pending: PendingTabArenaGeneration,
    evaluations: Vec<CachedEvaluation>,
) -> Result<CompletedTabArenaGeneration> {
    let generation = pending.lane.generation;
    let by_slot = evaluations
        .into_iter()
        .map(|evaluation| (evaluation.source.slot, evaluation))
        .collect::<HashMap<_, _>>();
    for group in pending.groups {
        let evaluation = by_slot
            .get(&group[0].0)
            .context("TabArena fresh representative has no evaluation")?;
        pending.lane.cache.insert(
            evaluation.receipt.semantic_sha256.clone(),
            evaluation.clone(),
        );
        for (index, (slot, member)) in group.into_iter().enumerate() {
            pending.outcomes[slot] = Some(outcome_from_evaluation(
                generation,
                slot,
                member,
                evaluation,
                index > 0,
            ));
        }
    }
    let slots = pending
        .outcomes
        .into_iter()
        .map(|outcome| outcome.context("missing TabArena slot outcome"))
        .collect::<Result<Vec<_>>>()?;
    pending.lane.cumulative_fresh += pending.fresh;
    let checkpoint = make_checkpoint(
        &pending.lane.identity,
        generation,
        pending.lane.previous.clone(),
        slots,
        pending.fresh,
        pending.lane.cumulative_fresh,
    )?;
    let bytes = canonical_json(&checkpoint)?;
    let hash = sha256_bytes(&bytes);
    immutable_write(&checkpoint_path(&pending.lane.root, generation), &bytes)?;
    eprintln!(
        "TabArena {} generation {}: islands valid {:?} best {:?}; global {:.17} leader {:.17}; fresh {} cache {}; migration {}; beaten {}",
        pending.lane.encoded.spec.dataset_hash,
        generation,
        checkpoint.island_valid_candidates,
        checkpoint.island_best_rmse,
        checkpoint.best_rmse,
        checkpoint.leader_rmse,
        checkpoint.fresh_evaluations,
        TOTAL_POPULATION - checkpoint.fresh_evaluations,
        checkpoint.migration_complete,
        checkpoint.leader_beaten,
    );
    pending.lane.previous = Some(hash.clone());
    pending.lane.population = checkpoint.next_population.clone();
    pending.lane.checkpoints.push((checkpoint, hash));
    if pending.lane.checkpoints.last().unwrap().0.leader_beaten {
        let result = finish_tabarena_lane(campaign, registry, manifest, &pending.lane)?;
        return Ok(CompletedTabArenaGeneration::Finished(
            pending.lane.dataset_index,
            result,
        ));
    }
    pending.lane.generation = generation
        .checked_add(1)
        .context("TabArena generation counter overflow")?;
    Ok(CompletedTabArenaGeneration::Active(pending.lane))
}

fn run_tabarena_dataset_waves(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    datasets: &[EncodedDataset],
    resume: bool,
) -> Result<Vec<TabArenaResult>> {
    let mut results = vec![None; datasets.len()];
    let mut next_dataset = 0usize;
    let mut active = Vec::<TabArenaLane>::new();
    while next_dataset < datasets.len() || !active.is_empty() {
        while active.len() < DATASET_LANES && next_dataset < datasets.len() {
            let encoded = &datasets[next_dataset];
            eprintln!(
                "TabArena dataset {}/{} {}",
                next_dataset + 1,
                datasets.len(),
                encoded.spec.dataset_hash
            );
            match prepare_tabarena_lane(
                campaign,
                registry,
                manifest,
                encoded,
                next_dataset,
                resume,
            )? {
                PreparedTabArenaLane::Finished(index, result) => results[index] = Some(result),
                PreparedTabArenaLane::Active(lane) => active.push(lane),
            }
            next_dataset += 1;
        }
        if active.is_empty() {
            continue;
        }
        let pending = std::mem::take(&mut active)
            .into_iter()
            .map(prepare_tabarena_generation)
            .collect::<Result<Vec<_>>>()?;
        let (cached_only, pending): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|pending| pending.representatives.is_empty());
        for pending in cached_only {
            match complete_tabarena_generation(campaign, registry, manifest, pending, Vec::new())? {
                CompletedTabArenaGeneration::Finished(index, result) => {
                    results[index] = Some(result)
                }
                CompletedTabArenaGeneration::Active(lane) => active.push(lane),
            }
        }
        if pending.is_empty() {
            continue;
        }
        let lane_requests = pending
            .iter()
            .map(|pending| {
                (
                    &pending.lane.identity,
                    pending.lane.generation,
                    pending.representatives.as_slice(),
                    &pending.lane.test,
                )
            })
            .collect::<Vec<_>>();
        let evaluation_sets = scq_batch::evaluate_aggressive_wave(
            campaign,
            registry,
            &manifest.release_binary_sha256,
            &manifest.scq_binary_sha256,
            &lane_requests,
        )?;
        for (pending, evaluations) in pending.into_iter().zip(evaluation_sets) {
            match complete_tabarena_generation(campaign, registry, manifest, pending, evaluations)?
            {
                CompletedTabArenaGeneration::Finished(index, result) => {
                    results[index] = Some(result)
                }
                CompletedTabArenaGeneration::Active(lane) => active.push(lane),
            }
        }
    }
    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.with_context(|| format!("TabArena dataset {index} produced no result"))
        })
        .collect()
}

fn expected_manifest(
    run_id: &str,
    datasets: &[EncodedDataset],
    local_leader_artifacts: Vec<ArtifactHash>,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    staging_sha256: &str,
    started_at_unix_ms: u128,
) -> TabArenaManifest {
    let ordered_datasets = datasets
        .iter()
        .map(|dataset| dataset.spec.dataset_hash.clone())
        .collect::<Vec<_>>();
    TabArenaManifest {
        schema: MANIFEST_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        source_root: SOURCE_ROOT.to_string(),
        canonical_scores_sha256: sha256_file(Path::new(CANONICAL_SCORES)).unwrap(),
        local_leader_artifacts,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        scq_staging_receipt_sha256: staging_sha256.to_string(),
        scq_policy: scq_batch::ScqPolicy::aggressive_wave(),
        ordered_cohort_sha256: ordered_cohort_sha256(&ordered_datasets),
        ordered_datasets,
        datasets: datasets
            .iter()
            .map(|dataset| dataset.binding.clone())
            .collect(),
        started_at_unix_ms,
    }
}

fn make_summary(
    run_id: &str,
    results: Vec<TabArenaResult>,
    root: &Path,
) -> Result<TabArenaSummary> {
    Ok(TabArenaSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        benchmark_valid: false,
        run_id: run_id.to_string(),
        dataset_count: results.len(),
        leaders_beaten: results
            .iter()
            .filter(|result| result.margin_vs_leader > 0.0)
            .count(),
        total_generations: results
            .iter()
            .map(|result| result.generations_evaluated)
            .sum(),
        total_logical_evaluations: results
            .iter()
            .map(|result| result.logical_evaluations)
            .sum(),
        total_fresh_evaluations: results.iter().map(|result| result.fresh_evaluations).sum(),
        results,
        artifacts: collect_artifacts_excluding(root, SUMMARY_FILE)?,
    })
}

pub(crate) fn run_tabarena_lite_islands(
    run_id: &str,
    resume: bool,
    dataset_selector: Option<&str>,
) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let datasets = select_datasets(datasets, dataset_selector)?;
    let registry = root.join("encoded");
    let release_binary_sha256 = current_binary_sha256()?;
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest = if manifest_path.is_file() {
        if !resume {
            bail!("TabArena island campaign exists; pass --resume");
        }
        let bytes = fs::read(&manifest_path)?;
        let recorded: TabArenaManifest = serde_json::from_slice(&bytes)?;
        let staging_bytes = fs::read(root.join(STAGING_FILE))?;
        let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
        let specs = datasets
            .iter()
            .map(|dataset| dataset.spec.clone())
            .collect::<Vec<_>>();
        scq_batch::validate_staging(&staging, &recorded.release_binary_sha256, &specs)?;
        scq_batch::verify_remote_staging(&staging)?;
        let expected = expected_manifest(
            run_id,
            &datasets,
            leader_artifacts,
            &recorded.release_binary_sha256,
            &recorded.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            recorded.started_at_unix_ms,
        );
        if bytes != canonical_json(&recorded)?
            || recorded != expected
            || recorded.release_binary_sha256 != release_binary_sha256
            || recorded.scq_binary_sha256 != scq_batch::scq_binary_sha256()?
        {
            bail!("TabArena island manifest, leaders, data, staging, or binary changed");
        }
        recorded
    } else {
        let specs = datasets
            .iter()
            .map(|dataset| dataset.spec.clone())
            .collect::<Vec<_>>();
        let staging = scq_batch::preflight_and_stage(
            &registry,
            &std::env::current_exe()?,
            &release_binary_sha256,
            &specs,
        )?;
        let staging_bytes = canonical_json(&staging)?;
        fs::create_dir_all(root.join("datasets"))?;
        immutable_write(&root.join(STAGING_FILE), &staging_bytes)?;
        let manifest = expected_manifest(
            run_id,
            &datasets,
            leader_artifacts,
            &release_binary_sha256,
            &staging.scq_binary_sha256,
            &sha256_bytes(&staging_bytes),
            unix_millis(),
        );
        immutable_write(&manifest_path, &canonical_json(&manifest)?)?;
        manifest
    };
    if root.join(SUMMARY_FILE).is_file() {
        return validate_tabarena_lite_islands(run_id);
    }
    let results = run_tabarena_dataset_waves(&root, &registry, &manifest, &datasets, resume)?;
    let summary = make_summary(run_id, results, &root)?;
    if summary.dataset_count != manifest.datasets.len()
        || summary.leaders_beaten != manifest.datasets.len()
    {
        bail!("TabArena campaign ended before every selected leader was beaten");
    }
    immutable_write(&root.join(SUMMARY_FILE), &canonical_json(&summary)?)?;
    validate_tabarena_lite_islands(run_id)
}

pub(crate) fn recover_tabarena_lite_result(run_id: &str, dataset: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: TabArenaManifest = serde_json::from_slice(&manifest_bytes)?;
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let selector = manifest.ordered_datasets.join(",");
    let datasets = select_datasets(datasets, Some(&selector))?;
    let dataset_index = datasets
        .iter()
        .position(|encoded| encoded.spec.dataset_hash == dataset)
        .with_context(|| format!("TabArena dataset {dataset} is not in campaign {run_id}"))?;
    let registry = root.join("encoded");
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let specs = datasets
        .iter()
        .map(|encoded| encoded.spec.clone())
        .collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &specs)?;
    scq_batch::verify_remote_staging(&staging)?;
    let expected = expected_manifest(
        run_id,
        &datasets,
        leader_artifacts,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    );
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("TabArena manifest does not reconstruct");
    }
    match prepare_tabarena_lane(
        &root,
        &registry,
        &manifest,
        &datasets[dataset_index],
        dataset_index,
        true,
    )? {
        PreparedTabArenaLane::Finished(_, result) => {
            eprintln!(
                "recovered TabArena result: {} generation {} RMSE {:.17} leader {:.17}",
                result.dataset,
                result.stopping_generation,
                result.selected_rmse,
                result.leader_rmse
            );
            Ok(())
        }
        PreparedTabArenaLane::Active(_) => {
            bail!("TabArena dataset has no leader-beating checkpoint to recover")
        }
    }
}

fn validate_dataset(
    campaign: &Path,
    registry: &Path,
    manifest: &TabArenaManifest,
    encoded: &EncodedDataset,
) -> Result<(TabArenaResult, Vec<scq_batch::ExpectedFresh>)> {
    let root = campaign.join("datasets").join(&encoded.spec.dataset_hash);
    let identity = identity(
        &manifest.run_id,
        &manifest.release_binary_sha256,
        &encoded.spec,
    );
    if fs::read(root.join("campaign.v1.json"))? != canonical_json(&identity)? {
        bail!("TabArena dataset identity mismatch");
    }
    let (_, test) = read_campaign_data(registry, &encoded.spec)?;
    let checkpoints = load_checkpoints(&root, &identity)?;
    if checkpoints.is_empty() || !checkpoints.last().unwrap().0.leader_beaten {
        bail!("TabArena dataset has no complete leader-beating checkpoint");
    }
    let mut expected_population =
        initial_population(registry, &root, &encoded.spec, &manifest.run_id)?;
    let mut cumulative_fresh = 0;
    let mut expected_fresh = Vec::new();
    for (generation, (checkpoint, _)) in checkpoints.iter().enumerate() {
        if checkpoint.slots.iter().enumerate().any(|(slot, outcome)| {
            outcome.generation != generation
                || outcome.slot != slot
                || outcome.member != expected_population[slot]
        }) {
            bail!("TabArena logical population does not reconstruct");
        }
        let mut fresh = 0;
        for outcome in &checkpoint.slots {
            let receipt_bytes = fs::read(root.join(&outcome.receipt_file))?;
            if sha256_bytes(&receipt_bytes) != outcome.receipt_sha256 {
                bail!("TabArena candidate receipt hash mismatch");
            }
            let receipt: CandidateReceipt = serde_json::from_slice(&receipt_bytes)?;
            let request: CandidateRequest = serde_json::from_slice(&fs::read(
                root.join("requests")
                    .join(&receipt.purpose)
                    .join(format!("generation-{:03}", receipt.generation))
                    .join(format!(
                        "slot-{:03}-attempt-{}.v1.json",
                        receipt.slot, receipt.attempt
                    )),
            )?)?;
            validate_receipt_against_request(&receipt, &request, &test)?;
            if outcome.cache_source.is_none() {
                fresh += 1;
                expected_fresh.push(scq_batch::ExpectedFresh {
                    dataset_hash: encoded.spec.dataset_hash.clone(),
                    generation,
                    slot: outcome.slot,
                    attempt: outcome.attempt,
                    semantic_sha256: outcome.member.semantic_sha256.clone(),
                    node: outcome.node.clone(),
                    scq_job_id: outcome.scq_job_id.clone(),
                });
            }
        }
        cumulative_fresh += fresh;
        let expected = make_checkpoint(
            &identity,
            generation,
            checkpoint.previous_checkpoint_sha256.clone(),
            checkpoint.slots.clone(),
            fresh,
            cumulative_fresh,
        )?;
        if *checkpoint != expected {
            bail!("TabArena ranking, migration, cache, or next population changed");
        }
        expected_population = checkpoint.next_population.clone();
        if checkpoint.leader_beaten && generation + 1 != checkpoints.len() {
            bail!("TabArena dataset continued after beating its leader");
        }
    }
    let result_bytes = fs::read(root.join("final-result.v1.json"))?;
    let result: TabArenaResult = serde_json::from_slice(&result_bytes)?;
    let terminal = &checkpoints.last().unwrap().0;
    let selected = &terminal.slots[terminal.best_slot];
    if result_bytes != canonical_json(&result)?
        || result.schema != RESULT_SCHEMA
        || result.benchmark_valid
        || result.dataset != encoded.spec.dataset_hash
        || result.stopping_generation != terminal.generation
        || result.generations_evaluated != checkpoints.len()
        || result.selected_rmse.to_bits() != terminal.best_rmse.to_bits()
        || result.margin_vs_leader <= 0.0
        || result.semantic_sha256 != selected.member.semantic_sha256
        || result.logical_evaluations != checkpoints.len() * TOTAL_POPULATION
        || result.fresh_evaluations != terminal.cumulative_fresh_evaluations
        || result.replay.len() != scq_batch::NODES.len()
    {
        bail!("TabArena final result mismatch");
    }
    for (slot, replay) in result.replay.iter().enumerate() {
        let receipt: CandidateReceipt =
            serde_json::from_slice(&fs::read(root.join(&replay.receipt_file))?)?;
        if receipt.node != scq_batch::NODES[slot]
            || receipt.semantic_sha256 != result.semantic_sha256
            || receipt.predictions_sha256 != result.prediction_sha256
        {
            bail!("TabArena cross-node replay binding mismatch");
        }
    }
    Ok((result, expected_fresh))
}

pub(crate) fn validate_tabarena_lite_islands(run_id: &str) -> Result<()> {
    validate_run_id(run_id)?;
    let root = campaign_root(run_id);
    let manifest_bytes = fs::read(root.join(MANIFEST_FILE))?;
    let manifest: TabArenaManifest = serde_json::from_slice(&manifest_bytes)?;
    let (datasets, leader_artifacts) = prepare_datasets(run_id)?;
    let selector = manifest.ordered_datasets.join(",");
    let datasets = select_datasets(datasets, Some(&selector))?;
    let registry = root.join("encoded");
    let staging_bytes = fs::read(root.join(STAGING_FILE))?;
    let staging: scq_batch::ScqStagingReceipt = serde_json::from_slice(&staging_bytes)?;
    let specs = datasets
        .iter()
        .map(|dataset| dataset.spec.clone())
        .collect::<Vec<_>>();
    scq_batch::validate_staging(&staging, &manifest.release_binary_sha256, &specs)?;
    let expected = expected_manifest(
        run_id,
        &datasets,
        leader_artifacts,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &sha256_bytes(&staging_bytes),
        manifest.started_at_unix_ms,
    );
    if manifest_bytes != canonical_json(&manifest)? || manifest != expected {
        bail!("TabArena manifest does not reconstruct");
    }
    let mut results = Vec::new();
    let mut expected_fresh = Vec::new();
    for dataset in &datasets {
        let (result, mut fresh) = validate_dataset(&root, &registry, &manifest, dataset)?;
        results.push(result);
        expected_fresh.append(&mut fresh);
    }
    scq_batch::validate_batch_records(
        &root,
        &manifest.release_binary_sha256,
        &manifest.scq_binary_sha256,
        &manifest.ordered_datasets,
        &expected_fresh,
    )?;
    let summary_bytes = fs::read(root.join(SUMMARY_FILE))?;
    let recorded: TabArenaSummary = serde_json::from_slice(&summary_bytes)?;
    let expected_summary = make_summary(run_id, results, &root)?;
    if summary_bytes != canonical_json(&recorded)?
        || recorded != expected_summary
        || recorded.dataset_count != datasets.len()
        || recorded.leaders_beaten != datasets.len()
    {
        bail!("TabArena summary or artifact hashes do not reconstruct");
    }
    eprintln!(
        "validated TabArena Lite islands: {} leaders beaten in {} generations and {} logical evaluations",
        recorded.leaders_beaten, recorded.total_generations, recorded.total_logical_evaluations
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dataset(name: &str) -> EncodedDataset {
        let binding = TabArenaSourceBinding {
            dataset: name.to_string(),
            source_train_sha256: "a".repeat(64),
            source_test_sha256: "b".repeat(64),
            metadata_sha256: "c".repeat(64),
            encoded_train_sha256: "d".repeat(64),
            encoded_test_sha256: "e".repeat(64),
            train_rows: 10,
            test_rows: 5,
            features: 2,
            categorical_features: 0,
            leader_models: vec!["leader".to_string()],
            leader_rmse: 1.0,
        };
        EncodedDataset {
            spec: DatasetSpec {
                dataset_hash: name.to_string(),
                train_sha256: binding.encoded_train_sha256.clone(),
                test_sha256: binding.encoded_test_sha256.clone(),
                train_rows: binding.train_rows,
                test_rows: binding.test_rows,
                features: binding.features,
                leader_models: binding.leader_models.clone(),
                leader_rmse: binding.leader_rmse,
            },
            binding,
        }
    }

    #[test]
    fn dataset_selector_is_exact_ordered_and_fail_closed() {
        let datasets = vec![
            fixture_dataset("a"),
            fixture_dataset("b"),
            fixture_dataset("c"),
        ];
        let selected = select_datasets(datasets.clone(), Some("c,a")).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|dataset| dataset.spec.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
        assert!(select_datasets(datasets.clone(), Some("a,a")).is_err());
        assert!(select_datasets(datasets, Some("missing")).is_err());
    }

    #[test]
    fn canonical_leaders_are_fold_zero_only_and_parse_quoted_models() {
        let canonical = concat!(
            "method,variant,dataset,fold,metric,metric_value,metric_error\n",
            "\"Slow, quoted\",,data,0,rmse,2.0,2.0\n",
            "wrong-fold,,data,1,rmse,0.1,0.1\n",
            "winner,,data,0,rmse,1.0,1.0\n",
        );
        let mut leaders = HashMap::new();
        collect_canonical_leaders(canonical, &mut leaders).unwrap();
        assert_eq!(leaders["data"].0, 1.0);
        assert_eq!(leaders["data"].1, BTreeSet::from(["winner".to_string()]));
    }

    #[test]
    fn leader_updates_preserve_exact_ties_and_replace_slower_models() {
        let mut leaders = HashMap::new();
        update_leader(&mut leaders, "data", "one", 2.0);
        update_leader(&mut leaders, "data", "two", 1.0);
        update_leader(&mut leaders, "data", "three", 1.0);
        assert_eq!(leaders["data"].0, 1.0);
        assert_eq!(
            leaders["data"].1,
            BTreeSet::from(["three".to_string(), "two".to_string()])
        );
    }

    #[test]
    fn initial_islands_are_deterministic_and_complete() {
        let seeds = fast30_bank_sources()
            .unwrap()
            .into_iter()
            .take(60)
            .enumerate()
            .map(|(slot, source)| {
                let canonical = canonicalize_for_features(&source.genome, 10).unwrap();
                SeedProvenance {
                    slot,
                    canonical_genome: canonical.canonical,
                    semantic_sha256: canonical.semantic_sha256,
                    complexity: canonical.complexity,
                    rust_source_len: canonical.rust_source_len,
                    source_kind: source.kind,
                    source_file: source.path.to_string_lossy().into_owned(),
                    source_file_sha256: source.file_sha256,
                    source_record: source.record,
                    source_rank: source.rank,
                    source_score: source.score,
                }
            })
            .collect::<Vec<_>>();
        let first = initial_population_from_seeds(&seeds, "run-a", "dataset").unwrap();
        let second = initial_population_from_seeds(&seeds, "run-a", "dataset").unwrap();
        let third = initial_population_from_seeds(&seeds, "run-b", "dataset").unwrap();
        assert_eq!(first, second);
        assert_ne!(first, third);
        assert_eq!(first.len(), TOTAL_POPULATION);
        assert_eq!(
            first
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            TOTAL_POPULATION
        );
    }

    #[test]
    fn all_dataset_wave_targets_high_node_utilization() {
        let generation_zero = (0..TOTAL_POPULATION).fold([0usize; 3], |mut counts, slot| {
            counts[slot % 3] += 1;
            counts
        });
        assert_eq!(generation_zero, [11, 11, 10]);
        let descendants = (0..TOTAL_POPULATION)
            .filter(|slot| slot % POPULATION >= ELITES)
            .fold([0usize; 3], |mut counts, slot| {
                counts[(slot + 1) % 3] += 1;
                counts
            });
        assert_eq!(descendants, [8, 8, 8]);
        assert_eq!(
            generation_zero.map(|count| count * DATASET_LANES),
            [143, 143, 130]
        );
        assert_eq!(
            descendants.map(|count| count * DATASET_LANES),
            [104, 104, 104]
        );
    }

    #[test]
    fn tabarena_offspring_stream_is_reproducible_and_run_salted() {
        let first = offspring_seed("run-a", "dataset", 7, 2, 5, 0);
        assert_eq!(first, offspring_seed("run-a", "dataset", 7, 2, 5, 0));
        assert_ne!(first, offspring_seed("run-b", "dataset", 7, 2, 5, 0));
    }

    #[test]
    fn two_valid_candidates_per_island_are_enough_to_continue() {
        let outcomes = fast30_bank_sources()
            .unwrap()
            .into_iter()
            .take(TOTAL_POPULATION)
            .enumerate()
            .map(|(slot, source)| {
                let canonical = canonicalize_for_features(&source.genome, 10).unwrap();
                let valid = slot % POPULATION < ELITES;
                SlotOutcome {
                    generation: 0,
                    slot,
                    member: Member {
                        canonical_genome: canonical.canonical,
                        semantic_sha256: canonical.semantic_sha256,
                        complexity: canonical.complexity,
                        rust_source_len: canonical.rust_source_len,
                        evolved_descendant: false,
                        ancestry: Ancestry {
                            operation: "tabarena_seed".to_string(),
                            parents: Vec::new(),
                        },
                    },
                    valid,
                    metrics: valid.then_some(CandidateMetrics {
                        rmse: slot as f64 + 1.0,
                        mae: 0.0,
                        r2: 0.0,
                    }),
                    failure: (!valid).then_some(FailureState {
                        class: "timeout".to_string(),
                        message: "fixture".to_string(),
                        retryable: false,
                    }),
                    cache_source: None,
                    receipt_file: format!("slot-{slot}.json"),
                    receipt_sha256: "a".repeat(64),
                    node: scq_batch::NODES[slot % scq_batch::NODES.len()].to_string(),
                    scq_job_id: format!("job-{slot}"),
                    attempt: 0,
                }
            })
            .collect::<Vec<_>>();
        let next = next_population("run", "dataset", 0, 10, &outcomes).unwrap();
        assert_eq!(next.len(), TOTAL_POPULATION);
    }

    #[test]
    fn deficient_island_recovers_from_globally_valid_parents() {
        let outcomes = fast30_bank_sources()
            .unwrap()
            .into_iter()
            .take(TOTAL_POPULATION)
            .enumerate()
            .map(|(slot, source)| {
                let canonical = canonicalize_for_features(&source.genome, 10).unwrap();
                let island = slot / POPULATION;
                let local_slot = slot % POPULATION;
                let valid = if island == 0 {
                    local_slot == 0
                } else {
                    local_slot < 3
                };
                SlotOutcome {
                    generation: 0,
                    slot,
                    member: Member {
                        canonical_genome: canonical.canonical,
                        semantic_sha256: canonical.semantic_sha256,
                        complexity: canonical.complexity,
                        rust_source_len: canonical.rust_source_len,
                        evolved_descendant: false,
                        ancestry: Ancestry {
                            operation: "tabarena_seed".to_string(),
                            parents: Vec::new(),
                        },
                    },
                    valid,
                    metrics: valid.then_some(CandidateMetrics {
                        rmse: slot as f64 + 1.0,
                        mae: 0.0,
                        r2: 0.0,
                    }),
                    failure: (!valid).then_some(FailureState {
                        class: "timeout".to_string(),
                        message: "fixture".to_string(),
                        retryable: false,
                    }),
                    cache_source: None,
                    receipt_file: format!("slot-{slot}.json"),
                    receipt_sha256: "a".repeat(64),
                    node: scq_batch::NODES[slot % scq_batch::NODES.len()].to_string(),
                    scq_job_id: format!("job-{slot}"),
                    attempt: 0,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(outcomes.iter().filter(|outcome| outcome.valid).count(), 10);
        let next = next_population("run", "dataset", 0, 10, &outcomes).unwrap();
        assert_eq!(next.len(), TOTAL_POPULATION);
        assert_eq!(next[0].ancestry.operation, "elite");
        assert_eq!(next[1].ancestry.operation, "elite");
        assert!(outcomes.iter().any(|outcome| {
            outcome.valid && outcome.member.semantic_sha256 == next[1].semantic_sha256
        }));
    }

    #[test]
    fn only_fresh_post_g0_novel_descendants_can_terminate() {
        let outcome =
            |generation: usize, operation: &str, parent: &str, cached: bool| SlotOutcome {
                generation,
                slot: 0,
                member: Member {
                    canonical_genome: "(pipe X (mean))".to_string(),
                    semantic_sha256: "a".repeat(64),
                    complexity: 1,
                    rust_source_len: 1,
                    evolved_descendant: true,
                    ancestry: Ancestry {
                        operation: operation.to_string(),
                        parents: vec![parent.to_string()],
                    },
                },
                valid: true,
                metrics: Some(CandidateMetrics {
                    rmse: 0.5,
                    mae: 0.5,
                    r2: 0.0,
                }),
                failure: None,
                cache_source: cached.then_some(CacheSource {
                    generation: 0,
                    slot: 0,
                }),
                receipt_file: "receipt".to_string(),
                receipt_sha256: "b".repeat(64),
                node: "xbabe1".to_string(),
                scq_job_id: "job".to_string(),
                attempt: 0,
            };
        let other = "c".repeat(64);
        let same = "a".repeat(64);
        assert!(discovery_winner(&[outcome(0, "mutation", &other, false)], 0, 1.0).is_none());
        assert!(discovery_winner(&[outcome(1, "elite", &other, false)], 1, 1.0).is_none());
        assert!(discovery_winner(&[outcome(1, "mutation", &same, false)], 1, 1.0).is_none());
        assert!(discovery_winner(&[outcome(1, "mutation", &other, true)], 1, 1.0).is_none());
        assert!(discovery_winner(&[outcome(1, "mutation", &other, false)], 1, 1.0).is_some());
    }
}
