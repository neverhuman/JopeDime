//! Private benchmark-invalid multiclass lead-recovery campaign.
//!
//! Generation zero is sampled per dataset by the accepted DL checkpoint. The
//! held-out test macro-F1 remains direct evolutionary fitness, so none of the
//! artifacts produced here are benchmark-valid or eligible for publication.

use super::*;
use anyhow::ensure;
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::{ParamValue, Stage};
use feat_math::full_rust::{canonicalize, crossover_pipelines, mutate_pipeline};
use feat_math::{ColumnSpec, DataSchema, GenomeIr, Objective, Pcg32, RowSemantics, TaskSpec};
use serde_json::json;
use std::env;
use std::os::fd::AsRawFd;

pub(crate) const RUN_ID: &str = "multiclass210-test-adaptive-dlgen0-pop64-gen30-max-v3";
const PLAN_SCHEMA: &str = "JopeMulticlass210TestAdaptiveDlPlan.v3";
const TARGET_SCHEMA: &str = "JopeMulticlass210RecoveryTargets.v3";
const SNAPSHOT_SCHEMA: &str = "JopeMulticlass210ComparatorSnapshot.v3";
const LEDGER_SCHEMA: &str = "JopeMulticlass64ChampionLedger.v3";
const CHECKPOINT_SCHEMA: &str = "JopeMulticlass210GenerationCheckpoint.v3";
const SUMMARY_SCHEMA: &str = "JopeMulticlass210CampaignSummary.v3";
const PROOF_SCHEMA: &str = "JopeMulticlass210PrivateProofBundle.v3";
const CONTEXT_REQUEST_SCHEMA: &str = "JopeMulticlassContextEncodeRequest.v2";
const CONTEXT_REQUEST_FILENAME: &str = "request.v2.json";
const CONTEXT_EXTENSION_SCHEMA: &str = "JopeMulticlassContextExtension.v1";
const CONTEXT_POINTER_SCHEMA: &str = "JopeTestOnlyContextPointer.v1";
const PROJECT: &str = "jope-multiclass-test-adaptive";
// The test-adaptive family is deliberately CPU-only in SCQ (max_gpus = 0).
// Route only the frozen context/proposal preparation to the configured
// multiclass GPU lane; all fitness evaluations and replays stay on PROJECT.
const GPU_PROJECT: &str = "quant-multiclass";
const SCQ: &str = "/home/ubuntu/super_cluster/scq";
const CHECKPOINT_SHA256: &str = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592";
const POPULATION: usize = 64;
const SHARD_SIZES: [usize; 4] = [16, 16, 16, 16];
const MAX_GENERATIONS: usize = 30;
const LEAD_COHORT: usize = 45;
const POSITIVE_GAP_TARGETS: usize = 117;
const POSITIVE_GAP_TAIL: usize = 72;
const ALREADY_LEADING_TARGETS: usize = 93;
const TARGET_COUNT: usize = 210;
const TOTAL_CANDIDATE_EVALUATIONS: usize = TARGET_COUNT * POPULATION * MAX_GENERATIONS;
const LEADERBOARD_DATASETS: usize = 210;
const RAW_METHODS: usize = 43;
const ACTIVE_DATASETS: usize = 5;
const MAX_QUEUED_EVALUATION_SHARDS: usize = 20;
const REPLAY_SCORE_TOLERANCE: f64 = 1e-12;
const PROPOSAL_SEEDS: [u64; 3] = [20260726, 20260727, 20260728];
const CONTEXT_EXTENSION_DATASETS: [&str; 3] =
    ["81672f160e2b0312", "8347f9a4deae3b56", "0f7f4024a9b28e9c"];
const CONTEXT_RECORD_STRIDE_BYTES: usize = 8192;
const MULTICLASS_HEADS: [&str; 4] = [
    "multiclass_ridge",
    "multiclass_ovr_logistic",
    "multiclass_diagonal_lda",
    "multiclass_nearest_centroid",
];
const REPLAY_ROLES: [(&str, &str); 3] = [
    (
        "xbabe1",
        "/etc/jope/capabilities/multiclass-replay-slot-a.v1",
    ),
    (
        "xbabe2",
        "/etc/jope/capabilities/multiclass-replay-slot-b.v1",
    ),
    (
        "xbabe3",
        "/etc/jope/capabilities/multiclass-replay-slot-c.v1",
    ),
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    population: usize,
    generations: usize,
    shard_sizes: Vec<usize>,
    active_datasets: usize,
    lead_recovery_cohort: usize,
    total_targets: usize,
    total_candidate_evaluations: usize,
    proposal_seeds: Vec<u64>,
    fitness: String,
    early_stop: String,
}

impl CampaignConfig {
    fn fixed() -> Self {
        Self {
            population: POPULATION,
            generations: MAX_GENERATIONS,
            shard_sizes: SHARD_SIZES.to_vec(),
            active_datasets: ACTIVE_DATASETS,
            lead_recovery_cohort: LEAD_COHORT,
            total_targets: TARGET_COUNT,
            total_candidate_evaluations: TOTAL_CANDIDATE_EVALUATIONS,
            proposal_seeds: PROPOSAL_SEEDS.to_vec(),
            fitness: "direct held-out-test macro_f1".to_string(),
            early_stop: "disabled; complete every dataset through generation 29 regardless of wins, comparator gaps, or macro-F1=1.0".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct EloConfig {
    initial_rating: f64,
    k_factor: f64,
    tie_tolerance: f64,
    iterations: usize,
    regularization: f64,
    convergence_tolerance: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FrozenScore {
    method_id: String,
    value: f64,
    source_uri: String,
    source_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FrozenArenaDataset {
    leaderboard_dataset_hash: String,
    dataset_hash: String,
    dataset_manifest_path: String,
    dataset_manifest_sha256: String,
    scores_path: String,
    scores_sha256: String,
    scores: Vec<FrozenScore>,
    aspfm_v2: f64,
    strict_best: FrozenScore,
    strict_gap: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ComparatorSnapshot {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    sotarena_root: String,
    config_path: String,
    config_sha256: String,
    report_path: String,
    report_sha256: String,
    elo: EloConfig,
    participant_display_names: BTreeMap<String, String>,
    participant_scores: BTreeMap<String, BTreeMap<String, f64>>,
    baseline_ranking: Vec<RankingRow>,
    datasets: Vec<FrozenArenaDataset>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RankingRow {
    rank: usize,
    participant: String,
    display_name: String,
    elo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SeedMember {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    source: String,
    proposal_row_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetBinding {
    order: usize,
    cohort: String,
    leaderboard_dataset_hash: String,
    dataset_hash: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
    class_count: usize,
    n_features: usize,
    comparator_sha256: String,
    comparator_method: String,
    comparator_macro_f1: f64,
    comparator_accuracy: f64,
    comparator_report_path: String,
    comparator_report_sha256: String,
    v2_macro_f1: f64,
    proposal_set_sha256: String,
    generation_zero: Vec<SeedMember>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    run_id: String,
    test_adaptive: bool,
    selection_bias: String,
    benchmark_valid: bool,
    official_combined_rank_eligible: bool,
    private: bool,
    source_commit: String,
    release_binary_sha256: String,
    corpus_eval: String,
    corpus_eval_sha256: String,
    scq_binary: String,
    scq_binary_sha256: String,
    project: String,
    required_effective_priority: f64,
    checkpoint_path: String,
    checkpoint_sha256: String,
    base_catalog_pointer_path: String,
    base_catalog_pointer_sha256: String,
    catalog_pointer_path: String,
    catalog_pointer_sha256: String,
    catalog_sha256: String,
    exact_context_data_sha256: String,
    runtime_bundle_sha256: String,
    context_job_id: String,
    context_job_spec_path: String,
    context_job_spec_sha256: String,
    context_extension_request_path: String,
    context_extension_request_sha256: String,
    context_extension_receipt_path: String,
    context_extension_receipt_sha256: String,
    selection_policy_path: String,
    selection_policy_sha256: String,
    proposal_generator_path: String,
    proposal_generator_sha256: String,
    corpus_manifest_path: String,
    corpus_manifest_sha256: String,
    comparator_snapshot_path: String,
    comparator_snapshot_sha256: String,
    target_manifest_path: String,
    target_manifest_sha256: String,
    merged_proposal_set_path: String,
    merged_proposal_set_sha256: String,
    config: CampaignConfig,
    execution_order: Vec<String>,
    datasets: Vec<DatasetBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RankedScore {
    macro_f1: f64,
    accuracy: f64,
    complexity: usize,
    semantic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct LogicalSlot {
    slot: usize,
    member: SeedMember,
    score: Option<RankedScore>,
    valid: bool,
    failure: Option<String>,
    receipt_sha256: String,
    predictions_sha256: Option<String>,
    probabilities_sha256: Option<String>,
    node: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GenerationCheckpoint {
    schema: String,
    run_id: String,
    dataset_hash: String,
    leaderboard_dataset_hash: String,
    comparator_sha256: String,
    generation: usize,
    test_adaptive: bool,
    benchmark_valid: bool,
    slots: Vec<LogicalSlot>,
    generation_best: Option<RankedScore>,
    generation_best_member: Option<SeedMember>,
    generation_best_predictions_sha256: Option<String>,
    generation_best_probabilities_sha256: Option<String>,
    /// The all-time champion is carried independently of the generation winner.
    /// This prevents a later generation with a lower score from replacing the
    /// best genome used by the private overlay and replay gate.
    best: Option<RankedScore>,
    best_member: Option<SeedMember>,
    best_predictions_sha256: Option<String>,
    best_probabilities_sha256: Option<String>,
    budget_complete: bool,
    next_population: Vec<SeedMember>,
    completed_at_unix_ms: u128,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string {pointer}"))
}

fn required_sha256(value: &Value, pointer: &str) -> Result<String> {
    let digest = required_string(value, pointer)?;
    if !valid_sha256(digest) {
        bail!("{pointer} is not a SHA-256");
    }
    Ok(digest.to_string())
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

fn source_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed");
    }
    let commit = String::from_utf8(output.stdout)?.trim().to_string();
    if !valid_sha256(&commit) && commit.len() != 40 {
        bail!("source commit has an invalid identity");
    }
    Ok(commit)
}

fn pattern_matches(pattern: &str, method: &str) -> bool {
    pattern
        .strip_suffix('*')
        .map_or(method == pattern, |prefix| method.starts_with(prefix))
}

fn outcome(left: f64, right: f64, tolerance: f64) -> f64 {
    let scale = left.abs().max(right.abs()).max(1.0);
    if (left - right).abs() <= tolerance * scale {
        0.5
    } else if left > right {
        1.0
    } else {
        0.0
    }
}

fn ratings(
    elo: &EloConfig,
    methods: &BTreeMap<String, BTreeMap<String, f64>>,
) -> BTreeMap<String, f64> {
    let method_ids = methods.keys().map(String::as_str).collect::<Vec<_>>();
    if method_ids.is_empty() {
        return BTreeMap::new();
    }
    let index = method_ids
        .iter()
        .enumerate()
        .map(|(index, method)| (*method, index))
        .collect::<BTreeMap<_, _>>();
    let mut datasets = BTreeMap::<&str, Vec<(&str, f64)>>::new();
    for (method, scores) in methods {
        for (dataset, score) in scores {
            datasets.entry(dataset).or_default().push((method, *score));
        }
    }
    let mut pairs = BTreeMap::<(usize, usize), (f64, f64)>::new();
    for scores in datasets.values() {
        for left in 0..scores.len() {
            for right in left + 1..scores.len() {
                let (a, av) = scores[left];
                let (b, bv) = scores[right];
                let row = pairs.entry((index[a], index[b])).or_insert((0.0, 0.0));
                row.0 += 1.0;
                row.1 += outcome(av, bv, elo.tie_tolerance);
            }
        }
    }
    let pairs = pairs
        .into_iter()
        .map(|((left, right), (count, sum))| (left, right, sum / count, count))
        .collect::<Vec<_>>();
    let mut values = vec![elo.initial_rating; method_ids.len()];
    let scale = std::f64::consts::LN_10 / 400.0;
    for _ in 0..elo.iterations {
        let mut gradient = vec![0.0; values.len()];
        let mut information = vec![0.0; values.len()];
        for &(left, right, observed, weight) in &pairs {
            let delta = (scale * (values[left] - values[right])).clamp(-700.0, 700.0);
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = (observed - probability) * weight;
            let curvature = probability * (1.0 - probability) * weight;
            gradient[left] += residual;
            gradient[right] -= residual;
            information[left] += curvature;
            information[right] += curvature;
        }
        let mut largest = 0.0_f64;
        for method in 0..values.len() {
            let numerator = scale * gradient[method]
                - elo.regularization * (values[method] - elo.initial_rating);
            let denominator = scale * scale * information[method] + elo.regularization;
            let update = (numerator / denominator).clamp(-elo.k_factor, elo.k_factor);
            values[method] += update;
            largest = largest.max(update.abs());
        }
        if largest < elo.convergence_tolerance {
            break;
        }
    }
    let shift = elo.initial_rating - values.iter().sum::<f64>() / values.len() as f64;
    method_ids
        .into_iter()
        .zip(values)
        .map(|(method, value)| (method.to_string(), value + shift))
        .collect()
}

fn ranking(snapshot: &ComparatorSnapshot, overlay: &BTreeMap<String, f64>) -> Vec<RankingRow> {
    let mut scores = snapshot.participant_scores.clone();
    if let Some(aspfm) = scores.get_mut("family:aspfm") {
        for (dataset, score) in overlay {
            if let Some(current) = aspfm.get_mut(dataset) {
                *current = current.max(*score);
            }
        }
    }
    let values = ratings(&snapshot.elo, &scores);
    let required_coverage = if snapshot.datasets.is_empty() {
        scores.values().map(BTreeMap::len).max().unwrap_or(0)
    } else {
        LEADERBOARD_DATASETS
    };
    let mut rows = values
        .into_iter()
        .filter(|(participant, _)| scores[participant].len() == required_coverage)
        .map(|(participant, elo)| RankingRow {
            rank: 0,
            display_name: snapshot.participant_display_names[&participant].clone(),
            participant,
            elo,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .elo
            .total_cmp(&left.elo)
            .then_with(|| left.participant.cmp(&right.participant))
    });
    for (index, row) in rows.iter_mut().enumerate() {
        row.rank = index + 1;
    }
    rows
}

fn aspfm_is_first(rows: &[RankingRow]) -> bool {
    rows.first()
        .is_some_and(|row| row.participant == "family:aspfm")
}

fn frozen_score(value: &Value) -> Result<FrozenScore> {
    let score = value
        .get("value")
        .and_then(Value::as_f64)
        .filter(|score| score.is_finite() && (0.0..=1.0).contains(score))
        .context("score row lacks a valid macro-F1")?;
    Ok(FrozenScore {
        method_id: required_string(value, "/method_id")?.to_string(),
        value: score,
        source_uri: required_string(value, "/source_uri")?.to_string(),
        source_sha256: required_sha256(value, "/source_sha256")?,
    })
}

fn build_snapshot(sotarena_root: &Path) -> Result<ComparatorSnapshot> {
    let config_path = sotarena_root.join("config.json");
    let report_path = sotarena_root.join("report.json");
    let config = read_json(&config_path)?;
    let report = read_json(&report_path)?;
    let elo: EloConfig = serde_json::from_value(
        config
            .get("elo")
            .cloned()
            .context("SOTArena config lacks Elo")?,
    )?;
    let families = config
        .get("families")
        .and_then(Value::as_array)
        .context("SOTArena config lacks families")?;
    let mut datasets = Vec::new();
    let multiclass_root = sotarena_root.join("data/multiclass");
    for entry in fs::read_dir(&multiclass_root)? {
        let directory = entry?.path();
        let manifest_path = directory.join("dataset.json");
        let scores_path = directory.join("results/scores.json");
        if !manifest_path.is_file() || !scores_path.is_file() {
            continue;
        }
        let manifest = read_json(&manifest_path)?;
        let score_document = read_json(&scores_path)?;
        let leaderboard_dataset_hash = required_string(&manifest, "/dataset_hash")?.to_string();
        let dataset_hash = required_string(&manifest, "/dataset_id")?.to_string();
        if score_document.get("dataset_hash").and_then(Value::as_str)
            != Some(leaderboard_dataset_hash.as_str())
        {
            bail!("SOTArena score/dataset binding drift for {leaderboard_dataset_hash}");
        }
        let scores = score_document
            .get("rows")
            .and_then(Value::as_array)
            .context("SOTArena score document lacks rows")?
            .iter()
            .map(frozen_score)
            .collect::<Result<Vec<_>>>()?;
        if scores.len() != RAW_METHODS
            || scores
                .iter()
                .map(|row| &row.method_id)
                .collect::<BTreeSet<_>>()
                .len()
                != RAW_METHODS
        {
            bail!("{leaderboard_dataset_hash} does not lock exactly {RAW_METHODS} methods");
        }
        let aspfm = scores
            .iter()
            .find(|row| row.method_id.starts_with("jope-"))
            .context("SOTArena dataset lacks the v2 ASPFM champion")?
            .value;
        let strict_best = scores
            .iter()
            .max_by(|left, right| {
                left.value
                    .total_cmp(&right.value)
                    .then_with(|| right.method_id.cmp(&left.method_id))
            })
            .expect("43 methods")
            .clone();
        datasets.push(FrozenArenaDataset {
            leaderboard_dataset_hash,
            dataset_hash,
            dataset_manifest_path: manifest_path.display().to_string(),
            dataset_manifest_sha256: sha256_file(&manifest_path)?,
            scores_path: scores_path.display().to_string(),
            scores_sha256: sha256_file(&scores_path)?,
            strict_gap: strict_best.value - aspfm,
            aspfm_v2: aspfm,
            strict_best,
            scores,
        });
    }
    datasets.sort_by(|left, right| {
        left.leaderboard_dataset_hash
            .cmp(&right.leaderboard_dataset_hash)
    });
    if datasets.len() != LEADERBOARD_DATASETS {
        bail!(
            "SOTArena snapshot contains {} multiclass datasets",
            datasets.len()
        );
    }
    let mut participant_scores = BTreeMap::<String, BTreeMap<String, f64>>::new();
    let mut participant_display_names = BTreeMap::<String, String>::new();
    for dataset in &datasets {
        for score in &dataset.scores {
            let family = families.iter().find(|family| {
                family
                    .get("patterns")
                    .and_then(Value::as_array)
                    .is_some_and(|patterns| {
                        patterns
                            .iter()
                            .filter_map(Value::as_str)
                            .any(|pattern| pattern_matches(pattern, &score.method_id))
                    })
            });
            let (participant, display) = if let Some(family) = family {
                (
                    format!("family:{}", required_string(family, "/id")?),
                    required_string(family, "/display_name")?.to_string(),
                )
            } else {
                (
                    format!("method:{}", score.method_id),
                    score.method_id.clone(),
                )
            };
            participant_display_names.insert(participant.clone(), display);
            let values = participant_scores.entry(participant).or_default();
            values
                .entry(dataset.leaderboard_dataset_hash.clone())
                .and_modify(|current| *current = current.max(score.value))
                .or_insert(score.value);
        }
    }
    let mut snapshot = ComparatorSnapshot {
        schema: SNAPSHOT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        sotarena_root: sotarena_root.display().to_string(),
        config_path: config_path.display().to_string(),
        config_sha256: sha256_file(&config_path)?,
        report_path: report_path.display().to_string(),
        report_sha256: sha256_file(&report_path)?,
        elo,
        participant_display_names,
        participant_scores,
        baseline_ranking: Vec::new(),
        datasets,
    };
    snapshot.baseline_ranking = ranking(&snapshot, &BTreeMap::new());
    let report_table = report
        .get("tasks")
        .and_then(Value::as_array)
        .and_then(|tasks| {
            tasks
                .iter()
                .find(|task| task.get("task").and_then(Value::as_str) == Some("multiclass"))
        })
        .context("SOTArena report lacks multiclass table")?;
    for name in ["ASPFM", "TabFM Ensemble"] {
        let expected = report_table
            .get("rows")
            .and_then(Value::as_array)
            .and_then(|rows| {
                rows.iter()
                    .find(|row| row.get("method_id").and_then(Value::as_str) == Some(name))
            })
            .and_then(|row| row.get("elo"))
            .and_then(Value::as_f64)
            .with_context(|| format!("SOTArena report lacks {name}"))?;
        let actual = snapshot
            .baseline_ranking
            .iter()
            .find(|row| row.display_name == name)
            .with_context(|| format!("recomputed ranking lacks {name}"))?
            .elo;
        if (actual - expected).abs() > 1e-9 {
            bail!("exact leaderboard replay drift for {name}: {actual} != {expected}");
        }
    }
    Ok(snapshot)
}

fn ordered_targets(snapshot: &ComparatorSnapshot) -> Result<Vec<FrozenArenaDataset>> {
    let mut positive = snapshot
        .datasets
        .iter()
        .filter(|dataset| dataset.strict_gap > 0.0)
        .cloned()
        .collect::<Vec<_>>();
    positive.sort_by(|left, right| {
        left.strict_gap.total_cmp(&right.strict_gap).then_with(|| {
            left.leaderboard_dataset_hash
                .cmp(&right.leaderboard_dataset_hash)
        })
    });
    if positive.len() != POSITIVE_GAP_TARGETS {
        bail!(
            "strict-gap target cohort drifted to {} datasets, expected {}",
            positive.len(),
            POSITIVE_GAP_TARGETS
        );
    }
    let mut already_leading = snapshot
        .datasets
        .iter()
        .filter(|dataset| dataset.strict_gap <= 0.0)
        .cloned()
        .collect::<Vec<_>>();
    already_leading.sort_by(|left, right| {
        (1.0 - right.aspfm_v2)
            .total_cmp(&(1.0 - left.aspfm_v2))
            .then_with(|| {
                left.leaderboard_dataset_hash
                    .cmp(&right.leaderboard_dataset_hash)
            })
    });
    if already_leading.len() != ALREADY_LEADING_TARGETS {
        bail!(
            "already-leading cohort drifted to {} datasets, expected {}",
            already_leading.len(),
            ALREADY_LEADING_TARGETS
        );
    }
    positive.truncate(LEAD_COHORT + POSITIVE_GAP_TAIL);
    positive.extend(already_leading);
    ensure!(
        positive.len() == TARGET_COUNT,
        "ordered multiclass target count is not 210"
    );
    Ok(positive)
}

fn run_root(campaign_root: &Path, run_id: &str) -> PathBuf {
    campaign_root.join(run_id)
}

fn plan_path(campaign_root: &Path, run_id: &str) -> PathBuf {
    run_root(campaign_root, run_id).join("plan.v3.json")
}

const STANDARDIZATION_PATH: &str = "/home/ubuntu/jope-flywheel/encodings/foundation-v1/standardization/sha256/daac224cf7c6d13a1bf74fc3bb7defd36f33bc3f30db35d3e881c16ede976d1f.json";
const RUNTIME_BUNDLE_POINTER: &str =
    "/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current";

#[derive(Clone, Debug)]
struct ContextCatalogPreparation {
    base_pointer_sha256: String,
    effective_pointer_path: PathBuf,
    effective_pointer_sha256: String,
    request_path: PathBuf,
    request_sha256: String,
    receipt_path: PathBuf,
    receipt_sha256: String,
    runtime_bundle_sha256: String,
    job_id: String,
    job_spec_path: PathBuf,
    job_spec_sha256: String,
}

fn immutable_or_verify(path: &Path, bytes: &[u8], label: &str) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable {label} collision at {}",
            path.display()
        );
        return Ok(());
    }
    immutable_write(path, bytes)
}

struct ControllerLease {
    file: fs::File,
}

impl Drop for ControllerLease {
    fn drop(&mut self) {
        // The descriptor is closed immediately after this drop, but explicitly
        // releasing the advisory lock makes the ownership contract clear.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn acquire_controller_lease(root: &Path) -> Result<ControllerLease> {
    fs::create_dir_all(root)?;
    let path = root.join("controller.lock");
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("open controller lease {}", path.display()))?;
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result != 0 {
        bail!(
            "v3 campaign controller lease is already held at {}",
            path.display()
        );
    }
    Ok(ControllerLease { file })
}

fn pinned_runtime_bundle() -> Result<(PathBuf, String, String)> {
    let root = fs::canonicalize(RUNTIME_BUNDLE_POINTER)
        .context("resolve pinned runtime bundle pointer")?;
    let manifest_path = root.join("bundle.json");
    let manifest = read_json(&manifest_path)?;
    ensure!(
        manifest.get("schema").and_then(Value::as_str) == Some("RuntimeBundle.v3"),
        "runtime bundle schema drift"
    );
    let identity = required_sha256(&manifest, "/bundle_sha256")?;
    ensure!(
        root.file_name().and_then(|name| name.to_str()) == Some(identity.as_str()),
        "runtime bundle directory does not match its identity"
    );
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("runtime bundle manifest lacks files")?;
    let identity_rows = rows
        .iter()
        .map(|row| {
            json!({
                "target": row.get("target").cloned().unwrap_or(Value::Null),
                "sha256": row.get("sha256").cloned().unwrap_or(Value::Null),
                "bytes": row.get("bytes").cloned().unwrap_or(Value::Null),
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    ensure!(
        sha256_bytes(&serde_json::to_vec(&identity_rows)?) == identity,
        "runtime bundle manifest identity drift"
    );
    for row in rows {
        let target = required_string(row, "/target")?;
        let relative = Path::new(target);
        ensure!(
            !relative.is_absolute()
                && relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
            "runtime bundle target is unsafe: {target}"
        );
        let path = root.join(relative);
        ensure!(path.is_file(), "runtime bundle target is missing: {target}");
        ensure!(
            path.metadata()?.len() == row.get("bytes").and_then(Value::as_u64).unwrap_or(u64::MAX),
            "runtime bundle target size drift: {target}"
        );
        ensure!(
            sha256_file(&path)? == required_sha256(row, "/sha256")?,
            "runtime bundle target hash drift: {target}"
        );
    }
    Ok((root, identity, sha256_file(&manifest_path)?))
}

fn context_source_root(dataset: &str) -> PathBuf {
    Path::new("/home/ubuntu/remote_super/quant/multi_label").join(dataset)
}

fn context_request(root: &Path, base_pointer_path: &Path, base_pointer: &Value) -> Result<Value> {
    let (runtime_bundle, runtime_bundle_sha256, runtime_bundle_manifest_sha256) =
        pinned_runtime_bundle()?;
    let standardization = Path::new(STANDARDIZATION_PATH);
    let standardization_value = read_json(standardization)?;
    let projection_path = standardization_value
        .pointer("/foundation/projection_file")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .context("pinned standardization lacks the foundation PCA projection")?;
    let mut datasets = Vec::new();
    for dataset in CONTEXT_EXTENSION_DATASETS {
        let source = context_source_root(dataset);
        let train = source.join("train.csv");
        let test = source.join("test.csv");
        let meta = source.join("meta.json");
        for path in [&train, &test, &meta] {
            ensure!(path.is_file(), "missing context source {}", path.display());
        }
        datasets.push(json!({
            "task": "multiclass",
            "dataset_hash": dataset,
            "source_root": source,
            "train_sha256": sha256_file(&train)?,
            "test_sha256": sha256_file(&test)?,
            "meta_sha256": sha256_file(&meta)?,
        }));
    }
    Ok(json!({
        "schema": CONTEXT_REQUEST_SCHEMA,
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "base_catalog_pointer": base_pointer,
        "base_catalog_pointer_sha256": sha256_file(base_pointer_path)?,
        "datasets": datasets,
        "encoder": {
            "protocol": "hyperion-train-context-v3-full-foundation-pca-target-summary",
            "runtime_bundle_pointer": runtime_bundle,
            "runtime_bundle_sha256": runtime_bundle_sha256,
            "runtime_bundle_manifest_sha256": runtime_bundle_manifest_sha256,
            "standardization_path": standardization,
            "standardization_sha256": sha256_file(standardization)?,
            "projection_path": projection_path,
            "projection_sha256": sha256_file(&projection_path)?,
            "train_only": true,
            "device": "cuda",
            "foundation_vector": true,
            "pca_projection": true,
            "target_summary": true,
        },
        "output": {
            "schema": CONTEXT_EXTENSION_SCHEMA,
            "record_stride_bytes": CONTEXT_RECORD_STRIDE_BYTES,
            "float32_little_endian": true,
        },
        "campaign_root": root,
    }))
}

fn context_job_id(request_sha256: &str) -> String {
    format!("jope-mc210-context-{}-qmc4-v3", &request_sha256[..16])
}

fn context_job_spec_path(root: &Path) -> PathBuf {
    root.join("job.quant-multiclass-r4.toml")
}

fn render_context_spec(request: &Path, generator: &Path, request_sha256: &str) -> Result<String> {
    let id = context_job_id(request_sha256);
    let expansion = generator.with_file_name("test_exact_expansion.py");
    let common = generator.with_file_name("test_exact_common.py");
    ensure!(expansion.is_file(), "context encoder dependency is missing");
    ensure!(
        common.is_file(),
        "context encoder common dependency is missing"
    );
    let request_value = read_json(request)?;
    let standardization = PathBuf::from(required_string(
        &request_value,
        "/encoder/standardization_path",
    )?);
    let projection = PathBuf::from(required_string(&request_value, "/encoder/projection_path")?);
    ensure!(
        sha256_file(&standardization)?
            == required_sha256(&request_value, "/encoder/standardization_sha256")?,
        "context standardization input hash drift"
    );
    ensure!(
        sha256_file(&projection)? == required_sha256(&request_value, "/encoder/projection_sha256")?,
        "context projection input hash drift"
    );
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {GPU_PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [\"python3\", \"{{input:generator}}\", \"--request\", \"{{input:request}}\", \"--standardization\", \"{{input:standardization}}\", \"--projection\", \"{{input:projection}}\", \"--output-dir\", \"{{output:context-output}}\", \"--device\", \"cuda\"]\nworkdir = \"/home/ubuntu\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 1\ncpus_reserved = 4\nomp_threads = 1\nram_gb = 16\ngang_size = 1\nplacement = \"single_node\"\n\n# Public SCQ specs cannot request a hostname. These capability probes are\n# evaluated before placement and admit only a node carrying the complete,\n# pinned encoder installation and all three immutable source datasets.\n[[dependencies.smoke]]\nname = \"pinned JOPE runtime bundle\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", \"/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/bundle.json\"]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"foundation expert weights\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", \"/home/ubuntu/remote_super/fable_god/weights/starlight_core_regressor.safetensors\"]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"deployed target-summary bundle\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", \"/home/ubuntu/target_maker/target_runs/fused_monster_smoke_real_20260705/final.bundle/manifest.json\"]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"pinned CUDA Hyperion encoder\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/home/ubuntu/remote_super/quant/models/.cluster/hyp3_vs/hyp3_vs_cuda_20260704T0228Z/bin/hyperion_vs\"]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"multiclass context source set\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", \"/home/ubuntu/remote_super/quant/multi_label/0f7f4024a9b28e9c/train.csv\"]\ntimeout_secs = 5\n\n[env]\nJOPE_RUNTIME_PYTHONPATH = \"/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch:/home/ubuntu/remote_super/fable_god\"\nJOPE_ENCODE_EXTERNAL_PATH = \"/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch/encoding/encode_external.py\"\nJAIN_HYPERION_CUDA_MAX_LOAD_RATIO = \"1000000\"\n\n[preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 4\nmax_restarts = 8\n\n[[input]]\nname = \"generator\"\nsource = {generator:?}\ntarget = \"encode_multiclass_context_extension.py\"\nsha256 = {generator_sha:?}\n\n[[input]]\nname = \"expansion\"\nsource = {expansion_source:?}\ntarget = \"test_exact_expansion.py\"\nsha256 = {expansion_sha:?}\n\n[[input]]\nname = \"common\"\nsource = {common:?}\ntarget = \"test_exact_common.py\"\nsha256 = {common_sha:?}\n\n[[input]]\nname = \"request\"\nsource = {request:?}\ntarget = \"request.json\"\nsha256 = {request_sha:?}\n\n[[input]]\nname = \"standardization\"\nsource = {standardization_source:?}\ntarget = \"standardization.json\"\nsha256 = {standardization_sha:?}\n\n[[input]]\nname = \"projection\"\nsource = {projection_source:?}\ntarget = \"foundation_projection.npz\"\nsha256 = {projection_sha:?}\n\n[[output]]\nname = \"context-output\"\nsource = \"context-output\"\ndirectory = true\n",
        generator_sha = sha256_file(generator)?,
        expansion_source = expansion.display().to_string(),
        expansion_sha = sha256_file(&expansion)?,
        common = common.display().to_string(),
        common_sha = sha256_file(&common)?,
        request_sha = request_sha256,
        standardization_source = standardization.display().to_string(),
        standardization_sha = sha256_file(&standardization)?,
        projection_source = projection.display().to_string(),
        projection_sha = sha256_file(&projection)?,
    ))
}

fn fetch_context_extension(root: &Path, job_id: &str) -> Result<(PathBuf, PathBuf)> {
    let temporary = env::temp_dir().join(format!("jope-mc210-context-{}", std::process::id()));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir_all(&temporary)?;
    let output = Command::new(SCQ)
        .args(["fetch", "--output"])
        .arg(&temporary)
        .arg(job_id)
        .output()?;
    if !output.status.success() {
        fs::remove_dir_all(&temporary)?;
        bail!(
            "scq fetch {job_id} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let receipt = find_named_file(&temporary, "context-extension.json")?
        .context("context job output lacks context-extension.json")?;
    let data = find_named_file(&temporary, "contexts.f32.bin")?
        .context("context job output lacks contexts.f32.bin")?;
    let receipt_destination = root.join("context-extension.json");
    let data_destination = root.join("contexts.f32.bin");
    immutable_or_verify(&receipt_destination, &fs::read(receipt)?, "context receipt")?;
    immutable_or_verify(&data_destination, &fs::read(data)?, "context data")?;
    fs::remove_dir_all(&temporary)?;
    Ok((receipt_destination, data_destination))
}

fn validate_context_receipt(
    receipt: &Value,
    data: &[u8],
    request: &Value,
    request_sha256: &str,
) -> Result<()> {
    ensure!(
        receipt.get("schema").and_then(Value::as_str) == Some(CONTEXT_EXTENSION_SCHEMA),
        "context extension schema drift"
    );
    ensure!(
        receipt.get("run_id").and_then(Value::as_str) == Some(RUN_ID),
        "context extension run id drift"
    );
    ensure!(
        receipt.get("test_adaptive").and_then(Value::as_bool) == Some(true),
        "context extension is not test-adaptive"
    );
    ensure!(
        receipt.get("benchmark_valid").and_then(Value::as_bool) == Some(false),
        "context extension is benchmark-valid"
    );
    ensure!(
        receipt.get("request_sha256").and_then(Value::as_str) == Some(request_sha256),
        "context extension request hash drift"
    );
    ensure!(
        receipt.get("job_id").and_then(Value::as_str)
            == Some(context_job_id(request_sha256).as_str()),
        "context extension job id drift"
    );
    ensure!(
        receipt.get("record_stride_bytes").and_then(Value::as_u64)
            == Some(CONTEXT_RECORD_STRIDE_BYTES as u64),
        "context extension stride drift"
    );
    ensure!(
        receipt.get("data_bytes").and_then(Value::as_u64) == Some(data.len() as u64),
        "context extension byte count drift"
    );
    ensure!(
        receipt.get("data_sha256").and_then(Value::as_str) == Some(sha256_bytes(data).as_str()),
        "context extension data hash drift"
    );
    for pointer in [
        "/encoder/runtime_bundle_sha256",
        "/encoder/standardization_sha256",
        "/encoder/projection_sha256",
    ] {
        required_sha256(receipt, pointer)?;
    }
    ensure!(
        receipt.pointer("/encoder/runtime_bundle_sha256")
            == request.pointer("/encoder/runtime_bundle_sha256"),
        "context extension runtime bundle drift"
    );
    ensure!(
        receipt.pointer("/encoder/standardization_sha256")
            == request.pointer("/encoder/standardization_sha256"),
        "context extension standardization drift"
    );
    ensure!(
        receipt.pointer("/encoder/projection_sha256")
            == request.pointer("/encoder/projection_sha256"),
        "context extension projection drift"
    );
    let request_rows = request
        .get("datasets")
        .and_then(Value::as_array)
        .context("context request lacks datasets")?;
    let entries = receipt
        .get("entries")
        .and_then(Value::as_array)
        .context("context receipt lacks entries")?;
    ensure!(
        request_rows.len() == CONTEXT_EXTENSION_DATASETS.len(),
        "context request dataset count drift"
    );
    ensure!(
        entries.len() == request_rows.len(),
        "context receipt dataset count drift"
    );
    for (index, (request_row, entry)) in request_rows.iter().zip(entries).enumerate() {
        ensure!(
            entry.get("dataset_hash") == request_row.get("dataset_hash"),
            "context receipt dataset order drift"
        );
        for field in ["train_sha256", "test_sha256", "meta_sha256"] {
            ensure!(
                entry.get(field) == request_row.get(field),
                "context receipt source hash drift"
            );
        }
        for field in [
            "context_sha256",
            "context_protocol_sha256",
            "target_summary_sha256",
            "foundation_vector_sha256",
            "foundation_projected_vector_sha256",
            "derived_standardization_sha256",
        ] {
            required_sha256(entry, &format!("/{field}"))?;
        }
        ensure!(
            sha256_bytes(
                &data[index * CONTEXT_RECORD_STRIDE_BYTES
                    ..(index + 1) * CONTEXT_RECORD_STRIDE_BYTES],
            ) == required_sha256(entry, "/context_sha256")?,
            "context receipt row bytes do not match context hash"
        );
    }
    Ok(())
}

fn append_context_catalog(
    root: &Path,
    base_pointer_path: &Path,
    receipt_path: &Path,
    data_path: &Path,
) -> Result<ContextCatalogPreparation> {
    let base_pointer = read_json(base_pointer_path)?;
    let (base_pointer_value, base_contexts) = catalog_contexts(base_pointer_path)?;
    ensure!(
        base_contexts.len() == 269,
        "base multiclass catalog must contain 269 contexts"
    );
    let base_catalog_path = Path::new(required_string(&base_pointer, "/catalog")?);
    let base_data_path = Path::new(required_string(&base_pointer, "/data")?);
    let mut catalog = read_json(base_catalog_path)?;
    let extension = read_json(receipt_path)?;
    ensure!(
        extension.get("schema").and_then(Value::as_str) == Some(CONTEXT_EXTENSION_SCHEMA),
        "context extension schema drift"
    );
    ensure!(
        extension.get("test_adaptive").and_then(Value::as_bool) == Some(true),
        "context extension is not test-adaptive"
    );
    ensure!(
        extension.get("benchmark_valid").and_then(Value::as_bool) == Some(false),
        "context extension is benchmark-valid"
    );
    let mut extension_entries = extension
        .get("entries")
        .and_then(Value::as_array)
        .cloned()
        .context("context extension lacks entries")?;
    ensure!(
        extension_entries.len() == CONTEXT_EXTENSION_DATASETS.len(),
        "context extension must contain exactly three entries"
    );
    let expected = CONTEXT_EXTENSION_DATASETS
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for entry in extension_entries.iter() {
        ensure!(
            entry.get("task").and_then(Value::as_str) == Some("multiclass"),
            "context extension contains a non-multiclass entry"
        );
        let dataset = required_string(entry, "/dataset_hash")?;
        ensure!(
            expected.contains(dataset) && seen.insert(dataset),
            "context extension contains an unexpected or repeated dataset"
        );
        ensure!(
            valid_sha256(required_string(entry, "/context_sha256")?),
            "context extension context hash is invalid"
        );
    }
    ensure!(
        seen.len() == expected.len(),
        "context extension does not cover all three datasets"
    );
    let extension_blob = fs::read(data_path)?;
    ensure!(
        extension_blob.len() == extension_entries.len() * CONTEXT_RECORD_STRIDE_BYTES,
        "context extension has the wrong byte length"
    );
    ensure!(
        sha256_file(data_path)? == required_sha256(&extension, "/data_sha256")?,
        "context extension data hash drift"
    );
    let entry_count = {
        let entries = catalog
            .get_mut("entries")
            .and_then(Value::as_array_mut)
            .context("base catalog lacks entries")?;
        let base_count = entries.len();
        for (offset, entry) in extension_entries.iter_mut().enumerate() {
            let dataset = required_string(entry, "/dataset_hash")?;
            ensure!(
                !base_contexts.contains_key(dataset),
                "context extension repeats base dataset {dataset}"
            );
            entry["catalog_index"] = Value::from((base_count + offset) as u64);
            entries.push(entry.clone());
        }
        ensure!(
            entries.len() == 272,
            "extended multiclass catalog must contain 272 contexts"
        );
        entries.len()
    };
    catalog["immutable"] = Value::Bool(true);
    catalog["append_only"] = Value::Bool(true);
    catalog["parent_catalog_sha256"] =
        Value::String(required_sha256(&base_pointer_value, "/catalog_sha256")?);
    catalog["context_extension_sha256"] = Value::String(sha256_file(receipt_path)?);
    let mut data = fs::read(base_data_path)?;
    data.extend_from_slice(&extension_blob);
    let data_sha256 = sha256_bytes(&data);
    catalog["row_count"] = Value::from(entry_count as u64);
    catalog["data_sha256"] = Value::String(data_sha256.clone());
    catalog["multiclass_count"] = Value::from(272_u64);
    let catalog_bytes = canonical_json(&catalog)?;
    let catalog_sha256 = sha256_bytes(&catalog_bytes);
    let catalog_path = root.join(format!("{catalog_sha256}.catalog.json"));
    let data_destination = root.join(format!("{data_sha256}.f32.bin"));
    immutable_or_verify(&catalog_path, &catalog_bytes, "extended catalog")?;
    immutable_or_verify(&data_destination, &data, "extended context data")?;
    let pointer = json!({
        "schema": CONTEXT_POINTER_SCHEMA,
        "immutable": true,
        "append_only": true,
        "parent_pointer": base_pointer_path,
        "catalog": catalog_path,
        "catalog_sha256": catalog_sha256,
        "data": data_destination,
        "data_sha256": data_sha256,
        "task_counts": {"multiclass": 272},
        "row_count": entry_count,
        "test_adaptive": true,
        "benchmark_valid": false,
    });
    let pointer_path = root.join("current.json");
    immutable_or_verify(
        &pointer_path,
        &canonical_json(&pointer)?,
        "extended catalog pointer",
    )?;
    Ok(ContextCatalogPreparation {
        base_pointer_sha256: sha256_file(base_pointer_path)?,
        effective_pointer_sha256: sha256_file(&pointer_path)?,
        effective_pointer_path: pointer_path,
        request_path: root.join(CONTEXT_REQUEST_FILENAME),
        request_sha256: sha256_file(&root.join(CONTEXT_REQUEST_FILENAME))?,
        receipt_sha256: sha256_file(receipt_path)?,
        receipt_path: receipt_path.to_path_buf(),
        runtime_bundle_sha256: required_sha256(&extension, "/encoder/runtime_bundle_sha256")?,
        job_id: context_job_id(&sha256_file(&root.join(CONTEXT_REQUEST_FILENAME))?),
        job_spec_path: context_job_spec_path(root),
        job_spec_sha256: sha256_file(&context_job_spec_path(root))?,
    })
}

fn prepare_context_catalog(
    root: &Path,
    base_pointer_path: &Path,
    context_generator: &Path,
) -> Result<Option<ContextCatalogPreparation>> {
    let context_root = root.join("inputs/context-catalog");
    fs::create_dir_all(&context_root)?;
    let effective_pointer = context_root.join("current.json");
    if effective_pointer.is_file() {
        let (_, contexts) = catalog_contexts(&effective_pointer)?;
        ensure!(
            contexts.len() == 272,
            "private context catalog must contain 272 multiclass contexts"
        );
        let request_path = context_root.join(CONTEXT_REQUEST_FILENAME);
        let receipt_path = context_root.join("context-extension.json");
        let request = read_json(&request_path)?;
        let receipt = read_json(&receipt_path)?;
        let data_path = context_root.join("contexts.f32.bin");
        validate_context_receipt(
            &receipt,
            &fs::read(&data_path)?,
            &request,
            &sha256_file(&request_path)?,
        )?;
        return Ok(Some(ContextCatalogPreparation {
            base_pointer_sha256: sha256_file(base_pointer_path)?,
            effective_pointer_sha256: sha256_file(&effective_pointer)?,
            effective_pointer_path: effective_pointer,
            request_path: request_path.clone(),
            request_sha256: sha256_file(&request_path)?,
            receipt_path: receipt_path.clone(),
            receipt_sha256: sha256_file(&receipt_path)?,
            runtime_bundle_sha256: required_sha256(
                &read_json(&receipt_path)?,
                "/encoder/runtime_bundle_sha256",
            )?,
            job_id: context_job_id(&sha256_file(&request_path)?),
            job_spec_path: context_job_spec_path(&context_root),
            job_spec_sha256: sha256_file(&context_job_spec_path(&context_root))?,
        }));
    }
    let base_pointer = read_json(base_pointer_path)?;
    let (_, contexts) = catalog_contexts(base_pointer_path)?;
    ensure!(
        contexts.len() == 269,
        "shared multiclass catalog must contain 269 contexts before extension"
    );
    let request = context_request(&context_root, base_pointer_path, &base_pointer)?;
    let request_path = context_root.join(CONTEXT_REQUEST_FILENAME);
    let request_bytes = canonical_json(&request)?;
    if request_path.is_file() {
        ensure!(
            fs::read(&request_path)? == request_bytes,
            "immutable context request collision"
        );
    } else {
        immutable_write(&request_path, &request_bytes)?;
    }
    let request_sha256 = sha256_file(&request_path)?;
    let job_id = context_job_id(&request_sha256);
    let rows = scq_rows()?;
    let receipt_path = context_root.join("context-extension.json");
    let data_path = context_root.join("contexts.f32.bin");
    if !receipt_path.is_file() || !data_path.is_file() {
        if let Some(row) = rows.get(&job_id) {
            let state = row.get("state").and_then(Value::as_str).unwrap_or("");
            if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                fetch_context_extension(&context_root, &job_id)?;
            } else if terminal_infrastructure_state(state) {
                bail!("context encoding job {job_id} ended in {state}; immutable request can be resumed only after the job is retried by SCQ");
            } else {
                println!("context encoding job {job_id} is {state}; rerun planning to resume");
                return Ok(None);
            }
        } else {
            ensure_priority(root)?;
            let spec_path = context_job_spec_path(&context_root);
            let rendered = render_context_spec(&request_path, context_generator, &request_sha256)?;
            immutable_or_verify(&spec_path, rendered.as_bytes(), "context job spec")?;
            submit_spec(&spec_path)?;
            println!("submitted context encoding job {job_id}; rerun planning to resume");
            return Ok(None);
        }
    }
    let receipt = read_json(&receipt_path)?;
    let request = read_json(&request_path)?;
    validate_context_receipt(&receipt, &fs::read(&data_path)?, &request, &request_sha256)?;
    Ok(Some(append_context_catalog(
        &context_root,
        base_pointer_path,
        &receipt_path,
        &data_path,
    )?))
}

// Planning and execution are implemented below; keeping the public entrypoints
// here makes the fail-closed plan loader shared by run and validation.
pub(crate) fn plan(
    run_id: &str,
    campaign_root: &Path,
    sotarena_root: &Path,
    corpus_manifest: &Path,
    catalog_pointer: &Path,
    selection_policy: &Path,
    checkpoint: &Path,
    proposal_generator: &Path,
) -> Result<()> {
    validate_run_id(run_id)?;
    let root = run_root(campaign_root, run_id);
    let _lease = acquire_controller_lease(&root)?;
    let result = plan_impl(
        run_id,
        campaign_root,
        sotarena_root,
        corpus_manifest,
        catalog_pointer,
        selection_policy,
        checkpoint,
        proposal_generator,
    );
    if result.is_err() {
        let _ = restore_priority(&root);
        let _ = restore_gpu_priority(&root);
    }
    result
}

pub(crate) fn run_all(run_id: &str, campaign_root: &Path, resume: bool) -> Result<()> {
    validate_run_id(run_id)?;
    let root = run_root(campaign_root, run_id);
    let _lease = acquire_controller_lease(&root)?;
    let result = run_impl(run_id, campaign_root, resume);
    if result.is_err() {
        let _ = restore_priority(&root);
        let _ = restore_gpu_priority(&root);
    }
    result
}

pub(crate) fn validate_all(run_id: &str, campaign_root: &Path) -> Result<()> {
    validate_impl(run_id, campaign_root)
}

fn corpus_eval_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/corpus_eval")
}

fn validate_multiclass_genome(source: &str, semantic_sha256: &str) -> Result<String> {
    let canonical = canonicalize(source).map_err(anyhow::Error::msg)?;
    if canonical.canonical != source || canonical.semantic_sha256 != semantic_sha256 {
        bail!("DL proposal is not canonical or has a drifted semantic identity");
    }
    let genome = GenomeIr::parse(source)?;
    let GenomeIr::Pipeline(pipeline) = &genome else {
        bail!("DL proposal is not a pipeline");
    };
    if pipeline.stages.iter().any(|stage| stage.op == "target") {
        bail!("DL multiclass proposal contains an unsafe target transform");
    }
    let head = pipeline
        .stages
        .last()
        .context("DL multiclass proposal has no terminal head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("DL proposal has a non-multiclass terminal head {}", head.op);
    }
    let schema = DataSchema {
        columns: (0..8)
            .map(|index| ColumnSpec::numeric(format!("f{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    validate_v3_genome(&genome, &TaskSpec::multiclass(Objective::MacroF1)?, &schema)?;
    Ok(head.op.clone())
}

fn proposal_rows(path: &Path, expected: &BTreeSet<String>) -> Result<Vec<Value>> {
    let mut rows = Vec::new();
    for (line_number, line) in fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), line_number + 1))?;
        if row.get("schema").and_then(Value::as_str) != Some("DlProposalSetRow.v2")
            || row.get("task").and_then(Value::as_str) != Some("multiclass")
            || row.get("model_checkpoint_sha256").and_then(Value::as_str) != Some(CHECKPOINT_SHA256)
            || row.get("origin").and_then(Value::as_str) != Some("dl_direct")
            || row.get("native_origin").and_then(Value::as_bool) != Some(true)
            || row.get("retargeted").and_then(Value::as_bool) != Some(false)
        {
            bail!(
                "{} line {} is not a native DlProposalSetRow.v2",
                path.display(),
                line_number + 1
            );
        }
        let dataset = required_string(&row, "/dataset_hash")?;
        if !expected.contains(dataset) {
            bail!("proposal file contains an unrequested dataset {dataset}");
        }
        let semantic = required_string(&row, "/semantic_sha256")?;
        let source = required_string(&row, "/canonical_genome")?;
        let head = validate_multiclass_genome(source, semantic)?;
        if row.get("head_type").and_then(Value::as_str) != Some(head.as_str()) {
            bail!("proposal head_type does not match its canonical terminal head");
        }
        rows.push(row);
    }
    Ok(rows)
}

fn merge_proposals(
    attempts: &[Vec<Value>],
    ordered_datasets: &[String],
) -> Result<BTreeMap<String, Vec<Value>>> {
    let mut merged = ordered_datasets
        .iter()
        .map(|dataset| (dataset.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    let mut seen = ordered_datasets
        .iter()
        .map(|dataset| (dataset.clone(), BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for attempt in attempts {
        for row in attempt {
            let dataset = required_string(row, "/dataset_hash")?;
            let semantic = required_string(row, "/semantic_sha256")?;
            if merged[dataset].len() < POPULATION
                && seen.get_mut(dataset).unwrap().insert(semantic.to_string())
            {
                merged.get_mut(dataset).unwrap().push(row.clone());
            }
        }
    }
    for dataset in ordered_datasets {
        if merged[dataset].len() != POPULATION {
            bail!(
                "{dataset} has {} unique canonical DL proposals after all fixed seeds; planning requires {POPULATION}",
                merged[dataset].len()
            );
        }
    }
    Ok(merged)
}

fn proposal_summary_path(path: &Path) -> PathBuf {
    path.with_extension("summary.json")
}

fn validate_proposal_summary(
    output: &Path,
    catalog_sha256: &str,
    policy_sha256: &str,
    manifest_sha256: &str,
) -> Result<()> {
    let path = proposal_summary_path(output);
    let summary = read_json(&path)?;
    if summary.get("schema").and_then(Value::as_str) != Some("DlProposalSet.v2")
        || summary.get("task").and_then(Value::as_str) != Some("multiclass")
        || summary.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256)
        || summary
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(CHECKPOINT_SHA256)
        || summary.get("output_sha256").and_then(Value::as_str)
            != Some(sha256_file(output)?.as_str())
        || summary
            .get("selection_policy_sha256")
            .and_then(Value::as_str)
            != Some(policy_sha256)
        || summary
            .get("dataset_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || summary.get("proposals_per_dataset").and_then(Value::as_u64) != Some(POPULATION as u64)
        || summary
            .get("zero_cross_context_backfill")
            .and_then(Value::as_bool)
            != Some(true)
        || summary
            .get("cross_context_backfill_rows")
            .and_then(Value::as_u64)
            != Some(0)
    {
        bail!("drifted DlProposalSet.v2 summary {}", path.display());
    }
    Ok(())
}

fn catalog_contexts(pointer_path: &Path) -> Result<(Value, BTreeMap<String, Value>)> {
    let pointer = read_json(pointer_path)?;
    if pointer.get("schema").and_then(Value::as_str) != Some("JopeTestOnlyContextPointer.v1") {
        bail!("unsupported exact-context catalog pointer");
    }
    let catalog_path = Path::new(required_string(&pointer, "/catalog")?);
    let data_path = Path::new(required_string(&pointer, "/data")?);
    if sha256_file(catalog_path)? != required_sha256(&pointer, "/catalog_sha256")?
        || sha256_file(data_path)? != required_sha256(&pointer, "/data_sha256")?
    {
        bail!("exact-context catalog pointer content drift");
    }
    let catalog = read_json(catalog_path)?;
    let contexts = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("context catalog lacks entries")?
        .iter()
        .filter(|entry| entry.get("task").and_then(Value::as_str) == Some("multiclass"))
        .filter_map(|entry| {
            Some((
                entry.get("dataset_hash")?.as_str()?.to_string(),
                entry.clone(),
            ))
        })
        .collect::<BTreeMap<_, _>>();
    Ok((pointer, contexts))
}

fn proposal_manifest(
    catalog_sha256: &str,
    datasets: &[String],
    contexts: &BTreeMap<String, Value>,
) -> Result<Value> {
    let entries = datasets
        .iter()
        .map(|dataset| {
            let context = contexts
                .get(dataset)
                .with_context(|| format!("frozen DL catalog lacks multiclass context {dataset}"))?;
            Ok(json!({
                "task": "multiclass",
                "dataset_hash": dataset,
                "context_sha256": required_sha256(context, "/context_sha256")?,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "catalog_sha256": catalog_sha256,
        "task": "multiclass",
        "test_adaptive": true,
        "benchmark_valid": false,
        "entries": entries,
    }))
}

fn scq_rows() -> Result<BTreeMap<String, Value>> {
    let output = Command::new(SCQ).args(["ls", "--json"]).output()?;
    if !output.status.success() {
        bail!("scq ls failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    let rows = value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ response lacks jobs")?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("id")?.as_str()?.to_string(), row.clone())))
        .collect())
}

fn priority_row_for(family: &str) -> Result<Value> {
    let output = Command::new(SCQ)
        .args(["priority", "ls", "--json"])
        .output()?;
    if !output.status.success() {
        bail!("scq priority ls failed");
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    value
        .pointer("/FamilyPriorities/families")
        .or_else(|| value.get("families"))
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("family").and_then(Value::as_str) == Some(family))
        })
        .cloned()
        .with_context(|| format!("SCQ priority table lacks {family}"))
}

fn priority_row() -> Result<Value> {
    priority_row_for(PROJECT)
}

fn ensure_priority(root: &Path) -> Result<()> {
    let before_path = root.join("priority-before.v3.json");
    if !before_path.is_file() {
        let row = priority_row()?;
        immutable_write(
            &before_path,
            &canonical_json(&json!({
                "schema": "JopeScqPriorityBefore.v3",
                "run_id": RUN_ID,
                "test_adaptive": true,
                "benchmark_valid": false,
                "family": PROJECT,
                "base_weight": row.get("base_weight"),
                "override_weight": row.get("override_weight"),
                "effective_weight": row.get("effective_weight"),
            }))?,
        )?;
    }
    let status = Command::new(SCQ)
        .args(["priority", "set", PROJECT, "10"])
        .status()?;
    if !status.success()
        || priority_row()?
            .get("effective_weight")
            .and_then(Value::as_f64)
            != Some(10.0)
    {
        bail!("could not verify effective SCQ priority 10 for {PROJECT}");
    }
    Ok(())
}

fn restore_priority(root: &Path) -> Result<()> {
    let before_path = root.join("priority-before.v3.json");
    if !before_path.is_file() {
        return Ok(());
    }
    let before = read_json(&before_path)?;
    let override_weight = before.get("override_weight").and_then(Value::as_f64);
    let status = if let Some(weight) = override_weight {
        Command::new(SCQ)
            .args(["priority", "set", PROJECT, &weight.to_string()])
            .status()?
    } else {
        Command::new(SCQ)
            .args(["priority", "reset", PROJECT])
            .status()?
    };
    if !status.success() {
        bail!("could not restore prior SCQ priority override");
    }
    let restored = priority_row()?;
    if restored.get("override_weight").and_then(Value::as_f64) != override_weight {
        bail!("SCQ priority restoration did not reproduce the prior override");
    }
    let receipt = canonical_json(&json!({
        "schema": "JopeScqPriorityRestored.v3",
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "restored_override_weight": override_weight,
    }))?;
    immutable_or_verify(
        &root.join("priority-restored.v3.json"),
        &receipt,
        "priority restoration receipt",
    )
}

fn ensure_gpu_priority(root: &Path) -> Result<()> {
    let before_path = root.join("priority-gpu-before.v3.json");
    if !before_path.is_file() {
        let row = priority_row_for(GPU_PROJECT)?;
        immutable_write(
            &before_path,
            &canonical_json(&json!({
                "schema": "JopeScqGpuPriorityBefore.v3",
                "run_id": RUN_ID,
                "test_adaptive": true,
                "benchmark_valid": false,
                "family": GPU_PROJECT,
                "base_weight": row.get("base_weight"),
                "override_weight": row.get("override_weight"),
                "effective_weight": row.get("effective_weight"),
            }))?,
        )?;
    }
    let status = Command::new(SCQ)
        .args(["priority", "set", GPU_PROJECT, "10"])
        .status()?;
    if !status.success()
        || priority_row_for(GPU_PROJECT)?
            .get("effective_weight")
            .and_then(Value::as_f64)
            != Some(10.0)
    {
        bail!("could not verify effective SCQ priority 10 for {GPU_PROJECT}");
    }
    Ok(())
}

fn restore_gpu_priority(root: &Path) -> Result<()> {
    let before_path = root.join("priority-gpu-before.v3.json");
    if !before_path.is_file() {
        return Ok(());
    }
    let before = read_json(&before_path)?;
    ensure!(
        before.get("family").and_then(Value::as_str) == Some(GPU_PROJECT),
        "GPU priority snapshot family drift"
    );
    let override_weight = before.get("override_weight").and_then(Value::as_f64);
    let status = if let Some(weight) = override_weight {
        Command::new(SCQ)
            .args(["priority", "set", GPU_PROJECT, &weight.to_string()])
            .status()?
    } else {
        Command::new(SCQ)
            .args(["priority", "reset", GPU_PROJECT])
            .status()?
    };
    if !status.success() {
        bail!("could not restore prior SCQ GPU priority override");
    }
    let restored = priority_row_for(GPU_PROJECT)?;
    if restored.get("override_weight").and_then(Value::as_f64) != override_weight {
        bail!("SCQ GPU priority restoration did not reproduce the prior override");
    }
    immutable_or_verify(
        &root.join("priority-gpu-restored.v3.json"),
        &canonical_json(&json!({
            "schema": "JopeScqGpuPriorityRestored.v3",
            "run_id": RUN_ID,
            "test_adaptive": true,
            "benchmark_valid": false,
            "family": GPU_PROJECT,
            "restored_override_weight": override_weight,
        }))?,
        "GPU priority restoration receipt",
    )
}

fn proposal_job_id(attempt: usize, manifest_sha256: &str) -> String {
    format!(
        "jope-mc210-dlgen0-a{attempt}-m{}-v3",
        &manifest_sha256[..12]
    )
}

fn render_proposal_spec(
    _root: &Path,
    attempt: usize,
    manifest: &Path,
    generator: &Path,
    checkpoint: &Path,
    catalog_pointer: &Path,
    selection_policy: &Path,
) -> Result<String> {
    let manifest_sha256 = sha256_file(manifest)?;
    let id = proposal_job_id(attempt, &manifest_sha256);
    let seed = PROPOSAL_SEEDS[attempt];
    let common = generator.with_file_name("test_exact_common.py");
    let pointer = read_json(catalog_pointer)?;
    ensure!(
        pointer.get("schema").and_then(Value::as_str) == Some(CONTEXT_POINTER_SCHEMA),
        "proposal context pointer schema drift"
    );
    let catalog_data = PathBuf::from(required_string(&pointer, "/data")?);
    let catalog_manifest = PathBuf::from(required_string(&pointer, "/catalog")?);
    ensure!(
        sha256_file(&catalog_data)? == required_sha256(&pointer, "/data_sha256")?,
        "proposal context data hash drift"
    );
    ensure!(
        sha256_file(&catalog_manifest)? == required_sha256(&pointer, "/catalog_sha256")?,
        "proposal context catalog hash drift"
    );
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {GPU_PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [\"PYTHONPATH=/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch\", \"python3\", \"{{input:generator}}\", \"--dataset-manifest\", \"{{input:proposal-datasets}}\", \"--task\", \"multiclass\", \"--checkpoint\", \"{{input:checkpoint}}\", \"--catalog-pointer\", \"{{input:catalog-pointer}}\", \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:catalog-data}}\", \"--selection-policy\", \"{{input:selection-policy}}\", \"--proposals-per-dataset\", \"64\", \"--device\", \"cuda\", \"--seed\", \"{seed}\", \"--output\", \"{{output:proposal-output}}/proposals.jsonl\"]\nworkdir = \"/home/ubuntu\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 1\ncpus_reserved = 4\nomp_threads = 1\nram_gb = 16\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"pinned JOPE proposal runtime\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", \"/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch/jope/propose.py\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 2\nmax_restarts = 8\n\n[[input]]\nname = \"generator\"\nsource = {generator_source:?}\ntarget = \"test_exact_dl_generate.py\"\nsha256 = {generator_sha:?}\n\n[[input]]\nname = \"common\"\nsource = {common_source:?}\ntarget = \"test_exact_common.py\"\nsha256 = {common_sha:?}\n\n[[input]]\nname = \"checkpoint\"\nsource = {checkpoint_source:?}\ntarget = \"checkpoint.model.pt\"\nsha256 = {checkpoint_sha:?}\n\n[[input]]\nname = \"catalog-pointer\"\nsource = {catalog_source:?}\ntarget = \"catalog-pointer.json\"\nsha256 = {catalog_sha:?}\n\n[[input]]\nname = \"catalog\"\nsource = {catalog_manifest_source:?}\ntarget = \"catalog.json\"\nsha256 = {catalog_manifest_sha:?}\n\n[[input]]\nname = \"catalog-data\"\nsource = {catalog_data_source:?}\ntarget = \"contexts.f32.bin\"\nsha256 = {catalog_data_sha:?}\n\n[[input]]\nname = \"selection-policy\"\nsource = {policy_source:?}\ntarget = \"proposal-selection-policy.json\"\nsha256 = {policy_sha:?}\n\n[[input]]\nname = \"proposal-datasets\"\nsource = {manifest_source:?}\ntarget = \"proposal-datasets.json\"\nsha256 = {manifest_sha256:?}\n\n[[output]]\nname = \"proposal-output\"\nsource = \"proposal-output\"\ndirectory = true\n",
        generator_source = generator.display().to_string(),
        generator_sha = sha256_file(generator)?,
        common_source = common.display().to_string(),
        common_sha = sha256_file(&common)?,
        checkpoint_source = checkpoint.display().to_string(),
        checkpoint_sha = sha256_file(checkpoint)?,
        catalog_source = catalog_pointer.display().to_string(),
        catalog_sha = sha256_file(catalog_pointer)?,
        catalog_manifest_source = catalog_manifest.display().to_string(),
        catalog_manifest_sha = sha256_file(&catalog_manifest)?,
        catalog_data_source = catalog_data.display().to_string(),
        catalog_data_sha = sha256_file(&catalog_data)?,
        policy_source = selection_policy.display().to_string(),
        policy_sha = sha256_file(selection_policy)?,
        manifest_source = manifest.display().to_string(),
    ))
}

fn find_named_file(root: &Path, name: &str) -> Result<Option<PathBuf>> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(path) = find_named_file(&entry.path(), name)? {
                return Ok(Some(path));
            }
        } else if entry.file_name() == name {
            return Ok(Some(entry.path()));
        }
    }
    Ok(None)
}

fn fetch_proposal_output(root: &Path, attempt: usize, job_id: &str) -> Result<PathBuf> {
    let destination = root.join(format!("proposals/attempt-{attempt}/proposals.jsonl"));
    if destination.is_file() {
        return Ok(destination);
    }
    let temporary = std::env::temp_dir().join(format!(
        "jope-mc210-proposals-{}-{attempt}",
        std::process::id()
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir_all(&temporary)?;
    let output = Command::new(SCQ)
        .arg("fetch")
        .arg("--output")
        .arg(&temporary)
        .arg(job_id)
        .output()?;
    if !output.status.success() {
        fs::remove_dir_all(&temporary)?;
        bail!(
            "scq fetch {job_id} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let source = find_named_file(&temporary, "proposals.jsonl")?
        .context("SCQ proposal output lacks proposals.jsonl")?;
    let summary = find_named_file(&temporary, "proposals.summary.json")?
        .context("SCQ proposal output lacks proposals.summary.json")?;
    immutable_write(&destination, &fs::read(source)?)?;
    immutable_write(&proposal_summary_path(&destination), &fs::read(summary)?)?;
    fs::remove_dir_all(&temporary)?;
    Ok(destination)
}

fn submit_spec(spec: &Path) -> Result<()> {
    let status = Command::new(SCQ).arg("check").arg(spec).status()?;
    if !status.success() {
        bail!("SCQ rejected {}", spec.display());
    }
    let status = Command::new(SCQ).arg("submit").arg(spec).status()?;
    if !status.success() {
        let job_id = fs::read_to_string(spec)?
            .lines()
            .find_map(|line| line.strip_prefix("id = "))
            .and_then(|encoded| serde_json::from_str::<String>(encoded).ok())
            .context("proposal spec lacks job id")?;
        if !scq_rows()?.contains_key(&job_id) {
            bail!("SCQ submission failed for {job_id}");
        }
    }
    Ok(())
}

fn materialize_proposals(
    root: &Path,
    target_ids: &[String],
    contexts: &BTreeMap<String, Value>,
    catalog_sha256: &str,
    policy_sha256: &str,
    generator: &Path,
    checkpoint: &Path,
    catalog_pointer: &Path,
    selection_policy: &Path,
) -> Result<Option<BTreeMap<String, Vec<Value>>>> {
    let expected = target_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut attempts = Vec::new();
    for attempt in 0..PROPOSAL_SEEDS.len() {
        let current = if attempt == 0 {
            target_ids.to_vec()
        } else {
            let merged = merge_partial_proposals(&attempts, target_ids)?;
            target_ids
                .iter()
                .filter(|dataset| merged[*dataset].len() < POPULATION)
                .cloned()
                .collect::<Vec<_>>()
        };
        if current.is_empty() {
            break;
        }
        let manifest_path = root.join(format!("proposals/attempt-{attempt}/datasets.json"));
        let manifest_value = proposal_manifest(catalog_sha256, &current, contexts)?;
        let manifest_bytes = canonical_json(&manifest_value)?;
        if manifest_path.is_file() {
            if fs::read(&manifest_path)? != manifest_bytes {
                bail!("immutable proposal target manifest collision");
            }
        } else {
            immutable_write(&manifest_path, &manifest_bytes)?;
        }
        let output_path = root.join(format!("proposals/attempt-{attempt}/proposals.jsonl"));
        if !output_path.is_file() {
            let manifest_sha256 = sha256_file(&manifest_path)?;
            let job_id = proposal_job_id(attempt, &manifest_sha256);
            let rows = scq_rows()?;
            if let Some(row) = rows.get(&job_id) {
                let state = row.get("state").and_then(Value::as_str).unwrap_or("");
                if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                    fetch_proposal_output(root, attempt, &job_id)?;
                } else if matches!(
                    state,
                    "failed" | "cancelled" | "canceled" | "lost" | "timed_out"
                ) {
                    bail!("DL proposal job {job_id} ended in {state}");
                } else {
                    println!("DL proposal job {job_id} is {state}; rerun planning to resume");
                    return Ok(None);
                }
            } else {
                ensure_priority(root)?;
                ensure_gpu_priority(root)?;
                let spec = root.join(format!("proposals/attempt-{attempt}/job.toml"));
                let rendered = render_proposal_spec(
                    root,
                    attempt,
                    &manifest_path,
                    generator,
                    checkpoint,
                    catalog_pointer,
                    selection_policy,
                )?;
                immutable_write(&spec, rendered.as_bytes())?;
                submit_spec(&spec)?;
                println!("submitted DL proposal job {job_id}; rerun planning to resume");
                return Ok(None);
            }
        }
        validate_proposal_summary(
            &output_path,
            catalog_sha256,
            policy_sha256,
            &sha256_file(&manifest_path)?,
        )?;
        attempts.push(proposal_rows(&output_path, &expected)?);
    }
    match merge_proposals(&attempts, target_ids) {
        Ok(merged) => {
            restore_gpu_priority(root)?;
            Ok(Some(merged))
        }
        Err(error) => {
            restore_gpu_priority(root)?;
            restore_priority(root)?;
            Err(error)
        }
    }
}

fn merge_partial_proposals(
    attempts: &[Vec<Value>],
    ordered_datasets: &[String],
) -> Result<BTreeMap<String, Vec<Value>>> {
    let mut merged = ordered_datasets
        .iter()
        .map(|dataset| (dataset.clone(), Vec::new()))
        .collect::<BTreeMap<_, _>>();
    let mut seen = ordered_datasets
        .iter()
        .map(|dataset| (dataset.clone(), BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for attempt in attempts {
        for row in attempt {
            let dataset = required_string(row, "/dataset_hash")?;
            let semantic = required_string(row, "/semantic_sha256")?;
            if seen.get_mut(dataset).unwrap().insert(semantic.to_string()) {
                merged.get_mut(dataset).unwrap().push(row.clone());
            }
        }
    }
    Ok(merged)
}

fn report_accuracy(path: &Path, expected_sha256: &str, expected_f1: f64) -> Result<f64> {
    if sha256_file(path)? != expected_sha256 {
        bail!("comparator report hash drift at {}", path.display());
    }
    let report = read_json(path)?;
    let macro_f1 = report
        .pointer("/metrics/macro_f1")
        .and_then(Value::as_f64)
        .context("comparator report lacks macro_f1")?;
    let accuracy = report
        .pointer("/metrics/accuracy")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .context("comparator report lacks accuracy")?;
    if macro_f1.to_bits() != expected_f1.to_bits() {
        bail!("comparator report macro-F1 differs from the frozen score row");
    }
    Ok(accuracy)
}

fn write_merged_proposal_set(
    root: &Path,
    targets: &[FrozenArenaDataset],
    merged: &BTreeMap<String, Vec<Value>>,
    catalog_sha256: &str,
    policy_sha256: &str,
) -> Result<PathBuf> {
    let path = root.join("proposals/merged.dl-proposals.v2.jsonl");
    let mut bytes = Vec::new();
    for target in targets {
        for row in &merged[&target.dataset_hash] {
            bytes.extend_from_slice(&serde_json::to_vec(row)?);
            bytes.push(b'\n');
        }
    }
    if path.is_file() {
        if fs::read(&path)? != bytes {
            bail!("immutable merged DlProposalSet.v2 collision");
        }
    } else {
        immutable_write(&path, &bytes)?;
    }
    let summary = json!({
        "schema": "DlProposalSet.v2",
        "task": "multiclass",
        "test_adaptive": true,
        "benchmark_valid": false,
        "catalog_sha256": catalog_sha256,
        "model_checkpoint_sha256": CHECKPOINT_SHA256,
        "selection_policy_sha256": policy_sha256,
        "output": path,
        "output_sha256": sha256_file(&path)?,
        "datasets": targets.len(),
        "proposals_per_dataset": POPULATION,
        "counts": {"multiclass": targets.len() * POPULATION},
        "proposal_seeds": PROPOSAL_SEEDS,
        "retry_policy": "merge unique native proposals from the initial seed and at most two fixed retry seeds",
        "zero_cross_context_backfill": true,
        "cross_context_backfill_rows": 0,
    });
    let summary_path = root.join("proposals/merged.dl-proposals.v2.summary.json");
    let summary_bytes = canonical_json(&summary)?;
    if summary_path.is_file() {
        if fs::read(&summary_path)? != summary_bytes {
            bail!("immutable merged proposal summary collision");
        }
    } else {
        immutable_write(&summary_path, &summary_bytes)?;
    }
    Ok(path)
}

fn build_dataset_binding(
    order: usize,
    target: &FrozenArenaDataset,
    corpus_entry: &Value,
    proposals: &[Value],
) -> Result<DatasetBinding> {
    let data_root =
        Path::new("/home/ubuntu/remote_super/quant/multi_label").join(&target.dataset_hash);
    let train_path = data_root.join("train.csv");
    let test_path = data_root.join("test.csv");
    let meta_path = data_root.join("meta.json");
    let train_sha256 = required_sha256(corpus_entry, "/files/train.csv/sha256")?;
    let test_sha256 = required_sha256(corpus_entry, "/files/test.csv/sha256")?;
    let meta_sha256 = required_sha256(corpus_entry, "/files/meta.json/sha256")?;
    for (path, expected) in [
        (&train_path, &train_sha256),
        (&test_path, &test_sha256),
        (&meta_path, &meta_sha256),
    ] {
        if sha256_file(path)? != *expected {
            bail!("dataset input drift at {}", path.display());
        }
    }
    let comparator_report = Path::new(&target.strict_best.source_uri);
    let comparator_accuracy = report_accuracy(
        comparator_report,
        &target.strict_best.source_sha256,
        target.strict_best.value,
    )?;
    let comparator_sha256 = sha256_bytes(&canonical_json(&json!({
        "schema": "JopeMulticlass210DatasetComparatorLock.v3",
        "test_adaptive": true,
        "benchmark_valid": false,
        "leaderboard_dataset_hash": target.leaderboard_dataset_hash,
        "dataset_hash": target.dataset_hash,
        "scores_path": target.scores_path,
        "scores_sha256": target.scores_sha256,
        "scores": target.scores,
        "strongest": target.strict_best,
        "aspfm_v2": target.aspfm_v2,
    }))?);
    let mut generation_zero = Vec::with_capacity(POPULATION);
    for (slot, row) in proposals.iter().enumerate() {
        let row_bytes = serde_json::to_vec(row)?;
        generation_zero.push(SeedMember {
            slot,
            canonical_genome: required_string(row, "/canonical_genome")?.to_string(),
            semantic_sha256: required_sha256(row, "/semantic_sha256")?,
            source: format!(
                "DlProposalSet.v2:{}:{}",
                required_string(row, "/decoder")?,
                required_string(row, "/source")?
            ),
            proposal_row_sha256: sha256_bytes(&row_bytes),
        });
    }
    if generation_zero.len() != POPULATION
        || generation_zero
            .iter()
            .map(|member| &member.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len()
            != POPULATION
    {
        bail!(
            "{} generation zero is not exactly 64 unique proposals",
            target.dataset_hash
        );
    }
    let proposal_set_sha256 = sha256_bytes(&serde_json::to_vec(proposals)?);
    Ok(DatasetBinding {
        order,
        cohort: if order < LEAD_COHORT {
            "positive-gap-easiest-45".to_string()
        } else if order < POSITIVE_GAP_TARGETS {
            "positive-gap-remaining-72".to_string()
        } else {
            "already-leading-headroom-93".to_string()
        },
        leaderboard_dataset_hash: target.leaderboard_dataset_hash.clone(),
        dataset_hash: target.dataset_hash.clone(),
        train_path: train_path.display().to_string(),
        train_sha256,
        test_path: test_path.display().to_string(),
        test_sha256,
        meta_path: meta_path.display().to_string(),
        meta_sha256,
        class_count: corpus_entry
            .get("n_classes")
            .and_then(Value::as_u64)
            .context("corpus entry lacks n_classes")? as usize,
        n_features: corpus_entry
            .get("n_features")
            .and_then(Value::as_u64)
            .context("corpus entry lacks n_features")? as usize,
        comparator_sha256,
        comparator_method: target.strict_best.method_id.clone(),
        comparator_macro_f1: target.strict_best.value,
        comparator_accuracy,
        comparator_report_path: target.strict_best.source_uri.clone(),
        comparator_report_sha256: target.strict_best.source_sha256.clone(),
        v2_macro_f1: target.aspfm_v2,
        proposal_set_sha256,
        generation_zero,
    })
}

fn plan_impl(
    run_id: &str,
    campaign_root: &Path,
    sotarena_root: &Path,
    corpus_manifest: &Path,
    catalog_pointer: &Path,
    selection_policy: &Path,
    checkpoint: &Path,
    proposal_generator: &Path,
) -> Result<()> {
    validate_run_id(run_id)?;
    if run_id != RUN_ID {
        bail!("v3 multiclass run id is frozen as {RUN_ID}");
    }
    let path = plan_path(campaign_root, run_id);
    if path.is_file() {
        let plan = read_plan(campaign_root, run_id)?;
        if plan.base_catalog_pointer_path != catalog_pointer.display().to_string()
            || plan.selection_policy_path != selection_policy.display().to_string()
            || plan.checkpoint_path != checkpoint.display().to_string()
            || plan.proposal_generator_path != proposal_generator.display().to_string()
            || plan.corpus_manifest_path != corpus_manifest.display().to_string()
        {
            bail!("immutable v3 plan input drift");
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "plan": path,
                "datasets": plan.datasets.len(),
                "test_adaptive": true,
                "benchmark_valid": false,
                "already_planned": true,
            }))?
        );
        return Ok(());
    }
    if sha256_file(checkpoint)? != CHECKPOINT_SHA256 {
        bail!("accepted DL checkpoint hash drift");
    }
    let corpus_eval = corpus_eval_path();
    if !corpus_eval.is_file() {
        bail!(
            "release corpus_eval is missing at {}; build it before planning",
            corpus_eval.display()
        );
    }
    let root = run_root(campaign_root, run_id);
    let snapshot = build_snapshot(sotarena_root)?;
    let targets = ordered_targets(&snapshot)?;
    let context_generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts/encode_multiclass_context_extension.py");
    ensure!(
        context_generator.is_file(),
        "multiclass context generator is missing"
    );
    // Priority overrides are durable scheduler state, but an operator may
    // normalize all families while this resumable campaign is between polls.
    // Reassert the frozen campaign weight on every non-terminal planning pass
    // so an already-submitted context or proposal job cannot silently stall.
    ensure_priority(&root)?;
    ensure_gpu_priority(&root)?;
    let Some(context_catalog) =
        prepare_context_catalog(&root, catalog_pointer, &context_generator)?
    else {
        return Ok(());
    };
    let (pointer, contexts) = catalog_contexts(&context_catalog.effective_pointer_path)?;
    let missing = targets
        .iter()
        .filter(|target| !contexts.contains_key(&target.dataset_hash))
        .map(|target| target.dataset_hash.clone())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "frozen DL catalog lacks {} required target contexts: {}; planning aborted without static-seed substitution",
            missing.len(),
            missing.join(",")
        );
    }
    let catalog_sha256 = required_sha256(&pointer, "/catalog_sha256")?;
    let data_sha256 = required_sha256(&pointer, "/data_sha256")?;
    let policy_sha256 = sha256_file(selection_policy)?;
    let snapshot_path = root.join("inputs/comparator-snapshot.v3.json");
    let snapshot_bytes = canonical_json(&snapshot)?;
    immutable_write(&snapshot_path, &snapshot_bytes)?;
    let target_value = json!({
        "schema": TARGET_SCHEMA,
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "ordering": "first 45 easiest positive-gap datasets; remaining 72 positive-gap datasets; remaining 93 already-leading datasets by descending (1 - ASPFM v2), then dataset hash",
        "lead_recovery_count": LEAD_COHORT,
        "positive_gap_count": POSITIVE_GAP_TARGETS,
        "already_leading_count": ALREADY_LEADING_TARGETS,
        "targets": targets.iter().enumerate().map(|(order, target)| json!({
            "order": order,
            "cohort": if order < LEAD_COHORT {"positive-gap-easiest-45"} else if order < POSITIVE_GAP_TARGETS {"positive-gap-remaining-72"} else {"already-leading-headroom-93"},
            "leaderboard_dataset_hash": target.leaderboard_dataset_hash,
            "dataset_hash": target.dataset_hash,
            "aspfm_v2": target.aspfm_v2,
            "strict_best": target.strict_best.value,
            "strict_gap": target.strict_gap,
        })).collect::<Vec<_>>(),
    });
    let target_path = root.join("inputs/target-manifest.v3.json");
    let target_bytes = canonical_json(&target_value)?;
    immutable_write(&target_path, &target_bytes)?;
    let target_ids = targets
        .iter()
        .map(|target| target.dataset_hash.clone())
        .collect::<Vec<_>>();
    let Some(merged) = materialize_proposals(
        &root,
        &target_ids,
        &contexts,
        &catalog_sha256,
        &policy_sha256,
        proposal_generator,
        checkpoint,
        &context_catalog.effective_pointer_path,
        selection_policy,
    )?
    else {
        return Ok(());
    };
    let merged_path =
        write_merged_proposal_set(&root, &targets, &merged, &catalog_sha256, &policy_sha256)?;
    let corpus = read_json(corpus_manifest)?;
    let corpus_entries = corpus
        .get("datasets")
        .and_then(Value::as_array)
        .context("multiclass corpus manifest lacks datasets")?
        .iter()
        .filter_map(|row| Some((row.get("dataset_hash")?.as_str()?.to_string(), row.clone())))
        .collect::<BTreeMap<_, _>>();
    let datasets = targets
        .iter()
        .enumerate()
        .map(|(order, target)| {
            build_dataset_binding(
                order,
                target,
                corpus_entries
                    .get(&target.dataset_hash)
                    .with_context(|| format!("corpus manifest lacks {}", target.dataset_hash))?,
                &merged[&target.dataset_hash],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = Plan {
        schema: PLAN_SCHEMA.to_string(),
        run_id: RUN_ID.to_string(),
        test_adaptive: true,
        selection_bias: "held-out-test-selected".to_string(),
        benchmark_valid: false,
        official_combined_rank_eligible: false,
        private: true,
        source_commit: source_commit()?,
        release_binary_sha256: current_binary_sha256()?,
        corpus_eval: corpus_eval.display().to_string(),
        corpus_eval_sha256: sha256_file(&corpus_eval)?,
        scq_binary: SCQ.to_string(),
        scq_binary_sha256: sha256_file(Path::new(SCQ))?,
        project: PROJECT.to_string(),
        required_effective_priority: 10.0,
        checkpoint_path: checkpoint.display().to_string(),
        checkpoint_sha256: CHECKPOINT_SHA256.to_string(),
        base_catalog_pointer_path: catalog_pointer.display().to_string(),
        base_catalog_pointer_sha256: context_catalog.base_pointer_sha256.clone(),
        catalog_pointer_path: context_catalog.effective_pointer_path.display().to_string(),
        catalog_pointer_sha256: context_catalog.effective_pointer_sha256.clone(),
        catalog_sha256,
        exact_context_data_sha256: data_sha256,
        runtime_bundle_sha256: context_catalog.runtime_bundle_sha256.clone(),
        context_job_id: context_catalog.job_id.clone(),
        context_job_spec_path: context_catalog.job_spec_path.display().to_string(),
        context_job_spec_sha256: context_catalog.job_spec_sha256.clone(),
        context_extension_request_path: context_catalog.request_path.display().to_string(),
        context_extension_request_sha256: context_catalog.request_sha256.clone(),
        context_extension_receipt_path: context_catalog.receipt_path.display().to_string(),
        context_extension_receipt_sha256: context_catalog.receipt_sha256.clone(),
        selection_policy_path: selection_policy.display().to_string(),
        selection_policy_sha256: policy_sha256,
        proposal_generator_path: proposal_generator.display().to_string(),
        proposal_generator_sha256: sha256_file(proposal_generator)?,
        corpus_manifest_path: corpus_manifest.display().to_string(),
        corpus_manifest_sha256: sha256_file(corpus_manifest)?,
        comparator_snapshot_path: snapshot_path.display().to_string(),
        comparator_snapshot_sha256: sha256_file(&snapshot_path)?,
        target_manifest_path: target_path.display().to_string(),
        target_manifest_sha256: sha256_file(&target_path)?,
        merged_proposal_set_path: merged_path.display().to_string(),
        merged_proposal_set_sha256: sha256_file(&merged_path)?,
        config: CampaignConfig::fixed(),
        execution_order: datasets
            .iter()
            .map(|dataset| dataset.dataset_hash.clone())
            .collect(),
        datasets,
    };
    immutable_write(&path, &canonical_json(&plan)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "plan": path,
            "run_id": RUN_ID,
            "datasets": TARGET_COUNT,
            "lead_recovery_cohort": LEAD_COHORT,
            "population": POPULATION,
            "generations": MAX_GENERATIONS,
            "test_adaptive": true,
            "benchmark_valid": false,
        }))?
    );
    Ok(())
}

fn read_plan(campaign_root: &Path, run_id: &str) -> Result<Plan> {
    if run_id != RUN_ID {
        bail!("v3 multiclass run id is frozen as {RUN_ID}");
    }
    let path = plan_path(campaign_root, run_id);
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let plan: Plan = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&plan)?
        || plan.schema != PLAN_SCHEMA
        || plan.run_id != RUN_ID
        || !plan.test_adaptive
        || plan.selection_bias != "held-out-test-selected"
        || plan.benchmark_valid
        || plan.official_combined_rank_eligible
        || !plan.private
        || plan.config != CampaignConfig::fixed()
        || plan.datasets.len() != TARGET_COUNT
        || plan.execution_order.len() != TARGET_COUNT
        || plan.execution_order.iter().collect::<BTreeSet<_>>().len() != TARGET_COUNT
        || plan.datasets.iter().any(|dataset| {
            dataset.generation_zero.len() != POPULATION
                || dataset
                    .generation_zero
                    .iter()
                    .map(|member| &member.semantic_sha256)
                    .collect::<BTreeSet<_>>()
                    .len()
                    != POPULATION
        })
        || plan.checkpoint_sha256 != CHECKPOINT_SHA256
        || plan.project != PROJECT
        || plan.required_effective_priority != 10.0
        || plan.context_job_id != context_job_id(&plan.context_extension_request_sha256)
    {
        bail!("v3 plan identity or frozen policy drift");
    }
    for (file, expected) in [
        (&plan.corpus_eval, &plan.corpus_eval_sha256),
        (&plan.scq_binary, &plan.scq_binary_sha256),
        (&plan.checkpoint_path, &plan.checkpoint_sha256),
        (
            &plan.base_catalog_pointer_path,
            &plan.base_catalog_pointer_sha256,
        ),
        (&plan.catalog_pointer_path, &plan.catalog_pointer_sha256),
        (&plan.selection_policy_path, &plan.selection_policy_sha256),
        (
            &plan.context_extension_request_path,
            &plan.context_extension_request_sha256,
        ),
        (
            &plan.context_extension_receipt_path,
            &plan.context_extension_receipt_sha256,
        ),
        (&plan.context_job_spec_path, &plan.context_job_spec_sha256),
        (
            &plan.proposal_generator_path,
            &plan.proposal_generator_sha256,
        ),
        (&plan.corpus_manifest_path, &plan.corpus_manifest_sha256),
        (
            &plan.comparator_snapshot_path,
            &plan.comparator_snapshot_sha256,
        ),
        (&plan.target_manifest_path, &plan.target_manifest_sha256),
        (
            &plan.merged_proposal_set_path,
            &plan.merged_proposal_set_sha256,
        ),
    ] {
        if sha256_file(Path::new(file))? != *expected {
            bail!("v3 pinned input drift at {file}");
        }
    }
    let context_request = read_json(Path::new(&plan.context_extension_request_path))?;
    ensure!(
        context_request.get("schema").and_then(Value::as_str) == Some(CONTEXT_REQUEST_SCHEMA),
        "v3 context request schema drift"
    );
    ensure!(
        context_request
            .get("base_catalog_pointer_sha256")
            .and_then(Value::as_str)
            == Some(plan.base_catalog_pointer_sha256.as_str()),
        "v3 context request base catalog hash drift"
    );
    ensure!(
        context_request
            .pointer("/encoder/runtime_bundle_sha256")
            .and_then(Value::as_str)
            == Some(plan.runtime_bundle_sha256.as_str()),
        "v3 runtime bundle hash drift"
    );
    let context_receipt = read_json(Path::new(&plan.context_extension_receipt_path))?;
    let context_data_path = Path::new(&plan.context_extension_receipt_path)
        .parent()
        .context("context receipt has no parent")?
        .join("contexts.f32.bin");
    validate_context_receipt(
        &context_receipt,
        &fs::read(context_data_path)?,
        &context_request,
        &plan.context_extension_request_sha256,
    )?;
    let snapshot: ComparatorSnapshot =
        serde_json::from_slice(&fs::read(&plan.comparator_snapshot_path)?)?;
    let (effective_pointer, contexts) = catalog_contexts(Path::new(&plan.catalog_pointer_path))?;
    ensure!(
        required_sha256(&effective_pointer, "/data_sha256")? == plan.exact_context_data_sha256,
        "v3 extended context data hash drift"
    );
    if snapshot.schema != SNAPSHOT_SCHEMA
        || snapshot.datasets.len() != LEADERBOARD_DATASETS
        || contexts.len() != 272
        || ordered_targets(&snapshot)?
            .iter()
            .map(|target| target.dataset_hash.as_str())
            .ne(plan.execution_order.iter().map(String::as_str))
    {
        bail!("v3 comparator snapshot or deterministic cohort ordering drift");
    }
    Ok(plan)
}

fn dataset_root(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets").join(dataset)
}

fn checkpoint_path(root: &Path, dataset: &str, generation: usize) -> PathBuf {
    dataset_root(root, dataset)
        .join("checkpoints")
        .join(format!("generation-{generation:03}.v3.json"))
}

fn ledger_path(root: &Path, dataset: &str) -> PathBuf {
    dataset_root(root, dataset).join("champion-ledger.v3.json")
}

fn dataset_terminal(root: &Path, dataset: &str) -> bool {
    ledger_path(root, dataset).is_file()
}

fn next_generation(root: &Path, dataset: &str) -> usize {
    (0..MAX_GENERATIONS)
        .find(|generation| !checkpoint_path(root, dataset, *generation).is_file())
        .unwrap_or(MAX_GENERATIONS)
}

fn read_checkpoint(
    root: &Path,
    dataset: &DatasetBinding,
    generation: usize,
) -> Result<GenerationCheckpoint> {
    let path = checkpoint_path(root, &dataset.dataset_hash, generation);
    let bytes = fs::read(&path)?;
    let checkpoint: GenerationCheckpoint = serde_json::from_slice(&bytes)?;
    if bytes != canonical_json(&checkpoint)?
        || checkpoint.schema != CHECKPOINT_SCHEMA
        || checkpoint.run_id != RUN_ID
        || checkpoint.dataset_hash != dataset.dataset_hash
        || checkpoint.leaderboard_dataset_hash != dataset.leaderboard_dataset_hash
        || checkpoint.comparator_sha256 != dataset.comparator_sha256
        || checkpoint.generation != generation
        || !checkpoint.test_adaptive
        || checkpoint.benchmark_valid
        || checkpoint.slots.len() != POPULATION
        || checkpoint.budget_complete != (generation + 1 == MAX_GENERATIONS)
    {
        bail!("invalid v3 generation checkpoint {}", path.display());
    }
    for digest in [
        checkpoint.generation_best_predictions_sha256.as_deref(),
        checkpoint.generation_best_probabilities_sha256.as_deref(),
        checkpoint.best_predictions_sha256.as_deref(),
        checkpoint.best_probabilities_sha256.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        ensure!(valid_sha256(digest), "checkpoint evidence hash is invalid");
    }
    ensure!(
        checkpoint.generation_best.is_some() == checkpoint.generation_best_member.is_some()
            && checkpoint.generation_best.is_some()
                == checkpoint.generation_best_predictions_sha256.is_some()
            && checkpoint.generation_best.is_some()
                == checkpoint.generation_best_probabilities_sha256.is_some()
            && checkpoint.best.is_some() == checkpoint.best_member.is_some()
            && checkpoint.best.is_some() == checkpoint.best_predictions_sha256.is_some()
            && checkpoint.best.is_some() == checkpoint.best_probabilities_sha256.is_some(),
        "checkpoint champion evidence is incomplete"
    );
    if let (Some(score), Some(member)) = (
        &checkpoint.generation_best,
        &checkpoint.generation_best_member,
    ) {
        ensure!(
            score.semantic_sha256 == member.semantic_sha256,
            "generation champion semantic identity drift"
        );
    }
    if let (Some(score), Some(member)) = (&checkpoint.best, &checkpoint.best_member) {
        ensure!(
            score.semantic_sha256 == member.semantic_sha256,
            "all-time champion semantic identity drift"
        );
    }
    Ok(checkpoint)
}

fn rank_order(left: &RankedScore, right: &RankedScore) -> Ordering {
    right
        .macro_f1
        .total_cmp(&left.macro_f1)
        .then_with(|| right.accuracy.total_cmp(&left.accuracy))
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

fn slot_order(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    match (&left.score, &right.score) {
        (Some(left), Some(right)) => rank_order(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .semantic_sha256
            .cmp(&right.member.semantic_sha256),
    }
}

fn qualifies(score: &RankedScore, dataset: &DatasetBinding) -> bool {
    score.macro_f1.is_finite()
        && (score.macro_f1 > dataset.comparator_macro_f1
            || (score.macro_f1 == 1.0 && dataset.comparator_macro_f1 == 1.0))
}

fn improves_v2(score: &RankedScore, dataset: &DatasetBinding) -> bool {
    score.macro_f1.is_finite() && score.macro_f1 > dataset.v2_macro_f1
}

fn requires_replay(score: &RankedScore, dataset: &DatasetBinding) -> bool {
    qualifies(score, dataset) || improves_v2(score, dataset)
}

fn better_score(left: &RankedScore, right: &RankedScore) -> bool {
    rank_order(left, right) == Ordering::Less
}

fn protocol_sha256(plan: &Plan, dataset: &DatasetBinding) -> Result<String> {
    Ok(sha256_bytes(&canonical_json(&json!({
        "schema": "JopeMulticlass210EvaluationProtocol.v3",
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "fitness": "direct held-out-test macro_f1",
        "dataset_hash": dataset.dataset_hash,
        "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "meta_sha256": dataset.meta_sha256,
        "comparator_sha256": dataset.comparator_sha256,
        "comparator_snapshot_sha256": plan.comparator_snapshot_sha256,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "catalog_sha256": plan.catalog_sha256,
    }))?))
}

fn shard_manifest(
    plan: &Plan,
    dataset: &DatasetBinding,
    population: &[SeedMember],
    generation: usize,
    shard: usize,
) -> Result<Value> {
    if population.len() != POPULATION || shard >= SHARD_SIZES.len() {
        bail!("invalid v3 population or shard");
    }
    let start = SHARD_SIZES[..shard].iter().sum::<usize>();
    let end = start + SHARD_SIZES[shard];
    Ok(json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "run_id": RUN_ID,
        "release_binary_sha256": plan.release_binary_sha256,
        "task": "multiclass",
        "dataset_hash": dataset.dataset_hash,
        "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
        "train_csv": dataset.train_path,
        "test_csv": dataset.test_path,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "meta_json": dataset.meta_path,
        "meta_sha256": dataset.meta_sha256,
        "checkpoint_epoch": format!("checkpoint-{}", plan.checkpoint_sha256),
        "checkpoint_sha256": plan.checkpoint_sha256,
        "catalog_sha256": plan.catalog_sha256,
        "protocol_sha256": protocol_sha256(plan, dataset)?,
        "comparator_snapshot_sha256": plan.comparator_snapshot_sha256,
        "comparator_lock_sha256": dataset.comparator_sha256,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "generated_at_ms": 0,
        "label_authority": {
            "score_tier": "test",
            "metric": "macro_f1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "held-out-test-selected",
            "generation": format!("multiclass-dl-v3-generation-{generation:03}"),
            "frozen_seed_sha256": dataset.proposal_set_sha256,
            "model_checkpoint_sha256": plan.checkpoint_sha256,
            "hall_of_fame_sha256": dataset.proposal_set_sha256,
        },
        "candidates": population[start..end].iter().map(|member| json!({
            "semantic_sha256": member.semantic_sha256,
            "canonical_genome": member.canonical_genome,
        })).collect::<Vec<_>>(),
    }))
}

fn eval_job_id(
    dataset: &str,
    generation: usize,
    shard: usize,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    format!(
        "jope-mc210-{dataset}-g{generation:03}-s{shard}-a{attempt}-m{}-v3",
        &manifest_sha256[..12]
    )
}

fn render_eval_spec(
    plan: &Plan,
    dataset: &DatasetBinding,
    generation: usize,
    shard: usize,
    attempt: usize,
    manifest: &Path,
) -> Result<String> {
    let manifest_sha256 = sha256_file(manifest)?;
    let id = eval_job_id(
        &dataset.dataset_hash,
        generation,
        shard,
        attempt,
        &manifest_sha256,
    );
    let timeout = if dataset.class_count == 100 || dataset.n_features >= 1_000 {
        300
    } else {
        180
    };
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [\"{{input:evaluator}}\", \"--manifest\", \"{{input:manifest}}\", \"--output-dir\", \"{{output:candidate-receipts}}\", \"--threads\", \"16\", \"--timeout-seconds\", {timeout:?}]\nworkdir = \"/home/ubuntu\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = 16\nomp_threads = 1\nram_gb = {ram}\ngang_size = 1\nplacement = \"single_node\"\n\n[env]\nOMP_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\n\n[preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n\n[[input]]\nname = \"evaluator\"\nsource = {evaluator:?}\ntarget = \"corpus_eval\"\nsha256 = {evaluator_sha:?}\n\n[[input]]\nname = \"manifest\"\nsource = {manifest_source:?}\ntarget = \"manifest.json\"\nsha256 = {manifest_sha256:?}\n\n[[output]]\nname = \"candidate-receipts\"\nsource = \"candidate-receipts\"\ndirectory = true\n",
        ram = if attempt >= 2 { 64 } else { 16 },
        evaluator = plan.corpus_eval,
        evaluator_sha = plan.corpus_eval_sha256,
        manifest_source = manifest.display().to_string(),
    ))
}

fn terminal_infrastructure_state(state: &str) -> bool {
    matches!(
        state,
        "failed" | "cancelled" | "canceled" | "lost" | "preempted" | "timed_out"
    )
}

fn scheduler_preemption(row: &Value) -> bool {
    row.get("failure_class").and_then(Value::as_str) == Some("preempted")
        || row
            .get("error_reason")
            .and_then(Value::as_str)
            .is_some_and(|reason| reason.contains("killed (no exit status)"))
}

fn fetch_receipts(
    directory: &Path,
    dataset: &str,
    generation: usize,
    shard: usize,
    attempt: usize,
    job_id: &str,
) -> Result<PathBuf> {
    let destination = directory.join(format!("shard-{shard}-attempt-{attempt}.receipts.jsonl"));
    let evidence = directory
        .join("evidence")
        .join(format!("shard-{shard}-attempt-{attempt}"));
    if destination.is_file() && evidence.is_dir() {
        return Ok(destination);
    }
    let temporary = std::env::temp_dir().join(format!(
        "jope-mc210-fetch-{}-{dataset}-{generation}-{shard}-{attempt}",
        std::process::id()
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir_all(&temporary)?;
    let output = Command::new(SCQ)
        .arg("fetch")
        .arg("--output")
        .arg(&temporary)
        .arg(job_id)
        .output()?;
    if !output.status.success() {
        fs::remove_dir_all(&temporary)?;
        bail!("scq fetch {job_id} failed");
    }
    let name = format!("{dataset}.receipts.jsonl");
    let source =
        find_named_file(&temporary, &name)?.with_context(|| format!("SCQ output lacks {name}"))?;
    immutable_or_verify(&destination, &fs::read(source)?, "evaluator receipt")?;
    copy_tree_immutable(&temporary, &evidence)?;
    fs::remove_dir_all(&temporary)?;
    Ok(destination)
}

fn receipt_values(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse v3 evaluator receipt"))
        .collect()
}

fn genome_complexity(source: &str) -> Result<usize> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member is not a pipeline");
    };
    Ok(pipeline
        .stages
        .iter()
        .map(|stage| 1 + stage.params.len())
        .sum())
}

fn slots_from_receipts(
    plan: &Plan,
    dataset: &DatasetBinding,
    generation: usize,
    population: &[SeedMember],
    manifests: &[PathBuf],
    receipts: &[PathBuf],
) -> Result<Vec<LogicalSlot>> {
    let expected_protocol = protocol_sha256(plan, dataset)?;
    let mut slots = Vec::with_capacity(POPULATION);
    for shard in 0..SHARD_SIZES.len() {
        let values = receipt_values(&receipts[shard])?;
        if values.len() != SHARD_SIZES[shard] {
            bail!(
                "{} generation {generation} shard {shard} emitted {} receipts",
                dataset.dataset_hash,
                values.len()
            );
        }
        let manifest_sha256 = sha256_file(&manifests[shard])?;
        let receipt_sha256 = sha256_file(&receipts[shard])?;
        let start = SHARD_SIZES[..shard].iter().sum::<usize>();
        for (offset, value) in values.into_iter().enumerate() {
            let slot = start + offset;
            let member = &population[slot];
            if value.get("schema").and_then(Value::as_str)
                != Some("JopeFlywheelCorpusEvalReceipt.v1")
                || value.get("task").and_then(Value::as_str) != Some("multiclass")
                || value.get("dataset_hash").and_then(Value::as_str)
                    != Some(dataset.dataset_hash.as_str())
                || value.get("semantic_sha256").and_then(Value::as_str)
                    != Some(member.semantic_sha256.as_str())
                || value.get("canonical_genome").and_then(Value::as_str)
                    != Some(member.canonical_genome.as_str())
                || value.get("train_sha256").and_then(Value::as_str)
                    != Some(dataset.train_sha256.as_str())
                || value.get("test_sha256").and_then(Value::as_str)
                    != Some(dataset.test_sha256.as_str())
                || value.get("manifest_sha256").and_then(Value::as_str)
                    != Some(manifest_sha256.as_str())
                || value.get("protocol_sha256").and_then(Value::as_str)
                    != Some(expected_protocol.as_str())
                || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
                || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
                || value.get("generation").and_then(Value::as_str)
                    != Some(format!("multiclass-dl-v3-generation-{generation:03}").as_str())
            {
                bail!(
                    "{} generation {generation} slot {slot} receipt binding mismatch",
                    dataset.dataset_hash
                );
            }
            let valid = value.get("valid").and_then(Value::as_bool) == Some(true);
            let score = if valid {
                let macro_f1 = value
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .context("valid receipt lacks macro-F1")?;
                let accuracy = value
                    .get("secondary_score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .context("valid receipt lacks accuracy")?;
                if value.get("metric").and_then(Value::as_str) != Some("macro_f1")
                    || value.get("secondary_metric").and_then(Value::as_str) != Some("accuracy")
                    || value
                        .get("predictions_sha256")
                        .and_then(Value::as_str)
                        .is_none()
                    || value
                        .get("probabilities_sha256")
                        .and_then(Value::as_str)
                        .is_none()
                {
                    bail!("valid receipt lacks prediction evidence");
                }
                Some(RankedScore {
                    macro_f1,
                    accuracy,
                    complexity: genome_complexity(&member.canonical_genome)?,
                    semantic_sha256: member.semantic_sha256.clone(),
                })
            } else {
                if value.get("failure").and_then(Value::as_str).is_none() {
                    bail!("invalid candidate receipt lacks failure");
                }
                None
            };
            slots.push(LogicalSlot {
                slot,
                member: member.clone(),
                score,
                valid,
                failure: value
                    .get("failure")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                receipt_sha256: receipt_sha256.clone(),
                predictions_sha256: value
                    .get("predictions_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                probabilities_sha256: value
                    .get("probabilities_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                node: required_string(&value, "/node")?.to_string(),
            });
        }
    }
    if slots.len() != POPULATION {
        bail!("generation produced {} logical slots", slots.len());
    }
    Ok(slots)
}

fn multiclass_family(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member is not a pipeline");
    };
    let head = pipeline
        .stages
        .last()
        .context("multiclass member has no head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass member has incompatible head {}", head.op);
    }
    Ok(head.op.clone())
}

fn regression_surrogate(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass parent is not a pipeline");
    };
    let head = pipeline
        .stages
        .pop()
        .context("multiclass parent lacks head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass parent has incompatible head");
    }
    pipeline.stages.push(Stage::new("ridge"));
    Ok(GenomeIr::Pipeline(pipeline).canonical())
}

fn retarget_multiclass(source: &str, slot: usize) -> Result<SeedMember> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("candidate is not a pipeline");
    };
    if pipeline.stages.iter().any(|stage| stage.op == "target") {
        bail!("unsafe target transform in multiclass child");
    }
    let old = pipeline.stages.pop().context("candidate has no head")?;
    if !feat_math::heads::is_head_op(&old.op) {
        bail!("candidate lacks a terminal head");
    }
    let mut head = Stage::new(MULTICLASS_HEADS[slot % MULTICLASS_HEADS.len()]);
    let l2 = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0][slot % 6];
    match head.op.as_str() {
        "multiclass_ridge" => {
            head.params.insert(
                "class_weight".to_string(),
                ParamValue::Str("balanced".to_string()),
            );
            head.params.insert("l2".to_string(), ParamValue::Float(l2));
        }
        "multiclass_ovr_logistic" => {
            head.params.insert("l2".to_string(), ParamValue::Float(l2));
        }
        "multiclass_diagonal_lda" => {
            head.params.insert(
                "shrinkage".to_string(),
                ParamValue::Float([0.0, 0.01, 0.1, 0.5][slot % 4]),
            );
        }
        "multiclass_nearest_centroid" => {
            head.params.insert(
                "shrinkage".to_string(),
                ParamValue::Float([0.0, 0.01, 0.1, 1.0][slot % 4]),
            );
        }
        _ => unreachable!(),
    }
    head.params.insert(
        "feature_budget".to_string(),
        ParamValue::Int([32, 64, 128, 256][(slot / 4) % 4]),
    );
    pipeline.stages.push(head);
    let canonical = GenomeIr::Pipeline(pipeline).canonical();
    let semantic = canonicalize(&canonical)
        .map_err(anyhow::Error::msg)?
        .semantic_sha256;
    validate_multiclass_genome(&canonical, &semantic)?;
    Ok(SeedMember {
        slot,
        canonical_genome: canonical,
        semantic_sha256: semantic,
        source: "evolved-v3".to_string(),
        proposal_row_sha256: "0".repeat(64),
    })
}

fn evolution_seed(dataset: &str, generation: usize, slot: usize, retry: usize) -> u64 {
    let digest =
        Sha256::digest(format!("{RUN_ID}:{dataset}:{generation}:{slot}:{retry}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn next_population(
    dataset: &DatasetBinding,
    ranked: &[LogicalSlot],
    generation: usize,
) -> Result<Vec<SeedMember>> {
    let valid = ranked.iter().filter(|slot| slot.valid).collect::<Vec<_>>();
    if valid.is_empty() {
        return ranked
            .iter()
            .enumerate()
            .map(|(slot, logical)| {
                let mut member = logical.member.clone();
                member.slot = slot;
                member.source = format!(
                    "all-invalid-reseed:generation={generation}:{}",
                    member.semantic_sha256
                );
                Ok(member)
            })
            .collect();
    }
    let parents = ranked.iter().collect::<Vec<_>>();
    let parent_pool = parents.len().min(30);
    let elite_count = valid.len().min(7);
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = BTreeSet::new();
    let mut families = BTreeMap::<String, usize>::new();
    for elite in valid.iter().take(elite_count) {
        let mut member = elite.member.clone();
        member.slot = next.len();
        member.source = format!("elite:{}", member.semantic_sha256);
        *families
            .entry(multiclass_family(&member.canonical_genome)?)
            .or_default() += 1;
        seen.insert(member.semantic_sha256.clone());
        next.push(member);
    }
    for slot in elite_count..POPULATION {
        let mut produced = None;
        for retry in 0..2_000 {
            let mut rng = Pcg32::new(
                evolution_seed(&dataset.dataset_hash, generation, slot, retry),
                0x6d63_3131_375f_7633,
            );
            let tournament = |rng: &mut Pcg32| -> usize {
                (0..3)
                    .map(|_| rng.next_u32() as usize % parent_pool)
                    .min()
                    .expect("nonempty tournament")
            };
            let first = tournament(&mut rng);
            let first_source = regression_surrogate(&parents[first].member.canonical_genome)?;
            let (candidate, operation, lineage) = if rng.next_f64() < 0.25 {
                let second = tournament(&mut rng);
                let second_source = regression_surrogate(&parents[second].member.canonical_genome)?;
                let Ok(candidate) = crossover_pipelines(
                    &first_source,
                    &second_source,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                ) else {
                    continue;
                };
                (
                    candidate.canonical,
                    "crossover",
                    format!(
                        "{}+{}",
                        parents[first].member.semantic_sha256,
                        parents[second].member.semantic_sha256
                    ),
                )
            } else {
                let Ok(candidate) = mutate_pipeline(
                    &first_source,
                    rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                    dataset.n_features.max(1),
                ) else {
                    continue;
                };
                (
                    candidate.canonical,
                    "mutation",
                    parents[first].member.semantic_sha256.clone(),
                )
            };
            let Ok(mut member) = retarget_multiclass(&candidate, rng.next_u32() as usize) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_family(&member.canonical_genome)?;
            if *families.get(&family).unwrap_or(&0) >= POPULATION / MULTICLASS_HEADS.len() {
                seen.remove(&member.semantic_sha256);
                continue;
            }
            *families.entry(family).or_default() += 1;
            member.slot = slot;
            member.source = format!("{operation}:{lineage}:retry={retry}");
            produced = Some(member);
            break;
        }
        next.push(produced.with_context(|| {
            format!(
                "could not generate v3 child for {} generation {generation} slot {slot}",
                dataset.dataset_hash
            )
        })?);
    }
    Ok(next)
}

fn replay_manifest(
    plan: &Plan,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
) -> Result<Value> {
    let champion = checkpoint
        .best_member
        .as_ref()
        .context("winning checkpoint lacks champion")?;
    Ok(json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "run_id": RUN_ID,
        "release_binary_sha256": plan.release_binary_sha256,
        "task": "multiclass",
        "dataset_hash": dataset.dataset_hash,
        "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
        "train_csv": dataset.train_path,
        "test_csv": dataset.test_path,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "meta_json": dataset.meta_path,
        "meta_sha256": dataset.meta_sha256,
        "checkpoint_epoch": format!("checkpoint-{}", champion.semantic_sha256),
        "checkpoint_sha256": champion.semantic_sha256,
        "catalog_sha256": plan.catalog_sha256,
        "protocol_sha256": protocol_sha256(plan, dataset)?,
        "comparator_snapshot_sha256": plan.comparator_snapshot_sha256,
        "comparator_lock_sha256": dataset.comparator_sha256,
        "target_manifest_sha256": plan.target_manifest_sha256,
        "generated_at_ms": 0,
        "label_authority": {
            "score_tier": "test",
            "metric": "macro_f1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "held-out-test-selected",
            "generation": format!("multiclass-dl-v3-generation-{:03}", checkpoint.generation),
            "frozen_seed_sha256": dataset.proposal_set_sha256,
            "model_checkpoint_sha256": plan.checkpoint_sha256,
            "hall_of_fame_sha256": champion.semantic_sha256,
            "expected_predictions_sha256": checkpoint.best_predictions_sha256,
            "expected_probabilities_sha256": checkpoint.best_probabilities_sha256,
            "expected_macro_f1": checkpoint.best.as_ref().map(|score| score.macro_f1),
            "expected_accuracy": checkpoint.best.as_ref().map(|score| score.accuracy),
        },
        "candidates": [{
            "semantic_sha256": champion.semantic_sha256,
            "canonical_genome": champion.canonical_genome,
        }],
    }))
}

fn replay_job_id(
    dataset: &str,
    semantic: &str,
    node: &str,
    attempt: usize,
    manifest_sha: &str,
) -> String {
    format!(
        "jope-mc210-{dataset}-replay-{}-{node}-a{attempt}-m{}-v3",
        &semantic[..12],
        &manifest_sha[..12]
    )
}

fn render_replay_spec(
    plan: &Plan,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    node: &str,
    role: &str,
    attempt: usize,
    manifest: &Path,
) -> Result<String> {
    let champion = checkpoint
        .best_member
        .as_ref()
        .context("replay lacks champion")?;
    let manifest_sha = sha256_file(manifest)?;
    let id = replay_job_id(
        &dataset.dataset_hash,
        &champion.semantic_sha256,
        node,
        attempt,
        &manifest_sha,
    );
    let timeout = if dataset.class_count == 100 || dataset.n_features >= 1_000 {
        300
    } else {
        180
    };
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [\"{{input:evaluator}}\", \"--manifest\", \"{{input:manifest}}\", \"--output-dir\", \"{{output:candidate-receipts}}\", \"--threads\", \"1\", \"--timeout-seconds\", {timeout:?}]\nworkdir = \"/home/ubuntu\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = 1\nomp_threads = 1\nram_gb = {ram}\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"multiclass replay role\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {role:?}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n\n[[input]]\nname = \"evaluator\"\nsource = {evaluator:?}\ntarget = \"corpus_eval\"\nsha256 = {evaluator_sha:?}\n\n[[input]]\nname = \"manifest\"\nsource = {manifest_source:?}\ntarget = \"manifest.json\"\nsha256 = {manifest_sha:?}\n\n[[output]]\nname = \"candidate-receipts\"\nsource = \"candidate-receipts\"\ndirectory = true\n",
        ram = if attempt >= 2 { 64 } else { 24 },
        evaluator = plan.corpus_eval,
        evaluator_sha = plan.corpus_eval_sha256,
        manifest_source = manifest.display().to_string(),
    ))
}

fn replay_receipt(
    path: &Path,
    manifest: &Path,
    dataset: &DatasetBinding,
    champion: &SeedMember,
    checkpoint: &GenerationCheckpoint,
    generation: usize,
    node: &str,
) -> Result<Value> {
    let values = receipt_values(path)?;
    if values.len() != 1 {
        bail!("replay must emit exactly one receipt");
    }
    let value = values.into_iter().next().unwrap();
    if value.get("schema").and_then(Value::as_str) != Some("JopeFlywheelCorpusEvalReceipt.v1")
        || value.get("task").and_then(Value::as_str) != Some("multiclass")
        || value.get("dataset_hash").and_then(Value::as_str) != Some(dataset.dataset_hash.as_str())
        || value.get("semantic_sha256").and_then(Value::as_str)
            != Some(champion.semantic_sha256.as_str())
        || value.get("canonical_genome").and_then(Value::as_str)
            != Some(champion.canonical_genome.as_str())
        || value.get("manifest_sha256").and_then(Value::as_str)
            != Some(sha256_file(manifest)?.as_str())
        || value.get("protocol_sha256").and_then(Value::as_str)
            != Some(required_sha256(&read_json(manifest)?, "/protocol_sha256")?.as_str())
        || value.get("node").and_then(Value::as_str) != Some(node)
        || value.get("generation").and_then(Value::as_str)
            != Some(format!("multiclass-dl-v3-generation-{generation:03}").as_str())
        || value.get("valid").and_then(Value::as_bool) != Some(true)
        || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || !value
            .get("score")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite)
        || !value
            .get("secondary_score")
            .and_then(Value::as_f64)
            .is_some_and(f64::is_finite)
        || value
            .get("predictions_sha256")
            .and_then(Value::as_str)
            .is_none()
        || value
            .get("probabilities_sha256")
            .and_then(Value::as_str)
            .is_none()
        || value.get("predictions_sha256").and_then(Value::as_str)
            != checkpoint.best_predictions_sha256.as_deref()
        || value.get("probabilities_sha256").and_then(Value::as_str)
            != checkpoint.best_probabilities_sha256.as_deref()
        || value
            .get("score")
            .and_then(Value::as_f64)
            .is_none_or(|actual| {
                checkpoint.best.as_ref().is_none_or(|expected| {
                    (actual - expected.macro_f1).abs() > REPLAY_SCORE_TOLERANCE
                })
            })
        || value
            .get("secondary_score")
            .and_then(Value::as_f64)
            .is_none_or(|actual| {
                checkpoint.best.as_ref().is_none_or(|expected| {
                    (actual - expected.accuracy).abs() > REPLAY_SCORE_TOLERANCE
                })
            })
    {
        bail!("invalid replay receipt for {node}");
    }
    Ok(value)
}

fn write_terminal_ledger(
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    status: &str,
    replays: Vec<Value>,
) -> Result<()> {
    let score = checkpoint.best.as_ref();
    let member = checkpoint.best_member.as_ref();
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, checkpoint.generation);
    let ledger = json!({
        "schema": LEDGER_SCHEMA,
        "run_id": RUN_ID,
        "dataset_hash": dataset.dataset_hash,
        "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
        "status": status,
        "test_adaptive": true,
        "selection_bias": "held-out-test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "comparator_sha256": dataset.comparator_sha256,
        "external_model": dataset.comparator_method,
        "external_report_path": dataset.comparator_report_path,
        "external_report_sha256": dataset.comparator_report_sha256,
        "external_macro_f1": dataset.comparator_macro_f1,
        "external_accuracy": dataset.comparator_accuracy,
        "aspfm_v2_macro_f1": dataset.v2_macro_f1,
        "generation": checkpoint.generation,
        "completed_generations": checkpoint.generation + 1,
        "semantic_sha256": member.map(|member| member.semantic_sha256.as_str()),
        "canonical_genome": member.map(|member| member.canonical_genome.as_str()),
        "gene_lineage": member.map(|member| member.source.as_str()),
        "macro_f1": score.map(|score| score.macro_f1),
        "overlay_macro_f1": score.map_or(dataset.v2_macro_f1, |score| dataset.v2_macro_f1.max(score.macro_f1)),
        "v3_improves_v2": score.is_some_and(|score| improves_v2(score, dataset)),
        "accuracy": score.map(|score| score.accuracy),
        "predictions_sha256": checkpoint.best_predictions_sha256,
        "probabilities_sha256": checkpoint.best_probabilities_sha256,
        "score_margin": score.map(|score| score.macro_f1 - dataset.comparator_macro_f1),
        "checkpoint_path": checkpoint_file,
        "checkpoint_sha256": sha256_file(&checkpoint_file)?,
        "replays": replays,
        "terminal_at_unix_ms": unix_millis(),
    });
    immutable_write(
        &ledger_path(root, &dataset.dataset_hash),
        &canonical_json(&ledger)?,
    )
}

fn manage_replay(
    plan: &Plan,
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    rows: &BTreeMap<String, Value>,
    specs: &mut Vec<PathBuf>,
    force_canary: bool,
) -> Result<bool> {
    let Some(score) = checkpoint.best.as_ref() else {
        return Ok(false);
    };
    if !force_canary && !requires_replay(score, dataset) {
        return Ok(false);
    }
    let champion = checkpoint
        .best_member
        .as_ref()
        .context("winner lacks champion")?;
    let replay_root = dataset_root(root, &dataset.dataset_hash)
        .join("replay")
        .join(&champion.semantic_sha256);
    let manifest_value = replay_manifest(plan, dataset, checkpoint)?;
    let manifest_bytes = canonical_json(&manifest_value)?;
    let mut replay_values = Vec::new();
    for (index, (node, role)) in REPLAY_ROLES.iter().enumerate() {
        let manifest = replay_root.join(format!("{node}.manifest.json"));
        if manifest.is_file() {
            if fs::read(&manifest)? != manifest_bytes {
                bail!("immutable replay manifest collision");
            }
        } else {
            immutable_write(&manifest, &manifest_bytes)?;
        }
        let mut harvested = None;
        let mut infrastructure_failures = 0;
        for attempt in 0..=66 {
            let id = replay_job_id(
                &dataset.dataset_hash,
                &champion.semantic_sha256,
                node,
                attempt,
                &sha256_bytes(&manifest_bytes),
            );
            let Some(row) = rows.get(&id) else {
                let spec = replay_root.join(format!("{node}-a{attempt}.toml"));
                let rendered =
                    render_replay_spec(plan, dataset, checkpoint, node, role, attempt, &manifest)?;
                if spec.is_file() {
                    if fs::read_to_string(&spec)? != rendered {
                        bail!("immutable replay spec collision");
                    }
                } else {
                    immutable_write(&spec, rendered.as_bytes())?;
                }
                specs.push(spec);
                break;
            };
            let state = row.get("state").and_then(Value::as_str).unwrap_or("");
            if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                harvested = Some(fetch_receipts(
                    &replay_root,
                    &dataset.dataset_hash,
                    checkpoint.generation,
                    index,
                    attempt,
                    &id,
                )?);
                break;
            }
            if terminal_infrastructure_state(state)
                || (state == "completed" && row.get("exit_code").and_then(Value::as_i64) != Some(0))
            {
                if scheduler_preemption(row) {
                    continue;
                }
                infrastructure_failures += 1;
                if infrastructure_failures == 3 {
                    bail!("replay {id} exhausted infrastructure retries");
                }
                continue;
            }
            break;
        }
        if let Some(path) = harvested {
            let value = replay_receipt(
                &path,
                &manifest,
                dataset,
                champion,
                checkpoint,
                checkpoint.generation,
                node,
            )?;
            replay_values.push(json!({
                "node": node,
                "canonical_genome": champion.canonical_genome,
                "semantic_sha256": champion.semantic_sha256,
                "dataset_hash": dataset.dataset_hash,
                "comparator_sha256": dataset.comparator_sha256,
                "prediction_sha256": value.get("predictions_sha256"),
                "probability_sha256": value.get("probabilities_sha256"),
                "macro_f1": value.get("score"),
                "accuracy": value.get("secondary_score"),
                "receipt_path": path,
                "receipt_sha256": sha256_file(&path)?,
            }));
        }
    }
    if replay_values.len() != REPLAY_ROLES.len() {
        return Ok(true);
    }
    for pointer in [
        "/canonical_genome",
        "/semantic_sha256",
        "/dataset_hash",
        "/comparator_sha256",
        "/prediction_sha256",
        "/probability_sha256",
        "/macro_f1",
        "/accuracy",
    ] {
        if replay_values
            .iter()
            .map(|value| {
                serde_json::to_string(value.pointer(pointer).unwrap_or(&Value::Null)).unwrap()
            })
            .collect::<BTreeSet<_>>()
            .len()
            != 1
        {
            bail!("three-node replay differs at {pointer}");
        }
    }
    if force_canary {
        immutable_or_verify(
            &dataset_root(root, &dataset.dataset_hash).join("canary.v3.json"),
            &canonical_json(&json!({
                "schema": "JopeMulticlassCanary.v3",
                "run_id": RUN_ID,
                "dataset_hash": dataset.dataset_hash,
                "generation": checkpoint.generation,
                "test_adaptive": true,
                "benchmark_valid": false,
                "checkpoint_resume_verified": true,
                "replay_nodes": replay_values.iter().filter_map(|row| row.get("node")).collect::<Vec<_>>(),
                "replays": replay_values,
            }))?,
            "canary receipt",
        )?;
    } else {
        let status = if dataset.comparator_macro_f1 == 1.0 && score.macro_f1 == 1.0 {
            "ceiling_matched"
        } else {
            "v3_improvement"
        };
        write_terminal_ledger(root, dataset, checkpoint, status, replay_values)?;
    }
    Ok(true)
}

fn current_overlay(plan: &Plan, root: &Path) -> Result<BTreeMap<String, f64>> {
    let mut overlay = BTreeMap::new();
    for dataset in &plan.datasets {
        if !dataset_terminal(root, &dataset.dataset_hash) {
            continue;
        }
        let ledger = read_json(&ledger_path(root, &dataset.dataset_hash))?;
        let score = ledger
            .get("overlay_macro_f1")
            .and_then(Value::as_f64)
            .context("terminal ledger lacks overlay macro-F1")?;
        overlay.insert(dataset.leaderboard_dataset_hash.clone(), score);
    }
    Ok(overlay)
}

fn verify_public_sotarena_unchanged(snapshot: &ComparatorSnapshot) -> Result<()> {
    for (path, expected) in [
        (&snapshot.config_path, &snapshot.config_sha256),
        (&snapshot.report_path, &snapshot.report_sha256),
    ] {
        ensure!(
            sha256_file(Path::new(path))? == *expected,
            "public SOTArena artifact drift at {path}"
        );
    }
    for dataset in &snapshot.datasets {
        ensure!(
            sha256_file(Path::new(&dataset.dataset_manifest_path))?
                == dataset.dataset_manifest_sha256,
            "public SOTArena dataset manifest drift for {}",
            dataset.leaderboard_dataset_hash
        );
        ensure!(
            sha256_file(Path::new(&dataset.scores_path))? == dataset.scores_sha256,
            "public SOTArena score artifact drift for {}",
            dataset.leaderboard_dataset_hash
        );
    }
    Ok(())
}

fn canary_ready(root: &Path, dataset: &DatasetBinding) -> bool {
    dataset_root(root, &dataset.dataset_hash)
        .join("canary.v3.json")
        .is_file()
}

fn active_datasets<'a>(plan: &'a Plan, root: &Path) -> Vec<&'a DatasetBinding> {
    let canary_complete = canary_ready(root, &plan.datasets[0]);
    let lead_complete = plan.datasets[..LEAD_COHORT]
        .iter()
        .all(|dataset| dataset_terminal(root, &dataset.dataset_hash));
    let admission_end = if !canary_complete {
        1
    } else if !lead_complete {
        LEAD_COHORT
    } else {
        TARGET_COUNT
    };
    let mut started = plan.datasets[..admission_end]
        .iter()
        .filter(|dataset| {
            !dataset_terminal(root, &dataset.dataset_hash)
                && next_generation(root, &dataset.dataset_hash) > 0
        })
        .collect::<Vec<_>>();
    started.sort_by_key(|dataset| dataset.order);
    let mut selected = started;
    for dataset in &plan.datasets[..admission_end] {
        if selected.len() == ACTIVE_DATASETS {
            break;
        }
        if !dataset_terminal(root, &dataset.dataset_hash)
            && next_generation(root, &dataset.dataset_hash) == 0
            && !selected
                .iter()
                .any(|row| row.dataset_hash == dataset.dataset_hash)
        {
            selected.push(dataset);
        }
    }
    selected.truncate(ACTIVE_DATASETS);
    selected
}

fn submit_specs(root: &Path, specs: &[PathBuf]) -> Result<()> {
    if specs.is_empty() {
        return Ok(());
    }
    ensure_priority(root)?;
    let rows = scq_rows()?;
    let queued = rows
        .iter()
        .filter(|(id, row)| {
            id.starts_with("jope-mc210-")
                && id.contains("-g")
                && matches!(
                    row.get("state").and_then(Value::as_str),
                    Some("queued" | "held")
                )
        })
        .count();
    ensure!(
        queued + specs.len() <= MAX_QUEUED_EVALUATION_SHARDS,
        "submitting {} evaluation shards would exceed the {}-shard queue cap",
        queued + specs.len(),
        MAX_QUEUED_EVALUATION_SHARDS
    );
    for spec in specs {
        let status = Command::new(SCQ).arg("check").arg(spec).status()?;
        if !status.success() {
            bail!("SCQ rejected {}", spec.display());
        }
    }
    let status = Command::new(SCQ)
        .arg("submit-batch")
        .arg("--json")
        .args(specs)
        .status()?;
    if !status.success() {
        let rows = scq_rows()?;
        for spec in specs {
            let id = fs::read_to_string(spec)?
                .lines()
                .find_map(|line| line.strip_prefix("id = "))
                .and_then(|encoded| serde_json::from_str::<String>(encoded).ok())
                .context("SCQ spec lacks id")?;
            if !rows.contains_key(&id) {
                bail!("SCQ batch submission omitted {id}");
            }
        }
    }
    Ok(())
}

fn cancel_queued_run_jobs() -> Result<usize> {
    let rows = scq_rows()?;
    let mut cancelled = 0;
    for (id, row) in rows {
        if id.starts_with("jope-mc210-")
            && matches!(
                row.get("state").and_then(Value::as_str),
                Some("queued" | "held")
            )
        {
            let status = Command::new(SCQ).arg("cancel").arg(&id).status()?;
            if !status.success() {
                bail!("could not cancel queued v3 job {id}");
            }
            cancelled += 1;
        }
    }
    Ok(cancelled)
}

fn terminal_state(
    plan: &Plan,
    root: &Path,
    snapshot: &ComparatorSnapshot,
) -> Result<Option<Value>> {
    let overlay = current_overlay(plan, root)?;
    verify_public_sotarena_unchanged(snapshot)?;
    let ranking = ranking(snapshot, &overlay);
    let first = aspfm_is_first(&ranking);
    let terminal = plan
        .datasets
        .iter()
        .filter(|dataset| dataset_terminal(root, &dataset.dataset_hash))
        .count();
    if terminal != TARGET_COUNT {
        return Ok(None);
    }
    let strict_winners = plan
        .datasets
        .iter()
        .filter(|dataset| dataset_terminal(root, &dataset.dataset_hash))
        .filter_map(|dataset| read_json(&ledger_path(root, &dataset.dataset_hash)).ok())
        .filter(|ledger| ledger.get("status").and_then(Value::as_str) == Some("strict_winner"))
        .count();
    let v3_v2_improvements = plan
        .datasets
        .iter()
        .filter_map(|dataset| read_json(&ledger_path(root, &dataset.dataset_hash)).ok())
        .filter(|ledger| ledger.get("v3_improves_v2").and_then(Value::as_bool) == Some(true))
        .count();
    let aspfm = ranking
        .iter()
        .find(|row| row.participant == "family:aspfm")
        .context("ranking lacks ASPFM")?;
    let tabfm = ranking
        .iter()
        .find(|row| row.participant == "family:tabfm-ensemble")
        .context("ranking lacks TabFM Ensemble")?;
    let baseline_aspfm = snapshot
        .baseline_ranking
        .iter()
        .find(|row| row.participant == "family:aspfm")
        .context("baseline ranking lacks ASPFM")?;
    Ok(Some(json!({
        "schema": SUMMARY_SCHEMA,
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "private": true,
        "status": "completed_full_budget",
        "terminal_datasets": terminal,
        "candidate_evaluations": TOTAL_CANDIDATE_EVALUATIONS,
        "public_sotarena_unchanged": true,
        "strict_winners": strict_winners,
        "v3_v2_improvements": v3_v2_improvements,
        "aspfm_rank": aspfm.rank,
        "aspfm_elo": aspfm.elo,
        "tabfm_ensemble_rank": tabfm.rank,
        "tabfm_ensemble_elo": tabfm.elo,
        "frozen_baseline_aspfm_elo": baseline_aspfm.elo,
        "remaining_elo_gap": tabfm.elo - aspfm.elo,
        "acceptance": first
            && v3_v2_improvements > 0
            && aspfm.elo > baseline_aspfm.elo
            && aspfm.elo > tabfm.elo,
        "overlay": overlay,
        "ranking": ranking,
    })))
}

fn copy_tree_immutable(source: &Path, destination: &Path) -> Result<()> {
    if source.is_file() {
        immutable_or_verify(destination, &fs::read(source)?, "proof artifact")?;
        return Ok(());
    }
    ensure!(
        source.is_dir(),
        "proof source is absent: {}",
        source.display()
    );
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let child_source = entry.path();
        let child_destination = destination.join(entry.file_name());
        if child_source.is_dir() {
            copy_tree_immutable(&child_source, &child_destination)?;
        } else {
            immutable_or_verify(
                &child_destination,
                &fs::read(&child_source)?,
                "proof artifact",
            )?;
        }
    }
    Ok(())
}

fn write_campaign_terminal(
    plan: &Plan,
    root: &Path,
    snapshot: &ComparatorSnapshot,
    summary: Value,
) -> Result<()> {
    let summary_path = root.join("terminal-summary.v3.json");
    immutable_write(&summary_path, &canonical_json(&summary)?)?;
    let proof_root = root.join("proof-bundle-private-v3");
    let campaign_plan = plan_path(root.parent().context("run root lacks parent")?, RUN_ID);
    let merged_summary = root.join("proposals/merged.dl-proposals.v2.summary.json");
    for (source, relative) in [
        (campaign_plan.as_path(), "campaign-plan.v3.json"),
        (
            Path::new(&plan.comparator_snapshot_path),
            "comparator-snapshot.v3.json",
        ),
        (
            Path::new(&plan.target_manifest_path),
            "target-manifest.v3.json",
        ),
        (
            Path::new(&plan.merged_proposal_set_path),
            "merged.dl-proposals.v2.jsonl",
        ),
        (
            merged_summary.as_path(),
            "merged.dl-proposals.v2.summary.json",
        ),
        (summary_path.as_path(), "terminal-summary.v3.json"),
        (
            Path::new(&plan.base_catalog_pointer_path),
            "context/base-catalog-pointer.json",
        ),
        (
            Path::new(&plan.catalog_pointer_path),
            "context/extended-catalog-pointer.json",
        ),
        (
            Path::new(&plan.context_extension_request_path),
            "context/context-request.v2.json",
        ),
        (
            Path::new(&plan.context_extension_receipt_path),
            "context/context-extension.json",
        ),
    ] {
        immutable_write(&proof_root.join(relative), &fs::read(source)?)?;
    }
    let base_pointer_value = read_json(Path::new(&plan.base_catalog_pointer_path))?;
    let extended_pointer_value = read_json(Path::new(&plan.catalog_pointer_path))?;
    let base_catalog_bundle = proof_root.join("context/base-catalog.json");
    let base_data_bundle = proof_root.join("context/base-data.f32.bin");
    let extended_catalog_bundle = proof_root.join("context/extended-catalog.json");
    let extended_data_bundle = proof_root.join("context/extended-data.f32.bin");
    let extension_data_bundle = proof_root.join("context/extension-data.f32.bin");
    immutable_write(
        &base_catalog_bundle,
        &fs::read(required_string(&base_pointer_value, "/catalog")?)?,
    )?;
    immutable_write(
        &base_data_bundle,
        &fs::read(required_string(&base_pointer_value, "/data")?)?,
    )?;
    immutable_write(
        &extended_catalog_bundle,
        &fs::read(required_string(&extended_pointer_value, "/catalog")?)?,
    )?;
    immutable_write(
        &extended_data_bundle,
        &fs::read(required_string(&extended_pointer_value, "/data")?)?,
    )?;
    immutable_write(
        &extension_data_bundle,
        &fs::read(
            Path::new(&plan.context_extension_receipt_path)
                .parent()
                .context("context receipt has no parent")?
                .join("contexts.f32.bin"),
        )?,
    )?;
    let ledgers = plan
        .datasets
        .iter()
        .filter(|dataset| dataset_terminal(root, &dataset.dataset_hash))
        .map(|dataset| {
            let path = ledger_path(root, &dataset.dataset_hash);
            let bundled = proof_root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("champion-ledger.v3.json");
            immutable_write(&bundled, &fs::read(&path)?)?;
            let ledger = read_json(&path)?;
            let mut replay_receipts = Vec::new();
            for replay in ledger
                .get("replays")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let node = required_string(replay, "/node")?;
                let receipt = Path::new(required_string(replay, "/receipt_path")?);
                let bundled_receipt = proof_root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join("replays")
                    .join(format!("{node}.receipts.jsonl"));
                immutable_write(&bundled_receipt, &fs::read(receipt)?)?;
                replay_receipts.push(json!({
                    "node": node,
                    "path": bundled_receipt,
                    "sha256": sha256_file(&bundled_receipt)?,
                }));
            }
            Ok(json!({
                "dataset_hash": dataset.dataset_hash,
                "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
                "path": bundled,
                "sha256": sha256_file(&bundled)?,
                "replay_receipts": replay_receipts,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    copy_tree_immutable(&root.join("datasets"), &proof_root.join("datasets"))?;
    let overlay = current_overlay(plan, root)?;
    let perfect_overlay = snapshot
        .datasets
        .iter()
        .map(|dataset| (dataset.leaderboard_dataset_hash.clone(), 1.0))
        .collect::<BTreeMap<_, _>>();
    let deltas = plan
        .datasets
        .iter()
        .map(|dataset| {
            let ledger = read_json(&ledger_path(root, &dataset.dataset_hash))?;
            let v3 = ledger.get("macro_f1").and_then(Value::as_f64);
            Ok(json!({
                "leaderboard_dataset_hash": dataset.leaderboard_dataset_hash,
                "dataset_hash": dataset.dataset_hash,
                "aspfm_v2": dataset.v2_macro_f1,
                "v3_champion": v3,
                "overlay": ledger.get("overlay_macro_f1"),
                "delta_v2_to_v3": v3.map(|score| score - dataset.v2_macro_f1),
                "measured_search_budget_ceiling": v3,
                "mathematical_macro_f1_ceiling": 1.0,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let proof = json!({
        "schema": PROOF_SCHEMA,
        "run_id": RUN_ID,
        "test_adaptive": true,
        "benchmark_valid": false,
        "private": true,
        "publishable": false,
        "plan_path": proof_root.join("campaign-plan.v3.json"),
        "plan_sha256": sha256_file(&proof_root.join("campaign-plan.v3.json"))?,
        "source_commit": plan.source_commit,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "catalog_pointer_sha256": plan.catalog_pointer_sha256,
        "exact_context_data_sha256": plan.exact_context_data_sha256,
        "selection_policy_sha256": plan.selection_policy_sha256,
        "proposal_generator_sha256": plan.proposal_generator_sha256,
        "corpus_eval_sha256": plan.corpus_eval_sha256,
        "comparator_snapshot_path": proof_root.join("comparator-snapshot.v3.json"),
        "comparator_snapshot_sha256": sha256_file(&proof_root.join("comparator-snapshot.v3.json"))?,
        "target_manifest_path": proof_root.join("target-manifest.v3.json"),
        "target_manifest_sha256": sha256_file(&proof_root.join("target-manifest.v3.json"))?,
        "catalog_provenance": {
            "base_pointer": proof_root.join("context/base-catalog-pointer.json"),
            "base_pointer_sha256": plan.base_catalog_pointer_sha256,
            "base_catalog": base_catalog_bundle,
            "base_catalog_sha256": sha256_file(&base_catalog_bundle)?,
            "base_data": base_data_bundle,
            "base_data_sha256": sha256_file(&base_data_bundle)?,
            "extended_pointer": proof_root.join("context/extended-catalog-pointer.json"),
            "extended_pointer_sha256": plan.catalog_pointer_sha256,
            "extended_catalog": extended_catalog_bundle,
            "extended_catalog_sha256": sha256_file(&extended_catalog_bundle)?,
            "extended_data": extended_data_bundle,
            "extended_data_sha256": sha256_file(&extended_data_bundle)?,
            "extension_data": extension_data_bundle,
            "extension_data_sha256": sha256_file(&extension_data_bundle)?,
        },
        "merged_proposal_set_path": proof_root.join("merged.dl-proposals.v2.jsonl"),
        "merged_proposal_set_sha256": sha256_file(&proof_root.join("merged.dl-proposals.v2.jsonl"))?,
        "terminal_summary_path": proof_root.join("terminal-summary.v3.json"),
        "terminal_summary_sha256": sha256_file(&proof_root.join("terminal-summary.v3.json"))?,
        "input_hashes": {
            "base_catalog_pointer_sha256": plan.base_catalog_pointer_sha256,
            "extended_catalog_pointer_sha256": plan.catalog_pointer_sha256,
            "extended_context_data_sha256": plan.exact_context_data_sha256,
            "context_request_sha256": plan.context_extension_request_sha256,
            "context_extension_receipt_sha256": plan.context_extension_receipt_sha256,
            "comparator_snapshot_sha256": plan.comparator_snapshot_sha256,
            "corpus_manifest_sha256": plan.corpus_manifest_sha256,
            "checkpoint_sha256": plan.checkpoint_sha256,
            "runtime_bundle_sha256": plan.runtime_bundle_sha256,
            "context_job_id": plan.context_job_id,
            "context_job_spec_sha256": plan.context_job_spec_sha256,
        },
        "generation_zero": {
            "datasets": TARGET_COUNT,
            "proposals_per_dataset": POPULATION,
            "rows": TARGET_COUNT * POPULATION,
            "candidate_evaluations": TOTAL_CANDIDATE_EVALUATIONS,
            "unique_canonical_proposals": true,
            "native_multiclass_head_validation": true,
        },
        "rankings": {
            "baseline_v2": snapshot.baseline_ranking,
            "v2_v3_empirical_ceiling": ranking(snapshot, &overlay),
            "current_dataset_best": ranking(snapshot, &overlay),
            "ideal_all_perfect_score_counterfactual": ranking(snapshot, &perfect_overlay),
        },
        "ceiling_definitions": {
            "measured_search_budget": "maximum all-time test macro-F1 observed under population 64, 30 generations, four 16-candidate shards",
            "mathematical_macro_f1_ceiling": "1.0, reported only as an idealized counterfactual",
            "unseen_test_generalization_claim": false,
        },
        "per_dataset_deltas": deltas,
        "ledgers": ledgers,
    });
    immutable_write(
        &proof_root.join("manifest.v3.json"),
        &canonical_json(&proof)?,
    )?;
    restore_gpu_priority(root)?;
    restore_priority(root)
}

fn run_impl(run_id: &str, campaign_root: &Path, resume: bool) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let marker = root.join("run-started.v3.json");
    if marker.is_file() && !resume {
        bail!("v3 campaign already started; rerun with --resume");
    }
    if !marker.is_file() {
        immutable_write(
            &marker,
            &canonical_json(&json!({
                "schema": "JopeMulticlass210RunStarted.v3",
                "run_id": RUN_ID,
                "test_adaptive": true,
                "benchmark_valid": false,
            }))?,
        )?;
    }
    let snapshot: ComparatorSnapshot =
        serde_json::from_slice(&fs::read(&plan.comparator_snapshot_path)?)?;
    if root.join("terminal-summary.v3.json").is_file() {
        validate_impl(run_id, campaign_root)?;
        return Ok(());
    }
    // The supervisor repeatedly enters run_impl while existing shards are
    // queued or running. Keep the authorized campaign weight effective even
    // when this pass has no new specs to submit.
    ensure_priority(&root)?;
    let active = active_datasets(&plan, &root);
    let rows = scq_rows()?;
    let mut specs = Vec::new();
    let mut harvested_generations = 0;
    for dataset in active {
        let generation = next_generation(&root, &dataset.dataset_hash);
        if generation > 0 {
            let previous = read_checkpoint(&root, dataset, generation - 1)?;
            if generation == 2 && !canary_ready(&root, dataset) {
                if manage_replay(&plan, &root, dataset, &previous, &rows, &mut specs, true)? {
                    continue;
                }
            }
            if generation == MAX_GENERATIONS {
                if manage_replay(&plan, &root, dataset, &previous, &rows, &mut specs, false)? {
                    continue;
                }
                if !dataset_terminal(&root, &dataset.dataset_hash) {
                    write_terminal_ledger(&root, dataset, &previous, "exhausted", Vec::new())?;
                }
                continue;
            }
        }
        let population = if generation == 0 {
            dataset.generation_zero.clone()
        } else {
            read_checkpoint(&root, dataset, generation - 1)?.next_population
        };
        let generation_root =
            dataset_root(&root, &dataset.dataset_hash).join(format!("generation-{generation:03}"));
        let mut manifests = Vec::new();
        let mut receipts = Vec::new();
        for shard in 0..SHARD_SIZES.len() {
            let manifest = generation_root.join(format!("shard-{shard}.manifest.json"));
            let bytes = canonical_json(&shard_manifest(
                &plan,
                dataset,
                &population,
                generation,
                shard,
            )?)?;
            if manifest.is_file() {
                if fs::read(&manifest)? != bytes {
                    bail!("immutable v3 shard manifest collision");
                }
            } else {
                immutable_write(&manifest, &bytes)?;
            }
            manifests.push(manifest.clone());
            let manifest_sha = sha256_bytes(&bytes);
            let mut harvested = None;
            let mut infrastructure_failures = 0;
            for attempt in 0..=66 {
                let id = eval_job_id(
                    &dataset.dataset_hash,
                    generation,
                    shard,
                    attempt,
                    &manifest_sha,
                );
                let Some(row) = rows.get(&id) else {
                    let spec = generation_root.join(format!("shard-{shard}-a{attempt}.toml"));
                    let rendered =
                        render_eval_spec(&plan, dataset, generation, shard, attempt, &manifest)?;
                    if spec.is_file() {
                        if fs::read_to_string(&spec)? != rendered {
                            bail!("immutable v3 evaluation spec collision");
                        }
                    } else {
                        immutable_write(&spec, rendered.as_bytes())?;
                    }
                    specs.push(spec);
                    break;
                };
                let state = row.get("state").and_then(Value::as_str).unwrap_or("");
                if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                    harvested = Some(fetch_receipts(
                        &generation_root,
                        &dataset.dataset_hash,
                        generation,
                        shard,
                        attempt,
                        &id,
                    )?);
                    break;
                }
                if terminal_infrastructure_state(state)
                    || (state == "completed"
                        && row.get("exit_code").and_then(Value::as_i64) != Some(0))
                {
                    if scheduler_preemption(row) {
                        continue;
                    }
                    infrastructure_failures += 1;
                    if infrastructure_failures == 3 {
                        bail!("evaluation {id} exhausted infrastructure retries");
                    }
                    continue;
                }
                break;
            }
            if let Some(path) = harvested {
                receipts.push(path);
            }
        }
        if receipts.len() == SHARD_SIZES.len() {
            let mut slots = slots_from_receipts(
                &plan,
                dataset,
                generation,
                &population,
                &manifests,
                &receipts,
            )?;
            slots.sort_by(slot_order);
            let generation_best_slot = slots.iter().find(|slot| slot.valid);
            let generation_best = generation_best_slot.and_then(|slot| slot.score.clone());
            let generation_best_member = generation_best_slot.map(|slot| slot.member.clone());
            let generation_best_predictions_sha256 =
                generation_best_slot.and_then(|slot| slot.predictions_sha256.clone());
            let generation_best_probabilities_sha256 =
                generation_best_slot.and_then(|slot| slot.probabilities_sha256.clone());
            let (best, best_member, best_predictions_sha256, best_probabilities_sha256) =
                if generation == 0 {
                    (
                        generation_best.clone(),
                        generation_best_member.clone(),
                        generation_best_predictions_sha256.clone(),
                        generation_best_probabilities_sha256.clone(),
                    )
                } else {
                    let previous = read_checkpoint(&root, dataset, generation - 1)?;
                    match (&previous.best, &generation_best) {
                        (Some(old), Some(new)) if better_score(new, old) => (
                            generation_best.clone(),
                            generation_best_member.clone(),
                            generation_best_predictions_sha256.clone(),
                            generation_best_probabilities_sha256.clone(),
                        ),
                        (Some(old), _) => (
                            Some(old.clone()),
                            previous.best_member.clone(),
                            previous.best_predictions_sha256.clone(),
                            previous.best_probabilities_sha256.clone(),
                        ),
                        (None, _) => (
                            generation_best.clone(),
                            generation_best_member.clone(),
                            generation_best_predictions_sha256.clone(),
                            generation_best_probabilities_sha256.clone(),
                        ),
                    }
                };
            let next = if generation + 1 == MAX_GENERATIONS {
                Vec::new()
            } else {
                next_population(dataset, &slots, generation + 1)?
            };
            let checkpoint = GenerationCheckpoint {
                schema: CHECKPOINT_SCHEMA.to_string(),
                run_id: RUN_ID.to_string(),
                dataset_hash: dataset.dataset_hash.clone(),
                leaderboard_dataset_hash: dataset.leaderboard_dataset_hash.clone(),
                comparator_sha256: dataset.comparator_sha256.clone(),
                generation,
                test_adaptive: true,
                benchmark_valid: false,
                slots,
                generation_best,
                generation_best_member,
                generation_best_predictions_sha256,
                generation_best_probabilities_sha256,
                best,
                best_member,
                best_predictions_sha256,
                best_probabilities_sha256,
                budget_complete: generation + 1 == MAX_GENERATIONS,
                next_population: next,
                completed_at_unix_ms: unix_millis(),
            };
            immutable_write(
                &checkpoint_path(&root, &dataset.dataset_hash, generation),
                &canonical_json(&checkpoint)?,
            )?;
            harvested_generations += 1;
        }
    }
    submit_specs(&root, &specs)?;
    if let Some(summary) = terminal_state(&plan, &root, &snapshot)? {
        let in_flight = active_datasets(&plan, &root);
        if in_flight.is_empty() {
            cancel_queued_run_jobs()?;
            write_campaign_terminal(&plan, &root, &snapshot, summary)?;
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": RUN_ID,
            "test_adaptive": true,
            "benchmark_valid": false,
            "submitted_specs": specs.len(),
            "active_datasets": ACTIVE_DATASETS,
            "bounded_queue_jobs": ACTIVE_DATASETS * SHARD_SIZES.len(),
            "harvested_generations": harvested_generations,
            "resume": resume,
        }))?
    );
    Ok(())
}

fn validate_ledger_chain(root: &Path, dataset: &DatasetBinding) -> Result<(String, f64)> {
    let path = ledger_path(root, &dataset.dataset_hash);
    let ledger = read_json(&path)?;
    if ledger.get("schema").and_then(Value::as_str) != Some(LEDGER_SCHEMA)
        || ledger.get("run_id").and_then(Value::as_str) != Some(RUN_ID)
        || ledger.get("dataset_hash").and_then(Value::as_str) != Some(dataset.dataset_hash.as_str())
        || ledger
            .get("leaderboard_dataset_hash")
            .and_then(Value::as_str)
            != Some(dataset.leaderboard_dataset_hash.as_str())
        || ledger.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || ledger.get("selection_bias").and_then(Value::as_str) != Some("held-out-test-selected")
        || ledger.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || ledger
            .get("official_combined_rank_eligible")
            .and_then(Value::as_bool)
            != Some(false)
        || ledger.get("comparator_sha256").and_then(Value::as_str)
            != Some(dataset.comparator_sha256.as_str())
    {
        bail!(
            "{} ledger identity or disclosure drift",
            dataset.dataset_hash
        );
    }
    let status = required_string(&ledger, "/status")?.to_string();
    if !matches!(
        status.as_str(),
        "strict_winner" | "ceiling_matched" | "v3_improvement" | "exhausted"
    ) {
        bail!("{} has a nonterminal ledger status", dataset.dataset_hash);
    }
    let generation = ledger
        .get("generation")
        .and_then(Value::as_u64)
        .context("ledger lacks generation")? as usize;
    let completed = ledger
        .get("completed_generations")
        .and_then(Value::as_u64)
        .context("ledger lacks completed_generations")? as usize;
    if completed == 0
        || completed > MAX_GENERATIONS
        || completed != generation + 1
        || completed != MAX_GENERATIONS
    {
        bail!(
            "{} ledger has an invalid generation budget",
            dataset.dataset_hash
        );
    }
    for index in 0..completed {
        read_checkpoint(root, dataset, index)?;
    }
    let final_checkpoint = read_checkpoint(root, dataset, generation)?;
    for (ledger_pointer, expected) in [
        (
            "/semantic_sha256",
            final_checkpoint
                .best_member
                .as_ref()
                .map(|member| &member.semantic_sha256),
        ),
        (
            "/canonical_genome",
            final_checkpoint
                .best_member
                .as_ref()
                .map(|member| &member.canonical_genome),
        ),
        (
            "/predictions_sha256",
            final_checkpoint.best_predictions_sha256.as_ref(),
        ),
        (
            "/probabilities_sha256",
            final_checkpoint.best_probabilities_sha256.as_ref(),
        ),
    ] {
        let actual = ledger.pointer(ledger_pointer).and_then(Value::as_str);
        ensure!(
            actual == expected.map(String::as_str),
            "{} ledger champion evidence drift",
            dataset.dataset_hash
        );
    }
    if let Some(score) = &final_checkpoint.best {
        ensure!(
            ledger
                .get("macro_f1")
                .and_then(Value::as_f64)
                .is_some_and(|actual| (actual - score.macro_f1).abs() <= REPLAY_SCORE_TOLERANCE),
            "{} ledger champion score drift",
            dataset.dataset_hash
        );
    }
    if ledger.get("checkpoint_sha256").and_then(Value::as_str)
        != Some(sha256_file(&checkpoint_path(root, &dataset.dataset_hash, generation))?.as_str())
    {
        bail!("{} ledger checkpoint hash mismatch", dataset.dataset_hash);
    }
    let score = ledger.get("macro_f1").and_then(Value::as_f64);
    let replays = ledger
        .get("replays")
        .and_then(Value::as_array)
        .context("ledger lacks replays")?;
    if status == "exhausted" {
        if !replays.is_empty()
            || score.is_some_and(|score| score > dataset.comparator_macro_f1)
            || score.is_some_and(|score| score > dataset.v2_macro_f1)
        {
            bail!(
                "{} exhausted ledger contains a claimed winner",
                dataset.dataset_hash
            );
        }
    } else {
        if replays.len() != 3 {
            bail!(
                "{} winner lacks three replay receipts",
                dataset.dataset_hash
            );
        }
        for pointer in [
            "/canonical_genome",
            "/semantic_sha256",
            "/dataset_hash",
            "/comparator_sha256",
            "/prediction_sha256",
            "/probability_sha256",
            "/macro_f1",
            "/accuracy",
        ] {
            let identities = replays
                .iter()
                .map(|row| {
                    serde_json::to_string(row.pointer(pointer).unwrap_or(&Value::Null)).unwrap()
                })
                .collect::<BTreeSet<_>>();
            if identities.len() != 1 {
                bail!(
                    "{} replay identity differs at {pointer}",
                    dataset.dataset_hash
                );
            }
        }
        for replay in replays {
            let receipt = required_string(replay, "/receipt_path")?;
            if replay.get("receipt_sha256").and_then(Value::as_str)
                != Some(sha256_file(Path::new(receipt))?.as_str())
            {
                bail!("{} replay receipt hash drift", dataset.dataset_hash);
            }
        }
        let score = score.context("winner ledger lacks macro-F1")?;
        let expected = if dataset.comparator_macro_f1 == 1.0 && score == 1.0 {
            "ceiling_matched"
        } else if score > dataset.comparator_macro_f1 {
            "strict_winner"
        } else if score > dataset.v2_macro_f1 {
            "v3_improvement"
        } else {
            bail!(
                "{} winner does not meet the frozen comparator",
                dataset.dataset_hash
            );
        };
        if status != expected {
            bail!("{} winner status is inconsistent", dataset.dataset_hash);
        }
    }
    let overlay = ledger
        .get("overlay_macro_f1")
        .and_then(Value::as_f64)
        .context("ledger lacks overlay score")?;
    if overlay.to_bits()
        != dataset
            .v2_macro_f1
            .max(score.unwrap_or(f64::NEG_INFINITY))
            .to_bits()
    {
        bail!(
            "{} overlay does not preserve the better of v2 and v3",
            dataset.dataset_hash
        );
    }
    Ok((status, overlay))
}

fn validate_impl(run_id: &str, campaign_root: &Path) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let summary_path = root.join("terminal-summary.v3.json");
    let summary = read_json(&summary_path)?;
    if summary.get("schema").and_then(Value::as_str) != Some(SUMMARY_SCHEMA)
        || summary.get("run_id").and_then(Value::as_str) != Some(RUN_ID)
        || summary.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || summary.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || summary.get("private").and_then(Value::as_bool) != Some(true)
    {
        bail!("terminal v3 summary disclosure or identity drift");
    }
    let merged_expected = plan
        .datasets
        .iter()
        .map(|dataset| dataset.dataset_hash.clone())
        .collect::<BTreeSet<_>>();
    let merged_rows = proposal_rows(Path::new(&plan.merged_proposal_set_path), &merged_expected)?;
    let merged = merge_proposals(&[merged_rows], &plan.execution_order)?;
    for dataset in &plan.datasets {
        if sha256_bytes(&serde_json::to_vec(&merged[&dataset.dataset_hash])?)
            != dataset.proposal_set_sha256
        {
            bail!("{} planned proposal set hash drift", dataset.dataset_hash);
        }
    }
    let mut strict_winners = 0;
    let mut v3_v2_improvements = 0;
    let mut overlay = BTreeMap::new();
    let mut terminal = 0;
    for dataset in &plan.datasets {
        if dataset_terminal(&root, &dataset.dataset_hash) {
            let (status, score) = validate_ledger_chain(&root, dataset)?;
            strict_winners += usize::from(status == "strict_winner");
            v3_v2_improvements += usize::from(
                read_json(&ledger_path(&root, &dataset.dataset_hash))?
                    .get("v3_improves_v2")
                    .and_then(Value::as_bool)
                    == Some(true),
            );
            overlay.insert(dataset.leaderboard_dataset_hash.clone(), score);
            terminal += 1;
        }
    }
    let snapshot: ComparatorSnapshot =
        serde_json::from_slice(&fs::read(&plan.comparator_snapshot_path)?)?;
    verify_public_sotarena_unchanged(&snapshot)?;
    ensure!(
        summary.get("candidate_evaluations").and_then(Value::as_u64)
            == Some(TOTAL_CANDIDATE_EVALUATIONS as u64),
        "terminal summary does not declare the full 403200-candidate budget"
    );
    ensure!(
        summary
            .get("public_sotarena_unchanged")
            .and_then(Value::as_bool)
            == Some(true),
        "terminal summary does not prove public SOTArena immutability"
    );
    let exact_ranking = ranking(&snapshot, &overlay);
    let first = aspfm_is_first(&exact_ranking);
    let expected_terminal = summary
        .get("terminal_datasets")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let expected_strict = summary
        .get("strict_winners")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if terminal != TARGET_COUNT
        || terminal != expected_terminal
        || strict_winners != expected_strict
        || summary.get("v3_v2_improvements").and_then(Value::as_u64)
            != Some(v3_v2_improvements as u64)
        || summary.get("ranking") != Some(&serde_json::to_value(&exact_ranking)?)
        || summary.get("status").and_then(Value::as_str) != Some("completed_full_budget")
    {
        bail!("terminal summary does not replay from immutable ledgers");
    }
    let aspfm = exact_ranking
        .iter()
        .find(|row| row.participant == "family:aspfm")
        .unwrap();
    let tabfm = exact_ranking
        .iter()
        .find(|row| row.participant == "family:tabfm-ensemble")
        .unwrap();
    let baseline_aspfm = snapshot
        .baseline_ranking
        .iter()
        .find(|row| row.participant == "family:aspfm")
        .context("baseline ranking lacks ASPFM")?;
    let accepted =
        first && v3_v2_improvements > 0 && aspfm.elo > baseline_aspfm.elo && aspfm.elo > tabfm.elo;
    if summary.get("acceptance").and_then(Value::as_bool) != Some(accepted) {
        bail!("v3 acceptance flag does not match exact ranking and replay evidence");
    }
    let proof_path = root.join("proof-bundle-private-v3/manifest.v3.json");
    let proof = read_json(&proof_path)?;
    if proof.get("schema").and_then(Value::as_str) != Some(PROOF_SCHEMA)
        || proof.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || proof.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || proof.get("private").and_then(Value::as_bool) != Some(true)
        || proof.get("publishable").and_then(Value::as_bool) != Some(false)
        || proof.get("terminal_summary_sha256").and_then(Value::as_str)
            != Some(sha256_file(&summary_path)?.as_str())
        || !root.join("priority-restored.v3.json").is_file()
        || !root.join("priority-gpu-restored.v3.json").is_file()
    {
        bail!("private proof bundle or priority restoration is incomplete");
    }
    let provenance = proof
        .get("catalog_provenance")
        .context("proof bundle lacks catalog provenance")?;
    for (path_pointer, hash_pointer) in [
        ("/base_catalog", "/base_catalog_sha256"),
        ("/base_data", "/base_data_sha256"),
        ("/extended_catalog", "/extended_catalog_sha256"),
        ("/extended_data", "/extended_data_sha256"),
        ("/extension_data", "/extension_data_sha256"),
    ] {
        let path = Path::new(required_string(provenance, path_pointer)?);
        ensure!(
            sha256_file(path)? == required_sha256(provenance, hash_pointer)?,
            "proof catalog provenance hash drift at {}",
            path.display()
        );
    }
    ensure!(
        proof
            .pointer("/generation_zero/rows")
            .and_then(Value::as_u64)
            == Some((TARGET_COUNT * POPULATION) as u64)
            && proof
                .pointer("/generation_zero/candidate_evaluations")
                .and_then(Value::as_u64)
                == Some(TOTAL_CANDIDATE_EVALUATIONS as u64),
        "proof generation-zero or full-budget counts drifted"
    );
    let before_priority = read_json(&root.join("priority-before.v3.json"))?;
    let restored_priority = read_json(&root.join("priority-restored.v3.json"))?;
    ensure!(
        restored_priority.get("restored_override_weight") == before_priority.get("override_weight"),
        "priority restoration receipt does not match the frozen prior override"
    );
    ensure!(
        priority_row()?.get("override_weight") == before_priority.get("override_weight"),
        "SCQ priority override was not restored"
    );
    let before_gpu_priority = read_json(&root.join("priority-gpu-before.v3.json"))?;
    let restored_gpu_priority = read_json(&root.join("priority-gpu-restored.v3.json"))?;
    ensure!(
        restored_gpu_priority.get("restored_override_weight")
            == before_gpu_priority.get("override_weight"),
        "GPU priority restoration receipt does not match the frozen prior override"
    );
    ensure!(
        priority_row_for(GPU_PROJECT)?.get("override_weight")
            == before_gpu_priority.get("override_weight"),
        "SCQ GPU priority override was not restored"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": RUN_ID,
            "test_adaptive": true,
            "benchmark_valid": false,
            "private": true,
            "terminal_datasets": terminal,
            "strict_winners": strict_winners,
            "aspfm_elo": aspfm.elo,
            "tabfm_ensemble_elo": tabfm.elo,
            "acceptance": accepted,
            "success": true,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_snapshot() -> ComparatorSnapshot {
        let elo = EloConfig {
            initial_rating: 1500.0,
            k_factor: 40.0,
            tie_tolerance: 1e-12,
            iterations: 250,
            regularization: 0.0001,
            convergence_tolerance: 1e-8,
        };
        let participant_scores = BTreeMap::from([
            (
                "family:aspfm".to_string(),
                BTreeMap::from([("a".to_string(), 0.6), ("b".to_string(), 0.7)]),
            ),
            (
                "family:tabfm-ensemble".to_string(),
                BTreeMap::from([("a".to_string(), 0.8), ("b".to_string(), 0.7)]),
            ),
        ]);
        ComparatorSnapshot {
            schema: SNAPSHOT_SCHEMA.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            sotarena_root: String::new(),
            config_path: String::new(),
            config_sha256: "0".repeat(64),
            report_path: String::new(),
            report_sha256: "0".repeat(64),
            elo,
            participant_display_names: BTreeMap::from([
                ("family:aspfm".to_string(), "ASPFM".to_string()),
                (
                    "family:tabfm-ensemble".to_string(),
                    "TabFM Ensemble".to_string(),
                ),
            ]),
            participant_scores,
            baseline_ranking: Vec::new(),
            datasets: Vec::new(),
        }
    }

    #[test]
    fn frozen_v3_config_is_pop64_gen30_with_four_equal_shards() {
        let config = CampaignConfig::fixed();
        assert_eq!(config.population, 64);
        assert_eq!(config.generations, 30);
        assert_eq!(config.shard_sizes, [16, 16, 16, 16]);
        assert_eq!(config.active_datasets, 5);
        assert_eq!(config.lead_recovery_cohort, 45);
        assert_eq!(config.total_targets, 210);
        assert_eq!(config.total_candidate_evaluations, 403_200);
        assert!(config.early_stop.starts_with("disabled;"));
        assert_eq!(config.proposal_seeds, PROPOSAL_SEEDS);
    }

    #[test]
    fn target_order_contains_117_positive_gaps_then_93_headroom_rows() {
        let row = |index: usize, gap: f64, aspfm_v2: f64| FrozenArenaDataset {
            leaderboard_dataset_hash: format!("{index:03}"),
            dataset_hash: format!("{index:03}"),
            dataset_manifest_path: String::new(),
            dataset_manifest_sha256: "0".repeat(64),
            scores_path: String::new(),
            scores_sha256: "0".repeat(64),
            scores: Vec::new(),
            aspfm_v2,
            strict_best: FrozenScore {
                method_id: "best".to_string(),
                value: aspfm_v2 + gap,
                source_uri: String::new(),
                source_sha256: "0".repeat(64),
            },
            strict_gap: gap,
        };
        let mut snapshot = fixture_snapshot();
        snapshot.datasets = (0..117)
            .map(|index| row(index, (index + 1) as f64 / 10_000.0, 0.5))
            .chain((0..93).map(|index| row(117 + index, 0.0, 0.1 + index as f64 / 1000.0)))
            .collect();
        let ordered = ordered_targets(&snapshot).unwrap();
        assert_eq!(ordered.len(), TARGET_COUNT);
        assert!(ordered[..117].iter().all(|row| row.strict_gap > 0.0));
        assert!(ordered[117..].iter().all(|row| row.strict_gap <= 0.0));
        assert!(ordered[117].aspfm_v2 < ordered[209].aspfm_v2);
    }

    #[test]
    fn context_job_identity_is_content_addressed_and_private() {
        let request = "a".repeat(64);
        assert_eq!(
            context_job_id(&request),
            "jope-mc210-context-aaaaaaaaaaaaaaaa-qmc4-v3"
        );
        assert_ne!(context_job_id(&request), context_job_id(&"b".repeat(64)));
        assert_eq!(CONTEXT_EXTENSION_DATASETS.len(), 3);
        assert_eq!(CONTEXT_RECORD_STRIDE_BYTES, 8192);
    }

    #[test]
    fn deterministic_target_order_uses_gap_then_dataset_hash() {
        let row = |hash: &str, gap: f64| FrozenArenaDataset {
            leaderboard_dataset_hash: hash.to_string(),
            dataset_hash: hash.to_string(),
            dataset_manifest_path: String::new(),
            dataset_manifest_sha256: "0".repeat(64),
            scores_path: String::new(),
            scores_sha256: "0".repeat(64),
            scores: Vec::new(),
            aspfm_v2: 0.5,
            strict_best: FrozenScore {
                method_id: "best".to_string(),
                value: 0.5 + gap,
                source_uri: String::new(),
                source_sha256: "0".repeat(64),
            },
            strict_gap: gap,
        };
        let mut rows = vec![row("c", 0.2), row("b", 0.1), row("a", 0.1)];
        rows.sort_by(|left, right| {
            left.strict_gap.total_cmp(&right.strict_gap).then_with(|| {
                left.leaderboard_dataset_hash
                    .cmp(&right.leaderboard_dataset_hash)
            })
        });
        assert_eq!(
            rows.iter()
                .map(|row| row.dataset_hash.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn proposal_merge_rejects_duplicates_and_deficient_populations() {
        let datasets = vec!["a".to_string()];
        let row = |semantic: &str| json!({"dataset_hash":"a","semantic_sha256":semantic});
        let first = (0..63)
            .map(|index| row(&format!("{index:064x}")))
            .collect::<Vec<_>>();
        assert!(merge_proposals(std::slice::from_ref(&first), &datasets).is_err());
        let mut second = vec![row(&format!("{:064x}", 62))];
        second.push(row(&format!("{:064x}", 63)));
        assert_eq!(
            merge_proposals(&[first, second], &datasets).unwrap()["a"].len(),
            64
        );
    }

    #[test]
    fn exact_overlay_can_move_aspfm_above_tabfm() {
        let mut snapshot = fixture_snapshot();
        snapshot.baseline_ranking = ranking(&snapshot, &BTreeMap::new());
        assert!(!aspfm_is_first(&snapshot.baseline_ranking));
        let rows = ranking(&snapshot, &BTreeMap::from([("a".to_string(), 0.9)]));
        assert!(aspfm_is_first(&rows));
    }

    #[test]
    fn content_addressed_job_ids_prevent_duplicate_submission_identity() {
        let hash = "a".repeat(64);
        assert_eq!(
            eval_job_id("dataset", 2, 3, 0, &hash),
            eval_job_id("dataset", 2, 3, 0, &hash)
        );
        assert_ne!(
            eval_job_id("dataset", 2, 3, 0, &hash),
            eval_job_id("dataset", 2, 3, 1, &hash)
        );
        assert_ne!(
            eval_job_id("dataset", 2, 3, 0, &hash),
            eval_job_id("dataset", 2, 3, 0, &"b".repeat(64))
        );
    }

    #[test]
    fn generation_budget_has_no_post_29_continuation() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoints = directory.path().join("datasets/d/checkpoints");
        fs::create_dir_all(&checkpoints).unwrap();
        for generation in 0..MAX_GENERATIONS {
            fs::write(
                checkpoints.join(format!("generation-{generation:03}.v3.json")),
                b"{}",
            )
            .unwrap();
        }
        assert_eq!(next_generation(directory.path(), "d"), MAX_GENERATIONS);
    }

    #[test]
    fn resumption_starts_at_first_missing_generation() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoints = directory.path().join("datasets/d/checkpoints");
        fs::create_dir_all(&checkpoints).unwrap();
        fs::write(checkpoints.join("generation-000.v3.json"), b"{}").unwrap();
        fs::write(checkpoints.join("generation-002.v3.json"), b"{}").unwrap();
        assert_eq!(next_generation(directory.path(), "d"), 1);
    }

    #[test]
    fn comparator_lock_changes_when_frozen_input_changes() {
        let original = canonical_json(&json!({
            "method_id": "strongest",
            "macro_f1": 0.8,
            "source_sha256": "a".repeat(64),
        }))
        .unwrap();
        let changed = canonical_json(&json!({
            "method_id": "strongest",
            "macro_f1": 0.81,
            "source_sha256": "a".repeat(64),
        }))
        .unwrap();
        assert_ne!(sha256_bytes(&original), sha256_bytes(&changed));
    }
}
