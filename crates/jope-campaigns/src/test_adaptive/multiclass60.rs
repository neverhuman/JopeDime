//! Frozen held-out-test-adaptive multiclass campaign contract.
//!
//! This lane is deliberately exploratory. Test macro-F1 is evolutionary
//! fitness, so every artifact is benchmark-invalid and excluded from official
//! combined rankings.

use super::*;
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::{ParamValue, Stage};
use feat_math::{ColumnSpec, DataSchema, GenomeIr, Objective, RowSemantics, TaskSpec};
use serde_json::json;

pub(crate) const RUN_ID: &str = "multiclass272-test-adaptive-union43-gen20-v2";
const PLAN_SCHEMA: &str = "JopeMulticlass272TestAdaptivePlan.v2";
const LEDGER_SCHEMA: &str = "JopeMulticlass60ChampionLedger.v2";
const CORPUS_SCHEMA: &str = "multiclass-union-runnable-v5";
const COMPARATOR_SCHEMA: &str = "test-adaptive-comparator-lock-v2";
const COMPARATOR_SET_SCHEMA: &str = "test-adaptive-comparator-set.v2";
const COMPARATOR_SET_FILE: &str = "comparator-set.v2.json";
const MC_COVERAGE_MANIFEST: &str =
    "/home/ubuntu/remote_super/quant/releases/mc-union-runnable272-v5/manifest.json";
const MC_COVERAGE_MANIFEST_SHA256: &str =
    "93888beb0d24f7cb7f0def66c88127a6cee6d0606c25c11ddff82161506e4938";
const MC_COVERAGE_RECEIPT: &str =
    "/home/ubuntu/remote_super/quant/releases/mc-union-runnable272-v5/receipts/final.json";
const TS_COVERAGE_MANIFEST: &str =
    "/home/ubuntu/remote_super/quant/releases/timeseries42-forecast3-v10/manifest.json";
const TS_COVERAGE_MANIFEST_SHA256: &str =
    "dda881dab5eeebcd7293081df5838062c97cfb5e15b9383776a4a3630f32c06d";
const TS_COVERAGE_RECEIPT: &str =
    "/home/ubuntu/remote_super/quant/releases/timeseries42-forecast3-v10/receipts/final.json";
const TS_BASELINE_RECEIPT: &str =
    "/home/ubuntu/remote_super/quant/releases/mc272-ts42-coverage-v1/receipts/timeseries-baseline-all21.json";
const DATA_ROOT: &str = "/home/ubuntu/remote_super/quant/multi_label";
const SCQ: &str = "/home/ubuntu/super_cluster/scq";
const CORPUS_EVAL: &str = "/home/ubuntu/jope-dime/target/release/corpus_eval";
const PROJECT: &str = "jope-multiclass-test-adaptive";
const JOPE_MULTICLASS_REPORT_FILE: &str = "jope-dime-test-adaptive-union43-gen20-v2.json";
const POPULATION: usize = 60;
const SHARD_SIZES: [usize; 4] = [15, 15, 15, 15];
const NODES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const PRIORITY_DATASETS: usize = 272;
// Frozen into the existing v1 plan. Capacity sharing with TimeSeries42 is
// enforced by SCQ admission, not by mutating this immutable identity.
const ACTIVE_DATASETS: usize = 5;
// Shards are often sub-second on small datasets, while a complete SCQ
// reconcile/harvest pass takes several seconds.  Keep multiple waves visible
// to SCQ so its global core and memory accounting, rather than controller
// latency, controls the number that actually run at once.
const QUEUE_BUFFER_DATASETS: usize = 64;
const SPEC_REVISION: usize = 2;
const REPLAY_SPEC_REVISION: usize = 3;
const REPLAY_ROLE_CAPABILITIES: [(&str, &str); 3] = [
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
const MIN_GENERATIONS: usize = 0;
const MAX_GENERATIONS: usize = 20;
const MAX_SCHEDULER_PREEMPTIONS: usize = 64;
const MULTICLASS_HEADS: [&str; 4] = [
    "multiclass_ridge",
    "multiclass_ovr_logistic",
    "multiclass_diagonal_lda",
    "multiclass_nearest_centroid",
];
const FEATURE_BUDGETS: [usize; 4] = [32, 64, 128, 256];
const FAST_BASELINES: [&str; 3] = [
    "mc20_hist_gradient_boosting",
    "mc20_extra_trees",
    "mc20_random_forest",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Config {
    population: usize,
    elites: usize,
    parent_pool: usize,
    tournament: usize,
    mutation_probability: f64,
    crossover_probability: f64,
    minimum_generations: usize,
    maximum_generations: usize,
    shard_sizes: Vec<usize>,
    nodes: Vec<String>,
    active_datasets: usize,
    candidate_threads: usize,
    candidate_timeout_seconds: usize,
    infrastructure_retries: usize,
    replay_nodes: usize,
    ranking: String,
    win_rule: String,
}

impl Config {
    fn fixed() -> Self {
        Self {
            population: POPULATION,
            elites: 7,
            parent_pool: 30,
            tournament: 3,
            mutation_probability: 0.75,
            crossover_probability: 0.25,
            minimum_generations: MIN_GENERATIONS,
            maximum_generations: MAX_GENERATIONS,
            shard_sizes: SHARD_SIZES.to_vec(),
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            active_datasets: ACTIVE_DATASETS,
            candidate_threads: 1,
            candidate_timeout_seconds: 180,
            infrastructure_retries: 1,
            replay_nodes: 3,
            ranking: "test macro_f1 descending, test accuracy descending, complexity ascending, semantic_sha256 ascending".to_string(),
            win_rule: "stop after the first confirmed strict beat of the frozen 43-model comparator, including generation 0; exact 1.0 matches an exact 1.0 incumbent as ceiling_matched; otherwise publish exhausted after generation 19".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SeedMember {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Incumbent {
    method: String,
    macro_f1: f64,
    accuracy: f64,
    report_path: String,
    report_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetBinding {
    order: usize,
    wave: String,
    dataset_hash: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
    class_count: usize,
    n_features: usize,
    fast_baselines_complete: bool,
    completed_incumbent_count: usize,
    comparator_sha256: String,
    strongest_available_incumbent: Option<Incumbent>,
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
    project: String,
    project_weight: usize,
    project_max_gpus: usize,
    scq_binary: String,
    scq_binary_sha256: String,
    release_binary_sha256: String,
    corpus_eval: String,
    corpus_eval_sha256: String,
    corpus_manifest: String,
    corpus_manifest_sha256: String,
    multiclass_coverage_final_receipt_sha256: String,
    timeseries_coverage_manifest_sha256: String,
    timeseries_coverage_final_receipt_sha256: String,
    comparator_root: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
    seed_catalog: String,
    seed_catalog_sha256: String,
    seed_population_sha256: String,
    priority_count: usize,
    execution_order: Vec<String>,
    fast_baselines: Vec<String>,
    config: Config,
    seeds: Vec<SeedMember>,
    datasets: Vec<DatasetBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CoverageGate {
    multiclass_receipt_sha256: String,
    timeseries_receipt_sha256: String,
    comparator_set_path: String,
    comparator_set_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct RankedScore {
    pub macro_f1: f64,
    pub accuracy: f64,
    pub complexity: usize,
    pub semantic_sha256: String,
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
    comparator_sha256: String,
    generation: usize,
    test_adaptive: bool,
    selection_bias: String,
    benchmark_valid: bool,
    slots: Vec<LogicalSlot>,
    best: Option<RankedScore>,
    best_member: Option<SeedMember>,
    initial_budget_complete: bool,
    continuation_lineage_salt: Option<String>,
    next_population: Vec<SeedMember>,
    completed_at_unix_ms: u128,
}

const CHECKPOINT_SCHEMA: &str = "JopeMulticlass272GenerationCheckpoint.v1";

pub(crate) fn rank_order(left: &RankedScore, right: &RankedScore) -> Ordering {
    right
        .macro_f1
        .total_cmp(&left.macro_f1)
        .then_with(|| right.accuracy.total_cmp(&left.accuracy))
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(crate) fn qualifies(
    candidate: &RankedScore,
    generation: usize,
    is_seed: bool,
    generation_zero_best: f64,
    incumbent: f64,
    final_twenty_complete: bool,
) -> bool {
    let _ = (is_seed, generation_zero_best, final_twenty_complete);
    if !candidate.macro_f1.is_finite() {
        return false;
    }
    let _ = generation;
    candidate.macro_f1 > incumbent || (incumbent == 1.0 && candidate.macro_f1 == 1.0)
}

fn canonical_manifest_sha256(mut value: Value) -> Result<String> {
    value
        .as_object_mut()
        .context("manifest must be an object")?
        .remove("manifest_sha256");
    Ok(sha256_bytes(&serde_json::to_vec(&value)?))
}

fn required_hash(value: &Value, pointer: &str) -> Result<String> {
    let hash = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing {pointer}"))?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{pointer} is not a SHA-256");
    }
    Ok(hash.to_string())
}

fn read_value(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

fn validate_source_manifest(path: &Path, schema: &str) -> Result<Value> {
    let value = read_value(path)?;
    if value.get("schema_version").and_then(Value::as_str) != Some(schema) {
        bail!("{} has the wrong schema", path.display());
    }
    let embedded = required_hash(&value, "/manifest_sha256")?;
    if canonical_manifest_sha256(value.clone())? != embedded {
        bail!("{} manifest SHA-256 mismatch", path.display());
    }
    Ok(value)
}

fn validate_coverage_receipt(
    path: &Path,
    campaign_id: &str,
    manifest_sha256: &str,
    task: &str,
    expected: usize,
) -> Result<String> {
    let receipt = read_value(path)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("coverage.final-receipt.v1")
        || receipt.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
        || receipt
            .get("campaign_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || receipt.get("task").and_then(Value::as_str) != Some(task)
        || receipt.get("expected").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("valid").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("missing").and_then(Value::as_u64) != Some(0)
        || receipt.get("invalid").and_then(Value::as_u64) != Some(0)
        || receipt.get("complete").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(expected)
    {
        bail!("invalid or incomplete coverage receipt {}", path.display());
    }
    let embedded = required_hash(&receipt, "/receipt_sha256")?;
    let mut unsigned = receipt;
    unsigned
        .as_object_mut()
        .context("coverage receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("coverage receipt digest drift at {}", path.display());
    }
    validate_receipt_report_files(&unsigned, expected, true)?;
    Ok(embedded)
}

fn coverage_gate(comparator_root: &Path) -> Result<CoverageGate> {
    let mc_manifest = validate_source_manifest(Path::new(MC_COVERAGE_MANIFEST), CORPUS_SCHEMA)?;
    if required_hash(&mc_manifest, "/manifest_sha256")? != MC_COVERAGE_MANIFEST_SHA256 {
        bail!("multiclass successor manifest pin drift");
    }
    let ts_manifest = read_value(Path::new(TS_COVERAGE_MANIFEST))?;
    if ts_manifest.get("schema_version").and_then(Value::as_str)
        != Some("timeseries42-forecast-campaign-v10")
        || required_hash(&ts_manifest, "/manifest_sha256")? != TS_COVERAGE_MANIFEST_SHA256
        || canonical_manifest_sha256(ts_manifest)? != TS_COVERAGE_MANIFEST_SHA256
    {
        bail!("timeseries successor manifest pin drift");
    }
    let multiclass_receipt_sha256 = validate_coverage_receipt(
        Path::new(MC_COVERAGE_RECEIPT),
        "mc-union-runnable272-v5",
        MC_COVERAGE_MANIFEST_SHA256,
        "multiclass",
        11_696,
    )?;
    let timeseries_receipt_sha256 = validate_coverage_receipt(
        Path::new(TS_COVERAGE_RECEIPT),
        "timeseries42-forecast3-v10",
        TS_COVERAGE_MANIFEST_SHA256,
        "timeseries",
        126,
    )?;
    let baseline = read_value(Path::new(TS_BASELINE_RECEIPT))?;
    if baseline.get("schema").and_then(Value::as_str)
        != Some("timeseries42.baseline-all21-final-receipt.v1")
        || baseline.get("expected").and_then(Value::as_u64) != Some(882)
        || baseline.get("completed_numeric").and_then(Value::as_u64) != Some(877)
        || baseline.get("skipped_ineligible").and_then(Value::as_u64) != Some(5)
        || baseline.get("missing").and_then(Value::as_u64) != Some(0)
        || baseline.get("invalid").and_then(Value::as_u64) != Some(0)
        || baseline.get("errored").and_then(Value::as_u64) != Some(0)
        || baseline.get("complete").and_then(Value::as_bool) != Some(true)
        || baseline
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(882)
    {
        bail!("invalid exact 877+5 time-series baseline receipt");
    }
    let baseline_receipt_sha256 = required_hash(&baseline, "/receipt_sha256")?;
    let mut unsigned_baseline = baseline;
    unsigned_baseline
        .as_object_mut()
        .context("baseline receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned_baseline)?) != baseline_receipt_sha256 {
        bail!("time-series baseline receipt digest drift");
    }
    validate_receipt_report_files(&unsigned_baseline, 882, false)?;
    let comparator_set_path = comparator_root.join(COMPARATOR_SET_FILE);
    let comparator_set = read_value(&comparator_set_path)?;
    if comparator_set.get("schema").and_then(Value::as_str) != Some(COMPARATOR_SET_SCHEMA)
        || comparator_set.get("root").and_then(Value::as_str)
            != Some(comparator_root.to_string_lossy().as_ref())
        || comparator_set
            .get("multiclass_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(multiclass_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_baseline_receipt_sha256")
            .and_then(Value::as_str)
            != Some(baseline_receipt_sha256.as_str())
        || comparator_set.get("lock_count").and_then(Value::as_u64) != Some(314)
    {
        bail!(
            "invalid post-coverage comparator set {}",
            comparator_set_path.display()
        );
    }
    let locks = comparator_set
        .get("locks")
        .and_then(Value::as_array)
        .context("comparator set lacks locks")?;
    let identities = locks
        .iter()
        .filter_map(|row| {
            Some((
                row.get("task")?.as_str()?,
                row.get("dataset_hash")?.as_str()?,
            ))
        })
        .collect::<BTreeSet<_>>();
    if locks.len() != 314
        || identities.len() != 314
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("multiclass"))
            .count()
            != 272
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("timeseries"))
            .count()
            != 42
    {
        bail!("post-coverage comparator set membership drift");
    }
    let embedded = required_hash(&comparator_set, "/comparator_set_sha256")?;
    let mut unsigned = comparator_set;
    unsigned
        .as_object_mut()
        .context("comparator set must be an object")?
        .remove("comparator_set_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("post-coverage comparator set digest drift");
    }
    Ok(CoverageGate {
        multiclass_receipt_sha256,
        timeseries_receipt_sha256,
        comparator_set_path: comparator_set_path.display().to_string(),
        comparator_set_sha256: embedded,
    })
}

fn rewrite_seed(source: &str, slot: usize) -> Result<SeedMember> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass generation-zero seeds must be pipelines");
    };
    if pipeline.stages.iter().any(|stage| stage.op == "target") {
        bail!("unsafe target transform in generation-zero seed");
    }
    let head = pipeline
        .stages
        .pop()
        .context("generation-zero seed has no terminal head")?;
    if !feat_math::heads::is_head_op(&head.op) {
        bail!("unknown terminal head in generation-zero seed");
    }
    let mut replacement = Stage::new(MULTICLASS_HEADS[slot % MULTICLASS_HEADS.len()]);
    let l2_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
    match replacement.op.as_str() {
        "multiclass_ridge" => {
            replacement.params.insert(
                "class_weight".to_string(),
                ParamValue::Str("balanced".to_string()),
            );
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_ovr_logistic" => {
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_diagonal_lda" => {
            replacement.params.insert(
                "shrinkage".to_string(),
                ParamValue::Float([0.0, 0.01, 0.1, 0.5][slot % 4]),
            );
        }
        "multiclass_nearest_centroid" => {}
        _ => unreachable!("frozen multiclass head family"),
    }
    replacement.params.insert(
        "feature_budget".to_string(),
        ParamValue::Int(
            FEATURE_BUDGETS[(slot / MULTICLASS_HEADS.len()) % FEATURE_BUDGETS.len()] as i64,
        ),
    );
    pipeline.stages.push(replacement);
    let canonical = GenomeIr::Pipeline(pipeline).canonical();
    let genome = GenomeIr::parse(&canonical)?;
    let schema = DataSchema {
        columns: (0..8)
            .map(|feature| ColumnSpec::numeric(format!("f{feature}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    validate_v3_genome(&genome, &TaskSpec::multiclass(Objective::MacroF1)?, &schema)?;
    Ok(SeedMember {
        slot,
        semantic_sha256: sha256_bytes(canonical.as_bytes()),
        canonical_genome: canonical,
        source:
            "non-test-adaptive generation-zero transformation body; native multiclass head rewrite"
                .to_string(),
    })
}

fn seed_population(path: &Path) -> Result<Vec<SeedMember>> {
    let mut seen = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut members = Vec::new();
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut sources = Vec::new();
    for source_path in [path, fallback.as_path()] {
        let content = fs::read_to_string(source_path)?;
        sources.extend(
            content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string),
        );
    }
    for rewrite_variant in 0..24 {
        for (source_index, line) in sources.iter().enumerate() {
            let Ok(mut member) = rewrite_seed(line, source_index + rewrite_variant) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_head_family(&member.canonical_genome)?;
            let count = family_counts.entry(family).or_default();
            if *count >= POPULATION / MULTICLASS_HEADS.len() {
                continue;
            }
            *count += 1;
            member.slot = members.len();
            members.push(member);
            if members.len() == POPULATION {
                break;
            }
        }
        if members.len() == POPULATION {
            break;
        }
    }
    if members.len() != POPULATION {
        bail!(
            "multiclass generation-zero requires exactly {POPULATION} members, found {}",
            members.len()
        );
    }
    Ok(members)
}

fn multiclass_head_family(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member must be a pipeline");
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

fn seed_sources_sha256(path: &Path) -> Result<String> {
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut bytes = fs::read(path)?;
    bytes.push(0);
    bytes.extend_from_slice(&fs::read(fallback)?);
    Ok(sha256_bytes(&bytes))
}

fn dataset_entries(trusted: &Value) -> Result<BTreeMap<String, Value>> {
    let rows = trusted
        .get("datasets")
        .and_then(Value::as_array)
        .context("trusted manifest lacks datasets")?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("dataset_hash")?.as_str()?.to_string(), row.clone())))
        .collect())
}

fn bind_frozen_comparator(
    mut binding: DatasetBinding,
    comparator_root: &Path,
    gate: &CoverageGate,
) -> Result<Option<DatasetBinding>> {
    let dataset_hash = binding.dataset_hash.as_str();
    let comparator_path = comparator_root
        .join("multiclass")
        .join(dataset_hash)
        .join("comparator.json");
    if !comparator_path.is_file() {
        return Ok(None);
    }
    let comparator = read_value(&comparator_path)?;
    if comparator.get("schema").and_then(Value::as_str) != Some(COMPARATOR_SCHEMA)
        || comparator.get("task").and_then(Value::as_str) != Some("multiclass")
        || comparator.get("dataset_hash").and_then(Value::as_str) != Some(dataset_hash)
        || comparator.get("run_id").and_then(Value::as_str) != Some(RUN_ID)
        || comparator.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || comparator.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || comparator
            .get("coverage_campaign_id")
            .and_then(Value::as_str)
            != Some("mc-union-runnable272-v5")
        || comparator
            .get("coverage_manifest_sha256")
            .and_then(Value::as_str)
            != Some(MC_COVERAGE_MANIFEST_SHA256)
        || comparator
            .get("coverage_final_receipt_sha256")
            .and_then(Value::as_str)
            != Some(gate.multiclass_receipt_sha256.as_str())
        || comparator
            .get("accepted_report_count")
            .and_then(Value::as_u64)
            != Some(43)
        || comparator
            .get("accepted_reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(43)
    {
        bail!("{dataset_hash} comparator identity drift");
    }
    let protocol_sha256 = required_hash(&comparator, "/protocol_sha256")?;
    let current_evaluator_sha256 = sha256_file(Path::new(CORPUS_EVAL))?;
    if comparator
        .pointer("/protocol/protocol_sha256")
        .and_then(Value::as_str)
        != Some(protocol_sha256.as_str())
        || comparator
            .pointer("/protocol/evaluator/sha256")
            .and_then(Value::as_str)
            != Some(current_evaluator_sha256.as_str())
        || comparator
            .pointer("/protocol/minimum_generations")
            .and_then(Value::as_u64)
            != Some(MIN_GENERATIONS as u64)
        || comparator
            .pointer("/protocol/maximum_generations")
            .and_then(Value::as_u64)
            != Some(MAX_GENERATIONS as u64)
        || comparator
            .pointer("/protocol/population")
            .and_then(Value::as_u64)
            != Some(POPULATION as u64)
    {
        bail!("{dataset_hash} comparator protocol drift");
    }
    let mut protocol = comparator
        .get("protocol")
        .cloned()
        .context("comparator lacks protocol")?;
    protocol
        .as_object_mut()
        .context("comparator protocol must be an object")?
        .remove("protocol_sha256");
    if sha256_bytes(&serde_json::to_vec(&protocol)?) != protocol_sha256 {
        bail!("{dataset_hash} comparator protocol digest drift");
    }
    let embedded = required_hash(&comparator, "/lock_sha256")?;
    let mut unsigned = comparator.clone();
    unsigned
        .as_object_mut()
        .context("comparator must be an object")?
        .remove("lock_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("{dataset_hash} comparator digest drift");
    }
    let locked_files = comparator
        .get("dataset_file_hashes")
        .and_then(Value::as_object)
        .context("comparator lacks dataset_file_hashes")?;
    if locked_files.get("train.csv").and_then(Value::as_str) != Some(binding.train_sha256.as_str())
        || locked_files.get("test.csv").and_then(Value::as_str)
            != Some(binding.test_sha256.as_str())
        || locked_files.get("meta.json").and_then(Value::as_str)
            != Some(binding.meta_sha256.as_str())
    {
        bail!("{dataset_hash} comparator input binding drift");
    }
    if sha256_file(Path::new(&binding.train_path))? != binding.train_sha256
        || sha256_file(Path::new(&binding.test_path))? != binding.test_sha256
        || sha256_file(Path::new(&binding.meta_path))? != binding.meta_sha256
    {
        bail!("{dataset_hash} frozen dataset input drift");
    }
    let winner = comparator
        .get("winner")
        .and_then(Value::as_object)
        .context("comparator lacks winner")?;
    let strongest = Incumbent {
        method: winner
            .get("model")
            .and_then(Value::as_str)
            .context("comparator winner lacks model")?
            .to_string(),
        macro_f1: winner
            .get("primary")
            .and_then(Value::as_f64)
            .context("comparator winner lacks primary score")?,
        accuracy: winner
            .get("secondary")
            .and_then(Value::as_f64)
            .context("comparator winner lacks secondary score")?,
        report_path: winner
            .get("report_path")
            .and_then(Value::as_str)
            .context("comparator winner lacks report path")?
            .to_string(),
        report_sha256: required_hash(&comparator, "/winner/report_sha256")?,
    };
    binding.fast_baselines_complete = true;
    binding.completed_incumbent_count = comparator
        .get("accepted_reports")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    binding.comparator_sha256 = embedded;
    binding.strongest_available_incumbent = Some(strongest);
    Ok(Some(binding))
}

fn build_plan(
    run_id: &str,
    corpus_path: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<Plan> {
    validate_run_id(run_id)?;
    if run_id != RUN_ID {
        bail!("multiclass campaign run-id is frozen as {RUN_ID}");
    }
    let gate = coverage_gate(comparator_root)?;
    let corpus = validate_source_manifest(corpus_path, CORPUS_SCHEMA)?;
    let corpus_sha = required_hash(&corpus, "/manifest_sha256")?;
    if corpus_path != Path::new(MC_COVERAGE_MANIFEST) || corpus_sha != MC_COVERAGE_MANIFEST_SHA256 {
        bail!("multiclass campaign must use the pinned v5 coverage manifest");
    }
    let rows = corpus
        .get("datasets")
        .and_then(Value::as_array)
        .context("corpus manifest lacks datasets")?;
    let execution_order = rows
        .iter()
        .map(|row| {
            row.get("dataset_hash")
                .and_then(Value::as_str)
                .map(str::to_string)
                .context("corpus dataset lacks dataset_hash")
        })
        .collect::<Result<Vec<_>>>()?;
    if execution_order.len() != 272 || execution_order.iter().collect::<BTreeSet<_>>().len() != 272
    {
        bail!("multiclass corpus must contain exactly 272 unique dataset hashes");
    }
    let entries = dataset_entries(&corpus)?;
    let seeds = seed_population(seed_catalog)?;
    let mut datasets = Vec::new();
    let mut comparator_hashes = Vec::new();
    for (order, dataset_hash) in execution_order.iter().enumerate() {
        let entry = entries
            .get(dataset_hash)
            .with_context(|| format!("corpus manifest lacks {dataset_hash}"))?;
        let root = Path::new(DATA_ROOT).join(dataset_hash);
        let train_path = root.join("train.csv");
        let test_path = root.join("test.csv");
        let meta_path = root.join("meta.json");
        let train_sha = required_hash(entry, "/files/train.csv/sha256")?;
        let test_sha = required_hash(entry, "/files/test.csv/sha256")?;
        let meta_sha = required_hash(entry, "/files/meta.json/sha256")?;
        if sha256_file(&train_path)? != train_sha
            || sha256_file(&test_path)? != test_sha
            || sha256_file(&meta_path)? != meta_sha
        {
            bail!("{dataset_hash} corpus input digest drift");
        }
        let class_count = entry.get("n_classes").and_then(Value::as_u64).unwrap_or(0) as usize;
        if class_count < 3 {
            bail!("{dataset_hash} has fewer than three classes");
        }
        let n_features = entry
            .get("n_features")
            .and_then(Value::as_u64)
            .context("corpus dataset lacks n_features")? as usize;
        let base = DatasetBinding {
            order,
            wave: "all-272".to_string(),
            dataset_hash: dataset_hash.clone(),
            train_path: train_path.display().to_string(),
            train_sha256: train_sha,
            test_path: test_path.display().to_string(),
            test_sha256: test_sha,
            meta_path: meta_path.display().to_string(),
            meta_sha256: meta_sha,
            class_count,
            n_features,
            fast_baselines_complete: false,
            completed_incumbent_count: 0,
            comparator_sha256: String::new(),
            strongest_available_incumbent: None,
        };
        let binding = bind_frozen_comparator(base, comparator_root, &gate)?
            .with_context(|| format!("missing post-coverage comparator for {dataset_hash}"))?;
        comparator_hashes.push((dataset_hash.clone(), binding.comparator_sha256.clone()));
        datasets.push(binding);
    }
    let seed_population_sha256 = sha256_bytes(&serde_json::to_vec(&seeds)?);
    if comparator_hashes.len() != 272 {
        bail!("multiclass comparator binding set is incomplete");
    }
    let comparator_set_sha256 = gate.comparator_set_sha256.clone();
    Ok(Plan {
        schema: PLAN_SCHEMA.to_string(),
        run_id: run_id.to_string(),
        test_adaptive: true,
        selection_bias: "test-selected".to_string(),
        benchmark_valid: false,
        official_combined_rank_eligible: false,
        project: PROJECT.to_string(),
        // New plans must match the live multiclass Priority-10 policy.  The
        // immutable-plan loader still reads older plans as-is; this only
        // changes the value emitted when a new plan is generated.
        project_weight: 10,
        project_max_gpus: 0,
        scq_binary: SCQ.to_string(),
        scq_binary_sha256: sha256_file(Path::new(SCQ))?,
        release_binary_sha256: current_binary_sha256()?,
        corpus_eval: CORPUS_EVAL.to_string(),
        corpus_eval_sha256: sha256_file(Path::new(CORPUS_EVAL))?,
        corpus_manifest: corpus_path.display().to_string(),
        corpus_manifest_sha256: corpus_sha,
        multiclass_coverage_final_receipt_sha256: gate.multiclass_receipt_sha256,
        timeseries_coverage_manifest_sha256: TS_COVERAGE_MANIFEST_SHA256.to_string(),
        timeseries_coverage_final_receipt_sha256: gate.timeseries_receipt_sha256,
        comparator_root: comparator_root.display().to_string(),
        comparator_set_path: gate.comparator_set_path,
        comparator_set_sha256,
        seed_catalog: format!(
            "{} + {}",
            seed_catalog.display(),
            seed_catalog
                .with_file_name("generation-zero-binary.txt")
                .display()
        ),
        seed_catalog_sha256: seed_sources_sha256(seed_catalog)?,
        seed_population_sha256,
        priority_count: PRIORITY_DATASETS,
        execution_order,
        fast_baselines: FAST_BASELINES
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        config: Config::fixed(),
        seeds,
        datasets,
    })
}

fn run_root(campaign_root: &Path, run_id: &str) -> PathBuf {
    campaign_root.join(run_id)
}

pub(crate) fn plan(
    run_id: &str,
    campaign_root: &Path,
    corpus_manifest: &Path,
    comparator_root: &Path,
    seed_catalog: &Path,
) -> Result<()> {
    let root = run_root(campaign_root, run_id);
    let path = root.join("plan.v1.json");
    let plan = if path.exists() {
        let existing = read_plan(campaign_root, run_id)?;
        if existing.corpus_manifest != corpus_manifest.display().to_string()
            || existing.comparator_root != comparator_root.display().to_string()
            || existing.seed_catalog_sha256 != seed_sources_sha256(seed_catalog)?
            || existing.scq_binary != SCQ
        {
            bail!(
                "immutable multiclass plan input drift at {}",
                path.display()
            );
        }
        existing
    } else {
        let plan = build_plan(run_id, corpus_manifest, comparator_root, seed_catalog)?;
        let bytes = canonical_json(&plan)?;
        immutable_write(&path, &bytes)?;
        plan
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "plan": path,
            "datasets": plan.datasets.len(),
            "priority": PRIORITY_DATASETS,
            "tail": plan.datasets.len() - PRIORITY_DATASETS,
            "fast_baseline_released": plan.datasets.iter().filter(|row| row.fast_baselines_complete).count(),
            "blocked_on_fast_baselines": plan.datasets.iter().filter(|row| !row.fast_baselines_complete).count(),
        }))?
    );
    Ok(())
}

fn read_plan(campaign_root: &Path, run_id: &str) -> Result<Plan> {
    let path = run_root(campaign_root, run_id).join("plan.v1.json");
    let plan: Plan = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("read {}", path.display()))?;
    let gate = coverage_gate(Path::new(&plan.comparator_root))?;
    if plan.schema != PLAN_SCHEMA
        || plan.run_id != RUN_ID
        || plan.test_adaptive != true
        || plan.selection_bias != "test-selected"
        || plan.benchmark_valid != false
        || plan.official_combined_rank_eligible != false
        || plan.config != Config::fixed()
        || plan.datasets.len() != 272
        || plan.seeds.len() != POPULATION
        || plan.corpus_eval != CORPUS_EVAL
        || plan.release_binary_sha256 != current_binary_sha256()?
        || plan.scq_binary != SCQ
        || sha256_file(Path::new(&plan.scq_binary))? != plan.scq_binary_sha256
        || plan.corpus_manifest != MC_COVERAGE_MANIFEST
        || plan.corpus_manifest_sha256 != MC_COVERAGE_MANIFEST_SHA256
        || plan.multiclass_coverage_final_receipt_sha256 != gate.multiclass_receipt_sha256
        || plan.timeseries_coverage_manifest_sha256 != TS_COVERAGE_MANIFEST_SHA256
        || plan.timeseries_coverage_final_receipt_sha256 != gate.timeseries_receipt_sha256
        || plan.comparator_set_path != gate.comparator_set_path
        || plan.comparator_set_sha256 != gate.comparator_set_sha256
        || plan.seed_catalog_sha256
            != seed_sources_sha256(Path::new(
                &plan.seed_catalog.split(" + ").next().unwrap_or(""),
            ))?
        || plan.seed_population_sha256 != sha256_bytes(&serde_json::to_vec(&plan.seeds)?)
        || sha256_file(Path::new(&plan.corpus_eval))? != plan.corpus_eval_sha256
    {
        bail!("multiclass plan identity or policy drift");
    }
    Ok(plan)
}

fn ready_dataset_bindings(plan: &Plan, root: &Path) -> Result<Vec<DatasetBinding>> {
    let comparator_root = Path::new(&plan.comparator_root);
    let gate = coverage_gate(comparator_root)?;
    let mut ready = Vec::new();
    for planned in &plan.datasets {
        let Some(binding) = bind_frozen_comparator(planned.clone(), comparator_root, &gate)? else {
            if planned.fast_baselines_complete {
                bail!(
                    "{} planned comparator lock disappeared",
                    planned.dataset_hash
                );
            }
            continue;
        };
        if binding != *planned {
            bail!("{} planned comparator drift", planned.dataset_hash);
        }
        let path = root
            .join("datasets")
            .join(&binding.dataset_hash)
            .join("comparator-binding.v1.json");
        let bytes = canonical_json(&binding)?;
        if path.is_file() {
            if fs::read(&path)? != bytes {
                bail!(
                    "immutable multiclass comparator binding collision at {}",
                    path.display()
                );
            }
        } else {
            immutable_write(&path, &bytes)?;
        }
        ready.push(binding);
    }
    Ok(ready)
}

fn ledger_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("champion-ledger.v1.json")
}

fn write_multiclass_terminal_report(
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    ledger: &Value,
) -> Result<()> {
    let semantic = ledger.get("semantic_sha256").and_then(Value::as_str);
    let slot = semantic.and_then(|hash| {
        checkpoint
            .slots
            .iter()
            .find(|row| row.member.semantic_sha256 == hash)
    });
    let report = json!({
        "schema_version": "multiclass-test-adaptive-report-v2",
        "algorithm": "jope-dime-test-adaptive-union43-gen20-v2",
        "model_name": "jope-dime-regressor",
        "task_type": "multiclass",
        "status": "completed",
        "terminal_status": ledger.get("status"),
        "dataset_hash": dataset.dataset_hash,
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "ranking_eligible": false,
        "metrics": {
            "macro_f1": ledger.get("macro_f1"),
            "accuracy": ledger.get("accuracy"),
            "comparator_margin": ledger.get("score_margin"),
        },
        "comparator": {
            "lock_sha256": dataset.comparator_sha256,
            "accepted_report_count": dataset.completed_incumbent_count,
            "winner_model": ledger.get("external_model"),
            "winner_report_sha256": ledger.get("external_report_sha256"),
        },
        "genome": ledger.get("canonical_genome"),
        "semantic_sha256": semantic,
        "generation": checkpoint.generation,
        "slot": slot.map(|row| row.slot),
        "timing": {
            "generation_completed_at_unix_ms": checkpoint.completed_at_unix_ms,
            "terminal_at_unix_ms": ledger.get("terminal_at_unix_ms"),
        },
        "prediction_sha256": slot.and_then(|row| row.predictions_sha256.as_deref()),
        "probability_sha256": slot.and_then(|row| row.probabilities_sha256.as_deref()),
        "receipt_sha256": slot.map(|row| row.receipt_sha256.as_str()),
        "checkpoint_sha256": ledger.get("checkpoint_sha256"),
        "confirmation": {
            "required": ledger.get("status").and_then(Value::as_str) != Some("exhausted"),
            "replays": ledger.get("replays"),
        },
    });
    let path = Path::new(DATA_ROOT)
        .join(&dataset.dataset_hash)
        .join("reports")
        .join(JOPE_MULTICLASS_REPORT_FILE);
    immutable_write(&path, &canonical_json(&report)?)
}

fn dataset_terminal(root: &Path, dataset_hash: &str) -> bool {
    let Ok(value) = read_value(&ledger_path(root, dataset_hash)) else {
        return false;
    };
    matches!(
        value.get("status").and_then(Value::as_str),
        Some("strict_winner" | "ceiling_matched" | "exhausted")
    )
}

fn dataset_started_at(root: &Path, dataset_hash: &str) -> Option<u128> {
    fs::metadata(
        root.join("datasets")
            .join(dataset_hash)
            .join("generation-000")
            .join("shard-0.manifest.json"),
    )
    .ok()?
    .modified()
    .ok()?
    .duration_since(std::time::UNIX_EPOCH)
    .ok()
    .map(|duration| duration.as_millis())
}

fn next_generation_index(root: &Path, dataset_hash: &str) -> usize {
    let directory = root.join("datasets").join(dataset_hash).join("checkpoints");
    fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| {
            name.strip_prefix("generation-")?
                .strip_suffix(".v1.json")?
                .parse::<usize>()
                .ok()
        })
        .max()
        .map_or(0, |generation| generation + 1)
}

fn initial_budget_complete(root: &Path, dataset_hash: &str) -> bool {
    checkpoint_path(root, dataset_hash, MAX_GENERATIONS - 1).is_file()
}

fn active_dataset_bindings<'a>(
    ready: &'a [DatasetBinding],
    root: &Path,
) -> Vec<&'a DatasetBinding> {
    let unresolved = ready
        .iter()
        .filter(|dataset| !dataset_terminal(root, &dataset.dataset_hash))
        .collect::<Vec<_>>();

    let mut bounded = unresolved;
    bounded.sort_by_key(|dataset| {
        (
            dataset_started_at(root, &dataset.dataset_hash).unwrap_or(u128::MAX),
            dataset.order,
        )
    });
    bounded.truncate(QUEUE_BUFFER_DATASETS);
    bounded
}

fn comparator_protocol_sha256(plan: &Plan, dataset_hash: &str) -> Result<String> {
    let path = Path::new(&plan.comparator_root)
        .join("multiclass")
        .join(dataset_hash)
        .join("comparator.json");
    let comparator = read_value(&path)?;
    required_hash(&comparator, "/protocol_sha256")
}

fn shard_manifest(
    plan: &Plan,
    dataset: &DatasetBinding,
    shard: usize,
    generation: usize,
    population: &[SeedMember],
) -> Result<Value> {
    let start = SHARD_SIZES[..shard].iter().sum::<usize>();
    let end = start + SHARD_SIZES[shard];
    if population.len() != POPULATION {
        bail!("generation population must contain exactly {POPULATION} members");
    }
    let candidates = population[start..end]
        .iter()
        .map(|member| {
            json!({
                "semantic_sha256": member.semantic_sha256,
                "canonical_genome": member.canonical_genome,
            })
        })
        .collect::<Vec<_>>();
    let protocol_sha256 = comparator_protocol_sha256(plan, &dataset.dataset_hash)?;
    Ok(json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "run_id": plan.run_id,
        "release_binary_sha256": plan.release_binary_sha256,
        "multiclass_coverage_final_receipt_sha256": plan.multiclass_coverage_final_receipt_sha256,
        "timeseries_coverage_final_receipt_sha256": plan.timeseries_coverage_final_receipt_sha256,
        "comparator_set_sha256": plan.comparator_set_sha256,
        "comparator_lock_sha256": dataset.comparator_sha256,
        "seed_population_sha256": plan.seed_population_sha256,
        "task": "multiclass",
        "dataset_hash": dataset.dataset_hash,
        "train_csv": dataset.train_path,
        "test_csv": dataset.test_path,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "checkpoint_epoch": format!("checkpoint-{}", plan.seed_population_sha256),
        "checkpoint_sha256": plan.seed_population_sha256,
        "catalog_sha256": plan.seed_catalog_sha256,
        "protocol_sha256": protocol_sha256,
        "generated_at_ms": 0,
        "meta_json": dataset.meta_path,
        "meta_sha256": dataset.meta_sha256,
        "label_authority": {
            "score_tier": "test",
            "metric": "macro_f1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-selected",
            "generation": format!("multiclass-generation-{generation:03}"),
            "frozen_seed_sha256": plan.seed_population_sha256,
            "model_checkpoint_sha256": plan.seed_population_sha256,
            "hall_of_fame_sha256": plan.seed_population_sha256,
        },
        "candidates": candidates,
    }))
}

fn render_spec(
    root: &Path,
    dataset: &DatasetBinding,
    shard: usize,
    generation: usize,
    attempt: usize,
    manifest: &Path,
    manifest_sha256: &str,
    evaluator_sha256: &str,
) -> String {
    let id = shard_job_id(
        &dataset.dataset_hash,
        generation,
        shard,
        attempt,
        manifest_sha256,
    );
    let timeout = if dataset.class_count == 100 || dataset.n_features >= 1_000 {
        300
    } else {
        180
    };
    let args = [
        "--manifest".to_string(),
        "{input:manifest}".to_string(),
        "--output-dir".to_string(),
        "{output:candidate-receipts}".to_string(),
        "--threads".to_string(),
        SHARD_SIZES[shard].to_string(),
        "--timeout-seconds".to_string(),
        timeout.to_string(),
    ];
    let args = args
        .iter()
        .map(|value| serde_json::to_string(value).expect("TOML JSON quoting"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [{evaluator:?}, {args}]\nworkdir = {workdir:?}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus}\nomp_threads = 1\nram_gb = {ram_gb}\ngang_size = 1\nplacement = \"single_node\"\n\n[env]\nOMP_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\n\n[preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n\n[[input]]\nname = \"evaluator\"\nsource = {evaluator_source:?}\ntarget = \"corpus_eval\"\nsha256 = {evaluator_sha256:?}\n\n[[input]]\nname = \"manifest\"\nsource = {manifest:?}\ntarget = \"manifest.json\"\nsha256 = {manifest_sha256:?}\n\n[[output]]\nname = \"candidate-receipts\"\nsource = \"candidate-receipts\"\ndirectory = true\n",
        workdir = root.display().to_string(),
        cpus = SHARD_SIZES[shard],
        ram_gb = if attempt >= 2 { 64 } else { 8 },
        evaluator = "{input:evaluator}",
        evaluator_source = CORPUS_EVAL,
        manifest = manifest.display().to_string(),
    )
}

fn scq_rows() -> Result<BTreeMap<String, Value>> {
    let output = Command::new(SCQ).arg("ls").arg("--json").output()?;
    if !output.status.success() {
        bail!(
            "scq ls failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    let rows = value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ job listing has no jobs array")?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("id")?.as_str()?.to_string(), row.clone())))
        .collect())
}

fn spec_job_id(path: &Path) -> Result<String> {
    let source = fs::read_to_string(path)?;
    let encoded = source
        .lines()
        .find_map(|line| line.strip_prefix("id = "))
        .with_context(|| format!("SCQ spec lacks job id: {}", path.display()))?;
    serde_json::from_str(encoded)
        .with_context(|| format!("SCQ spec has invalid job id: {}", path.display()))
}

fn shard_job_id(
    dataset_hash: &str,
    generation: usize,
    shard: usize,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    format!(
        "jope-mc272-{dataset_hash}-g{generation:03}-s{shard}-a{attempt}-m{}-r{SPEC_REVISION}",
        &manifest_sha256[..12]
    )
}

fn terminal_infrastructure_state(state: &str) -> bool {
    matches!(
        state,
        "failed" | "cancelled" | "canceled" | "lost" | "preempted" | "timed_out"
    )
}

fn scheduler_preemption(row: &Value) -> bool {
    row.get("state").and_then(Value::as_str) == Some("failed")
        && row.get("failure_class").and_then(Value::as_str) == Some("crash")
        && row.get("error_reason").and_then(Value::as_str) == Some("killed (no exit status)")
}

fn find_named_file(root: &Path, file_name: &str) -> Result<Option<PathBuf>> {
    if !root.is_dir() {
        return Ok(None);
    }
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named_file(&path, file_name)? {
                return Ok(Some(found));
            }
        } else if path.file_name().and_then(|value| value.to_str()) == Some(file_name) {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn fetch_shard_receipt(
    generation_root: &Path,
    dataset_hash: &str,
    generation: usize,
    shard: usize,
    attempt: usize,
    job_id: &str,
) -> Result<PathBuf> {
    let destination = generation_root
        .join("receipts")
        .join(format!("shard-{shard}-attempt-{attempt}.receipts.jsonl"));
    if destination.is_file() {
        return Ok(destination);
    }
    let temporary = std::env::temp_dir().join(format!(
        "jope-mc272-fetch-{}-{dataset_hash}-g{generation:03}-s{shard}-a{attempt}",
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
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let file_name = format!("{dataset_hash}.receipts.jsonl");
    let source = find_named_file(&temporary, &file_name)?
        .with_context(|| format!("SCQ output {job_id} lacks {file_name}"))?;
    let bytes = fs::read(&source)?;
    immutable_write(&destination, &bytes)?;
    fs::remove_dir_all(&temporary)?;
    Ok(destination)
}

fn read_receipt_values(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse multiclass candidate receipt"))
        .collect()
}

fn genome_complexity(source: &str) -> Result<usize> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member must be a pipeline");
    };
    Ok(pipeline
        .stages
        .iter()
        .map(|stage| 1 + stage.params.len())
        .sum())
}

fn slots_from_receipts(
    dataset: &DatasetBinding,
    generation: usize,
    population: &[SeedMember],
    manifests: &[PathBuf],
    receipts: &[PathBuf],
    expected_protocol_sha256: &str,
) -> Result<Vec<LogicalSlot>> {
    let mut slots = Vec::with_capacity(POPULATION);
    for shard in 0..SHARD_SIZES.len() {
        let values = read_receipt_values(&receipts[shard])?;
        if values.len() != SHARD_SIZES[shard] {
            bail!(
                "{} generation {generation} shard {shard} emitted {} receipts, expected {}",
                dataset.dataset_hash,
                values.len(),
                SHARD_SIZES[shard]
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
                    != Some(expected_protocol_sha256)
                || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
                || value.get("selection_bias").and_then(Value::as_str) != Some("test-selected")
                || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
                || value.get("generation").and_then(Value::as_str)
                    != Some(format!("multiclass-generation-{generation:03}").as_str())
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
                    .filter(|score| score.is_finite())
                    .context("valid multiclass receipt lacks finite macro-F1")?;
                let accuracy = value
                    .get("secondary_score")
                    .and_then(Value::as_f64)
                    .filter(|score| score.is_finite())
                    .context("valid multiclass receipt lacks finite accuracy")?;
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
                    bail!("valid multiclass receipt lacks metric or prediction evidence");
                }
                Some(RankedScore {
                    macro_f1,
                    accuracy,
                    complexity: genome_complexity(&member.canonical_genome)?,
                    semantic_sha256: member.semantic_sha256.clone(),
                })
            } else {
                if value.get("failure").and_then(Value::as_str).is_none() {
                    bail!("invalid multiclass candidate receipt lacks failure disposition");
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
                node: value
                    .get("node")
                    .and_then(Value::as_str)
                    .context("candidate receipt lacks node")?
                    .to_string(),
            });
        }
    }
    if slots.len() != POPULATION {
        bail!("generation produced {} logical slots", slots.len());
    }
    Ok(slots)
}

fn slot_rank_order(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
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

fn evolution_seed(dataset_hash: &str, generation: usize, slot: usize, retry: usize) -> u64 {
    let digest =
        Sha256::digest(format!("{RUN_ID}:{dataset_hash}:{generation}:{slot}:{retry}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("eight digest bytes"))
}

fn regression_surrogate(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass parent must be a pipeline");
    };
    let head = pipeline
        .stages
        .pop()
        .context("multiclass parent lacks head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass parent has incompatible head {}", head.op);
    }
    pipeline.stages.push(Stage::new("ridge"));
    Ok(GenomeIr::Pipeline(pipeline).canonical())
}

fn next_population(
    dataset: &DatasetBinding,
    ranked: &[LogicalSlot],
    generation: usize,
) -> Result<Vec<SeedMember>> {
    let valid = ranked.iter().filter(|slot| slot.valid).collect::<Vec<_>>();
    // A dataset-wide candidate-invalid generation is a model-logic result, not
    // an infrastructure failure and not a reason to kill every other active
    // dataset. Deterministically mutate the structurally valid failed genomes
    // with no elites so the next generation can escape the failed family.
    if valid.is_empty() {
        // Candidate mutation cannot repair a dataset-level evaluator
        // incompatibility.  Preserve the complete, canonical population and
        // advance deterministically; the corrected evaluator is introduced
        // through an immutable successor campaign rather than by changing the
        // binary pinned by this run.
        return ranked
            .iter()
            .enumerate()
            .map(|(slot, logical)| {
                let mut member = logical.member.clone();
                member.slot = slot;
                member.source = format!(
                    "all-invalid-reseed:generation={generation}:{}",
                    logical.member.semantic_sha256
                );
                Ok(member)
            })
            .collect();
    }
    // Keep valid candidates first for tournament selection, but retain the
    // remaining structurally valid genomes as diversity.  Restricting a
    // sparse generation to one or two successful parents can make the family
    // cap impossible to fill and burns thousands of deterministic retries.
    let parents = ranked.iter().collect::<Vec<_>>();
    if parents.is_empty() {
        bail!("{} generation has no candidate slots", dataset.dataset_hash);
    }
    let parent_pool = parents.len().min(30);
    let elite_count = valid.len().min(7);
    let mut next = Vec::with_capacity(POPULATION);
    let mut seen = BTreeSet::new();
    let mut families = BTreeMap::<String, usize>::new();
    for elite in valid.iter().take(elite_count) {
        let mut member = elite.member.clone();
        member.slot = next.len();
        member.source = format!("elite:{}", elite.member.semantic_sha256);
        *families
            .entry(multiclass_head_family(&member.canonical_genome)?)
            .or_default() += 1;
        seen.insert(member.semantic_sha256.clone());
        next.push(member);
    }
    for slot in elite_count..POPULATION {
        let mut produced = None;
        for retry in 0..2_000 {
            let mut rng = Pcg32::new(
                evolution_seed(&dataset.dataset_hash, generation, slot, retry),
                0x6d63_3237_326e_6578,
            );
            let tournament = |rng: &mut Pcg32| -> usize {
                (0..3)
                    .map(|_| rng.next_u32() as usize % parent_pool)
                    .min()
                    .expect("nonempty tournament")
            };
            let first = tournament(&mut rng);
            let first_source = regression_surrogate(&parents[first].member.canonical_genome)?;
            let (candidate, operation, parents) = if rng.next_f64() < 0.25 {
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
                    vec![
                        parents[first].member.semantic_sha256.clone(),
                        parents[second].member.semantic_sha256.clone(),
                    ],
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
                    vec![parents[first].member.semantic_sha256.clone()],
                )
            };
            let variant = rng.next_u32() as usize;
            let Ok(mut member) = rewrite_seed(&candidate, variant) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_head_family(&member.canonical_genome)?;
            if *families.get(&family).unwrap_or(&0) >= POPULATION / MULTICLASS_HEADS.len() {
                continue;
            }
            *families.entry(family).or_default() += 1;
            member.slot = slot;
            member.source = format!("{operation}:{}:retry={retry}", parents.join("+"));
            produced = Some(member);
            break;
        }
        next.push(produced.with_context(|| {
            format!(
                "could not generate canonical multiclass child for {} generation {generation} slot {slot}",
                dataset.dataset_hash
            )
        })?);
    }
    Ok(next)
}

fn checkpoint_path(root: &Path, dataset_hash: &str, generation: usize) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("checkpoints")
        .join(format!("generation-{generation:03}.v1.json"))
}

fn load_checkpoint(
    root: &Path,
    dataset: &DatasetBinding,
    generation: usize,
) -> Result<GenerationCheckpoint> {
    let path = checkpoint_path(root, &dataset.dataset_hash, generation);
    let checkpoint: GenerationCheckpoint = serde_json::from_slice(&fs::read(&path)?)?;
    if checkpoint.schema != CHECKPOINT_SCHEMA
        || checkpoint.run_id != RUN_ID
        || checkpoint.dataset_hash != dataset.dataset_hash
        || checkpoint.comparator_sha256 != dataset.comparator_sha256
        || checkpoint.generation != generation
        || !checkpoint.test_adaptive
        || checkpoint.selection_bias != "test-selected"
        || checkpoint.benchmark_valid
        || checkpoint.slots.len() != POPULATION
    {
        bail!("invalid multiclass checkpoint {}", path.display());
    }
    Ok(checkpoint)
}

fn replay_manifest(
    plan: &Plan,
    dataset: &DatasetBinding,
    generation: usize,
    champion: &SeedMember,
) -> Result<Value> {
    let protocol_sha256 = comparator_protocol_sha256(plan, &dataset.dataset_hash)?;
    Ok(json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "run_id": plan.run_id,
        "release_binary_sha256": plan.release_binary_sha256,
        "multiclass_coverage_final_receipt_sha256": plan.multiclass_coverage_final_receipt_sha256,
        "timeseries_coverage_final_receipt_sha256": plan.timeseries_coverage_final_receipt_sha256,
        "comparator_set_sha256": plan.comparator_set_sha256,
        "comparator_lock_sha256": dataset.comparator_sha256,
        "seed_population_sha256": plan.seed_population_sha256,
        "task": "multiclass",
        "dataset_hash": dataset.dataset_hash,
        "train_csv": dataset.train_path,
        "test_csv": dataset.test_path,
        "train_sha256": dataset.train_sha256,
        "test_sha256": dataset.test_sha256,
        "checkpoint_epoch": format!("checkpoint-{}", champion.semantic_sha256),
        "checkpoint_sha256": champion.semantic_sha256,
        "catalog_sha256": plan.seed_catalog_sha256,
        "protocol_sha256": protocol_sha256,
        "generated_at_ms": 0,
        "meta_json": dataset.meta_path,
        "meta_sha256": dataset.meta_sha256,
        "label_authority": {
            "score_tier": "test",
            "metric": "macro_f1",
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-selected",
            "generation": format!("multiclass-generation-{generation:03}"),
            "frozen_seed_sha256": plan.seed_population_sha256,
            "model_checkpoint_sha256": champion.semantic_sha256,
            "hall_of_fame_sha256": champion.semantic_sha256,
        },
        "candidates": [{
            "semantic_sha256": champion.semantic_sha256,
            "canonical_genome": champion.canonical_genome,
        }],
    }))
}

fn replay_job_id(
    dataset_hash: &str,
    semantic_sha256: &str,
    node: &str,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    format!(
        "jope-mc272-{dataset_hash}-replay-{}-{node}-a{attempt}-m{}-r{REPLAY_SPEC_REVISION}",
        &semantic_sha256[..12],
        &manifest_sha256[..12]
    )
}

fn replay_role_capability(node: &str) -> Result<&'static str> {
    REPLAY_ROLE_CAPABILITIES
        .iter()
        .find_map(|(candidate, marker)| (*candidate == node).then_some(*marker))
        .with_context(|| format!("no multiclass replay capability role is assigned to {node}"))
}

fn render_replay_spec(
    dataset: &DatasetBinding,
    champion: &SeedMember,
    node: &str,
    attempt: usize,
    manifest: &Path,
    manifest_sha256: &str,
    evaluator_sha256: &str,
) -> Result<String> {
    let id = replay_job_id(
        &dataset.dataset_hash,
        &champion.semantic_sha256,
        node,
        attempt,
        manifest_sha256,
    );
    let timeout = if dataset.class_count == 100 || dataset.n_features >= 1_000 {
        300
    } else {
        180
    };
    let args = [
        "--manifest".to_string(),
        "{input:manifest}".to_string(),
        "--output-dir".to_string(),
        "{output:candidate-receipts}".to_string(),
        "--threads".to_string(),
        "1".to_string(),
        "--timeout-seconds".to_string(),
        timeout.to_string(),
    ]
    .iter()
    .map(|value| serde_json::to_string(value).expect("TOML JSON quoting"))
    .collect::<Vec<_>>()
    .join(", ");
    let role_capability = replay_role_capability(node)?;
    Ok(format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/usr/bin/env\"\nargs = [{evaluator:?}, {args}]\nworkdir = \"/home/ubuntu\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = 1\nomp_threads = 1\nram_gb = {ram_gb}\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"multiclass replay role capability available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-f\", {role_capability:?}]\ntimeout_secs = 5\n\n[[dependencies.smoke]]\nname = \"multiclass evaluator compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[env]\nOMP_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\n\n[preempt]\npreemptible = false\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n\n[[input]]\nname = \"evaluator\"\nsource = {evaluator_source:?}\ntarget = \"corpus_eval\"\nsha256 = {evaluator_sha256:?}\n\n[[input]]\nname = \"manifest\"\nsource = {manifest:?}\ntarget = \"manifest.json\"\nsha256 = {manifest_sha256:?}\n\n[[output]]\nname = \"candidate-receipts\"\nsource = \"candidate-receipts\"\ndirectory = true\n",
        ram_gb = if attempt >= 2 { 64 } else { 24 },
        evaluator = "{input:evaluator}",
        evaluator_source = CORPUS_EVAL,
        manifest = manifest.display().to_string(),
    ))
}

fn validate_replay_receipt(
    path: &Path,
    manifest: &Path,
    dataset: &DatasetBinding,
    champion: &SeedMember,
    generation: usize,
    required_node: &str,
) -> Result<Value> {
    let values = read_receipt_values(path)?;
    if values.len() != 1 {
        bail!("replay must emit exactly one candidate receipt");
    }
    let value = values.into_iter().next().expect("one replay receipt");
    let manifest_value = read_value(manifest)?;
    let protocol_sha256 = required_hash(&manifest_value, "/protocol_sha256")?;
    let score = value.get("score").and_then(Value::as_f64);
    let secondary = value.get("secondary_score").and_then(Value::as_f64);
    if value.get("schema").and_then(Value::as_str) != Some("JopeFlywheelCorpusEvalReceipt.v1")
        || value.get("task").and_then(Value::as_str) != Some("multiclass")
        || value.get("dataset_hash").and_then(Value::as_str) != Some(dataset.dataset_hash.as_str())
        || value.get("semantic_sha256").and_then(Value::as_str)
            != Some(champion.semantic_sha256.as_str())
        || value.get("canonical_genome").and_then(Value::as_str)
            != Some(champion.canonical_genome.as_str())
        || value.get("manifest_sha256").and_then(Value::as_str)
            != Some(sha256_file(manifest)?.as_str())
        || value.get("protocol_sha256").and_then(Value::as_str) != Some(protocol_sha256.as_str())
        || value.get("node").and_then(Value::as_str) != Some(required_node)
        || value.get("generation").and_then(Value::as_str)
            != Some(format!("multiclass-generation-{generation:03}").as_str())
        || value.get("valid").and_then(Value::as_bool) != Some(true)
        || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || !score.is_some_and(f64::is_finite)
        || !secondary.is_some_and(f64::is_finite)
        || value
            .get("predictions_sha256")
            .and_then(Value::as_str)
            .is_none()
        || value
            .get("probabilities_sha256")
            .and_then(Value::as_str)
            .is_none()
    {
        bail!("invalid or drifted replay receipt for {required_node}");
    }
    Ok(value)
}

fn manage_winner_replay(
    plan: &Plan,
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
    rows: &BTreeMap<String, Value>,
    specs: &mut Vec<PathBuf>,
) -> Result<bool> {
    let Some(score) = checkpoint.best.as_ref() else {
        return Ok(false);
    };
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset has no frozen incumbent")?;
    if !qualifies(
        score,
        checkpoint.generation,
        checkpoint.generation == 0,
        score.macro_f1,
        incumbent.macro_f1,
        checkpoint.generation >= MAX_GENERATIONS - 1,
    ) {
        return Ok(false);
    }
    let champion = checkpoint
        .best_member
        .as_ref()
        .context("winning checkpoint lacks champion member")?;
    let replay_root = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("replay")
        .join(&champion.semantic_sha256);
    let mut replay_values = Vec::new();
    let mut replay_paths = Vec::new();
    for (node_index, node) in NODES[..3].iter().enumerate() {
        let manifest = replay_root.join(format!("{node}.manifest.json"));
        let manifest_bytes = canonical_json(&replay_manifest(
            plan,
            dataset,
            checkpoint.generation,
            champion,
        )?)?;
        if manifest.is_file() {
            if fs::read(&manifest)? != manifest_bytes {
                bail!(
                    "immutable replay manifest collision at {}",
                    manifest.display()
                );
            }
        } else {
            immutable_write(&manifest, &manifest_bytes)?;
        }
        let mut terminal_failures = 0usize;
        let mut harvested = None;
        for attempt in 0..=(MAX_SCHEDULER_PREEMPTIONS + 2) {
            let job_id = replay_job_id(
                &dataset.dataset_hash,
                &champion.semantic_sha256,
                node,
                attempt,
                &sha256_bytes(&manifest_bytes),
            );
            let Some(row) = rows.get(&job_id) else {
                let spec_path =
                    replay_root.join(format!("{node}-a{attempt}-r{REPLAY_SPEC_REVISION}.toml"));
                let rendered = render_replay_spec(
                    dataset,
                    champion,
                    node,
                    attempt,
                    &manifest,
                    &sha256_bytes(&manifest_bytes),
                    &plan.corpus_eval_sha256,
                )?;
                if spec_path.is_file() {
                    if fs::read_to_string(&spec_path)? != rendered {
                        bail!(
                            "immutable replay SCQ spec collision at {}",
                            spec_path.display()
                        );
                    }
                } else {
                    immutable_write(&spec_path, rendered.as_bytes())?;
                }
                specs.push(spec_path);
                break;
            };
            let state = row.get("state").and_then(Value::as_str).unwrap_or("");
            if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                harvested = Some(fetch_shard_receipt(
                    &replay_root,
                    &dataset.dataset_hash,
                    checkpoint.generation,
                    node_index,
                    attempt,
                    &job_id,
                )?);
                break;
            }
            if terminal_infrastructure_state(state)
                || (state == "completed" && row.get("exit_code").and_then(Value::as_i64) != Some(0))
            {
                if scheduler_preemption(row) {
                    continue;
                }
                terminal_failures += 1;
                if terminal_failures == 3 {
                    bail!("replay job {job_id} exhausted infrastructure retries");
                }
                continue;
            }
            break;
        }
        if let Some(path) = harvested {
            replay_values.push(validate_replay_receipt(
                &path,
                &manifest,
                dataset,
                champion,
                checkpoint.generation,
                node,
            )?);
            replay_paths.push(path);
        }
    }
    if replay_values.len() != 3 {
        return Ok(true);
    }
    let predictions = replay_values
        .iter()
        .filter_map(|value| value.get("predictions_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let probabilities = replay_values
        .iter()
        .filter_map(|value| value.get("probabilities_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let primary = replay_values
        .iter()
        .filter_map(|value| value.get("score").and_then(Value::as_f64).map(f64::to_bits))
        .collect::<BTreeSet<_>>();
    let secondary = replay_values
        .iter()
        .filter_map(|value| {
            value
                .get("secondary_score")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    if predictions.len() != 1
        || probabilities.len() != 1
        || primary.len() != 1
        || secondary.len() != 1
    {
        bail!(
            "three-node replay is not bit-identical for {}",
            dataset.dataset_hash
        );
    }
    let status = if incumbent.macro_f1 == 1.0 && score.macro_f1 == 1.0 {
        "ceiling_matched"
    } else {
        "strict_winner"
    };
    let replays = replay_values
        .iter()
        .zip(&replay_paths)
        .map(|(value, path)| {
            json!({
                "node": value.get("node").and_then(Value::as_str),
                "prediction_sha256": value.get("predictions_sha256").and_then(Value::as_str),
                "probability_sha256": value.get("probabilities_sha256").and_then(Value::as_str),
                "macro_f1": value.get("score").and_then(Value::as_f64),
                "accuracy": value.get("secondary_score").and_then(Value::as_f64),
                "receipt_path": path,
                "receipt_sha256": sha256_file(path).expect("validated replay receipt"),
            })
        })
        .collect::<Vec<_>>();
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, checkpoint.generation);
    let ledger = json!({
        "schema": LEDGER_SCHEMA,
        "run_id": RUN_ID,
        "dataset_hash": dataset.dataset_hash,
        "status": status,
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "comparator_sha256": dataset.comparator_sha256,
        "external_model": incumbent.method,
        "external_report_path": incumbent.report_path,
        "external_report_sha256": incumbent.report_sha256,
        "external_macro_f1": incumbent.macro_f1,
        "external_accuracy": incumbent.accuracy,
        "generation": checkpoint.generation,
        "completed_generations": checkpoint.generation + 1,
        "semantic_sha256": champion.semantic_sha256,
        "canonical_genome": champion.canonical_genome,
        "gene_lineage": champion.source,
        "macro_f1": score.macro_f1,
        "accuracy": score.accuracy,
        "score_margin": score.macro_f1 - incumbent.macro_f1,
        "checkpoint_path": checkpoint_file,
        "checkpoint_sha256": sha256_file(&checkpoint_file)?,
        "final_incumbent_count": dataset.completed_incumbent_count,
        "replays": replays,
        "sotarena_report_path": format!("tracks/multiclass272-test-adaptive/{}/jope.json", dataset.dataset_hash),
        "started_at_unix_ms": checkpoint.completed_at_unix_ms,
        "terminal_at_unix_ms": unix_millis(),
    });
    immutable_write(
        &ledger_path(root, &dataset.dataset_hash),
        &canonical_json(&ledger)?,
    )?;
    write_multiclass_terminal_report(dataset, checkpoint, &ledger)?;
    Ok(true)
}

fn write_exhausted_ledger(
    root: &Path,
    dataset: &DatasetBinding,
    checkpoint: &GenerationCheckpoint,
) -> Result<()> {
    if checkpoint.generation + 1 != MAX_GENERATIONS || !checkpoint.initial_budget_complete {
        bail!(
            "{} cannot terminate exhausted before generation {}",
            dataset.dataset_hash,
            MAX_GENERATIONS - 1
        );
    }
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset has no frozen incumbent")?;
    if checkpoint.best.as_ref().is_some_and(|score| {
        qualifies(
            score,
            checkpoint.generation,
            false,
            score.macro_f1,
            incumbent.macro_f1,
            true,
        )
    }) {
        bail!(
            "{} has a qualifying candidate and cannot be marked exhausted",
            dataset.dataset_hash
        );
    }
    let best_slot = checkpoint
        .slots
        .iter()
        .filter(|slot| slot.valid)
        .min_by(|left, right| match (&left.score, &right.score) {
            (Some(left), Some(right)) => rank_order(left, right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left.slot.cmp(&right.slot),
        });
    let score = best_slot.and_then(|slot| slot.score.as_ref());
    let member = best_slot.map(|slot| &slot.member);
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, checkpoint.generation);
    let ledger = json!({
        "schema": LEDGER_SCHEMA,
        "run_id": RUN_ID,
        "dataset_hash": dataset.dataset_hash,
        "status": "exhausted",
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "benchmark_valid": false,
        "official_combined_rank_eligible": false,
        "comparator_sha256": dataset.comparator_sha256,
        "external_model": incumbent.method,
        "external_report_path": incumbent.report_path,
        "external_report_sha256": incumbent.report_sha256,
        "external_macro_f1": incumbent.macro_f1,
        "external_accuracy": incumbent.accuracy,
        "generation": checkpoint.generation,
        "completed_generations": MAX_GENERATIONS,
        "semantic_sha256": member.map(|value| value.semantic_sha256.as_str()),
        "canonical_genome": member.map(|value| value.canonical_genome.as_str()),
        "gene_lineage": member.map(|value| value.source.as_str()),
        "macro_f1": score.map(|value| value.macro_f1),
        "accuracy": score.map(|value| value.accuracy),
        "score_margin": score.map(|value| value.macro_f1 - incumbent.macro_f1),
        "prediction_sha256": best_slot.and_then(|slot| slot.predictions_sha256.as_deref()),
        "probability_sha256": best_slot.and_then(|slot| slot.probabilities_sha256.as_deref()),
        "checkpoint_path": checkpoint_file,
        "checkpoint_sha256": sha256_file(&checkpoint_file)?,
        "final_incumbent_count": dataset.completed_incumbent_count,
        "replays": [],
        "sotarena_report_path": format!("tracks/multiclass272-test-adaptive/{}/jope.json", dataset.dataset_hash),
        "started_at_unix_ms": checkpoint.completed_at_unix_ms,
        "terminal_at_unix_ms": unix_millis(),
    });
    immutable_write(
        &ledger_path(root, &dataset.dataset_hash),
        &canonical_json(&ledger)?,
    )?;
    write_multiclass_terminal_report(dataset, checkpoint, &ledger)
}

pub(crate) fn run_all(run_id: &str, campaign_root: &Path, resume: bool) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let ready = ready_dataset_bindings(&plan, &root)?;
    let active = active_dataset_bindings(&ready, &root);
    if active.is_empty() {
        let terminal = plan
            .datasets
            .iter()
            .filter(|dataset| dataset_terminal(&root, &dataset.dataset_hash))
            .count();
        if terminal == plan.datasets.len() {
            println!("all 272 multiclass datasets already have terminal champion ledgers");
        } else {
            println!(
                "no unresolved comparator-ready multiclass datasets (ready={}, terminal={}, waiting={})",
                ready.len(),
                terminal,
                plan.datasets.len() - ready.len()
            );
        }
        return Ok(());
    }
    if !Path::new(CORPUS_EVAL).is_file() {
        bail!("release corpus_eval is missing: {CORPUS_EVAL}");
    }
    let rows = scq_rows()?;
    let mut specs = Vec::<PathBuf>::new();
    let mut harvested_generations = 0usize;
    for dataset in active {
        let generation = (0..)
            .find(|generation| {
                !checkpoint_path(&root, &dataset.dataset_hash, *generation).is_file()
            })
            .expect("unbounded generation iterator");
        if generation > 0 {
            let previous = load_checkpoint(&root, dataset, generation - 1)?;
            if manage_winner_replay(&plan, &root, dataset, &previous, &rows, &mut specs)? {
                continue;
            }
            if generation == MAX_GENERATIONS {
                write_exhausted_ledger(&root, dataset, &previous)?;
                continue;
            }
            if generation > MAX_GENERATIONS {
                bail!(
                    "{} exceeded the maximum generation budget",
                    dataset.dataset_hash
                );
            }
        }
        let population = if generation == 0 {
            plan.seeds.clone()
        } else {
            load_checkpoint(&root, dataset, generation - 1)?.next_population
        };
        let generation_root = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join(format!("generation-{generation:03}"));
        let mut manifest_paths = Vec::new();
        let mut receipt_paths = Vec::new();
        for shard in 0..SHARD_SIZES.len() {
            let manifest_path = generation_root.join(format!("shard-{shard}.manifest.json"));
            let manifest_bytes = canonical_json(&shard_manifest(
                &plan,
                dataset,
                shard,
                generation,
                &population,
            )?)?;
            if manifest_path.exists() {
                if !resume || fs::read(&manifest_path)? != manifest_bytes {
                    bail!(
                        "immutable multiclass shard collision at {}",
                        manifest_path.display()
                    );
                }
            } else {
                immutable_write(&manifest_path, &manifest_bytes)?;
            }
            manifest_paths.push(manifest_path.clone());
            let mut terminal_failures = 0usize;
            let mut harvested = None;
            for attempt in 0..=(MAX_SCHEDULER_PREEMPTIONS + 2) {
                let job_id = shard_job_id(
                    &dataset.dataset_hash,
                    generation,
                    shard,
                    attempt,
                    &sha256_bytes(&manifest_bytes),
                );
                let Some(row) = rows.get(&job_id) else {
                    let spec_path = generation_root
                        .join(format!("shard-{shard}-a{attempt}-r{SPEC_REVISION}.toml"));
                    let rendered = render_spec(
                        Path::new("/home/ubuntu"),
                        dataset,
                        shard,
                        generation,
                        attempt,
                        &manifest_path,
                        &sha256_bytes(&manifest_bytes),
                        &plan.corpus_eval_sha256,
                    );
                    if spec_path.exists() {
                        // r1 specs already staged before the admission
                        // tuning retain their original 24 GiB request.
                        // Accept that exact immutable legacy form while
                        // rendering new r1 specs with the leaner request.
                        let existing = fs::read_to_string(&spec_path)?;
                        let legacy_rendered = rendered.replace("ram_gb = 8\n", "ram_gb = 24\n");
                        if existing != rendered && existing != legacy_rendered {
                            bail!(
                                "immutable multiclass SCQ spec collision at {}",
                                spec_path.display()
                            );
                        }
                    } else {
                        immutable_write(&spec_path, rendered.as_bytes())?;
                    }
                    specs.push(spec_path);
                    break;
                };
                let state = row.get("state").and_then(Value::as_str).unwrap_or("");
                if state == "completed" && row.get("exit_code").and_then(Value::as_i64) == Some(0) {
                    harvested = Some(fetch_shard_receipt(
                        &generation_root,
                        &dataset.dataset_hash,
                        generation,
                        shard,
                        attempt,
                        &job_id,
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
                    terminal_failures += 1;
                    if terminal_failures == 3 {
                        bail!("multiclass shard {job_id} exhausted infrastructure retries");
                    }
                    continue;
                }
                break;
            }
            if let Some(path) = harvested {
                receipt_paths.push(path);
            }
        }
        if receipt_paths.len() == SHARD_SIZES.len() {
            let mut slots = slots_from_receipts(
                dataset,
                generation,
                &population,
                &manifest_paths,
                &receipt_paths,
                &comparator_protocol_sha256(&plan, &dataset.dataset_hash)?,
            )?;
            slots.sort_by(slot_rank_order);
            let best_slot = slots.iter().find(|slot| slot.valid);
            let best = best_slot.and_then(|slot| slot.score.clone());
            let best_member = best_slot.map(|slot| slot.member.clone());
            let incumbent = dataset
                .strongest_available_incumbent
                .as_ref()
                .context("dataset has no frozen incumbent")?;
            let winner = best.as_ref().is_some_and(|score| {
                qualifies(
                    score,
                    generation,
                    generation == 0,
                    score.macro_f1,
                    incumbent.macro_f1,
                    generation >= 19,
                )
            });
            let next = if winner || generation + 1 == MAX_GENERATIONS {
                Vec::new()
            } else {
                next_population(dataset, &slots, generation + 1)?
            };
            let checkpoint = GenerationCheckpoint {
                schema: CHECKPOINT_SCHEMA.to_string(),
                run_id: run_id.to_string(),
                dataset_hash: dataset.dataset_hash.clone(),
                comparator_sha256: dataset.comparator_sha256.clone(),
                generation,
                test_adaptive: true,
                selection_bias: "test-selected".to_string(),
                benchmark_valid: false,
                slots,
                best,
                best_member,
                initial_budget_complete: generation + 1 == MAX_GENERATIONS,
                continuation_lineage_salt: None,
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
    let check_failures = std::sync::Mutex::new(Vec::<String>::new());
    thread::scope(|scope| {
        for spec in &specs {
            let check_failures = &check_failures;
            scope.spawn(move || {
                let failure = match Command::new(SCQ).arg("check").arg(spec).status() {
                    Ok(status) if status.success() => None,
                    Ok(status) => Some(format!(
                        "scq check failed for {} with {status}",
                        spec.display()
                    )),
                    Err(error) => Some(format!(
                        "could not run scq check for {}: {error}",
                        spec.display()
                    )),
                };
                if let Some(failure) = failure {
                    check_failures.lock().unwrap().push(failure);
                }
            });
        }
    });
    let mut check_failures = check_failures.into_inner().unwrap();
    check_failures.sort();
    if !check_failures.is_empty() {
        bail!("{}", check_failures.join("; "));
    }
    if !specs.is_empty() {
        let status = Command::new(SCQ)
            .arg("submit-batch")
            .arg("--json")
            .args(&specs)
            .status()?;
        if !status.success() {
            let reconciled = scq_rows()?;
            let missing = specs
                .iter()
                .map(|spec| spec_job_id(spec))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|job_id| !reconciled.contains_key(job_id))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                bail!(
                    "SCQ multiclass generation submission failed; missing jobs: {}",
                    missing.join(",")
                );
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "submitted_specs": specs.len(),
            "active_datasets": ACTIVE_DATASETS,
            "queue_buffer_datasets": QUEUE_BUFFER_DATASETS,
            "harvested_generations": harvested_generations,
            "resume": resume,
        }))?
    );
    Ok(())
}

fn validate_ledger(root: &Path, value: &Value, dataset: &DatasetBinding) -> Result<u128> {
    if value.get("schema").and_then(Value::as_str) != Some(LEDGER_SCHEMA)
        || value.get("dataset_hash").and_then(Value::as_str) != Some(&dataset.dataset_hash)
        || value.get("test_adaptive").and_then(Value::as_bool) != Some(true)
        || value.get("selection_bias").and_then(Value::as_str) != Some("test-selected")
        || value.get("benchmark_valid").and_then(Value::as_bool) != Some(false)
        || value
            .get("official_combined_rank_eligible")
            .and_then(Value::as_bool)
            != Some(false)
        || value.get("comparator_sha256").and_then(Value::as_str)
            != Some(dataset.comparator_sha256.as_str())
    {
        bail!(
            "{} champion ledger identity or labels drift",
            dataset.dataset_hash
        );
    }
    let status = value.get("status").and_then(Value::as_str);
    if !matches!(
        status,
        Some("strict_winner" | "ceiling_matched" | "exhausted")
    ) {
        bail!("{} is not terminal", dataset.dataset_hash);
    }
    if value.get("final_incumbent_count").and_then(Value::as_u64)
        != Some(dataset.completed_incumbent_count as u64)
    {
        bail!(
            "{} winner was not measured against its frozen comparator report set",
            dataset.dataset_hash
        );
    }
    let generation = value
        .get("generation")
        .and_then(Value::as_u64)
        .context("champion ledger lacks generation")? as usize;
    let completed_generations = value
        .get("completed_generations")
        .and_then(Value::as_u64)
        .context("champion ledger lacks completed_generations")?
        as usize;
    if completed_generations == 0
        || completed_generations > MAX_GENERATIONS
        || completed_generations != generation + 1
        || (status == Some("exhausted") && completed_generations != MAX_GENERATIONS)
    {
        bail!("{} generation budget is invalid", dataset.dataset_hash);
    }
    for initial in 0..completed_generations {
        let checkpoint = load_checkpoint(root, dataset, initial)?;
        if initial == MAX_GENERATIONS - 1 && !checkpoint.initial_budget_complete {
            bail!(
                "{} final checkpoint lacks budget completion",
                dataset.dataset_hash
            );
        }
    }
    let checkpoint_file = checkpoint_path(root, &dataset.dataset_hash, generation);
    if value.get("checkpoint_sha256").and_then(Value::as_str)
        != Some(sha256_file(&checkpoint_file)?.as_str())
    {
        bail!("{} ledger checkpoint hash mismatch", dataset.dataset_hash);
    }
    let replays = value
        .get("replays")
        .and_then(Value::as_array)
        .context("champion ledger lacks replay evidence")?;
    if status == Some("exhausted") {
        if !replays.is_empty() {
            bail!(
                "{} exhausted ledger unexpectedly has replays",
                dataset.dataset_hash
            );
        }
        let incumbent = dataset
            .strongest_available_incumbent
            .as_ref()
            .context("dataset lacks frozen incumbent")?;
        if value
            .get("macro_f1")
            .and_then(Value::as_f64)
            .is_some_and(|score| {
                score > incumbent.macro_f1 || (score == 1.0 && incumbent.macro_f1 == 1.0)
            })
        {
            bail!(
                "{} best-effort ledger contains a qualifying winner",
                dataset.dataset_hash
            );
        }
        return value
            .get("terminal_at_unix_ms")
            .and_then(Value::as_u64)
            .map(u128::from)
            .context("best-effort ledger lacks terminal timestamp");
    }
    let nodes = replays
        .iter()
        .filter_map(|row| row.get("node").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let hashes = replays
        .iter()
        .filter_map(|row| row.get("prediction_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let probability_hashes = replays
        .iter()
        .filter_map(|row| row.get("probability_sha256").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let primary = replays
        .iter()
        .filter_map(|row| {
            row.get("macro_f1")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    let secondary = replays
        .iter()
        .filter_map(|row| {
            row.get("accuracy")
                .and_then(Value::as_f64)
                .map(f64::to_bits)
        })
        .collect::<BTreeSet<_>>();
    if replays.len() != 3
        || nodes.len() != 3
        || hashes.len() != 1
        || probability_hashes.len() != 1
        || primary.len() != 1
        || secondary.len() != 1
    {
        bail!(
            "{} lacks three-node bit-identical replay",
            dataset.dataset_hash
        );
    }
    for replay in replays {
        let path = replay
            .get("receipt_path")
            .and_then(Value::as_str)
            .context("replay lacks receipt path")?;
        if replay.get("receipt_sha256").and_then(Value::as_str)
            != Some(sha256_file(Path::new(path))?.as_str())
        {
            bail!("{} replay receipt hash mismatch", dataset.dataset_hash);
        }
    }
    let incumbent = dataset
        .strongest_available_incumbent
        .as_ref()
        .context("dataset lacks frozen incumbent")?;
    let champion = value
        .get("macro_f1")
        .and_then(Value::as_f64)
        .filter(|score| score.is_finite())
        .context("champion ledger lacks finite macro-F1")?;
    let expected_status = if incumbent.macro_f1 == 1.0 && champion == 1.0 {
        "ceiling_matched"
    } else if champion > incumbent.macro_f1 {
        "strict_winner"
    } else {
        bail!(
            "{} ledger does not beat its comparator",
            dataset.dataset_hash
        );
    };
    if value.get("status").and_then(Value::as_str) != Some(expected_status) {
        bail!("{} champion status is inconsistent", dataset.dataset_hash);
    }
    value
        .get("terminal_at_unix_ms")
        .and_then(Value::as_u64)
        .map(u128::from)
        .context("champion ledger lacks terminal timestamp")
}

pub(crate) fn validate_all(run_id: &str, campaign_root: &Path) -> Result<()> {
    let plan = read_plan(campaign_root, run_id)?;
    let root = run_root(campaign_root, run_id);
    let ready = ready_dataset_bindings(&plan, &root)?;
    if ready.len() != plan.datasets.len() {
        bail!(
            "multiclass validation requires 272 frozen comparators, found {}",
            ready.len()
        );
    }
    let mut terminal_times = Vec::new();
    for dataset in &ready {
        let value = read_value(&ledger_path(&root, &dataset.dataset_hash))?;
        terminal_times.push(validate_ledger(&root, &value, dataset)?);
    }
    let latest_terminal = terminal_times.iter().max().copied();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "terminal_datasets": 272,
            "latest_terminal_at_unix_ms": latest_terminal,
            "all_datasets_included": true,
            "frozen_comparators": 272,
            "three_node_replay": true,
            "success": true,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(f1: f64, accuracy: f64, complexity: usize, hash: &str) -> RankedScore {
        RankedScore {
            macro_f1: f1,
            accuracy,
            complexity,
            semantic_sha256: hash.repeat(64),
        }
    }

    #[test]
    fn frozen_config_matches_campaign_budget_contract() {
        let config = Config::fixed();
        assert_eq!(config.population, 60);
        assert_eq!(config.shard_sizes, [15, 15, 15, 15]);
        assert_eq!(config.elites, 7);
        assert_eq!(config.parent_pool, 30);
        assert_eq!(config.tournament, 3);
        assert_eq!(config.mutation_probability, 0.75);
        assert_eq!(config.crossover_probability, 0.25);
        assert_eq!(config.minimum_generations, 0);
        assert_eq!(config.maximum_generations, 20);
        assert_eq!(config.candidate_timeout_seconds, 180);
        assert_eq!(config.infrastructure_retries, 1);
    }

    #[test]
    fn replay_specs_route_three_roles_by_fail_closed_capability() {
        let dataset = DatasetBinding {
            order: 0,
            wave: "fixture".to_string(),
            dataset_hash: "fixture".to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 12,
            fast_baselines_complete: true,
            completed_incumbent_count: 43,
            comparator_sha256: "4".repeat(64),
            strongest_available_incumbent: None,
        };
        let champion = SeedMember {
            slot: 0,
            canonical_genome: "input | standardize | multiclass_ridge".to_string(),
            semantic_sha256: "5".repeat(64),
            source: "fixture".to_string(),
        };
        for (node, marker) in REPLAY_ROLE_CAPABILITIES {
            let rendered = render_replay_spec(
                &dataset,
                &champion,
                node,
                0,
                Path::new("manifest.json"),
                &"6".repeat(64),
                &"7".repeat(64),
            )
            .unwrap();
            assert!(rendered.contains("[[dependencies.smoke]]"));
            assert!(rendered.contains(marker));
            assert!(!rendered.contains("/etc/hostname"));
            assert!(!rendered.contains("\nnode = "));
            assert!(rendered.contains("-r3\""));
        }
        assert!(replay_role_capability(NODES[3]).is_err());
    }

    #[test]
    fn scheduler_kill_without_exit_status_is_preemption_not_retry_failure() {
        let preempted = json!({
            "state": "failed",
            "failure_class": "crash",
            "error_reason": "killed (no exit status)",
        });
        assert!(scheduler_preemption(&preempted));
        assert!(!scheduler_preemption(&json!({
            "state": "failed",
            "failure_class": "crash",
            "error_reason": "segmentation fault",
        })));
    }

    #[test]
    fn scq_specs_use_integer_runtime_limits_and_revisioned_ids() {
        let dataset = DatasetBinding {
            order: 0,
            wave: "fixture".to_string(),
            dataset_hash: "fixture".to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 12,
            fast_baselines_complete: true,
            completed_incumbent_count: 41,
            comparator_sha256: "4".repeat(64),
            strongest_available_incumbent: None,
        };
        let rendered = render_spec(
            Path::new("/tmp"),
            &dataset,
            0,
            0,
            0,
            Path::new("manifest.json"),
            &"5".repeat(64),
            &"6".repeat(64),
        );
        assert!(rendered.contains("max_runtime_hours = 1"));
        assert!(!rendered.contains("max_runtime_hours = 0.25"));
        assert_eq!(
            shard_job_id("fixture", 0, 0, 0, &"5".repeat(64)),
            "jope-mc272-fixture-g000-s0-a0-m555555555555-r2"
        );
        assert_ne!(
            shard_job_id("fixture", 0, 0, 0, &"5".repeat(64)),
            shard_job_id("fixture", 0, 0, 0, &"7".repeat(64)),
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("job.toml");
        fs::write(&path, rendered).unwrap();
        assert_eq!(
            spec_job_id(&path).unwrap(),
            "jope-mc272-fixture-g000-s0-a0-m555555555555-r2"
        );
    }

    #[test]
    fn missing_comparator_keeps_dataset_waiting_without_blocking_plan() {
        let directory = tempfile::tempdir().unwrap();
        let dataset = DatasetBinding {
            order: 0,
            wave: "all-272".to_string(),
            dataset_hash: "fixture".to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 12,
            fast_baselines_complete: false,
            completed_incumbent_count: 0,
            comparator_sha256: String::new(),
            strongest_available_incumbent: None,
        };
        let gate = CoverageGate {
            multiclass_receipt_sha256: "a".repeat(64),
            timeseries_receipt_sha256: "b".repeat(64),
            comparator_set_path: "fixture".to_string(),
            comparator_set_sha256: "c".repeat(64),
        };
        assert!(bind_frozen_comparator(dataset, directory.path(), &gate)
            .unwrap()
            .is_none());
    }

    #[test]
    fn started_datasets_keep_admission_priority_over_new_comparators() {
        let directory = tempfile::tempdir().unwrap();
        let binding = |order, dataset_hash: &str| DatasetBinding {
            order,
            wave: "fixture".to_string(),
            dataset_hash: dataset_hash.to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 12,
            fast_baselines_complete: true,
            completed_incumbent_count: 41,
            comparator_sha256: "4".repeat(64),
            strongest_available_incumbent: None,
        };
        let generation = directory.path().join("datasets/started/generation-000");
        fs::create_dir_all(&generation).unwrap();
        fs::write(generation.join("shard-0.manifest.json"), b"{}").unwrap();
        let ready = vec![binding(0, "waiting"), binding(1, "started")];
        let active = active_dataset_bindings(&ready, directory.path());
        assert_eq!(active[0].dataset_hash, "started");
        assert_eq!(active[1].dataset_hash, "waiting");
    }

    #[test]
    fn scheduler_never_invents_post_generation_19_continuations() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoints = directory.path().join("datasets/dataset/checkpoints");
        fs::create_dir_all(&checkpoints).unwrap();
        for generation in 0..MAX_GENERATIONS {
            fs::write(
                checkpoints.join(format!("generation-{generation:03}.v1.json")),
                b"{}",
            )
            .unwrap();
        }
        assert_eq!(
            next_generation_index(directory.path(), "dataset"),
            MAX_GENERATIONS
        );
        assert!(initial_budget_complete(directory.path(), "dataset"));
    }

    #[test]
    fn ranking_uses_macro_f1_accuracy_complexity_then_hash() {
        let mut values = vec![
            score(0.8, 0.9, 4, "c"),
            score(0.8, 0.9, 3, "b"),
            score(0.8, 0.8, 1, "a"),
            score(0.9, 0.1, 99, "d"),
        ];
        values.sort_by(rank_order);
        assert_eq!(values[0].macro_f1, 0.9);
        assert_eq!(values[1].complexity, 3);
        assert_eq!(values[2].complexity, 4);
    }

    #[test]
    fn strict_win_and_ceiling_match_can_terminate_at_generation_zero() {
        let winner = score(0.91, 0.8, 3, "a");
        assert!(qualifies(&winner, 0, false, 0.8, 0.9, true));
        assert!(qualifies(&winner, 8, true, 0.8, 0.9, true));
        assert!(qualifies(&winner, 9, false, 0.8, 0.9, true));
        let saturated = score(1.0, 1.0, 3, "b");
        assert!(qualifies(&saturated, 0, true, 1.0, 1.0, false));
        assert!(qualifies(&saturated, 9, false, 0.9, 1.0, true));
        let tied = score(0.9, 1.0, 1, "c");
        assert!(!qualifies(&tied, 9, false, 0.8, 0.9, true));
    }

    #[test]
    fn generation_zero_rewrite_is_native_multiclass_and_deterministic() {
        let source = "(pipe X (standardize) (ridge lambda_frac=0.02))";
        let first = rewrite_seed(source, 0).unwrap();
        let second = rewrite_seed(source, 0).unwrap();
        assert_eq!(first, second);
        assert!(first.canonical_genome.contains("multiclass_ridge"));
        assert!(!first.canonical_genome.contains("(ridge "));
        assert!(rewrite_seed("(pipe X (target kind=log1p) (ridge lambda_frac=0.02))", 0).is_err());
        let binary_source = rewrite_seed("(pipe X (standardize) (logistic))", 0).unwrap();
        assert!(binary_source.canonical_genome.contains("multiclass_ridge"));
        assert!(!binary_source.canonical_genome.contains("logistic"));
    }

    #[test]
    fn checked_in_seed_file_rewrites_to_exact_unique_population() {
        let members = seed_population(Path::new("seeds/generation-zero.txt")).unwrap();
        assert_eq!(members.len(), 60);
        assert_eq!(
            members
                .iter()
                .map(|member| &member.semantic_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            60
        );
        let mut families = BTreeMap::new();
        for member in &members {
            assert!(!member.canonical_genome.contains("(target "));
            *families
                .entry(multiclass_head_family(&member.canonical_genome).unwrap())
                .or_insert(0usize) += 1;
        }
        assert_eq!(families.len(), 4);
        assert!(families.values().all(|count| *count == 15));
    }

    #[test]
    fn mutation_and_crossover_keep_one_native_multiclass_head_and_family_cap() {
        let seeds = seed_population(Path::new("seeds/generation-zero.txt")).unwrap();
        let mut ranked = seeds
            .iter()
            .enumerate()
            .map(|(slot, member)| LogicalSlot {
                slot,
                member: member.clone(),
                score: Some(RankedScore {
                    macro_f1: 1.0 - slot as f64 / 1_000.0,
                    accuracy: 1.0 - slot as f64 / 2_000.0,
                    complexity: genome_complexity(&member.canonical_genome).unwrap(),
                    semantic_sha256: member.semantic_sha256.clone(),
                }),
                valid: true,
                failure: None,
                receipt_sha256: "a".repeat(64),
                predictions_sha256: Some("b".repeat(64)),
                probabilities_sha256: Some("c".repeat(64)),
                node: "xbabe1".to_string(),
            })
            .collect::<Vec<_>>();
        ranked.sort_by(slot_rank_order);
        let dataset = DatasetBinding {
            order: 0,
            wave: "all-272".to_string(),
            dataset_hash: "fixture".to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 48,
            fast_baselines_complete: true,
            completed_incumbent_count: 41,
            comparator_sha256: "4".repeat(64),
            strongest_available_incumbent: None,
        };
        let first = next_population(&dataset, &ranked, 1).unwrap();
        let second = next_population(&dataset, &ranked, 1).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 60);
        let mut families = BTreeMap::new();
        for member in &first {
            let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(&member.canonical_genome).unwrap()
            else {
                panic!("member is not a pipeline")
            };
            assert_eq!(
                pipeline
                    .stages
                    .iter()
                    .filter(|stage| MULTICLASS_HEADS.contains(&stage.op.as_str()))
                    .count(),
                1
            );
            assert!(pipeline.stages.iter().all(|stage| stage.op != "target"));
            *families
                .entry(multiclass_head_family(&member.canonical_genome).unwrap())
                .or_insert(0usize) += 1;
        }
        assert!(families.values().all(|count| *count <= 15));

        let mut sparse = ranked.clone();
        for slot in sparse.iter_mut().skip(21) {
            slot.valid = false;
            slot.score = None;
            slot.failure = Some("candidate-invalid fixture".to_string());
        }
        sparse.sort_by(slot_rank_order);
        let sparse_first = next_population(&dataset, &sparse, 1).unwrap();
        let sparse_second = next_population(&dataset, &sparse, 1).unwrap();
        assert_eq!(sparse_first, sparse_second);
        assert_eq!(sparse_first.len(), 60);

        let mut all_invalid = ranked.clone();
        for slot in &mut all_invalid {
            slot.valid = false;
            slot.score = None;
            slot.failure = Some("candidate-invalid fixture".to_string());
        }
        all_invalid.sort_by(slot_rank_order);
        let recovered_first = next_population(&dataset, &all_invalid, 1).unwrap();
        let recovered_second = next_population(&dataset, &all_invalid, 1).unwrap();
        assert_eq!(recovered_first, recovered_second);
        assert_eq!(recovered_first.len(), 60);
        assert!(recovered_first.iter().all(|member| member
            .source
            .starts_with("all-invalid-reseed:generation=1:")));
    }

    #[test]
    fn scq_spec_content_addresses_native_evaluator_and_manifest() {
        let dataset = DatasetBinding {
            order: 0,
            wave: "all-272".to_string(),
            dataset_hash: "fixture".to_string(),
            train_path: "train.csv".to_string(),
            train_sha256: "1".repeat(64),
            test_path: "test.csv".to_string(),
            test_sha256: "2".repeat(64),
            meta_path: "meta.json".to_string(),
            meta_sha256: "3".repeat(64),
            class_count: 4,
            n_features: 48,
            fast_baselines_complete: true,
            completed_incumbent_count: 41,
            comparator_sha256: "4".repeat(64),
            strongest_available_incumbent: None,
        };
        let manifest_sha256 = "a".repeat(64);
        let evaluator_sha256 = "b".repeat(64);
        let spec = render_spec(
            Path::new("/home/ubuntu"),
            &dataset,
            0,
            0,
            0,
            Path::new("/tmp/manifest.json"),
            &manifest_sha256,
            &evaluator_sha256,
        );
        assert!(spec.contains("command = \"/usr/bin/env\""));
        assert!(spec.contains("{input:evaluator}"));
        assert!(spec.contains("{input:manifest}"));
        assert!(spec.contains(&format!("source = {CORPUS_EVAL:?}")));
        assert!(spec.contains(&format!("sha256 = {evaluator_sha256:?}")));
        assert!(spec.contains(&format!("sha256 = {manifest_sha256:?}")));
        assert!(!spec.to_ascii_lowercase().contains("python"));
    }
}
