use super::*;

pub(crate) fn ledger_path(root: &Path, dataset_hash: &str) -> PathBuf {
    root.join("datasets")
        .join(dataset_hash)
        .join("champion-ledger.v1.json")
}

pub(crate) fn write_multiclass_terminal_report(
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
    let path = Path::new(data_root_str())
        .join(&dataset.dataset_hash)
        .join("reports")
        .join(JOPE_MULTICLASS_REPORT_FILE);
    immutable_write(&path, &canonical_json(&report)?)
}

pub(crate) fn dataset_terminal(root: &Path, dataset_hash: &str) -> bool {
    let Ok(value) = read_value(&ledger_path(root, dataset_hash)) else {
        return false;
    };
    matches!(
        value.get("status").and_then(Value::as_str),
        Some("strict_winner" | "ceiling_matched" | "exhausted")
    )
}

pub(crate) fn dataset_started_at(root: &Path, dataset_hash: &str) -> Option<u128> {
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

#[cfg(test)]
pub(crate) fn next_generation_index(root: &Path, dataset_hash: &str) -> usize {
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

#[cfg(test)]
pub(crate) fn initial_budget_complete(root: &Path, dataset_hash: &str) -> bool {
    checkpoint_path(root, dataset_hash, MAX_GENERATIONS - 1).is_file()
}

pub(crate) fn active_dataset_bindings<'a>(
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

pub(crate) fn comparator_protocol_sha256(plan: &Plan, dataset_hash: &str) -> Result<String> {
    let path = Path::new(&plan.comparator_root)
        .join("multiclass")
        .join(dataset_hash)
        .join("comparator.json");
    let comparator = read_value(&path)?;
    required_hash(&comparator, "/protocol_sha256")
}

pub(crate) fn shard_manifest(
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

// Keep the worker specification inputs explicit without changing its published format.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_spec(
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
        evaluator_source = corpus_eval_str(),
        manifest = manifest.display().to_string(),
    )
}

pub(crate) fn scq_rows() -> Result<BTreeMap<String, Value>> {
    let output = Command::new(scq_str()).arg("ls").arg("--json").output()?;
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

pub(crate) fn spec_job_id(path: &Path) -> Result<String> {
    let source = fs::read_to_string(path)?;
    let encoded = source
        .lines()
        .find_map(|line| line.strip_prefix("id = "))
        .with_context(|| format!("SCQ spec lacks job id: {}", path.display()))?;
    serde_json::from_str(encoded)
        .with_context(|| format!("SCQ spec has invalid job id: {}", path.display()))
}

pub(crate) fn shard_job_id(
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

pub(crate) fn terminal_infrastructure_state(state: &str) -> bool {
    matches!(
        state,
        "failed" | "cancelled" | "canceled" | "lost" | "preempted" | "timed_out"
    )
}

pub(crate) fn scheduler_preemption(row: &Value) -> bool {
    row.get("state").and_then(Value::as_str) == Some("failed")
        && row.get("failure_class").and_then(Value::as_str) == Some("crash")
        && row.get("error_reason").and_then(Value::as_str) == Some("killed (no exit status)")
}

pub(crate) fn find_named_file(root: &Path, file_name: &str) -> Result<Option<PathBuf>> {
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

pub(crate) fn fetch_shard_receipt(
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
    let output = Command::new(scq_str())
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

pub(crate) fn read_receipt_values(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("parse multiclass candidate receipt"))
        .collect()
}
