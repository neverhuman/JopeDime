//! Rust-only promising-candidate replay and fitted-artifact cache.
//!
//! Audit is the default and never executes an evaluator or compiler. `--replay`
//! invokes the precompiled `corpus_eval` ELF over a deterministic 32-dataset
//! family-diverse panel. `--build` requires that replay receipt and lowers at
//! most two qualified candidates per dataset/checkpoint through feat-math's
//! native fitted-C++17 codec. Python and script executors are rejected.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use feat_math::{
    auc, encode_fitted_cpp17, fit_legacy_numeric_bound, fit_table, r2, supports_fitted_cpp17,
    validate_fitted_cpp_codec, ColumnSpec, DataSchema, EncodedTable, FittedCppRuntime, GenomeIr,
    Objective, RawTable, RowSemantics, TaskSpec,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_core::{
    compiled_cache_key, compiled_finalists, exact_parity, family_diverse_panel, CompileCandidate,
    PanelDataset,
};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, canonical_json_bytes, default_root_str, now_ms, read_json, sha256_bytes,
    sha256_file,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

const REQUEST_SCHEMA: &str = "JopeCompileEconomyRequest.v1";
const REPLAY_SCHEMA: &str = "JopeCompileReplayReceipt.v1";
const ARTIFACT_SCHEMA: &str = "JopeCompiledCacheArtifact.v1";
const OPERATION_SCHEMA: &str = "JopeCompiledCacheOperation.v1";
const PANEL_SIZE: usize = 32;
jope_paths::once_str!(
    fn default_evaluator_str => paths::corpus_eval_binary()
);

#[derive(Parser, Debug)]
#[command(
    name = "jope-compile-cache",
    about = "Audit/replay/build the Rust-only fitted candidate cache"
)]
struct Args {
    /// Immutable compile-economy request.
    #[arg(long)]
    request: PathBuf,
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_evaluator_str())]
    evaluator: PathBuf,
    /// Run the precompiled broad evaluator over the selected 32-dataset panel.
    #[arg(long, conflicts_with = "build")]
    replay: bool,
    /// Explicitly compile qualified target-dataset finalists.
    #[arg(long, conflicts_with = "replay")]
    build: bool,
    /// Required by --build; produced by --replay for the same request.
    #[arg(long, requires = "build")]
    replay_receipt: Option<PathBuf>,
    /// Required by --build. Reuse is idempotent.
    #[arg(long, requires = "build")]
    idempotency_key: Option<String>,
    #[arg(long, default_value_t = 1)]
    replay_threads: usize,
    #[arg(long, default_value_t = 300)]
    timeout_seconds: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    task: String,
    checkpoint_sha256: String,
    catalog_sha256: String,
    compiler_bundle_sha256: String,
    target_triple: String,
    frozen_seed_sha256: String,
    hall_of_fame_sha256: String,
    generation: String,
    #[serde(default = "default_panel_seed")]
    panel_seed: u64,
    datasets: Vec<Dataset>,
    candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    dataset_hash: String,
    source_family: String,
    train_csv: PathBuf,
    test_csv: PathBuf,
    train_sha256: String,
    test_sha256: String,
    protocol_sha256: String,
    incumbent_score: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    dataset_hash: String,
    semantic_sha256: String,
    canonical_genome: String,
    score: f64,
    valid: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Qualified {
    dataset_hash: String,
    semantic_sha256: String,
    score: f64,
    rank: usize,
    top_two: bool,
    strict_incumbent_beat: bool,
    incumbent_score: Option<f64>,
}

#[derive(Clone, Debug)]
struct NumericData {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    features: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayReceipt {
    schema: String,
    request_sha256: String,
    task: String,
    checkpoint_sha256: String,
    evaluator: String,
    evaluator_sha256: String,
    panel_size: usize,
    panel_dataset_hashes: Vec<String>,
    semantic_sha256s: Vec<String>,
    receipt_files: Vec<ReplayFile>,
    expected_receipts: usize,
    observed_receipts: usize,
    valid_receipts: usize,
    complete: bool,
    created_at_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayFile {
    dataset_hash: String,
    path: String,
    sha256: String,
    receipts: usize,
    valid: usize,
}

fn default_panel_seed() -> u64 {
    20_260_726
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.replay_threads > 0,
        "--replay-threads must be greater than zero"
    );
    let (request, request_sha256) = load_request(&args.request)?;
    let panel = select_panel(&request)?;
    let qualified = qualified_candidates(&request)?;

    let report = if args.replay {
        run_replay(&args, &request, &request_sha256, &panel, &qualified)?
    } else if args.build {
        run_build(&args, &request, &request_sha256, &panel, &qualified)?
    } else {
        audit_report(&args, &request, &request_sha256, &panel, &qualified)?
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn load_request(path: &Path) -> Result<(Request, String)> {
    let value = read_json(path)?;
    let request_sha256 = sha256_bytes(&canonical_json_bytes(&value)?);
    let request: Request = serde_json::from_value(value)
        .with_context(|| format!("parse compile request {}", path.display()))?;
    ensure!(
        request.schema == REQUEST_SCHEMA,
        "unsupported compile request schema '{}'",
        request.schema
    );
    ensure!(
        matches!(
            request.task.as_str(),
            "regression" | "binary" | "timeseries" | "multiclass"
        ),
        "task must be regression, binary, timeseries, or multiclass"
    );
    for (label, digest) in [
        ("checkpoint_sha256", &request.checkpoint_sha256),
        ("catalog_sha256", &request.catalog_sha256),
        ("compiler_bundle_sha256", &request.compiler_bundle_sha256),
        ("frozen_seed_sha256", &request.frozen_seed_sha256),
        ("hall_of_fame_sha256", &request.hall_of_fame_sha256),
    ] {
        validate_sha256(label, digest)?;
    }
    ensure!(!request.target_triple.is_empty(), "target_triple is empty");
    ensure!(!request.generation.is_empty(), "generation is empty");
    ensure!(
        request.datasets.len() >= PANEL_SIZE,
        "request needs at least {PANEL_SIZE} datasets"
    );

    let mut datasets = BTreeSet::new();
    for dataset in &request.datasets {
        ensure!(
            datasets.insert(dataset.dataset_hash.clone()),
            "duplicate dataset {}",
            dataset.dataset_hash
        );
        ensure!(
            !dataset.dataset_hash.is_empty() && !dataset.source_family.is_empty(),
            "dataset identity/family is empty"
        );
        validate_sha256("dataset train_sha256", &dataset.train_sha256)?;
        validate_sha256("dataset test_sha256", &dataset.test_sha256)?;
        validate_sha256("dataset protocol_sha256", &dataset.protocol_sha256)?;
        if let Some(score) = dataset.incumbent_score {
            ensure!(score.is_finite(), "incumbent score is non-finite");
        }
    }
    let mut identities = BTreeSet::new();
    for candidate in &request.candidates {
        ensure!(
            datasets.contains(&candidate.dataset_hash),
            "candidate references unknown dataset {}",
            candidate.dataset_hash
        );
        validate_sha256("candidate semantic_sha256", &candidate.semantic_sha256)?;
        ensure!(
            sha256_bytes(candidate.canonical_genome.as_bytes()) == candidate.semantic_sha256,
            "candidate semantic hash disagrees with canonical genome"
        );
        ensure!(
            identities.insert((
                candidate.dataset_hash.clone(),
                candidate.semantic_sha256.clone()
            )),
            "duplicate candidate identity"
        );
        if candidate.valid {
            ensure!(
                candidate.score.is_finite(),
                "valid candidate score is non-finite"
            );
        }
    }
    Ok((request, request_sha256))
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} is not a lowercase SHA-256"
    );
    Ok(())
}

fn select_panel(request: &Request) -> Result<Vec<Dataset>> {
    let policy_rows = request
        .datasets
        .iter()
        .map(|row| PanelDataset {
            dataset_hash: row.dataset_hash.clone(),
            source_family: row.source_family.clone(),
        })
        .collect::<Vec<_>>();
    let selected = family_diverse_panel(&policy_rows, PANEL_SIZE, request.panel_seed)
        .map_err(anyhow::Error::msg)?;
    let by_hash = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    selected
        .into_iter()
        .map(|row| {
            by_hash
                .get(row.dataset_hash.as_str())
                .map(|value| (*value).clone())
                .context("selected panel dataset disappeared")
        })
        .collect()
}

fn qualified_candidates(request: &Request) -> Result<Vec<Qualified>> {
    let rows = request
        .candidates
        .iter()
        .map(|row| CompileCandidate {
            task: request.task.clone(),
            dataset_hash: row.dataset_hash.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            score: row.score,
            valid: row.valid,
        })
        .collect::<Vec<_>>();
    let incumbents = request
        .datasets
        .iter()
        .filter_map(|row| {
            row.incumbent_score
                .map(|score| ((request.task.clone(), row.dataset_hash.clone()), score))
        })
        .collect::<BTreeMap<_, _>>();
    let selected = compiled_finalists(&rows, &incumbents);
    let mut rank_by_identity = BTreeMap::new();
    for dataset in &request.datasets {
        let mut ranked = request
            .candidates
            .iter()
            .filter(|row| {
                row.dataset_hash == dataset.dataset_hash && row.valid && row.score.is_finite()
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        for (index, row) in ranked.into_iter().enumerate() {
            rank_by_identity.insert(
                (row.dataset_hash.clone(), row.semantic_sha256.clone()),
                index + 1,
            );
        }
    }
    let datasets = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::with_capacity(selected.len());
    for row in selected {
        let dataset = datasets
            .get(row.dataset_hash.as_str())
            .context("qualified candidate dataset disappeared")?;
        let rank = *rank_by_identity
            .get(&(row.dataset_hash.clone(), row.semantic_sha256.clone()))
            .context("qualified candidate rank disappeared")?;
        let strict_incumbent_beat = dataset
            .incumbent_score
            .is_some_and(|incumbent| row.score > incumbent);
        ensure!(
            rank <= 2 || strict_incumbent_beat,
            "candidate was qualified without top-two or strict-win evidence"
        );
        output.push(Qualified {
            dataset_hash: row.dataset_hash,
            semantic_sha256: row.semantic_sha256,
            score: row.score,
            rank,
            top_two: rank <= 2,
            strict_incumbent_beat,
            incumbent_score: dataset.incumbent_score,
        });
    }
    let mut per_dataset = BTreeMap::<&str, usize>::new();
    for row in &output {
        *per_dataset.entry(&row.dataset_hash).or_default() += 1;
    }
    ensure!(
        per_dataset.values().all(|count| *count <= 2),
        "more than two finalists selected for a dataset/checkpoint"
    );
    Ok(output)
}

fn audit_report(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    let status = cache_status(&cache_root(args))?;
    Ok(json!({
        "schema": "JopeCompileEconomyAudit.v1",
        "mode": "audit",
        "mutated": false,
        "request_sha256": request_sha256,
        "task": request.task,
        "checkpoint_sha256": request.checkpoint_sha256,
        "panel_size": panel.len(),
        "panel_dataset_hashes": panel.iter().map(|row| &row.dataset_hash).collect::<Vec<_>>(),
        "qualified_finalists": qualified,
        "qualified_count": qualified.len(),
        "max_finalists_per_dataset_checkpoint": 2,
        "broad_evaluator": args.evaluator,
        "compile_policy": "explicit_build_only_after_complete_native_replay",
        "fitted_state_scope": "dataset_train_sha256",
        "cache": status,
    }))
}

fn run_replay(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    validate_native_elf(&args.evaluator, "evaluator")?;
    ensure!(!qualified.is_empty(), "no candidates qualify for replay");
    let candidate_map = candidate_map(request);
    let mut semantic_rows = BTreeMap::<String, String>::new();
    for row in qualified {
        let candidate = candidate_map
            .get(&(row.dataset_hash.as_str(), row.semantic_sha256.as_str()))
            .context("qualified candidate body disappeared")?;
        semantic_rows
            .entry(row.semantic_sha256.clone())
            .or_insert_with(|| candidate.canonical_genome.clone());
    }

    let replay_root = cache_root(args).join("replays").join(request_sha256);
    let final_receipt = replay_root.join("replay-receipt.json");
    if final_receipt.is_file() {
        let existing: ReplayReceipt = serde_json::from_value(read_json(&final_receipt)?)?;
        validate_replay_receipt(&existing, request, request_sha256, panel, &semantic_rows)?;
        return Ok(json!({
            "mode": "replay",
            "idempotent": true,
            "receipt": final_receipt,
            "receipt_sha256": sha256_file(&final_receipt)?,
            "complete": existing.complete,
            "observed_receipts": existing.observed_receipts,
            "valid_receipts": existing.valid_receipts,
        }));
    }
    fs::create_dir_all(&replay_root)?;
    let manifest_root = replay_root.join("manifests");
    let output_root = replay_root.join("outputs");
    fs::create_dir_all(&manifest_root)?;
    fs::create_dir_all(&output_root)?;
    let evaluator_sha256 = sha256_file(&args.evaluator)?;
    let candidates = semantic_rows
        .iter()
        .map(|(semantic, genome)| json!({"semantic_sha256": semantic, "canonical_genome": genome}))
        .collect::<Vec<_>>();
    let checkpoint_epoch = format!("checkpoint-{}", request.checkpoint_sha256);
    let generated_at_ms = now_ms();
    let metric = if request.task == "binary" {
        "auc"
    } else {
        "r2"
    };
    let mut receipt_files = Vec::new();
    let mut observed = 0usize;
    let mut valid = 0usize;
    for dataset in panel {
        verify_dataset_files(dataset)?;
        let manifest = json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "task": request.task,
            "dataset_hash": dataset.dataset_hash,
            "train_csv": dataset.train_csv,
            "test_csv": dataset.test_csv,
            "train_sha256": dataset.train_sha256,
            "test_sha256": dataset.test_sha256,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_sha256": request.checkpoint_sha256,
            "catalog_sha256": request.catalog_sha256,
            "protocol_sha256": dataset.protocol_sha256,
            "generated_at_ms": generated_at_ms,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": metric,
                "test_adaptive": true,
                "benchmark_valid": false,
                "selection_bias": "test-exact-reused",
                "generation": request.generation,
                "frozen_seed_sha256": request.frozen_seed_sha256,
                "model_checkpoint_sha256": request.checkpoint_sha256,
                "hall_of_fame_sha256": request.hall_of_fame_sha256,
            },
            "candidates": candidates,
        });
        let manifest_path = manifest_root.join(format!("{}.json", dataset.dataset_hash));
        atomic_json(&manifest_path, &manifest)?;
        let output_dir = output_root.join(&dataset.dataset_hash);
        let output = Command::new(&args.evaluator)
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--threads")
            .arg(args.replay_threads.to_string())
            .arg("--timeout-seconds")
            .arg(args.timeout_seconds.to_string())
            .output()
            .with_context(|| format!("execute native evaluator {}", args.evaluator.display()))?;
        if !output.status.success() {
            bail!(
                "native replay failed for {}: {}",
                dataset.dataset_hash,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let receipt_path = output_dir.join(format!("{}.receipts.jsonl", dataset.dataset_hash));
        let (rows, valid_rows) = validate_replay_jsonl(
            &receipt_path,
            request,
            dataset,
            semantic_rows.keys().map(String::as_str).collect(),
        )?;
        observed += rows;
        valid += valid_rows;
        receipt_files.push(ReplayFile {
            dataset_hash: dataset.dataset_hash.clone(),
            path: receipt_path.display().to_string(),
            sha256: sha256_file(&receipt_path)?,
            receipts: rows,
            valid: valid_rows,
        });
    }
    let expected = panel
        .len()
        .checked_mul(semantic_rows.len())
        .context("replay receipt count overflow")?;
    let receipt = ReplayReceipt {
        schema: REPLAY_SCHEMA.to_string(),
        request_sha256: request_sha256.to_owned(),
        task: request.task.clone(),
        checkpoint_sha256: request.checkpoint_sha256.clone(),
        evaluator: args.evaluator.display().to_string(),
        evaluator_sha256,
        panel_size: panel.len(),
        panel_dataset_hashes: panel.iter().map(|row| row.dataset_hash.clone()).collect(),
        semantic_sha256s: semantic_rows.keys().cloned().collect(),
        receipt_files,
        expected_receipts: expected,
        observed_receipts: observed,
        valid_receipts: valid,
        complete: observed == expected,
        created_at_ms: now_ms(),
    };
    ensure!(receipt.complete, "native replay receipt set is incomplete");
    atomic_json(&final_receipt, &serde_json::to_value(&receipt)?)?;
    Ok(json!({
        "mode": "replay",
        "idempotent": false,
        "receipt": final_receipt,
        "receipt_sha256": sha256_file(&final_receipt)?,
        "complete": receipt.complete,
        "expected_receipts": expected,
        "observed_receipts": observed,
        "valid_receipts": valid,
    }))
}

fn validate_native_elf(path: &Path, label: &str) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("{label} {} is unavailable", path.display()))?;
    ensure!(metadata.is_file(), "{label} is not a regular file");
    ensure!(
        metadata.permissions().mode() & 0o111 != 0,
        "{label} is not executable"
    );
    let mut bytes = [0u8; 4];
    File::open(path)?.read_exact(&mut bytes)?;
    ensure!(
        bytes == *b"\x7fELF",
        "{label} must be a native ELF; Python and scripts are forbidden"
    );
    Ok(())
}

fn validate_replay_jsonl(
    path: &Path,
    request: &Request,
    dataset: &Dataset,
    expected: BTreeSet<&str>,
) -> Result<(usize, usize)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read replay receipts {}", path.display()))?;
    let mut observed = BTreeSet::new();
    let mut valid = 0usize;
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let row: Value = serde_json::from_str(line)
            .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
        ensure!(
            row["schema"] == "JopeFlywheelCorpusEvalReceipt.v1",
            "unexpected replay receipt schema"
        );
        ensure!(row["task"] == request.task, "replay task drift");
        ensure!(
            row["dataset_hash"] == dataset.dataset_hash,
            "replay dataset drift"
        );
        ensure!(
            row["train_sha256"] == dataset.train_sha256
                && row["test_sha256"] == dataset.test_sha256,
            "replay dataset content drift"
        );
        ensure!(
            row["protocol_sha256"] == dataset.protocol_sha256,
            "replay protocol drift"
        );
        ensure!(
            row["model_checkpoint_sha256"] == request.checkpoint_sha256,
            "replay checkpoint drift"
        );
        ensure!(
            row["test_adaptive"] == true && row["benchmark_valid"] == false,
            "replay lost adaptive/benchmark-invalid disclosure"
        );
        let semantic = row["semantic_sha256"]
            .as_str()
            .context("replay semantic is missing")?;
        ensure!(
            expected.contains(semantic),
            "replay contains an unexpected semantic"
        );
        ensure!(
            observed.insert(semantic.to_owned()),
            "duplicate replay semantic receipt"
        );
        valid += usize::from(row["valid"] == true);
    }
    ensure!(
        observed.len() == expected.len(),
        "replay receipt count differs from requested semantics"
    );
    Ok((observed.len(), valid))
}

fn validate_replay_receipt(
    receipt: &ReplayReceipt,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    semantics: &BTreeMap<String, String>,
) -> Result<()> {
    ensure!(
        receipt.schema == REPLAY_SCHEMA,
        "replay receipt schema drift"
    );
    ensure!(
        receipt.request_sha256 == request_sha256
            && receipt.task == request.task
            && receipt.checkpoint_sha256 == request.checkpoint_sha256,
        "replay receipt provenance drift"
    );
    ensure!(
        receipt.complete
            && receipt.panel_size == PANEL_SIZE
            && receipt.panel_dataset_hashes
                == panel
                    .iter()
                    .map(|row| row.dataset_hash.clone())
                    .collect::<Vec<_>>()
            && receipt.semantic_sha256s == semantics.keys().cloned().collect::<Vec<_>>(),
        "replay receipt cohort drift"
    );
    let evaluator = Path::new(&receipt.evaluator);
    validate_native_elf(evaluator, "recorded replay evaluator")?;
    ensure!(
        sha256_file(evaluator)? == receipt.evaluator_sha256,
        "recorded replay evaluator hash drift"
    );
    let datasets = panel
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut observed = 0usize;
    let mut valid = 0usize;
    for row in &receipt.receipt_files {
        let path = Path::new(&row.path);
        ensure!(sha256_file(path)? == row.sha256, "replay file hash drift");
        let dataset = datasets
            .get(row.dataset_hash.as_str())
            .context("replay receipt references a dataset outside the panel")?;
        let (file_rows, file_valid) = validate_replay_jsonl(
            path,
            request,
            dataset,
            semantics.keys().map(String::as_str).collect(),
        )?;
        ensure!(
            file_rows == row.receipts && file_valid == row.valid,
            "replay file aggregate drift"
        );
        observed += file_rows;
        valid += file_valid;
    }
    ensure!(
        observed == receipt.observed_receipts
            && receipt.observed_receipts == receipt.expected_receipts
            && valid == receipt.valid_receipts,
        "replay aggregate counts drift"
    );
    Ok(())
}

fn run_build(
    args: &Args,
    request: &Request,
    request_sha256: &str,
    panel: &[Dataset],
    qualified: &[Qualified],
) -> Result<Value> {
    ensure!(
        !qualified.is_empty(),
        "no candidates qualify for compilation"
    );
    let replay_path = args
        .replay_receipt
        .as_deref()
        .context("--replay-receipt is required with --build")?;
    let replay: ReplayReceipt = serde_json::from_value(read_json(replay_path)?)?;
    let candidate_map = candidate_map(request);
    let semantics = qualified
        .iter()
        .map(|row| {
            let candidate = candidate_map
                .get(&(row.dataset_hash.as_str(), row.semantic_sha256.as_str()))
                .expect("request was validated");
            (
                row.semantic_sha256.clone(),
                candidate.canonical_genome.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    validate_replay_receipt(&replay, request, request_sha256, panel, &semantics)?;
    let replay_receipt_sha256 = sha256_file(replay_path)?;
    let idempotency_key = args
        .idempotency_key
        .as_deref()
        .context("--idempotency-key is required with --build")?;
    ensure!(
        !idempotency_key.trim().is_empty(),
        "idempotency key is empty"
    );

    let datasets = request
        .datasets
        .iter()
        .map(|row| (row.dataset_hash.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let runtime_root = cache_root(args).join("shared-runtime");
    let runtime = FittedCppRuntime::new(&runtime_root)?;
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut entries = Vec::new();
    for finalist in qualified {
        let dataset = datasets
            .get(finalist.dataset_hash.as_str())
            .context("finalist dataset disappeared")?;
        let candidate = candidate_map
            .get(&(
                finalist.dataset_hash.as_str(),
                finalist.semantic_sha256.as_str(),
            ))
            .context("finalist body disappeared")?;
        let key = compiled_cache_key(
            &request.task,
            &dataset.train_sha256,
            &candidate.semantic_sha256,
            &dataset.protocol_sha256,
            &request.compiler_bundle_sha256,
            &request.target_triple,
        )
        .map_err(anyhow::Error::msg)?;
        let entry = entry_root(args, &key);
        let hit = if entry.join("artifact-manifest.json").is_file() {
            validate_entry(&entry, request, dataset, candidate, &key)?;
            true
        } else {
            build_entry(
                &entry,
                request,
                dataset,
                candidate,
                finalist,
                &key,
                &replay_receipt_sha256,
                &runtime,
            )?;
            false
        };
        hits += usize::from(hit);
        misses += usize::from(!hit);
        let operation = write_operation(
            args,
            request_sha256,
            idempotency_key,
            dataset,
            candidate,
            &key,
            hit,
        )?;
        entries.push(json!({
            "cache_key": key,
            "dataset_hash": dataset.dataset_hash,
            "semantic_sha256": candidate.semantic_sha256,
            "cache_hit": hit,
            "artifact_manifest": entry.join("artifact-manifest.json"),
            "operation_receipt": operation,
        }));
    }
    Ok(json!({
        "schema": "JopeCompileEconomyBuild.v1",
        "mode": "build",
        "request_sha256": request_sha256,
        "replay_receipt": replay_path,
        "replay_receipt_sha256": replay_receipt_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "entries": entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits as f64 / (hits + misses) as f64,
        "cache": cache_status(&cache_root(args))?,
    }))
}

#[allow(clippy::too_many_arguments)]
fn build_entry(
    entry: &Path,
    request: &Request,
    dataset: &Dataset,
    candidate: &Candidate,
    finalist: &Qualified,
    key: &str,
    replay_receipt_sha256: &str,
    runtime: &FittedCppRuntime,
) -> Result<()> {
    verify_dataset_files(dataset)?;
    let train = read_numeric_csv(&dataset.train_csv)?;
    let test = read_numeric_csv(&dataset.test_csv)?;
    ensure!(
        train.features == test.features,
        "train/test feature widths differ"
    );
    validate_targets(&request.task, &train.y, "train")?;
    validate_targets(&request.task, &test.y, "test")?;
    let genome = GenomeIr::parse(&candidate.canonical_genome).context("parse finalist genome")?;
    ensure!(
        genome.canonical() == candidate.canonical_genome,
        "finalist genome is not canonical"
    );
    let train_table = raw_table(train.features, &train.x)?;
    let test_table = raw_table(test.features, &test.x)?;
    let model = if request.task == "regression" {
        fit_legacy_numeric_bound(&genome, &train.x, &train.y)?
    } else {
        let task = TaskSpec::binary(Objective::RocAuc)?;
        fit_table(
            &genome,
            &task,
            &numeric_schema(train.features),
            &train_table,
            &train.y,
        )?
    };
    supports_fitted_cpp17(&model)?;
    let codec = encode_fitted_cpp17(&model)?;
    validate_fitted_cpp_codec(&codec.bytes)?;
    ensure!(
        sha256_bytes(&codec.bytes) == codec.sha256,
        "codec identity drift"
    );
    let interpreted = model.predict_table(&test_table)?;
    let encoded_test = EncodedTable::from_bound_model(&model, &test_table)?;
    let compiled_model = runtime.load(&codec.bytes)?;
    let compiled = compiled_model.predict(&encoded_test)?;
    let interpreted_score = score(&request.task, &test.y, &interpreted)?;
    let compiled_score = score(&request.task, &test.y, &compiled)?;
    let parity = exact_parity(&interpreted, &compiled, interpreted_score, compiled_score);
    ensure!(
        parity.qualified,
        "compiled/interpreted prediction or score bit parity failed"
    );
    let interpreted_prediction_sha256 = prediction_sha256(&interpreted);
    let compiled_prediction_sha256 = prediction_sha256(&compiled);
    ensure!(
        interpreted_prediction_sha256 == compiled_prediction_sha256,
        "prediction hashes differ despite parity"
    );

    let parent = entry.parent().context("cache entry has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".{key}.tmp-{}", std::process::id()));
    if temporary.exists() {
        bail!("temporary cache entry exists: {}", temporary.display());
    }
    fs::create_dir(&temporary)?;
    let result = (|| -> Result<()> {
        let artifact = temporary.join("fitted-model.codec");
        atomic_bytes(&artifact, &codec.bytes)?;
        let manifest = json!({
            "schema": ARTIFACT_SCHEMA,
            "cache_key": key,
            "task": request.task,
            "dataset_hash": dataset.dataset_hash,
            "dataset_train_sha256": dataset.train_sha256,
            "dataset_test_sha256": dataset.test_sha256,
            "semantic_sha256": candidate.semantic_sha256,
            "evaluator_protocol_sha256": dataset.protocol_sha256,
            "compiler_bundle_sha256": request.compiler_bundle_sha256,
            "target_triple": request.target_triple,
            "checkpoint_sha256": request.checkpoint_sha256,
            "artifact": "fitted-model.codec",
            "artifact_sha256": codec.sha256,
            "score": finalist.score,
            "within_dataset_rank": finalist.rank,
            "strict_incumbent_beat": finalist.strict_incumbent_beat,
            "top_two": finalist.top_two,
            "replay_receipt_sha256": replay_receipt_sha256,
            "prediction_bit_parity": parity.prediction_bit_parity,
            "score_bit_parity": parity.score_bit_parity,
            "qualified": parity.qualified,
            "interpreted_score": interpreted_score,
            "compiled_score": compiled_score,
            "interpreted_predictions_sha256": interpreted_prediction_sha256,
            "compiled_predictions_sha256": compiled_prediction_sha256,
            "fitted_state_scope": {
                "task": request.task,
                "dataset_train_sha256": dataset.train_sha256,
                "semantic_sha256": candidate.semantic_sha256,
            },
            "created_at_ms": now_ms(),
        });
        atomic_json(&temporary.join("artifact-manifest.json"), &manifest)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }
    match fs::rename(&temporary, entry) {
        Ok(()) => {}
        Err(_) if entry.join("artifact-manifest.json").is_file() => {
            fs::remove_dir_all(&temporary)?;
            validate_entry(entry, request, dataset, candidate, key)?;
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temporary);
            return Err(error.into());
        }
    }
    validate_entry(entry, request, dataset, candidate, key)
}

fn validate_entry(
    entry: &Path,
    request: &Request,
    dataset: &Dataset,
    candidate: &Candidate,
    key: &str,
) -> Result<()> {
    let manifest_path = entry.join("artifact-manifest.json");
    let manifest = read_json(&manifest_path)?;
    ensure!(
        manifest["schema"] == ARTIFACT_SCHEMA,
        "artifact schema drift"
    );
    ensure!(manifest["cache_key"] == key, "artifact cache key drift");
    ensure!(
        manifest["task"] == request.task
            && manifest["dataset_hash"] == dataset.dataset_hash
            && manifest["dataset_train_sha256"] == dataset.train_sha256
            && manifest["semantic_sha256"] == candidate.semantic_sha256
            && manifest["evaluator_protocol_sha256"] == dataset.protocol_sha256
            && manifest["compiler_bundle_sha256"] == request.compiler_bundle_sha256
            && manifest["target_triple"] == request.target_triple,
        "artifact fitted identity drift"
    );
    ensure!(
        manifest["prediction_bit_parity"] == true
            && manifest["score_bit_parity"] == true
            && manifest["qualified"] == true,
        "artifact lacks exact parity qualification"
    );
    let relative = manifest["artifact"]
        .as_str()
        .context("artifact path is missing")?;
    ensure!(
        relative == "fitted-model.codec",
        "unexpected artifact filename"
    );
    let artifact = entry.join(relative);
    let expected = manifest["artifact_sha256"]
        .as_str()
        .context("artifact hash is missing")?;
    ensure!(sha256_file(&artifact)? == expected, "artifact hash drift");
    validate_fitted_cpp_codec(&fs::read(artifact)?)?;
    Ok(())
}

fn candidate_map(request: &Request) -> BTreeMap<(&str, &str), &Candidate> {
    request
        .candidates
        .iter()
        .map(|row| {
            (
                (row.dataset_hash.as_str(), row.semantic_sha256.as_str()),
                row,
            )
        })
        .collect()
}

fn verify_dataset_files(dataset: &Dataset) -> Result<()> {
    ensure!(
        sha256_file(&dataset.train_csv)? == dataset.train_sha256,
        "train content hash drift for {}",
        dataset.dataset_hash
    );
    ensure!(
        sha256_file(&dataset.test_csv)? == dataset.test_sha256,
        "test content hash drift for {}",
        dataset.dataset_hash
    );
    Ok(())
}

fn read_numeric_csv(path: &Path) -> Result<NumericData> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read numeric CSV {}", path.display()))?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut features = None;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field.trim().parse::<f64>().with_context(|| {
                    format!("{}:{} invalid numeric field", path.display(), line_no + 1)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.len() >= 2,
            "{}:{} needs a feature and target",
            path.display(),
            line_no + 1
        );
        let width = values.len() - 1;
        if let Some(expected) = features {
            ensure!(
                width == expected,
                "{}:{} feature width drift",
                path.display(),
                line_no + 1
            );
        } else {
            features = Some(width);
        }
        let target = values.pop().expect("length checked");
        ensure!(
            target.is_finite(),
            "{}:{} non-finite target",
            path.display(),
            line_no + 1
        );
        for value in &mut values {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        x.push(values);
        y.push(target);
    }
    Ok(NumericData {
        x,
        y,
        features: features.context("empty numeric CSV")?,
    })
}

fn validate_targets(task: &str, y: &[f64], split: &str) -> Result<()> {
    ensure!(!y.is_empty(), "{split} targets are empty");
    if task == "binary" {
        let positive = y.iter().filter(|value| **value == 1.0).count();
        ensure!(
            y.iter().all(|value| *value == 0.0 || *value == 1.0),
            "{split} binary targets are not exactly 0/1"
        );
        ensure!(
            positive > 0 && positive < y.len(),
            "{split} binary targets lack both classes"
        );
    }
    Ok(())
}

fn numeric_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|index| ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

fn raw_table(features: usize, rows: &[Vec<f64>]) -> Result<RawTable> {
    RawTable::new(
        (0..features).map(|index| format!("c{index}")).collect(),
        rows.iter()
            .map(|row| row.iter().map(|value| format!("{value:?}")).collect())
            .collect(),
    )
}

fn score(task: &str, target: &[f64], prediction: &[f64]) -> Result<f64> {
    ensure!(
        target.len() == prediction.len() && prediction.iter().all(|value| value.is_finite()),
        "prediction output is missing or non-finite"
    );
    let value = if task == "binary" {
        ensure!(
            prediction.iter().all(|value| (0.0..=1.0).contains(value)),
            "binary compiled predictions are outside [0,1]"
        );
        auc(target, prediction)
    } else {
        r2(target, prediction)
    };
    ensure!(value.is_finite(), "compiled score is non-finite");
    Ok(value)
}

fn prediction_sha256(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_bytes(&bytes)
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .context("output filename is not UTF-8")?,
        std::process::id()
    ));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_operation(
    args: &Args,
    request_sha256: &str,
    idempotency_key: &str,
    dataset: &Dataset,
    candidate: &Candidate,
    cache_key: &str,
    hit: bool,
) -> Result<PathBuf> {
    let identity = json!({
        "request_sha256": request_sha256,
        "idempotency_key": idempotency_key,
        "cache_key": cache_key,
    });
    let operation_sha256 = sha256_bytes(&canonical_json_bytes(&identity)?);
    let path = cache_root(args)
        .join("operations")
        .join(format!("{operation_sha256}.json"));
    let value = json!({
        "schema": OPERATION_SCHEMA,
        "operation_sha256": operation_sha256,
        "request_sha256": request_sha256,
        "idempotency_key": idempotency_key,
        "cache_key": cache_key,
        "dataset_hash": dataset.dataset_hash,
        "semantic_sha256": candidate.semantic_sha256,
        "outcome": if hit { "hit" } else { "miss" },
        "created_at_ms": now_ms(),
    });
    if path.is_file() {
        let prior = read_json(&path)?;
        for field in [
            "schema",
            "operation_sha256",
            "request_sha256",
            "idempotency_key",
            "cache_key",
            "dataset_hash",
            "semantic_sha256",
        ] {
            ensure!(prior[field] == value[field], "idempotency receipt drift");
        }
        return Ok(path);
    }
    atomic_json(&path, &value)?;
    Ok(path)
}

fn cache_root(args: &Args) -> PathBuf {
    args.root.join("compiled-cache")
}

fn entry_root(args: &Args, key: &str) -> PathBuf {
    cache_root(args).join("entries").join(&key[..2]).join(key)
}

fn cache_status(root: &Path) -> Result<Value> {
    let mut entries = 0usize;
    let mut valid_entries = 0usize;
    let entries_root = root.join("entries");
    if entries_root.is_dir() {
        for prefix in fs::read_dir(&entries_root)? {
            let prefix = prefix?.path();
            if !prefix.is_dir() {
                continue;
            }
            for entry in fs::read_dir(prefix)? {
                let manifest = entry?.path().join("artifact-manifest.json");
                if !manifest.is_file() {
                    continue;
                }
                entries += 1;
                if generic_entry_valid(&manifest).unwrap_or(false) {
                    valid_entries += 1;
                }
            }
        }
    }
    let mut hits = 0usize;
    let mut misses = 0usize;
    let operations = root.join("operations");
    if operations.is_dir() {
        for row in fs::read_dir(operations)? {
            let path = row?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let value = match read_json(&path) {
                Ok(value) if value["schema"] == OPERATION_SCHEMA => value,
                _ => continue,
            };
            match value["outcome"].as_str() {
                Some("hit") => hits += 1,
                Some("miss") => misses += 1,
                _ => {}
            }
        }
    }
    let requests = hits + misses;
    Ok(json!({
        "entries": entries,
        "valid_entries": valid_entries,
        "invalid_entries": entries - valid_entries,
        "hits": hits,
        "misses": misses,
        "hit_rate": if requests == 0 { Value::Null } else { json!(hits as f64 / requests as f64) },
    }))
}

fn generic_entry_valid(manifest_path: &Path) -> Result<bool> {
    let manifest = read_json(manifest_path)?;
    if manifest["schema"] != ARTIFACT_SCHEMA
        || manifest["prediction_bit_parity"] != true
        || manifest["score_bit_parity"] != true
        || manifest["qualified"] != true
    {
        return Ok(false);
    }
    let entry = manifest_path.parent().context("manifest has no parent")?;
    let relative = manifest["artifact"]
        .as_str()
        .context("artifact is missing")?;
    let expected = manifest["artifact_sha256"]
        .as_str()
        .context("artifact hash is missing")?;
    Ok(relative == "fitted-model.codec" && sha256_file(&entry.join(relative))? == expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn dataset(index: usize, family: &str) -> Dataset {
        Dataset {
            dataset_hash: format!("d{index:03}"),
            source_family: family.to_owned(),
            train_csv: PathBuf::from(format!("/train/{index}")),
            test_csv: PathBuf::from(format!("/test/{index}")),
            train_sha256: digest((index + 1) as u8),
            test_sha256: digest((index + 2) as u8),
            protocol_sha256: digest((index + 3) as u8),
            incumbent_score: Some(0.5),
        }
    }

    fn request() -> Request {
        let datasets = (0..40)
            .map(|index| dataset(index, &format!("f{index:03}")))
            .collect::<Vec<_>>();
        let genomes = [
            "(pipe X (ridge))",
            "(pipe X (ridge l2=0.1))",
            "(pipe X (mean))",
        ];
        let candidates = genomes
            .iter()
            .enumerate()
            .map(|(index, genome)| Candidate {
                dataset_hash: "d000".to_owned(),
                semantic_sha256: sha256_bytes(genome.as_bytes()),
                canonical_genome: (*genome).to_owned(),
                score: 0.7 - index as f64 / 10.0,
                valid: true,
            })
            .collect();
        Request {
            schema: REQUEST_SCHEMA.to_owned(),
            task: "regression".to_owned(),
            checkpoint_sha256: digest(240),
            catalog_sha256: digest(241),
            compiler_bundle_sha256: digest(242),
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            frozen_seed_sha256: digest(243),
            hall_of_fame_sha256: digest(244),
            generation: "strict-wave-000001".to_owned(),
            panel_seed: default_panel_seed(),
            datasets,
            candidates,
        }
    }

    #[test]
    fn panel_is_exact_family_diverse_and_deterministic() {
        let request = request();
        let first = select_panel(&request).unwrap();
        let second = select_panel(&request).unwrap();
        assert_eq!(first.len(), PANEL_SIZE);
        assert_eq!(
            first
                .iter()
                .map(|row| &row.dataset_hash)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|row| &row.dataset_hash)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            first
                .iter()
                .map(|row| &row.source_family)
                .collect::<BTreeSet<_>>()
                .len(),
            PANEL_SIZE
        );
    }

    #[test]
    fn finalists_are_top_two_or_strict_and_bounded() {
        let request = request();
        let finalists = qualified_candidates(&request).unwrap();
        assert_eq!(finalists.len(), 2);
        assert!(finalists.iter().all(|row| row.top_two));
        assert!(finalists.iter().all(|row| row.strict_incumbent_beat));
    }

    #[test]
    fn native_executor_rejects_scripts() {
        let temporary = TempDir::new().unwrap();
        let path = temporary.path().join("fake");
        fs::write(&path, b"#!/usr/bin/python3\n").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        assert!(validate_native_elf(&path, "evaluator").is_err());
    }

    #[test]
    fn parity_is_bit_exact() {
        assert!(exact_parity(&[0.0, 1.0], &[0.0, 1.0], 1.0, 1.0).qualified);
        assert!(!exact_parity(&[-0.0], &[0.0], 1.0, 1.0).qualified);
    }

    #[test]
    fn cache_status_counts_append_only_operations() {
        let temporary = TempDir::new().unwrap();
        let operations = temporary.path().join("operations");
        fs::create_dir_all(&operations).unwrap();
        atomic_json(
            &operations.join("a.json"),
            &json!({"schema": OPERATION_SCHEMA, "outcome": "hit"}),
        )
        .unwrap();
        atomic_json(
            &operations.join("b.json"),
            &json!({"schema": OPERATION_SCHEMA, "outcome": "miss"}),
        )
        .unwrap();
        let status = cache_status(temporary.path()).unwrap();
        assert_eq!(status["hits"], 1);
        assert_eq!(status["misses"], 1);
        assert_eq!(status["hit_rate"], 0.5);
    }

    #[test]
    fn native_build_emits_dataset_bound_bit_parity_artifact() {
        let temporary = TempDir::new().unwrap();
        let train = temporary.path().join("train.csv");
        let test = temporary.path().join("test.csv");
        fs::write(&train, b"0,0\n1,1\n2,2\n3,3\n").unwrap();
        fs::write(&test, b"4,4\n5,5\n6,6\n7,7\n").unwrap();
        let genome = "(pipe X (ridge))";
        let candidate = Candidate {
            dataset_hash: "fixture".to_owned(),
            semantic_sha256: sha256_bytes(genome.as_bytes()),
            canonical_genome: genome.to_owned(),
            score: 0.9,
            valid: true,
        };
        let dataset = Dataset {
            dataset_hash: candidate.dataset_hash.clone(),
            source_family: "fixture-family".to_owned(),
            train_csv: train.clone(),
            test_csv: test.clone(),
            train_sha256: sha256_file(&train).unwrap(),
            test_sha256: sha256_file(&test).unwrap(),
            protocol_sha256: digest(230),
            incumbent_score: Some(0.8),
        };
        let mut request = request();
        request.datasets = vec![dataset.clone()];
        request.candidates = vec![candidate.clone()];
        let key = compiled_cache_key(
            &request.task,
            &dataset.train_sha256,
            &candidate.semantic_sha256,
            &dataset.protocol_sha256,
            &request.compiler_bundle_sha256,
            &request.target_triple,
        )
        .unwrap();
        let entry = temporary.path().join("entry");
        let runtime = FittedCppRuntime::new(temporary.path().join("runtime")).unwrap();
        build_entry(
            &entry,
            &request,
            &dataset,
            &candidate,
            &Qualified {
                dataset_hash: dataset.dataset_hash.clone(),
                semantic_sha256: candidate.semantic_sha256.clone(),
                score: candidate.score,
                rank: 1,
                top_two: true,
                strict_incumbent_beat: true,
                incumbent_score: dataset.incumbent_score,
            },
            &key,
            &digest(231),
            &runtime,
        )
        .unwrap();
        validate_entry(&entry, &request, &dataset, &candidate, &key).unwrap();
        let manifest = read_json(&entry.join("artifact-manifest.json")).unwrap();
        assert_eq!(manifest["prediction_bit_parity"], true);
        assert_eq!(manifest["score_bit_parity"], true);
        assert_eq!(
            manifest["fitted_state_scope"]["dataset_train_sha256"],
            dataset.train_sha256
        );
    }
}
