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

mod build;
mod replay;

pub(super) use build::{build_entry, run_build, validate_entry};
pub(super) use replay::{run_replay, validate_native_elf, validate_replay_receipt};

mod cache;
mod numeric;
mod selection;
pub(super) use cache::{
    atomic_bytes, cache_root, cache_status, entry_root, generic_entry_valid, write_operation,
};
pub(super) use numeric::{
    numeric_schema, prediction_sha256, raw_table, read_numeric_csv, score, validate_targets,
};
pub(super) use selection::{
    audit_report, candidate_map, qualified_candidates, select_panel, verify_dataset_files,
};

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

#[cfg(test)]
mod tests;
