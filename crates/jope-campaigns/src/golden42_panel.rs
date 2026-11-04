use crate::data::read_regression_csv;
use anyhow::{bail, Context, Result};
use feat_math::{
    emit_fitted_cpp17, encode_fitted_cpp17, fit_legacy_numeric_bound, fit_table,
    initial_population, next_generation, validate_fitted_cpp_codec, BoundModel, ColumnSpec,
    DataSchema, EncodedTable, FittedCppRuntime, GenomeIr, Objective, RawTable, RowSemantics,
    TaskSpec, V3Archive, V3Candidate, V3EvolutionConfig, V3ScoredCandidate,
};
use jope_dime_regressor::paths;
/// Streamed (bounded memory); the same digest as hashing the whole file.
use jope_dime_regressor::selfplay_runtime::sha256_file;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DATASET: &str = "24713c7f7bfa7807";
const TRAIN_SHA256: &str = "792b5536a28dc7180265ea0a0f5a770e9d7eb002b150298e20c66f3278690836";
const TEST_SHA256: &str = "89b77683e7c4899d043f6bbad31d291cbae40d90639fc2d161655d965a2d66b2";
const WINNER_RMSE: f64 = 0.06295124334370077;
const WORKERS: usize = 80;
const ADMISSION_RELATIVE: &str = "manifests/advanced-v3/fitted-cpp-v3-admission.v1.json";
jope_paths::once_str!(
    fn node_lease_str => paths::legacy_tree("jope-golden42-controller-node-leases").join("xbabe2-v3.lock")
);
const MIN_RAM: u64 = 64 * 1024 * 1024 * 1024;
const MIN_DISK: u64 = 100 * 1024 * 1024 * 1024;
const MAX_SWAP: u64 = 1024 * 1024 * 1024;
const NEAR_WINNER: &str = "(pipe X (target kind=log1p) (quantile_bins append=true bins=25) (blom append=true) (pair_prod append=true topk=12) (pair_prod append=true topk=13) (quantile_bins append=true bins=25) (blom append=true) (elasticnet alpha=0.028669353704044713 l1_ratio=0.09375316848524005 sweeps=60))";
const COMPATIBLE_WINNER: &str = "(pipe X (target kind=log1p) (quantile_bins append=true bins=25) (blom append=true) (pair_prod append=true topk=12) (quantile_bins append=true bins=2) (blom append=true) (pair_prod append=true topk=12) (elasticnet alpha=0.025 l1_ratio=0.09375316848524005 sweeps=60))";
const GOLDEN42_FORMULA_SEED: &str = "(pipe X (target kind=log1p) (rank append=true) (quantile_bins bins=16 append=true) (pair_ratio topk=12 append=true) (elasticnet alpha=0.02 l1_ratio=0.1 sweeps=60))";
const NEAREST_PROFILE_SEED: &str =
    "(pipe X (bin_onehot append=true bins=11) (standardize) (gbm depth=3 rounds=83))";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissionReceipt {
    schema: String,
    admitted: bool,
    test_adaptive: bool,
    benchmark_valid: bool,
    compiler: String,
    compiler_flags: Vec<String>,
    release_sha256: String,
    source_sha256: String,
    gates: BTreeMap<String, GateReceipt>,
    performance: PerformanceReceipt,
    smoke: SmokeReceipt,
    created_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateReceipt {
    passed: bool,
    command: String,
    elapsed_ms: u128,
    output_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerformanceReceipt {
    runs: usize,
    rust_median_ns: u128,
    cpp_median_ns: u128,
    bit_parity: bool,
    codec_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SmokeReceipt {
    candidates: usize,
    worker_processes: usize,
    distinct_worker_ids: usize,
    cpp_only: bool,
    ranking_sha256: String,
    next_population_sha256: String,
    completion_order_independent: bool,
    best_rmse: f64,
    elapsed_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignManifest {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    test_sha256: String,
    node: String,
    workers: usize,
    population: usize,
    islands: usize,
    island_size: usize,
    strict_threshold: f64,
    test_adaptive: bool,
    benchmark_valid: bool,
    prediction_backend: String,
    release_sha256: String,
    admission_sha256: String,
    initial_population: Vec<V3Candidate>,
    created_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerBootstrap {
    schema: String,
    run_id: String,
    dataset_hash: String,
    worker_id: usize,
    cpu: usize,
    release_sha256: String,
    train_path: PathBuf,
    test_features_path: PathBuf,
    output_root: PathBuf,
    runtime_root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequest {
    schema: String,
    generation: usize,
    slot: usize,
    candidate: V3Candidate,
    context_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerResponse {
    schema: String,
    worker_id: usize,
    generation: usize,
    slot: usize,
    semantic_hash: String,
    context_sha256: String,
    predictions: Vec<f64>,
    prediction_sha256: String,
    codec_sha256: String,
    codec_file: PathBuf,
    fit_ns: u128,
    codec_ns: u128,
    cpp_load_ns: u128,
    cpp_predict_ns: u128,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    slot: usize,
    worker_id: usize,
    cpu: usize,
    candidate: V3Candidate,
    context_sha256: String,
    rmse: f64,
    predictions: Vec<f64>,
    prediction_sha256: String,
    codec_sha256: String,
    codec_file: PathBuf,
    fit_ns: u128,
    codec_ns: u128,
    cpp_load_ns: u128,
    cpp_predict_ns: u128,
    started_at_unix_ms: u128,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    previous_sha256: Option<String>,
    ranked_semantic_hashes: Vec<String>,
    scored: Vec<V3ScoredCandidate>,
    archive: V3Archive,
    next_population: Vec<V3Candidate>,
    created_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WinnerClaim {
    schema: String,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    slot: usize,
    semantic_hash: String,
    rmse: f64,
    threshold: f64,
    codec_sha256: String,
    prediction_sha256: String,
    claimed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmationRequest {
    schema: String,
    cpu: usize,
    candidate: V3Candidate,
    train_path: PathBuf,
    test_features_path: PathBuf,
    runtime_root: PathBuf,
    expected_codec_sha256: String,
    expected_prediction_sha256: String,
    output: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmationReceipt {
    schema: String,
    cpu: usize,
    pid: u32,
    codec_sha256: String,
    prediction_sha256: String,
    predictions: Vec<f64>,
    confirmed_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalResult {
    schema: String,
    run_id: String,
    dataset_hash: String,
    outcome: String,
    generation: usize,
    slot: usize,
    rmse: f64,
    strict_threshold: f64,
    candidate: V3Candidate,
    winner_claim_sha256: String,
    confirmation_sha256: Vec<String>,
    codec_sha256: String,
    prediction_sha256: String,
    standalone_source_sha256: String,
    frozen_codec_path: PathBuf,
    standalone_source_path: PathBuf,
    test_adaptive: bool,
    benchmark_valid: bool,
    finished_at_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize)]
struct Telemetry<'a> {
    schema: &'static str,
    unix_ms: u128,
    generation: usize,
    completed: usize,
    pending: usize,
    workers_alive: usize,
    workers_affinity_ok: usize,
    worker_states: Vec<WorkerTelemetry>,
    cpu_load_1m: f64,
    available_ram_bytes: u64,
    swap_used_bytes: u64,
    free_disk_bytes: u64,
    cache_hits: usize,
    invalids: usize,
    diversity: usize,
    candidate_latency_p50_ms: f64,
    throughput_candidates_per_minute: f64,
    operator_success: bool,
    run_id: &'a str,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerTelemetry {
    worker_id: usize,
    pid: u32,
    assigned_cpu: usize,
    allowed_cpu_list: String,
    state: String,
    cpu_time_ticks: u64,
    rss_bytes: u64,
    alive: bool,
}

struct Lease {
    file: File,
}

impl Lease {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        if file.try_lock().is_err() {
            bail!("campaign lease is already held at {}", path.display());
        }
        Ok(Self { file })
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

struct WorkerProcess {
    id: usize,
    cpu: usize,
    child: Child,
    stdin: BufWriter<ChildStdin>,
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let pid = self.child.id() as i32;
            // SAFETY: kill reads no memory; try_wait just saw this child unreaped, so its PID
            // (never 0 or 1) is not reused and -pid can only name the group the child leads.
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
    }
}

struct CampaignTraining {
    table: RawTable,
    schema: DataSchema,
    task: TaskSpec,
}

struct Health {
    ram: u64,
    swap: u64,
    disk: u64,
    load: f64,
}

mod admission;
mod artifacts;
mod campaign;
mod confirmation;
mod ranking;
mod telemetry;
mod training;
mod validation;
mod worker_control;
mod workers;

pub(super) use admission::*;
use artifacts::*;
pub(super) use campaign::*;
pub(super) use confirmation::*;
use ranking::*;
use telemetry::*;
use training::*;
pub(super) use validation::*;
use worker_control::*;
pub(super) use workers::*;

#[cfg(test)]
mod tests;
