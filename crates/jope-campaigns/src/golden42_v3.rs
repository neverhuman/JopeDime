use crate::data::read_regression_csv;
use anyhow::{bail, Context, Result};
use feat_math::{
    emit_fitted_cpp17, encode_fitted_cpp17, fit_legacy_numeric_bound, fit_table,
    initial_population, next_generation, validate_fitted_cpp_codec, BoundModel, ColumnSpec,
    DataSchema, EncodedTable, FittedCppRuntime, GenomeIr, Objective, RawTable, RowSemantics,
    TaskSpec, V3Archive, V3Candidate, V3EvolutionConfig, V3ScoredCandidate,
};
use jope_dime_regressor::paths;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::fd::AsRawFd;
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
        let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if status != 0 {
            bail!("campaign lease is already held at {}", path.display());
        }
        Ok(Self { file })
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
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

pub(crate) fn admit(output: &Path) -> Result<()> {
    let workspace = workspace_root();
    let compiler_output = Command::new("g++").arg("--version").output()?;
    if !compiler_output.status.success() {
        bail!("g++ --version failed");
    }
    let compiler = String::from_utf8(compiler_output.stdout)?;
    if !compiler.lines().next().unwrap_or_default().contains("13.3") {
        bail!("fitted C++ admission requires g++ 13.3, found {compiler:?}");
    }
    let commands = [
        (
            "default_workspace_check",
            vec!["check", "--workspace", "--all-targets"],
            None,
        ),
        (
            "all_feature_workspace_check",
            vec!["check", "--workspace", "--all-targets", "--all-features"],
            None,
        ),
        (
            "legacy_freeze_golden_champion_token_portable_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "legacy_v2_freeze",
                "--test",
                "golden",
                "--test",
                "champion_parity",
                "--test",
                "tokens",
                "--test",
                "portable_parity",
            ],
            None,
        ),
        ("gpu_backend_parity", vec!["test", "-p", "battle-gpu"], None),
        (
            "normal_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            None,
        ),
        (
            "asan_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            Some(("JOPE_FITTED_CPP_SANITIZER", "address")),
        ),
        (
            "ubsan_nonrecovering_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            Some(("JOPE_FITTED_CPP_SANITIZER", "undefined")),
        ),
        (
            "codec_malformed_once_compiled_dynamic_runtime",
            vec!["test", "-p", "feat-math", "fitted_cpp_runtime"],
            None,
        ),
        (
            "mutation_crossover_100k",
            vec![
                "test",
                "-p",
                "feat-math",
                "one_hundred_thousand_typed_children_compile_without_retry",
            ],
            None,
        ),
    ];
    let mut gates = BTreeMap::new();
    for (name, args, environment) in commands {
        let started = Instant::now();
        let mut command = Command::new("cargo");
        command.current_dir(&workspace).args(&args);
        if let Some((key, value)) = environment {
            command.env(key, value);
        }
        let result = command
            .output()
            .with_context(|| format!("launch cargo gate {name}"))?;
        let mut combined = result.stdout;
        combined.extend_from_slice(&result.stderr);
        if !result.status.success() {
            bail!(
                "fitted C++ admission gate {name} failed:\n{}",
                String::from_utf8_lossy(&combined)
            );
        }
        gates.insert(
            name.to_string(),
            GateReceipt {
                passed: true,
                command: format!(
                    "{}cargo {}",
                    environment.map_or(String::new(), |(key, value)| format!("{key}={value} ")),
                    args.join(" ")
                ),
                elapsed_ms: started.elapsed().as_millis(),
                output_sha256: sha256_bytes(&combined),
            },
        );
    }
    let performance = performance_gate()?;
    let smoke = smoke_gate()?;
    let release = std::env::current_exe()?;
    let receipt = AdmissionReceipt {
        schema: "jope-dime.fitted-cpp-v3-admission.v1".to_string(),
        admitted: true,
        test_adaptive: true,
        benchmark_valid: false,
        compiler: compiler.lines().next().unwrap_or_default().to_string(),
        compiler_flags: [
            "-std=c++17",
            "-O3",
            "-march=znver2",
            "-ffp-contract=off",
            "-fno-fast-math",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        release_sha256: sha256_file(&release)?,
        source_sha256: source_tree_hash()?,
        gates,
        performance,
        smoke,
        created_at_unix_ms: unix_ms(),
    };
    atomic_immutable_write(output, &json_bytes(&receipt)?)
}

fn performance_gate() -> Result<PerformanceReceipt> {
    let registry = paths::jope_registry();
    validate_dataset_hashes(&registry)?;
    let root = registry.join(DATASET);
    let train = read_regression_csv(&root.join("train.csv"))?;
    let test = read_regression_csv(&root.join("test.csv"))?;
    let genome = GenomeIr::parse(NEAR_WINNER)?;
    let training = campaign_training(&train.x)?;
    let model = fit_campaign_candidate(&genome, &training, &train.y)?;
    let legacy = fit_legacy_numeric_bound(&genome, &train.x, &train.y)?;
    let codec = encode_fitted_cpp17(&model)?;
    validate_fitted_cpp_codec(&codec.bytes)?;
    let encoded = EncodedTable::from_numeric_rows(&test.x)?;
    let runtime_root = workspace_root().join("target/fitted-cpp-admission-runtime");
    fs::create_dir_all(&runtime_root)?;
    let runtime = FittedCppRuntime::new(&runtime_root)?;
    let cpp = runtime.load(&codec.bytes)?;
    let raw = numeric_raw_table(&test.x)?;
    if prediction_bits(&legacy.predict_table(&raw)?) != prediction_bits(&model.predict_table(&raw)?)
    {
        bail!("V3 campaign fit is not bit-identical to the frozen legacy near-winner fit");
    }
    let mut rust_times = Vec::with_capacity(30);
    let mut cpp_times = Vec::with_capacity(30);
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        expected = model.predict_table(&raw)?;
        rust_times.push(started.elapsed().as_nanos());
        let started = Instant::now();
        actual = cpp.predict(&encoded)?;
        cpp_times.push(started.elapsed().as_nanos());
    }
    let parity = prediction_bits(&expected) == prediction_bits(&actual);
    if !parity {
        let (index, rust, cpp) = first_prediction_mismatch(&expected, &actual)
            .context("30-run fitted C++ performance gate lost bit parity without a mismatch")?;
        bail!(
            "30-run fitted C++ performance gate lost bit parity at prediction {index}: Rust {rust:?} ({:#018x}), C++ {cpp:?} ({:#018x})",
            rust.to_bits(),
            cpp.to_bits()
        );
    }
    rust_times.sort_unstable();
    cpp_times.sort_unstable();
    let rust_median = rust_times[15];
    let cpp_median = cpp_times[15];
    if cpp_median > rust_median.saturating_mul(4).saturating_add(1_000_000) {
        bail!(
            "fitted C++ prediction median {cpp_median}ns exceeds correctness-first envelope against Rust {rust_median}ns"
        );
    }
    drop(cpp);
    Ok(PerformanceReceipt {
        runs: 30,
        rust_median_ns: rust_median,
        cpp_median_ns: cpp_median,
        bit_parity: parity,
        codec_sha256: codec.sha256,
    })
}

fn smoke_gate() -> Result<SmokeReceipt> {
    let started = Instant::now();
    let registry = paths::jope_registry();
    validate_dataset_hashes(&registry)?;
    let dataset_root = registry.join(DATASET);
    let test = read_regression_csv(&dataset_root.join("test.csv"))?;
    let release = std::env::current_exe()?;
    let release_sha256 = sha256_file(&release)?;
    let root = workspace_root().join("target").join(format!(
        "golden42-v3-smoke-{}-{}",
        std::process::id(),
        unix_ms()
    ));
    fs::create_dir_all(&root)?;
    let result = (|| -> Result<SmokeReceipt> {
        wait_for_resources(&root, "admission-smoke", 0, 0, WORKERS, 0, 0)?;
        let test_features = root.join("test-features.v1.json");
        atomic_immutable_write(&test_features, &json_bytes(&test.x)?)?;
        let cpu_ids = choose_cpus(WORKERS)?;
        let config = V3EvolutionConfig::default();
        let population = initial_population(&config, &v3_seed_sources());
        if population.len() != WORKERS {
            bail!("V3 smoke population is not 80 candidates");
        }
        let mut workers = spawn_workers(
            &release,
            &registry,
            &root,
            "admission-smoke",
            &dataset_root.join("train.csv"),
            &test_features,
            &cpu_ids,
        )?;
        let smoke_result = (|| -> Result<SmokeReceipt> {
            write_worker_registry(&root, &workers)?;
            let (_sender, receiver) = worker_channel(&mut workers)?;
            for (slot, candidate) in population.iter().enumerate() {
                let request = WorkerRequest {
                    schema: "jope-dime.golden42-v3-worker-request.v1".to_string(),
                    generation: 0,
                    slot,
                    candidate: candidate.clone(),
                    context_sha256: context_hash(&release_sha256, candidate),
                };
                serde_json::to_writer(&mut workers[slot].stdin, &request)?;
                workers[slot].stdin.write_all(b"\n")?;
                workers[slot].stdin.flush()?;
            }
            let mut completion_order = Vec::with_capacity(WORKERS);
            let mut worker_ids = BTreeSet::new();
            while completion_order.len() < WORKERS {
                match receiver.recv_timeout(Duration::from_secs(1)) {
                    Ok((worker_id, response)) => {
                        let response = response?;
                        if response.worker_id != worker_id
                            || response.slot >= population.len()
                            || response.generation != 0
                            || response.semantic_hash != population[response.slot].semantic_hash
                            || response.context_sha256
                                != context_hash(&release_sha256, &population[response.slot])
                            || response.predictions.len() != test.y.len()
                            || response.prediction_sha256 != prediction_hash(&response.predictions)
                        {
                            bail!("V3 smoke worker returned an invalid response");
                        }
                        validate_codec_artifact(
                            &root,
                            &response.codec_file,
                            &response.codec_sha256,
                        )?;
                        let rmse = raw_rmse(&test.y, &response.predictions)?;
                        worker_ids.insert(worker_id);
                        completion_order.push(V3ScoredCandidate {
                            candidate: population[response.slot].clone(),
                            rmse,
                            novelty: 1.0,
                        });
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if started.elapsed() > Duration::from_secs(3_600) {
                            bail!("V3 80-candidate smoke generation timed out");
                        }
                        for worker in &mut workers {
                            if let Some(status) = worker.child.try_wait()? {
                                bail!("V3 smoke worker {} exited early with {status}", worker.id);
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        bail!("V3 smoke worker response channels disconnected")
                    }
                }
            }
            let next_completion = next_generation(&config, 1, &completion_order, &[0; 4])?;
            let mut ranked = completion_order.clone();
            ranked.sort_by(scored_order);
            let next_ranked = next_generation(&config, 1, &ranked, &[0; 4])?;
            if next_completion != next_ranked {
                bail!("V3 next generation changed with 80-worker completion order");
            }
            let ranking_sha256 = sha256_bytes(
                ranked
                    .iter()
                    .flat_map(|item| {
                        [
                            item.candidate.semantic_hash.as_bytes(),
                            &item.rmse.to_bits().to_le_bytes(),
                        ]
                        .concat()
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            let best_rmse = ranked
                .iter()
                .map(|item| item.rmse)
                .min_by(f64::total_cmp)
                .context("empty V3 smoke ranking")?;
            Ok(SmokeReceipt {
                candidates: completion_order.len(),
                worker_processes: workers.len(),
                distinct_worker_ids: worker_ids.len(),
                cpp_only: true,
                ranking_sha256,
                next_population_sha256: sha256_bytes(&json_bytes(&next_ranked)?),
                completion_order_independent: true,
                best_rmse,
                elapsed_ms: started.elapsed().as_millis(),
            })
        })();
        terminate_workers(&mut workers);
        smoke_result
    })();
    let cleanup = fs::remove_dir_all(&root);
    match (result, cleanup) {
        (Ok(receipt), Ok(())) => Ok(receipt),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
    }
}

pub(crate) fn run(
    registry: &Path,
    run_id: &str,
    dataset: &str,
    node: &str,
    workers: usize,
    population: usize,
    resume: bool,
) -> Result<()> {
    if dataset != DATASET || node != "xbabe2" || workers != WORKERS || population != WORKERS {
        bail!("Golden42 V3 requires dataset {DATASET}, node xbabe2, and 80 workers/population");
    }
    let hostname = command_text("hostname", &["-s"])?;
    if hostname.trim() != "xbabe2" {
        bail!("Golden42 V3 must run on xbabe2, found {}", hostname.trim());
    }
    validate_dataset_hashes(registry)?;
    let workspace = workspace_root();
    let root = workspace.join("runs").join(run_id);
    let admission_path = workspace.join(ADMISSION_RELATIVE);
    let admission_bytes = fs::read(&admission_path)
        .with_context(|| format!("read admission receipt {}", admission_path.display()))?;
    let admission: AdmissionReceipt = serde_json::from_slice(&admission_bytes)?;
    if !admission.admitted || admission.benchmark_valid || !admission.test_adaptive {
        bail!("fitted C++ admission receipt is not production-admitted for this campaign");
    }
    if admission.source_sha256 != source_tree_hash()? {
        bail!("fitted C++ admission receipt does not bind the current source tree");
    }
    let release = std::env::current_exe()?;
    let release_sha256 = sha256_file(&release)?;
    if admission.release_sha256 != release_sha256 {
        bail!("admission receipt does not bind the running release binary");
    }
    if root.exists() && !resume {
        bail!("run root {} already exists; use --resume", root.display());
    }
    fs::create_dir_all(&root)?;
    let _node_lease = Lease::acquire(Path::new(node_lease_str()))?;
    let _run_lease = Lease::acquire(&root.join("run.lock"))?;
    wait_for_resources(&root, run_id, 0, 0, 0, 0, 0)?;
    let cpu_ids = choose_cpus(WORKERS)?;
    let dataset_root = registry.join(DATASET);
    let test = read_regression_csv(&dataset_root.join("test.csv"))?;
    let test_features = root.join("test-features.v1.json");
    atomic_immutable_write(&test_features, &json_bytes(&test.x)?)?;
    let config = V3EvolutionConfig::default();
    let supplied = v3_seed_sources();
    let initial = initial_population(&config, &supplied);
    let manifest_path = root.join("manifest.v3.json");
    let admission_sha256 = sha256_bytes(&admission_bytes);
    let manifest = if manifest_path.exists() {
        let existing: CampaignManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        validate_manifest_contract(
            &existing,
            run_id,
            node,
            workers,
            population,
            &release_sha256,
            &admission_sha256,
            &initial,
        )?;
        existing
    } else {
        let created = CampaignManifest {
            schema: "jope-dime.golden42-v3-max-manifest.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            train_sha256: TRAIN_SHA256.to_string(),
            test_sha256: TEST_SHA256.to_string(),
            node: node.to_string(),
            workers,
            population,
            islands: 4,
            island_size: 20,
            strict_threshold: WINNER_RMSE,
            test_adaptive: true,
            benchmark_valid: false,
            prediction_backend: "fitted-cpp17-only".to_string(),
            release_sha256: release_sha256.clone(),
            admission_sha256,
            initial_population: initial.clone(),
            created_at_unix_ms: unix_ms(),
        };
        atomic_immutable_write(&manifest_path, &json_bytes(&created)?)?;
        created
    };
    if root.join("final-result.v1.json").exists() {
        validate(run_id)?;
        return Ok(());
    }
    let (mut generation, mut population, mut archive, mut previous_checkpoint) =
        resume_state(&root, &manifest)?;
    let mut retry = 0usize;
    loop {
        let mut workers = spawn_workers(
            &release,
            registry,
            &root,
            run_id,
            &dataset_root.join("train.csv"),
            &test_features,
            &cpu_ids,
        )?;
        write_worker_registry(&root, &workers)?;
        let result = campaign_loop(
            &root,
            run_id,
            &release_sha256,
            &test.y,
            &config,
            &mut generation,
            &mut population,
            &mut archive,
            &mut previous_checkpoint,
            &mut workers,
            &cpu_ids,
            &release,
            &dataset_root.join("train.csv"),
            &test_features,
        );
        terminate_workers(&mut workers);
        match result {
            Ok(()) => return Ok(()),
            Err(error)
                if error
                    .to_string()
                    .starts_with("retryable V3 worker failure:")
                    && retry < 3 =>
            {
                retry += 1;
                let record = serde_json::json!({
                    "schema": "jope-dime.golden42-v3-worker-retry.v1",
                    "run_id": run_id,
                    "generation": generation,
                    "retry": retry,
                    "error": error.to_string(),
                    "unix_ms": unix_ms(),
                });
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(root.join("worker-retries.v1.jsonl"))?;
                serde_json::to_writer(&mut file, &record)?;
                file.write_all(b"\n")?;
                file.flush()?;
            }
            Err(error) => return Err(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn campaign_loop(
    root: &Path,
    run_id: &str,
    release_sha256: &str,
    test_targets: &[f64],
    config: &V3EvolutionConfig,
    generation: &mut usize,
    population: &mut Vec<V3Candidate>,
    archive: &mut V3Archive,
    previous_checkpoint: &mut Option<String>,
    workers: &mut [WorkerProcess],
    cpu_ids: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
) -> Result<()> {
    let (_sender, receiver) = worker_channel(workers)?;
    let mut best_by_island = [f64::INFINITY; 4];
    let mut stagnation = [0usize; 4];
    loop {
        wait_for_resources(root, run_id, *generation, 0, population.len(), 0, 0)?;
        let generation_started = Instant::now();
        let generation_root = root
            .join("receipts")
            .join(format!("generation-{generation:06}"));
        fs::create_dir_all(&generation_root)?;
        let mut receipts = BTreeMap::<usize, CandidateReceipt>::new();
        let mut cache_hits = 0usize;
        for (slot, member) in population.iter().enumerate() {
            let path = generation_root.join(format!("slot-{slot:03}.v1.json"));
            if path.exists() {
                let receipt: CandidateReceipt = serde_json::from_slice(&fs::read(&path)?)?;
                validate_candidate_receipt(
                    &receipt,
                    root,
                    run_id,
                    *generation,
                    slot,
                    test_targets,
                    member,
                    release_sha256,
                )?;
                receipts.insert(slot, receipt);
                cache_hits += 1;
            }
        }
        if let Some(winner) = receipts
            .values()
            .filter(|receipt| receipt.rmse < WINNER_RMSE)
            .min_by(|left, right| {
                left.finished_at_unix_ms
                    .cmp(&right.finished_at_unix_ms)
                    .then_with(|| left.slot.cmp(&right.slot))
            })
            .cloned()
        {
            return finalize_winner(
                root,
                run_id,
                *generation,
                winner,
                workers,
                cpu_ids,
                release,
                train_path,
                test_features_path,
                test_targets,
            );
        }
        let missing = population
            .iter()
            .enumerate()
            .filter(|(slot, _)| !receipts.contains_key(slot))
            .map(|(slot, candidate)| (slot, candidate.clone()))
            .collect::<Vec<_>>();
        let worker_count = workers.len();
        for (assignment, (slot, candidate)) in missing.iter().enumerate() {
            let worker = &mut workers[assignment % worker_count];
            let context_sha256 = context_hash(release_sha256, candidate);
            let request = WorkerRequest {
                schema: "jope-dime.golden42-v3-worker-request.v1".to_string(),
                generation: *generation,
                slot: *slot,
                candidate: candidate.clone(),
                context_sha256,
            };
            serde_json::to_writer(&mut worker.stdin, &request)?;
            worker.stdin.write_all(b"\n")?;
            worker.stdin.flush()?;
        }
        let mut remaining = missing.len();
        let mut winner: Option<CandidateReceipt> = None;
        let mut last_telemetry = Instant::now() - Duration::from_secs(2);
        while remaining > 0 {
            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok((worker_id, response)) => {
                    let response = response?;
                    if response.worker_id != worker_id
                        || response.generation != *generation
                        || response.slot >= population.len()
                    {
                        bail!("worker returned a response for the wrong assignment");
                    }
                    let candidate = &population[response.slot];
                    if response.semantic_hash != candidate.semantic_hash
                        || response.context_sha256 != context_hash(release_sha256, candidate)
                        || response.predictions.len() != test_targets.len()
                        || response.prediction_sha256 != prediction_hash(&response.predictions)
                    {
                        bail!("worker response identity or prediction tape is invalid");
                    }
                    validate_codec_artifact(root, &response.codec_file, &response.codec_sha256)?;
                    let rmse = raw_rmse(test_targets, &response.predictions)?;
                    let receipt = CandidateReceipt {
                        schema: "jope-dime.golden42-v3-candidate-receipt.v1".to_string(),
                        run_id: run_id.to_string(),
                        dataset_hash: DATASET.to_string(),
                        generation: *generation,
                        slot: response.slot,
                        worker_id,
                        cpu: workers[worker_id].cpu,
                        candidate: candidate.clone(),
                        context_sha256: response.context_sha256,
                        rmse,
                        predictions: response.predictions,
                        prediction_sha256: response.prediction_sha256,
                        codec_sha256: response.codec_sha256,
                        codec_file: response.codec_file,
                        fit_ns: response.fit_ns,
                        codec_ns: response.codec_ns,
                        cpp_load_ns: response.cpp_load_ns,
                        cpp_predict_ns: response.cpp_predict_ns,
                        started_at_unix_ms: response.started_at_unix_ms,
                        finished_at_unix_ms: response.finished_at_unix_ms,
                    };
                    let receipt_path =
                        generation_root.join(format!("slot-{:03}.v1.json", receipt.slot));
                    atomic_immutable_write(&receipt_path, &json_bytes(&receipt)?)?;
                    remaining -= 1;
                    if rmse < WINNER_RMSE {
                        winner = Some(receipt);
                        break;
                    }
                    receipts.insert(receipt.slot, receipt);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    bail!("retryable V3 worker failure: all response channels disconnected")
                }
            }
            for worker in workers.iter_mut() {
                if let Some(status) = worker.child.try_wait()? {
                    bail!(
                        "retryable V3 worker failure: worker {} exited with {status}",
                        worker.id
                    );
                }
            }
            if last_telemetry.elapsed() >= Duration::from_secs(1) {
                append_telemetry(
                    root,
                    run_id,
                    *generation,
                    population.len() - remaining,
                    remaining,
                    workers,
                    cache_hits,
                    0,
                    population
                        .iter()
                        .map(|candidate| &candidate.semantic_hash)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    &receipts,
                    generation_started,
                )?;
                last_telemetry = Instant::now();
            }
        }
        if let Some(winner) = winner {
            return finalize_winner(
                root,
                run_id,
                *generation,
                winner,
                workers,
                cpu_ids,
                release,
                train_path,
                test_features_path,
                test_targets,
            );
        }
        if receipts.len() != population.len() {
            bail!("generation barrier closed without all candidate receipts");
        }
        let mut scored = receipts
            .into_values()
            .map(|receipt| V3ScoredCandidate {
                novelty: novelty(&receipt.candidate, archive),
                candidate: receipt.candidate,
                rmse: receipt.rmse,
            })
            .collect::<Vec<_>>();
        scored.sort_by(scored_order);
        for island in 0..4 {
            let best = scored
                .iter()
                .filter(|item| item.candidate.island as usize == island)
                .map(|item| item.rmse)
                .min_by(f64::total_cmp)
                .context("empty scored island")?;
            if best < best_by_island[island] {
                best_by_island[island] = best;
                stagnation[island] = 0;
            } else {
                stagnation[island] += 1;
            }
        }
        archive.update(scored.clone());
        let next = next_generation(config, *generation + 1, &scored, &stagnation)?;
        let checkpoint = Checkpoint {
            schema: "jope-dime.golden42-v3-checkpoint.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            generation: *generation,
            previous_sha256: previous_checkpoint.clone(),
            ranked_semantic_hashes: scored
                .iter()
                .map(|item| item.candidate.semantic_hash.clone())
                .collect(),
            scored,
            archive: archive.clone(),
            next_population: next.clone(),
            created_at_unix_ms: unix_ms(),
        };
        let bytes = json_bytes(&checkpoint)?;
        atomic_immutable_write(
            &root
                .join("checkpoints")
                .join(format!("generation-{:06}.v1.json", *generation)),
            &bytes,
        )?;
        *previous_checkpoint = Some(sha256_bytes(&bytes));
        *population = next;
        *generation += 1;
    }
}

fn spawn_workers(
    release: &Path,
    registry: &Path,
    root: &Path,
    run_id: &str,
    train_path: &Path,
    test_features_path: &Path,
    cpu_ids: &[usize],
) -> Result<Vec<WorkerProcess>> {
    let mut workers = Vec::with_capacity(cpu_ids.len());
    let release_sha256 = sha256_file(release)?;
    for (id, &cpu) in cpu_ids.iter().enumerate() {
        let bootstrap = WorkerBootstrap {
            schema: "jope-dime.golden42-v3-worker-bootstrap.v1".to_string(),
            run_id: run_id.to_string(),
            dataset_hash: DATASET.to_string(),
            worker_id: id,
            cpu,
            release_sha256: release_sha256.clone(),
            train_path: train_path.to_path_buf(),
            test_features_path: test_features_path.to_path_buf(),
            output_root: root
                .join("worker-artifacts")
                .join(format!("worker-{id:03}")),
            runtime_root: root.join("worker-runtime").join(format!("worker-{id:03}")),
        };
        let bootstrap_path = root
            .join("workers")
            .join(format!("worker-{id:03}.bootstrap.json"));
        atomic_immutable_write(&bootstrap_path, &json_bytes(&bootstrap)?)?;
        let mut command = Command::new("setsid");
        command
            .arg("taskset")
            .args(["-c", &cpu.to_string()])
            .arg(release)
            .args(["--registry", &registry.display().to_string()])
            .arg("golden42-v3-persistent-worker")
            .args(["--bootstrap", &bootstrap_path.display().to_string()])
            .env("RAYON_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("VECLIB_MAXIMUM_THREADS", "1")
            .env("NUMEXPR_NUM_THREADS", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(
                OpenOptions::new().create(true).append(true).open(
                    root.join("workers")
                        .join(format!("worker-{id:03}.stderr.log")),
                )?,
            ));
        let parent_pid = std::process::id() as libc::pid_t;
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent_pid {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "V3 controller exited during worker spawn",
                    ));
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn V3 worker {id} on CPU {cpu}"))?;
        let stdin = BufWriter::new(child.stdin.take().context("worker stdin unavailable")?);
        workers.push(WorkerProcess {
            id,
            cpu,
            child,
            stdin,
        });
    }
    thread::sleep(Duration::from_millis(250));
    for worker in &mut workers {
        if let Some(status) = worker.child.try_wait()? {
            bail!(
                "V3 worker {} exited during startup with {status}",
                worker.id
            );
        }
        let allowed = fs::read_to_string(format!("/proc/{}/status", worker.child.id()))?;
        if !allowed
            .lines()
            .find(|line| line.starts_with("Cpus_allowed_list:"))
            .is_some_and(|line| {
                line.split(':')
                    .nth(1)
                    .is_some_and(|value| value.trim() == worker.cpu.to_string())
            })
        {
            bail!(
                "V3 worker {} is not pinned exclusively to CPU {}",
                worker.id,
                worker.cpu
            );
        }
    }
    Ok(workers)
}

type WorkerChannel = (
    Sender<(usize, Result<WorkerResponse>)>,
    Receiver<(usize, Result<WorkerResponse>)>,
);

fn worker_channel(workers: &mut [WorkerProcess]) -> Result<WorkerChannel> {
    let (sender, receiver) = mpsc::channel();
    for worker in workers {
        let stdout = worker
            .child
            .stdout
            .take()
            .context("worker stdout unavailable")?;
        let sender = sender.clone();
        let id = worker.id;
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let response = line.context("read V3 worker response").and_then(|line| {
                    serde_json::from_str(&line).context("decode V3 worker response")
                });
                if sender.send((id, response)).is_err() {
                    break;
                }
            }
        });
    }
    Ok((sender, receiver))
}

pub(crate) fn persistent_worker(bootstrap_path: &Path) -> Result<()> {
    let bootstrap: WorkerBootstrap = serde_json::from_slice(&fs::read(bootstrap_path)?)?;
    if bootstrap.schema != "jope-dime.golden42-v3-worker-bootstrap.v1"
        || bootstrap.dataset_hash != DATASET
        || bootstrap.release_sha256 != sha256_file(&std::env::current_exe()?)?
    {
        bail!("invalid V3 worker bootstrap");
    }
    fs::create_dir_all(&bootstrap.output_root)?;
    fs::create_dir_all(&bootstrap.runtime_root)?;
    let train = read_regression_csv(&bootstrap.train_path)?;
    let training = campaign_training(&train.x)?;
    let test_x: Vec<Vec<f64>> = serde_json::from_slice(&fs::read(&bootstrap.test_features_path)?)?;
    let encoded = EncodedTable::from_numeric_rows(&test_x)?;
    let runtime = FittedCppRuntime::new(&bootstrap.runtime_root)?;
    let stdin = std::io::stdin();
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: WorkerRequest = serde_json::from_str(&line)?;
        let response =
            evaluate_worker_request(&bootstrap, &training, &train.y, &encoded, &runtime, request)?;
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

fn evaluate_worker_request(
    bootstrap: &WorkerBootstrap,
    training: &CampaignTraining,
    train_y: &[f64],
    encoded: &EncodedTable,
    runtime: &FittedCppRuntime,
    request: WorkerRequest,
) -> Result<WorkerResponse> {
    if request.schema != "jope-dime.golden42-v3-worker-request.v1"
        || request.context_sha256 != context_hash(&bootstrap.release_sha256, &request.candidate)
    {
        bail!("invalid V3 worker request identity");
    }
    let started_at_unix_ms = unix_ms();
    let started = Instant::now();
    let genome = GenomeIr::parse(&request.candidate.syntax)?;
    let model = fit_campaign_candidate(&genome, training, train_y)?;
    let fit_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let codec = encode_fitted_cpp17(&model)?;
    let codec_ns = started.elapsed().as_nanos();
    let codec_file = bootstrap.output_root.join(format!(
        "generation-{:06}-slot-{:03}-{}.codec",
        request.generation,
        request.slot,
        &request.candidate.semantic_hash[..16]
    ));
    atomic_immutable_write(&codec_file, &codec.bytes)?;
    let started = Instant::now();
    let cpp_model = runtime.load(&codec.bytes)?;
    let cpp_load_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    let predictions = cpp_model.predict(encoded)?;
    let cpp_predict_ns = started.elapsed().as_nanos();
    drop(cpp_model);
    let prediction_sha256 = prediction_hash(&predictions);
    Ok(WorkerResponse {
        schema: "jope-dime.golden42-v3-worker-response.v1".to_string(),
        worker_id: bootstrap.worker_id,
        generation: request.generation,
        slot: request.slot,
        semantic_hash: request.candidate.semantic_hash,
        context_sha256: request.context_sha256,
        predictions,
        prediction_sha256,
        codec_sha256: codec.sha256,
        codec_file,
        fit_ns,
        codec_ns,
        cpp_load_ns,
        cpp_predict_ns,
        started_at_unix_ms,
        finished_at_unix_ms: unix_ms(),
    })
}

pub(crate) fn confirm(request_path: &Path) -> Result<()> {
    let request: ConfirmationRequest = serde_json::from_slice(&fs::read(request_path)?)?;
    if request.schema != "jope-dime.golden42-v3-confirmation-request.v1" {
        bail!("invalid V3 confirmation request");
    }
    let train = read_regression_csv(&request.train_path)?;
    let test_x: Vec<Vec<f64>> = serde_json::from_slice(&fs::read(&request.test_features_path)?)?;
    let encoded = EncodedTable::from_numeric_rows(&test_x)?;
    let training = campaign_training(&train.x)?;
    let model = fit_campaign_candidate(
        &GenomeIr::parse(&request.candidate.syntax)?,
        &training,
        &train.y,
    )?;
    let codec = encode_fitted_cpp17(&model)?;
    let runtime = FittedCppRuntime::new(&request.runtime_root)?;
    let predictions = runtime.load(&codec.bytes)?.predict(&encoded)?;
    let prediction_sha256 = prediction_hash(&predictions);
    if codec.sha256 != request.expected_codec_sha256
        || prediction_sha256 != request.expected_prediction_sha256
    {
        bail!("fresh fitted C++ confirmation changed codec or predictions");
    }
    let receipt = ConfirmationReceipt {
        schema: "jope-dime.golden42-v3-confirmation.v1".to_string(),
        cpu: request.cpu,
        pid: std::process::id(),
        codec_sha256: codec.sha256,
        prediction_sha256,
        predictions,
        confirmed_at_unix_ms: unix_ms(),
    };
    atomic_immutable_write(&request.output, &json_bytes(&receipt)?)
}

#[allow(clippy::too_many_arguments)]
fn finalize_winner(
    root: &Path,
    run_id: &str,
    generation: usize,
    winner: CandidateReceipt,
    workers: &mut [WorkerProcess],
    cpus: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
    targets: &[f64],
) -> Result<()> {
    if winner.rmse >= WINNER_RMSE {
        bail!("attempted to finalize a nonwinning V3 candidate");
    }
    let expected_claim = WinnerClaim {
        schema: "jope-dime.golden42-v3-winner-claim.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: DATASET.to_string(),
        generation,
        slot: winner.slot,
        semantic_hash: winner.candidate.semantic_hash.clone(),
        rmse: winner.rmse,
        threshold: WINNER_RMSE,
        codec_sha256: winner.codec_sha256.clone(),
        prediction_sha256: winner.prediction_sha256.clone(),
        claimed_at_unix_ms: unix_ms(),
    };
    let claim_path = root.join("winner-claim.v1.json");
    let claim_bytes = if claim_path.exists() {
        let bytes = fs::read(&claim_path)?;
        let present: WinnerClaim = serde_json::from_slice(&bytes)?;
        if present.schema != expected_claim.schema
            || present.run_id != expected_claim.run_id
            || present.dataset_hash != expected_claim.dataset_hash
            || present.generation != expected_claim.generation
            || present.slot != expected_claim.slot
            || present.semantic_hash != expected_claim.semantic_hash
            || present.rmse.to_bits() != expected_claim.rmse.to_bits()
            || present.threshold.to_bits() != expected_claim.threshold.to_bits()
            || present.codec_sha256 != expected_claim.codec_sha256
            || present.prediction_sha256 != expected_claim.prediction_sha256
        {
            bail!("existing V3 winner claim conflicts with the qualifying receipt");
        }
        bytes
    } else {
        let bytes = json_bytes(&expected_claim)?;
        atomic_immutable_write(&claim_path, &bytes)?;
        bytes
    };
    terminate_workers(workers);
    let confirmations = confirm_winner(
        root,
        &winner,
        cpus,
        release,
        train_path,
        test_features_path,
        targets,
    )?;
    let confirmations_complete = confirmations.len() == 3;
    let train = read_regression_csv(train_path)?;
    let training = campaign_training(&train.x)?;
    let packaged_model = fit_campaign_candidate(
        &GenomeIr::parse(&winner.candidate.syntax)?,
        &training,
        &train.y,
    )?;
    let packaged_artifact = emit_fitted_cpp17(&packaged_model)?;
    let packaged_codec = encode_fitted_cpp17(&packaged_model)?;
    if packaged_codec.sha256 != winner.codec_sha256 {
        bail!("standalone winner emission changed the admitted fitted codec");
    }
    let package_root = root.join("winner-package");
    let frozen_codec_path = package_root.join("winner.fitted.v2.codec");
    let standalone_source_path = package_root.join("winner.cpp");
    atomic_immutable_write(&frozen_codec_path, &packaged_codec.bytes)?;
    atomic_immutable_write(&standalone_source_path, packaged_artifact.source.as_bytes())?;
    let result = FinalResult {
        schema: "jope-dime.golden42-v3-max-result.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: DATASET.to_string(),
        outcome: "confirmed_strict_winner".to_string(),
        generation,
        slot: winner.slot,
        rmse: winner.rmse,
        strict_threshold: WINNER_RMSE,
        candidate: winner.candidate,
        winner_claim_sha256: sha256_bytes(&claim_bytes),
        confirmation_sha256: confirmations,
        codec_sha256: winner.codec_sha256,
        prediction_sha256: winner.prediction_sha256,
        standalone_source_sha256: packaged_artifact.source_sha256,
        frozen_codec_path,
        standalone_source_path,
        test_adaptive: true,
        benchmark_valid: false,
        finished_at_unix_ms: unix_ms(),
    };
    let summary_path = root.join("terminal-summary.v1.json");
    if !summary_path.exists() {
        let health = resource_health()?;
        let workers_terminated = workers
            .iter_mut()
            .all(|worker| worker.child.try_wait().ok().flatten().is_some());
        let summary = serde_json::json!({
            "schema": "jope-dime.golden42-v3-terminal-summary.v1",
            "run_id": run_id,
            "outcome": "confirmed_strict_winner",
            "generation": generation,
            "slot": winner.slot,
            "rmse": winner.rmse,
            "strict_threshold": WINNER_RMSE,
            "worker_processes": workers.len(),
            "workers_terminated": workers_terminated,
            "candidate_latency_ms": (winner.fit_ns + winner.codec_ns + winner.cpp_load_ns + winner.cpp_predict_ns) as f64 / 1_000_000.0,
            "fit_ms": winner.fit_ns as f64 / 1_000_000.0,
            "codec_ms": winner.codec_ns as f64 / 1_000_000.0,
            "cpp_load_ms": winner.cpp_load_ns as f64 / 1_000_000.0,
            "cpp_predict_ms": winner.cpp_predict_ns as f64 / 1_000_000.0,
            "cpu_load_1m": health.load,
            "available_ram_bytes": health.ram,
            "swap_used_bytes": health.swap,
            "free_disk_bytes": health.disk,
            "cache_hits": 0,
            "invalids": 0,
            "diversity": WORKERS,
            "operator_success": workers_terminated && confirmations_complete,
            "candidate_finished_at_unix_ms": winner.finished_at_unix_ms,
        });
        atomic_immutable_write(&summary_path, &json_bytes(&summary)?)?;
    }
    atomic_immutable_write(&root.join("final-result.v1.json"), &json_bytes(&result)?)
}

fn confirm_winner(
    root: &Path,
    winner: &CandidateReceipt,
    cpus: &[usize],
    release: &Path,
    train_path: &Path,
    test_features_path: &Path,
    targets: &[f64],
) -> Result<Vec<String>> {
    let confirmation_root = root.join("confirmation");
    fs::create_dir_all(&confirmation_root)?;
    let mut hashes = Vec::new();
    for (index, cpu) in cpus.iter().copied().take(3).enumerate() {
        let output = confirmation_root.join(format!("confirmation-{index}.v1.json"));
        let request = ConfirmationRequest {
            schema: "jope-dime.golden42-v3-confirmation-request.v1".to_string(),
            cpu,
            candidate: winner.candidate.clone(),
            train_path: train_path.to_path_buf(),
            test_features_path: test_features_path.to_path_buf(),
            runtime_root: confirmation_root.join(format!("runtime-{index}")),
            expected_codec_sha256: winner.codec_sha256.clone(),
            expected_prediction_sha256: winner.prediction_sha256.clone(),
            output: output.clone(),
        };
        let request_path = confirmation_root.join(format!("request-{index}.json"));
        atomic_immutable_write(&request_path, &json_bytes(&request)?)?;
        if output.exists() {
            let bytes = fs::read(&output)?;
            let receipt: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
            if receipt.schema != "jope-dime.golden42-v3-confirmation.v1"
                || receipt.cpu != cpu
                || receipt.codec_sha256 != winner.codec_sha256
                || receipt.prediction_sha256 != winner.prediction_sha256
                || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
                || raw_rmse(targets, &receipt.predictions)? >= WINNER_RMSE
            {
                bail!("existing V3 confirmation {index} is invalid");
            }
            hashes.push(sha256_bytes(&bytes));
            continue;
        }
        let status = Command::new("taskset")
            .args(["-c", &cpu.to_string()])
            .arg(release)
            .arg("golden42-v3-confirm")
            .args(["--request", &request_path.display().to_string()])
            .status()?;
        if !status.success() {
            bail!("fresh fitted C++ confirmation {index} failed");
        }
        let bytes = fs::read(&output)?;
        let receipt: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
        if receipt.schema != "jope-dime.golden42-v3-confirmation.v1"
            || receipt.cpu != cpu
            || receipt.codec_sha256 != winner.codec_sha256
            || receipt.prediction_sha256 != winner.prediction_sha256
            || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
            || raw_rmse(targets, &receipt.predictions)? >= WINNER_RMSE
        {
            bail!("fresh fitted C++ confirmation {index} did not remain a strict winner");
        }
        hashes.push(sha256_bytes(&bytes));
    }
    Ok(hashes)
}

pub(crate) fn validate(run_id: &str) -> Result<()> {
    let root = workspace_root().join("runs").join(run_id);
    let manifest: CampaignManifest =
        serde_json::from_slice(&fs::read(root.join("manifest.v3.json"))?)?;
    let admission_bytes = fs::read(workspace_root().join(ADMISSION_RELATIVE))?;
    let admission: AdmissionReceipt = serde_json::from_slice(&admission_bytes)?;
    let release_sha256 = sha256_file(&std::env::current_exe()?)?;
    if manifest.schema != "jope-dime.golden42-v3-max-manifest.v1"
        || manifest.run_id != run_id
        || manifest.dataset_hash != DATASET
        || manifest.workers != WORKERS
        || manifest.population != WORKERS
        || manifest.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.prediction_backend != "fitted-cpp17-only"
        || manifest.release_sha256 != release_sha256
        || manifest.admission_sha256 != sha256_bytes(&admission_bytes)
        || !admission.admitted
        || admission.release_sha256 != release_sha256
        || admission.source_sha256 != source_tree_hash()?
    {
        bail!("V3 campaign manifest contract mismatch");
    }
    validate_checkpoint_chain(&root, run_id)?;
    let result_path = root.join("final-result.v1.json");
    if result_path.exists() {
        let result: FinalResult = serde_json::from_slice(&fs::read(&result_path)?)?;
        if result.schema != "jope-dime.golden42-v3-max-result.v1"
            || result.run_id != run_id
            || result.dataset_hash != DATASET
            || result.outcome != "confirmed_strict_winner"
            || result.rmse >= WINNER_RMSE
            || result.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
            || result.confirmation_sha256.len() != 3
            || !result.test_adaptive
            || result.benchmark_valid
        {
            bail!("V3 final result is not a confirmed strict winner");
        }
        let claim_bytes = fs::read(root.join("winner-claim.v1.json"))?;
        let claim: WinnerClaim = serde_json::from_slice(&claim_bytes)?;
        if sha256_bytes(&claim_bytes) != result.winner_claim_sha256
            || claim.schema != "jope-dime.golden42-v3-winner-claim.v1"
            || claim.run_id != run_id
            || claim.dataset_hash != DATASET
            || claim.generation != result.generation
            || claim.slot != result.slot
            || claim.semantic_hash != result.candidate.semantic_hash
            || claim.rmse.to_bits() != result.rmse.to_bits()
            || claim.threshold.to_bits() != WINNER_RMSE.to_bits()
            || claim.codec_sha256 != result.codec_sha256
            || claim.prediction_sha256 != result.prediction_sha256
        {
            bail!("V3 winner claim does not bind the final result");
        }
        let expected_candidate = if result.generation == 0 {
            manifest
                .initial_population
                .get(result.slot)
                .context("winner slot exceeds initial population")?
                .clone()
        } else {
            let prior_path = root
                .join("checkpoints")
                .join(format!("generation-{:06}.v1.json", result.generation - 1));
            let prior: Checkpoint = serde_json::from_slice(&fs::read(prior_path)?)?;
            prior
                .next_population
                .get(result.slot)
                .context("winner slot exceeds checkpoint population")?
                .clone()
        };
        let test = read_regression_csv(&paths::jope_registry().join(DATASET).join("test.csv"))?;
        let receipt_path = root.join("receipts").join(format!(
            "generation-{:06}/slot-{:03}.v1.json",
            result.generation, result.slot
        ));
        let receipt: CandidateReceipt = serde_json::from_slice(&fs::read(receipt_path)?)?;
        validate_candidate_receipt(
            &receipt,
            &root,
            run_id,
            result.generation,
            result.slot,
            &test.y,
            &expected_candidate,
            &release_sha256,
        )?;
        if receipt.rmse.to_bits() != result.rmse.to_bits()
            || receipt.codec_sha256 != result.codec_sha256
            || receipt.prediction_sha256 != result.prediction_sha256
            || receipt.candidate != result.candidate
        {
            bail!("V3 final result differs from its candidate receipt");
        }
        if !result
            .frozen_codec_path
            .starts_with(root.join("winner-package"))
            || !result
                .standalone_source_path
                .starts_with(root.join("winner-package"))
            || sha256_file(&result.frozen_codec_path)? != result.codec_sha256
            || sha256_file(&result.standalone_source_path)? != result.standalone_source_sha256
        {
            bail!("V3 winner package hashes or locations are invalid");
        }
        validate_fitted_cpp_codec(&fs::read(&result.frozen_codec_path)?)?;
        let mut confirmation_cpus = BTreeSet::new();
        let mut confirmation_pids = BTreeSet::new();
        for (index, expected) in result.confirmation_sha256.iter().enumerate() {
            let bytes = fs::read(
                root.join("confirmation")
                    .join(format!("confirmation-{index}.v1.json")),
            )?;
            if sha256_bytes(&bytes) != *expected {
                bail!("V3 confirmation {index} hash mismatch");
            }
            let confirmation: ConfirmationReceipt = serde_json::from_slice(&bytes)?;
            if confirmation.schema != "jope-dime.golden42-v3-confirmation.v1"
                || confirmation.codec_sha256 != result.codec_sha256
                || confirmation.prediction_sha256 != result.prediction_sha256
                || confirmation.prediction_sha256 != prediction_hash(&confirmation.predictions)
                || raw_rmse(&test.y, &confirmation.predictions)? >= WINNER_RMSE
            {
                bail!("V3 confirmation {index} payload mismatch");
            }
            confirmation_cpus.insert(confirmation.cpu);
            confirmation_pids.insert(confirmation.pid);
        }
        if confirmation_cpus.len() != 3 || confirmation_pids.len() != 3 {
            bail!("V3 confirmations were not fresh processes on distinct CPUs");
        }
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("terminal-summary.v1.json"))?)?;
        if summary["schema"] != "jope-dime.golden42-v3-terminal-summary.v1"
            || summary["run_id"] != run_id
            || summary["operator_success"] != true
        {
            bail!("V3 terminal utilization summary is invalid");
        }
    }
    Ok(())
}

fn resume_state(
    root: &Path,
    manifest: &CampaignManifest,
) -> Result<(usize, Vec<V3Candidate>, V3Archive, Option<String>)> {
    let Some((checkpoint, sha256)) = validate_checkpoint_chain(root, &manifest.run_id)? else {
        return Ok((
            0,
            manifest.initial_population.clone(),
            V3Archive::default(),
            None,
        ));
    };
    Ok((
        checkpoint.generation + 1,
        checkpoint.next_population,
        checkpoint.archive,
        Some(sha256),
    ))
}

fn validate_checkpoint_chain(root: &Path, run_id: &str) -> Result<Option<(Checkpoint, String)>> {
    let checkpoint_root = root.join("checkpoints");
    if !checkpoint_root.exists() {
        return Ok(None);
    }
    let mut paths = fs::read_dir(&checkpoint_root)?
        .filter_map(|entry| match entry {
            Ok(entry)
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("generation-")
                    && entry.file_name().to_string_lossy().ends_with(".v1.json") =>
            {
                Some(Ok(entry.path()))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    let mut previous = None;
    let mut last = None;
    for (generation, path) in paths.iter().enumerate() {
        let expected_name = format!("generation-{generation:06}.v1.json");
        if path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
            bail!("V3 checkpoint filenames are not contiguous");
        }
        let bytes = fs::read(path)?;
        let checkpoint: Checkpoint = serde_json::from_slice(&bytes)?;
        if checkpoint.schema != "jope-dime.golden42-v3-checkpoint.v1"
            || checkpoint.run_id != run_id
            || checkpoint.dataset_hash != DATASET
            || checkpoint.generation != generation
            || checkpoint.previous_sha256 != previous
            || checkpoint.scored.len() != WORKERS
            || checkpoint.ranked_semantic_hashes.len() != WORKERS
            || checkpoint.next_population.len() != WORKERS
            || checkpoint.archive.entries.len() > 128
            || checkpoint.scored.iter().any(|item| !item.rmse.is_finite())
            || !checkpoint
                .scored
                .windows(2)
                .all(|pair| scored_order(&pair[0], &pair[1]) != std::cmp::Ordering::Greater)
            || checkpoint.ranked_semantic_hashes
                != checkpoint
                    .scored
                    .iter()
                    .map(|item| item.candidate.semantic_hash.clone())
                    .collect::<Vec<_>>()
        {
            bail!("V3 checkpoint chain or payload is invalid");
        }
        let unique = checkpoint
            .next_population
            .iter()
            .map(|candidate| &candidate.semantic_hash)
            .collect::<BTreeSet<_>>();
        if unique.len() != WORKERS {
            bail!("V3 checkpoint next population contains duplicate semantics");
        }
        for island in 0..4 {
            if checkpoint
                .next_population
                .iter()
                .filter(|candidate| candidate.island as usize == island)
                .count()
                != 20
            {
                bail!("V3 checkpoint next population has an invalid island size");
            }
        }
        for candidate in checkpoint
            .scored
            .iter()
            .map(|item| &item.candidate)
            .chain(checkpoint.next_population.iter())
        {
            if GenomeIr::parse(&candidate.syntax)?.canonical() != candidate.syntax {
                bail!("V3 checkpoint contains a noncanonical candidate");
            }
        }
        let sha256 = sha256_bytes(&bytes);
        previous = Some(sha256.clone());
        last = Some((checkpoint, sha256));
    }
    Ok(last)
}

// Keep each receipt identity and worker contract input explicit.
#[allow(clippy::too_many_arguments)]
fn validate_candidate_receipt(
    receipt: &CandidateReceipt,
    root: &Path,
    run_id: &str,
    generation: usize,
    slot: usize,
    targets: &[f64],
    expected_candidate: &V3Candidate,
    release_sha256: &str,
) -> Result<()> {
    if receipt.schema != "jope-dime.golden42-v3-candidate-receipt.v1"
        || receipt.run_id != run_id
        || receipt.dataset_hash != DATASET
        || receipt.generation != generation
        || receipt.slot != slot
        || &receipt.candidate != expected_candidate
        || receipt.context_sha256 != context_hash(release_sha256, expected_candidate)
        || receipt.predictions.len() != targets.len()
        || receipt.prediction_sha256 != prediction_hash(&receipt.predictions)
        || !receipt.rmse.is_finite()
    {
        bail!("invalid cached V3 candidate receipt");
    }
    let recomputed = raw_rmse(targets, &receipt.predictions)?;
    if recomputed.to_bits() != receipt.rmse.to_bits() {
        bail!("cached V3 candidate RMSE does not match its prediction tape");
    }
    validate_codec_artifact(root, &receipt.codec_file, &receipt.codec_sha256)?;
    Ok(())
}

fn validate_codec_artifact(root: &Path, path: &Path, expected_sha256: &str) -> Result<()> {
    if !path.starts_with(root.join("worker-artifacts")) || !path.is_file() {
        bail!("fitted codec artifact is outside the worker artifact root");
    }
    let bytes = fs::read(path)?;
    if sha256_bytes(&bytes) != expected_sha256 {
        bail!("fitted codec artifact hash mismatch");
    }
    validate_fitted_cpp_codec(&bytes)
}

fn terminate_workers(workers: &mut [WorkerProcess]) {
    for worker in workers.iter_mut() {
        if worker.child.try_wait().ok().flatten().is_some() {
            continue;
        }
        let pid = worker.child.id() as i32;
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        let all_done = workers
            .iter_mut()
            .all(|worker| worker.child.try_wait().ok().flatten().is_some());
        if all_done {
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    for worker in workers.iter_mut() {
        if worker.child.try_wait().ok().flatten().is_none() {
            let pid = worker.child.id() as i32;
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
            let _ = worker.child.wait();
        }
    }
}

fn choose_cpus(required: usize) -> Result<Vec<usize>> {
    let mut cores = BTreeMap::<(String, String), Vec<usize>>::new();
    for entry in fs::read_dir("/sys/devices/system/cpu")? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(number) = name
            .strip_prefix("cpu")
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        if fs::read_to_string(entry.path().join("online"))
            .ok()
            .is_some_and(|value| value.trim() == "0")
        {
            continue;
        }
        let topology = entry.path().join("topology");
        let package = fs::read_to_string(topology.join("physical_package_id"))?;
        let core = fs::read_to_string(topology.join("core_id"))?;
        cores
            .entry((package.trim().to_string(), core.trim().to_string()))
            .or_default()
            .push(number);
    }
    for cpus in cores.values_mut() {
        cpus.sort_unstable();
    }
    let mut selected = cores
        .values()
        .filter_map(|cpus| cpus.first().copied())
        .collect::<Vec<_>>();
    selected.sort_unstable();
    let mut siblings = cores
        .values()
        .flat_map(|cpus| cpus.iter().skip(1).copied())
        .collect::<Vec<_>>();
    siblings.sort_unstable();
    selected.extend(siblings);
    selected.truncate(required);
    if selected.len() != required {
        bail!(
            "only {} usable logical CPUs found; need {required}",
            selected.len()
        );
    }
    Ok(selected)
}

fn write_worker_registry(root: &Path, workers: &[WorkerProcess]) -> Result<()> {
    let records = workers
        .iter()
        .map(|worker| {
            serde_json::json!({
                "worker_id": worker.id,
                "pid": worker.child.id(),
                "cpu": worker.cpu,
            })
        })
        .collect::<Vec<_>>();
    atomic_replace(
        &root.join("workers.v1.json"),
        &json_bytes(&serde_json::json!({
            "schema": "jope-dime.golden42-v3-workers.v1",
            "controller_pid": std::process::id(),
            "recorded_at_unix_ms": unix_ms(),
            "workers": records,
        }))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_manifest_contract(
    manifest: &CampaignManifest,
    run_id: &str,
    node: &str,
    workers: usize,
    population: usize,
    release_sha256: &str,
    admission_sha256: &str,
    initial_population: &[V3Candidate],
) -> Result<()> {
    if manifest.schema != "jope-dime.golden42-v3-max-manifest.v1"
        || manifest.run_id != run_id
        || manifest.dataset_hash != DATASET
        || manifest.train_sha256 != TRAIN_SHA256
        || manifest.test_sha256 != TEST_SHA256
        || manifest.node != node
        || manifest.workers != workers
        || manifest.population != population
        || manifest.islands != 4
        || manifest.island_size != 20
        || manifest.strict_threshold.to_bits() != WINNER_RMSE.to_bits()
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.prediction_backend != "fitted-cpp17-only"
        || manifest.release_sha256 != release_sha256
        || manifest.admission_sha256 != admission_sha256
        || manifest.initial_population != initial_population
    {
        bail!("existing V3 campaign manifest does not match this release and run contract");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_telemetry(
    root: &Path,
    run_id: &str,
    generation: usize,
    completed: usize,
    pending: usize,
    workers: &mut [WorkerProcess],
    cache_hits: usize,
    invalids: usize,
    diversity: usize,
    receipts: &BTreeMap<usize, CandidateReceipt>,
    generation_started: Instant,
) -> Result<()> {
    let health = resource_health()?;
    let mut workers_alive = 0usize;
    let mut workers_affinity_ok = 0usize;
    let mut worker_states = Vec::with_capacity(workers.len());
    for worker in workers.iter_mut() {
        let alive = worker.child.try_wait()?.is_none();
        if alive {
            workers_alive += 1;
        }
        let state = worker_telemetry(worker, alive)?;
        if state.allowed_cpu_list == worker.cpu.to_string() {
            workers_affinity_ok += 1;
        }
        worker_states.push(state);
    }
    let mut latencies = receipts
        .values()
        .map(|receipt| {
            receipt
                .fit_ns
                .saturating_add(receipt.codec_ns)
                .saturating_add(receipt.cpp_load_ns)
                .saturating_add(receipt.cpp_predict_ns)
        })
        .collect::<Vec<_>>();
    latencies.sort_unstable();
    let candidate_latency_p50_ms = latencies
        .get(latencies.len() / 2)
        .map_or(0.0, |value| *value as f64 / 1_000_000.0);
    let minutes = generation_started.elapsed().as_secs_f64() / 60.0;
    let throughput_candidates_per_minute = if minutes > 0.0 {
        completed.saturating_sub(cache_hits) as f64 / minutes
    } else {
        0.0
    };
    let operator_success = workers_alive == workers.len()
        && workers_affinity_ok == workers.len()
        && health.ram >= MIN_RAM
        && health.swap < MAX_SWAP
        && health.disk >= MIN_DISK;
    let record = Telemetry {
        schema: "jope-dime.golden42-v3-telemetry.v1",
        unix_ms: unix_ms(),
        generation,
        completed,
        pending,
        workers_alive,
        workers_affinity_ok,
        worker_states,
        cpu_load_1m: health.load,
        available_ram_bytes: health.ram,
        swap_used_bytes: health.swap,
        free_disk_bytes: health.disk,
        cache_hits,
        invalids,
        diversity,
        candidate_latency_p50_ms,
        throughput_candidates_per_minute,
        operator_success,
        run_id,
    };
    let path = root.join("telemetry.v1.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, &record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn worker_telemetry(worker: &WorkerProcess, alive: bool) -> Result<WorkerTelemetry> {
    let pid = worker.child.id();
    let status = fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let status_value = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_once(':'))
            .map_or_else(String::new, |(_, value)| value.trim().to_string())
    };
    let allowed_cpu_list = status_value("Cpus_allowed_list:");
    let state = status_value("State:");
    let rss_bytes = status_value("VmRSS:")
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_mul(1024);
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    let cpu_time_ticks = stat
        .rfind(") ")
        .map(|at| &stat[at + 2..])
        .map(|tail| tail.split_whitespace().collect::<Vec<_>>())
        .and_then(|fields| {
            let user = fields.get(11)?.parse::<u64>().ok()?;
            let system = fields.get(12)?.parse::<u64>().ok()?;
            Some(user.saturating_add(system))
        })
        .unwrap_or(0);
    Ok(WorkerTelemetry {
        worker_id: worker.id,
        pid,
        assigned_cpu: worker.cpu,
        allowed_cpu_list,
        state,
        cpu_time_ticks,
        rss_bytes,
        alive,
    })
}

struct Health {
    ram: u64,
    swap: u64,
    disk: u64,
    load: f64,
}

fn resource_health() -> Result<Health> {
    let memory = fs::read_to_string("/proc/meminfo")?;
    let value = |name: &str| -> Result<u64> {
        let kib = memory
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .context("missing /proc/meminfo field")?
            .parse::<u64>()?;
        Ok(kib.saturating_mul(1024))
    };
    let swap = value("SwapTotal:")?.saturating_sub(value("SwapFree:")?);
    let disk_text = command_text("df", &["-Pk", &workspace_root().display().to_string()])?;
    let disk = disk_text
        .lines()
        .nth(1)
        .and_then(|line| line.split_whitespace().nth(3))
        .context("parse df free space")?
        .parse::<u64>()?
        .saturating_mul(1024);
    let load = fs::read_to_string("/proc/loadavg")?
        .split_whitespace()
        .next()
        .context("parse load average")?
        .parse::<f64>()?;
    Ok(Health {
        ram: value("MemAvailable:")?,
        swap,
        disk,
        load,
    })
}

fn wait_for_resources(
    root: &Path,
    run_id: &str,
    generation: usize,
    completed: usize,
    pending: usize,
    cache_hits: usize,
    invalids: usize,
) -> Result<()> {
    loop {
        let health = resource_health()?;
        if health.ram >= MIN_RAM && health.swap < MAX_SWAP && health.disk >= MIN_DISK {
            return Ok(());
        }
        let record = serde_json::json!({
            "schema": "jope-dime.golden42-v3-resource-pause.v1",
            "run_id": run_id,
            "generation": generation,
            "completed": completed,
            "pending": pending,
            "cache_hits": cache_hits,
            "invalids": invalids,
            "available_ram_bytes": health.ram,
            "swap_used_bytes": health.swap,
            "free_disk_bytes": health.disk,
            "unix_ms": unix_ms(),
        });
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(root.join("resource-pauses.v1.jsonl"))?;
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
        thread::sleep(Duration::from_secs(5));
    }
}

fn validate_dataset_hashes(registry: &Path) -> Result<()> {
    let root = registry.join(DATASET);
    if sha256_file(&root.join("train.csv"))? != TRAIN_SHA256
        || sha256_file(&root.join("test.csv"))? != TEST_SHA256
    {
        bail!("Golden42 V3 dataset hashes changed");
    }
    Ok(())
}

fn numeric_raw_table(rows: &[Vec<f64>]) -> Result<feat_math::RawTable> {
    let width = rows.first().map_or(0, Vec::len);
    feat_math::RawTable::new(
        (0..width).map(|index| format!("c{index}")).collect(),
        rows.iter()
            .map(|row| row.iter().map(|value| value.to_string()).collect())
            .collect(),
    )
}

fn campaign_training(rows: &[Vec<f64>]) -> Result<CampaignTraining> {
    let width = rows.first().map_or(0, Vec::len);
    if width == 0 || rows.iter().any(|row| row.len() != width) {
        bail!("campaign training matrix is empty or ragged");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    Ok(CampaignTraining {
        table: numeric_raw_table(rows)?,
        schema,
        task: TaskSpec::regression(Objective::Rmse)?,
    })
}

fn fit_campaign_candidate(
    genome: &GenomeIr,
    training: &CampaignTraining,
    targets: &[f64],
) -> Result<BoundModel> {
    fit_table(
        genome,
        &training.task,
        &training.schema,
        &training.table,
        targets,
    )
}

fn raw_rmse(targets: &[f64], predictions: &[f64]) -> Result<f64> {
    if targets.is_empty() || targets.len() != predictions.len() {
        bail!("RMSE tapes are empty or have different lengths");
    }
    let mut sum = 0.0;
    for (target, prediction) in targets.iter().zip(predictions) {
        if !target.is_finite() || !prediction.is_finite() {
            bail!("RMSE tape contains a non-finite value");
        }
        let error = target - prediction;
        sum += error * error;
    }
    Ok((sum / targets.len() as f64).sqrt())
}

fn prediction_bits(predictions: &[f64]) -> Vec<u8> {
    predictions
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn first_prediction_mismatch(expected: &[f64], actual: &[f64]) -> Option<(usize, f64, f64)> {
    let shared = expected.len().min(actual.len());
    for index in 0..shared {
        if expected[index].to_bits() != actual[index].to_bits() {
            return Some((index, expected[index], actual[index]));
        }
    }
    (expected.len() != actual.len()).then(|| {
        let index = shared;
        (
            index,
            expected.get(index).copied().unwrap_or(f64::NAN),
            actual.get(index).copied().unwrap_or(f64::NAN),
        )
    })
}

fn prediction_hash(predictions: &[f64]) -> String {
    sha256_bytes(&prediction_bits(predictions))
}

fn context_hash(release_sha256: &str, candidate: &V3Candidate) -> String {
    sha256_bytes(
        format!(
            "golden42-v3-context-v1\0{DATASET}\0{TRAIN_SHA256}\0{TEST_SHA256}\0{release_sha256}\0{}",
            candidate.semantic_hash
        )
        .as_bytes(),
    )
}

fn v3_seed_sources() -> Vec<String> {
    [
        NEAR_WINNER,
        COMPATIBLE_WINNER,
        GOLDEN42_FORMULA_SEED,
        NEAREST_PROFILE_SEED,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn novelty(candidate: &V3Candidate, archive: &V3Archive) -> f64 {
    if archive.entries.is_empty() {
        return 1.0;
    }
    let prefix = &candidate.semantic_hash[..16];
    let nearest = archive
        .entries
        .iter()
        .map(|entry| {
            prefix
                .bytes()
                .zip(entry.candidate.semantic_hash[..16].bytes())
                .filter(|(left, right)| left != right)
                .count()
        })
        .min()
        .unwrap_or(0);
    nearest as f64 / 16.0
}

fn scored_order(left: &V3ScoredCandidate, right: &V3ScoredCandidate) -> std::cmp::Ordering {
    left.rmse
        .total_cmp(&right.rmse)
        .then_with(|| {
            left.candidate
                .complexity()
                .cmp(&right.candidate.complexity())
        })
        .then_with(|| {
            left.candidate
                .semantic_hash
                .cmp(&right.candidate.semantic_hash)
        })
}

fn source_tree_hash() -> Result<String> {
    let workspace = workspace_root();
    let mut paths = vec![
        workspace.join("Cargo.toml"),
        workspace.join("Cargo.lock"),
        workspace.join("scripts/run_golden42_v3_max_xbabe2.sh"),
    ];
    for root in [workspace.join("src"), workspace.join("vendor")] {
        collect_source_files(&root, &mut paths)?;
    }
    paths.sort();
    let mut hasher = Sha256::new();
    for path in paths {
        let relative = path.strip_prefix(&workspace)?.to_string_lossy();
        let bytes = fs::read(&path)?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_source_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_source_files(&path, output)?;
        } else if kind.is_file() {
            output.push(path);
        }
    }
    Ok(())
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
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id(),
        unix_ms()
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

fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id(),
        unix_ms()
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

fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_selection_prefers_distinct_physical_cores() {
        let cpus = choose_cpus(80).unwrap();
        assert_eq!(cpus.len(), 80);
        assert_eq!(cpus.iter().copied().collect::<BTreeSet<_>>().len(), 80);
        assert_eq!(&cpus[..64], &(0..64).collect::<Vec<_>>());
    }

    #[test]
    fn strict_rmse_is_sequential_and_rejects_bad_tapes() {
        assert_eq!(raw_rmse(&[0.0, 1.0], &[0.0, 1.0]).unwrap().to_bits(), 0);
        assert!(raw_rmse(&[], &[]).is_err());
        assert!(raw_rmse(&[0.0], &[f64::NAN]).is_err());
    }

    #[test]
    fn worker_group_cancellation_leaves_no_survivor() {
        let mut child = Command::new("setsid")
            .args(["sleep", "60"])
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id() as i32;
        let stdin = BufWriter::new(child.stdin.take().unwrap());
        let mut workers = vec![WorkerProcess {
            id: 0,
            cpu: 0,
            child,
            stdin,
        }];
        terminate_workers(&mut workers);
        assert!(workers[0].child.try_wait().unwrap().is_some());
        let status = unsafe { libc::kill(-pid, 0) };
        assert_eq!(status, -1, "cancelled worker process group survived");
    }

    #[test]
    fn checkpoint_resume_is_hash_chained_and_fail_closed() {
        let root = std::env::temp_dir().join(format!(
            "golden42-v3-checkpoint-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("checkpoints")).unwrap();
        let config = V3EvolutionConfig::default();
        let population = initial_population(&config, &[]);
        let mut scored = population
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, candidate)| V3ScoredCandidate {
                candidate,
                rmse: 0.1 + index as f64 * 1e-6,
                novelty: 1.0,
            })
            .collect::<Vec<_>>();
        scored.sort_by(scored_order);
        let next = next_generation(&config, 1, &scored, &[0; 4]).unwrap();
        let checkpoint = Checkpoint {
            schema: "jope-dime.golden42-v3-checkpoint.v1".to_string(),
            run_id: "fixture".to_string(),
            dataset_hash: DATASET.to_string(),
            generation: 0,
            previous_sha256: None,
            ranked_semantic_hashes: scored
                .iter()
                .map(|item| item.candidate.semantic_hash.clone())
                .collect(),
            scored: scored.clone(),
            archive: V3Archive::default(),
            next_population: next.clone(),
            created_at_unix_ms: 1,
        };
        let first = json_bytes(&checkpoint).unwrap();
        atomic_immutable_write(&root.join("checkpoints/generation-000000.v1.json"), &first)
            .unwrap();
        let loaded = validate_checkpoint_chain(&root, "fixture")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.0.next_population, next);
        assert_eq!(loaded.1, sha256_bytes(&first));
        let broken = Checkpoint {
            generation: 1,
            previous_sha256: Some("wrong".to_string()),
            next_population: next_generation(&config, 2, &scored, &[0; 4]).unwrap(),
            ..checkpoint
        };
        atomic_immutable_write(
            &root.join("checkpoints/generation-000001.v1.json"),
            &json_bytes(&broken).unwrap(),
        )
        .unwrap();
        assert!(validate_checkpoint_chain(&root, "fixture").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    fn assert_target_campaign_v3_fit_preserves_frozen_near_winner_bits() {
        let root = paths::jope_registry().join(DATASET);
        if !root.join("train.csv").is_file() || !root.join("test.csv").is_file() {
            return;
        }
        validate_dataset_hashes(&paths::jope_registry()).unwrap();
        let train = read_regression_csv(&root.join("train.csv")).unwrap();
        let test = read_regression_csv(&root.join("test.csv")).unwrap();
        let genome = GenomeIr::parse(NEAR_WINNER).unwrap();
        let training = campaign_training(&train.x).unwrap();
        let v3 = fit_campaign_candidate(&genome, &training, &train.y).unwrap();
        let legacy = fit_legacy_numeric_bound(&genome, &train.x, &train.y).unwrap();
        let test_table = numeric_raw_table(&test.x).unwrap();
        let expected = v3.predict_table(&test_table).unwrap();
        assert_eq!(
            prediction_bits(&expected),
            prediction_bits(&legacy.predict_table(&test_table).unwrap())
        );
        let codec = encode_fitted_cpp17(&v3).unwrap();
        let runtime_root = workspace_root().join("target/fitted-cpp-target-parity-runtime");
        fs::create_dir_all(&runtime_root).unwrap();
        let runtime = FittedCppRuntime::new(&runtime_root).unwrap();
        let cpp = runtime.load(&codec.bytes).unwrap();
        let encoded = EncodedTable::from_numeric_rows(&test.x).unwrap();
        let mut rust_times = Vec::with_capacity(30);
        let mut cpp_times = Vec::with_capacity(30);
        let mut actual = Vec::new();
        for _ in 0..30 {
            let started = Instant::now();
            assert_eq!(
                prediction_bits(&v3.predict_table(&test_table).unwrap()),
                prediction_bits(&expected)
            );
            rust_times.push(started.elapsed().as_nanos());
            let started = Instant::now();
            actual = cpp.predict(&encoded).unwrap();
            cpp_times.push(started.elapsed().as_nanos());
        }
        assert_eq!(expected.len(), actual.len());
        if let Some((index, rust, cpp)) = first_prediction_mismatch(&expected, &actual) {
            panic!(
                "target C++ parity mismatch at {index}: Rust {rust:?} ({:#018x}), C++ {cpp:?} ({:#018x})",
                rust.to_bits(),
                cpp.to_bits()
            );
        }
        rust_times.sort_unstable();
        cpp_times.sort_unstable();
        assert!(
            cpp_times[15] <= rust_times[15].saturating_mul(4).saturating_add(1_000_000),
            "target C++ median {}ns exceeds Rust {}ns correctness envelope",
            cpp_times[15],
            rust_times[15]
        );
    }

    #[test]
    #[ignore = "unavailable: real-data parity must reuse original registry model artifacts; legacy refits are disallowed"]
    fn target_campaign_v3_fit_preserves_frozen_near_winner_bits() {
        std::thread::Builder::new()
            .name("golden42-v3-frozen-near-winner".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(assert_target_campaign_v3_fit_preserves_frozen_near_winner_bits)
            .expect("spawn Golden42 V3 parity test with its qualified stack")
            .join()
            .expect("Golden42 V3 parity test thread panicked");
    }
}
