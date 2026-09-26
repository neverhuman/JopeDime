//! Unit-level positive fitting contracts behind the mandatory fit authority.
//!
//! Shipped binaries refuse every fit until a production namespace verifier
//! exists; `tests/fit_boundary.rs`, `tests/gp_contracts.rs` and
//! `tests/view_contracts.rs` prove those refusals through the real CLI. The
//! tests below exist only in this crate's `cfg(test)` build, where:
//!
//! * the synthetic scope is activated by the reviewed test-only issuer
//!   `Authority::activate_synthetic_for_test` (no product path can do this);
//! * the authority service runs without the independent event witness, an
//!   allowance that exists only under `cfg(test)`;
//! * the service host, its numerical workers and the CLI-shaped clients are
//!   separate processes of this test executable, entered through the ignored
//!   fixture tests below. Every fit still crosses `fit_boundary::fit`, the
//!   service claim/authorize/start custody and `fit_boundary::execute_service`.
use super::*;
use crate::eval_registry::{
    numerical_contract_sha256, ordered_training_sha256,
    service::{self, Service, ServiceConfig},
    Authority, HistoricalExclusion, HistoryInventory, NumericalContract, ScopePolicy,
};
use crate::prepare::load_view;
use rusqlite::OptionalExtension;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::{Child, Output},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const SCOPE: &str = "synthetic";
const PROGRAM: &str = "(pipe X (ridge))";
const REFUSED: &str = "namespace capability inactive: explicit verified activation required";
const HOST_SPEC: &str = "JOPE_FIT_CONTRACT_SERVICE_HOST";
const HOST_ENTRY: &str = "gp::fit_contract_tests::service_host_entry";
const WORKER_ENTRY: &str = "gp::fit_contract_tests::service_worker_entry";
/// Routes a CLI-shaped argv into [`cli_entry`] of this test executable.
pub(super) const CLI_ENTRY_ARGS: [&str; 7] = [
    "--exact",
    "gp::fit_contract_tests::cli_entry",
    "--ignored",
    "--nocapture",
    "--test-threads",
    "1",
    "--",
];
const PINS: [(&str, &str); 4] = [
    ("OMP_NUM_THREADS", "1"),
    ("RAYON_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
];
/// Never inherit native allocation, worker custody or crash-injection state.
const INHERITED: [&str; 13] = [
    "SCQ_JOB_ID",
    "SCQ_CONTROL_FD",
    "JOPE_GPU_POOL",
    "JOPE_GPU_LEASE_FD",
    "JOPE_GPU_WORKER_ACTIVE",
    "JOPE_SERVICE_INPUT_FD",
    "JOPE_SERVICE_TICKET",
    "JOPE_SERVICE_SOCKET",
    "JOPE_SERVICE_OUTPUT",
    "JOPE_EXPERIMENT_DIR",
    "JOPE_EXPERIMENT_CRASH_AFTER",
    "JOPE_FIT_DEADLINE_MONOTONIC_NS",
    HOST_SPEC,
];

/// Fixture child: the CLI subcommands the original integration contracts used.
#[test]
#[ignore = "fixture CLI entry, launched only by the fit contract tests"]
fn cli_entry() {
    let arguments: Vec<String> = std::env::args().collect();
    let Some(split) = arguments.iter().position(|argument| argument == "--") else {
        return;
    };
    if let Err(error) = run_cli(&arguments[split + 1..]) {
        eprintln!("Error: {error:?}");
        std::process::exit(1);
    }
}

fn run_cli(argv: &[String]) -> Result<()> {
    let (operation, rest) = argv.split_first().context("fixture CLI operation absent")?;
    let mut values = BTreeMap::new();
    let mut qualification = false;
    let mut index = 0;
    while index < rest.len() {
        let key = rest[index]
            .strip_prefix("--")
            .context("fixture CLI flag expected")?;
        if key == "qualification" {
            qualification = true;
            index += 1;
            continue;
        }
        let value = rest
            .get(index + 1)
            .context("fixture CLI flag value absent")?;
        ensure!(
            values.insert(key.to_owned(), value.clone()).is_none(),
            "repeated fixture CLI flag --{key}"
        );
        index += 2;
    }
    let text = |key: &str| {
        values
            .get(key)
            .cloned()
            .with_context(|| format!("fixture CLI --{key} absent"))
    };
    let path = |key: &str| text(key).map(PathBuf::from);
    match operation.as_str() {
        "fit" => fit(&path("train")?, &text("genome")?, &path("output")?),
        "fit-view" if std::env::var(FAULT).as_deref() == Ok("fail_export") => {
            anyhow::bail!("fixture: fitted-view export unavailable")
        }
        "fit-view" => fit_view(
            &path("prepared")?,
            text("cut")?.parse()?,
            &text("genome")?,
            &path("output")?,
        ),
        // The isolated export's worker (W1 follow-up 2c).
        "fit-memory" if std::env::var(FAULT).as_deref() == Ok("fail_export") => {
            anyhow::bail!("fixture: fitted-view export unavailable")
        }
        "fit-memory" => fit_memory(
            text("fd")?.parse()?,
            &text("training-sha256")?,
            &text("preprocessing-sha256")?,
            &text("genome")?,
            &path("output")?,
        ),
        // Same library call the original test made in-process, run inside a
        // configured client so its spawned worker inherits the fit bindings.
        "fit-view-isolated" => fit_view_isolated(
            &path("executable")?,
            &path("prepared")?,
            text("cut")?.parse()?,
            &text("genome")?,
            &path("output")?,
        ),
        "evaluate-worker" => evaluate(
            &path("train")?,
            &path("validation")?,
            &path("genome")?,
            &path("output")?,
        ),
        "evaluate-memory" => {
            let prepared = values.get("prepared").map(PathBuf::from);
            let cut = values
                .get("cut")
                .map(|cut| cut.parse::<usize>())
                .transpose()?;
            ensure!(
                prepared.is_some() == cut.is_some(),
                "--prepared and --cut go together"
            );
            evaluator_fault()?;
            evaluate_prepared_memory(
                text("fd")?.parse()?,
                &path("genome")?,
                &path("output")?,
                prepared.as_deref().zip(cut),
            )
        }
        "search" => search_seeded_with_rng_in_root(
            &path("root")?,
            &path("prepared")?,
            text("cut")?.parse()?,
            &path("output")?,
            text("evaluations")?.parse()?,
            text("seconds")?.parse()?,
            qualification,
            values.get("seeds").map(Path::new),
            values.get("vocabulary").map(Path::new),
            values
                .get("rng-seed")
                .map(|seed| seed.parse::<u64>())
                .transpose()?
                .unwrap_or(SEED),
        ),
        // The shipped `score` subcommand's library call (outer scoring).
        "score" => score(
            &path("model")?,
            &path("test")?,
            &path("receipt")?,
            &path("output")?,
        ),
        // The cycle pipeline's predictor handoff for one completed search.
        "export-selected" => {
            let root = path("root")?;
            let executable = path("executable")?;
            let executable_sha = file_hash(&executable)?;
            let seeds = values.get("seeds").map(PathBuf::from);
            let vocabulary = values.get("vocabulary").map(PathBuf::from);
            let receipt = handoff::export_selected(&handoff::SelectedExport {
                root: &root,
                search_output: &path("output")?,
                prepared: &path("prepared")?,
                cut: text("cut")?.parse()?,
                seeds: seeds.as_deref(),
                vocabulary: vocabulary.as_deref(),
                executable: &executable,
                executable_sha: &executable_sha,
                predictor: &path("predictor")?,
            })?;
            println!("{receipt}");
            Ok(())
        }
        other => anyhow::bail!("unsupported fixture CLI operation {other}"),
    }
}

/// Fixture-only evaluator faults. A test sets them on a search client; its
/// evaluator children inherit them. Nothing outside this harness reads them.
const FAULT: &str = "JOPE_FIXTURE_EVALUATOR_FAULT";
/// Directory where a faulting evaluator records the pids it leaves behind.
const CENSUS: &str = "JOPE_FIXTURE_EVALUATOR_CENSUS";

/// Inject the configured evaluator fault before any numerical work.
fn evaluator_fault() -> Result<()> {
    let Ok(fault) = std::env::var(FAULT) else {
        return Ok(());
    };
    // A helper process the evaluator starts and never waits for.
    let leave_behind = || -> Result<()> {
        let helper = Command::new("sleep")
            .arg("300")
            .stdin(Stdio::null())
            .spawn()?;
        let census = PathBuf::from(std::env::var_os(CENSUS).context("fixture census absent")?);
        fs::create_dir_all(&census)?;
        fs::write(census.join(format!("{}.pid", helper.id())), b"")?;
        std::mem::forget(helper);
        Ok(())
    };
    let hang = || std::thread::sleep(Duration::from_secs(3600));
    match fault.as_str() {
        "block" => hang(),
        "fork_block" => {
            leave_behind()?;
            hang();
        }
        "fork_exit" => leave_behind()?,
        "oom" => {
            // Twice the evaluator's 8 GiB address-space limit.
            let buffer: Vec<u8> = Vec::with_capacity(16 << 30);
            std::hint::black_box(&buffer);
        }
        other => anyhow::bail!("unknown fixture evaluator fault {other}"),
    }
    Ok(())
}

/// Fixture child: the long-lived synthetic authority service.
#[test]
#[ignore = "fixture service host, launched only by the fit contract tests"]
fn service_host_entry() {
    let Some(spec) = std::env::var_os(HOST_SPEC) else {
        return;
    };
    let spec: Value = read_json(Path::new(&spec)).unwrap();
    let path = |key: &str| PathBuf::from(spec[key].as_str().unwrap());
    let mut config = ServiceConfig::regression(
        path("authority"),
        path("witness"),
        path("socket"),
        std::env::current_exe().unwrap(),
        4,
    );
    config.worker_arguments = ["--exact", WORKER_ENTRY, "--ignored", "--nocapture"]
        .map(String::from)
        .to_vec();
    config.handshake_timeout = Duration::from_secs(60);
    let service = Service::bind(config).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    let requested = path("stop");
    let watcher = std::thread::spawn(move || {
        while !requested.exists() {
            std::thread::sleep(Duration::from_millis(10));
        }
        signal.store(true, Ordering::SeqCst);
    });
    let stats = service.run(stop).unwrap();
    watcher.join().unwrap();
    write_json(&path("stats"), &stats).unwrap();
}

/// Fixture child: the real service-owned numerical executor.
#[test]
#[ignore = "fixture numerical worker, launched only by the fixture service host"]
fn service_worker_entry() {
    if std::env::var_os("JOPE_SERVICE_INPUT_FD").is_none() {
        return;
    }
    crate::fit_boundary::execute_service().unwrap();
}

fn contract() -> NumericalContract {
    NumericalContract {
        protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
        precision: "f64".into(),
        omp_threads: 1,
        rayon_threads: 1,
        blas_threads: 1,
        mkl_threads: 1,
    }
}

fn training_hash(rows: &[Vec<f64>]) -> String {
    let x = rows
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect::<Vec<_>>();
    let y = rows
        .iter()
        .map(|row| *row.last().unwrap())
        .collect::<Vec<_>>();
    ordered_training_sha256(&x, &y).unwrap()
}

fn cut_training_hash(prepared: &Path) -> String {
    training_hash(&load_view(prepared, 0, true).unwrap())
}

fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "fixture CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Synthetic authority, service host and configured CLI clients.
struct Fixture {
    root: PathBuf,
    authority: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    host: Option<Child>,
}

impl Fixture {
    /// Synthetic scope with imported synthetic coverage; not yet activated.
    fn covered(root: &Path, training: Vec<String>) -> Self {
        let fixture = Self {
            root: root.into(),
            authority: root.join("synthetic-authority"),
            witness: root.join("independent-witness"),
            socket: root.join("authority.sock"),
            host: None,
        };
        Authority::initialize(
            &fixture.authority,
            &fixture.witness,
            vec![ScopePolicy {
                scope: SCOPE.into(),
                training_sha256: training,
                numerical_contract_sha256: vec![numerical_contract_sha256(&contract()).unwrap()],
                fit_seeds: vec![0],
            }],
        )
        .unwrap();
        fixture.import_history(&["synthetic-tombstone.json"]);
        fixture
    }

    /// The only issuer used here is the reviewed `cfg(test)` synthetic one.
    fn activated(root: &Path, training: Vec<String>) -> Self {
        let fixture = Self::covered(root, training);
        fixture.open().activate_synthetic_for_test().unwrap();
        assert_eq!(fixture.activation().unwrap().0, "active");
        fixture
    }

    fn open(&self) -> Authority {
        Authority::open(&self.authority, &self.witness).unwrap()
    }

    /// Explicit synthetic tombstones only. Each call covers every listed file.
    fn import_history(&self, names: &[&str]) -> String {
        let history = self.root.join("synthetic-history");
        fs::create_dir_all(&history).unwrap();
        let fence = self.root.join("synthetic-fence.json");
        if !fence.exists() {
            write_json(
                &fence,
                &json!({"synthetic_only":true,"production_dispatchers":0}),
            )
            .unwrap();
        }
        let mut files = BTreeMap::new();
        let mut occurrences = Vec::new();
        for name in names {
            let source = history.join(name);
            if !source.exists() {
                write_json(
                    &source,
                    &json!({"synthetic_only":true,"kind":"artificial excluded identity",
                    "name":name,"training_sha256":"0".repeat(64),"physical_fits":0}),
                )
                .unwrap();
            }
            let digest = file_hash(&source).unwrap();
            files.insert(source, digest.clone());
            occurrences.push(HistoricalExclusion {
                scope: SCOPE.into(),
                canonical_program: None,
                training_sha256: Some("0".repeat(64)),
                evidence_sha256: digest,
                reason: "explicit synthetic non-production tombstone".into(),
            });
        }
        self.open()
            .import_history(&HistoryInventory {
                scope: SCOPE.into(),
                source_roots: vec![history],
                files,
                occurrences,
                legacy_fence_sha256: file_hash(&fence).unwrap(),
                legacy_fence_receipt: fence,
            })
            .unwrap()
    }

    fn stop_path(&self) -> PathBuf {
        self.root.join("service-host.stop")
    }
    fn stats_path(&self) -> PathBuf {
        self.root.join("service-host.stats.json")
    }
    fn identity_path(&self) -> PathBuf {
        let mut path = self.socket.as_os_str().to_os_string();
        path.push(".identity.json");
        PathBuf::from(path)
    }
    fn host_log(&self) -> String {
        fs::read_to_string(self.root.join("service-host.log")).unwrap_or_default()
    }

    fn start(&mut self) {
        assert!(self.host.is_none(), "service host already running");
        let spec = self.root.join("service-host.json");
        write_json(
            &spec,
            &json!({"authority":self.authority,"witness":self.witness,"socket":self.socket,
            "stop":self.stop_path(),"stats":self.stats_path()}),
        )
        .unwrap();
        let log = fs::File::create(self.root.join("service-host.log")).unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command.args([
            "--exact",
            HOST_ENTRY,
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ]);
        for name in INHERITED {
            command.env_remove(name);
        }
        command
            .envs(PINS)
            .env(HOST_SPEC, &spec)
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log);
        self.host = Some(command.spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            if let Some(status) = self.host.as_mut().unwrap().try_wait().unwrap() {
                self.host = None;
                panic!(
                    "service host exited before readiness ({status}): {}",
                    self.host_log()
                );
            }
            if self.socket.exists()
                && self.identity_path().exists()
                && service::stats(&self.socket, &self.authority, &self.witness).is_ok()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "service host readiness exceeded 60 seconds: {}",
                self.host_log()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A configured client of this fixture's authority, in the given scope.
    fn command_in_scope(&self, scope: &str) -> Command {
        let executable = std::env::current_exe().unwrap();
        let mut command = Command::new(&executable);
        command.args(CLI_ENTRY_ARGS);
        for name in INHERITED {
            command.env_remove(name);
        }
        command
            .envs(PINS)
            .env("JOPE_CPU_EVALUATOR", &executable)
            .env("JOPE_FIT_AUTHORITY", &self.authority)
            .env("JOPE_FIT_WITNESS", &self.witness)
            .env("JOPE_FIT_SCOPE", scope)
            .env("JOPE_FIT_SERVICE_SOCKET", &self.socket);
        command
    }
    fn command(&self) -> Command {
        self.command_in_scope(SCOPE)
    }

    /// Bounded graceful drain of the original host; returns its service stats.
    fn stop_service(&mut self) -> Value {
        let mut host = self.host.take().expect("service host not running");
        fs::write(self.stop_path(), b"stop\n").unwrap();
        let deadline = Instant::now() + Duration::from_secs(40);
        let status = loop {
            if let Some(status) = host.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = host.kill();
                let _ = host.wait();
                panic!("service host failed bounded drain: {}", self.host_log());
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success(), "service host failed: {}", self.host_log());
        assert!(!self.socket.exists(), "graceful stop retained socket");
        assert!(
            !self.identity_path().exists(),
            "graceful stop retained readiness"
        );
        let stats: Value = read_json(&self.stats_path()).unwrap();
        assert_eq!(stats["active"], 0);
        stats
    }

    fn database(&self) -> rusqlite::Connection {
        rusqlite::Connection::open_with_flags(
            self.authority.join("registry.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap()
    }
    /// (all entries, completed entries)
    fn entries(&self) -> (i64, i64) {
        self.database()
            .query_row(
                "SELECT count(*),coalesce(sum(state='completed'),0) FROM entries",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap()
    }
    fn assert_one_original_fit(&self) {
        assert_eq!(self.entries(), (1, 1));
    }
    /// Sorted terminal states of every original fit entry.
    fn entry_states(&self) -> Vec<String> {
        let db = self.database();
        let mut statement = db
            .prepare("SELECT state FROM entries ORDER BY state")
            .unwrap();
        let states = statement
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<String>>>()
            .unwrap();
        states
    }
    fn assert_no_fit_starts(&self) {
        assert_eq!(
            self.entries().0,
            0,
            "refused request must never claim or start numerical work"
        );
    }
    /// (state, epoch) of the synthetic activation row.
    fn activation(&self) -> Option<(String, String)> {
        self.database()
            .query_row(
                "SELECT state,epoch FROM activations WHERE scope=?",
                [SCOPE],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .unwrap()
    }
    fn current_epoch(&self) -> String {
        self.database()
            .query_row(
                "SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),(SELECT instance FROM meta))",
                [],
                |r| r.get(0),
            )
            .unwrap()
    }
    fn coverage(&self) -> String {
        self.database()
            .query_row(
                "SELECT inventory_sha FROM coverage WHERE scope=?",
                [SCOPE],
                |r| r.get(0),
            )
            .unwrap()
    }
}

impl Drop for Fixture {
    /// Never leave a fixture service behind; a failed drain is killed/reaped.
    fn drop(&mut self) {
        if let Some(mut host) = self.host.take() {
            let _ = fs::write(self.stop_path(), b"stop\n");
            let deadline = Instant::now() + Duration::from_secs(40);
            while matches!(host.try_wait(), Ok(None)) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
            let _ = host.kill();
            let _ = host.wait();
        }
    }
}

/// The pointer-view preparation of `tests/view_contracts.rs`.
fn view_fixture() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let prepared = view_fixture_in(t.path(), "example", &[8, 9]);
    (t, prepared)
}

/// A pointer-view preparation under `root`: cut 0 trains on raw rows 0..8 and
/// validates on `validation`. Every variant shares the training rows and their
/// preprocessing, so only the scored membership differs between variants.
fn view_fixture_in(root: &Path, name: &str, validation: &[usize]) -> PathBuf {
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join("data/originals/raw.csv"), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared").join(name);
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let scored: Vec<Vec<String>> = validation.iter().map(|&i| raw[i].clone()).collect();
    let va = csv_bytes(&pp.transform(&scored).unwrap()).unwrap();
    let mapped = validation.iter().max().map_or(10, |&i| (i + 1).max(10));
    let p = json!({"schema":"parallel.prepared.v1","dataset":name,"family":"example","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),"selected_train_indices":(0..mapped).collect::<Vec<_>>(),"selected_test_indices":[0,1],"target_last":true,"rows_total":12,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":validation,"training_sha256":hash(&tr),"validation_sha256":hash(&va),"preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],"storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&root.join("data/originals/raw.csv")).unwrap(),"feature_columns":["x"],"target_column":"target","original_training_row_map":(0..mapped).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

/// The 40-row raw training file of the fitted predictor contract.
fn raw_training(root: &Path) -> (PathBuf, String) {
    let train = root.join("train.csv");
    let rows = (0..40)
        .map(|i| vec![i as f64, (i * i) as f64, 100.0 + i as f64 * 2.0])
        .collect::<Vec<_>>();
    fs::write(&train, csv_bytes(&rows).unwrap()).unwrap();
    let raw = Table::read(&train, &Value::Null).unwrap();
    let preprocessing = Preprocessing::fit(&raw.rows, "regression").unwrap();
    let training = training_hash(&preprocessing.transform(&raw.rows).unwrap());
    (train, training)
}

/// A refused public fit publishes nothing and claims nothing.
fn assert_fit_refused(authority: &Fixture, mut client: Command, train: &Path, message: &str) {
    let output = authority.root.join("refused-model.json");
    let result = client
        .args(["fit", "--train"])
        .arg(train)
        .args(["--genome", PROGRAM, "--output"])
        .arg(&output)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "refused fit succeeded");
    assert!(error.contains(message), "expected {message:?}: {error}");
    assert!(!output.exists(), "refused fit published a predictor");
    authority.assert_no_fit_starts();
}

#[test]
fn fitted_predictor_roundtrip_matches_native_gp_and_restores_target_units() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let model = tmp.path().join("model.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit", "--train"])
            .arg(&train)
            .args(["--genome", PROGRAM, "--output"])
            .arg(&model)
            .output()
            .unwrap(),
    );
    authority.assert_one_original_fit();
    let features = tmp.path().join("features.csv");
    let x = vec![vec![2.0, 4.0], vec![10.0, 100.0]];
    fs::write(&features, csv_bytes(&x).unwrap()).unwrap();
    let out = tmp.path().join("predictions.csv");
    predict(&model, &features, &out).unwrap();
    let predicted = fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert!(predicted.iter().all(|p| *p > 90.0 && *p < 190.0));
    let p: Predictor = read_json(&model).unwrap();
    let decoded: Model = serde_json::from_slice(&serde_json::to_vec(&p.model).unwrap()).unwrap();
    let tx = x
        .iter()
        .map(|r| {
            p.preprocessing
                .features(&r.iter().map(f64::to_string).collect::<Vec<_>>())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(p.model.predict(&tx), decoded.predict(&tx));
    let expected = p
        .model
        .predict(&tx)
        .iter()
        .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
        .collect::<Vec<_>>();
    assert_eq!(predicted, expected);
    let completion_ref = p.fit_completion.as_ref().unwrap();
    let completion: Value = read_json(Path::new(completion_ref["path"].as_str().unwrap())).unwrap();
    // Library half of `verify-fit-completion`: the committed original is
    // returned byte-for-byte and is bound to this authority instance only.
    let instance = authority.open().instance().to_owned();
    let verified = authority
        .open()
        .verify_committed_completion(completion_ref["sha256"].as_str().unwrap())
        .unwrap();
    assert_eq!(serde_json::to_value(&verified).unwrap(), completion);
    assert_eq!(verified.authority_instance, instance);
    assert_ne!(verified.authority_instance, "f".repeat(64));

    // Evicting a retained model leaves the original exclusion intact. Exporting
    // under a different destination cannot authorize replacement fitting.
    fs::remove_file(
        authority
            .authority
            .join("objects")
            .join(completion["model_sha256"].as_str().unwrap()),
    )
    .unwrap();
    let replacement = tmp.path().join("replacement-model.json");
    let unavailable = authority
        .command()
        .args(["fit", "--train"])
        .arg(&train)
        .args(["--genome", PROGRAM, "--output"])
        .arg(&replacement)
        .output()
        .unwrap();
    assert!(!unavailable.status.success());
    assert!(!replacement.exists());
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
}

#[test]
fn sealed_memory_view_matches_file_evaluation_and_rejects_writes() {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let tmp = tempfile::tempdir().unwrap();
    let rows = (0..30)
        .map(|i| vec![i as f64, (i * i) as f64, 3.0 + i as f64 * 0.8])
        .collect::<Vec<_>>();
    let mut memory = memory_view(&rows[..24], &rows[24..]).unwrap();
    assert!(memory.write_all(b"corrupt").is_err());
    let train = tmp.path().join("train.csv");
    let heldout = tmp.path().join("heldout.csv");
    let genome = tmp.path().join("p.dsl");
    fs::write(&train, csv_bytes(&rows[..24]).unwrap()).unwrap();
    fs::write(&heldout, csv_bytes(&rows[24..]).unwrap()).unwrap();
    fs::write(&genome, PROGRAM).unwrap();
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training_hash(&rows[..24])]);
    authority.start();
    success(
        authority
            .command()
            .args(["evaluate-worker", "--train"])
            .arg(&train)
            .arg("--validation")
            .arg(&heldout)
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&a)
            .output()
            .unwrap(),
    );
    success(
        authority
            .command()
            .args(["evaluate-memory", "--fd"])
            .arg(memory.as_raw_fd().to_string())
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&b)
            .output()
            .unwrap(),
    );
    let a: Value = read_json(&a).unwrap();
    let b: Value = read_json(&b).unwrap();
    assert_eq!(a["semantic_sha256"], b["semantic_sha256"]);
    assert_eq!(a["rmse"], b["rmse"]);
    assert_eq!(a["physical_fit_starts"], 1);
    assert_eq!(b["physical_fit_starts"], 0);
    assert_eq!(b["evaluation_origin"], "reuse");
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
    assert_eq!(stats["committed"], 1);
    assert_eq!(stats["reused"], 1);
    assert_eq!(stats["authority_opens"], 1);
}

#[test]
fn fitted_view_predictor_uses_the_exact_cut_and_original_target_units() {
    let (temp, prepared) = view_fixture();
    let output = temp.path().join("predictor.json");
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit-view", "--prepared"])
            .arg(&prepared)
            .args(["--cut", "0", "--genome", PROGRAM, "--output"])
            .arg(&output)
            .output()
            .unwrap(),
    );
    let predictor: Predictor = read_json(&output).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(predictor.training_sha256, p.cuts[0].training_sha256);
    let rows = load_view(&prepared, 0, true).unwrap();
    let x = rows.iter().map(|r| r[..1].to_vec()).collect::<Vec<_>>();
    let y = rows.iter().map(|r| r[1]).collect::<Vec<_>>();
    let expected = feat_math::fit(&GenomeIr::parse(PROGRAM).unwrap(), &x, &y).unwrap();
    assert_eq!(predictor.model.predict(&x), expected.predict(&x));
    let features = temp.path().join("features.csv");
    fs::write(&features, "3\n7\n").unwrap();
    let predictions = temp.path().join("predictions.csv");
    predict(&output, &features, &predictions).unwrap();
    let predictions = fs::read_to_string(predictions).unwrap();
    assert!(predictions
        .lines()
        .all(|r| r.parse::<f64>().unwrap() > 100.0));
    assert!(!prepared.join("cut-0/train.csv").exists());
    authority.assert_one_original_fit();
}

#[test]
fn isolated_fitted_view_export_binds_success_and_preserves_worker_failures() {
    let (temp, prepared) = view_fixture();
    let executable = std::env::current_exe().unwrap();
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    let isolated = |genome: &str, output: &Path| {
        authority
            .command()
            .args(["fit-view-isolated", "--executable"])
            .arg(&executable)
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", genome, "--output"])
            .arg(output)
            .output()
            .unwrap()
    };
    let output = temp.path().join("isolated.json");
    success(isolated(PROGRAM, &output));
    let receipt: Value = read_json(&output.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["predictor_sha256"], file_hash(&output).unwrap());
    assert_eq!(receipt["threads"], 1);
    assert_eq!(receipt["memory_limit_gib"], 8);
    let failed = temp.path().join("failed.json");
    assert!(!isolated("(pipe X (unknown))", &failed).status.success());
    let receipt: Value = read_json(&failed.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(!receipt["failure"].as_str().unwrap().is_empty());
    assert!(!failed.exists());
    assert!(!fs::read_to_string(failed.with_extension("fit.log"))
        .unwrap()
        .is_empty());
    // The unknown operator is a deterministic numerical failure: it reaches the
    // authorized worker once and is retained as that original terminal outcome.
    assert_eq!(authority.entry_states(), ["completed", "terminal_failure"]);
}

#[test]
fn experiment_crashes_at_durable_boundaries_recover_without_repeating_committed_fits() {
    for boundary in [
        "candidate_intent",
        "candidate_metric",
        "observation_journal",
        "observation_progress",
        "observations_snapshot",
        "search_summary",
    ] {
        let (temp, prepared) = view_fixture();
        let root = temp.path();
        let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
        authority.start();
        let experiment = root.join("experiment");
        fs::create_dir_all(&experiment).unwrap();
        write_json(
            &experiment.join("manifest.json"),
            &json!({"schema":"parallel.seed_experiment.v2"}),
        )
        .unwrap();
        let output = experiment.join("search");
        let command = || {
            let mut c = authority.command();
            c.args(["search", "--root"])
                .arg(root)
                .args([
                    "--cut",
                    "0",
                    "--evaluations",
                    "3",
                    "--seconds",
                    "180",
                    "--qualification",
                ])
                .arg("--prepared")
                .arg(&prepared)
                .arg("--output")
                .arg(&output)
                .env("JOPE_EXPERIMENT_DIR", &experiment);
            c
        };
        let failed = command()
            .env("JOPE_EXPERIMENT_CRASH_AFTER", boundary)
            .output()
            .unwrap();
        assert!(
            !failed.status.success(),
            "failpoint was not reached: {boundary}"
        );
        let before = if output.join("observations.jsonl").exists() {
            read_journal(&output.join("observations.jsonl")).unwrap()
        } else {
            vec![]
        };
        let metric: Option<Value> = if boundary == "candidate_metric" {
            Some(read_json(&output.join("candidate-0.json")).unwrap())
        } else {
            None
        };
        if !output.join("search.json").exists() {
            let resumed = command().output().unwrap();
            assert!(
                resumed.status.success(),
                "{boundary}: {}",
                String::from_utf8_lossy(&resumed.stderr)
            );
        }
        let after: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        assert_eq!(after.len(), 3, "{boundary}");
        // The independent replay reaches the published winner and stop from the
        // recovered ledger, including recovered metrics and unobserved intents.
        let replayed = selection_replay::verify_search(root, &output)
            .unwrap_or_else(|error| panic!("{boundary}: {error:#}"));
        assert_eq!(
            replayed.replay.stop,
            selection_replay::StopReason::EvaluationBudget,
            "{boundary}"
        );
        if boundary == "candidate_intent" {
            assert!(matches!(
                replayed.replay.rows[0].eligibility,
                selection_replay::Eligibility::Failed { .. }
            ));
        }
        assert_eq!(
            after[..before.len()],
            before,
            "committed observations changed at {boundary}"
        );
        if let Some(metric) = metric {
            assert_eq!(after[0]["metrics"], metric);
            assert_eq!(after[0]["recovered_completed_artifact"], true);
        }
        let counters = after
            .iter()
            .map(|r| r["proposal_attempts"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert!(
            counters.windows(2).all(|w| w[0] < w[1]),
            "RNG counter did not advance: {boundary}"
        );
        // Authority-side proof of "without repeating committed fits": every
        // original fit key started exactly once and nothing was replayed.
        let stats = authority.stop_service();
        let (entries, _) = authority.entries();
        let fitted = after.iter().filter(|r| !r["fit_outcome"].is_null()).count();
        assert_eq!(stats["worker_spawns"], entries, "{boundary}: {stats}");
        assert_eq!(stats["reused"], 0, "{boundary}: {stats}");
        assert_eq!(fitted as i64, entries, "{boundary}");
        if boundary == "candidate_intent" {
            // An intent without a completion is never dispatched again.
            assert_eq!(
                after[0]["failure"],
                "unknown_original_execution_without_completion"
            );
            assert!(after[0]["fit_outcome"].is_null());
            assert_eq!(entries, 2);
        }
    }
}

#[test]
fn one_evaluation_budget_runs_the_incumbent_before_neural_candidates() {
    let (temp, prepared) = view_fixture();
    let vocabulary = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let vocab: crate::grammar::Vocab = read_json(&vocabulary).unwrap();
    let champion = PROGRAM.to_string();
    let mut seen = BTreeSet::from([champion.clone()]);
    let mut programs = Vec::new();
    for seed in 0..10000 {
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        if let Ok(program) = vocab.project_proposal(&raw) {
            if seen.insert(program.clone()) {
                programs.push(program);
            }
        }
        if programs.len() == 79 {
            break;
        }
    }
    assert_eq!(programs.len(), 79);
    programs.insert(64, champion.clone());
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    let seeds = temp.path().join("seeds.json");
    write_json(
        &seeds,
        &SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: file_hash(&prepared.join("prepared.json")).unwrap(),
            cut: 0,
            training_sha256: p.cuts[0].training_sha256.clone(),
            context_sha256: "0".repeat(64),
            checkpoint_sha256: "1".repeat(64),
            candidates: programs
                .into_iter()
                .enumerate()
                .map(|(i, genome)| SeedCandidate {
                    genome,
                    origin: if i == 64 {
                        "incumbent_cut_teacher"
                    } else {
                        "neural_policy"
                    }
                    .into(),
                    policy_logprob: None,
                })
                .collect(),
        },
    )
    .unwrap();
    let output = temp.path().join("teacher-first-search");
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    let result = authority
        .command()
        .args(["search", "--root"])
        .arg(temp.path())
        .arg("--prepared")
        .arg(&prepared)
        .args(["--cut", "0", "--output"])
        .arg(&output)
        .args([
            "--evaluations",
            "1",
            "--seconds",
            "90",
            "--qualification",
            "--seeds",
        ])
        .arg(&seeds)
        .arg("--vocabulary")
        .arg(&vocabulary)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let observations: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0]["origin"], "incumbent_cut_teacher");
    assert_eq!(observations[0]["initial_seed_slot"], 64);
    assert_eq!(observations[0]["genome"], champion);
    assert!(observations[0]["failure"].is_null());
    authority.assert_one_original_fit();
}

#[test]
fn deactivated_synthetic_namespace_refuses_a_configured_fit_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    authority.open().namespace_deactivate(SCOPE).unwrap();
    assert_eq!(authority.activation().unwrap().0, "inactive");
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
fn coverage_reimport_deactivates_the_synthetic_namespace_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    let prior = authority.coverage();
    authority.import_history(&["synthetic-tombstone.json", "synthetic-tombstone-2.json"]);
    assert_ne!(authority.coverage(), prior, "coverage was not replaced");
    assert_eq!(authority.activation().unwrap().0, "inactive");
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
fn out_of_scope_training_and_foreign_scope_are_refused_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let other = "a".repeat(64);
    assert_ne!(training, other);
    let mut authority = Fixture::activated(tmp.path(), vec![other]);
    authority.start();
    assert_fit_refused(
        &authority,
        authority.command(),
        &train,
        "training data outside authorized scope",
    );
    assert_fit_refused(
        &authority,
        authority.command_in_scope("synthetic-foreign"),
        &train,
        "scope denied",
    );
    // The namespace itself stayed active: the scope checks refused alone.
    assert_eq!(authority.activation().unwrap().0, "active");
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
fn recovery_epoch_change_invalidates_the_synthetic_activation_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    let (_, activated_epoch) = authority.activation().unwrap();
    assert_eq!(activated_epoch, authority.current_epoch());
    let report = Authority::recovery_reconcile(&authority.authority, &authority.witness).unwrap();
    Authority::recovery_commit_epoch(&authority.authority, &authority.witness, &report).unwrap();
    // The row is untouched; only its binding to the prior epoch is now stale.
    assert_eq!(
        authority.activation().unwrap(),
        ("active".to_string(), activated_epoch.clone())
    );
    assert_ne!(authority.current_epoch(), activated_epoch);
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

/// A deterministic numerical failure that parses but has no fitted operator.
const UNFITTABLE: &str = "(pipe X (unknown))";

/// An 80-slot seed population for the view fixture's cut 0. The first slots are
/// `leading`; every slot carries the given critic rank weight (never a score).
fn seed_population(
    prepared: &Path,
    path: &Path,
    leading: &[&str],
    weight: impl Fn(usize) -> Option<f64>,
) {
    let mut programs: Vec<String> = leading.iter().map(|p| p.to_string()).collect();
    let mut seen: BTreeSet<String> = programs.iter().cloned().collect();
    let mut seed = 0;
    while programs.len() < 80 {
        let program = feat_math::evolve::random_canonical_pipeline(seed, 6);
        seed += 1;
        if seen.insert(program.clone()) {
            programs.push(program);
        }
    }
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    write_json(
        path,
        &SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: file_hash(&prepared.join("prepared.json")).unwrap(),
            cut: 0,
            training_sha256: p.cuts[0].training_sha256.clone(),
            context_sha256: "0".repeat(64),
            checkpoint_sha256: "1".repeat(64),
            candidates: programs
                .into_iter()
                .enumerate()
                .map(|(i, genome)| SeedCandidate {
                    genome,
                    origin: "neural_policy".into(),
                    policy_logprob: weight(i),
                })
                .collect(),
        },
    )
    .unwrap();
}

/// A configured CLI search of cut 0 under `root`.
fn search_client(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    evaluations: usize,
) -> Command {
    timed_search_client(authority, root, prepared, output, evaluations, 180)
}

/// A configured CLI search of cut 0 with an explicit search deadline.
fn timed_search_client(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    evaluations: usize,
    seconds: u64,
) -> Command {
    let mut command = authority.command();
    command
        .args(["search", "--root"])
        .arg(root)
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .args(["--cut", "0", "--qualification"])
        .args(["--seconds", &seconds.to_string()])
        .args(["--evaluations", &evaluations.to_string()]);
    command
}

/// Rewrite a completed search's journal, snapshot and snapshot digest together,
/// so that only the deliberately tampered rows differ from the publication.
fn republish_ledger(output: &Path, rows: &[Value]) {
    let mut journal = Vec::new();
    for row in rows {
        journal.extend(serde_json::to_vec(row).unwrap());
        journal.push(b'\n');
    }
    fs::write(output.join("observations.jsonl"), journal).unwrap();
    write_json(&output.join("observations.json"), &rows.to_vec()).unwrap();
    let mut search: Value = read_json(&output.join("search.json")).unwrap();
    search["observations_sha256"] = json!(file_hash(&output.join("observations.json")).unwrap());
    write_json(&output.join("search.json"), &search).unwrap();
}

#[test]
fn independent_replay_reproduces_real_search_winners_and_refuses_tampered_ledgers() {
    use selection_replay::{verify_search, Eligibility, StopReason};
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // The critic ranks the unfittable program first and the linear head last;
    // the second search reverses every rank weight. Neither may matter.
    let leading = ["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM];
    let forward = root.join("seeds-forward.json");
    let reversed = root.join("seeds-reversed.json");
    seed_population(&prepared, &forward, &leading, |i| Some(-(i as f64)));
    seed_population(&prepared, &reversed, &leading, |i| Some(i as f64 - 80.0));
    let mut outputs = Vec::new();
    for (name, seeds) in [("search-forward", &forward), ("search-reversed", &reversed)] {
        let output = root.join(name);
        success(
            search_client(&authority, root, &prepared, &output, 3)
                .arg("--seeds")
                .arg(seeds)
                .output()
                .unwrap(),
        );
        outputs.push(output);
    }
    let output = &outputs[0];
    let before = crate::smartcluster_worker::hash_directory(output).unwrap();
    let replayed = verify_search(root, output).unwrap();
    // Replay is read-only: no authority-bearing or published byte changed.
    assert_eq!(
        crate::smartcluster_worker::hash_directory(output).unwrap(),
        before
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let search: Value = read_json(&output.join("search.json")).unwrap();
    let replay = &replayed.replay;
    assert_eq!(replay.stop, StopReason::EvaluationBudget);
    assert_eq!(rows[1]["genome"], UNFITTABLE);
    assert!(
        matches!(&replay.rows[1].eligibility, Eligibility::Failed { failure } if !failure.is_empty())
    );
    assert_eq!((replay.measured, replay.failed), (2, 1));
    // A third, trivially independent computation over the published rows.
    let scale = rows[0]["target_scale"].as_f64().unwrap();
    let measured: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row["failure"].is_null())
        .map(|(i, row)| (i, row["metrics"]["rmse"].as_f64().unwrap() * scale))
        .collect();
    let best = measured.iter().fold(
        measured[0],
        |best, next| if next.1 < best.1 { *next } else { best },
    );
    let winner = replay.winner.clone().unwrap();
    assert_eq!(winner.position, best.0 as i64);
    assert_eq!(winner.objective.to_bits(), best.1.to_bits());
    assert_eq!(search["best_program"], rows[best.0]["genome"]);
    // Identical predictions are one ranked entry represented by the earliest row.
    if rows[0]["metrics"]["semantic_sha256"] == rows[2]["metrics"]["semantic_sha256"] {
        assert_eq!(replay.ranking.len(), 1);
        assert_eq!(winner.position, 0);
        assert!(matches!(
            replay.rows[2].eligibility,
            Eligibility::SemanticDuplicate { of_position: 0, .. }
        ));
    } else {
        assert_eq!(replay.ranking.len(), 2);
    }
    // Reversing every predicted rank weight changes no measurement or winner.
    let other = verify_search(root, &outputs[1]).unwrap().replay;
    assert_eq!(other.winner, replay.winner);
    assert_eq!(other.ranking, replay.ranking);
    assert_eq!(other.rows, replay.rows);

    // Tamper suite: each change refuses; the original bytes are then restored.
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(output.join(name)).unwrap())
        .collect();
    let restore = || {
        for (name, bytes) in files.iter().zip(&original) {
            fs::write(output.join(name), bytes).unwrap();
        }
    };
    let refused = |expected: &str| {
        let error = format!("{:#}", verify_search(root, output).unwrap_err());
        assert!(error.contains(expected), "expected {expected:?}: {error}");
        restore();
        assert!(verify_search(root, output).is_ok());
    };
    let mut swapped = rows.clone();
    swapped.swap(0, 2);
    republish_ledger(output, &swapped);
    refused("ledger order");
    let mut forged = rows.clone();
    forged[best.0]["metrics"]["rmse"] = json!(0.0);
    republish_ledger(output, &forged);
    refused("original evidence refused");
    let mut revived = rows.clone();
    revived[1]["failure"] = Value::Null;
    revived[1]["metrics"] = rows[best.0]["metrics"].clone();
    republish_ledger(output, &revived);
    refused("original evidence refused");
    let mut published = search.clone();
    published["best_program"] = rows[1]["genome"].clone();
    write_json(&output.join("search.json"), &published).unwrap();
    refused("published winner differs");
    let mut published = search.clone();
    published["best_validation_rmse"] = json!(best.1 / 2.0);
    write_json(&output.join("search.json"), &published).unwrap();
    refused("published winner differs");
    let mut torn = original[0].clone();
    torn.extend_from_slice(b"{\"schema\"");
    fs::write(output.join("observations.jsonl"), torn).unwrap();
    refused("torn tail");
    // A late or stray result outside the ledger is reported, never ranked.
    fs::write(
        output.join("candidate-3.json"),
        serde_json::to_vec(&json!({"rmse":0.0,"semantic_sha256":"9".repeat(64)})).unwrap(),
    )
    .unwrap();
    fs::write(output.join("candidate-3.attempt.json"), b"{}").unwrap();
    let late = verify_search(root, output).unwrap();
    assert_eq!(late.replay.winner, replay.winner);
    assert_eq!(
        late.unledgered_artifacts,
        [
            PathBuf::from("candidate-3.attempt.json"),
            PathBuf::from("candidate-3.json")
        ]
    );
    authority.stop_service();
    // Three original fit entries; the reversed search started no new fit.
    assert_eq!(authority.entries().0, 3);
    let reused: Vec<Value> = read_json(&outputs[1].join("observations.json")).unwrap();
    for row in reused.iter().filter(|row| row["failure"].is_null()) {
        assert_eq!(row["evaluation_origin"], "reuse");
        assert_eq!(row["metrics"]["physical_fit_starts"], 0);
    }
}

#[test]
fn resumed_search_refuses_journaled_measurements_its_original_completion_does_not_support() {
    for forged in [json!(0.0), json!(-1.0)] {
        let (temp, prepared) = view_fixture();
        let root = temp.path();
        let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
        authority.start();
        let experiment = root.join("experiment");
        fs::create_dir_all(&experiment).unwrap();
        write_json(
            &experiment.join("manifest.json"),
            &json!({"schema":"parallel.seed_experiment.v2"}),
        )
        .unwrap();
        let output = experiment.join("search");
        let command = || {
            let mut command = search_client(&authority, root, &prepared, &output, 2);
            command.env("JOPE_EXPERIMENT_DIR", &experiment);
            command
        };
        let crashed = command()
            .env("JOPE_EXPERIMENT_CRASH_AFTER", "observation_progress")
            .output()
            .unwrap();
        assert!(!crashed.status.success());
        let journal = output.join("observations.jsonl");
        let mut rows = read_journal(&journal).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["failure"].is_null());
        assert_ne!(rows[0]["metrics"]["rmse"], forged);
        rows[0]["metrics"]["rmse"] = forged.clone();
        let mut bytes = serde_json::to_vec(&rows[0]).unwrap();
        bytes.push(b'\n');
        fs::write(&journal, bytes).unwrap();
        let resumed = command().output().unwrap();
        let published: Option<Value> = read_json(&output.join("search.json")).ok();
        assert!(
            !resumed.status.success(),
            "resume adopted the journaled RMSE {forged} as measured evidence: {:?}",
            published.map(|s| (s["best_program"].clone(), s["best_validation_rmse"].clone()))
        );
        assert!(!output.join("search.json").exists());
        // The refusal precedes any new dispatch: only the original fit exists.
        authority.stop_service();
        assert_eq!(authority.entries().0, 1);
    }
}

/// The featmath grammar vocabulary shipped with this crate.
fn vocabulary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json")
}

/// Independent enumeration of the generic bootstrap: `(original generator
/// draw, its single adaptation)` per slot, starting with the fixed reference.
fn generic_bootstrap(vocab: &crate::grammar::Vocab, slots: usize) -> Vec<(String, String, u64)> {
    let mut seen = BTreeSet::from([PROGRAM.to_string()]);
    let mut expected = vec![(PROGRAM.to_string(), PROGRAM.to_string(), 0)];
    let mut seed = SEED;
    while expected.len() < slots {
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        if let Ok(adapted) = vocab.project_proposal(&raw) {
            if seen.insert(adapted.clone()) {
                expected.push((raw, adapted, seed));
            }
        }
        seed += 1;
    }
    expected
}

#[test]
fn generic_bootstrap_candidates_retain_their_original_proposal_and_one_adaptation() {
    let vocabulary = vocabulary_path();
    let vocab: crate::grammar::Vocab = read_json(&vocabulary).unwrap();
    let expected = generic_bootstrap(&vocab, 12);
    let adapted = expected
        .iter()
        .position(|(raw, adapted, _)| raw != adapted)
        .expect("the fixture must exercise an adapted bootstrap slot");
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("generic");
    success(
        search_client(&authority, root, &prepared, &output, adapted + 1)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), adapted + 1);
    for (i, row) in rows.iter().enumerate() {
        let (raw, executable, _) = &expected[i];
        assert_eq!(row["genome"], *executable, "slot {i}: executable recipe");
        let attempt: Value =
            read_json(&output.join(format!("candidate-{i}.attempt.json"))).unwrap();
        assert_eq!(
            attempt["raw_genome"], *raw,
            "slot {i}: the original generator proposal was replaced by its adaptation"
        );
        assert_eq!(
            row["proposal_before_projection"],
            if raw != executable {
                json!(raw)
            } else {
                Value::Null
            },
            "slot {i}: ledger original proposal"
        );
    }
    // The envelope re-derives each original from its generator draw, exactly one
    // adaptation, and the executable; the selected subject links to its fold fit.
    let verified = lineage::verify_search(root, &output, None, Some(vocabulary.as_path())).unwrap();
    assert!(verified.lineage.adapted >= 1);
    for (i, chain) in verified.lineage.chains.iter().enumerate() {
        let (raw, executable, seed) = &expected[i];
        assert_eq!(chain.original_sha256, hash(raw.as_bytes()));
        assert_eq!(chain.executable_sha256, hash(executable.as_bytes()));
        assert_eq!(chain.adapted, raw != executable);
        if i == 0 {
            assert_eq!(chain.source, "fixed_reference");
        } else {
            assert_eq!(chain.source, "generic_bootstrap");
            assert_eq!(rows[i]["lineage"]["source"]["generator_seed"], *seed);
        }
    }
    // A wrong or missing vocabulary is not the frozen request's adaptation.
    assert!(lineage::verify_search(root, &output, None, None).is_err());
    let selected = verified.selected.as_ref().expect("a measured winner");
    let winner = verified.selection.replay.winner.clone().unwrap().genome;
    // Final refit of the selected recipe on the same cut training rows: an
    // explicit transformation that reuses the selected fold fit's state.
    let fit_view = |genome: &str, predictor: &Path| {
        success(
            authority
                .command()
                .args(["fit-view", "--prepared"])
                .arg(&prepared)
                .args(["--cut", "0", "--genome", genome, "--output"])
                .arg(predictor)
                .output()
                .unwrap(),
        );
    };
    let predictor = root.join("final-predictor.json");
    fit_view(&winner, &predictor);
    let refit = lineage::verify_final_refit(root, &verified, &predictor).unwrap();
    assert_eq!(refit.model_sha256, selected.fold_model["sha256"]);
    // Another recipe, or the selected recipe's name over another fitted state,
    // is not the selected subject's refit, whatever its predictions.
    let other = root.join("other-predictor.json");
    fit_view("(pipe X (standardize) (ridge))", &other);
    let error = format!(
        "{:#}",
        lineage::verify_final_refit(root, &verified, &other).unwrap_err()
    );
    assert!(
        error.contains("not the selected executable recipe"),
        "{error}"
    );
    let mut swapped: Predictor = read_json(&other).unwrap();
    swapped.canonical = GenomeIr::parse(&winner).unwrap().canonical();
    let swapped_path = root.join("swapped-predictor.json");
    write_json(&swapped_path, &swapped).unwrap();
    let error = format!(
        "{:#}",
        lineage::verify_final_refit(root, &verified, &swapped_path).unwrap_err()
    );
    assert!(error.contains("not the selected fold fit"), "{error}");
    authority.stop_service();
}

#[test]
fn variation_children_record_parents_and_cannot_alter_parent_fitted_state() {
    let vocabulary = vocabulary_path();
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // Slots 1..3 adapt onto slot 0's program, so their first proposals are
    // refused as duplicates and each slot is filled by an explicit variation.
    let seeds = root.join("seeds.json");
    seed_population(
        &prepared,
        &seeds,
        &[
            PROGRAM,
            "(pipe X  (ridge))",
            "(pipe X   (ridge))",
            "(pipe X    (ridge))",
        ],
        |_| None,
    );
    let output = root.join("variations");
    success(
        search_client(&authority, &root, &prepared, &output, 4)
            .arg("--seeds")
            .arg(&seeds)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let verified = lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path()),
    )
    .unwrap();
    assert_eq!(verified.lineage.chains[0].source, "seed_population_slot");
    assert!(verified.lineage.variations >= 3, "{:?}", verified.lineage);
    // Frozen inputs are authenticated: another seed population is refused.
    let other = root.join("other-seeds.json");
    seed_population(&prepared, &other, &[PROGRAM], |_| Some(-1.0));
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(other.as_path()),
        Some(vocabulary.as_path())
    )
    .is_err());
    // Every parent's original fitted state still reproduces its recorded
    // predictions after all of its children were fitted: children are new
    // subjects, never in-place edits of their parents.
    let validation = load_view(&prepared, 0, false).unwrap();
    let x: Vec<Vec<f64>> = validation
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect();
    let parents: BTreeSet<i64> = verified
        .lineage
        .chains
        .iter()
        .flat_map(|chain| chain.parents.iter().copied())
        .collect();
    assert!(
        !parents.is_empty(),
        "no variation recorded a measured parent"
    );
    for parent in parents {
        let row = &rows[parent as usize];
        let completion = crate::label_admission::artifact(&root, &row["clean_completion"]).unwrap();
        let model: feat_math::Model = serde_json::from_slice(
            &crate::label_admission::artifact_bytes(&root, &completion["model"]).unwrap(),
        )
        .unwrap();
        let predictions = model.predict(&x);
        assert_eq!(
            hash(&serde_json::to_vec(&predictions).unwrap()),
            row["metrics"]["semantic_sha256"],
            "parent {parent} fitted state changed"
        );
    }
    // Tampering with the pre-dispatch attempt breaks its completion binding.
    let attempt_path = output.join("candidate-1.attempt.json");
    let original = fs::read(&attempt_path).unwrap();
    let mut attempt: Value = serde_json::from_slice(&original).unwrap();
    attempt["lineage"]["source"]["operator"] = json!("fresh_random");
    write_json(&attempt_path, &attempt).unwrap();
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path())
    )
    .is_err());
    fs::write(&attempt_path, original).unwrap();
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path())
    )
    .is_ok());
    authority.stop_service();
}

#[test]
fn reused_fits_are_rescored_on_their_own_rows_and_never_inherit_scores() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    // Same training rows and preprocessing; only the scored membership differs.
    let first = view_fixture_in(&root, "example", &[8, 9]);
    let second = view_fixture_in(&root, "example-b", &[10, 11]);
    let a: Prepared = read_json(&first.join("prepared.json")).unwrap();
    let b: Prepared = read_json(&second.join("prepared.json")).unwrap();
    assert_eq!(a.cuts[0].training_sha256, b.cuts[0].training_sha256);
    assert_eq!(
        a.cuts[0].preprocessing_sha256,
        b.cuts[0].preprocessing_sha256
    );
    assert_ne!(a.cuts[0].validation_sha256, b.cuts[0].validation_sha256);
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&first)]);
    authority.start();
    let run = |prepared: &Path, name: &str| {
        let output = root.join(name);
        success(
            search_client(&authority, &root, prepared, &output, 3)
                .output()
                .unwrap(),
        );
        let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        (output, rows)
    };
    let (_, cold) = run(&first, "cold");
    let fitted = authority.entries().0;
    let (warm_output, warm) = run(&second, "warm");
    // Identical fitting work was reused: no new physical fit anywhere.
    assert_eq!(authority.entries().0, fitted);
    let search: Value = read_json(&warm_output.join("search.json")).unwrap();
    let measured = warm.iter().filter(|row| row["failure"].is_null()).count();
    assert!(measured > 0);
    assert_eq!(search["execution_accounting"]["physical_fit_starts"], 0);
    assert_eq!(
        search["execution_accounting"]["reused_completions"],
        measured
    );
    let validation = load_view(&second, 0, false).unwrap();
    let x: Vec<Vec<f64>> = validation
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect();
    let mut fit_keys = BTreeSet::new();
    for (old, new) in cold.iter().zip(&warm) {
        assert_eq!(old["genome"], new["genome"]);
        if !new["failure"].is_null() {
            continue;
        }
        let (was, now) = (&old["metrics"], &new["metrics"]);
        assert_eq!(now["evaluation_origin"], "reuse");
        assert_eq!(now["physical_fit_starts"], 0);
        // Same fitted subject (fit key and model) ...
        assert_eq!(
            was["score_identity"]["fit_key"],
            now["score_identity"]["fit_key"]
        );
        assert_eq!(
            was["score_identity"]["model_sha256"],
            now["score_identity"]["model_sha256"]
        );
        fit_keys.insert(now["score_identity"]["fit_key"].to_string());
        // ... but a distinct score identity measured on its own rows.
        assert_eq!(
            now["score_identity"]["scoring_rows_sha256"],
            b.cuts[0].validation_sha256
        );
        assert_ne!(was["score_identity_sha256"], now["score_identity_sha256"]);
        // The stored fitted state, applied to the new rows, reproduces the new
        // predictions and RMSE: the score was measured, not carried over.
        let completion = crate::label_admission::artifact(&root, &new["clean_completion"]).unwrap();
        let model: feat_math::Model = serde_json::from_slice(
            &crate::label_admission::artifact_bytes(&root, &completion["model"]).unwrap(),
        )
        .unwrap();
        let predictions = model.predict(&x);
        assert_eq!(
            hash(&serde_json::to_vec(&predictions).unwrap()),
            now["semantic_sha256"],
            "the reused fit was not rescored on the new rows"
        );
        let squared = validation
            .iter()
            .zip(&predictions)
            .map(|(row, p)| (row[row.len() - 1] - p).powi(2))
            .sum::<f64>();
        let rmse = (squared / predictions.len() as f64).sqrt();
        let recorded = now["rmse"].as_f64().unwrap();
        assert!(
            (rmse - recorded).abs() <= 1e-12 * rmse.abs().max(1.0),
            "recorded RMSE {recorded} is not the new rows' RMSE {rmse}"
        );
        assert_ne!(was["rmse"], now["rmse"], "a score carried over unchanged");
    }
    // Different recipes never share a fit key.
    assert_eq!(fit_keys.len(), measured);
    // Both searches replay; a score carried over from the other membership,
    // even with its own consistent identity, is refused.
    let cold_output = root.join("cold");
    selection_replay::verify_search(&root, &cold_output).unwrap();
    selection_replay::verify_search(&root, &warm_output).unwrap();
    let position = warm
        .iter()
        .position(|row| row["failure"].is_null())
        .unwrap();
    let mut stale = warm.clone();
    stale[position]["metrics"] = cold[position]["metrics"].clone();
    republish_ledger(&warm_output, &stale);
    let error = format!(
        "{:#}",
        selection_replay::verify_search(&root, &warm_output).unwrap_err()
    );
    assert!(
        error.contains("another fold") || error.contains("outside the frozen fold"),
        "{error}"
    );
    authority.stop_service();
}

#[test]
fn feasibility_estimates_are_recorded_beside_observed_outcomes_under_one_rule() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // A covered recipe, one the fitter provably refuses (a misplaced target
    // stage), and one with an uncovered head.
    let refused = "(pipe X (standardize) (target) (ridge))";
    let uncovered = "(pipe X (standardize) (gpr m=50 steps=5))";
    let seeds = root.join("seeds.json");
    seed_population(&prepared, &seeds, &[PROGRAM, refused, uncovered], |_| None);
    let output = root.join("feasibility");
    success(
        search_client(&authority, &root, &prepared, &output, 3)
            .arg("--seeds")
            .arg(&seeds)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let verdicts: Vec<&str> = rows
        .iter()
        .map(|row| row["feasibility"]["verdict"].as_str().unwrap())
        .collect();
    assert_eq!(
        verdicts,
        ["proven_feasible", "proven_outside_limits", "unknown"]
    );
    // Estimate beside observation: the proven-feasible recipe fitted, and the
    // proven refusal was observed as a deterministic fit failure. Recording the
    // estimate did not change dispatch: all three reached the fitter.
    assert!(rows[0]["failure"].is_null());
    assert!(!rows[1]["failure"].is_null());
    assert_eq!(rows[1]["fit_outcome"], "deterministic_failure");
    assert_eq!(authority.entries().0, 3);
    // One rule for every arm: each recorded estimate is exactly the classifier
    // applied to the program and the cut's declared shape (8 rows, 1 feature).
    let shape = feat_math::feasibility::Shape { rows: 8, cols: 1 };
    for (i, row) in rows.iter().enumerate() {
        let attempt: Value =
            read_json(&output.join(format!("candidate-{i}.attempt.json"))).unwrap();
        let expected = serde_json::to_value(feat_math::feasibility::classify(
            &GenomeIr::parse(row["genome"].as_str().unwrap()).unwrap(),
            shape,
            &feat_math::feasibility::Limits::default(),
        ))
        .unwrap();
        assert_eq!(attempt["feasibility"], expected, "candidate {i}");
    }
    // Selection replay and lineage remain valid with the diagnostic present.
    lineage::verify_search(&root, &output, Some(seeds.as_path()), None).unwrap();
    authority.stop_service();
}

#[test]
fn a_resume_under_other_selection_rules_is_refused_not_restarted() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let experiment = root.join("experiment");
    fs::create_dir_all(&experiment).unwrap();
    write_json(
        &experiment.join("manifest.json"),
        &json!({"schema":"parallel.seed_experiment.v2"}),
    )
    .unwrap();
    let output = experiment.join("search");
    let command = || {
        let mut command = search_client(&authority, root, &prepared, &output, 2);
        command.env("JOPE_EXPERIMENT_DIR", &experiment);
        command
    };
    let crashed = command()
        .env("JOPE_EXPERIMENT_CRASH_AFTER", "observation_progress")
        .output()
        .unwrap();
    assert!(!crashed.status.success());
    // The objective, evidence, tie and stop rules were frozen with the request.
    let request_path = output.join("search-request.json");
    let original = fs::read(&request_path).unwrap();
    let request: Value = serde_json::from_slice(&original).unwrap();
    assert_eq!(request["selection_contract"], selection_replay::CONTRACT);
    assert_eq!(
        request["proposal_cap_per_slot"],
        selection_replay::PROPOSAL_CAP_PER_SLOT
    );
    let journal = fs::read(output.join("observations.jsonl")).unwrap();
    for (key, value) in [
        (
            "selection_contract",
            json!("parallel.gp_selection_contract.v0"),
        ),
        ("proposal_cap_per_slot", json!(1)),
    ] {
        let mut changed = request.clone();
        changed[key] = value;
        write_json(&request_path, &changed).unwrap();
        let resumed = command().output().unwrap();
        assert!(
            !resumed.status.success(),
            "{key}: resumed under other rules"
        );
        assert!(
            String::from_utf8_lossy(&resumed.stderr).contains("request changed"),
            "{key}: {}",
            String::from_utf8_lossy(&resumed.stderr)
        );
        // Refused before any new work: the ledger and the fit registry are unchanged.
        assert_eq!(
            fs::read(output.join("observations.jsonl")).unwrap(),
            journal
        );
        assert!(!output.join("search.json").exists());
        assert_eq!(authority.entries().0, 1);
    }
    // The original rules resume and complete; the replay accepts the result.
    fs::write(&request_path, &original).unwrap();
    success(command().output().unwrap());
    selection_replay::verify_search(root, &output).unwrap();
    authority.stop_service();
}

/// Live (non-zombie) helper processes among those a faulting evaluator
/// recorded. Only the fixture's own `sleep 300` helpers are counted, so a
/// reused pid is never mistaken for one.
fn live_left_behind(census: &Path) -> Vec<i32> {
    let Ok(entries) = fs::read_dir(census) else {
        return Vec::new();
    };
    let mut live = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = name
            .strip_suffix(".pid")
            .and_then(|pid| pid.parse::<i32>().ok())
        else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let state = stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().next());
        let helper = fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|cmdline| cmdline.starts_with(b"sleep\0300\0"));
        if helper && state != Some("Z") {
            live.push(pid);
        }
    }
    live.sort_unstable();
    live
}

/// A killed orphan is reaped by init asynchronously: poll briefly.
fn settled_left_behind(census: &Path) -> Vec<i32> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let live = live_left_behind(census);
        if live.is_empty() || Instant::now() >= deadline {
            return live;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A failing test must not leak the helpers it provoked.
struct CensusGuard(PathBuf);

impl Drop for CensusGuard {
    fn drop(&mut self) {
        for pid in live_left_behind(&self.0) {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}

fn recorded(census: &Path) -> usize {
    fs::read_dir(census).map_or(0, |entries| entries.count())
}

fn completion(root: &Path, row: &Value) -> Value {
    crate::label_admission::artifact(root, &row["clean_completion"]).unwrap()
}

#[test]
fn a_blocked_fitter_and_its_helpers_are_killed_at_the_original_deadline() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let census = root.join("census");
    let _guard = CensusGuard(census.clone());
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("blocked");
    let started = Instant::now();
    success(
        timed_search_client(&authority, root, &prepared, &output, 2, 4)
            .env(FAULT, "fork_block")
            .env(CENSUS, &census)
            .output()
            .unwrap(),
    );
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!(recorded(&census), 1, "the fault fixture did not run");
    // No-orphan census: the helper dies with the killed evaluator.
    assert_eq!(
        settled_left_behind(&census),
        Vec::<i32>::new(),
        "a helper outlived its killed evaluator"
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 1, "work was dispatched after the deadline");
    assert_eq!(rows[0]["failure"], "candidate_or_view_timeout");
    let dispatch = completion(root, &rows[0]);
    assert_eq!(dispatch["deadline_exceeded"], true);
    assert_eq!(dispatch["process_group"]["isolated"], true);
    assert_eq!(dispatch["process_group"]["members_after_drain"], 0);
    let search: Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(search["stop_reason"], "search_deadline");
    let replayed = selection_replay::verify_search(root, &output).unwrap();
    assert_eq!(
        replayed.replay.stop,
        selection_replay::StopReason::SearchDeadline
    );
    assert!(replayed.replay.winner.is_none() && search["best_program"].is_null());
    // A publication claiming another terminal disposition is refused.
    let mut forged = search.clone();
    forged["stop_reason"] = json!("evaluation_budget");
    write_json(&output.join("search.json"), &forged).unwrap();
    let error = format!(
        "{:#}",
        selection_replay::verify_search(root, &output).unwrap_err()
    );
    assert!(error.contains("stop reason"), "{error}");
    // The blocked evaluator never reached the authority: nothing started.
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

#[test]
fn helpers_left_by_a_finished_evaluator_are_drained_before_the_next_dispatch() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let census = root.join("census");
    let _guard = CensusGuard(census.clone());
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("leaky");
    success(
        search_client(&authority, root, &prepared, &output, 2)
            .env(FAULT, "fork_exit")
            .env(CENSUS, &census)
            .output()
            .unwrap(),
    );
    assert_eq!(recorded(&census), 2, "the fault fixture did not run");
    assert_eq!(
        settled_left_behind(&census),
        Vec::<i32>::new(),
        "a helper outlived its finished evaluator"
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert!(row["failure"].is_null(), "{row}");
        let dispatch = completion(root, row);
        // Each candidate's helper was found and killed before the next one ran.
        assert_eq!(dispatch["process_group"]["members_at_exit"], 1);
        assert_eq!(dispatch["process_group"]["members_after_drain"], 0);
    }
    selection_replay::verify_search(root, &output).unwrap();
    authority.stop_service();
}

#[test]
fn a_second_searcher_cannot_dispatch_into_a_live_search() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("contended");
    let first = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .env(FAULT, "block")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Wait until the first searcher has reserved and dispatched candidate 0.
    let intent = output.join("candidate-0.attempt.json");
    let deadline = Instant::now() + Duration::from_secs(60);
    while !intent.exists() {
        assert!(Instant::now() < deadline, "first searcher never dispatched");
        std::thread::sleep(Duration::from_millis(20));
    }
    // Identical request, second process, while candidate 0 is still in flight.
    let second = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .output()
        .unwrap();
    let first = first.wait_with_output().unwrap();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(
        !second.status.success(),
        "a concurrent searcher dispatched into a live search"
    );
    assert!(
        String::from_utf8_lossy(&second.stderr).contains("held by another live searcher"),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    // One reservation, one terminal outcome, one publication.
    let journal = read_journal(&output.join("observations.jsonl")).unwrap();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0]["failure"], "candidate_or_view_timeout");
    selection_replay::verify_search(root, &output).unwrap();
    // A completed search stays closed to any later searcher.
    let late = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .output()
        .unwrap();
    assert!(!late.status.success());
    assert!(String::from_utf8_lossy(&late.stderr).contains("already complete"));
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

#[test]
fn an_evaluator_allocation_failure_is_a_contained_failed_row() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("oom");
    success(
        search_client(&authority, root, &prepared, &output, 2)
            .env(FAULT, "oom")
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        let failure = row["failure"].as_str().unwrap();
        assert!(failure.starts_with("worker_exit:"), "{failure}");
        assert!(row["metrics"].is_null());
        assert_eq!(completion(root, row)["outcome"], "infrastructure_failure");
    }
    // No measured candidate: a typed no-result, never a substitute winner.
    let search: Value = read_json(&output.join("search.json")).unwrap();
    assert!(search["best_program"].is_null());
    let replayed = selection_replay::verify_search(root, &output).unwrap();
    assert!(replayed.replay.winner.is_none());
    assert_eq!(
        replayed.replay.stop,
        selection_replay::StopReason::EvaluationBudget
    );
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

/// A completed three-candidate search whose seed population leads with two
/// fittable programs and one deterministic fit failure.
fn handoff_search(authority: &Fixture, root: &Path, prepared: &Path) -> (PathBuf, PathBuf) {
    let seeds = root.join("handoff-seeds.json");
    seed_population(
        prepared,
        &seeds,
        &["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM],
        |_| None,
    );
    let output = root.join("handoff-search");
    success(
        search_client(authority, root, prepared, &output, 3)
            .arg("--seeds")
            .arg(&seeds)
            .output()
            .unwrap(),
    );
    (output, seeds)
}

/// The cycle pipeline's handoff, run as a configured client.
fn export_selected(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    seeds: &Path,
    predictor: &Path,
) -> Command {
    let mut command = authority.command();
    command
        .args(["export-selected", "--root"])
        .arg(root)
        .arg("--output")
        .arg(output)
        .arg("--prepared")
        .arg(prepared)
        .args(["--cut", "0", "--seeds"])
        .arg(seeds)
        .arg("--executable")
        .arg(std::env::current_exe().unwrap())
        .arg("--predictor")
        .arg(predictor);
    command
}

#[test]
fn the_predictor_handoff_exports_only_the_sealed_replayed_selection() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    let winner = selection_replay::verify_search(&root, &output)
        .unwrap()
        .replay
        .winner
        .unwrap();
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let other = rows
        .iter()
        .find(|row| row["failure"].is_null() && row["genome"] != winner.genome.as_str())
        .expect("a second measured program")["genome"]
        .clone();
    // A summary naming another evaluated program is not the selection.
    let search_path = output.join("search.json");
    let original = fs::read(&search_path).unwrap();
    let mut forged: Value = serde_json::from_slice(&original).unwrap();
    forged["best_program"] = other.clone();
    write_json(&search_path, &forged).unwrap();
    let tampered = root.join("tampered.json");
    let refused = export_selected(&authority, &root, &prepared, &output, &seeds, &tampered)
        .output()
        .unwrap();
    assert!(
        !refused.status.success(),
        "the handoff exported the summary's program {other}"
    );
    assert!(!tampered.exists(), "a predictor was published for {other}");
    fs::write(&search_path, &original).unwrap();
    // The honest path exports exactly the replayed winner on its training rows.
    let predictor = root.join("selected.json");
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    let exported: Predictor = read_json(&predictor).unwrap();
    assert_eq!(
        exported.canonical,
        GenomeIr::parse(&winner.genome).unwrap().canonical()
    );
    let seal: Value = read_json(&predictor.with_extension("seal.json")).unwrap();
    assert_eq!(seal["status"], "selected");
    assert_eq!(seal["selected"]["executable_program"], winner.genome);
    let scope = &seal["final_fit_scope"];
    assert_eq!(scope["kind"], handoff::FINAL_FIT_SCOPE);
    assert_eq!(scope["held_out_labels_in_fit"], false);
    assert_eq!(exported.training_sha256, scope["training_sha256"]);
    assert_ne!(
        scope["training_sha256"],
        scope["held_out_validation_sha256"]
    );
    let handoff: Value = read_json(&predictor.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "exported");
    assert_eq!(
        handoff["final_refit"]["transformation"],
        "selected_cut_training_refit_reuses_fold_fit"
    );
    assert_eq!(
        handoff["final_refit"]["model_sha256"],
        seal["selected"]["fold_model"]["sha256"]
    );
    // A resumed cycle re-verifies and reuses the recorded handoff.
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    authority.stop_service();
}

#[test]
fn a_sealed_selection_refuses_swaps_and_keeps_the_selection_when_export_fails() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    let inputs = handoff::Inputs {
        root: &root,
        output: &output,
        seeds: Some(&seeds),
        vocabulary: None,
    };
    let seal_path = root.join("selection.seal.json");
    let seal = handoff::write_seal(&inputs, &seal_path).unwrap();
    assert_eq!(seal["status"], "selected");
    handoff::verify_seal(&inputs, &seal_path).unwrap();
    // Each swap is refused by identity, whatever the predictions would be:
    // another candidate's fitted state or recipe, another scope, another status.
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let position = seal["selected"]["position"].as_u64().unwrap() as usize;
    let other = rows
        .iter()
        .enumerate()
        .find(|(i, row)| *i != position && row["failure"].is_null())
        .expect("a second measured candidate")
        .1;
    let other_completion = completion(&root, other);
    for (pointer, value) in [
        ("/selected/fold_model", other_completion["model"].clone()),
        (
            "/selected/fold_fit_completion",
            other_completion["fit_completion"].clone(),
        ),
        ("/selected/executable_program", other["genome"].clone()),
        ("/final_fit_scope/training_sha256", json!("0".repeat(64))),
        ("/final_fit_scope/held_out_labels_in_fit", json!(true)),
        ("/status", json!("no_measured_candidate")),
    ] {
        let mut swapped = seal.clone();
        *swapped.pointer_mut(pointer).unwrap() = value;
        write_json(&seal_path, &swapped).unwrap();
        let error = format!(
            "{:#}",
            handoff::verify_seal(&inputs, &seal_path).unwrap_err()
        );
        assert!(
            error.contains("differs from the replayed selection"),
            "{pointer}: {error}"
        );
    }
    write_json(&seal_path, &seal).unwrap();
    // A ledger row missing its lineage event cannot be sealed at all.
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(output.join(name)).unwrap())
        .collect();
    let mut missing = rows.clone();
    missing[position].as_object_mut().unwrap().remove("lineage");
    republish_ledger(&output, &missing);
    let error = format!("{:#}", handoff::seal(&inputs).unwrap_err());
    assert!(error.contains("lineage"), "{error}");
    for (name, bytes) in files.iter().zip(&original) {
        fs::write(output.join(name), bytes).unwrap();
    }
    handoff::verify_seal(&inputs, &seal_path).unwrap();
    // A failed export keeps the sealed selection and substitutes nothing.
    let predictor = root.join("failed.json");
    let failed = export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
        .env(FAULT, "fail_export")
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let recorded: Value = read_json(&predictor.with_extension("handoff.json")).unwrap();
    assert_eq!(recorded["status"], "export_failed");
    assert!(
        recorded["failure"]
            .as_str()
            .unwrap()
            .contains("worker exited"),
        "{recorded}"
    );
    assert!(recorded["predictor"].is_null() && !predictor.exists());
    assert_eq!(
        recorded["selected_fitted_state"],
        seal["selected"]["fold_model"]
    );
    assert_eq!(
        recorded["program_sha256"],
        seal["selected"]["program_sha256"]
    );
    // Only the sealed recipe was attempted.
    let receipt: Value = read_json(&predictor.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert_eq!(
        receipt["program_sha256"],
        hash(
            seal["selected"]["executable_program"]
                .as_str()
                .unwrap()
                .as_bytes()
        )
    );
    // The recorded failure is terminal: a retry re-verifies and reports it.
    assert!(
        !export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap()
            .status
            .success()
    );
    authority.stop_service();
}

/// Seed population whose programs equal `base`'s but whose origin labels are
/// all `origin`: only the labels differ.
fn relabelled_population(base: &Path, path: &Path, origin: &str) {
    let mut population: SeedPopulation = read_json(base).unwrap();
    for candidate in &mut population.candidates {
        candidate.origin = origin.into();
    }
    write_json(path, &population).unwrap();
}

#[test]
fn a_paired_comparison_names_its_sole_arm_difference_and_refuses_any_other() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let neural = root.join("seeds-neural.json");
    seed_population(
        &prepared,
        &neural,
        &["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM],
        |_| None,
    );
    let shuffled = root.join("seeds-shuffled.json");
    relabelled_population(&neural, &shuffled, "shuffled_context");
    let run = |name: &str, extra: &[&std::ffi::OsStr]| {
        let output = root.join(name);
        success(
            search_client(&authority, &root, &prepared, &output, 3)
                .args(extra)
                .output()
                .unwrap(),
        );
        output
    };
    let flag = std::ffi::OsStr::new;
    let model = run("arm-model", &[flag("--seeds"), neural.as_os_str()]);
    let relabelled = run("arm-relabelled", &[flag("--seeds"), shuffled.as_os_str()]);
    let generic = run("arm-generic", &[]);
    let reseeded = run(
        "arm-reseeded",
        &[
            flag("--seeds"),
            shuffled.as_os_str(),
            flag("--rng-seed"),
            flag("7"),
        ],
    );
    fn arm<'a>(label: &'a str, output: &'a Path) -> paired::Arm<'a> {
        paired::Arm { label, output }
    }
    // Origin labels are the sole difference: every shared program carries the
    // same measurement, admissibility and budget in both arms.
    let manifest = paired::compare(
        &root,
        &[arm("model", &model), arm("relabelled", &relabelled)],
        &["seeds_sha256"],
    )
    .unwrap();
    assert_eq!(
        manifest["observed_request_differences"],
        json!(["seeds_sha256"])
    );
    assert_eq!(manifest["shared_programs"], 3);
    assert_eq!(
        manifest["shared_programs_with_divergent_outcomes"],
        json!([])
    );
    let arms = manifest["arms"].as_array().unwrap();
    assert_eq!(arms[0]["winner"]["program"], arms[1]["winner"]["program"]);
    assert_eq!(
        arms[0]["winner"]["objective_rmse_original_units"],
        arms[1]["winner"]["objective_rmse_original_units"]
    );
    for key in [
        "completed_evaluations",
        "measured",
        "failed",
        "proposal_draws",
    ] {
        assert_eq!(arms[0]["counters"][key], arms[1]["counters"][key], "{key}");
    }
    assert_eq!(arms[0]["generation_zero"]["origins"]["neural_policy"], 3);
    assert_eq!(arms[1]["generation_zero"]["origins"]["shuffled_context"], 3);
    // The second arm reused the first arm's fits: counted apart, not hidden.
    assert_eq!(arms[0]["counters"]["recorded_physical_fit_starts"], 2);
    assert_eq!(arms[1]["counters"]["recorded_physical_fit_starts"], 0);
    assert_eq!(arms[1]["counters"]["reused_completions"], 2);
    // The deterministic fit failure has no recorded start outcome in either arm.
    assert_eq!(arms[0]["counters"]["unknown_start_outcomes"], 1);
    assert!(manifest["feasibility_protocol"].is_string());
    // Seeded against generic: the seed population is the declared difference.
    paired::compare(
        &root,
        &[arm("model", &model), arm("generic", &generic)],
        &["seeds_sha256"],
    )
    .unwrap();
    // Undeclared, overdeclared and empty declarations are refused.
    for (arms, declared, expected) in [
        (
            vec![arm("model", &model), arm("reseeded", &reseeded)],
            vec!["seeds_sha256"],
            "not exactly the declared",
        ),
        (
            vec![arm("model", &model), arm("relabelled", &relabelled)],
            vec!["seeds_sha256", "rng_seed"],
            "not exactly the declared",
        ),
        (
            vec![arm("model", &model), arm("relabelled", &relabelled)],
            vec![],
            "must declare",
        ),
        (
            vec![arm("model", &model), arm("model", &relabelled)],
            vec!["seeds_sha256"],
            "unique",
        ),
    ] {
        let error = format!(
            "{:#}",
            paired::compare(&root, &arms, &declared).unwrap_err()
        );
        assert!(error.contains(expected), "{declared:?}: {error}");
    }
    let manifest = paired::compare(
        &root,
        &[arm("model", &model), arm("reseeded", &reseeded)],
        &["seeds_sha256", "rng_seed"],
    )
    .unwrap();
    assert_eq!(
        manifest["observed_request_differences"],
        json!(["rng_seed", "seeds_sha256"])
    );
    // A shared program whose measurement differs between arms is refused.
    let rows: Vec<Value> = read_json(&relabelled.join("observations.json")).unwrap();
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(relabelled.join(name)).unwrap())
        .collect();
    let measured = rows
        .iter()
        .position(|row| row["failure"].is_null())
        .unwrap();
    let mut forged = rows.clone();
    forged[measured]["metrics"]["rmse"] = json!(0.5);
    republish_ledger(&relabelled, &forged);
    assert!(paired::compare(
        &root,
        &[arm("model", &model), arm("relabelled", &relabelled)],
        &["seeds_sha256"],
    )
    .is_err());
    for (name, bytes) in files.iter().zip(&original) {
        fs::write(relabelled.join(name), bytes).unwrap();
    }
    authority.stop_service();
}

#[test]
fn a_refused_proposal_in_one_slot_does_not_shift_another_slots_random_streams() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let run = |seeds: &Path, name: &str| -> Vec<Value> {
        let output = root.join(name);
        success(
            search_client(&authority, &root, &prepared, &output, 3)
                .arg("--seeds")
                .arg(seeds)
                .output()
                .unwrap(),
        );
        read_json(&output.join("observations.json")).unwrap()
    };
    // Arm B: slot 1's initial proposal duplicates slot 0 and is refused, so
    // slot 1 is filled by a variation draw. In both arms slot 2's initial
    // proposal is a duplicate too, so slot 2 is filled by variation.
    let b = root.join("streams-b.json");
    seed_population(
        &prepared,
        &b,
        &[PROGRAM, "(pipe X   (ridge))", "(pipe X  (ridge))"],
        |_| None,
    );
    let rows_b = run(&b, "streams-arm-b");
    assert_eq!(rows_b[1]["lineage"]["source"]["kind"], "variation");
    // Arm A seeds slot 1 with exactly the program arm B's variation produced:
    // both arms enter slot 2 with the same ledger, parent pool and evaluated
    // set, and differ only by one refused draw in slot 1.
    let a = root.join("streams-a.json");
    let slot_one = rows_b[1]["genome"].as_str().unwrap();
    seed_population(
        &prepared,
        &a,
        &[PROGRAM, slot_one, "(pipe X  (ridge))"],
        |_| None,
    );
    let rows_a = run(&a, "streams-arm-a");
    assert_eq!(
        rows_a[1]["lineage"]["source"]["kind"],
        "seed_population_slot"
    );
    for row in 0..2 {
        assert_eq!(rows_a[row]["genome"], rows_b[row]["genome"]);
        for key in ["rmse", "semantic_sha256"] {
            assert_eq!(rows_a[row]["metrics"][key], rows_b[row]["metrics"][key]);
        }
    }
    let counter = |row: &Value| row["proposal_attempts"].as_u64().unwrap();
    assert!(
        counter(&rows_b[1]) > counter(&rows_a[1]),
        "arm B drew more proposals before slot 2"
    );
    // Slot 2 walks the same keyed stream sequence in both arms: the same
    // accepted draw, words, operator, parents and program.
    let slot_two = |rows: &[Value]| {
        let source = &rows[2]["lineage"]["source"];
        json!({"draw":counter(&rows[2]) - counter(&rows[1]) - 1,
            "kind":source["kind"],"operator":source["operator"],"parents":source["parents"],
            "variation_seed":source["variation_seed"],"operator_word":source["operator_word"],
            "genome":rows[2]["genome"]})
    };
    assert_eq!(
        slot_two(&rows_a),
        slot_two(&rows_b),
        "a refused proposal in slot 1 shifted slot 2's random streams"
    );
    for rows in [&rows_a, &rows_b] {
        let source = &rows[2]["lineage"]["source"];
        assert_eq!(source["kind"], "variation", "{source}");
        assert_eq!(source["slot"], 2, "{source}");
        assert_eq!(source["slot_draw"], slot_two(rows)["draw"], "{source}");
    }
    authority.stop_service();
}

/// Rewrite the fixture preparation's family partition (identity recomputed by
/// every consumer, as a producer of that partition would publish it).
fn set_partition(prepared: &Path, partition: &str) {
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["meta_partition"] = json!(partition);
    write_json(&prepared.join("prepared.json"), &p).unwrap();
}

/// W1-T04, consumer-map route R13: `--qualification` admitted a search of a
/// sealed `meta_test` family. Its observations are teacher/learning inputs, so
/// the sealed family would reach learning through the search. The refusal
/// precedes every view load, output reservation and fit claim.
#[test]
fn qualification_search_of_a_sealed_meta_test_family_is_refused_before_any_view_or_fit() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    set_partition(&prepared, "meta_test");
    let sealed = root.join("search-sealed-family");
    let refused = search_client(&authority, &root, &prepared, &sealed, 1)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&refused.stderr);
    assert!(
        !refused.status.success(),
        "a sealed meta_test family was searched under --qualification"
    );
    assert!(error.contains("sealed meta_test"), "{error}");
    assert!(
        !sealed.exists(),
        "a refused search left search state behind"
    );
    authority.assert_no_fit_starts();
    // Positive control: the identical view as a meta-validation family is
    // still admitted for qualification (adaptive development evidence).
    set_partition(&prepared, "meta_validation");
    let admitted = root.join("search-validation-family");
    success(
        search_client(&authority, &root, &prepared, &admitted, 1)
            .output()
            .unwrap(),
    );
    let search: Value = read_json(&admitted.join("search.json")).unwrap();
    assert_eq!(search["final_tests_opened"], false);
    assert_eq!(authority.entries(), (1, 1));
    authority.stop_service();
}

/// The view fixture's preparation over its own backing file, with the given
/// raw label cells replaced. Cut 0 trains on rows 0..8 and validates on 8, 9;
/// rows 10 and 11 sit in the backing file outside every membership, where a
/// shared source keeps rows no stage may use.
fn label_variant(root: &Path, name: &str, labels: &[(usize, &str)]) -> PathBuf {
    fs::create_dir_all(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let mut raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    for (row, label) in labels {
        raw[*row][1] = label.to_string();
    }
    let backing = format!("data/originals/{name}.csv");
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join(&backing), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared").join(name);
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let va = csv_bytes(&pp.transform(&raw[8..10]).unwrap()).unwrap();
    let p = json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example",
        "task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,
        "clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),
        "original_test_sha256":"1".repeat(64),"selected_train_indices":(0..10).collect::<Vec<_>>(),
        "selected_test_indices":[0,1],"target_last":true,"rows_total":12,
        "cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],
            "training_sha256":hash(&tr),"validation_sha256":hash(&va),
            "preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],
        "storage":{"source":backing,"source_sha256":file_hash(&root.join(&backing)).unwrap(),
            "feature_columns":["x"],"target_column":"target",
            "original_training_row_map":(0..10).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    prepared
}

/// W1-T06 at the search: what one search reads, fits, proposes and scores.
/// Backing rows outside every membership are forbidden parents: proposals,
/// fitting inputs and scores are identical and every fit is reused. The
/// cut's held-out labels are its authorized selection labels: fitting inputs
/// stay identical (fits reused) while scores change. A training label is an
/// authorized fitting input: the fitted subject changes.
#[test]
fn search_proposals_fits_and_scores_depend_only_on_their_permitted_rows() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let variants = [
        ("baseline", vec![]),
        (
            "unreferenced-backing-rows-changed",
            vec![(10, "900"), (11, "-7")],
        ),
        ("heldout-labels-changed", vec![(8, "150"), (9, "95.5")]),
        ("training-label-changed", vec![(3, "140")]),
    ];
    let prepared: Vec<PathBuf> = variants
        .iter()
        .map(|(name, labels)| label_variant(&root, name, labels))
        .collect();
    let training: BTreeSet<String> = prepared.iter().map(|p| cut_training_hash(p)).collect();
    assert_eq!(
        training.len(),
        2,
        "only the training-label variant changes its fitting rows"
    );
    let mut authority = Fixture::activated(&root, training.into_iter().collect());
    authority.start();
    // Duplicate initial proposals force variation draws whose parents are
    // chosen from measured candidates: later proposals depend on scores.
    let run = |index: usize| -> (Vec<Value>, i64) {
        let (name, _) = &variants[index];
        let seeds = root.join(format!("seeds-{name}.json"));
        seed_population(
            &prepared[index],
            &seeds,
            &[
                PROGRAM,
                "(pipe X   (ridge))",
                "(pipe X (standardize) (ridge))",
                "(pipe X  (ridge))",
            ],
            |_| None,
        );
        let output = root.join(format!("search-{name}"));
        success(
            search_client(&authority, &root, &prepared[index], &output, 4)
                .arg("--seeds")
                .arg(&seeds)
                .output()
                .unwrap(),
        );
        let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        (rows, authority.entries().0)
    };
    // Permitted identifier differences: `prepared_sha256` and the seed file's
    // `seeds_sha256` name the (changed) preparation; they are not compared.
    let project = |rows: &[Value], key: &str| -> Vec<Value> {
        rows.iter()
            .map(|row| match key {
                "proposals" => {
                    let mut source = row["lineage"]["source"].clone();
                    if let Some(source) = source.as_object_mut() {
                        source.remove("seeds_sha256");
                    }
                    json!([row["genome"], source])
                }
                "fitting_inputs" => json!([
                    row["training_sha256"],
                    row["metrics"]["score_identity"]["fit_key"]
                ]),
                "fitted_state" => row["metrics"]["score_identity"]["model_sha256"].clone(),
                "scores" => json!([row["failure"].is_null(), row["metrics"]["rmse"]]),
                "predictions" => row["metrics"]["semantic_sha256"].clone(),
                _ => unreachable!(),
            })
            .collect()
    };
    let keys = [
        "proposals",
        "fitting_inputs",
        "fitted_state",
        "scores",
        "predictions",
    ];
    let (baseline, fits) = run(0);
    assert!(
        baseline
            .iter()
            .filter(|row| row["failure"].is_null())
            .count()
            >= 2
    );
    assert!(baseline[1]["lineage"]["source"]["kind"] == "variation");
    // (variant, per-key expectation: Some(true) equal, Some(false) differs at
    // slot 0, None declared; whether the fit registry must grow). A changed
    // score may steer later variation to new programs, so the held-out
    // variant's proposals and registry growth are recorded, not asserted.
    let expectations = [
        (
            1usize,
            [Some(true), Some(true), Some(true), Some(true), Some(true)],
            Some(false),
        ),
        (2, [None, None, None, Some(false), None], None),
        (3, [None, Some(false), Some(false), None, None], Some(true)),
    ];
    for (index, expected, new_fits) in expectations {
        let before_fits = authority.entries().0;
        let (rows, after_fits) = run(index);
        let mut comparisons = Vec::new();
        for (key, expect) in keys.iter().zip(expected) {
            let (a, b) = (project(&baseline, key), project(&rows, key));
            // The first slot is the same program in every variant.
            let (a0, b0) = (&a[0], &b[0]);
            match expect {
                Some(true) => assert_eq!(a, b, "{}: {key} changed", variants[index].0),
                Some(false) => assert_ne!(a0, b0, "{}: {key} ignored its input", variants[index].0),
                None => {}
            }
            comparisons.push(json!({"artifact":key,"expected":match expect {
                    Some(true) => "equal", Some(false) => "differs (slot 0)", None => "declared"},
                "observed":if a == b {"equal"} else {"differs"},"baseline":a,"mutated":b}));
        }
        if index == 2 {
            // Selection labels are not fitting inputs: slot 0 reuses its fit.
            assert_eq!(
                project(&baseline, "fitting_inputs")[0],
                project(&rows, "fitting_inputs")[0]
            );
            assert_eq!(rows[0]["metrics"]["evaluation_origin"], "reuse");
        }
        if let Some(grows) = new_fits {
            assert_eq!(
                after_fits > before_fits,
                grows,
                "{}: fit registry growth {before_fits} -> {after_fits}",
                variants[index].0
            );
        }
        if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
            let dir = Path::new(&dir).join("counterfactual");
            fs::create_dir_all(&dir).unwrap();
            write_json(
                &dir.join(format!("search-{}.json", variants[index].0)),
                &json!({"schema":"jope.w1-counterfactual-receipt/v1","task":"W1-T06",
                    "mutation":format!("search-{}", variants[index].0),
                    "perturbed":variants[index].1.iter().map(|(r, l)| json!({"backing_row":r,"label":l})).collect::<Vec<_>>(),
                    "frozen":"engine source, seed population programs, rng seed, evaluator, budget (4 evaluations), cut membership",
                    "tolerance":"exact equality of recorded identities and f64 bits; fits are single-threaded fixed-seed",
                    "permitted_identifier_differences":["prepared_sha256","lineage.source.seeds_sha256"],
                    "comparisons":comparisons,"fit_registry_entries":{"before":before_fits,"after":after_fits},
                    "receiver":"gp::search_seeded_with_rng_in_root through the fixture CLI"}),
            )
            .unwrap();
        }
    }
    assert!(fits > 0);
    authority.stop_service();
}

/// Raw-unit sealed final-test rows for the view fixture (`x`, `target`).
const SEALED_TEST: &str = "12,136\n13,139\n14,142.5\n";

/// A completed search of the view fixture and its exported selected predictor.
fn exported_subject(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
) -> (PathBuf, PathBuf, PathBuf) {
    let (output, seeds) = handoff_search(authority, root, prepared);
    let predictor = root.join("selected.json");
    success(
        export_selected(authority, root, prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    (output, seeds, predictor)
}

/// W1-T04, consumer-map route R14: `score` hashed whatever test file the
/// caller named against the caller's own selection receipt, so any label file
/// could be scored — it was never bound to the preparation's sealed-test
/// identity — and the metrics were published straight to the caller.
#[test]
fn a_selection_receipt_alone_neither_reads_caller_chosen_labels_nor_publishes_a_score() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (_, _, predictor) = exported_subject(&authority, &root, &prepared);
    authority.stop_service();
    // Caller-chosen labels: not the preparation's sealed identity ("1"*64).
    let chosen = root.join("caller-chosen.csv");
    fs::write(&chosen, SEALED_TEST).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_ne!(file_hash(&chosen).unwrap(), p.original_test_sha256);
    let evidence = root.join("selection-evidence.json");
    write_json(
        &evidence,
        &json!({"predictor_sha256":file_hash(&predictor).unwrap(),"partition":"meta_validation"}),
    )
    .unwrap();
    let receipt = root.join("selection.json");
    write_json(
        &receipt,
        &SelectionReceipt {
            schema: "parallel.selection.v1".into(),
            predictor_sha256: file_hash(&predictor).unwrap(),
            final_test_sha256: file_hash(&chosen).unwrap(),
            selection_evidence_path: evidence.clone(),
            selection_evidence_sha256: file_hash(&evidence).unwrap(),
            selected_on: "meta_validation".into(),
            frozen: true,
        },
    )
    .unwrap();
    let output = root.join("outer-score.json");
    let scored = score(&predictor, &chosen, &receipt, &output);
    assert!(
        scored.is_err(),
        "scored caller-chosen labels without a release grant: {}",
        fs::read_to_string(&output).unwrap_or_default()
    );
    let error = format!("{:#}", scored.unwrap_err());
    assert!(error.contains("outer release"), "{error}");
    assert!(!output.exists(), "a refused score was published");
}

/// Copy a retained receipt to `$JOPE_W1_RECEIPTS/<group>/` when set.
fn retain(group: &str, name: &str, path: &Path) {
    if let Some(dir) = std::env::var_os("JOPE_W1_RECEIPTS") {
        let dir = Path::new(&dir).join(group);
        fs::create_dir_all(&dir).unwrap();
        fs::copy(path, dir.join(name)).unwrap();
    }
}

fn refusal<T: std::fmt::Debug>(result: Result<T>) -> String {
    format!(
        "{:#}",
        result.expect_err("the release authority accepted a request it must refuse")
    )
}

/// W1-T04 acceptance: sealed labels reach only a granted outer scorer; the
/// score stays withheld until the separate authority releases it, once.
#[test]
fn outer_scores_need_a_grant_read_only_the_granted_labels_and_release_once() {
    use outer_release::{Authority as Release, GrantRequest};
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    // The preparation binds its sealed-test identity; the file itself does
    // not exist while search, selection and export run (denied upstream).
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let sealed = root.join("data/originals/sealed-test.csv");
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    assert!(!sealed.exists());
    let fits = authority.entries();
    fs::write(&sealed, SEALED_TEST).unwrap();
    let outer_dir = root.join("outer-authority");
    let outer = Release::initialize(&outer_dir).unwrap();
    let search = handoff::Inputs {
        root: &root,
        output: &output,
        seeds: Some(&seeds),
        vocabulary: None,
    };
    // Grants: a symlinked label path and an unexported subject are refused.
    let link = root.join("data/originals/sealed-link.csv");
    std::os::unix::fs::symlink(&sealed, &link).unwrap();
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &predictor,
        test: &link,
    }));
    assert!(error.contains("symlink"), "{error}");
    let orphan = root.join("unexported.json");
    fs::copy(&predictor, &orphan).unwrap();
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &orphan,
        test: &sealed,
    }));
    assert!(error.contains("handoff"), "{error}");
    let grant = outer
        .grant(&GrantRequest {
            search,
            predictor: &predictor,
            test: &sealed,
        })
        .unwrap();
    let granted: Value = read_json(&grant).unwrap();
    assert_eq!(granted["scoring_labels"]["role"], "sealed_final");
    assert_eq!(
        granted["scoring_labels"]["path"],
        "data/originals/sealed-test.csv"
    );
    assert_eq!(
        granted["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes())
    );
    assert_eq!(
        granted["subject"]["predictor"]["sha256"],
        file_hash(&predictor).unwrap()
    );
    let error = refusal(outer.grant(&GrantRequest {
        search,
        predictor: &predictor,
        test: &sealed,
    }));
    assert!(error.contains("already granted"), "{error}");

    // Refused label reads, each before the first label byte is read.
    let receipts = root.join("outer-receipts");
    let mut forged = granted.clone();
    forged["scoring_labels"]["path"] = json!("data/originals/raw.csv");
    let forged_path = root.join("forged-grant.json");
    write_json(&forged_path, &forged).unwrap();
    let refused_forged = receipts.join("refused-forged-grant.json");
    let error = refusal(outer.score(&forged_path, &predictor, &sealed, &refused_forged));
    assert!(error.contains("not recorded"), "{error}");
    let other_model = root.join("other-model.json");
    let reencoded: Value = read_json(&predictor).unwrap();
    fs::write(&other_model, serde_json::to_vec(&reencoded).unwrap()).unwrap();
    let refused_model = receipts.join("refused-other-model.json");
    let error = refusal(outer.score(&grant, &other_model, &sealed, &refused_model));
    assert!(error.contains("not the granted fitted subject"), "{error}");
    let copy = root.join("data/originals/sealed-copy.csv");
    fs::write(&copy, SEALED_TEST).unwrap();
    let refused_copy = receipts.join("refused-other-label-file.json");
    let error = refusal(outer.score(&grant, &predictor, &copy, &refused_copy));
    assert!(error.contains("not the granted one"), "{error}");
    for path in [&refused_forged, &refused_model, &refused_copy] {
        let receipt: Value = read_json(path).unwrap();
        assert_eq!(receipt["status"], "refused", "{receipt}");
        assert_eq!(receipt["labels_opened"], false, "{receipt}");
    }

    // The valid score, through the shipped `score` entry: withheld.
    let scored_receipt = receipts.join("scored-withheld.json");
    success(
        authority
            .command()
            .args(["score", "--model"])
            .arg(&predictor)
            .arg("--test")
            .arg(&sealed)
            .arg("--receipt")
            .arg(&grant)
            .arg("--output")
            .arg(&scored_receipt)
            .env(outer_release::AUTHORITY_ENV, &outer_dir)
            .output()
            .unwrap(),
    );
    let scored: Value = read_json(&scored_receipt).unwrap();
    assert_eq!(scored["status"], "scored_withheld");
    assert_eq!(scored["released"], false);
    assert!(!fs::read_to_string(&scored_receipt)
        .unwrap()
        .contains("rmse"));
    // A retry cannot obtain a second look.
    let refused_retry = receipts.join("refused-retry.json");
    let error = refusal(outer.score(&grant, &predictor, &sealed, &refused_retry));
    assert!(error.contains("consumed"), "{error}");

    // Release, once; the released metrics are the granted labels' metrics.
    let grant_id = granted["grant_id"].as_str().unwrap();
    let visible = root.join("released-outer-score.json");
    outer.release(grant_id, &visible).unwrap();
    let released: Value = read_json(&visible).unwrap();
    let fitted: Predictor = read_json(&predictor).unwrap();
    let rows: Vec<(&str, f64)> = SEALED_TEST
        .lines()
        .map(|line| {
            let (x, y) = line.split_once(',').unwrap();
            (x, y.parse().unwrap())
        })
        .collect();
    let x: Vec<Vec<f64>> = rows
        .iter()
        .map(|(x, _)| fitted.preprocessing.features(&[x.to_string()]).unwrap())
        .collect();
    let predicted = fitted.model.predict(&x);
    let squared = rows
        .iter()
        .zip(&predicted)
        .map(|((_, y), p)| {
            (y - (p * fitted.preprocessing.target_scale + fitted.preprocessing.target_mean)).powi(2)
        })
        .sum::<f64>();
    let rmse = (squared / rows.len() as f64).sqrt();
    let reported = released["metrics"]["rmse"].as_f64().unwrap();
    assert!(
        (reported - rmse).abs() <= 1e-12 * rmse.abs().max(1.0),
        "released RMSE {reported} is not the granted labels' RMSE {rmse}"
    );
    assert_eq!(released["scoring_labels"]["rows"], 3);
    assert_eq!(released["per_row_values_included"], false);
    assert_eq!(
        released["preparation"]["clean_status"], p["clean_status"],
        "a released score must carry its preparation's evaluation status"
    );
    let again = root.join("released-again.json");
    let error = refusal(outer.release(grant_id, &again));
    assert!(error.contains("already released"), "{error}");
    assert!(!again.exists());

    // The append-only history records every decision; only the one granted
    // attempt ever opened sealed labels. No fit was started for scoring.
    let events = outer.events().unwrap();
    let kinds: Vec<&str> = events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "grant_refused",
            "grant_refused",
            "granted",
            "grant_refused",
            "score_refused",
            "score_refused",
            "score_refused",
            "score_started",
            "scored",
            "score_refused",
            "released",
            "release_refused"
        ]
    );
    assert!(events
        .iter()
        .filter(|event| event["event"] == "score_refused")
        .all(|event| event["labels_opened"] == false));
    assert_eq!(authority.entries(), fits);
    let ledger = outer_dir.join("ledger.jsonl");
    for (name, path) in [
        ("ledger.jsonl", &ledger),
        ("grant.json", &grant),
        ("scored-withheld-receipt.json", &scored_receipt),
        ("released-outer-score.json", &visible),
        ("refused-label-read-forged-grant.json", &refused_forged),
        ("refused-label-read-other-file.json", &refused_copy),
        ("refused-retry.json", &refused_retry),
    ] {
        retain("outer-release", name, path);
    }
    // Removing a recorded event breaks the chain: the authority refuses to open.
    let original = fs::read(&ledger).unwrap();
    let kept: Vec<u8> = original
        .split_inclusive(|b| *b == b'\n')
        .enumerate()
        .filter(|(i, _)| *i != 10)
        .flat_map(|(_, line)| line.iter().copied())
        .collect();
    fs::write(&ledger, kept).unwrap();
    assert!(refusal(Release::open(&outer_dir)).contains("chain"));
    fs::write(&ledger, original).unwrap();
    Release::open(&outer_dir).unwrap();
    authority.stop_service();
}

/// The shipped `outer-release` clap command, parsed from argv and run by the
/// same `outer_release_cli::run` the binary calls. The shipped binary cannot
/// produce an exported subject (every fit is fenced), so its positive path is
/// proven here; `tests/outer_release_cli.rs` proves the binary's wiring.
#[derive(clap::Parser)]
#[command(name = "jope-parallel")]
struct OuterReleaseArgv {
    #[command(subcommand)]
    command: OuterReleaseVerb,
}
#[derive(clap::Subcommand)]
enum OuterReleaseVerb {
    OuterRelease {
        #[command(subcommand)]
        command: outer_release_cli::OuterReleaseCommand,
    },
}
fn outer_release_cli_run(argv: &[&str]) -> Result<Value> {
    use clap::Parser;
    let OuterReleaseVerb::OuterRelease { command } = OuterReleaseArgv::try_parse_from(
        std::iter::once("jope-parallel").chain(argv.iter().copied()),
    )?
    .command;
    outer_release_cli::run(command)
}

/// W1 follow-up 2b: init, grant, score, release and events through the CLI
/// verbs, with the same one-grant, one-attempt, one-release rules.
#[test]
fn the_outer_release_cli_grants_scores_and_releases_once() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    authority.stop_service();
    let sealed = root.join("data/originals/sealed-test.csv");
    fs::write(&sealed, SEALED_TEST).unwrap();
    let dir = root.join("outer-cli-authority");
    let s = |p: &Path| p.to_str().unwrap().to_owned();
    let (a, r) = (s(&dir), s(&root));
    let init = outer_release_cli_run(&["outer-release", "init", "--authority", &a]).unwrap();
    assert_eq!(init["instance"].as_str().map(str::len), Some(64));
    let grant_args = [
        "outer-release",
        "grant",
        "--authority",
        &a,
        "--root",
        &r,
        "--search-output",
        &s(&output),
        "--seeds",
        &s(&seeds),
        "--predictor",
        &s(&predictor),
        "--test",
        &s(&sealed),
    ];
    let granted = outer_release_cli_run(&grant_args).unwrap();
    let grant_id = granted["grant_id"].as_str().unwrap().to_owned();
    let error = format!("{:#}", outer_release_cli_run(&grant_args).unwrap_err());
    assert!(error.contains("already granted"), "{error}");
    let receipt = root.join("cli-scored.json");
    let score = |receipt: &Path| {
        outer_release_cli_run(&[
            "outer-release",
            "score",
            "--authority",
            &a,
            "--grant",
            granted["grant"].as_str().unwrap(),
            "--model",
            &s(&predictor),
            "--test",
            &s(&sealed),
            "--receipt",
            &s(receipt),
        ])
    };
    let scored = score(&receipt).unwrap();
    assert_eq!(scored["status"], "scored_withheld");
    assert!(!fs::read_to_string(&receipt).unwrap().contains("rmse"));
    assert!(score(&root.join("cli-retry.json")).is_err());
    let visible = root.join("cli-released.json");
    let release = [
        "outer-release",
        "release",
        "--authority",
        &a,
        "--grant-id",
        &grant_id,
        "--output",
        &s(&visible),
    ];
    outer_release_cli_run(&release).unwrap();
    let released: Value = read_json(&visible).unwrap();
    assert!(released["metrics"]["rmse"].as_f64().unwrap().is_finite());
    assert!(outer_release_cli_run(&release).is_err());
    let events = outer_release_cli_run(&["outer-release", "events", "--authority", &a]).unwrap();
    let kinds: Vec<&str> = events["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "granted",
            "grant_refused",
            "score_started",
            "scored",
            "score_refused",
            "released",
            "release_refused"
        ]
    );
    // An unknown verb or a missing required flag is a parse refusal, not a call.
    assert!(outer_release_cli_run(&["outer-release", "publish", "--authority", &a]).is_err());
    assert!(outer_release_cli_run(&["outer-release", "grant", "--authority", &a]).is_err());
}

/// W1 follow-up 2c (consumer-map route R2 at the export worker): the fitted
/// predictor export worker loaded its own view from the prepared path and
/// resolved the backing table under the nearest ancestor holding
/// `manifests/` and `Cargo.toml` - not under the search's declared root. The
/// parent must project the fitting-only input under the declared root and
/// hand the worker only that sealed projection.
#[test]
fn the_export_worker_receives_its_projected_input_and_never_resolves_the_view_itself() {
    let temp = tempfile::tempdir().unwrap();
    let outer = temp.path().canonicalize().unwrap();
    let root = outer.join("declared");
    fs::create_dir_all(&root).unwrap();
    let prepared = view_fixture_in(&root, "example", &[8, 9]);
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    // The declared root is a relocated copy without workspace markers, and an
    // ancestor is a workspace holding a byte-identical backing table.
    fs::remove_dir_all(root.join("manifests")).unwrap();
    fs::remove_file(root.join("Cargo.toml")).unwrap();
    let backing = root.join("data/originals/raw.csv");
    let decoy = outer.join("data/originals/raw.csv");
    fs::create_dir_all(outer.join("manifests")).unwrap();
    fs::write(outer.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(decoy.parent().unwrap()).unwrap();
    fs::copy(&backing, &decoy).unwrap();

    // The backing table exists only outside the declared root: the export
    // must refuse rather than read it there.
    let away = root.join("raw.csv.away");
    fs::rename(&backing, &away).unwrap();
    let outside = root.join("exported-outside.json");
    let refused = export_selected(&authority, &root, &prepared, &output, &seeds, &outside)
        .output()
        .unwrap();
    assert!(
        !refused.status.success(),
        "the export read its fitting view from outside the declared root"
    );
    assert!(!outside.exists());
    let handoff: Value = read_json(&outside.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "export_failed");
    let receipt: Value = read_json(&outside.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(receipt["worker_argv"].as_array().is_none_or(Vec::is_empty));

    // The backing table only inside the declared root, the decoy ancestor
    // still a workspace without it: the export succeeds, and its worker was
    // handed the sealed projection and no prepared path.
    fs::rename(&away, &backing).unwrap();
    fs::remove_file(&decoy).unwrap();
    let inside = root.join("exported.json");
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &inside)
            .output()
            .unwrap(),
    );
    let receipt: Value = read_json(&inside.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["worker_input"]["kind"], "sealed_memory_projection");
    assert_eq!(receipt["artifact_root"], json!(root));
    let argv = receipt["worker_argv"].as_array().unwrap();
    assert_eq!(argv[0], "fit-memory");
    assert!(argv
        .iter()
        .all(|a| !a.as_str().unwrap().contains("prepared")));
    let handoff: Value = read_json(&inside.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "exported");
    let exported: Predictor = read_json(&inside).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(exported.training_sha256, p.cuts[0].training_sha256);
    authority.stop_service();
}

// ---------------------------------------------------------------------------
// Review of #111 (W1-T04, route R8 at the outer release authority): every
// grant field, every recorded digest and every consumed object comes from
// exactly the bytes the authority verified or wrote. The test-only
// instrumentation point `manifest::after_verification` replaces a file
// between its check (or its write) and its consumption.
// ---------------------------------------------------------------------------

const OTHER_TEST: &str = "12,1000\n13,1000\n14,1000\n";

/// An exported selected subject of the view fixture, the sealed labels its
/// preparation binds (written after export), and another label file.
struct OuterCase {
    root: PathBuf,
    prepared: PathBuf,
    output: PathBuf,
    seeds: PathBuf,
    predictor: PathBuf,
    sealed: PathBuf,
    other: PathBuf,
    // Rust drops fields in declaration order: stop the service before its files.
    _authority: Fixture,
    _temp: tempfile::TempDir,
}

fn outer_case() -> OuterCase {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    let sealed = root.join("data/originals/sealed-test.csv");
    fs::write(&sealed, SEALED_TEST).unwrap();
    let other = root.join("data/originals/other-labels.csv");
    fs::write(&other, OTHER_TEST).unwrap();
    OuterCase {
        root,
        prepared,
        output,
        seeds,
        predictor,
        sealed,
        other,
        _authority: authority,
        _temp: temp,
    }
}

impl OuterCase {
    fn release(&self, name: &str) -> outer_release::Authority {
        outer_release::Authority::initialize(&self.root.join(name)).unwrap()
    }
    fn grant(&self, name: &str, test: &Path) -> Result<Value> {
        let path = self.release(name).grant(&outer_release::GrantRequest {
            search: handoff::Inputs {
                root: &self.root,
                output: &self.output,
                seeds: Some(&self.seeds),
                vocabulary: None,
            },
            predictor: &self.predictor,
            test,
        })?;
        read_json(&path)
    }
    fn handoff(&self) -> PathBuf {
        self.predictor.with_extension("handoff.json")
    }
}

/// How often a receiver reports `target` verified while `run` runs.
fn verifications(target: &Path, run: impl FnOnce()) -> usize {
    let target = target.canonicalize().unwrap();
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = std::rc::Rc::clone(&count);
    let _hook = crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen.set(seen.get() + 1);
        }
    });
    run();
    count.get()
}

/// Replace `target` (new inode, via rename) with `bytes` at its `nth` reported
/// verification: the last one, as counted by an identical run.
fn swap_at(target: &Path, nth: usize, bytes: Vec<u8>) -> crate::manifest::verification_hook::Guard {
    assert!(nth > 0, "the receiver never reported {target:?} verified");
    let target = target.canonicalize().unwrap();
    let mut seen = 0;
    crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen += 1;
            if seen == nth {
                let staged = target.with_extension("swap");
                fs::write(&staged, &bytes).unwrap();
                fs::rename(&staged, &target).unwrap();
            }
        }
    })
}

/// The reviewed finding [P1]: `grant_body` hashed `prepared.json` against the
/// sealed selection, then reopened it. A preparation substituted between the
/// two supplied the grant's sealed-label identity and metadata.
#[test]
fn an_outer_grant_takes_its_labels_and_metadata_from_the_verified_preparation() {
    let case = outer_case();
    let prepared = case.prepared.join("prepared.json");
    let bound: Value = read_json(&prepared).unwrap();
    // Positive control: an honest grant binds the preparation's sealed labels.
    let mut honest = Value::Null;
    let nth = verifications(&prepared, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(
        honest["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes())
    );
    assert_eq!(honest["preparation"]["dataset"], bound["dataset"]);
    // A preparation naming another label file and other metadata replaces
    // the verified one right after its check.
    let mut substituted = bound.clone();
    substituted["original_test_sha256"] = json!(hash(OTHER_TEST.as_bytes()));
    substituted["dataset"] = json!("substituted-dataset");
    substituted["family"] = json!("substituted-family");
    substituted["clean_status"] = json!("substituted");
    let _hook = swap_at(
        &prepared,
        nth,
        serde_json::to_vec_pretty(&substituted).unwrap(),
    );
    let granted = case.grant("swapped", &case.other).unwrap();
    assert_eq!(
        granted["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes()),
        "a grant was issued for labels the sealed preparation never bound: {granted}"
    );
    for key in ["dataset", "family", "clean_status"] {
        assert_eq!(granted["preparation"][key], bound[key], "{key}: {granted}");
    }
    // The scorer checks the named file against the verified identity, so the
    // substituted labels are never parsed.
    let outer = outer_release::Authority::open(&case.root.join("swapped")).unwrap();
    let grant_path = case
        .root
        .join("swapped/grants")
        .join(format!("{}.json", granted["grant_id"].as_str().unwrap()));
    let receipt = case.root.join("swapped-score.json");
    let error = refusal(outer.score(&grant_path, &case.predictor, &case.other, &receipt));
    assert!(
        error.contains("differ from the granted identity"),
        "{error}"
    );
}

/// The same receiver, for the sealed selection: the seal the handoff records
/// is the seal that is compared with the replayed selection.
#[test]
fn an_outer_grant_is_bound_to_the_seal_it_checked() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let seal = case.root.join(handoff["seal"]["path"].as_str().unwrap());
    let replayed = fs::read(&seal).unwrap();
    // Positive control, counting the seal's verifications.
    let nth = verifications(&seal, || {
        case.grant("honest", &case.sealed).unwrap();
    });
    // The handoff records a seal that is not the replayed selection.
    let mut tampered: Value = serde_json::from_slice(&replayed).unwrap();
    tampered["stop_reason"] = json!("tampered");
    let tampered = serde_json::to_vec_pretty(&tampered).unwrap();
    fs::write(&seal, &tampered).unwrap();
    handoff["seal"]["sha256"] = json!(hash(&tampered));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("tampered", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "{error}"
    );
    // The replayed seal replaces it right after its check against the handoff.
    let _hook = swap_at(&seal, nth, replayed);
    let error = refusal(case.grant("swapped", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "a grant was issued for a handoff whose seal is not the replayed selection: {error}"
    );
}

/// The same receiver, for the fitted subject: the predictor the grant names
/// is the predictor verified as the selected final refit.
#[test]
fn an_outer_grant_is_for_the_predictor_it_verified() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let exported = fs::read(&case.predictor).unwrap();
    // Positive control, counting the predictor's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.predictor, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["predictor"]["sha256"], hash(&exported));
    // The handoff records a predictor that declares another training scope.
    let mut forged: Value = serde_json::from_slice(&exported).unwrap();
    forged["training_sha256"] = json!("0".repeat(64));
    let forged = serde_json::to_vec_pretty(&forged).unwrap();
    fs::write(&case.predictor, &forged).unwrap();
    handoff["predictor"]["sha256"] = json!(hash(&forged));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("forged", &case.sealed));
    assert!(error.contains("another training scope"), "{error}");
    // The verified export replaces it right after the subject check.
    let _hook = swap_at(&case.predictor, nth, exported);
    let result = case.grant("swapped", &case.sealed);
    assert!(
        matches!(&result, Err(e) if format!("{e:#}").contains("another training scope")),
        "a grant was issued for a subject never verified as the final refit: {result:?}"
    );
}

/// The same receiver, for the handoff: the digest the grant records is the
/// digest of the handoff whose content was checked and recorded.
#[test]
fn an_outer_grant_records_the_handoff_it_checked() {
    let case = outer_case();
    let recorded = fs::read(case.handoff()).unwrap();
    // Positive control, counting the handoff's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.handoff(), || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["handoff"]["sha256"], hash(&recorded));
    // Another handoff replaces it right after it was read.
    let mut other: Value = serde_json::from_slice(&recorded).unwrap();
    other["final_refit"]["model_sha256"] = json!("f".repeat(64));
    let _hook = swap_at(
        &case.handoff(),
        nth,
        serde_json::to_vec_pretty(&other).unwrap(),
    );
    let granted = case.grant("swapped", &case.sealed).unwrap();
    assert_eq!(
        granted["subject"]["handoff"]["sha256"],
        hash(&recorded),
        "the grant records a handoff other than the one it checked: {granted}"
    );
    assert_eq!(
        granted["subject"]["model_sha256"],
        honest["subject"]["model_sha256"]
    );
}

/// The same receiver, for the final refit's fit completion
/// (`lineage::verify_final_refit`): the completion consumed is the one whose
/// digest the predictor records.
#[test]
fn an_outer_grant_consumes_the_fit_completion_it_verified() {
    let case = outer_case();
    let exported: Value = read_json(&case.predictor).unwrap();
    let completion = PathBuf::from(exported["fit_completion"]["path"].as_str().unwrap());
    // Positive control, counting the completion's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&completion, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    let _hook = swap_at(&completion, nth, b"{}".to_vec());
    let granted = case
        .grant("swapped", &case.sealed)
        .map_err(|e| format!("{e:#}"));
    assert!(
        matches!(&granted, Ok(g) if g["subject"]["fit_key"] == honest["subject"]["fit_key"]),
        "the refit check consumed a completion other than the verified one: {granted:?}"
    );
}

/// Replace the first file the authority reports written under `dir` with
/// `edit` applied to it.
fn swap_written(
    dir: PathBuf,
    edit: impl Fn(&mut Value) + 'static,
) -> crate::manifest::verification_hook::Guard {
    let mut done = false;
    crate::manifest::verification_hook::install(move |path| {
        if !done && path.parent() == Some(dir.as_path()) {
            done = true;
            let mut value: Value = read_json(path).unwrap();
            edit(&mut value);
            let staged = path.with_extension("swap");
            fs::write(&staged, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            fs::rename(&staged, path).unwrap();
        }
    })
}

/// The authority's own records: the ledger binds the digest of exactly the
/// grant and withheld score the authority wrote, never a later reread.
#[test]
fn the_release_ledger_binds_the_bytes_the_authority_wrote() {
    let case = outer_case();
    let other = hash(OTHER_TEST.as_bytes());
    // A grant replaced right after it is written, naming other labels.
    let dir = case.root.join("grant-swap");
    let outer = case.release("grant-swap");
    let grants = dir.canonicalize().unwrap().join("grants");
    let relative = case.other.strip_prefix(&case.root).unwrap().to_path_buf();
    let path = {
        let _hook = swap_written(grants, move |grant| {
            grant["scoring_labels"]["path"] = json!(relative);
            grant["scoring_labels"]["sha256"] = json!(other);
        });
        outer
            .grant(&outer_release::GrantRequest {
                search: handoff::Inputs {
                    root: &case.root,
                    output: &case.output,
                    seeds: Some(&case.seeds),
                    vocabulary: None,
                },
                predictor: &case.predictor,
                test: &case.sealed,
            })
            .unwrap()
    };
    let receipt = case.root.join("grant-swap-score.json");
    let error = refusal(outer.score(&path, &case.predictor, &case.other, &receipt));
    assert!(
        error.contains("not recorded by this release authority"),
        "labels the preparation never bound were scored under a replaced grant: {error}"
    );

    // A withheld score replaced right after it is written.
    let dir = case.root.join("withheld-swap");
    let outer = case.release("withheld-swap");
    let withheld = dir.canonicalize().unwrap().join("withheld");
    let path = outer
        .grant(&outer_release::GrantRequest {
            search: handoff::Inputs {
                root: &case.root,
                output: &case.output,
                seeds: Some(&case.seeds),
                vocabulary: None,
            },
            predictor: &case.predictor,
            test: &case.sealed,
        })
        .unwrap();
    let scored = {
        let _hook = swap_written(withheld, |result| {
            result["metrics"]["rmse"] = json!(0.0);
        });
        outer
            .score(
                &path,
                &case.predictor,
                &case.sealed,
                &case.root.join("withheld-score.json"),
            )
            .unwrap()
    };
    let grant_id = scored["grant_id"].as_str().unwrap().to_string();
    let visible = case.root.join("withheld-swap-released.json");
    let released = outer
        .release(&grant_id, &visible)
        .map_err(|e| format!("{e:#}"));
    assert!(
        matches!(&released, Err(e) if e.contains("withheld score changed after scoring")),
        "a withheld score other than the one computed was released: {released:?}"
    );
    assert!(!visible.exists());
}
