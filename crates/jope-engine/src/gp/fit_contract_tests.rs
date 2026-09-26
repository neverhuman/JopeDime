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
        "fit-view" => fit_view(
            &path("prepared")?,
            text("cut")?.parse()?,
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
        other => anyhow::bail!("unsupported fixture CLI operation {other}"),
    }
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
    let mut command = authority.command();
    command
        .args(["search", "--root"])
        .arg(root)
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .args(["--cut", "0", "--seconds", "180", "--qualification"])
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
