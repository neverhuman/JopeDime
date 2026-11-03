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

mod cli;
mod support;

pub(super) use support::*;

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
    if let Err(error) = cli::run_cli(&arguments[split + 1..]) {
        eprintln!("Error: {error:?}");
        // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=verified-exit-behavior expires=2026-12-31
        std::process::exit(1);
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

mod fit_views;
pub(super) use fit_views::*;
mod search_trials;
pub(super) use search_trials::*;
mod search_lineage;
pub(super) use search_lineage::*;
mod search_lifecycle;
pub(super) use search_lifecycle::*;
mod selection;
pub(super) use selection::*;
mod outer_release;
pub(super) use outer_release::*;
mod view_exports;
pub(super) use view_exports::*;
mod outer_grants;
pub(super) use outer_grants::*;
