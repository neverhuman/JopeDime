//! Batch ownership and interruption at the shared engine/API boundary (§3.1, §6.5).
//!
//! CLI discovery/dispatch must opt into these helpers. They do not implement run,
//! control, or export handlers; a refused control never counts as a committed pause.
#![cfg(not(windows))] // signal-hook's iterator is unavailable on Windows.
#![allow(dead_code)] // Store-bound transport wiring awaits the shared engine owner.

use jope_api::{ids::RunId, RunSnapshot, RunState};
use jope_client::{Request, Transport};
#[cfg(feature = "engine")]
use jope_service::{engine::InProcessEngine, store::Store};
use signal_hook::iterator::{Handle, Signals};
#[cfg(feature = "engine")]
use std::path::Path;
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

/// Scope the store and its advisory lock to this command, on both success and error.
/// The borrowed store cannot escape the operation. No listener or child is started.
#[cfg(feature = "engine")]
pub fn with_engine<R>(
    home: &Path,
    operation: impl FnOnce(&Store, &Interrupts) -> anyhow::Result<R>,
) -> anyhow::Result<R> {
    let engine = InProcessEngine::open(home)?;
    // Reverse local drop order joins/unregisters the signal owner before the
    // engine closes its Store and releases the advisory lock, including unwind.
    let interrupts = Interrupts::install()?;
    operation(engine.store(), &interrupts)
}

/// A durable paused/terminal acknowledgment, suitable for the batch entrypoint.
#[derive(Debug)]
pub struct Interrupted {
    pub run_id: RunId,
    pub state: RunState,
}

impl Interrupted {
    pub fn exit_code(&self) -> u8 {
        130
    }
}

impl std::fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "run {} acknowledged {:?} after SIGINT",
            self.run_id, self.state
        )
    }
}

impl std::error::Error for Interrupted {}

/// Safe signal delivery; API calls and store IO never run in a signal handler.
pub struct Interrupts {
    count: Arc<AtomicUsize>,
    handle: Handle,
    reader: Option<JoinHandle<()>>,
}

impl Interrupts {
    pub fn install() -> std::io::Result<Self> {
        let mut signals = Signals::new([signal_hook::consts::SIGINT])?;
        let handle = signals.handle();
        let count = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&count);
        let reader = thread::Builder::new()
            .name("jope-batch-signals".into())
            .spawn(move || {
                for _ in signals.forever() {
                    // SIGINT may be coalesced by the OS; retain every delivered
                    // occurrence, including a second received during API/store IO.
                    let _ = observed.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                        Some(n.saturating_add(1).min(2))
                    });
                }
            })?;
        Ok(Self {
            count,
            handle,
            reader: Some(reader),
        })
    }

    pub fn requested(&self) -> bool {
        self.count.load(Ordering::SeqCst) != 0
    }

    #[cfg(test)]
    pub(crate) fn delivered_for_test(count: usize) -> Self {
        let interrupts = Self::install().unwrap();
        interrupts.count.store(count.min(2), Ordering::SeqCst);
        interrupts
    }

    /// Called by the command at a request boundary after SIGINT. Wait for the
    /// engine's committed acknowledgment; a second SIGINT escalates to cancel.
    /// Errors propagate instead of manufacturing a successful exit130.
    pub fn finish<T: Transport>(
        &self,
        transport: &T,
        mut snapshot: RunSnapshot,
    ) -> anyhow::Result<Option<Interrupted>> {
        if !self.requested() {
            return Ok(None);
        }
        let run_id = snapshot.run_id.clone();
        // The command may have retained a snapshot across a long request. A
        // stale paused snapshot is not proof of the run's current durable state.
        snapshot = transport.run_status(&run_id)?;
        let mut cancelling = false;
        let mut pausing = false;
        loop {
            anyhow::ensure!(
                snapshot.run_id == run_id,
                "control returned a different run"
            );
            let cancel = self.count.load(Ordering::SeqCst) >= 2;
            let terminal = matches!(
                snapshot.state,
                RunState::Completed | RunState::Cancelled | RunState::Failed
            );
            if terminal || (snapshot.state == RunState::Paused && !cancel) {
                return Ok(Some(Interrupted {
                    run_id,
                    state: snapshot.state,
                }));
            }
            let action = if cancel && !cancelling {
                cancelling = true;
                Some("cancel")
            } else if !cancel && !pausing && snapshot.state == RunState::Running {
                pausing = true;
                Some("pause")
            } else {
                None
            };
            if let Some(action) = action {
                let key = format!("cli_{}", ulid::Ulid::new());
                snapshot = transport
                    .execute(
                        Request::new(8, 0, &[("id", &run_id.to_string())])?
                            .key(Some(&key))?
                            .if_match(snapshot.revision)
                            .json(&serde_json::json!({"action": action}))?,
                    )?
                    .json()?;
            } else {
                // A short status request leaves room for cancellation; do not
                // enter the batch command's normal 30-second long poll here.
                thread::sleep(Duration::from_millis(20));
                snapshot = transport.run_status(&run_id)?;
            }
        }
    }
}

impl Drop for Interrupts {
    fn drop(&mut self) {
        self.handle.close();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
        // The reader and this final handle drop this owner's registrations.
    }
}

#[cfg(all(test, feature = "engine"))]
mod tests {
    use super::*;
    use jope_client::{transport::header, Response};
    use jope_service::{
        engine::EngineError,
        store::{Record, RecordKey},
    };
    use serde_json::{json, Value};
    use std::{
        fs,
        io::Cursor,
        process::{Child, Command, Stdio},
        sync::Mutex,
        time::Instant,
    };

    fn snapshot() -> RunSnapshot {
        serde_json::from_value(json!({
            "run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0", "revision":"1",
            "state":"running", "phase":"searching", "pending":null, "effort":"high",
            "budget_used":{"evaluations":"7","wall_seconds":"1"},
            "budget_total":{"evaluations":"128","wall_seconds":"60"}, "best":null,
            "event_watermark":"0", "warnings":[], "completion_reason":null
        }))
        .unwrap()
    }

    fn fixture() -> tempfile::TempDir {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/batch-proof");
        fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }

    fn persist(store: &Store, snapshot: &RunSnapshot) {
        store
            .transaction(|write| {
                write.put(
                    &RecordKey::Run(snapshot.run_id.clone()),
                    &Record::run(
                        json!({"snapshot":snapshot,"checkpoint":{"committed_round":7}}),
                        vec![],
                        snapshot.state,
                        snapshot.revision.0,
                    ),
                )
            })
            .unwrap();
    }

    /// Protocol fixture, deliberately not a discovery/export handler or H2 proof.
    struct ControlFixture<'a> {
        store: &'a Store,
        current: Mutex<RunSnapshot>,
        actions: Mutex<Vec<String>>,
        pending_pause: bool,
        pause_marker: Option<std::path::PathBuf>,
        refuse: bool,
    }

    impl<'a> ControlFixture<'a> {
        fn new(store: &'a Store) -> Self {
            Self {
                store,
                current: Mutex::new(snapshot()),
                actions: Mutex::new(vec![]),
                pending_pause: false,
                pause_marker: None,
                refuse: false,
            }
        }
    }

    impl Transport for ControlFixture<'_> {
        fn execute(&self, request: Request) -> jope_client::Result<Response> {
            let mut current = self.current.lock().unwrap();
            assert!(request.target().contains(&current.run_id.to_string()));
            if request.method().as_str() == "POST" {
                if self.refuse {
                    return Err(jope_client::Error::Protocol("control refused"));
                }
                assert_eq!(
                    request.headers[header::IF_MATCH],
                    format!("\"{}\"", current.revision.0)
                );
                assert!(request.headers["idempotency-key"]
                    .to_str()
                    .unwrap()
                    .starts_with("cli_"));
                let body: Value = serde_json::from_slice(request.body()).unwrap();
                let action = body["action"].as_str().unwrap();
                self.actions.lock().unwrap().push(action.into());
                current.state = match action {
                    "pause" if self.pending_pause => RunState::Pausing,
                    "pause" => RunState::Paused,
                    "cancel" => RunState::Cancelled,
                    "continue" if current.state == RunState::Paused => RunState::Running,
                    _ => panic!("unexpected control action"),
                };
                current.revision.0 += 1;
                persist(self.store, &current);
                if action == "pause" {
                    if let Some(marker) = &self.pause_marker {
                        fs::write(marker, b"pausing").unwrap();
                    }
                }
            }
            let mut headers = jope_client::transport::HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
            headers.insert("jope-api-version", "1.0".parse().unwrap());
            Ok(Response::new(
                200,
                headers,
                Cursor::new(serde_json::to_vec(&*current).unwrap()),
            ))
        }
    }

    #[test]
    fn no_interrupt_does_not_mutate_or_read_a_run() {
        let home = fixture();
        with_engine(home.path(), |store, interrupts| {
            let transport = ControlFixture::new(store);
            assert!(interrupts.finish(&transport, snapshot())?.is_none());
            assert!(transport.actions.lock().unwrap().is_empty());
            assert!(store.get(&RecordKey::Run(snapshot().run_id))?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn failed_control_is_not_reported_as_a_committed_interruption() {
        let home = fixture();
        with_engine(home.path(), |store, interrupts| {
            let mut transport = ControlFixture::new(store);
            transport.refuse = true;
            interrupts.count.store(1, Ordering::SeqCst);
            assert!(interrupts
                .finish(&transport, snapshot())
                .unwrap_err()
                .to_string()
                .contains("control refused"));
            assert!(store.get(&RecordKey::Run(snapshot().run_id))?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn two_already_observed_interrupts_cancel_without_a_pause() {
        let home = fixture();
        with_engine(home.path(), |store, interrupts| {
            let transport = ControlFixture::new(store);
            interrupts.count.store(2, Ordering::SeqCst);
            let result = interrupts.finish(&transport, snapshot())?.unwrap();
            assert_eq!(result.state, RunState::Cancelled);
            assert_eq!(result.exit_code(), 130);
            assert_eq!(*transport.actions.lock().unwrap(), ["cancel"]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn stale_paused_snapshot_cannot_skip_the_current_runs_pause_commit() {
        let home = fixture();
        with_engine(home.path(), |store, interrupts| {
            let transport = ControlFixture::new(store);
            let mut stale = snapshot();
            stale.state = RunState::Paused;
            interrupts.count.store(1, Ordering::SeqCst);
            let result = interrupts.finish(&transport, stale)?.unwrap();
            assert_eq!(result.state, RunState::Paused);
            assert_eq!(*transport.actions.lock().unwrap(), ["pause"]);
            assert_eq!(
                store
                    .get(&RecordKey::Run(result.run_id))?
                    .unwrap()
                    .run
                    .unwrap()
                    .state,
                RunState::Paused
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn operation_error_releases_store_before_lock_and_preserves_checkpoint() {
        let home = fixture();
        let result: anyhow::Result<()> = with_engine(home.path(), |store, _| {
            persist(store, &snapshot());
            anyhow::bail!("operation failed")
        });
        assert!(result.unwrap_err().to_string().contains("operation failed"));
        with_engine(home.path(), |store, _| {
            let saved = store.get(&RecordKey::Run(snapshot().run_id))?.unwrap();
            assert_eq!(saved.run.unwrap().state, RunState::Paused);
            assert_eq!(saved.data["checkpoint"]["committed_round"], 7);
            Ok(())
        })
        .unwrap();
    }

    // Signals are process-wide: real SIGINT tests run only in this isolated child.
    #[test]
    #[ignore = "invoked by the bounded subprocess lifecycle test"]
    fn process_fixture() {
        let home = std::env::var_os("JOPE_BATCH_TEST_HOME").unwrap();
        let home = Path::new(&home);
        let mode = std::env::var("JOPE_BATCH_TEST_MODE").unwrap();
        let code = with_engine(home, |store, interrupts| {
            let mut transport = ControlFixture::new(store);
            transport.pending_pause = mode == "cancel";
            transport.pause_marker = Some(home.join("pausing"));
            persist(store, &snapshot());
            fs::write(home.join("ready"), b"ready")?;
            if mode == "normal" {
                return Ok(0);
            }
            let deadline = Instant::now() + Duration::from_secs(15);
            while !interrupts.requested() {
                anyhow::ensure!(Instant::now() < deadline, "SIGINT fixture timed out");
                thread::sleep(Duration::from_millis(5));
            }
            let outcome = interrupts.finish(&transport, snapshot())?.unwrap();
            fs::write(home.join("outcome"), format!("{:?}", outcome.state))?;
            Ok(outcome.exit_code())
        })
        .unwrap();
        // with_engine has dropped the store, lock, registration and joined reader.
        assert!(fs::read_dir("/proc/self/task").unwrap().all(|entry| {
            fs::read_to_string(entry.unwrap().path().join("comm"))
                .map_or(true, |name| !name.starts_with("jope-batch-sign"))
        }));
        std::process::exit(i32::from(code));
    }

    struct OwnChild(Child);
    impl Drop for OwnChild {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
            }
            let _ = self.0.wait();
        }
    }

    fn await_file(path: &Path, child: &mut OwnChild) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.is_file() {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "fixture exited early"
            );
            assert!(Instant::now() < deadline, "fixture readiness timed out");
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn subprocess_pause_cancel_busy_reopen_and_no_surviving_process() {
        let evidence_root = fixture();
        let mut receipts = vec![];
        for mode in ["normal", "pause", "cancel"] {
            let home = evidence_root.path().join(mode);
            fs::create_dir(&home).unwrap();
            let log = fs::File::create(home.join("child.log")).unwrap();
            let mut child = OwnChild(
                Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "cmd::inproc::tests::process_fixture",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("JOPE_BATCH_TEST_HOME", &home)
                    .env("JOPE_BATCH_TEST_MODE", mode)
                    .stdin(Stdio::null())
                    .stdout(log.try_clone().unwrap())
                    .stderr(log)
                    .spawn()
                    .unwrap(),
            );
            let pid = child.0.id();
            if mode != "normal" {
                await_file(&home.join("ready"), &mut child);
                let error = with_engine(&home, |_, _| Ok(())).unwrap_err();
                assert!(matches!(error.downcast_ref::<EngineError>(),
                    Some(EngineError::Busy {pid:Some(owner),command})
                    if *owner == pid && command.contains("process_fixture")));
                for index in 0..if mode == "cancel" { 2 } else { 1 } {
                    if index == 1 {
                        await_file(&home.join("pausing"), &mut child);
                    }
                    assert!(Command::new("sh")
                        .args(["-c", "kill -INT \"$1\"", "sh", &pid.to_string()])
                        .status()
                        .unwrap()
                        .success());
                }
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            let status = loop {
                if let Some(status) = child.0.try_wait().unwrap() {
                    break status;
                }
                assert!(Instant::now() < deadline, "fixture did not exit");
                thread::sleep(Duration::from_millis(5));
            };
            assert_eq!(
                status.code(),
                Some(if mode == "normal" { 0 } else { 130 }),
                "{}",
                fs::read_to_string(home.join("child.log")).unwrap()
            );
            assert!(!Path::new(&format!("/proc/{pid}")).exists());
            assert!(!home.join("daemon.json").exists());
            with_engine(&home, |store, _| {
                let saved = store.get(&RecordKey::Run(snapshot().run_id))?.unwrap();
                let recovery = saved.run.as_ref().unwrap();
                assert_eq!(
                    recovery.state,
                    if mode == "cancel" {
                        RunState::Cancelled
                    } else {
                        RunState::Paused
                    }
                );
                assert_eq!(saved.data["checkpoint"]["committed_round"], 7);
                if mode == "pause" {
                    // A later injected client can reopen and continue the saved
                    // control fixture; this does not qualify discovery/H2.
                    let transport = ControlFixture::new(store);
                    *transport.current.lock().unwrap() =
                        serde_json::from_value(saved.data["snapshot"].clone())?;
                    let resumed: RunSnapshot = transport
                        .execute(
                            Request::new(8, 0, &[("id", &snapshot().run_id.to_string())])?
                                .key(Some("cli_later_fixture_client"))?
                                .if_match(jope_api::decimal::DecimalU64(recovery.revision))
                                .json(&json!({"action":"continue"}))?,
                        )?
                        .json()?;
                    assert_eq!(resumed.state, RunState::Running);
                }
                Ok(())
            })
            .unwrap();
            receipts.push(
                json!({"mode":mode,"child_pid":pid,"exit_code":status.code(),
                "pid_absent_from_procfs":true,
                "no_daemon_json":true,"store_reopened_under_same_lock":true,
                "checkpoint_round":7,"control_backend":"injected fixture, not H2",
                "child_log":fs::read_to_string(home.join("child.log")).unwrap()}),
            );
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/batch-proof");
        fs::write(
            root.join(format!("process-receipt-{}.json", std::process::id())),
            serde_json::to_vec_pretty(&receipts).unwrap(),
        )
        .unwrap();
    }
}
