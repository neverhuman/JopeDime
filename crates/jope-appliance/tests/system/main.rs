//! INT-02 PRE-STAGE: fixture reconnects and public store crash boundaries.
//! These tests do not qualify live terminal/MCP, discovery determinism, or the
//! internal CAS failpoint matrix required by release gates R4, S2, and R1.
#![cfg(feature = "engine")]

mod r4_event_parity;

use jope_api::{decimal::DecimalU64, EventBody, PublicEvent, RunSnapshot, RunState, RunStateBody};
use jope_client::{
    transport::{header, HeaderMap, HeaderValue},
    EventStream, Request, Response, StreamEvent, Transport,
};
use jope_service::{
    events::{EventLog, RunUpdate},
    store::{Record, RecordKey, Store},
};
use serde_json::json;
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

fn fixture_events() -> Vec<PublicEvent> {
    include_str!("events.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn frame(event: &PublicEvent) -> String {
    format!(
        "id: {}\ndata: {}\n\n",
        event.sequence.0,
        serde_json::to_string(event).unwrap()
    )
}

/// A fixture observer deliberately replays the last acknowledged event. Its
/// first connection can end halfway through the following SSE frame.
struct FixtureTransport {
    events: Vec<PublicEvent>,
    initial_cut: usize,
    cursors: Mutex<Vec<Option<DecimalU64>>>,
}

impl FixtureTransport {
    fn new(initial_cut: usize) -> Self {
        Self {
            events: fixture_events(),
            initial_cut,
            cursors: Mutex::new(vec![]),
        }
    }
}

impl Transport for FixtureTransport {
    fn execute(&self, request: Request) -> jope_client::Result<Response> {
        let mut headers = HeaderMap::new();
        headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
        let body = if request.headers.get(header::ACCEPT)
            == Some(&HeaderValue::from_static("text/event-stream"))
        {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            let after = request
                .headers
                .get("last-event-id")
                .map(|value| DecimalU64(value.to_str().unwrap().parse().expect("decimal cursor")));
            let mut cursors = self.cursors.lock().unwrap();
            let first = cursors.is_empty();
            cursors.push(after);
            let events = self
                .events
                .iter()
                .filter(|event| after.is_none_or(|after| event.sequence >= after))
                .collect::<Vec<_>>();
            let mut text = String::new();
            for event in events.iter().take(if first {
                self.initial_cut
            } else {
                events.len()
            }) {
                text.push_str(&frame(event));
            }
            if first {
                if let Some(event) = events.get(self.initial_cut) {
                    let partial = frame(event);
                    text.push_str(&partial[..partial.len() / 2]);
                }
            }
            text.into_bytes()
        } else {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            // The second observer uses the public run_events client operation.
            assert!(request.target().contains("after=0"));
            serde_json::to_vec(&json!({
                "events": self.events,
                "next_after": self.events.last().unwrap().sequence,
            }))
            .unwrap()
        };
        Ok(Response::new(200, headers, Cursor::new(body)))
    }
}

/// Reopen the durable journal for every HTTP request so a reconnect crosses
/// the same persistence boundary as an engine restart.
struct PersistedEventTransport {
    home: PathBuf,
    run: jope_api::ids::RunId,
    initial_cut: usize,
    cursors: Mutex<Vec<Option<DecimalU64>>>,
}

impl PersistedEventTransport {
    fn page(&self, after: u64) -> Vec<PublicEvent> {
        let store = Arc::new(Store::open(&self.home).unwrap());
        EventLog::new(store)
            .page(&self.run, after, 256, &[])
            .unwrap()
            .unwrap()
            .events
    }
}

impl Transport for PersistedEventTransport {
    fn execute(&self, request: Request) -> jope_client::Result<Response> {
        let mut headers = HeaderMap::new();
        headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
        let streaming = request.headers.get(header::ACCEPT)
            == Some(&HeaderValue::from_static("text/event-stream"));
        let body = if streaming {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            let after = request
                .headers
                .get("last-event-id")
                .map(|value| DecimalU64(value.to_str().unwrap().parse().expect("decimal cursor")));
            let first = {
                let mut cursors = self.cursors.lock().unwrap();
                cursors.push(after);
                cursors.len() == 1
            };
            // Deliberately include the last acknowledged event so the public
            // stream must suppress a replay after reopening the store.
            let events = self.page(after.map_or(0, |cursor| cursor.0.saturating_sub(1)));
            let mut text = String::new();
            for event in events.iter().take(if first {
                self.initial_cut
            } else {
                events.len()
            }) {
                text.push_str(&frame(event));
            }
            if first {
                if let Some(event) = events.get(self.initial_cut) {
                    let partial = frame(event);
                    text.push_str(&partial[..partial.len() / 2]);
                }
            }
            text.into_bytes()
        } else {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            assert!(request.target().contains("after=0"));
            let events = self.page(0);
            serde_json::to_vec(&json!({
                "next_after": events.last().map(|event| event.sequence),
                "events": events,
            }))
            .unwrap()
        };
        Ok(Response::new(200, headers, Cursor::new(body)))
    }
}

fn durable(event: jope_client::Result<StreamEvent>) -> PublicEvent {
    match event.unwrap() {
        StreamEvent::Durable(event) => *event,
        StreamEvent::Telemetry(_) => panic!("fixture contains only durable events"),
    }
}

#[test]
fn r4_fixture_reconnect_does_not_acknowledge_partial_frames_or_duplicate_replays() {
    let expected = fixture_events();
    let run = expected[0].run_id.clone();
    for cut in 0..=expected.len() {
        let transport = FixtureTransport::new(cut);
        let mut stream = EventStream::new(&transport, run.clone(), None).unwrap();
        let mut observed = stream.by_ref().map(durable).collect::<Vec<_>>();
        assert_eq!(observed, expected[..cut], "disconnect at {cut}");
        let cursor = (cut > 0).then_some(DecimalU64(cut as u64));
        assert_eq!(stream.last_event_id(), cursor);
        drop(stream); // Closing an observer retains its last complete watermark.

        let stream = EventStream::new(&transport, run.clone(), cursor).unwrap();
        observed.extend(stream.map(durable));
        assert_eq!(observed, expected, "reattach at {cut}");
        assert_eq!(*transport.cursors.lock().unwrap(), vec![None, cursor]);
        let page =
            serde_json::to_value(transport.run_events(&run, DecimalU64(0), 256).unwrap()).unwrap();
        let page_events: Vec<PublicEvent> = serde_json::from_value(page["events"].clone()).unwrap();
        assert_eq!(observed, page_events, "stream and page observer at {cut}");
    }
}

#[test]
fn s2_fixture_champion_scores_and_ordered_events_survive_every_reconnect_schedule() {
    let expected = fixture_events();
    for interval in 1..=expected.len() {
        let transport = FixtureTransport::new(expected.len());
        let mut stream = EventStream::new(&transport, expected[0].run_id.clone(), None).unwrap();
        let mut observed = Vec::new();
        while let Some(event) = stream.next() {
            observed.push(durable(event));
            if observed.len().is_multiple_of(interval) {
                stream.reconnect().unwrap();
            }
        }
        assert_eq!(observed, expected, "reconnect interval {interval}");
        let EventBody::RunFinished(finished) = &observed.last().unwrap().event else {
            panic!("fixture must finish");
        };
        assert!(finished.champion.is_some());
        assert_eq!(finished.scores.cv, Some(0.42));
        assert_eq!(stream.last_event_id(), Some(DecimalU64(6)));
    }
}

#[test]
fn s2_durable_event_log_reopen_preserves_partial_reconnect_and_page_equivalence() {
    let fixture = fixture_events();
    let run = fixture[0].run_id.clone();
    for cut in 0..=fixture.len() {
        let home = tempfile::tempdir().unwrap();
        let expected = {
            let store = Arc::new(Store::open(home.path()).unwrap());
            let log = EventLog::new(store);
            log.commit(
                RunUpdate {
                    expected_revision: None,
                    snapshot: snapshot(),
                    checkpoint: json!({"round": 1}),
                    blobs: vec![],
                },
                fixture.iter().map(|event| event.event.clone()).collect(),
                |_| Ok(()),
            )
            .unwrap();
            log.page(&run, 0, 256, &[]).unwrap().unwrap().events
        };

        let transport = PersistedEventTransport {
            home: home.path().to_path_buf(),
            run: run.clone(),
            initial_cut: cut,
            cursors: Mutex::new(vec![]),
        };
        let mut stream = EventStream::new(&transport, run.clone(), None).unwrap();
        let mut observed = stream.by_ref().map(durable).collect::<Vec<_>>();
        assert_eq!(observed, expected[..cut], "first disconnect at frame {cut}");
        let cursor = stream.last_event_id();
        assert_eq!(cursor, (cut > 0).then_some(DecimalU64(cut as u64)));
        drop(stream);

        let stream = EventStream::new(&transport, run.clone(), cursor).unwrap();
        observed.extend(stream.map(durable));
        assert_eq!(observed, expected, "reconnect at frame {cut}");
        assert_eq!(*transport.cursors.lock().unwrap(), vec![None, cursor]);

        let page =
            serde_json::to_value(transport.run_events(&run, DecimalU64(0), 256).unwrap()).unwrap();
        let page_events: Vec<PublicEvent> = serde_json::from_value(page["events"].clone()).unwrap();
        assert_eq!(observed, page_events, "stream and page at frame {cut}");
    }
}

fn snapshot() -> RunSnapshot {
    serde_json::from_value(json!({
        "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0", "revision": "1",
        "state": "paused", "phase": "searching", "pending": null, "effort": "fast",
        "budget_used": {"evaluations": "0", "wall_seconds": "0"},
        "budget_total": {"evaluations": "4", "wall_seconds": "10"},
        "best": null, "event_watermark": "0", "warnings": [], "completion_reason": null
    }))
    .unwrap()
}

fn state_body(snapshot: &RunSnapshot) -> EventBody {
    EventBody::RunState(RunStateBody {
        state: snapshot.state,
        phase: snapshot.phase,
        pending: None,
        reason: None,
    })
}

fn stop_at(home: &Path, wanted: &str, point: &str) {
    if wanted == point {
        fs::write(home.join("ready"), point).unwrap();
        loop {
            std::thread::park_timeout(Duration::from_secs(60));
        }
    }
}

fn crash_artifact(round: u64) -> jope_api::ids::ArtifactId {
    format!("art_{round:024x}").parse().unwrap()
}

// Only the supervised Unix test child installs this handler. It never returns
// to abort's default action; the parent must observe the exact hook marker and
// deliver SIGKILL, or its existing deadline/ChildGuard fails and reaps the child.
#[cfg(unix)]
#[expect(
    unsafe_code,
    reason = "test-only registration of an async-signal-safe SIGSTOP handler"
)]
fn hold_store_abort_for_supervisor() {
    // SAFETY: this capture-free callback only calls signal-hook's documented
    // async-signal-safe raise wrapper. It allocates nothing and takes no locks.
    let _handler = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGABRT, || loop {
            let _ = signal_hook::low_level::raise(signal_hook::consts::SIGSTOP);
        })
    }
    .unwrap();
}

#[test]
#[ignore = "subprocess-only crash target"]
fn crash_child() {
    let home = std::env::var_os("INT02_CRASH_HOME").expect("supervisor sets isolated home");
    let home = Path::new(&home);
    let point = std::env::var("INT02_CRASH_POINT").unwrap();
    if point.starts_with("cas:") {
        #[cfg(unix)]
        hold_store_abort_for_supervisor();
        #[cfg(not(unix))]
        panic!("internal CAS SIGKILL controls require native Unix signals");
    }
    let store = Arc::new(Store::open(home).unwrap());
    let log = EventLog::new(Arc::clone(&store));
    let mut next = log.snapshot(&snapshot().run_id).unwrap().unwrap();
    let mut roots = vec![];
    let EventBody::AlgorithmAccepted(mut artifact) = fixture_events()[3].event.clone() else {
        unreachable!();
    };
    for round in 1..=3 {
        next.state = RunState::Running;
        next.revision.0 += 1;
        stop_at(home, &point, &format!("{round}:before_blob_publish"));
        let bytes = format!("synthetic model fixture {round}");
        let blob = store.publish_blob(Cursor::new(bytes.as_bytes())).unwrap();
        stop_at(home, &point, &format!("{round}:after_blob_publish"));
        roots.push(blob.clone());
        artifact.model_artifact = Some(crash_artifact(round));
        next = log
            .commit(
                RunUpdate {
                    expected_revision: Some(next.revision.0 - 1),
                    snapshot: next.clone(),
                    checkpoint: json!({"round": round}),
                    blobs: roots.clone(),
                },
                vec![
                    state_body(&next),
                    EventBody::AlgorithmAccepted(artifact.clone()),
                ],
                |write| {
                    write.put(
                        &RecordKey::Artifact(crash_artifact(round)),
                        &Record::metadata(json!({"fixture": true, "round": round}), vec![blob]),
                    )?;
                    stop_at(home, &point, &format!("{round}:before_journal_commit"));
                    Ok(())
                },
            )
            .unwrap()
            .snapshot;
        stop_at(home, &point, &format!("{round}:after_journal_commit"));
    }
    panic!("unknown crash boundary {point}");
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Check all tables, including rows past the watermark that paging would hide.
fn recovered_prefix(
    home: &Path,
    committed: u64,
    point: &str,
) -> (RunSnapshot, Record, Vec<PublicEvent>) {
    let store = Arc::new(Store::open(home).unwrap());
    let log = EventLog::new(Arc::clone(&store));
    let recovered = log.snapshot(&snapshot().run_id).unwrap().unwrap();
    assert_eq!(recovered.state, RunState::Paused, "{point}");
    assert_eq!(
        recovered.revision.0,
        if committed == 0 { 1 } else { committed + 2 },
        "{point}"
    );
    let watermark = 1 + 2 * committed;
    assert_eq!(recovered.event_watermark.0, watermark, "{point}");
    let record = store
        .get(&RecordKey::Run(recovered.run_id.clone()))
        .unwrap()
        .unwrap();
    let recovery = record.run.as_ref().unwrap();
    assert_eq!(recovery.state, RunState::Paused, "{point}");
    assert_eq!(recovery.revision, recovered.revision.0, "{point}");
    assert_eq!(recovery.interrupted, committed > 0, "{point}");
    assert_eq!(record.data["checkpoint"]["round"], committed, "{point}");
    assert_eq!(record.blobs.len() as u64, committed, "{point}");
    let events = log
        .page(&recovered.run_id, 0, 256, &[])
        .unwrap()
        .unwrap()
        .events;
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence.0)
            .collect::<Vec<_>>(),
        (1..=watermark).collect::<Vec<_>>(),
        "{point}"
    );
    assert!(
        events.iter().all(|event| event.run_id == recovered.run_id),
        "{point}"
    );
    assert_eq!(events[0].event, state_body(&snapshot()), "{point}");
    let mut running = snapshot();
    running.state = RunState::Running;
    for round in 1..=3 {
        let artifact = store
            .get(&RecordKey::Artifact(crash_artifact(round)))
            .unwrap();
        assert_eq!(
            artifact.is_some(),
            round <= committed,
            "{point}, artifact {round}"
        );
        if let Some(artifact) = artifact {
            assert_eq!(
                artifact.data,
                json!({"fixture": true, "round": round}),
                "{point}"
            );
            assert_eq!(artifact.blobs.len(), 1, "{point}");
            assert!(record.blobs.contains(&artifact.blobs[0]), "{point}");
            assert_eq!(
                store.read_blob(&artifact.blobs[0]).unwrap(),
                format!("synthetic model fixture {round}").as_bytes(),
                "{point}"
            );
            assert_eq!(
                events[(2 * round - 1) as usize].event,
                state_body(&running),
                "{point}"
            );
            let EventBody::AlgorithmAccepted(mut accepted) = fixture_events()[3].event.clone()
            else {
                unreachable!()
            };
            accepted.model_artifact = Some(crash_artifact(round));
            assert_eq!(
                events[(2 * round) as usize].event,
                EventBody::AlgorithmAccepted(accepted),
                "{point}"
            );
        }
    }
    for sequence in watermark + 1..=7 {
        assert!(
            store
                .get(&RecordKey::Event {
                    run: recovered.run_id.clone(),
                    sequence
                })
                .unwrap()
                .is_none(),
            "{point}, uncommitted event {sequence}"
        );
    }
    (recovered, record, events)
}

fn prepare_crash_fixture(home: &Path) {
    let store = Arc::new(Store::open(home).unwrap());
    let log = EventLog::new(store);
    let initial = snapshot();
    log.commit(
        RunUpdate {
            expected_revision: None,
            snapshot: initial.clone(),
            checkpoint: json!({"round": 0}),
            blobs: vec![],
        },
        vec![state_body(&initial)],
        |_| Ok(()),
    )
    .unwrap();
}

fn verify_crash_continuation(home: &Path, committed: u64, point: &str) {
    let (mut recovered, record, prefix) = recovered_prefix(home, committed, point);
    // A second automatic open must not recover the same run twice.
    let store = Arc::new(Store::open(home).unwrap());
    let log = EventLog::new(Arc::clone(&store));
    assert_eq!(
        log.snapshot(&recovered.run_id).unwrap(),
        Some(recovered.clone()),
        "{point}"
    );
    assert_eq!(
        store
            .get(&RecordKey::Run(recovered.run_id.clone()))
            .unwrap(),
        Some(record.clone()),
        "{point}"
    );
    let expected_revision = recovered.revision.0;
    recovered.revision.0 += 1;
    let watermark = recovered.event_watermark.0;
    let appended = log
        .commit(
            RunUpdate {
                expected_revision: Some(expected_revision),
                snapshot: recovered.clone(),
                checkpoint: record.data["checkpoint"].clone(),
                blobs: record.blobs,
            },
            vec![state_body(&recovered)],
            |_| Ok(()),
        )
        .unwrap();
    assert_eq!(appended.events[0].sequence.0, watermark + 1, "{point}");
    let events = log
        .page(&recovered.run_id, 0, 256, &[])
        .unwrap()
        .unwrap()
        .events;
    assert_eq!(events[..prefix.len()], prefix, "{point}");
    assert_eq!(events.last().unwrap(), &appended.events[0], "{point}");
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence.0)
            .collect::<Vec<_>>(),
        (1..=watermark + 1).collect::<Vec<_>>(),
        "{point}"
    );
    eprintln!(
        "INT02 cut={point} committed={committed} watermark={watermark} next={}",
        appended.events[0].sequence.0
    );
}

/// Explicit opt-in: dependency features are not visible as this crate's cfgs.
/// Run with INT02_INTERNAL_CAS_CUTS=1 and --features jope-service/failpoints.
/// Selecting these cuts without the hooks must fail, never silently skip.
#[cfg(unix)]
fn internal_cas_sigkill_cuts() {
    let bytes = b"synthetic model fixture 1";
    let hash = jope_assets::sha256_hex(bytes);
    let reference = jope_service::store::BlobRef {
        sha256: std::array::from_fn(|i| u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).unwrap()),
        bytes: bytes.len() as u64,
    };
    for cut in [
        "after_write",
        "after_file_sync",
        "after_hash_verify",
        "after_rename",
        "after_directory_sync",
    ] {
        let point = format!("cas:{cut}");
        let home = tempfile::tempdir().unwrap();
        prepare_crash_fixture(home.path());
        let stderr = home.path().join("child.stderr");
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash_child", "--ignored", "--nocapture"])
                .env("INT02_CRASH_HOME", home.path())
                .env("INT02_CRASH_POINT", &point)
                .env("JOPE_STORE_FAILPOINT", cut)
                .stdout(Stdio::null())
                .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
                .spawn()
                .unwrap(),
        );
        let marker = format!("JOPE_STORE_CRASH:{cut}");
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            let output = fs::read_to_string(&stderr).unwrap();
            if output.lines().any(|line| line == marker) {
                break;
            }
            assert!(child.0.try_wait().unwrap().is_none(), "{point}: child exited before hook; build with --features jope-service/failpoints: {output}");
            assert!(
                Instant::now() < deadline,
                "{point}: hook not reached: {output}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill().unwrap();
        let status = child.0.wait().unwrap();
        assert!(!status.success(), "{point}");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                Some(9),
                "{point}: must be supervisor SIGKILL, not abort"
            );
        }
        let renamed = matches!(cut, "after_rename" | "after_directory_sync");
        let staged = fs::read_dir(home.path().join("tmp"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(staged.len(), usize::from(!renamed), "{point}");
        if let Some(staged) = staged.first() {
            assert_eq!(fs::read(staged).unwrap(), bytes, "{point}");
        }
        {
            let store = Store::open(home.path()).unwrap();
            assert_eq!(store.verify_blob(&reference).is_ok(), renamed, "{point}");
            if renamed {
                assert_eq!(store.read_blob(&reference).unwrap(), bytes, "{point}");
            }
            assert!(
                fs::read_dir(home.path().join("tmp"))
                    .unwrap()
                    .next()
                    .is_none(),
                "{point}: staging not recovered"
            );
        }
        verify_crash_continuation(home.path(), 0, &point);
        eprintln!("INT02 CAS hook={cut} signal=9 renamed={renamed} staging_recovered=true");
    }
}

#[cfg(not(unix))]
fn internal_cas_sigkill_cuts() {
    panic!("internal CAS cuts require native Unix signals");
}

#[test]
fn r1_sigkill_at_public_blob_and_journal_boundaries_recovers_atomic_records_and_sequences() {
    for (round, boundary) in (1..=3).flat_map(|round| {
        [
            "before_blob_publish",
            "after_blob_publish",
            "before_journal_commit",
            "after_journal_commit",
        ]
        .map(|boundary| (round, boundary))
    }) {
        let point = format!("{round}:{boundary}");
        let home = tempfile::tempdir().unwrap();
        prepare_crash_fixture(home.path());
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash_child", "--ignored", "--nocapture"])
                .env("INT02_CRASH_HOME", home.path())
                .env("INT02_CRASH_POINT", &point)
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        while !fs::read_to_string(home.path().join("ready")).is_ok_and(|marker| marker == point) {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "child exited: {point}"
            );
            assert!(Instant::now() < deadline, "child did not reach {point}");
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill().unwrap();
        let status = child.0.wait().unwrap();
        assert!(!status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(9), "{point}");
        }
        let committed = round - u64::from(boundary != "after_journal_commit");
        verify_crash_continuation(home.path(), committed, &point);
    }
    if std::env::var("INT02_INTERNAL_CAS_CUTS").as_deref() == Ok("1") {
        internal_cas_sigkill_cuts();
    } else {
        eprintln!("INT02 internal CAS cuts not selected; require INT02_INTERNAL_CAS_CUTS=1 and --features jope-service/failpoints");
    }
}

#[test]
#[ignore = "pending executable run handlers"]
fn live_terminal_attach_and_mcp_reconnect_preserve_the_same_run_projection() {
    panic!(
        "PRE-STAGE: wire terminal/MCP observers to operational run handlers before qualification"
    );
}

#[path = "s2_determinism.rs"]
mod s2_determinism;
