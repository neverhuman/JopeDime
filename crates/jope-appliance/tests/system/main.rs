//! INT-02 reconnect, deterministic rerun and real engine crash suites.
//! Full internal engine cuts require INT02_ENGINE_BOUNDARY_CUTS=1 and the
//! non-default jope-service/failpoints feature; fixture cuts are separate.
#![cfg(feature = "engine")]

#[cfg(all(feature = "mcp", any(target_os = "linux", target_os = "macos")))]
mod live_driver;
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

mod crash_tests;
mod event_tests;
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
    let committed = if point.starts_with("cas:") {
        std::env::var("INT02_CAS_COMMITTED")
            .map(|value| value.parse::<u64>().unwrap())
            .unwrap_or(0)
    } else {
        0
    };
    assert!(committed <= 2);
    let mut roots = store
        .get(&RecordKey::Run(next.run_id.clone()))
        .unwrap()
        .unwrap()
        .blobs;
    assert_eq!(roots.len() as u64, committed);
    if committed > 0 {
        // Store::open has recovered the seeded running prefix. Persist an actual
        // resume before the next CAS write so the killed child owns a running run.
        assert_eq!(next.state, RunState::Paused);
        next.state = RunState::Running;
        next.revision.0 += 1;
        next = log
            .commit(
                RunUpdate {
                    expected_revision: Some(next.revision.0 - 1),
                    snapshot: next.clone(),
                    checkpoint: json!({"round": committed}),
                    blobs: roots.clone(),
                },
                vec![state_body(&next)],
                |_| Ok(()),
            )
            .unwrap()
            .snapshot;
    }
    for round in committed + 1..=3 {
        publish_crash_round(&store, &mut next, &mut roots, round, home, &point);
    }
    panic!("unknown crash boundary {point}");
}

fn publish_crash_round(
    store: &Arc<Store>,
    next: &mut RunSnapshot,
    roots: &mut Vec<jope_service::store::BlobRef>,
    round: u64,
    home: &Path,
    point: &str,
) {
    let log = EventLog::new(Arc::clone(store));
    let EventBody::AlgorithmAccepted(mut artifact) = fixture_events()[3].event.clone() else {
        unreachable!();
    };
    next.state = RunState::Running;
    next.revision.0 += 1;
    stop_at(home, point, &format!("{round}:before_blob_publish"));
    let bytes = format!("synthetic model fixture {round}");
    let blob = store.publish_blob(Cursor::new(bytes.as_bytes())).unwrap();
    stop_at(home, point, &format!("{round}:after_blob_publish"));
    roots.push(blob.clone());
    artifact.model_artifact = Some(crash_artifact(round));
    *next = log
        .commit(
            RunUpdate {
                expected_revision: Some(next.revision.0 - 1),
                snapshot: next.clone(),
                checkpoint: json!({"round": round}),
                blobs: roots.clone(),
            },
            vec![state_body(next), EventBody::AlgorithmAccepted(artifact)],
            |write| {
                write.put(
                    &RecordKey::Artifact(crash_artifact(round)),
                    &Record::metadata(json!({"fixture": true, "round": round}), vec![blob]),
                )?;
                stop_at(home, point, &format!("{round}:before_journal_commit"));
                Ok(())
            },
        )
        .unwrap()
        .snapshot;
    stop_at(home, point, &format!("{round}:after_journal_commit"));
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
    let resumed = point.starts_with("cas:") && committed > 0;
    let store = Arc::new(Store::open(home).unwrap());
    let log = EventLog::new(Arc::clone(&store));
    let recovered = log.snapshot(&snapshot().run_id).unwrap().unwrap();
    assert_eq!(recovered.state, RunState::Paused, "{point}");
    assert_eq!(
        recovered.revision.0,
        if committed == 0 {
            1
        } else {
            committed + 2 + 2 * u64::from(resumed)
        },
        "{point}"
    );
    let watermark = 1 + 2 * committed + u64::from(resumed);
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
            assert_eq!(
                record.blobs[(round - 1) as usize],
                artifact.blobs[0],
                "{point}"
            );
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
    if resumed {
        assert_eq!(
            events.last().unwrap().event,
            state_body(&running),
            "{point}"
        );
    }
    for sequence in watermark + 1..=8 {
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

fn prepare_crash_prefix(home: &Path, committed: u64) {
    prepare_crash_fixture(home);
    if committed > 0 {
        let store = Arc::new(Store::open(home).unwrap());
        let log = EventLog::new(Arc::clone(&store));
        let mut next = log.snapshot(&snapshot().run_id).unwrap().unwrap();
        let mut roots = vec![];
        for round in 1..=committed {
            publish_crash_round(&store, &mut next, &mut roots, round, home, "no cut");
        }
    }
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
