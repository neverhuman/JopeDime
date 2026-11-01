//! Durable per-run journal. Share one EventLog per engine/store. The caller's
//! snapshot, opaque checkpoint, events and transaction hook commit atomically.
//! Hooks must not write journal run/event keys or publish effects externally.
//! Notifications are hints; reconnect and delivery always read committed rows.
//! Controller installation, recovery events and engine routing remain adapters.

use crate::store::{BlobRef, Record, RecordKey, Result, Store, StoreError, Write};
use jope_api::{
    decimal::DecimalU64, ids::RunId, EventBody, PublicEvent, RunSnapshot, TelemetryEvent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{sync::Arc, time::Duration};
use tokio::sync::{broadcast, watch};

pub struct EventLog {
    store: Arc<Store>,
    wake: watch::Sender<()>,
    telemetry: broadcast::Sender<TelemetryEvent>,
}

/// The expected revision is None only when creating a previously absent run.
/// snapshot.event_watermark must be the last observed committed watermark.
pub struct RunUpdate {
    pub expected_revision: Option<u64>,
    pub snapshot: RunSnapshot,
    pub checkpoint: Value,
    pub blobs: Vec<BlobRef>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JournalRun {
    snapshot: RunSnapshot,
    checkpoint: Value,
}

#[derive(Debug)]
pub struct JournalCommit<T> {
    pub effect: T,
    pub snapshot: RunSnapshot,
    pub events: Vec<PublicEvent>,
}

#[derive(Debug, Serialize)]
pub struct EventPage {
    pub events: Vec<PublicEvent>,
    pub next_after: DecimalU64,
    #[serde(skip)]
    pub(crate) watermark: u64,
}

impl EventLog {
    pub fn new(store: Arc<Store>) -> Self {
        let (wake, _) = watch::channel(());
        let (telemetry, _) = broadcast::channel(16);
        Self {
            store,
            wake,
            telemetry,
        }
    }

    fn journal_run(record: Record, run: &RunId) -> Result<JournalRun> {
        let mut stored: JournalRun = serde_json::from_value(record.data)?;
        let recovery = record
            .run
            .ok_or(StoreError::Integrity("journal run metadata missing"))?;
        if &stored.snapshot.run_id != run {
            return Err(StoreError::Integrity("journal run identity mismatch"));
        }
        // Store recovery updates these fields without rewriting opaque data.
        stored.snapshot.state = recovery.state;
        stored.snapshot.revision = DecimalU64(recovery.revision);
        Ok(stored)
    }

    pub fn snapshot(&self, run: &RunId) -> Result<Option<RunSnapshot>> {
        self.store
            .get(&RecordKey::Run(run.clone()))?
            .map(|r| Self::journal_run(r, run).map(|r| r.snapshot))
            .transpose()
    }

    /// Blocking store operation. Install a candidate controller only after this
    /// succeeds. A commit IO error has an unknown outcome: recover/read back.
    pub fn commit<T>(
        &self,
        mut update: RunUpdate,
        bodies: Vec<EventBody>,
        apply: impl FnOnce(&mut Write<'_>) -> Result<T>,
    ) -> Result<JournalCommit<T>> {
        let run = update.snapshot.run_id.clone();
        let result = self.store.transaction(|tx| {
            let key = RecordKey::Run(run.clone());
            let previous = tx.get(&key)?;
            // A paused run remains interrupted until Continue makes it running.
            let interrupted = previous
                .as_ref()
                .and_then(|r| r.run.as_ref())
                .is_some_and(|r| r.interrupted)
                && update.snapshot.state != jope_api::RunState::Running;
            let old = previous.map(|r| Self::journal_run(r, &run)).transpose()?;
            let revision = old.as_ref().map(|r| r.snapshot.revision.0);
            let watermark = old.as_ref().map_or(0, |r| r.snapshot.event_watermark.0);
            if revision != update.expected_revision
                || update.snapshot.event_watermark.0 != watermark
                || update.snapshot.revision.0 == 0
                || revision.is_some_and(|r| update.snapshot.revision.0 < r)
            {
                return Err(StoreError::Invalid("stale journal revision or watermark"));
            }
            let count = u64::try_from(bodies.len())
                .map_err(|_| StoreError::Invalid("event batch too large"))?;
            let last = watermark
                .checked_add(count)
                .ok_or(StoreError::Invalid("event sequence exhausted"))?;
            let time = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            let mut events = Vec::with_capacity(bodies.len());
            for (index, event) in bodies.into_iter().enumerate() {
                let sequence = watermark + index as u64 + 1;
                let event = PublicEvent {
                    run_id: run.clone(),
                    sequence: DecimalU64(sequence),
                    time: time.clone(),
                    event,
                };
                let key = RecordKey::Event {
                    run: run.clone(),
                    sequence,
                };
                if tx.get(&key)?.is_some() {
                    return Err(StoreError::Integrity("event sequence already exists"));
                }
                tx.put(
                    &key,
                    &Record::metadata(serde_json::to_value(&event)?, vec![]),
                )?;
                events.push(event);
            }
            update.snapshot.event_watermark = DecimalU64(last);
            let snapshot = update.snapshot;
            let stored = JournalRun {
                snapshot: snapshot.clone(),
                checkpoint: update.checkpoint,
            };
            let mut record = Record::run(
                serde_json::to_value(stored)?,
                update.blobs,
                snapshot.state,
                snapshot.revision.0,
            );
            record.run.as_mut().unwrap().interrupted = interrupted;
            tx.put(&key, &record)?;
            // Tests hold this real transaction after event and checkpoint writes.
            let effect = apply(tx)?;
            Ok(JournalCommit {
                effect,
                snapshot,
                events,
            })
        })?;
        if !result.events.is_empty() {
            self.wake.send_replace(());
        }
        Ok(result)
    }

    /// Scan at most limit consecutive records; filtered pages can be empty while
    /// next_after advances. An absent/corrupt durable row is an error, never a gap.
    pub fn page(
        &self,
        run: &RunId,
        after: u64,
        limit: usize,
        kinds: &[String],
    ) -> Result<Option<EventPage>> {
        if !(1..=256).contains(&limit) || kinds.iter().any(|k| !EVENT_KINDS.contains(&k.as_str())) {
            return Err(StoreError::Invalid("invalid event limit or kinds"));
        }
        let Some(snapshot) = self.snapshot(run)? else {
            return Ok(None);
        };
        let watermark = snapshot.event_watermark.0;
        if after > watermark {
            return Err(StoreError::Invalid("event cursor exceeds watermark"));
        }
        let end = after.saturating_add(limit as u64).min(watermark);
        let mut events = Vec::new();
        for offset in 0..end - after {
            let sequence = after + offset + 1;
            let row = self
                .store
                .get(&RecordKey::Event {
                    run: run.clone(),
                    sequence,
                })?
                .ok_or(StoreError::Integrity("durable event missing"))?;
            if row.run.is_some() || !row.blobs.is_empty() {
                return Err(StoreError::Integrity("invalid durable event record"));
            }
            let event: PublicEvent = serde_json::from_value(row.data)?;
            if event.run_id != *run || event.sequence.0 != sequence {
                return Err(StoreError::Integrity("durable event identity mismatch"));
            }
            if kinds.is_empty() || kinds.iter().any(|k| k == event_kind(&event.event)) {
                events.push(event);
            }
        }
        Ok(Some(EventPage {
            events,
            next_after: DecimalU64(end),
            watermark,
        }))
    }

    pub(crate) fn subscribe(&self) -> watch::Receiver<()> {
        self.wake.subscribe()
    }
    pub(crate) fn subscribe_telemetry(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.telemetry.subscribe()
    }

    /// Ephemeral samples never allocate a sequence or touch the store.
    pub fn publish_telemetry(&self, sample: TelemetryEvent) {
        let _ = self.telemetry.send(sample);
    }

    pub async fn wait_after(
        self: &Arc<Self>,
        run: &RunId,
        after: u64,
        wait: Duration,
    ) -> Result<Option<RunSnapshot>> {
        if wait > Duration::from_secs(60) {
            return Err(StoreError::Invalid("wait exceeds 60 seconds"));
        }
        // Register before the first read, so a commit in between cannot be lost.
        let mut wake = self.subscribe();
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let log = Arc::clone(self);
            let id = run.clone();
            let snapshot = tokio::task::spawn_blocking(move || log.snapshot(&id))
                .await
                .map_err(|_| StoreError::Integrity("snapshot reader failed"))??;
            let Some(snapshot) = snapshot else {
                return Ok(None);
            };
            if after > snapshot.event_watermark.0 {
                return Err(StoreError::Invalid("event cursor exceeds watermark"));
            }
            if snapshot.event_watermark.0 > after || tokio::time::Instant::now() >= deadline {
                return Ok(Some(snapshot));
            }
            tokio::select! {
                changed = wake.changed() => changed.map_err(|_| StoreError::Integrity("event notifier closed"))?,
                _ = tokio::time::sleep_until(deadline) => {},
            }
        }
    }
}

pub(crate) const EVENT_KINDS: [&str; 10] = [
    "run.state",
    "run.config",
    "plan",
    "experiment.started",
    "experiment.finished",
    "algorithm.accepted",
    "reasoning",
    "warning",
    "export.completed",
    "run.finished",
];

pub(crate) fn event_kind(body: &EventBody) -> &'static str {
    match body {
        EventBody::RunState(_) => "run.state",
        EventBody::RunConfig(_) => "run.config",
        EventBody::Plan(_) => "plan",
        EventBody::ExperimentStarted(_) => "experiment.started",
        EventBody::ExperimentFinished(_) => "experiment.finished",
        EventBody::AlgorithmAccepted(_) => "algorithm.accepted",
        EventBody::Reasoning(_) => "reasoning",
        EventBody::Warning(_) => "warning",
        EventBody::ExportCompleted(_) => "export.completed",
        EventBody::RunFinished(_) => "run.finished",
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use jope_api::{
        BudgetTotal, BudgetUsed, Effort, ReasoningBody, RunPhase, RunState, TelemetryBody,
    };
    use serde_json::json;

    pub(crate) fn test_body(summary: &str) -> EventBody {
        EventBody::Reasoning(ReasoningBody {
            stage: "searching".into(),
            algorithm_id: None,
            summary: summary.into(),
            evidence: None,
        })
    }

    pub(crate) fn fixture() -> (tempfile::TempDir, Arc<EventLog>, RunSnapshot) {
        let directory = tempfile::tempdir().unwrap();
        let log = Arc::new(EventLog::new(Arc::new(
            Store::open(directory.path()).unwrap(),
        )));
        let snapshot = RunSnapshot {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            revision: DecimalU64(1),
            state: RunState::Paused,
            phase: RunPhase::Searching,
            pending: None,
            effort: Effort::Medium,
            budget_used: BudgetUsed {
                evaluations: DecimalU64(0),
                wall_seconds: "0".into(),
            },
            budget_total: BudgetTotal {
                evaluations: DecimalU64(128),
                wall_seconds: DecimalU64(60),
            },
            best: None,
            event_watermark: DecimalU64(0),
            warnings: vec![],
            completion_reason: None,
            evaluation_receipt: None,
        };
        log.commit(
            RunUpdate {
                expected_revision: None,
                snapshot: snapshot.clone(),
                checkpoint: json!({"round": 0}),
                blobs: vec![],
            },
            vec![],
            |_| Ok(()),
        )
        .unwrap();
        (directory, log, snapshot)
    }

    pub(crate) fn append(
        log: &EventLog,
        snapshot: &RunSnapshot,
        bodies: Vec<EventBody>,
    ) -> RunSnapshot {
        let mut next = snapshot.clone();
        next.revision.0 += 1;
        log.commit(
            RunUpdate {
                expected_revision: Some(snapshot.revision.0),
                snapshot: next,
                checkpoint: json!({"round": snapshot.revision.0}),
                blobs: vec![],
            },
            bodies,
            |_| Ok(()),
        )
        .unwrap()
        .snapshot
    }

    #[tokio::test]
    async fn held_real_transaction_hides_events_checkpoint_and_other_effects_then_wakes_under_50ms()
    {
        let (_dir, log, snapshot) = fixture();
        let mut wake = log.subscribe();
        let reader = Arc::clone(&log);
        let run = snapshot.run_id.clone();
        let waiter = tokio::spawn(async move {
            reader
                .wait_after(&run, 0, Duration::from_secs(60))
                .await
                .unwrap()
                .unwrap()
        });
        tokio::task::yield_now().await;
        let (staged_tx, staged_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let (committed_tx, committed_rx) = tokio::sync::oneshot::channel();
        let writer = Arc::clone(&log);
        let mut next = snapshot.clone();
        next.revision.0 += 1;
        let run = snapshot.run_id.clone();
        let worker = std::thread::spawn(move || {
            let result = writer.commit(
                RunUpdate {
                    expected_revision: Some(1),
                    snapshot: next,
                    checkpoint: json!({"round": 1}),
                    blobs: vec![],
                },
                vec![test_body("committed")],
                |tx| {
                    assert!(tx
                        .get(&RecordKey::Event {
                            run: run.clone(),
                            sequence: 1
                        })?
                        .is_some());
                    tx.put(
                        &RecordKey::Evaluation { run, number: 7 },
                        &Record::metadata(json!({"finished": true}), vec![]),
                    )?;
                    staged_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                },
            );
            let _ = committed_tx.send(std::time::Instant::now());
            result
        });
        staged_rx.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(!wake.has_changed().unwrap());
        assert_eq!(
            log.snapshot(&snapshot.run_id).unwrap(),
            Some(snapshot.clone())
        );
        assert!(log
            .page(&snapshot.run_id, 0, 10, &[])
            .unwrap()
            .unwrap()
            .events
            .is_empty());
        assert!(log
            .store
            .get(&RecordKey::Evaluation {
                run: snapshot.run_id.clone(),
                number: 7
            })
            .unwrap()
            .is_none());
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!waiter.is_finished());
        release_tx.send(()).unwrap();
        // Publication/fsync is setup; measure delivery once the event is committed.
        let start = tokio::time::timeout(Duration::from_secs(20), committed_rx)
            .await
            .unwrap()
            .unwrap();
        let visible = tokio::time::timeout(
            Duration::from_millis(50).saturating_sub(start.elapsed()),
            waiter,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(start.elapsed() < Duration::from_millis(50));
        worker.join().unwrap().unwrap();
        wake.changed().await.unwrap();
        assert_eq!(visible.event_watermark.0, 1);
        let record = log
            .store
            .get(&RecordKey::Run(snapshot.run_id.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(record.data["checkpoint"], json!({"round": 1}));
        assert!(log
            .store
            .get(&RecordKey::Evaluation {
                run: snapshot.run_id,
                number: 7
            })
            .unwrap()
            .is_some());
    }

    #[test]
    fn aborted_and_stale_transactions_leave_no_events_or_effects() {
        let (_dir, log, snapshot) = fixture();
        let wake = log.subscribe();
        let mut next = snapshot.clone();
        next.revision.0 += 1;
        let result: Result<JournalCommit<()>> = log.commit(
            RunUpdate {
                expected_revision: Some(1),
                snapshot: next,
                checkpoint: json!("uncommitted"),
                blobs: vec![],
            },
            vec![test_body("uncommitted")],
            |tx| {
                tx.put(
                    &RecordKey::Evaluation {
                        run: snapshot.run_id.clone(),
                        number: 9,
                    },
                    &Record::metadata(json!("abort"), vec![]),
                )?;
                Err(StoreError::Invalid("simulated refusal"))
            },
        );
        assert!(result.is_err());
        assert!(!wake.has_changed().unwrap());
        assert_eq!(
            log.snapshot(&snapshot.run_id).unwrap(),
            Some(snapshot.clone())
        );
        assert!(log
            .store
            .get(&RecordKey::Event {
                run: snapshot.run_id.clone(),
                sequence: 1
            })
            .unwrap()
            .is_none());
        assert!(log
            .store
            .get(&RecordKey::Evaluation {
                run: snapshot.run_id.clone(),
                number: 9
            })
            .unwrap()
            .is_none());
        let committed = append(&log, &snapshot, vec![test_body("visible")]);
        let stale = log.commit(
            RunUpdate {
                expected_revision: Some(1),
                snapshot,
                checkpoint: json!(null),
                blobs: vec![],
            },
            vec![test_body("stale")],
            |_| Ok(()),
        );
        assert!(stale.is_err());
        let page = log.page(&committed.run_id, 0, 256, &[]).unwrap().unwrap();
        assert_eq!(page.events.len(), 1);
        assert_eq!(page.events[0].sequence.0, 1);
    }

    #[test]
    fn pages_have_contiguous_sequences_and_filtered_empty_pages_advance_the_cursor() {
        let (_dir, log, snapshot) = fixture();
        let current = append(
            &log,
            &snapshot,
            (0..300).map(|i| test_body(&i.to_string())).collect(),
        );
        let mut cursor = 0;
        let mut all = vec![];
        while cursor < 300 {
            let page = log
                .page(&current.run_id, cursor, 128, &[])
                .unwrap()
                .unwrap();
            all.extend(page.events.into_iter().map(|e| e.sequence.0));
            cursor = page.next_after.0;
        }
        assert_eq!(all, (1..=300).collect::<Vec<_>>());
        let filtered = log
            .page(&current.run_id, 0, 100, &["warning".into()])
            .unwrap()
            .unwrap();
        assert!(filtered.events.is_empty());
        assert_eq!(filtered.next_after.0, 100);
        assert!(log.page(&current.run_id, 0, 0, &[]).is_err());
        assert!(log.page(&current.run_id, 0, 257, &[]).is_err());
        assert!(log.page(&current.run_id, 301, 1, &[]).is_err());
        assert!(log
            .page(&current.run_id, 0, 1, &["telemetry".into()])
            .is_err());
    }

    #[test]
    fn reopen_retains_sequence_and_merges_store_recovery_metadata_without_losing_checkpoint() {
        let (dir, log, snapshot) = fixture();
        let mut running = snapshot.clone();
        running.state = RunState::Running;
        running.revision.0 += 1;
        let committed = log
            .commit(
                RunUpdate {
                    expected_revision: Some(1),
                    snapshot: running,
                    checkpoint: json!({"seed": 917}),
                    blobs: vec![],
                },
                vec![test_body("before recovery")],
                |_| Ok(()),
            )
            .unwrap();
        let before = serde_json::to_value(&committed.events).unwrap();
        drop(log);
        let reopened = EventLog::new(Arc::new(Store::open(dir.path()).unwrap()));
        let recovered = reopened.snapshot(&snapshot.run_id).unwrap().unwrap();
        assert_eq!(recovered.state, RunState::Paused);
        assert_eq!(recovered.revision.0, 3);
        assert_eq!(recovered.event_watermark.0, 1);
        assert_eq!(
            reopened
                .store
                .get(&RecordKey::Run(snapshot.run_id.clone()))
                .unwrap()
                .unwrap()
                .data["checkpoint"],
            json!({"seed": 917})
        );
        assert_eq!(
            serde_json::to_value(
                reopened
                    .page(&snapshot.run_id, 0, 256, &[])
                    .unwrap()
                    .unwrap()
                    .events
            )
            .unwrap(),
            before
        );
        let next = append(&reopened, &recovered, vec![test_body("after recovery")]);
        assert_eq!(next.event_watermark.0, 2);
        assert!(
            reopened
                .store
                .get(&RecordKey::Run(snapshot.run_id))
                .unwrap()
                .unwrap()
                .run
                .unwrap()
                .interrupted,
            "ordinary paused journal commits must retain the recovery marker"
        );
    }

    #[test]
    fn existing_event_keys_corrupt_rows_and_exhausted_sequences_are_not_silently_overwritten() {
        let (_dir, log, snapshot) = fixture();
        log.store
            .transaction(|tx| {
                tx.put(
                    &RecordKey::Event {
                        run: snapshot.run_id.clone(),
                        sequence: 1,
                    },
                    &Record::metadata(json!({"bad": true}), vec![]),
                )
            })
            .unwrap();
        assert!(log
            .commit(
                RunUpdate {
                    expected_revision: Some(1),
                    snapshot: snapshot.clone(),
                    checkpoint: json!(null),
                    blobs: vec![]
                },
                vec![test_body("collision")],
                |_| Ok(())
            )
            .is_err());
        let mut high = snapshot.clone();
        high.event_watermark = DecimalU64(u64::MAX);
        log.store
            .transaction(|tx| {
                tx.put(
                    &RecordKey::Run(high.run_id.clone()),
                    &Record::run(
                        serde_json::to_value(JournalRun {
                            snapshot: high.clone(),
                            checkpoint: json!(null),
                        })?,
                        vec![],
                        high.state,
                        1,
                    ),
                )
            })
            .unwrap();
        assert!(log
            .page(&high.run_id, u64::MAX, 256, &[])
            .unwrap()
            .unwrap()
            .events
            .is_empty());
        assert!(log
            .commit(
                RunUpdate {
                    expected_revision: Some(1),
                    snapshot: high.clone(),
                    checkpoint: json!(null),
                    blobs: vec![]
                },
                vec![test_body("overflow")],
                |_| Ok(())
            )
            .is_err());
        assert!(log.page(&snapshot.run_id, 0, 1, &[]).is_err());
        assert_eq!(
            log.snapshot(&high.run_id)
                .unwrap()
                .unwrap()
                .event_watermark
                .0,
            u64::MAX
        );
    }

    #[tokio::test]
    async fn long_poll_times_out_without_waking_for_revision_only_or_telemetry() {
        let (_dir, log, snapshot) = fixture();
        // Finish the synchronous revision-only write before starting the deadline.
        let current = append(&log, &snapshot, vec![]);
        let reader = Arc::clone(&log);
        let id = snapshot.run_id.clone();
        let start = std::time::Instant::now();
        let waiter = tokio::spawn(async move {
            reader
                .wait_after(&id, 0, Duration::from_millis(40))
                .await
                .unwrap()
                .unwrap()
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        // An unrelated/spurious wake hint must also recheck the event watermark.
        log.wake.send_replace(());
        log.publish_telemetry(TelemetryEvent {
            run_id: snapshot.run_id.clone(),
            time: "2026-10-30T00:00:00Z".into(),
            event: TelemetryBody::Telemetry(vec![]),
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!waiter.is_finished());
        let result = waiter.await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert!(start.elapsed() < Duration::from_millis(250));
        assert_eq!(result.revision, current.revision);
        assert_eq!(result.event_watermark.0, 0);
    }

    #[cfg(feature = "failpoints")]
    #[test]
    #[ignore = "subprocess helper explicitly invoked by the crash supervisor"]
    fn event_crash_child() {
        let home = std::env::var("JOPE_EVENT_TEST_HOME").unwrap();
        let log = EventLog::new(Arc::new(Store::open(home).unwrap()));
        let run: RunId = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap();
        let old = log.snapshot(&run).unwrap().unwrap();
        let mut next = old.clone();
        next.revision.0 += 1;
        log.commit(
            RunUpdate {
                expected_revision: Some(old.revision.0),
                snapshot: next,
                checkpoint: json!({"child": true}),
                blobs: vec![],
            },
            vec![test_body("child event")],
            |_| {
                if std::env::var("JOPE_EVENT_TEST_POINT").as_deref() == Ok("before") {
                    eprintln!("JOPE_EVENT_TEST_CRASH:before");
                    std::process::abort();
                }
                Ok(())
            },
        )
        .unwrap();
        panic!("crash boundary did not execute");
    }

    #[cfg(feature = "failpoints")]
    #[test]
    fn process_crash_before_and_after_commit_never_reissues_a_published_sequence() {
        for point in ["before", "after"] {
            let (dir, log, snapshot) = fixture();
            let first = append(&log, &snapshot, vec![test_body("first")]);
            drop(log);
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "events::tests::event_crash_child",
                    "--ignored",
                    "--nocapture",
                ])
                .env("JOPE_EVENT_TEST_HOME", dir.path())
                .env("JOPE_EVENT_TEST_POINT", point)
                .env_remove("JOPE_STORE_FAILPOINT");
            if point == "after" {
                command.env("JOPE_STORE_FAILPOINT", "after_commit");
            }
            let output = command.output().unwrap();
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(if point == "before" {
                    "JOPE_EVENT_TEST_CRASH:before"
                } else {
                    "JOPE_STORE_CRASH:after_commit"
                }),
                "{stderr}"
            );
            let log = EventLog::new(Arc::new(Store::open(dir.path()).unwrap()));
            let recovered = log.snapshot(&first.run_id).unwrap().unwrap();
            assert_eq!(
                recovered.event_watermark.0,
                if point == "before" { 1 } else { 2 }
            );
            let existing = log
                .page(&first.run_id, 0, 256, &[])
                .unwrap()
                .unwrap()
                .events;
            let saved = serde_json::to_value(&existing).unwrap();
            let next = append(&log, &recovered, vec![test_body("next")]);
            let all = log
                .page(&first.run_id, 0, 256, &[])
                .unwrap()
                .unwrap()
                .events;
            assert_eq!(serde_json::to_value(&all[..existing.len()]).unwrap(), saved);
            assert_eq!(
                all.last().unwrap().sequence.0,
                recovered.event_watermark.0 + 1
            );
            assert_eq!(next.event_watermark.0, all.len() as u64);
        }
    }
}
