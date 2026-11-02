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
pub(crate) mod tests;
