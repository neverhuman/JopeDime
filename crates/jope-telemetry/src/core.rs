//! Bounded run lifecycle, wait-free producer queues and one asynchronous drain.

use crate::event::Event;
use crate::manifest::{RunManifest, RunMetadata, RunStatus, SourceDef};
use crate::record::RecordSink;
use crate::sink::jsonl::{JsonlWriter, MAX_RECORD_BYTES, RECORD_QUEUE_CAPACITY};
use anyhow::{anyhow, bail, Context, Result};
use rtrb::{Consumer, Producer, RingBuffer};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use ulid::Ulid;

pub const MAX_PRODUCERS: usize = 16;
pub const EVENT_QUEUE_CAPACITY: usize = 4096;
pub const DRAIN_BATCH_PER_PRODUCER: usize = 256;
const MANIFEST_PERIOD: Duration = Duration::from_secs(1);
const DRAIN_IDLE_SLEEP: Duration = Duration::from_millis(1);

#[derive(Default)]
struct ProducerStats {
    drops: AtomicU64,
    late_drops: AtomicU64,
    in_flight: AtomicU64,
}

struct DrainProducer {
    consumer: Consumer<Event>,
    stats: Arc<ProducerStats>,
    capacity: usize,
    dequeued: u64,
    high_water: u64,
}

/// Configures all producer rings before starting the drain thread.
pub struct RecorderBuilder {
    run_dir: PathBuf,
    run_id: Ulid,
    metadata: RunMetadata,
    source_names: Vec<String>,
    queue_capacity: usize,
}

/// Per-thread SPSC producer. It owns no heap storage after construction and never waits.
pub struct ProducerHandle {
    producer: Producer<Event>,
    stats: Arc<ProducerStats>,
    stopping: Arc<AtomicBool>,
    started: Instant,
    run_id: [u8; 16],
    source_id: u16,
    sequence: u64,
    metric_layout: Arc<Vec<(u16, u16)>>,
}

/// Run owner. Joining and rollup happen only when `close` is called after training.
pub struct Recorder {
    run_dir: PathBuf,
    run_id: Ulid,
    stopping: Arc<AtomicBool>,
    close_requested: Arc<AtomicBool>,
    join: Option<JoinHandle<DrainOutcome>>,
    record_tx: SyncSender<Vec<u8>>,
    record_drops: Arc<AtomicU64>,
    record_in_flight: Arc<AtomicU64>,
    oversized_record_drops: Arc<AtomicU64>,
    sink_errors: Arc<AtomicU64>,
}

#[derive(Clone, Debug, Default)]
pub struct HealthSnapshot {
    pub events_written: u64,
    pub records_written: u64,
    pub event_drops: u64,
    pub record_drops: u64,
    pub oversized_record_drops: u64,
    pub sink_errors: u64,
}

impl Recorder {
    pub fn run_id(&self) -> Ulid {
        self.run_id
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn health(&self) -> HealthSnapshot {
        HealthSnapshot {
            record_drops: self.record_drops.load(Ordering::Relaxed),
            oversized_record_drops: self.oversized_record_drops.load(Ordering::Relaxed),
            sink_errors: self.sink_errors.load(Ordering::Relaxed),
            ..HealthSnapshot::default()
        }
    }

    pub fn close(self) -> Result<RunManifest> {
        self.stop_and_finalize(true)
    }

    /// Stop the drain and finalize a run whose producer exited before normal completion.
    pub fn interrupt(self) -> Result<RunManifest> {
        self.stop_and_finalize(false)
    }

    fn stop_and_finalize(mut self, complete: bool) -> Result<RunManifest> {
        if complete {
            self.close_requested.store(true, Ordering::Release);
        }
        self.stopping.store(true, Ordering::Release);
        let mut outcome = self
            .join
            .take()
            .ok_or_else(|| anyhow!("telemetry drain already joined"))?
            .join()
            .map_err(|_| anyhow!("telemetry drain thread panicked"))?;
        self.finish_rollup_and_manifest(&mut outcome)?;
        Ok(outcome.manifest)
    }

    fn finish_rollup_and_manifest(&self, outcome: &mut DrainOutcome) -> Result<()> {
        #[cfg(feature = "parquet")]
        {
            let parquet_path = self.run_dir.join("metrics.parquet");
            if let Err(error) = crate::sink::parquet::regenerate(&outcome.event_path, &parquet_path)
            {
                outcome.manifest.sink_errors += 1;
                outcome.manifest.sink_error = Some(format!("Parquet rollup failed: {error:#}"));
                outcome.manifest.status = RunStatus::SinkFailed;
                self.sink_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(not(feature = "parquet"))]
        let _ = (&self, &outcome.event_path);

        outcome.manifest.write_atomic(&outcome.manifest_path)?;
        Ok(())
    }
}

impl RecordSink for Recorder {
    fn record_line(&self, _kind: &str, line: Vec<u8>) -> bool {
        queue_record_line(
            &self.record_tx,
            &self.stopping,
            &self.record_drops,
            &self.record_in_flight,
            &self.oversized_record_drops,
            line,
        )
    }

    fn record_encoding_error(&self, oversized: bool) {
        note_record_encoding_error(&self.record_drops, &self.oversized_record_drops, oversized);
    }
}

/// A cloneable, `Send + Sync` handle on a run's record queue, for record producers on other
/// threads (evaluator workers, a background evolution queue). It shares the recorder's drop and
/// in-flight counters, so `close` still waits for records in flight. After close every record is
/// counted as a drop.
#[derive(Clone)]
pub struct RecordHandle {
    record_tx: SyncSender<Vec<u8>>,
    stopping: Arc<AtomicBool>,
    record_drops: Arc<AtomicU64>,
    record_in_flight: Arc<AtomicU64>,
    oversized_record_drops: Arc<AtomicU64>,
}

impl Recorder {
    pub fn record_handle(&self) -> RecordHandle {
        RecordHandle {
            record_tx: self.record_tx.clone(),
            stopping: self.stopping.clone(),
            record_drops: self.record_drops.clone(),
            record_in_flight: self.record_in_flight.clone(),
            oversized_record_drops: self.oversized_record_drops.clone(),
        }
    }
}

impl RecordSink for RecordHandle {
    fn record_line(&self, _kind: &str, line: Vec<u8>) -> bool {
        queue_record_line(
            &self.record_tx,
            &self.stopping,
            &self.record_drops,
            &self.record_in_flight,
            &self.oversized_record_drops,
            line,
        )
    }

    fn record_encoding_error(&self, oversized: bool) {
        note_record_encoding_error(&self.record_drops, &self.oversized_record_drops, oversized);
    }
}

fn queue_record_line(
    record_tx: &SyncSender<Vec<u8>>,
    stopping: &AtomicBool,
    record_drops: &AtomicU64,
    record_in_flight: &AtomicU64,
    oversized_record_drops: &AtomicU64,
    line: Vec<u8>,
) -> bool {
    record_in_flight.fetch_add(1, Ordering::AcqRel);
    let _in_flight = AtomicCountGuard(record_in_flight);
    if line.len() > MAX_RECORD_BYTES || line.last() != Some(&b'\n') {
        oversized_record_drops.fetch_add(1, Ordering::Relaxed);
        record_drops.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    if stopping.load(Ordering::Relaxed) {
        record_drops.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    match record_tx.try_send(line) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
            record_drops.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

fn note_record_encoding_error(
    record_drops: &AtomicU64,
    oversized_record_drops: &AtomicU64,
    oversized: bool,
) {
    if oversized {
        oversized_record_drops.fetch_add(1, Ordering::Relaxed);
    }
    record_drops.fetch_add(1, Ordering::Relaxed);
}

pub const MISSING_VALUE_BITS: u64 = 0x7ff8_4a4f_5045_5445;

struct AtomicCountGuard<'a>(&'a AtomicU64);

impl Drop for AtomicCountGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            self.stopping.store(true, Ordering::Release);
            if let Err(error) = join.join() {
                eprintln!("jope-telemetry drain stopped during drop: {error:?}");
            }
        }
    }
}

struct DrainOutcome {
    manifest: RunManifest,
    event_path: PathBuf,
    manifest_path: PathBuf,
}

#[allow(clippy::too_many_arguments)]
fn drain_loop(
    run_dir: &Path,
    mut manifest: RunManifest,
    writer: &mut JsonlWriter,
    mut producers: Vec<DrainProducer>,
    record_rx: Receiver<Vec<u8>>,
    record_in_flight: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
    close_requested: Arc<AtomicBool>,
    record_drops: Arc<AtomicU64>,
    oversized_record_drops: Arc<AtomicU64>,
    sink_errors: Arc<AtomicU64>,
) -> DrainOutcome {
    let event_path = writer.path().to_owned();
    let manifest_path = run_dir.join("manifest.json");
    let mut sink_failed = false;
    let mut wrote_anything = false;
    let mut last_manifest = Instant::now();

    loop {
        let mut worked = false;
        for producer in &mut producers {
            for _ in 0..DRAIN_BATCH_PER_PRODUCER {
                let queued = producer.capacity.saturating_sub(producer.consumer.slots());
                producer.high_water = producer.high_water.max(queued as u64);
                match producer.consumer.pop() {
                    Ok(event) => {
                        worked = true;
                        producer.dequeued = producer.dequeued.saturating_add(1);
                        if !sink_failed {
                            match writer.append_event(&event) {
                                Ok(()) => {
                                    manifest.events_written += 1;
                                    wrote_anything = true;
                                }
                                Err(error) => {
                                    sink_failed = true;
                                    note_sink_error(&mut manifest, &sink_errors, error);
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        while let Ok(line) = record_rx.try_recv() {
            worked = true;
            if !sink_failed {
                match writer.append_record_line(&line) {
                    Ok(()) => {
                        manifest.records_written += 1;
                        wrote_anything = true;
                    }
                    Err(error) => {
                        sink_failed = true;
                        note_sink_error(&mut manifest, &sink_errors, error);
                    }
                }
            }
        }

        if wrote_anything && last_manifest.elapsed() >= MANIFEST_PERIOD {
            if let Err(error) = writer.flush() {
                sink_failed = true;
                note_sink_error(&mut manifest, &sink_errors, error);
            }
            refresh_health(
                &mut manifest,
                &producers,
                &record_drops,
                &oversized_record_drops,
                &sink_errors,
            );
            if let Err(error) = manifest.write_atomic(&manifest_path) {
                note_sink_error(&mut manifest, &sink_errors, error);
            }
            wrote_anything = false;
            last_manifest = Instant::now();
        }

        let queues_empty = producers.iter().all(|p| p.consumer.is_empty());
        let producers_idle = producers
            .iter()
            .all(|p| p.stats.in_flight.load(Ordering::Acquire) == 0);
        let records_idle = record_in_flight.load(Ordering::Acquire) == 0;
        if stopping.load(Ordering::Acquire) && queues_empty && producers_idle && records_idle {
            break;
        }
        if !worked {
            thread::sleep(DRAIN_IDLE_SLEEP);
        }
    }

    if let Err(error) = writer.flush() {
        sink_failed = true;
        note_sink_error(&mut manifest, &sink_errors, error);
    }
    refresh_health(
        &mut manifest,
        &producers,
        &record_drops,
        &oversized_record_drops,
        &sink_errors,
    );
    let status = if sink_failed {
        RunStatus::SinkFailed
    } else if close_requested.load(Ordering::Acquire) {
        RunStatus::Complete
    } else {
        RunStatus::Interrupted
    };
    manifest.finalize(status);
    if let Err(error) = manifest.write_atomic(&manifest_path) {
        note_sink_error(&mut manifest, &sink_errors, error);
        eprintln!(
            "jope-telemetry could not finalize manifest: {}",
            manifest.sink_error.as_deref().unwrap_or("unknown error")
        );
    }

    DrainOutcome {
        manifest,
        event_path,
        manifest_path,
    }
}

fn refresh_health(
    manifest: &mut RunManifest,
    producers: &[DrainProducer],
    record_drops: &AtomicU64,
    oversized_record_drops: &AtomicU64,
    sink_errors: &AtomicU64,
) {
    // At finalization every producer is quiescent and each queue is empty, so all successful
    // enqueues have been observed here. During a run this is the number drained so far.
    manifest.events_enqueued = producers.iter().map(|p| p.dequeued).sum();
    manifest.event_drops = producers
        .iter()
        .map(|p| p.stats.drops.load(Ordering::Relaxed))
        .sum();
    manifest.late_event_drops = producers
        .iter()
        .map(|p| p.stats.late_drops.load(Ordering::Relaxed))
        .sum();
    manifest.event_queue_high_water = producers.iter().map(|p| p.high_water).sum();
    manifest.record_drops = record_drops.load(Ordering::Relaxed);
    manifest.oversized_record_drops = oversized_record_drops.load(Ordering::Relaxed);
    manifest.sink_errors = sink_errors.load(Ordering::Relaxed);
    manifest.effective_sampling = manifest.requested_sampling.clone();
}

fn note_sink_error(manifest: &mut RunManifest, errors: &AtomicU64, error: impl std::fmt::Display) {
    errors.fetch_add(1, Ordering::Relaxed);
    manifest.sink_errors = errors.load(Ordering::Relaxed);
    if manifest.sink_error.is_none() {
        manifest.sink_error = Some(error.to_string());
    }
}

mod builder;
mod producer;
#[cfg(test)]
mod tests;
