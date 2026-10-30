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

impl RecorderBuilder {
    pub fn new(run_dir: impl Into<PathBuf>, metadata: RunMetadata) -> Self {
        Self {
            run_dir: run_dir.into(),
            run_id: Ulid::new(),
            metadata,
            source_names: Vec::new(),
            queue_capacity: EVENT_QUEUE_CAPACITY,
        }
    }

    pub fn with_run_id(mut self, run_id: Ulid) -> Self {
        self.run_id = run_id;
        self
    }

    pub fn with_queue_capacity(mut self, capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > EVENT_QUEUE_CAPACITY {
            bail!("event queue capacity must be in 1..={EVENT_QUEUE_CAPACITY}");
        }
        self.queue_capacity = capacity;
        Ok(self)
    }

    /// Register a bounded producer at run start. Duplicate names and producer 17 are refused.
    pub fn add_producer(&mut self, name: impl Into<String>) -> Result<u16> {
        if self.source_names.len() >= MAX_PRODUCERS {
            bail!("a telemetry run supports at most {MAX_PRODUCERS} producers");
        }
        let name = name.into();
        if name.is_empty() || name.len() > 128 {
            bail!("producer name must contain 1..=128 UTF-8 bytes");
        }
        if self.source_names.iter().any(|s| s == &name) {
            bail!("producer {name:?} is already registered");
        }
        self.source_names.push(name);
        Ok(self.source_names.len() as u16)
    }

    pub fn start(mut self) -> Result<(Recorder, Vec<ProducerHandle>)> {
        let group_count = self
            .metadata
            .metric_groups
            .values()
            .map(BTreeMap::len)
            .sum::<usize>();
        let group_label_bytes = self
            .metadata
            .metric_groups
            .values()
            .flat_map(BTreeMap::values)
            .map(String::len)
            .sum::<usize>();
        if self.metadata.metric_groups.len() > 4096
            || group_count > 4096
            || group_label_bytes > 1 << 20
            || self
                .metadata
                .metric_groups
                .values()
                .flat_map(BTreeMap::values)
                .any(|label| label.is_empty() || label.len() > 128)
        {
            bail!("metric group registry exceeds the frozen 1 MiB / 4,096-label bounds");
        }
        if self.source_names.is_empty() {
            self.source_names.push("jope-telemetry".to_owned());
        }
        let parent = self
            .run_dir
            .parent()
            .ok_or_else(|| anyhow!("run directory must have a parent"))?;
        fs::create_dir_all(parent).context("create telemetry output parent")?;
        fs::create_dir(&self.run_dir).with_context(|| {
            format!(
                "create new telemetry run directory {}",
                self.run_dir.display()
            )
        })?;

        let sources = self
            .source_names
            .iter()
            .enumerate()
            .map(|(i, name)| SourceDef {
                id: (i + 1) as u16,
                name: name.clone(),
            })
            .collect::<Vec<_>>();
        self.metadata.sources = sources.clone();
        let mut manifest = RunManifest::new(self.run_id, self.metadata);
        manifest.event_queue_capacity = self.queue_capacity;
        manifest.event_queue_producers = sources.len();
        manifest.record_queue_capacity = RECORD_QUEUE_CAPACITY;
        manifest.max_record_bytes = MAX_RECORD_BYTES;
        let manifest_path = self.run_dir.join("manifest.json");
        manifest
            .write_atomic(&manifest_path)
            .context("write initial telemetry manifest")?;
        let mut writer = JsonlWriter::create(&self.run_dir).context("create telemetry JSONL")?;

        let run_id_bytes = self.run_id.to_bytes();
        let started = Instant::now();
        let stopping = Arc::new(AtomicBool::new(false));
        let close_requested = Arc::new(AtomicBool::new(false));
        let record_drops = Arc::new(AtomicU64::new(0));
        let record_in_flight = Arc::new(AtomicU64::new(0));
        let oversized_record_drops = Arc::new(AtomicU64::new(0));
        let sink_errors = Arc::new(AtomicU64::new(0));
        let (record_tx, record_rx) = mpsc::sync_channel(RECORD_QUEUE_CAPACITY);

        let mut producers = Vec::with_capacity(sources.len());
        let mut drain_producers = Vec::with_capacity(sources.len());
        for source in &sources {
            let (producer, consumer) = RingBuffer::new(self.queue_capacity);
            let stats = Arc::new(ProducerStats::default());
            producers.push(ProducerHandle {
                producer,
                stats: stats.clone(),
                stopping: stopping.clone(),
                started,
                run_id: run_id_bytes,
                source_id: source.id,
                sequence: 0,
                metric_layout: Arc::new(Vec::new()),
            });
            drain_producers.push(DrainProducer {
                consumer,
                stats,
                capacity: self.queue_capacity,
                dequeued: 0,
                high_water: 0,
            });
        }

        let worker_stop = stopping.clone();
        let worker_close = close_requested.clone();
        let worker_record_drops = record_drops.clone();
        let worker_record_in_flight = record_in_flight.clone();
        let worker_oversized_drops = oversized_record_drops.clone();
        let worker_sink_errors = sink_errors.clone();
        let run_dir = self.run_dir.clone();
        let run_id = self.run_id;
        let join = thread::Builder::new()
            .name("jope-telemetry-drain".to_owned())
            .stack_size(512 * 1024)
            .spawn(move || {
                drain_loop(
                    &run_dir,
                    manifest,
                    &mut writer,
                    drain_producers,
                    record_rx,
                    worker_record_in_flight,
                    worker_stop,
                    worker_close,
                    worker_record_drops,
                    worker_oversized_drops,
                    worker_sink_errors,
                )
            })
            .context("spawn telemetry drain thread")?;

        Ok((
            Recorder {
                run_dir: self.run_dir,
                run_id,
                stopping,
                close_requested,
                join: Some(join),
                record_tx,
                record_drops,
                record_in_flight,
                oversized_record_drops,
                sink_errors,
            },
            producers,
        ))
    }
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

impl ProducerHandle {
    pub fn source_id(&self) -> u16 {
        self.source_id
    }

    /// Wall-clock seconds since the recorder started, on the clock behind `monotonic_ns`.
    pub fn run_elapsed_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn set_metric_layout(&mut self, columns: Vec<(u16, u16)>) -> Result<()> {
        if columns.len() > 4096 {
            bail!("metric layout exceeds the 4,096-column registry cap");
        }
        for &(metric_id, _) in &columns {
            if crate::train::metric(metric_id).is_none() {
                bail!("unknown metric id {metric_id:#06x}");
            }
        }
        self.metric_layout = Arc::new(columns);
        Ok(())
    }

    #[inline]
    pub fn emit_scalar(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> bool {
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let event = self.event(metric_id, group_id, step, value, sampled);
        let pushed = self.try_push(event);
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        pushed
    }

    #[inline]
    pub fn emit_unavailable(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        reason_id: u16,
        sampled: bool,
    ) -> bool {
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let event = Event::unavailable(
            self.run_id,
            self.next_sequence(),
            self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            self.source_id,
            metric_id,
            group_id,
            step,
            reason_id,
            sampled,
        );
        let pushed = self.try_push(event);
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        pushed
    }

    /// Submit row-major float64 values and exact u64 steps from one Python batch.
    /// Missing values use the `StepBatch` sentinel and are not turned into fake zero samples.
    pub fn submit_values(&mut self, steps: &[u64], rows: usize, values: &[f64]) -> Result<u64> {
        let metric_layout = self.metric_layout.clone();
        let width = metric_layout.len();
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow!("metric batch dimensions overflow"))?;
        if rows > 32 || steps.len() < rows || values.len() < expected {
            bail!("metric batch dimensions exceed or do not match the registered layout");
        }
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let mut dropped = 0;
        for (row, &step) in steps[..rows].iter().enumerate() {
            for (column, &(metric_id, group_id)) in metric_layout.iter().enumerate() {
                let value = values[row * width + column];
                if value.to_bits() == MISSING_VALUE_BITS {
                    continue;
                }
                let event = self.event(metric_id, group_id, step, value, false);
                if !self.try_push(event) {
                    dropped += 1;
                }
            }
        }
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        Ok(dropped)
    }

    #[inline]
    fn event(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> Event {
        let sequence = self.next_sequence();
        Event::scalar(
            self.run_id,
            sequence,
            self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            self.source_id,
            metric_id,
            group_id,
            step,
            value,
            sampled,
        )
    }

    #[inline]
    fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        sequence
    }

    #[inline]
    fn try_push(&mut self, event: Event) -> bool {
        if self.stopping.load(Ordering::Relaxed) {
            self.stats.late_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        match self.producer.push(event) {
            Ok(()) => true,
            Err(_) => {
                self.stats.drops.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::RunMetadata;
    use std::fs;
    use std::thread;

    #[test]
    fn producer_is_nonblocking_bounded_and_counted() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default())
            .with_queue_capacity(1)
            .unwrap();
        b.add_producer("test").unwrap();
        let (recorder, mut producers) = b.start().unwrap();
        let producer = &mut producers[0];
        assert!(producer.emit_scalar(crate::train::TRAIN_LOSS, 0, 0, 1.0, false));
        assert!(!producer.emit_scalar(crate::train::TRAIN_LOSS, 0, 1, 0.9, false));
        // Allow the one drain thread to consume the event before shutdown.
        thread::sleep(Duration::from_millis(10));
        let manifest = recorder.close().unwrap();
        assert_eq!(manifest.event_drops, 1);
        assert_eq!(manifest.events_written, 1);
        assert!(dir.path().join("run/events.jsonl").exists());
    }

    #[test]
    fn record_lane_is_bounded_and_reports_oversize() {
        let dir = tempfile::tempdir().unwrap();
        let mut b = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default());
        b.add_producer("test").unwrap();
        let (recorder, _) = b.start().unwrap();
        let too_large = serde_json::json!({"payload": "x".repeat(MAX_RECORD_BYTES)});
        assert!(!recorder.record_value("oversize/v1", too_large));
        let manifest = recorder.close().unwrap();
        assert_eq!(manifest.oversized_record_drops, 1);
        assert_eq!(manifest.record_drops, 1);
        let text = fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
        assert!(text.is_empty());
    }
    #[test]
    fn interrupt_finalizes_manifest_as_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default());
        builder.add_producer("test").unwrap();
        let (recorder, mut producers) = builder.start().unwrap();
        assert!(producers[0].emit_scalar(crate::train::TRAIN_LOSS, 0, 0, 1.0, false));

        let manifest = recorder.interrupt().unwrap();

        assert!(matches!(manifest.status, RunStatus::Interrupted));
        #[cfg(feature = "parquet")]
        assert!(dir.path().join("run/metrics.parquet").exists());
    }
}
