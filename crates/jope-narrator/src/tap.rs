//! Bounded nonblocking admission. Producers never wait for capacity, clients or disk.
use crate::LineageFact;
use serde::{Deserialize, Serialize};
#[cfg(feature = "testing")]
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError},
    Arc,
};

pub const DEFAULT_MAX_FACTS: usize = 8192;
pub const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const MIN_FRAME_BYTES: usize = 8 * 1024;
const ENVELOPE_HEADROOM_BYTES: usize = 4 * 1024;

#[derive(Clone, Debug)]
pub struct Limits {
    /// Maximum facts admitted to the in-memory queue.
    pub max_facts: usize,
    /// Maximum aggregate queued payload bytes.
    pub max_bytes: usize,
    /// Maximum encoded size of an individual fact frame.
    pub max_frame_bytes: usize,
    /// Maximum records written to one journal.
    pub max_records: u64,
    /// Maximum bytes written to one journal.
    pub max_journal_bytes: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_facts: DEFAULT_MAX_FACTS,
            max_bytes: DEFAULT_MAX_BYTES,
            max_frame_bytes: MAX_FRAME_BYTES,
            max_records: 1_000_000,
            max_journal_bytes: 1024 * 1024 * 1024,
        }
    }
}
impl Limits {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.max_facts > 0
                && self.max_facts <= 1_000_000
                && self.max_bytes > 0
                && self.max_bytes <= 1024 * 1024 * 1024
                && self.max_frame_bytes >= MIN_FRAME_BYTES
                && self.max_frame_bytes <= MAX_FRAME_BYTES
                && (1..=1_000_000).contains(&self.max_records)
                && (1..=16 * 1024 * 1024 * 1024).contains(&self.max_journal_bytes),
            "invalid observer limits"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Links {
    /// Draw identifier associated with the fact, when available.
    pub draw_id: Option<String>,
    /// Request identifier associated with the fact, when available.
    pub request_id: Option<String>,
    /// Attempt identifier associated with the fact, when available.
    pub attempt_id: Option<String>,
    /// Search-event identifier associated with the fact, when available.
    pub search_event_id: Option<String>,
    /// Lowercase SHA-256 identity of the linked observation, when available.
    pub observation_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitStatus {
    Queued,
    Dropped,
    Closed,
    Invalid,
}

#[derive(Debug)]
struct Pending {
    bytes: Vec<u8>,
}
#[derive(Debug)]
struct Shared {
    closing: AtomicBool,
    active: AtomicUsize,
    dropped: AtomicU64,
    bytes: AtomicUsize,
    queued: AtomicUsize,
    limits: Limits,
}
#[cfg(feature = "testing")]
type RecordedFacts = Arc<Mutex<Vec<(LineageFact, Links)>>>;
#[derive(Clone, Debug)]
pub struct Tap {
    shared: Arc<Shared>,
    sender: SyncSender<Pending>,
    #[cfg(feature = "testing")]
    recording: Option<RecordedFacts>,
}
impl Tap {
    pub fn emit(&self, fact: LineageFact) -> EmitStatus {
        self.emit_linked(fact, Links::default())
    }
    pub fn emit_linked(&self, fact: LineageFact, links: Links) -> EmitStatus {
        if self.shared.closing.load(Ordering::SeqCst) {
            return EmitStatus::Closed;
        }
        self.shared.active.fetch_add(1, Ordering::SeqCst);
        struct Admission<'a>(&'a Shared);
        impl Drop for Admission<'_> {
            fn drop(&mut self) {
                self.0.active.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let _admission = Admission(&self.shared);
        if self.shared.closing.load(Ordering::SeqCst) {
            return EmitStatus::Closed;
        }
        let bytes = match validate_fact(&fact)
            .and_then(|()| validate_links(&links))
            .and_then(|()| {
                crate::encode_bounded(
                    &(&fact, &links),
                    self.shared
                        .limits
                        .max_frame_bytes
                        .saturating_sub(ENVELOPE_HEADROOM_BYTES),
                )
            }) {
            Ok(bytes) => bytes,
            Err(_) => {
                self.shared.dropped.fetch_add(1, Ordering::Relaxed);
                return EmitStatus::Invalid;
            }
        };
        #[cfg(feature = "testing")]
        if let Some(records) = &self.recording {
            let Ok(mut records) = records.lock() else {
                return EmitStatus::Invalid;
            };
            records.push((fact, links));
            return EmitStatus::Queued;
        }
        let length = bytes.len();
        if self
            .shared
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(length)
                    .filter(|&n| n <= self.shared.limits.max_bytes)
            })
            .is_err()
        {
            self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            return EmitStatus::Dropped;
        }
        self.shared.queued.fetch_add(1, Ordering::AcqRel);
        match self.sender.try_send(Pending {
            bytes: bytes.into_boxed_slice().into_vec(),
        }) {
            Ok(()) => EmitStatus::Queued,
            Err(error) => {
                self.shared.bytes.fetch_sub(length, Ordering::AcqRel);
                self.shared.queued.fetch_sub(1, Ordering::AcqRel);
                match error {
                    TrySendError::Full(_) => {
                        self.shared.dropped.fetch_add(1, Ordering::Relaxed);
                        EmitStatus::Dropped
                    }
                    TrySendError::Disconnected(_) => {
                        self.close();
                        EmitStatus::Closed
                    }
                }
            }
        }
    }
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }
    pub fn close(&self) {
        self.shared.closing.store(true, Ordering::SeqCst);
    }
}
pub struct FactReceiver {
    shared: Arc<Shared>,
    receiver: Receiver<Pending>,
    reported: u64,
}
impl FactReceiver {
    pub(crate) fn recv_linked(&mut self) -> anyhow::Result<Option<(LineageFact, Links)>> {
        loop {
            // SeqCst orders producer registration/check and receiver closure/check
            // across both atomics; Acquire/Release alone permits store buffering.
            // Snapshot closure before inspecting the queue. Once no admitted producer is
            // active, an empty queue cannot race a previously accepted in-flight send.
            let closed = self.shared.closing.load(Ordering::SeqCst)
                && self.shared.active.load(Ordering::SeqCst) == 0;
            let total = self.shared.dropped.load(Ordering::Acquire);
            if total > self.reported {
                let dropped = total - self.reported;
                self.reported = total;
                return Ok(Some((
                    LineageFact::TapDropped { dropped },
                    Links::default(),
                )));
            }
            let pending = if closed {
                match self.receiver.try_recv() {
                    Ok(p) => p,
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(None),
                }
            } else {
                match self
                    .receiver
                    .recv_timeout(std::time::Duration::from_millis(10))
                {
                    Ok(p) => p,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => {
                        self.close();
                        continue;
                    }
                }
            };
            self.shared
                .bytes
                .fetch_sub(pending.bytes.len(), Ordering::AcqRel);
            self.shared.queued.fetch_sub(1, Ordering::AcqRel);
            // Decode after the channel pop, without holding any queue lock. Malformed
            // bytes are a failure, never an implicit successful end-of-stream.
            return Ok(Some(serde_json::from_slice(&pending.bytes)?));
        }
    }
    pub fn recv(&mut self) -> anyhow::Result<Option<LineageFact>> {
        Ok(self.recv_linked()?.map(|p| p.0))
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.shared.queued.load(Ordering::Acquire) == 0
    }
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }
    pub fn close(&self) {
        self.shared.closing.store(true, Ordering::SeqCst);
    }
}
impl Drop for FactReceiver {
    fn drop(&mut self) {
        self.close();
    }
}
pub fn fact_channel(max_facts: usize, max_bytes: usize) -> (Tap, FactReceiver) {
    channel_with(Limits {
        max_facts: max_facts.clamp(1, 1_000_000),
        max_bytes: max_bytes.clamp(1, 1024 * 1024 * 1024),
        ..Limits::default()
    })
}
pub(crate) fn channel_with(limits: Limits) -> (Tap, FactReceiver) {
    let (sender, receiver) = mpsc::sync_channel(limits.max_facts);
    let shared = Arc::new(Shared {
        closing: AtomicBool::new(false),
        active: AtomicUsize::new(0),
        dropped: AtomicU64::new(0),
        bytes: AtomicUsize::new(0),
        queued: AtomicUsize::new(0),
        limits,
    });
    (
        Tap {
            shared: shared.clone(),
            sender,
            #[cfg(feature = "testing")]
            recording: None,
        },
        FactReceiver {
            shared,
            receiver,
            reported: 0,
        },
    )
}

/// Explicit test support; absent from normal production builds.
#[cfg(feature = "testing")]
pub struct RecordingTap {
    tap: Tap,
    facts: RecordedFacts,
}
#[cfg(feature = "testing")]
impl Default for RecordingTap {
    fn default() -> Self {
        let (mut tap, receiver) = channel_with(Limits::default());
        drop(receiver);
        // The collector has its own explicit lifetime; no production receiver is used.
        tap.shared.closing.store(false, Ordering::Release);
        let facts = Arc::new(Mutex::new(Vec::new()));
        tap.recording = Some(facts.clone());
        Self { tap, facts }
    }
}
#[cfg(feature = "testing")]
impl RecordingTap {
    pub fn tap(&self) -> Tap {
        self.tap.clone()
    }
    pub fn facts(&self) -> Vec<LineageFact> {
        self.linked_facts().into_iter().map(|p| p.0).collect()
    }
    pub fn linked_facts(&self) -> Vec<(LineageFact, Links)> {
        self.facts.lock().expect("test collector poisoned").clone()
    }
}

pub(crate) fn validate_fact(fact: &LineageFact) -> anyhow::Result<()> {
    use LineageFact::*;
    let finite = |x: f64| -> anyhow::Result<()> {
        anyhow::ensure!(x.is_finite(), "nonfinite fact value");
        Ok(())
    };
    match fact {
        Drawn { parents, child, .. } => {
            anyhow::ensure!(parents.len() <= 16, "too many parent roles");
            for program in parents.iter().chain(child.iter()) {
                anyhow::ensure!(
                    program.canonical.len() <= crate::program::MAX_PROGRAM_BYTES,
                    "program text exceeds capture limit"
                );
                anyhow::ensure!(
                    program.sha256 == crate::digest(program.canonical.as_bytes()),
                    "program text identity mismatch"
                );
            }
        }
        SearchStarted {
            exploration_seconds,
            ..
        } => {
            finite(*exploration_seconds)?;
        }
        Measured {
            rmse,
            hold_r2,
            train_r2,
            objective_loss,
            seconds,
            ..
        } => {
            for v in [rmse, hold_r2, train_r2, objective_loss]
                .into_iter()
                .flatten()
            {
                finite(*v)?;
            }
            finite(*seconds)?;
        }
        GenerationClosed { survivors, .. } => {
            for s in survivors {
                finite(s.rmse)?;
                finite(s.score)?;
            }
        }
        Migrated { rmse, .. } => {
            finite(*rmse)?;
        }
        PredictionBound {
            seconds: Some(v), ..
        } => {
            finite(*v)?;
        }
        _ => {}
    }
    // No intermediate JSON tree: even rejected caller-owned payloads remain bounded.
    let safe = |n: usize| -> anyhow::Result<()> {
        anyhow::ensure!((n as u128) <= 9_007_199_254_740_991, "unsafe integer field");
        Ok(())
    };
    match fact {
        SearchStarted {
            cut,
            train_rows,
            hold_rows,
            features,
            population,
            elites,
            max_transforms,
            islands,
            migration_interval,
            exploration_evaluations,
            incumbents,
            ..
        } => {
            for n in [
                cut,
                train_rows,
                hold_rows,
                features,
                population,
                elites,
                max_transforms,
                islands,
                migration_interval,
                exploration_evaluations,
                incumbents,
            ] {
                safe(*n)?;
            }
        }
        Drawn {
            cut,
            generation,
            island,
            slot,
            ..
        } => {
            for n in [Some(*cut), Some(*generation), Some(*island), *slot]
                .into_iter()
                .flatten()
            {
                safe(n)?;
            }
        }
        Admitted {
            cut,
            generation,
            island,
            ..
        } => {
            for n in [cut, generation, island] {
                safe(*n)?;
            }
        }
        Started {
            cut,
            generation,
            island,
            ..
        }
        | Measured {
            cut,
            generation,
            island,
            ..
        } => {
            safe(*cut)?;
            safe(*generation)?;
            if let Some(n) = island {
                safe(*n)?;
            }
        }
        GenerationClosed {
            cut,
            generation,
            island,
            evaluations,
            barren,
            ..
        } => {
            for n in [cut, generation, island, evaluations, barren] {
                safe(*n)?;
            }
        }
        Migrated {
            cut,
            after_generation,
            from_island,
            to_island,
            ..
        } => {
            for n in [cut, after_generation, from_island, to_island] {
                safe(*n)?;
            }
        }
        ExplorationStopped {
            cut,
            generations,
            evaluations,
            ..
        } => {
            for n in [cut, generations, evaluations] {
                safe(*n)?;
            }
        }
        ComparisonStarted { cuts, .. } => {
            for n in cuts {
                safe(*n)?;
            }
        }
        PredictionBound { rows, .. } => {
            safe(*rows)?;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn validate_links(links: &Links) -> anyhow::Result<()> {
    for id in [
        &links.draw_id,
        &links.request_id,
        &links.attempt_id,
        &links.search_event_id,
        &links.observation_sha256,
    ]
    .into_iter()
    .flatten()
    {
        anyhow::ensure!(crate::is_sha256(id), "invalid native evidence identity");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_owned_frame_is_an_error_not_clean_closure() {
        let (tap, mut receiver) = fact_channel(2, 4096);
        tap.shared.bytes.store(1, Ordering::Release);
        tap.shared.queued.store(1, Ordering::Release);
        tap.sender.try_send(Pending { bytes: vec![b'x'] }).unwrap();
        tap.close();
        assert!(receiver.recv_linked().is_err());
    }
}
