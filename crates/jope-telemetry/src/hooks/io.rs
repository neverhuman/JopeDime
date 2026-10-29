//! IO instrumentation for the flywheel's resident-NumPy input path (rows I1-I4).
//!
//! * [`BatchIo`] writes the data-supply counters into the trainer's step row.
//! * [`H2dSampler`] is the slot state machine for sampled CUDA-event H2D copies. The events
//!   themselves are recorded and queried from Python, where the torch stream lives; Python only
//!   reports the elapsed milliseconds of a completed slot.
//! * [`ReadStats`] aggregates logical file sizes and wall time around startup reads.

use crate::hooks::step::{missing, StepBatch};
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::time::Instant;

pub const IO_COLUMNS: [&str; 9] = [
    "io.batch_prep",
    "io.queue.depth",
    "io.queue.wait",
    "io.queue.fallback",
    "io.data_wait",
    "io.batches_per_s",
    "io.samples_per_s",
    "io.batch.bytes",
    "io.batch_bytes_per_s",
];
pub const H2D_COLUMNS: [&str; 3] = ["io.h2d.bytes", "io.h2d", "io.h2d_gb_s"];
/// End-to-end GB/s: batch bytes over data wait plus the completed H2D copy.
pub const END_TO_END_COLUMN: &str = "io.batch_gb_s";
/// The batch-dict arrays the flywheel trainer copies to the device each step.
pub const ARRAY_KEYS: [&str; 11] = [
    "ctx_b",
    "prog_b",
    "proj_b",
    "prior_b",
    "scale_b",
    "sos_b",
    "target_f_b",
    "gate_b",
    "rev_idx_b",
    "rev_f_b",
    "rev_gate_b",
];
pub const H2D_SAMPLE_STRIDE: u64 = 64;
pub const H2D_SLOTS: usize = 4;

/// Evolution-queue counters for one popped batch (`AsyncEvoBatchQueue.last_*`).
#[derive(Clone, Copy, Debug, Default)]
pub struct QueueSample {
    pub depth: f64,
    pub wait_ms: f64,
    pub fallback: f64,
}

/// One step's data-supply observation.
#[derive(Clone, Copy, Debug)]
pub struct IoSample {
    pub prep_ms: f64,
    pub queue: Option<QueueSample>,
    pub data_wait_ms: f64,
    pub samples: u64,
    pub bytes: u64,
}

/// Column indices of [`IO_COLUMNS`] in a step layout, plus the inter-batch clock.
pub struct BatchIo {
    cols: [usize; 9],
    last: Instant,
}

impl BatchIo {
    pub fn new(batch: &StepBatch) -> Result<Self> {
        let mut cols = [0; 9];
        for (slot, name) in cols.iter_mut().zip(IO_COLUMNS) {
            *slot = batch
                .col(name)
                .ok_or_else(|| anyhow!("step layout has no {name} column"))?;
        }
        Ok(Self {
            cols,
            last: Instant::now(),
        })
    }

    pub fn record(&mut self, batch: &mut StepBatch, s: &IoSample) -> Result<()> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64().max(1e-9);
        self.last = now;
        let c = &self.cols;
        batch.put(c[0], s.prep_ms)?;
        if let Some(q) = s.queue {
            batch.put(c[1], q.depth)?;
            batch.put(c[2], q.wait_ms)?;
            batch.put(c[3], q.fallback)?;
        }
        batch.put(c[4], s.data_wait_ms)?;
        batch.put(c[5], 1.0 / elapsed)?;
        batch.put(c[6], s.samples as f64 / elapsed)?;
        batch.put(c[7], s.bytes as f64)?;
        batch.put(c[8], s.bytes as f64 / elapsed)?;
        Ok(())
    }
}

/// A completed H2D sample, ready for the H2D source's row: `(step, values)` where values are
/// `io.h2d.bytes`, `io.h2d` (ms), `io.h2d_gb_s` and, for the main batch, `io.batch_gb_s`.
pub struct H2dRow {
    pub step: u64,
    pub values: [f64; 4],
}

/// Slot state machine for sampled, unsynchronized CUDA-event H2D timing.
pub struct H2dSampler {
    stride: u64,
    pending: Vec<bool>,
    steps: Vec<u64>,
    bytes: Vec<u64>,
    data_wait_ms: Vec<Option<f64>>,
    next_slot: usize,
    end_to_end: bool,
    pub enabled: bool,
    pub dropped_samples: u64,
}

impl H2dSampler {
    pub fn new(enabled: bool, end_to_end: bool, stride: u64, slots: usize) -> Self {
        let slots = slots.max(1);
        Self {
            stride: stride.max(1),
            pending: vec![false; slots],
            steps: vec![0; slots],
            bytes: vec![0; slots],
            data_wait_ms: vec![None; slots],
            next_slot: 0,
            end_to_end,
            enabled,
            dropped_samples: 0,
        }
    }

    pub fn will_sample(&self, step: u64) -> bool {
        self.enabled && step.is_multiple_of(self.stride)
    }

    /// Reserve a free slot for `step`, or `None` (not sampled, disabled, or every slot pending,
    /// which counts as a dropped sample). Python records the start event on the returned slot.
    pub fn begin(&mut self, step: u64, bytes: u64, data_wait_ms: Option<f64>) -> Option<usize> {
        if !self.will_sample(step) {
            return None;
        }
        let n = self.pending.len();
        for offset in 0..n {
            let slot = (self.next_slot + offset) % n;
            if !self.pending[slot] {
                self.next_slot = (slot + 1) % n;
                self.steps[slot] = step;
                self.bytes[slot] = bytes;
                self.data_wait_ms[slot] = data_wait_ms;
                return Some(slot);
            }
        }
        self.dropped_samples += 1;
        None
    }

    /// Python recorded the end event of `slot`; it is now waiting for completion.
    pub fn recorded(&mut self, slot: usize) {
        if let Some(p) = self.pending.get_mut(slot) {
            *p = true;
        }
    }

    pub fn pending_slots(&self) -> Vec<usize> {
        (0..self.pending.len())
            .filter(|&s| self.pending[s])
            .collect()
    }

    /// The end event of `slot` completed after `elapsed_ms`.
    pub fn complete(&mut self, slot: usize, elapsed_ms: f64) -> Option<H2dRow> {
        if !self.pending.get(slot).copied().unwrap_or(false) {
            return None;
        }
        self.pending[slot] = false;
        let size = self.bytes[slot] as f64;
        let gb_s = if elapsed_ms > 0.0 {
            size / (elapsed_ms * 1e6)
        } else {
            missing()
        };
        let end_to_end = match (self.end_to_end, self.data_wait_ms[slot].take()) {
            (true, Some(wait)) => {
                let total = wait + elapsed_ms;
                if total > 0.0 {
                    size / (total * 1e6)
                } else {
                    missing()
                }
            }
            _ => missing(),
        };
        Some(H2dRow {
            step: self.steps[slot],
            values: [size, elapsed_ms, gb_s, end_to_end],
        })
    }

    /// A CUDA event call on `slot` raised: drop that sample and stop sampling.
    pub fn fail(&mut self, slot: Option<usize>) {
        self.dropped_samples += 1;
        if let Some(p) = slot.and_then(|s| self.pending.get_mut(s)) {
            *p = false;
        }
        self.enabled = false;
    }

    /// The shutdown synchronization failed: every pending sample is lost.
    pub fn abandon_pending(&mut self) {
        self.dropped_samples += self.pending.iter().filter(|p| **p).count() as u64;
        self.pending.fill(false);
        self.enabled = false;
    }
}

/// Startup read totals, as recorded in the flywheel step manifest (`io_startup_reads`).
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct ReadSnapshot {
    pub enabled: bool,
    pub logical_bytes: u64,
    pub read_ms: f64,
    pub read_operation_ms: f64,
    pub reads: u64,
}

/// Logical bytes and wall time of local reads. `read_ms` spans the first read's start to the
/// last read's end, so overlapping reads aren't double-counted; `read_operation_ms` sums them.
#[derive(Default)]
pub struct ReadStats {
    bytes: u64,
    read_operation_ms: f64,
    reads: u64,
    first: Option<Instant>,
    last: Option<Instant>,
}

impl ReadStats {
    pub fn record(&mut self, size: u64, started: Instant, finished: Instant) {
        self.bytes += size;
        self.read_operation_ms += finished.duration_since(started).as_nanos() as f64 / 1e6;
        self.first = Some(self.first.map_or(started, |f| f.min(started)));
        self.last = Some(self.last.map_or(finished, |l| l.max(finished)));
        self.reads += 1;
    }

    pub fn snapshot(&self) -> ReadSnapshot {
        let read_ms = match (self.first, self.last) {
            (Some(f), Some(l)) => l.duration_since(f).as_nanos() as f64 / 1e6,
            _ => 0.0,
        };
        ReadSnapshot {
            enabled: true,
            logical_bytes: self.bytes,
            read_ms,
            read_operation_ms: self.read_operation_ms,
            reads: self.reads,
        }
    }

    pub fn take_snapshot(&mut self) -> ReadSnapshot {
        let snap = self.snapshot();
        *self = Self::default();
        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn h2d_slots_stride_and_drops() {
        let mut s = H2dSampler::new(true, true, 64, 2);
        assert_eq!(s.begin(1, 10, None), None);
        assert_eq!(s.begin(64, 1_000_000, Some(1.0)), Some(0));
        s.recorded(0);
        assert_eq!(s.begin(128, 2_000_000, None), Some(1));
        s.recorded(1);
        assert_eq!(s.begin(192, 1, None), None);
        assert_eq!(s.dropped_samples, 1);
        assert_eq!(s.pending_slots(), vec![0, 1]);
        let row = s.complete(0, 1.0).unwrap();
        assert_eq!(row.step, 64);
        assert_eq!(row.values[0], 1_000_000.0);
        assert_eq!(row.values[2], 1.0);
        assert_eq!(row.values[3], 0.5);
        let row = s.complete(1, 0.0).unwrap();
        assert_eq!(row.values[2].to_bits(), missing().to_bits());
        assert_eq!(row.values[3].to_bits(), missing().to_bits());
        assert!(s.complete(1, 1.0).is_none());
        s.begin(256, 1, None);
        s.recorded(0);
        s.abandon_pending();
        assert_eq!(s.dropped_samples, 2);
        assert!(!s.enabled && s.begin(320, 1, None).is_none());
    }

    #[test]
    fn evolution_sampler_never_writes_end_to_end() {
        let mut s = H2dSampler::new(true, false, 1, 1);
        let slot = s.begin(5, 100, Some(3.0)).unwrap();
        s.recorded(slot);
        let row = s.complete(slot, 2.0).unwrap();
        assert_eq!(row.values[3].to_bits(), missing().to_bits());
    }

    #[test]
    fn read_span_is_union_not_sum() {
        let t0 = Instant::now();
        let mut r = ReadStats::default();
        r.record(10, t0, t0 + Duration::from_millis(4));
        r.record(
            5,
            t0 + Duration::from_millis(1),
            t0 + Duration::from_millis(3),
        );
        let s = r.take_snapshot();
        assert_eq!((s.logical_bytes, s.reads), (15, 2));
        assert!((s.read_ms - 4.0).abs() < 1e-9);
        assert!((s.read_operation_ms - 6.0).abs() < 1e-9);
        assert_eq!(r.snapshot().reads, 0);
    }
}
