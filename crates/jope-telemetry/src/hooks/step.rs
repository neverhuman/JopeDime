//! Per-step host scalars (TELEMETRY_SPEC rows T1-T4, T8, T11-T12) and host wall-clock spans.
//!
//! The frozen contract: trainers already materialize some per-step values on the host (W's
//! `float(loss.detach().cpu().item())`). Telemetry reuses exactly those values; it adds no
//! device-to-host copy and no synchronization.
//!
//! [`StepBatch`] keeps one run-scoped, preallocated `rows x columns` block of `f64` plus `rows`
//! step numbers. `put` stores one value in the current row, `commit` closes the row, and a full
//! block goes to the producer in one `submit_values` call. A value never put in a row stays the
//! missing sentinel, so a missing value is never reported as zero. [`StepTimer`] gives host
//! wall-clock spans; they are never GPU execution time.

use crate::core::{ProducerHandle, MISSING_VALUE_BITS};
use anyhow::{bail, Result};
use std::time::Instant;

/// Frozen memory contract: at most 4,096 float64 columns x 32 rows (1 MiB) per block.
pub const MAX_COLUMNS: usize = 4096;
pub const MAX_ROWS: usize = 32;

#[inline]
pub fn missing() -> f64 {
    f64::from_bits(MISSING_VALUE_BITS)
}

pub struct StepBatch {
    columns: Vec<String>,
    rows: usize,
    values: Vec<f64>,
    steps: Vec<u64>,
    row: usize,
    pub submitted_rows: u64,
    pub submit_errors: u64,
    pub dropped_events: u64,
}

impl StepBatch {
    pub fn new(columns: Vec<String>, rows: usize) -> Result<Self> {
        if columns.is_empty() {
            bail!("StepBatch needs at least one column");
        }
        if columns.len() > MAX_COLUMNS {
            bail!(
                "StepBatch has {} columns; the frozen maximum is {MAX_COLUMNS}",
                columns.len()
            );
        }
        let mut seen = std::collections::HashSet::new();
        if !columns.iter().all(|c| seen.insert(c.as_str())) {
            bail!("StepBatch columns must be unique");
        }
        if !(1..=MAX_ROWS).contains(&rows) {
            bail!("rows must be in 1..={MAX_ROWS}, got {rows}");
        }
        Ok(Self {
            values: vec![missing(); rows * columns.len()],
            steps: vec![0; rows],
            columns,
            rows,
            row: 0,
            submitted_rows: 0,
            submit_errors: 0,
            dropped_events: 0,
        })
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    pub fn col(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Store one already-materialized host value in the current row.
    #[inline]
    pub fn put(&mut self, col: usize, value: f64) -> Result<()> {
        if col >= self.columns.len() {
            bail!(
                "column {col} is outside the {}-column layout",
                self.columns.len()
            );
        }
        self.values[self.row * self.columns.len() + col] = value;
        Ok(())
    }

    /// The value put in the current row, if any (tests and the unattributed-span check).
    pub fn current(&self, col: usize) -> Option<f64> {
        let v = *self.values.get(self.row * self.columns.len() + col)?;
        (v.to_bits() != MISSING_VALUE_BITS).then_some(v)
    }

    /// Close the current row as optimizer step `step`; submit when the block is full.
    #[inline]
    pub fn commit(&mut self, step: u64, producer: &mut ProducerHandle) {
        self.steps[self.row] = step;
        self.row += 1;
        if self.row == self.rows {
            self.submit(producer);
        }
    }

    /// Submit a partial block (epoch end, shutdown). Never called per step.
    pub fn flush(&mut self, producer: &mut ProducerHandle) {
        if self.row > 0 {
            self.submit(producer);
        }
    }

    pub fn pending_rows(&self) -> usize {
        self.row
    }

    fn submit(&mut self, producer: &mut ProducerHandle) {
        let rows = self.row;
        let width = self.columns.len();
        match producer.submit_values(&self.steps[..rows], rows, &self.values[..rows * width]) {
            Ok(dropped) => {
                self.submitted_rows += rows as u64;
                self.dropped_events += dropped;
            }
            // The producer counts its own drops; a failing sink must never stop training.
            Err(_) => self.submit_errors += 1,
        }
        self.values[..rows * width].fill(missing());
        self.row = 0;
    }
}

/// Host wall-clock spans for one step, in milliseconds. `begin` at the start of a step, `mark`
/// at the end of each stage (the span runs from the previous mark), `end` at the end of the
/// step. Spans are exclusive, so `unattributed = wall - sum(spans)`.
pub struct StepTimer {
    t0: Instant,
    last: Instant,
    spans_ns: u128,
}

impl Default for StepTimer {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            t0: now,
            last: now,
            spans_ns: 0,
        }
    }
}

impl StepTimer {
    pub fn begin(&mut self) {
        let now = Instant::now();
        self.t0 = now;
        self.last = now;
        self.spans_ns = 0;
    }

    /// The span since the previous mark, in ms.
    pub fn mark(&mut self) -> f64 {
        let now = Instant::now();
        let d = now.duration_since(self.last).as_nanos();
        self.last = now;
        self.spans_ns += d;
        d as f64 / 1e6
    }

    /// `(wall_ms, unattributed_ms)` since `begin`.
    pub fn end(&self) -> (f64, f64) {
        let wall = self.t0.elapsed().as_nanos();
        (
            wall as f64 / 1e6,
            (wall as i128 - self.spans_ns as i128) as f64 / 1e6,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_layouts() {
        assert!(StepBatch::new(vec![], 32).is_err());
        assert!(StepBatch::new(vec!["a".into(), "a".into()], 32).is_err());
        assert!(StepBatch::new(vec!["a".into()], 0).is_err());
        assert!(StepBatch::new(vec!["a".into()], 33).is_err());
        assert!(StepBatch::new((0..4097).map(|i| i.to_string()).collect(), 1).is_err());
        let mut b = StepBatch::new(vec!["a".into(), "b".into()], 2).unwrap();
        assert!(b.put(2, 1.0).is_err());
        b.put(1, 2.5).unwrap();
        assert_eq!(b.current(1), Some(2.5));
        assert_eq!(b.current(0), None);
        assert_eq!(b.col("b"), Some(1));
    }

    #[test]
    fn timer_spans_sum_to_wall() {
        let mut t = StepTimer::default();
        t.begin();
        let a = t.mark();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = t.mark();
        let (wall, unattributed) = t.end();
        assert!(b >= 2.0 && a >= 0.0);
        assert!((wall - (a + b + unattributed)).abs() < 1e-6);
        assert!(unattributed >= 0.0);
    }
}
