//! Rust fitness evaluator timings (TELEMETRY_SPEC rows E5-E7).
//!
//! The flywheel evaluates candidates by running the Rust worker as a subprocess from a thread
//! pool (`rust_evaluator.py`: `evaluate_gp_batch_rust*` -> `_eval_single_dsl_worker`). Each
//! worker first waits for host CPU headroom (the 85% cap), runs `--fit-s1`, and optionally waits
//! again and runs the outer `--fit`.
//!
//! [`EvalBatch`] splits each candidate's latency into executor queue wait, CPU-guard wait,
//! fit-s1 and outer fit, keeps the subprocess return code (timeouts are `rc == -1`), and reports
//! throughput and child CPU per candidate from `getrusage(RUSAGE_CHILDREN)`. Workers write
//! per-slot atomics, each slot written only by its own task, so the evaluation path takes no
//! lock. Use one instance per concurrently running batch stream: a `batch_begin` while a batch
//! is still active shares the slots and the process-wide child CPU counter, so that record is
//! marked `overlapping_batches`.

use crate::hooks::{quantile, sorted, sum};
use serde::Serialize;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const RECORD_KIND: &str = "jope.telemetry.eval-batch/v1";
pub const MAX_SLOTS: usize = 1024;
/// Bucket k counts latencies in [2^k, 2^(k+1)) ms, k = 0..15; bucket 0 also takes anything
/// under 1 ms, and the last bucket everything from 32.8 s up.
pub const HIST_BUCKETS: usize = 16;

pub fn bucket(ms: f64) -> usize {
    if ms < 2.0 {
        return 0;
    }
    ((63 - (ms as u64).leading_zeros()) as usize).min(HIST_BUCKETS - 1)
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Q4 {
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

impl Q4 {
    fn of(v: &[f64]) -> Self {
        Self {
            p50: quantile(v, 0.5),
            p95: quantile(v, 0.95),
            p99: quantile(v, 0.99),
            max: v.last().copied(),
        }
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct EvalBatchRecord {
    pub kind: String,
    pub n: usize,
    pub completed: usize,
    pub workers_requested: i64,
    pub workers_effective: i64,
    pub wall_s: f64,
    pub throughput_candidates_per_s: Option<f64>,
    pub latency_ms: Q4,
    pub latency_hist_log2_ms: [u64; HIST_BUCKETS],
    pub executor_queue_wait_ms: Q4,
    pub cpu_guard_wait_ms: Q4,
    pub cpu_guard_wait_share: Option<f64>,
    pub fit_s1_ms: Q4,
    pub outer_fit_ms: Q4,
    pub timeouts: u64,
    pub nonzero_rc: u64,
    pub timeout_rate: Option<f64>,
    pub cpu_s_per_candidate: Option<f64>,
    pub worker_efficiency: Option<f64>,
    pub overlapping_batches: bool,
    pub slots_dropped_total: u64,
}

struct Meta {
    kind: String,
    n: usize,
    workers: (i64, i64),
    t0: Instant,
    ru0: f64,
    active: u32,
    overlapped: bool,
}

pub struct EvalBatch {
    epoch: Instant,
    submit: Box<[AtomicI64]>,
    start: Box<[AtomicI64]>,
    guard: Box<[AtomicI64]>,
    fit: Box<[AtomicI64]>,
    outer: Box<[AtomicI64]>,
    end: Box<[AtomicI64]>,
    rc: Box<[AtomicI64]>,
    meta: Mutex<Meta>,
    n: AtomicI64,
    pub batches: AtomicI64,
    pub slots_dropped: AtomicI64,
}

fn slots() -> Box<[AtomicI64]> {
    (0..MAX_SLOTS).map(|_| AtomicI64::new(0)).collect()
}

/// User plus system CPU seconds of terminated, waited-for children of this process, from
/// `/proc/self/stat` (`cutime + cstime`, in that interface's fixed 100 Hz USER_HZ). The PyO3
/// layer passes `resource.getrusage(RUSAGE_CHILDREN)` instead, which has µs resolution.
pub fn children_cpu_s() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    // Fields after the parenthesized command name start at field 3; cutime is field 16.
    let rest = stat.rsplit_once(')').map_or("", |(_, r)| r);
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| {
        fields
            .get(i)
            .and_then(|f| f.parse::<u64>().ok())
            .unwrap_or(0)
    };
    (ticks(13) + ticks(14)) as f64 / 100.0
}

impl Default for EvalBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalBatch {
    pub fn new() -> Self {
        let epoch = Instant::now();
        Self {
            epoch,
            submit: slots(),
            start: slots(),
            guard: slots(),
            fit: slots(),
            outer: slots(),
            end: slots(),
            rc: slots(),
            meta: Mutex::new(Meta {
                kind: String::new(),
                n: 0,
                workers: (0, 0),
                t0: epoch,
                ru0: 0.0,
                active: 0,
                overlapped: false,
            }),
            n: AtomicI64::new(0),
            batches: AtomicI64::new(0),
            slots_dropped: AtomicI64::new(0),
        }
    }

    /// Nanoseconds since construction, never zero (zero marks an unset slot).
    fn now(&self) -> i64 {
        self.epoch.elapsed().as_nanos().min(i64::MAX as u128 - 1) as i64 + 1
    }

    fn slot(&self, i: usize) -> bool {
        (i as i64) < self.n.load(Ordering::Acquire)
    }

    /// `children_cpu_s`: child CPU seconds now ([`children_cpu_s`] or `getrusage`).
    pub fn batch_begin(
        &self,
        kind: &str,
        n: usize,
        workers_requested: i64,
        workers_effective: i64,
        children_cpu_s: f64,
    ) {
        let mut m = self.meta.lock().unwrap_or_else(|e| e.into_inner());
        if m.active > 0 {
            m.overlapped = true;
        }
        m.active += 1;
        m.kind = kind.to_owned();
        m.n = n.min(MAX_SLOTS);
        self.slots_dropped
            .fetch_add(n.saturating_sub(MAX_SLOTS) as i64, Ordering::Relaxed);
        m.workers = (workers_requested, workers_effective);
        let now = self.now();
        for i in 0..m.n {
            self.submit[i].store(now, Ordering::Relaxed);
            for a in [
                &self.start,
                &self.guard,
                &self.fit,
                &self.outer,
                &self.end,
                &self.rc,
            ] {
                a[i].store(0, Ordering::Relaxed);
            }
        }
        m.t0 = Instant::now();
        m.ru0 = children_cpu_s;
        self.n.store(m.n as i64, Ordering::Release);
    }

    // Worker-side calls; `i` is the task's index in the batch.
    pub fn worker_start(&self, i: usize) {
        if self.slot(i) {
            self.start[i].store(self.now(), Ordering::Relaxed);
        }
    }

    pub fn guard_wait(&self, i: usize, ns: i64) {
        if self.slot(i) {
            self.guard[i].fetch_add(ns, Ordering::Relaxed);
        }
    }

    pub fn fit_done(&self, i: usize, ns: i64, rc: i64) {
        if self.slot(i) {
            self.fit[i].store(ns, Ordering::Relaxed);
            self.rc[i].store(rc, Ordering::Relaxed);
        }
    }

    pub fn outer_done(&self, i: usize, ns: i64) {
        if self.slot(i) {
            self.outer[i].store(ns, Ordering::Relaxed);
        }
    }

    pub fn worker_end(&self, i: usize) {
        if self.slot(i) {
            self.end[i].store(self.now(), Ordering::Release);
        }
    }

    pub fn batch_end(&self, children_cpu_s: f64) -> EvalBatchRecord {
        let mut m = self.meta.lock().unwrap_or_else(|e| e.into_inner());
        let wall_s = m.t0.elapsed().as_nanos() as f64 / 1e9;
        let cpu_s = children_cpu_s - m.ru0;
        let n = m.n;
        let ld = |a: &[AtomicI64], i: usize| a[i].load(Ordering::Acquire);
        let done: Vec<usize> = (0..n).filter(|&i| ld(&self.end, i) != 0).collect();
        let ms = |ns: i64| ns as f64 / 1e6;
        let total_ms = sorted(
            done.iter()
                .map(|&i| ms(ld(&self.end, i) - ld(&self.submit, i)))
                .collect(),
        );
        let queue_ms = sorted(
            done.iter()
                .filter(|&&i| ld(&self.start, i) != 0)
                .map(|&i| ms(ld(&self.start, i) - ld(&self.submit, i)))
                .collect(),
        );
        let guard_ms = sorted(done.iter().map(|&i| ms(ld(&self.guard, i))).collect());
        let fit_ms = sorted(
            done.iter()
                .filter(|&&i| ld(&self.fit, i) != 0)
                .map(|&i| ms(ld(&self.fit, i)))
                .collect(),
        );
        let outer_ms = sorted(
            done.iter()
                .filter(|&&i| ld(&self.outer, i) != 0)
                .map(|&i| ms(ld(&self.outer, i)))
                .collect(),
        );
        let mut hist = [0u64; HIST_BUCKETS];
        for &v in &total_ms {
            hist[bucket(v)] += 1;
        }
        let rcs: Vec<i64> = done.iter().map(|&i| ld(&self.rc, i)).collect();
        let timeouts = rcs.iter().filter(|&&rc| rc == -1).count() as u64;
        let nonzero = rcs.iter().filter(|&&rc| rc != 0 && rc != -1).count() as u64;
        let completed = done.len();
        let workers = if m.workers.1 != 0 { m.workers.1 } else { 1 };
        let total_sum = sum(&total_ms);
        let rec = EvalBatchRecord {
            kind: m.kind.clone(),
            n,
            completed,
            workers_requested: m.workers.0,
            workers_effective: m.workers.1,
            wall_s,
            throughput_candidates_per_s: (wall_s > 0.0).then(|| completed as f64 / wall_s),
            latency_ms: Q4::of(&total_ms),
            latency_hist_log2_ms: hist,
            executor_queue_wait_ms: Q4::of(&queue_ms),
            cpu_guard_wait_ms: Q4::of(&guard_ms),
            cpu_guard_wait_share: (!total_ms.is_empty() && total_sum > 0.0)
                .then(|| sum(&guard_ms) / total_sum),
            fit_s1_ms: Q4::of(&fit_ms),
            outer_fit_ms: Q4::of(&outer_ms),
            timeouts,
            nonzero_rc: nonzero,
            timeout_rate: (completed > 0).then(|| timeouts as f64 / completed as f64),
            cpu_s_per_candidate: (completed > 0).then(|| cpu_s / completed as f64),
            worker_efficiency: (wall_s > 0.0).then(|| cpu_s / (wall_s * workers as f64)),
            overlapping_batches: m.overlapped,
            slots_dropped_total: self.slots_dropped.load(Ordering::Relaxed) as u64,
        };
        m.active = m.active.saturating_sub(1);
        if m.active == 0 {
            m.overlapped = false;
        }
        self.batches.fetch_add(1, Ordering::Relaxed);
        rec
    }

    /// The return code of task `i` in the last batch, for the generation recorder.
    pub fn rc_of(&self, i: usize) -> Option<i64> {
        (self.slot(i) && self.end[i].load(Ordering::Acquire) != 0)
            .then(|| self.rc[i].load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_log2_ms() {
        assert_eq!(bucket(0.2), 0);
        assert_eq!(bucket(1.9), 0);
        assert_eq!(bucket(2.0), 1);
        assert_eq!(bucket(3.9), 1);
        assert_eq!(bucket(4.0), 2);
        assert_eq!(bucket(1e9), HIST_BUCKETS - 1);
    }

    #[test]
    fn splits_latency_and_counts_rc() {
        let t = EvalBatch::new();
        t.batch_begin("strat300", 3, 4, 2, 1.0);
        for i in 0..3 {
            t.worker_start(i);
            t.guard_wait(i, 1_000_000);
            t.guard_wait(i, 1_000_000);
            t.fit_done(i, 5_000_000, [0, -1, 3][i]);
            t.worker_end(i);
        }
        t.worker_start(7); // outside the batch: ignored
        let r = t.batch_end(1.3);
        assert_eq!((r.n, r.completed, r.timeouts, r.nonzero_rc), (3, 3, 1, 1));
        assert_eq!(r.cpu_guard_wait_ms.p50, Some(2.0));
        assert_eq!(r.fit_s1_ms.max, Some(5.0));
        assert_eq!(r.outer_fit_ms.p50, None);
        assert_eq!(r.latency_hist_log2_ms.iter().sum::<u64>(), 3);
        assert_eq!(r.timeout_rate, Some(1.0 / 3.0));
        assert!((r.cpu_s_per_candidate.unwrap() - 0.1).abs() < 1e-12);
        assert!(!r.overlapping_batches);
        assert_eq!(t.rc_of(1), Some(-1));
        assert_eq!(t.rc_of(5), None);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["kind"], "strat300");
        assert_eq!(v["latency_hist_log2_ms"].as_array().unwrap().len(), 16);
    }

    #[test]
    fn overlap_and_slot_cap() {
        let t = EvalBatch::new();
        t.batch_begin("full", MAX_SLOTS + 5, 1, 1, 0.0);
        t.batch_begin("strat300", 1, 1, 1, 0.0);
        let r = t.batch_end(0.0);
        assert!(r.overlapping_batches);
        assert_eq!(r.slots_dropped_total, 5);
        let r = t.batch_end(0.0);
        assert!(r.overlapping_batches);
        t.batch_begin("full", 1, 1, 1, children_cpu_s());
        assert!(!t.batch_end(children_cpu_s()).overlapping_batches);
    }
}
