//! Trainer-side hooks, ported from the former `jope.telemetry` Python package (owner rule
//! 2026-10-29: no Python telemetry outside DL training loops).
//!
//! Everything here is plain Rust and unit-tested without Python. `python.rs` wraps it for the
//! torch loops, which keep only one-line calls and the two things that must stay in Python: the
//! CUDA event record/query on the torch stream, and the DOPE parameter walk.

pub mod dope;
pub mod eval;
pub mod generation;
pub mod io;
pub mod json;
pub mod step;
pub mod trainers;
pub mod validation;

/// Linear-interpolated quantile of an ascending slice, the same formula the Python hooks used
/// (`h = (n - 1) p`, `lo = floor(h)`, `hi = min(lo + 1, n - 1)`).
pub fn quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    Some(sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo]))
}

/// Left-to-right sum starting from `0.0`, matching Python's `sum()` on floats.
pub fn sum(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, v| acc + v)
}

pub(crate) fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_python_formula() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[3.0], 0.95), Some(3.0));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), Some(2.5));
        let h = 3.0 * 0.95;
        assert_eq!(
            quantile(&[1.0, 2.0, 3.0, 4.0], 0.95),
            Some(3.0 + (h - 2.0) * 1.0)
        );
        assert_eq!(sum(&[0.1, 0.2, 0.3]), 0.1 + 0.2 + 0.3);
    }
}

/// The ported hooks end to end through a real recorder: step rows with the missing sentinel,
/// H2D rows, startup availability, and records from another thread through a `RecordHandle`.
#[cfg(test)]
mod pipeline_tests {
    use crate::core::RecorderBuilder;
    use crate::hooks::io::{BatchIo, H2dSampler, IoSample};
    use crate::hooks::step::{StepBatch, StepTimer};
    use crate::hooks::trainers;
    use crate::manifest::{RunMetadata, RunStatus};
    use crate::record::RecordSink;
    use serde_json::Value;

    #[test]
    fn flywheel_round_writes_step_h2d_and_round_records() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("w-step-r1-s7-test");
        let groups = trainers::flywheel_metric_groups(2);
        let mut builder = RecorderBuilder::new(
            &run,
            RunMetadata {
                metric_groups: groups.clone(),
                ..RunMetadata::default()
            },
        );
        for p in &trainers::W_PRODUCERS[..3] {
            builder.add_producer(*p).unwrap();
        }
        let (recorder, mut producers) = builder.start().unwrap();
        let cols = trainers::flywheel_step_columns();
        producers[0]
            .set_metric_layout(trainers::resolve_layout(&cols, &groups).unwrap())
            .unwrap();
        let mut batch = StepBatch::new(cols, 32).unwrap();
        let h2d_cols = trainers::flywheel_h2d_columns(false);
        producers[1]
            .set_metric_layout(trainers::resolve_layout(&h2d_cols, &groups).unwrap())
            .unwrap();
        let mut h2d_batch = StepBatch::new(h2d_cols, 32).unwrap();
        let mut io = BatchIo::new(&batch).unwrap();
        let mut timer = StepTimer::default();
        let mut sampler = H2dSampler::new(true, true, 2, 2);
        trainers::emit_flywheel_startup(&mut producers[0], None);
        for step in 1..=40u64 {
            timer.begin();
            let data_ms = timer.mark();
            batch
                .put(batch.col("step.span.data").unwrap(), data_ms)
                .unwrap();
            let sample = IoSample {
                prep_ms: 0.5,
                queue: None,
                data_wait_ms: data_ms,
                samples: 8,
                bytes: 4_000_000,
            };
            io.record(&mut batch, &sample).unwrap();
            if let Some(slot) = sampler.begin(step, 4_000_000, Some(1.0)) {
                sampler.recorded(slot);
                let row = sampler.complete(slot, 1.0).unwrap();
                for (c, v) in row.values.iter().enumerate() {
                    h2d_batch.put(c, *v).unwrap();
                }
                h2d_batch.commit(row.step, &mut producers[1]);
            }
            batch.put(0, 1.0 / step as f64).unwrap();
            let (wall, unattributed) = timer.end();
            batch.put(batch.col("step.wall").unwrap(), wall).unwrap();
            batch
                .put(batch.col("step.unattributed").unwrap(), unattributed)
                .unwrap();
            batch.commit(step, &mut producers[0]);
        }
        batch.flush(&mut producers[0]);
        h2d_batch.flush(&mut producers[1]);
        let sink = recorder.record_handle();
        std::thread::spawn(move || {
            assert!(sink.record_value(
                "jope.telemetry.eval-batch/v1",
                serde_json::json!({"kind": "strat300"})
            ));
        })
        .join()
        .unwrap();
        assert_eq!((batch.submitted_rows, h2d_batch.submitted_rows), (40, 20));
        drop(producers);
        let manifest = recorder.close().unwrap();
        assert!(matches!(manifest.status, RunStatus::Complete));
        assert_eq!(manifest.event_drops, 0);
        assert_eq!(manifest.records_written, 1);

        let text = std::fs::read_to_string(run.join("events.jsonl")).unwrap();
        let mut per_metric = std::collections::BTreeMap::<u64, (u64, u64)>::new();
        for line in text.lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            if v["kind"] == "event" {
                let e = &v["event"];
                let slot = per_metric
                    .entry(e["metric_id"].as_u64().unwrap())
                    .or_default();
                if e.get("availability_reason_id")
                    .is_some_and(|r| !r.is_null())
                {
                    slot.1 += 1;
                } else {
                    slot.0 += 1;
                }
            }
        }
        let id = |n: &str| u64::from(crate::train::metric_by_name(n).unwrap().id);
        // Every step wrote loss, IO counters and spans; queue columns were never put: absent.
        assert_eq!(per_metric[&id("train.loss")], (40, 0));
        assert_eq!(per_metric[&id("io.batch_bytes_per_s")], (40, 0));
        assert!(!per_metric.contains_key(&id("io.queue.depth")));
        assert!(!per_metric.contains_key(&id("train.loss.component")));
        // Every second step was sampled; end-to-end GB/s is bytes / (wait + copy) = 2 GB/s.
        assert_eq!(per_metric[&id("io.h2d_gb_s")], (20, 0));
        assert_eq!(per_metric[&id("io.batch_gb_s")], (20, 0));
        assert_eq!(per_metric[&id("throughput.tokens_per_s")], (0, 1));
        assert_eq!(per_metric[&id("io.local.read_mib_s")], (0, 1));
        assert!(text.contains("\"body\":{\"kind\":\"strat300\"}"));
    }
}
