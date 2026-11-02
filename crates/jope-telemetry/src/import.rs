//! `jtel import`: backfill telemetry records from outputs the trainers and the flywheel already
//! write, so thresholds (alarm A2) can be calibrated on history without running anything.
//!
//! Every imported record is tagged `source = "import"` with the source file's SHA-256 and base
//! name. DSL text and local paths are never copied into a record.
//!
//! Supported inputs:
//! - `parallel.neural_training.v1` state JSONL (jope-engine trainer)
//! - `train_metrics.jsonl` and other per-epoch JSONL (generic numeric flattening)
//! - `jope.flywheel-selfplay-results/v2` report JSON
//! - `gp_distilled/v1` round JSON (`round_metrics`, `retention_summary`)
//! - `jope.hwm/v1` files, one per context
//! - `jope.score/v1` retained-score JSONL

mod hwm;
mod queue;
mod retention;
mod scalar;

#[cfg(test)]
mod tests;

pub use hwm::{
    hwm_consistency, hwm_file, HwmBeat, HwmConsistency, HWM_CONSISTENCY_SCHEMA,
    HWM_IDENTITY_FIELDS, HWM_SCHEMA,
};
pub use queue::{retained_scores, selfplay_report, QueueRecord, SelfplayReport, QUEUE_SCHEMA};
pub use retention::{distilled_round, parse_utc, Retention, RETENTION_SCHEMA};
pub use scalar::{
    engine_state_jsonl, generic_epoch_jsonl, ImportedScalar, SourceRef, MAX_INPUT_BYTES,
    SCALAR_SCHEMA,
};

use scalar::{num, read_bounded};
