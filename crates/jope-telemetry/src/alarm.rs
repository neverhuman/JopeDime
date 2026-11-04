//! TRAIN+GP alarms A1–A10 (TELEMETRY_SPEC, "Alarms (TRAIN+GP)").
//!
//! The engine is fed already-collected values: step scalars from the drain, generation
//! summaries from the GP loop, and guard results from the epoch boundary. It never touches the
//! training thread. The fail-closed checks (A1, A8, A9) are decided where they are observed: the
//! caller raises when `observe_*` returns a `FailClosed` alarm. The engine records them too.
//!
//! Every default threshold below is provisional until calibrated on the R1–R3 history
//! (`jtel import`).

use crate::gp::GenSummary;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlarmId {
    A1DopeFreeze,
    A2GpCollapse,
    A3Divergence,
    A4SelectionCollapse,
    A5EloRegression,
    A6ScheduleDrift,
    A7EvaluatorHealth,
    A8SealedTouch,
    A9RustOnlyTarget,
    A10TrainerHeartbeat,
}

impl AlarmId {
    pub fn code(self) -> &'static str {
        match self {
            Self::A1DopeFreeze => "A1",
            Self::A2GpCollapse => "A2",
            Self::A3Divergence => "A3",
            Self::A4SelectionCollapse => "A4",
            Self::A5EloRegression => "A5",
            Self::A6ScheduleDrift => "A6",
            Self::A7EvaluatorHealth => "A7",
            Self::A8SealedTouch => "A8",
            Self::A9RustOnlyTarget => "A9",
            Self::A10TrainerHeartbeat => "A10",
        }
    }
    pub fn severity(self) -> Severity {
        match self {
            Self::A1DopeFreeze | Self::A8SealedTouch | Self::A9RustOnlyTarget => {
                Severity::FailClosed
            }
            _ => Severity::Alarm,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Training or emission must stop: the caller raises or returns an error.
    FailClosed,
    /// Loud log line, alarm event and comms note; training continues.
    Alarm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Alarm {
    pub schema: String,
    pub id: AlarmId,
    pub code: String,
    pub severity: Severity,
    pub run_id: String,
    /// The step, generation or round the alarm refers to.
    pub at: u64,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Thresholds {
    pub collapse_consecutive_gens: u32,
    pub min_unique_genotype_frac: f64,
    pub min_behavior_unique_frac: f64,
    pub min_op_entropy_norm: f64,
    pub max_dedup_reject_frac: f64,
    pub min_passer_fitness_iqr: f64,
    pub min_novel_passer_frac: f64,
    pub grad_spike_factor: f64,
    pub grad_spike_window: usize,
    pub grad_spike_consecutive: u32,
    pub val_rise_epochs: u32,
    pub max_selection_top1_share: f64,
    pub min_selection_unique_frac: f64,
    pub min_prediction_spread: f64,
    pub elo_drop_rounds: u32,
    pub lr_rel_tol: f64,
    pub max_eval_timeout_rate: f64,
    pub max_eval_fail_rate: f64,
    pub heartbeat_median_factor: f64,
    pub heartbeat_min_secs: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            collapse_consecutive_gens: 3,
            min_unique_genotype_frac: 0.5,
            min_behavior_unique_frac: 0.3,
            min_op_entropy_norm: 0.5,
            max_dedup_reject_frac: 0.5,
            min_passer_fitness_iqr: 1e-3,
            min_novel_passer_frac: 0.1,
            grad_spike_factor: 10.0,
            grad_spike_window: 100,
            grad_spike_consecutive: 5,
            val_rise_epochs: 3,
            max_selection_top1_share: 0.5,
            min_selection_unique_frac: 0.1,
            min_prediction_spread: 1e-6,
            elo_drop_rounds: 2,
            lr_rel_tol: 1e-6,
            max_eval_timeout_rate: 0.05,
            max_eval_fail_rate: 0.10,
            heartbeat_median_factor: 10.0,
            heartbeat_min_secs: 60.0,
        }
    }
}

pub const ALARM_SCHEMA: &str = "jope.telemetry.alarm/v1";

/// DOPE freeze guard outcome at one boundary (rows D1–D4). Every field is optional on input,
/// so the guard's sampled per-step records (D4 only) deserialize too.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DopeGuardObservation {
    pub epoch: u64,
    /// `None` at a boundary with nothing to compare against yet (run start).
    pub hashes_equal: Option<bool>,
    pub changed_tensors: Vec<String>,
    pub trainable_dope_params: u64,
    /// `None` when the record doesn't count them (sampled step checks).
    pub trainable_jope_params: Option<u64>,
    pub dope_params_in_optimizer: u64,
    pub dope_grads_present: u64,
    pub dope_training_mode: bool,
}

/// Fused-model validation collapse indicators (rows V5–V6).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct SelectionObservation {
    pub epoch: u64,
    pub top1_share: Option<f64>,
    pub unique_frac: Option<f64>,
    pub across_dataset_prediction_spread: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct EvalBatchObservation {
    pub batch: u64,
    pub n: u64,
    pub timeouts: u64,
    pub failures: u64,
    pub repro_mismatches: u64,
}

pub struct Engine {
    run_id: String,
    th: Thresholds,
    collapse_streak: HashMap<String, u32>,
    grad_window: VecDeque<f64>,
    grad_spike_streak: u32,
    val_hist: Vec<(f64, f64)>,
    elo_hist: HashMap<String, Vec<f64>>,
    step_secs: VecDeque<f64>,
    last_step_at: Option<f64>,
    notifier: Option<CommsNotifier>,
}

fn median(it: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = it.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    Some(if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    })
}

/// Appends `## [TEL-alarm] …` notes to comms.md under its flock, at most one per alarm id per
/// run per `min_interval_secs`. Write failures are counted, never propagated: an alarm writer
/// that can itself fail training would be a second thing to monitor.
pub struct CommsNotifier {
    pub comms: PathBuf,
    pub lock: PathBuf,
    pub min_interval_secs: f64,
    last: HashMap<AlarmId, f64>,
    pub write_failures: u64,
    pub suppressed: u64,
}

/// `YYYY-MM-DDTHH:MM:SSZ` from the system clock, without a date crate.
pub fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

mod engine;
mod notifier;

#[cfg(test)]
mod tests;
