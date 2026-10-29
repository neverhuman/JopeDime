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

impl Engine {
    pub fn new(run_id: impl Into<String>, th: Thresholds) -> Self {
        Self {
            run_id: run_id.into(),
            th,
            collapse_streak: HashMap::new(),
            grad_window: VecDeque::new(),
            grad_spike_streak: 0,
            val_hist: Vec::new(),
            elo_hist: HashMap::new(),
            step_secs: VecDeque::new(),
            last_step_at: None,
            notifier: None,
        }
    }

    pub fn with_notifier(mut self, n: CommsNotifier) -> Self {
        self.notifier = Some(n);
        self
    }

    fn fire(&mut self, id: AlarmId, at: u64, reasons: Vec<String>, now_secs: f64) -> Alarm {
        let a = Alarm {
            schema: ALARM_SCHEMA.into(),
            id,
            code: id.code().into(),
            severity: id.severity(),
            run_id: self.run_id.clone(),
            at,
            reasons,
        };
        eprintln!(
            "JTEL-ALARM {} {:?} run={} at={} {}",
            a.code,
            a.severity,
            a.run_id,
            a.at,
            a.reasons.join("; ")
        );
        if let Some(n) = &mut self.notifier {
            n.notify(&a, now_secs);
        }
        a
    }

    /// A1: any freeze invariant broken. `FailClosed`; the caller raises.
    pub fn observe_dope_guard(&mut self, o: &DopeGuardObservation, now: f64) -> Option<Alarm> {
        let mut r = Vec::new();
        if o.hashes_equal == Some(false) {
            r.push(format!(
                "DOPE hash changed: {}",
                o.changed_tensors.join(",")
            ));
        }
        if o.trainable_dope_params > 0 {
            r.push(format!(
                "{} DOPE params have requires_grad=True",
                o.trainable_dope_params
            ));
        }
        if o.dope_params_in_optimizer > 0 {
            r.push(format!(
                "{} DOPE params in optimizer groups",
                o.dope_params_in_optimizer
            ));
        }
        if o.dope_grads_present > 0 {
            r.push(format!("{} DOPE params hold a .grad", o.dope_grads_present));
        }
        if o.dope_training_mode {
            r.push("dope_model.training is True".into());
        }
        if o.trainable_jope_params == Some(0) {
            // Not a freeze violation, but nothing is learning: report under A1 for visibility.
            r.push("no trainable JopeDime params".into());
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A1DopeFreeze, o.epoch, r, now))
    }

    /// A2: a collapse condition held for N consecutive generations of one context.
    pub fn observe_generation(&mut self, g: &GenSummary, now: f64) -> Option<Alarm> {
        let th = &self.th;
        let mut r = Vec::new();
        let lt = |v: Option<f64>, t: f64| v.is_some_and(|v| v < t);
        if lt(g.unique_genotype_frac, th.min_unique_genotype_frac) {
            r.push(format!(
                "unique_genotype_frac={:.3}",
                g.unique_genotype_frac.unwrap()
            ));
        }
        if lt(
            g.behavior.fitness_profile_unique_frac,
            th.min_behavior_unique_frac,
        ) {
            r.push(format!(
                "fitness_profile_unique_frac={:.3}",
                g.behavior.fitness_profile_unique_frac.unwrap()
            ));
        }
        if lt(g.op_entropy_norm, th.min_op_entropy_norm) {
            r.push(format!("op_entropy_norm={:.3}", g.op_entropy_norm.unwrap()));
        }
        if g.dedup_reject_frac
            .is_some_and(|v| v > th.max_dedup_reject_frac)
        {
            r.push(format!(
                "dedup_reject_frac={:.3}",
                g.dedup_reject_frac.unwrap()
            ));
        }
        if g.fitness_passers.n >= 4 && lt(g.fitness_passers.iqr, th.min_passer_fitness_iqr) {
            r.push(format!(
                "passer_fitness_iqr={:.2e}",
                g.fitness_passers.iqr.unwrap()
            ));
        }
        if g.provenance_passers.n > 0 && lt(g.provenance_passers.novel, th.min_novel_passer_frac) {
            r.push(format!(
                "novel_passer_frac={:.3}",
                g.provenance_passers.novel.unwrap()
            ));
        }
        let streak = self.collapse_streak.entry(g.context.clone()).or_default();
        if r.is_empty() {
            *streak = 0;
            return None;
        }
        *streak += 1;
        if *streak < th.collapse_consecutive_gens {
            return None;
        }
        let s = *streak;
        r.push(format!("{s} consecutive generations"));
        Some(self.fire(AlarmId::A2GpCollapse, g.generation as u64, r, now))
    }

    /// A3 (step part) and A10 bookkeeping. `wall_secs` is the step's completion time.
    pub fn observe_step(
        &mut self,
        step: u64,
        loss: f64,
        grad_norm: Option<f64>,
        wall_secs: f64,
    ) -> Option<Alarm> {
        if let Some(prev) = self.last_step_at.replace(wall_secs) {
            if self.step_secs.len() == 256 {
                self.step_secs.pop_front();
            }
            self.step_secs.push_back(wall_secs - prev);
        }
        let mut r = Vec::new();
        if !loss.is_finite() {
            r.push(format!("non-finite loss {loss}"));
        }
        if let Some(g) = grad_norm {
            if !g.is_finite() {
                r.push(format!("non-finite grad norm {g}"));
            } else {
                let spiking = self.grad_window.len() >= self.th.grad_spike_window / 2
                    && median(self.grad_window.iter().copied())
                        .is_some_and(|m| m > 0.0 && g > self.th.grad_spike_factor * m);
                self.grad_spike_streak = if spiking {
                    self.grad_spike_streak + 1
                } else {
                    0
                };
                if self.grad_spike_streak >= self.th.grad_spike_consecutive {
                    r.push(format!(
                        "grad norm {g:.3e} > {}x running median for {} steps",
                        self.th.grad_spike_factor, self.grad_spike_streak
                    ));
                }
                if !spiking {
                    if self.grad_window.len() == self.th.grad_spike_window {
                        self.grad_window.pop_front();
                    }
                    self.grad_window.push_back(g);
                }
            }
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A3Divergence, step, r, wall_secs))
    }

    /// A3 (epoch part): validation loss rising for N epochs while training loss falls.
    pub fn observe_epoch(
        &mut self,
        epoch: u64,
        train_loss: f64,
        val_loss: f64,
        now: f64,
    ) -> Option<Alarm> {
        self.val_hist.push((train_loss, val_loss));
        let n = self.th.val_rise_epochs as usize;
        if self.val_hist.len() <= n {
            return None;
        }
        let w = &self.val_hist[self.val_hist.len() - n - 1..];
        let val_rising = w.windows(2).all(|p| p[1].1 > p[0].1);
        let train_falling = w.windows(2).all(|p| p[1].0 < p[0].0);
        (val_rising && train_falling).then(|| {
            self.fire(
                AlarmId::A3Divergence,
                epoch,
                vec![format!(
                    "val loss rising {n} epochs while train loss falls (overfit)"
                )],
                now,
            )
        })
    }

    /// A4: the fused model picks the same winner everywhere, or predicts a constant.
    pub fn observe_selection(&mut self, o: &SelectionObservation, now: f64) -> Option<Alarm> {
        let th = &self.th;
        let mut r = Vec::new();
        if o.top1_share
            .is_some_and(|v| v > th.max_selection_top1_share)
        {
            r.push(format!("top1_share={:.3}", o.top1_share.unwrap()));
        }
        if o.unique_frac
            .is_some_and(|v| v < th.min_selection_unique_frac)
        {
            r.push(format!(
                "selection_unique_frac={:.3}",
                o.unique_frac.unwrap()
            ));
        }
        if o.across_dataset_prediction_spread
            .is_some_and(|v| v < th.min_prediction_spread)
        {
            r.push(format!(
                "prediction_spread={:.2e}",
                o.across_dataset_prediction_spread.unwrap()
            ));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A4SelectionCollapse, o.epoch, r, now))
    }

    /// A5: Elo of one task type down for N consecutive rounds within one `elo_fit_id` series.
    /// `series` names the fit series and task; values from different fits never share a key.
    pub fn observe_elo(
        &mut self,
        series: &str,
        round: u64,
        elo: f64,
        below_baseline: bool,
        now: f64,
    ) -> Option<Alarm> {
        let h = self.elo_hist.entry(series.to_owned()).or_default();
        h.push(elo);
        let n = self.th.elo_drop_rounds as usize;
        let mut r = Vec::new();
        if h.len() > n && h[h.len() - n - 1..].windows(2).all(|p| p[1] < p[0]) {
            r.push(format!("{series}: Elo down {n} consecutive rounds"));
        }
        if below_baseline && round >= 3 {
            r.push(format!("{series}: below H baseline at round {round}"));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A5EloRegression, round, r, now))
    }

    /// A6: the optimizer's LR differs from the schedule the config declares.
    pub fn observe_lr(&mut self, step: u64, lr: f64, expected: f64, now: f64) -> Option<Alarm> {
        let rel = (lr - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
        (rel > self.th.lr_rel_tol).then(|| {
            self.fire(
                AlarmId::A6ScheduleDrift,
                step,
                vec![format!("lr={lr:.6e} expected={expected:.6e} rel={rel:.2e}")],
                now,
            )
        })
    }

    /// A7: evaluator timeouts, failures, or non-reproducible fitness.
    pub fn observe_eval_batch(&mut self, o: &EvalBatchObservation, now: f64) -> Option<Alarm> {
        if o.n == 0 {
            return None;
        }
        let mut r = Vec::new();
        let tr = o.timeouts as f64 / o.n as f64;
        let fr = o.failures as f64 / o.n as f64;
        if tr > self.th.max_eval_timeout_rate {
            r.push(format!("timeout_rate={tr:.3}"));
        }
        if fr > self.th.max_eval_fail_rate {
            r.push(format!("fail_rate={fr:.3}"));
        }
        if o.repro_mismatches > 0 {
            r.push(format!(
                "{} fitness re-evaluations differed",
                o.repro_mismatches
            ));
        }
        (!r.is_empty()).then(|| self.fire(AlarmId::A7EvaluatorHealth, o.batch, r, now))
    }

    /// A8: any evaluation touched the sealed partition. `FailClosed`.
    pub fn observe_sealed_touch(&mut self, at: u64, count: u64, now: f64) -> Option<Alarm> {
        (count > 0).then(|| {
            self.fire(
                AlarmId::A8SealedTouch,
                at,
                vec![format!("{count} sealed-partition evaluations")],
                now,
            )
        })
    }

    /// A9: a flywheel target other than Rust was emitted. `FailClosed` for emission.
    pub fn observe_rust_only(&mut self, at: u64, violations: u64, now: f64) -> Option<Alarm> {
        (violations > 0).then(|| {
            self.fire(
                AlarmId::A9RustOnlyTarget,
                at,
                vec![format!("{violations} non-Rust targets")],
                now,
            )
        })
    }

    /// A10: called by the drain on its 1 s tick while the producer process is alive.
    pub fn heartbeat(&mut self, now: f64) -> Option<Alarm> {
        let last = self.last_step_at?;
        let med = median(self.step_secs.iter().copied()).unwrap_or(0.0);
        let limit = (self.th.heartbeat_median_factor * med).max(self.th.heartbeat_min_secs);
        let idle = now - last;
        (idle > limit).then(|| {
            self.fire(
                AlarmId::A10TrainerHeartbeat,
                0,
                vec![format!("no step for {idle:.0}s (limit {limit:.0}s)")],
                now,
            )
        })
    }
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

impl CommsNotifier {
    pub fn new(comms: impl Into<PathBuf>, lock: impl Into<PathBuf>) -> Self {
        Self {
            comms: comms.into(),
            lock: lock.into(),
            min_interval_secs: 3600.0,
            last: HashMap::new(),
            write_failures: 0,
            suppressed: 0,
        }
    }

    pub fn notify(&mut self, a: &Alarm, now_secs: f64) {
        if self
            .last
            .get(&a.id)
            .is_some_and(|&t| now_secs - t < self.min_interval_secs)
        {
            self.suppressed += 1;
            return;
        }
        self.last.insert(a.id, now_secs);
        if self.append(a).is_err() {
            self.write_failures += 1;
        }
    }

    fn append(&self, a: &Alarm) -> std::io::Result<()> {
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.lock)?;
        lock.lock()?;
        let ts = utc_now_rfc3339();
        let mut f = std::fs::OpenOptions::new().append(true).open(&self.comms)?;
        let sev = match a.severity {
            Severity::FailClosed => "FAIL-CLOSED",
            Severity::Alarm => "alarm",
        };
        let body = format!(
            "\n## [TEL-alarm] {ts} — {} {sev} run `{}` at {}\n{}\n",
            a.code,
            a.run_id,
            a.at,
            a.reasons
                .iter()
                .map(|r| format!("- {r}\n"))
                .collect::<String>()
                .trim_end()
        );
        f.write_all(body.as_bytes())?;
        f.flush()
        // The lock is released when `lock` drops.
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::{BehaviorStats, ProvenanceFracs, Quantiles};

    fn gen(context: &str, g: u32, unique: f64) -> GenSummary {
        GenSummary {
            schema: crate::gp::GEN_SCHEMA.into(),
            context: context.into(),
            round: 0,
            generation: g,
            pop_size: 10,
            candidates_truncated: 0,
            genomes_truncated: 0,
            unique_genotype_frac: Some(unique),
            dedup: Default::default(),
            dedup_reject_frac: None,
            edit_distance_pairs: 0,
            edit_distance_mean: None,
            edit_distance_p10: None,
            parent_child_hist: [0; 8],
            behavior: BehaviorStats::default(),
            proj16_unique_frac: None,
            proj16_mean_hamming: None,
            op_counts: Default::default(),
            op_entropy_nats: None,
            op_entropy_norm: None,
            ops_distinct: 0,
            ops_unused: None,
            edit_ops: Default::default(),
            size_clauses: Quantiles::default(),
            complexity: Quantiles::default(),
            status: Default::default(),
            fitness_passers: Quantiles::default(),
            best_fitness: None,
            gens_since_improved: 0,
            provenance_population: ProvenanceFracs::default(),
            provenance_passers: ProvenanceFracs::default(),
            provenance_hwm: ProvenanceFracs::default(),
        }
    }

    #[test]
    fn collapse_needs_consecutive_generations_per_context() {
        let mut e = Engine::new("r", Thresholds::default());
        assert!(e.observe_generation(&gen("a", 0, 0.2), 0.0).is_none());
        assert!(e.observe_generation(&gen("b", 0, 0.2), 0.0).is_none());
        assert!(e.observe_generation(&gen("a", 1, 0.2), 0.0).is_none());
        assert!(
            e.observe_generation(&gen("a", 2, 0.9), 0.0).is_none(),
            "streak resets"
        );
        assert!(e.observe_generation(&gen("a", 3, 0.2), 0.0).is_none());
        assert!(e.observe_generation(&gen("a", 4, 0.2), 0.0).is_none());
        let a = e.observe_generation(&gen("a", 5, 0.2), 0.0).unwrap();
        assert_eq!(a.id, AlarmId::A2GpCollapse);
        assert_eq!(a.severity, Severity::Alarm);
    }

    #[test]
    fn dope_guard_fails_closed() {
        let mut e = Engine::new("r", Thresholds::default());
        let ok = DopeGuardObservation {
            hashes_equal: Some(true),
            trainable_jope_params: Some(5),
            ..Default::default()
        };
        assert!(e.observe_dope_guard(&ok, 0.0).is_none());
        let bad = DopeGuardObservation {
            hashes_equal: Some(false),
            changed_tensors: vec!["dope_model.w".into()],
            trainable_jope_params: Some(5),
            ..Default::default()
        };
        let a = e.observe_dope_guard(&bad, 0.0).unwrap();
        assert_eq!(a.severity, Severity::FailClosed);
        assert!(a.reasons[0].contains("dope_model.w"));
    }

    #[test]
    fn grad_spike_needs_consecutive_steps_and_nonfinite_fires_at_once() {
        let mut e = Engine::new("r", Thresholds::default());
        for s in 0..100 {
            assert!(e.observe_step(s, 1.0, Some(1.0), s as f64).is_none());
        }
        for s in 100..104 {
            assert!(e.observe_step(s, 1.0, Some(50.0), s as f64).is_none());
        }
        assert!(e.observe_step(104, 1.0, Some(50.0), 104.0).is_some());
        assert!(e.observe_step(105, f64::NAN, Some(1.0), 105.0).is_some());
    }

    #[test]
    fn heartbeat_uses_median_step_time_with_floor() {
        let mut e = Engine::new("r", Thresholds::default());
        for s in 0..10 {
            e.observe_step(s, 1.0, None, s as f64);
        }
        assert!(e.heartbeat(60.0).is_none());
        assert!(e.heartbeat(80.0).is_some());
    }

    #[test]
    fn utc_format() {
        let s = utc_now_rfc3339();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T');
    }
}
