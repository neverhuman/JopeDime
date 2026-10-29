//! JSON-in, JSON-out entry points for the TRAIN+GP modules, so the PyO3 layer ([TEL-io]'s
//! `python.rs`) can wrap them as thin `#[pyclass]`es taking and returning `str`.
//!
//! Everything here runs once per generation, epoch, guard or round, never per step, so the JSON
//! cost is off the hot path. DSL text is accepted as input, used only to parse and hash, and
//! never appears in an output.

use crate::alarm::{
    Alarm, CommsNotifier, DopeGuardObservation, Engine, EvalBatchObservation, SelectionObservation,
    Thresholds,
};
use crate::gp::{
    self, BehaviorKind, Candidate, Config, EditOp, EvalStatus, Genome, Interner, ProposalCounters,
    Provenance, Tracker,
};
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

/// Upper bound on trackers (contexts) per GP session. The real flywheel evolves up to 29
/// contexts per round (R1, 2026-10-24), so 16 was too few. With `gp::Config`'s archive bound
/// this keeps a session's archives under 64 × 64 KiB = 4 MiB.
pub const MAX_TRACKERS: usize = 64;
/// The op vocabulary is small (the stage and head pools), so its interner is bounded tighter.
pub const OPS_MAX_ENTRIES: usize = 1024;
/// Largest generation payload accepted. A real flywheel generation (about 16 candidates) is
/// under 10 KB, so this only bounds a pathological caller.
pub const MAX_GENERATION_JSON_BYTES: usize = 1024 * 1024;
pub const OPS_MAX_BYTES: usize = 64 * 1024;

/// One evaluated candidate as the Python recorder sends it. Field names follow the flywheel
/// evaluator's result dicts where one exists.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateIn {
    pub dsl: String,
    #[serde(default)]
    pub parent_dsl: Option<String>,
    #[serde(default)]
    pub operator: String,
    /// `completed`, `failed`, or a specific cause (`timeout`, `nonzero_rc`, `parse_fail`).
    pub status: String,
    #[serde(default)]
    pub cv_gate_passed: bool,
    pub fitness: f64,
    #[serde(default)]
    pub improved: Option<bool>,
    #[serde(default)]
    pub hwm_winner: bool,
    #[serde(default)]
    pub complexity: Option<u32>,
    /// Sign bits of the evaluator's 16-D `proj16` projection (bit i set when component i > 0).
    #[serde(default)]
    pub proj16_bits: Option<u16>,
    /// The G4 descriptor: the candidate's error profile, or its probe-panel log ratios when the
    /// session was opened with `probe_panel`. `null` entries are missing components.
    #[serde(default)]
    pub behavior: Vec<Option<f64>>,
    /// `novel`, `replayed` or `bank`.
    pub provenance: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationIn {
    pub context: String,
    pub round: u32,
    pub generation: u32,
    #[serde(default)]
    pub proposals: u64,
    #[serde(default)]
    pub seen_rejects: u64,
    pub candidates: Vec<CandidateIn>,
}

fn status(s: &str) -> Result<EvalStatus> {
    Ok(match s {
        "completed" | "ok" => EvalStatus::Ok,
        "failed" => EvalStatus::Failed,
        "timeout" => EvalStatus::Timeout,
        "nonzero_rc" => EvalStatus::NonzeroRc,
        "parse_fail" => EvalStatus::ParseFail,
        other => bail!("unknown candidate status {other:?}"),
    })
}

fn provenance(s: &str) -> Result<Provenance> {
    Ok(match s {
        "novel" => Provenance::Novel,
        "replayed" => Provenance::Replayed,
        "bank" => Provenance::Bank,
        other => bail!("unknown provenance {other:?}"),
    })
}

/// A GP loop's telemetry state: one interner pair and one tracker per context.
pub struct GpSession {
    cfg: Config,
    clauses: Interner,
    ops: Interner,
    trackers: BTreeMap<String, Tracker>,
}

impl GpSession {
    /// `probe_panel`: behavior vectors are probe-panel log ratios instead of error profiles.
    pub fn new(op_vocab_size: Option<usize>, seed: Option<u64>, probe_panel: bool) -> Self {
        let behavior_kind = if probe_panel {
            BehaviorKind::ProbePanel
        } else {
            BehaviorKind::ErrorProfile
        };
        let mut cfg = Config {
            op_vocab_size,
            behavior_kind,
            ..Default::default()
        };
        if let Some(s) = seed {
            cfg.seed = s;
        }
        Self {
            cfg,
            clauses: Interner::default(),
            ops: Interner::with_bounds(OPS_MAX_ENTRIES, OPS_MAX_BYTES),
            trackers: BTreeMap::new(),
        }
    }

    /// Summarize one generation (`GenerationIn` JSON) and return the `GenSummary` JSON.
    pub fn summarize_json(&mut self, json: &str) -> Result<String> {
        if json.len() > MAX_GENERATION_JSON_BYTES {
            bail!(
                "generation JSON is {} bytes, over the {MAX_GENERATION_JSON_BYTES}-byte bound",
                json.len()
            );
        }
        let g: GenerationIn = serde_json::from_str(json).context("generation JSON")?;
        let keep = g.candidates.len().min(self.cfg.max_candidates);
        let excess = (g.candidates.len() - keep) as u32;
        let mut cands = Vec::with_capacity(keep);
        for c in &g.candidates[..keep] {
            cands.push(Candidate {
                program: gp::program_sha256(&c.dsl),
                genome: Genome::from_dsl(&c.dsl, &mut self.clauses, &mut self.ops),
                parent: c
                    .parent_dsl
                    .as_deref()
                    .map(|p| Genome::from_dsl(p, &mut self.clauses, &mut self.ops)),
                edit_op: EditOp::from_label(&c.operator),
                status: status(&c.status)?,
                gate_passed: c.cv_gate_passed,
                fitness: c.fitness,
                improved: c.improved,
                hwm_winner: c.hwm_winner,
                complexity: c.complexity,
                proj16: c.proj16_bits,
                behavior: c
                    .behavior
                    .iter()
                    .take(self.cfg.max_behavior_len)
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect(),
                provenance: provenance(&c.provenance)?,
            });
        }
        if !self.trackers.contains_key(&g.context) && self.trackers.len() >= MAX_TRACKERS {
            bail!("GP session already tracks {MAX_TRACKERS} contexts");
        }
        let tracker = self
            .trackers
            .entry(g.context.clone())
            .or_insert_with(|| Tracker::new(g.context.clone(), self.cfg.clone()));
        let dedup = ProposalCounters {
            proposals: g.proposals,
            seen_rejects: g.seen_rejects,
        };
        let mut summary = tracker.summarize(g.round, g.generation, &cands, dedup);
        summary.candidates_truncated += excess;
        Ok(serde_json::to_string(&summary)?)
    }

    /// The clause and op dictionaries for the manifest, plus overflow counts.
    pub fn dictionaries_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&serde_json::json!({
            "clauses": self.clauses.dictionary(),
            "clauses_overflowed": self.clauses.overflowed(),
            "ops": self.ops.dictionary(),
            "ops_overflowed": self.ops.overflowed(),
        }))?)
    }
}

/// The alarm engine behind a string interface. `observe_json` returns the fired alarm as JSON,
/// or `None`; the caller raises when the alarm's `severity` is `fail_closed`.
pub struct AlarmSession {
    engine: Engine,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepIn {
    step: u64,
    loss: f64,
    #[serde(default)]
    grad_norm: Option<f64>,
    wall_secs: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EpochIn {
    epoch: u64,
    train_loss: f64,
    val_loss: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EloIn {
    series: String,
    round: u64,
    elo: f64,
    below_baseline: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LrIn {
    step: u64,
    lr: f64,
    expected: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CountIn {
    at: u64,
    count: u64,
}

impl AlarmSession {
    /// `comms` is `(comms.md, comms.md.lock)`; `None` disables comms notes (tests, dry runs).
    pub fn new(run_id: &str, comms: Option<(&str, &str)>) -> Self {
        let mut engine = Engine::new(run_id, Thresholds::default());
        if let Some((c, l)) = comms {
            engine = engine.with_notifier(CommsNotifier::new(c, l));
        }
        Self { engine }
    }

    pub fn observe_json(
        &mut self,
        kind: &str,
        json: &str,
        now_secs: f64,
    ) -> Result<Option<String>> {
        let e = &mut self.engine;
        let fired: Option<Alarm> = match kind {
            "dope_guard" => e.observe_dope_guard(&parse::<DopeGuardObservation>(json)?, now_secs),
            "generation" => e.observe_generation(&parse::<gp::GenSummary>(json)?, now_secs),
            "step" => {
                let s: StepIn = parse(json)?;
                e.observe_step(s.step, s.loss, s.grad_norm, s.wall_secs)
            }
            "epoch" => {
                let s: EpochIn = parse(json)?;
                e.observe_epoch(s.epoch, s.train_loss, s.val_loss, now_secs)
            }
            "selection" => e.observe_selection(&parse::<SelectionObservation>(json)?, now_secs),
            "elo" => {
                let s: EloIn = parse(json)?;
                e.observe_elo(&s.series, s.round, s.elo, s.below_baseline, now_secs)
            }
            "lr" => {
                let s: LrIn = parse(json)?;
                e.observe_lr(s.step, s.lr, s.expected, now_secs)
            }
            "eval_batch" => e.observe_eval_batch(&parse::<EvalBatchObservation>(json)?, now_secs),
            "sealed_touch" => {
                let s: CountIn = parse(json)?;
                e.observe_sealed_touch(s.at, s.count, now_secs)
            }
            "rust_only" => {
                let s: CountIn = parse(json)?;
                e.observe_rust_only(s.at, s.count, now_secs)
            }
            "heartbeat" => e.heartbeat(now_secs),
            other => bail!("unknown alarm input kind {other:?}"),
        };
        fired
            .map(|a| serde_json::to_string(&a).map_err(Into::into))
            .transpose()
    }
}

fn parse<T: for<'de> Deserialize<'de>>(json: &str) -> Result<T> {
    serde_json::from_str(json).context("alarm input JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn generation_round_trip_never_echoes_dsl() {
        let mut s = GpSession::new(Some(10), None, false);
        let gen = json!({
            "context": "ctx", "round": 1, "generation": 0, "proposals": 4, "seen_rejects": 1,
            "candidates": [
                {"dsl": "(pipe X (a q=1) (h))", "operator": "edit_parameter", "status": "completed",
                 "cv_gate_passed": true, "fitness": -1.0, "improved": true, "proj16_bits": 5,
                 "behavior": [0.1, null, 0.3, 0.4], "provenance": "novel"},
                {"dsl": "(pipe X (b) (h))", "parent_dsl": "(pipe X (a q=1) (h))", "operator": "edit_replace_head",
                 "status": "failed", "fitness": 100.0, "provenance": "replayed"}
            ]
        });
        let out = s.summarize_json(&gen.to_string()).unwrap();
        assert!(!out.contains("pipe X"));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["status"]["failed_unknown_cause"], 1);
        assert_eq!(v["dedup_reject_frac"], 0.25);
        assert_eq!(v["parent_child_hist"][1], 1);
        let dict: Value = serde_json::from_str(&s.dictionaries_json().unwrap()).unwrap();
        assert_eq!(dict["ops"].as_array().unwrap().len(), 3);
        assert!(s
            .summarize_json(&gen.to_string().replace("\"novel\"", "\"mystery\""))
            .is_err());
    }

    #[test]
    fn tracker_cap_covers_a_real_round_and_refuses_beyond_it() {
        let mut s = GpSession::new(None, None, false);
        let gen = |ctx: usize| {
            json!({"context": format!("c{ctx}"), "round": 0, "generation": 0, "candidates": [
                {"dsl": "(pipe X (h))", "status": "completed", "fitness": -1.0, "provenance": "novel"}
            ]})
            .to_string()
        };
        for ctx in 0..MAX_TRACKERS {
            s.summarize_json(&gen(ctx)).unwrap();
        }
        assert!(
            s.summarize_json(&gen(0)).is_ok(),
            "an existing context is still accepted"
        );
        assert!(s.summarize_json(&gen(MAX_TRACKERS)).is_err());
        assert!(s
            .summarize_json(&"x".repeat(MAX_GENERATION_JSON_BYTES + 1))
            .is_err());
    }

    #[test]
    fn alarm_session_dispatches_and_reports_fail_closed() {
        let mut a = AlarmSession::new("run", None);
        let bad = json!({"epoch": 3, "hashes_equal": false, "changed_tensors": ["dope_model.w"],
                         "trainable_dope_params": 0, "trainable_jope_params": 5,
                         "dope_params_in_optimizer": 0, "dope_grads_present": 0, "dope_training_mode": false});
        let fired = a
            .observe_json("dope_guard", &bad.to_string(), 0.0)
            .unwrap()
            .unwrap();
        let v: Value = serde_json::from_str(&fired).unwrap();
        assert_eq!(
            (v["code"].as_str(), v["severity"].as_str()),
            (Some("A1"), Some("fail_closed"))
        );
        assert!(a
            .observe_json("lr", r#"{"step":1,"lr":1e-3,"expected":1e-3}"#, 0.0)
            .unwrap()
            .is_none());
        assert!(a.observe_json("bogus", "{}", 0.0).is_err());
    }
}
