use super::retention::round_record_from_metrics;
use super::{num, read_bounded, SourceRef};
use crate::flywheel::{Contrast, RoundRecord};
use crate::gp::{
    self, Candidate, EditOp, EvalStatus, Genome, Interner, ProposalCounters, Provenance, Tracker,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub const QUEUE_SCHEMA: &str = "jope.telemetry.evo-queue/v1";

/// Row E10 plus the run-level W optimizer stats, per (round, arm) from
/// `evo_batch_queue_telemetry_by_round`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueueRecord {
    pub schema: String,
    pub round: String,
    pub arm: String,
    pub values: BTreeMap<String, f64>,
    pub availability: Vec<String>,
    #[serde(flatten)]
    pub src: SourceRef,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SelfplayReport {
    pub rounds: Vec<RoundRecord>,
    pub queue: Vec<QueueRecord>,
}

/// A `jope.flywheel-selfplay-results/v2` report: live rounds (with the outer win rate on the
/// last one), arm contrasts per evaluation panel, and per-round queue telemetry.
pub fn selfplay_report(path: &Path) -> Result<SelfplayReport> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let v: Map<String, Value> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if v.get("schema").and_then(Value::as_str) != Some("jope.flywheel-selfplay-results/v2") {
        bail!(
            "{} is not jope.flywheel-selfplay-results/v2",
            path.display()
        );
    }
    let mut rep = SelfplayReport::default();
    if let Some(live) = v
        .get("live_rust_gp_selfplay_rounds")
        .and_then(Value::as_array)
    {
        for r in live.iter().filter_map(Value::as_object) {
            let Some(round) = r.get("round").and_then(Value::as_u64) else {
                continue;
            };
            let mut rec = round_record_from_metrics(round as u32, r);
            rec.source = Some(format!("import:{}", src.file_sha256));
            rep.rounds.push(rec);
        }
    }
    let mut contrasts = BTreeMap::new();
    for (panel, pv) in &v {
        let Some(cs) = pv.get("contrasts").and_then(Value::as_object) else {
            continue;
        };
        for (name, c) in cs {
            let Some(mean) = c.get("mean").and_then(Value::as_f64) else {
                continue;
            };
            let ci = c.get("CI_central_98_75").and_then(Value::as_array);
            let at = |i: usize| ci.and_then(|a| a.get(i)).and_then(Value::as_f64);
            contrasts.insert(
                format!("{panel}.{name}"),
                Contrast {
                    mean,
                    ci_low: at(0),
                    ci_high: at(1),
                    ci_level: ci.map(|_| 0.9875),
                },
            );
        }
    }
    let outer = v
        .get("unscored_outer_split_live_rust_evaluation")
        .and_then(Value::as_object);
    // The contrasts and the outer win rate describe the finished run, so they go on the last
    // live round instead of being spread over rounds they weren't measured at.
    if let Some(last) = rep.rounds.last_mut() {
        last.arm_contrasts = contrasts;
        last.selfplay_win_rate = outer
            .and_then(|o| o.get("flywheel_win_rate"))
            .and_then(Value::as_f64);
        last.selfplay_comparisons = outer
            .and_then(|o| o.get("n_unscored_outer_contexts_evaluated"))
            .and_then(Value::as_u64)
            .map(|n| n as u32);
    }
    if let Some(q) = v
        .get("evo_batch_queue_telemetry_by_round")
        .and_then(Value::as_object)
    {
        for (round, arms) in q {
            for (arm, stats) in arms.as_object().into_iter().flatten() {
                let Some(m) = stats.as_object() else { continue };
                let values: BTreeMap<String, f64> = m
                    .iter()
                    .filter_map(|(k, x)| Some((k.clone(), num(x)?)))
                    .collect();
                let mut availability = vec![
                    // Accumulated by AsyncEvoBatchQueue but never emitted (row E10).
                    "gpu_starvation_wait_ms:not_emitted_by_producer".to_owned(),
                ];
                if !values.contains_key("p95_popped_batch_age_ms") {
                    availability.push("popped_batch_age_ms:mean_only".into());
                }
                rep.queue.push(QueueRecord {
                    schema: QUEUE_SCHEMA.into(),
                    round: round.clone(),
                    arm: arm.clone(),
                    values,
                    availability,
                    src: src.clone(),
                });
            }
        }
    }
    Ok(rep)
}

/// Generation summaries from `jope.score/v1` retained scores, one per round.
///
/// These rows are what survived the smart retention filter (top-k winners plus hard
/// negatives), not the population, so every summary built here is tagged
/// `context = "retained:<round>"`. Diversity numbers from them are biased upward against
/// collapse, and are only good for coarse threshold calibration.
pub fn retained_scores(
    path: &Path,
    round: u32,
    op_vocab_size: Option<usize>,
) -> Result<gp::GenSummary> {
    let bytes = read_bounded(path)?;
    let mut clauses = Interner::default();
    let mut ops = Interner::default();
    let mut cands = Vec::new();
    let lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let r: Map<String, Value> = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(_) if i + 1 == lines.len() => continue,
            Err(e) => bail!("{}:{}: {e}", path.display(), i + 1),
        };
        if r.get("schema").and_then(Value::as_str) != Some("jope.score/v1") {
            continue;
        }
        let Some(dsl) = r.get("dsl").and_then(Value::as_str) else {
            continue;
        };
        let program = match r
            .get("program_sha256")
            .and_then(Value::as_str)
            .and_then(hex32)
        {
            Some(p) => p,
            None => gp::program_sha256(dsl),
        };
        let gate = r
            .get("cv_gate_passed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let fitness = r.get("fitness").and_then(Value::as_f64).unwrap_or(f64::NAN);
        cands.push(Candidate {
            program,
            genome: Genome::from_dsl(dsl, &mut clauses, &mut ops),
            parent: None,
            edit_op: EditOp::from_label(r.get("operator").and_then(Value::as_str).unwrap_or("")),
            status: EvalStatus::Ok,
            gate_passed: gate,
            fitness,
            improved: None,
            // HWM beats come from `hwm/*.json` (see `hwm_file`); retained scores don't mark them.
            hwm_winner: false,
            complexity: None,
            proj16: None,
            behavior: Vec::new(),
            // Retained scores carry no bank/corpus membership, so provenance is unknown;
            // `Novel` here would overstate novelty, so these count as replayed.
            provenance: Provenance::Replayed,
        });
    }
    let cfg = gp::Config {
        op_vocab_size,
        ..Default::default()
    };
    let mut t = Tracker::new(format!("retained:{round}"), cfg);
    let mut g = t.summarize(round, 0, &cands, ProposalCounters::default());
    // Provenance is unknown for retained scores; don't report a fraction.
    for p in [
        &mut g.provenance_population,
        &mut g.provenance_passers,
        &mut g.provenance_hwm,
    ] {
        p.novel = None;
        p.replayed = None;
        p.bank = None;
    }
    Ok(g)
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}
