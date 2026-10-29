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

use crate::flywheel::{Contrast, RoundRecord, ROUND_SCHEMA};
use crate::gp::{
    self, Candidate, EditOp, EvalStatus, Genome, Interner, ProposalCounters, Provenance, Tracker,
};
use crate::train;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const SCALAR_SCHEMA: &str = "jope.telemetry.import-scalar/v1";
pub const HWM_SCHEMA: &str = "jope.telemetry.hwm-beat/v1";
pub const RETENTION_SCHEMA: &str = "jope.telemetry.retention/v1";
pub const QUEUE_SCHEMA: &str = "jope.telemetry.evo-queue/v1";

/// Inputs larger than this are refused rather than read whole.
pub const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub source: String,
    pub file_name: String,
    pub file_sha256: String,
}

impl SourceRef {
    fn of(path: &Path, bytes: &[u8]) -> Self {
        Self {
            source: "import".into(),
            file_name: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            file_sha256: Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        }
    }
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    if len > MAX_INPUT_BYTES {
        bail!(
            "{} is {len} bytes, over the {MAX_INPUT_BYTES}-byte import limit",
            path.display()
        );
    }
    std::fs::read(path).with_context(|| format!("read {}", path.display()))
}

/// One imported scalar, keyed to a step or epoch. `metric` is the spec name when the field
/// maps onto a registered metric, otherwise `imported.<json path>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportedScalar {
    pub schema: String,
    pub metric: String,
    pub metric_id: Option<u16>,
    pub group: Option<String>,
    pub step: Option<u64>,
    pub epoch: Option<f64>,
    pub value: f64,
    #[serde(flatten)]
    pub src: SourceRef,
}

/// Field names of `parallel.neural_training.v1` state rows mapped onto spec metrics.
const ENGINE_FIELDS: &[(&str, &str, Option<&str>)] = &[
    ("loss", "train.loss", None),
    ("sequence_nll", "train.loss.component", Some("sequence_nll")),
    ("ranking_bt", "train.loss.component", Some("ranking_bt")),
    (
        "ranking_listwise",
        "train.loss.component",
        Some("ranking_listwise"),
    ),
    (
        "runtime_huber",
        "train.loss.component",
        Some("runtime_huber"),
    ),
    (
        "contextfree_bt",
        "train.loss.component",
        Some("contextfree_bt"),
    ),
    ("gradient_norm", "optim.grad_norm.global", None),
    ("gradient_clipped", "optim.clipped", None),
    ("step_seconds", "step.wall", None),
    ("examples_per_second", "throughput.samples_per_s", None),
    ("telemetry_seconds", "train.existing_telemetry_s", None),
    ("coordination_seconds", "train.coordination_s", None),
    ("communication_seconds", "train.communication_s", None),
    ("checkpoint_seconds", "train.checkpoint_s", None),
];

fn metric_id(name: &str) -> Option<u16> {
    train::METRICS.iter().find(|m| m.name == name).map(|m| m.id)
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    }
}

/// `parallel.neural_training.v1` rows. `step_seconds` is converted to milliseconds to match
/// `step.wall`. Validation panels become `val.loss.component` scalars at `evaluated_step`.
pub fn engine_state_jsonl(path: &Path) -> Result<Vec<ImportedScalar>> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let mut out = Vec::new();
    let mut seen_val = HashSet::new();
    let lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: Map<String, Value> = match serde_json::from_slice(line) {
            Ok(r) => r,
            // A torn last line from a crashed writer is expected; anything else is not.
            Err(_) if i + 1 == lines.len() => continue,
            Err(e) => bail!("{}:{}: {e}", path.display(), i + 1),
        };
        if row.get("schema").and_then(Value::as_str) != Some("parallel.neural_training.v1") {
            continue;
        }
        // The state's `step` field is the 1-based count of completed optimizer steps.
        let step = row.get("step").and_then(Value::as_u64);
        let epoch = row.get("epoch").and_then(Value::as_f64);
        for &(field, metric, group) in ENGINE_FIELDS {
            if let Some(mut v) = row.get(field).and_then(num) {
                if field == "step_seconds" {
                    v *= 1e3;
                }
                out.push(ImportedScalar {
                    schema: SCALAR_SCHEMA.into(),
                    metric: metric.into(),
                    metric_id: metric_id(metric),
                    group: group.map(Into::into),
                    step,
                    epoch,
                    value: v,
                    src: src.clone(),
                });
            }
        }
        if let Some(val) = row.get("validation").and_then(Value::as_object) {
            let at = val.get("evaluated_step").and_then(Value::as_u64);
            // Every state row repeats the latest panel; import each evaluation once.
            if at.is_some_and(|s| seen_val.insert(s)) {
                for (k, v) in val {
                    if k == "evaluated_step" || k == "source_families" {
                        continue;
                    }
                    if let Some(x) = num(v) {
                        out.push(ImportedScalar {
                            schema: SCALAR_SCHEMA.into(),
                            metric: "val.loss.component".into(),
                            metric_id: metric_id("val.loss.component"),
                            group: Some(k.clone()),
                            step: at,
                            epoch: None,
                            value: x,
                            src: src.clone(),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Any per-epoch JSONL, such as `train_metrics.jsonl` (rows are free-form maps keyed by
/// `epoch`): every finite numeric leaf becomes `imported.<dotted path>`. Lists are skipped.
pub fn generic_epoch_jsonl(path: &Path) -> Result<Vec<ImportedScalar>> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let mut out = Vec::new();
    let lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: Map<String, Value> = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(_) if i + 1 == lines.len() => continue,
            Err(e) => bail!("{}:{}: {e}", path.display(), i + 1),
        };
        let epoch = row.get("epoch").and_then(Value::as_f64);
        let step = row.get("step").and_then(Value::as_u64);
        let mut leaves = Vec::new();
        flatten("", &Value::Object(row), &mut leaves);
        for (k, v) in leaves {
            if k == "epoch" || k == "step" || !v.is_finite() {
                continue;
            }
            out.push(ImportedScalar {
                schema: SCALAR_SCHEMA.into(),
                metric: format!("imported.{k}"),
                metric_id: None,
                group: None,
                step,
                epoch,
                value: v,
                src: src.clone(),
            });
        }
    }
    Ok(out)
}

fn flatten(prefix: &str, v: &Value, out: &mut Vec<(String, f64)>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&p, x, out);
            }
        }
        _ => {
            if let Some(x) = num(v) {
                out.push((prefix.to_owned(), x));
            }
        }
    }
}

/// Row E3 from one `jope.hwm/v1` file. The flywheel's HWM json lacks most of the AGENTS.md
/// comparison identity, so the missing fields are listed instead of guessed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HwmBeat {
    pub schema: String,
    pub context_sha256: String,
    pub program_sha256: String,
    pub operator: EditOp,
    pub parents: Vec<String>,
    pub previous_best_fitness: Option<f64>,
    pub new_best_fitness: Option<f64>,
    pub fitness_improvement: Option<f64>,
    pub cv_gate_passed: Option<bool>,
    pub cv_test_agreed: Option<bool>,
    pub updated_at_utc: Option<String>,
    pub missing_identity_fields: Vec<String>,
    #[serde(flatten)]
    pub src: SourceRef,
}

/// The comparison identity a target improvement must replay under (AGENTS.md).
pub const HWM_IDENTITY_FIELDS: &[&str] = &[
    "task",
    "dataset_hash",
    "cut_id",
    "evaluator",
    "vocabulary",
    "protocol",
];

pub fn hwm_file(path: &Path) -> Result<HwmBeat> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let v: Map<String, Value> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if v.get("schema").and_then(Value::as_str) != Some("jope.hwm/v1") {
        bail!("{} is not jope.hwm/v1", path.display());
    }
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned);
    let f = |k: &str| v.get(k).and_then(Value::as_f64);
    let b = |k: &str| v.get(k).and_then(Value::as_bool);
    Ok(HwmBeat {
        schema: HWM_SCHEMA.into(),
        context_sha256: s("context_sha256").context("hwm without context_sha256")?,
        program_sha256: s("program_sha256").context("hwm without program_sha256")?,
        operator: EditOp::from_label(v.get("operator").and_then(Value::as_str).unwrap_or("")),
        parents: v
            .get("parents")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        previous_best_fitness: f("previous_best_fitness"),
        new_best_fitness: f("new_best_fitness"),
        fitness_improvement: f("fitness_improvement"),
        cv_gate_passed: b("cv_gate_passed"),
        cv_test_agreed: b("cv_test_agreed"),
        updated_at_utc: s("updated_at_utc"),
        missing_identity_fields: HWM_IDENTITY_FIELDS
            .iter()
            .filter(|k| !v.contains_key(**k))
            .map(|k| (*k).to_owned())
            .collect(),
        src,
    })
}

/// Row E3 cross-check: do a round's HWM files belong to that round?
///
/// A beat counts as in the round when its `updated_at_utc` falls inside
/// `[created_at − wall_s − slack, created_at + slack]`. On the real 2026-10-24 run, all four
/// `round-01/hwm` files predate round 1 by about 4.8 h. They are leftovers of an earlier run,
/// which is why that tree has 9 HWM files but only 5 counted winners.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HwmConsistency {
    pub schema: String,
    pub round: u32,
    pub files: u32,
    pub in_round_window: u32,
    pub stale: Vec<String>,
    pub undated: u32,
    pub retention_new_hwm_count: Option<u64>,
    /// `in_round_window == new_hwm_count`, when both are known.
    pub count_matches: Option<bool>,
    pub availability: Option<String>,
}

pub const HWM_CONSISTENCY_SCHEMA: &str = "jope.telemetry.hwm-consistency/v1";

pub fn hwm_consistency(
    round: u32,
    beats: &[HwmBeat],
    ret: &Retention,
    slack_s: i64,
) -> HwmConsistency {
    let window = ret
        .created_at_utc
        .as_deref()
        .and_then(parse_utc)
        .map(|end| {
            let start = end - ret.wall_s.unwrap_or(0.0).ceil() as i64 - slack_s;
            (start, end + slack_s)
        });
    let (mut inside, mut undated, mut stale) = (0u32, 0u32, Vec::new());
    for b in beats {
        match (b.updated_at_utc.as_deref().and_then(parse_utc), window) {
            (Some(t), Some((lo, hi))) if (lo..=hi).contains(&t) => inside += 1,
            (Some(_), Some(_)) => stale.push(b.context_sha256.clone()),
            _ => undated += 1,
        }
    }
    let new = ret.counts.get("new_hwm_count").copied();
    HwmConsistency {
        schema: HWM_CONSISTENCY_SCHEMA.into(),
        round,
        files: beats.len() as u32,
        in_round_window: inside,
        stale,
        undated,
        retention_new_hwm_count: new,
        count_matches: window.and(new).map(|n| n == inside as u64),
        availability: window.is_none().then(|| "no_round_window".to_owned()),
    }
}

/// Seconds since the epoch for `YYYY-MM-DDTHH:MM:SS[.frac]Z`; `None` for anything else.
pub fn parse_utc(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    if !s.ends_with('Z') {
        return None;
    }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hh, mm, ss) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // Howard Hinnant's days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// `retention_summary` of a round (part of the E2 funnel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retention {
    pub schema: String,
    pub round: u32,
    pub created_at_utc: Option<String>,
    pub wall_s: Option<f64>,
    pub counts: BTreeMap<String, u64>,
    #[serde(flatten)]
    pub src: SourceRef,
}

/// A `gp_distilled/v1` round file: its `round_metrics` become a [`RoundRecord`] (target deltas,
/// wall time) and its `retention_summary` a [`Retention`].
pub fn distilled_round(path: &Path) -> Result<(RoundRecord, Retention)> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let v: Map<String, Value> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if v.get("schema").and_then(Value::as_str) != Some("gp_distilled/v1") {
        bail!("{} is not gp_distilled/v1", path.display());
    }
    let round = v
        .get("round")
        .and_then(Value::as_u64)
        .context("distilled round without round")? as u32;
    let rm = v
        .get("round_metrics")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut rec = round_record_from_metrics(round, &rm);
    rec.source = Some(format!("import:{}", src.file_sha256));
    let counts = v
        .get("retention_summary")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, x)| Some((k.clone(), x.as_u64()?)))
                .collect()
        })
        .unwrap_or_default();
    let created_at_utc = v
        .get("created_at_utc")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let wall_s = rec.wall_s;
    Ok((
        rec,
        Retention {
            schema: RETENTION_SCHEMA.into(),
            round,
            created_at_utc,
            wall_s,
            counts,
            src,
        },
    ))
}

fn round_record_from_metrics(round: u32, rm: &Map<String, Value>) -> RoundRecord {
    let mut target_deltas = BTreeMap::new();
    for k in [
        "mean_strat300_evo_fitness_delta",
        "mean_best_new_fitness_delta_vs_incumbent",
    ] {
        if let Some(x) = rm.get(k).and_then(Value::as_f64) {
            target_deltas.insert(k.to_owned(), x);
        }
    }
    for k in [
        "candidates_screened_in_rust_strat300",
        "candidates_confirmed_in_rust_full_jtt1",
        "new_champions_added",
        "contexts_expanded",
    ] {
        if let Some(x) = rm.get(k).and_then(Value::as_f64) {
            target_deltas.insert(format!("count.{k}"), x);
        }
    }
    RoundRecord {
        schema: ROUND_SCHEMA.into(),
        round,
        generation: round,
        wall_s: rm.get("wall_seconds").and_then(Value::as_f64),
        target_deltas,
        ..Default::default()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpfile(dir: &tempfile::TempDir, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn engine_rows_map_and_validation_is_deduplicated() {
        let d = tempfile::tempdir().unwrap();
        let row = |step: u64, vstep: u64| {
            format!(
                r#"{{"schema":"parallel.neural_training.v1","step":{step},"epoch":0.5,"loss":2.0,"sequence_nll":1.5,"gradient_norm":0.7,"gradient_clipped":false,"step_seconds":0.25,"validation":{{"sequence_nll":1.1,"evaluated_step":{vstep},"source_families":3}}}}"#
            )
        };
        let body = format!("{}\n{}\n{{\"torn", row(1, 1), row(2, 1));
        let p = tmpfile(&d, "state.jsonl", &body);
        let out = engine_state_jsonl(&p).unwrap();
        let wall: Vec<_> = out.iter().filter(|s| s.metric == "step.wall").collect();
        assert_eq!(wall.len(), 2);
        assert_eq!(wall[0].value, 250.0);
        assert_eq!(
            out.iter()
                .filter(|s| s.metric == "val.loss.component")
                .count(),
            1
        );
        assert!(out
            .iter()
            .all(|s| s.src.source == "import" && s.src.file_sha256.len() == 64));
        let clipped = out.iter().find(|s| s.metric == "optim.clipped").unwrap();
        assert_eq!(
            (clipped.value, clipped.metric_id),
            (0.0, Some(train::GRAD_CLIPPED))
        );
    }

    #[test]
    fn parse_utc_known_values() {
        // Reference values from Python's calendar.timegm.
        assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_utc("2026-10-24T05:52:15Z"), Some(1_792_821_135));
        assert_eq!(parse_utc("2000-02-29T12:00:00.5Z"), Some(951_825_600));
        assert_eq!(parse_utc("2026-10-24 05:52:15Z"), None);
        assert_eq!(parse_utc("2026-13-01T00:00:00Z"), None);
    }

    #[test]
    fn hwm_consistency_flags_beats_from_an_earlier_run() {
        let src = || SourceRef {
            source: "import".into(),
            file_name: "x".into(),
            file_sha256: "0".into(),
        };
        let beat = |cid: &str, t: &str| HwmBeat {
            schema: HWM_SCHEMA.into(),
            context_sha256: cid.into(),
            program_sha256: "p".into(),
            operator: EditOp::EditParameter,
            parents: vec![],
            previous_best_fitness: None,
            new_best_fitness: None,
            fitness_improvement: None,
            cv_gate_passed: None,
            cv_test_agreed: None,
            updated_at_utc: Some(t.into()),
            missing_identity_fields: vec![],
            src: src(),
        };
        let ret = Retention {
            schema: RETENTION_SCHEMA.into(),
            round: 1,
            created_at_utc: Some("2026-10-24T05:52:15Z".into()),
            wall_s: Some(183.26),
            counts: [("new_hwm_count".to_owned(), 0u64)].into(),
            src: src(),
        };
        let beats = [
            beat("a", "2026-10-24T00:59:18Z"),
            beat("b", "2026-10-24T01:03:35Z"),
        ];
        let c = hwm_consistency(1, &beats, &ret, 60);
        assert_eq!((c.files, c.in_round_window, c.stale.len()), (2, 0, 2));
        assert_eq!(
            c.count_matches,
            Some(true),
            "0 in-window beats == new_hwm_count 0"
        );
        let fresh = [beat("c", "2026-10-24T05:50:00Z")];
        let c = hwm_consistency(1, &fresh, &ret, 60);
        assert_eq!((c.in_round_window, c.count_matches), (1, Some(false)));
    }

    #[test]
    fn hwm_lists_missing_identity_and_drops_dsl() {
        let d = tempfile::tempdir().unwrap();
        let p = tmpfile(
            &d,
            "c.json",
            r#"{"schema":"jope.hwm/v1","context_sha256":"aa","program_sha256":"bb","dsl":"(pipe X (a) (b))","operator":"edit_replace_head","parents":["cc"],"previous_best_fitness":-1.0,"new_best_fitness":-1.1,"fitness_improvement":0.1,"cv_gate_passed":true}"#,
        );
        let h = hwm_file(&p).unwrap();
        assert_eq!(h.operator, EditOp::EditReplaceHead);
        assert_eq!(h.missing_identity_fields.len(), HWM_IDENTITY_FIELDS.len());
        assert!(!serde_json::to_string(&h).unwrap().contains("pipe X"));
    }

    #[test]
    fn retained_scores_summary_has_no_provenance_claim() {
        let d = tempfile::tempdir().unwrap();
        let rows = [
            r#"{"schema":"jope.score/v1","dsl":"(pipe X (a q=1) (h))","fitness":-0.5,"cv_gate_passed":true,"operator":"edit_parameter","retention_tier":"top_k_winner"}"#,
            r#"{"schema":"jope.score/v1","dsl":"(pipe X (a q=2) (h))","fitness":-0.4,"cv_gate_passed":true,"operator":"edit_parameter","retention_tier":"top_k_winner"}"#,
            r#"{"schema":"jope.score/v1","dsl":"(pipe X (b) (h))","fitness":100.0,"cv_gate_passed":false,"operator":"edit_insert_stage","retention_tier":"hard_negative"}"#,
        ];
        let p = tmpfile(&d, "s.jsonl", &rows.join("\n"));
        let g = retained_scores(&p, 1, Some(4)).unwrap();
        assert_eq!(g.pop_size, 3);
        assert_eq!(g.status.gate_passed, 2);
        assert_eq!(
            g.fitness_passers.max,
            Some(-0.4),
            "sentinel 100.0 stays out of the quantiles"
        );
        assert!(g.provenance_population.novel.is_none());
        assert_eq!(g.ops_distinct, 3);
        assert_eq!(g.ops_unused, Some(1));
    }
}
