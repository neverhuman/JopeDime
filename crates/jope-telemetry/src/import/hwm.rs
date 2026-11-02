use super::{parse_utc, read_bounded, Retention, SourceRef};
use crate::gp::EditOp;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;

pub const HWM_SCHEMA: &str = "jope.telemetry.hwm-beat/v1";

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
