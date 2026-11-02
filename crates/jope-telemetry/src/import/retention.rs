use super::{read_bounded, SourceRef};
use crate::flywheel::{RoundRecord, ROUND_SCHEMA};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::Path;

pub const RETENTION_SCHEMA: &str = "jope.telemetry.retention/v1";

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
