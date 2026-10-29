//! The TRAIN+GP part of `jtel summary`: a machine-readable report plus a text rendering.
//!
//! Input is normalized scalars (metric name, group, step, value) and typed records (kind plus
//! JSON body), so it works on a live run's events, on `jtel import` output, or on both. Missing
//! data is reported as missing, with its reason when one was recorded; nothing is filled in.

use crate::alarm::{Alarm, Severity};
use crate::flywheel::RoundRecord;
use crate::gp::{GenSummary, GEN_SCHEMA};
use crate::import::{HwmConsistency, Retention, HWM_CONSISTENCY_SCHEMA, RETENTION_SCHEMA};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const SUMMARY_SCHEMA: &str = "jope.telemetry.train-gp-summary/v1";

/// A scalar from the hot path or an import, by spec metric name.
#[derive(Clone, Copy, Debug)]
pub struct ScalarIn<'a> {
    pub metric: &'a str,
    pub group: Option<&'a str>,
    pub step: Option<u64>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Dist {
    pub n: u64,
    pub nonfinite: u64,
    pub first: Option<f64>,
    pub last: Option<f64>,
    pub min: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
}

impl Dist {
    fn of(values: &[f64]) -> Self {
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let first = finite.first().copied();
        let last = finite.last().copied();
        finite.sort_by(f64::total_cmp);
        let q = |p| crate::gp::quantile_sorted(&finite, p);
        Self {
            n: values.len() as u64,
            nonfinite: (values.len() - finite.len()) as u64,
            first,
            last,
            min: finite.first().copied(),
            p50: q(0.5),
            p95: q(0.95),
            max: finite.last().copied(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TrainingSummary {
    pub steps: u64,
    pub loss: Dist,
    pub grad_norm: Dist,
    pub clipped_frac: Option<f64>,
    pub step_wall_ms: Dist,
    /// Mean share of the host step wall time per host span (rows T11), largest first.
    pub host_span_share: Vec<(String, f64)>,
    pub unattributed_share: Option<f64>,
    pub val_loss: Dist,
    pub generalization_gap_last: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DopeSummary {
    pub guard_records: u64,
    pub compared: u64,
    pub all_equal: Option<bool>,
    pub violations: Vec<String>,
    pub last_composite_sha256: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GpContextSummary {
    pub generations: u64,
    pub last_unique_genotype_frac: Option<f64>,
    pub last_fitness_profile_unique_frac: Option<f64>,
    pub last_op_entropy_norm: Option<f64>,
    pub best_fitness: Option<f64>,
    pub gens_since_improved: u32,
    pub failed: u64,
    pub evaluated: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EvolutionSummary {
    pub rounds: Vec<u32>,
    /// Per target-delta key, values by round order.
    pub target_deltas: BTreeMap<String, Vec<f64>>,
    /// Keys whose last value is under 10% of the first in magnitude: the round-over-round gain
    /// is fading.
    pub diminishing: Vec<String>,
    pub hwm_rounds_inconsistent: Vec<u32>,
    pub hwm_stale_files: u64,
    pub retention_totals: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AlarmSummary {
    pub by_code: BTreeMap<String, u64>,
    pub fail_closed: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub schema: String,
    pub training: TrainingSummary,
    pub dope: DopeSummary,
    pub gp: BTreeMap<String, GpContextSummary>,
    pub evolution: EvolutionSummary,
    pub alarms: AlarmSummary,
    /// Metrics that were looked for and not found, with the recorded reason when there is one.
    pub missing: BTreeMap<String, String>,
    pub unparsed_records: u64,
}

const HOST_SPANS: &[&str] = &[
    "step.span.fwd",
    "step.span.bwd",
    "step.span.opt",
    "step.span.val",
    "step.span.ckpt",
    "step.span.guard",
];

pub fn summarize<'a>(
    scalars: impl IntoIterator<Item = ScalarIn<'a>>,
    records: impl IntoIterator<Item = (&'a str, &'a Value)>,
) -> Summary {
    let mut by_metric: BTreeMap<&str, Vec<(Option<u64>, f64)>> = BTreeMap::new();
    for s in scalars {
        by_metric
            .entry(s.metric)
            .or_default()
            .push((s.step, s.value));
    }
    let vals = |m: &str| -> Vec<f64> {
        by_metric
            .get(m)
            .map(|v| v.iter().map(|x| x.1).collect())
            .unwrap_or_default()
    };
    let mut out = Summary {
        schema: SUMMARY_SCHEMA.into(),
        ..Default::default()
    };

    let t = &mut out.training;
    t.loss = Dist::of(&vals("train.loss"));
    t.steps = by_metric
        .get("train.loss")
        .map(|v| {
            v.iter()
                .filter_map(|x| x.0)
                .collect::<std::collections::BTreeSet<_>>()
                .len() as u64
        })
        .unwrap_or(0);
    t.grad_norm = Dist::of(&vals("optim.grad_norm.global"));
    let clipped = vals("optim.clipped");
    t.clipped_frac =
        (!clipped.is_empty()).then(|| clipped.iter().sum::<f64>() / clipped.len() as f64);
    let wall = vals("step.wall");
    t.step_wall_ms = Dist::of(&wall);
    let wall_sum: f64 = wall.iter().filter(|v| v.is_finite()).sum();
    if wall_sum > 0.0 {
        let mut shares: Vec<(String, f64)> = HOST_SPANS
            .iter()
            .filter_map(|m| {
                let s: f64 = vals(m).iter().filter(|v| v.is_finite()).sum();
                (s > 0.0).then(|| ((*m).to_owned(), s / wall_sum))
            })
            .collect();
        shares.sort_by(|a, b| b.1.total_cmp(&a.1));
        t.host_span_share = shares;
        let un: f64 = vals("step.unattributed")
            .iter()
            .filter(|v| v.is_finite())
            .sum();
        t.unattributed_share = by_metric
            .contains_key("step.unattributed")
            .then(|| un / wall_sum);
    }
    t.val_loss = Dist::of(&vals("val.loss"));
    t.generalization_gap_last = t.val_loss.last.zip(t.loss.last).map(|(v, l)| v - l);
    for (m, label) in [
        ("train.loss", "train.loss"),
        ("optim.grad_norm.global", "optim.grad_norm.global"),
        ("step.wall", "step.wall"),
        ("val.loss", "val.loss"),
    ] {
        if !by_metric.contains_key(m) {
            out.missing.insert(label.into(), "no events".into());
        }
    }

    let mut target: BTreeMap<String, Vec<(u32, f64)>> = BTreeMap::new();
    for (kind, body) in records {
        let ok = match kind {
            GEN_SCHEMA => serde_json::from_value::<GenSummary>(body.clone()).map(|g| {
                let c = out.gp.entry(g.context.clone()).or_default();
                c.generations += 1;
                c.last_unique_genotype_frac = g.unique_genotype_frac;
                c.last_fitness_profile_unique_frac = g.behavior.fitness_profile_unique_frac;
                c.last_op_entropy_norm = g.op_entropy_norm;
                c.best_fitness = g.best_fitness;
                c.gens_since_improved = g.gens_since_improved;
                let st = &g.status;
                c.failed +=
                    (st.timeout + st.nonzero_rc + st.parse_fail + st.failed_unknown_cause) as u64;
                c.evaluated += g.pop_size as u64;
            }),
            HWM_CONSISTENCY_SCHEMA => {
                serde_json::from_value::<HwmConsistency>(body.clone()).map(|h| {
                    out.evolution.hwm_stale_files += h.stale.len() as u64;
                    if h.count_matches == Some(false) || !h.stale.is_empty() {
                        out.evolution.hwm_rounds_inconsistent.push(h.round);
                    }
                })
            }
            RETENTION_SCHEMA => serde_json::from_value::<Retention>(body.clone()).map(|r| {
                for (k, v) in r.counts {
                    *out.evolution.retention_totals.entry(k).or_default() += v;
                }
            }),
            crate::flywheel::ROUND_SCHEMA => serde_json::from_value::<RoundRecord>(body.clone())
                .map(|r| {
                    out.evolution.rounds.push(r.round);
                    for (k, v) in r.target_deltas {
                        if !k.starts_with("count.") {
                            target.entry(k).or_default().push((r.round, v));
                        }
                    }
                }),
            crate::alarm::ALARM_SCHEMA => serde_json::from_value::<Alarm>(body.clone()).map(|a| {
                *out.alarms.by_code.entry(a.code.clone()).or_default() += 1;
                if a.severity == Severity::FailClosed {
                    out.alarms.fail_closed.push(format!(
                        "{} at {}: {}",
                        a.code,
                        a.at,
                        a.reasons.join("; ")
                    ));
                }
            }),
            "jope.telemetry.dope-guard/v1" => {
                let d = &mut out.dope;
                d.guard_records += 1;
                if let Some(eq) = body.get("hashes_equal").and_then(Value::as_bool) {
                    d.compared += 1;
                    d.all_equal = Some(d.all_equal.unwrap_or(true) && eq);
                }
                if let Some(s) = body.get("composite_sha256").and_then(Value::as_str) {
                    d.last_composite_sha256 = Some(s.to_owned());
                }
                for v in body
                    .get("violations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(s) = v.as_str() {
                        d.violations.push(s.to_owned());
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        };
        if ok.is_err() {
            out.unparsed_records += 1;
        }
    }
    out.evolution.rounds.sort_unstable();
    out.evolution.rounds.dedup();
    out.evolution.hwm_rounds_inconsistent.sort_unstable();
    for (k, mut v) in target {
        v.sort_by_key(|x| x.0);
        let series: Vec<f64> = v.into_iter().map(|x| x.1).collect();
        if series.len() >= 3 {
            let (f, l) = (series[0].abs(), series[series.len() - 1].abs());
            if f > 0.0 && l < 0.1 * f {
                out.evolution.diminishing.push(k.clone());
            }
        }
        out.evolution.target_deltas.insert(k, series);
    }
    if out.dope.guard_records == 0 {
        out.missing.insert(
            "dope.freeze".into(),
            "no guard records (no fused trainer run)".into(),
        );
    }
    out
}

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |x| format!("{x:.4}"))
}

impl Summary {
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        let t = &self.training;
        let _ = writeln!(s, "== training");
        let _ = writeln!(
            s,
            "steps {}  loss first/last/min {}/{}/{}  non-finite {}",
            t.steps,
            opt(t.loss.first),
            opt(t.loss.last),
            opt(t.loss.min),
            t.loss.nonfinite
        );
        let _ = writeln!(
            s,
            "grad norm p50/p95/max {}/{}/{}  clipped {}",
            opt(t.grad_norm.p50),
            opt(t.grad_norm.p95),
            opt(t.grad_norm.max),
            opt(t.clipped_frac)
        );
        let _ = writeln!(
            s,
            "step wall ms p50/p95 {}/{}",
            opt(t.step_wall_ms.p50),
            opt(t.step_wall_ms.p95)
        );
        for (span, share) in &t.host_span_share {
            let _ = writeln!(s, "  {span:<16} {:>5.1}%", share * 100.0);
        }
        if let Some(u) = t.unattributed_share {
            let _ = writeln!(s, "  {:<16} {:>5.1}%", "unattributed", u * 100.0);
        }
        let _ = writeln!(
            s,
            "val loss last/min {}/{}  gap {}",
            opt(t.val_loss.last),
            opt(t.val_loss.min),
            opt(t.generalization_gap_last)
        );
        let d = &self.dope;
        let _ = writeln!(s, "== DOPE freeze");
        let _ = writeln!(
            s,
            "guard records {}  compared {}  all equal {}  violations {}",
            d.guard_records,
            d.compared,
            d.all_equal.map_or("-".into(), |b| b.to_string()),
            d.violations.len()
        );
        let _ = writeln!(s, "== GP ({} contexts)", self.gp.len());
        for (ctx, c) in &self.gp {
            let short: String = ctx.chars().take(16).collect();
            let _ = writeln!(
                s,
                "{short:<16} gens {:>3}  unique {}  profile-unique {}  op-entropy {}  best {}  stale {}  failed {}/{}",
                c.generations,
                opt(c.last_unique_genotype_frac),
                opt(c.last_fitness_profile_unique_frac),
                opt(c.last_op_entropy_norm),
                opt(c.best_fitness),
                c.gens_since_improved,
                c.failed,
                c.evaluated
            );
        }
        let e = &self.evolution;
        let _ = writeln!(s, "== evolution (rounds {:?})", e.rounds);
        for (k, v) in &e.target_deltas {
            let flag = if e.diminishing.contains(k) {
                "  DIMINISHING"
            } else {
                ""
            };
            let vals: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
            let _ = writeln!(s, "{k}: {}{flag}", vals.join(" -> "));
        }
        if !e.hwm_rounds_inconsistent.is_empty() {
            let _ = writeln!(
                s,
                "HWM inconsistent in rounds {:?} ({} stale files)",
                e.hwm_rounds_inconsistent, e.hwm_stale_files
            );
        }
        if !e.retention_totals.is_empty() {
            let r: Vec<String> = e
                .retention_totals
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            let _ = writeln!(s, "retention {}", r.join(" "));
        }
        let _ = writeln!(s, "== alarms {:?}", self.alarms.by_code);
        for f in &self.alarms.fail_closed {
            let _ = writeln!(s, "FAIL-CLOSED {f}");
        }
        if !self.missing.is_empty() {
            let _ = writeln!(s, "== missing");
            for (k, why) in &self.missing {
                let _ = writeln!(s, "{k}: {why}");
            }
        }
        if self.unparsed_records > 0 {
            let _ = writeln!(s, "unparsed records: {}", self.unparsed_records);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn summary_reports_spans_diminishing_deltas_and_missing() {
        let sc = [
            ScalarIn {
                metric: "train.loss",
                group: None,
                step: Some(1),
                value: 2.0,
            },
            ScalarIn {
                metric: "train.loss",
                group: None,
                step: Some(2),
                value: f64::NAN,
            },
            ScalarIn {
                metric: "step.wall",
                group: None,
                step: Some(1),
                value: 10.0,
            },
            ScalarIn {
                metric: "step.span.fwd",
                group: None,
                step: Some(1),
                value: 6.0,
            },
            ScalarIn {
                metric: "step.unattributed",
                group: None,
                step: Some(1),
                value: 4.0,
            },
        ];
        let rounds: Vec<Value> = [(1, -0.179), (2, -0.070), (3, -0.003)]
            .iter()
            .map(|(r, d)| {
                json!({"schema": crate::flywheel::ROUND_SCHEMA, "round": r, "generation": r,
                       "target_deltas": {"mean_strat300_evo_fitness_delta": d, "count.x": 1.0},
                       "arm_contrasts": {}})
            })
            .collect();
        let guard = json!({"hashes_equal": true, "composite_sha256": "abc", "violations": []});
        let mut recs: Vec<(&str, &Value)> = rounds
            .iter()
            .map(|r| (crate::flywheel::ROUND_SCHEMA, r))
            .collect();
        recs.push(("jope.telemetry.dope-guard/v1", &guard));
        let s = summarize(sc, recs);
        assert_eq!(s.training.steps, 2);
        assert_eq!(s.training.loss.nonfinite, 1);
        assert_eq!(
            s.training.host_span_share,
            vec![("step.span.fwd".to_owned(), 0.6)]
        );
        assert_eq!(s.training.unattributed_share, Some(0.4));
        assert_eq!(s.evolution.rounds, vec![1, 2, 3]);
        assert_eq!(
            s.evolution.diminishing,
            vec!["mean_strat300_evo_fitness_delta".to_owned()]
        );
        assert_eq!(s.dope.all_equal, Some(true));
        assert!(s.missing.contains_key("val.loss") && !s.missing.contains_key("dope.freeze"));
        let text = s.render_text();
        assert!(text.contains("DIMINISHING") && text.contains("step.span.fwd"));
    }
}
