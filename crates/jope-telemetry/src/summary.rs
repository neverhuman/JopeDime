//! The TRAIN+GP part of `jtel summary`: a machine-readable report plus a text rendering.
//!
//! Input is normalized scalars (metric name, group, step, value) and typed records (kind plus
//! JSON body), so it works on a live run's events, on `jtel import` output, or on both. Missing
//! data is reported as missing, with its reason when one was recorded; nothing is filled in.

use crate::alarm::{Alarm, Severity};
use crate::flywheel::{EloSnapshot, RoundRecord, TaskType, ELO_SCHEMA};
use crate::gp::{GenSummary, GEN_SCHEMA};
use crate::hooks::io::REASONING_KIND;
use crate::import::{HwmConsistency, Retention, HWM_CONSISTENCY_SCHEMA, RETENTION_SCHEMA};
use serde_json::Value;
use std::collections::BTreeMap;

mod types;
use std::fmt::Write as _;
pub use types::*;

fn elo_summary(snap: &EloSnapshot, seen: u32) -> EloSummary {
    let mut tasks = BTreeMap::new();
    for (task, ratings) in &snap.ratings {
        let mut ranked: Vec<(String, f64)> =
            ratings.iter().map(|(k, r)| (k.clone(), r.elo)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let flywheel = ranked
            .iter()
            .enumerate()
            .find(|(_, (n, _))| is_flywheel_contender(n))
            .map(|(i, (n, e))| (n.clone(), *e, i as u32 + 1));
        let best_method = ranked
            .iter()
            .find(|(n, _)| !is_flywheel_contender(n))
            .cloned();
        let delta = flywheel
            .as_ref()
            .zip(best_method.as_ref())
            .map(|(f, m)| f.1 - m.1);
        tasks.insert(
            *task,
            EloTaskSummary {
                contenders: ranked.len() as u32,
                flywheel,
                best_method,
                delta_vs_best_method: delta,
                top: ranked.into_iter().take(5).collect(),
            },
        );
    }
    let mut caveats = Vec::new();
    if let Some(a) = &snap.availability {
        caveats.push(a.clone());
    }
    if snap.panel_complete == Some(false) {
        caveats.push("incomplete_panel".into());
    }
    if snap.solver_version.is_none() {
        caveats.push("solver_version_unrecorded".into());
    }
    if snap
        .family_source
        .as_deref()
        .is_none_or(|f| f != "authoritative_roster")
    {
        caveats.push("family_source_not_authoritative".into());
    }
    EloSummary {
        round: snap.round,
        fit_digest: snap.fit_digest.clone(),
        snapshots_seen: seen,
        tasks,
        caveats,
    }
}

/// The part of a `reasoning-tok-s` record the summary reads; the numbers come from its events.
#[derive(serde::Deserialize)]
struct TokenizerIn {
    tokenizer: String,
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

    let k = &mut out.tok_s;
    k.rust_only = Dist::of(&vals(TOK_S_RUST));
    k.all_languages = Dist::of(&vals(TOK_S_ALL_LANGUAGES));
    for v in vals(TOK_S_RUST_TIMED_ALONE) {
        if v == 1.0 {
            k.rust_timed_alone_steps += 1;
        } else if v == 0.0 {
            k.rust_shared_span_steps += 1;
        }
    }
    let legacy = vals(TOK_S_LEGACY);
    k.legacy_unlabeled = legacy
        .iter()
        .any(|v| v.is_finite())
        .then(|| Dist::of(&legacy));
    let last = |m: &str| vals(m).into_iter().rev().find(|v| v.is_finite());
    k.reasoning_tok_per_s = last(TOK_S_REASONING);
    k.reasoning_tokens = last(REASONING_TOKENS);
    k.reasoning_wall_s = last(REASONING_WALL_S);
    let reasoning = k.reasoning_tok_per_s;
    for (metric, value) in [TOK_S_RUST, TOK_S_ALL_LANGUAGES, TOK_S_REASONING]
        .into_iter()
        .zip([
            out.tok_s.rust_only.p50,
            out.tok_s.all_languages.p50,
            reasoning,
        ])
    {
        if value.is_none() {
            out.missing.insert(
                metric.into(),
                "no values (unavailable or not emitted)".into(),
            );
        }
    }

    let mut target: BTreeMap<String, Vec<(u32, f64)>> = BTreeMap::new();
    let mut latest_elo: Option<EloSnapshot> = None;
    let mut elo_seen = 0u32;
    for (kind, body) in records {
        let ok = match kind {
            REASONING_KIND => serde_json::from_value::<TokenizerIn>(body.clone())
                .map(|r| out.tok_s.reasoning_tokenizer = Some(r.tokenizer)),
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
            ELO_SCHEMA => serde_json::from_value::<EloSnapshot>(body.clone()).map(|e| {
                elo_seen += 1;
                // The trend lives inside one fit, so only the latest snapshot is summarized.
                if latest_elo.as_ref().is_none_or(|l| e.round >= l.round) {
                    latest_elo = Some(e);
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
    out.elo = latest_elo.as_ref().map(|e| elo_summary(e, elo_seen));
    if out.elo.is_none() {
        out.missing
            .insert("fly.elo".into(), "no Elo snapshot records".into());
    }
    if out.dope.guard_records == 0 {
        out.missing.insert(
            "dope.freeze".into(),
            "no guard records (no fused trainer run)".into(),
        );
    }
    out
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
        let k = &self.tok_s;
        let _ = writeln!(s, "== tok/s");
        let timing = format!(
            "  ({})",
            rust_timing(k.rust_timed_alone_steps, k.rust_shared_span_steps)
        );
        for ((_, label), (d, note)) in TOK_S_LABELS
            .iter()
            .zip([(&k.rust_only, timing.as_str()), (&k.all_languages, "")])
        {
            let state = if d.p50.is_none() { "  unavailable" } else { "" };
            let _ = writeln!(
                s,
                "{label:<26} p50/p95/last {}/{}/{}  n {}{state}{note}",
                opt(d.p50),
                opt(d.p95),
                opt(d.last),
                d.n - d.nonfinite
            );
        }
        if let Some(d) = &k.legacy_unlabeled {
            let _ = writeln!(
                s,
                "{:<26} p50 {}  n {}  (legacy {TOK_S_LEGACY}; neither of the two above)",
                "unlabeled tok/s",
                opt(d.p50),
                d.n - d.nonfinite
            );
        }
        let _ = match (
            k.reasoning_tok_per_s,
            k.reasoning_tokens,
            k.reasoning_wall_s,
        ) {
            (Some(rate), tokens, wall) => writeln!(
                s,
                "{TOK_S_REASONING_LABEL:<26} {}  = {} tokens / {} s wall-clock, run total  \
                 (tokenizer {})",
                opt(Some(rate)),
                opt(tokens),
                opt(wall),
                k.reasoning_tokenizer.as_deref().unwrap_or("unrecorded")
            ),
            (None, ..) => writeln!(
                s,
                "{TOK_S_REASONING_LABEL:<26} -  unavailable  (run total / wall-clock; not measured)"
            ),
        };
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
        if let Some(e) = &self.elo {
            let short: String = e.fit_digest.chars().take(12).collect();
            let _ = writeln!(
                s,
                "== Elo vs methods (round {}, fit {short}, {} snapshots)",
                e.round, e.snapshots_seen
            );
            for (task, t) in &e.tasks {
                let fly = t.flywheel.as_ref().map_or("-".to_owned(), |(n, v, r)| {
                    format!("{n} {v:.1} (rank {r}/{})", t.contenders)
                });
                let best = t
                    .best_method
                    .as_ref()
                    .map_or("-".to_owned(), |(n, v)| format!("{n} {v:.1}"));
                let _ = writeln!(
                    s,
                    "{task:?}: flywheel {fly}; best method {best}; delta {}",
                    opt(t.delta_vs_best_method)
                );
            }
            if !e.caveats.is_empty() {
                let _ = writeln!(s, "caveats: {}", e.caveats.join(", "));
            }
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
mod tests;
