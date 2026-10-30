//! `jtel fleet`: per-host and fleet roll-up of run directories (GB/s, tok/s, host utilization).
//!
//! Layout: `ROOT/<host>/<...>/<run_dir>/` with `manifest.json` and `events.jsonl` in each run dir.
//! The host is the first path component under ROOT. Per host: median / p95 / mean over every
//! finite event value of each metric across that host's runs, plus unavailable counts by reason
//! (an unavailable value is never counted as 0). Fleet: the sum of per-host medians for rate
//! metrics, which assumes the hosts ran concurrently. tok/s is always the labeled pair, Rust-only
//! first; both lines print even when a host has no values for them.

use crate::gp::quantile_sorted;
use crate::summary::{
    rust_timing, TOK_S_ALL_LANGUAGES, TOK_S_LABELS, TOK_S_LEGACY, TOK_S_RUST,
    TOK_S_RUST_TIMED_ALONE,
};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const FLEET_SCHEMA: &str = "jope.telemetry.fleet/v1";

/// Reported metrics in print order; the bool is whether it is a rate summed across the fleet.
const METRICS: &[(&str, bool)] = &[
    ("io.batch_gb_s", true),
    ("io.h2d_gb_s", true),
    ("io.batch_bytes_per_s", true),
    ("io.local.read_mib_s", false),
    (TOK_S_RUST, true),
    (TOK_S_ALL_LANGUAGES, true),
    (TOK_S_LEGACY, true),
    ("throughput.samples_per_s", true),
    ("io.samples_per_s", true),
    ("host.gpu.util", false),
    ("host.gpu.memory", false),
    ("host.cpu.node_util", false),
    ("host.cpu.process_util", false),
];

#[derive(Clone, Debug, Default, Serialize)]
pub struct MetricStats {
    pub n: u64,
    pub median: Option<f64>,
    pub p95: Option<f64>,
    pub mean: Option<f64>,
    /// Unavailable events by reason name.
    pub unavailable: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct HostReport {
    pub runs: u64,
    pub status: BTreeMap<String, u64>,
    pub event_drops: u64,
    pub sink_errors: u64,
    /// Torn or unparsable `events.jsonl` lines that were skipped.
    pub skipped_lines: u64,
    pub metrics: BTreeMap<String, MetricStats>,
    /// Steps whose Rust-only tok/s was timed on the Rust output alone, or on the shared span.
    pub rust_timed_alone_steps: u64,
    pub rust_shared_span_steps: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FleetRate {
    pub sum_of_host_medians: f64,
    pub hosts: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FleetReport {
    pub schema: &'static str,
    pub hosts: BTreeMap<String, HostReport>,
    pub fleet: BTreeMap<String, FleetRate>,
}

#[derive(Default)]
struct HostValues {
    report: HostReport,
    values: BTreeMap<&'static str, Vec<f64>>,
}

/// Run dirs under `root` (any directory holding both files), with their host, in path order.
fn runs_under(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .with_context(|| format!("read {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries.iter().rev() {
            // Not followed through symlinks, so a link cycle can't loop the walk.
            if e.file_type()?.is_dir() {
                stack.push(e.path());
            }
        }
        if dir.join("manifest.json").is_file() && dir.join("events.jsonl").is_file() {
            let rel = dir.strip_prefix(root).unwrap_or(&dir);
            let mut parts = rel.components();
            let host = match (parts.next(), parts.next()) {
                (Some(first), Some(_)) => first.as_os_str().to_string_lossy().into_owned(),
                _ => "(root)".to_owned(),
            };
            out.push((host, dir));
        }
    }
    Ok(out)
}

fn read_run(run: &Path, host: &mut HostValues) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(run.join("manifest.json"))?)
        .with_context(|| format!("{}/manifest.json", run.display()))?;
    let names: BTreeMap<u64, &str> = manifest
        .get("metrics")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| Some((m.get("id")?.as_u64()?, m.get("name")?.as_str()?)))
        .collect();
    let reasons: BTreeMap<u64, &str> = manifest
        .get("availability_reasons")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(k, v)| Some((k.parse().ok()?, v.as_str()?)))
        .collect();
    let r = &mut host.report;
    r.runs += 1;
    let status = manifest.get("status").map_or("None".to_owned(), |s| {
        s.as_str().map_or_else(|| s.to_string(), str::to_owned)
    });
    *r.status.entry(status).or_default() += 1;
    let count = |k: &str| manifest.get(k).and_then(Value::as_u64).unwrap_or(0);
    r.event_drops += count("event_drops");
    r.sink_errors += count("sink_errors");
    let file = fs::File::open(run.join("events.jsonl"))?;
    for line in BufReader::new(file).lines() {
        let Ok(row) = serde_json::from_str::<Value>(&line?) else {
            host.report.skipped_lines += 1; // a torn final line from an interrupted writer
            continue;
        };
        if row.get("kind").and_then(Value::as_str) != Some("event") {
            continue;
        }
        let ev = row.get("event").unwrap_or(&Value::Null);
        let Some(name) = ev
            .get("metric_id")
            .and_then(Value::as_u64)
            .and_then(|id| names.get(&id))
        else {
            continue;
        };
        let value = ev.get("value").and_then(Value::as_f64);
        if *name == TOK_S_RUST_TIMED_ALONE {
            match value {
                Some(1.0) => host.report.rust_timed_alone_steps += 1,
                Some(0.0) => host.report.rust_shared_span_steps += 1,
                _ => {}
            }
            continue;
        }
        let Some(&(metric, _)) = METRICS.iter().find(|(m, _)| m == name) else {
            continue;
        };
        let reason = ev
            .get("availability_reason_id")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if reason != 0 {
            let why = reasons
                .get(&reason)
                .map_or_else(|| format!("reason {reason}"), |s| (*s).to_owned());
            let stats = host.report.metrics.entry(metric.to_owned()).or_default();
            *stats.unavailable.entry(why).or_default() += 1;
        } else if let Some(v) = value.filter(|v| v.is_finite()) {
            host.values.entry(metric).or_default().push(v);
        }
    }
    Ok(())
}

pub fn roll_up(root: &Path) -> Result<FleetReport> {
    let mut hosts: BTreeMap<String, HostValues> = BTreeMap::new();
    for (host, run) in runs_under(root)? {
        read_run(&run, hosts.entry(host).or_default())?;
    }
    let mut report = FleetReport {
        schema: FLEET_SCHEMA,
        hosts: BTreeMap::new(),
        fleet: BTreeMap::new(),
    };
    for (host, mut h) in hosts {
        for (metric, mut vals) in h.values {
            vals.sort_by(f64::total_cmp);
            let stats = h.report.metrics.entry(metric.to_owned()).or_default();
            stats.n = vals.len() as u64;
            stats.median = quantile_sorted(&vals, 0.5);
            stats.p95 = quantile_sorted(&vals, 0.95);
            stats.mean = Some(vals.iter().sum::<f64>() / vals.len() as f64);
        }
        report.hosts.insert(host, h.report);
    }
    for &(metric, _) in METRICS.iter().filter(|m| m.1) {
        let meds: Vec<(&String, f64)> = report
            .hosts
            .iter()
            .filter_map(|(host, r)| Some((host, r.metrics.get(metric)?.median?)))
            .collect();
        if !meds.is_empty() {
            report.fleet.insert(
                metric.to_owned(),
                FleetRate {
                    sum_of_host_medians: meds.iter().map(|m| m.1).sum(),
                    hosts: meds.into_iter().map(|m| m.0.clone()).collect(),
                },
            );
        }
    }
    Ok(report)
}

fn g4(v: f64) -> String {
    // Four significant digits, like Python's `:.4g` without the exponent switch for big rates.
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let digits = 3 - v.abs().log10().floor() as i32;
    if digits > 0 {
        format!("{v:.*}", digits as usize)
    } else {
        let scale = 10f64.powi(-digits);
        format!("{}", (v / scale).round() * scale)
    }
}

fn label(metric: &str) -> String {
    TOK_S_LABELS.iter().find(|(m, _)| *m == metric).map_or_else(
        || {
            if metric == TOK_S_LEGACY {
                "tok/s unlabeled (legacy)".to_owned()
            } else {
                metric.to_owned()
            }
        },
        |(_, l)| (*l).to_owned(),
    )
}

impl FleetReport {
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        for (host, r) in &self.hosts {
            let _ = writeln!(
                s,
                "== {host}: {} runs {:?}  event_drops={} sink_errors={} skipped_lines={}",
                r.runs, r.status, r.event_drops, r.sink_errors, r.skipped_lines
            );
            for &(metric, _) in METRICS {
                let stats = r.metrics.get(metric);
                let pair = TOK_S_LABELS.iter().any(|(m, _)| *m == metric);
                let name = label(metric);
                match stats {
                    Some(e) if e.median.is_some() => {
                        let _ = write!(
                            s,
                            "  {name:<30} n={:<6} median={} p95={} mean={}",
                            e.n,
                            g4(e.median.unwrap_or(f64::NAN)),
                            g4(e.p95.unwrap_or(f64::NAN)),
                            g4(e.mean.unwrap_or(f64::NAN))
                        );
                        if metric == TOK_S_RUST {
                            let timing =
                                rust_timing(r.rust_timed_alone_steps, r.rust_shared_span_steps);
                            let _ = write!(s, "  ({timing})");
                        }
                        let _ = writeln!(s);
                    }
                    None if pair => {
                        let _ = writeln!(s, "  {name:<30} no values");
                    }
                    _ => {}
                }
                for (reason, n) in stats.map(|e| &e.unavailable).into_iter().flatten() {
                    let _ = writeln!(s, "  {name:<30} unavailable x{n}: {reason}");
                }
            }
        }
        let _ = writeln!(
            s,
            "== fleet (sum of per-host medians; assumes concurrent hosts)"
        );
        for &(metric, _) in METRICS.iter().filter(|m| m.1) {
            let pair = TOK_S_LABELS.iter().any(|(m, _)| *m == metric);
            match self.fleet.get(metric) {
                Some(f) => {
                    let _ = writeln!(
                        s,
                        "  {:<30} {}  over {}",
                        label(metric),
                        g4(f.sum_of_host_medians),
                        f.hosts.join(", ")
                    );
                }
                None if pair => {
                    let _ = writeln!(s, "  {:<30} no values", label(metric));
                }
                None => {}
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run(dir: &Path, events: &[Value], torn: bool) {
        fs::create_dir_all(dir).unwrap();
        let manifest = json!({
            "status": "complete", "event_drops": 0, "sink_errors": 0,
            "availability_reasons": {"10": "no_text_token_path"},
            "metrics": [
                {"id": 0x012a, "name": TOK_S_RUST},
                {"id": 0x012b, "name": TOK_S_ALL_LANGUAGES},
                {"id": 0x012c, "name": TOK_S_RUST_TIMED_ALONE},
                {"id": 0x041b, "name": "io.batch_gb_s"},
            ],
        });
        fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        let mut text: String = events
            .iter()
            .map(|e| json!({"kind": "event", "event": e}).to_string() + "\n")
            .collect();
        if torn {
            text.push_str("{\"kind\":\"ev");
        }
        fs::write(dir.join("events.jsonl"), text).unwrap();
    }

    fn ev(id: u16, step: u64, value: Option<f64>, reason: u16) -> Value {
        json!({"metric_id": id, "group": 0, "step": step, "value": value,
               "availability_reason_id": reason})
    }

    #[test]
    fn fleet_reports_the_labeled_tok_s_pair_per_host_and_in_total() {
        let root = tempfile::tempdir().unwrap();
        let mut a = Vec::new();
        for step in 1..=3 {
            let s = step as f64;
            a.push(ev(0x012a, step, Some(100.0 * s), 0));
            a.push(ev(0x012b, step, Some(1000.0 * s), 0));
            a.push(ev(0x012c, step, Some(0.0), 0));
            a.push(ev(0x041b, step, Some(2.0), 0));
        }
        run(&root.path().join("xbabe1/fused/run-a"), &a, true);
        let b = [
            ev(0x012a, 0, None, 10),
            ev(0x012b, 0, None, 10),
            ev(0x041b, 1, Some(1.0), 0),
        ];
        run(&root.path().join("xbabe3/run-b"), &b, false);
        let r = roll_up(root.path()).unwrap();
        let x1 = &r.hosts["xbabe1"];
        assert_eq!((x1.runs, x1.skipped_lines), (1, 1));
        assert_eq!(x1.metrics[TOK_S_RUST].median, Some(200.0));
        assert_eq!(x1.metrics[TOK_S_ALL_LANGUAGES].p95, Some(2900.0));
        assert_eq!(
            (x1.rust_timed_alone_steps, x1.rust_shared_span_steps),
            (0, 3)
        );
        let x3 = &r.hosts["xbabe3"];
        assert_eq!(x3.metrics[TOK_S_RUST].n, 0);
        assert_eq!(x3.metrics[TOK_S_RUST].unavailable["no_text_token_path"], 1);
        assert_eq!(r.fleet["io.batch_gb_s"].sum_of_host_medians, 3.0);
        assert_eq!(r.fleet[TOK_S_RUST].hosts, vec!["xbabe1".to_owned()]);
        let text = r.render_text();
        let line = |parts: &[&str]| {
            text.lines()
                .filter(|l| parts.iter().all(|p| l.contains(p)))
                .count()
        };
        let rust = "Rust-only tok/s (headline)";
        let shared = "(timed on the shared all-languages span)";
        assert_eq!(line(&[rust, "n=3", "median=200.0", shared]), 1);
        assert_eq!(line(&[rust, "unavailable x1: no_text_token_path"]), 1);
        assert_eq!(line(&["all-languages tok/s", "2000  over xbabe1"]), 1);
        assert!(!text.contains("legacy"), "no legacy values, no legacy line");
    }

    #[test]
    fn both_tok_s_lines_print_without_any_values() {
        let root = tempfile::tempdir().unwrap();
        run(
            &root.path().join("xbabe2/r"),
            &[ev(0x041b, 1, Some(1.5), 0)],
            false,
        );
        let text = roll_up(root.path()).unwrap().render_text();
        for (_, label) in TOK_S_LABELS {
            let both = |l: &&str| l.contains(label) && l.ends_with("no values");
            assert_eq!(text.lines().filter(both).count(), 2, "host and fleet");
        }
        assert_eq!(g4(5_032_491.14), "5032000");
        assert_eq!(g4(34_059.2), "34060");
        assert_eq!(g4(1.50432), "1.504");
        assert_eq!(g4(0.000123456), "0.0001235");
    }
}
