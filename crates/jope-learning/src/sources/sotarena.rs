//! SOTArena `results/<version>/scores.csv` → `jope.score/v1` rows (benchmark scores, not HWM).

use crate::canon::{canonical_sha256, sha256_hex};
use crate::civil::parse_rfc3339;
use crate::metric::Metric;
use crate::record::*;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Constant identifying the columns.
pub const COLUMNS: [&str; 13] = [
    "track_id",
    "dataset_id",
    "method_id",
    "protocol",
    "metric",
    "value",
    "status",
    "benchmark_valid",
    "source_kind",
    "source_revision",
    "source_sha256",
    "source_uri",
    "notes",
];

#[derive(Serialize)]
struct CutIdentity<'a> {
    track: &'a str,
    protocol: &'a str,
}

/// Performs the cut ID operation and returns an error if it cannot be completed.
pub fn cut_id(track: &str, protocol: &str) -> Result<String> {
    canonical_sha256(&CutIdentity { track, protocol })
}

#[derive(Clone, Debug, Serialize, Default)]
/// SOTArena version metadata and the `scores.csv` file found for that version.
pub struct TrackScan {
    /// Version recorded for this track scan.
    pub version: String,
    /// Scores csv recorded for this track scan.
    pub scores_csv: String,
    /// SHA-256 digest identifying the associated scores.
    pub scores_sha256: String,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Mapped recorded for this track scan.
    pub mapped: u64,
    /// Collection of rejects values carried by this track scan.
    pub rejects: Vec<(u64, String)>,
}

/// One `scores.csv` (comma-separated, no quoting in the published files; a quoted field is
/// rejected rather than mis-split).
pub fn parse_scores(
    bytes: &[u8],
    version: &str,
    git_sha: &str,
    fallback_ts: u64,
) -> Result<(Vec<ScoreRecord>, TrackScan)> {
    let text = std::str::from_utf8(bytes).context("scores.csv is not UTF-8")?;
    let mut lines = text.lines();
    let header = lines.next().context("empty scores.csv")?;
    let cols: Vec<&str> = header.split(',').collect();
    ensure!(cols == COLUMNS, "unexpected scores.csv header: {header}");
    let mut scan = TrackScan {
        version: version.into(),
        scores_sha256: sha256_hex(bytes),
        ..Default::default()
    };
    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        let n = i as u64 + 2;
        if line.trim().is_empty() {
            continue;
        }
        scan.rows += 1;
        match parse_line(line, version, git_sha, fallback_ts) {
            Ok(r) => {
                scan.mapped += 1;
                out.push(r);
            }
            Err(e) => scan.rejects.push((n, format!("{e:#}"))),
        }
    }
    Ok((out, scan))
}

fn parse_line(line: &str, version: &str, git_sha: &str, fallback_ts: u64) -> Result<ScoreRecord> {
    if line.contains('"') {
        bail!("quoted field");
    }
    let f: Vec<&str> = line.splitn(COLUMNS.len(), ',').collect();
    ensure!(f.len() == COLUMNS.len(), "{} fields", f.len());
    let metric = Metric::parse(f[4]).with_context(|| format!("unknown metric {}", f[4]))?;
    let value: f64 = f[5].parse().with_context(|| format!("value {}", f[5]))?;
    let benchmark_valid = match f[7].to_ascii_lowercase().as_str() {
        "true" | "1" => true,
        "false" | "0" => false,
        other => bail!("benchmark_valid {other}"),
    };
    ensure!(crate::canon::is_hex64(f[10]), "source_sha256 {}", f[10]);
    let opt = |s: &str| (!s.is_empty()).then(|| s.to_string());
    let mut r = ScoreRecord {
        schema: SCORE_SCHEMA.into(),
        dataset_hash: f[1].into(),
        cut_id: cut_id(f[0], f[3])?,
        method_id: f[2].into(),
        evaluator_sha256: UNBOUND.into(),
        protocol: f[3].into(),
        metric,
        value,
        direction: metric.direction(),
        cost: None,
        benchmark_valid,
        receipt_sha256: f[10].into(),
        run_id: f[11].into(),
        git_sha: git_sha.into(),
        ts_unix: parse_rfc3339(f[9]).unwrap_or(fallback_ts),
        source: ScoreSource {
            kind: "sotarena".into(),
            track: opt(f[0]),
            version: Some(version.into()),
            status: opt(f[6]),
            source_kind: opt(f[8]),
            source_revision: opt(f[9]),
            source_uri: opt(f[11]),
            notes: opt(f[12]),
        },
        row_sha256: String::new(),
    };
    r.row_sha256 = r.compute_row_sha256()?;
    r.validate()?;
    Ok(r)
}

/// Every `<results>/<version>/scores.csv`.
pub fn track_versions(results: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(results).with_context(|| format!("read {}", results.display()))?
    {
        let entry = entry?;
        let p = entry.path().join("scores.csv");
        if entry.file_type()?.is_dir() && p.exists() {
            out.push((entry.file_name().to_string_lossy().into_owned(), p));
        }
    }
    out.sort();
    ensure!(
        !out.is_empty(),
        "no <version>/scores.csv under {}",
        results.display()
    );
    Ok(out)
}

/// `git rev-parse HEAD` of the results tree when it is inside a git checkout.
pub fn git_sha_of(path: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (s.len() == 40).then_some(s)
}

/// Performs the scan source operation and returns an error if it cannot be completed.
pub fn scan_source(results: &Path, git_sha: &str, scans: &BTreeMap<String, TrackScan>) -> Value {
    crate::hwm::source_value(
        "sotarena",
        results,
        json!({ "git_sha": git_sha, "tracks": scans }),
    )
}
