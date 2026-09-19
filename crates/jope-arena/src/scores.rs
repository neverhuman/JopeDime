//! The SOTArena `scores.csv` schema (`registry/schemas/score.schema.json` plus the provenance
//! columns every published track carries) and score emission from campaign receipts.
//!
//! Columns, in order: `track_id, dataset_id, method_id, protocol, metric, value, status,
//! benchmark_valid, source_kind, source_revision, source_sha256, source_uri, notes`.
//! `status ∈ {ok, failed, missing, withdrawn}`, `protocol ∈ {blind, public-development,
//! aspfm-generated}`, `benchmark_valid` is written as Python's `True`/`False`.
//!
//! Emission reads the receipts a campaign leaves behind — `sotarena.score-receipt.v1`
//! (`sotarena score`), `JopeDimeDatasetReport.v3` (the Golden42 `score-panel` reports) and
//! `SotarenaPullArtifact.v1` (`sotarena_pull emit` metadata) — and turns each into one row whose
//! `source_sha256` is the receipt file's digest and `source_uri` its path relative to the
//! campaign root, exactly as the published rows were made.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::io::{read_text, sha256_file};
use crate::pyfmt::{py_bool, py_repr};

/// The column order of every published `scores.csv`.
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

/// Accepted `protocol` values.
pub const PROTOCOLS: [&str; 3] = ["blind", "public-development", "aspfm-generated"];
/// Accepted `status` values.
pub const STATUSES: [&str; 4] = ["ok", "failed", "missing", "withdrawn"];

/// One `scores.csv` row. `value` is kept as the text that was written so a re-serialisation is
/// byte-identical; [`ScoreRow::value_f64`] parses it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ScoreRow {
    /// Track id (`tabular-regression-v2`).
    pub track_id: String,
    /// Dataset id (16-hex hash or a named split).
    pub dataset_id: String,
    /// Method id.
    pub method_id: String,
    /// Protocol.
    pub protocol: String,
    /// Metric id.
    pub metric: String,
    /// Metric value as text (empty when not scored).
    pub value: String,
    /// Status.
    pub status: String,
    /// `True` / `False`.
    pub benchmark_valid: String,
    /// Kind of the source document.
    pub source_kind: String,
    /// Revision label of the source (free text, often empty).
    pub source_revision: String,
    /// SHA-256 of the source document.
    pub source_sha256: String,
    /// Path of the source document relative to the benchmark root.
    pub source_uri: String,
    /// Free text.
    pub notes: String,
}

impl ScoreRow {
    /// The numeric value, if any.
    pub fn value_f64(&self) -> Option<f64> {
        if self.value.is_empty() {
            None
        } else {
            self.value.parse().ok()
        }
    }

    /// Whether the row enters the ladder (`status == ok` and a value present).
    pub fn is_rankable(&self) -> bool {
        self.status == "ok" && !self.value.is_empty()
    }

    /// Schema checks a row must pass before publication.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.track_id.is_empty(), "empty track_id");
        ensure!(!self.dataset_id.is_empty(), "empty dataset_id");
        ensure!(!self.method_id.is_empty(), "empty method_id");
        ensure!(
            PROTOCOLS.contains(&self.protocol.as_str()),
            "unknown protocol {}",
            self.protocol
        );
        ensure!(!self.metric.is_empty(), "empty metric");
        ensure!(
            STATUSES.contains(&self.status.as_str()),
            "unknown status {}",
            self.status
        );
        if self.status == "ok" {
            let value = self
                .value_f64()
                .with_context(|| format!("ok row without a numeric value: {}", self.value))?;
            ensure!(value.is_finite(), "non-finite value {}", self.value);
        }
        ensure!(
            matches!(self.benchmark_valid.as_str(), "True" | "False"),
            "benchmark_valid must be True or False, not {}",
            self.benchmark_valid
        );
        ensure!(
            self.source_sha256.is_empty()
                || (self.source_sha256.len() == 64
                    && self.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())),
            "invalid source_sha256 {}",
            self.source_sha256
        );
        Ok(())
    }
}

/// Parse a `scores.csv` text (header must be exactly [`COLUMNS`]).
pub fn parse_scores(text: &str) -> Result<Vec<ScoreRow>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(text.as_bytes());
    let header: Vec<&str> = reader.headers()?.iter().collect();
    ensure!(
        header == COLUMNS,
        "scores.csv header {:?} is not the SOTArena schema",
        header
    );
    let mut rows = Vec::new();
    for (index, record) in reader.deserialize::<ScoreRow>().enumerate() {
        rows.push(record.with_context(|| format!("scores.csv row {}", index + 2))?);
    }
    Ok(rows)
}

/// Read and parse `path` (or `path.gz`).
pub fn read_scores(path: &Path) -> Result<Vec<ScoreRow>> {
    parse_scores(&read_text(path)?)
}

/// Serialise rows as the benchmark writes them (`csv.DictWriter`, `\n` line ends, minimal
/// quoting).
pub fn write_scores(rows: &[ScoreRow]) -> Result<String> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new());
    for row in rows {
        writer.serialize(row)?;
    }
    let bytes = writer.into_inner()?;
    Ok(String::from_utf8(bytes)?)
}

/// What a receipt file turned out to be.
#[derive(Clone, Debug, PartialEq)]
pub struct Receipt {
    /// Schema id of the document.
    pub schema: String,
    /// Dataset id.
    pub dataset_id: String,
    /// Metric id.
    pub metric: String,
    /// Metric value.
    pub value: f64,
    /// `benchmark_valid` as recorded (defaults to `true` for a score receipt).
    pub benchmark_valid: bool,
    /// `protocol` as recorded, when the receipt names one.
    pub protocol: Option<String>,
    /// `source_kind` for the row.
    pub source_kind: &'static str,
}

/// Interpret one JSON receipt.
pub fn parse_receipt(value: &Value, metric: &str) -> Result<Receipt> {
    let schema = value["schema"].as_str().unwrap_or_default().to_owned();
    let number = |v: &Value, what: &str| -> Result<f64> {
        v.as_f64()
            .with_context(|| format!("{what} is not a number"))
    };
    match schema.as_str() {
        "sotarena.score-receipt.v1" => Ok(Receipt {
            dataset_id: value["dataset_id"]
                .as_str()
                .context("score receipt without dataset_id")?
                .to_owned(),
            metric: value["metric"].as_str().unwrap_or(metric).to_owned(),
            value: number(&value["value"], "value")?,
            benchmark_valid: value["benchmark_valid"].as_bool().unwrap_or(true),
            protocol: value["protocol"].as_str().map(str::to_owned),
            source_kind: "score-receipt",
            schema,
        }),
        "JopeDimeDatasetReport.v3" => Ok(Receipt {
            dataset_id: value["dataset_hash"]
                .as_str()
                .context("dataset report without dataset_hash")?
                .to_owned(),
            metric: metric.to_owned(),
            value: number(&value["metrics"][metric], &format!("metrics.{metric}"))?,
            benchmark_valid: value["benchmark_valid"].as_bool().unwrap_or(false),
            protocol: value["protocol"].as_str().map(str::to_owned),
            source_kind: "reproducible-run-report",
            schema,
        }),
        "SotarenaPullArtifact.v1" => Ok(Receipt {
            dataset_id: value["dataset_hash"]
                .as_str()
                .context("pull artifact without dataset_hash")?
                .to_owned(),
            metric: metric.to_owned(),
            value: number(
                &value["reproduced_metrics"][metric],
                &format!("reproduced_metrics.{metric}"),
            )?,
            benchmark_valid: value["benchmark_valid"].as_bool().unwrap_or(false),
            protocol: None,
            source_kind: "replay-artifact",
            schema,
        }),
        other => bail!("unsupported receipt schema {other:?}"),
    }
}

/// Emission parameters.
#[derive(Clone, Debug)]
pub struct Emission {
    /// Track id for every row.
    pub track_id: String,
    /// Method id for every row.
    pub method_id: String,
    /// Protocol for every row (a receipt's own `protocol` must agree when present).
    pub protocol: String,
    /// Metric id.
    pub metric: String,
    /// `source_revision` for every row.
    pub source_revision: String,
    /// `notes` for every row.
    pub notes: String,
    /// Directory the `source_uri` is relative to.
    pub root: PathBuf,
}

/// Every `*.json` under `root` (recursively, sorted) that parses as a supported receipt.
pub fn collect_receipts(root: &Path) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .with_context(|| format!("read {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else if path.extension() == Some(OsStr::new("json")) {
                out.push(path);
            }
        }
        Ok(())
    }
    ensure!(
        root.is_dir(),
        "campaign root {} is not a directory",
        root.display()
    );
    let mut out = Vec::new();
    walk(root, &mut out)?;
    Ok(out)
}

/// Turn every receipt under `emission.root` into rows; one row per dataset, duplicates rejected.
pub fn emit_scores(emission: &Emission) -> Result<Vec<ScoreRow>> {
    ensure!(
        PROTOCOLS.contains(&emission.protocol.as_str()),
        "unknown protocol {}",
        emission.protocol
    );
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for path in collect_receipts(&emission.root)? {
        let value: Value = match serde_json::from_slice(&fs::read(&path)?) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(schema) = value["schema"].as_str() else {
            continue;
        };
        if !matches!(
            schema,
            "sotarena.score-receipt.v1" | "JopeDimeDatasetReport.v3" | "SotarenaPullArtifact.v1"
        ) {
            continue;
        }
        let receipt =
            parse_receipt(&value, &emission.metric).with_context(|| path.display().to_string())?;
        ensure!(
            receipt.metric == emission.metric,
            "{} scores {} but the track metric is {}",
            path.display(),
            receipt.metric,
            emission.metric
        );
        if let Some(protocol) = &receipt.protocol {
            ensure!(
                protocol == &emission.protocol,
                "{} records protocol {protocol} but emission asked for {}",
                path.display(),
                emission.protocol
            );
        }
        ensure!(
            seen.insert(receipt.dataset_id.clone()),
            "two receipts for dataset {} (second: {})",
            receipt.dataset_id,
            path.display()
        );
        let (source_sha256, _) = sha256_file(&path)?;
        let relative = path
            .strip_prefix(&emission.root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        let row = ScoreRow {
            track_id: emission.track_id.clone(),
            dataset_id: receipt.dataset_id,
            method_id: emission.method_id.clone(),
            protocol: emission.protocol.clone(),
            metric: emission.metric.clone(),
            value: py_repr(receipt.value),
            status: "ok".to_owned(),
            benchmark_valid: py_bool(receipt.benchmark_valid).to_owned(),
            source_kind: receipt.source_kind.to_owned(),
            source_revision: emission.source_revision.clone(),
            source_sha256,
            source_uri: relative,
            notes: emission.notes.clone(),
        };
        row.validate()?;
        rows.push(row);
    }
    ensure!(
        !rows.is_empty(),
        "no supported receipts under {} (expected sotarena.score-receipt.v1, JopeDimeDatasetReport.v3 or SotarenaPullArtifact.v1)",
        emission.root.display()
    );
    rows.sort_by(|a, b| a.dataset_id.cmp(&b.dataset_id));
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n\
tabular-regression-v2,019de5d76a718163,jope-dime-regressor,public-development,rmse,0.02830482835368847,ok,True,reproducible-run-report,,7dbfa1d3b007a74358b14b6b0b498bc4e850be83f1aed3ab1189ac45f0bcd2e6,results/external-references/golden42/019de5d76a718163/jope-dime-regressor.json,Exact Golden42 report; method identity is the report filename.\n\
tabular-regression-v2,ZTS_small,foo,public-development,rmse,,missing,False,validated-controller-report,2026-06-21T08:06:56+00:00,,,\n";

    #[test]
    fn scores_round_trip_byte_for_byte() {
        let rows = parse_scores(SAMPLE).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].value_f64(), Some(0.028_304_828_353_688_47));
        assert!(rows[0].is_rankable());
        assert!(!rows[1].is_rankable());
        rows[0].validate().unwrap();
        rows[1].validate().unwrap();
        assert_eq!(write_scores(&rows).unwrap(), SAMPLE);
    }

    #[test]
    fn header_and_enums_are_enforced() {
        assert!(parse_scores("a,b\n1,2\n").is_err());
        let mut row = parse_scores(SAMPLE).unwrap().remove(0);
        row.protocol = "secret".to_owned();
        assert!(row.validate().is_err());
        row.protocol = "blind".to_owned();
        row.status = "ok".to_owned();
        row.value = "nan".to_owned();
        assert!(row.validate().is_err());
    }

    #[test]
    fn emission_reads_every_receipt_kind() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("a")).unwrap();
        fs::write(
            root.join("a/receipt.json"),
            r#"{"schema":"sotarena.score-receipt.v1","track_id":"t","dataset_id":"d1","metric":"rmse","value":0.5,"rows":3,"predictions_sha256":"","test_sha256":""}"#,
        )
        .unwrap();
        fs::write(
            root.join("report.json"),
            r#"{"schema":"JopeDimeDatasetReport.v3","dataset_hash":"d2","metrics":{"rmse":0.25,"r2":0.9},"benchmark_valid":true,"protocol":"aspfm-generated"}"#,
        )
        .unwrap();
        fs::write(
            root.join("metadata.json"),
            r#"{"schema":"SotarenaPullArtifact.v1","dataset_hash":"d3","reproduced_metrics":{"rmse":0.125},"benchmark_valid":true}"#,
        )
        .unwrap();
        fs::write(root.join("noise.json"), r#"{"schema":"something.else"}"#).unwrap();
        fs::write(root.join("broken.json"), "{").unwrap();
        let emission = Emission {
            track_id: "tabular-regression-v2".to_owned(),
            method_id: "jope-dime".to_owned(),
            protocol: "aspfm-generated".to_owned(),
            metric: "rmse".to_owned(),
            source_revision: "rev".to_owned(),
            notes: "n".to_owned(),
            root: root.to_path_buf(),
        };
        let rows = emit_scores(&emission).unwrap();
        assert_eq!(
            rows.iter()
                .map(|r| r.dataset_id.as_str())
                .collect::<Vec<_>>(),
            ["d1", "d2", "d3"]
        );
        assert_eq!(rows[0].value, "0.5");
        assert_eq!(rows[0].source_uri, "a/receipt.json");
        assert_eq!(rows[0].source_kind, "score-receipt");
        assert_eq!(rows[1].source_kind, "reproducible-run-report");
        assert_eq!(rows[2].source_kind, "replay-artifact");
        assert_eq!(rows[0].benchmark_valid, "True");
        let (sha, _) = sha256_file(&root.join("a/receipt.json")).unwrap();
        assert_eq!(rows[0].source_sha256, sha);
        let text = write_scores(&rows).unwrap();
        assert_eq!(parse_scores(&text).unwrap(), rows);

        // A receipt that names another protocol is refused, as is a duplicate dataset.
        let mut wrong = emission.clone();
        wrong.protocol = "public-development".to_owned();
        assert!(emit_scores(&wrong)
            .unwrap_err()
            .to_string()
            .contains("records protocol aspfm-generated"));
        fs::write(
            root.join("dup.json"),
            r#"{"schema":"sotarena.score-receipt.v1","dataset_id":"d1","metric":"rmse","value":0.5}"#,
        )
        .unwrap();
        assert!(emit_scores(&emission)
            .unwrap_err()
            .to_string()
            .contains("two receipts for dataset d1"));
    }

    #[test]
    fn empty_campaign_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let emission = Emission {
            track_id: "t".to_owned(),
            method_id: "m".to_owned(),
            protocol: "blind".to_owned(),
            metric: "rmse".to_owned(),
            source_revision: String::new(),
            notes: String::new(),
            root: dir.path().to_path_buf(),
        };
        assert!(emit_scores(&emission)
            .unwrap_err()
            .to_string()
            .contains("no supported receipts"));
    }
}
