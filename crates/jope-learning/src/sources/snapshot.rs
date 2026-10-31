//! Lineage-A corpus snapshots (`JopeTestOnlyTrainingCorpus.v1`): the per-(task, dataset_hash)
//! `test_exact` maximum of the strict rows, evaluator `unbound`.
//!
//! The manifest carries counts and the sha256 of each JSONL split; the rows themselves are
//! streamed from `--rows <split>=<path>` (a file, `-` for stdin, or a process substitution over
//! `tar --zstd -xOf`), hashed as they pass so the manifest's `train_sha256`/`validation_sha256`
//! bind what was read. Nothing is extracted to disk.

use crate::canon::{canonical_json, canonical_sha256, sha256_hex};
use crate::hwm::Candidate;
use crate::metric::Metric;
use crate::record::*;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Schema identifier for  values.
pub const SCHEMA: &str = "JopeTestOnlyTrainingCorpus.v1";
/// Constant identifying the tier.
pub const TIER: &str = "test_exact";

#[derive(Serialize)]
struct CutIdentity<'a> {
    tier: &'a str,
    protocol: &'a str,
}

/// Performs the cut ID operation and returns an error if it cannot be completed.
pub fn cut_id(protocol: &str) -> Result<String> {
    canonical_sha256(&CutIdentity {
        tier: TIER,
        protocol,
    })
}

#[derive(Clone, Debug, Deserialize)]
/// `Manifest` is a record for the manifest contract in this module: Lineage-A corpus snapshots (JopeTestOnlyTrainingCorpus.v1): the per-(task, datasethash) testexact maximum of the strict rows, evaluator unbound.
pub struct Manifest {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Source label watermark recorded for this manifest.
    pub source_label_watermark: u64,
    /// Mapping of counts values carried by this manifest.
    pub counts: BTreeMap<String, u64>,
    /// Strict filter recorded for this manifest.
    pub strict_filter: Value,
    /// SHA-256 digest identifying the associated train.
    pub train_sha256: String,
    /// SHA-256 digest identifying the associated validation.
    pub validation_sha256: String,
    #[serde(default)]
    /// SHA-256 digest identifying the associated source db.
    pub source_db_sha256: Option<String>,
    #[serde(default)]
    /// SHA-256 digest identifying the associated catalog.
    pub catalog_sha256: Option<String>,
}

/// The score-bearing part of a corpus row (embeddings are skipped, not stored).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Row {
    /// Identifier for the associated dataset.
    pub dataset_id: String,
    /// Provenance source for this value.
    pub source: String,
    /// Canonical program or genome representation.
    pub genome: String,
    /// Canonical genome recorded for this row.
    pub canonical_genome: String,
    /// Valid recorded for this row.
    pub valid: bool,
    /// Metric used to evaluate the represented result.
    pub metric: String,
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: f64,
    #[serde(default)]
    /// Optional cost ms value; `None` means it is not available.
    pub cost_ms: Option<f64>,
    #[serde(default)]
    /// Optional engine value; `None` means it is not available.
    pub engine: Option<String>,
    /// Metadata recorded for this row.
    pub metadata: RowMeta,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
/// `RowMeta` is a record for the row meta contract in this module: Lineage-A corpus snapshots (JopeTestOnlyTrainingCorpus.v1): the per-(task, datasethash) testexact maximum of the strict rows, evaluator unbound.
pub struct RowMeta {
    /// Task type recorded for this row meta.
    pub task_type: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// Split recorded for this row meta.
    pub split: String,
    /// Score tier recorded for this row meta.
    pub score_tier: String,
    #[serde(default)]
    /// Optional selection bias value; `None` means it is not available.
    pub selection_bias: Option<String>,
    #[serde(default)]
    /// Optional benchmark valid value; `None` means it is not available.
    pub benchmark_valid: Option<bool>,
    #[serde(default)]
    /// Optional generation value; `None` means it is not available.
    pub generation: Option<String>,
    #[serde(default)]
    /// Hash identifying the associated semantic plan.
    pub semantic_plan_hash: Option<String>,
    #[serde(default)]
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: Option<String>,
    #[serde(default)]
    /// SHA-256 digest identifying the associated model checkpoint.
    pub model_checkpoint_sha256: Option<String>,
    #[serde(default)]
    /// Optional label created at ms value; `None` means it is not available.
    pub label_created_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `SplitStats` is a record for the split stats contract in this module: Lineage-A corpus snapshots (JopeTestOnlyTrainingCorpus.v1): the per-(task, datasethash) testexact maximum of the strict rows, evaluator unbound.
pub struct SplitStats {
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Sha-256 recorded for this split stats.
    pub sha256: String,
    /// SHA-256 digest identifying the associated expected.
    pub expected_sha256: String,
    /// Sha-256 matches recorded for this split stats.
    pub sha256_matches: bool,
    /// Mapping of counts values carried by this split stats.
    pub counts: BTreeMap<String, u64>,
    /// Skipped invalid recorded for this split stats.
    pub skipped_invalid: u64,
    /// Skipped tier recorded for this split stats.
    pub skipped_tier: u64,
    /// Skipped metric recorded for this split stats.
    pub skipped_metric: u64,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `SnapshotScan` is a record for the snapshot scan contract in this module: Lineage-A corpus snapshots (JopeTestOnlyTrainingCorpus.v1): the per-(task, datasethash) testexact maximum of the strict rows, evaluator unbound.
pub struct SnapshotScan {
    /// Watermark recorded for this snapshot scan.
    pub watermark: u64,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// Protocol recorded for this snapshot scan.
    pub protocol: String,
    /// Mapping of splits values carried by this snapshot scan.
    pub splits: BTreeMap<String, SplitStats>,
    /// Counts match manifest recorded for this snapshot scan.
    pub counts_match_manifest: bool,
    /// Maxima recorded for this snapshot scan.
    pub maxima: u64,
}

/// Reads manifest from the configured source.
pub fn read_manifest(path: &Path) -> Result<(Manifest, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let m: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        m.schema == SCHEMA,
        "manifest schema {} is not {SCHEMA}",
        m.schema
    );
    Ok((m, sha256_hex(&bytes)))
}

fn expected_metric(manifest: &Manifest, task: &str) -> Option<Metric> {
    manifest.strict_filter[task]["metric"]
        .as_str()
        .and_then(Metric::parse)
}

struct Best {
    row: Row,
    split: String,
}

/// Stream one split, updating per-(task, dataset) maxima.
fn scan_split<R: Read>(
    manifest: &Manifest,
    split: &str,
    reader: R,
    best: &mut BTreeMap<(String, String), Best>,
) -> Result<SplitStats> {
    let expected_sha = match split {
        "train" => manifest.train_sha256.clone(),
        "validation" => manifest.validation_sha256.clone(),
        other => bail!("unknown split {other} (train|validation)"),
    };
    let mut stats = SplitStats {
        expected_sha256: expected_sha,
        ..Default::default()
    };
    let mut hasher = Sha256::new();
    let mut reader = BufReader::with_capacity(1 << 20, reader);
    let mut line = Vec::with_capacity(64 * 1024);
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        hasher.update(&line);
        stats.bytes += n as u64;
        let trimmed = line.strip_suffix(b"\n").unwrap_or(&line);
        if trimmed.is_empty() {
            continue;
        }
        stats.rows += 1;
        let row: Row = serde_json::from_slice(trimmed)
            .with_context(|| format!("{split} row {}", stats.rows))?;
        *stats
            .counts
            .entry(format!("{}/{}", row.metadata.task_type, split))
            .or_default() += 1;
        if !row.valid || !row.score.is_finite() {
            stats.skipped_invalid += 1;
            continue;
        }
        if row.metadata.score_tier != TIER {
            stats.skipped_tier += 1;
            continue;
        }
        let Some(metric) = expected_metric(manifest, &row.metadata.task_type) else {
            stats.skipped_metric += 1;
            continue;
        };
        if Metric::parse(&row.metric) != Some(metric) {
            stats.skipped_metric += 1;
            continue;
        }
        let key = (
            row.metadata.task_type.clone(),
            row.metadata.dataset_hash.clone(),
        );
        match best.get(&key) {
            Some(b) if !metric.direction().strictly_improves(row.score, b.row.score) => {}
            _ => {
                best.insert(
                    key,
                    Best {
                        row,
                        split: split.to_string(),
                    },
                );
            }
        }
    }
    stats.sha256 = hex::encode(hasher.finalize());
    stats.sha256_matches = stats.sha256 == stats.expected_sha256;
    Ok(stats)
}

/// Performs the map best operation and returns an error if it cannot be completed.
pub fn map_best(
    manifest: &Manifest,
    manifest_sha256: &str,
    protocol: &str,
    row: &Row,
    split: &str,
) -> Result<Candidate> {
    let metric = Metric::parse(&row.metric).context("metric")?;
    let identity = HwmIdentity {
        comparison: None,
        task: row.metadata.task_type.clone(),
        dataset_hash: row.metadata.dataset_hash.clone(),
        cut_id: cut_id(protocol)?,
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: protocol.into(),
    };
    let blob = canonical_json(row)?;
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256()?,
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: sha256_hex(row.canonical_genome.as_bytes()),
            dsl: row.canonical_genome.clone(),
            semantic_sha256: row
                .metadata
                .semantic_plan_hash
                .clone()
                .filter(|s| crate::canon::is_hex64(s)),
        },
        score: Score {
            metric,
            value: row.score,
            direction: metric.direction(),
            units: "dimensionless".into(),
        },
        cost: Cost {
            runtime_seconds: row.cost_ms.map(|ms| ms / 1000.0),
            search_seconds: None,
        },
        evaluator: Evaluator {
            sha256: UNBOUND.into(),
            version: format!("snapshot-{}", manifest.source_label_watermark),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: false,
        provenance: Provenance {
            source: format!("snapshot-{}", manifest.source_label_watermark),
            run_id: format!(
                "{}/{split}",
                row.metadata
                    .generation
                    .clone()
                    .unwrap_or_else(|| "unknown-generation".into())
            ),
            git_sha: UNBOUND.into(),
            bundle_id: manifest_sha256.into(),
            record_sha256: sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: row.metadata.label_created_at_ms.unwrap_or(0) / 1000,
    };
    record.validate()?;
    Ok(Candidate {
        record,
        blob,
        chain_from_source: false,
    })
}

/// `splits`: (split name, reader). Returns the candidates (one per (task, dataset)) and the scan.
pub fn scan(
    manifest_path: &Path,
    splits: Vec<(String, Box<dyn Read>)>,
) -> Result<(Vec<Candidate>, SnapshotScan)> {
    let (manifest, manifest_sha256) = read_manifest(manifest_path)?;
    ensure!(!splits.is_empty(), "no --rows given: nothing to scan");
    let protocol = manifest.strict_filter["selection_bias"]
        .as_str()
        .unwrap_or("test-exact")
        .to_string();
    let mut scan = SnapshotScan {
        watermark: manifest.source_label_watermark,
        manifest_sha256: manifest_sha256.clone(),
        protocol: protocol.clone(),
        ..Default::default()
    };
    let mut best = BTreeMap::new();
    for (split, reader) in splits {
        ensure!(
            !scan.splits.contains_key(&split),
            "split {split} given twice"
        );
        let stats = scan_split(&manifest, &split, reader, &mut best)?;
        ensure!(
            stats.sha256_matches,
            "{split} rows hash to {} but the manifest binds {}",
            stats.sha256,
            stats.expected_sha256
        );
        scan.splits.insert(split, stats);
    }
    let mut counted: BTreeMap<String, u64> = BTreeMap::new();
    for s in scan.splits.values() {
        for (k, v) in &s.counts {
            *counted.entry(k.clone()).or_default() += v;
        }
    }
    scan.counts_match_manifest = counted
        .iter()
        .all(|(k, v)| manifest.counts.get(k) == Some(v));
    let mut out = Vec::with_capacity(best.len());
    for b in best.values() {
        out.push(map_best(
            &manifest,
            &manifest_sha256,
            &protocol,
            &b.row,
            &b.split,
        )?);
    }
    scan.maxima = out.len() as u64;
    Ok((out, scan))
}

/// Performs the scan source operation.
pub fn scan_source(manifest: &Path, scan: &SnapshotScan) -> Value {
    crate::hwm::source_value(
        "snapshot",
        manifest,
        json!({
            "watermark": scan.watermark,
            "manifest_sha256": scan.manifest_sha256,
            "protocol": scan.protocol,
            "splits": scan.splits,
            "counts_match_manifest": scan.counts_match_manifest,
            "maxima": scan.maxima,
        }),
    )
}
