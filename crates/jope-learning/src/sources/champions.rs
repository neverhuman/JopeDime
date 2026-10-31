//! `parallel.cut_champion.v1` → `jope.hwm/v1` (11,030 training/validation winners + 873 sealed).
//!
//! One file per identity, named `Identity::key()`, holding the engine's `current.json` bytes.
//! Those bytes become the record blob unchanged; `prev_event_sha256` and `improvement_number`
//! are copied from the champion so the engine's chain is preserved verbatim.

use crate::canon::{canonical_sha256, sha256_hex};
use crate::champion::Champion;
use crate::hwm::Candidate;
use crate::metric::Metric;
use crate::record::*;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

/// What names an exact cut: everything in the identity except the dataset (already a field) and
/// the evaluator/vocabulary (already fields of the HWM identity).
#[derive(Serialize)]
struct CutIdentity<'a> {
    family: &'a str,
    partition: &'a str,
    cut: u64,
    prepared_sha256: &'a str,
    training_sha256: &'a str,
    validation_sha256: &'a str,
}

/// Performs the cut ID operation and returns an error if it cannot be completed.
pub fn cut_id(identity: &crate::champion::Identity) -> Result<String> {
    canonical_sha256(&CutIdentity {
        family: &identity.family,
        partition: &identity.partition,
        cut: identity.cut,
        prepared_sha256: &identity.prepared_sha256,
        training_sha256: &identity.training_sha256,
        validation_sha256: &identity.validation_sha256,
    })
}

#[derive(Clone, Debug, Serialize)]
/// `ChampionReject` is a record for the champion reject contract in this module: parallel.cutchampion.v1 → jope.hwm/v1 (11,030 training/validation winners + 873 sealed).
pub struct ChampionReject {
    /// File recorded for this champion reject.
    pub file: String,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Default)]
/// `ChampionScan` is a record for the champion scan contract in this module: parallel.cutchampion.v1 → jope.hwm/v1 (11,030 training/validation winners + 873 sealed).
pub struct ChampionScan {
    /// Files recorded for this champion scan.
    pub files: u64,
    /// Mapped recorded for this champion scan.
    pub mapped: u64,
    /// Sealed recorded for this champion scan.
    pub sealed: u64,
    /// Collection of rejects values carried by this champion scan.
    pub rejects: Vec<ChampionReject>,
}

/// Map one `current.json` byte string. `expected_key` is the file stem (must equal
/// `Identity::key()`); `bundle_id` names the restore bundle.
pub fn map_champion(
    bytes: &[u8],
    expected_key: Option<&str>,
    bundle_id: &str,
) -> Result<Candidate> {
    let ch = Champion::parse(bytes).context("parse parallel.cut_champion.v1")?;
    ch.validate()?;
    let key = ch.identity.key()?;
    if let Some(k) = expected_key {
        ensure!(k == key, "file name {k} is not Identity::key() {key}");
    }
    let sealed = ch.identity.is_sealed();
    ensure!(
        matches!(
            ch.identity.partition.as_str(),
            "meta_train" | "meta_validation" | "meta_test"
        ),
        "unknown partition {}",
        ch.identity.partition
    );
    let protocol = ch.observation["evaluation_bias"]
        .as_str()
        .context("evaluation_bias absent")?
        .to_string();
    let identity = HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: ch.identity.dataset.clone(),
        cut_id: cut_id(&ch.identity)?,
        evaluator_sha256: ch.identity.evaluator_sha256.clone(),
        vocabulary_sha256: ch.identity.vocabulary_sha256.clone(),
        protocol,
    };
    let dsl = ch.genome()?.to_string();
    let program_sha256 = ch.program_sha256()?.to_string();
    ensure!(
        program_sha256 == sha256_hex(dsl.as_bytes()),
        "program_sha256 does not hash the genome"
    );
    let version = ch.observation["schema"]
        .as_str()
        .unwrap_or("parallel.gp_observation.v1")
        .to_string();
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256()?,
        identity,
        legacy_identity_sha256: Some(key),
        genome: Genome {
            dsl,
            program_sha256,
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value: ch.rmse_original_units,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost {
            runtime_seconds: ch.observation["runtime_seconds"].as_f64(),
            search_seconds: ch.observation["completed_search_seconds"].as_f64(),
        },
        evaluator: Evaluator {
            sha256: ch.identity.evaluator_sha256.clone(),
            version,
        },
        benchmark_valid: false,
        sealed,
        replayable: true,
        provenance: Provenance {
            source: if sealed {
                "champions-sealed".into()
            } else {
                "champions".into()
            },
            run_id: ch.source.to_string_lossy().into_owned(),
            git_sha: UNBOUND.into(),
            bundle_id: bundle_id.into(),
            record_sha256: sha256_hex(bytes),
        },
        prev_event_sha256: ch.previous_event_sha256.clone(),
        improvement_number: ch.improvement_number,
        ts_unix: ch.timestamp,
    };
    record.validate()?;
    Ok(Candidate {
        record,
        blob: bytes.to_vec(),
        chain_from_source: true,
    })
}

/// Every `*.json` in `dir` (flat restore layout) or every `<key>/current.json` (engine layout).
pub fn champion_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_dir() {
            let cur = path.join("current.json");
            if cur.exists() {
                out.push((name, cur));
            }
        } else if let Some(stem) = name.strip_suffix(".json") {
            out.push((stem.to_string(), path));
        }
    }
    out.sort();
    Ok(out)
}

/// Map a whole restore directory; rejects are returned, never dropped.
pub fn scan(dir: &Path, bundle_id: &str) -> Result<(Vec<Candidate>, ChampionScan)> {
    let files = champion_files(dir)?;
    if files.is_empty() {
        bail!("no champion files under {}", dir.display());
    }
    let mut scan = ChampionScan::default();
    let mut out = Vec::with_capacity(files.len());
    for (stem, path) in files {
        scan.files += 1;
        let bytes = std::fs::read(&path)?;
        let expected = crate::canon::is_hex64(&stem).then_some(stem.as_str());
        match map_champion(&bytes, expected, bundle_id) {
            Ok(c) => {
                if c.record.sealed {
                    scan.sealed += 1;
                }
                scan.mapped += 1;
                out.push(c);
            }
            Err(e) => scan.rejects.push(ChampionReject {
                file: path.display().to_string(),
                reason: format!("{e:#}"),
            }),
        }
    }
    Ok((out, scan))
}

/// Performs the scan source operation and returns an error if it cannot be completed.
pub fn scan_source(dir: &Path, bundle_id: &str, scan: &ChampionScan) -> serde_json::Value {
    crate::hwm::source_value(
        "champions",
        dir,
        json!({
            "bundle_id": bundle_id,
            "files": scan.files,
            "mapped": scan.mapped,
            "sealed": scan.sealed,
            "unmapped": scan.rejects.len(),
        }),
    )
}
