//! Read-only source adapters. They translate existing evidence into `MeasurementRow`s and
//! feature lines without changing any value or grade, and refuse inputs they do not recognize.

use super::pool::{MeasurementRow, Status};
use super::selectors::FeatureLine;
use crate::canon::{is_hex64, sha256_hex};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

/// Schema identifier for coverage values.
pub const COVERAGE_SCHEMA: &str = "jope.r22-lpi-scratch-coverage/v1";
/// Constant identifying the roster feature dtype.
pub const ROSTER_FEATURE_DTYPE: &str = "<f4";
/// `q_grid` is micro-q; a row whose `q` disagrees with it by more than this is refused.
const Q_GRID_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Serialize)]
/// `CoverageReport` is a record for the coverage report contract in this module: Read-only source adapters. They translate existing evidence into MeasurementRows and feature lines without changing any value or grade, and refuse inputs they do not recognize.
pub struct CoverageReport {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Source schema recorded for this coverage report.
    pub source_schema: String,
    /// Source label recorded for this coverage report.
    pub source_label: String,
    /// SHA-256 digest identifying the associated source receipt.
    pub source_receipt_sha256: String,
    /// Partition recorded for this coverage report.
    pub partition: String,
    /// Contexts in partition recorded for this coverage report.
    pub contexts_in_partition: usize,
    /// Excluded by list recorded for this coverage report.
    pub excluded_by_list: usize,
    /// Excluded pairs listed recorded for this coverage report.
    pub excluded_pairs_listed: usize,
    /// Excluded pairs removed recorded for this coverage report.
    pub excluded_pairs_removed: usize,
    /// Ineligible recorded for this coverage report.
    pub ineligible: usize,
    /// Contexts emitted recorded for this coverage report.
    pub contexts_emitted: usize,
    /// Number of rows represented by this value.
    pub rows: usize,
}

fn status(outcome: &str) -> Status {
    match outcome {
        "completed" => Status::Complete,
        "failed" | "program_failed" => Status::ProgramFailed,
        "timeout" => Status::Timeout,
        _ => Status::Unknown,
    }
}

fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string {key}"))
}

/// Rows for every member of `measured_in_trie` in `partition` contexts. Only the `train`
/// partition may be adapted: shadow pools never read validation, sealed or test outcomes.
/// `excluded_pairs` removes named (context, program) members, e.g. each context's own incumbent.
pub fn coverage_rows(
    doc: &Value,
    partition: &str,
    excluded: &BTreeSet<String>,
    excluded_pairs: &BTreeSet<(String, String)>,
    protocol_sha256: &str,
) -> Result<(Vec<MeasurementRow>, CoverageReport)> {
    ensure!(
        partition == "train",
        "only the train partition may enter a shadow pool"
    );
    ensure!(
        is_hex64(protocol_sha256),
        "protocol_sha256 must be a full sha256"
    );
    let schema = text(doc, "schema")?;
    ensure!(
        schema == COVERAGE_SCHEMA,
        "unknown coverage schema {schema:?}"
    );
    let contexts = doc
        .get("contexts")
        .and_then(Value::as_object)
        .context("coverage contexts must be an object keyed by context id")?;
    let mut report = CoverageReport {
        schema: "jope.j9-coverage-ingest/v1",
        source_schema: schema.to_owned(),
        source_label: text(doc, "label")?.to_owned(),
        source_receipt_sha256: text(doc, "receipt_sha256")?.to_owned(),
        partition: partition.to_owned(),
        contexts_in_partition: 0,
        excluded_by_list: 0,
        excluded_pairs_listed: excluded_pairs.len(),
        excluded_pairs_removed: 0,
        ineligible: 0,
        contexts_emitted: 0,
        rows: 0,
    };
    let mut rows = Vec::new();
    for (cid, ctx) in contexts {
        if text(ctx, "partition")? != partition {
            continue;
        }
        report.contexts_in_partition += 1;
        if excluded.contains(cid) {
            report.excluded_by_list += 1;
            continue;
        }
        if ctx.pointer("/counts/eligible").and_then(Value::as_bool) != Some(true) {
            report.ineligible += 1;
            continue;
        }
        let lineage = text(ctx, "lineage")?;
        let incumbent = ctx.pointer("/incumbent/q").and_then(Value::as_f64);
        let members = ctx
            .get("measured_in_trie")
            .and_then(Value::as_array)
            .with_context(|| format!("{cid}: measured_in_trie"))?;
        for m in members {
            let program = text(m, "program_sha256")?;
            if excluded_pairs.contains(&(cid.clone(), program.to_owned())) {
                report.excluded_pairs_removed += 1;
                continue;
            }
            let st = status(text(m, "outcome")?);
            let q = m.get("q").and_then(Value::as_f64);
            if st == Status::Complete {
                let q = q.with_context(|| format!("{cid}: completed member without q"))?;
                let grid = m
                    .get("q_grid")
                    .and_then(Value::as_i64)
                    .with_context(|| format!("{cid}: q_grid"))?;
                ensure!(
                    (q - grid as f64 / 1e6).abs() <= Q_GRID_TOLERANCE,
                    "{cid}: q {q} disagrees with q_grid {grid}"
                );
            }
            let tokens = m
                .get("tokens")
                .and_then(Value::as_array)
                .with_context(|| format!("{cid}: tokens"))?
                .iter()
                .map(|t| {
                    t.as_u64()
                        .and_then(|t| u32::try_from(t).ok())
                        .context("token id")
                })
                .collect::<Result<Vec<u32>>>()?;
            rows.push(MeasurementRow {
                context_id: cid.clone(),
                parent_id: lineage.to_owned(),
                role: partition.to_owned(),
                program_sha256: program.to_owned(),
                status: st,
                q: if st == Status::Complete { q } else { None },
                incumbent_q: incumbent,
                protocol_sha256: protocol_sha256.to_owned(),
                tokens: Some(tokens),
            });
        }
        report.contexts_emitted += 1;
    }
    report.rows = rows.len();
    Ok((rows, report))
}

/// Descriptor projection for global representation/retrieval. No legacy quality, incumbent,
/// runtime or failure label can escape through this versioned adapter.
#[derive(Debug, Serialize)]
pub struct CoverageDescriptor {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent.
    pub parent_id: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Optional tokens value; `None` means it is not available.
    pub tokens: Option<Vec<u32>>,
}

/// Versioned global ingress companion. The entire legacy document is bound to a verified
/// exposure receipt, so changing partition, `eligible`, scores, tokens, or incumbent changes
/// the subject identity. V1 `coverage_rows` remains a diagnostic shadow-pool adapter.
pub fn coverage_rows_for(
    doc: &Value,
    protocol_sha256: &str,
    verifier: &crate::replay::admission::AdmissionVerifier<'_>,
    receipt_sha256: &str,
    purpose: crate::replay::admission::Purpose,
    at_ms: u64,
) -> Result<(
    Vec<CoverageDescriptor>,
    CoverageReport,
    crate::replay::admission::ReplayAdmission,
)> {
    ensure!(
        purpose.is_global(),
        "coverage global ingress requires a global purpose"
    );
    let subject = crate::canon::canonical_sha256(doc)?;
    let admission = verifier.admit(&subject, receipt_sha256, purpose, None, at_ms)?;
    let (rows, report) = coverage_rows(
        doc,
        "train",
        &BTreeSet::new(),
        &BTreeSet::new(),
        protocol_sha256,
    )?;
    // A legacy aggregate cannot authenticate per-attempt labels or policy targets. This
    // companion admits descriptors for representation/retrieval only; measured global labels
    // must enter individually through ExposureReceipt/Observation.
    ensure!(
        matches!(
            purpose,
            crate::replay::admission::Purpose::RepresentationLearning
                | crate::replay::admission::Purpose::RetrievalIndexUpdate
        ),
        "HELD: v1 coverage cannot supply global per-label supervision"
    );
    let descriptors = rows
        .into_iter()
        .map(|row| CoverageDescriptor {
            context_id: row.context_id,
            parent_id: row.parent_id,
            program_sha256: row.program_sha256,
            tokens: row.tokens,
        })
        .collect();
    Ok((descriptors, report, admission))
}

#[derive(Deserialize)]
struct RosterFeatures {
    bytes: usize,
    dimensions: usize,
    dtype: String,
    offset: usize,
    pack_bytes: usize,
    pack_sha256: String,
    sha256: String,
}

#[derive(Deserialize)]
struct RosterRow {
    context_id: String,
    features: RosterFeatures,
}

/// Raw context vectors for `wanted` contexts from a joined native roster and its f32 pack. The
/// pack digest and every row digest are checked; a missing context is an error.
pub fn roster_vectors(
    roster: &str,
    pack: &[u8],
    wanted: &BTreeSet<String>,
) -> Result<Vec<FeatureLine>> {
    let pack_sha = sha256_hex(pack);
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, line) in roster
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        // Only the two fields used here are read; the roster's other fields stay untouched.
        let v: Value =
            serde_json::from_str(line).with_context(|| format!("roster line {}", i + 1))?;
        let row: RosterRow = serde_json::from_value(serde_json::json!({
            "context_id": v.get("context_id"),
            "features": v.get("features"),
        }))
        .with_context(|| format!("roster line {}", i + 1))?;
        if !wanted.contains(&row.context_id) {
            continue;
        }
        let f = &row.features;
        ensure!(
            f.dtype == ROSTER_FEATURE_DTYPE,
            "unsupported feature dtype {}",
            f.dtype
        );
        ensure!(
            f.pack_bytes == pack.len() && f.pack_sha256 == pack_sha,
            "feature pack identity mismatch"
        );
        ensure!(f.bytes == f.dimensions * 4, "feature byte count");
        let end = f.offset.checked_add(f.bytes).context("feature range")?;
        ensure!(end <= pack.len(), "feature range outside the pack");
        let slice = &pack[f.offset..end];
        ensure!(
            sha256_hex(slice) == f.sha256,
            "{}: feature digest mismatch",
            row.context_id
        );
        let vector: Vec<f64> = slice
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        ensure!(
            vector.iter().all(|x| x.is_finite()),
            "{}: non-finite feature",
            row.context_id
        );
        if !seen.insert(row.context_id.clone()) {
            bail!("duplicate roster row for {}", row.context_id);
        }
        out.push(FeatureLine {
            id: row.context_id,
            vector,
        });
    }
    let missing: Vec<&String> = wanted.difference(&seen).collect();
    ensure!(
        missing.is_empty(),
        "{} pooled contexts have no roster vector, e.g. {:?}",
        missing.len(),
        missing.first()
    );
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hex(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    #[test]
    fn coverage_adapter_keeps_train_only_and_honours_exclusions() {
        let member = |p: char, q: f64| json!({"outcome": "completed", "program_sha256": hex(p), "q": q, "q_grid": (q * 1e6).round() as i64, "q_opt": q, "tokens": [1, 5, 2]});
        let doc = json!({
            "schema": COVERAGE_SCHEMA, "label": "R22 scratch-only, non-qualifying", "receipt_sha256": hex('f'),
            "contexts": {
                hex('a'): {"partition": "train", "lineage": hex('1'), "counts": {"eligible": true}, "incumbent": {"q": 0.9}, "measured_in_trie": [member('b', 0.5), member('c', 0.25)]},
                hex('d'): {"partition": "train", "lineage": hex('2'), "counts": {"eligible": true}, "incumbent": {"q": 0.9}, "measured_in_trie": [member('b', 0.5)]},
                hex('e'): {"partition": "validation", "lineage": hex('3'), "counts": {"eligible": true}, "incumbent": {"q": 0.9}, "measured_in_trie": [member('b', 0.5)]},
            }
        });
        let excluded: BTreeSet<String> = [hex('d')].into();
        let none = BTreeSet::new();
        let (rows, report) = coverage_rows(&doc, "train", &excluded, &none, &hex('9')).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows
            .iter()
            .all(|r| r.context_id == hex('a') && r.role == "train"));
        assert_eq!(
            (
                report.contexts_in_partition,
                report.excluded_by_list,
                report.contexts_emitted
            ),
            (2, 1, 1)
        );
        assert!(coverage_rows(&doc, "validation", &excluded, &none, &hex('9')).is_err());
        let own: BTreeSet<(String, String)> = [(hex('a'), hex('b'))].into();
        let (rows, report) = coverage_rows(&doc, "train", &excluded, &own, &hex('9')).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].program_sha256, hex('c'));
        assert_eq!(
            (report.excluded_pairs_listed, report.excluded_pairs_removed),
            (1, 1)
        );
        let mut bad = doc.clone();
        bad["contexts"][hex('a')]["measured_in_trie"][0]["q_grid"] = json!(1);
        assert!(coverage_rows(&bad, "train", &excluded, &none, &hex('9')).is_err());
    }

    #[test]
    fn roster_vectors_check_digests() {
        let a: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let b: Vec<u8> = [0.5f32, 4.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        let pack: Vec<u8> = a.iter().chain(&b).copied().collect();
        let line = |id: char, offset: usize, bytes: &[u8]| {
            json!({"context_id": hex(id), "other": 1, "features": {"bytes": 8, "dimensions": 2, "dtype": "<f4", "offset": offset, "pack_bytes": 16, "pack_sha256": sha256_hex(&pack), "path": "x", "sha256": sha256_hex(bytes)}}).to_string()
        };
        let roster = format!("{}\n{}\n", line('a', 0, &a), line('b', 8, &b));
        let wanted: BTreeSet<String> = [hex('b')].into();
        let out = roster_vectors(&roster, &pack, &wanted).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].vector, vec![0.5, 4.0]);
        let tampered = format!("{}\n{}\n", line('a', 0, &a), line('b', 8, &a));
        assert!(roster_vectors(&tampered, &pack, &wanted).is_err());
        let missing: BTreeSet<String> = [hex('c')].into();
        assert!(roster_vectors(&roster, &pack, &missing).is_err());
    }
}
