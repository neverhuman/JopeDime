//! Pure, deterministic census of supplied immutable index bytes. No estate discovery.

use crate::canon::{is_hex64, sha256_hex};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Inputs supplied by the producer, never filled from the importer's checkout or clock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerSpec {
    /// Optional source commit value; `None` means it is not available.
    pub source_commit: Option<String>,
    /// Optional code version value; `None` means it is not available.
    pub code_version: Option<String>,
    /// Optional command value; `None` means it is not available.
    pub command: Option<Vec<String>>,
    /// Optional collection cut value; `None` means it is not available.
    pub collection_cut: Option<String>,
    /// Timestamp at which the associated collected occurred.
    pub collected_at: Option<String>,
    /// Input uri recorded for this producer spec.
    pub input_uri: String,
    /// Identifier for the associated input version.
    pub input_version_id: Option<String>,
    /// Other immutable roots consumed by the producer (including source/code receipts).
    pub input_roots: Vec<InputRoot>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `InputRoot` is a record for the input root contract in this module: Pure, deterministic census of supplied immutable index bytes. No estate discovery.
pub struct InputRoot {
    /// Uri recorded for this input root.
    pub uri: String,
    /// Identifier for the associated version.
    pub version_id: Option<String>,
    /// Sha-256 recorded for this input root.
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RowIdentity` is a record for the row identity contract in this module: Pure, deterministic census of supplied immutable index bytes. No estate discovery.
pub struct RowIdentity {
    /// Locator recorded for this row identity.
    pub locator: String,
    /// Hash of the parsed row's compact JSON; the input hash preserves its exact original bytes.
    pub row_sha256: String,
    /// Identifier for the associated occurrence.
    pub occurrence_id: String,
}

/// Local archive format, pending the integrator's shared schema allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerManifest {
    /// Archive version recorded for this producer manifest.
    pub archive_version: u32,
    /// Producer recorded for this producer manifest.
    pub producer: ProducerSpec,
    /// SHA-256 digest identifying the associated input.
    pub input_sha256: String,
    /// Index schema recorded for this producer manifest.
    pub index_schema: String,
    /// Ordering recorded for this producer manifest.
    pub ordering: String,
    /// Number of rows represented by this value.
    pub rows: Vec<RowIdentity>,
}

/// Performs the parse index operation and returns an error if it cannot be completed.
pub fn parse_index(bytes: &[u8]) -> Result<Value> {
    let index =
        crate::clover::round_qual::parse_strict_json_value(bytes, "supplied observation index")?;
    ensure!(
        index["schema"] == "jopedime.gp-results-index/v1",
        "unsupported index schema"
    );
    ensure!(index["runs"].is_array(), "index missing runs array");
    Ok(index)
}

/// Performs the row identities operation and returns an error if it cannot be completed.
pub fn row_identities(index: &Value, input_sha256: &str) -> Result<Vec<RowIdentity>> {
    index["runs"]
        .as_array()
        .context("index missing runs")?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let locator = format!("/runs/{i}");
            let row_sha256 = sha256_hex(&serde_json::to_vec(row)?);
            // Reuse I02's adopted canonical codec and normative domain separation.
            let payload = jope_records::wire_contract::CanonicalPayload::parse_and_canonicalize(
                &serde_json::to_vec(&serde_json::json!({
                    "schema": "jope.observation-occurrence/v2", "input_sha256": input_sha256,
                    "locator": locator, "row_sha256": row_sha256,
                }))?,
            )?;
            let occurrence_id = payload.domain_separated_id()?.sha256_hex().to_owned();
            Ok(RowIdentity {
                locator,
                row_sha256,
                occurrence_id,
            })
        })
        .collect()
}

/// Performs the produce manifest operation and returns an error if it cannot be completed.
pub fn produce_manifest(bytes: &[u8], producer: ProducerSpec) -> Result<ProducerManifest> {
    let index = parse_index(bytes)?;
    let input_sha256 = sha256_hex(bytes);
    Ok(ProducerManifest {
        archive_version: 2,
        rows: row_identities(&index, &input_sha256)?,
        producer,
        input_sha256,
        index_schema: "jopedime.gp-results-index/v1".into(),
        ordering: "original_runs_array_occurrences".into(),
    })
}

/// Pin comes from the controller's independently authenticated custody, never from this
/// manifest itself. A self-asserted commit, hash or success Boolean is insufficient.
#[derive(Clone, Copy, Debug)]
pub struct ProducerInput<'a> {
    /// Byte count for the associated manifest.
    pub manifest_bytes: &'a [u8],
    /// SHA-256 digest identifying the associated authenticated manifest.
    pub authenticated_manifest_sha256: Option<&'a str>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
/// `ProducerClosure` is an enum for the producer closure contract in this module: Pure, deterministic census of supplied immutable index bytes. No estate discovery.
pub enum ProducerClosure {
    /// The `Bound` case of `ProducerClosure`.
    Bound {
        /// SHA-256 digest identifying the associated manifest.
        manifest_sha256: String,
    },
    /// The `Held` case of `ProducerClosure`.
    Held {
        /// Reasons recorded for this outcome.
        reasons: Vec<String>,
    },
}

/// Authentication and immutable binding are separate from observation retention. Any refusal
/// returns HELD so the complete supplied denominator can still be archived.
pub fn verify_producer(bytes: &[u8], input: Option<ProducerInput<'_>>) -> ProducerClosure {
    let check = || -> Result<String> {
        let input = input.context("missing producer manifest")?;
        let pin = input
            .authenticated_manifest_sha256
            .context("missing authenticated producer pin")?;
        ensure!(
            is_hex64(pin) && sha256_hex(input.manifest_bytes) == pin,
            "producer manifest digest mismatch"
        );
        let manifest: ProducerManifest =
            serde_json::from_value(crate::clover::round_qual::parse_strict_json_value(
                input.manifest_bytes,
                "producer manifest",
            )?)?;
        ensure!(
            manifest.archive_version == 2,
            "unsupported producer version"
        );
        let spec = &manifest.producer;
        let commit = spec
            .source_commit
            .as_ref()
            .context("missing producer source commit")?;
        ensure!(
            commit.len() == 40
                && commit
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "invalid full source commit"
        );
        ensure!(
            spec.code_version
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty()),
            "missing producer code version"
        );
        ensure!(
            spec.command
                .as_ref()
                .is_some_and(|a| !a.is_empty() && a.iter().all(|s| !s.trim().is_empty())),
            "missing canonical producer command"
        );
        ensure!(
            spec.collection_cut
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty()),
            "missing collection cut"
        );
        ensure!(
            spec.collected_at
                .as_ref()
                .is_some_and(|s| !s.trim().is_empty()),
            "missing collection time"
        );
        ensure!(!spec.input_uri.trim().is_empty(), "missing input URI");
        ensure!(
            !spec.input_roots.is_empty()
                && spec
                    .input_roots
                    .iter()
                    .all(|r| !r.uri.trim().is_empty() && is_hex64(&r.sha256)),
            "missing or malformed authenticated input roots"
        );
        ensure!(
            produce_manifest(bytes, spec.clone())? == manifest,
            "producer input/row census mismatch"
        );
        Ok(pin.to_owned())
    };
    match check() {
        Ok(manifest_sha256) => ProducerClosure::Bound { manifest_sha256 },
        Err(e) => ProducerClosure::Held {
            reasons: vec![format!("{e:#}")],
        },
    }
}

/// 18,380 is only the dated task-card target. It never supplies today's estate denominator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalFixture {
    /// Date recorded for this historical fixture.
    pub date: String,
    /// Number of expected rows represented here.
    pub expected_rows: usize,
    /// SHA-256 digest identifying the associated input.
    pub input_sha256: String,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// Expected dispositions recorded for this historical fixture.
    pub expected_dispositions: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
/// `HistoricalStatus` is an enum for the historical status contract in this module: Pure, deterministic census of supplied immutable index bytes. No estate discovery.
pub enum HistoricalStatus {
    /// Field value recorded by this public API type.
    /// The `VerifiedDatedFixture` case of `HistoricalStatus`.
    VerifiedDatedFixture {
        /// Date recorded for this outcome.
        date: String,
        /// Rows recorded for this outcome.
        rows: usize,
    },
    /// Field value recorded by this public API type.
    /// The `Held` case of `HistoricalStatus`.
    Held {
        /// Dated target recorded for this outcome.
        dated_target: usize,
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

/// Performs the verify historical operation and returns an error if it cannot be completed.
pub fn verify_historical(
    expected: Option<&HistoricalFixture>,
    input_sha256: &str,
    closure: &ProducerClosure,
    actual: &std::collections::BTreeMap<String, usize>,
) -> HistoricalStatus {
    if let (Some(e), ProducerClosure::Bound { manifest_sha256 }) = (expected, closure) {
        if !e.date.trim().is_empty()
            && e.expected_rows == 18_380
            && e.input_sha256 == input_sha256
            && &e.manifest_sha256 == manifest_sha256
            && &e.expected_dispositions == actual
            && actual.values().sum::<usize>() == e.expected_rows
        {
            return HistoricalStatus::VerifiedDatedFixture {
                date: e.date.clone(),
                rows: e.expected_rows,
            };
        }
    }
    HistoricalStatus::Held { dated_target: 18_380, reason: "UNKNOWN: original dated input, authenticated producer and expected disposition census not supplied or do not match; current estate unknown".into() }
}
