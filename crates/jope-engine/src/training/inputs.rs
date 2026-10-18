//! Resolved training inputs (W6-T01).
//!
//! The trainer consumes exactly the bytes it verified: every input is read
//! once, hashed and parsed from the same buffer, and every cached staging is
//! keyed by the content it was built from. A block's resolved inputs are
//! summarized in a frozen manifest (`jope.training_inputs.v1`) whose digest
//! checkpoints and receipts carry.
//!
//! Units kept apart (see `exposure_units` in the manifest):
//! - **available records** are corpus records in admitted groups;
//! - **unique programs** are distinct token sequences among them; one
//!   program can appear in many contexts;
//! - **presentations** are sampled rows. A record may be presented many
//!   times across comparisons and steps; a repeat is exposure of the same
//!   record, never new data. They are counted by the T04 journal.
use crate::{
    corpus::{Index, Record, SEQUENCE},
    manifest::{hash, resolve},
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub const SCHEMA: &str = "jope.training_inputs.v1";
/// One context vector: 1536 dataset + 512 target dimensions of f32.
pub const CONTEXT_BYTES: usize = 8192;

/// Read `path` once and return its bytes only if they hash to `expected`.
/// Callers parse these bytes; the path is never re-read after verification.
pub fn read_verified(path: &Path, expected_sha256: &str) -> Result<Vec<u8>> {
    let bytes =
        fs::read(path).with_context(|| format!("resolved input missing: {}", path.display()))?;
    ensure!(
        hash(&bytes) == expected_sha256,
        "resolved input changed: {}",
        path.display()
    );
    Ok(bytes)
}

/// Cache identity of the staged context tensor: the digest of the context
/// rows actually listed (path, digest, partition, provenance), never the
/// index's declared context-index digest, which the rows need not match.
pub fn context_identity(index: &Index) -> Result<String> {
    Ok(hash(&serde_json::to_vec(&index.contexts)?))
}

/// The resident index must be the index whose bytes were verified. A file
/// substituted between verification and the resident load is refused.
pub fn bind_resident(verified: &Value, index: &Index) -> Result<()> {
    for field in [
        "schema",
        "vocabulary_sha256",
        "context_index_sha256",
        "records_sha256",
    ] {
        let resident = match field {
            "schema" => &index.schema,
            "vocabulary_sha256" => &index.vocabulary_sha256,
            "context_index_sha256" => &index.context_index_sha256,
            _ => &index.records_sha256,
        };
        ensure!(
            verified[field] == resident.as_str(),
            "resident corpus {field} differs from the verified index"
        );
    }
    ensure!(
        verified["qualification"] == index.qualification
            && verified["records"].as_u64() == Some(index.records as u64)
            && verified["contexts"] == serde_json::to_value(&index.contexts)?,
        "resident corpus records, qualification or contexts differ from the verified index"
    );
    if index.schema == "parallel.teaching_corpus.v1" {
        ensure!(
            verified["groups"] == serde_json::to_value(&index.groups)?
                && verified["legal_sets"] == serde_json::to_value(&index.legal_sets)?,
            "resident corpus groups or legal sets differ from the verified index"
        );
    }
    Ok(())
}

/// Every context vector, verified against the index before use.
pub fn stage_contexts(root: &Path, index: &Index) -> Result<Vec<f32>> {
    ensure!(
        index
            .contexts
            .iter()
            .all(|c| crate::corpus_projection::admitted_partition(&c.partition)),
        "non-admitted context partition in training closure"
    );
    let mut contexts = Vec::with_capacity(index.contexts.len() * CONTEXT_BYTES / 4);
    for c in &index.contexts {
        let bytes = read_verified(&resolve(root, &c.path)?, &c.sha256)?;
        ensure!(
            bytes.len() == CONTEXT_BYTES,
            "context artifact dimensions changed: {}",
            c.path.display()
        );
        contexts.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
        );
    }
    Ok(contexts)
}

/// Host-side training tensors in the exact layout `rank_impl` stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staged {
    /// `[records, SEQUENCE]` token ids; the sixteen bits are preserved in i16.
    pub tokens: Vec<i16>,
    /// `[records, SEQUENCE]` legal-set ids.
    pub legal: Vec<i16>,
    pub context_ids: Vec<i64>,
    /// `[legal_sets, vocabulary]` Boolean masks.
    pub masks: Vec<u8>,
}

pub fn stage_records(
    index: &Index,
    vocabulary_len: usize,
    get: impl Fn(usize) -> Record,
) -> Result<Staged> {
    let mut tokens = Vec::with_capacity(index.records * SEQUENCE);
    let mut legal = Vec::with_capacity(index.records * SEQUENCE);
    let mut context_ids = Vec::with_capacity(index.records);
    for i in 0..index.records {
        let r = get(i);
        ensure!(
            r.context < index.contexts.len(),
            "record {i} names context {} outside the corpus",
            r.context
        );
        ensure!(
            r.legal
                .iter()
                .all(|&id| (id as u16 as usize) < index.legal_sets.len()),
            "record {i} names a legal set outside the corpus"
        );
        // LibTorch's short tensor is signed. Preserve all sixteen identifier
        // bits and restore unsigned values only after gathering a minibatch.
        tokens.extend(r.tokens.map(|id| id as u16 as i16));
        legal.extend(r.legal.map(|id| id as u16 as i16));
        context_ids.push(r.context as i64);
    }
    let mut masks = vec![0u8; index.legal_sets.len() * vocabulary_len];
    for (i, set) in index.legal_sets.iter().enumerate() {
        for &t in set {
            ensure!(
                t < vocabulary_len,
                "legal set {i} names token {t} outside the vocabulary"
            );
            masks[i * vocabulary_len + t] = 1;
        }
    }
    Ok(Staged {
        tokens,
        legal,
        context_ids,
        masks,
    })
}

/// Identity of a program: its supervised token sequence.
pub fn program_identity(record: &Record) -> String {
    let length = record.length.min(SEQUENCE);
    hash(
        &record.tokens[..length]
            .iter()
            .flat_map(|&t| (t as u16).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// Distinct programs among `ids`.
pub fn unique_programs(
    ids: impl IntoIterator<Item = usize>,
    get: impl Fn(usize) -> Record,
) -> usize {
    ids.into_iter()
        .map(|i| program_identity(&get(i)))
        .collect::<BTreeSet<_>>()
        .len()
}

/// Available records, groups and families per admitted partition.
pub fn availability(index: &Index) -> Value {
    let mut partitions = BTreeMap::<&str, (usize, usize, BTreeSet<&str>)>::new();
    let mut grouped = 0usize;
    for group in &index.groups {
        let entry = partitions.entry(&group.partition).or_default();
        entry.0 += 1;
        entry.1 += group.records.len();
        entry.2.insert(&group.family);
        grouped += group.records.len();
    }
    json!({
        "partitions": partitions
            .into_iter()
            .map(|(partition, (groups, records, families))| {
                (partition.to_string(), json!({"groups":groups,"records":records,"families":families.len()}))
            })
            .collect::<serde_json::Map<_, _>>(),
        "records_in_groups": grouped,
        "records_outside_groups": index.records.saturating_sub(grouped),
    })
}

/// The frozen manifest of one block's resolved inputs. `unique_programs`
/// is recorded only when it was actually computed; it is never estimated.
pub fn freeze(
    index: &Index,
    index_sha256: &str,
    vocabulary_sha256: &str,
    unique_programs: Option<usize>,
) -> Result<Value> {
    ensure!(
        vocabulary_sha256 == index.vocabulary_sha256,
        "frozen vocabulary differs from the corpus"
    );
    Ok(json!({
        "schema": SCHEMA,
        "corpus": {
            "index_sha256": index_sha256,
            "schema": index.schema,
            "qualification": index.qualification,
            "records_sha256": index.records_sha256,
            "records": index.records,
            "record_bytes": index.record_bytes,
        },
        "vocabulary_sha256": vocabulary_sha256,
        "contexts": {
            "declared_index_sha256": index.context_index_sha256,
            "rows_sha256": context_identity(index)?,
            "count": index.contexts.len(),
        },
        "legal_sets_sha256": hash(&serde_json::to_vec(&index.legal_sets)?),
        "groups_sha256": hash(&serde_json::to_vec(&index.groups)?),
        "availability": availability(index),
        "unique_programs": match unique_programs {
            Some(count) => json!({"status":"computed","count":count}),
            None => json!({"status":"not_computed_at_launch","count":null}),
        },
        "exposure_units": {
            "available_records": "corpus records in admitted groups of this frozen corpus",
            "unique_programs": "distinct supervised token sequences among available records",
            "presentations": "sampled rows; a record may be presented repeatedly across comparisons and steps; a repeat is exposure of the same record, never new data",
            "presentation_ledger": super::consumption::SCHEMA,
        },
        "revocation": "not_available_in_this_source",
    }))
}

pub fn digest(frozen: &Value) -> Result<String> {
    Ok(hash(&serde_json::to_vec(frozen)?))
}

/// Freeze the per-head eligibility projection and sampled occurrence identities
/// alongside the already frozen corpus inputs. This is additive: legacy input
/// manifests and their digests are left byte-for-byte unchanged.
pub fn freeze_intent_closure(
    frozen_inputs: &Value,
    per_head_eligibility: &Value,
    fresh_occurrences: &[String],
    replay_occurrences: &[String],
) -> Result<Value> {
    use anyhow::ensure;
    ensure!(
        frozen_inputs["schema"].as_str() == Some(SCHEMA),
        "intent closure requires jope.training_inputs.v1"
    );
    ensure!(
        per_head_eligibility.is_object(),
        "per-head eligibility must be an object"
    );
    ensure!(
        !per_head_eligibility.as_object().unwrap().is_empty(),
        "per-head eligibility is empty"
    );
    for (head, proof) in per_head_eligibility.as_object().unwrap() {
        ensure!(!head.is_empty(), "eligibility head identity is empty");
        ensure!(
            proof.as_str().is_some_and(valid_sha256),
            "eligibility proof for {head} is not a SHA256 digest"
        );
    }
    validate_occurrences("fresh", fresh_occurrences)?;
    validate_occurrences("replay", replay_occurrences)?;
    let fresh = fresh_occurrences.iter().collect::<BTreeSet<_>>();
    ensure!(
        replay_occurrences.iter().all(|id| !fresh.contains(id)),
        "an occurrence cannot be both fresh and replay"
    );
    Ok(json!({
        "schema": "jope.training_intent_inputs/v1",
        "base_inputs": frozen_inputs,
        "base_inputs_sha256": digest(frozen_inputs)?,
        "per_head_eligibility": per_head_eligibility,
        "fresh_occurrences": fresh_occurrences,
        "replay_occurrences": replay_occurrences,
    }))
}

fn validate_occurrences(kind: &str, occurrences: &[String]) -> Result<()> {
    use anyhow::ensure;
    ensure!(
        !occurrences.is_empty(),
        "{kind} occurrence identity is empty"
    );
    let unique = occurrences.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == occurrences.len(),
        "duplicate {kind} occurrence identity"
    );
    ensure!(
        occurrences.iter().all(|id| !id.trim().is_empty()),
        "empty {kind} occurrence identity"
    );
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
