//! Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub mod cost;
pub mod reader;

use crate::canon::{canonical_json, is_hex64, sha256_hex};
use anyhow::{ensure, Context, Result};
use jope_records::custody_archive::{self, Blob, MAX_MEMBERS, MAX_PAYLOAD};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
};

/// Configured upper or lower bound for max manifest.
pub const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
/// Constant identifying the byte cap.
pub const BYTE_CAP: u64 = 25_000_000_000;
/// Constant identifying the hot cap.
pub const HOT_CAP: u64 = 1_000_000_000_000;
/// Constant identifying the put cap.
pub const PUT_CAP: u64 = 1_500_000;
/// Constant identifying the keeper facts.
pub const KEEPER_FACTS: [&str; 4] = [
    "new_context_best",
    "parity_failure",
    "cited",
    "sealed_salt_audit",
];

/// Performs the unsigned operation and returns an error if it cannot be completed.
pub fn unsigned(value: &Value, name: &str) -> Result<u64> {
    value
        .as_u64()
        .with_context(|| format!("{name} must be a nonnegative integer"))
}
/// Performs the identity operation and returns an error if it cannot be completed.
pub fn identity(row: &Value) -> Result<[String; 3]> {
    let mut result = [String::new(), String::new(), String::new()];
    for (i, field) in ["bucket", "key", "version_id"].iter().enumerate() {
        result[i] = row[*field]
            .as_str()
            .filter(|s| !s.is_empty())
            .context("complete object version identity")?
            .to_owned();
    }
    Ok(result)
}

/// Performs the classify operation and returns an error if it cannot be completed.
pub fn classify(row: &Value, facts: Option<&Value>) -> Result<Value> {
    let id = identity(row)?;
    unsigned(&row["bytes"], "object bytes")?;
    let latest = row["is_latest"].as_bool().context("latest-version flag")?;
    let key = &id[1];
    let role = key.rsplit('/').next().unwrap_or_default();
    let mut kind = "keeper";
    let mut reason = "unmeasured_keeper_or_hold_facts";
    if id[2] == "null" {
        reason = "immutable_version_unavailable";
    } else if !latest {
        reason = "noncurrent_version_review";
    } else if key
        .split('/')
        .any(|p| ["sealed", "judge", "checkpoints", "gp-hwm", "provenance"].contains(&p))
    {
        reason = "protected_evidence";
    } else if [
        "score_records",
        "evaluation",
        "rust_parity",
        "input",
        "program",
        "rust_source",
        "record.json",
    ]
    .contains(&role)
    {
        reason = "permanent_ledger_or_small_artifact";
    } else if role != "rust_executable" {
        reason = "unknown_role";
    } else if let Some(facts) = facts.filter(|f| !f.is_null()) {
        ensure!(facts.is_object(), "keeper facts object");
        ensure!(
            facts["identity"] == json!(id),
            "keeper facts bind the exact physical version"
        );
        let measured = facts["legal_hold_checked"] == true
            && facts["retention_checked"] == true
            && KEEPER_FACTS.iter().all(|k| facts[*k].is_boolean())
            && facts["audit_salt_commitment"]
                .as_str()
                .is_some_and(is_hex64)
            && facts
                .as_object()
                .is_some_and(|o| o.contains_key("retention_until"));
        if measured {
            if facts["legal_hold"] != "OFF" || !facts["retention_until"].is_null() {
                reason = "protected_or_unknown_hold";
            } else if KEEPER_FACTS.iter().any(|k| facts[*k] == true) {
                reason = "verified_keeper_fact";
            } else {
                kind = "other";
                reason = "complete_negative_keeper_facts";
            }
        }
    }
    let mut result = row.as_object().context("object version row")?.clone();
    result.extend(
        json!({"role":role,"retention_class":kind,"reason":reason,
        "proposed_retention_days":if kind == "other" {Some(14)} else {None},
        "applied":false,"qualified":false})
        .as_object()
        .context("classification")?
        .clone(),
    );
    Ok(Value::Object(result))
}

/// Unknown counters and exhausted limits refuse a recommendation; no service is controlled.
pub fn budget_status(values: &Value) -> Result<Value> {
    let mut unknown = Vec::new();
    let mut alarm = Vec::new();
    let mut exhausted = Vec::new();
    for (name, cap) in [
        ("new_version_bytes", BYTE_CAP),
        ("raw_hot_bytes", HOT_CAP),
        ("puts", PUT_CAP),
    ] {
        if values[name].is_null() {
            unknown.push(name);
        } else {
            let n = unsigned(&values[name], name)?;
            if n >= cap * 4 / 5 {
                alarm.push(name);
            }
            if n >= cap {
                exhausted.push(name);
            }
        }
    }
    Ok(
        json!({"caps":{"new_version_bytes":BYTE_CAP,"raw_hot_bytes":HOT_CAP,"puts":PUT_CAP},
        "values":{"new_version_bytes":values["new_version_bytes"],"raw_hot_bytes":values["raw_hot_bytes"],"puts":values["puts"]},
        "admit_new_scoring":unknown.is_empty() && exhausted.is_empty(),
        "unknown":unknown,"alarm":alarm,"exhausted":exhausted,"control_applied":false}),
    )
}

/// Performs the bounded read operation and returns an error if it cannot be completed.
pub fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut raw = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut raw)?;
    ensure!(raw.len() as u64 <= limit, "bounded input bytes");
    Ok(raw)
}

fn manifest(raw: &[u8]) -> Result<(Value, Vec<Blob>)> {
    ensure!(raw.len() as u64 <= MAX_MANIFEST, "bounded manifest bytes");
    let index: Value = serde_json::from_slice(raw)?;
    ensure!(index.is_object(), "pack manifest object");
    ensure!(
        index["schema"] == "jope.custody-local-pack/v2"
            && index["qualified"] == false
            && index["remote_publication"] == false,
        "closed non-qualifying pack"
    );
    let refs = index["refs"]
        .as_array()
        .context("bounded physical reference roster")?;
    let blobs = index["blobs"].as_array().context("bounded blob roster")?;
    ensure!(
        !refs.is_empty() && refs.len() <= MAX_MEMBERS,
        "bounded physical reference roster"
    );
    ensure!(
        !blobs.is_empty() && blobs.len() <= MAX_MEMBERS,
        "bounded blob roster"
    );
    ensure!(
        index["pack_sha256"].as_str().is_some_and(is_hex64),
        "content-addressed pack SHA"
    );
    let mut roster = Vec::new();
    let mut expected = BTreeMap::new();
    for b in blobs {
        let sha = b["sha256"].as_str().context("safe content member")?;
        ensure!(
            is_hex64(sha) && b["name"] == format!("members/{sha}"),
            "safe content member"
        );
        let bytes = unsigned(&b["bytes"], "blob bytes")?;
        ensure!(
            expected.insert(sha.to_owned(), bytes).is_none(),
            "duplicate blob declaration"
        );
        roster.push(Blob {
            name: format!("members/{sha}"),
            sha256: sha.to_owned(),
            bytes,
        });
    }
    let mut physical = BTreeSet::new();
    let mut used = BTreeSet::new();
    for row in refs {
        let id = identity(row)?;
        ensure!(
            id[2] != "null" && physical.insert(id),
            "unique immutable captured reference"
        );
        let sha = row["sha256"]
            .as_str()
            .context("every reference joins a verified blob")?;
        let bytes = unsigned(&row["bytes"], "reference bytes")?;
        ensure!(
            expected.get(sha) == Some(&bytes),
            "every reference joins a verified blob"
        );
        used.insert(sha.to_owned());
    }
    ensure!(
        used.len() == expected.len(),
        "every blob has a source reference"
    );
    Ok((index, roster))
}

/// Report the hash of the exact manifest bytes verified, even if its pathname is replaced.
pub fn verify_manifest_bytes(raw: &[u8], directory: &Path) -> Result<Value> {
    let (index, roster) = manifest(raw)?;
    let sha = index["pack_sha256"].as_str().context("pack SHA")?;
    verify_parsed(
        File::open(directory.join(format!("{sha}.tar.gz")))?,
        raw,
        &index,
        &roster,
    )
}

/// Performs the verify pack operation and returns an error if it cannot be completed.
pub fn verify_pack(path: &Path, raw: &[u8]) -> Result<Value> {
    let (index, roster) = manifest(raw)?;
    verify_parsed(File::open(path)?, raw, &index, &roster)
}

fn verify_parsed(file: File, raw: &[u8], index: &Value, roster: &[Blob]) -> Result<Value> {
    custody_archive::verify(
        file,
        index["pack_sha256"].as_str().context("pack SHA")?,
        roster,
    )?;
    Ok(
        json!({"physical_refs":index["refs"].as_array().context("refs")?.len(),
        "verified_blobs":roster.len(),"all_member_shas_verified":true,
        "manifest_sha256":sha256_hex(raw),"qualified":false,"remote_calls":0}),
    )
}
/// Performs the verify saved operation and returns an error if it cannot be completed.
pub fn verify_saved(path: &Path) -> Result<Value> {
    let raw = bounded_read(path, MAX_MANIFEST).context("bounded manifest bytes")?;
    verify_manifest_bytes(&raw, path.parent().context("manifest directory")?)
}

#[derive(Clone)]
/// `CapturedMember` is a record for the captured member contract in this module: Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub struct CapturedMember {
    /// Row recorded for this captured member.
    pub row: Value,
    /// Body recorded for this captured member.
    pub body: Arc<[u8]>,
}

/// Fault hooks run before the real operation; they cannot skip persistence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistStep {
    /// The `Write` case of `PersistStep`.
    Write,
    /// The `FileSync` case of `PersistStep`.
    FileSync,
    /// The `ParentSync` case of `PersistStep`.
    ParentSync,
}

/// Performs the durable file operation and returns an error if it cannot be completed.
pub fn durable_file(
    path: &Path,
    raw: &[u8],
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<()> {
    hook(path, PersistStep::Write)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(raw)?;
    hook(path, PersistStep::FileSync)?;
    file.sync_all()?;
    hook(path, PersistStep::ParentSync)?;
    File::open(path.parent().context("parent directory")?)?.sync_all()?;
    Ok(())
}

/// Performs the pack local operation and returns an error if it cannot be completed.
pub fn pack_local(destination: &Path, members: &[CapturedMember]) -> Result<Value> {
    pack_with(destination, members, &mut |_, _| Ok(()))
}

fn durable_directory(
    path: &Path,
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<()> {
    fs::DirBuilder::new().mode(0o700).create(path)?;
    let result = (|| {
        hook(path, PersistStep::ParentSync)?;
        File::open(path.parent().context("directory parent")?)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        mark_aborted(path);
    }
    result
}

/// Performs the pack with operation and returns an error if it cannot be completed.
pub fn pack_with(
    destination: &Path,
    members: &[CapturedMember],
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<Value> {
    ensure!(
        !members.is_empty() && members.len() <= MAX_MEMBERS,
        "bounded physical reference roster"
    );
    let mut refs = Vec::new();
    let mut physical = BTreeSet::new();
    let mut blobs = BTreeMap::new();
    let mut unique_bytes = 0u64;
    for m in members {
        let id = identity(&m.row)?;
        ensure!(
            physical.insert(id.clone()),
            "duplicate physical object version"
        );
        ensure!(id[2] != "null", "immutable version required for capture");
        ensure!(
            unsigned(&m.row["bytes"], "member bytes")? == m.body.len() as u64,
            "captured size mismatch"
        );
        let sha = sha256_hex(&m.body);
        if let Some(old) = blobs.get(&sha) {
            ensure!(old == &m.body, "content digest collision");
        } else {
            unique_bytes = unique_bytes
                .checked_add(m.body.len() as u64)
                .context("payload overflow")?;
            ensure!(unique_bytes <= MAX_PAYLOAD, "local pack raw-byte cap");
            blobs.insert(sha.clone(), m.body.clone());
        }
        refs.push(json!({"bucket":id[0],"key":id[1],"version_id":id[2],"bytes":m.body.len(),"sha256":sha}));
    }
    refs.sort_by_key(|r| identity(r).unwrap_or_default());
    let raw = custody_archive::encode(&blobs)?;
    let pack_sha = sha256_hex(&raw);
    let index = json!({"schema":"jope.custody-local-pack/v2","refs":refs,
        "blobs":blobs.iter().map(|(h,b)|json!({"name":format!("members/{h}"),"sha256":h,"bytes":b.len()})).collect::<Vec<_>>(),
        "pack_sha256":pack_sha,"qualified":false,"remote_publication":false});
    durable_directory(destination, hook)?;
    // Retain incomplete bytes and an abort witness on every caught persistence/readback error.
    let result = (|| {
        durable_file(&destination.join(format!("{pack_sha}.tar.gz")), &raw, hook)?;
        let manifest_raw = canonical_json(&index)?;
        let verified = verify_manifest_bytes(&manifest_raw, destination)?;
        durable_file(&destination.join("manifest.json"), &manifest_raw, hook)?;
        Ok(
            json!({"manifest_sha256":verified["manifest_sha256"],"physical_refs":refs.len(),
            "unique_blobs":blobs.len(),"pack_bytes":raw.len(),"qualified":false,"remote_writes":0}),
        )
    })();
    if result.is_err() {
        mark_aborted(destination);
    }
    result
}

fn mark_aborted(directory: &Path) {
    // This witness is defensive. Failure to persist it never turns an error into success.
    let _ = durable_file(
        &directory.join("ABORTED"),
        b"incomplete custody publication\n",
        &mut |_, _| Ok(()),
    );
}

/// Verify completion with the receipt SHA returned by a successful publisher, not a directory guess.
pub fn verify_complete(directory: &Path, receipt_sha: &str) -> Result<Value> {
    ensure!(
        is_hex64(receipt_sha) && !directory.join("ABORTED").exists(),
        "completed publication identity required"
    );
    let raw = bounded_read(&directory.join("receipt.json"), MAX_MANIFEST)?;
    ensure!(
        sha256_hex(&raw) == receipt_sha,
        "terminal receipt SHA mismatch"
    );
    let receipt: Value = serde_json::from_slice(&raw)?;
    ensure!(
        receipt["schema"] == "jope.custody-v2-local-dry-run/v1"
            && receipt["complete"] == true
            && receipt["qualified"] == false
            && receipt["remote_calls"] == 0
            && receipt["remote_mutations"] == 0,
        "nonqualifying completed receipt required"
    );
    let ledger = bounded_read(&directory.join("object-ledger.json"), 32 * 1024 * 1024)?;
    ensure!(
        receipt["ledger_sha256"] == sha256_hex(&ledger),
        "ledger SHA mismatch"
    );
    let verified = verify_saved(&directory.join("pack/manifest.json"))?;
    ensure!(
        receipt["pack_sample"]["manifest_sha256"] == verified["manifest_sha256"],
        "manifest receipt join"
    );
    Ok(
        json!({"receipt_sha256":receipt_sha,"readback":verified,"complete":true,"qualified":false,"remote_calls":0}),
    )
}

/// Local capture only. Budget status is reported and never authorizes scoring or storage policy.
pub struct Pricing<'a> {
    /// Requests recorded for this pricing.
    pub requests: &'a [Value],
    /// Rates recorded for this pricing.
    pub rates: &'a Value,
    /// As of recorded for this pricing.
    pub as_of: &'a str,
}
/// `DryRunRequest` is a record for the dry run request contract in this module: Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub struct DryRunRequest<'a> {
    /// Number of rows represented by this value.
    pub rows: &'a [Value],
    /// Facts recorded for this dry run request.
    pub facts: &'a [Value],
    /// Budgets recorded for this dry run request.
    pub budgets: &'a Value,
    /// Members recorded for this dry run request.
    pub members: &'a [CapturedMember],
    /// Optional input sha value; `None` means it is not available.
    pub input_sha: Option<&'a str>,
    /// Optional pricing value; `None` means it is not available.
    pub pricing: Option<Pricing<'a>>,
}

/// Performs the dry run operation and returns an error if it cannot be completed.
pub fn dry_run(
    destination: &Path,
    request: &DryRunRequest<'_>,
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<Value> {
    let DryRunRequest {
        rows,
        facts,
        budgets,
        members,
        input_sha,
        pricing,
    } = request;
    if let Some(sha) = input_sha {
        ensure!(is_hex64(sha), "pinned input SHA required");
    }
    let estimated_cost = pricing
        .as_ref()
        .map(|p| cost::report(rows, p.requests, p.rates, p.as_of))
        .transpose()?;
    let mut known = BTreeMap::new();
    for fact in *facts {
        let id: [String; 3] = serde_json::from_value(fact["identity"].clone())?;
        ensure!(known.insert(id, fact).is_none(), "duplicate keeper facts");
    }
    let mut seen = BTreeSet::new();
    let mut ledger = Vec::new();
    for row in *rows {
        let id = identity(row)?;
        ensure!(seen.insert(id.clone()), "duplicate ledger physical version");
        ledger.push(classify(row, known.get(&id).copied())?);
    }
    ensure!(
        known.keys().all(|id| seen.contains(id)),
        "facts outside observed ledger"
    );
    for member in *members {
        let id = identity(&member.row)?;
        let original = rows
            .iter()
            .find(|r| identity(r).ok().as_ref() == Some(&id))
            .context("captured version outside observed ledger")?;
        ensure!(
            original["bytes"] == member.row["bytes"],
            "capture joins ledger size"
        );
    }
    let guard = budget_status(budgets)?;
    let ledger_raw = canonical_json(&ledger)?;
    ensure!(ledger_raw.len() <= 32 * 1024 * 1024, "bounded ledger bytes");
    durable_directory(destination, hook)?;
    let result = (|| {
        durable_file(&destination.join("object-ledger.json"), &ledger_raw, hook)?;
        let pack = pack_with(&destination.join("pack"), members, hook)?;
        let receipt = json!({"schema":"jope.custody-v2-local-dry-run/v1","complete":true,"qualified":false,
            "remote_calls":0,"remote_mutations":0,"ledger_sha256":sha256_hex(&ledger_raw),"listed_versions":rows.len(),
            "unpacked_versions":rows.len().checked_sub(members.len()).context("sample count")?,
            "entire_prefix_packed":false,"point_in_time_snapshot":false,"pack_sample":pack,"guard":guard,
            "input_sha256":input_sha,"cost":estimated_cost,"actual_billed_usd":null});
        let raw = canonical_json(&receipt)?;
        durable_file(&destination.join("receipt.json"), &raw, hook)?;
        let pin = sha256_hex(&raw);
        verify_complete(destination, &pin)?;
        Ok(json!({"receipt":receipt,"receipt_sha256":pin}))
    })();
    if result.is_err() {
        mark_aborted(destination);
    }
    result
}
