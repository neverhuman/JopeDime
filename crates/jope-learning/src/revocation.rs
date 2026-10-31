//! W12-T05 at the live consumers: revocation lookup.
//!
//! The revocation authority is the claims and revocation graph (`jope.core-ip-claims/v1`), verified
//! by [`jope_records::claims`]. Revoking an evidence node there revokes exactly its descendants.
//! A live consumer needs that closure as digests, held on its own host, so that a later edit of
//! the document can never silently un-revoke anything:
//!
//! - [`import`] verifies the document and appends every newly revoked node to this state tree's
//!   append-only, hash-chained `revocations` ledger. Each record carries:
//!   - the digest the node's bytes were pinned at;
//!   - the revocation that reaches it: issuer, cause, effective time, sequence, and the node it
//!     names;
//!   - the sha256 of the claims document it came from.
//!
//!   Nothing ever leaves the ledger. Requalification needs new evidence under a new identity,
//!   never a cleared flag.
//! - [`Revocations::load`] reads that ledger, and [`Revocations::refuse`] refuses a subject whose
//!   dependency closure names a revoked digest.
//!
//! The consumers:
//! - **W3 teacher admission** (`eligibility`, which `qualify`, `hwm::ingest`, `freeze_teaching`,
//!   `revalidate_qualification` and `revalidate_replay` all recompute through). It covers the
//!   record, the freeze, the experiment and window, and every fold's predictions, fitted state,
//!   request, job, result and receipts. It also covers the teaching context receipt, its fitted
//!   state and output, and every parent receipt with its own fitted state and output.
//! - **HWM ingest** (`hwm::ingest`): a candidate whose record is revoked is rejected as `revoked`.
//! - **W8 publication** (`progress::publish::prepare`): a commit whose closure names a revoked
//!   object (by domain digest or byte sha256) or blob never moves the run pointer. The product
//!   driver refuses to append a revoked item at all.
//!
//! An absent or empty ledger means only that no revocation was imported on this host, which
//! [`Status`] reports. It does not mean that nothing is revoked anywhere.

use crate::canon::is_hex64;
use crate::ledger::{BatchReceipt, Ledger};
use crate::state::State;
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Schema identifier for record values.
pub const RECORD_SCHEMA: &str = "jope.revocation-record/v1";
/// The ledger (under the state tree) that holds imported revocations.
pub const LEDGER: &str = "revocations";

/// One revoked evidence node as a consumer sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevocationRecord {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Node recorded for this revocation record.
    pub node: String,
    /// The digest the node's bytes were pinned at: what consumers match.
    pub sha256: String,
    /// The node the revocation names: this node, or its revoked ancestor.
    pub revoked_node: String,
    /// Issuer recorded for this revocation record.
    pub issuer: String,
    /// Cause recorded for this revocation record.
    pub cause: String,
    /// Effective utc recorded for this revocation record.
    pub effective_utc: String,
    /// Sequence recorded for this revocation record.
    pub sequence: i64,
    /// sha256 of the claims document bytes the record was imported from.
    pub claims_sha256: String,
}

/// What the local ledger holds.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Status {
    /// Revoked digests known on this host.
    pub revoked: u64,
    /// The ledger batch last appended, if any.
    pub head: Option<String>,
}

/// The revocations imported on this host, by digest.
#[derive(Clone, Debug, Default)]
pub struct Revocations {
    by_digest: BTreeMap<String, RevocationRecord>,
    status: Status,
}

impl Revocations {
    /// Read the host's revocation ledger. The chain must verify; an absent ledger is empty.
    pub fn load(state: &State) -> Result<Revocations> {
        let ledger = Ledger::open(state.root(), LEDGER);
        let mut out = Revocations::default();
        if !ledger.root().exists() {
            return Ok(out);
        }
        ledger.require_intact()?;
        for (i, event) in ledger.events()?.into_iter().enumerate() {
            let record: RevocationRecord = serde_json::from_value(event)
                .with_context(|| format!("revocation ledger event #{i} does not parse"))?;
            ensure!(
                record.schema == RECORD_SCHEMA && is_hex64(&record.sha256),
                "revocation ledger event #{i} is not a {RECORD_SCHEMA} record"
            );
            out.by_digest.entry(record.sha256.clone()).or_insert(record);
        }
        out.status = Status {
            revoked: out.by_digest.len() as u64,
            head: ledger.head()?.map(|h| h.path),
        };
        Ok(out)
    }

    /// Performs the status operation.
    pub fn status(&self) -> &Status {
        &self.status
    }

    /// Returns whether empty.
    pub fn is_empty(&self) -> bool {
        self.by_digest.is_empty()
    }

    /// All locally imported revoked digests, in stable order. This iterator cannot mutate
    /// or clear revocation state and makes no claim about revocations not imported here.
    pub fn digests(&self) -> impl Iterator<Item = &str> + '_ {
        self.by_digest.keys().map(String::as_str)
    }

    /// The revocation that covers `digest`, if any.
    pub fn revoked(&self, digest: &str) -> Option<&RevocationRecord> {
        self.by_digest.get(digest)
    }

    /// Refuse `what` when any of `digests` (its dependency closure) is revoked. The error names
    /// the revoked node, the revocation that reaches it and the requalification rule.
    pub fn refuse<'a>(&self, what: &str, digests: impl IntoIterator<Item = &'a str>) -> Result<()> {
        if self.by_digest.is_empty() {
            return Ok(());
        }
        for digest in digests {
            if let Some(r) = self.by_digest.get(digest) {
                bail!(
                    "RevokedEvidence: {what} depends on revoked evidence {} ({digest}), revoked by \
                     {} at {} (sequence {}{}): {}. Requalification needs new evidence under a new \
                     identity",
                    r.node,
                    r.issuer,
                    r.effective_utc,
                    r.sequence,
                    if r.revoked_node == r.node {
                        String::new()
                    } else {
                        format!(", through its ancestor {}", r.revoked_node)
                    },
                    r.cause
                );
            }
        }
        Ok(())
    }
}

/// What an import would append, or appended.
#[derive(Clone, Debug, Serialize)]
pub struct Import {
    /// SHA-256 digest identifying the associated claims.
    pub claims_sha256: String,
    /// Revoked nodes the document names, in total.
    pub revoked: u64,
    /// Newly revoked nodes: appended with `apply`, otherwise only reported.
    pub new: Vec<RevocationRecord>,
    /// Nodes this host holds as revoked that the document no longer revokes. They stay revoked
    /// here: an edited document never un-revokes anything.
    pub no_longer_in_document: Vec<String>,
    /// Applied recorded for this import.
    pub applied: bool,
    /// Ledger batch receipt produced by the operation, if one was appended.
    pub batch: Option<BatchReceipt>,
}

/// Import the revocation closure of the claims record at `claims` (repository evidence paths
/// relative to `root`). The record must verify ([`jope_records::claims::revoked_evidence`]). A
/// node this host already holds under another digest is refused: the document would have
/// re-pinned revoked evidence. With `apply` false nothing is written and no lock is taken.
pub fn import(state: &State, claims: &Path, root: Option<&Path>, apply: bool) -> Result<Import> {
    let _lock = if apply {
        Some(state.lock(LEDGER)?)
    } else {
        None
    };
    let closure = jope_records::claims::revoked_evidence(claims, root, false)
        .map_err(|e| anyhow!("{}: {e}", claims.display()))?;
    let existing = Revocations::load(state)?;
    let by_node: BTreeMap<&str, &RevocationRecord> = existing
        .by_digest
        .values()
        .map(|r| (r.node.as_str(), r))
        .collect();
    let mut new = Vec::new();
    for r in &closure.revoked {
        ensure!(
            is_hex64(&r.sha256),
            "revoked node {} has no sha256 digest",
            r.node
        );
        match by_node.get(r.node.as_str()) {
            Some(held) => ensure!(
                held.sha256 == r.sha256,
                "revoked node {} is held here at {} but the document pins it at {}; refusing to \
                 re-pin revoked evidence. Nothing was written",
                r.node,
                held.sha256,
                r.sha256
            ),
            None if existing.revoked(&r.sha256).is_some() => {}
            None => new.push(RevocationRecord {
                schema: RECORD_SCHEMA.into(),
                node: r.node.clone(),
                sha256: r.sha256.clone(),
                revoked_node: r.revoked_node.clone(),
                issuer: r.issuer.clone(),
                cause: r.cause.clone(),
                effective_utc: r.effective_utc.clone(),
                sequence: r.sequence,
                claims_sha256: closure.claims_sha256.clone(),
            }),
        }
    }
    let named: BTreeSet<&str> = closure.revoked.iter().map(|r| r.node.as_str()).collect();
    let no_longer_in_document = by_node
        .keys()
        .filter(|n| !named.contains(*n))
        .map(|n| n.to_string())
        .collect();
    let batch = if apply && !new.is_empty() {
        let events = new
            .iter()
            .map(|r| serde_json::to_value(r).map_err(Into::into))
            .collect::<Result<Vec<Value>>>()?;
        Some(Ledger::open(state.root(), LEDGER).append(
            &events,
            json!({"source": "claims", "claims": claims, "claims_sha256": closure.claims_sha256}),
        )?)
    } else {
        None
    };
    Ok(Import {
        claims_sha256: closure.claims_sha256,
        revoked: closure.revoked.len() as u64,
        new,
        no_longer_in_document,
        applied: apply,
        batch,
    })
}

/// Builders for tests that need a verified claims record with revocations.
#[cfg(test)]
pub(crate) mod testing {
    use super::import;
    use crate::state::State;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    /// A repository-local claims record under `root`: one evidence node per
    /// `(name, bytes, parents)`, pinned at the sha256 of `bytes`, and `revoked` revoked in
    /// sequence order.
    pub(crate) fn claims(
        root: &Path,
        nodes: &[(&str, &[u8], &[&str])],
        revoked: &[&str],
    ) -> PathBuf {
        let mut evidence = serde_json::Map::new();
        std::fs::create_dir_all(root.join("evidence")).unwrap();
        for (name, bytes, parents) in nodes {
            let path = format!("evidence/{name}");
            std::fs::write(root.join(&path), bytes).unwrap();
            evidence.insert(
                (*name).into(),
                json!({"kind": "synthetic-fixture", "location": "repository", "path": path,
                    "sha256": crate::canon::sha256_hex(bytes), "source_commit": "a".repeat(40),
                    "subject": "fixture subject", "scope": "fixture", "depends_on": parents}),
            );
        }
        let revocations: Vec<_> = revoked
            .iter()
            .enumerate()
            .map(|(i, node)| {
                json!({"node": node, "issuer": "reviewer", "cause": "fixture cause",
                    "effective_utc": "2026-09-26T12:00:00Z", "sequence": i + 1})
            })
            .collect();
        let path = root.join("docs/internal/core-ip/claims/claims.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({"schema": "jope.core-ip-claims/v1",
                "evidence": evidence, "revocations": revocations, "claims": []}))
            .unwrap(),
        )
        .unwrap();
        path
    }

    /// Write [`claims`] under `root` and import it into `state`: first as a plan, which must
    /// write nothing, then applied.
    pub(crate) fn revoke(
        state: &State,
        root: &Path,
        nodes: &[(&str, &[u8], &[&str])],
        revoked: &[&str],
    ) {
        let path = claims(root, nodes, revoked);
        let ledger = state.root().join(super::LEDGER);
        let existed = ledger.exists();
        let plan = import(state, &path, Some(root), false).unwrap();
        assert!(!plan.applied && plan.batch.is_none());
        assert_eq!(ledger.exists(), existed, "an import plan wrote the ledger");
        import(state, &path, Some(root), true).unwrap();
    }
}
