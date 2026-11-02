//! W8-T03: dependency-first conditional publication.
//!
//! A run's committed state is its `HEAD` pointer ([`Head`], at `progress/runs/<run>/HEAD.json`)
//! naming a [`Commit`]. Everything a commit reaches is content-addressed and immutable. A commit
//! is published in three ordered steps, and the pointer is always last:
//!
//! 1. **Dependencies, then manifests.** The commit is appended to the [`Outbox`] behind
//!    everything it names. The outbox delivers in sequence order: blobs and other dependencies
//!    first, then the objects that name them, then the commit itself. Every write is
//!    create-only.
//! 2. **Verification.** The commit's full closure is read back from the remote and verified by
//!    [`verify_closure`]:
//!    - every object is present, canonical and hashes to its digest;
//!    - every blob is present and hashes to its sha256;
//!    - the graph rules hold ([`check_rules`]). These cover run membership, commit height and
//!      lineage, checkpoint and exposure contiguity, one consecutive chain of update indexes
//!      (no forked or repeated update), and that decisions and evidence come only from
//!      completed attempts;
//!    - every HWM decision the commit adds is one this host's own HWM ledger recorded
//!      ([`HwmDecision::verify_against`]);
//!    - nothing in the closure, ancestors included, is revoked evidence
//!      ([`crate::revocation`]). This is checked on the closure as the outbox recorded it,
//!      before anything is delivered.
//!
//!    If an object is missing remotely but the local outbox holds it, it is uploaded again
//!    create-only from its verified payload. Any other defect blocks publication.
//! 3. **Conditional advance.** The pointer is read once, together with its version. It must
//!    name exactly the commit's parent, or be absent for a first commit. It is then replaced
//!    conditionally on that version ([`Remote::put_pointer`]).
//!    - A pointer naming anything else is a [`Publication::Conflict`], and nothing is written.
//!    - A backend that cannot express the precondition fails. There is no unconditional
//!      fallback.
//!
//! A losing commit is never overwritten or deleted. It stays on the remote as an immutable,
//! unreferenced object, and in the local outbox, so the divergence can be reconciled. The winner
//! is never displaced.
//!
//! Corpora and checkpoints become reusable only through [`reusable`]. It verifies the pointer,
//! the commit it names and that commit's whole closure before returning anything.

use crate::canon::sha256_hex;
use crate::ledger::Ledger;
use crate::progress::inventory::{
    decode_as, head_key, object_key, BlobRef, CandidateEvidence, Checkpoint, Commit, Decoded,
    Dependency, Encoded, Event, EventBody, Exposure, Head, HwmDecision, Inventoried, Kind,
    ObjectRef, RunManifest, HEAD_SCHEMA,
};
use crate::progress::outbox::Outbox;
use crate::progress::remote::Remote;
use crate::revocation::Revocations;
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::store::ObservedObject;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod publication;
pub use publication::{advance, prepare, publish, reusable, Plan, Prepared, Publication, Reusable};

/// Upper bound on the items one closure walk visits.
pub const MAX_CLOSURE_ITEMS: usize = 1 << 20;

/// What is wrong with a remote closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DefectKind {
    /// Not on the remote.
    Missing,
    /// On the remote, but not the bytes its reference names.
    Corrupt,
    /// Every object verifies, but the graph breaks a rule.
    Invalid,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, thiserror::Error)]
#[error("{kind:?} {key}: {detail}")]
/// `Defect` is a record for the defect contract in this module: W8-T03: dependency-first conditional publication.
pub struct Defect {
    /// Record kind or category represented by this value.
    pub kind: DefectKind,
    /// Key recorded for this defect.
    pub key: String,
    /// Detail recorded for this defect.
    pub detail: String,
}

fn defect(kind: DefectKind, key: &str, detail: impl Into<String>) -> anyhow::Error {
    Defect {
        kind,
        key: key.into(),
        detail: detail.into(),
    }
    .into()
}

/// A verified remote closure: every object's bytes and every blob reference.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    /// Number of objects represented by this value.
    pub objects: BTreeMap<ObjectRef, Vec<u8>>,
    /// Blobs recorded for this graph.
    pub blobs: BTreeSet<BlobRef>,
    /// Remote reads made (request accounting).
    pub reads: u64,
}

impl Graph {
    /// Performs the get operation.
    pub fn get<T: Inventoried>(&self, digest: &str) -> Option<Decoded<T>> {
        let key = ObjectRef {
            kind: T::KIND,
            digest: digest.into(),
        };
        self.objects
            .get(&key)
            .and_then(|bytes| decode_as::<T>(bytes, digest).ok())
    }

    /// Performs the all operation.
    pub fn all<T: Inventoried>(&self) -> Vec<Decoded<T>> {
        self.objects
            .iter()
            .filter(|(r, _)| r.kind == T::KIND)
            .filter_map(|(r, bytes)| decode_as::<T>(bytes, &r.digest).ok())
            .collect()
    }

    /// Performs the digests operation.
    pub fn digests(&self, kind: Kind) -> Vec<String> {
        self.objects
            .keys()
            .filter(|r| r.kind == kind)
            .map(|r| r.digest.clone())
            .collect()
    }

    /// Corpora whose whole closure verified.
    pub fn corpora(&self) -> Vec<String> {
        self.digests(Kind::CorpusManifest)
    }

    /// Complete checkpoints whose whole closure verified.
    pub fn checkpoints(&self) -> Vec<String> {
        self.digests(Kind::Checkpoint)
    }
}

/// Read and verify the full dependency closure of `root` from the remote. Every failure is a
/// [`Defect`]: the first missing or corrupt item met, or the first broken graph rule.
pub fn verify_closure(remote: &dyn Remote, root: &Dependency) -> Result<Graph> {
    let mut graph = Graph::default();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([root.clone()]);
    while let Some(item) = queue.pop_front() {
        if !seen.insert(item.clone()) {
            continue;
        }
        ensure!(
            seen.len() <= MAX_CLOSURE_ITEMS,
            "closure of {} exceeds {MAX_CLOSURE_ITEMS} items",
            root.key()
        );
        let key = item.key();
        graph.reads += 1;
        let Some(bytes) = remote.get(&key)? else {
            return Err(defect(DefectKind::Missing, &key, "not on the remote"));
        };
        let deps = item
            .verify(&bytes)
            .map_err(|e| defect(DefectKind::Corrupt, &key, format!("{e:#}")))?;
        match item {
            Dependency::Object(o) => {
                graph.objects.insert(o, bytes);
            }
            Dependency::Blob(b) => {
                graph.blobs.insert(b);
            }
        }
        queue.extend(deps);
    }
    if let Dependency::Object(o) = root {
        check_rules(&graph, o)?;
    }
    Ok(graph)
}

/// The graph rules no single object can check on its own. Applies to a closure that already
/// verified item by item, so every object a rule names is present.
pub fn check_rules(graph: &Graph, root: &ObjectRef) -> Result<()> {
    fn need<T>(value: Option<T>, key: &str, what: &str) -> Result<T> {
        value.ok_or_else(|| defect(DefectKind::Invalid, key, format!("{what} does not decode")))
    }
    let invalid = |key: String, detail: String| defect(DefectKind::Invalid, &key, detail);

    // One run: every object in the closure belongs to the root's run.
    let run_of = |bytes: &[u8]| -> Option<String> {
        serde_json::from_slice::<serde_json::Value>(bytes)
            .ok()?
            .get("run_id")?
            .as_str()
            .map(str::to_string)
    };
    let root_key = object_key(root.kind, &root.digest);
    let run_id = need(
        graph.objects.get(root).and_then(|b| run_of(b)),
        &root_key,
        "the closure root",
    )?;
    for (r, bytes) in &graph.objects {
        let theirs = run_of(bytes);
        if theirs.as_deref() != Some(run_id.as_str()) {
            return Err(invalid(
                object_key(r.kind, &r.digest),
                format!("belongs to run {theirs:?}, not {run_id}"),
            ));
        }
    }

    // Commits: one manifest along the chain, and heights that count up by one.
    for c in graph.all::<Commit>() {
        let key = object_key(Kind::Commit, c.digest());
        let c = c.value();
        if let Some(p) = &c.parent {
            let parent = need(graph.get::<Commit>(p), &key, "its parent")?;
            let parent = parent.value();
            if parent.height.checked_add(1) != Some(c.height)
                || parent.run_manifest != c.run_manifest
            {
                return Err(invalid(
                    key,
                    format!(
                        "height {} on a parent at height {} (manifest {} on {})",
                        c.height, parent.height, c.run_manifest, parent.run_manifest
                    ),
                ));
            }
        }
    }

    // Exposures: update n starts from the checkpoint of update n - 1, or from nothing at 0.
    let index_of = |digest: &str, key: &str| -> Result<u64> {
        Ok(need(graph.get::<Checkpoint>(digest), key, "a checkpoint")?
            .value()
            .update_index)
    };
    for e in graph.all::<Exposure>() {
        let key = object_key(Kind::Exposure, e.digest());
        let e = e.value();
        let expected = match &e.checkpoint_before {
            Some(c) => index_of(c, &key)?.checked_add(1),
            None => Some(0),
        };
        if expected != Some(e.update_index) {
            return Err(invalid(
                key,
                format!(
                    "update {} does not follow its starting checkpoint (expected {expected:?})",
                    e.update_index
                ),
            ));
        }
    }

    // Checkpoints: commit exactly their exposure, contiguous with their parent.
    for k in graph.all::<Checkpoint>() {
        let key = object_key(Kind::Checkpoint, k.digest());
        let k = k.value();
        let exposure = need(graph.get::<Exposure>(&k.exposure), &key, "its exposure")?;
        let e = exposure.value();
        let expected = match &k.parent {
            Some(p) => index_of(p, &key)?.checked_add(1),
            None => Some(0),
        };
        if expected != Some(k.update_index)
            || e.update_index != k.update_index
            || e.corpus != k.corpus
            || e.checkpoint_before != k.parent
        {
            return Err(invalid(
                key,
                format!(
                    "update {} (expected {expected:?}) does not commit exposure {} (update {}, \
                     corpus {}, from {:?})",
                    k.update_index, k.exposure, e.update_index, e.corpus, e.checkpoint_before
                ),
            ));
        }
    }

    // Updates form one consecutive chain. The rules above make every exposure and checkpoint
    // follow the checkpoint before it; this makes the chain single: exactly one exposure and at
    // most one checkpoint per update index, and the indexes 0, 1, 2, ... without a gap. A forked
    // update (a second update 0 from nothing, or two updates from one checkpoint) or a repeated
    // one (an update index committed again later) is therefore never committed.
    fn consecutive<T: Inventoried>(
        graph: &Graph,
        index: impl Fn(&T) -> u64,
        invalid: &dyn Fn(String, String) -> anyhow::Error,
    ) -> Result<()> {
        let mut seen: BTreeMap<u64, String> = BTreeMap::new();
        for item in graph.all::<T>() {
            let i = index(item.value());
            if let Some(other) = seen.insert(i, item.digest().to_string()) {
                return Err(invalid(
                    object_key(T::KIND, item.digest()),
                    format!(
                        "update {i} is already committed by {} {other}: updates form one \
                         consecutive chain, so a forked or repeated update index is refused",
                        T::KIND.dir()
                    ),
                ));
            }
        }
        if let Some((gap, _)) = seen.keys().enumerate().find(|(n, i)| *n as u64 != **i) {
            return Err(invalid(
                object_key(T::KIND, &seen[&(gap as u64)]),
                format!("update indexes {:?} are not consecutive", seen.keys()),
            ));
        }
        Ok(())
    }
    consecutive::<Exposure>(graph, |e| e.update_index, &invalid)?;
    consecutive::<Checkpoint>(graph, |k| k.update_index, &invalid)?;

    // Decisions and candidate evidence come only from a unit's completed attempt.
    let completed = |digest: &str, work_id: &str, attempt: u32| -> bool {
        graph.get::<Event>(digest).is_some_and(|e| {
            let e = e.value();
            e.work_id == work_id
                && e.attempt == attempt
                && matches!(e.body, EventBody::Completed { .. })
        })
    };
    for d in graph.all::<HwmDecision>() {
        let v = d.value();
        if !completed(&v.completion, &v.work_id, v.attempt) {
            return Err(invalid(
                object_key(Kind::HwmDecision, d.digest()),
                format!(
                    "decides attempt {} from {}, which is not that unit's completed attempt",
                    v.attempt, v.completion
                ),
            ));
        }
    }
    for c in graph.all::<CandidateEvidence>() {
        let v = c.value();
        if !completed(&v.completion, &v.work_id, v.attempt) {
            return Err(invalid(
                object_key(Kind::CandidateEvidence, c.digest()),
                format!(
                    "evidence for attempt {} from {}, which is not that unit's completed attempt",
                    v.attempt, v.completion
                ),
            ));
        }
    }
    Ok(())
}
