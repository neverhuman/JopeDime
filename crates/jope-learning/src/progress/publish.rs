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

/// The outcome of one publication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "outcome")]
pub enum Publication {
    /// The pointer moved from `from` to the commit.
    Advanced {
        /// Commit recorded for this publication.
        commit: String,
        /// Height recorded for this publication.
        height: u64,
        /// Optional from value; `None` means it is not available.
        from: Option<String>,
    },
    /// The pointer already named this commit, as after a retry once a response was lost.
    /// Field value recorded by this public API type.
    AlreadyPublished {
        /// Commit recorded for this outcome.
        commit: String,
    },
    /// The pointer names something other than the commit's parent. Nothing was written. The
    /// commit stays on the remote and in the outbox for reconciliation.
    Conflict {
        /// Commit recorded for this publication.
        commit: String,
        /// Optional expected value; `None` means it is not available.
        expected: Option<String>,
        /// Optional found value; `None` means it is not available.
        found: Option<String>,
    },
    /// The commit's closure is not durable and verified on the remote, or the pointer is not a
    /// valid pointer of this run. The pointer was not moved.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Blocked {
        /// Commit recorded for this outcome.
        commit: String,
        /// Explanation accompanying this outcome.
        reason: String,
    },
}

/// Every digest `commit`'s closure reaches through this outbox's records: each item's reference
/// digest and the sha256 of its recorded bytes (an object's bytes and domain digest differ).
fn local_closure_digests(outbox: &Outbox, commit: &Encoded) -> Vec<String> {
    let mut digests = vec![commit.digest.clone(), sha256_hex(&commit.bytes)];
    let mut seen = BTreeSet::new();
    let mut pending: Vec<Dependency> = commit.dependencies.clone();
    while let Some(item) = pending.pop() {
        if !seen.insert(item.clone()) {
            continue;
        }
        digests.push(match &item {
            Dependency::Blob(b) => b.sha256.clone(),
            Dependency::Object(o) => o.digest.clone(),
        });
        if let Some(record) = outbox.seq_of(&item).and_then(|seq| outbox.record(seq)) {
            digests.push(record.bytes_sha256.clone());
            pending.extend(record.dependencies.iter().cloned());
        }
    }
    digests
}

/// A publication that has done everything except move the pointer.
#[derive(Debug)]
pub struct Plan {
    /// Identifier for the associated run.
    pub run_id: String,
    /// Head key recorded for this plan.
    pub head_key: String,
    /// Commit recorded for this plan.
    pub commit: String,
    /// Height recorded for this plan.
    pub height: u64,
    /// Optional parent value; `None` means it is not available.
    pub parent: Option<String>,
    /// Objects missing remotely that were uploaded again from the local outbox.
    pub reuploaded: Vec<String>,
    /// Remote reads made while verifying the closure.
    pub reads: u64,
    observed: Option<ObservedObject>,
}

#[derive(Debug)]
/// `Prepared` is an enum for the prepared contract in this module: W8-T03: dependency-first conditional publication.
pub enum Prepared {
    /// The `Ready` case of `Prepared`.
    Ready(Plan),
    /// The `Done` case of `Prepared`.
    Done(Publication),
}

/// Steps 1 and 2, then one read of the pointer: everything up to, but not including, the pointer
/// write. Refuses (errors, writing nothing) when `publisher` is not the run's publisher or the
/// commit is not the run's, and when the commit names anything the outbox has not recorded.
pub fn prepare(
    outbox: &mut Outbox,
    remote: &dyn Remote,
    publisher: &str,
    manifest: &Decoded<RunManifest>,
    commit: &Encoded,
) -> Result<Prepared> {
    ensure!(
        commit.kind == Kind::Commit,
        "only a commit advances a run pointer"
    );
    let c = decode_as::<Commit>(&commit.bytes, &commit.digest)?.into_value();
    let m = manifest.value();
    ensure!(
        publisher == m.publisher,
        "{publisher} is not the publisher of run {} ({}); nothing was written",
        m.run_id,
        m.publisher
    );
    ensure!(
        c.run_id == m.run_id && c.run_manifest == manifest.digest(),
        "commit {} belongs to run {} manifest {}, not {} {}; nothing was written",
        commit.digest,
        c.run_id,
        c.run_manifest,
        m.run_id,
        manifest.digest()
    );
    let blocked = |reason: String| {
        Ok(Prepared::Done(Publication::Blocked {
            commit: commit.digest.clone(),
            reason,
        }))
    };

    // W12-T05: nothing revoked is delivered or committed. The commit's whole closure, ancestors
    // included, is checked as this outbox recorded it, before anything is appended or delivered.
    // The outbox records every dependency before its dependent and delivery verifies that the
    // remote holds exactly those bytes, so the verified remote closure is the same set.
    let revocations = Revocations::load(outbox.state())?;
    if !revocations.is_empty() {
        if let Err(e) = revocations.refuse(
            &format!("commit {}", commit.digest),
            local_closure_digests(outbox, commit)
                .iter()
                .map(String::as_str),
        ) {
            return blocked(format!("{e:#}; nothing was delivered"));
        }
    }

    outbox.append_object(commit)?;
    let delivery = outbox.deliver(remote, None)?;
    if let Some(f) = delivery.failure {
        return blocked(format!(
            "delivery of outbox record {} ({}) failed: {}",
            f.seq, f.key, f.error
        ));
    }

    let root = Dependency::Object(ObjectRef {
        kind: Kind::Commit,
        digest: commit.digest.clone(),
    });
    let mut reuploaded: Vec<String> = Vec::new();
    let graph = loop {
        let error = match verify_closure(remote, &root) {
            Ok(graph) => break graph,
            Err(error) => error,
        };
        let Some(d) = error.downcast_ref::<Defect>() else {
            return Err(error);
        };
        if d.kind != DefectKind::Missing || reuploaded.contains(&d.key) {
            return blocked(format!(
                "the commit's closure does not verify on the remote: {d}"
            ));
        }
        match outbox.redeliver(remote, &d.key) {
            Ok(true) => reuploaded.push(d.key.clone()),
            Ok(false) => {
                return blocked(format!(
                    "the commit's closure does not verify on the remote: {d}; this outbox \
                     cannot supply it either"
                ))
            }
            Err(e) => return blocked(format!("re-uploading {} failed: {e:#}", d.key)),
        }
    };
    let reads = graph.reads;

    // An HWM decision is a link to the W3 HWM ledger, which stays the authority. Every decision
    // this commit adds must be one this host's own HWM ledger (the state tree of its outbox)
    // recorded, with that record, identity, transition and qualification. Decisions already
    // committed by an ancestor were checked by the host that added them, so a host restored from
    // the remote can publish on top of them without their batches.
    let ledger = Ledger::open(outbox.state().root(), "hwm");
    for o in c.objects.iter().filter(|o| o.kind == Kind::HwmDecision) {
        let key = object_key(o.kind, &o.digest);
        let Some(decision) = graph.get::<HwmDecision>(&o.digest) else {
            return blocked(format!("{key} is not in the verified closure"));
        };
        if let Err(e) = decision.value().verify_against(&ledger) {
            return blocked(format!(
                "{key} is not a decision this host's HWM ledger ({}) recorded: {e:#}",
                ledger.root().display()
            ));
        }
    }

    let head_key = head_key(&c.run_id)?;
    let observed = remote.get_observed(&head_key)?;
    let found = match &observed {
        None => None,
        Some(o) => match Head::decode(&o.bytes) {
            Ok(h) if h.run_id == c.run_id => Some(h),
            Ok(h) => {
                return blocked(format!(
                    "{head_key} names run {}; it is left as it is",
                    h.run_id
                ))
            }
            Err(e) => {
                return blocked(format!(
                    "{head_key} is not a valid run pointer ({e:#}); it is left as it is"
                ))
            }
        },
    };
    if found.as_ref().is_some_and(|h| h.commit == commit.digest) {
        return Ok(Prepared::Done(Publication::AlreadyPublished {
            commit: commit.digest.clone(),
        }));
    }
    if found.as_ref().map(|h| &h.commit) != c.parent.as_ref() {
        return Ok(Prepared::Done(Publication::Conflict {
            commit: commit.digest.clone(),
            expected: c.parent.clone(),
            found: found.map(|h| h.commit),
        }));
    }
    if let Some(h) = &found {
        if h.height.checked_add(1) != Some(c.height) {
            return blocked(format!(
                "{head_key} claims height {} for the parent, which is at height {}; it is left \
                 as it is",
                h.height,
                c.height - 1
            ));
        }
    }
    Ok(Prepared::Ready(Plan {
        run_id: c.run_id,
        head_key,
        commit: commit.digest.clone(),
        height: c.height,
        parent: c.parent,
        reuploaded,
        reads,
        observed,
    }))
}

/// Step 3: move the pointer from the observed predecessor to the commit, conditionally.
///
/// When the write fails, only the pointer itself can say what happened:
/// - it names the commit: the write landed and its response was lost, so the result is
///   `Advanced`;
/// - it still names the predecessor: nothing was written, and the error is returned;
/// - it names anything else: another writer won, and the result is a `Conflict`.
pub fn advance(remote: &dyn Remote, plan: Plan) -> Result<Publication> {
    let bytes = Head {
        schema: HEAD_SCHEMA.into(),
        run_id: plan.run_id.clone(),
        commit: plan.commit.clone(),
        height: plan.height,
    }
    .encode()?;
    let Err(error) = remote.put_pointer(&plan.head_key, &bytes, plan.observed.as_ref()) else {
        return Ok(Publication::Advanced {
            commit: plan.commit,
            height: plan.height,
            from: plan.parent,
        });
    };
    let now = remote
        .get_observed(&plan.head_key)?
        .map(|o| Head::decode(&o.bytes))
        .transpose()
        .with_context(|| format!("{}: unreadable after a failed write", plan.head_key))?
        .map(|h| h.commit);
    match now {
        Some(c) if c == plan.commit => Ok(Publication::Advanced {
            commit: plan.commit,
            height: plan.height,
            from: plan.parent,
        }),
        now if now == plan.parent => Err(error.context(format!(
            "{}: the conditional pointer write failed and the pointer is unchanged; nothing \
             was written, and the publication can be retried",
            plan.head_key
        ))),
        now => Ok(Publication::Conflict {
            commit: plan.commit,
            expected: plan.parent,
            found: now,
        }),
    }
}

/// Publish `commit` for `manifest`'s run: [`prepare`], then [`advance`].
pub fn publish(
    outbox: &mut Outbox,
    remote: &dyn Remote,
    publisher: &str,
    manifest: &Decoded<RunManifest>,
    commit: &Encoded,
) -> Result<Publication> {
    match prepare(outbox, remote, publisher, manifest, commit)? {
        Prepared::Ready(plan) => advance(remote, plan),
        Prepared::Done(publication) => Ok(publication),
    }
}

/// A run's remotely committed state, verified end to end.
#[derive(Debug)]
pub struct Reusable {
    /// Head recorded for this reusable.
    pub head: Head,
    /// Graph recorded for this reusable.
    pub graph: Graph,
}

/// Read the run pointer, then verify the commit it names and that commit's whole closure. Only
/// what this returns may be reused. Every failure is an error, never a partial result. That
/// includes a corrupt pointer, a dangling or misdescribed one, and any missing or corrupt
/// object, blob or broken rule.
pub fn reusable(remote: &dyn Remote, run_id: &str) -> Result<Option<Reusable>> {
    let key = head_key(run_id)?;
    let Some(bytes) = remote.get(&key)? else {
        return Ok(None);
    };
    let head = Head::decode(&bytes).with_context(|| format!("{key} is not a valid run pointer"))?;
    ensure!(head.run_id == run_id, "{key} names run {}", head.run_id);
    let root = Dependency::Object(ObjectRef {
        kind: Kind::Commit,
        digest: head.commit.clone(),
    });
    let graph = verify_closure(remote, &root).with_context(|| {
        format!(
            "{key} names commit {} whose closure does not verify",
            head.commit
        )
    })?;
    let commit = graph
        .get::<Commit>(&head.commit)
        .context("the pointer's commit did not decode")?;
    let commit = commit.value();
    if commit.height != head.height || commit.run_id != run_id {
        bail!(
            "{key} claims height {} for commit {} of run {} at height {}",
            head.height,
            head.commit,
            commit.run_id,
            commit.height
        );
    }
    Ok(Some(Reusable { head, graph }))
}
