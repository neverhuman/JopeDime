//! W8-T04/T06: views rebuilt from committed records, and state reconstructed from the remote
//! committed graph.
//!
//! **Derived views (T04).**
//! - [`summarize`] rebuilds a run's [`Summary`] from its verified committed closure alone.
//!   It covers:
//!   - the census: coverage, completion, failures by class, exclusions, decisions (refusals
//!     included) by transition, improvements and violations;
//!   - candidate yield and optimizer exposure;
//!   - corpora, unique teachers and checkpoints;
//!   - its freshness: the commit and height it was derived from.
//! - [`publish_summary`] stores the summary as an immutable object, and points a disposable
//!   index (`progress/runs/<run>/summary.json`) at it. The index is replaced only conditionally
//!   on the version just read.
//! - Deleting the index or the summary object and rebuilding yields the same bytes. Neither is
//!   ever an input to a decision.
//! - [`read_summary`] reports whether a summary is fresh or stale against the current pointer.
//!   A corrupt or inconsistent index is an error, not a view.
//!
//! **Reconstruction (T06).** [`reconstruct`] classifies what the remote, and optionally a
//! surviving local outbox, hold for a run:
//! - **committed**: the verified closure of the run pointer;
//! - **uncommitted**: uploaded but not reachable from the pointer, including divergent
//!   commits;
//! - **quarantined**: objects that do not verify;
//! - **local-only**: recorded, but the remote does not hold these exact bytes (never uploaded,
//!   or shadowed by a partial or foreign object);
//! - **unacknowledged**: uploaded but never acknowledged;
//! - **acknowledged-missing**: acknowledged but absent remotely.
//!
//! Only the committed state is recoverable from the remote. [`restore`] rebuilds a fresh local
//! outbox from exactly that, verifying every item, so the host can publish on top of the
//! committed tip.

use crate::canon::canonical_json;
use crate::progress::census::{reconcile, Census, Disposition};
use crate::progress::inventory::{
    blob_key, decode_as, encode, head_key, object_key, verify_object, BlobRef, CandidateEvidence,
    Checkpoint, Commit, CorpusManifest, Decoded, Dependency, Encoded, Event, Exposure, Fraction,
    Head, HwmDecision, Kind, ObjectRef, RunManifest, Summary,
};
use crate::progress::outbox::Outbox;
use crate::progress::publish::{reusable, Reusable};
use crate::progress::remote::Remote;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for index values.
pub const INDEX_SCHEMA: &str = "jope.progress.summary-index/v1";

/// `progress/runs/<run>/summary.json`: the disposable derived index.
pub fn index_key(run_id: &str) -> Result<String> {
    head_key(run_id)?;
    Ok(format!("progress/runs/{run_id}/summary.json"))
}

/// The derived index: which summary object is the run's current view.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Index {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Digest of the [`Summary`] object.
    pub summary: String,
    /// Derived from recorded for this index.
    pub derived_from: String,
    /// Height recorded for this index.
    pub height: u64,
}

/// Rebuild the run's summary from its verified committed closure. The census is reconciled
/// against the run manifest that the head commit names.
pub fn summarize(state: &Reusable) -> Result<(Summary, Census)> {
    let graph = &state.graph;
    let commit = graph
        .get::<Commit>(&state.head.commit)
        .context("the head commit is not in its closure")?;
    let manifest: Decoded<RunManifest> = graph
        .get::<RunManifest>(&commit.value().run_manifest)
        .context("the run manifest is not in the closure")?;
    let census = reconcile(
        &manifest,
        &graph.all::<Event>(),
        &graph.all::<HwmDecision>(),
    );
    let evidenced: BTreeSet<String> = graph
        .all::<CandidateEvidence>()
        .into_iter()
        .map(|c| c.value().completion.clone())
        .collect();
    let yielded = census
        .units
        .iter()
        .filter(|u| {
            matches!(&u.disposition, Disposition::Completed { completion, .. }
                if evidenced.contains(completion))
        })
        .count() as u64;
    let exposures = graph.all::<Exposure>();
    let corpora = graph.all::<CorpusManifest>();
    let teachers: BTreeSet<BlobRef> = corpora
        .iter()
        .flat_map(|c| c.value().members.iter().cloned())
        .collect();
    let checkpoints = graph.all::<Checkpoint>();
    let latest = checkpoints
        .iter()
        .max_by(|a, b| {
            (a.value().update_index, a.digest()).cmp(&(b.value().update_index, b.digest()))
        })
        .map(|c| c.digest().to_string());
    let summary = Summary {
        schema: Kind::Summary.schema().into(),
        run_id: state.head.run_id.clone(),
        derived_from: state.head.commit.clone(),
        height: state.head.height,
        counts: census.counts.clone(),
        candidate_yield: Fraction {
            numerator: yielded,
            denominator: census.counts.completed,
        },
        exposures: exposures.len() as u64,
        consumed_rows: exposures.iter().map(|e| e.value().consumed_rows).sum(),
        corpora: corpora.len() as u64,
        unique_teachers: teachers.len() as u64,
        checkpoints: checkpoints.len() as u64,
        latest_checkpoint: latest,
    };
    Ok((summary, census))
}

/// Rebuild the summary from the committed graph, store it, and point the derived index at it.
/// `None` when the run has no committed pointer. Fails, leaving the index untouched, when the
/// committed state does not verify.
pub fn publish_summary(remote: &dyn Remote, run_id: &str) -> Result<Option<Summary>> {
    let Some(state) = reusable(remote, run_id)? else {
        return Ok(None);
    };
    let (summary, _) = summarize(&state)?;
    let encoded = encode(&summary)?;
    remote.put_immutable(&encoded.key(), &encoded.bytes)?;
    let index = Index {
        schema: INDEX_SCHEMA.into(),
        run_id: run_id.into(),
        summary: encoded.digest.clone(),
        derived_from: summary.derived_from.clone(),
        height: summary.height,
    };
    let key = index_key(run_id)?;
    let bytes = canonical_json(&index)?;
    let observed = remote.get_observed(&key)?;
    if observed.as_ref().map(|o| o.bytes.as_slice()) != Some(bytes.as_slice()) {
        remote.put_pointer(&key, &bytes, observed.as_ref())?;
    }
    Ok(Some(summary))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
/// `Freshness` is an enum for the freshness contract in this module: W8-T04/T06: views rebuilt from committed records, and state reconstructed from the remote committed graph.
pub enum Freshness {
    /// Derived from the commit the run pointer names now.
    Fresh,
    /// Derived from another commit than the one the pointer names now.
    Stale {
        /// Derived from recorded for this freshness.
        derived_from: String,
        /// Derived height recorded for this freshness.
        derived_height: u64,
        /// Optional head value; `None` means it is not available.
        head: Option<String>,
        /// Optional head height value; `None` means it is not available.
        head_height: Option<u64>,
    },
    /// No summary has been derived.
    Missing,
}

/// The derived summary and how fresh it is. The index and the summary object it names must
/// agree and decode canonically; anything else is an error, not a view.
pub fn read_summary(remote: &dyn Remote, run_id: &str) -> Result<(Freshness, Option<Summary>)> {
    let key = index_key(run_id)?;
    let Some(bytes) = remote.get(&key)? else {
        return Ok((Freshness::Missing, None));
    };
    let index: Index =
        serde_json::from_slice(&bytes).with_context(|| format!("{key} does not parse"))?;
    ensure!(
        index.schema == INDEX_SCHEMA && index.run_id == run_id && canonical_json(&index)? == bytes,
        "{key} is not a canonical summary index of run {run_id}"
    );
    let object = object_key(Kind::Summary, &index.summary);
    let summary_bytes = remote
        .get(&object)?
        .with_context(|| format!("{key} names summary {object}, which is missing"))?;
    let summary = decode_as::<Summary>(&summary_bytes, &index.summary)
        .with_context(|| format!("{key} names summary {object}, which does not verify"))?
        .into_value();
    ensure!(
        summary.run_id == run_id
            && summary.derived_from == index.derived_from
            && summary.height == index.height,
        "{key} disagrees with the summary it names"
    );
    let head = match remote.get(&head_key(run_id)?)? {
        Some(b) => Some(Head::decode(&b).context("the run pointer is not valid")?),
        None => None,
    };
    let fresh = head
        .as_ref()
        .is_some_and(|h| h.commit == summary.derived_from && h.height == summary.height);
    let freshness = if fresh {
        Freshness::Fresh
    } else {
        Freshness::Stale {
            derived_from: summary.derived_from.clone(),
            derived_height: summary.height,
            head: head.as_ref().map(|h| h.commit.clone()),
            head_height: head.as_ref().map(|h| h.height),
        }
    };
    Ok((freshness, Some(summary)))
}

/// What the remote and a surviving outbox hold for a run, by store key.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Reconstruction {
    /// Identifier for the associated run.
    pub run_id: String,
    /// Optional head value; `None` means it is not available.
    pub head: Option<Head>,
    /// The committed closure: remotely acknowledged, committed and verified. The only progress
    /// recoverable from the remote.
    pub committed: BTreeSet<String>,
    /// This run's verified objects on the remote that the committed closure does not reach.
    /// Derived summaries are not progress and are not listed.
    pub uncommitted: BTreeSet<String>,
    /// Digests of this run's commits outside the committed chain (divergent branches).
    pub divergent: BTreeSet<String>,
    /// Keys under the object prefix that do not verify. Never read as state.
    pub quarantined: BTreeSet<String>,
    /// Outbox entries whose exact bytes are not on the remote. They are lost unless this host
    /// survives.
    pub local_only: BTreeSet<String>,
    /// Outbox entries on the remote that this host never acknowledged, for example after a
    /// crash or a lost response following the upload.
    pub unacknowledged: BTreeSet<String>,
    /// Outbox entries acknowledged, yet absent remotely: only this host can supply them.
    pub acknowledged_missing: BTreeSet<String>,
}

fn parse_object_key(key: &str) -> Option<ObjectRef> {
    let rest = key.strip_prefix("progress/objects/")?;
    let (dir, file) = rest.split_once('/')?;
    let digest = file.strip_suffix(".json")?;
    let kind = Kind::ALL.into_iter().find(|k| k.dir() == dir)?;
    Some(ObjectRef {
        kind,
        digest: digest.into(),
    })
}

fn run_of(bytes: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()?
        .get("run_id")?
        .as_str()
        .map(str::to_string)
}

/// Classify a run's progress. A present but invalid pointer, or a closure that does not
/// verify, is an error: a visible failure, never a partial "committed" set.
pub fn reconstruct(
    remote: &dyn Remote,
    run_id: &str,
    outbox: Option<&Outbox>,
) -> Result<Reconstruction> {
    let mut out = Reconstruction {
        run_id: run_id.into(),
        ..Default::default()
    };
    let mut chain = BTreeSet::new();
    if let Some(state) = reusable(remote, run_id)? {
        out.committed.extend(
            state
                .graph
                .objects
                .keys()
                .map(|o| object_key(o.kind, &o.digest)),
        );
        out.committed
            .extend(state.graph.blobs.iter().map(|b| blob_key(&b.sha256)));
        let mut next = Some(state.head.commit.clone());
        while let Some(c) = next {
            next = state
                .graph
                .get::<Commit>(&c)
                .and_then(|commit| commit.value().parent.clone());
            chain.insert(c);
        }
        out.head = Some(state.head);
    }
    let listed: BTreeSet<String> = remote.list("progress/objects/")?.into_iter().collect();
    for key in &listed {
        let Some(object) = parse_object_key(key) else {
            out.quarantined.insert(key.clone());
            continue;
        };
        let Some(bytes) = remote.get(key)? else {
            continue;
        };
        if verify_object(object.kind, &object.digest, &bytes).is_err() {
            out.quarantined.insert(key.clone());
            continue;
        }
        if run_of(&bytes).as_deref() != Some(run_id)
            || object.kind == Kind::Summary
            || out.committed.contains(key)
        {
            continue;
        }
        out.uncommitted.insert(key.clone());
        if object.kind == Kind::Commit && !chain.contains(&object.digest) {
            out.divergent.insert(object.digest);
        }
    }
    if let Some(o) = outbox {
        for seq in 1..=o.records() {
            let record = o.record(seq).context("outbox record vanished")?;
            let key = record.item.key();
            // On the remote means the remote holds exactly these bytes: a partial or foreign
            // object under the key does not count.
            let on_remote = match &record.item {
                Dependency::Object(_) => listed.contains(&key) && !out.quarantined.contains(&key),
                Dependency::Blob(b) => remote
                    .get(&key)?
                    .is_some_and(|bytes| BlobRef::of(&bytes) == *b),
            };
            match (o.ack(seq).is_some(), on_remote) {
                (true, false) => out.acknowledged_missing.insert(key),
                (false, true) => out.unacknowledged.insert(key),
                (false, false) => out.local_only.insert(key),
                (true, true) => false,
            };
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `Restored` is a record for the restored contract in this module: W8-T04/T06: views rebuilt from committed records, and state reconstructed from the remote committed graph.
pub struct Restored {
    /// Head recorded for this restored.
    pub head: Head,
    /// Items appended to the outbox, in dependency order.
    pub appended: u64,
    /// Items the outbox already held.
    pub present: u64,
    /// Remote reads made, closure verification included.
    pub reads: u64,
}

/// Restore the run's committed closure into `outbox` in dependency order, verifying every
/// object and blob again as it is read, then acknowledge it all against the remote (where it
/// exists with the same bytes). The host can then publish the next commit on the restored tip.
/// `None` when the run has no committed pointer.
pub fn restore(remote: &dyn Remote, run_id: &str, outbox: &mut Outbox) -> Result<Option<Restored>> {
    let Some(state) = reusable(remote, run_id)? else {
        return Ok(None);
    };
    let mut reads = state.graph.reads;
    let mut deps: BTreeMap<Dependency, Vec<Dependency>> = BTreeMap::new();
    for (o, bytes) in &state.graph.objects {
        deps.insert(
            Dependency::Object(o.clone()),
            verify_object(o.kind, &o.digest, bytes)?,
        );
    }
    // Dependency order: a post-order walk from the head commit, children in sorted order.
    let root = Dependency::Object(ObjectRef {
        kind: Kind::Commit,
        digest: state.head.commit.clone(),
    });
    let mut order = Vec::new();
    let mut done = BTreeSet::new();
    let mut stack = vec![(root, false)];
    while let Some((item, expanded)) = stack.pop() {
        if done.contains(&item) {
            continue;
        }
        if expanded {
            done.insert(item.clone());
            order.push(item);
            continue;
        }
        stack.push((item.clone(), true));
        for d in deps.get(&item).into_iter().flatten().rev() {
            if !done.contains(d) {
                stack.push((d.clone(), false));
            }
        }
    }
    let (mut appended, mut present) = (0, 0);
    for item in order {
        if outbox.seq_of(&item).is_some() {
            present += 1;
            continue;
        }
        match &item {
            Dependency::Object(o) => {
                let bytes = &state.graph.objects[o];
                outbox.append_object(&Encoded::from_bytes(o.kind, &o.digest, bytes)?)?;
            }
            Dependency::Blob(b) => {
                reads += 1;
                let bytes = remote
                    .get(&blob_key(&b.sha256))?
                    .with_context(|| format!("blob {} vanished during the restore", b.sha256))?;
                outbox.append_result(&bytes, b)?;
            }
        }
        appended += 1;
    }
    let delivery = outbox.deliver(remote, None)?;
    if let Some(f) = delivery.failure {
        bail!(
            "restored item {} could not be acknowledged against the remote: {}",
            f.key,
            f.error
        );
    }
    Ok(Some(Restored {
        head: state.head,
        appended,
        present,
        reads,
    }))
}
