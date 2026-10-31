//! Bringing a mirror up to this host's tip.
//!
//! The pointer writes in [`crate::mirror`] are conditional, which is what stops a second host's
//! batch being orphaned — but a condition with no way to satisfy it is a wedge. Local state is the
//! authority and it moves first: one `hwm ingest` run without `--store`, or one transient push
//! failure, leaves the mirror behind, and from then on the writer's pre-append `HEAD.json` never
//! matches what the mirror holds again.
//!
//! So the push *reconciles* instead of asserting. It reads the mirror's `HEAD.json`, asks
//! [`Ledger::mirror_position`] whether that tip is a batch this host already has, and:
//!
//! * **gap** — uploads every local batch after it, and everything each of those batches *names*
//!   ahead of the batch object itself (all of it content-addressed, immutable and created with
//!   `If-None-Match: *`, so re-uploading one is a no-op and uploading one twice from two hosts
//!   cannot corrupt it), and only then moves `HEAD.json`, still conditionally;
//! * **divergence** — the mirror's tip is a batch this host cannot produce, so catching up would
//!   mean orphaning it. Fail closed, before anything local is touched, and say what to run.
//!
//! The expected-previous value handed to [`Mirror::put_pointer`] is the fresh read that
//! `mirror_position` just proved to be an ancestor of the local chain. That is *not* the
//! self-satisfying precondition the mirror module warns about: the read is used as the CAS witness
//! only after it has been checked against local history, and a tip that fails that check is
//! refused rather than overwritten. The version from that same read remains the write precondition; a
//! concurrent pointer change fails even after the ancestor proof and uploads.

use crate::ledger::{Ledger, MirrorPosition};
use crate::mirror::{Mirror, MirrorReport};
use anyhow::{anyhow, Result};
use jope_datasets::store::ObservedObject;
use std::fs;
use std::path::{Path, PathBuf};

/// What the push is about to do to one ledger's copy on the mirror.
pub struct Plan {
    /// `<ledger>/ledger/HEAD.json`.
    pub head_rel: String,
    /// The mirror's `HEAD.json` exactly as read, already proved to be an ancestor of the local
    /// chain (or `None` when the mirror has none). This is the CAS witness for the pointer move.
    pub observed_head: Option<ObservedObject>,
    /// Local batches the mirror is missing, oldest first.
    pub missing: Vec<(String, PathBuf)>,
}

/// Performs the head rel operation and returns an error if it cannot be completed.
pub fn head_rel(ledger: &Ledger) -> String {
    format!("{}/ledger/HEAD.json", ledger.name())
}

fn diverged(m: &Mirror, rel: &str, detail: &str) -> anyhow::Error {
    anyhow!(
        "{rel}: the mirror at {url} holds a chain this host cannot continue — {detail}. \
         Nothing was written; catching it up would orphan its tip. \
         To record this run locally, re-run the same command without --store and with \
         JOPE_LEARNING_URL unset; to push it, point --store at a mirror whose tip this host's \
         ledger already contains. A mirror is only ever advanced by a host that holds its tip.",
        url = m.url
    )
}

/// Read the mirror's `HEAD.json` and work out whether it can be caught up, without writing
/// anything. Errors — fails closed — on a true divergence.
///
/// Run this *before* the local append so that a diverged mirror costs nothing: no batch object
/// uploaded, no local batch sealed, no local pointer moved. Re-running it after the append is what
/// produces the [`Plan`] the push executes.
pub fn plan(m: &Mirror, ledger: &Ledger) -> Result<Plan> {
    let head_rel = head_rel(ledger);
    let observed_head = m.get_observed(&head_rel)?;
    match ledger.mirror_position(observed_head.as_ref().map(|object| object.bytes.as_slice()))? {
        MirrorPosition::UpToDate => Ok(Plan {
            head_rel,
            observed_head,
            missing: Vec::new(),
        }),
        MirrorPosition::Behind { missing } => Ok(Plan {
            head_rel,
            observed_head,
            missing,
        }),
        MirrorPosition::Diverged { detail } => Err(diverged(m, &head_rel, &detail)),
    }
}

/// Fail closed if the mirror has diverged from this host's chain; otherwise do nothing.
pub fn preflight(m: &Mirror, ledger: &Ledger) -> Result<()> {
    plan(m, ledger).map(|_| ())
}

/// Upload every batch the mirror is missing — each one preceded by `prepare`, which puts whatever
/// that batch *names* on the mirror — and only then move its `HEAD.json`, conditionally on the
/// bytes [`plan`] observed and validated.
///
/// `HEAD.json` is the commit point, and the order around it is the whole guarantee. It is the only
/// key that makes a batch object reachable, and it is also the only key [`plan`] consults to decide
/// what is missing. So while it still holds the old tip, a failed push costs nothing: the next
/// `plan` recomputes the same gap and re-runs `prepare` over the same batches. Once it holds the
/// new tip, `plan` returns [`MirrorPosition::UpToDate`], `missing` is empty and `prepare` never
/// runs for those batches again — so anything it was going to upload must already be there.
///
/// That last clause is a *contract on `prepare`*, not a property this function can enforce, and it
/// is the one that was broken: `prepare` must upload everything its batch names or return an
/// error. Quietly skipping one object — because the local store could not supply it, say — is
/// indistinguishable here from having uploaded it, and the skip is permanent: the batch object
/// goes up, `HEAD.json` moves past it, and no later `plan` ever offers that batch again.
///
/// Uploading early is always safe: blobs and batch objects are content-addressed, immutable and
/// written with `If-None-Match: *`, so a repeat is an `Exists` no-op and two hosts uploading the
/// same object cannot corrupt it. Uploading late is not: it turns one transient object-store
/// failure into a mirror whose own ledger permanently names records it does not hold.
pub fn push(
    m: &Mirror,
    ledger: &Ledger,
    plan: Plan,
    report: &mut MirrorReport,
    mut prepare: impl FnMut(&Path, &mut MirrorReport) -> Result<()>,
) -> Result<()> {
    for (rel, path) in &plan.missing {
        prepare(path, report)?;
        let rel = format!("{}/ledger/{rel}", ledger.name());
        report.count(m.put_immutable(&rel, &fs::read(path)?)?);
        report.batches += 1;
    }
    if plan.missing.is_empty() {
        return Ok(());
    }
    let head_bytes = fs::read(ledger.root().join("HEAD.json"))?;
    m.put_pointer(&plan.head_rel, &head_bytes, plan.observed_head.as_ref())?;
    report.pointers += 1;
    Ok(())
}
