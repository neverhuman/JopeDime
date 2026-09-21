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
//! * **gap** — uploads every local batch after it (batches are content-addressed, immutable and
//!   created with `If-None-Match: *`, so re-uploading one is a no-op and uploading one twice from
//!   two hosts cannot corrupt it), and only then moves `HEAD.json`, still conditionally;
//! * **divergence** — the mirror's tip is a batch this host cannot produce, so catching up would
//!   mean orphaning it. Fail closed, before anything local is touched, and say what to run.
//!
//! The expected-previous value handed to [`Mirror::put_pointer`] is the fresh read that
//! `mirror_position` just proved to be an ancestor of the local chain. That is *not* the
//! self-satisfying precondition the mirror module warns about: the read is used as the CAS witness
//! only after it has been checked against local history, and a tip that fails that check is
//! refused rather than overwritten. The compare-then-put is still TOCTOU (the `Store` API has no
//! server-side conditional PUT), exactly as before.

use crate::ledger::{Ledger, MirrorPosition};
use crate::mirror::{Mirror, MirrorReport};
use anyhow::{anyhow, Result};
use std::fs;
use std::path::PathBuf;

/// What the push is about to do to one ledger's copy on the mirror.
pub struct Plan {
    /// `<ledger>/ledger/HEAD.json`.
    pub head_rel: String,
    /// The mirror's `HEAD.json` exactly as read, already proved to be an ancestor of the local
    /// chain (or `None` when the mirror has none). This is the CAS witness for the pointer move.
    pub observed_head: Option<Vec<u8>>,
    /// Local batches the mirror is missing, oldest first.
    pub missing: Vec<(String, PathBuf)>,
}

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
    let observed_head = m.get(&head_rel)?;
    match ledger.mirror_position(observed_head.as_deref())? {
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

/// Upload every batch the mirror is missing, then move its `HEAD.json` conditionally on the bytes
/// [`plan`] observed and validated. Returns the batches that were uploaded, oldest first, so the
/// caller can push whatever those batches reference (blobs, pointers).
pub fn push(
    m: &Mirror,
    ledger: &Ledger,
    plan: Plan,
    report: &mut MirrorReport,
) -> Result<Vec<(String, PathBuf)>> {
    for (rel, path) in &plan.missing {
        let rel = format!("{}/ledger/{rel}", ledger.name());
        report.count(m.put_immutable(&rel, &fs::read(path)?)?);
        report.batches += 1;
    }
    if plan.missing.is_empty() {
        return Ok(plan.missing);
    }
    let head_bytes = fs::read(ledger.root().join("HEAD.json"))?;
    m.put_pointer(&plan.head_rel, &head_bytes, plan.observed_head.as_deref())?;
    report.pointers += 1;
    Ok(plan.missing)
}
