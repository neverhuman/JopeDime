//! Durable learning progress (W8): what a run leaves behind, and how it is accounted for.
//!
//! W3 made individual decisions recoverable on one host: the HWM ledger, its pointers and the
//! teaching qualifications link to one another and roll forward after a crash. What it does not
//! say is *what a run was asked to do and what became of each piece of it*. A retried attempt,
//! a worker that timed out, a preflight refusal, a submission delivered twice by the transport —
//! none of those has a durable identity today, so none of them can be counted, and anything that
//! cannot be counted can silently disappear from a denominator.
//!
//! [`inventory`] defines the durable objects (W8-T01): a run manifest that freezes the complete
//! planned census before any work executes, the per-attempt events (scheduled, retry, started,
//! failed, completed, excluded), candidate evidence, links to W3 HWM decisions, corpus manifests,
//! exposure/update records, complete checkpoints, commits and rebuildable summaries. Every object
//! has one finite canonical encoding, schema limits, a domain-separated digest and an explicit
//! list of the objects and blobs it depends on.
//!
//! [`census`] reconciles a run's planned census against the events delivered for it: every
//! planned logical unit must reach exactly one terminal disposition, every failed attempt stays
//! counted after it is retried, a transport replay has no effect, and a retry can never become a
//! second logical completion or a second improvement.
//!
//! [`outbox`] persists every item locally before anything is acknowledged (W8-T02). It is a
//! bounded, fsynced, hash-chained append log with sequence numbers, dependency references and
//! delivery acknowledgements. It pauses new work at its high-water mark instead of dropping or
//! growing. [`remote`] is the store interface delivery uses: create-only immutable writes and
//! conditional pointer writes, with no unconditional fallback.
//!
//! [`publish`] advances a run's committed pointer (W8-T03), in this order:
//! 1. deliver the dependencies, then the manifests, then the commit;
//! 2. verify the commit's whole remote closure;
//! 3. move the pointer conditionally from its expected predecessor.
//!
//! Two publishers racing the same predecessor produce one winner and one explicit conflict.
//! Corpora and checkpoints are reusable only from a verified closure.
//!
//! [`rebuild`] rebuilds derived views from committed records (W8-T04). It also reconstructs
//! and restores fresh local state from the remote committed graph (W8-T06).
//!
//! [`cadence`] decides when to flush and when to checkpoint, and what to retain (W8-T05):
//! - a flush is due every 60 s, at 1 MiB, or immediately after an urgent event;
//! - a checkpoint is due every 100 updates or 5 minutes, and at shutdown;
//! - winners keep full artifacts and every other attempt keeps compact evidence.
//!
//! All of this runs against an injected clock. A failed flush never advances the durable
//! watermark.
//!
//! Logical work identity ([`inventory::WorkSpec::work_id`]) is derived from the run and the
//! work's own specification and never from an attempt; `(work_id, attempt)` is the attempt
//! identity; an object's digest is its delivery identity. Nothing here is authority for teaching
//! or HWM admission — those stay in [`crate::eligibility`] and [`crate::hwm`]. The remote store
//! is only ever the one named explicitly by `--store`/`JOPE_LEARNING_URL`; the tests use local
//! and in-memory stores only.

pub mod cadence;
pub mod census;
pub mod inventory;
pub mod outbox;
pub mod publish;
pub mod rebuild;
pub mod remote;

#[cfg(test)]
mod tests {
    mod cadence;
    mod census;
    mod encoding;
    mod faults;
    pub mod fixture;
    mod outbox;
    mod publish;
    mod rebuild;
    mod receiving;
}
