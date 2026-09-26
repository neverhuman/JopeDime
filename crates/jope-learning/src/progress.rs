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
//! Logical work identity ([`inventory::WorkSpec::work_id`]) is derived from the run and the
//! work's own specification and never from an attempt; `(work_id, attempt)` is the attempt
//! identity; an object's digest is its delivery identity. Nothing here is authority for teaching
//! or HWM admission — those stay in [`crate::eligibility`] and [`crate::hwm`] — and nothing here
//! talks to a remote store.

pub mod census;
pub mod inventory;

#[cfg(test)]
mod tests {
    mod census;
    mod encoding;
    pub mod fixture;
}
