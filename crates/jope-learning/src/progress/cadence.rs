//! W8-T05: cadence and retention classes.
//!
//! **Flush cadence** ([`Cadence`]). Compact progress goes from the outbox to the remote when
//! anything is pending and any of these holds:
//! - an urgent request is outstanding ([`Urgent`]): an HWM change, a corpus freeze, a
//!   checkpoint commit, a terminal failure, controlled shutdown, or a restart that found
//!   pending entries;
//! - at least [`Policy::flush_bytes`] (1 MiB) are pending;
//! - at least [`Policy::flush_interval_ms`] (60 s) have passed since the last flush that left
//!   nothing pending.
//!
//! Only a flush that leaves nothing pending clears urgency and restarts the interval. A failed,
//! partial or response-lost flush keeps everything pending and keeps its urgency, and the
//! interval keeps running, so the next tick retries it.
//!
//! The durable watermark ([`durable_through`]) is computed from the outbox's durable
//! acknowledgements alone: the highest `n` such that records `1..=n` are all acknowledged.
//! Nothing in memory can advance it. A failed flush therefore never claims remote durability,
//! and a restarted process reports exactly what it reported before.
//!
//! **Checkpoint cadence** ([`Checkpoints`]). A checkpoint is due at a committed update:
//! - once [`Policy::checkpoint_updates`] (100) updates are uncheckpointed;
//! - or once [`Policy::checkpoint_interval_ms`] (5 min) have passed since the last checkpoint;
//! - and at controlled shutdown, whenever any committed update is uncheckpointed.
//!
//! **Retention** ([`retention`]):
//! - winners, admitted teachers and declared diagnostic samples keep their full fitted
//!   artifacts ([`Retention::Full`]);
//! - every other completed attempt keeps compact evidence ([`Retention::Compact`]). That is a
//!   smaller record, never no record, so a non-winner outcome is never dropped.
//!
//! Nothing here deletes anything or changes a retention policy.
//!
//! All time comes from an injected [`Clock`]. The [`Policy::ADOPTED`] values, like the outbox
//! limits, were adopted by owner delegation on 2026-09-26 and remain configuration
//! (`progress::product::Config`).

use crate::progress::inventory::Retention;
use crate::progress::outbox::{Delivery, Outbox};
use crate::progress::remote::Remote;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Milliseconds on a monotonic, injected clock.
pub trait Clock {
    /// Performs the now ms operation and returns an error if it cannot be completed.
    fn now_ms(&self) -> u64;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Policy` is a record for the policy contract in this module: W8-T05: cadence and retention classes.
pub struct Policy {
    /// Flush interval ms recorded for this policy.
    pub flush_interval_ms: u64,
    /// Byte count for the associated flush.
    pub flush_bytes: u64,
    /// Checkpoint updates recorded for this policy.
    pub checkpoint_updates: u64,
    /// Checkpoint interval ms recorded for this policy.
    pub checkpoint_interval_ms: u64,
}

impl Policy {
    /// Adopted by owner delegation on 2026-09-26: flush every 60 s or 1 MiB; checkpoint every
    /// 100 updates or 5 minutes.
    pub const ADOPTED: Policy = Policy {
        flush_interval_ms: 60_000,
        flush_bytes: 1 << 20,
        checkpoint_updates: 100,
        checkpoint_interval_ms: 300_000,
    };
}

/// Events that request an immediate flush.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Urgent {
    /// The `HwmChange` case of `Urgent`.
    HwmChange,
    /// The `CorpusFreeze` case of `Urgent`.
    CorpusFreeze,
    /// The `CheckpointCommit` case of `Urgent`.
    CheckpointCommit,
    /// The `TerminalFailure` case of `Urgent`.
    TerminalFailure,
    /// The `Shutdown` case of `Urgent`.
    Shutdown,
    /// A restarted process found pending entries.
    Restart,
}

/// Why a flush is due.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "reason")]
pub enum FlushDue {
    /// The `Urgent` case of `FlushDue`.
    Urgent {
        /// Requests recorded for this outcome.
        requests: Vec<Urgent>,
    },
    /// The `Size` case of `FlushDue`.
    Size {
        /// Pending bytes recorded for this outcome.
        pending_bytes: u64,
    },
    /// The `Interval` case of `FlushDue`.
    Interval {
        /// Waited ms recorded for this outcome.
        waited_ms: u64,
    },
}

/// The durable watermark: the highest `n` such that outbox records `1..=n` are all
/// acknowledged by the remote. Derived from durable acknowledgements only.
pub fn durable_through(outbox: &Outbox) -> u64 {
    (1..=outbox.records())
        .take_while(|&seq| outbox.ack(seq).is_some())
        .last()
        .unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
/// `Tick` is a record for the tick contract in this module: W8-T05: cadence and retention classes.
pub struct Tick {
    /// Optional due value; `None` means it is not available.
    pub due: Option<FlushDue>,
    /// `None` when nothing was due.
    pub delivery: Option<Delivery>,
    /// True only when the flush left nothing pending.
    pub flushed: bool,
    /// Durable through recorded for this tick.
    pub durable_through: u64,
}

/// Flush scheduling state.
#[derive(Clone, Debug)]
pub struct Cadence {
    /// Policy recorded for this cadence.
    pub policy: Policy,
    last_flush_ms: u64,
    urgent: BTreeSet<Urgent>,
}

impl Cadence {
    /// Start (or restart) the cadence. Pending entries found in the outbox are flushed
    /// immediately: they may be owed since before a crash.
    pub fn start(policy: Policy, now_ms: u64, outbox: &Outbox) -> Cadence {
        let mut cadence = Cadence::new(policy, now_ms);
        if outbox.pending_entries() > 0 {
            cadence.request(Urgent::Restart);
        }
        cadence
    }

    /// A cadence for an outbox the caller knows owed nothing when it started: nothing is urgent
    /// and the interval starts now.
    pub fn new(policy: Policy, now_ms: u64) -> Cadence {
        Cadence {
            policy,
            last_flush_ms: now_ms,
            urgent: BTreeSet::new(),
        }
    }

    /// Performs the request operation and returns an error if it cannot be completed.
    pub fn request(&mut self, why: Urgent) {
        self.urgent.insert(why);
    }

    /// Performs the pending urgent operation and returns an error if it cannot be completed.
    pub fn pending_urgent(&self) -> Vec<Urgent> {
        self.urgent.iter().copied().collect()
    }

    /// Whether a flush is due now, and why. Nothing is due while nothing is pending.
    pub fn due(&self, now_ms: u64, outbox: &Outbox) -> Option<FlushDue> {
        self.due_for(now_ms, outbox.pending_entries(), outbox.pending_bytes())
    }

    /// [`Cadence::due`] for a caller that owes more than the outbox's pending entries, such as
    /// a publisher whose appended objects are delivered but not yet committed.
    pub fn due_for(
        &self,
        now_ms: u64,
        pending_entries: u64,
        pending_bytes: u64,
    ) -> Option<FlushDue> {
        if pending_entries == 0 {
            return None;
        }
        if !self.urgent.is_empty() {
            return Some(FlushDue::Urgent {
                requests: self.pending_urgent(),
            });
        }
        if pending_bytes >= self.policy.flush_bytes {
            return Some(FlushDue::Size { pending_bytes });
        }
        let waited_ms = now_ms.saturating_sub(self.last_flush_ms);
        (waited_ms >= self.policy.flush_interval_ms).then_some(FlushDue::Interval { waited_ms })
    }

    /// Flush if due. Only a flush that leaves nothing pending clears urgency and restarts the
    /// interval; a failed or partial one keeps everything owed.
    pub fn tick(
        &mut self,
        clock: &dyn Clock,
        outbox: &mut Outbox,
        remote: &dyn Remote,
    ) -> Result<Tick> {
        let now = clock.now_ms();
        if outbox.pending_entries() == 0 {
            // Nothing is owed: any urgency is satisfied and the interval restarts.
            self.urgent.clear();
            self.last_flush_ms = now;
        }
        let Some(due) = self.due(now, outbox) else {
            return Ok(Tick {
                due: None,
                delivery: None,
                flushed: false,
                durable_through: durable_through(outbox),
            });
        };
        let delivery = outbox.deliver(remote, None)?;
        let flushed = delivery.failure.is_none() && delivery.pending == 0;
        if flushed {
            self.urgent.clear();
            self.last_flush_ms = now;
        }
        Ok(Tick {
            due: Some(due),
            delivery: Some(delivery),
            flushed,
            durable_through: durable_through(outbox),
        })
    }
}

/// Why a checkpoint is due.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "reason")]
pub enum CheckpointDue {
    /// The `Updates` case of `CheckpointDue`.
    Updates {
        /// Uncheckpointed recorded for this outcome.
        uncheckpointed: u64,
    },
    /// The `Interval` case of `CheckpointDue`.
    Interval {
        /// Waited ms recorded for this outcome.
        waited_ms: u64,
    },
    /// The `Shutdown` case of `CheckpointDue`.
    Shutdown {
        /// Uncheckpointed recorded for this outcome.
        uncheckpointed: u64,
    },
}

/// Checkpoint scheduling at committed updates.
#[derive(Clone, Debug)]
pub struct Checkpoints {
    /// Policy recorded for this checkpoints.
    pub policy: Policy,
    last_ms: u64,
    uncheckpointed: u64,
}

impl Checkpoints {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(policy: Policy, now_ms: u64) -> Checkpoints {
        Checkpoints {
            policy,
            last_ms: now_ms,
            uncheckpointed: 0,
        }
    }

    /// Record one committed update, and say whether a checkpoint is due at it.
    pub fn committed_update(&mut self, now_ms: u64) -> Option<CheckpointDue> {
        self.uncheckpointed += 1;
        if self.uncheckpointed >= self.policy.checkpoint_updates {
            return Some(CheckpointDue::Updates {
                uncheckpointed: self.uncheckpointed,
            });
        }
        let waited_ms = now_ms.saturating_sub(self.last_ms);
        (waited_ms >= self.policy.checkpoint_interval_ms)
            .then_some(CheckpointDue::Interval { waited_ms })
    }

    /// Record `n` committed updates observed at once (an engine checkpoint published after
    /// several optimizer steps), and say whether a checkpoint is due at the last of them. The
    /// update trigger fires as soon as the count reaches the policy, however many updates
    /// arrived together; `n == 0` records nothing and is never due.
    pub fn committed_updates(&mut self, n: u64, now_ms: u64) -> Option<CheckpointDue> {
        if n == 0 {
            return None;
        }
        self.uncheckpointed = self.uncheckpointed.saturating_add(n - 1);
        self.committed_update(now_ms)
    }

    /// Controlled shutdown: due whenever any committed update is uncheckpointed.
    pub fn shutdown(&self) -> Option<CheckpointDue> {
        (self.uncheckpointed > 0).then_some(CheckpointDue::Shutdown {
            uncheckpointed: self.uncheckpointed,
        })
    }

    /// A checkpoint was committed. Call only once its commit is durable.
    pub fn checkpointed(&mut self, now_ms: u64) {
        self.uncheckpointed = 0;
        self.last_ms = now_ms;
    }

    /// Performs the uncheckpointed operation and returns an error if it cannot be completed.
    pub fn uncheckpointed(&self) -> u64 {
        self.uncheckpointed
    }
}

/// What an attempt's candidate turned out to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Role {
    /// Initialized or improved an HWM.
    Winner,
    /// The `AdmittedTeacher` case of `Role`.
    AdmittedTeacher,
    /// A declared diagnostic sample.
    DiagnosticSample,
    /// Any other completed attempt.
    Other,
}

/// The retention class for a role. There is no class that retains nothing.
pub fn retention(role: Role) -> Retention {
    match role {
        Role::Winner | Role::AdmittedTeacher | Role::DiagnosticSample => Retention::Full,
        Role::Other => Retention::Compact,
    }
}
