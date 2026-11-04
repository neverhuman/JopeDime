//! Controller operations for `jope.native_bounded_run.v1` (W7-T02, W7-T05).
//!
//! * **Plan** freezes the canonical contract bytes (`plan.json`) and opens the journal.
//!   Execution binds to that artifact and its digest, never to mutable flags.
//! * **Status** and **Verify** are read-only: they create no directory, lock or file, take
//!   no lock, start no process and contact nothing. Verify re-checks the plan, the whole
//!   hash-chained journal and every committed result; it never starts numerical work.
//! * **Cancel** is idempotent: the first request is published without replacement and
//!   later calls return it unchanged.
//! * [`Coordinator`] persists each rank's dispatch intent before launching it and its launch
//!   identity before rendezvous, persists the full-group admission before any participant
//!   is told of it (ranks start work only on that acknowledgement), drains the *whole*
//!   synchronous group on rank failure, lost control, grant expiry, deadline, cancellation
//!   or restart, distinguishes never-started work from unacknowledged completion, and
//!   commits a group at most once.
//!
//! The execution backend is the [`Supervisor`] trait. The live implementation is the
//! SmartCluster-backed [`super::supervisor::SmartClusterSupervisor`]; the in-process fake and
//! the same-host process supervisor in the tests are fixtures only.
use super::{
    census::{self, HostCensus, PreflightReceipt, RefusalReason},
    contract::{AttemptOutcome, BoundedRunContract, PlanRefusal, MAX_CONTRACT_BYTES},
    rendezvous::{
        self, publish_immutable, Connected, GroupAdmission, GroupRefusal, Rendezvous,
        VerifiedResult,
    },
};
use crate::manifest::hash;
use anyhow::{anyhow, bail, ensure, Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub const JOURNAL_SCHEMA: &str = "jope.native_bounded_journal.v1";
pub const STATUS_SCHEMA: &str = "jope.native_bounded_status.v1";
pub const VERIFY_SCHEMA: &str = "jope.native_bounded_verification.v1";
pub const CANCEL_SCHEMA: &str = "jope.native_bounded_cancel.v1";
/// A rank that loses its control channel stops itself and exits with this code.
pub const EXIT_LOST_CONTROL: i32 = 75;
const MAX_RECORD_BYTES: u64 = 1024 * 1024;
const MAX_RECORDS: usize = 100_000;
const PLAN: &str = "plan.json";
const JOURNAL: &str = "journal";
const LOCK: &str = "controller.lock";
const CANCEL: &str = "cancel.json";
const RESULTS: &str = "results";
const ENVELOPES: &str = "envelopes";

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    ensure!(
        f.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(b)
}

mod replay;
use replay::*;

mod operations;
pub use operations::*;

/// Liveness of supervised rank work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum Liveness {
    Running,
    Exited {
        code: Option<i32>,
        signal: Option<i32>,
    },
    Gone,
}
/// Everything a supervisor may use to start rank work. The rank itself must still join the
/// rendezvous and hold a verified full-group admission before starting any work.
pub struct RankLaunch<'a> {
    pub contract: &'a BoundedRunContract,
    pub plan_sha256: &'a str,
    pub global_rank: u32,
    pub dispatch_id: &'a str,
    /// Original durable intent time; remote launch/recovery cannot renew the wall allowance.
    pub dispatch_intent_unix_ms: u64,
}
/// Repository supervisor of rank work. The live implementation is SmartCluster-backed
/// ([`super::supervisor::SmartClusterSupervisor`]); implementations must never start work
/// from anything but a [`RankLaunch`].
pub trait Supervisor {
    fn launch(&mut self, launch: &RankLaunch<'_>) -> Result<SupervisorHandle>;
    /// Find work started under `dispatch_id` whose launch was never journaled.
    fn find(&mut self, dispatch_id: &str) -> Result<Option<SupervisorHandle>>;
    fn probe(&mut self, handle: &SupervisorHandle) -> Result<Liveness>;
    /// Stop and reap; idempotent; returns the final liveness, never `Running`.
    fn terminate(&mut self, handle: &SupervisorHandle) -> Result<Liveness>;
    /// Result retained by the supervisor for this work: `(signed envelope, bytes)`.
    fn retained_result(&mut self, handle: &SupervisorHandle) -> Result<Option<(Vec<u8>, Vec<u8>)>>;
}
mod coordinator;
mod types;
pub use coordinator::{Coordinator, PersistedAdmission};
pub use types::*;
pub(super) use types::{group_window_end, journal_window_closed};
