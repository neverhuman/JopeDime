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
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SupervisorHandle {
    pub supervisor: String,
    pub identity: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "cause", rename_all = "snake_case", deny_unknown_fields)]
pub enum DrainCause {
    RankFailed { global_rank: u32, detail: String },
    ControlLost { global_rank: u32 },
    GrantExpired { global_rank: u32 },
    DeadlineElapsed,
    WallBudgetElapsed { global_rank: u32 },
    Cancelled,
    CoordinatorRestart,
}
/// The one execution window of the synchronous group: it closes at the original deadline
/// or at the first participant grant expiry, whichever comes first. Supervision, restart
/// recovery, dispatch, admission, delivery, acknowledgement and every new commit use this
/// single gate; only finishing an already-journaled commit or drain may happen later.
pub fn group_window_closed(contract: &BoundedRunContract, now: u64) -> Option<DrainCause> {
    if now >= contract.original_deadline_unix_ms {
        return Some(DrainCause::DeadlineElapsed);
    }
    contract
        .participants
        .iter()
        .find(|p| now >= p.grant.body.expires_unix_ms)
        .map(|p| DrainCause::GrantExpired {
            global_rank: p.global_rank,
        })
}
/// Wall consumption starts at the first durable dispatch intent, not at recovery.
fn journal_window_closed(
    contract: &BoundedRunContract,
    records: &[(JournalRecord, String)],
    now: u64,
) -> Option<DrainCause> {
    if let Some(cause) = group_window_closed(contract, now) {
        return Some(cause);
    }
    records.iter().find_map(|(record, _)| {
        let Event::DispatchIntent { global_rank, .. } = record.event else {
            return None;
        };
        let participant = contract.participant(global_rank)?;
        (now.saturating_sub(record.recorded_unix_ms) >= participant.ceilings.wall_ms)
            .then_some(DrainCause::WallBudgetElapsed { global_rank })
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Committed,
    /// The rank whose failure caused the drain.
    Failed,
    /// Stopped by the whole-group drain while running (or already stopped).
    Drained,
    /// No work was ever started for this rank.
    NeverStarted,
    /// An authenticated result was retained but never acknowledged; not committed.
    CompletedUnacknowledged,
    /// An authenticated result was acknowledged but the group drained before commit.
    CompletedUncommitted,
    /// The group was refused before any rank started.
    Refused,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Committed,
    Drained,
    Cancelled,
    Refused,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Planned {
        original_deadline_unix_ms: u64,
        committed_updates_at_start: u64,
        planned_updates: u64,
    },
    PreflightAdmitted {
        receipt: PreflightReceipt,
    },
    PreflightRefused {
        census_sha256: String,
        reasons: Vec<RefusalReason>,
    },
    GroupAdmitted {
        admission_sha256: String,
        admission: GroupAdmission,
    },
    /// Every participant was sent the persisted admission; ranks may now start work.
    AdmissionDelivered {
        admission_sha256: String,
    },
    GroupRefused {
        refusal: GroupRefusal,
    },
    DispatchIntent {
        global_rank: u32,
        dispatch_id: String,
    },
    Launched {
        global_rank: u32,
        dispatch_id: String,
        handle: SupervisorHandle,
    },
    ResultAcknowledged {
        global_rank: u32,
        delivery_id: u64,
        result_sha256: String,
        envelope_sha256: String,
        updates: u64,
    },
    GroupCommitted {
        committed_updates_total: u64,
        /// `(global_rank, result_sha256)` in rank order (JSON map keys cannot be integers
        /// inside a tagged record).
        results: Vec<(u32, String)>,
    },
    CancelRequested {
        request_sha256: String,
    },
    CoordinatorRestarted {
        boot_id: String,
        pid: u32,
    },
    DrainStarted {
        cause: DrainCause,
    },
    Disposition {
        global_rank: u32,
        disposition: Disposition,
        detail: String,
    },
    Terminal {
        outcome: Outcome,
        committed_updates_total: u64,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JournalRecord {
    pub schema: String,
    pub sequence: u64,
    pub previous_sha256: Option<String>,
    pub plan_sha256: String,
    pub recorded_unix_ms: u64,
    pub event: Event,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CancelRequest {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub reason: String,
    pub requested_unix_ms: u64,
}

/// The seven externally visible rank states.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RankState {
    Scheduled,
    Admitted,
    Running,
    Committed,
    Failed,
    Drained,
    Unknown,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RankStatus {
    pub global_rank: u32,
    pub host: String,
    pub device_uuid: Option<String>,
    pub state: RankState,
    pub disposition: Option<Disposition>,
    pub detail: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StatusReport {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub original_deadline_unix_ms: u64,
    pub journal_records: u64,
    pub journal_tip_sha256: Option<String>,
    pub coordinator_live: bool,
    pub cancel_requested: bool,
    pub outcome: Option<Outcome>,
    pub committed_updates_at_start: u64,
    pub committed_updates_total: u64,
    pub ranks: Vec<RankStatus>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifyReport {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub journal_records: u64,
    pub journal_tip_sha256: Option<String>,
    pub outcome: Option<Outcome>,
    pub committed_updates_total: u64,
    pub results_verified: u64,
    pub numerical_work_started: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlanReport {
    pub plan_sha256: String,
    pub created: bool,
    pub review: serde_json::Value,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CancelReport {
    pub plan_sha256: String,
    pub request: CancelRequest,
    pub already_requested: bool,
    pub journaled: bool,
    pub delivered_to_live_coordinator: bool,
    pub drain_pending_reconcile: bool,
    pub outcome: Option<Outcome>,
}

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
/// Existing physical run directory; never created by read-only operations.
fn existing_dir(dir: &Path) -> Result<()> {
    ensure!(dir.is_absolute(), "run directory must be absolute");
    let meta = fs::symlink_metadata(dir)
        .with_context(|| format!("run directory {} is absent", dir.display()))?;
    ensure!(
        meta.is_dir() && fs::canonicalize(dir)? == dir,
        "run directory must be a physical directory without symlinks"
    );
    Ok(())
}
fn create_dir(dir: &Path) -> Result<()> {
    ensure!(dir.is_absolute(), "run directory must be absolute");
    let parent = dir.parent().context("run directory parent")?;
    ensure!(
        fs::canonicalize(parent)? == parent,
        "run directory parent must be physical"
    );
    match fs::symlink_metadata(dir) {
        Ok(m) => ensure!(
            m.is_dir() && !m.file_type().is_symlink(),
            "run directory must be physical"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir(dir)?;
            fs::File::open(parent)?.sync_all()?;
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
fn lock_file(dir: &Path, create: bool) -> Result<fs::File> {
    let f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(create)
        .truncate(false)
        .custom_flags(libc::O_NOFOLLOW)
        .open(dir.join(LOCK))
        .context("controller lock (run directory not planned?)")?;
    Ok(f)
}
fn try_lock(f: &fs::File) -> bool {
    unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
}
/// Non-interfering liveness probe: reads `/proc/locks` for an exclusive flock on the
/// controller lock's inode. It never takes a lock itself, so it cannot block a writer.
fn coordinator_lock_holder(dir: &Path) -> Result<Option<u32>> {
    let meta = match fs::symlink_metadata(dir.join(LOCK)) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    ensure!(meta.is_file(), "controller lock is not a regular file");
    let dev = meta.dev();
    let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
    let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
    for line in fs::read_to_string("/proc/locks")?.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 6 || f[1] != "FLOCK" || f[3] != "WRITE" {
            continue;
        }
        let id: Vec<&str> = f[5].split(':').collect();
        if id.len() == 3
            && u64::from_str_radix(id[0], 16) == Ok(major)
            && u64::from_str_radix(id[1], 16) == Ok(minor)
            && id[2].parse::<u64>() == Ok(meta.ino())
        {
            return Ok(Some(f[4].parse().unwrap_or(0)));
        }
    }
    Ok(None)
}
fn load_plan(dir: &Path) -> Result<(BoundedRunContract, String)> {
    let bytes = read_bounded(&dir.join(PLAN), MAX_CONTRACT_BYTES as u64)?;
    let contract = BoundedRunContract::parse(&bytes).map_err(|e| anyhow!("{e}"))?;
    ensure!(
        contract.canonical_bytes()? == bytes,
        "frozen plan bytes are not the canonical contract"
    );
    Ok((contract, hash(&bytes)))
}
fn load_journal(dir: &Path) -> Result<Vec<(JournalRecord, String)>> {
    let jdir = dir.join(JOURNAL);
    match fs::symlink_metadata(&jdir) {
        Ok(m) => ensure!(
            m.is_dir() && !m.file_type().is_symlink(),
            "journal must be a physical directory"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    }
    let mut sequences = BTreeSet::new();
    for entry in fs::read_dir(&jdir)? {
        let name = entry?
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("non-UTF8 journal entry"))?;
        if name.starts_with(".exchange-pending-") {
            continue; // interrupted staging; never part of the chain
        }
        let seq = name
            .strip_suffix(".json")
            .filter(|s| s.len() == 8 && s.bytes().all(|b| b.is_ascii_digit()))
            .with_context(|| format!("unknown journal entry `{name}`"))?
            .parse::<u64>()?;
        sequences.insert(seq);
        ensure!(sequences.len() <= MAX_RECORDS, "journal record bound");
    }
    ensure!(
        sequences.iter().copied().eq(1..=sequences.len() as u64),
        "journal sequence is not contiguous from 1"
    );
    let mut out = Vec::with_capacity(sequences.len());
    let mut previous = None;
    for seq in sequences {
        let bytes = read_bounded(&jdir.join(format!("{seq:08}.json")), MAX_RECORD_BYTES)?;
        let r: JournalRecord = serde_json::from_slice(&bytes)?;
        ensure!(
            r.schema == JOURNAL_SCHEMA && r.sequence == seq && r.previous_sha256 == previous,
            "journal chain broken at record {seq}"
        );
        let sha = hash(&bytes);
        previous = Some(sha.clone());
        out.push((r, sha));
    }
    Ok(out)
}

/// State derived by replaying the journal under every ordering invariant.
#[derive(Default, Clone)]
struct Replay {
    preflight: Option<PreflightReceipt>,
    admission: Option<(GroupAdmission, String)>,
    delivered: bool,
    refused: bool,
    intents: BTreeMap<u32, String>,
    launched: BTreeMap<u32, SupervisorHandle>,
    acked: BTreeMap<u32, String>,
    committed: Option<u64>,
    cancel: bool,
    drain: Option<DrainCause>,
    dispositions: BTreeMap<u32, (Disposition, String)>,
    terminal: Option<(Outcome, u64)>,
}
fn replay(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    records: &[(JournalRecord, String)],
) -> Result<Replay> {
    let n = contract.world_size() as u32;
    let start = contract.counters.committed_updates_at_start;
    let mut s = Replay::default();
    for (i, (r, _)) in records.iter().enumerate() {
        let at = r.sequence;
        ensure!(
            r.plan_sha256 == plan_sha256,
            "journal record {at} belongs to another plan"
        );
        ensure!(
            s.terminal.is_none(),
            "journal record {at} follows the terminal record"
        );
        let rank_ok = |rank: u32| -> Result<()> {
            ensure!(
                rank < n,
                "journal record {at} names rank {rank} outside the plan"
            );
            Ok(())
        };
        // Every step that starts work or confers authority falls inside the group
        // window. Finishing records (drain, dispositions, terminal) may come later.
        if matches!(
            r.event,
            Event::DispatchIntent { .. }
                | Event::Launched { .. }
                | Event::GroupAdmitted { .. }
                | Event::AdmissionDelivered { .. }
                | Event::ResultAcknowledged { .. }
                | Event::GroupCommitted { .. }
        ) {
            if let Some(cause) = journal_window_closed(contract, &records[..i], r.recorded_unix_ms)
            {
                bail!(
                    "journal record {at} was journaled after the group window closed ({cause:?})"
                );
            }
        }
        match &r.event {
            Event::Planned {
                original_deadline_unix_ms,
                committed_updates_at_start,
                planned_updates,
            } => ensure!(
                i == 0
                    && *original_deadline_unix_ms == contract.original_deadline_unix_ms
                    && *committed_updates_at_start == start
                    && *planned_updates == contract.counters.planned_updates,
                "journal must open with the plan's original deadline and counters"
            ),
            _ if i == 0 => bail!("journal must open with the Planned record"),
            Event::PreflightAdmitted { receipt } => {
                ensure!(
                    s.admission.is_none()
                        && !s.refused
                        && receipt.plan_sha256 == plan_sha256
                        && receipt.run == contract.run
                        && receipt.attempt == contract.attempt,
                    "journal record {at}: preflight after admission or for another plan"
                );
                s.preflight = Some(receipt.clone());
            }
            Event::PreflightRefused { .. } => ensure!(
                s.admission.is_none() && !s.refused,
                "journal record {at}: preflight after admission"
            ),
            Event::DispatchIntent {
                global_rank,
                dispatch_id,
            } => {
                rank_ok(*global_rank)?;
                ensure!(
                    s.preflight.is_some()
                        && s.admission.is_none()
                        && !s.refused
                        && !s.cancel
                        && s.drain.is_none()
                        && *dispatch_id
                            == rendezvous::dispatch_id(plan_sha256, contract.attempt, *global_rank)?
                        && s.intents.insert(*global_rank, dispatch_id.clone()).is_none(),
                    "journal record {at}: dispatch intent without preflight, after admission/refusal/drain, or repeated"
                );
            }
            Event::Launched {
                global_rank,
                dispatch_id,
                handle,
            } => {
                rank_ok(*global_rank)?;
                ensure!(
                    s.intents.get(global_rank) == Some(dispatch_id)
                        && s.admission.is_none()
                        && !s.refused
                        && s.drain.is_none()
                        && s.launched.insert(*global_rank, handle.clone()).is_none(),
                    "journal record {at}: launch without its persisted intent or repeated"
                );
            }
            Event::GroupAdmitted {
                admission_sha256,
                admission,
            } => {
                ensure!(
                    s.admission.is_none()
                        && !s.refused
                        && s.preflight.is_some()
                        && s.drain.is_none()
                        && s.launched.len() as u32 == n
                        && *admission_sha256 == rendezvous::admission_sha256(admission)?,
                    "journal record {at}: admission without preflight and persisted ownership of every rank, or repeated"
                );
                rendezvous::verify_group_admission(contract, plan_sha256, admission)?;
                s.admission = Some((admission.clone(), admission_sha256.clone()));
            }
            Event::AdmissionDelivered { admission_sha256 } => {
                ensure!(
                    s.admission.as_ref().map(|(_, h)| h) == Some(admission_sha256)
                        && !s.delivered
                        && s.drain.is_none(),
                    "journal record {at}: delivery of an unpersisted admission"
                );
                s.delivered = true;
            }
            Event::GroupRefused { refusal } => {
                ensure!(
                    s.admission.is_none()
                        && !s.refused
                        && s.drain.is_none()
                        && refusal.started_ranks == 0
                        && refusal.committed_updates == 0,
                    "journal record {at}: refusal after admission"
                );
                s.refused = true;
            }
            Event::ResultAcknowledged {
                global_rank,
                result_sha256,
                updates,
                ..
            } => {
                rank_ok(*global_rank)?;
                ensure!(
                    s.delivered
                        && s.committed.is_none()
                        && *updates == contract.counters.planned_updates
                        && s.acked
                            .insert(*global_rank, result_sha256.clone())
                            .is_none(),
                    "journal record {at}: result acknowledgement invalid or repeated"
                );
            }
            Event::GroupCommitted {
                committed_updates_total,
                results,
            } => {
                ensure!(
                    s.committed.is_none()
                        && s.drain.is_none()
                        && s.acked.len() as u32 == n
                        && results.iter().cloned().eq(s.acked.clone())
                        && *committed_updates_total == start + contract.counters.planned_updates,
                    "journal record {at}: group commit without every authenticated result, after drain, or repeated"
                );
                s.committed = Some(*committed_updates_total);
            }
            Event::CancelRequested { .. } => {
                ensure!(!s.cancel, "journal record {at}: repeated cancellation");
                s.cancel = true;
            }
            Event::CoordinatorRestarted { .. } => {}
            Event::DrainStarted { cause } => {
                ensure!(
                    s.drain.is_none()
                        && s.committed.is_none()
                        && !s.refused
                        && !s.intents.is_empty(),
                    "journal record {at}: drain repeated, after commit/refusal, or with nothing dispatched"
                );
                s.drain = Some(cause.clone());
            }
            Event::Disposition {
                global_rank,
                disposition,
                detail,
            } => {
                rank_ok(*global_rank)?;
                let allowed = match disposition {
                    Disposition::Committed => s.committed.is_some(),
                    Disposition::Refused => s.refused,
                    Disposition::CompletedUncommitted => {
                        s.drain.is_some() && s.acked.contains_key(global_rank)
                    }
                    Disposition::NeverStarted => {
                        (s.drain.is_some() || (s.cancel && s.intents.is_empty()))
                            && !s.launched.contains_key(global_rank)
                    }
                    Disposition::Failed
                    | Disposition::Drained
                    | Disposition::CompletedUnacknowledged => s.drain.is_some(),
                };
                ensure!(
                    allowed
                        && s.dispositions
                            .insert(*global_rank, (*disposition, detail.clone()))
                            .is_none(),
                    "journal record {at}: disposition not permitted by the recorded state"
                );
            }
            Event::Terminal {
                outcome,
                committed_updates_total,
            } => {
                let consistent = match outcome {
                    Outcome::Committed => s.committed == Some(*committed_updates_total),
                    Outcome::Refused => s.refused && *committed_updates_total == start,
                    Outcome::Drained | Outcome::Cancelled => {
                        (s.drain.is_some() || (s.cancel && s.intents.is_empty()))
                            && s.committed.is_none()
                            && *committed_updates_total == start
                    }
                };
                ensure!(
                    consistent && s.dispositions.len() as u32 == n,
                    "journal record {at}: terminal outcome inconsistent or a rank lacks a disposition"
                );
                s.terminal = Some((*outcome, *committed_updates_total));
            }
        }
    }
    Ok(s)
}
fn rank_status(contract: &BoundedRunContract, s: &Replay, live: bool) -> Vec<RankStatus> {
    contract
        .participants
        .iter()
        .map(|p| {
            let r = p.global_rank;
            let (state, disposition, detail) = if let Some((d, detail)) = s.dispositions.get(&r) {
                let state = match d {
                    Disposition::Committed => RankState::Committed,
                    Disposition::Failed => RankState::Failed,
                    _ => RankState::Drained,
                };
                (state, Some(*d), detail.clone())
            } else if !s.intents.contains_key(&r) {
                (RankState::Scheduled, None, "planned; not dispatched".into())
            } else if !s.launched.contains_key(&r) {
                (
                    RankState::Unknown,
                    None,
                    "dispatch intent persisted without a launch record; reconciliation decides"
                        .into(),
                )
            } else if !live {
                (
                    RankState::Unknown,
                    None,
                    "launched but no live coordinator holds the run; reconciliation required"
                        .into(),
                )
            } else if s.admission.is_none() {
                (
                    RankState::Scheduled,
                    None,
                    "launched; awaiting full-group admission, no work permitted".into(),
                )
            } else if !s.delivered {
                (
                    RankState::Admitted,
                    None,
                    "full-group admission persisted; not yet acknowledged to participants".into(),
                )
            } else if s.acked.contains_key(&r) {
                (
                    RankState::Running,
                    None,
                    "authenticated result acknowledged; awaiting group".into(),
                )
            } else {
                (
                    RankState::Running,
                    None,
                    "launched under a live coordinator".into(),
                )
            };
            RankStatus {
                global_rank: r,
                host: p.host.host.clone(),
                device_uuid: p.device.as_ref().map(|d| d.uuid.clone()),
                state,
                disposition,
                detail,
            }
        })
        .collect()
}
fn build_status(
    dir: &Path,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    records: &[(JournalRecord, String)],
    s: &Replay,
    live: bool,
) -> Result<StatusReport> {
    let start = contract.counters.committed_updates_at_start;
    Ok(StatusReport {
        schema: STATUS_SCHEMA.into(),
        run: contract.run.clone(),
        attempt: contract.attempt,
        plan_sha256: plan_sha256.into(),
        original_deadline_unix_ms: contract.original_deadline_unix_ms,
        journal_records: records.len() as u64,
        journal_tip_sha256: records.last().map(|(_, h)| h.clone()),
        coordinator_live: live,
        cancel_requested: s.cancel || fs::symlink_metadata(dir.join(CANCEL)).is_ok(),
        outcome: s.terminal.map(|(o, _)| o),
        committed_updates_at_start: start,
        committed_updates_total: s.committed.unwrap_or(start),
        ranks: rank_status(contract, s, live),
    })
}

/// Read-only preflight of a pinned, unstarted plan. This never journals admission,
/// holds an allocation, or contacts a host; the coordinator must recheck at dispatch.
pub fn preflight_dry_run(
    dir: &Path,
    expected_plan_sha256: &str,
    census_path: &Path,
    now: u64,
) -> Result<PreflightReceipt> {
    verify(dir, Some(expected_plan_sha256))?;
    let (contract, plan_sha256) = load_plan(dir)?;
    ensure!(
        plan_sha256 == expected_plan_sha256,
        "frozen plan changed during preflight"
    );
    contract.validate_at(now).map_err(|e| anyhow!("{e}"))?;
    let records = load_journal(dir)?;
    let state = replay(&contract, &plan_sha256, &records)?;
    ensure!(
        state.terminal.is_none()
            && state.intents.is_empty()
            && state.admission.is_none()
            && !state.cancel
            && fs::symlink_metadata(dir.join(CANCEL)).is_err()
            && coordinator_lock_holder(dir)?.is_none(),
        "dry-run preflight requires an unstarted, uncancelled plan without a live coordinator"
    );
    let bytes = read_bounded(census_path, census::MAX_CENSUS_BYTES as u64)?;
    let census = HostCensus::from_bytes(&bytes)?;
    Ok(census::preflight(
        &contract,
        &plan_sha256,
        &census,
        &hash(&bytes),
        now,
    )?)
}

/// Read-only Status. Creates nothing, takes no lock and contacts no participant.
pub fn status(dir: &Path) -> Result<StatusReport> {
    existing_dir(dir)?;
    let (contract, plan_sha256) = load_plan(dir)?;
    let records = load_journal(dir)?;
    let s = replay(&contract, &plan_sha256, &records)?;
    let live = coordinator_lock_holder(dir)?.is_some();
    build_status(dir, &contract, &plan_sha256, &records, &s, live)
}

/// Read-only verification of the frozen plan, the journal chain and invariants, any
/// cancellation request and every acknowledged result. It never launches, probes or
/// contacts a participant, and never starts numerical work.
pub fn verify(dir: &Path, expected_plan_sha256: Option<&str>) -> Result<VerifyReport> {
    existing_dir(dir)?;
    let (contract, plan_sha256) = load_plan(dir)?;
    if let Some(expected) = expected_plan_sha256 {
        ensure!(
            expected == plan_sha256,
            "frozen plan digest differs from the reviewed plan"
        );
    }
    let records = load_journal(dir)?;
    ensure!(!records.is_empty(), "planned run has no journal");
    let s = replay(&contract, &plan_sha256, &records)?;
    if fs::symlink_metadata(dir.join(CANCEL)).is_ok() {
        let request: CancelRequest =
            serde_json::from_slice(&read_bounded(&dir.join(CANCEL), MAX_RECORD_BYTES)?)?;
        ensure!(
            request.schema == CANCEL_SCHEMA
                && request.plan_sha256 == plan_sha256
                && request.run == contract.run
                && request.attempt == contract.attempt,
            "cancellation request is bound to another plan"
        );
    }
    let mut verified = 0u64;
    if let Some((_, admission)) = &s.admission {
        for (rank, sha) in &s.acked {
            let p = contract.participant(*rank).context("acknowledged rank")?;
            let bytes = read_bounded(
                &dir.join(RESULTS).join(&p.result_destination.path),
                p.result_destination.max_bytes,
            )?;
            let envelope = read_bounded(
                &dir.join(ENVELOPES).join(format!("rank-{rank}.json")),
                u64::from(contract.rendezvous.max_message_bytes),
            )?;
            let v =
                rendezvous::verify_result(&contract, &plan_sha256, admission, &envelope, &bytes)?;
            ensure!(
                v.body().result_sha256 == *sha && v.body().global_rank == *rank,
                "committed result bytes differ from the acknowledged digest"
            );
            let acknowledged = records.iter().any(|(r, _)| {
                matches!(&r.event, Event::ResultAcknowledged { global_rank, envelope_sha256, .. }
                    if global_rank == rank && *envelope_sha256 == hash(&envelope))
            });
            ensure!(
                acknowledged,
                "result envelope differs from the acknowledged envelope"
            );
            verified += 1;
        }
    }
    Ok(VerifyReport {
        schema: VERIFY_SCHEMA.into(),
        run: contract.run.clone(),
        attempt: contract.attempt,
        plan_sha256,
        journal_records: records.len() as u64,
        journal_tip_sha256: records.last().map(|(_, h)| h.clone()),
        outcome: s.terminal.map(|(o, _)| o),
        committed_updates_total: s
            .committed
            .unwrap_or(contract.counters.committed_updates_at_start),
        results_verified: verified,
        numerical_work_started: false,
    })
}

/// Outcome of a terminal attempt, for planning its successor.
pub fn attempt_outcome(dir: &Path, plan_sha256: &str) -> Result<AttemptOutcome> {
    let report = verify(dir, Some(plan_sha256))?;
    Ok(AttemptOutcome {
        plan_sha256: report.plan_sha256,
        journal_tip_sha256: report.journal_tip_sha256.context("empty journal")?,
        terminal: report.outcome.is_some(),
        committed_updates_total: report.committed_updates_total,
    })
}

fn append(
    dir: &Path,
    records: &mut Vec<(JournalRecord, String)>,
    plan_sha256: &str,
    event: Event,
    now: u64,
) -> Result<()> {
    let record = JournalRecord {
        schema: JOURNAL_SCHEMA.into(),
        sequence: records.len() as u64 + 1,
        previous_sha256: records.last().map(|(_, h)| h.clone()),
        plan_sha256: plan_sha256.into(),
        recorded_unix_ms: now,
        event,
    };
    let bytes = serde_json::to_vec(&record)?;
    ensure!(
        bytes.len() as u64 <= MAX_RECORD_BYTES,
        "journal record bound"
    );
    let name = PathBuf::from(format!("{:08}.json", record.sequence));
    let jdir = dir.join(JOURNAL);
    if !jdir.exists() {
        fs::create_dir(&jdir)?;
        fs::File::open(dir)?.sync_all()?;
    }
    // No-replace publication: a conflicting record at this sequence is refused.
    publish_immutable(&jdir, &name, &bytes)?;
    records.push((record, hash(&bytes)));
    Ok(())
}

/// Freeze a reviewed plan. Refuses legacy, invalid and underfunded contracts before any
/// directory is created; re-planning the identical bytes is idempotent.
pub fn plan(
    dir: &Path,
    contract_bytes: &[u8],
    now: u64,
    predecessor_dir: Option<&Path>,
) -> Result<PlanReport> {
    let contract = BoundedRunContract::parse(contract_bytes).map_err(|e| anyhow!("{e}"))?;
    contract.validate_at(now).map_err(|e| anyhow!("{e}"))?;
    match (&contract.predecessor, predecessor_dir) {
        (Some(p), Some(previous_dir)) => {
            let (previous, previous_sha) = {
                existing_dir(previous_dir)?;
                load_plan(previous_dir)?
            };
            ensure!(
                previous_sha == p.plan_sha256,
                "predecessor plan digest differs"
            );
            let outcome = attempt_outcome(previous_dir, &previous_sha)?;
            contract
                .validate_successor(&previous, &outcome)
                .map_err(|e| anyhow!("{e}"))?;
        }
        (Some(_), None) => bail!(
            "{}",
            PlanRefusal::Invalid("successor attempt requires the predecessor run directory".into())
        ),
        (None, Some(_)) => bail!(
            "{}",
            PlanRefusal::Invalid("first attempt cannot name a predecessor run directory".into())
        ),
        (None, None) => {}
    }
    let canonical = contract.canonical_bytes()?;
    let plan_sha256 = hash(&canonical);
    create_dir(dir)?;
    let lock = lock_file(dir, true)?;
    ensure!(
        try_lock(&lock),
        "another controller holds this run directory"
    );
    let path = dir.join(PLAN);
    let created = if fs::symlink_metadata(&path).is_ok() {
        ensure!(
            read_bounded(&path, MAX_CONTRACT_BYTES as u64)? == canonical,
            "run directory already holds a different plan"
        );
        false
    } else {
        publish_immutable(dir, Path::new(PLAN), &canonical)?;
        true
    };
    let mut records = load_journal(dir)?;
    if records.is_empty() {
        append(
            dir,
            &mut records,
            &plan_sha256,
            Event::Planned {
                original_deadline_unix_ms: contract.original_deadline_unix_ms,
                committed_updates_at_start: contract.counters.committed_updates_at_start,
                planned_updates: contract.counters.planned_updates,
            },
            now,
        )?;
    }
    replay(&contract, &plan_sha256, &records)?;
    Ok(PlanReport {
        review: contract.review(&plan_sha256),
        plan_sha256,
        created,
    })
}

/// Idempotent cancellation bound to the reviewed plan digest.
pub fn cancel(dir: &Path, plan_sha256: &str, reason: &str, now: u64) -> Result<CancelReport> {
    existing_dir(dir)?;
    let (contract, actual) = load_plan(dir)?;
    ensure!(actual == plan_sha256, "cancellation names a different plan");
    ensure!(
        !reason.trim().is_empty() && reason.len() <= 512 && !reason.chars().any(|c| c.is_control()),
        "cancellation reason must be 1..512 printable characters"
    );
    let path = dir.join(CANCEL);
    let (request, already) = if fs::symlink_metadata(&path).is_ok() {
        let existing: CancelRequest =
            serde_json::from_slice(&read_bounded(&path, MAX_RECORD_BYTES)?)?;
        ensure!(
            existing.schema == CANCEL_SCHEMA && existing.plan_sha256 == plan_sha256,
            "existing cancellation is bound to another plan"
        );
        (existing, true)
    } else {
        let request = CancelRequest {
            schema: CANCEL_SCHEMA.into(),
            run: contract.run.clone(),
            attempt: contract.attempt,
            plan_sha256: plan_sha256.into(),
            reason: reason.into(),
            requested_unix_ms: now,
        };
        publish_immutable(dir, Path::new(CANCEL), &serde_json::to_vec(&request)?)?;
        // A racing identical request would have been refused as a conflict above.
        let stored: CancelRequest =
            serde_json::from_slice(&read_bounded(&path, MAX_RECORD_BYTES)?)?;
        (stored, false)
    };
    let request_sha256 = hash(&serde_json::to_vec(&request)?);
    let lock = lock_file(dir, false)?;
    let mut journaled = false;
    let mut drain_pending = false;
    let delivered_live = !try_lock(&lock);
    let mut records = load_journal(dir)?;
    let s = replay(&contract, plan_sha256, &records)?;
    if !delivered_live && s.terminal.is_none() {
        if !s.cancel {
            append(
                dir,
                &mut records,
                plan_sha256,
                Event::CancelRequested { request_sha256 },
                now,
            )?;
            journaled = true;
        }
        if s.intents.is_empty() {
            // Nothing was dispatched, so nothing can have started: finish here.
            finish_cancel_before_dispatch(dir, &contract, plan_sha256, &mut records, now)?;
            journaled = true;
        } else {
            drain_pending = true;
        }
    }
    let s = replay(&contract, plan_sha256, &records)?;
    Ok(CancelReport {
        plan_sha256: plan_sha256.into(),
        request,
        already_requested: already,
        journaled,
        delivered_to_live_coordinator: delivered_live,
        drain_pending_reconcile: drain_pending,
        outcome: s.terminal.map(|(o, _)| o),
    })
}
/// A journaled cancellation with nothing dispatched ends the attempt: every rank is
/// never-started and nothing is committed.
fn finish_cancel_before_dispatch(
    dir: &Path,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    records: &mut Vec<(JournalRecord, String)>,
    now: u64,
) -> Result<()> {
    let s = replay(contract, plan_sha256, records)?;
    ensure!(
        s.cancel && s.intents.is_empty() && s.terminal.is_none(),
        "cancellation before dispatch requires a journaled request and no dispatch"
    );
    for p in &contract.participants {
        if !s.dispositions.contains_key(&p.global_rank) {
            append(
                dir,
                records,
                plan_sha256,
                Event::Disposition {
                    global_rank: p.global_rank,
                    disposition: Disposition::NeverStarted,
                    detail: "cancelled before dispatch".into(),
                },
                now,
            )?;
        }
    }
    append(
        dir,
        records,
        plan_sha256,
        Event::Terminal {
            outcome: Outcome::Cancelled,
            committed_updates_total: contract.counters.committed_updates_at_start,
        },
        now,
    )
}

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
enum Retained {
    Absent,
    Valid(Box<VerifiedResult>),
    Invalid(String),
}
/// Proof that an admission was journaled before any participant was told of it.
pub struct PersistedAdmission {
    admission: GroupAdmission,
    admission_sha256: String,
}
impl PersistedAdmission {
    pub fn admission_sha256(&self) -> &str {
        &self.admission_sha256
    }
}

/// The single writer of one planned run attempt.
pub struct Coordinator<S: Supervisor> {
    dir: PathBuf,
    contract: BoundedRunContract,
    plan_sha256: String,
    records: Vec<(JournalRecord, String)>,
    supervisor: S,
    needs_reconcile: bool,
    _lock: fs::File,
}
impl<S: Supervisor> Coordinator<S> {
    /// Opens the run bound to the reviewed plan digest. A non-terminal journal that already
    /// dispatched work is a coordinator restart: nothing further may be dispatched until
    /// [`Coordinator::reconcile`] has settled every owned rank.
    pub fn open(dir: &Path, plan_sha256: &str, supervisor: S, now: u64) -> Result<Self> {
        existing_dir(dir)?;
        let (contract, actual) = load_plan(dir)?;
        ensure!(
            actual == plan_sha256,
            "coordinator binds only the reviewed plan artifact"
        );
        let lock = lock_file(dir, false)?;
        ensure!(try_lock(&lock), "another coordinator owns this run");
        let records = load_journal(dir)?;
        let s = replay(&contract, plan_sha256, &records)?;
        ensure!(!records.is_empty(), "planned run has no journal");
        let mut c = Self {
            dir: dir.into(),
            contract,
            plan_sha256: plan_sha256.into(),
            records,
            supervisor,
            needs_reconcile: false,
            _lock: lock,
        };
        if s.terminal.is_none() && !s.intents.is_empty() {
            let me = crate::eval_registry::ProcessIdentity::current()?;
            c.append(
                Event::CoordinatorRestarted {
                    boot_id: me.boot_id,
                    pid: me.pid,
                },
                now,
            )?;
            c.needs_reconcile = true;
        }
        Ok(c)
    }
    pub fn contract(&self) -> &BoundedRunContract {
        &self.contract
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn needs_reconcile(&self) -> bool {
        self.needs_reconcile
    }
    pub fn supervisor(&mut self) -> &mut S {
        &mut self.supervisor
    }
    pub fn journal(&self) -> Vec<JournalRecord> {
        self.records.iter().map(|(r, _)| r.clone()).collect()
    }
    fn append(&mut self, event: Event, now: u64) -> Result<()> {
        append(&self.dir, &mut self.records, &self.plan_sha256, event, now)?;
        replay(&self.contract, &self.plan_sha256, &self.records)?;
        Ok(())
    }
    fn state(&self) -> Result<Replay> {
        replay(&self.contract, &self.plan_sha256, &self.records)
    }
    fn window_closed(&self, now: u64) -> Option<DrainCause> {
        journal_window_closed(&self.contract, &self.records, now)
    }
    pub fn status(&self) -> Result<StatusReport> {
        build_status(
            &self.dir,
            &self.contract,
            &self.plan_sha256,
            &self.records,
            &self.state()?,
            true,
        )
    }
    /// Journal a preflight of an untrusted census record. A refusal is journaled too
    /// (for audit) and returned; it is not terminal while the window remains.
    pub fn preflight(&mut self, census_bytes: &[u8], now: u64) -> Result<PreflightReceipt> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && !self.needs_reconcile,
            "preflight is only permitted before admission"
        );
        self.contract.validate_at(now).map_err(|e| anyhow!("{e}"))?;
        let census = HostCensus::from_bytes(census_bytes)?;
        let census_sha256 = hash(census_bytes);
        match census::preflight(
            &self.contract,
            &self.plan_sha256,
            &census,
            &census_sha256,
            now,
        ) {
            Ok(receipt) => {
                self.append(
                    Event::PreflightAdmitted {
                        receipt: receipt.clone(),
                    },
                    now,
                )?;
                Ok(receipt)
            }
            Err(refusal) => {
                self.append(
                    Event::PreflightRefused {
                        census_sha256,
                        reasons: refusal.reasons.clone(),
                    },
                    now,
                )?;
                Err(refusal.into())
            }
        }
    }
    /// Open a rendezvous; requires a preflight receipt still fresh under the frozen
    /// telemetry policy.
    pub fn open_rendezvous(
        &self,
        coordinator: &SigningKey,
        coordinator_boot_id: &str,
        nonce: &[u8; 16],
        now: u64,
    ) -> Result<Rendezvous> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && !self.needs_reconcile,
            "rendezvous is only opened before admission"
        );
        let receipt = s
            .preflight
            .context("rendezvous requires an admitted preflight")?;
        ensure!(
            now.saturating_sub(receipt.checked_unix_ms) <= self.contract.telemetry.max_age_ms,
            "preflight receipt is stale; preflight a fresh census"
        );
        Rendezvous::open(
            &self.contract,
            &self.plan_sha256,
            coordinator,
            coordinator_boot_id,
            nonce,
            now,
        )
    }
    /// Persist the full-group admission *before* any participant learns of it. Requires
    /// the persisted dispatch and launch identity (ownership) of every rank.
    pub fn record_admission(
        &mut self,
        admission: GroupAdmission,
        now: u64,
    ) -> Result<PersistedAdmission> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none()
                && s.admission.is_none()
                && s.drain.is_none()
                && !self.needs_reconcile,
            "admission is recorded once, while the run is live"
        );
        ensure!(
            s.launched.len() == self.contract.world_size(),
            "ownership of every rank must be persisted before admission"
        );
        if let Some(cause) = self.window_closed(now) {
            bail!("no admission is journaled after the group window closed ({cause:?})");
        }
        rendezvous::verify_group_admission(&self.contract, &self.plan_sha256, &admission)?;
        let admission_sha256 = rendezvous::admission_sha256(&admission)?;
        self.append(
            Event::GroupAdmitted {
                admission_sha256: admission_sha256.clone(),
                admission: admission.clone(),
            },
            now,
        )?;
        Ok(PersistedAdmission {
            admission,
            admission_sha256,
        })
    }
    /// Acknowledge an admission to participants; only a persisted admission can be sent.
    /// Every participant must receive it; a failed delivery drains the whole group.
    pub fn deliver_admission(
        &mut self,
        persisted: &PersistedAdmission,
        connected: &mut [Connected],
        now: u64,
    ) -> Result<()> {
        self.deliver_admission_with_clock(persisted, connected, || now)
    }
    /// Refresh the original execution window around each potentially blocking delivery.
    pub fn deliver_admission_with_clock(
        &mut self,
        persisted: &PersistedAdmission,
        connected: &mut [Connected],
        clock: impl Fn() -> u64,
    ) -> Result<()> {
        let now = clock();
        let s = self.state()?;
        ensure!(
            s.admission.as_ref().map(|(_, h)| h.as_str())
                == Some(persisted.admission_sha256.as_str())
                && !s.delivered
                && s.drain.is_none(),
            "admission is not the journaled, undelivered admission"
        );
        // Delivery is when ranks may start work: after the window closed, drain instead.
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause.clone(), now)?;
            bail!("admission not delivered: the group window closed ({cause:?}); group drained");
        }
        let ranks: BTreeSet<u32> = connected.iter().map(|c| c.global_rank).collect();
        if !ranks
            .iter()
            .copied()
            .eq(0..self.contract.world_size() as u32)
            || ranks.len() != connected.len()
        {
            let missing = (0..self.contract.world_size() as u32)
                .find(|r| !ranks.contains(r))
                .unwrap_or(0);
            self.drain(
                DrainCause::ControlLost {
                    global_rank: missing,
                },
                now,
            )?;
            bail!("admission delivery requires one control stream per planned rank");
        }
        let bytes = serde_json::to_vec(&persisted.admission)?;
        let limit = self.contract.rendezvous.max_message_bytes;
        for c in connected.iter_mut() {
            let now = clock();
            if let Some(cause) = self.window_closed(now) {
                self.drain(cause.clone(), now)?;
                bail!("admission delivery window closed ({cause:?}); group drained");
            }
            if let Err(e) = rendezvous::write_frame(&mut c.stream, &bytes, limit) {
                let rank = c.global_rank;
                self.drain(DrainCause::ControlLost { global_rank: rank }, clock())?;
                return Err(e.context(format!(
                    "admission delivery to rank {rank} failed; group drained"
                )));
            }
        }
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause.clone(), now)?;
            bail!("admission delivery window closed ({cause:?}); group drained");
        }
        self.append(
            Event::AdmissionDelivered {
                admission_sha256: persisted.admission_sha256.clone(),
            },
            now,
        )
    }
    /// A refused rendezvous ends the attempt: launched rank work is stopped before it
    /// could start any optimizer attempt, and nothing is committed.
    pub fn record_refusal(&mut self, refusal: &GroupRefusal, now: u64) -> Result<()> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && s.drain.is_none(),
            "refusal after admission"
        );
        self.append(
            Event::GroupRefused {
                refusal: refusal.clone(),
            },
            now,
        )?;
        for rank in 0..self.contract.world_size() as u32 {
            let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                (Some(h), _) => Some(h.clone()),
                (None, Some(id)) => self.supervisor.find(id)?,
                (None, None) => None,
            };
            let stopped = match handle {
                Some(h) => {
                    let last = self.supervisor.terminate(&h)?;
                    ensure!(last != Liveness::Running, "rank {rank} survived refusal");
                    format!("; unadmitted rank work stopped ({last:?})")
                }
                None => "; no rank work was started".into(),
            };
            self.append(
                Event::Disposition {
                    global_rank: rank,
                    disposition: Disposition::Refused,
                    detail: format!("{}{stopped}", serde_json::to_string(&refusal.cause)?),
                },
                now,
            )?;
        }
        self.append(
            Event::Terminal {
                outcome: Outcome::Refused,
                committed_updates_total: self.contract.counters.committed_updates_at_start,
            },
            now,
        )
    }
    /// Dispatch is permitted only after a fresh admitted preflight, before admission, while
    /// no cancellation, drain or restart is pending, and inside the group window: the
    /// original deadline and every participant's grant, not only this rank's. A group that
    /// can no longer run as a whole starts nothing more.
    fn require_dispatchable(&mut self, rank: u32, now: u64) -> Result<Replay> {
        ensure!(
            !self.needs_reconcile,
            "coordinator restarted: reconcile before dispatch"
        );
        ensure!(
            !self.cancel_pending(now)?,
            "cancellation requested; nothing is dispatched"
        );
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.drain.is_none() && !s.refused && s.admission.is_none(),
            "run is admitted, refused, draining or terminal; no further dispatch"
        );
        let receipt = s
            .preflight
            .as_ref()
            .context("dispatch requires an admitted preflight")?;
        ensure!(
            now.saturating_sub(receipt.checked_unix_ms) <= self.contract.telemetry.max_age_ms,
            "preflight receipt is stale; preflight a fresh census before dispatch"
        );
        self.contract
            .participant(rank)
            .context("rank outside the plan")?;
        if let Some(cause) = self.window_closed(now) {
            bail!("rank {rank} is not dispatched: the group window closed ({cause:?})");
        }
        Ok(s)
    }
    /// Persist the dispatch intent (idempotent) before any launch side effect.
    pub fn record_dispatch_intent(&mut self, rank: u32, now: u64) -> Result<String> {
        let s = self.require_dispatchable(rank, now)?;
        if let Some(id) = s.intents.get(&rank) {
            return Ok(id.clone());
        }
        let id = rendezvous::dispatch_id(&self.plan_sha256, self.contract.attempt, rank)?;
        self.append(
            Event::DispatchIntent {
                global_rank: rank,
                dispatch_id: id.clone(),
            },
            now,
        )?;
        Ok(id)
    }
    /// Launch a rank whose intent is persisted. Never relaunches: journaled or already
    /// supervised work under the same dispatch id is recorded, not restarted.
    pub fn launch(&mut self, rank: u32, now: u64) -> Result<SupervisorHandle> {
        self.launch_with_clock(rank, || now)
    }
    pub fn launch_with_clock(
        &mut self,
        rank: u32,
        clock: impl Fn() -> u64,
    ) -> Result<SupervisorHandle> {
        let s = self.require_dispatchable(rank, clock())?;
        let id = s
            .intents
            .get(&rank)
            .cloned()
            .context("dispatch intent must be journaled before launch")?;
        if let Some(h) = s.launched.get(&rank) {
            return Ok(h.clone());
        }
        let intent = self
            .records
            .iter()
            .find_map(|(r, _)| match &r.event {
                Event::DispatchIntent {
                    global_rank,
                    dispatch_id,
                } if *global_rank == rank && *dispatch_id == id => Some(r.recorded_unix_ms),
                _ => None,
            })
            .context("dispatch intent timestamp absent")?;
        let found = self.supervisor.find(&id)?;
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            bail!("execution window closed while finding rank {rank}");
        }
        self.require_dispatchable(rank, now)?;
        let handle = match found {
            Some(h) => h,
            None => self.supervisor.launch(&RankLaunch {
                contract: &self.contract,
                plan_sha256: &self.plan_sha256,
                global_rank: rank,
                dispatch_id: &id,
                dispatch_intent_unix_ms: intent,
            })?,
        };
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            bail!("execution window closed while launching rank {rank}");
        }
        self.append(
            Event::Launched {
                global_rank: rank,
                dispatch_id: id,
                handle: handle.clone(),
            },
            now,
        )?;
        Ok(handle)
    }
    pub fn dispatch_all(&mut self, now: u64) -> Result<()> {
        self.dispatch_all_with_clock(|| now)
    }
    pub fn dispatch_all_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<()> {
        for rank in 0..self.contract.world_size() as u32 {
            self.record_dispatch_intent(rank, clock())?;
            self.launch_with_clock(rank, &clock)?;
        }
        Ok(())
    }
    fn retained(&mut self, handle: &SupervisorHandle) -> Result<Retained> {
        let s = self.state()?;
        // A result can only bind an admission; before one exists nothing is a result.
        let Some((_, adm)) = s.admission.as_ref() else {
            return Ok(Retained::Absent);
        };
        Ok(match self.supervisor.retained_result(handle)? {
            None => Retained::Absent,
            Some((envelope, bytes)) => match rendezvous::verify_result(
                &self.contract,
                &self.plan_sha256,
                adm,
                &envelope,
                &bytes,
            ) {
                Ok(v) => Retained::Valid(Box::new(v)),
                Err(e) => Retained::Invalid(format!("{e:#}")),
            },
        })
    }
    fn acknowledge(
        &mut self,
        v: &VerifiedResult,
        rank: u32,
        clock: &impl Fn() -> u64,
    ) -> Result<bool> {
        let now = clock();
        let s = self.state()?;
        ensure!(
            v.body().global_rank == rank && s.intents.contains_key(&rank),
            "result does not belong to dispatched rank {rank}"
        );
        if let Some(previous) = s.acked.get(&rank) {
            ensure!(
                *previous == v.body().result_sha256,
                "conflicting second result for rank {rank}"
            );
            return Ok(true);
        }
        // Adopting a retained result is authority-bearing: the same gate as poll and
        // recovery, whichever path reaches it.
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            return Ok(false);
        }
        let p = self.contract.participant(rank).context("rank")?.clone();
        publish_immutable(
            &self.dir.join(RESULTS),
            &p.result_destination.path,
            v.bytes(),
        )?;
        publish_immutable(
            &self.dir.join(ENVELOPES),
            Path::new(&format!("rank-{rank}.json")),
            v.envelope(),
        )?;
        let now = clock();
        let cause = if self.cancel_pending(now)? {
            Some(DrainCause::Cancelled)
        } else {
            self.window_closed(now)
        };
        if let Some(cause) = cause {
            self.drain(cause, now)?;
            return Ok(false);
        }
        self.append(
            Event::ResultAcknowledged {
                global_rank: rank,
                delivery_id: v.body().delivery_id,
                result_sha256: v.body().result_sha256.clone(),
                envelope_sha256: hash(v.envelope()),
                updates: v.body().updates,
            },
            now,
        )?;
        Ok(true)
    }
    /// Journal the single group commit (inside the group window only), or finish a commit
    /// that is already journaled (at any time, exactly once, never drained).
    fn commit(&mut self, clock: &impl Fn() -> u64) -> Result<()> {
        let s = self.state()?;
        let now = clock();
        if s.committed.is_none() {
            ensure!(
                s.drain.is_none() && s.acked.len() == self.contract.world_size(),
                "group commit requires every authenticated result"
            );
            if let Some(cause) = self.window_closed(now) {
                self.drain(cause, now)?;
                return Ok(());
            }
            self.append(
                Event::GroupCommitted {
                    committed_updates_total: self.contract.counters.committed_updates_at_start
                        + self.contract.counters.planned_updates,
                    results: s.acked.clone().into_iter().collect(),
                },
                now,
            )?;
        }
        let s = self.state()?;
        let total = s.committed.context("commit")?;
        for rank in 0..self.contract.world_size() as u32 {
            if !s.dispositions.contains_key(&rank) {
                self.append(
                    Event::Disposition {
                        global_rank: rank,
                        disposition: Disposition::Committed,
                        detail: "authenticated result committed with the whole group".into(),
                    },
                    now,
                )?;
            }
        }
        if self.state()?.terminal.is_none() {
            self.append(
                Event::Terminal {
                    outcome: Outcome::Committed,
                    committed_updates_total: total,
                },
                now,
            )?;
        }
        Ok(())
    }
    /// Drain the whole synchronous group. Idempotent and resumable; the first recorded
    /// cause governs. Every rank ends with exactly one disposition and no supervised work
    /// survives.
    pub fn drain(&mut self, cause: DrainCause, now: u64) -> Result<StatusReport> {
        let s = self.state()?;
        if s.terminal.is_some() {
            return self.status();
        }
        ensure!(s.committed.is_none(), "a committed group is never drained");
        ensure!(
            !s.intents.is_empty(),
            "nothing was dispatched; there is no group to drain"
        );
        if s.drain.is_none() {
            self.append(Event::DrainStarted { cause }, now)?;
        }
        let cause = self.state()?.drain.context("drain cause")?;
        let failing = match &cause {
            DrainCause::RankFailed { global_rank, .. } => Some(*global_rank),
            _ => None,
        };
        for rank in 0..self.contract.world_size() as u32 {
            let s = self.state()?;
            if s.dispositions.contains_key(&rank) {
                continue;
            }
            let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                (Some(h), _) => Some(h.clone()),
                (None, Some(id)) => self.supervisor.find(id)?,
                (None, None) => None,
            };
            let (disposition, detail) = match handle {
                None => (
                    Disposition::NeverStarted,
                    if s.intents.contains_key(&rank) {
                        "dispatch intent persisted; supervisor holds no started work".to_string()
                    } else {
                        "never dispatched".to_string()
                    },
                ),
                Some(h) => {
                    let last = self.supervisor.terminate(&h)?;
                    ensure!(last != Liveness::Running, "rank {rank} survived the drain");
                    let started_unrecorded = !s.launched.contains_key(&rank);
                    if failing == Some(rank) {
                        (Disposition::Failed, format!("{cause:?}; final {last:?}"))
                    } else if s.acked.contains_key(&rank) {
                        (
                            Disposition::CompletedUncommitted,
                            "authenticated result acknowledged; group drained before commit".into(),
                        )
                    } else if let Retained::Valid(v) = self.retained(&h)? {
                        (
                            Disposition::CompletedUnacknowledged,
                            format!(
                                "authenticated result {} retained, never acknowledged; not committed",
                                v.body().result_sha256
                            ),
                        )
                    } else {
                        (
                            Disposition::Drained,
                            format!(
                                "stopped by whole-group drain; final {last:?}{}",
                                if started_unrecorded {
                                    "; launch was never journaled"
                                } else {
                                    ""
                                }
                            ),
                        )
                    }
                }
            };
            self.append(
                Event::Disposition {
                    global_rank: rank,
                    disposition,
                    detail,
                },
                now,
            )?;
        }
        self.append(
            Event::Terminal {
                outcome: if cause == DrainCause::Cancelled {
                    Outcome::Cancelled
                } else {
                    Outcome::Drained
                },
                committed_updates_total: self.contract.counters.committed_updates_at_start,
            },
            now,
        )?;
        self.status()
    }
    /// Report a control channel failure observed by the transport (heartbeat timeout).
    pub fn control_lost(&mut self, rank: u32, now: u64) -> Result<StatusReport> {
        self.drain(DrainCause::ControlLost { global_rank: rank }, now)
    }
    fn cancel_pending(&mut self, now: u64) -> Result<bool> {
        let s = self.state()?;
        if s.cancel {
            return Ok(true);
        }
        let path = self.dir.join(CANCEL);
        if fs::symlink_metadata(&path).is_err() {
            return Ok(false);
        }
        let bytes = read_bounded(&path, MAX_RECORD_BYTES)?;
        let request: CancelRequest = serde_json::from_slice(&bytes)?;
        ensure!(
            request.plan_sha256 == self.plan_sha256,
            "cancellation bound to another plan"
        );
        self.append(
            Event::CancelRequested {
                request_sha256: hash(&bytes),
            },
            now,
        )?;
        Ok(true)
    }
    /// One supervision step: cancellation, deadline, grant expiry, rank exits, result
    /// acknowledgement and — once every rank's result is authenticated — the single commit.
    pub fn poll(&mut self, now: u64) -> Result<StatusReport> {
        self.poll_with_clock(|| now)
    }
    /// Live callers refresh time after blocking supervisor operations. The fixed-time
    /// wrapper remains available for deterministic journal replay fixtures.
    pub fn poll_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<StatusReport> {
        let now = clock();
        ensure!(
            !self.needs_reconcile,
            "coordinator restarted: reconcile before polling"
        );
        let s = self.state()?;
        if s.terminal.is_some() {
            return self.status();
        }
        if s.intents.is_empty() {
            if self.cancel_pending(now)? {
                finish_cancel_before_dispatch(
                    &self.dir,
                    &self.contract,
                    &self.plan_sha256,
                    &mut self.records,
                    now,
                )?;
            }
            return self.status();
        }
        if let Some(cause) = s.drain.clone() {
            return self.drain(cause, now);
        }
        if self.cancel_pending(now)? {
            return self.drain(DrainCause::Cancelled, now);
        }
        if let Some(cause) = self.window_closed(now) {
            return self.drain(cause, now);
        }
        for rank in 0..self.contract.world_size() as u32 {
            let s = self.state()?;
            if s.acked.contains_key(&rank) {
                continue;
            }
            let Some(handle) = s.launched.get(&rank).cloned() else {
                continue;
            };
            let observed = self.supervisor.probe(&handle)?;
            let now = clock();
            if let Some(cause) = self.window_closed(now) {
                return self.drain(cause, now);
            }
            match observed {
                Liveness::Running => {}
                Liveness::Exited { code: Some(0), .. } if !s.delivered => {
                    return self.drain(
                        DrainCause::RankFailed {
                            global_rank: rank,
                            detail: "exited before the full-group admission was delivered".into(),
                        },
                        now,
                    )
                }
                Liveness::Exited { code: Some(0), .. } => {
                    let retained = self.retained(&handle)?;
                    let now = clock();
                    if self.cancel_pending(now)? {
                        return self.drain(DrainCause::Cancelled, now);
                    }
                    if let Some(cause) = self.window_closed(now) {
                        return self.drain(cause, now);
                    }
                    match retained {
                        Retained::Valid(v) => {
                            if !self.acknowledge(&v, rank, &clock)? {
                                return self.status();
                            }
                        }
                        Retained::Absent => {
                            return self.drain(
                                DrainCause::RankFailed {
                                    global_rank: rank,
                                    detail: "exited without a retained result".into(),
                                },
                                now,
                            )
                        }
                        Retained::Invalid(detail) => {
                            return self.drain(
                                DrainCause::RankFailed {
                                    global_rank: rank,
                                    detail: format!("result refused: {detail}"),
                                },
                                now,
                            )
                        }
                    }
                }
                Liveness::Exited {
                    code: Some(EXIT_LOST_CONTROL),
                    ..
                } => return self.drain(DrainCause::ControlLost { global_rank: rank }, now),
                other => {
                    return self.drain(
                        DrainCause::RankFailed {
                            global_rank: rank,
                            detail: format!("{other:?}"),
                        },
                        now,
                    )
                }
            }
        }
        let now = clock();
        if self.cancel_pending(now)? {
            return self.drain(DrainCause::Cancelled, now);
        }
        if let Some(cause) = self.window_closed(now) {
            return self.drain(cause, now);
        }
        let s = self.state()?;
        if s.delivered && s.acked.len() == self.contract.world_size() {
            self.commit(&clock)?;
        }
        self.status()
    }
    /// Settle every owned rank after a coordinator restart.
    ///
    /// * A commit already journaled before the crash is finished (dispositions and the
    ///   terminal record), exactly once and at any time; it is never drained.
    /// * A drain already journaled is resumed with its original cause.
    /// * Otherwise a *new* recovery commit is possible only through the same gates as
    ///   [`Coordinator::poll`]: no pending cancellation, and the group window (original
    ///   deadline and every participant grant) still open. If either fails, the group is
    ///   drained with that cause before any retained result is adopted.
    /// * Inside the window, complete, authenticated, unacknowledged results for the
    ///   *whole* group are acknowledged and committed once; anything less drains, and each
    ///   rank is marked never-started, completed-unacknowledged or drained.
    ///
    /// A restart therefore never widens the window `poll` enforces (review of #111).
    pub fn reconcile(&mut self, now: u64) -> Result<StatusReport> {
        self.reconcile_with_clock(|| now)
    }
    pub fn reconcile_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<StatusReport> {
        let now = clock();
        let s = self.state()?;
        if s.terminal.is_none() && s.intents.is_empty() {
            if self.cancel_pending(now)? {
                finish_cancel_before_dispatch(
                    &self.dir,
                    &self.contract,
                    &self.plan_sha256,
                    &mut self.records,
                    now,
                )?;
            }
        } else if s.terminal.is_none() {
            if s.committed.is_some() {
                self.commit(&clock)?;
            } else if let Some(cause) = s.drain.clone() {
                self.drain(cause, now)?;
            } else if self.cancel_pending(now)? {
                self.drain(DrainCause::Cancelled, now)?;
            } else if let Some(cause) = self.window_closed(now) {
                // Poll's gate, applied before any retained result is adopted.
                self.drain(cause, now)?;
            } else {
                // Unacknowledged completion is committed only when the *whole* group
                // completed with authenticated results; anything less drains.
                let mut complete = s.delivered;
                let mut found = Vec::new();
                for rank in 0..self.contract.world_size() as u32 {
                    if s.acked.contains_key(&rank) {
                        continue;
                    }
                    let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                        (Some(h), _) => Some(h.clone()),
                        (None, Some(id)) => self.supervisor.find(id)?,
                        (None, None) => None,
                    };
                    let Some(h) = handle else {
                        complete = false;
                        continue;
                    };
                    match self.supervisor.probe(&h)? {
                        Liveness::Exited { code: Some(0), .. } => match self.retained(&h)? {
                            Retained::Valid(v) => found.push((rank, v)),
                            _ => complete = false,
                        },
                        _ => complete = false,
                    }
                }
                let now = clock();
                if self.cancel_pending(now)? {
                    self.drain(DrainCause::Cancelled, now)?;
                } else if let Some(cause) = self.window_closed(now) {
                    self.drain(cause, now)?;
                } else if complete {
                    for (rank, v) in &found {
                        if !self.acknowledge(v, *rank, &clock)? {
                            self.needs_reconcile = false;
                            return self.status();
                        }
                    }
                    let now = clock();
                    if self.cancel_pending(now)? {
                        self.drain(DrainCause::Cancelled, now)?;
                    } else if let Some(cause) = self.window_closed(now) {
                        self.drain(cause, now)?;
                    } else {
                        self.commit(&clock)?;
                    }
                } else {
                    self.drain(DrainCause::CoordinatorRestart, now)?;
                }
            }
        }
        self.needs_reconcile = false;
        self.status()
    }
}
