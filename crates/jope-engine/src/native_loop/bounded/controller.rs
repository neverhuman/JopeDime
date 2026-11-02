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
    f.try_lock().is_ok()
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
    dispositions: BTreeMap<u32, (BoundedControllerDisposition, String)>,
    terminal: Option<(BoundedControllerOutcome, u64)>,
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
            BoundedControllerEvent::DispatchIntent { .. }
                | BoundedControllerEvent::Launched { .. }
                | BoundedControllerEvent::GroupAdmitted { .. }
                | BoundedControllerEvent::AdmissionDelivered { .. }
                | BoundedControllerEvent::ResultAcknowledged { .. }
                | BoundedControllerEvent::GroupCommitted { .. }
        ) {
            if let Some(cause) = journal_window_closed(contract, &records[..i], r.recorded_unix_ms)
            {
                bail!(
                    "journal record {at} was journaled after the group window closed ({cause:?})"
                );
            }
        }
        match &r.event {
            BoundedControllerEvent::Planned {
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
            BoundedControllerEvent::PreflightAdmitted { receipt } => {
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
            BoundedControllerEvent::PreflightRefused { .. } => ensure!(
                s.admission.is_none() && !s.refused,
                "journal record {at}: preflight after admission"
            ),
            BoundedControllerEvent::DispatchIntent {
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
            BoundedControllerEvent::Launched {
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
            BoundedControllerEvent::GroupAdmitted {
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
            BoundedControllerEvent::AdmissionDelivered { admission_sha256 } => {
                ensure!(
                    s.admission.as_ref().map(|(_, h)| h) == Some(admission_sha256)
                        && !s.delivered
                        && s.drain.is_none(),
                    "journal record {at}: delivery of an unpersisted admission"
                );
                s.delivered = true;
            }
            BoundedControllerEvent::GroupRefused { refusal } => {
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
            BoundedControllerEvent::ResultAcknowledged {
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
            BoundedControllerEvent::GroupCommitted {
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
            BoundedControllerEvent::CancelRequested { .. } => {
                ensure!(!s.cancel, "journal record {at}: repeated cancellation");
                s.cancel = true;
            }
            BoundedControllerEvent::CoordinatorRestarted { .. } => {}
            BoundedControllerEvent::DrainStarted { cause } => {
                ensure!(
                    s.drain.is_none()
                        && s.committed.is_none()
                        && !s.refused
                        && !s.intents.is_empty(),
                    "journal record {at}: drain repeated, after commit/refusal, or with nothing dispatched"
                );
                s.drain = Some(cause.clone());
            }
            BoundedControllerEvent::BoundedControllerDisposition {
                global_rank,
                disposition,
                detail,
            } => {
                rank_ok(*global_rank)?;
                let allowed = match disposition {
                    BoundedControllerDisposition::Committed => s.committed.is_some(),
                    BoundedControllerDisposition::Refused => s.refused,
                    BoundedControllerDisposition::CompletedUncommitted => {
                        s.drain.is_some() && s.acked.contains_key(global_rank)
                    }
                    BoundedControllerDisposition::NeverStarted => {
                        (s.drain.is_some() || (s.cancel && s.intents.is_empty()))
                            && !s.launched.contains_key(global_rank)
                    }
                    BoundedControllerDisposition::Failed
                    | BoundedControllerDisposition::Drained
                    | BoundedControllerDisposition::CompletedUnacknowledged => s.drain.is_some(),
                };
                ensure!(
                    allowed
                        && s.dispositions
                            .insert(*global_rank, (*disposition, detail.clone()))
                            .is_none(),
                    "journal record {at}: disposition not permitted by the recorded state"
                );
            }
            BoundedControllerEvent::Terminal {
                outcome,
                committed_updates_total,
            } => {
                let consistent = match outcome {
                    BoundedControllerOutcome::Committed => {
                        s.committed == Some(*committed_updates_total)
                    }
                    BoundedControllerOutcome::Refused => {
                        s.refused && *committed_updates_total == start
                    }
                    BoundedControllerOutcome::Drained | BoundedControllerOutcome::Cancelled => {
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
                    BoundedControllerDisposition::Committed => RankState::Committed,
                    BoundedControllerDisposition::Failed => RankState::Failed,
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
                matches!(&r.event, BoundedControllerEvent::ResultAcknowledged { global_rank, envelope_sha256, .. }
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
    event: BoundedControllerEvent,
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
            BoundedControllerEvent::Planned {
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
                BoundedControllerEvent::CancelRequested { request_sha256 },
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
                BoundedControllerEvent::BoundedControllerDisposition {
                    global_rank: p.global_rank,
                    disposition: BoundedControllerDisposition::NeverStarted,
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
        BoundedControllerEvent::Terminal {
            outcome: BoundedControllerOutcome::Cancelled,
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
mod coordinator;
mod types;
pub use coordinator::{Coordinator, PersistedAdmission};
pub use types::*;
pub(super) use types::{group_window_end, journal_window_closed};
