use super::*;

/// Existing physical run directory; never created by read-only operations.
pub(super) fn existing_dir(dir: &Path) -> Result<()> {
    ensure!(dir.is_absolute(), "run directory must be absolute");
    let meta = fs::symlink_metadata(dir)
        .with_context(|| format!("run directory {} is absent", dir.display()))?;
    ensure!(
        meta.is_dir() && fs::canonicalize(dir)? == dir,
        "run directory must be a physical directory without symlinks"
    );
    Ok(())
}
pub(super) fn create_dir(dir: &Path) -> Result<()> {
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
pub(super) fn lock_file(dir: &Path, create: bool) -> Result<fs::File> {
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
pub(super) fn try_lock(f: &fs::File) -> bool {
    f.try_lock().is_ok()
}
/// Non-interfering liveness probe: reads `/proc/locks` for an exclusive flock on the
/// controller lock's inode. It never takes a lock itself, so it cannot block a writer.
pub(super) fn coordinator_lock_holder(dir: &Path) -> Result<Option<u32>> {
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
pub(super) fn load_plan(dir: &Path) -> Result<(BoundedRunContract, String)> {
    let bytes = read_bounded(&dir.join(PLAN), MAX_CONTRACT_BYTES as u64)?;
    let contract = BoundedRunContract::parse(&bytes).map_err(|e| anyhow!("{e}"))?;
    ensure!(
        contract.canonical_bytes()? == bytes,
        "frozen plan bytes are not the canonical contract"
    );
    Ok((contract, hash(&bytes)))
}
pub(super) fn load_journal(dir: &Path) -> Result<Vec<(JournalRecord, String)>> {
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
pub(super) struct Replay {
    pub(super) preflight: Option<PreflightReceipt>,
    pub(super) admission: Option<(GroupAdmission, String)>,
    pub(super) delivered: bool,
    pub(super) refused: bool,
    pub(super) intents: BTreeMap<u32, String>,
    pub(super) launched: BTreeMap<u32, SupervisorHandle>,
    pub(super) acked: BTreeMap<u32, String>,
    pub(super) committed: Option<u64>,
    pub(super) cancel: bool,
    pub(super) drain: Option<DrainCause>,
    pub(super) dispositions: BTreeMap<u32, (BoundedControllerDisposition, String)>,
    pub(super) terminal: Option<(BoundedControllerOutcome, u64)>,
}
pub(super) fn replay(
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
pub(super) fn rank_status(
    contract: &BoundedRunContract,
    s: &Replay,
    live: bool,
) -> Vec<RankStatus> {
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
pub(super) fn build_status(
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
