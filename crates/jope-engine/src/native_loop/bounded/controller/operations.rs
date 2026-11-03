use super::*;

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

pub(super) fn append(
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
