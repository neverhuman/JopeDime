use super::*;

/// The next sampler position after a journal's durable commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Cursor {
    pub optimizer_step: u64,
    pub serial: u64,
    pub dropout_seed: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reconciliation {
    pub header: Header,
    /// The rank's partition-v2 shard; `None` under v1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard: Option<Shard>,
    /// Every journaled microbatch whose update outcome was recorded,
    /// including rejected and re-executed work.
    pub computed: Units,
    /// Microbatches of a step interrupted before its update outcome.
    pub unfinished: Units,
    pub unfinished_step: Option<u64>,
    pub attempted_updates: u64,
    pub applied_updates: u64,
    pub rejected_updates: u64,
    pub ended_by_rejection: bool,
    pub committed_updates: u64,
    pub committed: Units,
    pub committed_through: u64,
    pub applied_through: u64,
    /// Applied updates no durable checkpoint covers: lost at attempt end.
    pub uncommitted_applied_updates: u64,
    pub commit_events: u64,
    /// The durable checkpoint is ahead of the journal's last commit event.
    pub commit_event_missing: bool,
    pub torn_tail: bool,
    pub duplicate_events_ignored: u64,
    pub unique_committed_records: u64,
    #[serde(skip)]
    pub committed_records: BTreeSet<u64>,
    pub next: Cursor,
}

/// Complete lines of a journal and whether a torn final line was dropped.
fn complete_lines(bytes: &[u8]) -> (Vec<&[u8]>, bool) {
    match bytes.iter().rposition(|&b| b == b'\n') {
        Some(end) => (
            bytes[..end].split(|&b| b == b'\n').collect(),
            end + 1 < bytes.len(),
        ),
        None => (Vec::new(), !bytes.is_empty()),
    }
}

/// Validate a journal and derive its consumption. `durable_step` is the
/// optimizer step of the run's durable `latest.json`, when known: it may be
/// ahead of the last commit event (a crash between checkpoint publication and
/// the journal commit) but never behind it and never beyond applied work.
pub fn reconcile(bytes: &[u8], durable_step: Option<u64>) -> Result<Reconciliation> {
    let (lines, torn_tail) = complete_lines(bytes);
    let mut tracker = Tracker {
        track_records: true,
        ..Tracker::default()
    };
    let mut seen = BTreeMap::<String, &[u8]>::new();
    let mut duplicates = 0;
    for (n, line) in lines.into_iter().enumerate() {
        let event: TrainingConsumptionEvent = serde_json::from_slice(line)
            .with_context(|| format!("malformed consumption journal line {}", n + 1))?;
        let key = event.key();
        if let Some(previous) = seen.get(&key) {
            ensure!(
                *previous == line,
                "conflicting duplicate consumption event {key}"
            );
            duplicates += 1;
            continue;
        }
        seen.insert(key, line);
        tracker.accept(&event)?;
    }
    let header = tracker
        .header
        .clone()
        .context("consumption journal has no attempt header")?;
    let commit_events = tracker.commits;
    let by_events = tracker.committed_through;
    let mut commit_event_missing = false;
    if let Some(durable) = durable_step {
        ensure!(
            durable >= by_events,
            "consumption journal commits step {by_events} beyond the durable checkpoint {durable}"
        );
        ensure!(
            durable <= tracker.applied_through,
            "durable checkpoint {durable} is beyond the journal's applied updates"
        );
        if durable > by_events {
            tracker.commit_through(durable)?;
            commit_event_missing = true;
        }
    }
    let committed_through = tracker.committed_through;
    let unfinished = tracker.step.units;
    Ok(Reconciliation {
        computed: tracker.computed,
        unfinished,
        unfinished_step: (unfinished.microbatches > 0).then_some(tracker.next_step),
        attempted_updates: tracker.applied + tracker.rejected,
        applied_updates: tracker.applied,
        rejected_updates: tracker.rejected,
        ended_by_rejection: tracker.ended.is_some(),
        committed_updates: committed_through - header.initial_optimizer_step,
        committed: tracker.committed,
        committed_through,
        applied_through: tracker.applied_through,
        uncommitted_applied_updates: tracker.applied_through - committed_through,
        commit_events,
        commit_event_missing,
        torn_tail,
        duplicate_events_ignored: duplicates,
        unique_committed_records: tracker.committed_records.len() as u64,
        next: Cursor {
            optimizer_step: committed_through,
            serial: header.serial_in(tracker.shard.as_ref(), committed_through, 0)?,
            dropout_seed: header.dropout_seed(committed_through, 0)?,
        },
        committed_records: tracker.committed_records,
        shard: tracker.shard,
        header,
    })
}

/// A prefix recorded in a checkpoint receipt must still be the journal's prefix.
pub fn verify_binding(bytes: &[u8], binding: &Binding) -> Result<()> {
    ensure!(
        binding.schema == SCHEMA
            && bytes.len() as u64 >= binding.bytes
            && hex::encode(Sha256::digest(&bytes[..binding.bytes as usize])) == binding.sha256,
        "consumption journal no longer contains its committed prefix"
    );
    Ok(())
}

/// Result of reconciling a checkpoint that became durable before its journal
/// commit acknowledgement. `acknowledged` is false only when later applied
/// work exists beyond the durable checkpoint; that work remains visible and
/// uncommitted for the next attempt.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PublicationAckRecovery {
    pub binding: Binding,
    pub reconciliation: Reconciliation,
    pub appended: bool,
    pub acknowledged: bool,
}

/// Idempotently append the logical commit acknowledgement after a published
/// checkpoint. Physical retries remain represented by their attempt events;
/// this function never adds exposure or update events. It refuses torn tails,
/// conflicting acknowledgements, and checkpoints ahead of applied work.
pub fn recover_published_ack(
    path: &Path,
    durable_step: u64,
    state_sha256: &str,
) -> Result<PublicationAckRecovery> {
    ensure!(
        is_sha256(state_sha256),
        "published state digest is malformed"
    );
    let bytes = fs::read(path)?;
    let mut reconciliation = reconcile(&bytes, Some(durable_step))?;
    let (lines, torn) = complete_lines(&bytes);
    ensure!(
        !torn,
        "cannot append a publication acknowledgement after a torn journal tail"
    );
    let mut matching_ack = false;
    for line in &lines {
        let event: TrainingConsumptionEvent = serde_json::from_slice(line)?;
        if let TrainingConsumptionEvent::Commit {
            step,
            state_sha256: ack_sha,
        } = event
        {
            if step == durable_step {
                ensure!(
                    ack_sha == state_sha256,
                    "conflicting publication acknowledgement digest"
                );
                matching_ack = true;
            }
        }
    }
    let mut appended = false;
    let mut acknowledged = matching_ack;
    if reconciliation.commit_event_missing {
        ensure!(
            reconciliation.applied_through >= durable_step,
            "published checkpoint is ahead of journal-applied work"
        );
        if reconciliation.applied_through >= durable_step {
            let event = TrainingConsumptionEvent::Commit {
                step: durable_step,
                state_sha256: state_sha256.into(),
            };
            let mut next = Tracker {
                track_records: true,
                ..Tracker::default()
            };
            let mut seen = BTreeMap::<String, &[u8]>::new();
            for line in &lines {
                let event: TrainingConsumptionEvent = serde_json::from_slice(line)?;
                let key = event.key();
                if let Some(previous) = seen.get(&key) {
                    ensure!(
                        *previous == *line,
                        "conflicting duplicate consumption event {key}"
                    );
                    continue;
                }
                seen.insert(key, line);
                next.accept(&event)?;
            }
            next.accept(&event)?;
            let mut encoded = serde_json::to_vec(&event)?;
            encoded.push(b'\n');
            let mut file = fs::OpenOptions::new().append(true).open(path)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            appended = true;
            acknowledged = true;
            let updated = fs::read(path)?;
            reconciliation = reconcile(&updated, Some(durable_step))?;
        }
    }
    let updated = fs::read(path)?;
    let binding = Binding {
        schema: SCHEMA.into(),
        bytes: updated.len() as u64,
        sha256: hex::encode(Sha256::digest(&updated)),
    };
    Ok(PublicationAckRecovery {
        binding,
        reconciliation,
        appended,
        acknowledged,
    })
}

/// Re-derive every journaled microbatch from its cursor with the frozen
/// sampler; the recorded records must be exactly what the sampler yields.
pub fn verify_records(
    bytes: &[u8],
    mut sampler: impl FnMut(usize, usize) -> Vec<usize>,
) -> Result<u64> {
    let (lines, _) = complete_lines(bytes);
    let mut verified = 0;
    for line in lines {
        if let TrainingConsumptionEvent::Microbatch {
            step,
            microbatch,
            serial,
            records,
            ..
        } = serde_json::from_slice::<TrainingConsumptionEvent>(line)?
        {
            let expected = sampler(usize::try_from(serial)?, records.len());
            ensure!(
                expected.len() == records.len()
                    && expected.iter().zip(&records).all(|(&e, &r)| e as u64 == r),
                "recorded records of step {step} microbatch {microbatch} differ from the sampler"
            );
            verified += 1;
        }
    }
    Ok(verified)
}

/// Run-level totals across consecutive attempts of one rank.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunTotals {
    pub attempts: usize,
    pub initial_optimizer_step: u64,
    pub committed_through: u64,
    pub committed_updates: u64,
    /// In-memory optimizer steps, including lost and re-executed ones.
    pub applied_updates: u64,
    pub lost_applied_updates: u64,
    pub rejected_updates: u64,
    pub computed: Units,
    pub committed: Units,
    pub unique_committed_records: u64,
}

/// Each attempt must resume at its predecessor's durable commit with the
/// same run constants; unresolved gaps are refused, never bridged.
pub fn reconcile_run(attempts: &[Reconciliation]) -> Result<RunTotals> {
    let first = attempts.first().context("run has no attempts")?;
    let same = |a: &Header, b: &Header| {
        let strip = |h: &Header| Header {
            initial_optimizer_step: 0,
            resume_state_sha256: None,
            ..h.clone()
        };
        strip(a) == strip(b)
    };
    let mut totals = RunTotals {
        attempts: attempts.len(),
        initial_optimizer_step: first.header.initial_optimizer_step,
        committed_through: first.header.initial_optimizer_step,
        committed_updates: 0,
        applied_updates: 0,
        lost_applied_updates: 0,
        rejected_updates: 0,
        computed: Units::default(),
        committed: Units::default(),
        unique_committed_records: 0,
    };
    let mut records = BTreeSet::new();
    for (i, attempt) in attempts.iter().enumerate() {
        if i > 0 {
            let previous = &attempts[i - 1];
            ensure!(
                same(&previous.header, &attempt.header) && previous.shard == attempt.shard,
                "attempt {i} changed the run constants"
            );
            ensure!(
                attempt.header.initial_optimizer_step == previous.committed_through
                    && attempt.header.resume_state_sha256.is_some(),
                "attempt {i} does not resume at its predecessor's durable commit"
            );
            totals.lost_applied_updates += previous.uncommitted_applied_updates;
        }
        totals.committed_through = attempt.committed_through;
        totals.committed_updates += attempt.committed_updates;
        totals.applied_updates += attempt.applied_updates;
        totals.rejected_updates += attempt.rejected_updates;
        totals.computed.add(&attempt.computed)?;
        totals.computed.add(&attempt.unfinished)?;
        totals.committed.add(&attempt.committed)?;
        records.extend(attempt.committed_records.iter().copied());
    }
    totals.unique_committed_records = records.len() as u64;
    Ok(totals)
}
