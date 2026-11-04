use super::*;

fn transaction_required(db: &Connection) -> Result<()> {
    ensure!(
        !db.is_autocommit(),
        "funding mutation requires the original authority transaction"
    );
    Ok(())
}

pub fn register_authorization(
    db: &Connection,
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    clock: FundingClock,
) -> Result<String> {
    transaction_required(db)?;
    let id = verify_authorization(trust, authorization, clock)?;
    let existing: Option<String> = db
        .query_row(
            "SELECT id FROM funding_authorizations WHERE domain_id=?",
            [&authorization.body.domain_id],
            |row| row.get(0),
        )
        .optional()?;
    ensure!(
        existing.is_none(),
        "funding domain already registered; reset/reissue forbidden"
    );
    db.execute(
        "INSERT INTO funding_authorizations VALUES(?,?,?,?)",
        params![
            id,
            authorization.body.domain_id,
            authorization.body.authority_evidence_sha256,
            serde_json::to_string(authorization)?
        ],
    )?;
    Ok(id)
}

fn stored_authorization(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let bytes: String = db.query_row(
        "SELECT json FROM funding_authorizations WHERE id=?",
        [verified.authorization_id()?],
        |row| row.get(0),
    )?;
    ensure!(
        serde_json::from_str::<SignedAuthorization>(&bytes)? == *verified.authorization(),
        "original registered funding authorization differs"
    );
    Ok(())
}

pub fn reserve(db: &Connection, verified: &VerifiedReservation) -> Result<String> {
    transaction_required(db)?;
    stored_authorization(db, verified)?;
    let auth = &verified.authorization().body;
    let mut core = auth.historical_core_ms;
    let mut node = auth.historical_node_ms;
    let mut statement =
        db.prepare("SELECT json FROM funding_reservations WHERE authorization_id=?")?;
    let rows = statement.query_map([verified.authorization_id()?], |row| {
        row.get::<_, String>(0)
    })?;
    for row in rows {
        let old: SignedReservation = serde_json::from_str(&row?)?;
        core = core
            .checked_add(old.body.core_ms)
            .context("funding core charge overflow")?;
        node = node
            .checked_add(old.body.node_ms)
            .context("funding node charge overflow")?;
    }
    let body = &verified.reservation().body;
    ensure!(
        core.checked_add(body.core_ms)
            .is_some_and(|sum| sum <= auth.core_ms)
            && node
                .checked_add(body.node_ms)
                .is_some_and(|sum| sum <= auth.node_ms),
        "original funding ceiling exhausted"
    );
    db.execute(
        "INSERT INTO funding_reservations VALUES(?,?,?,?,?,NULL)",
        params![
            body.reservation_id,
            verified.authorization_id()?,
            body.demand.task_sha256,
            serde_json::to_string(verified.reservation())?,
            "reserved"
        ],
    )?;
    Ok(body.reservation_id.clone())
}

fn original_reservation(
    db: &Connection,
    verified: &VerifiedReservation,
) -> Result<(String, Option<String>)> {
    stored_authorization(db, verified)?;
    let (json, state, start): (String, String, Option<String>) = db.query_row(
        "SELECT json,state,start_json FROM funding_reservations WHERE id=?",
        [verified.reservation_id()],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        serde_json::from_str::<SignedReservation>(&json)? == *verified.reservation(),
        "original registered reservation differs"
    );
    Ok((state, start))
}

pub fn preflight(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let (state, _) = original_reservation(db, verified)?;
    ensure!(
        state == "reserved",
        "original task start already consumed or unavailable"
    );
    Ok(())
}

fn execution_valid(
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<()> {
    let demand = verified.demand();
    ensure!(
        execution.process.boot_id == demand.boot_id
            && execution.process.pid > 0
            && execution.process.birth_ticks > 0
            && execution.executable_sha256 == demand.executable_sha256
            && execution.runtime_sha256 == demand.runtime_sha256
            && execution.request_sha256 == demand.task_sha256,
        "authenticated execution custody does not match the original funded task"
    );
    ensure!(
        clock.unix_ms >= verified.reservation().body.not_before_unix_ms
            && clock.unix_ms < demand.deadline_unix_ms
            && clock.monotonic_ns < demand.deadline_monotonic_ns,
        "original task deadline elapsed before funding consumption"
    );
    Ok(())
}

pub fn participant_id(reservation_id: &str, slot: u32) -> Result<String> {
    digest(&("parallel.funding_participant.v1", reservation_id, slot))
}

pub fn event(
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    participant_slot: Option<u32>,
    clock: FundingClock,
) -> Result<FundingEvent> {
    let event_key = match participant_slot {
        Some(slot) => participant_id(verified.reservation_id(), slot)?,
        None => digest(&("parallel.funding_start.v1", verified.reservation_id()))?,
    };
    Ok(FundingEvent {
        schema: "parallel.funding_event.v1".into(),
        event_key,
        authorization: verified.authorization().clone(),
        reservation: verified.reservation().clone(),
        demand: verified.demand().clone(),
        execution: execution.clone(),
        participant_slot,
        claimed_unix_ms: clock.unix_ms,
        claimed_monotonic_ns: clock.monotonic_ns,
    })
}

pub fn consume_start(
    db: &Connection,
    verified: &VerifiedReservation,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<FundingEvent> {
    transaction_required(db)?;
    execution_valid(verified, execution, clock)?;
    preflight(db, verified)?;
    concurrent_admission(db, verified)?;
    let event = event(verified, execution, None, clock)?;
    ensure!(db.execute("UPDATE funding_reservations SET state='started',start_json=? WHERE id=? AND state='reserved'",
        params![serde_json::to_string(&event)?, verified.reservation_id()])? == 1, "funding start consumed concurrently");
    Ok(event)
}

/// Check every still-started reservation atomically, across funding domains.
/// An unknown or incomplete termination keeps its original allocation occupied;
/// successful task completion never refunds the charged funding ceilings.
fn concurrent_admission(db: &Connection, verified: &VerifiedReservation) -> Result<()> {
    let caps = &verified.demand().resources;
    let allocation = &verified.authorization().body.allocation;
    let (mut cores, mut memory, mut searches) =
        (caps.cpu_cores, caps.host_memory_bytes, caps.cpu_searches);
    let requested: BTreeSet<u32> = caps.gpu_devices.iter().copied().collect();
    let mut statement =
        db.prepare("SELECT json FROM funding_reservations WHERE state='started'")?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let old: SignedReservation = serde_json::from_str(&row?)?;
        if old.body.demand.host_id != verified.demand().host_id {
            continue;
        }
        // A host reboot is not evidence that the original allocation/attempt
        // was independently reconciled. It cannot silently release custody.
        ensure!(
            old.body.demand.boot_id == verified.demand().boot_id,
            "unreconciled funding allocation from an earlier host boot"
        );
        let old_caps = &old.body.demand.resources;
        ensure!(
            old_caps
                .gpu_devices
                .iter()
                .all(|id| !requested.contains(id)),
            "funded GPU allocation already occupied"
        );
        cores = cores
            .checked_add(old_caps.cpu_cores)
            .context("concurrent CPU allocation overflow")?;
        memory = memory
            .checked_add(old_caps.host_memory_bytes)
            .context("concurrent host memory overflow")?;
        searches = searches
            .checked_add(old_caps.cpu_searches)
            .context("concurrent search allocation overflow")?;
    }
    ensure!(
        cores <= allocation.cpu_cores
            && memory <= allocation.host_memory_bytes
            && searches <= verified.authorization().body.max_concurrent_cpu_searches,
        "funded concurrent host allocation exhausted"
    );
    Ok(())
}

pub fn claim_participant(
    db: &Connection,
    verified: &VerifiedReservation,
    slot: u32,
    execution: &ExecutionClaim,
    clock: FundingClock,
) -> Result<FundingEvent> {
    transaction_required(db)?;
    execution_valid(verified, execution, clock)?;
    ensure!(
        verified
            .reservation()
            .body
            .participant_slots
            .contains(&slot),
        "undeclared funding participant slot"
    );
    let (state, start) = original_reservation(db, verified)?;
    ensure!(
        state == "started" && start.is_some(),
        "participant cannot precede the original task start"
    );
    let start: FundingEvent =
        serde_json::from_str(start.as_deref().context("original task start absent")?)?;
    ensure!(
        start.reservation == *verified.reservation() && start.participant_slot.is_none(),
        "participant original supervisor start mismatch"
    );
    // A start whose witness response was lost cannot release any rank.
    ensure!(
        db.query_row(
            "SELECT EXISTS(SELECT 1 FROM funding_witness WHERE event_key=?)",
            [&start.event_key],
            |row| row.get::<_, bool>(0)
        )?,
        "original task start has no retained independent witness ACK"
    );
    let mut statement =
        db.prepare("SELECT json FROM funding_participants WHERE reservation_id=?")?;
    let originals =
        statement.query_map([verified.reservation_id()], |row| row.get::<_, String>(0))?;
    for original in originals {
        let original: FundingEvent = serde_json::from_str(&original?)?;
        ensure!(
            original.execution.process != execution.process,
            "original participant process already consumed a rank slot"
        );
    }
    let event = event(verified, execution, Some(slot), clock)?;
    db.execute(
        "INSERT INTO funding_participants VALUES(?,?,?,?)",
        params![
            event.participant_id()?,
            verified.reservation_id(),
            slot.to_string(),
            serde_json::to_string(&event)?
        ],
    )?;
    Ok(event)
}

/// Validate the full signed event against installed trust and the independently
/// observed execution identity. This is suitable for the existing witness's
/// strict funding-event branch; it does not accept arbitrary journal payloads.
pub fn verify_event(trust: &FundingTrust, value: &FundingEvent, clock: FundingClock) -> Result<()> {
    validate_witness_event(
        trust,
        value,
        &trust.authority_instance,
        &trust.authority_epoch,
    )?;
    let verified = verify_reservation(
        trust,
        &value.authorization,
        &value.reservation,
        &value.demand,
        clock,
    )?;
    execution_valid(&verified, &value.execution, clock)
}

/// Verify original event bytes at their recorded claim time. This supports
/// retrieval of a previously committed ACK after the original deadline without
/// restamping it or authorizing another start. New witness writes must separately
/// enforce the returned original monotonic deadline on their transport path.
pub fn validate_witness_event(
    trust: &FundingTrust,
    value: &FundingEvent,
    authority_instance: &str,
    authority_epoch: &str,
) -> Result<(String, u64)> {
    ensure!(
        trust.authority_instance == authority_instance && trust.authority_epoch == authority_epoch,
        "funding witness authority instance/epoch mismatch"
    );
    ensure!(
        value.schema == "parallel.funding_event.v1",
        "unsupported funding event"
    );
    let clock = FundingClock {
        unix_ms: value.claimed_unix_ms,
        monotonic_ns: value.claimed_monotonic_ns,
    };
    let verified = verify_reservation(
        trust,
        &value.authorization,
        &value.reservation,
        &value.demand,
        clock,
    )?;
    execution_valid(&verified, &value.execution, clock)?;
    if let Some(slot) = value.participant_slot {
        ensure!(
            value.reservation.body.participant_slots.contains(&slot),
            "funding event participant outside reservation"
        );
    }
    ensure!(
        event(&verified, &value.execution, value.participant_slot, clock)? == *value,
        "funding event original identity changed"
    );
    Ok((value.event_key.clone(), value.demand.deadline_monotonic_ns))
}
