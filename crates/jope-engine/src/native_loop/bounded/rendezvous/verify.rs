use super::*;

/// Participant-side check of the coordinator's signed challenge.
pub fn verify_challenge(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    frame: &[u8],
    now_unix_ms: u64,
) -> Result<Challenge, RefusalCause> {
    if frame.len() as u64 > u64::from(contract.rendezvous.max_message_bytes) {
        return Err(RefusalCause::Oversized {
            bytes: frame.len() as u64,
            limit: u64::from(contract.rendezvous.max_message_bytes),
        });
    }
    let c: Challenge = serde_json::from_slice(frame).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    verify(
        &contract.coordinator_public_key,
        CHALLENGE_DOMAIN,
        &c.body,
        &c.signature,
    )
    .map_err(|_| RefusalCause::ForgedCoordinator)?;
    let b = &c.body;
    if b.schema != CHALLENGE_SCHEMA || !is_digest(&b.rendezvous_id) {
        return Err(RefusalCause::Malformed {
            detail: "challenge schema".into(),
        });
    }
    if b.run != contract.run {
        return Err(RefusalCause::WrongRun);
    }
    if b.attempt != contract.attempt {
        return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
    }
    if b.plan_sha256 != plan_sha256 {
        return Err(RefusalCause::WrongPlan);
    }
    if now_unix_ms > b.join_deadline_unix_ms
        || b.join_deadline_unix_ms > contract.original_deadline_unix_ms
        || b.opened_unix_ms > now_unix_ms.saturating_add(contract.telemetry.clock_skew_ms)
    {
        return Err(RefusalCause::StaleRendezvous);
    }
    Ok(c)
}

/// Proof, held by one participant, that the whole planned group was admitted. Only
/// [`verify_admission`] constructs it; a rank must hold one before starting any work.
#[derive(Clone, Debug)]
pub struct AdmittedMember {
    admission: GroupAdmission,
    admission_sha256: String,
    global_rank: u32,
    world_size: u32,
    bootstrap: Vec<u8>,
}
impl AdmittedMember {
    pub fn admission(&self) -> &GroupAdmission {
        &self.admission
    }
    pub fn admission_sha256(&self) -> &str {
        &self.admission_sha256
    }
    pub fn global_rank(&self) -> u32 {
        self.global_rank
    }
    pub fn world_size(&self) -> u32 {
        self.world_size
    }
    /// The communicator bootstrap (join point for the trainer's communicator creation).
    pub fn communicator_bootstrap(&self) -> &[u8] {
        &self.bootstrap
    }
}
pub fn admission_sha256(admission: &GroupAdmission) -> Result<String> {
    digest(admission)
}
/// Identity under which a rank's work is dispatched (one per plan attempt and rank); also
/// how a restarted coordinator finds work whose launch it never journaled.
pub fn dispatch_id(plan_sha256: &str, attempt: u32, global_rank: u32) -> Result<String> {
    digest(&(
        "jope.native_dispatch_id.v1",
        plan_sha256,
        attempt,
        global_rank,
    ))
}
/// Fresh rendezvous nonce from the kernel.
pub fn fresh_nonce() -> Result<[u8; 16]> {
    let mut nonce = [0u8; 16];
    fs::File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    Ok(nonce)
}
/// Controller-side check that a group admission is the coordinator's own full-group
/// admission for this plan (used when persisting and when verifying the journal).
pub fn verify_group_admission(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    admission: &GroupAdmission,
) -> Result<()> {
    verify(
        &contract.coordinator_public_key,
        ADMISSION_DOMAIN,
        &admission.body,
        &admission.signature,
    )
    .context("group admission is not signed by the planned coordinator")?;
    let b = &admission.body;
    let grants: Vec<_> = contract.participants.iter().map(|p| &p.grant).collect();
    ensure!(
        b.schema == ADMISSION_SCHEMA
            && b.run == contract.run
            && b.attempt == contract.attempt
            && b.plan_sha256 == plan_sha256
            && b.original_deadline_unix_ms == contract.original_deadline_unix_ms
            && b.grants_sha256 == digest(&grants)?
            && b.members.len() == contract.world_size()
            && b.members
                .iter()
                .zip(&contract.participants)
                .all(|(m, p)| m.global_rank == p.global_rank
                    && m.host == p.host.host
                    && m.boot_id == p.host.boot_id
                    && m.device_uuid == p.device.as_ref().map(|d| d.uuid.clone())
                    && is_digest(&m.hello_sha256)),
        "group admission does not bind the full planned group"
    );
    Ok(())
}

/// Participant-side verification of a full-group admission against its own hello.
pub fn verify_admission(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    own_hello: &Hello,
    frame: &[u8],
    now_unix_ms: u64,
) -> Result<AdmittedMember, RefusalCause> {
    if frame.len() as u64 > u64::from(contract.rendezvous.max_message_bytes) {
        return Err(RefusalCause::Oversized {
            bytes: frame.len() as u64,
            limit: u64::from(contract.rendezvous.max_message_bytes),
        });
    }
    let a: GroupAdmission = serde_json::from_slice(frame).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    verify(
        &contract.coordinator_public_key,
        ADMISSION_DOMAIN,
        &a.body,
        &a.signature,
    )
    .map_err(|_| RefusalCause::ForgedCoordinator)?;
    let b = &a.body;
    let rank = own_hello.body.global_rank;
    if b.schema != ADMISSION_SCHEMA || b.run != contract.run {
        return Err(RefusalCause::WrongRun);
    }
    if b.attempt != contract.attempt {
        return Err(RefusalCause::ReplayedAttempt { attempt: b.attempt });
    }
    if b.plan_sha256 != plan_sha256 {
        return Err(RefusalCause::WrongPlan);
    }
    if b.rendezvous_id != own_hello.body.rendezvous_id {
        return Err(RefusalCause::StaleRendezvous);
    }
    if b.original_deadline_unix_ms != contract.original_deadline_unix_ms
        || now_unix_ms >= contract.original_deadline_unix_ms
    {
        return Err(RefusalCause::DeadlineElapsed);
    }
    let grants: Vec<_> = contract.participants.iter().map(|p| &p.grant).collect();
    if digest(&grants).ok().as_deref() != Some(b.grants_sha256.as_str()) {
        return Err(RefusalCause::WrongGrant { rank });
    }
    let Some(p) = contract.participant(rank) else {
        return Err(RefusalCause::UnknownRank { rank });
    };
    if now_unix_ms >= p.grant.body.expires_unix_ms {
        return Err(RefusalCause::GrantExpired { rank });
    }
    let ranks: Vec<u32> = b.members.iter().map(|m| m.global_rank).collect();
    if !ranks.iter().copied().eq(0..contract.world_size() as u32) {
        return Err(RefusalCause::IncompleteGroup);
    }
    for (m, planned) in b.members.iter().zip(&contract.participants) {
        if m.host != planned.host.host
            || m.boot_id != planned.host.boot_id
            || m.device_uuid != planned.device.as_ref().map(|d| d.uuid.clone())
        {
            return Err(RefusalCause::WrongParticipant {
                rank: m.global_rank,
            });
        }
    }
    let own = serde_json::to_vec(own_hello).map_err(|e| RefusalCause::Malformed {
        detail: e.to_string(),
    })?;
    if b.members[rank as usize].hello_sha256 != crate::manifest::hash(&own) {
        return Err(RefusalCause::Equivocation { rank });
    }
    let bootstrap =
        hex::decode(&b.communicator_bootstrap).map_err(|e| RefusalCause::Malformed {
            detail: e.to_string(),
        })?;
    if bootstrap.is_empty() || bootstrap.len() > MAX_BOOTSTRAP_BYTES {
        return Err(RefusalCause::Malformed {
            detail: "communicator bootstrap".into(),
        });
    }
    Ok(AdmittedMember {
        admission_sha256: admission_sha256(&a).map_err(|e| RefusalCause::Malformed {
            detail: e.to_string(),
        })?,
        admission: a,
        global_rank: rank,
        world_size: contract.world_size() as u32,
        bootstrap,
    })
}

/// A result whose envelope and bytes authenticated against the admitted plan.
/// Only [`verify_result`] constructs it.
#[derive(Clone, Debug)]
pub struct VerifiedResult {
    envelope: Vec<u8>,
    body: ResultBody,
    bytes: Vec<u8>,
}
impl VerifiedResult {
    pub fn body(&self) -> &ResultBody {
        &self.body
    }
    pub fn envelope(&self) -> &[u8] {
        &self.envelope
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
/// Authenticate one rank's retained result. Refuses oversize envelopes, forged or
/// wrong-key signatures, other runs/attempts/plans/admissions, bytes that differ from the
/// signed digest, results larger than the planned destination and any update count
/// other than the synchronous group's planned count.
pub fn verify_result(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    admission_sha256: &str,
    envelope: &[u8],
    bytes: &[u8],
) -> Result<VerifiedResult> {
    ensure!(
        envelope.len() as u64 <= u64::from(contract.rendezvous.max_message_bytes),
        "result envelope exceeds message limit"
    );
    let signed: SignedResult = serde_json::from_slice(envelope)?;
    let b = &signed.body;
    let p = contract
        .participant(b.global_rank)
        .context("result names no planned participant")?;
    verify(&p.public_key, RESULT_DOMAIN, b, &signed.signature)
        .context("result is not signed by the planned participant key")?;
    ensure!(
        b.schema == RESULT_SCHEMA
            && b.run == contract.run
            && b.attempt == contract.attempt
            && b.plan_sha256 == plan_sha256
            && b.admission_sha256 == admission_sha256
            && b.delivery_id > 0,
        "result is bound to a different run, attempt, plan or admission"
    );
    ensure!(
        b.result_bytes <= p.result_destination.max_bytes
            && bytes.len() as u64 == b.result_bytes
            && crate::manifest::hash(bytes) == b.result_sha256,
        "result bytes differ from the authenticated digest or planned destination bound"
    );
    ensure!(
        b.updates == contract.counters.planned_updates,
        "result update count differs from the synchronous plan"
    );
    Ok(VerifiedResult {
        envelope: envelope.to_vec(),
        body: signed.body,
        bytes: bytes.to_vec(),
    })
}
