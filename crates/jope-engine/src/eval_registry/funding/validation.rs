use super::*;

pub fn verify_authorization(
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    clock: FundingClock,
) -> Result<String> {
    ensure!(
        trust.schema == "parallel.funding_trust.v1"
            && text_valid(&trust.domain_id)
            && text_valid(&trust.authority_instance)
            && text_valid(&trust.authority_epoch)
            && hash_valid(&trust.authority_manifest_sha256)
            && hash_valid(&trust.authority_executable_sha256)
            && text_valid(&trust.host_id)
            && text_valid(&trust.boot_id),
        "installed funding trust invalid"
    );
    let body = &authorization.body;
    ensure!(
        body.approved_task_sha256.len() <= MAX_APPROVED_TASKS
            && serde_json::to_vec(body)?.len() <= MAX_AUTHORIZATION_BYTES,
        "funding authorization exceeds bounded witness input limits"
    );
    ensure!(
        body.schema == "parallel.funding_authorization.v1"
            && body.domain_id == trust.domain_id
            && text_valid(&body.scope),
        "funding authorization domain/scope mismatch"
    );
    let id = digest(body)?;
    ensure!(
        hash_valid(&trust.authorization_sha256) && id == trust.authorization_sha256,
        "funding authorization is not the independently pinned decision"
    );
    verify_signature(
        &trust.authorization_public_key,
        AUTH_TAG,
        body,
        &authorization.signature,
    )?;
    ensure!(
        body.reservation_public_key == trust.reservation_public_key
            && hash_valid(&body.reservation_public_key),
        "funding reservation delegation mismatch"
    );
    ensure!(
        hash_valid(&body.authority_evidence_sha256)
            && hash_valid(&body.historical_reconciliation_sha256)
            && hash_valid(&body.release_prerequisites_sha256)
            && !body.approved_task_sha256.is_empty()
            && body.approved_task_sha256.iter().all(|id| hash_valid(id)),
        "original authority/accounting/prerequisite evidence or frozen task list missing"
    );
    ensure!(
        body.period_start_unix_ms <= clock.unix_ms && clock.unix_ms < body.period_end_unix_ms,
        "funding accounting period unavailable or elapsed"
    );
    ensure!(
        body.period_start_unix_ms < body.period_end_unix_ms
            && body.historical_core_ms <= body.core_ms
            && body.historical_node_ms <= body.node_ms
            && (body.core_ms > 0 || body.node_ms > 0)
            && body.max_concurrent_cpu_searches <= 104,
        "invalid funding ceilings/history"
    );
    ensure!(
        body.allocation.cpu_cores > 0
            && body.allocation.host_memory_bytes > 0
            && body.allocation.gpu_memory_bytes_per_device > 0
            && !body.allocation.gpu_devices.is_empty()
            && body.allocation.gpu_devices.len() <= 8
            && body.allocation.gpu_devices.iter().all(|&device| device < 8)
            && body
                .allocation
                .gpu_devices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                == body.allocation.gpu_devices.len(),
        "invalid independently authorized host allocation"
    );
    ensure!(
        !body.capabilities.is_empty() && body.capabilities.iter().all(|s| text_valid(s)),
        "funding capabilities missing"
    );
    Ok(id)
}

pub fn verify_reservation(
    trust: &FundingTrust,
    authorization: &SignedAuthorization,
    reservation: &SignedReservation,
    demand: &TaskDemand,
    clock: FundingClock,
) -> Result<VerifiedReservation> {
    let authorization_id = verify_authorization(trust, authorization, clock)?;
    let body = &reservation.body;
    ensure!(
        body.schema == "parallel.task_reservation.v1"
            && hash_valid(&body.reservation_id)
            && body.authorization_sha256 == authorization_id
            && body.domain_id == trust.domain_id
            && body.authority_instance == trust.authority_instance
            && body.authority_epoch == trust.authority_epoch
            && body.authority_manifest_sha256 == trust.authority_manifest_sha256
            && body.authority_executable_sha256 == trust.authority_executable_sha256,
        "reservation original domain/authority/epoch mismatch"
    );
    verify_signature(
        &trust.reservation_public_key,
        RESERVATION_TAG,
        body,
        &reservation.signature,
    )?;
    ensure!(
        &body.demand == demand,
        "actual task/deadline/runtime/resource demand differs from signed reservation"
    );
    ensure!(
        hash_valid(&demand.task_sha256)
            && hash_valid(&demand.executable_sha256)
            && hash_valid(&demand.runtime_sha256)
            && demand.host_id == trust.host_id
            && demand.boot_id == trust.boot_id
            && authorization.body.capabilities.contains(&demand.capability)
            && authorization
                .body
                .approved_task_sha256
                .contains(&demand.task_sha256),
        "task identity/host/boot/capability not funded"
    );
    ensure!(
        body.not_before_unix_ms >= authorization.body.period_start_unix_ms
            && body.not_before_unix_ms <= clock.unix_ms
            && clock.unix_ms < demand.deadline_unix_ms
            && demand.deadline_unix_ms <= authorization.body.period_end_unix_ms
            && clock.monotonic_ns < demand.deadline_monotonic_ns,
        "original funding task window elapsed or changed"
    );
    let window = demand
        .deadline_unix_ms
        .checked_sub(body.not_before_unix_ms)
        .context("invalid task window")?;
    let remaining_ns = demand.deadline_monotonic_ns - clock.monotonic_ns;
    ensure!(
        remaining_ns
            <= window
                .checked_mul(1_000_000)
                .context("task window overflow")?,
        "monotonic task deadline exceeds original reserved window"
    );
    let caps = &demand.resources;
    ensure!(
        [1, 2, 4, 8].contains(&caps.world)
            && caps.gpus == caps.world
            && caps.cpu_cores > 0
            && caps.host_memory_bytes > 0
            && caps.gpu_memory_bytes_per_rank > 0
            && caps.cpu_searches <= authorization.body.max_concurrent_cpu_searches
            && caps.gpu_devices.len() == caps.world as usize
            && caps
                .gpu_devices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                == caps.gpu_devices.len()
            && caps.gpu_devices.iter().all(|device| authorization
                .body
                .allocation
                .gpu_devices
                .contains(device))
            && caps.cpu_cores <= authorization.body.allocation.cpu_cores
            && caps.host_memory_bytes <= authorization.body.allocation.host_memory_bytes
            && caps.gpu_memory_bytes_per_rank
                <= authorization.body.allocation.gpu_memory_bytes_per_device,
        "invalid native task resource caps"
    );
    ensure!(
        body.participant_slots == (0..caps.world).collect::<Vec<_>>(),
        "original participant slots do not exactly cover the declared world"
    );
    let core_ceiling = window
        .checked_mul(u64::from(caps.cpu_cores))
        .context("core reservation overflow")?;
    ensure!(
        body.core_ms >= core_ceiling
            && body.node_ms >= window
            && body.core_ms <= authorization.body.core_ms
            && body.node_ms <= authorization.body.node_ms,
        "task reservation does not cover its full original resource/window ceiling"
    );
    Ok(VerifiedReservation {
        authorization: authorization.clone(),
        reservation: reservation.clone(),
    })
}

/// Authority signing only: the caller must possess the installed delegated key;
/// successful signing does not register, charge or start the reservation.
pub fn sign_reservation(body: TaskReservation, key: &SigningKey) -> Result<SignedReservation> {
    let signature = hex::encode(key.sign(&signed_bytes(RESERVATION_TAG, &body)?).to_bytes());
    Ok(SignedReservation { body, signature })
}
