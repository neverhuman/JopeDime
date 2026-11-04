use super::*;

/// Witness checks signed current policy provenance, never a caller's boolean.
pub(crate) fn verify_witness_event(
    trust: &policy::NativeWitnessTrust,
    event: &Value,
    identity: &super::super::authority::FitIdentity,
    start: &super::super::authority::StartToken,
    witness: &super::super::witness::WitnessPolicy,
) -> Result<()> {
    ensure!(
        event["schema"] == "jope.native_fit_authorization.v1",
        "native witness authorization schema differs"
    );
    let signed: policy::SignedNativeGrant = serde_json::from_value(event["signed_grant"].clone())?;
    policy::verify_signed(
        trust,
        &signed,
        event["authorized_unix_ms"]
            .as_u64()
            .context("original native authorization time absent")?,
    )?;
    let grant = &signed.body;
    ensure!(
        event["provenance"]["scope"] == grant.scope
            && event["provenance"]["task_id"] == start.task_id
            && identity.fit_seed == grant.fit_seed,
        "native witness provenance differs"
    );
    ensure!(
        grant.authority_instance == witness.authority_instance
            && grant.authority_epoch == witness.authority_epoch
            && grant.authority_executable_sha256 == witness.authority_executable_sha256
            && grant.witness_public_key == witness.witness_public_key
            && grant
                .allowed_runtime_sha256
                .contains(&identity.runtime_sha256)
            && grant.training_sha256.contains(&identity.training_sha256)
            && grant.numerical_contract_sha256 == identity.numerical_contract_sha256,
        "native witness grant/authority/fit binding mismatch"
    );
    let task = event["task_manifest_sha256"]
        .as_str()
        .context("native task digest absent")?;
    ensure!(
        grant.approved_task_sha256.contains(task)
            && event["candidate_index"]
                .as_u64()
                .context("native candidate absent")?
                < grant.max_candidates_per_task as u64
            && event["fit_key"] == start.fit_key
            && event["deadline_monotonic_ns"]
                .as_u64()
                .context("native deadline absent")?
                >= start.deadline_monotonic_ns,
        "native witness task/budget/start differs"
    );
    let placement: TrustedFitChildPlacement = serde_json::from_value(event["placement"].clone())?;
    let mut body = serde_json::to_value(&placement)?;
    body.as_object_mut().unwrap().remove("receipt_sha256");
    ensure!(
        placement.schema == "jope.smartcluster_fit_placement.v1"
            && placement.child_executable_sha256 == grant.evaluator_executable_sha256
            && placement.cgroup_path
                == grant.cgroup_root.join(format!(
                    "scq-{}-attempt-{}-fence-{}",
                    placement.target_attempt.job_id,
                    placement.target_attempt.number,
                    placement.target_attempt.fence
                ))
            && hash(&serde_json::to_vec(&body)?) == placement.receipt_sha256
            && placement.child_pid == start.executor.pid
            && placement.child_birth_ticks == start.executor.birth_ticks
            && placement.child_boot_id == start.executor.boot_id
            && event["target_attempt"] == serde_json::to_value(placement.target_attempt)?
            && event["authority_attempt"] == serde_json::to_value(placement.authority_attempt)?
            && placement.task_manifest_sha256 == task
            && event["descriptor_sha256"] == placement.descriptor_sha256
            && event["candidate_index"] == placement.candidate_index,
        "native witness original placement differs"
    );
    Ok(())
}
