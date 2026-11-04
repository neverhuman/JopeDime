use super::*;

pub fn submit(
    socket: &Path,
    inputs: &ServiceInputs,
    provenance: RequestProvenance,
) -> Result<ServiceResult> {
    let root = PathBuf::from(
        std::env::var_os("JOPE_FIT_AUTHORITY").context("approved fit authority absent")?,
    );
    let witness =
        PathBuf::from(std::env::var_os("JOPE_FIT_WITNESS").context("approved fit witness absent")?);
    submit_to(socket, &root, &witness, inputs, provenance)
}
pub fn submit_to(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &ServiceInputs,
    provenance: RequestProvenance,
) -> Result<ServiceResult> {
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let descriptor_sha256 = sha(&bytes);
    let (mut stream, identity) = connect_approved(socket, root, witness)?;
    send_json(
        &mut stream,
        &Request::Submit {
            provenance,
            descriptor_sha256,
        },
        Duration::from_secs(5),
    )?;
    send_fd(&stream, descriptor.as_raw_fd(), Duration::from_secs(5))?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(120))?;
    ensure!(
        response["ok"] == true,
        "fit service rejected request: {}",
        response["error"]
    );
    validate_response(
        serde_json::from_value(response["result"].clone())?,
        &identity,
    )
}
pub fn verify_original(
    socket: &Path,
    root: &Path,
    witness: &Path,
    completion: &str,
    scope: &str,
) -> Result<ServiceResult> {
    let (mut stream, identity) = connect_approved(socket, root, witness)?;
    send_json(
        &mut stream,
        &Request::Verify {
            completion_sha256: completion.into(),
            scope: scope.into(),
        },
        Duration::from_secs(5),
    )?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(30))?;
    ensure!(
        response["ok"] == true,
        "original completion verification rejected: {}",
        response["error"]
    );
    validate_response(
        serde_json::from_value(response["result"].clone())?,
        &identity,
    )
}
/// No Submit request, fit descriptor, start token, or retry exists on this path.
pub fn query_evaluate(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &query::QueryInputs,
) -> Result<ServiceQueryResult> {
    let deadline = inputs.deadline_monotonic_ns;
    remaining_request(deadline)?;
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let (mut stream, identity) = connect_approved_until(socket, root, witness, deadline)?;
    send_json(
        &mut stream,
        &Request::QueryEvaluate {
            descriptor_sha256: sha(&bytes),
        },
        remaining_request(deadline)?,
    )?;
    send_fd(
        &stream,
        descriptor.as_raw_fd(),
        remaining_request(deadline)?,
    )?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline)?)?;
    remaining_request(deadline)?;
    ensure!(
        response["ok"] == true,
        "query evaluation rejected: {}",
        response["error"]
    );
    validate_query_response(
        serde_json::from_value(response["result"].clone())?,
        inputs,
        &identity,
        deadline,
    )
}
/// Retrieve the committed original query without accepting or scoring any work.
/// Its execution deadline remains immutable; only transport gets a new deadline.
pub fn query_retrieve(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &query::QueryInputs,
    deadline_monotonic_ns: u64,
) -> Result<ServiceQueryResult> {
    remaining_request(deadline_monotonic_ns)?;
    let bytes = serde_json::to_vec(inputs)?;
    let descriptor = sealed(&bytes)?;
    let (mut stream, identity) =
        connect_approved_until(socket, root, witness, deadline_monotonic_ns)?;
    send_json(
        &mut stream,
        &Request::QueryRetrieve {
            descriptor_sha256: sha(&bytes),
            deadline_monotonic_ns,
        },
        remaining_request(deadline_monotonic_ns)?,
    )?;
    send_fd(
        &stream,
        descriptor.as_raw_fd(),
        remaining_request(deadline_monotonic_ns)?,
    )?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline_monotonic_ns)?)?;
    remaining_request(deadline_monotonic_ns)?;
    ensure!(
        response["ok"] == true,
        "original query retrieval rejected: {}",
        response["error"]
    );
    let result: ServiceQueryResult = serde_json::from_value(response["result"].clone())?;
    ensure!(result.reused, "query retrieval cannot create an execution");
    validate_query_response(result, inputs, &identity, deadline_monotonic_ns)
}
pub(crate) fn validate_query_response(
    result: ServiceQueryResult,
    inputs: &query::QueryInputs,
    identity: &ServiceIdentity,
    response_deadline: u64,
) -> Result<ServiceQueryResult> {
    let record = &result.record;
    ensure!(
        result.authority_instance == identity.authority_instance
            && record.request_sha256 == inputs.descriptor_sha256()?
            && record.completion_sha256 == inputs.completion_sha256,
        "query response original authority/request mismatch"
    );
    for (path, digest) in [
        (&result.request_path, &record.request_sha256),
        (&result.completion_path, &record.completion_sha256),
        (&result.model_path, &record.model_sha256),
        (&result.predictions_path, &record.predictions_sha256),
        (&result.metrics_path, &record.metrics_sha256),
        (&result.receipt_path, &record.receipt_sha256),
    ] {
        remaining_request(response_deadline)?;
        ensure!(
            path == &identity.authority_root.join("objects").join(digest),
            "query artifact outside approved authority"
        );
        read_content(path, digest)?;
    }
    ensure!(
        read_content(&result.request_path, &record.request_sha256)? == serde_json::to_vec(inputs)?,
        "query response descriptor differs from original request"
    );
    let receipt = query::verify_artifacts(
        &read_content(&result.completion_path, &record.completion_sha256)?,
        &read_content(&result.model_path, &record.model_sha256)?,
        inputs,
        &read_content(&result.receipt_path, &record.receipt_sha256)?,
        &read_content(&result.predictions_path, &record.predictions_sha256)?,
        &read_content(&result.metrics_path, &record.metrics_sha256)?,
    )?;
    ensure!(
        serde_json::to_value(&receipt)? == serde_json::to_value(&result.receipt)?
            && receipt.query_key == record.query_key
            && receipt.authority_instance == identity.authority_instance
            && receipt.producer_executable_sha256 == identity.executable_sha256,
        "query response does not match original producer/receipt"
    );
    remaining_request(response_deadline)?;
    Ok(result)
}
pub fn stats(socket: &Path, root: &Path, witness: &Path) -> Result<ServiceStats> {
    let (mut stream, _) = connect_approved(socket, root, witness)?;
    send_json(&mut stream, &Request::Stats, Duration::from_secs(5))?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(5))?;
    Ok(serde_json::from_value(response["stats"].clone())?)
}
pub(crate) fn validate_response(
    result: ServiceResult,
    identity: &ServiceIdentity,
) -> Result<ServiceResult> {
    ensure!(
        result.authority_instance == identity.authority_instance
            && result.completion_path
                == identity
                    .authority_root
                    .join("objects")
                    .join(&result.completion_sha256),
        "response is outside approved authority"
    );
    let receipt = verify_completion_receipt(&read_content(
        &result.completion_path,
        &result.completion_sha256,
    )?)?;
    ensure!(
        serde_json::to_value(&receipt)? == serde_json::to_value(&result.receipt)?
            && receipt.fit_key == result.fit_key
            && receipt.model_sha256 == result.model_sha256,
        "response does not match original immutable receipt"
    );
    ensure!(
        result.model_path
            == result
                .model_sha256
                .as_ref()
                .map(|sha| identity.authority_root.join("objects").join(sha)),
        "model reference outside authority"
    );
    if receipt.outcome == ExecutionOutcome::Success {
        result.model_bytes()?;
    }
    Ok(result)
}
