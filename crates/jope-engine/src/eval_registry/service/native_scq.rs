//! Separate native wire route; the retired/legacy request path is unchanged.
use super::*;
pub(super) fn connection(
    state: &Arc<State>,
    stream: &mut UnixStream,
    peer: &ProcessIdentity,
    request: super::super::native_scq::NativeRequest,
    stopping: bool,
) -> Result<()> {
    ensure!(!stopping, "authority draining: native requests unavailable");
    let policy = state
        .native_fit_policy
        .as_ref()
        .context("installed native fit policy unavailable")?
        .clone();
    ensure!(
        state
            .active
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < state
                .config
                .max_workers)
                .then_some(n + 1))
            .is_ok(),
        "all native fit service slots occupied"
    );
    let _active = CountGuard {
        counter: &state.active,
    };
    state.requests.fetch_add(1, Ordering::SeqCst);
    let input = File::from(receive_fd(stream, Duration::from_secs(5))?);
    let bytes = read_sealed(&input, state.config.max_descriptor_bytes)?;
    ensure!(
        sha(&bytes) == request.descriptor_sha256,
        "native sealed descriptor changed"
    );
    let inputs: ServiceInputs = serde_json::from_slice(&bytes)?;
    let fit = inputs.verify(
        &state.identity.executable_sha256,
        &state.runtime,
        &state.config,
    )?;
    let mut native =
        super::super::native_scq::verify_request(policy, peer, &request, &inputs, &fit)?;
    let claim = {
        let _dispatch = state.dispatch_gate()?;
        lock(&state.authority)?.claim_native(&fit, &mut native)?
    };
    let result = match claim {
        ClaimOutcome::Claimed(claim) => {
            let deadline = native.deadline_monotonic_ns;
            supervise(
                state,
                &fit,
                &inputs,
                &input,
                &claim,
                deadline,
                Some(&mut native),
            )?
        }
        ClaimOutcome::Existing {
            state: status,
            completion_sha256: Some(sha),
            ..
        } if status == "completed" || status == "terminal_failure" => {
            state.reused.fetch_add(1, Ordering::SeqCst);
            result_from_original(state, &sha, true)?
        }
        other => bail!("original native fit unavailable; never replay: {other:?}"),
    };
    send_json(
        stream,
        &json!({"ok":true,"result":result}),
        Duration::from_secs(5),
    )
}
pub(crate) fn submit(
    socket: &Path,
    root: &Path,
    witness: &Path,
    inputs: &ServiceInputs,
    request: super::super::native_scq::NativeRequest,
) -> Result<ServiceResult> {
    let bytes = serde_json::to_vec(inputs)?;
    ensure!(
        sha(&bytes) == request.descriptor_sha256,
        "native input request differs"
    );
    let descriptor = sealed(&bytes)?;
    let (mut stream, identity) = connect_approved(socket, root, witness)?;
    send_json(
        &mut stream,
        &Request::SubmitNative { request },
        Duration::from_secs(5),
    )?;
    send_fd(&stream, descriptor.as_raw_fd(), Duration::from_secs(5))?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(120))?;
    ensure!(
        response["ok"] == true,
        "native fit service rejected request: {}",
        response["error"]
    );
    let result: ServiceResult = serde_json::from_value(response["result"].clone())?;
    ensure!(
        result.receipt.authority_instance == identity.authority_instance,
        "native fit result authority changed"
    );
    validate_response(result, &identity)
}
