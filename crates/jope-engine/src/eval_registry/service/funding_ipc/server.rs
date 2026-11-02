use super::*;

pub(super) fn lock_with_deadline<T>(
    mutex: &std::sync::Mutex<T>,
    deadline: u64,
) -> Result<std::sync::MutexGuard<'_, T>> {
    let mut sleep = std::time::Duration::from_nanos(FUNDING_LOCK_MIN_SLEEP_NS);
    loop {
        remaining_request(deadline)?;
        match mutex.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                bail!("authority service mutex poisoned")
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }
        remaining_request(deadline)?;
        std::thread::sleep(
            sleep.min(std::time::Duration::from_nanos(
                deadline
                    .checked_sub(request_monotonic_ns()?)
                    .context("funding admission deadline expired")?,
            )),
        );
        sleep = (sleep * 2).min(std::time::Duration::from_nanos(FUNDING_LOCK_MAX_SLEEP_NS));
    }
}

fn dispatch_gate_until(
    state: &Arc<State>,
    deadline: u64,
) -> Result<std::sync::MutexGuard<'_, bool>> {
    let mut gate = lock_with_deadline(&state.dispatch_open, deadline)?;
    ensure!(*gate, "authority dispatch fenced");
    if let Err(error) = state
        .lease
        .verify()
        .and_then(|_| check_socket(&state.identity, &state.config.socket))
    {
        *gate = false;
        return Err(error);
    }
    remaining_request(deadline)?;
    Ok(gate)
}

pub(super) fn serve_request(
    state: &Arc<State>,
    stream: &mut UnixStream,
    peer: &ProcessIdentity,
    request_sha: &str,
    deadline: u64,
) -> Result<()> {
    remaining_request(deadline)?;
    ensure!(
        deadline.saturating_sub(request_monotonic_ns()?) <= 30_000_000_000,
        "funding transport window exceeds bounded admission"
    );
    state
        .funding_active
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |active| {
            (active < state.config.max_workers.min(16)).then_some(active + 1)
        })
        .map_err(|_| anyhow::anyhow!("funding admission slots exhausted"))?;
    let _slot = CountGuard {
        counter: &state.funding_active,
    };
    let installation = state
        .funding_installation
        .as_ref()
        .context("native funding not installed")?;
    // Both service and Authority retain the installation loaded at bind time.
    // A valid but replaced config file must not silently choose another trust.
    installation.revalidate()?;
    let input = File::from(receive_fd(stream, remaining_request(deadline)?)?);
    let bytes = read_sealed(&input, MAX_FUNDING_BYTES)?;
    ensure!(
        sha(&bytes) == request_sha,
        "funding request sealed bytes changed"
    );
    let request: FundingRequest = serde_json::from_slice(&bytes)?;
    request.validate(installation.trust()?)?;
    if let Some((_, reservation, _)) = request.task() {
        ensure!(
            deadline <= reservation.body.demand.deadline_monotonic_ns,
            "funding transport exceeds original task deadline"
        );
    }
    // Observer derives this independently. Client JSON never supplies the
    // execution claim, runtime digest, process birth or executable identity.
    let runtime = if matches!(
        request,
        FundingRequest::Start { .. } | FundingRequest::Participant { .. }
    ) {
        Some(crate::runtime_custody::loaded_runtime_descriptor_until(
            peer, deadline,
        )?)
    } else {
        None
    };
    remaining_request(deadline)?;
    let result = {
        let _dispatch = dispatch_gate_until(state, deadline)?;
        remaining_request(deadline)?;
        let mut authority = lock_with_deadline(&state.authority, deadline)?;
        installation.revalidate()?;
        remaining_request(deadline)?;
        ensure!(
            authenticated_peer(stream)? == *peer,
            "funding peer incarnation changed"
        );
        match &request {
            FundingRequest::Register { authorization } => {
                json!({"authorization_id":authority.funding_register(authorization, deadline)?})
            }
            FundingRequest::Reserve {
                authorization,
                reservation,
                ..
            } => {
                json!({"reservation_id":authority.funding_reserve(authorization, reservation, deadline)?})
            }
            FundingRequest::Preflight {
                authorization,
                reservation,
                ..
            } => {
                authority.funding_preflight(authorization, reservation, deadline)?;
                json!({"reservation_id":reservation.body.reservation_id,"state":"reserved","numerical_start_released":false})
            }
            FundingRequest::Start {
                authorization,
                reservation,
                descriptor_json,
            }
            | FundingRequest::Participant {
                authorization,
                reservation,
                descriptor_json,
                ..
            } => {
                let runtime = runtime
                    .as_ref()
                    .context("funding runtime observation absent")?;
                let descriptor = serde_json::to_vec(descriptor_json)?;
                let execution = f::ExecutionClaim {
                    process: peer.clone(),
                    executable_sha256: runtime["executable_sha256"]
                        .as_str()
                        .context("observed executable absent")?
                        .into(),
                    runtime_sha256: runtime["runtime_sha256"]
                        .as_str()
                        .context("observed runtime absent")?
                        .into(),
                    request_sha256: sha(&descriptor),
                };
                let slot = match &request {
                    FundingRequest::Participant {
                        participant_slot, ..
                    } => Some(*participant_slot),
                    _ => None,
                };
                authority.funding_admit(
                    authorization,
                    reservation,
                    &execution,
                    slot,
                    &descriptor,
                    deadline,
                )?
            }
        }
    };
    // ACK consumption survives a lost reply. No automatic restart or retry is
    // introduced on timeout, process loss or a runtime change after admission.
    if let Some(original) = runtime {
        let current = crate::runtime_custody::loaded_runtime_descriptor_until(peer, deadline)?;
        ensure!(
            current["runtime_sha256"] == original["runtime_sha256"]
                && current["executable_sha256"] == original["executable_sha256"],
            "funding peer runtime changed during admission"
        );
    }
    remaining_request(deadline)?;
    #[cfg(test)]
    if state.config.drop_funding_reply && matches!(request, FundingRequest::Start { .. }) {
        return Ok(());
    }
    let bytes = serde_json::to_vec(&result)?;
    ensure!(
        bytes.len() as u64 <= MAX_FUNDING_BYTES,
        "funding proof exceeds sealed transport bound"
    );
    let output = sealed(&bytes)?;
    send_json(
        stream,
        &json!({"ok":true,"funding_response_sha256":sha(&bytes)}),
        remaining_request(deadline)?,
    )?;
    send_fd(stream, output.as_raw_fd(), remaining_request(deadline)?)?;
    Ok(())
}
