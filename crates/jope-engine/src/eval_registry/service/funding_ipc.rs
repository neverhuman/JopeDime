//! Separate pinned cross-executable funding transport. The ordinary fit client
//! retains its same-executable rule and original 64 KiB frame bound.
use super::*;
use crate::eval_registry::{funding as f, funding_config::InstalledFunding};

const MAX_FUNDING_BYTES: u64 = 16 * 1024 * 1024;
const FUNDING_LOCK_MIN_SLEEP_NS: u64 = 1_000_000;
const FUNDING_LOCK_MAX_SLEEP_NS: u64 = 25_000_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum FundingRequest {
    Register { authorization: f::SignedAuthorization },
    Reserve { authorization: f::SignedAuthorization, reservation: f::SignedReservation, descriptor_json: Value },
    Preflight { authorization: f::SignedAuthorization, reservation: f::SignedReservation, descriptor_json: Value },
    Start { authorization: f::SignedAuthorization, reservation: f::SignedReservation, descriptor_json: Value },
    Participant { authorization: f::SignedAuthorization, reservation: f::SignedReservation, descriptor_json: Value, participant_slot: u32 },
}

/// Opaque proof of one witnessed native admission. This is not a complete GPU
/// execution permit: device lease, hold revision, inputs and post-context
/// runtime qualification remain separate mandatory task-contract checks.
pub struct VerifiedFundingAdmission {
    receipt: f::FundingReceipt,
}
impl VerifiedFundingAdmission {
    pub fn receipt(&self) -> &f::FundingReceipt { &self.receipt }
    pub fn check_current(&self, descriptor: &Value, participant: Option<u32>) -> Result<()> {
        let event = &self.receipt.event;
        ensure!(event.execution.process == ProcessIdentity::current()?
            && event.execution.request_sha256 == sha(&serde_json::to_vec(descriptor)?)
            && event.participant_slot == participant,
            "witnessed funding admission belongs to another task or participant");
        let now = f::FundingClock::current()?;
        ensure!(now.unix_ms < event.demand.deadline_unix_ms && now.monotonic_ns < event.demand.deadline_monotonic_ns,
            "witnessed funding admission expired");
        Ok(())
    }
}

pub fn funding_admission(config: &Path, request: &FundingRequest) -> Result<VerifiedFundingAdmission> {
    let started = request_monotonic_ns()?;
    let installation = InstalledFunding::load(config)?;
    funding_admission_installed(&installation, request, started)
}

pub(super) fn funding_admission_installed(installation: &InstalledFunding, request: &FundingRequest,
    started: u64) -> Result<VerifiedFundingAdmission> {
    ensure!(matches!(request, FundingRequest::Start { .. } | FundingRequest::Participant { .. }),
        "preflight/reservation is not numerical admission");
    let original = funding_request_installed(installation, request, started)?;
    Ok(VerifiedFundingAdmission { receipt:serde_json::from_value(original["receipt"].clone())? })
}
impl FundingRequest {
    fn task(&self) -> Option<(&f::SignedAuthorization, &f::SignedReservation, &Value)> {
        match self {
            Self::Register { .. } => None,
            Self::Reserve { authorization, reservation, descriptor_json }
            | Self::Preflight { authorization, reservation, descriptor_json }
            | Self::Start { authorization, reservation, descriptor_json }
            | Self::Participant { authorization, reservation, descriptor_json, .. } => Some((authorization, reservation, descriptor_json)),
        }
    }
    fn validate(&self, trust: &f::FundingTrust) -> Result<()> {
        let clock = f::FundingClock::current()?;
        if let Some((authorization, reservation, descriptor)) = self.task() {
            ensure!(descriptor.is_object() && sha(&serde_json::to_vec(descriptor)?) == reservation.body.demand.task_sha256,
                "sealed common task descriptor differs from approved signed task");
            f::verify_reservation(trust, authorization, reservation, &reservation.body.demand, clock)?;
            if descriptor["schema"] == "parallel.resident_common_task.v1" {
                crate::task_contract::verify_resident_funding_demand(descriptor, &reservation.body.demand)?;
            }
        } else if let Self::Register { authorization } = self {
            f::verify_authorization(trust, authorization, clock)?;
        }
        Ok(())
    }
}

pub(super) fn lock_with_deadline<T>(mutex: &std::sync::Mutex<T>, deadline: u64) -> Result<std::sync::MutexGuard<'_, T>> {
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
        std::thread::sleep(sleep.min(std::time::Duration::from_nanos(
            deadline.checked_sub(request_monotonic_ns()?).context("funding admission deadline expired")?,
        )));
        sleep = (sleep * 2).min(std::time::Duration::from_nanos(FUNDING_LOCK_MAX_SLEEP_NS));
    }
}

fn dispatch_gate_until(state: &Arc<State>, deadline: u64) -> Result<std::sync::MutexGuard<'_, bool>> {
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

pub(super) fn serve_request(state: &Arc<State>, stream: &mut UnixStream, peer: &ProcessIdentity,
    request_sha: &str, deadline: u64) -> Result<()> {
    remaining_request(deadline)?;
    ensure!(deadline.saturating_sub(request_monotonic_ns()?) <= 30_000_000_000,
        "funding transport window exceeds bounded admission");
    state.funding_active.fetch_update(Ordering::SeqCst, Ordering::SeqCst,
        |active| (active < state.config.max_workers.min(16)).then_some(active + 1))
        .map_err(|_| anyhow::anyhow!("funding admission slots exhausted"))?;
    let _slot = CountGuard { counter:&state.funding_active };
    let installation = state.funding_installation.as_ref().context("native funding not installed")?;
    // Both service and Authority retain the installation loaded at bind time.
    // A valid but replaced config file must not silently choose another trust.
    installation.revalidate()?;
    let input = File::from(receive_fd(stream, remaining_request(deadline)?)?);
    let bytes = read_sealed(&input, MAX_FUNDING_BYTES)?;
    ensure!(sha(&bytes) == request_sha, "funding request sealed bytes changed");
    let request: FundingRequest = serde_json::from_slice(&bytes)?;
    request.validate(installation.trust()?)?;
    if let Some((_, reservation, _)) = request.task() {
        ensure!(deadline <= reservation.body.demand.deadline_monotonic_ns, "funding transport exceeds original task deadline");
    }
    // Observer derives this independently. Client JSON never supplies the
    // execution claim, runtime digest, process birth or executable identity.
    let runtime = if matches!(request, FundingRequest::Start { .. } | FundingRequest::Participant { .. }) {
        Some(crate::runtime_custody::loaded_runtime_descriptor_until(peer, deadline)?)
    } else { None };
    remaining_request(deadline)?;
    let result = {
        let _dispatch = dispatch_gate_until(state, deadline)?;
        remaining_request(deadline)?;
        let mut authority = lock_with_deadline(&state.authority, deadline)?;
        installation.revalidate()?;
        remaining_request(deadline)?;
        ensure!(authenticated_peer(stream)? == *peer, "funding peer incarnation changed");
        match &request {
            FundingRequest::Register { authorization } => json!({"authorization_id":authority.funding_register(authorization, deadline)?}),
            FundingRequest::Reserve { authorization, reservation, .. } => json!({"reservation_id":authority.funding_reserve(authorization, reservation, deadline)?}),
            FundingRequest::Preflight { authorization, reservation, .. } => {
                authority.funding_preflight(authorization, reservation, deadline)?;
                json!({"reservation_id":reservation.body.reservation_id,"state":"reserved","numerical_start_released":false})
            }
            FundingRequest::Start { authorization, reservation, descriptor_json }
            | FundingRequest::Participant { authorization, reservation, descriptor_json, .. } => {
                let runtime = runtime.as_ref().context("funding runtime observation absent")?;
                let descriptor = serde_json::to_vec(descriptor_json)?;
                let execution = f::ExecutionClaim { process:peer.clone(),
                    executable_sha256:runtime["executable_sha256"].as_str().context("observed executable absent")?.into(),
                    runtime_sha256:runtime["runtime_sha256"].as_str().context("observed runtime absent")?.into(),
                    request_sha256:sha(&descriptor) };
                let slot = match &request { FundingRequest::Participant { participant_slot, .. } => Some(*participant_slot), _ => None };
                authority.funding_admit(authorization, reservation, &execution, slot, &descriptor, deadline)?
            }
        }
    };
    // ACK consumption survives a lost reply. No automatic restart or retry is
    // introduced on timeout, process loss or a runtime change after admission.
    if let Some(original) = runtime {
        let current = crate::runtime_custody::loaded_runtime_descriptor_until(peer, deadline)?;
        ensure!(current["runtime_sha256"] == original["runtime_sha256"]
            && current["executable_sha256"] == original["executable_sha256"], "funding peer runtime changed during admission");
    }
    remaining_request(deadline)?;
    #[cfg(test)]
    if state.config.drop_funding_reply && matches!(request, FundingRequest::Start { .. }) {
        return Ok(());
    }
    let bytes = serde_json::to_vec(&result)?;
    ensure!(bytes.len() as u64 <= MAX_FUNDING_BYTES, "funding proof exceeds sealed transport bound");
    let output = sealed(&bytes)?;
    send_json(stream, &json!({"ok":true,"funding_response_sha256":sha(&bytes)}), remaining_request(deadline)?)?;
    send_fd(stream, output.as_raw_fd(), remaining_request(deadline)?)?;
    Ok(())
}

fn connect_funding(installation: &InstalledFunding, deadline: u64) -> Result<(UnixStream, ServiceIdentity)> {
    let config = installation.revalidate()?;
    let trust = &config.trust;
    let root = fs::canonicalize(&config.authority_root)?;
    let witness = fs::canonicalize(&config.witness_path)?;
    let identity: ServiceIdentity = serde_json::from_slice(&fs::read(identity_path(&config.socket))?)?;
    let manifest = fs::read(root.join("authority.json"))?;
    ensure!(identity.schema == "parallel.fit_service.v1" && identity.authority_root == root
        && identity.witness == witness && identity.authority_instance == trust.authority_instance
        && identity.manifest_sha256 == trust.authority_manifest_sha256 && sha(&manifest) == trust.authority_manifest_sha256
        && identity.executable_sha256 == trust.authority_executable_sha256
        && verified_executable(&identity.process)? == trust.authority_executable_sha256,
        "funding service differs from independently installed pins");
    check_socket(&identity, &config.socket)?;
    let stream = connect_bounded(&config.socket, remaining_request(deadline)?)?;
    ensure!(authenticated_peer(&stream)? == identity.process, "funding service socket peer replaced");
    remaining_request(deadline)?;
    Ok((stream, identity))
}

pub fn funding_request(config: &Path, request: &FundingRequest) -> Result<Value> {
    let started = request_monotonic_ns()?;
    let installation = InstalledFunding::load(config)?;
    funding_request_installed(&installation, request, started)
}

// The public entry always loads a protected root-owned installation. Private
// unit fixtures inject an InstalledFunding into this same transport path.
pub(super) fn funding_request_installed(installation: &InstalledFunding, request: &FundingRequest,
    started: u64) -> Result<Value> {
    request.validate(installation.trust()?)?;
    match request {
        FundingRequest::Start { reservation, .. } | FundingRequest::Participant { reservation, .. } => {
            // Numerical peers are pinned by the signed approved task, while
            // the installed CPU CLI performs only control-plane operations.
            // Requiring the CPU CLI inode here would exclude every real rank.
            ensure!(verified_executable(&ProcessIdentity::current()?)? == reservation.body.demand.executable_sha256,
                "numerical funding peer differs from the signed task executable");
        }
        _ => installation.verify_client()?,
    }
    let mut deadline = started.checked_add(30_000_000_000).context("funding transport deadline overflow")?;
    if let Some((_, reservation, _)) = request.task() { deadline = deadline.min(reservation.body.demand.deadline_monotonic_ns); }
    let bytes = serde_json::to_vec(request)?;
    ensure!(bytes.len() as u64 <= MAX_FUNDING_BYTES, "funding request exceeds sealed transport bound");
    let input = sealed(&bytes)?;
    let (mut stream, identity) = connect_funding(installation, deadline)?;
    send_json(&mut stream, &Request::Funding { descriptor_sha256:sha(&bytes), deadline_monotonic_ns:deadline }, remaining_request(deadline)?)?;
    send_fd(&stream, input.as_raw_fd(), remaining_request(deadline)?)?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline)?)?;
    ensure!(response["ok"] == true, "native funding rejected: {}", response["error"]);
    let output = File::from(receive_fd(&stream, remaining_request(deadline)?)?);
    let bytes = read_sealed(&output, MAX_FUNDING_BYTES)?;
    ensure!(response["funding_response_sha256"] == sha(&bytes), "funding response sealed bytes changed");
    let result: Value = serde_json::from_slice(&bytes)?;
    installation.revalidate()?;
    ensure!(authenticated_peer(&stream)? == identity.process
        && verified_executable(&identity.process)? == identity.executable_sha256,
        "funding authority replaced before receipt verification");
    match request {
        FundingRequest::Register { authorization } => ensure!(result["authorization_id"] == f::digest(&authorization.body)?, "registered funding identity mismatch"),
        FundingRequest::Reserve { reservation, .. } => ensure!(result["reservation_id"] == reservation.body.reservation_id, "reserved funding identity mismatch"),
        FundingRequest::Preflight { reservation, .. } => ensure!(result["reservation_id"] == reservation.body.reservation_id
            && result["state"] == "reserved" && result["numerical_start_released"] == false, "invalid preflight response"),
        FundingRequest::Start { .. } | FundingRequest::Participant { .. } => verify_proof(installation, request, &result)?,
    }
    remaining_request(deadline)?;
    Ok(result)
}

fn verify_proof(installation: &InstalledFunding, request: &FundingRequest, result: &Value) -> Result<()> {
    let config = installation.revalidate()?;
    ensure!(result["schema"] == "parallel.funding_witness_proof.v1", "funding witness proof absent");
    let receipt: f::FundingReceipt = serde_json::from_value(result["receipt"].clone())?;
    let policy: witness::WitnessPolicy = serde_json::from_value(result["policy"].clone())?;
    let signed: witness::SignedRequest = serde_json::from_value(result["request"].clone())?;
    let ack: witness::SignedAck = serde_json::from_value(result["ack"].clone())?;
    let event_bytes: Vec<u8> = serde_json::from_value(result["event_bytes"].clone())?;
    ensure!(policy.sha256()? == config.witness_policy_sha256 && policy.funding_trust.as_ref() == Some(&config.trust)
        && sha(&event_bytes) == receipt.event_sha256 && f::digest(&signed)? == receipt.request_sha256
        && f::digest(&ack)? == receipt.ack_sha256 && policy.sha256()? == receipt.policy_sha256,
        "funding original witness object identities differ");
    ack.verify(&policy, &signed)?;
    ensure!(ack.body.event_sha256 == receipt.event_sha256, "funding ACK refers to another event");
    let event: Value = serde_json::from_slice(&event_bytes)?;
    ensure!(event["payload"]["original"] == serde_json::to_value(&receipt.event)?, "funding receipt differs from witnessed original");
    let (authorization, reservation, descriptor) = request.task().context("funding task absent")?;
    let slot = match request { FundingRequest::Participant { participant_slot, .. } => Some(*participant_slot), _ => None };
    ensure!(&receipt.event.authorization == authorization && &receipt.event.reservation == reservation
        && receipt.event.participant_slot == slot && receipt.event.execution.process == ProcessIdentity::current()?
        && receipt.event.execution.request_sha256 == sha(&serde_json::to_vec(descriptor)?),
        "funding receipt does not belong to this original task/participant");
    f::verify_event(&config.trust, &receipt.event, f::FundingClock::current()?)?;
    Ok(())
}
