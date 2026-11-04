use super::*;

fn connect_funding(
    installation: &InstalledFunding,
    deadline: u64,
) -> Result<(UnixStream, ServiceIdentity)> {
    let config = installation.revalidate()?;
    let trust = &config.trust;
    let root = fs::canonicalize(&config.authority_root)?;
    let witness = fs::canonicalize(&config.witness_path)?;
    let identity: ServiceIdentity =
        serde_json::from_slice(&fs::read(identity_path(&config.socket))?)?;
    let manifest = fs::read(root.join("authority.json"))?;
    ensure!(
        identity.schema == "parallel.fit_service.v1"
            && identity.authority_root == root
            && identity.witness == witness
            && identity.authority_instance == trust.authority_instance
            && identity.manifest_sha256 == trust.authority_manifest_sha256
            && sha(&manifest) == trust.authority_manifest_sha256
            && identity.executable_sha256 == trust.authority_executable_sha256
            && verified_executable(&identity.process)? == trust.authority_executable_sha256,
        "funding service differs from independently installed pins"
    );
    check_socket(&identity, &config.socket)?;
    let stream = connect_bounded(&config.socket, remaining_request(deadline)?)?;
    ensure!(
        authenticated_peer(&stream)? == identity.process,
        "funding service socket peer replaced"
    );
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
pub(crate) fn funding_request_installed(
    installation: &InstalledFunding,
    request: &FundingRequest,
    started: u64,
) -> Result<Value> {
    request.validate(installation.trust()?)?;
    match request {
        FundingRequest::Start { reservation, .. }
        | FundingRequest::Participant { reservation, .. } => {
            // Numerical peers are pinned by the signed approved task, while
            // the installed CPU CLI performs only control-plane operations.
            // Requiring the CPU CLI inode here would exclude every real rank.
            ensure!(
                verified_executable(&ProcessIdentity::current()?)?
                    == reservation.body.demand.executable_sha256,
                "numerical funding peer differs from the signed task executable"
            );
        }
        _ => installation.verify_client()?,
    }
    let mut deadline = started
        .checked_add(30_000_000_000)
        .context("funding transport deadline overflow")?;
    if let Some((_, reservation, _)) = request.task() {
        deadline = deadline.min(reservation.body.demand.deadline_monotonic_ns);
    }
    let bytes = serde_json::to_vec(request)?;
    ensure!(
        bytes.len() as u64 <= MAX_FUNDING_BYTES,
        "funding request exceeds sealed transport bound"
    );
    let input = sealed(&bytes)?;
    let (mut stream, identity) = connect_funding(installation, deadline)?;
    send_json(
        &mut stream,
        &Request::Funding {
            descriptor_sha256: sha(&bytes),
            deadline_monotonic_ns: deadline,
        },
        remaining_request(deadline)?,
    )?;
    send_fd(&stream, input.as_raw_fd(), remaining_request(deadline)?)?;
    let response: Value = receive_json(&mut stream, remaining_request(deadline)?)?;
    ensure!(
        response["ok"] == true,
        "native funding rejected: {}",
        response["error"]
    );
    let output = File::from(receive_fd(&stream, remaining_request(deadline)?)?);
    let bytes = read_sealed(&output, MAX_FUNDING_BYTES)?;
    ensure!(
        response["funding_response_sha256"] == sha(&bytes),
        "funding response sealed bytes changed"
    );
    let result: Value = serde_json::from_slice(&bytes)?;
    installation.revalidate()?;
    ensure!(
        authenticated_peer(&stream)? == identity.process
            && verified_executable(&identity.process)? == identity.executable_sha256,
        "funding authority replaced before receipt verification"
    );
    match request {
        FundingRequest::Register { authorization } => ensure!(
            result["authorization_id"] == f::digest(&authorization.body)?,
            "registered funding identity mismatch"
        ),
        FundingRequest::Reserve { reservation, .. } => ensure!(
            result["reservation_id"] == reservation.body.reservation_id,
            "reserved funding identity mismatch"
        ),
        FundingRequest::Preflight { reservation, .. } => ensure!(
            result["reservation_id"] == reservation.body.reservation_id
                && result["state"] == "reserved"
                && result["numerical_start_released"] == false,
            "invalid preflight response"
        ),
        FundingRequest::Start { .. } | FundingRequest::Participant { .. } => {
            verify_proof(installation, request, &result)?
        }
    }
    remaining_request(deadline)?;
    Ok(result)
}

fn verify_proof(
    installation: &InstalledFunding,
    request: &FundingRequest,
    result: &Value,
) -> Result<()> {
    let config = installation.revalidate()?;
    ensure!(
        result["schema"] == "parallel.funding_witness_proof.v1",
        "funding witness proof absent"
    );
    let receipt: f::FundingReceipt = serde_json::from_value(result["receipt"].clone())?;
    let policy: witness::WitnessPolicy = serde_json::from_value(result["policy"].clone())?;
    let signed: witness::SignedRequest = serde_json::from_value(result["request"].clone())?;
    let ack: witness::SignedAck = serde_json::from_value(result["ack"].clone())?;
    let event_bytes: Vec<u8> = serde_json::from_value(result["event_bytes"].clone())?;
    ensure!(
        policy.sha256()? == config.witness_policy_sha256
            && policy.funding_trust.as_ref() == Some(&config.trust)
            && sha(&event_bytes) == receipt.event_sha256
            && f::digest(&signed)? == receipt.request_sha256
            && f::digest(&ack)? == receipt.ack_sha256
            && policy.sha256()? == receipt.policy_sha256,
        "funding original witness object identities differ"
    );
    ack.verify(&policy, &signed)?;
    ensure!(
        ack.body.event_sha256 == receipt.event_sha256,
        "funding ACK refers to another event"
    );
    let event: Value = serde_json::from_slice(&event_bytes)?;
    ensure!(
        event["payload"]["original"] == serde_json::to_value(&receipt.event)?,
        "funding receipt differs from witnessed original"
    );
    let (authorization, reservation, descriptor) = request.task().context("funding task absent")?;
    let slot = match request {
        FundingRequest::Participant {
            participant_slot, ..
        } => Some(*participant_slot),
        _ => None,
    };
    ensure!(
        &receipt.event.authorization == authorization
            && &receipt.event.reservation == reservation
            && receipt.event.participant_slot == slot
            && receipt.event.execution.process == ProcessIdentity::current()?
            && receipt.event.execution.request_sha256 == sha(&serde_json::to_vec(descriptor)?),
        "funding receipt does not belong to this original task/participant"
    );
    f::verify_event(&config.trust, &receipt.event, f::FundingClock::current()?)?;
    Ok(())
}
