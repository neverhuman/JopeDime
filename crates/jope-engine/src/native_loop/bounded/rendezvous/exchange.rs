use super::*;

/// A connected, authenticated participant control stream.
pub struct Connected {
    pub global_rank: u32,
    pub stream: Box<dyn ControlStream>,
}
/// Accept and authenticate participants from `acceptor` until the group is admitted or
/// refused. Every handshake, challenge and hello ends by the join deadline itself (an
/// absolute deadline over all of their reads and writes, however a peer paces its bytes) and
/// is bounded by the frozen message limit, and each hello is bound to its peer's channel
/// identity. Peers are served one at a time, so a slow peer delays the others, but never past
/// the join deadline. Admission is returned *undelivered*: only the controller delivers it,
/// after journaling.
pub fn gather(
    acceptor: &mut dyn Acceptor,
    rendezvous: &mut Rendezvous,
    coordinator: &SigningKey,
    now: &dyn Fn() -> u64,
) -> Result<(GroupAdmission, Vec<Connected>), GroupRefusal> {
    let limit = rendezvous.contract().rendezvous.max_message_bytes;
    let challenge =
        serde_json::to_vec(rendezvous.challenge()).expect("rendezvous challenge is serializable");
    let mut connected = Vec::new();
    loop {
        if let Some(admission) = rendezvous.admit(coordinator, now())? {
            return Ok((admission, connected));
        }
        // The join deadline on this host's monotonic clock.
        let join_deadline = Instant::now()
            + Duration::from_millis(
                rendezvous
                    .challenge()
                    .body
                    .join_deadline_unix_ms
                    .saturating_sub(now()),
            );
        match acceptor.next_incoming(join_deadline) {
            Incoming::Stream(mut stream, peer) => {
                let remaining = join_deadline
                    .saturating_duration_since(Instant::now())
                    .max(Duration::from_millis(1));
                let _ = stream.set_timeouts(Some(remaining), Some(remaining));
                if stream.set_deadline(Some(join_deadline)).is_err()
                    || write_frame(&mut stream, &challenge, limit).is_err()
                {
                    continue; // the peer went away before authenticating; nothing trusted
                }
                let frame = read_frame(&mut stream, limit);
                // Admission delivery and heartbeats set their own deadlines.
                if stream.set_deadline(None).is_err() {
                    continue;
                }
                let frame = match frame {
                    Ok(f) => f,
                    Err(e) => {
                        let cause = if e.to_string().contains("exceeds message limit") {
                            RefusalCause::Oversized {
                                bytes: u64::from(limit) + 1,
                                limit: u64::from(limit),
                            }
                        } else {
                            // An unauthenticated peer that timed out or closed early
                            // proves nothing; the join deadline still governs the group.
                            continue;
                        };
                        return Err(rendezvous.refuse(cause));
                    }
                };
                let rank = rendezvous.offer_from(&frame, &peer, now())?;
                connected.retain(|c: &Connected| c.global_rank != rank);
                connected.push(Connected {
                    global_rank: rank,
                    stream,
                });
            }
            Incoming::Pending | Incoming::Unauthenticated(_) => {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
/// [`gather`] on a same-host Unix socket (fixture transport).
pub fn gather_unix(
    listener: &UnixListener,
    rendezvous: &mut Rendezvous,
    coordinator: &SigningKey,
    now: &dyn Fn() -> u64,
) -> Result<(GroupAdmission, Vec<Connected>), GroupRefusal> {
    let _ = listener.set_nonblocking(true);
    gather(&mut UnixAcceptor(listener), rendezvous, coordinator, now)
}
/// Participant side on a same-host Unix socket (fixture transport); see [`join`].
pub fn join_unix(
    stream: &mut UnixStream,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    key: &SigningKey,
    hello: impl FnOnce(&Challenge) -> HelloBody,
    now: &dyn Fn() -> u64,
) -> Result<AdmittedMember> {
    join(
        &mut DeadlineSocket::new(&*stream),
        contract,
        plan_sha256,
        key,
        hello,
        now,
    )
}
/// Participant side: read and verify the challenge, send the signed hello built by `hello`,
/// then verify the full-group admission. The whole exchange ends by one absolute deadline,
/// the join timeout (cut to the original deadline) from now, however the coordinator paces
/// its bytes.
pub fn join(
    stream: &mut dyn ControlStream,
    contract: &BoundedRunContract,
    plan_sha256: &str,
    key: &SigningKey,
    hello: impl FnOnce(&Challenge) -> HelloBody,
    now: &dyn Fn() -> u64,
) -> Result<AdmittedMember> {
    let limit = contract.rendezvous.max_message_bytes;
    let remaining = Duration::from_millis(
        contract
            .rendezvous
            .join_timeout_ms
            .min(contract.original_deadline_unix_ms.saturating_sub(now()))
            .max(1),
    );
    stream.set_timeouts(Some(remaining), None)?;
    by_deadline(stream, Instant::now() + remaining, |stream| {
        let frame = read_frame(stream, limit)?;
        let challenge = verify_challenge(contract, plan_sha256, &frame, now())
            .map_err(|c| anyhow::anyhow!("challenge refused: {c:?}"))?;
        let signed = sign_hello(key, hello(&challenge))?;
        write_frame(stream, &serde_json::to_vec(&signed)?, limit)?;
        let frame = read_frame(stream, limit)?;
        verify_admission(contract, plan_sha256, &signed, &frame, now())
            .map_err(|c| anyhow::anyhow!("admission refused: {c:?}"))
    })
}
