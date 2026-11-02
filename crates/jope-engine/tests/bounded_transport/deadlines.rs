use super::*;

fn quick(ms: u64) -> Timeouts {
    let t = Duration::from_millis(ms);
    Timeouts {
        connect: t,
        handshake: t,
        io: t,
    }
}

/// Review 5339604155: a peer that sent a partial handshake record and then one byte inside
/// every per-read timeout held `poll_accept` for as long as it kept sending. The handshake
/// deadline is absolute.
#[test]
fn a_trickling_client_cannot_hold_the_server_handshake_past_its_deadline() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let deadline = Duration::from_millis(300);
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&a),
        pins(&[(&b, "host-b")]),
        quick(300),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    // No certificate and no key: a partial record, trickled.
    let peer = TcpStream::connect(addr).unwrap();
    thread::spawn(move || trickle(peer, &PARTIAL_RECORD));
    let start = Instant::now();
    assert!(accept(&acceptor).is_err(), "a trickling peer was accepted");
    ended_at_deadline(start.elapsed(), deadline, "a trickled server handshake");
    // A silent peer ends at the same deadline.
    let _silent = TcpStream::connect(addr).unwrap();
    let start = Instant::now();
    assert!(accept(&acceptor).is_err(), "a silent peer was accepted");
    ended_at_deadline(start.elapsed(), deadline, "a silent server handshake");
    // Positive control: a genuine client still completes its handshake on the same listener.
    let (cc, pin) = (credential(&b), a.sha256.clone());
    let t = thread::spawn(move || transport::connect(addr, &cc, (&pin, "host-a"), timeouts(5)));
    let mut server = accept(&acceptor).unwrap();
    let mut client = t.join().unwrap().unwrap();
    write_frame(&mut client, b"after the trickle", 1024).unwrap();
    assert_eq!(read_frame(&mut server, 1024).unwrap(), b"after the trickle");
}

/// The client loop had the same shape: a server that trickles a partial record held
/// `connect` for as long as it kept sending.
#[test]
fn a_trickling_server_cannot_hold_the_client_handshake_past_its_deadline() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let deadline = Duration::from_millis(300);
    let listener = TcpListener::bind(loopback()).unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (peer, _) = listener.accept().unwrap();
        trickle(peer, &PARTIAL_RECORD);
    });
    let start = Instant::now();
    let refused = transport::connect(addr, &credential(&b), (&a.sha256, "host-a"), quick(300));
    assert!(refused.is_err(), "a trickling server was accepted");
    ended_at_deadline(start.elapsed(), deadline, "a trickled client handshake");
    server.join().unwrap();
    // A server that accepts the connection and never answers ends at the same deadline.
    let silent = TcpListener::bind(loopback()).unwrap();
    let start = Instant::now();
    let refused = transport::connect(
        silent.local_addr().unwrap(),
        &credential(&b),
        (&a.sha256, "host-a"),
        quick(300),
    );
    assert!(refused.is_err(), "a silent server was accepted");
    ended_at_deadline(start.elapsed(), deadline, "a silent client handshake");
    // Positive control: the same client deadline against a genuine server.
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&a),
        pins(&[(&b, "host-b")]),
        timeouts(5),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    let (cc, pin) = (credential(&b), a.sha256.clone());
    let t = thread::spawn(move || transport::connect(addr, &cc, (&pin, "host-a"), quick(300)));
    let mut server = accept(&acceptor).unwrap();
    let mut client = t.join().unwrap().unwrap();
    write_frame(&mut server, b"genuine", 1024).unwrap();
    assert_eq!(read_frame(&mut client, 1024).unwrap(), b"genuine");
}

/// The fixture plan with a 300 ms join timeout.
fn quick_join(k: &Keys) -> (BoundedRunContract, Vec<TestCertificate>, Duration) {
    let (mut c, certs) = planned_hosts(k);
    c.rendezvous.join_timeout_ms = 300;
    resign(&mut c, k);
    c.validate_structure().unwrap();
    (c, certs, Duration::from_millis(300))
}
/// Open a rendezvous and gather on `acceptor` until it is refused: the refusal, and how long
/// after opening it came.
fn gather_until_refused(
    c: &BoundedRunContract,
    k: &Keys,
    acceptor: &mut dyn rendezvous::Acceptor,
) -> (rendezvous::GroupRefusal, Duration) {
    let opened = Instant::now();
    let mut rv = rendezvous::Rendezvous::open(
        c,
        &sha(&bytes(c)),
        &k.coordinator,
        "fixture-coordinator-boot",
        &rendezvous::fresh_nonce().unwrap(),
        now_ms(),
    )
    .unwrap();
    let refusal = rendezvous::gather(acceptor, &mut rv, &k.coordinator, &now_ms)
        .err()
        .expect("a group was admitted");
    (refusal, opened.elapsed())
}
fn assert_join_timeout(refusal: &rendezvous::GroupRefusal) {
    assert!(
        matches!(refusal.cause, RefusalCause::JoinTimeout { .. }),
        "{refusal}"
    );
    assert_eq!((refusal.started_ranks, refusal.committed_updates), (0, 0));
}

/// An authenticated rank that reads the challenge and then trickles its hello held the
/// synchronous gathering, and so every other rank's admission, past the join deadline.
#[test]
fn a_trickled_hello_cannot_hold_the_rendezvous_past_its_join_deadline() {
    let k = keys(3);
    let (c, certs, join) = quick_join(&k);
    let mut acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&certs[1]),
        transport::planned_certificates(&c).unwrap(),
        Timeouts::from_plan(&c),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    let (cc, pin, limit) = (
        credential(&certs[0]),
        certs[1].sha256.clone(),
        c.rendezvous.max_message_bytes,
    );
    let peer = thread::spawn(move || {
        let mut s = transport::connect(addr, &cc, (&pin, "fixture-host-1"), timeouts(5)).unwrap();
        read_frame(&mut s, limit).unwrap();
        trickle(s, &1_000u32.to_be_bytes());
    });
    let (refusal, elapsed) = gather_until_refused(&c, &k, &mut acceptor);
    assert_join_timeout(&refusal);
    ended_at_deadline(elapsed, join, "a rendezvous held by a trickled hello");
    peer.join().expect("the rank never reached its trickle");
}

/// A peer that trickles a partial handshake record reaches no hello at all; its handshake
/// ends by the join deadline, not by its own handshake allowance measured from its arrival.
#[test]
fn a_trickled_handshake_cannot_hold_the_rendezvous_past_its_join_deadline() {
    let k = keys(3);
    let (c, certs, join) = quick_join(&k);
    let mut acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&certs[1]),
        transport::planned_certificates(&c).unwrap(),
        Timeouts::from_plan(&c),
    )
    .unwrap();
    let peer = TcpStream::connect(acceptor.local_addr().unwrap()).unwrap();
    thread::spawn(move || trickle(peer, &PARTIAL_RECORD));
    let (refusal, elapsed) = gather_until_refused(&c, &k, &mut acceptor);
    assert_join_timeout(&refusal);
    ended_at_deadline(elapsed, join, "a rendezvous held by a trickled handshake");
}

/// The same-host fixture transport obeys the same join deadline.
#[test]
fn a_trickled_hello_cannot_hold_the_fixture_rendezvous_past_its_join_deadline() {
    let k = keys(3);
    let (c, _, join) = quick_join(&k);
    let (_t, root) = tempdir();
    let socket = root.join("control.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let limit = c.rendezvous.max_message_bytes;
    let peer = thread::spawn(move || {
        let mut s = std::os::unix::net::UnixStream::connect(&socket).unwrap();
        read_frame(&mut s, limit).unwrap();
        trickle(s, &1_000u32.to_be_bytes());
    });
    let (refusal, elapsed) = gather_until_refused(&c, &k, &mut rendezvous::UnixAcceptor(&listener));
    assert_join_timeout(&refusal);
    ended_at_deadline(
        elapsed,
        join,
        "a fixture rendezvous held by a trickled hello",
    );
    peer.join().expect("the rank never reached its trickle");
}

/// A joining rank's whole exchange (challenge, hello and admission) ends by its join timeout:
/// a coordinator that trickles the challenge (or, after a genuine challenge, the admission)
/// cannot hold the rank longer.
fn join_against_a_trickling_coordinator(admission: bool) {
    let k = keys(3);
    let (c, certs, join) = quick_join(&k);
    let sha = sha(&bytes(&c));
    let closure = rendezvous::closure_digest(&c.artifact_closure).unwrap();
    let limit = c.rendezvous.max_message_bytes;
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&certs[1]),
        transport::planned_certificates(&c).unwrap(),
        timeouts(5),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    let rv = rendezvous::Rendezvous::open(
        &c,
        &sha,
        &k.coordinator,
        "fixture-coordinator-boot",
        &rendezvous::fresh_nonce().unwrap(),
        now_ms(),
    )
    .unwrap();
    let challenge = serde_json::to_vec(rv.challenge()).unwrap();
    let coordinator = thread::spawn(move || {
        let mut s = accept(&acceptor).unwrap();
        if admission {
            write_frame(&mut s, &challenge, limit).unwrap();
            read_frame(&mut s, limit).unwrap();
        }
        trickle(s, &1_000u32.to_be_bytes());
    });
    let mut s = transport::connect(
        addr,
        &credential(&certs[0]),
        (&certs[1].sha256, "fixture-host-1"),
        Timeouts::from_plan(&c),
    )
    .unwrap();
    let start = Instant::now();
    let joined = rendezvous::join(
        &mut s,
        &c,
        &sha,
        &SigningKey::from_bytes(&rank_seed(0)),
        |ch| hello_body(&c, &sha, ch, 0, now_ms(), &closure),
        &now_ms,
    );
    let elapsed = start.elapsed();
    assert!(joined.is_err(), "admitted by a trickling coordinator");
    ended_at_deadline(elapsed, join, "a join held by a trickling coordinator");
    drop(s);
    coordinator
        .join()
        .expect("the coordinator never reached its trickle");
}
/// An exchange on an established stream ends by the deadline its caller set, including the
/// several socket reads or writes a TLS stream makes inside one call; per-read timeouts of
/// 5 s never fire first.
#[test]
fn an_exchange_deadline_bounds_trickled_frames_and_stalled_writes() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let deadline = Duration::from_millis(300);
    // A frame whose body trickles.
    let (mut server, client) = tls_pair(&a, &b);
    let peer = thread::spawn(move || trickle(client, &1_000u32.to_be_bytes()));
    let start = Instant::now();
    assert!(rendezvous::read_frame_by(&mut server, 1024, Instant::now() + deadline).is_err());
    ended_at_deadline(start.elapsed(), deadline, "a trickled frame");
    drop(server);
    peer.join().unwrap();
    // A deadline that has already passed refuses at once and consumes nothing: with no
    // deadline, the same stream then reads the whole frame (positive control).
    let (mut server, mut client) = tls_pair(&a, &b);
    write_frame(&mut client, b"intact", 1024).unwrap();
    thread::sleep(Duration::from_millis(50));
    let start = Instant::now();
    assert!(rendezvous::read_frame_by(&mut server, 1024, Instant::now()).is_err());
    assert!(start.elapsed() < Duration::from_millis(100));
    assert_eq!(read_frame(&mut server, 1024).unwrap(), b"intact");
    // A peer that stops reading: the write fills the socket buffers and ends at the deadline.
    let big = vec![0u8; 64 << 20];
    let start = Instant::now();
    assert!(
        rendezvous::write_frame_by(&mut server, &big, u32::MAX, Instant::now() + deadline).is_err()
    );
    ended_at_deadline(
        start.elapsed(),
        deadline,
        "a write to a peer that stopped reading",
    );
    drop(client);
}

/// Verifier note on `c9ca8b547`: a dropped stream's courtesy `close_notify` was capped at one
/// write timeout from the drop, so after an agent's reply write failed at its 30 s bound the
/// close could hold `Agent::serve` about 30 s more before it saw `stop`. The close now ends
/// within `CLOSE_TIMEOUT` of the drop, whatever the stream's write timeout.
#[test]
fn dropping_a_stream_whose_peer_stopped_reading_ends_within_the_close_timeout() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let (mut server, client) = tls_pair(&a, &b);
    // Fill the socket buffers. With a 100 ms write timeout a write fails only once a socket
    // write has made no progress for that long: the peer never reads, so the buffers are full
    // and TLS records are still queued behind them.
    let io = Some(Duration::from_secs(5));
    server
        .set_timeouts(io, Some(Duration::from_millis(100)))
        .unwrap();
    let chunk = vec![0u8; 64 << 10];
    let start = Instant::now();
    while matches!(server.write(&chunk), Ok(n) if n > 0) {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the socket buffers never filled"
        );
    }
    // Five times the close timeout, and no deadline: uncapped, the close waited all of it.
    server.set_timeouts(io, io).unwrap();
    let start = Instant::now();
    drop(server);
    ended_at_deadline(
        start.elapsed(),
        transport::CLOSE_TIMEOUT,
        "the close of a stream whose peer stopped reading",
    );
    drop(client);
}

#[test]
fn a_trickled_challenge_cannot_hold_a_joining_rank_past_its_join_timeout() {
    join_against_a_trickling_coordinator(false);
}
#[test]
fn a_trickled_admission_cannot_hold_a_joining_rank_past_its_join_timeout() {
    join_against_a_trickling_coordinator(true);
}
