use super::*;
use jope_parallel::native_loop::bounded::controller;
use jope_parallel::native_loop::bounded::rendezvous as rendezvous_api;

/// The fixture plan with each host's certificate digest set to an in-test certificate.
fn planned_hosts(k: &Keys) -> (BoundedRunContract, Vec<TestCertificate>) {
    let certs: Vec<_> = (0..3)
        .map(|i| current(&format!("fixture-host-{i}"), 21 + i as u8))
        .collect();
    let mut c = contract(now_ms(), 3, k);
    let unique = format!("w7-transport-{}", certs[0].sha256.get(..8).unwrap());
    c.run = unique;
    for (p, cert) in c.participants.iter_mut().zip(&certs) {
        p.host.certificate_sha256 = cert.sha256.clone();
    }
    resign(&mut c, k);
    c.validate_structure().unwrap();
    (c, certs)
}

#[test]
fn the_plan_pins_every_host_and_the_coordinator() {
    let k = keys(3);
    let (c, certs) = planned_hosts(&k);
    let planned = transport::planned_certificates(&c).unwrap();
    assert_eq!(planned.len(), 3);
    for (i, cert) in certs.iter().enumerate() {
        assert_eq!(planned[&cert.sha256], format!("fixture-host-{i}"));
    }
    let (sha, host) = transport::coordinator_certificate(&c).unwrap();
    assert_eq!(
        (sha.as_str(), host.as_str()),
        (certs[1].sha256.as_str(), "fixture-host-1")
    );
    let t = Timeouts::from_plan(&c);
    assert_eq!(
        t.io,
        Duration::from_millis(c.rendezvous.heartbeat_timeout_ms)
    );
    assert_eq!(t.handshake, Duration::from_secs(10));
    // Two hosts presenting one certificate cannot be told apart: refused.
    let mut shared = c.clone();
    shared.participants[2].host.certificate_sha256 = certs[0].sha256.clone();
    assert!(transport::planned_certificates(&shared).is_err());
    // A coordinator host outside the plan has no pinned certificate: refused.
    let mut outside = c.clone();
    outside.coordinator_host = "fixture-host-9".into();
    assert!(transport::coordinator_certificate(&outside).is_err());
}

/// Coordinator on a TLS acceptor presenting host 1's certificate; each rank connects with the
/// certificate of the host in `cert_of_rank`.
fn run_tls_rendezvous(cert_of_rank: [usize; 3]) -> (Result<Vec<u32>>, controller::StatusReport) {
    let now = now_ms();
    let k = keys(3);
    let (c, certs) = planned_hosts(&k);
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let sha = controller::plan(&dir, &bytes(&c), now, None)
        .unwrap()
        .plan_sha256;
    let mut co = Coordinator::open(&dir, &sha, FakeSupervisor::default(), now).unwrap();
    co.preflight(&serde_json::to_vec(&census(&c, now)).unwrap(), now)
        .unwrap();
    co.dispatch_all(now).unwrap();
    let mut rv = co
        .open_rendezvous(
            &k.coordinator,
            "fixture-coordinator-boot",
            &rendezvous_api::fresh_nonce().unwrap(),
            now,
        )
        .unwrap();
    let mut acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&certs[1]),
        transport::planned_certificates(&c).unwrap(),
        Timeouts::from_plan(&c),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    let pin = transport::coordinator_certificate(&c).unwrap();
    let closure = rendezvous_api::closure_digest(&c.artifact_closure).unwrap();
    let threads: Vec<_> = (0..3u32)
        .map(|rank| {
            let cred = credential(&certs[cert_of_rank[rank as usize]]);
            let (contract, sha, closure, pin) =
                (c.clone(), sha.clone(), closure.clone(), pin.clone());
            thread::spawn(move || -> Result<u32> {
                let mut s = transport::connect(
                    addr,
                    &cred,
                    (&pin.0, &pin.1),
                    Timeouts::from_plan(&contract),
                )?;
                rendezvous_api::join(
                    &mut s,
                    &contract,
                    &sha,
                    &SigningKey::from_bytes(&rank_seed(rank as usize)),
                    |ch| hello_body(&contract, &sha, ch, rank, now_ms(), &closure),
                    &now_ms,
                )
                .map(|m| m.global_rank())
            })
        })
        .collect();
    let joined = match rendezvous_api::gather(&mut acceptor, &mut rv, &k.coordinator, &now_ms) {
        Ok((admission, mut connected)) => {
            let persisted = co.record_admission(admission, now_ms()).unwrap();
            co.deliver_admission(&persisted, &mut connected, now_ms())
                .unwrap();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Result<Vec<_>>>()
        }
        Err(refusal) => {
            co.record_refusal(&refusal, now_ms()).unwrap();
            drop(acceptor);
            for t in threads {
                assert!(
                    t.join().unwrap().is_err(),
                    "a rank was admitted after refusal"
                );
            }
            Err(anyhow::anyhow!(
                "{}",
                serde_json::to_string(&refusal.cause).unwrap()
            ))
        }
    };
    let status = co.status().unwrap();
    (joined, status)
}

#[test]
fn the_rendezvous_admits_only_hosts_presenting_their_planned_certificates() {
    // Positive control: every rank on its own host's certificate is admitted.
    let (joined, status) = run_tls_rendezvous([0, 1, 2]);
    assert_eq!(joined.unwrap(), vec![0, 1, 2]);
    assert!(status.ranks.iter().all(|r| r.state == RankState::Running));
    // Rank 2 connecting with host 0's (pinned, valid) certificate is another host: the whole
    // group is refused with nothing started or committed.
    let (joined, status) = run_tls_rendezvous([0, 1, 0]);
    let cause = joined.unwrap_err().to_string();
    assert!(
        cause.contains("wrong_participant") && cause.contains("\"rank\":2"),
        "{cause}"
    );
    assert_eq!(status.outcome, Some(Outcome::Refused));
    assert_eq!(status.committed_updates_total, 60_000);
}

#[test]
fn a_hello_replayed_over_a_fresh_authenticated_connection_is_refused() {
    let now = now_ms();
    let k = keys(3);
    let (c, certs) = planned_hosts(&k);
    let sha = sha(&bytes(&c));
    let closure = rendezvous_api::closure_digest(&c.artifact_closure).unwrap();
    let open = || {
        rendezvous_api::Rendezvous::open(
            &c,
            &sha,
            &k.coordinator,
            "fixture-coordinator-boot",
            &rendezvous_api::fresh_nonce().unwrap(),
            now,
        )
        .unwrap()
    };
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&certs[1]),
        transport::planned_certificates(&c).unwrap(),
        timeouts(5),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    // Rank 0 sends one honest hello to the first rendezvous; the frame is captured.
    let mut first = open();
    let hello = hello_frame(
        &k,
        hello_body(&c, &sha, first.challenge(), 0, now_ms(), &closure),
    );
    let (cc, pin) = (credential(&certs[0]), certs[1].sha256.clone());
    let captured = hello.clone();
    let t = thread::spawn(move || {
        let mut s = transport::connect(addr, &cc, (&pin, "fixture-host-1"), timeouts(5)).unwrap();
        write_frame(&mut s, &captured, 64 * 1024).unwrap();
        thread::sleep(Duration::from_millis(200));
    });
    let mut s = accept(&acceptor).unwrap();
    let peer = PeerIdentity::Certificate {
        sha256: s.peer_certificate_sha256().to_string(),
    };
    let received = read_frame(&mut s, 64 * 1024).unwrap();
    assert_eq!(first.offer_from(&received, &peer, now_ms()).unwrap(), 0);
    t.join().unwrap();
    // The same bytes over a fresh, genuinely authenticated connection to a new rendezvous
    // (a coordinator restart) are stale.
    let mut second = open();
    let (cc, pin) = (credential(&certs[0]), certs[1].sha256.clone());
    let replay = hello.clone();
    let t = thread::spawn(move || {
        let mut s = transport::connect(addr, &cc, (&pin, "fixture-host-1"), timeouts(5)).unwrap();
        write_frame(&mut s, &replay, 64 * 1024).unwrap();
        thread::sleep(Duration::from_millis(200));
    });
    let mut s = accept(&acceptor).unwrap();
    let peer = PeerIdentity::Certificate {
        sha256: s.peer_certificate_sha256().to_string(),
    };
    let received = read_frame(&mut s, 64 * 1024).unwrap();
    assert_eq!(received, hello);
    assert_eq!(
        second
            .offer_from(&received, &peer, now_ms())
            .unwrap_err()
            .cause,
        RefusalCause::StaleRendezvous
    );
    t.join().unwrap();
    // A genuine hello for rank 0 offered from host 1's certificate is another host.
    let mut third = open();
    let honest = hello_frame(
        &k,
        hello_body(&c, &sha, third.challenge(), 0, now_ms(), &closure),
    );
    let host1 = PeerIdentity::Certificate {
        sha256: certs[1].sha256.clone(),
    };
    assert_eq!(
        third
            .offer_from(&honest, &host1, now_ms())
            .unwrap_err()
            .cause,
        RefusalCause::WrongParticipant { rank: 0 }
    );
}

/// W7-D4: identity is the pinned leaf certificate. A host that rotates its certificate is
/// refused by every plan that pinned the old one, whether it dials the coordinator or is the
/// coordinator; only a new plan that pins the new certificate admits it.
#[test]
fn a_rotated_certificate_needs_a_new_plan() {
    let k = keys(3);
    let (c, certs) = planned_hosts(&k);
    let rotated = current("fixture-host-0", 77);
    assert_ne!(rotated.sha256, certs[0].sha256);
    // Host 0 dials the planned coordinator (host 1) with its rotated certificate; the
    // coordinator's plan pins only the old one.
    let (refusal, _) = attempt(
        &certs[1],
        transport::planned_certificates(&c).unwrap(),
        &rotated,
        &certs[1].sha256,
    );
    assert!(
        !refusal.is_empty(),
        "a rotated certificate was accepted under the old plan"
    );
    // The coordinator rotates: every rank still pins its planned certificate and refuses the
    // new one.
    let rotated_coordinator = current("fixture-host-1", 78);
    let (_, client_refused) = attempt(
        &rotated_coordinator,
        transport::planned_certificates(&c).unwrap(),
        &certs[0],
        &certs[1].sha256,
    );
    assert!(
        client_refused,
        "a rank accepted a coordinator certificate its plan does not pin"
    );
    // A new plan pinning the rotated certificate is a different plan, and admits it.
    let mut next = c.clone();
    next.participants[0].host.certificate_sha256 = rotated.sha256.clone();
    next.validate_structure().unwrap();
    assert_ne!(sha(&bytes(&next)), sha(&bytes(&c)));
    let (refusal, _) = attempt(
        &certs[1],
        transport::planned_certificates(&next).unwrap(),
        &rotated,
        &certs[1].sha256,
    );
    assert!(
        refusal.is_empty(),
        "the new plan refused its own pinned certificate: {refusal}"
    );
    // A coordinator outside the plan has no pinned identity at all.
    let mut outside = next.clone();
    outside.coordinator_host = "fixture-host-9".into();
    assert!(transport::coordinator_certificate(&outside).is_err());
}

// ------------------------------------------- absolute deadlines against peers that trickle
