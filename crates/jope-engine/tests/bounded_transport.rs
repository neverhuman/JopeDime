//! W7-T04/T07 cross-host transport fixtures on loopback TCP: TLS 1.3 with mutual pinned
//! certificate identity, bounded frames, deadlines and delivery ids, and the authenticated
//! rendezvous over it. Certificates are generated in-test (self-signed Ed25519 built from DER,
//! `support/tls_fixture.rs`). Loopback on one host is a fixture, not the three-host proof.
#[path = "support/bounded_fixture.rs"]
mod fixture;
#[path = "support/tls_fixture.rs"]
mod tls;
use anyhow::Result;
use ed25519_dalek::SigningKey;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    contract::BoundedRunContract,
    controller::{self, Coordinator, Outcome, RankState},
    rendezvous::{self, read_frame, write_frame, PeerIdentity, RefusalCause},
    transport::{self, HostCredential, Sequenced, Timeouts, TlsAcceptor, TlsStream},
};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    os::unix::fs::PermissionsExt,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tls::*;

fn credential(c: &TestCertificate) -> HostCredential {
    HostCredential::from_der(c.der.clone(), c.pkcs8.clone()).unwrap()
}
fn timeouts(seconds: u64) -> Timeouts {
    let t = Duration::from_secs(seconds);
    Timeouts {
        connect: t,
        handshake: t,
        io: t,
    }
}
fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}
fn pins(entries: &[(&TestCertificate, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(c, host)| (c.sha256.clone(), host.to_string()))
        .collect()
}
/// Wait for the next connection; an error when its handshake failed.
fn accept(acceptor: &TlsAcceptor) -> Result<TlsStream> {
    let start = Instant::now();
    loop {
        if let Some(s) = acceptor.poll_accept()? {
            return Ok(s);
        }
        anyhow::ensure!(start.elapsed() < Duration::from_secs(20), "no connection");
        thread::sleep(Duration::from_millis(5));
    }
}
/// A connected, mutually authenticated pair: `(server side, client side)`.
fn tls_pair(server: &TestCertificate, client: &TestCertificate) -> (TlsStream, TlsStream) {
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(server),
        pins(&[(client, "host-b")]),
        timeouts(5),
    )
    .unwrap();
    let addr = acceptor.local_addr().unwrap();
    let (cc, pin) = (credential(client), server.sha256.clone());
    let t = thread::spawn(move || transport::connect(addr, &cc, (&pin, "host-a"), timeouts(5)));
    let s = accept(&acceptor).unwrap();
    (s, t.join().unwrap().unwrap())
}
/// One connection attempt: `(the server's refusal, lowercased or empty, client refused)`.
fn attempt(
    server: &TestCertificate,
    server_pins: BTreeMap<String, String>,
    client: &TestCertificate,
    client_pin: &str,
) -> (String, bool) {
    let acceptor =
        TlsAcceptor::bind(loopback(), &credential(server), server_pins, timeouts(5)).unwrap();
    let addr = acceptor.local_addr().unwrap();
    let (cc, pin) = (credential(client), client_pin.to_string());
    let t =
        thread::spawn(
            move || match transport::connect(addr, &cc, (&pin, "host-a"), timeouts(5)) {
                Err(_) => true,
                // TLS 1.3 client authentication completes on the client first; the server's
                // refusal then arrives as an alert, so nothing can be read.
                Ok(mut s) => read_frame(&mut s, 1024).is_err(),
            },
        );
    // An accepted stream is closed at once, so an accepted client reads end-of-stream.
    let refusal = match accept(&acceptor) {
        Ok(stream) => {
            drop(stream);
            String::new()
        }
        Err(e) => format!("{e:#}").to_lowercase(),
    };
    (refusal, t.join().unwrap())
}
fn frame(id: u64, body: &[u8]) -> Vec<u8> {
    let mut f = id.to_be_bytes().to_vec();
    f.extend_from_slice(body);
    f
}

#[test]
fn mutually_pinned_tls_carries_bounded_frames() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let (mut server, mut client) = tls_pair(&a, &b);
    assert_eq!(
        (server.peer(), server.peer_certificate_sha256()),
        ("host-b", b.sha256.as_str())
    );
    assert_eq!(
        (client.peer(), client.peer_certificate_sha256()),
        ("host-a", a.sha256.as_str())
    );
    write_frame(&mut client, b"hello over tls", 1024).unwrap();
    assert_eq!(read_frame(&mut server, 1024).unwrap(), b"hello over tls");
    write_frame(&mut server, b"reply", 1024).unwrap();
    assert_eq!(read_frame(&mut client, 1024).unwrap(), b"reply");
    // Credentials load from PEM files; the private key must be owner-only and no symlink.
    let (_t, root) = tempdir();
    let (cert, key) = (root.join("host.pem"), root.join("host.key"));
    std::fs::write(&cert, pem("CERTIFICATE", &a.der)).unwrap();
    std::fs::write(&key, pem("PRIVATE KEY", &a.pkcs8)).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let loaded = HostCredential::load(&cert, &key).unwrap();
    assert_eq!(loaded.certificate_sha256(), a.sha256);
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o640)).unwrap();
    let e = HostCredential::load(&cert, &key).err().unwrap();
    assert!(e.to_string().contains("owner only"), "{e}");
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let link = root.join("link.key");
    std::os::unix::fs::symlink(&key, &link).unwrap();
    assert!(HostCredential::load(&cert, &link).is_err());
    assert!(HostCredential::from_der(a.der.clone(), b"not a key".to_vec()).is_err());
    assert!(HostCredential::from_der(b"not a certificate".to_vec(), a.pkcs8.clone()).is_err());
}

#[test]
fn a_wrong_certificate_is_refused_in_both_directions() {
    let (a, b, stranger) = (
        current("host-a", 11),
        current("host-b", 12),
        current("host-c", 13),
    );
    // The client presents a certificate the server has not pinned.
    let (refusal, client) = attempt(&a, pins(&[(&b, "host-b")]), &stranger, &a.sha256);
    assert!(!refusal.is_empty(), "an unpinned client was accepted");
    assert!(client);
    // The server presents a certificate the client has not pinned.
    let (refusal, client) = attempt(&stranger, pins(&[(&b, "host-b")]), &b, &a.sha256);
    assert!(!refusal.is_empty());
    assert!(client, "an unpinned server was accepted");
    // Positive control with the same certificates correctly pinned.
    let (refusal, client) = attempt(&a, pins(&[(&b, "host-b")]), &b, &a.sha256);
    assert_eq!(refusal, "");
    assert!(
        client,
        "the accepted client read no frame, as none was sent"
    );
}

#[test]
fn expired_and_not_yet_valid_certificates_are_refused() {
    let now = unix_now();
    let a = current("host-a", 11);
    let expired = certificate("host-b", 12, now - 7_200, now - 3_600);
    let future = certificate("host-b", 14, now + 3_600, now + 7_200);
    for (stale, word) in [(&expired, "expired"), (&future, "notvalidyet")] {
        // A stale client certificate, although pinned.
        let (refusal, client) = attempt(&a, pins(&[(stale, "host-b")]), stale, &a.sha256);
        assert!(refusal.contains(word), "{word}: {refusal}");
        assert!(client);
        // A stale server certificate, although pinned.
        let b = current("host-b", 15);
        let (refusal, client) = attempt(stale, pins(&[(&b, "host-b")]), &b, &stale.sha256);
        assert!(!refusal.is_empty());
        assert!(client, "a {word} server certificate was accepted");
    }
}

#[test]
fn oversize_frames_are_refused_before_their_body_is_read() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let (mut server, mut client) = tls_pair(&a, &b);
    // Refused locally: nothing is sent.
    assert!(write_frame(&mut client, &[7u8; 1025], 1024).is_err());
    // A header claiming more than the limit is refused without waiting for any body.
    client.write_all(&1025u32.to_be_bytes()).unwrap();
    client.flush().unwrap();
    let start = Instant::now();
    let e = read_frame(&mut server, 1024).unwrap_err().to_string();
    assert!(e.contains("exceeds message limit"), "{e}");
    assert!(start.elapsed() < Duration::from_secs(2));
    // The sequenced channel applies the same bound to the whole frame, id included.
    let (server, client) = tls_pair(&a, &b);
    let mut tx = Sequenced::new(client, 1024);
    assert!(tx.send(&[0u8; 1017]).is_err());
    assert_eq!(tx.send(&[0u8; 1016]).unwrap(), 1);
    let mut rx = Sequenced::new(server, 1024);
    assert_eq!(rx.receive().unwrap().0, 1);
}

#[test]
fn silent_peers_time_out() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&a),
        pins(&[(&b, "host-b")]),
        timeouts(1),
    )
    .unwrap();
    // A TCP peer that never starts the handshake.
    let _silent = TcpStream::connect(acceptor.local_addr().unwrap()).unwrap();
    let start = Instant::now();
    assert!(accept(&acceptor).is_err());
    assert!(
        start.elapsed() < Duration::from_secs(4),
        "{:?}",
        start.elapsed()
    );
    // An authenticated peer that sends nothing: reads end at the I/O deadline.
    let addr = acceptor.local_addr().unwrap();
    let (cc, pin) = (credential(&b), a.sha256.clone());
    let t = thread::spawn(move || transport::connect(addr, &cc, (&pin, "host-a"), timeouts(1)));
    let mut server = accept(&acceptor).unwrap();
    let _client = t.join().unwrap().unwrap();
    let start = Instant::now();
    assert!(read_frame(&mut server, 1024).is_err());
    assert!(
        start.elapsed() < Duration::from_secs(4),
        "{:?}",
        start.elapsed()
    );
    // A connect to a closed port fails within the connect deadline.
    let closed = TcpListener::bind(loopback()).unwrap().local_addr().unwrap();
    assert!(
        transport::connect(closed, &credential(&b), (&a.sha256, "host-a"), timeouts(1)).is_err()
    );
}

/// A TCP relay that records every byte the client sent.
fn recording_relay(target: SocketAddr) -> (SocketAddr, Arc<Mutex<Vec<u8>>>) {
    let listener = TcpListener::bind(loopback()).unwrap();
    let addr = listener.local_addr().unwrap();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&recorded);
    thread::spawn(move || {
        let (client, _) = listener.accept().unwrap();
        let server = TcpStream::connect(target).unwrap();
        let (mut c_read, mut s_write) = (client.try_clone().unwrap(), server.try_clone().unwrap());
        let up = thread::spawn(move || {
            let mut buf = [0u8; 4096];
            while let Ok(n) = c_read.read(&mut buf) {
                if n == 0 || s_write.write_all(&buf[..n]).is_err() {
                    break;
                }
                sink.lock().unwrap().extend_from_slice(&buf[..n]);
            }
        });
        let (mut s_read, mut c_write) = (server, client);
        let mut buf = [0u8; 4096];
        while let Ok(n) = s_read.read(&mut buf) {
            if n == 0 || c_write.write_all(&buf[..n]).is_err() {
                break;
            }
        }
        let _ = up.join();
    });
    (addr, recorded)
}

#[test]
fn replayed_bytes_and_delivery_ids_are_refused() {
    let (a, b) = (current("host-a", 11), current("host-b", 12));
    let acceptor = TlsAcceptor::bind(
        loopback(),
        &credential(&a),
        pins(&[(&b, "host-b")]),
        timeouts(2),
    )
    .unwrap();
    let (relay, recorded) = recording_relay(acceptor.local_addr().unwrap());
    let (cc, pin) = (credential(&b), a.sha256.clone());
    let t = thread::spawn(move || {
        let mut s = transport::connect(relay, &cc, (&pin, "host-a"), timeouts(2)).unwrap();
        write_frame(&mut s, b"one authenticated frame", 1024).unwrap();
        s
    });
    let mut server = accept(&acceptor).unwrap();
    assert_eq!(
        read_frame(&mut server, 1024).unwrap(),
        b"one authenticated frame"
    );
    drop(t.join().unwrap());
    drop(server);
    thread::sleep(Duration::from_millis(100));
    // Replaying the recorded client bytes on a new connection cannot complete a handshake:
    // the server's fresh key share makes the recorded records useless.
    let replay = recorded.lock().unwrap().clone();
    assert!(!replay.is_empty());
    let mut raw = TcpStream::connect(acceptor.local_addr().unwrap()).unwrap();
    raw.write_all(&replay).unwrap();
    assert!(
        accept(&acceptor).is_err(),
        "a replayed session was accepted"
    );

    // Delivery ids on an authenticated stream: only the next consecutive id is accepted.
    let (server, mut client) = tls_pair(&a, &b);
    let mut rx = Sequenced::new(server, 1024);
    for f in [
        frame(1, b"first"),
        frame(2, b"second"),
        frame(2, b"again"),
        frame(3, b"third"),
    ] {
        write_frame(&mut client, &f, 1024).unwrap();
    }
    assert_eq!(rx.receive().unwrap(), (1, b"first".to_vec()));
    assert_eq!(rx.receive().unwrap(), (2, b"second".to_vec()));
    let e = rx.receive().unwrap_err().to_string();
    assert!(e.contains("replayed, duplicated or reordered"), "{e}");
    assert!(rx.receive().is_err(), "a refused channel stays refused");
    // A skipped id, and a frame too short to carry one.
    let (server, mut client) = tls_pair(&a, &b);
    let mut rx = Sequenced::new(server, 1024);
    write_frame(&mut client, &frame(2, b"skipped"), 1024).unwrap();
    assert!(rx.receive().is_err());
    let (server, mut client) = tls_pair(&a, &b);
    let mut rx = Sequenced::new(server, 1024);
    write_frame(&mut client, b"short", 1024).unwrap();
    assert!(rx.receive().is_err());
}

// ---------------------------------------------------------- the rendezvous over the transport

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
            &rendezvous::fresh_nonce().unwrap(),
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
    let closure = rendezvous::closure_digest(&c.artifact_closure).unwrap();
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
                rendezvous::join(
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
    let joined = match rendezvous::gather(&mut acceptor, &mut rv, &k.coordinator, &now_ms) {
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
    let closure = rendezvous::closure_digest(&c.artifact_closure).unwrap();
    let open = || {
        rendezvous::Rendezvous::open(
            &c,
            &sha,
            &k.coordinator,
            "fixture-coordinator-boot",
            &rendezvous::fresh_nonce().unwrap(),
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
