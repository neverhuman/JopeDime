use super::*;

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
