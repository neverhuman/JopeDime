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
    rendezvous::{self, read_frame, write_frame, ControlStream, PeerIdentity, RefusalCause},
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

#[path = "bounded_transport/connection.rs"]
mod connection;
#[path = "bounded_transport/deadlines.rs"]
mod deadlines;
#[path = "bounded_transport/rendezvous.rs"]
mod rendezvous_tests;
