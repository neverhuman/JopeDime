//! Authenticated cross-host transport for the bounded run (W7-T04/T07).
//!
//! TCP with TLS 1.3 (rustls with the aws-lc-rs provider, always named explicitly, never
//! inferred from crate features) and **mutual pinned certificate identity**. Each side presents
//! its host certificate and accepts the peer only if:
//! - the SHA-256 of the peer's end-entity certificate is the certificate digest the frozen plan
//!   records for that host (`HostIdentity::certificate_sha256`, collected by the census);
//! - the certificate is inside its validity window;
//! - the TLS handshake signature verifies with that certificate's key.
//!
//! There is no certificate authority and no name-based trust: the reviewed plan, built from
//! the census, is the only trust root. A peer that fails any check never delivers a byte to
//! the rendezvous.
//!
//! Every frame is length-prefixed and bounded before its body is read ([`read_frame`]), and
//! [`Sequenced`] carries strictly consecutive delivery ids for request/response protocols.
//! Rendezvous messages keep their own ed25519 signatures, rendezvous binding and delivery
//! ids, so an application message replayed over a fresh authenticated connection is still
//! refused by the rendezvous.
//!
//! **Deadlines are absolute.** A socket timeout bounds one read or write, and a peer that
//! sends one byte inside every timeout never trips it, while a single TLS call (a handshake
//! step, a record read, a flush) can make many socket reads and writes. So every socket
//! here is a [`DeadlineSocket`]: before each read or write its kernel timeout is set to the
//! smaller of the per-operation timeout ([`Timeouts`]) and the time left before an absolute
//! deadline, and at the deadline the operation is refused without touching the socket. A
//! handshake has one deadline for all of its I/O; callers put one on each exchange
//! ([`ControlStream::set_deadline`], `rendezvous::read_frame_by`, [`Sequenced::receive_by`]).
//! A peer, authenticated or not, therefore cannot hold a caller past the allowance it was
//! given, however it paces its bytes, and nothing here extends an allowance.
use super::{
    collect::{certificate_der, der_element},
    contract::BoundedRunContract,
    rendezvous::{read_frame, write_frame, Acceptor, ControlStream, Incoming, PeerIdentity},
};
use crate::manifest::hash;
use anyhow::{anyhow, bail, ensure, Context, Result};
use rustls::{
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{aws_lc_rs, verify_tls12_signature, verify_tls13_signature, CryptoProvider},
    pki_types::{
        pem::PemObject, CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime,
    },
    server::danger::{ClientCertVerified, ClientCertVerifier},
    CertificateError, ClientConfig, ClientConnection, DigitallySignedStruct, DistinguishedName,
    ServerConfig, ServerConnection, SignatureScheme, StreamOwned,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt},
        net::UnixStream,
    },
    path::Path,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

const MAX_CREDENTIAL_BYTES: u64 = 64 * 1024;
mod certificate;
mod sequenced;
mod socket;
mod tls;

pub use certificate::{certificate_validity, HostCredential};
pub(super) use certificate::{provider, Pinned};
pub use sequenced::Sequenced;
pub(super) use socket::SERVER_NAME;
pub use socket::{DeadlineSocket, TimedSocket, CLOSE_TIMEOUT};
pub use tls::{
    connect, coordinator_certificate, planned_certificates, Timeouts, TlsAcceptor, TlsStream,
};
