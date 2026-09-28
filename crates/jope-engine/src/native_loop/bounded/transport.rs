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
/// The TLS server name a client sends; identity is the pinned certificate, never this name.
const SERVER_NAME: &str = "jope-bounded-peer";
/// Bound on the courtesy `close_notify` of a dropped [`TlsStream`], whatever its write timeout.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// A socket whose kernel timeouts bound one read or one write at a time.
pub trait TimedSocket: Read + Write {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()>;
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()>;
}
impl TimedSocket for TcpStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}
impl TimedSocket for UnixStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}
impl TimedSocket for &UnixStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Limits {
    read: Option<Duration>,
    write: Option<Duration>,
    deadline: Option<Instant>,
}
/// A socket whose every blocking read and write is bounded twice: by a per-operation timeout
/// (inactivity) and by an optional absolute deadline. Before each operation the kernel
/// timeout is set to the smaller of the two, so no single read or write can outlast the
/// deadline; at or after the deadline an operation is refused with `TimedOut` and the socket
/// is not touched. The deadline covers every operation a caller makes through this socket,
/// including each of the many a TLS connection can make inside one call.
pub struct DeadlineSocket<S> {
    socket: S,
    limits: Mutex<Limits>,
}
impl<S: TimedSocket> DeadlineSocket<S> {
    /// No per-operation timeout and no deadline until [`ControlStream::set_timeouts`] and
    /// [`ControlStream::set_deadline`] set them.
    pub fn new(socket: S) -> Self {
        Self {
            socket,
            limits: Mutex::new(Limits::default()),
        }
    }
    pub fn get_ref(&self) -> &S {
        &self.socket
    }
    fn limits(&self) -> MutexGuard<'_, Limits> {
        self.limits.lock().unwrap_or_else(PoisonError::into_inner)
    }
    /// Bring the deadline forward to `cap` if that is sooner; it is never moved later.
    fn cap_deadline(&self, cap: Instant) {
        let mut limits = self.limits();
        limits.deadline = Some(limits.deadline.map_or(cap, |d| d.min(cap)));
    }
    /// The kernel timeout for the next read (`write == false`) or write: its per-operation
    /// timeout, cut to what is left before the deadline.
    fn next_timeout(&self, write: bool) -> std::io::Result<Option<Duration>> {
        let limits = *self.limits();
        let per_operation = if write { limits.write } else { limits.read };
        let Some(deadline) = limits.deadline else {
            return Ok(per_operation);
        };
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "the I/O deadline has passed",
            ));
        }
        Ok(Some(per_operation.map_or(left, |t| t.min(left))))
    }
}
impl<S: TimedSocket> Read for DeadlineSocket<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let timeout = self.next_timeout(false)?;
        self.socket.apply_read_timeout(timeout)?;
        self.socket.read(buf)
    }
}
impl<S: TimedSocket> Write for DeadlineSocket<S> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let timeout = self.next_timeout(true)?;
        self.socket.apply_write_timeout(timeout)?;
        self.socket.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.socket.flush()
    }
}
impl<S: TimedSocket + Send> ControlStream for DeadlineSocket<S> {
    fn set_timeouts(&self, read: Option<Duration>, write: Option<Duration>) -> std::io::Result<()> {
        if [read, write].into_iter().flatten().any(|t| t.is_zero()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot set a 0 duration timeout",
            ));
        }
        let mut limits = self.limits();
        (limits.read, limits.write) = (read, write);
        Ok(())
    }
    fn set_deadline(&self, deadline: Option<Instant>) -> std::io::Result<()> {
        self.limits().deadline = deadline;
        Ok(())
    }
}
type Socket = DeadlineSocket<TcpStream>;

fn provider() -> Arc<CryptoProvider> {
    Arc::new(aws_lc_rs::default_provider())
}

/// This host's certificate and PKCS#8 private key.
pub struct HostCredential {
    certificate: CertificateDer<'static>,
    key: PrivateKeyDer<'static>,
    certificate_sha256: String,
}
impl HostCredential {
    /// A certificate (DER, or PEM with one certificate) and a DER PKCS#8 private key.
    pub fn from_der(certificate: Vec<u8>, pkcs8_key: Vec<u8>) -> Result<Self> {
        let certificate = certificate_der(&certificate)?;
        certificate_validity(&certificate)?;
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8_key));
        provider()
            .key_provider
            .load_private_key(key.clone_key())
            .map_err(|e| anyhow!("unsupported private key: {e}"))?;
        Ok(Self {
            certificate_sha256: hash(&certificate),
            certificate: CertificateDer::from(certificate),
            key,
        })
    }
    /// Load the certificate (DER, or PEM with one certificate) and the PKCS#8 private key
    /// (PEM `PRIVATE KEY` or DER). Neither may be a symlink; the key must be readable by its
    /// owner only.
    pub fn load(certificate: &Path, key: &Path) -> Result<Self> {
        let certificate = read_credential(certificate, false)?;
        let key = read_credential(key, true)?;
        let key = if key.starts_with(b"-----BEGIN") {
            match PrivateKeyDer::from_pem_slice(&key)
                .map_err(|e| anyhow!("private key PEM: {e:?}"))?
            {
                PrivateKeyDer::Pkcs8(k) => k.secret_pkcs8_der().to_vec(),
                _ => bail!("the private key must be PKCS#8 (`PRIVATE KEY`)"),
            }
        } else {
            key
        };
        Self::from_der(certificate, key)
    }
    pub fn certificate_sha256(&self) -> &str {
        &self.certificate_sha256
    }
}
fn read_credential(path: &Path, private: bool) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let meta = f.metadata()?;
    ensure!(meta.is_file(), "{} is not a regular file", path.display());
    ensure!(
        !private || meta.mode() & 0o077 == 0,
        "private key {} must be readable by its owner only",
        path.display()
    );
    let mut b = Vec::new();
    f.take(MAX_CREDENTIAL_BYTES + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= MAX_CREDENTIAL_BYTES,
        "{} exceeds {MAX_CREDENTIAL_BYTES} bytes",
        path.display()
    );
    Ok(b)
}

fn next_field<'a>(fields: &mut &'a [u8]) -> Result<(u8, &'a [u8])> {
    let ((tag, contents), rest) = der_element(fields)?;
    *fields = rest;
    Ok((tag, contents))
}
/// `(not_before, not_after)` of a DER certificate, in Unix seconds.
pub fn certificate_validity(der: &[u8]) -> Result<(i64, i64)> {
    let ((tag, certificate), rest) = der_element(der)?;
    ensure!(tag == 0x30 && rest.is_empty(), "certificate framing");
    let ((tag, tbs), _) = der_element(certificate)?;
    ensure!(tag == 0x30, "TBSCertificate framing");
    let mut fields = tbs;
    let (mut tag, _) = next_field(&mut fields)?;
    if tag == 0xa0 {
        tag = next_field(&mut fields)?.0; // the explicit version, then the serial number
    }
    ensure!(tag == 0x02, "certificate serial number");
    ensure!(
        next_field(&mut fields)?.0 == 0x30,
        "certificate signature algorithm"
    );
    ensure!(next_field(&mut fields)?.0 == 0x30, "certificate issuer");
    let (tag, validity) = next_field(&mut fields)?;
    ensure!(tag == 0x30, "certificate validity");
    let ((t0, before), rest) = der_element(validity)?;
    let ((t1, after), rest) = der_element(rest)?;
    ensure!(rest.is_empty(), "certificate validity holds two times");
    let (not_before, not_after) = (der_time(t0, before)?, der_time(t1, after)?);
    ensure!(not_before <= not_after, "certificate validity is inverted");
    Ok((not_before, not_after))
}
/// UTCTime (`YYMMDDHHMMSSZ`) or GeneralizedTime (`YYYYMMDDHHMMSSZ`) to Unix seconds.
fn der_time(tag: u8, bytes: &[u8]) -> Result<i64> {
    let s = std::str::from_utf8(bytes).context("certificate time is not ASCII")?;
    ensure!(
        s.len() >= 2 && s.ends_with('Z') && s[..s.len() - 1].bytes().all(|b| b.is_ascii_digit()),
        "certificate time `{s}` is not a UTC time"
    );
    let n = |r: std::ops::Range<usize>| -> i64 { s[r].parse().unwrap_or(-1) };
    let (year, at) = match (tag, s.len()) {
        (0x17, 13) => {
            let yy = n(0..2);
            (if yy >= 50 { 1900 + yy } else { 2000 + yy }, 2)
        }
        (0x18, 15) => (n(0..4), 4),
        _ => bail!("unsupported certificate time encoding"),
    };
    let (month, day) = (n(at..at + 2), n(at + 2..at + 4));
    let (hour, minute, second) = (n(at + 4..at + 6), n(at + 6..at + 8), n(at + 8..at + 10));
    ensure!(
        (1..=12).contains(&month)
            && (1..=31).contains(&day)
            && (0..24).contains(&hour)
            && (0..60).contains(&minute)
            && (0..60).contains(&second),
        "certificate time `{s}` is out of range"
    );
    Ok(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}
/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The certificates one side accepts (`certificate SHA-256 -> host`).
#[derive(Debug)]
struct Pinned {
    accepted: BTreeMap<String, String>,
    provider: Arc<CryptoProvider>,
}
impl Pinned {
    fn check(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> std::result::Result<(), rustls::Error> {
        let bad = rustls::Error::InvalidCertificate;
        if !intermediates.is_empty() {
            return Err(rustls::Error::General(
                "a pinned host certificate carries no chain".into(),
            ));
        }
        if !self.accepted.contains_key(&hash(end_entity.as_ref())) {
            return Err(bad(CertificateError::ApplicationVerificationFailure));
        }
        let (not_before, not_after) = certificate_validity(end_entity.as_ref())
            .map_err(|_| bad(CertificateError::BadEncoding))?;
        let now = now.as_secs() as i64;
        if now < not_before {
            return Err(bad(CertificateError::NotValidYet));
        }
        if now > not_after {
            return Err(bad(CertificateError::Expired));
        }
        Ok(())
    }
    fn tls12(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }
    fn tls13(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }
    fn schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}
impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity, intermediates, now)
            .map(|()| ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls12(message, cert, dss)
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls13(message, cert, dss)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}
impl ClientCertVerifier for Pinned {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }
    fn client_auth_mandatory(&self) -> bool {
        true
    }
    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> std::result::Result<ClientCertVerified, rustls::Error> {
        self.check(end_entity, intermediates, now)
            .map(|()| ClientCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls12(message, cert, dss)
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        self.tls13(message, cert, dss)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}

/// Connect, handshake and I/O allowances. `connect` bounds the TCP connect and `handshake` the
/// whole TLS handshake, as one absolute deadline over all of its reads and writes. `io` is the
/// per-operation timeout of an established stream; each exchange on it also gets an absolute
/// deadline from its caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timeouts {
    pub connect: Duration,
    pub handshake: Duration,
    pub io: Duration,
}
impl Timeouts {
    /// Deadlines derived from a frozen plan: I/O is bounded by the heartbeat timeout, and
    /// connect and handshake by the join timeout, at most 10 s each.
    pub fn from_plan(contract: &BoundedRunContract) -> Self {
        let bound =
            Duration::from_millis(contract.rendezvous.join_timeout_ms).min(Duration::from_secs(10));
        Self {
            connect: bound,
            handshake: bound,
            io: Duration::from_millis(contract.rendezvous.heartbeat_timeout_ms),
        }
    }
}

/// Every planned host's pinned certificate (`certificate SHA-256 -> host`). Two hosts that
/// present the same certificate cannot be told apart and are refused.
pub fn planned_certificates(contract: &BoundedRunContract) -> Result<BTreeMap<String, String>> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for p in &contract.participants {
        let h = &p.host;
        if let Some(other) = out.insert(h.certificate_sha256.clone(), h.host.clone()) {
            ensure!(
                other == h.host,
                "hosts `{other}` and `{}` present the same certificate",
                h.host
            );
        }
    }
    Ok(out)
}
/// The coordinator's pinned certificate: that of the planned host named `coordinator_host`.
/// A coordinator host outside the plan has no pinned identity and is refused.
pub fn coordinator_certificate(contract: &BoundedRunContract) -> Result<(String, String)> {
    let host = contract
        .participants
        .iter()
        .map(|p| &p.host)
        .find(|h| h.host == contract.coordinator_host)
        .with_context(|| {
            format!(
                "coordinator host `{}` is not a planned host, so it has no pinned certificate",
                contract.coordinator_host
            )
        })?;
    Ok((host.certificate_sha256.clone(), host.host.clone()))
}

enum Inner {
    Client(Box<StreamOwned<ClientConnection, Socket>>),
    Server(Box<StreamOwned<ServerConnection, Socket>>),
}
/// An authenticated TLS stream whose peer presented a pinned certificate.
pub struct TlsStream {
    inner: Inner,
    peer_certificate_sha256: String,
    peer: String,
}
impl TlsStream {
    pub fn peer_certificate_sha256(&self) -> &str {
        &self.peer_certificate_sha256
    }
    /// The planned host whose certificate the peer presented.
    pub fn peer(&self) -> &str {
        &self.peer
    }
    fn socket(&self) -> &Socket {
        match &self.inner {
            Inner::Client(s) => &s.sock,
            Inner::Server(s) => &s.sock,
        }
    }
}
impl Read for TlsStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match &mut self.inner {
            Inner::Client(s) => s.read(buf),
            Inner::Server(s) => s.read(buf),
        }
    }
}
impl Write for TlsStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match &mut self.inner {
            Inner::Client(s) => s.write(buf),
            Inner::Server(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match &mut self.inner {
            Inner::Client(s) => s.flush(),
            Inner::Server(s) => s.flush(),
        }
    }
}
impl ControlStream for TlsStream {
    fn set_timeouts(&self, read: Option<Duration>, write: Option<Duration>) -> std::io::Result<()> {
        self.socket().set_timeouts(read, write)
    }
    fn set_deadline(&self, deadline: Option<Instant>) -> std::io::Result<()> {
        self.socket().set_deadline(deadline)
    }
}
impl Drop for TlsStream {
    fn drop(&mut self) {
        // A courtesy close: never past the stream's deadline, and at most CLOSE_TIMEOUT from now.
        let socket = self.socket();
        let write = socket.limits().write;
        let wait = write.map_or(CLOSE_TIMEOUT, |w| w.min(CLOSE_TIMEOUT));
        socket.cap_deadline(Instant::now() + wait);
        match &mut self.inner {
            Inner::Client(s) => s.conn.send_close_notify(),
            Inner::Server(s) => s.conn.send_close_notify(),
        }
        let _ = self.flush();
    }
}

fn peer_of(
    certificates: Option<&[CertificateDer<'static>]>,
    accepted: &BTreeMap<String, String>,
) -> Result<(String, String)> {
    let certificate = certificates
        .and_then(|c| c.first())
        .context("peer presented no certificate")?;
    let sha = hash(certificate.as_ref());
    let host = accepted
        .get(&sha)
        .context("peer certificate is not pinned")?
        .clone();
    Ok((sha, host))
}
/// A socket for a TLS handshake: every one of its reads and writes ends by `deadline`.
fn handshaking(sock: TcpStream, timeout: Duration, deadline: Instant) -> Result<Socket> {
    sock.set_nodelay(true)?;
    let sock = DeadlineSocket::new(sock);
    sock.set_timeouts(Some(timeout), Some(timeout))?;
    sock.set_deadline(Some(deadline))?;
    Ok(sock)
}
/// After the handshake: the per-operation I/O timeout, and no deadline until a caller sets one
/// for an exchange.
fn established(sock: &Socket, io: Duration) -> Result<()> {
    sock.set_deadline(None)?;
    sock.set_timeouts(Some(io), Some(io))?;
    Ok(())
}

/// Connect to `address` and accept it only if it presents the pinned `server` certificate
/// (`(SHA-256, host)`), presenting `credential` in return.
pub fn connect(
    address: SocketAddr,
    credential: &HostCredential,
    server: (&str, &str),
    timeouts: Timeouts,
) -> Result<TlsStream> {
    let accepted = BTreeMap::from([(server.0.to_string(), server.1.to_string())]);
    let mut config = ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned {
            accepted: accepted.clone(),
            provider: provider(),
        }))
        .with_client_auth_cert(
            vec![credential.certificate.clone()],
            credential.key.clone_key(),
        )?;
    config.enable_sni = false;
    let sock = TcpStream::connect_timeout(&address, timeouts.connect)
        .with_context(|| format!("connect {address}"))?;
    // One `complete_io` step can make several blocking reads; the deadline bounds them all.
    let deadline = Instant::now() + timeouts.handshake;
    let mut sock = handshaking(sock, timeouts.handshake, deadline)?;
    let mut conn = ClientConnection::new(Arc::new(config), ServerName::try_from(SERVER_NAME)?)?;
    while conn.is_handshaking() {
        ensure!(
            Instant::now() < deadline,
            "TLS handshake with {address} exceeded {:?}",
            timeouts.handshake
        );
        conn.complete_io(&mut sock)
            .with_context(|| format!("TLS handshake with {address}"))?;
    }
    let (peer_certificate_sha256, peer) = peer_of(conn.peer_certificates(), &accepted)?;
    established(&sock, timeouts.io)?;
    Ok(TlsStream {
        inner: Inner::Client(Box::new(StreamOwned::new(conn, sock))),
        peer_certificate_sha256,
        peer,
    })
}

/// A TCP listener that completes a mutually authenticated handshake before yielding a stream.
pub struct TlsAcceptor {
    listener: TcpListener,
    config: Arc<ServerConfig>,
    accepted: BTreeMap<String, String>,
    timeouts: Timeouts,
}
impl TlsAcceptor {
    /// Listen on `address`, presenting `credential` and accepting only the pinned `accepted`
    /// certificates (`SHA-256 -> host`).
    pub fn bind(
        address: SocketAddr,
        credential: &HostCredential,
        accepted: BTreeMap<String, String>,
        timeouts: Timeouts,
    ) -> Result<Self> {
        ensure!(!accepted.is_empty(), "no pinned peer certificate");
        let config = ServerConfig::builder_with_provider(provider())
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(Arc::new(Pinned {
                accepted: accepted.clone(),
                provider: provider(),
            }))
            .with_single_cert(
                vec![credential.certificate.clone()],
                credential.key.clone_key(),
            )?;
        let listener = TcpListener::bind(address).with_context(|| format!("listen {address}"))?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            config: Arc::new(config),
            accepted,
            timeouts,
        })
    }
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }
    /// Accept one pending connection and authenticate it within the handshake deadline.
    /// `Ok(None)` when no connection is pending; an error when the peer failed to
    /// authenticate (nothing it sent is trusted).
    pub fn poll_accept(&self) -> Result<Option<TlsStream>> {
        self.accept_by(None)
    }
    /// [`TlsAcceptor::poll_accept`] whose handshake also ends by `deadline` (for example a
    /// join deadline) when that comes before the handshake allowance runs out.
    pub fn poll_accept_by(&self, deadline: Instant) -> Result<Option<TlsStream>> {
        self.accept_by(Some(deadline))
    }
    fn accept_by(&self, cap: Option<Instant>) -> Result<Option<TlsStream>> {
        let (sock, from) = match self.listener.accept() {
            Ok(accepted) => accepted,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        sock.set_nonblocking(false)?;
        // One absolute deadline for the whole handshake: one `complete_io` step can make
        // several blocking reads, and a peer that trickles a record inside every socket
        // timeout must not keep the step, or the caller, going.
        let start = Instant::now();
        let own = start + self.timeouts.handshake;
        let deadline = cap.map_or(own, |cap| cap.min(own));
        let allowance = deadline.saturating_duration_since(start);
        let mut sock = handshaking(sock, self.timeouts.handshake, deadline)?;
        let mut conn = ServerConnection::new(Arc::clone(&self.config))?;
        while conn.is_handshaking() {
            ensure!(
                Instant::now() < deadline,
                "TLS handshake from {from} exceeded {allowance:?}"
            );
            conn.complete_io(&mut sock)
                .with_context(|| format!("TLS handshake from {from}"))?;
        }
        let (peer_certificate_sha256, peer) = peer_of(conn.peer_certificates(), &self.accepted)?;
        established(&sock, self.timeouts.io)?;
        Ok(Some(TlsStream {
            inner: Inner::Server(Box::new(StreamOwned::new(conn, sock))),
            peer_certificate_sha256,
            peer,
        }))
    }
}
impl Acceptor for TlsAcceptor {
    fn next_incoming(&mut self, deadline: Instant) -> Incoming {
        match self.poll_accept_by(deadline) {
            Ok(None) => Incoming::Pending,
            Err(e) => Incoming::Unauthenticated(format!("{e:#}")),
            Ok(Some(stream)) => {
                let sha256 = stream.peer_certificate_sha256().to_string();
                Incoming::Stream(Box::new(stream), PeerIdentity::Certificate { sha256 })
            }
        }
    }
}

/// Frames carrying strictly consecutive delivery ids (1, 2, 3, ...) on one stream, for
/// request/response protocols. The frame limit covers the 8-byte id. A repeated, regressed
/// or skipped id refuses the channel for good.
pub struct Sequenced<S: Read + Write> {
    stream: S,
    limit: u32,
    sent: u64,
    received: u64,
    refused: bool,
}
impl<S: Read + Write> Sequenced<S> {
    pub fn new(stream: S, limit: u32) -> Self {
        Self {
            stream,
            limit,
            sent: 0,
            received: 0,
            refused: false,
        }
    }
    pub fn get_ref(&self) -> &S {
        &self.stream
    }
    /// Send `body` under the next delivery id and return that id.
    pub fn send(&mut self, body: &[u8]) -> Result<u64> {
        ensure!(!self.refused, "sequenced channel already refused");
        let id = self.sent + 1;
        let mut frame = id.to_be_bytes().to_vec();
        frame.extend_from_slice(body);
        write_frame(&mut self.stream, &frame, self.limit)?;
        self.sent = id;
        Ok(id)
    }
    /// Receive the next frame; only the next consecutive delivery id is accepted.
    pub fn receive(&mut self) -> Result<(u64, Vec<u8>)> {
        ensure!(!self.refused, "sequenced channel already refused");
        let frame = read_frame(&mut self.stream, self.limit);
        self.accept(frame)
    }
    fn accept(&mut self, frame: Result<Vec<u8>>) -> Result<(u64, Vec<u8>)> {
        let result = frame.and_then(|frame| {
            ensure!(frame.len() >= 8, "frame without a delivery id");
            let id = u64::from_be_bytes(frame[..8].try_into()?);
            ensure!(
                id == self.received + 1,
                "delivery id {id} is not the next id {}: replayed, duplicated or reordered",
                self.received + 1
            );
            Ok((id, frame[8..].to_vec()))
        });
        match result {
            Ok((id, body)) => {
                self.received = id;
                Ok((id, body))
            }
            Err(e) => {
                self.refused = true;
                Err(e)
            }
        }
    }
}
impl<S: ControlStream> Sequenced<S> {
    /// Run `op` with every blocking read and write on the stream ending by `deadline`, then
    /// remove the deadline.
    fn capped<T>(
        &mut self,
        deadline: Instant,
        op: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.stream.set_deadline(Some(deadline))?;
        let result = op(self);
        let cleared = self.stream.set_deadline(None);
        let value = result?;
        cleared?;
        Ok(value)
    }
    /// [`Sequenced::send`], ending by `deadline` however slowly the peer reads.
    pub fn send_by(&mut self, body: &[u8], deadline: Instant) -> Result<u64> {
        self.capped(deadline, |s| s.send(body))
    }
    /// [`Sequenced::receive`], ending by `deadline` however the peer paces its bytes.
    pub fn receive_by(&mut self, deadline: Instant) -> Result<(u64, Vec<u8>)> {
        self.capped(deadline, Self::receive)
    }
    /// Like [`Sequenced::receive`], for a server waiting on an idle peer: `Ok(None)` when no
    /// byte arrived within `wait` (nothing was consumed, the channel stays usable). A frame
    /// that has started must arrive whole within `within` of its first byte, however its
    /// bytes are paced, or the channel is refused.
    pub fn poll_receive(
        &mut self,
        wait: Duration,
        within: Duration,
    ) -> Result<Option<(u64, Vec<u8>)>> {
        ensure!(!self.refused, "sequenced channel already refused");
        let mut first = [0u8; 1];
        match self.capped(Instant::now() + wait, |s| Ok(s.stream.read(&mut first)))? {
            Ok(0) => {
                self.refused = true;
                bail!("peer closed the sequenced channel")
            }
            Ok(_) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Ok(None)
            }
            Err(e) => {
                self.refused = true;
                return Err(e.into());
            }
        }
        let limit = self.limit;
        let frame = self.capped(Instant::now() + within, |s| {
            let mut rest = [0u8; 3];
            s.stream.read_exact(&mut rest)?;
            let len = u32::from_be_bytes([first[0], rest[0], rest[1], rest[2]]);
            ensure!(
                len <= limit,
                "frame of {len} bytes exceeds message limit {limit}"
            );
            let mut body = vec![0u8; len as usize];
            s.stream.read_exact(&mut body)?;
            Ok(body)
        });
        self.accept(frame).map(Some)
    }
}
