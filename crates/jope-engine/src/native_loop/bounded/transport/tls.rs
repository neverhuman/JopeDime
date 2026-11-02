use super::*;

type Socket = DeadlineSocket<TcpStream>;

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
