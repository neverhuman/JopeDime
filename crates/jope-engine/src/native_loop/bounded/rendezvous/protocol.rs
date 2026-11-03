use super::*;

/// A bidirectional control stream: an authenticated TLS stream
/// ([`super::transport::TlsStream`]) or a same-host Unix socket (fixtures) behind a
/// [`DeadlineSocket`].
pub trait ControlStream: Read + Write + Send {
    /// Per-operation timeouts: each single read or write may wait this long (inactivity).
    fn set_timeouts(&self, read: Option<Duration>, write: Option<Duration>) -> std::io::Result<()>;
    /// An absolute deadline over every later blocking read and write, including each of the
    /// several a TLS stream can make inside one call, or none. Bytes that keep arriving
    /// inside a per-operation timeout do not extend it.
    fn set_deadline(&self, deadline: Option<Instant>) -> std::io::Result<()>;
}
/// Run `exchange` on `stream` with every blocking read and write ending by `deadline`, then
/// remove the deadline.
pub fn by_deadline<S, T>(
    stream: &mut S,
    deadline: Instant,
    exchange: impl FnOnce(&mut S) -> Result<T>,
) -> Result<T>
where
    S: ControlStream + ?Sized,
{
    stream.set_deadline(Some(deadline))?;
    let result = exchange(stream);
    let cleared = stream.set_deadline(None);
    let value = result?;
    cleared?;
    Ok(value)
}
/// [`read_frame`] that ends by `deadline`, however the peer paces its bytes.
pub fn read_frame_by<S: ControlStream + ?Sized>(
    stream: &mut S,
    limit: u32,
    deadline: Instant,
) -> Result<Vec<u8>> {
    by_deadline(stream, deadline, |s| read_frame(s, limit))
}
/// [`write_frame`] that ends by `deadline`, however slowly the peer reads.
pub fn write_frame_by<S: ControlStream + ?Sized>(
    stream: &mut S,
    bytes: &[u8],
    limit: u32,
    deadline: Instant,
) -> Result<()> {
    by_deadline(stream, deadline, |s| write_frame(s, bytes, limit))
}
/// The channel identity of a peer, established before any frame is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerIdentity {
    /// A same-host Unix socket (fixture transport): no identity beyond the host.
    SameHostSocket,
    /// A mutually authenticated TLS peer that presented this pinned certificate; it may only
    /// speak for ranks planned on the host that owns that certificate.
    Certificate { sha256: String },
}
/// One step of an [`Acceptor`].
pub enum Incoming {
    /// Nothing is waiting.
    Pending,
    /// A peer connected but did not authenticate; nothing it sent is trusted.
    Unauthenticated(String),
    Stream(Box<dyn ControlStream>, PeerIdentity),
}
/// A non-blocking source of authenticated control streams.
pub trait Acceptor {
    /// Authenticate one pending connection, if any, finishing by `deadline` at the latest.
    fn next_incoming(&mut self, deadline: Instant) -> Incoming;
}
/// The same-host fixture acceptor. It authenticates nothing, so it has no handshake to bound.
pub struct UnixAcceptor<'a>(pub &'a UnixListener);
impl Acceptor for UnixAcceptor<'_> {
    fn next_incoming(&mut self, _deadline: Instant) -> Incoming {
        match self.0.accept() {
            Ok((stream, _)) => match stream.set_nonblocking(false) {
                Ok(()) => Incoming::Stream(
                    Box::new(DeadlineSocket::new(stream)),
                    PeerIdentity::SameHostSocket,
                ),
                Err(e) => Incoming::Unauthenticated(e.to_string()),
            },
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Incoming::Pending,
            Err(e) => Incoming::Unauthenticated(e.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChallengeBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub coordinator_boot_id: String,
    pub opened_unix_ms: u64,
    pub join_deadline_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Challenge {
    pub body: ChallengeBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HelloBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub global_rank: u32,
    pub delivery_id: u64,
    pub sent_unix_ms: u64,
    /// Observed on the participant's own host, never copied from the plan.
    pub host: String,
    pub machine_id: String,
    pub boot_id: String,
    pub device_uuid: Option<String>,
    pub source_commit: String,
    pub executable_sha256: String,
    pub runtime: RuntimeIdentity,
    pub installed_policy_sha256: String,
    /// [`observed_closure_digest`] over the participant's local input copy.
    pub artifact_closure_sha256: String,
    pub grant_sha256: String,
    pub communicator_bootstrap: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Hello {
    pub body: HelloBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Member {
    pub global_rank: u32,
    pub hello_sha256: String,
    pub host: String,
    pub boot_id: String,
    pub device_uuid: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub rendezvous_id: String,
    pub admitted_unix_ms: u64,
    pub original_deadline_unix_ms: u64,
    pub grants_sha256: String,
    pub members: Vec<Member>,
    pub communicator_bootstrap: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GroupAdmission {
    pub body: AdmissionBody,
    pub signature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResultBody {
    pub schema: String,
    pub run: String,
    pub attempt: u32,
    pub plan_sha256: String,
    pub admission_sha256: String,
    pub global_rank: u32,
    pub delivery_id: u64,
    pub result_sha256: String,
    pub result_bytes: u64,
    pub updates: u64,
    pub completed_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignedResult {
    pub body: ResultBody,
    pub signature: String,
}
/// Post-admission control. Heartbeats carry strictly increasing delivery ids.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlFrame {
    Heartbeat { delivery_id: u64 },
    Stop { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "cause", rename_all = "snake_case", deny_unknown_fields)]
pub enum RefusalCause {
    Oversized { bytes: u64, limit: u64 },
    Malformed { detail: String },
    UnknownRank { rank: u32 },
    ForgedPeer { claimed_rank: u32 },
    ForgedCoordinator,
    WrongRun,
    ReplayedAttempt { attempt: u32 },
    WrongPlan,
    StaleRendezvous,
    ReplayedBoot { rank: u32 },
    WrongParticipant { rank: u32 },
    SubstitutedSource { rank: u32 },
    SubstitutedInput { rank: u32 },
    WrongGrant { rank: u32 },
    GrantExpired { rank: u32 },
    DeadlineElapsed,
    NotYetOpen,
    StaleMessage { rank: u32 },
    JoinTimeout { missing: Vec<u32> },
    Equivocation { rank: u32 },
    IncompleteGroup,
}
/// Terminal refusal of the whole synchronous group. No rank may start and no update is
/// committed; `committed_updates` is structurally zero.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GroupRefusal {
    pub run: String,
    pub attempt: u32,
    pub rendezvous_id: String,
    pub cause: RefusalCause,
    pub started_ranks: u32,
    pub committed_updates: u64,
}
impl std::fmt::Display for GroupRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bounded-run group refused: {}",
            serde_json::to_string(&self.cause).unwrap_or_default()
        )
    }
}
impl std::error::Error for GroupRefusal {}

/// Deterministic digest of the planned (path, sha256, bytes) closure.
pub fn closure_digest(closure: &[ClosureEntry]) -> Result<String> {
    digest(&closure)
}
/// Hash the participant's own local copy of every closure entry (bounded by the planned
/// length). A substituted byte changes this digest, which the coordinator refuses.
pub fn observed_closure_digest(root: &Path, closure: &[ClosureEntry]) -> Result<String> {
    let mut observed = Vec::with_capacity(closure.len());
    for e in closure {
        let path = crate::manifest::resolve(root, &e.path)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)?;
        ensure!(
            file.metadata()?.is_file(),
            "closure entry is not a regular file"
        );
        let mut bytes = Vec::new();
        file.take(e.bytes + 1).read_to_end(&mut bytes)?;
        observed.push(ClosureEntry {
            path: e.path.clone(),
            sha256: crate::manifest::hash(&bytes),
            bytes: bytes.len() as u64,
        });
    }
    closure_digest(&observed)
}
/// Immutable verified artifact receipt: the bytes must match the planned entry exactly and
/// are published without replacement. An identical existing file is idempotent; any
/// conflicting existing file is refused and retained.
pub fn receive_artifact(root: &Path, entry: &ClosureEntry, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() as u64 == entry.bytes && crate::manifest::hash(bytes) == entry.sha256,
        "exchanged artifact `{}` differs from the planned closure",
        entry.path.display()
    );
    publish_immutable(root, &entry.path, bytes)
}
pub(crate) fn publish_immutable(root: &Path, relative: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(super::super::is_relative(relative), "unsafe artifact path");
    let path = root.join(relative);
    let parent = path.parent().context("artifact parent")?;
    fs::create_dir_all(parent)?;
    let (mut f, staged) = tempfile::Builder::new()
        .prefix(".exchange-pending-")
        .tempfile_in(parent)?
        .keep()?;
    f.write_all(bytes)?;
    f.sync_all()?;
    match crate::native_persistence::publish_staged(&staged, &path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            crate::native_persistence::discard_staging(&staged)?;
            let existing = fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&path)?;
            let mut old = Vec::new();
            existing
                .take(bytes.len() as u64 + 1)
                .read_to_end(&mut old)?;
            ensure!(
                old == bytes,
                "conflicting immutable artifact `{}`",
                relative.display()
            );
        }
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

/// Length-prefixed frame. The limit is enforced before the body is read or allocated. Length
/// and body go to the stream in one write, so a TLS stream queues a small frame (a heartbeat,
/// an admission) whole, and a write cut short by a deadline does not leave its length queued
/// without its body.
pub fn write_frame<W: Write + ?Sized>(w: &mut W, bytes: &[u8], limit: u32) -> Result<()> {
    ensure!(
        bytes.len() as u64 <= u64::from(limit),
        "frame exceeds message limit"
    );
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(bytes);
    w.write_all(&frame)?;
    w.flush()?;
    Ok(())
}
pub fn read_frame<R: Read + ?Sized>(r: &mut R, limit: u32) -> Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len);
    ensure!(
        len <= limit,
        "frame of {len} bytes exceeds message limit {limit}"
    );
    let mut body = vec![0u8; len as usize];
    r.read_exact(&mut body)?;
    Ok(body)
}

pub fn sign_hello(key: &SigningKey, body: HelloBody) -> Result<Hello> {
    Ok(Hello {
        signature: sign(key, HELLO_DOMAIN, &body)?,
        body,
    })
}
pub fn sign_result(key: &SigningKey, body: ResultBody) -> Result<SignedResult> {
    Ok(SignedResult {
        signature: sign(key, RESULT_DOMAIN, &body)?,
        body,
    })
}
