use std::{
    fs::{self, OpenOptions},
    io::Read,
    net::SocketAddr,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use nix::unistd::Uid;
use rustls::{
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName},
    ClientConfig, RootCertStore,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};
use tokio_rustls::{client::TlsStream, TlsConnector};

use crate::{ExecutionError, NodeControlFrame, OutboxError, ProtocolError};

const MAX_TLS_FRAME_BYTES: usize = 1024 * 1024;
const MAX_CREDENTIAL_BYTES: u64 = 1024 * 1024;
const MAX_CONTROL_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_RECONNECT_FAILURES: u32 = 10_000;

#[derive(Clone, Debug)]
pub struct MtlsClientConfig {
    pub hub_address: SocketAddr,
    pub server_name: String,
    pub ca_certificate: PathBuf,
    pub client_certificate: PathBuf,
    pub client_private_key: PathBuf,
    pub connect_timeout: Duration,
    pub frame_timeout: Duration,
}

impl MtlsClientConfig {
    pub fn validate(&self) -> Result<(), ControlError> {
        self.validate_shape()?;
        let _material = load_tls_material(self)?;
        Ok(())
    }

    pub(crate) fn validate_shape(&self) -> Result<(), ControlError> {
        if self.server_name.is_empty()
            || self.connect_timeout.is_zero()
            || self.frame_timeout.is_zero()
            || self.connect_timeout > MAX_CONTROL_TIMEOUT
            || self.frame_timeout > MAX_CONTROL_TIMEOUT
            || !self.ca_certificate.is_absolute()
            || !self.client_certificate.is_absolute()
            || !self.client_private_key.is_absolute()
        {
            return Err(ControlError::InvalidConfiguration);
        }
        Ok(())
    }
}

pub struct MtlsControlClient {
    stream: TlsStream<TcpStream>,
    frame_timeout: Duration,
}

impl MtlsControlClient {
    pub async fn connect(config: &MtlsClientConfig) -> Result<Self, ControlError> {
        config.validate_shape()?;
        let material = load_tls_material(config)?;
        let mut roots = RootCertStore::empty();
        for certificate in material.ca_certificates {
            roots
                .add(certificate)
                .map_err(|_| ControlError::InvalidCertificate)?;
        }
        let tls = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_client_auth_cert(material.client_certificates, material.client_private_key)
            .map_err(|_| ControlError::InvalidCertificate)?;
        let tcp = tokio::time::timeout(
            config.connect_timeout,
            TcpStream::connect(config.hub_address),
        )
        .await
        .map_err(|_| ControlError::Timeout)?
        .map_err(ControlError::Io)?;
        tcp.set_nodelay(true).map_err(ControlError::Io)?;
        let server_name = ServerName::try_from(config.server_name.clone())
            .map_err(|_| ControlError::InvalidConfiguration)?;
        let stream = tokio::time::timeout(
            config.connect_timeout,
            TlsConnector::from(Arc::new(tls)).connect(server_name, tcp),
        )
        .await
        .map_err(|_| ControlError::Timeout)?
        .map_err(|_| ControlError::TlsHandshake)?;
        Ok(Self {
            stream,
            frame_timeout: config.frame_timeout,
        })
    }

    pub async fn send(&mut self, frame: &NodeControlFrame) -> Result<(), ControlError> {
        let bytes = frame.encode().map_err(ControlError::Protocol)?;
        if bytes.len() > MAX_TLS_FRAME_BYTES {
            return Err(ControlError::OversizedFrame);
        }
        tokio::time::timeout(self.frame_timeout, async {
            self.stream.write_u32(bytes.len() as u32).await?;
            self.stream.write_all(&bytes).await?;
            self.stream.flush().await
        })
        .await
        .map_err(|_| ControlError::Timeout)?
        .map_err(ControlError::Io)
    }

    pub async fn receive(&mut self) -> Result<NodeControlFrame, ControlError> {
        let bytes = tokio::time::timeout(self.frame_timeout, async {
            let length = self.stream.read_u32().await? as usize;
            if length == 0 || length > MAX_TLS_FRAME_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid control frame length",
                ));
            }
            let mut bytes = vec![0; length];
            self.stream.read_exact(&mut bytes).await?;
            Ok::<_, std::io::Error>(bytes)
        })
        .await
        .map_err(|_| ControlError::Timeout)?
        .map_err(ControlError::Io)?;
        NodeControlFrame::decode(&bytes).map_err(ControlError::Protocol)
    }
}

#[async_trait]
pub trait ControlConnection: Send {
    async fn send_frame(&mut self, frame: &NodeControlFrame) -> Result<(), ControlError>;
    async fn receive_frame(&mut self) -> Result<NodeControlFrame, ControlError>;
}

#[async_trait]
impl ControlConnection for MtlsControlClient {
    async fn send_frame(&mut self, frame: &NodeControlFrame) -> Result<(), ControlError> {
        Self::send(self, frame).await
    }

    async fn receive_frame(&mut self) -> Result<NodeControlFrame, ControlError> {
        Self::receive(self).await
    }
}

#[async_trait]
pub trait ControlConnector: Send + Sync {
    async fn connect(
        &self,
        config: &MtlsClientConfig,
    ) -> Result<Box<dyn ControlConnection>, ControlError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MtlsControlConnector;

#[async_trait]
impl ControlConnector for MtlsControlConnector {
    async fn connect(
        &self,
        config: &MtlsClientConfig,
    ) -> Result<Box<dyn ControlConnection>, ControlError> {
        Ok(Box::new(MtlsControlClient::connect(config).await?))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlFlow {
    Continue,
    Stop,
}

#[async_trait]
pub trait ControlFrameHandler: Send {
    async fn handle(&mut self, frame: NodeControlFrame) -> Result<ControlFlow, ControlError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReconnectPolicy {
    pub initial_delay: Duration,
    pub maximum_delay: Duration,
    pub maximum_failures: u32,
}

impl ReconnectPolicy {
    pub(crate) fn validate(self) -> Result<(), ControlError> {
        if self.initial_delay.is_zero()
            || self.initial_delay > self.maximum_delay
            || self.maximum_delay > MAX_CONTROL_TIMEOUT
            || self.maximum_failures == 0
            || self.maximum_failures > MAX_RECONNECT_FAILURES
        {
            return Err(ControlError::InvalidConfiguration);
        }
        Ok(())
    }
}

pub async fn supervise_outbound_control<F>(
    connector: &dyn ControlConnector,
    config: &MtlsClientConfig,
    policy: ReconnectPolicy,
    handler: &mut dyn ControlFrameHandler,
    mut hello_factory: F,
) -> Result<(), ControlError>
where
    F: FnMut() -> Result<NodeControlFrame, ControlError> + Send,
{
    config.validate_shape()?;
    policy.validate()?;
    let mut failures = 0;
    let mut delay = policy.initial_delay;
    loop {
        let mut connection = match connector.connect(config).await {
            Ok(connection) => connection,
            Err(error) => {
                wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
                continue;
            }
        };
        let hello = hello_factory()?;
        validate_outbound_session_hello(&hello)?;
        if let Err(error) = connection.send_frame(&hello).await {
            wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
            continue;
        }
        loop {
            let frame = match connection.receive_frame().await {
                Ok(frame) => frame,
                Err(error) => {
                    wait_before_reconnect(error, &mut failures, &mut delay, policy).await?;
                    break;
                }
            };
            if !controller_to_node(&frame) {
                return Err(ControlError::UnexpectedFrameDirection);
            }
            if matches!(frame, NodeControlFrame::ConfirmOutbound { .. }) {
                return Err(ControlError::UnexpectedFrameDirection);
            }
            match handler.handle(frame).await? {
                ControlFlow::Continue => {}
                ControlFlow::Stop => return Ok(()),
            }
        }
    }
}

pub(crate) fn controller_to_node(frame: &NodeControlFrame) -> bool {
    matches!(
        frame,
        NodeControlFrame::Infer { .. }
            | NodeControlFrame::Cancel { .. }
            | NodeControlFrame::Rotate { .. }
            | NodeControlFrame::Revoke { .. }
            | NodeControlFrame::ConfirmOutbound { .. }
            | NodeControlFrame::StorageControl { .. }
    )
}

pub(crate) fn validate_outbound_session_hello(
    frame: &NodeControlFrame,
) -> Result<(), ControlError> {
    let NodeControlFrame::SessionHello {
        node_id,
        session_id,
        profile,
        provision,
        qualification_digest,
        at_ms,
        signature,
        ..
    } = frame
    else {
        return Err(ControlError::UnexpectedFrameDirection);
    };
    frame
        .verify_session_hello(
            node_id,
            session_id,
            *profile,
            *provision,
            *qualification_digest,
            &signature.key_id,
            signature.public_key,
            *at_ms,
            1,
        )
        .map_err(ControlError::Protocol)
}

pub(crate) async fn wait_before_reconnect(
    error: ControlError,
    failures: &mut u32,
    delay: &mut Duration,
    policy: ReconnectPolicy,
) -> Result<(), ControlError> {
    if !error.reconnectable() {
        return Err(error);
    }
    *failures = (*failures).saturating_add(1);
    if *failures >= policy.maximum_failures {
        return Err(ControlError::ReconnectExhausted);
    }
    tokio::time::sleep(*delay).await;
    *delay = std::cmp::min((*delay).saturating_mul(2), policy.maximum_delay);
    Ok(())
}

struct TlsMaterial {
    ca_certificates: Vec<CertificateDer<'static>>,
    client_certificates: Vec<CertificateDer<'static>>,
    client_private_key: PrivateKeyDer<'static>,
}

fn load_tls_material(config: &MtlsClientConfig) -> Result<TlsMaterial, ControlError> {
    let ca_pem = read_physical_credential(&config.ca_certificate, false)?;
    let ca_certificates = CertificateDer::pem_slice_iter(&ca_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ControlError::InvalidCertificate)?;
    let client_pem = read_physical_credential(&config.client_certificate, false)?;
    let client_certificates = CertificateDer::pem_slice_iter(&client_pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ControlError::InvalidCertificate)?;
    if ca_certificates.is_empty() || client_certificates.is_empty() {
        return Err(ControlError::InvalidCertificate);
    }
    let mut private_key_pem = read_physical_credential(&config.client_private_key, true)?;
    let private_keys =
        PrivateKeyDer::pem_slice_iter(&private_key_pem).collect::<Result<Vec<_>, _>>();
    private_key_pem.fill(0);
    let mut private_keys = private_keys.map_err(|_| ControlError::InvalidCertificate)?;
    if private_keys.len() != 1 {
        return Err(ControlError::InvalidCertificate);
    }
    let client_private_key = private_keys.pop().ok_or(ControlError::InvalidCertificate)?;
    Ok(TlsMaterial {
        ca_certificates,
        client_certificates,
        client_private_key,
    })
}

fn read_physical_credential(path: &Path, owner_only: bool) -> Result<Vec<u8>, ControlError> {
    if !path.is_absolute() {
        return Err(ControlError::InvalidConfiguration);
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(ControlError::Io)?;
    let opened = file.metadata().map_err(ControlError::Io)?;
    let named = fs::symlink_metadata(path).map_err(ControlError::Io)?;
    let mode = opened.mode() & 0o777;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.len() == 0
        || opened.len() > MAX_CREDENTIAL_BYTES
        || (owner_only && mode != 0o600)
        || (!owner_only && (mode & 0o111 != 0 || mode & 0o022 != 0 || mode & 0o400 == 0))
        || path.canonicalize().map_err(ControlError::Io)? != path
    {
        return Err(ControlError::InvalidConfiguration);
    }
    validate_physical_components(path)?;
    let mut bytes = Vec::new();
    let read_result = Read::by_ref(&mut file)
        .take(MAX_CREDENTIAL_BYTES + 1)
        .read_to_end(&mut bytes);
    if let Err(error) = read_result {
        bytes.fill(0);
        return Err(ControlError::Io(error));
    }
    if bytes.len() as u64 != opened.len() || bytes.len() as u64 > MAX_CREDENTIAL_BYTES {
        bytes.fill(0);
        return Err(ControlError::InvalidConfiguration);
    }
    Ok(bytes)
}

fn validate_physical_components(path: &Path) -> Result<(), ControlError> {
    let mut current = PathBuf::from("/");
    for component in path.components().skip(1) {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)
            .map_err(ControlError::Io)?
            .file_type()
            .is_symlink()
        {
            return Err(ControlError::InvalidConfiguration);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        fs, io,
        os::unix::fs::{symlink, PermissionsExt},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use ed25519_dalek::SigningKey;
    use jain_smartcluster_fabric::{DomainSignature, SignatureAuthority};

    use super::*;
    use crate::{supervise_durable_outbound_control, DurableControlOutbox};

    fn config() -> MtlsClientConfig {
        MtlsClientConfig {
            hub_address: SocketAddr::from(([127, 0, 0, 1], 9443)),
            server_name: "hub.internal".to_owned(),
            ca_certificate: PathBuf::from("/etc/jain/ca.pem"),
            client_certificate: PathBuf::from("/etc/jain/node.pem"),
            client_private_key: PathBuf::from("/etc/jain/node-key.pem"),
            connect_timeout: Duration::from_secs(1),
            frame_timeout: Duration::from_secs(1),
        }
    }

    fn policy(maximum_failures: u32) -> ReconnectPolicy {
        ReconnectPolicy {
            initial_delay: Duration::from_millis(1),
            maximum_delay: Duration::from_millis(2),
            maximum_failures,
        }
    }

    fn hello(sequence: u64) -> NodeControlFrame {
        NodeControlFrame::session_hello(
            "node-1".to_owned(),
            "session-1".to_owned(),
            crate::NodeProfile::Cpu,
            crate::WorkerProvision::CodeRunner,
            [7; 32],
            sequence,
            10_000 + sequence,
            "node-key-1".to_owned(),
            &SigningKey::from_bytes(&[53; 32]),
        )
        .unwrap()
    }

    fn hub_key() -> SigningKey {
        SigningKey::from_bytes(&[79; 32])
    }

    fn durable_identity() -> crate::NodeRuntimeIdentity {
        crate::NodeRuntimeIdentity::new(
            "node-1".to_owned(),
            "session-1".to_owned(),
            "node-key-1".to_owned(),
            "hub-issuer".to_owned(),
            hub_key().verifying_key().to_bytes(),
            "image-issuer".to_owned(),
            [4; 32],
            SigningKey::from_bytes(&[53; 32]),
        )
        .unwrap()
    }

    fn durable_event(cursor: u64) -> crate::NodeEvent {
        crate::NodeEvent::Progress {
            identity: crate::EventIdentity {
                run_id: "run-1".to_owned(),
                attempt: 1,
                fence: 2,
                cursor,
                stream_offset: 0,
                control_position: 3,
                phase: crate::ExecutionPhase::Run,
            },
            completed: cursor,
            total: Some(2),
            message: "running".to_owned(),
        }
    }

    fn confirmation(pending: &crate::OutboundPending) -> NodeControlFrame {
        NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            pending.sequence,
            pending.digest,
            50_000,
            "hub-issuer".to_owned(),
            "hub-key-1".to_owned(),
            &hub_key(),
        )
        .unwrap()
    }

    fn controller_frame() -> NodeControlFrame {
        NodeControlFrame::Revoke {
            session_id: "session-1".to_owned(),
            sequence: 90,
            reason: "test stop".to_owned(),
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key-1".to_owned(),
                public_key: [9; 32],
                signature: vec![1; 64],
            },
        }
    }

    #[test]
    fn direction_allowlist_rejects_every_node_owned_frame_class() {
        let identity = crate::NodeRuntimeIdentity::new(
            "node-1".to_owned(),
            "session-1".to_owned(),
            "node-key-1".to_owned(),
            "hub-issuer".to_owned(),
            [3; 32],
            "image-issuer".to_owned(),
            [4; 32],
            SigningKey::from_bytes(&[53; 32]),
        )
        .unwrap();
        let event = crate::NodeEvent::SmokeStarted(crate::EventIdentity {
            run_id: "run-1".to_owned(),
            attempt: 1,
            fence: 2,
            cursor: 1,
            stream_offset: 0,
            control_position: 3,
            phase: crate::ExecutionPhase::Smoke,
        });
        assert!(!controller_to_node(&hello(1)));
        assert!(!controller_to_node(
            &NodeControlFrame::outbound_event(&identity, 1, event).unwrap()
        ));
        assert!(controller_to_node(&controller_frame()));
        assert!(controller_to_node(
            &NodeControlFrame::outbound_confirmation(
                "session-1".to_owned(),
                1,
                [9; 32],
                1,
                "hub-issuer".to_owned(),
                "hub-key-1".to_owned(),
                &hub_key(),
            )
            .unwrap()
        ));
    }

    struct ScriptedConnection {
        sent: Arc<Mutex<Vec<NodeControlFrame>>>,
        send_error_on: Option<usize>,
        send_calls: usize,
        received: VecDeque<Result<NodeControlFrame, ControlError>>,
    }

    #[async_trait]
    impl ControlConnection for ScriptedConnection {
        async fn send_frame(&mut self, frame: &NodeControlFrame) -> Result<(), ControlError> {
            self.sent.lock().unwrap().push(frame.clone());
            self.send_calls += 1;
            if self.send_error_on == Some(self.send_calls) {
                self.send_error_on = None;
                return Err(ControlError::Io(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "scripted ambiguous send",
                )));
            }
            Ok(())
        }

        async fn receive_frame(&mut self) -> Result<NodeControlFrame, ControlError> {
            self.received.pop_front().unwrap_or_else(|| {
                Err(ControlError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "scripted EOF",
                )))
            })
        }
    }

    struct ScriptedConnector {
        attempts: Mutex<VecDeque<Result<ScriptedConnection, ControlError>>>,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ControlConnector for ScriptedConnector {
        async fn connect(
            &self,
            _config: &MtlsClientConfig,
        ) -> Result<Box<dyn ControlConnection>, ControlError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let attempt = {
                self.attempts
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected connection attempt")
            };
            attempt.map(|connection| Box::new(connection) as Box<dyn ControlConnection>)
        }
    }

    struct ScriptedHandler {
        seen: Vec<NodeControlFrame>,
        failure: Option<ControlError>,
        stop_after: usize,
    }

    #[async_trait]
    impl ControlFrameHandler for ScriptedHandler {
        async fn handle(&mut self, frame: NodeControlFrame) -> Result<ControlFlow, ControlError> {
            self.seen.push(frame);
            if let Some(error) = self.failure.take() {
                return Err(error);
            }
            Ok(if self.seen.len() >= self.stop_after {
                ControlFlow::Stop
            } else {
                ControlFlow::Continue
            })
        }
    }

    #[test]
    fn credentials_require_one_physical_bounded_owner_custody() {
        let root = tempfile::tempdir().unwrap();
        let key = root.path().join("client-key.pem");
        fs::write(&key, b"key").unwrap();
        fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(read_physical_credential(&key, true).unwrap(), b"key");

        for mode in [0o640, 0o604, 0o444] {
            fs::set_permissions(&key, fs::Permissions::from_mode(mode)).unwrap();
            assert!(
                read_physical_credential(&key, true).is_err(),
                "mode {mode:o}"
            );
        }
        fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();

        let hardlink = root.path().join("client-key-hardlink.pem");
        fs::hard_link(&key, &hardlink).unwrap();
        assert!(read_physical_credential(&key, true).is_err());
        fs::remove_file(&hardlink).unwrap();

        let symlink_path = root.path().join("client-key-symlink.pem");
        symlink(&key, &symlink_path).unwrap();
        assert!(read_physical_credential(&symlink_path, true).is_err());

        let certificate = root.path().join("client.pem");
        fs::write(&certificate, b"certificate").unwrap();
        fs::set_permissions(&certificate, fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            read_physical_credential(&certificate, false).unwrap(),
            b"certificate"
        );
        for mode in [0o666, 0o744, 0o244] {
            fs::set_permissions(&certificate, fs::Permissions::from_mode(mode)).unwrap();
            assert!(
                read_physical_credential(&certificate, false).is_err(),
                "mode {mode:o}"
            );
        }

        let oversized = root.path().join("oversized.pem");
        fs::write(&oversized, vec![1; MAX_CREDENTIAL_BYTES as usize + 1]).unwrap();
        fs::set_permissions(&oversized, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(read_physical_credential(&oversized, true).is_err());
    }

    #[tokio::test]
    async fn reconnect_sends_a_fresh_hello_before_any_dispatch() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([
                Err(ControlError::Timeout),
                Ok(ScriptedConnection {
                    sent: Arc::clone(&sent),
                    send_error_on: Some(1),
                    send_calls: 0,
                    received: VecDeque::new(),
                }),
                Ok(ScriptedConnection {
                    sent: Arc::clone(&sent),
                    send_error_on: None,
                    send_calls: 0,
                    received: VecDeque::from([Err(ControlError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "first session ended",
                    )))]),
                }),
                Ok(ScriptedConnection {
                    sent: Arc::clone(&sent),
                    send_error_on: None,
                    send_calls: 0,
                    received: VecDeque::from([Ok(controller_frame())]),
                }),
            ])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: None,
            stop_after: 1,
        };
        let mut sequence = 0;
        supervise_outbound_control(&connector, &config(), policy(5), &mut handler, || {
            sequence += 1;
            Ok(hello(sequence))
        })
        .await
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(handler.seen, vec![controller_frame()]);
        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 3);
        for (index, frame) in sent.iter().enumerate() {
            let NodeControlFrame::SessionHello { sequence, .. } = frame else {
                panic!("first outbound frame was not a session hello");
            };
            assert_eq!(*sequence, index as u64 + 1);
        }
    }

    #[tokio::test]
    async fn durable_supervisor_replays_ambiguous_send_and_retires_only_signed_confirmations() {
        let root = tempfile::tempdir().unwrap();
        let outbox =
            DurableControlOutbox::open(root.path().join("outbox"), durable_identity()).unwrap();
        let first = outbox.enqueue_event(durable_event(1)).unwrap();
        let second = outbox.enqueue_event(durable_event(2)).unwrap();
        let first_confirmation = confirmation(&first);
        let second_confirmation = confirmation(&second);
        let sent = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([
                Ok(ScriptedConnection {
                    sent: Arc::clone(&sent),
                    send_error_on: Some(2),
                    send_calls: 0,
                    received: VecDeque::new(),
                }),
                Ok(ScriptedConnection {
                    sent: Arc::clone(&sent),
                    send_error_on: None,
                    send_calls: 0,
                    received: VecDeque::from([
                        Ok(controller_frame()),
                        Ok(first_confirmation.clone()),
                        Ok(first_confirmation),
                        Ok(second_confirmation),
                        Ok(controller_frame()),
                    ]),
                }),
            ])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: None,
            stop_after: 2,
        };
        let mut hello_sequence = 0;
        supervise_durable_outbound_control(
            &connector,
            &config(),
            policy(4),
            &mut handler,
            &outbox,
            100,
            || Ok(50_010),
            || {
                hello_sequence += 1;
                Ok(hello(hello_sequence))
            },
        )
        .await
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(handler.seen, vec![controller_frame(), controller_frame()]);
        assert_eq!(outbox.pending_count().unwrap(), 0);
        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 5);
        assert!(matches!(sent[0], NodeControlFrame::SessionHello { .. }));
        assert_eq!(sent[1], first.frame);
        assert!(matches!(sent[2], NodeControlFrame::SessionHello { .. }));
        assert_eq!(sent[3], first.frame);
        assert_eq!(sent[4], second.frame);
    }

    #[tokio::test]
    async fn terminal_direction_dispatch_and_budget_failures_never_reconnect() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: None,
                send_calls: 0,
                received: VecDeque::from([Ok(hello(90))]),
            })])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: None,
            stop_after: 1,
        };
        let error =
            supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
                Ok(hello(1))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, ControlError::UnexpectedFrameDirection));
        assert!(handler.seen.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([Ok(ScriptedConnection {
                sent: Arc::clone(&sent),
                send_error_on: None,
                send_calls: 0,
                received: VecDeque::from([Ok(controller_frame())]),
            })])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: Some(ControlError::Timeout),
            stop_after: 1,
        };
        let error =
            supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
                Ok(hello(2))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, ControlError::Timeout));
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let calls = Arc::new(AtomicUsize::new(0));
        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([
                Err(ControlError::Timeout),
                Err(ControlError::TlsHandshake),
            ])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: None,
            stop_after: 1,
        };
        let error =
            supervise_outbound_control(&connector, &config(), policy(2), &mut handler, || {
                Ok(hello(3))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, ControlError::ReconnectExhausted));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn invalid_certificate_is_terminal_before_hello_or_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let connector = ScriptedConnector {
            attempts: Mutex::new(VecDeque::from([Err(ControlError::InvalidCertificate)])),
            calls: Arc::clone(&calls),
        };
        let mut handler = ScriptedHandler {
            seen: Vec::new(),
            failure: None,
            stop_after: 1,
        };
        let mut hello_calls = 0;
        let error =
            supervise_outbound_control(&connector, &config(), policy(4), &mut handler, || {
                hello_calls += 1;
                Ok(hello(hello_calls))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, ControlError::InvalidCertificate));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(hello_calls, 0);
        assert!(handler.seen.is_empty());
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    #[error("invalid mTLS control configuration")]
    InvalidConfiguration,
    #[error("invalid mTLS certificate material")]
    InvalidCertificate,
    #[error("mTLS handshake failed")]
    TlsHandshake,
    #[error("mTLS control deadline exceeded")]
    Timeout,
    #[error("mTLS control frame exceeded its bound")]
    OversizedFrame,
    #[error("control frame direction violated the node protocol")]
    UnexpectedFrameDirection,
    #[error("storage composition is unavailable until the governed Shard dependency is pinned")]
    StorageCompositionUnavailable,
    #[error("outbound control reconnect budget exhausted")]
    ReconnectExhausted,
    #[error("node protocol failed: {0}")]
    Protocol(ProtocolError),
    #[error("durable control outbox failed: {0}")]
    Outbox(#[from] OutboxError),
    #[error("node execution dispatch failed: {0}")]
    Execution(ExecutionError),
    #[error("mTLS control I/O failed: {0}")]
    Io(std::io::Error),
}

impl ControlError {
    fn reconnectable(&self) -> bool {
        matches!(self, Self::TlsHandshake | Self::Timeout | Self::Io(_))
    }
}
