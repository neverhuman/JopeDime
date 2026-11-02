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
        // Never let the crypto provider be inferred from the rustls crate features: any crate in
        // the workspace that turns on a second provider (e.g. an HTTP client pulling
        // `rustls/ring`) makes `CryptoProvider::from_crate_features()` return `None`, and a bare
        // `ClientConfig::builder()` then panics at the first outbound control connection.
        let tls = ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|_| ControlError::InvalidConfiguration)?
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
mod tests;
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
