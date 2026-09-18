use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use jain_smartcluster::{FrameCodec, FrameError};
use jain_smartcluster_core::{
    ClientFrame, ErrorCode, ProtocolVersion, Request, Response, ResponseEnvelope, ServerError,
    ServerFrame, StreamId, StreamInputFrame, Validate,
};
use thiserror::Error;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot, Notify, Semaphore};

use crate::auth::{AuthError, AuthenticatedPeer, Authorizer};

const CONNECTION_OUTPUT_CAPACITY: usize = 256;
const MAX_CONNECTIONS: usize = 1_024;
const MAX_CONNECTIONS_PER_UID: usize = 128;
const INITIAL_FRAME_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECTION_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
pub struct ConnectionSink {
    sender: mpsc::Sender<Outbound>,
    disconnected: Arc<AtomicBool>,
    disconnect: Arc<Notify>,
}

struct Outbound {
    frame: ServerFrame,
    written: Option<oneshot::Sender<()>>,
}

impl ConnectionSink {
    pub async fn send(&self, frame: ServerFrame) -> Result<(), ConnectionClosed> {
        let (written, receiver) = oneshot::channel();
        self.sender
            .send(Outbound {
                frame,
                written: Some(written),
            })
            .await
            .map_err(|_| ConnectionClosed)?;
        receiver.await.map_err(|_| ConnectionClosed)
    }

    pub fn try_send(&self, frame: ServerFrame) -> Result<(), StreamBackpressure> {
        let result = self
            .sender
            .try_send(Outbound {
                frame,
                written: None,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => StreamBackpressure::Lagged,
                mpsc::error::TrySendError::Closed(_) => StreamBackpressure::Closed,
            });
        if result.is_err() {
            self.disconnect();
        }
        result
    }

    pub fn close_stream(
        &self,
        stream_id: StreamId,
        error: Option<ServerError>,
    ) -> Result<(), StreamBackpressure> {
        self.try_send(ServerFrame::StreamClosed { stream_id, error })
    }

    pub fn disconnect(&self) {
        if !self.disconnected.swap(true, Ordering::AcqRel) {
            self.disconnect.notify_waiters();
        }
    }

    pub fn is_disconnected(&self) -> bool {
        self.disconnected.load(Ordering::Acquire)
    }

    pub async fn disconnected(&self) {
        loop {
            if self.is_disconnected() {
                return;
            }
            let notified = self.disconnect.notified();
            if self.is_disconnected() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("connection closed")]
pub struct ConnectionClosed;

#[derive(Clone, Copy, Debug, Error)]
pub enum StreamBackpressure {
    #[error("connection output queue is full; subscriber must resume by offset")]
    Lagged,
    #[error("connection is closed")]
    Closed,
}

#[async_trait]
pub trait RequestHandler: Send + Sync + 'static {
    async fn request(
        &self,
        peer: &AuthenticatedPeer,
        request: Request,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError>;

    async fn stream_input(
        &self,
        peer: &AuthenticatedPeer,
        frame: StreamInputFrame,
    ) -> Result<(), ServerError>;

    async fn close_stream(&self, peer: &AuthenticatedPeer, stream_id: StreamId);

    async fn stream_response_written(&self, _peer: &AuthenticatedPeer, _stream_id: StreamId) {}

    async fn connection_closed(&self, _peer: &AuthenticatedPeer) {}
}

pub struct UnixServer<H> {
    listener: UnixListener,
    socket_path: PathBuf,
    authorizer: Authorizer,
    handler: Arc<H>,
    codec: FrameCodec,
    connections: Arc<Semaphore>,
    connections_by_uid: Arc<Mutex<HashMap<u32, usize>>>,
}

impl<H: RequestHandler> UnixServer<H> {
    pub fn bind(
        socket_path: impl Into<PathBuf>,
        socket_mode: u32,
        authorizer: Authorizer,
        handler: Arc<H>,
        codec: FrameCodec,
    ) -> Result<Self, ServerStartupError> {
        let socket_path = socket_path.into();
        let parent = socket_path
            .parent()
            .ok_or_else(|| ServerStartupError::NoParent(socket_path.clone()))?;
        prepare_parent(parent)?;
        remove_abandoned_socket(&socket_path)?;
        let listener =
            UnixListener::bind(&socket_path).map_err(|source| ServerStartupError::Bind {
                path: socket_path.clone(),
                source,
            })?;
        fs::set_permissions(&socket_path, fs::Permissions::from_mode(socket_mode)).map_err(
            |source| ServerStartupError::Permissions {
                path: socket_path.clone(),
                source,
            },
        )?;
        Ok(Self {
            listener,
            socket_path,
            authorizer,
            handler,
            codec,
            connections: Arc::new(Semaphore::new(MAX_CONNECTIONS)),
            connections_by_uid: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub async fn serve(&self) -> Result<(), ServerErrorKind> {
        loop {
            let (stream, _) = self.listener.accept().await?;
            let permit = match Arc::clone(&self.connections).try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => continue,
            };
            let peer = match self.authorizer.authenticate(&stream) {
                Ok(peer) => peer,
                Err(_) => continue,
            };
            let Some(peer_permit) = PeerConnectionPermit::try_acquire(
                Arc::clone(&self.connections_by_uid),
                peer.identity.uid,
            ) else {
                continue;
            };
            let handler = Arc::clone(&self.handler);
            let codec = self.codec;
            tokio::spawn(async move {
                let _permit = permit;
                let _peer_permit = peer_permit;
                let _ = serve_connection(stream, peer.clone(), Arc::clone(&handler), codec).await;
                handler.connection_closed(&peer).await;
            });
        }
    }

    pub fn remove_socket(self) -> Result<(), ServerStartupError> {
        drop(self.listener);
        match fs::remove_file(&self.socket_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(source) => Err(ServerStartupError::Remove {
                path: self.socket_path,
                source,
            }),
        }
    }
}

async fn serve_connection<H: RequestHandler>(
    stream: UnixStream,
    peer: AuthenticatedPeer,
    handler: Arc<H>,
    codec: FrameCodec,
) -> Result<(), ServerErrorKind> {
    let (mut reader, mut writer) = stream.into_split();
    let (sender, mut receiver) = mpsc::channel(CONNECTION_OUTPUT_CAPACITY);
    let disconnected = Arc::new(AtomicBool::new(false));
    let disconnect = Arc::new(Notify::new());
    let sink = ConnectionSink {
        sender,
        disconnected: Arc::clone(&disconnected),
        disconnect: Arc::clone(&disconnect),
    };
    let writer_disconnect = sink.clone();
    let writer_task = tokio::spawn(async move {
        while let Some(outbound) = receiver.recv().await {
            match tokio::time::timeout(
                CONNECTION_WRITE_TIMEOUT,
                codec.write_frame(&mut writer, &outbound.frame),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    writer_disconnect.disconnect();
                    return Err(error);
                }
                Err(_) => {
                    writer_disconnect.disconnect();
                    return Ok(());
                }
            }
            if let Some(written) = outbound.written {
                let _ = written.send(());
            }
        }
        Ok::<(), FrameError>(())
    });

    let first_frame = tokio::select! {
        () = sink.disconnected() => None,
        result = tokio::time::timeout(
            INITIAL_FRAME_TIMEOUT,
            codec.read_frame::<ClientFrame, _>(&mut reader),
        ) => match result {
            Ok(frame) => frame?,
            Err(_) => None,
        },
    };
    let Some(first_frame) = first_frame else {
        sink.disconnect();
        writer_task.abort();
        let _ = writer_task.await;
        return Ok(());
    };
    if !matches!(&first_frame, ClientFrame::Request(_)) {
        sink.disconnect();
        writer_task.abort();
        let _ = writer_task.await;
        return Ok(());
    }
    let mut next_frame = Some(first_frame);
    let mut stream_open = false;
    'connection: loop {
        if disconnected.load(Ordering::Acquire) {
            break;
        }
        let frame = match next_frame.take() {
            Some(frame) => frame,
            None => {
                let frame = tokio::select! {
                    () = sink.disconnected() => break,
                    frame = codec.read_frame::<ClientFrame, _>(&mut reader) => frame?,
                };
                let Some(frame) = frame else { break };
                frame
            }
        };
        match frame {
            ClientFrame::Request(envelope) => {
                if stream_open {
                    break;
                }
                let request_id = envelope.request_id;
                if !ProtocolVersion::CURRENT.is_compatible_with(envelope.version) {
                    sink.send(ServerFrame::Response(Box::new(ResponseEnvelope::error(
                        request_id,
                        ServerError::new(
                            ErrorCode::ProtocolMismatch,
                            format!(
                                "unsupported protocol {}.{}",
                                envelope.version.major, envelope.version.minor
                            ),
                        ),
                    ))))
                    .await?;
                    break;
                }
                let mut close_after_response = false;
                let result = match envelope.request.validate() {
                    Ok(()) => {
                        let request = handler.request(&peer, envelope.request, sink.clone());
                        tokio::pin!(request);
                        tokio::select! {
                            result = &mut request => result,
                            next = codec.read_frame::<ClientFrame, _>(&mut reader) => {
                                close_after_response = true;
                                match next? {
                                    None => break 'connection,
                                    Some(_) => Err(ServerError::new(
                                        ErrorCode::Conflict,
                                        "a connection may have only one request in flight",
                                    )),
                                }
                            }
                            () = sink.disconnected() => break 'connection,
                        }
                    }
                    Err(error) => Err(ServerError::new(
                        ErrorCode::InvalidRequest,
                        error.to_string(),
                    )),
                };
                let opened_stream = result.as_ref().ok().and_then(|response| match response {
                    Response::Subscribed(opened) | Response::Attached(opened) => {
                        Some(opened.stream_id)
                    }
                    _ => None,
                });
                sink.send(ServerFrame::Response(Box::new(ResponseEnvelope {
                    version: ProtocolVersion::CURRENT,
                    request_id,
                    result,
                })))
                .await?;
                if let Some(stream_id) = opened_stream {
                    handler.stream_response_written(&peer, stream_id).await;
                    stream_open = true;
                } else {
                    break;
                }
                if close_after_response {
                    break;
                }
            }
            ClientFrame::StreamInput(frame) => {
                if let Err(error) = frame.input.validate() {
                    let _ = sink.close_stream(
                        frame.stream_id,
                        Some(ServerError::new(
                            ErrorCode::InvalidRequest,
                            error.to_string(),
                        )),
                    );
                    continue;
                }
                if let Err(error) = handler.stream_input(&peer, frame.clone()).await {
                    let _ = sink.close_stream(frame.stream_id, Some(error));
                }
            }
            ClientFrame::CloseStream { stream_id } => {
                handler.close_stream(&peer, stream_id).await;
                let _ = sink.close_stream(stream_id, None);
                break;
            }
        }
    }
    sink.disconnect();
    drop(sink);
    writer_task.abort();
    let _ = writer_task.await;
    Ok(())
}

struct PeerConnectionPermit {
    counts: Arc<Mutex<HashMap<u32, usize>>>,
    uid: u32,
}

impl PeerConnectionPermit {
    fn try_acquire(counts: Arc<Mutex<HashMap<u32, usize>>>, uid: u32) -> Option<Self> {
        {
            let mut current = counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let count = current.entry(uid).or_default();
            if *count >= MAX_CONNECTIONS_PER_UID {
                return None;
            }
            *count += 1;
        }
        Some(Self { counts, uid })
    }
}

impl Drop for PeerConnectionPermit {
    fn drop(&mut self) {
        let mut current = self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(count) = current.get_mut(&self.uid) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            current.remove(&self.uid);
        }
    }
}

fn prepare_parent(parent: &Path) -> Result<(), ServerStartupError> {
    if !parent.exists() {
        fs::create_dir_all(parent).map_err(|source| ServerStartupError::Parent {
            path: parent.to_path_buf(),
            source,
        })?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o750)).map_err(|source| {
            ServerStartupError::Permissions {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }
    let metadata = fs::metadata(parent).map_err(|source| ServerStartupError::Parent {
        path: parent.to_path_buf(),
        source,
    })?;
    if !metadata.is_dir() || metadata.mode() & 0o002 != 0 {
        return Err(ServerStartupError::InsecureParent(parent.to_path_buf()));
    }
    Ok(())
}

fn remove_abandoned_socket(path: &Path) -> Result<(), ServerStartupError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(ServerStartupError::Inspect {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !metadata.file_type().is_socket() {
        return Err(ServerStartupError::NonSocket(path.to_path_buf()));
    }
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(_) => return Err(ServerStartupError::AlreadyRunning(path.to_path_buf())),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
            ) => {}
        Err(source) => {
            return Err(ServerStartupError::Inspect {
                path: path.to_path_buf(),
                source,
            });
        }
    }
    fs::remove_file(path).map_err(|source| ServerStartupError::Remove {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Debug, Error)]
pub enum ServerStartupError {
    #[error("socket path has no parent: {0}")]
    NoParent(PathBuf),
    #[error("cannot prepare socket parent {path}: {source}")]
    Parent {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("socket parent is not a secure directory: {0}")]
    InsecureParent(PathBuf),
    #[error("cannot inspect socket path {path}: {source}")]
    Inspect {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("refusing to replace non-socket path: {0}")]
    NonSocket(PathBuf),
    #[error("another scqd is already listening at {0}")]
    AlreadyRunning(PathBuf),
    #[error("cannot remove abandoned socket {path}: {source}")]
    Remove {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot bind socket {path}: {source}")]
    Bind {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot set permissions on {path}: {source}")]
    Permissions {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Error)]
pub enum ServerErrorKind {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error(transparent)]
    Connection(#[from] ConnectionClosed),
    #[error("connection writer task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("peer authentication failed: {0}")]
    Authentication(#[from] AuthError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_uid_connection_limit_releases_capacity() {
        let counts = Arc::new(Mutex::new(HashMap::from([(
            42,
            MAX_CONNECTIONS_PER_UID - 1,
        )])));
        let permit = PeerConnectionPermit::try_acquire(Arc::clone(&counts), 42)
            .expect("last per-uid permit");
        assert!(PeerConnectionPermit::try_acquire(Arc::clone(&counts), 42).is_none());
        drop(permit);
        assert!(PeerConnectionPermit::try_acquire(counts, 42).is_some());
    }

    #[test]
    fn refuses_to_replace_regular_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("scqd.sock");
        fs::write(&path, "do not replace").expect("file");
        assert!(matches!(
            remove_abandoned_socket(&path),
            Err(ServerStartupError::NonSocket(_))
        ));
        assert_eq!(
            fs::read_to_string(path).expect("contents"),
            "do not replace"
        );
    }

    #[test]
    fn world_writable_parent_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o777)).expect("chmod");
        assert!(matches!(
            prepare_parent(temp.path()),
            Err(ServerStartupError::InsecureParent(_))
        ));
    }
}
