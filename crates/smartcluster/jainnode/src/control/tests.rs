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

mod admission;
mod supervisor;
