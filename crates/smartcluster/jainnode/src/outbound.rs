use jain_smartcluster_fabric::{
    sha256, valid_identifier, Digest32, DomainSignature, SignatureAuthority, WorkerAcknowledgement,
};
use serde::{Deserialize, Serialize};

use crate::{
    EventIdentity, ExecutionError, ExecutionReceipt, NodeControlFrame, NodeEvent,
    NodeRuntimeIdentity, ProtocolError, PublishedArtifact, SignedStorageFragmentReceipt,
    SignedStorageTelemetry, MAX_STREAM_CHUNK_BYTES,
};

const ASSIGNMENT_ACCEPTED_DOMAIN: &[u8] = b"jain.node.assignment-accepted.v1\0";
const EVENT_DOMAIN: &[u8] = b"jain.node.event.v1\0";
const TERMINAL_DOMAIN: &[u8] = b"jain.node.terminal.v1\0";
const STORAGE_TELEMETRY_FRAME_DOMAIN: &[u8] = b"jain.node.storage-telemetry-frame.v1\0";
const STORAGE_FRAGMENT_RECEIPT_FRAME_DOMAIN: &[u8] =
    b"jain.node.storage-fragment-receipt-frame.v1\0";
const MAX_EVENT_MESSAGE_BYTES: usize = 1024;
const MAX_EVENT_ARTIFACTS: usize = 128;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeFailureCode {
    UnsupportedHost,
    InvalidWorkUnit,
    InvalidStaging,
    NetworkPolicyUnavailable,
    DurabilityUnavailable,
    Cancelled,
    CancellationStateInvalid,
    DeadlineExceeded,
    MemoryLimitExceeded,
    NonZeroExit(i32),
    LogBudgetExceeded,
    StreamProtocolViolation,
    ClockUnavailable,
    StaleAuthority,
    AssignmentStateInvalid,
    IsolationUnavailable,
    StorageUnavailable,
    NetworkUnavailable,
    ContainerRuntimeUnavailable,
    OutboundControlUnavailable,
}

impl From<&ExecutionError> for NodeFailureCode {
    fn from(error: &ExecutionError) -> Self {
        match error {
            ExecutionError::UnsupportedHost => Self::UnsupportedHost,
            ExecutionError::InvalidWorkUnit => Self::InvalidWorkUnit,
            ExecutionError::InvalidStaging => Self::InvalidStaging,
            ExecutionError::NetworkPolicyUnavailable => Self::NetworkPolicyUnavailable,
            ExecutionError::DurabilityUnavailable => Self::DurabilityUnavailable,
            ExecutionError::Cancelled => Self::Cancelled,
            ExecutionError::CancellationState => Self::CancellationStateInvalid,
            ExecutionError::DeadlineExceeded => Self::DeadlineExceeded,
            ExecutionError::MemoryLimitExceeded => Self::MemoryLimitExceeded,
            ExecutionError::NonZeroExit(code) => Self::NonZeroExit(*code),
            ExecutionError::LogBudgetExceeded => Self::LogBudgetExceeded,
            ExecutionError::StreamProtocolViolation => Self::StreamProtocolViolation,
            ExecutionError::ClockUnavailable => Self::ClockUnavailable,
            ExecutionError::StaleAuthority => Self::StaleAuthority,
            ExecutionError::Assignment(_) => Self::AssignmentStateInvalid,
            ExecutionError::IsolationBroker(_) => Self::IsolationUnavailable,
            ExecutionError::Storage(_) => Self::StorageUnavailable,
            ExecutionError::Network(_) => Self::NetworkUnavailable,
            ExecutionError::Docker(_) => Self::ContainerRuntimeUnavailable,
            ExecutionError::OutboundControlUnavailable => Self::OutboundControlUnavailable,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeTerminalOutcome {
    Success(ExecutionReceipt),
    Failure {
        code: NodeFailureCode,
        detail_digest: Digest32,
    },
}

impl NodeTerminalOutcome {
    pub fn failure(error: &ExecutionError) -> Self {
        Self::Failure {
            code: NodeFailureCode::from(error),
            detail_digest: sha256(error.to_string().as_bytes()),
        }
    }
}

impl NodeControlFrame {
    pub fn assignment_accepted(
        identity: &NodeRuntimeIdentity,
        outbound_sequence: u64,
        acknowledgement: WorkerAcknowledgement,
    ) -> Result<Self, ProtocolError> {
        let mut frame = Self::AssignmentAccepted {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            outbound_sequence,
            acknowledgement: Box::new(acknowledgement),
            signature: empty_node_signature(identity),
        };
        frame.validate_outbound_shape(identity.node_id(), identity.session_id())?;
        frame.sign_outbound(identity)?;
        frame.verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )?;
        Ok(frame)
    }

    pub fn outbound_event(
        identity: &NodeRuntimeIdentity,
        outbound_sequence: u64,
        event: NodeEvent,
    ) -> Result<Self, ProtocolError> {
        let mut frame = Self::Event {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            outbound_sequence,
            event: Box::new(event),
            signature: empty_node_signature(identity),
        };
        frame.validate_outbound_shape(identity.node_id(), identity.session_id())?;
        frame.sign_outbound(identity)?;
        frame.verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )?;
        Ok(frame)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn terminal(
        identity: &NodeRuntimeIdentity,
        outbound_sequence: u64,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        outcome: NodeTerminalOutcome,
    ) -> Result<Self, ProtocolError> {
        let mut frame = Self::Terminal {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            outbound_sequence,
            run_id,
            attempt,
            attempt_fence,
            control_position,
            at_ms,
            outcome: Box::new(outcome),
            signature: empty_node_signature(identity),
        };
        frame.validate_outbound_shape(identity.node_id(), identity.session_id())?;
        frame.sign_outbound(identity)?;
        frame.verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )?;
        Ok(frame)
    }

    pub fn outbound_storage_telemetry(
        identity: &NodeRuntimeIdentity,
        outbound_sequence: u64,
        telemetry: SignedStorageTelemetry,
    ) -> Result<Self, ProtocolError> {
        let mut frame = Self::StorageTelemetry {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            outbound_sequence,
            telemetry: Box::new(telemetry),
            signature: empty_node_signature(identity),
        };
        frame.validate_outbound_shape(identity.node_id(), identity.session_id())?;
        frame.sign_outbound(identity)?;
        frame.verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )?;
        Ok(frame)
    }

    pub fn outbound_storage_fragment_receipt(
        identity: &NodeRuntimeIdentity,
        outbound_sequence: u64,
        receipt: SignedStorageFragmentReceipt,
    ) -> Result<Self, ProtocolError> {
        let mut frame = Self::StorageFragmentReceipt {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            outbound_sequence,
            receipt: Box::new(receipt),
            signature: empty_node_signature(identity),
        };
        frame.validate_outbound_shape(identity.node_id(), identity.session_id())?;
        frame.sign_outbound(identity)?;
        frame.verify_outbound(
            identity.node_id(),
            identity.session_id(),
            identity.key_id(),
            identity.public_key(),
        )?;
        Ok(frame)
    }

    pub fn outbound_sequence(&self) -> Option<u64> {
        match self {
            Self::AssignmentAccepted {
                outbound_sequence, ..
            }
            | Self::Event {
                outbound_sequence, ..
            }
            | Self::Terminal {
                outbound_sequence, ..
            }
            | Self::StorageTelemetry {
                outbound_sequence, ..
            }
            | Self::StorageFragmentReceipt {
                outbound_sequence, ..
            } => Some(*outbound_sequence),
            _ => None,
        }
    }

    pub fn signed_outbound_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let (domain, mut unsigned) = match self {
            Self::AssignmentAccepted { .. } => (ASSIGNMENT_ACCEPTED_DOMAIN, self.clone()),
            Self::Event { .. } => (EVENT_DOMAIN, self.clone()),
            Self::Terminal { .. } => (TERMINAL_DOMAIN, self.clone()),
            Self::StorageTelemetry { .. } => (STORAGE_TELEMETRY_FRAME_DOMAIN, self.clone()),
            Self::StorageFragmentReceipt { .. } => {
                (STORAGE_FRAGMENT_RECEIPT_FRAME_DOMAIN, self.clone())
            }
            _ => return Err(ProtocolError::InvalidOutbound),
        };
        match &mut unsigned {
            Self::AssignmentAccepted { signature, .. }
            | Self::Event { signature, .. }
            | Self::Terminal { signature, .. }
            | Self::StorageTelemetry { signature, .. }
            | Self::StorageFragmentReceipt { signature, .. } => clear_signature(signature),
            _ => return Err(ProtocolError::InvalidOutbound),
        }
        let payload = postcard::to_stdvec(&unsigned).map_err(|_| ProtocolError::InvalidOutbound)?;
        let mut bytes = Vec::with_capacity(domain.len() + payload.len());
        bytes.extend_from_slice(domain);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn verify_outbound(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
        expected_key_id: &str,
        expected_public_key: Digest32,
    ) -> Result<(), ProtocolError> {
        if !valid_identifier(expected_node_id)
            || !valid_identifier(expected_session_id)
            || !valid_identifier(expected_key_id)
            || expected_public_key == [0; 32]
        {
            return Err(ProtocolError::InvalidOutbound);
        }
        self.validate_outbound_shape(expected_node_id, expected_session_id)?;
        if let Self::AssignmentAccepted {
            acknowledgement, ..
        } = self
        {
            acknowledgement
                .signature
                .verify_exact(
                    SignatureAuthority::WorkerAcknowledgement,
                    expected_node_id,
                    expected_public_key,
                    &acknowledgement
                        .signed_bytes()
                        .map_err(|_| ProtocolError::InvalidOutbound)?,
                )
                .map_err(|_| ProtocolError::InvalidOutbound)?;
        }
        match self {
            Self::StorageTelemetry { telemetry, .. } => telemetry.verify(
                expected_node_id,
                expected_session_id,
                expected_key_id,
                expected_public_key,
            )?,
            Self::StorageFragmentReceipt { receipt, .. } => receipt.verify(
                expected_node_id,
                expected_session_id,
                expected_key_id,
                expected_public_key,
            )?,
            _ => {}
        }
        let signature = self.outbound_signature()?;
        if signature.key_id != expected_key_id {
            return Err(ProtocolError::InvalidOutbound);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                expected_node_id,
                expected_public_key,
                &self.signed_outbound_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidOutbound)
    }

    fn sign_outbound(&mut self, identity: &NodeRuntimeIdentity) -> Result<(), ProtocolError> {
        let bytes = self.signed_outbound_bytes()?;
        let signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            identity.node_id().to_owned(),
            identity.key_id().to_owned(),
            identity.signing_key(),
            &bytes,
        );
        *self.outbound_signature_mut()? = signature;
        Ok(())
    }

    fn validate_outbound_shape(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
    ) -> Result<(), ProtocolError> {
        let (node_id, session_id, sequence) = match self {
            Self::AssignmentAccepted {
                node_id,
                session_id,
                outbound_sequence,
                acknowledgement,
                ..
            } => {
                validate_acknowledgement(acknowledgement, expected_node_id, expected_session_id)?;
                (node_id, session_id, outbound_sequence)
            }
            Self::Event {
                node_id,
                session_id,
                outbound_sequence,
                event,
                ..
            } => {
                validate_event(event)?;
                (node_id, session_id, outbound_sequence)
            }
            Self::Terminal {
                node_id,
                session_id,
                outbound_sequence,
                run_id,
                attempt,
                attempt_fence,
                control_position,
                at_ms,
                outcome,
                ..
            } => {
                if !valid_identifier(run_id)
                    || *attempt == 0
                    || *attempt_fence == 0
                    || *control_position == 0
                    || *at_ms == 0
                {
                    return Err(ProtocolError::InvalidOutbound);
                }
                validate_outcome(outcome)?;
                (node_id, session_id, outbound_sequence)
            }
            Self::StorageTelemetry {
                node_id,
                session_id,
                outbound_sequence,
                telemetry,
                ..
            } => {
                telemetry.validate_shape(expected_node_id, expected_session_id)?;
                (node_id, session_id, outbound_sequence)
            }
            Self::StorageFragmentReceipt {
                node_id,
                session_id,
                outbound_sequence,
                receipt,
                ..
            } => {
                receipt.validate_shape(expected_node_id, expected_session_id)?;
                (node_id, session_id, outbound_sequence)
            }
            _ => return Err(ProtocolError::InvalidOutbound),
        };
        if node_id != expected_node_id || session_id != expected_session_id || *sequence == 0 {
            return Err(ProtocolError::InvalidOutbound);
        }
        Ok(())
    }

    fn outbound_signature(&self) -> Result<&DomainSignature, ProtocolError> {
        match self {
            Self::AssignmentAccepted { signature, .. }
            | Self::Event { signature, .. }
            | Self::Terminal { signature, .. }
            | Self::StorageTelemetry { signature, .. }
            | Self::StorageFragmentReceipt { signature, .. } => Ok(signature),
            _ => Err(ProtocolError::InvalidOutbound),
        }
    }

    fn outbound_signature_mut(&mut self) -> Result<&mut DomainSignature, ProtocolError> {
        match self {
            Self::AssignmentAccepted { signature, .. }
            | Self::Event { signature, .. }
            | Self::Terminal { signature, .. }
            | Self::StorageTelemetry { signature, .. }
            | Self::StorageFragmentReceipt { signature, .. } => Ok(signature),
            _ => Err(ProtocolError::InvalidOutbound),
        }
    }
}

fn empty_node_signature(identity: &NodeRuntimeIdentity) -> DomainSignature {
    DomainSignature {
        authority: SignatureAuthority::NodeSession,
        issuer: identity.node_id().to_owned(),
        key_id: identity.key_id().to_owned(),
        public_key: identity.public_key(),
        signature: Vec::new(),
    }
}

fn clear_signature(signature: &mut DomainSignature) {
    signature.issuer.clear();
    signature.key_id.clear();
    signature.public_key = [0; 32];
    signature.signature.clear();
}

fn validate_acknowledgement(
    acknowledgement: &WorkerAcknowledgement,
    expected_node_id: &str,
    expected_session_id: &str,
) -> Result<(), ProtocolError> {
    if acknowledgement.node_id != expected_node_id
        || acknowledgement.session_id != expected_session_id
        || acknowledgement.work_unit_digest == [0; 32]
        || acknowledgement.placement_digest == [0; 32]
        || acknowledgement.assignment_journal_digest == [0; 32]
        || acknowledgement.attempt == 0
        || acknowledgement.attempt_fence == 0
        || acknowledgement.hub_epoch == 0
        || acknowledgement.control_position == 0
        || acknowledgement.acknowledged_at_ms == 0
    {
        return Err(ProtocolError::InvalidOutbound);
    }
    acknowledgement
        .signature
        .verify_exact(
            SignatureAuthority::WorkerAcknowledgement,
            expected_node_id,
            acknowledgement.signature.public_key,
            &acknowledgement
                .signed_bytes()
                .map_err(|_| ProtocolError::InvalidOutbound)?,
        )
        .map_err(|_| ProtocolError::InvalidOutbound)
}

fn validate_event(event: &NodeEvent) -> Result<(), ProtocolError> {
    validate_event_identity(event.identity())?;
    match event {
        NodeEvent::SmokeStarted(_) | NodeEvent::SmokePassed(_) | NodeEvent::Metrics { .. } => {}
        NodeEvent::SmokeFailed { reason, .. } => validate_message(reason, false)?,
        NodeEvent::StdoutChunk { bytes, .. } | NodeEvent::StderrChunk { bytes, .. } => {
            if bytes.is_empty() || bytes.len() > MAX_STREAM_CHUNK_BYTES {
                return Err(ProtocolError::InvalidOutbound);
            }
        }
        NodeEvent::Progress {
            completed,
            total,
            message,
            ..
        } => {
            if total.is_some_and(|total| *completed > total) {
                return Err(ProtocolError::InvalidOutbound);
            }
            validate_message(message, true)?;
        }
        NodeEvent::ArtifactPublished { artifact, .. } => validate_artifact(artifact)?,
        NodeEvent::StreamGap {
            first_available_cursor,
            durable_log_artifact_ids,
            ..
        } => {
            if *first_available_cursor == 0
                || durable_log_artifact_ids.is_empty()
                || durable_log_artifact_ids.len() > MAX_EVENT_ARTIFACTS
                || durable_log_artifact_ids
                    .iter()
                    .any(|artifact_id| !valid_identifier(artifact_id))
            {
                return Err(ProtocolError::InvalidOutbound);
            }
        }
    }
    Ok(())
}

fn validate_event_identity(identity: &EventIdentity) -> Result<(), ProtocolError> {
    if !valid_identifier(&identity.run_id)
        || identity.attempt == 0
        || identity.fence == 0
        || identity.cursor == 0
        || identity.control_position == 0
    {
        return Err(ProtocolError::InvalidOutbound);
    }
    Ok(())
}

fn validate_outcome(outcome: &NodeTerminalOutcome) -> Result<(), ProtocolError> {
    match outcome {
        NodeTerminalOutcome::Success(receipt) => {
            if receipt.outputs.len() > MAX_EVENT_ARTIFACTS
                || receipt.logs.len() > MAX_EVENT_ARTIFACTS
                || receipt
                    .outputs
                    .iter()
                    .any(|artifact| validate_artifact(artifact).is_err())
                || receipt
                    .logs
                    .iter()
                    .any(|artifact| validate_artifact(artifact).is_err())
            {
                return Err(ProtocolError::InvalidOutbound);
            }
        }
        NodeTerminalOutcome::Failure {
            code,
            detail_digest,
        } => {
            if matches!(code, NodeFailureCode::NonZeroExit(0)) || *detail_digest == [0; 32] {
                return Err(ProtocolError::InvalidOutbound);
            }
        }
    }
    Ok(())
}

fn validate_artifact(artifact: &PublishedArtifact) -> Result<(), ProtocolError> {
    if !valid_identifier(&artifact.artifact_id)
        || artifact.relative_path.is_empty()
        || artifact.relative_path.starts_with('/')
        || artifact.relative_path.contains('\0')
        || artifact
            .relative_path
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
        || artifact.digest == [0; 32]
        || artifact.durable_receipt_digest == [0; 32]
        || artifact.writable_domains == 0
    {
        return Err(ProtocolError::InvalidOutbound);
    }
    Ok(())
}

fn validate_message(message: &str, allow_empty: bool) -> Result<(), ProtocolError> {
    if (!allow_empty && message.is_empty()) || message.len() > MAX_EVENT_MESSAGE_BYTES {
        return Err(ProtocolError::InvalidOutbound);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use jain_smartcluster_fabric::DomainSignature;

    use super::*;
    use crate::{ExecutionPhase, NodeProfile};

    fn identity() -> NodeRuntimeIdentity {
        NodeRuntimeIdentity::new(
            "node-1".to_owned(),
            "session-1".to_owned(),
            "node-key-1".to_owned(),
            "hub-issuer".to_owned(),
            [3; 32],
            "image-issuer".to_owned(),
            [4; 32],
            SigningKey::from_bytes(&[61; 32]),
        )
        .unwrap()
    }

    fn event() -> NodeEvent {
        NodeEvent::StdoutChunk {
            identity: EventIdentity {
                run_id: "run-1".to_owned(),
                attempt: 1,
                fence: 2,
                cursor: 3,
                stream_offset: 0,
                control_position: 4,
                phase: ExecutionPhase::Run,
            },
            bytes: b"bounded".to_vec(),
        }
    }

    fn acknowledgement(identity: &NodeRuntimeIdentity) -> WorkerAcknowledgement {
        let mut acknowledgement = WorkerAcknowledgement {
            node_id: identity.node_id().to_owned(),
            session_id: identity.session_id().to_owned(),
            work_unit_digest: [1; 32],
            placement_digest: [2; 32],
            assignment_journal_digest: [3; 32],
            attempt: 1,
            attempt_fence: 2,
            hub_epoch: 3,
            control_position: 4,
            acknowledged_at_ms: 5,
            signature: DomainSignature {
                authority: SignatureAuthority::WorkerAcknowledgement,
                issuer: identity.node_id().to_owned(),
                key_id: "jainnode-session".to_owned(),
                public_key: identity.public_key(),
                signature: Vec::new(),
            },
        };
        acknowledgement.signature = DomainSignature::sign(
            SignatureAuthority::WorkerAcknowledgement,
            identity.node_id().to_owned(),
            "jainnode-session".to_owned(),
            identity.signing_key(),
            &acknowledgement.signed_bytes().unwrap(),
        );
        acknowledgement
    }

    #[test]
    fn outbound_frames_bind_node_fence_cursor_sequence_and_domain() {
        let identity = identity();
        let frame = NodeControlFrame::outbound_event(&identity, 7, event()).unwrap();
        frame
            .verify_outbound(
                identity.node_id(),
                identity.session_id(),
                identity.key_id(),
                identity.public_key(),
            )
            .unwrap();
        let canonical = frame.signed_outbound_bytes().unwrap();

        let mut tampered = frame.clone();
        let NodeControlFrame::Event {
            outbound_sequence, ..
        } = &mut tampered
        else {
            unreachable!();
        };
        *outbound_sequence += 1;
        assert!(tampered
            .verify_outbound(
                identity.node_id(),
                identity.session_id(),
                identity.key_id(),
                identity.public_key(),
            )
            .is_err());

        let mut tampered_cursor = frame.clone();
        let NodeControlFrame::Event {
            event: boxed_event, ..
        } = &mut tampered_cursor
        else {
            unreachable!();
        };
        let NodeEvent::StdoutChunk {
            identity: event_identity,
            ..
        } = boxed_event.as_mut()
        else {
            unreachable!();
        };
        event_identity.cursor += 1;
        event_identity.fence += 1;
        assert!(tampered_cursor
            .verify_outbound(
                identity.node_id(),
                identity.session_id(),
                identity.key_id(),
                identity.public_key(),
            )
            .is_err());

        let mut wrong_domain = frame;
        let NodeControlFrame::Event { signature, .. } = &mut wrong_domain else {
            unreachable!();
        };
        *signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            identity.node_id().to_owned(),
            identity.key_id().to_owned(),
            identity.signing_key(),
            &[b"wrong-domain".as_slice(), &canonical[EVENT_DOMAIN.len()..]].concat(),
        );
        assert!(wrong_domain
            .verify_outbound(
                identity.node_id(),
                identity.session_id(),
                identity.key_id(),
                identity.public_key(),
            )
            .is_err());

        let mut oversized = event();
        let NodeEvent::StdoutChunk { bytes, .. } = &mut oversized else {
            unreachable!();
        };
        *bytes = vec![0; MAX_STREAM_CHUNK_BYTES + 1];
        assert!(NodeControlFrame::outbound_event(&identity, 8, oversized).is_err());
    }

    #[test]
    fn terminal_failure_is_closed_and_detail_is_digest_only() {
        let identity = identity();
        let outcome = NodeTerminalOutcome::failure(&ExecutionError::NonZeroExit(19));
        assert!(matches!(
            outcome,
            NodeTerminalOutcome::Failure {
                code: NodeFailureCode::NonZeroExit(19),
                ..
            }
        ));
        let frame =
            NodeControlFrame::terminal(&identity, 8, "run-1".to_owned(), 1, 2, 4, 9, outcome)
                .unwrap();
        frame
            .verify_outbound(
                identity.node_id(),
                identity.session_id(),
                identity.key_id(),
                identity.public_key(),
            )
            .unwrap();
        assert!(!String::from_utf8_lossy(&frame.encode().unwrap()).contains("process exited"));

        let impossible = NodeTerminalOutcome::Failure {
            code: NodeFailureCode::NonZeroExit(0),
            detail_digest: [7; 32],
        };
        assert!(NodeControlFrame::terminal(
            &identity,
            9,
            "run-1".to_owned(),
            1,
            2,
            4,
            9,
            impossible,
        )
        .is_err());
    }

    #[test]
    fn acknowledgement_cannot_be_self_signed_by_a_different_node_key() {
        let identity = identity();
        let alternate = NodeRuntimeIdentity::new(
            "node-1".to_owned(),
            "session-1".to_owned(),
            "alternate-key".to_owned(),
            "hub-issuer".to_owned(),
            [3; 32],
            "image-issuer".to_owned(),
            [4; 32],
            SigningKey::from_bytes(&[62; 32]),
        )
        .unwrap();
        assert!(
            NodeControlFrame::assignment_accepted(&identity, 1, acknowledgement(&alternate),)
                .is_err()
        );
    }

    #[test]
    fn append_only_discriminants_keep_existing_v1_prefixes() {
        let identity = identity();
        let signature = empty_node_signature(&identity);
        let enrollment = NodeControlFrame::Enroll {
            ticket: crate::EnrollmentTicket {
                node_id: "node-1".to_owned(),
                nonce: [1; 32],
                issued_at_ms: 1,
                expires_at_ms: 2,
                signature: signature.clone(),
            },
        };
        let hello = identity
            .session_hello(
                NodeProfile::Cpu,
                crate::WorkerProvision::CodeRunner,
                [7; 32],
                1,
                1,
            )
            .unwrap();
        let cancel = NodeControlFrame::Cancel {
            session_id: "session-1".to_owned(),
            sequence: 1,
            run_id: "run-1".to_owned(),
            attempt: 1,
            attempt_fence: 1,
            signature: signature.clone(),
        };
        let rotate = NodeControlFrame::Rotate {
            session_id: "session-1".to_owned(),
            sequence: 1,
            next_session_id: "session-2".to_owned(),
            expires_at_ms: 2,
            signature: signature.clone(),
        };
        let revoke = NodeControlFrame::Revoke {
            session_id: "session-1".to_owned(),
            sequence: 1,
            reason: "closed".to_owned(),
            signature,
        };
        assert_eq!(enrollment.encode().unwrap()[0], 0);
        assert_eq!(hello.encode().unwrap()[0], 1);
        assert_eq!(cancel.encode().unwrap()[0], 3);
        assert_eq!(rotate.encode().unwrap()[0], 4);
        assert_eq!(revoke.encode().unwrap()[0], 5);
        assert_eq!(
            NodeControlFrame::assignment_accepted(&identity, 1, acknowledgement(&identity))
                .unwrap()
                .encode()
                .unwrap()[0],
            6
        );
        assert_eq!(
            NodeControlFrame::outbound_event(&identity, 2, event())
                .unwrap()
                .encode()
                .unwrap()[0],
            7
        );
        assert_eq!(
            NodeControlFrame::terminal(
                &identity,
                3,
                "run-1".to_owned(),
                1,
                2,
                4,
                5,
                NodeTerminalOutcome::failure(&ExecutionError::Cancelled),
            )
            .unwrap()
            .encode()
            .unwrap()[0],
            8
        );
        assert_eq!(
            NodeControlFrame::outbound_confirmation(
                "session-1".to_owned(),
                3,
                [9; 32],
                5,
                "hub-issuer".to_owned(),
                "hub-key".to_owned(),
                &SigningKey::from_bytes(&[11; 32]),
            )
            .unwrap()
            .encode()
            .unwrap()[0],
            9
        );
    }
}
