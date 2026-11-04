use super::*;

pub(super) fn empty_node_signature(identity: &NodeRuntimeIdentity) -> DomainSignature {
    DomainSignature {
        authority: SignatureAuthority::NodeSession,
        issuer: identity.node_id().to_owned(),
        key_id: identity.key_id().to_owned(),
        public_key: identity.public_key(),
        signature: Vec::new(),
    }
}

pub(crate) fn clear_signature(signature: &mut DomainSignature) {
    signature.issuer.clear();
    signature.key_id.clear();
    signature.public_key = [0; 32];
    signature.signature.clear();
}

pub(crate) fn validate_acknowledgement(
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

pub(crate) fn validate_event(event: &NodeEvent) -> Result<(), ProtocolError> {
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

pub(crate) fn validate_event_identity(identity: &EventIdentity) -> Result<(), ProtocolError> {
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

pub(crate) fn validate_outcome(outcome: &NodeTerminalOutcome) -> Result<(), ProtocolError> {
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

pub(crate) fn validate_artifact(artifact: &PublishedArtifact) -> Result<(), ProtocolError> {
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

pub(crate) fn validate_message(message: &str, allow_empty: bool) -> Result<(), ProtocolError> {
    if (!allow_empty && message.is_empty()) || message.len() > MAX_EVENT_MESSAGE_BYTES {
        return Err(ProtocolError::InvalidOutbound);
    }
    Ok(())
}
