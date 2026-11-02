use super::validation::empty_node_signature;
use super::*;

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
