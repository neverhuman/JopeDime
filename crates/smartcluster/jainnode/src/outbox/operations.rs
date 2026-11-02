use crate::{EventSink, ExecutionError, ExecutionReceipt};
use async_trait::async_trait;

use super::*;

impl DurableControlOutbox {
    pub fn open(root: PathBuf, identity: NodeRuntimeIdentity) -> Result<Self, OutboxError> {
        validate_or_create_root(&root)?;
        if root.join(POSITION_PENDING_FILE).exists() || root.join(ENQUEUE_PENDING_FILE).exists() {
            return Err(OutboxError::InvalidState);
        }
        let position = read_position(&root)?;
        let mut pending = BTreeMap::new();
        let mut confirmed_residue = None;
        let mut entries = fs::read_dir(&root)
            .map_err(OutboxError::Io)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(OutboxError::Io)?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        if entries.len() > MAX_OUTBOX_RECORDS.saturating_add(1) {
            return Err(OutboxError::CapacityExceeded);
        }
        for entry in entries {
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| OutboxError::InvalidState)?;
            if name == POSITION_FILE {
                continue;
            }
            let (sequence, filename_digest) = parse_record_name(&name)?;
            let path = entry.path();
            let bytes = read_physical_file(&path, MAX_OUTBOX_RECORD_BYTES)?;
            let stored: StoredOutbound =
                postcard::from_bytes(&bytes).map_err(|_| OutboxError::InvalidState)?;
            let pending_record =
                validate_record(stored, sequence, filename_digest, &identity, path.clone())?;
            if sequence < position.confirmed_sequence {
                return Err(OutboxError::InvalidState);
            }
            if sequence == position.confirmed_sequence {
                if sequence == 0
                    || pending_record.pending.digest != position.confirmed_digest
                    || confirmed_residue.replace(path).is_some()
                {
                    return Err(OutboxError::InvalidState);
                }
                continue;
            }
            if pending.insert(sequence, pending_record).is_some() {
                return Err(OutboxError::Conflict);
            }
        }
        if let Some(path) = confirmed_residue {
            fs::remove_file(path).map_err(OutboxError::Io)?;
            sync_directory(&root)?;
        }
        validate_sequence_continuity(&position, &pending)?;
        if position.confirmed_sequence == 0 && position.confirmed_digest != [0; 32] {
            return Err(OutboxError::InvalidState);
        }
        if position.confirmed_sequence != 0 && position.confirmed_digest == [0; 32] {
            return Err(OutboxError::InvalidState);
        }
        Ok(Self {
            root,
            identity,
            state: Mutex::new(OutboxState { position, pending }),
        })
    }

    pub fn enqueue_acknowledgement(
        &self,
        acknowledgement: WorkerAcknowledgement,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::assignment_accepted(&self.identity, sequence, acknowledgement)
        })
    }

    pub fn contains_acknowledgement(
        &self,
        acknowledgement: &WorkerAcknowledgement,
    ) -> Result<bool, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if state.pending.values().any(|record| {
            matches!(
                &record.pending.frame,
                NodeControlFrame::AssignmentAccepted {
                    acknowledgement: pending,
                    ..
                } if pending.as_ref() == acknowledgement
            )
        }) {
            return Ok(true);
        }
        if state.position.confirmed_sequence == 0 {
            return Ok(false);
        }
        let frame = NodeControlFrame::assignment_accepted(
            &self.identity,
            state.position.confirmed_sequence,
            acknowledgement.clone(),
        )
        .map_err(OutboxError::Protocol)?;
        let digest = sha256(&frame.encode().map_err(OutboxError::Protocol)?);
        Ok(digest == state.position.confirmed_digest)
    }

    pub fn enqueue_event(&self, event: NodeEvent) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| NodeControlFrame::outbound_event(&self.identity, sequence, event))
    }

    pub fn enqueue_storage_telemetry(
        &self,
        telemetry: SignedStorageTelemetry,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::outbound_storage_telemetry(&self.identity, sequence, telemetry)
        })
    }

    pub fn enqueue_storage_fragment_receipt(
        &self,
        receipt: SignedStorageFragmentReceipt,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::outbound_storage_fragment_receipt(&self.identity, sequence, receipt)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        outcome: NodeTerminalOutcome,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue(|sequence| {
            NodeControlFrame::terminal(
                &self.identity,
                sequence,
                run_id,
                attempt,
                attempt_fence,
                control_position,
                at_ms,
                outcome,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal_success(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        receipt: ExecutionReceipt,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue_terminal(
            run_id,
            attempt,
            attempt_fence,
            control_position,
            at_ms,
            NodeTerminalOutcome::Success(receipt),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_terminal_failure(
        &self,
        run_id: String,
        attempt: u32,
        attempt_fence: u64,
        control_position: u64,
        at_ms: u64,
        error: &ExecutionError,
    ) -> Result<OutboundPending, OutboxError> {
        self.enqueue_terminal(
            run_id,
            attempt,
            attempt_fence,
            control_position,
            at_ms,
            NodeTerminalOutcome::failure(error),
        )
    }

    pub fn oldest(&self) -> Result<Option<OutboundPending>, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        Ok(state
            .pending
            .first_key_value()
            .map(|(_, record)| record.pending.clone()))
    }

    pub fn pending_count(&self) -> Result<usize, OutboxError> {
        let state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        Ok(state.pending.len())
    }

    pub(crate) fn verify_session_hello_identity(
        &self,
        frame: &NodeControlFrame,
    ) -> Result<(), OutboxError> {
        let NodeControlFrame::SessionHello {
            node_id,
            session_id,
            signature,
            ..
        } = frame
        else {
            return Err(OutboxError::Conflict);
        };
        if node_id != self.identity.node_id()
            || session_id != self.identity.session_id()
            || signature.key_id != self.identity.key_id()
            || signature.public_key != self.identity.public_key()
        {
            return Err(OutboxError::Conflict);
        }
        Ok(())
    }

    pub fn confirm_controller(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<OutboundConfirmation, OutboxError> {
        let (sequence, digest) = frame
            .verify_outbound_confirmation(
                self.identity.session_id(),
                self.identity.hub_issuer(),
                self.identity.hub_public_key(),
                now_ms,
                maximum_clock_skew_ms,
            )
            .map_err(OutboxError::Protocol)?;
        self.retire_exact(sequence, digest)
    }

    pub(super) fn retire_exact(
        &self,
        sequence: u64,
        digest: Digest32,
    ) -> Result<OutboundConfirmation, OutboxError> {
        let mut state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if sequence != 0
            && digest != [0; 32]
            && sequence == state.position.confirmed_sequence
            && digest == state.position.confirmed_digest
        {
            return Ok(OutboundConfirmation::AlreadyConfirmed);
        }
        let Some((&oldest_sequence, record)) = state.pending.first_key_value() else {
            return Err(OutboxError::Conflict);
        };
        if sequence != oldest_sequence || digest != record.pending.digest || digest == [0; 32] {
            return Err(OutboxError::Conflict);
        }
        let path = record.path.clone();
        let next_position = OutboxPosition {
            schema: POSITION_SCHEMA.to_owned(),
            confirmed_sequence: sequence,
            confirmed_digest: digest,
            checksum: position_checksum(sequence, digest),
        };
        persist_position(&self.root, &next_position)?;
        fs::remove_file(path).map_err(OutboxError::Io)?;
        sync_directory(&self.root)?;
        state.position = next_position;
        state.pending.remove(&sequence);
        Ok(OutboundConfirmation::Retired)
    }

    fn enqueue<F>(&self, build: F) -> Result<OutboundPending, OutboxError>
    where
        F: FnOnce(u64) -> Result<NodeControlFrame, ProtocolError>,
    {
        let mut state = self.state.lock().map_err(|_| OutboxError::InvalidState)?;
        if state.pending.len() >= MAX_OUTBOX_RECORDS {
            return Err(OutboxError::CapacityExceeded);
        }
        let sequence = state
            .position
            .confirmed_sequence
            .checked_add(
                u64::try_from(state.pending.len()).map_err(|_| OutboxError::CapacityExceeded)?,
            )
            .and_then(|sequence| sequence.checked_add(1))
            .ok_or(OutboxError::CapacityExceeded)?;
        let frame = build(sequence).map_err(OutboxError::Protocol)?;
        frame
            .verify_outbound(
                self.identity.node_id(),
                self.identity.session_id(),
                self.identity.key_id(),
                self.identity.public_key(),
            )
            .map_err(OutboxError::Protocol)?;
        if frame.outbound_sequence() != Some(sequence) {
            return Err(OutboxError::InvalidState);
        }
        let frame_bytes = frame.encode().map_err(OutboxError::Protocol)?;
        let digest = sha256(&frame_bytes);
        let stored = StoredOutbound {
            schema: OUTBOX_SCHEMA.to_owned(),
            sequence,
            digest,
            frame_bytes,
        };
        let bytes = postcard::to_stdvec(&stored).map_err(|_| OutboxError::InvalidState)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_OUTBOX_RECORD_BYTES {
            return Err(OutboxError::CapacityExceeded);
        }
        let path = self.root.join(record_name(sequence, digest));
        let pending_path = self.root.join(ENQUEUE_PENDING_FILE);
        write_new_synced(&pending_path, &bytes)?;
        fs::rename(&pending_path, &path).map_err(OutboxError::Io)?;
        sync_directory(&self.root)?;
        let pending = OutboundPending {
            sequence,
            digest,
            frame,
        };
        let record = PendingRecord {
            pending: pending.clone(),
            path,
        };
        if state.pending.insert(sequence, record).is_some() {
            return Err(OutboxError::Conflict);
        }
        Ok(pending)
    }
}

#[async_trait]
impl EventSink for DurableControlOutbox {
    async fn emit(&self, event: NodeEvent) -> Result<(), ExecutionError> {
        self.enqueue_event(event)
            .map(|_| ())
            .map_err(|_| ExecutionError::OutboundControlUnavailable)
    }
}
