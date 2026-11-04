use super::*;

#[derive(Debug)]
pub struct NodeProtocolState {
    root: PathBuf,
    node_id: String,
    session_id: String,
    profile: NodeProfile,
    provision: WorkerProvision,
    issuer: String,
    hub_public_key: Digest32,
    image_issuer: String,
    image_public_key: Digest32,
    max_inflight: u32,
    inflight: u32,
    read_only_fuse_enabled: bool,
    session_state: crate::session_state::DurableSessionState,
    revoked: bool,
}

impl NodeProtocolState {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        root: PathBuf,
        node_id: String,
        session_id: String,
        profile: NodeProfile,
        provision: WorkerProvision,
        issuer: String,
        hub_public_key: Digest32,
        image_issuer: String,
        image_public_key: Digest32,
        max_inflight: u32,
    ) -> Result<Self, ProtocolError> {
        if !root.is_absolute()
            || profile == NodeProfile::StorageOnly
            || !valid_identifier(&node_id)
            || !valid_identifier(&session_id)
            || !valid_identifier(&issuer)
            || !valid_identifier(&image_issuer)
            || hub_public_key == [0; 32]
            || image_public_key == [0; 32]
            || max_inflight == 0
        {
            return Err(ProtocolError::InvalidConfiguration);
        }
        fs::create_dir_all(&root).map_err(ProtocolError::Io)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).map_err(ProtocolError::Io)?;
        if root.canonicalize().map_err(ProtocolError::Io)? != root || {
            let metadata = fs::symlink_metadata(&root).map_err(ProtocolError::Io)?;
            metadata.file_type().is_symlink() || !metadata.is_dir() || metadata.mode() & 0o022 != 0
        } {
            return Err(ProtocolError::InvalidConfiguration);
        }
        let session_state = crate::session_state::load_or_create(&root, &node_id, &session_id)?;
        let session_id = session_state.current_session_id.clone();
        let revoked = session_state.revoked;
        Ok(Self {
            root,
            node_id,
            session_id,
            profile,
            provision,
            issuer,
            hub_public_key,
            image_issuer,
            image_public_key,
            max_inflight,
            inflight: 0,
            read_only_fuse_enabled: false,
            session_state,
            revoked,
        })
    }

    pub fn read_only_fuse_enabled(&self) -> bool {
        self.read_only_fuse_enabled
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub const fn revoked(&self) -> bool {
        self.revoked
    }

    pub const fn control_position(&self) -> u64 {
        self.session_state.control_position
    }

    pub fn consume_enrollment(
        &mut self,
        ticket: &EnrollmentTicket,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        ticket.verify(&self.node_id, &self.issuer, self.hub_public_key, now_ms)?;
        let consumed = self
            .root
            .join(format!("enrollment-{}.consumed", hex_digest(ticket.nonce)));
        write_new_synced(&self.root, &consumed, &ticket.signed_bytes()?).map_err(
            |error| match error {
                ProtocolError::Io(io) if io.kind() == std::io::ErrorKind::AlreadyExists => {
                    ProtocolError::EnrollmentReplayed
                }
                other => other,
            },
        )
    }

    pub fn verify_infer(
        &self,
        frame: NodeControlFrame,
        now_ms: u64,
    ) -> Result<AdmittedInfer, ProtocolError> {
        if self.revoked || self.profile == NodeProfile::StorageOnly {
            return Err(ProtocolError::SessionRevoked);
        }
        let NodeControlFrame::Infer { request } = frame else {
            return Err(ProtocolError::MalformedFrame);
        };
        let InferRequest {
            session_id,
            sequence,
            deadline_ms,
            work,
            placement,
        } = *request;
        if session_id != self.session_id
            || deadline_ms <= now_ms
            || deadline_ms > placement.lease_expires_at_ms
            || self.inflight >= self.max_inflight
        {
            return Err(ProtocolError::AdmissionDenied);
        }
        let work = VerifiedWorkUnit::verify(
            work,
            &self.issuer,
            self.hub_public_key,
            &self.image_issuer,
            self.image_public_key,
        )
        .map_err(|_| ProtocolError::InvalidWorkUnit)?;
        if !self.provision.admits(work.workload_class) {
            return Err(ProtocolError::AdmissionDenied);
        }
        if matches!(
            (self.profile, work.gpu.is_some()),
            (NodeProfile::Cpu, true) | (NodeProfile::Gpu, false)
        ) {
            return Err(ProtocolError::AdmissionDenied);
        }
        placement
            .verify_for(
                &work,
                &self.node_id,
                &self.session_id,
                self.hub_public_key,
                now_ms,
            )
            .map_err(|_| ProtocolError::InvalidWorkUnit)?;
        Ok(AdmittedInfer {
            work,
            placement,
            sequence,
            deadline_ms,
        })
    }

    pub fn commit_verified_infer(&mut self, admitted: &AdmittedInfer) -> Result<(), ProtocolError> {
        if self.revoked
            || self.profile == NodeProfile::StorageOnly
            || self.inflight >= self.max_inflight
        {
            return Err(ProtocolError::AdmissionDenied);
        }
        self.accept_sequence(admitted.sequence)?;
        self.inflight += 1;
        Ok(())
    }

    pub fn admit_infer(
        &mut self,
        frame: NodeControlFrame,
        now_ms: u64,
    ) -> Result<AdmittedInfer, ProtocolError> {
        let admitted = self.verify_infer(frame, now_ms)?;
        self.commit_verified_infer(&admitted)?;
        Ok(admitted)
    }

    pub fn complete_infer(&mut self) -> Result<(), ProtocolError> {
        self.inflight = self
            .inflight
            .checked_sub(1)
            .ok_or(ProtocolError::AdmissionDenied)?;
        Ok(())
    }

    pub fn restore_inflight(&mut self, count: usize) -> Result<(), ProtocolError> {
        let count = u32::try_from(count).map_err(|_| ProtocolError::AdmissionDenied)?;
        if self.inflight != 0 || count > self.max_inflight {
            return Err(ProtocolError::AdmissionDenied);
        }
        self.inflight = count;
        Ok(())
    }

    pub fn accept_cancel(
        &mut self,
        frame: &NodeControlFrame,
        work: &VerifiedWorkUnit,
    ) -> Result<(), ProtocolError> {
        let sequence = self.verify_cancel(frame, work)?;
        self.accept_sequence(sequence)
    }

    pub fn verify_cancel(
        &self,
        frame: &NodeControlFrame,
        work: &VerifiedWorkUnit,
    ) -> Result<u64, ProtocolError> {
        if self.revoked {
            return Err(ProtocolError::SessionRevoked);
        }
        let NodeControlFrame::Cancel {
            session_id,
            sequence,
            run_id,
            attempt,
            attempt_fence,
            signature,
        } = frame
        else {
            return Err(ProtocolError::MalformedFrame);
        };
        if session_id != &self.session_id
            || run_id != &work.run_id.to_string()
            || *attempt != work.attempt
            || *attempt_fence != work.attempt_fence
        {
            return Err(ProtocolError::FenceRejected);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                &self.issuer,
                self.hub_public_key,
                &frame.signed_control_bytes()?,
            )
            .map_err(|_| ProtocolError::SessionRevoked)?;
        if *sequence == 0 || *sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        Ok(*sequence)
    }

    pub fn accept_session_control(
        &mut self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<(), ProtocolError> {
        let verified = self.verify_session_control(frame, now_ms)?;
        self.apply_verified_session_control(&verified)
    }

    pub fn verify_session_control(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<VerifiedSessionControl, ProtocolError> {
        if self.revoked {
            return Err(ProtocolError::SessionRevoked);
        }
        let (session_id, sequence, signature, action) = match frame {
            NodeControlFrame::Rotate {
                session_id,
                sequence,
                next_session_id,
                expires_at_ms,
                signature,
            } => {
                if !valid_identifier(next_session_id) || *expires_at_ms <= now_ms {
                    return Err(ProtocolError::AdmissionDenied);
                }
                (
                    session_id,
                    *sequence,
                    signature,
                    SessionControlAction::Rotate {
                        next_session_id: next_session_id.clone(),
                    },
                )
            }
            NodeControlFrame::Revoke {
                session_id,
                sequence,
                reason,
                signature,
            } => {
                if reason.is_empty() || reason.len() > 1024 {
                    return Err(ProtocolError::AdmissionDenied);
                }
                (
                    session_id,
                    *sequence,
                    signature,
                    SessionControlAction::Revoke,
                )
            }
            _ => return Err(ProtocolError::MalformedFrame),
        };
        if session_id != &self.session_id {
            return Err(ProtocolError::SessionRevoked);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                &self.issuer,
                self.hub_public_key,
                &frame.signed_control_bytes()?,
            )
            .map_err(|_| ProtocolError::SessionRevoked)?;
        if sequence == 0 || sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        Ok(VerifiedSessionControl {
            sequence,
            current_session_id: self.session_id.clone(),
            action,
        })
    }

    pub fn apply_verified_session_control(
        &mut self,
        verified: &VerifiedSessionControl,
    ) -> Result<(), ProtocolError> {
        if verified.current_session_id != self.session_id {
            return Err(ProtocolError::SessionRevoked);
        }
        let (next_session_id, revoked) = match &verified.action {
            SessionControlAction::Rotate { next_session_id } => {
                (next_session_id.clone(), self.revoked)
            }
            SessionControlAction::Revoke => (self.session_id.clone(), true),
        };
        let next =
            self.session_state
                .successor(verified.sequence, next_session_id.clone(), revoked)?;
        crate::session_state::persist(&self.root, &next)?;
        self.session_state = next;
        self.session_id = next_session_id;
        self.revoked = revoked;
        Ok(())
    }

    pub fn commit_verified_sequence(&mut self, sequence: u64) -> Result<(), ProtocolError> {
        self.accept_sequence(sequence)
    }

    pub fn verify_storage_control(
        &self,
        frame: &NodeControlFrame,
        now_ms: u64,
    ) -> Result<u64, ProtocolError> {
        let NodeControlFrame::StorageControl { request } = frame else {
            return Err(ProtocolError::MalformedFrame);
        };
        request.verify(&self.session_id, &self.issuer, self.hub_public_key, now_ms)?;
        if self.revoked || request.sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedStorageRevision);
        }
        Ok(request.sequence)
    }

    fn accept_sequence(&mut self, sequence: u64) -> Result<(), ProtocolError> {
        if sequence == 0 || sequence <= self.session_state.control_position {
            return Err(ProtocolError::ReplayedPosition);
        }
        let next = self
            .session_state
            .successor(sequence, self.session_id.clone(), self.revoked)?;
        crate::session_state::persist(&self.root, &next)?;
        self.session_state = next;
        Ok(())
    }
}
