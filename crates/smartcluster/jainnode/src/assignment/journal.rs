use super::*;

impl AssignmentJournal {
    pub fn open(root: PathBuf) -> Result<Self, AssignmentError> {
        if !root.is_absolute() {
            return Err(AssignmentError::UnsafeJournalPath);
        }
        fs::create_dir_all(&root).map_err(AssignmentError::Io)?;
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
            .map_err(AssignmentError::Io)?;
        let metadata = fs::symlink_metadata(&root).map_err(AssignmentError::Io)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() || metadata.mode() & 0o022 != 0 {
            return Err(AssignmentError::UnsafeJournalPath);
        }
        let canonical = root.canonicalize().map_err(AssignmentError::Io)?;
        if canonical != root {
            return Err(AssignmentError::UnsafeJournalPath);
        }
        Ok(Self { root })
    }

    pub fn accept_before_ack(
        &self,
        identity: &NodeRuntimeIdentity,
        work: VerifiedWorkUnit,
        placement: Placement,
        session_sequence: u64,
        now_ms: u64,
    ) -> Result<(AcceptedAssignment, WorkerAcknowledgement), AssignmentError> {
        placement
            .verify_for(
                &work,
                identity.node_id(),
                identity.session_id(),
                identity.hub_public_key,
                now_ms,
            )
            .map_err(AssignmentError::Fabric)?;
        let record = AssignmentRecord::new(&work, &placement, session_sequence, now_ms)?;
        let bytes = postcard::to_stdvec(&record).map_err(|_| AssignmentError::InvalidRecord)?;
        if bytes.len() > MAX_ASSIGNMENT_BYTES {
            return Err(AssignmentError::InvalidRecord);
        }
        let digest = sha256(&bytes);
        let record_path = self.record_path(&record);
        self.persist_exact(&record_path, &bytes, digest)?;
        let accepted = AcceptedAssignment {
            work,
            placement,
            journal_digest: digest,
            session_sequence,
            acknowledged_at_ms: now_ms,
            record_path,
        };
        let acknowledgement = accepted.acknowledgement(identity)?;
        Ok((accepted, acknowledgement))
    }

    pub fn mark_running(&self, accepted: &AcceptedAssignment) -> Result<(), AssignmentError> {
        self.transition(
            accepted,
            AssignmentState::Accepted,
            AssignmentState::Running,
        )
    }

    pub fn mark_terminal(&self, accepted: &AcceptedAssignment) -> Result<(), AssignmentError> {
        self.transition(
            accepted,
            AssignmentState::Running,
            AssignmentState::Terminal,
        )
    }

    pub fn mark_rejected(&self, accepted: &AcceptedAssignment) -> Result<(), AssignmentError> {
        self.transition(
            accepted,
            AssignmentState::Accepted,
            AssignmentState::Terminal,
        )
    }

    pub fn recover(
        &self,
        identity: &NodeRuntimeIdentity,
        now_ms: u64,
    ) -> Result<RecoveryReport, AssignmentError> {
        self.recover_interrupted_writes(identity, now_ms)?;
        let mut report = RecoveryReport::default();
        let mut entries = fs::read_dir(&self.root)
            .map_err(AssignmentError::Io)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(AssignmentError::Io)?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        if entries.len() > MAX_RECOVERY_RECORDS {
            return Err(AssignmentError::RecoveryBoundExceeded);
        }
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("assignment") {
                continue;
            }
            match self.replay_record(identity, &path, now_ms) {
                Ok(Some(accepted)) => {
                    report.pending.push(accepted.work.run_id.to_string());
                    report.replayable.push(accepted);
                }
                Ok(None) => {}
                Err(_) => {
                    let quarantine = quarantine_file(&self.root, &path)?;
                    report.quarantined.push(quarantine.display().to_string());
                }
            }
        }
        Ok(report)
    }

    fn replay_record(
        &self,
        identity: &NodeRuntimeIdentity,
        path: &Path,
        now_ms: u64,
    ) -> Result<Option<AcceptedAssignment>, AssignmentError> {
        let bytes = read_bounded(path)?;
        let record: AssignmentRecord =
            postcard::from_bytes(&bytes).map_err(|_| AssignmentError::InvalidRecord)?;
        let work = VerifiedWorkUnit::verify(
            record.signed_work.clone(),
            &identity.hub_issuer,
            identity.hub_public_key,
            &identity.image_issuer,
            identity.image_public_key,
        )
        .map_err(AssignmentError::Fabric)?;
        record.validate_exact(identity, &work, now_ms)?;
        match record.state {
            AssignmentState::Accepted => {}
            AssignmentState::Terminal => return Ok(None),
            AssignmentState::Running => return Err(AssignmentError::ConflictingRecord),
        }
        Ok(Some(AcceptedAssignment {
            work,
            placement: record.placement,
            journal_digest: sha256(&bytes),
            session_sequence: record.session_sequence,
            acknowledged_at_ms: record.acknowledged_at_ms,
            record_path: path.to_path_buf(),
        }))
    }

    fn recover_interrupted_writes(
        &self,
        identity: &NodeRuntimeIdentity,
        now_ms: u64,
    ) -> Result<(), AssignmentError> {
        let mut entries = fs::read_dir(&self.root)
            .map_err(AssignmentError::Io)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(AssignmentError::Io)?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        if entries.len() > MAX_RECOVERY_RECORDS {
            return Err(AssignmentError::RecoveryBoundExceeded);
        }
        for entry in entries {
            let path = entry.path();
            let extension = path.extension().and_then(|value| value.to_str());
            if !matches!(extension, Some("pending" | "transition")) {
                continue;
            }
            let destination = path.with_extension("assignment");
            let candidate = read_record(&path);
            let promote = match (extension, candidate, destination.exists()) {
                (Some("pending"), Ok(record), false) => {
                    self.validate_interrupted(identity, &record, now_ms).is_ok()
                        && record.state == AssignmentState::Accepted
                }
                (Some("transition"), Ok(record), true) => {
                    let current = read_record(&destination);
                    current.is_ok_and(|current| valid_transition(&current, &record))
                        && self.validate_interrupted(identity, &record, now_ms).is_ok()
                }
                _ => false,
            };
            if promote {
                fs::rename(&path, &destination).map_err(AssignmentError::Io)?;
                sync_directory(&self.root)?;
            } else {
                quarantine_file(&self.root, &path)?;
            }
        }
        Ok(())
    }

    fn validate_interrupted(
        &self,
        identity: &NodeRuntimeIdentity,
        record: &AssignmentRecord,
        now_ms: u64,
    ) -> Result<(), AssignmentError> {
        let work = VerifiedWorkUnit::verify(
            record.signed_work.clone(),
            &identity.hub_issuer,
            identity.hub_public_key,
            &identity.image_issuer,
            identity.image_public_key,
        )
        .map_err(AssignmentError::Fabric)?;
        record.validate_exact(identity, &work, now_ms)
    }

    fn transition(
        &self,
        accepted: &AcceptedAssignment,
        expected: AssignmentState,
        next: AssignmentState,
    ) -> Result<(), AssignmentError> {
        let mut record = read_record(&accepted.record_path)?;
        if record.state != expected
            || record.work_unit_digest != accepted.work.work_unit_digest()
            || record.placement_digest
                != accepted
                    .placement
                    .digest()
                    .map_err(AssignmentError::Fabric)?
        {
            return Err(AssignmentError::ConflictingRecord);
        }
        record.state = next;
        let bytes = postcard::to_stdvec(&record).map_err(|_| AssignmentError::InvalidRecord)?;
        atomic_replace(&self.root, &accepted.record_path, &bytes)
    }

    fn record_path(&self, record: &AssignmentRecord) -> PathBuf {
        self.root.join(format!(
            "{}-{}-{}.assignment",
            record.run_id, record.attempt, record.attempt_fence
        ))
    }

    fn persist_exact(
        &self,
        record_path: &Path,
        bytes: &[u8],
        digest: Digest32,
    ) -> Result<(), AssignmentError> {
        if record_path.exists() {
            let existing = read_bounded(record_path)?;
            if existing == bytes {
                return Ok(());
            }
            let quarantine = self.root.join(format!(
                "{}.quarantine-{}",
                record_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .ok_or(AssignmentError::UnsafeJournalPath)?,
                hex_digest(digest)
            ));
            write_new_synced(&quarantine, bytes)?;
            sync_directory(&self.root)?;
            return Err(AssignmentError::ConflictingRecord);
        }
        let pending = record_path.with_extension("pending");
        write_new_synced(&pending, bytes)?;
        fs::rename(&pending, record_path).map_err(AssignmentError::Io)?;
        sync_directory(&self.root)
    }
}
