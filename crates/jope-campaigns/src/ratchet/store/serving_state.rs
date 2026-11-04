use super::*;

impl RatchetStore {
    pub(crate) fn commit_dataset_state(
        &self,
        state: &DatasetLearningStateV2,
        committed_at_unix_seconds: u64,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        state.validate()?;
        if let Some((_, current)) = self.load_dataset_state(&state.dataset_sha256)? {
            current.validate_successor(state)?;
        }
        let bytes = canonical_json(state)?;
        let digest = sha256_bytes(&bytes);
        let dataset_root = self.root.join("datasets").join(&state.dataset_sha256);
        immutable_write(
            &dataset_root.join("states").join(format!("{digest}.json")),
            &bytes,
        )?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: committed_at_unix_seconds,
            serving_authority: false,
        };
        atomic_pointer_write(&dataset_root.join("current.json"), &pointer)?;
        Ok(digest)
    }

    pub(crate) fn load_dataset_state(
        &self,
        dataset_sha256: &str,
    ) -> Result<Option<(String, DatasetLearningStateV2)>> {
        validate_sha256("dataset state lookup", dataset_sha256)?;
        let dataset_root = self.root.join("datasets").join(dataset_sha256);
        let pointer_path = dataset_root.join("current.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(false)?;
        let bytes = fs::read(
            dataset_root
                .join("states")
                .join(format!("{}.json", pointer.artifact_sha256)),
        )
        .context("dataset state object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("dataset state pointer digest does not match its immutable object");
        }
        let state: DatasetLearningStateV2 = serde_json::from_slice(&bytes)?;
        state.validate()?;
        if state.dataset_sha256 != dataset_sha256
            || state.snapshot_sha256()? != pointer.artifact_sha256
        {
            bail!("dataset state object does not bind the requested dataset");
        }
        Ok(Some((pointer.artifact_sha256, state)))
    }

    pub(crate) fn publish_serving_wave(
        &self,
        wave: &LearningWaveManifestV1,
        switches: &RolloutSwitchesV1,
        sealed_holdout_hashes: &BTreeSet<String>,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        switches.require_wave_authority()?;
        let bytes = canonical_json(wave)?;
        let digest = sha256_bytes(&bytes);
        let current = self.load_serving_wave()?;
        if current
            .as_ref()
            .is_some_and(|(current_digest, current_manifest)| {
                current_digest == &digest && current_manifest == wave
            })
        {
            return Ok(digest);
        }
        let parent = current
            .as_ref()
            .map(|(digest, manifest)| (digest.as_str(), manifest));
        wave.validate(parent, sealed_holdout_hashes)?;
        if parent.is_some() && !self.has_eligible_training_head(wave)? {
            bail!("recurring serving wave is not backed by an eligible training-head checkpoint");
        }
        immutable_write(
            &self
                .root
                .join("waves")
                .join("objects")
                .join(format!("{digest}.json")),
            &bytes,
        )?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: wave.accepted_at_unix_seconds,
            serving_authority: true,
        };
        // `waves/current.json` is deliberately written before the compatibility
        // mirror: a crash can leave a stale mirror, never a false authority.
        atomic_pointer_write(&self.root.join("waves/current.json"), &pointer)?;
        let mirror = RatchetArtifactPointerV1 {
            serving_authority: false,
            ..pointer
        };
        atomic_pointer_write(&self.root.join("dl/current.json"), &mirror)?;
        Ok(digest)
    }

    pub(crate) fn admit_attempt(
        &self,
        generation: &Generation0ReceiptV3,
        admission: &AttemptAdmissionManifestV3,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        if !valid_attempt_id(&admission.attempt_id) {
            bail!("attempt identity is unsafe");
        }
        let admission_bytes = canonical_json(admission)?;
        let admission_sha256 = sha256_bytes(&admission_bytes);
        let attempt_root = self.root.join("attempts").join(&admission.attempt_id);
        let committed_admission = attempt_root.join("admission.v3.json");
        if committed_admission.exists() {
            if fs::read(&committed_admission)? == admission_bytes {
                return Ok(admission_sha256);
            }
            bail!("attempt identity is already bound to different immutable admission bytes");
        }
        if self.root.join("admissions-stopped.json").exists() {
            bail!("new attempt admissions are stopped by an authority fault");
        }
        let (wave_sha256, wave) = self
            .load_serving_wave()?
            .context("no accepted serving wave exists")?;
        let (state_sha256, state) = self
            .load_dataset_state(&admission.dataset_sha256)?
            .context("dataset has no authoritative v2 learning state")?;
        if generation.learning_wave_sha256 != wave_sha256
            || generation.dataset_state_sha256 != state_sha256
        {
            bail!("new attempt did not use the latest serving wave and exact dataset state");
        }
        admission.validate(generation, &wave, &state)?;

        let generation_bytes = canonical_json(generation)?;
        let generation_sha256 = sha256_bytes(&generation_bytes);
        immutable_write(
            &attempt_root
                .join("generation-zero")
                .join(format!("{generation_sha256}.json")),
            &generation_bytes,
        )?;
        // Admission is the final commit marker. A crash before this write
        // leaves no admitted attempt; retrying reconstructs identical bytes.
        immutable_write(&committed_admission, &admission_bytes)?;
        Ok(admission_sha256)
    }

    pub(crate) fn load_serving_wave(&self) -> Result<Option<(String, LearningWaveManifestV1)>> {
        let pointer_path = self.root.join("waves/current.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(true)?;
        let object_path = self
            .root
            .join("waves/objects")
            .join(format!("{}.json", pointer.artifact_sha256));
        let bytes = fs::read(&object_path).context("serving wave object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("serving wave pointer digest does not match its immutable object");
        }
        let manifest: LearningWaveManifestV1 = serde_json::from_slice(&bytes)?;
        if manifest.manifest_sha256()? != pointer.artifact_sha256 {
            bail!("serving wave object is not canonical");
        }
        Ok(Some((pointer.artifact_sha256, manifest)))
    }

    /// Stop future admissions without modifying any frozen attempt manifest.
    #[cfg(test)]
    pub(crate) fn stop_admissions(&self, fault_receipt_sha256: &str, at: u64) -> Result<()> {
        let _writer = WriterGuard::acquire(&self.root)?;
        validate_sha256("authority fault receipt", fault_receipt_sha256)?;
        let value = serde_json::json!({
            "schema": "RatchetAdmissionStop.v1",
            "fault_receipt_sha256": fault_receipt_sha256,
            "stopped_at_unix_seconds": at,
        });
        immutable_write(
            &self.root.join("admissions-stopped.json"),
            &canonical_json(&value)?,
        )
    }

    /// Fault rollback only moves the authoritative pointer to an already
    /// accepted immutable ancestor. It never republishes or edits that wave.
    #[cfg(test)]
    pub(crate) fn restore_previous_wave(&self, previous_wave_sha256: &str, at: u64) -> Result<()> {
        let _writer = WriterGuard::acquire(&self.root)?;
        validate_sha256("previous accepted wave", previous_wave_sha256)?;
        let object_path = self
            .root
            .join("waves/objects")
            .join(format!("{previous_wave_sha256}.json"));
        let bytes = fs::read(&object_path).context("previous accepted wave object is missing")?;
        if sha256_bytes(&bytes) != previous_wave_sha256 {
            bail!("previous accepted wave object is corrupt");
        }
        let manifest: LearningWaveManifestV1 = serde_json::from_slice(&bytes)?;
        if manifest.manifest_sha256()? != previous_wave_sha256 {
            bail!("previous accepted wave object is noncanonical");
        }
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: previous_wave_sha256.into(),
            updated_at_unix_seconds: at,
            serving_authority: true,
        };
        atomic_pointer_write(&self.root.join("waves/current.json"), &pointer)?;
        let mirror = RatchetArtifactPointerV1 {
            serving_authority: false,
            ..pointer
        };
        atomic_pointer_write(&self.root.join("dl/current.json"), &mirror)
    }
}
