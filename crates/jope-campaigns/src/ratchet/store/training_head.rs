use super::*;

impl RatchetStore {
    pub(crate) fn publish_training_head(
        &self,
        head: &TrainingHeadManifestV1,
        batch: &LearningMicroBatchManifestV1,
        replay: &ReplayManifestV2,
    ) -> Result<String> {
        let _writer = WriterGuard::acquire(&self.root)?;
        head.validate()?;
        batch.validate(replay)?;
        if head.micro_batch_manifest_sha256 != batch.manifest_sha256(replay)?
            || head.replay_manifest_sha256 != replay.manifest_sha256()?
            || head.checkpoint_sha256 != batch.candidate_training_checkpoint_sha256
            || head.resume_checkpoint_sha256 != batch.candidate_resume_checkpoint_sha256
            || head.label_cutoff != replay.label_cutoff
        {
            bail!("training head does not reconstruct from its completed micro-batch and replay");
        }
        let bytes = canonical_json(head)?;
        let digest = sha256_bytes(&bytes);
        let head_object = self
            .root
            .join("training-head/objects")
            .join(format!("{digest}.json"));
        if head_object.exists() {
            if fs::read(&head_object)? != bytes {
                bail!("content-addressed training head object is corrupt");
            }
            let pointer = RatchetArtifactPointerV1 {
                schema: POINTER_SCHEMA.into(),
                artifact_sha256: digest.clone(),
                updated_at_unix_seconds: head.published_at_unix_seconds,
                serving_authority: false,
            };
            atomic_pointer_write(&self.root.join("training-head.json"), &pointer)?;
            return Ok(digest);
        }
        let historical_sets = self.historical_replay_sets()?;
        replay.validate(&historical_sets)?;
        if let Some((_, current)) = self.load_training_head()? {
            if batch.base_training_checkpoint_sha256 != current.checkpoint_sha256
                || head.label_cutoff < current.label_cutoff
                || head.published_at_unix_seconds <= current.published_at_unix_seconds
            {
                bail!("training head does not continue the current non-serving training branch");
            }
        }
        let replay_bytes = canonical_json(replay)?;
        let replay_sha256 = sha256_bytes(&replay_bytes);
        immutable_write(
            &self
                .root
                .join("replay/objects")
                .join(format!("{replay_sha256}.json")),
            &replay_bytes,
        )?;
        let batch_bytes = canonical_json(batch)?;
        let batch_sha256 = sha256_bytes(&batch_bytes);
        immutable_write(
            &self
                .root
                .join("micro-batches/objects")
                .join(format!("{batch_sha256}.json")),
            &batch_bytes,
        )?;
        immutable_write(&head_object, &bytes)?;
        let pointer = RatchetArtifactPointerV1 {
            schema: POINTER_SCHEMA.into(),
            artifact_sha256: digest.clone(),
            updated_at_unix_seconds: head.published_at_unix_seconds,
            serving_authority: false,
        };
        atomic_pointer_write(&self.root.join("training-head.json"), &pointer)?;
        Ok(digest)
    }

    pub(crate) fn load_training_head(&self) -> Result<Option<(String, TrainingHeadManifestV1)>> {
        let pointer_path = self.root.join("training-head.json");
        if !pointer_path.exists() {
            return Ok(None);
        }
        let pointer: RatchetArtifactPointerV1 = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        pointer.validate(false)?;
        let bytes = fs::read(
            self.root
                .join("training-head/objects")
                .join(format!("{}.json", pointer.artifact_sha256)),
        )
        .context("training-head object is missing")?;
        if sha256_bytes(&bytes) != pointer.artifact_sha256 {
            bail!("training-head pointer does not match its immutable object");
        }
        let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
        head.validate()?;
        if canonical_json(&head)? != bytes {
            bail!("training-head object is not canonical");
        }
        Ok(Some((pointer.artifact_sha256, head)))
    }

    pub(super) fn historical_replay_sets(&self) -> Result<BTreeSet<Vec<String>>> {
        let root = self.root.join("training-head/objects");
        if !root.exists() {
            return Ok(BTreeSet::new());
        }
        let mut paths = fs::read_dir(root)?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?;
        paths.sort();
        let mut sets = BTreeSet::new();
        for path in paths {
            let bytes = fs::read(&path)?;
            let expected = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .context("training-head object filename is invalid")?;
            validate_sha256("training-head object filename", expected)?;
            if sha256_bytes(&bytes) != expected {
                bail!("training-head object filename does not match its content");
            }
            let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
            head.validate()?;
            if canonical_json(&head)? != bytes {
                bail!("training-head object is not canonical");
            }
            let replay_bytes = fs::read(
                self.root
                    .join("replay/objects")
                    .join(format!("{}.json", head.replay_manifest_sha256)),
            )
            .context("accepted training head is missing its replay object")?;
            if sha256_bytes(&replay_bytes) != head.replay_manifest_sha256 {
                bail!("accepted training head replay object is corrupt");
            }
            let replay: ReplayManifestV2 = serde_json::from_slice(&replay_bytes)?;
            replay.validate(&sets)?;
            for step in replay.steps {
                let mut normalized = step.datasets;
                normalized.sort();
                sets.insert(normalized);
            }
        }
        Ok(sets)
    }

    pub(super) fn has_eligible_training_head(&self, wave: &LearningWaveManifestV1) -> Result<bool> {
        let root = self.root.join("training-head/objects");
        if !root.exists() {
            return Ok(false);
        }
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            let bytes = fs::read(&path)?;
            let expected = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .context("training-head object filename is invalid")?;
            validate_sha256("training-head object filename", expected)?;
            if sha256_bytes(&bytes) != expected {
                bail!("training-head object filename does not match its content");
            }
            let head: TrainingHeadManifestV1 = serde_json::from_slice(&bytes)?;
            head.validate()?;
            if canonical_json(&head)? != bytes {
                bail!("training-head object is not canonical");
            }
            if head.checkpoint_sha256 == wave.policy_checkpoint_sha256
                && head.resume_checkpoint_sha256 == wave.resume_checkpoint_sha256
                && head.corrector_sha256 == wave.corrector_sha256
                && head.label_cutoff == wave.training_label_cutoff
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
