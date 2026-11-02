use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]

pub(crate) struct LearningMicroBatchManifestV1 {
    pub(crate) schema: String,
    pub(crate) batch_index: u64,
    pub(crate) serving_wave_sha256: String,
    pub(crate) base_training_checkpoint_sha256: String,
    pub(crate) candidate_training_checkpoint_sha256: String,
    pub(crate) candidate_resume_checkpoint_sha256: String,
    pub(crate) replay_manifest_sha256: String,
    pub(crate) scheduler_draws: Vec<SchedulerDrawV4>,
    pub(crate) update_steps: usize,
    pub(crate) completed: bool,
}

impl LearningMicroBatchManifestV1 {
    pub(crate) fn validate(&self, replay: &ReplayManifestV2) -> Result<()> {
        if self.schema != MICRO_BATCH_SCHEMA
            || self.scheduler_draws.len() != DATASETS_PER_MICRO_BATCH
            || self.update_steps != REPLAY_STEPS
            || !self.completed
        {
            bail!("micro-batch is not eight datasets plus one complete 100-step update");
        }
        validate_hashes([
            (
                "micro-batch serving wave",
                self.serving_wave_sha256.as_str(),
            ),
            (
                "micro-batch base training checkpoint",
                self.base_training_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch candidate checkpoint",
                self.candidate_training_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch candidate resume checkpoint",
                self.candidate_resume_checkpoint_sha256.as_str(),
            ),
            (
                "micro-batch replay manifest",
                self.replay_manifest_sha256.as_str(),
            ),
        ])?;
        for draw in &self.scheduler_draws {
            draw.validate()?;
        }
        require_unique(
            "micro-batch scheduler datasets",
            self.scheduler_draws
                .iter()
                .map(|draw| draw.dataset_sha256.as_str()),
        )?;
        if self.replay_manifest_sha256 != replay.manifest_sha256()?
            || self
                .scheduler_draws
                .iter()
                .map(|draw| draw.dataset_sha256.as_str())
                .collect::<BTreeSet<_>>()
                != replay
                    .scheduled_datasets
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
        {
            bail!("micro-batch scheduler draws and immutable replay disagree");
        }
        Ok(())
    }

    pub(crate) fn manifest_sha256(&self, replay: &ReplayManifestV2) -> Result<String> {
        self.validate(replay)?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TrainingHeadManifestV1 {
    pub(crate) schema: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) resume_checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) micro_batch_manifest_sha256: String,
    pub(crate) replay_manifest_sha256: String,
    pub(crate) update_steps: usize,
    pub(crate) label_cutoff: u64,
    pub(crate) published_at_unix_seconds: u64,
}

impl TrainingHeadManifestV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != TRAINING_HEAD_SCHEMA || self.update_steps != REPLAY_STEPS {
            bail!("training head does not represent one valid 100-step update");
        }
        validate_hashes([
            ("training head checkpoint", self.checkpoint_sha256.as_str()),
            (
                "training head resume checkpoint",
                self.resume_checkpoint_sha256.as_str(),
            ),
            ("training head corrector", self.corrector_sha256.as_str()),
            (
                "training head micro-batch",
                self.micro_batch_manifest_sha256.as_str(),
            ),
            (
                "training head replay manifest",
                self.replay_manifest_sha256.as_str(),
            ),
        ])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct RolloutSwitchesV1 {
    #[serde(default)]
    pub(crate) dataset_state_v2: bool,
    #[serde(default)]
    pub(crate) scheduler_v4: bool,
    #[serde(default)]
    pub(crate) replay_v2: bool,
    #[serde(default)]
    pub(crate) generation_zero_v3: bool,
    #[serde(default)]
    pub(crate) wave_authority: bool,
}

impl RolloutSwitchesV1 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.wave_authority
            && !(self.dataset_state_v2
                && self.scheduler_v4
                && self.replay_v2
                && self.generation_zero_v3)
        {
            bail!("wave authority requires every dependent rollout switch");
        }
        Ok(())
    }

    pub(super) fn require_wave_authority(&self) -> Result<()> {
        self.validate()?;
        if !self.wave_authority {
            bail!("serving-wave publication is disabled");
        }
        Ok(())
    }
}
