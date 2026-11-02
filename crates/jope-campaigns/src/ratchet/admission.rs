use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]

pub(crate) enum ProofSlateLaneV3 {
    Selectable,
    CvOnlyCalibration,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ProofSlateCandidateV3 {
    pub(crate) candidate: CandidateIdentityV1,
    pub(crate) lane: ProofSlateLaneV3,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct AttemptAdmissionManifestV3 {
    pub(crate) schema: String,
    pub(crate) attempt_id: String,
    pub(crate) learning_wave_sha256: String,
    pub(crate) dataset_state_sha256: String,
    pub(crate) scheduler_draw_sha256: String,
    pub(crate) generation_zero_sha256: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) corrector_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) folds_sha256: String,
    pub(crate) proof_slate_seed: u64,
    pub(crate) proof_slate: Vec<ProofSlateCandidateV3>,
}

impl AttemptAdmissionManifestV3 {
    pub(crate) fn validate(
        &self,
        generation: &Generation0ReceiptV3,
        wave: &LearningWaveManifestV1,
        state: &DatasetLearningStateV2,
    ) -> Result<()> {
        if self.schema != ADMISSION_V3_SCHEMA || !valid_attempt_id(&self.attempt_id) {
            bail!("attempt admission is not a valid v3 manifest");
        }
        validate_hashes([
            (
                "admission learning wave",
                self.learning_wave_sha256.as_str(),
            ),
            (
                "admission dataset state",
                self.dataset_state_sha256.as_str(),
            ),
            (
                "admission scheduler draw",
                self.scheduler_draw_sha256.as_str(),
            ),
            (
                "admission generation zero",
                self.generation_zero_sha256.as_str(),
            ),
            ("admission checkpoint", self.checkpoint_sha256.as_str()),
            ("admission corrector", self.corrector_sha256.as_str()),
            ("admission dataset", self.dataset_sha256.as_str()),
            ("admission folds", self.folds_sha256.as_str()),
        ])?;
        generation.validate_against_state(state)?;
        let wave_sha256 = wave.manifest_sha256()?;
        if self.learning_wave_sha256 != wave_sha256
            || self.learning_wave_sha256 != generation.learning_wave_sha256
            || self.dataset_state_sha256 != generation.dataset_state_sha256
            || self.dataset_sha256 != generation.dataset_sha256
            || self.generation_zero_sha256 != generation.receipt_sha256()?
            || self.checkpoint_sha256 != wave.policy_checkpoint_sha256
            || self.corrector_sha256 != wave.corrector_sha256
        {
            bail!("attempt admission does not bind one frozen wave/dataset/generation");
        }
        if self.proof_slate.len() != 12 {
            bail!("attempt proof slate is not precommitted as 10+2");
        }
        require_unique(
            "attempt proof slate",
            self.proof_slate
                .iter()
                .map(|entry| entry.candidate.semantic_sha256.as_str()),
        )?;
        let selectable = self
            .proof_slate
            .iter()
            .filter(|entry| entry.lane == ProofSlateLaneV3::Selectable)
            .count();
        let calibration = self
            .proof_slate
            .iter()
            .filter(|entry| entry.lane == ProofSlateLaneV3::CvOnlyCalibration)
            .count();
        if (selectable, calibration) != (10, 2) {
            bail!("attempt proof slate lanes drifted from 10 selectable + 2 CV-only");
        }
        for entry in &self.proof_slate {
            entry.candidate.validate()?;
        }
        Ok(())
    }
}

pub(crate) fn valid_attempt_id(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.'))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayLabelRevisionV2 {
    pub(crate) group_sha256: String,
    pub(crate) dataset_sha256: String,
    pub(crate) revision: u64,
    pub(crate) label_revision_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayStepV2 {
    pub(crate) step: usize,
    pub(crate) datasets: Vec<String>,
    pub(crate) revisions: Vec<ReplayLabelRevisionV2>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayManifestV2 {
    pub(crate) schema: String,
    pub(crate) label_cutoff: u64,
    pub(crate) reservoir_manifest_sha256: String,
    pub(crate) reservoir_group_count: usize,
    pub(crate) reservoir_groups: Vec<String>,
    pub(crate) scheduled_datasets: Vec<String>,
    pub(crate) steps: Vec<ReplayStepV2>,
}

impl ReplayManifestV2 {
    pub(crate) fn validate(
        &self,
        historical_unordered_dataset_sets: &BTreeSet<Vec<String>>,
    ) -> Result<()> {
        if self.schema != REPLAY_V2_SCHEMA
            || self.reservoir_group_count != 200
            || self.reservoir_groups.len() != 200
            || self.steps.len() != REPLAY_STEPS
            || self.scheduled_datasets.len() != DATASETS_PER_MICRO_BATCH
        {
            bail!("replay v2 manifest does not bind 200 groups, eight datasets, and 100 steps");
        }
        validate_sha256("replay reservoir", &self.reservoir_manifest_sha256)?;
        for group in &self.reservoir_groups {
            validate_sha256("replay reservoir group", group)?;
        }
        require_unique(
            "replay reservoir groups",
            self.reservoir_groups.iter().map(String::as_str),
        )?;
        if self.reservoir_manifest_sha256 != sha256_bytes(&canonical_json(&self.reservoir_groups)?)
        {
            bail!("replay reservoir manifest does not bind its frozen 200 groups");
        }
        let reservoir_groups = self
            .reservoir_groups
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        validate_dataset_set("scheduled micro-batch", &self.scheduled_datasets)?;
        let mut local_sets = BTreeSet::new();
        for (index, step) in self.steps.iter().enumerate() {
            if step.step != index
                || step.datasets.len() != DATASETS_PER_MICRO_BATCH
                || step.revisions.len() != DATASETS_PER_MICRO_BATCH
            {
                bail!("replay step is not an exact unordered eight-dataset draw");
            }
            validate_dataset_set("replay step", &step.datasets)?;
            let mut normalized = step.datasets.clone();
            normalized.sort();
            if historical_unordered_dataset_sets.contains(&normalized)
                || !local_sets.insert(normalized)
            {
                bail!("replay contains a previously drawn unordered eight-dataset set");
            }
            let revision_datasets = step
                .revisions
                .iter()
                .map(|revision| revision.dataset_sha256.as_str())
                .collect::<BTreeSet<_>>();
            let step_datasets = step
                .datasets
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if revision_datasets != step_datasets {
                bail!("replay revisions do not bind exactly the step datasets");
            }
            for revision in &step.revisions {
                validate_hashes([
                    ("replay group", revision.group_sha256.as_str()),
                    ("replay dataset", revision.dataset_sha256.as_str()),
                    (
                        "replay label revision",
                        revision.label_revision_sha256.as_str(),
                    ),
                ])?;
                if revision.revision == 0 {
                    bail!("replay references label revision zero");
                }
                if !reservoir_groups.contains(revision.group_sha256.as_str()) {
                    bail!("replay step references a group outside the frozen reservoir");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn manifest_sha256(&self) -> Result<String> {
        self.validate(&BTreeSet::new())?;
        Ok(sha256_bytes(&canonical_json(self)?))
    }
}
