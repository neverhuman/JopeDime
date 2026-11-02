use super::*;

mod arm;
mod dataset;

pub use arm::Best5ArmV1;
pub use dataset::{DatasetStageV1, FrozenDatasetV1};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenCheckpointV1 {
    pub artifact_path: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub regression_head_schema_sha256: String,
    pub regression_heads: Vec<String>,
    pub inference_protocol_sha256: String,
}

impl FrozenCheckpointV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("checkpoint artifact path", &self.artifact_path)?;
        if self.artifact_sha256 != FROZEN_CHECKPOINT_SHA256 {
            return Err(format!(
                "checkpoint must be exactly {FROZEN_CHECKPOINT_SHA256}; substitution is forbidden"
            ));
        }
        if self.artifact_bytes == 0 {
            return Err("checkpoint artifact byte length must be positive".into());
        }
        validate_sha256(
            "regression head schema",
            &self.regression_head_schema_sha256,
        )?;
        if self.regression_heads.is_empty()
            || !self
                .regression_heads
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self
                .regression_heads
                .iter()
                .any(|head| !task_head_valid(GpTaskV1::Regression, head))
            || canonical_sha256(&self.regression_heads)? != self.regression_head_schema_sha256
        {
            return Err(
                "regression head schema is unsupported, unsorted, or hash-mismatched".into(),
            );
        }
        validate_sha256("inference protocol", &self.inference_protocol_sha256)
    }

    pub fn verify_artifact(&self) -> Result<(), String> {
        self.validate()?;
        let bytes = fs::read(&self.artifact_path)
            .map_err(|error| format!("read checkpoint {}: {error}", self.artifact_path))?;
        if bytes.len() as u64 != self.artifact_bytes {
            return Err("frozen checkpoint byte length changed".into());
        }
        if raw_sha256(&bytes) != self.artifact_sha256 {
            return Err("frozen checkpoint artifact hash changed".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmBudgetV1 {
    pub population: u32,
    pub generations: u32,
    pub search_splits: u32,
    pub archive_members: u32,
    pub cv7_finalists: u32,
    pub cv_folds: u32,
    pub logical_operation_cap: u32,
    pub predicted_cpu_second_cap: u64,
}

impl ArmBudgetV1 {
    pub fn frozen() -> Self {
        Self {
            population: POPULATION as u32,
            generations: GENERATIONS as u32,
            search_splits: SEARCH_SPLITS as u32,
            archive_members: 20,
            cv7_finalists: 10,
            cv_folds: 7,
            logical_operation_cap: LOGICAL_OPERATION_CAP,
            predicted_cpu_second_cap: PREDICTED_CPU_SECOND_CAP,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::frozen() {
            return Err(
                "arm budget must use the frozen 80/8/8/20/10/CV7/1100/7200 schedule".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScqPolicyV1 {
    pub project: String,
    pub priority_weight: u32,
    pub preemptible: bool,
    pub cpus: u32,
    pub memory_mib: u32,
    pub candidate_timeout_seconds: u32,
    pub job_timeout_seconds: u32,
    pub retries_same_identity: u32,
    pub min_worker_disk_free_fraction_micros: u32,
}

impl ScqPolicyV1 {
    pub fn frozen() -> Self {
        Self {
            project: PROJECT_ID.into(),
            priority_weight: 10,
            preemptible: true,
            cpus: 4,
            memory_mib: 8 * 1024,
            candidate_timeout_seconds: 30,
            job_timeout_seconds: 2 * 60 * 60,
            retries_same_identity: 2,
            // Owner directive 2026-08-07: 2% free-disk floor, was 10%. The box
            // runs at ~91% used by design (corpora + archives); 10% blocked
            // every launch while 2% still leaves ~150G headroom on this array.
            min_worker_disk_free_fraction_micros: 20_000,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self != &Self::frozen() {
            return Err("SCQ policy differs from the frozen best-five policy".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub input_manifest_sha256: String,
    pub panel_sha256: String,
    pub observation_snapshot_sha256: String,
    pub generated_registry_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
    pub six_win_manifest_sha256: String,
    pub proposal_generator_sha256: String,
    pub proposal_runtime_sha256: String,
    pub sotarena_root: String,
    pub input_manifest_artifact: FrozenFileV1,
    pub panel_artifact: FrozenFileV1,
    pub observation_snapshot_artifact: FrozenFileV1,
    pub six_win_manifest_artifact: FrozenFileV1,
    pub registry_artifact: FrozenFileV1,
    pub evaluator_artifact: FrozenFileV1,
    pub math_policy_artifact: FrozenFileV1,
    pub backend_policy_artifact: FrozenFileV1,
    pub proposal_generator_artifact: FrozenFileV1,
    pub proposal_runtime_artifact: FrozenFileV1,
    pub checkpoint: FrozenCheckpointV1,
    pub datasets: Vec<FrozenDatasetV1>,
    pub search_seeds: Vec<u64>,
    pub bootstrap_seed: u64,
    pub budgets: BTreeMap<Best5ArmV1, ArmBudgetV1>,
    pub scq: ScqPolicyV1,
    pub production_pointer_writes: bool,
    pub production_learning_ingest: bool,
    pub legacy_strategy_state_reused: bool,
}

impl CampaignFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5CampaignFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self.campaign_id != CAMPAIGN_ID {
            return Err(format!("campaign ID must be {CAMPAIGN_ID}"));
        }
        for (name, digest) in [
            ("input manifest", self.input_manifest_sha256.as_str()),
            ("panel", self.panel_sha256.as_str()),
            (
                "observation snapshot",
                self.observation_snapshot_sha256.as_str(),
            ),
            (
                "generated registry",
                self.generated_registry_sha256.as_str(),
            ),
            ("evaluator", self.evaluator_sha256.as_str()),
            ("math policy", self.math_policy_sha256.as_str()),
            ("backend policy", self.backend_policy_sha256.as_str()),
            ("six-win manifest", self.six_win_manifest_sha256.as_str()),
            (
                "proposal generator",
                self.proposal_generator_sha256.as_str(),
            ),
            ("proposal runtime", self.proposal_runtime_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.input_manifest_sha256 != FROZEN_INPUT_MANIFEST_SHA256
            || self.panel_sha256 != FROZEN_PANEL_SHA256
        {
            return Err(
                "campaign substituted the frozen input manifest or 42-dataset panel".into(),
            );
        }
        self.checkpoint.validate()?;
        self.scq.validate()?;
        nonempty("SOTArena root", &self.sotarena_root)?;
        self.observation_snapshot_artifact
            .validate("observation snapshot")?;
        for (name, artifact, expected) in [
            (
                "input manifest",
                &self.input_manifest_artifact,
                self.input_manifest_sha256.as_str(),
            ),
            ("panel", &self.panel_artifact, self.panel_sha256.as_str()),
            (
                "six-win manifest",
                &self.six_win_manifest_artifact,
                self.six_win_manifest_sha256.as_str(),
            ),
            (
                "registry",
                &self.registry_artifact,
                self.generated_registry_sha256.as_str(),
            ),
            (
                "evaluator",
                &self.evaluator_artifact,
                self.evaluator_sha256.as_str(),
            ),
            (
                "math policy",
                &self.math_policy_artifact,
                self.math_policy_sha256.as_str(),
            ),
            (
                "backend policy",
                &self.backend_policy_artifact,
                self.backend_policy_sha256.as_str(),
            ),
            (
                "proposal generator",
                &self.proposal_generator_artifact,
                self.proposal_generator_sha256.as_str(),
            ),
            (
                "proposal runtime",
                &self.proposal_runtime_artifact,
                self.proposal_runtime_sha256.as_str(),
            ),
        ] {
            artifact.validate(name)?;
            if artifact.sha256 != expected {
                return Err(format!("{name} artifact hash does not match its binding"));
            }
        }
        if self.datasets.len() != ALL_DATASETS {
            return Err("campaign must retain the complete frozen 42-dataset panel".into());
        }
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for dataset in &self.datasets {
            dataset.validate()?;
            if !ids.insert(dataset.dataset_id.as_str())
                || !digests.insert(dataset.dataset_sha256.as_str())
            {
                return Err("campaign dataset freeze contains a duplicate".into());
            }
        }
        let embedded = embedded_input_manifest()?;
        if embedded.datasets.len() != ALL_DATASETS {
            return Err("compiled input manifest does not contain 42 datasets".into());
        }
        for (actual, expected) in self.datasets.iter().zip(&embedded.datasets) {
            if actual.dataset_id != expected.dataset_hash
                || actual.train_sha256 != expected.train_sha256
                || actual.test_sha256 != expected.test_sha256
                || actual.metadata_sha256 != expected.meta_sha256
                || actual.train_rows != expected.train_rows
                || actual.feature_count != expected.features
            {
                return Err(format!(
                    "frozen input-manifest identity or hash drift for {}",
                    expected.dataset_hash
                ));
            }
        }
        let expected_pilot: BTreeSet<_> = PILOT_DATASET_IDS.into_iter().collect();
        let actual_pilot: BTreeSet<_> = self
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Pilot)
            .map(|dataset| dataset.dataset_id.as_str())
            .collect();
        if actual_pilot != expected_pilot {
            return Err("pilot must be the frozen first 20 input-manifest datasets".into());
        }
        if self
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Confirmation)
            .count()
            != CONFIRMATION_DATASETS
        {
            return Err("confirmation must contain the remaining 22 datasets".into());
        }
        if self
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Pilot && dataset.train_rows < 300)
            .count()
            != 9
        {
            return Err("pilot must retain all nine datasets below 300 training rows".into());
        }
        if self
            .datasets
            .iter()
            .filter(|dataset| dataset.current_sotarena_win)
            .count()
            != 6
        {
            return Err("campaign must freeze all six current SOTArena wins".into());
        }
        if self.search_seeds != SEARCH_SEEDS || self.bootstrap_seed != BOOTSTRAP_SEED {
            return Err("campaign seeds differ from the frozen 101/202/20260807 protocol".into());
        }
        if self.budgets.keys().copied().collect::<BTreeSet<_>>()
            != Best5ArmV1::ALL.into_iter().collect()
        {
            return Err("campaign requires five experimental arms and one matched control".into());
        }
        for budget in self.budgets.values() {
            budget.validate()?;
        }
        if self.production_pointer_writes
            || self.production_learning_ingest
            || self.legacy_strategy_state_reused
        {
            return Err(
                "isolated campaign cannot reuse legacy mutable state or mutate production".into(),
            );
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalBatchFreezeV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub proposals: Vec<ProposalFreezeV1>,
}

impl ProposalBatchFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5ProposalBatchFreeze.v1";

    /// Completion order is not evidence. Normalize it to the campaign's
    /// frozen dataset order before validation or publication.
    pub fn normalize_completion_order(&mut self, campaign: &CampaignFreezeV1) {
        let order = campaign
            .datasets
            .iter()
            .enumerate()
            .map(|(index, dataset)| (dataset.dataset_sha256.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        self.proposals.sort_by_key(|proposal| {
            order
                .get(proposal.dataset_sha256.as_str())
                .copied()
                .unwrap_or(usize::MAX)
        });
    }

    pub fn validate(&self, campaign: &CampaignFreezeV1) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self.campaign_freeze_sha256 != campaign.identity()? {
            return Err("proposal batch does not bind the supplied campaign".into());
        }
        if self.proposals.len() != ALL_DATASETS {
            return Err("proposal job must cover all 42 datasets up front".into());
        }
        let expected: BTreeSet<_> = campaign
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_sha256.as_str())
            .collect();
        let mut actual = BTreeSet::new();
        for proposal in &self.proposals {
            proposal.validate()?;
            if proposal.checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256
                || proposal.proposals.iter().any(|genome| {
                    genome.task != GpTaskV1::Regression
                        || !campaign
                            .checkpoint
                            .regression_heads
                            .contains(&genome.head_family)
                })
                || !actual.insert(proposal.dataset_sha256.as_str())
            {
                return Err("proposal batch substituted a checkpoint or repeated inference".into());
            }
        }
        if actual != expected {
            return Err(
                "proposal batch dataset coverage differs from the 42-dataset freeze".into(),
            );
        }
        if self
            .proposals
            .iter()
            .zip(&campaign.datasets)
            .any(|(proposal, dataset)| proposal.dataset_sha256 != dataset.dataset_sha256)
        {
            return Err("proposal batch is not in frozen dataset order".into());
        }
        Ok(())
    }

    pub fn identity(&self, campaign: &CampaignFreezeV1) -> Result<String, String> {
        self.validate(campaign)?;
        canonical_sha256(self)
    }
}
