use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use serde::{Deserialize, Serialize};

use crate::jope_generalized_gp::{
    raw_sha256, task_head_valid, GenomeV1, GpTaskV1, ProposalFreezeV1,
};
use crate::search_v3::{canonical_sha256, validate_sha256, PredictionObjectV1};

pub const CAMPAIGN_ID: &str = "jope-dlgp-best5";
pub const PROJECT_ID: &str = "jope-dlgp-best5";
pub const FROZEN_CHECKPOINT_SHA256: &str =
    "e0aef7d74afd75d3003bdb049fa0bac06653f66408c7bfc458b1aba5ae1b98cb";
pub const FROZEN_INPUT_MANIFEST_SHA256: &str =
    "9e47eb9a10f6d92cab839093f3817eb1e0a17c1ac83f63a2ed3633af585bc6c0";
pub const FROZEN_PANEL_SHA256: &str =
    "b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75";
pub const BOOTSTRAP_SEED: u64 = 20_260_807;
pub const SEARCH_SEEDS: [u64; 2] = [101, 202];
pub const POPULATION: usize = 80;
pub const GENERATIONS: usize = 8;
pub const SEARCH_SPLITS: usize = 8;
pub const LOGICAL_OPERATION_CAP: u32 = 1_100;
pub const PREDICTED_CPU_SECOND_CAP: u64 = 7_200;
pub const PILOT_DATASETS: usize = 20;
pub const CONFIRMATION_DATASETS: usize = 22;
pub const ALL_DATASETS: usize = 42;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenFileV1 {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl FrozenFileV1 {
    pub fn validate(&self, name: &str) -> Result<(), String> {
        nonempty(&format!("{name} path"), &self.path)?;
        validate_sha256(name, &self.sha256)?;
        if self.bytes == 0 {
            return Err(format!("{name} byte length must be positive"));
        }
        Ok(())
    }

    pub fn verify(&self, name: &str) -> Result<(), String> {
        self.validate(name)?;
        let bytes =
            fs::read(&self.path).map_err(|error| format!("read {name} {}: {error}", self.path))?;
        if bytes.len() as u64 != self.bytes || raw_sha256(&bytes) != self.sha256 {
            return Err(format!("{name} bytes differ from their freeze"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignSourceV1 {
    pub schema: String,
    pub sotarena_root: String,
    pub input_manifest: FrozenFileV1,
    pub panel: FrozenFileV1,
    pub observation_snapshot: FrozenFileV1,
    pub six_win_manifest: FrozenFileV1,
    pub registry: FrozenFileV1,
    pub evaluator: FrozenFileV1,
    pub math_policy: FrozenFileV1,
    pub backend_policy: FrozenFileV1,
    pub proposal_generator: FrozenFileV1,
    pub proposal_runtime: FrozenFileV1,
    pub checkpoint: FrozenCheckpointV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SixWinManifestV1 {
    pub schema: String,
    pub dataset_ids: Vec<String>,
}

impl SixWinManifestV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SixWinManifest.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self.dataset_ids.len() != 6 || !self.dataset_ids.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err("six-win manifest must contain six sorted unique dataset IDs".into());
        }
        for id in &self.dataset_ids {
            short_dataset_id(id)?;
        }
        Ok(())
    }
}

impl CampaignSourceV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5CampaignSource.v1";

    pub fn verify(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        nonempty("SOTArena root", &self.sotarena_root)?;
        let root = std::path::Path::new(&self.sotarena_root);
        if !root.is_dir() {
            return Err("frozen SOTArena root is not a directory".into());
        }
        for (name, artifact) in [
            ("input manifest", &self.input_manifest),
            ("panel", &self.panel),
            ("observation snapshot", &self.observation_snapshot),
            ("six-win manifest", &self.six_win_manifest),
            ("registry", &self.registry),
            ("evaluator", &self.evaluator),
            ("math policy", &self.math_policy),
            ("backend policy", &self.backend_policy),
            ("proposal generator", &self.proposal_generator),
            ("proposal runtime", &self.proposal_runtime),
        ] {
            artifact.verify(name)?;
        }
        if self.input_manifest.sha256 != FROZEN_INPUT_MANIFEST_SHA256
            || self.panel.sha256 != FROZEN_PANEL_SHA256
        {
            return Err("source substituted the frozen input manifest or panel".into());
        }
        self.checkpoint.verify_artifact()
    }
}

pub const PILOT_DATASET_IDS: [&str; PILOT_DATASETS] = [
    "1cb97b1d7314d43d",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "43b4cc3ab0ebbd97",
    "55b144de19659467",
    "5f91a00bd3b66742",
    "79e600bc5fcd5da7",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b78d323566930bd7",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "fe3cc716a05d036f",
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "1a803b27c2400dc4",
];

const FROZEN_INPUT_MANIFEST_JSON: &str = include_str!("../../manifests/input-manifest.v1.json");

#[derive(Deserialize)]
struct EmbeddedInputManifestV1 {
    datasets: Vec<EmbeddedInputDatasetV1>,
}

#[derive(Deserialize)]
struct EmbeddedInputDatasetV1 {
    dataset_hash: String,
    features: u32,
    meta_sha256: String,
    train_rows: u32,
    train_sha256: String,
    test_sha256: String,
}

fn embedded_input_manifest() -> Result<EmbeddedInputManifestV1, String> {
    if raw_sha256(FROZEN_INPUT_MANIFEST_JSON.as_bytes()) != FROZEN_INPUT_MANIFEST_SHA256 {
        return Err("compiled frozen input manifest bytes drifted".into());
    }
    serde_json::from_str(FROZEN_INPUT_MANIFEST_JSON)
        .map_err(|error| format!("decode compiled frozen input manifest: {error}"))
}

fn schema(actual: &str, expected: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("schema must be {expected}, found {actual}"))
    }
}

fn finite(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}

fn nonempty(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must not be empty"))
    } else {
        Ok(())
    }
}

fn short_dataset_id(value: &str) -> Result<(), String> {
    if value.len() == 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "dataset ID must be 16 lowercase hex characters: {value}"
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Best5ArmV1 {
    EvolvabilityUcbLineages,
    CrossFittedResidualDlgp,
    SparseOofPortfolioGp,
    HeadFamilyMapElites,
    SurrogateUcbOffspringSearch,
    ConfidenceGpControl,
}

impl Best5ArmV1 {
    pub const EXPERIMENTS: [Self; 5] = [
        Self::EvolvabilityUcbLineages,
        Self::CrossFittedResidualDlgp,
        Self::SparseOofPortfolioGp,
        Self::HeadFamilyMapElites,
        Self::SurrogateUcbOffspringSearch,
    ];
    pub const ALL: [Self; 6] = [
        Self::EvolvabilityUcbLineages,
        Self::CrossFittedResidualDlgp,
        Self::SparseOofPortfolioGp,
        Self::HeadFamilyMapElites,
        Self::SurrogateUcbOffspringSearch,
        Self::ConfidenceGpControl,
    ];

    pub fn is_experiment(self) -> bool {
        self != Self::ConfidenceGpControl
    }

    pub fn simplicity_order(self) -> usize {
        match self {
            Self::EvolvabilityUcbLineages => 0,
            Self::HeadFamilyMapElites => 1,
            Self::SurrogateUcbOffspringSearch => 2,
            Self::CrossFittedResidualDlgp => 3,
            Self::SparseOofPortfolioGp => 4,
            Self::ConfidenceGpControl => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetStageV1 {
    Pilot,
    Confirmation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenDatasetV1 {
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub train_relative_path: String,
    pub test_relative_path: String,
    pub metadata_relative_path: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub metadata_sha256: String,
    pub train_rows: u32,
    pub feature_count: u32,
    pub source_family: String,
    pub stage: DatasetStageV1,
    pub current_sotarena_win: bool,
}

impl FrozenDatasetV1 {
    pub fn validate(&self) -> Result<(), String> {
        short_dataset_id(&self.dataset_id)?;
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.train_sha256.as_str()),
            ("test", self.test_sha256.as_str()),
            ("metadata", self.metadata_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if !self.dataset_sha256.starts_with(&self.dataset_id) {
            return Err("full dataset digest must start with its frozen short ID".into());
        }
        if self.train_rows == 0 || self.feature_count == 0 {
            return Err("dataset rows and features must be positive".into());
        }
        for (name, path) in [
            ("training", self.train_relative_path.as_str()),
            ("test", self.test_relative_path.as_str()),
            ("metadata", self.metadata_relative_path.as_str()),
        ] {
            let path = std::path::Path::new(path);
            if path.is_absolute()
                || path.as_os_str().is_empty()
                || path
                    .components()
                    .any(|component| matches!(component, std::path::Component::ParentDir))
            {
                return Err(format!("{name} dataset path must be a safe relative path"));
            }
        }
        nonempty("source family", &self.source_family)
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRowV1 {
    pub stable_row_id: String,
    pub atomic_group: String,
    pub target: f64,
}

impl TargetRowV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("stable row ID", &self.stable_row_id)?;
        nonempty("atomic group", &self.atomic_group)?;
        finite("regression target", self.target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSplitV1 {
    pub split_index: u32,
    pub split_seed: u64,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSplitPlanV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub master_seed: u64,
    pub stable_row_ids: Vec<String>,
    pub row_groups: BTreeMap<String, String>,
    pub splits: Vec<DataSplitV1>,
}

impl SearchSplitPlanV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchSplitPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.validate_n(Self::SCHEMA, SEARCH_SPLITS, true)
    }

    pub(crate) fn validate_n(
        &self,
        expected_schema: &str,
        count: usize,
        distinct: bool,
    ) -> Result<(), String> {
        schema(&self.schema, expected_schema)?;
        validate_sha256("split dataset", &self.dataset_sha256)?;
        validate_sha256("split training", &self.training_sha256)?;
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("split plan master seed must be 101 or 202".into());
        }
        let all: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all.len() != self.stable_row_ids.len() || all.len() < count {
            return Err("split rows are duplicate or insufficient".into());
        }
        if self.row_groups.len() != all.len()
            || self
                .row_groups
                .iter()
                .any(|(row, group)| !all.contains(row) || group.is_empty())
        {
            return Err("split plan must bind every row to an atomic group".into());
        }
        if self.splits.len() != count {
            return Err(format!("split plan requires exactly {count} splits"));
        }
        let mut validation_sets = BTreeSet::new();
        for (index, split) in self.splits.iter().enumerate() {
            if split.split_index != index as u32 {
                return Err("split indices must be contiguous".into());
            }
            let train: BTreeSet<_> = split.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = split.validation_row_ids.iter().collect();
            if train.len() != split.train_row_ids.len()
                || validation.len() != split.validation_row_ids.len()
                || train.is_empty()
                || validation.is_empty()
                || !train.is_disjoint(&validation)
                || train.union(&validation).copied().collect::<BTreeSet<_>>() != all
            {
                return Err("split must be a complete disjoint 80/20 partition".into());
            }
            let groups: BTreeSet<_> = split.validation_groups.iter().collect();
            if validation
                .iter()
                .any(|row| !groups.contains(&self.row_groups[*row]))
                || train
                    .iter()
                    .any(|row| groups.contains(&self.row_groups[*row]))
            {
                return Err("split crosses or omits an atomic group".into());
            }
            let validation_fraction = validation.len() as f64 / all.len() as f64;
            let target_fraction = if expected_schema == Self::SCHEMA {
                0.20
            } else {
                1.0 / count as f64
            };
            let max_group = self
                .row_groups
                .values()
                .fold(BTreeMap::<&str, usize>::new(), |mut counts, group| {
                    *counts.entry(group).or_default() += 1;
                    counts
                })
                .values()
                .copied()
                .max()
                .unwrap_or(1) as f64
                / all.len() as f64;
            if (validation_fraction - target_fraction).abs() > max_group + 0.02 {
                return Err(
                    "atomic-aware validation partition has the wrong approximate size".into(),
                );
            }
            validation_sets.insert(split.validation_row_ids.clone());
        }
        if distinct && validation_sets.len() != count {
            return Err("all eight search validation partitions must be distinct".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cv7PlanV1(pub SearchSplitPlanV1);

impl Cv7PlanV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5Cv7Plan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate_n(Self::SCHEMA, 7, false)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionKeyV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub split_sha256: String,
    pub semantic_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl PredictionKeyV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5PredictionKey.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.training_sha256.as_str()),
            ("split", self.split_sha256.as_str()),
            ("semantic", self.semantic_sha256.as_str()),
            ("evaluator", self.evaluator_sha256.as_str()),
            ("math policy", self.math_policy_sha256.as_str()),
            ("backend policy", self.backend_policy_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

/// A complete, deployable predictor. Residual and portfolio winners are
/// represented directly instead of pretending that their last component is
/// the selected model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseTrackV1 {
    ProductionCompatible,
    ResearchOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalModelFamilyV1 {
    CatBoost,
    LightGbm,
    XgBoost,
    GpControl,
    TabFm,
    TabPfn,
    TabIcl,
    TabDpt,
}

impl ExternalModelFamilyV1 {
    pub fn license_track(self) -> LicenseTrackV1 {
        match self {
            Self::CatBoost | Self::LightGbm | Self::XgBoost | Self::GpControl => {
                LicenseTrackV1::ProductionCompatible
            }
            Self::TabFm | Self::TabPfn | Self::TabIcl | Self::TabDpt => {
                LicenseTrackV1::ResearchOnly
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseIdentityV1 {
    pub spdx_expression: String,
    pub license_text_sha256: String,
    pub upstream_source: String,
    pub upstream_revision: String,
}

impl LicenseIdentityV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("SPDX expression", &self.spdx_expression)?;
        validate_sha256("license text", &self.license_text_sha256)?;
        nonempty("upstream source", &self.upstream_source)?;
        nonempty("upstream revision", &self.upstream_revision)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrozenPredictorComponentV1 {
    GpPipeline {
        pipeline: GenomeV1,
    },
    ExternalModel {
        family: ExternalModelFamilyV1,
        artifact_sha256: String,
        artifact_bytes: u64,
        inference_protocol_sha256: String,
        license: LicenseIdentityV1,
        complexity: u64,
    },
}

impl FrozenPredictorComponentV1 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::GpPipeline { pipeline } => {
                pipeline.validate()?;
                if pipeline.task != GpTaskV1::Regression {
                    return Err("predictor component GP pipeline must be a regressor".into());
                }
                Ok(())
            }
            Self::ExternalModel {
                artifact_sha256,
                artifact_bytes,
                inference_protocol_sha256,
                license,
                ..
            } => {
                validate_sha256("external model artifact", artifact_sha256)?;
                if *artifact_bytes == 0 {
                    return Err("external model artifact byte length must be positive".into());
                }
                validate_sha256("external inference protocol", inference_protocol_sha256)?;
                license.validate()
            }
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn license_track(&self) -> LicenseTrackV1 {
        match self {
            Self::GpPipeline { .. } => LicenseTrackV1::ProductionCompatible,
            Self::ExternalModel { family, .. } => family.license_track(),
        }
    }

    pub fn complexity(&self) -> u64 {
        match self {
            Self::GpPipeline { pipeline } => pipeline.complexity,
            Self::ExternalModel { complexity, .. } => *complexity,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrozenPredictorV1 {
    Single {
        pipeline: GenomeV1,
    },
    ResidualComposite {
        base: GenomeV1,
        residual_pipeline: GenomeV1,
        eta: f64,
    },
    SparsePortfolio {
        pipelines: Vec<GenomeV1>,
        weights: Vec<f64>,
    },
    ComponentSingle {
        component: FrozenPredictorComponentV1,
        license_track: LicenseTrackV1,
    },
    ComponentResidualComposite {
        base: FrozenPredictorComponentV1,
        residual: FrozenPredictorComponentV1,
        eta: f64,
        cross_fit_protocol_sha256: String,
        license_track: LicenseTrackV1,
    },
    ComponentSparsePortfolio {
        components: Vec<FrozenPredictorComponentV1>,
        weights: Vec<f64>,
        cross_fit_protocol_sha256: String,
        license_track: LicenseTrackV1,
    },
}

impl FrozenPredictorV1 {
    pub fn single(pipeline: GenomeV1) -> Self {
        Self::Single { pipeline }
    }

    pub fn pipelines(&self) -> Vec<&GenomeV1> {
        match self {
            Self::Single { pipeline } => vec![pipeline],
            Self::ResidualComposite {
                base,
                residual_pipeline,
                ..
            } => vec![base, residual_pipeline],
            Self::SparsePortfolio { pipelines, .. } => pipelines.iter().collect(),
            Self::ComponentSingle { component, .. } => match component {
                FrozenPredictorComponentV1::GpPipeline { pipeline } => vec![pipeline],
                FrozenPredictorComponentV1::ExternalModel { .. } => Vec::new(),
            },
            Self::ComponentResidualComposite { base, residual, .. } => [base, residual]
                .into_iter()
                .filter_map(|component| match component {
                    FrozenPredictorComponentV1::GpPipeline { pipeline } => Some(pipeline),
                    FrozenPredictorComponentV1::ExternalModel { .. } => None,
                })
                .collect(),
            Self::ComponentSparsePortfolio { components, .. } => components
                .iter()
                .filter_map(|component| match component {
                    FrozenPredictorComponentV1::GpPipeline { pipeline } => Some(pipeline),
                    FrozenPredictorComponentV1::ExternalModel { .. } => None,
                })
                .collect(),
        }
    }

    pub fn component_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::ResidualComposite { .. } => 2,
            Self::SparsePortfolio { pipelines, .. } => pipelines.len(),
            Self::ComponentSingle { .. } => 1,
            Self::ComponentResidualComposite { .. } => 2,
            Self::ComponentSparsePortfolio { components, .. } => components.len(),
        }
    }

    pub fn license_track(&self) -> LicenseTrackV1 {
        match self {
            Self::Single { .. } | Self::ResidualComposite { .. } | Self::SparsePortfolio { .. } => {
                LicenseTrackV1::ProductionCompatible
            }
            Self::ComponentSingle { license_track, .. }
            | Self::ComponentResidualComposite { license_track, .. }
            | Self::ComponentSparsePortfolio { license_track, .. } => *license_track,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.component_count() == 0 || self.component_count() > 4 {
            return Err("frozen predictor requires one through four pipelines".into());
        }
        let mut semantics = BTreeSet::new();
        for pipeline in self.pipelines() {
            pipeline.validate()?;
            if pipeline.task != GpTaskV1::Regression
                || !semantics.insert(pipeline.semantic_sha256.as_str())
            {
                return Err("frozen predictor pipelines must be unique regressors".into());
            }
        }
        match self {
            Self::Single { .. } => Ok(()),
            Self::ResidualComposite { eta, .. } => {
                if eta.is_finite() && (0.0..=1.0).contains(eta) {
                    Ok(())
                } else {
                    Err("residual eta must be finite and in [0,1]".into())
                }
            }
            Self::SparsePortfolio { pipelines, weights } => {
                if pipelines.len() != weights.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
                {
                    return Err(
                        "portfolio weights must be a nonnegative simplex aligned to pipelines"
                            .into(),
                    );
                }
                Ok(())
            }
            Self::ComponentSingle {
                component,
                license_track,
            } => {
                component.validate()?;
                validate_component_track(*license_track, std::slice::from_ref(component))
            }
            Self::ComponentResidualComposite {
                base,
                residual,
                eta,
                cross_fit_protocol_sha256,
                license_track,
            } => {
                base.validate()?;
                residual.validate()?;
                validate_sha256("residual cross-fit protocol", cross_fit_protocol_sha256)?;
                if !eta.is_finite() || !(0.0..=1.0).contains(eta) {
                    return Err("residual eta must be finite and in [0,1]".into());
                }
                if base.identity()? == residual.identity()? {
                    return Err("residual predictor components must be unique".into());
                }
                validate_component_track(*license_track, &[base.clone(), residual.clone()])
            }
            Self::ComponentSparsePortfolio {
                components,
                weights,
                cross_fit_protocol_sha256,
                license_track,
            } => {
                validate_sha256("portfolio cross-fit protocol", cross_fit_protocol_sha256)?;
                let mut identities = BTreeSet::new();
                for component in components {
                    component.validate()?;
                    if !identities.insert(component.identity()?) {
                        return Err("portfolio predictor components must be unique".into());
                    }
                }
                if components.len() != weights.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
                {
                    return Err(
                        "portfolio weights must be a nonnegative simplex aligned to components"
                            .into(),
                    );
                }
                validate_component_track(*license_track, components)
            }
        }
    }

    pub fn semantic_sha256(&self) -> Result<String, String> {
        self.validate()?;
        match self {
            Self::Single { pipeline } => Ok(pipeline.semantic_sha256.clone()),
            Self::ResidualComposite { .. }
            | Self::SparsePortfolio { .. }
            | Self::ComponentSingle { .. }
            | Self::ComponentResidualComposite { .. }
            | Self::ComponentSparsePortfolio { .. } => canonical_sha256(self),
        }
    }

    pub fn complexity(&self) -> u64 {
        match self {
            Self::ComponentSingle { component, .. } => component.complexity(),
            Self::ComponentResidualComposite { base, residual, .. } => {
                base.complexity() + residual.complexity()
            }
            Self::ComponentSparsePortfolio { components, .. } => components
                .iter()
                .map(FrozenPredictorComponentV1::complexity)
                .sum(),
            _ => self
                .pipelines()
                .into_iter()
                .map(|pipeline| pipeline.complexity)
                .sum(),
        }
    }

    pub fn reconstruct(&self, component_predictions: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        self.validate()?;
        if component_predictions.len() != self.component_count() || component_predictions.is_empty()
        {
            return Err("predictor reconstruction has incomplete component coverage".into());
        }
        let rows = component_predictions[0].len();
        if rows == 0
            || component_predictions
                .iter()
                .any(|values| values.len() != rows || values.iter().any(|value| !value.is_finite()))
        {
            return Err("predictor component predictions are nonfinite or row-misaligned".into());
        }
        let values = match self {
            Self::Single { .. } => component_predictions[0].clone(),
            Self::ResidualComposite { eta, .. } => component_predictions[0]
                .iter()
                .zip(&component_predictions[1])
                .map(|(base, residual)| base + eta * residual)
                .collect(),
            Self::SparsePortfolio { weights, .. } => (0..rows)
                .map(|row| {
                    weights
                        .iter()
                        .zip(component_predictions)
                        .map(|(weight, prediction)| weight * prediction[row])
                        .sum()
                })
                .collect(),
            Self::ComponentSingle { .. } => component_predictions[0].clone(),
            Self::ComponentResidualComposite { eta, .. } => component_predictions[0]
                .iter()
                .zip(&component_predictions[1])
                .map(|(base, residual)| base + eta * residual)
                .collect(),
            Self::ComponentSparsePortfolio { weights, .. } => (0..rows)
                .map(|row| {
                    weights
                        .iter()
                        .zip(component_predictions)
                        .map(|(weight, prediction)| weight * prediction[row])
                        .sum()
                })
                .collect(),
        };
        Ok(values)
    }
}

fn validate_component_track(
    track: LicenseTrackV1,
    components: &[FrozenPredictorComponentV1],
) -> Result<(), String> {
    let contains_research = components
        .iter()
        .any(|component| component.license_track() == LicenseTrackV1::ResearchOnly);
    if track == LicenseTrackV1::ProductionCompatible && contains_research {
        return Err(
            "research-only component cannot enter a production-compatible predictor".into(),
        );
    }
    if track == LicenseTrackV1::ResearchOnly && !contains_research {
        return Err(
            "research-only predictor must bind at least one research-only component".into(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AncestryV1 {
    pub predictor: FrozenPredictorV1,
    pub founder_semantic_sha256: String,
    pub parent_semantic_sha256s: Vec<String>,
    pub model_seeded: bool,
}

impl AncestryV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.predictor.validate()?;
        validate_sha256("founder semantic", &self.founder_semantic_sha256)?;
        for parent in &self.parent_semantic_sha256s {
            validate_sha256("parent semantic", parent)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidenceV1 {
    pub predictor: FrozenPredictorV1,
    pub ancestry: AncestryV1,
    pub prediction_receipt_sha256: String,
    pub split_index: u32,
    pub r2_clipped: f64,
    pub anchor_median_r2_clipped: f64,
    pub paired_uplift: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
    pub failure: Option<FailureCodeV1>,
}

impl CandidateEvidenceV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.predictor.validate()?;
        self.ancestry.validate()?;
        validate_sha256(
            "candidate prediction receipt",
            &self.prediction_receipt_sha256,
        )?;
        if self.predictor != self.ancestry.predictor {
            return Err("candidate ancestry binds a different frozen predictor".into());
        }
        for (name, value) in [
            ("R2", self.r2_clipped),
            ("anchor R2", self.anchor_median_r2_clipped),
            ("uplift", self.paired_uplift),
            ("RMSE", self.rmse),
            ("runtime", self.runtime_seconds),
        ] {
            finite(name, value)?;
        }
        if !(-1.0..=1.0).contains(&self.r2_clipped)
            || !(-1.0..=1.0).contains(&self.anchor_median_r2_clipped)
            || self.rmse < 0.0
            || self.runtime_seconds < 0.0
            || (self.r2_clipped - self.anchor_median_r2_clipped - self.paired_uplift).abs() > 1e-12
        {
            return Err("candidate evidence metrics are outside the frozen math policy".into());
        }
        match (self.valid, &self.failure) {
            (true, None) | (false, Some(_)) => Ok(()),
            _ => Err("candidate validity and typed failure disagree".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMemberV1 {
    pub predictor: FrozenPredictorV1,
    pub ancestry: AncestryV1,
    pub split_evidence: Vec<CandidateEvidenceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cv7EvidenceV1 {
    pub semantic_sha256: String,
    pub predictor: FrozenPredictorV1,
    pub fold_prediction_receipt_sha256s: Vec<String>,
    pub fold_r2: Vec<f64>,
    pub pooled_oof_r2: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub complexity: u64,
    pub valid: bool,
}

impl Cv7EvidenceV1 {
    pub fn robust_score(&self) -> f64 {
        let mean = self.fold_r2.iter().sum::<f64>() / self.fold_r2.len() as f64;
        let variance = self
            .fold_r2
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / self.fold_r2.len() as f64;
        self.pooled_oof_r2 - 0.25 * variance.sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("CV7 semantic", &self.semantic_sha256)?;
        self.predictor.validate()?;
        if self.predictor.semantic_sha256()? != self.semantic_sha256 {
            return Err("CV7 semantic does not identify its frozen predictor".into());
        }
        if self.fold_r2.len() != 7
            || self.fold_prediction_receipt_sha256s.len() != 7
            || self
                .fold_prediction_receipt_sha256s
                .iter()
                .any(|receipt| validate_sha256("CV7 prediction receipt", receipt).is_err())
            || self
                .fold_r2
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err("CV7 requires seven finite clipped fold R2 scores".into());
        }
        finite("pooled OOF R2", self.pooled_oof_r2)?;
        finite("CV7 RMSE", self.rmse)?;
        finite("CV7 runtime", self.runtime_seconds)?;
        if !(-1.0..=1.0).contains(&self.pooled_oof_r2)
            || self.rmse < 0.0
            || self.runtime_seconds < 0.0
            || self.complexity != self.predictor.complexity()
        {
            return Err("CV7 metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationPopulationV1 {
    pub generation: u32,
    pub semantic_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetLedgerV1 {
    pub logical_operations: u32,
    pub predicted_cpu_seconds: u64,
    pub mandatory_anchor_evaluations: u32,
    pub completed_anchor_evaluations: u32,
    pub mandatory_search_evaluations: u32,
    pub completed_search_evaluations: u32,
    pub mandatory_archive_evaluations: u32,
    pub completed_archive_evaluations: u32,
    pub mandatory_cv7_evaluations: u32,
    pub completed_cv7_evaluations: u32,
    pub arm_specific_operations: u32,
}

impl BudgetLedgerV1 {
    pub fn validate(&self, arm: Best5ArmV1) -> Result<(), String> {
        if self.logical_operations > LOGICAL_OPERATION_CAP
            || self.predicted_cpu_seconds > PREDICTED_CPU_SECOND_CAP
        {
            return Err("search exceeded a frozen logical or predicted-CPU budget".into());
        }
        let expected_arm_specific = match arm {
            Best5ArmV1::EvolvabilityUcbLineages => 20 * 7,
            // Four residual generations each recreate five OOF base fits plus
            // one outer-validation base fit. These shared predictions are
            // still charged logical credit on cache hits.
            Best5ArmV1::CrossFittedResidualDlgp => 4 * 6,
            _ => 0,
        };
        if self.mandatory_anchor_evaluations != 10 * 8
            || self.mandatory_search_evaluations != 80 * 8
            || self.mandatory_archive_evaluations != 20 * 8
            || self.mandatory_cv7_evaluations != 10 * 7
            || self.arm_specific_operations != expected_arm_specific
            || self.logical_operations
                != self.completed_anchor_evaluations
                    + self.completed_search_evaluations
                    + self.completed_archive_evaluations
                    + self.completed_cv7_evaluations
                    + self.arm_specific_operations
        {
            return Err("logical operation ledger differs from frozen mandatory accounting".into());
        }
        if self.mandatory_anchor_evaluations != self.completed_anchor_evaluations
            || self.mandatory_search_evaluations != self.completed_search_evaluations
            || self.mandatory_archive_evaluations != self.completed_archive_evaluations
            || self.mandatory_cv7_evaluations != self.completed_cv7_evaluations
        {
            return Err("incomplete_mandatory_coverage".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchResultV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub proposal_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub master_seed: u64,
    pub split_plan_sha256: String,
    pub populations: Vec<GenerationPopulationV1>,
    pub archive: Vec<ArchiveMemberV1>,
    pub cv7_plan_sha256: String,
    pub finalists: Vec<Cv7EvidenceV1>,
    pub winner_semantic_sha256: String,
    pub ledger: BudgetLedgerV1,
    pub test_data_accessed: bool,
    pub ranking_state_shared: bool,
}

impl SearchResultV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchResult.v1";

    /// Worker completion order is deliberately outside the artifact identity.
    /// Rankings are represented by their scores and winner fields, so these
    /// set-like collections can be ordered by their stable semantic keys.
    pub fn normalize_completion_order(&mut self) {
        for population in &mut self.populations {
            population.semantic_sha256s.sort();
        }
        for member in &mut self.archive {
            member
                .split_evidence
                .sort_by_key(|evidence| evidence.split_index);
        }
        self.archive.sort_by(|left, right| {
            left.predictor
                .semantic_sha256()
                .unwrap_or_default()
                .cmp(&right.predictor.semantic_sha256().unwrap_or_default())
        });
        self.finalists
            .sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("campaign", self.campaign_freeze_sha256.as_str()),
            ("proposal", self.proposal_freeze_sha256.as_str()),
            ("dataset", self.dataset_sha256.as_str()),
            ("search split plan", self.split_plan_sha256.as_str()),
            ("CV7 plan", self.cv7_plan_sha256.as_str()),
            ("winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("search seed must be 101 or 202".into());
        }
        if self.populations.len() != GENERATIONS {
            return Err("search result requires exactly eight generations".into());
        }
        for (generation, population) in self.populations.iter().enumerate() {
            if population.generation != generation as u32
                || population.semantic_sha256s.len() != POPULATION
                || population
                    .semantic_sha256s
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != POPULATION
                || !population
                    .semantic_sha256s
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            {
                return Err(
                    "every generation must contain 80 unique, canonically ordered semantic genomes"
                        .into(),
                );
            }
            for semantic in &population.semantic_sha256s {
                validate_sha256("population semantic", semantic)?;
            }
        }
        if self.archive.len() != 20 {
            return Err("search archive must contain exactly 20 genomes".into());
        }
        let mut archive_semantics = BTreeSet::new();
        for (member_index, member) in self.archive.iter().enumerate() {
            member.predictor.validate()?;
            member.ancestry.validate()?;
            let member_semantic = member.predictor.semantic_sha256()?;
            if member.ancestry.predictor != member.predictor {
                return Err("archive ancestry binds a different predictor".into());
            }
            if !archive_semantics.insert(member_semantic.clone())
                || member.split_evidence.len() != SEARCH_SPLITS
                || member_index > 0
                    && self.archive[member_index - 1].predictor.semantic_sha256()?
                        >= member_semantic
            {
                return Err(
                    "archive must contain 20 canonically ordered unique genomes with eight-split evidence"
                        .into(),
                );
            }
            let split_indices: BTreeSet<_> = member
                .split_evidence
                .iter()
                .map(|evidence| {
                    evidence.validate()?;
                    if evidence.predictor != member.predictor {
                        return Err("archive evidence semantic mismatch".into());
                    }
                    Ok(evidence.split_index)
                })
                .collect::<Result<_, String>>()?;
            if split_indices != (0..SEARCH_SPLITS as u32).collect() {
                return Err("archive evidence does not cover all eight splits".into());
            }
            if member
                .split_evidence
                .iter()
                .enumerate()
                .any(|(index, evidence)| evidence.split_index != index as u32)
            {
                return Err("archive split evidence is not in canonical split order".into());
            }
        }
        if self.finalists.len() != 10 {
            return Err("search result requires ten CV7 finalists".into());
        }
        let mut finalists = BTreeSet::new();
        let mut derived_portfolios = 0usize;
        for (index, finalist) in self.finalists.iter().enumerate() {
            finalist.validate()?;
            let archive_member = archive_semantics.contains(finalist.semantic_sha256.as_str());
            let derived_portfolio =
                if let FrozenPredictorV1::SparsePortfolio { pipelines, .. } = &finalist.predictor {
                    derived_portfolios += 1;
                    self.arm == Best5ArmV1::SparseOofPortfolioGp
                        && pipelines
                            .iter()
                            .all(|pipeline| archive_semantics.contains(&pipeline.semantic_sha256))
                } else {
                    false
                };
            if !finalists.insert(finalist.semantic_sha256.as_str())
                || (!archive_member && !derived_portfolio)
                || index > 0
                    && self.finalists[index - 1].semantic_sha256.as_str()
                        >= finalist.semantic_sha256.as_str()
            {
                return Err(
                    "CV7 finalists must be ten canonically ordered unique archive members".into(),
                );
            }
        }
        if derived_portfolios > 1 {
            return Err("CV7 may bind only one derived sparse portfolio".into());
        }
        let winner = self
            .finalists
            .iter()
            .find(|candidate| candidate.semantic_sha256 == self.winner_semantic_sha256)
            .ok_or_else(|| "winner is not a CV7 finalist".to_string())?;
        if !winner.valid {
            return Err("search cannot freeze an invalid winner".into());
        }
        let best = self
            .finalists
            .iter()
            .filter(|candidate| candidate.valid)
            .min_by(|left, right| {
                right
                    .robust_score()
                    .total_cmp(&left.robust_score())
                    .then_with(|| left.rmse.total_cmp(&right.rmse))
                    .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                    .then_with(|| left.complexity.cmp(&right.complexity))
                    .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
            })
            .unwrap();
        if best.semantic_sha256 != self.winner_semantic_sha256 {
            return Err("winner does not follow the frozen CV7 ordering".into());
        }
        self.ledger.validate(self.arm)?;
        if self.test_data_accessed || self.ranking_state_shared {
            return Err("search result violated sealed-test or arm-isolation policy".into());
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
pub struct SeedChoiceV1 {
    pub master_seed: u64,
    pub search_result_sha256: String,
    pub winner_semantic_sha256: String,
    pub winner_predictor: FrozenPredictorV1,
    pub pooled_oof_r2: f64,
    pub fold_r2_sd: f64,
    pub rmse: f64,
    pub runtime_seconds: f64,
    pub complexity: u64,
}

impl SeedChoiceV1 {
    pub fn robust_score(&self) -> f64 {
        self.pooled_oof_r2 - 0.25 * self.fold_r2_sd
    }

    pub fn validate(&self) -> Result<(), String> {
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("seed choice must use seed 101 or 202".into());
        }
        validate_sha256("search result", &self.search_result_sha256)?;
        validate_sha256("seed winner", &self.winner_semantic_sha256)?;
        self.winner_predictor.validate()?;
        if self.winner_predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("seed winner semantic does not identify its frozen predictor".into());
        }
        for (name, value) in [
            ("pooled OOF R2", self.pooled_oof_r2),
            ("fold R2 SD", self.fold_r2_sd),
            ("RMSE", self.rmse),
            ("runtime", self.runtime_seconds),
        ] {
            finite(name, value)?;
        }
        if self.fold_r2_sd < 0.0 || self.rmse < 0.0 || self.runtime_seconds < 0.0 {
            return Err("seed selection metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionFreezeV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub choices: Vec<SeedChoiceV1>,
    pub selected_seed: u64,
    pub winner_semantic_sha256: String,
    pub winner_predictor: FrozenPredictorV1,
}

impl SelectionFreezeV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SelectionFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("selection campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("selection dataset", &self.dataset_sha256)?;
        validate_sha256("selection winner", &self.winner_semantic_sha256)?;
        self.winner_predictor.validate()?;
        if self.winner_predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("selection winner semantic does not identify its predictor".into());
        }
        if self.choices.len() != 2 {
            return Err("selection freeze requires both search seeds".into());
        }
        for choice in &self.choices {
            choice.validate()?;
        }
        if self
            .choices
            .iter()
            .map(|choice| choice.master_seed)
            .collect::<BTreeSet<_>>()
            != SEARCH_SEEDS.into_iter().collect()
            || self
                .choices
                .iter()
                .map(|choice| choice.master_seed)
                .collect::<Vec<_>>()
                != SEARCH_SEEDS
        {
            return Err("selection choices must cover seeds 101 and 202 in canonical order".into());
        }
        let best = self
            .choices
            .iter()
            .min_by(|left, right| {
                right
                    .robust_score()
                    .total_cmp(&left.robust_score())
                    .then_with(|| left.rmse.total_cmp(&right.rmse))
                    .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
                    .then_with(|| left.complexity.cmp(&right.complexity))
                    .then_with(|| {
                        left.winner_semantic_sha256
                            .cmp(&right.winner_semantic_sha256)
                    })
            })
            .unwrap();
        if best.master_seed != self.selected_seed
            || best.winner_semantic_sha256 != self.winner_semantic_sha256
            || best.winner_predictor != self.winner_predictor
        {
            return Err("selection does not follow the frozen seed ordering".into());
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
pub struct SealedScoreV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub selection_freeze_sha256: String,
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub winner_semantic_sha256: String,
    pub predictor: FrozenPredictorV1,
    pub test_access_ordinal: u32,
    pub r2: f64,
    pub rmse: f64,
    pub latency_seconds: f64,
    pub validity: f64,
    pub ancestry: AncestryV1,
}

impl SealedScoreV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedScore.v1";

    pub fn validate(&self, selection: &SelectionFreezeV1) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("sealed campaign", self.campaign_freeze_sha256.as_str()),
            ("sealed selection", self.selection_freeze_sha256.as_str()),
            ("sealed dataset", self.dataset_sha256.as_str()),
            ("sealed test", self.test_sha256.as_str()),
            ("sealed winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.selection_freeze_sha256 != selection.identity()?
            || self.campaign_freeze_sha256 != selection.campaign_freeze_sha256
            || self.arm != selection.arm
            || self.dataset_sha256 != selection.dataset_sha256
            || self.winner_semantic_sha256 != selection.winner_semantic_sha256
            || self.predictor != selection.winner_predictor
        {
            return Err("sealed score does not bind the selection freeze".into());
        }
        if self.test_access_ordinal != 1 {
            return Err("sealed test may be evaluated exactly once".into());
        }
        self.predictor.validate()?;
        if self.predictor.semantic_sha256()? != self.winner_semantic_sha256 {
            return Err("sealed predictor semantic binding changed".into());
        }
        for (name, value) in [
            ("sealed R2", self.r2),
            ("sealed RMSE", self.rmse),
            ("sealed latency", self.latency_seconds),
            ("sealed validity", self.validity),
        ] {
            finite(name, value)?;
        }
        if self.rmse < 0.0 || self.latency_seconds < 0.0 || !(0.0..=1.0).contains(&self.validity) {
            return Err("sealed metrics are outside their domains".into());
        }
        self.ancestry.validate()
    }

    pub fn identity(&self, selection: &SelectionFreezeV1) -> Result<String, String> {
        self.validate(selection)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionReceiptV1 {
    pub schema: String,
    pub key: PredictionKeyV1,
    pub prediction_object_sha256: String,
    pub logical_operation_ordinal: u32,
}

impl PredictionReceiptV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5PredictionReceipt.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.key.validate()?;
        validate_sha256("prediction object", &self.prediction_object_sha256)?;
        if self.logical_operation_ordinal == 0 {
            return Err("prediction receipt must charge a logical operation".into());
        }
        Ok(())
    }

    pub fn validate_object(&self, prediction: &PredictionObjectV1) -> Result<(), String> {
        self.validate()?;
        prediction.validate()?;
        if canonical_sha256(prediction)? != self.prediction_object_sha256
            || prediction.training_view_sha256 != self.key.training_sha256
            || prediction.fold_plan_sha256 != self.key.split_sha256
            || prediction.candidate_semantic_sha256 != self.key.semantic_sha256
            || prediction.evaluator_sha256 != self.key.evaluator_sha256
            || prediction.math_policy_sha256 != self.key.math_policy_sha256
            || prediction.backend_policy_sha256 != self.key.backend_policy_sha256
        {
            return Err("prediction receipt does not bind its content-addressed object".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchStagingAttestationV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub proposal_batch_sha256: String,
    pub dataset_sha256: String,
    pub training: FrozenFileV1,
    pub metadata: FrozenFileV1,
    pub test_artifacts_staged: bool,
}

impl SearchStagingAttestationV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchStagingAttestation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("staged campaign", self.campaign_freeze_sha256.as_str()),
            ("staged proposals", self.proposal_batch_sha256.as_str()),
            ("staged dataset", self.dataset_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        self.training.validate("staged training")?;
        self.metadata.validate("staged metadata")?;
        if self.test_artifacts_staged {
            return Err("search staging must not contain test files or hashes".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBundleV1 {
    pub schema: String,
    pub staging: SearchStagingAttestationV1,
    pub target_rows: Vec<TargetRowV1>,
    pub split_plan: SearchSplitPlanV1,
    pub cv7_plan: Cv7PlanV1,
    pub anchor_receipt_sets: Vec<AnchorReceiptSetV1>,
    pub prediction_receipts: Vec<PredictionReceiptV1>,
    pub result: SearchResultV1,
}

impl SearchBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchBundle.v1";

    pub fn normalize_completion_order(&mut self) {
        self.result.normalize_completion_order();
        self.anchor_receipt_sets
            .sort_by_key(|anchors| anchors.split_index);
        self.prediction_receipts.sort_by(|left, right| {
            left.logical_operation_ordinal
                .cmp(&right.logical_operation_ordinal)
                .then_with(|| left.key.identity().ok().cmp(&right.key.identity().ok()))
        });
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.staging.validate()?;
        self.split_plan.validate()?;
        self.cv7_plan.validate()?;
        self.result.validate()?;
        if self.target_rows.is_empty() {
            return Err("search bundle lacks staged training targets".into());
        }
        let mut row_ids = BTreeSet::new();
        for row in &self.target_rows {
            row.validate()?;
            if !row_ids.insert(row.stable_row_id.as_str()) {
                return Err("search bundle contains duplicate training row IDs".into());
            }
        }
        if self.staging.campaign_freeze_sha256 != self.result.campaign_freeze_sha256
            || self.staging.proposal_batch_sha256 != self.result.proposal_freeze_sha256
            || self.staging.dataset_sha256 != self.result.dataset_sha256
            || self.staging.training.sha256 != self.split_plan.training_sha256
            || self.split_plan.dataset_sha256 != self.result.dataset_sha256
            || self.split_plan.master_seed != self.result.master_seed
            || self.split_plan.identity()? != self.result.split_plan_sha256
            || self.cv7_plan.identity()? != self.result.cv7_plan_sha256
        {
            return Err("search bundle bindings disagree".into());
        }
        if self.prediction_receipts.len() != self.result.ledger.logical_operations as usize {
            return Err("search bundle lacks one receipt per charged logical operation".into());
        }
        if self.anchor_receipt_sets.len() != SEARCH_SPLITS
            || self
                .anchor_receipt_sets
                .iter()
                .enumerate()
                .any(|(index, anchors)| {
                    anchors.split_index != index as u32 || anchors.validate().is_err()
                })
        {
            return Err("search bundle requires ten anchor receipts on every search split".into());
        }
        let mut identities = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        for receipt in &self.prediction_receipts {
            receipt.validate()?;
            if receipt.key.dataset_sha256 != self.result.dataset_sha256
                || receipt.key.training_sha256 != self.staging.training.sha256
                || !identities.insert(receipt.identity()?)
                || !ordinals.insert(receipt.logical_operation_ordinal)
            {
                return Err("search prediction receipts are duplicate or foreign".into());
            }
        }
        if ordinals != (1..=self.result.ledger.logical_operations).collect::<BTreeSet<_>>() {
            return Err("search logical-operation receipt ordinals are not contiguous".into());
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
pub struct AnchorReceiptSetV1 {
    pub split_index: u32,
    pub prediction_receipt_sha256s: Vec<String>,
    pub median_r2_clipped: f64,
}

impl AnchorReceiptSetV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.prediction_receipt_sha256s.len() != 10
            || self
                .prediction_receipt_sha256s
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != 10
            || !self.median_r2_clipped.is_finite()
            || !(-1.0..=1.0).contains(&self.median_r2_clipped)
        {
            return Err("anchor receipt set is incomplete or invalid".into());
        }
        for receipt in &self.prediction_receipt_sha256s {
            validate_sha256("anchor prediction receipt", receipt)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedClaimV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub dataset_sha256: String,
    pub selection_freeze_sha256s: Vec<String>,
    pub access_ordinal: u32,
}

impl SealedClaimV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedClaim.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed claim campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("sealed claim dataset", &self.dataset_sha256)?;
        if self.access_ordinal != 1
            || self.selection_freeze_sha256s.len() < 2
            || self.selection_freeze_sha256s.len() > Best5ArmV1::ALL.len()
            || !self
                .selection_freeze_sha256s
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        {
            return Err("sealed claim requires one canonical dataset-level selection set".into());
        }
        for selection in &self.selection_freeze_sha256s {
            validate_sha256("sealed claim selection", selection)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedStagingAttestationV1 {
    pub schema: String,
    pub claim_sha256: String,
    pub training: FrozenFileV1,
    pub test: FrozenFileV1,
}

impl SealedStagingAttestationV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedStagingAttestation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed staged claim", &self.claim_sha256)?;
        self.training.validate("sealed training")?;
        self.test.validate("sealed test")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedBundleV1 {
    pub schema: String,
    pub claim: SealedClaimV1,
    pub staging: SealedStagingAttestationV1,
    pub selections: Vec<SelectionFreezeV1>,
    pub prediction_receipts: Vec<PredictionReceiptV1>,
    pub scores: Vec<SealedScoreV1>,
}

impl SealedBundleV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedBundle.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.claim.validate()?;
        self.staging.validate()?;
        if self.staging.claim_sha256 != self.claim.identity()?
            || self.selections.len() != self.claim.selection_freeze_sha256s.len()
            || self.scores.len() != self.selections.len()
            || self.prediction_receipts.len() != self.selections.len()
        {
            return Err("sealed bundle has incomplete dataset-level coverage".into());
        }
        let mut selection_ids = Vec::new();
        let mut arms = BTreeSet::new();
        for selection in &self.selections {
            selection.validate()?;
            if selection.campaign_freeze_sha256 != self.claim.campaign_freeze_sha256
                || selection.dataset_sha256 != self.claim.dataset_sha256
                || !arms.insert(selection.arm)
            {
                return Err("sealed selection set contains a foreign or duplicate arm".into());
            }
            selection_ids.push(selection.identity()?);
        }
        selection_ids.sort();
        if selection_ids != self.claim.selection_freeze_sha256s {
            return Err("sealed claim does not identify the supplied selection set".into());
        }
        for score in &self.scores {
            let selection = self
                .selections
                .iter()
                .find(|selection| selection.arm == score.arm)
                .ok_or_else(|| "sealed score has no matching selection".to_string())?;
            score.validate(selection)?;
            if score.test_sha256 != self.staging.test.sha256 {
                return Err("sealed score test hash differs from staged bytes".into());
            }
        }
        for receipt in &self.prediction_receipts {
            receipt.validate()?;
            if receipt.key.dataset_sha256 != self.claim.dataset_sha256
                || receipt.key.training_sha256 != self.staging.training.sha256
            {
                return Err("sealed prediction receipt is foreign".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedDatasetReceiptV1 {
    pub schema: String,
    pub sealed_bundle_sha256: String,
    pub claim_sha256: String,
    pub score_sha256s: Vec<String>,
}

impl SealedDatasetReceiptV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SealedDatasetReceipt.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("sealed bundle", &self.sealed_bundle_sha256)?;
        validate_sha256("sealed dataset claim", &self.claim_sha256)?;
        if self.score_sha256s.len() < 2
            || !self.score_sha256s.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err("sealed dataset receipt requires sorted unique score artifacts".into());
        }
        for score in &self.score_sha256s {
            validate_sha256("sealed score artifact", score)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonScoreV1 {
    pub arm: Best5ArmV1,
    pub dataset_sha256: String,
    pub rmse: f64,
    pub latency_seconds: f64,
    pub finalist_validity: f64,
    pub overall_validity: f64,
    pub strict_benchmark_win: bool,
    pub preserved_existing_win: bool,
    pub predicted_cpu_seconds: f64,
}

impl ComparisonScoreV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("comparison dataset", &self.dataset_sha256)?;
        for (name, value) in [
            ("RMSE", self.rmse),
            ("latency", self.latency_seconds),
            ("finalist validity", self.finalist_validity),
            ("overall validity", self.overall_validity),
            ("predicted CPU", self.predicted_cpu_seconds),
        ] {
            finite(name, value)?;
        }
        if self.rmse <= 0.0
            || self.latency_seconds < 0.0
            || self.predicted_cpu_seconds < 0.0
            || !(0.0..=1.0).contains(&self.finalist_validity)
            || !(0.0..=1.0).contains(&self.overall_validity)
        {
            return Err("comparison metrics are outside their domains".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmComparisonV1 {
    pub arm: Best5ArmV1,
    pub mean_delta: f64,
    pub median_delta: f64,
    pub one_sided_95_lower_bound: f64,
    pub paired_wins: u32,
    pub strict_benchmark_wins: u32,
    pub predicted_cpu_seconds: f64,
    pub advances: bool,
    pub failure_codes: Vec<FailureCodeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCodeV1 {
    MalformedProposal,
    UnsupportedHead,
    MissingCheckpoint,
    CheckpointHashMismatch,
    CandidateTimeout,
    StaleEvidence,
    OutputCollision,
    CasCorruption,
    IncompleteSplitCoverage,
    InvalidCv7,
    TestAccessBeforeSelection,
    NonpositiveLowerBound,
    NegativeMedian,
    IncompleteSealedResults,
    FinalistValidity,
    OverallValidity,
    ExistingBenchmarkWinLost,
    LatencyGate,
    ConfirmationPairedWins,
    ConfirmationWinPreservation,
    LaunchGateBlocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CampaignDecisionV1 {
    PilotWinner { arm: Best5ArmV1 },
    Adopt { arm: Best5ArmV1 },
    NoImprovement { failure_codes: Vec<FailureCodeV1> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub stage: DatasetStageV1,
    pub bootstrap_seed: u64,
    pub bootstrap_repetitions: u32,
    pub dataset_count: u32,
    pub arms: Vec<ArmComparisonV1>,
    pub decision: CampaignDecisionV1,
}

impl ComparisonV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5Comparison.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("comparison campaign", &self.campaign_freeze_sha256)?;
        let expected_count = match self.stage {
            DatasetStageV1::Pilot => PILOT_DATASETS,
            DatasetStageV1::Confirmation => CONFIRMATION_DATASETS,
        };
        if self.bootstrap_seed != BOOTSTRAP_SEED
            || self.bootstrap_repetitions != 10_000
            || self.dataset_count != expected_count as u32
        {
            return Err("comparison bootstrap or dataset cardinality is not frozen".into());
        }
        let expected_arms: BTreeSet<_> = match self.stage {
            DatasetStageV1::Pilot => Best5ArmV1::EXPERIMENTS.into_iter().collect(),
            DatasetStageV1::Confirmation => self.arms.iter().map(|arm| arm.arm).collect(),
        };
        if self.arms.iter().map(|arm| arm.arm).collect::<BTreeSet<_>>() != expected_arms
            || (self.stage == DatasetStageV1::Pilot && self.arms.len() != 5)
            || (self.stage == DatasetStageV1::Confirmation && self.arms.len() != 1)
        {
            return Err("comparison contains the wrong arm set".into());
        }
        for arm in &self.arms {
            for value in [
                arm.mean_delta,
                arm.median_delta,
                arm.one_sided_95_lower_bound,
                arm.predicted_cpu_seconds,
            ] {
                finite("arm comparison", value)?;
            }
            if !arm.failure_codes.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err("comparison failure codes must be sorted and unique".into());
            }
        }
        match self.decision {
            CampaignDecisionV1::PilotWinner { arm } if self.stage == DatasetStageV1::Pilot => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("pilot winner did not clear every gate".into());
                }
            }
            CampaignDecisionV1::Adopt { arm } if self.stage == DatasetStageV1::Confirmation => {
                if !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
                {
                    return Err("adopted arm did not clear confirmation".into());
                }
            }
            CampaignDecisionV1::NoImprovement { ref failure_codes }
                if !failure_codes.is_empty()
                    && failure_codes.windows(2).all(|pair| pair[0] < pair[1]) => {}
            _ => return Err("comparison decision does not match its stage or gate state".into()),
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
pub struct LaunchEvidenceV1 {
    pub schema: String,
    pub scq_doctor_success: bool,
    pub evaluator_evidence_age_seconds: u64,
    pub supply_evidence_age_seconds: u64,
    pub max_evidence_age_seconds: u64,
    pub reviewed_action_acknowledged: bool,
    pub observed_project_priority_weight: u32,
    pub eligible_worker_disk_free_fractions: Vec<f64>,
}

impl LaunchEvidenceV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5LaunchEvidence.v1";

    pub fn launch_allowed(&self) -> Result<bool, String> {
        self.launch_allowed_for(&ScqPolicyV1::frozen())
    }

    pub fn launch_allowed_for(&self, policy: &ScqPolicyV1) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        policy.validate()?;
        if self.max_evidence_age_seconds == 0 || self.eligible_worker_disk_free_fractions.is_empty()
        {
            return Err("launch evidence lacks freshness or eligible workers".into());
        }
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self.scq_doctor_success
            && self.evaluator_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.supply_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.reviewed_action_acknowledged
            && self.observed_project_priority_weight == policy.priority_weight
            && self
                .eligible_worker_disk_free_fractions
                .iter()
                .all(|fraction| {
                    *fraction >= policy.min_worker_disk_free_fraction_micros as f64 / 1_000_000.0
                }))
    }

    pub fn disk_warning(&self) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| *fraction < 0.05))
    }
}
