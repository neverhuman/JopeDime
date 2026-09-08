use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use feat_math::genome::GenomeIr;
use serde::{Deserialize, Serialize};

use crate::search_v3::{canonical_sha256, validate_sha256, PredictionObjectV1};

pub const REGRESSION_SHADOW_PREFIX: &str = "e0aef7d74afd";
pub const BINARY_SHADOW_PREFIX: &str = "13b1c6bbe3e7";
pub const DEVELOPMENT_REGRESSION_DATASETS: u64 = 17_263;
pub const DEVELOPMENT_BINARY_DATASETS: u64 = 5_030;
pub const VALIDATION_REGRESSION_DATASETS: u64 = 1_928;
pub const VALIDATION_BINARY_DATASETS: u64 = 569;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpTaskV1 {
    Regression,
    Binary,
}

impl GpTaskV1 {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Binary => "binary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogPartitionV1 {
    Development,
    Validation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetOriginV1 {
    Real,
    Synthetic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogDatasetV1 {
    pub dataset_sha256: String,
    pub task: GpTaskV1,
    pub partition: CatalogPartitionV1,
    pub origin: DatasetOriginV1,
    pub source_family: String,
    pub row_count: u64,
    pub feature_count: u64,
    pub missing_fraction: f64,
    /// Regression target dispersion/skew proxy or binary positive prevalence.
    pub target_distribution: f64,
    pub sotarena_dataset: bool,
    pub current_strict_win: bool,
}

impl CatalogDatasetV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("catalog dataset", &self.dataset_sha256)?;
        nonempty("source_family", &self.source_family)?;
        if self.row_count == 0 || self.feature_count == 0 {
            return Err("catalog datasets must declare positive rows and features".into());
        }
        if !self.missing_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.missing_fraction)
            || !self.target_distribution.is_finite()
        {
            return Err("catalog coverage values are invalid".into());
        }
        if self.current_strict_win && !self.sotarena_dataset {
            return Err("current strict wins must identify their SOTArena dataset".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogCountsV1 {
    pub development_regression: u64,
    pub development_binary: u64,
    pub validation_regression: u64,
    pub validation_binary: u64,
}

impl CatalogCountsV1 {
    pub const PRODUCTION: Self = Self {
        development_regression: DEVELOPMENT_REGRESSION_DATASETS,
        development_binary: DEVELOPMENT_BINARY_DATASETS,
        validation_regression: VALIDATION_REGRESSION_DATASETS,
        validation_binary: VALIDATION_BINARY_DATASETS,
    };

    pub fn total_development(self) -> u64 {
        self.development_regression + self.development_binary
    }

    pub fn total_validation(self) -> u64 {
        self.validation_regression + self.validation_binary
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogFreezeV1 {
    pub schema: String,
    pub observation_snapshot_sha256: String,
    pub counts: CatalogCountsV1,
    pub datasets: Vec<CatalogDatasetV1>,
}

impl CatalogFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCatalogFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("observation snapshot", &self.observation_snapshot_sha256)?;
        let mut seen = BTreeSet::new();
        let mut actual = CatalogCountsV1::default();
        for dataset in &self.datasets {
            dataset.validate()?;
            if dataset.origin != DatasetOriginV1::Real {
                return Err("a frozen benchmark catalog may contain only real datasets".into());
            }
            if !seen.insert(dataset.dataset_sha256.as_str()) {
                return Err(format!(
                    "duplicate catalog dataset {}",
                    dataset.dataset_sha256
                ));
            }
            match (dataset.partition, dataset.task) {
                (CatalogPartitionV1::Development, GpTaskV1::Regression) => {
                    actual.development_regression += 1
                }
                (CatalogPartitionV1::Development, GpTaskV1::Binary) => {
                    actual.development_binary += 1
                }
                (CatalogPartitionV1::Validation, GpTaskV1::Regression) => {
                    actual.validation_regression += 1
                }
                (CatalogPartitionV1::Validation, GpTaskV1::Binary) => actual.validation_binary += 1,
            }
        }
        if actual != self.counts {
            return Err(format!(
                "catalog count freeze {:?} does not match entries {:?}",
                self.counts, actual
            ));
        }
        Ok(())
    }

    pub fn validate_production_cardinality(&self) -> Result<(), String> {
        self.validate()?;
        if self.counts != CatalogCountsV1::PRODUCTION {
            return Err(format!(
                "production catalog must freeze {:?}, found {:?}",
                CatalogCountsV1::PRODUCTION,
                self.counts
            ));
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
pub struct FrozenArtifactV1 {
    pub path: String,
    pub sha256: String,
    pub byte_length: u64,
}

impl FrozenArtifactV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("artifact path", &self.path)?;
        validate_sha256("artifact", &self.sha256)?;
        if self.byte_length == 0 {
            return Err("artifact byte length must be positive".into());
        }
        Ok(())
    }

    pub fn verify(&self) -> Result<(), String> {
        self.validate()?;
        let bytes = fs::read(&self.path)
            .map_err(|error| format!("read frozen artifact {}: {error}", self.path))?;
        if bytes.len() as u64 != self.byte_length {
            return Err(format!("frozen artifact {} byte length changed", self.path));
        }
        // Artifacts are executable/checkpoint bytes, not JSON values.
        let actual = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(&bytes))
        };
        if actual != self.sha256 {
            return Err(format!("frozen artifact {} hash changed", self.path));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelShadowFreezeV1 {
    pub task: GpTaskV1,
    pub artifact: FrozenArtifactV1,
    pub task_head_schema_sha256: String,
    pub inference_protocol_sha256: String,
    pub inference_only: bool,
    pub production_promoted: bool,
}

impl ModelShadowFreezeV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.artifact.validate()?;
        validate_sha256("task-head schema", &self.task_head_schema_sha256)?;
        validate_sha256("inference protocol", &self.inference_protocol_sha256)?;
        let prefix = match self.task {
            GpTaskV1::Regression => REGRESSION_SHADOW_PREFIX,
            GpTaskV1::Binary => BINARY_SHADOW_PREFIX,
        };
        if !self.artifact.sha256.starts_with(prefix) {
            return Err(format!(
                "{} shadow must use the frozen {prefix}… checkpoint",
                self.task.as_str()
            ));
        }
        if !self.inference_only || self.production_promoted {
            return Err("campaign shadows must be inference-only and unpromoted".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFreezeV1 {
    pub schema: String,
    pub regression: ModelShadowFreezeV1,
    pub binary: ModelShadowFreezeV1,
}

impl CheckpointFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCheckpointFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.regression.validate()?;
        self.binary.validate()?;
        if self.regression.task != GpTaskV1::Regression || self.binary.task != GpTaskV1::Binary {
            return Err("checkpoint freeze task slots are swapped".into());
        }
        Ok(())
    }

    pub fn verify_artifacts(&self) -> Result<(), String> {
        self.validate()?;
        self.regression.artifact.verify()?;
        self.binary.artifact.verify()
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorFreezeV1 {
    pub schema: String,
    pub artifact: FrozenArtifactV1,
    pub evaluator_protocol_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl EvaluatorFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpEvaluatorFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        self.artifact.validate()?;
        validate_sha256("evaluator protocol", &self.evaluator_protocol_sha256)?;
        validate_sha256("math policy", &self.math_policy_sha256)?;
        validate_sha256("backend policy", &self.backend_policy_sha256)
    }

    pub fn verify_artifact(&self) -> Result<(), String> {
        self.validate()?;
        self.artifact.verify()
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignArmV1 {
    RotatingMean,
    ConfidenceGp,
    QualityDiversityGp,
    LiteRaceControl,
}

impl CampaignArmV1 {
    pub const EXPERIMENTS: [Self; 3] = [
        Self::RotatingMean,
        Self::ConfidenceGp,
        Self::QualityDiversityGp,
    ];

    pub fn eligible(self) -> bool {
        self != Self::LiteRaceControl
    }

    pub fn simplicity_order(self) -> usize {
        match self {
            Self::RotatingMean => 0,
            Self::ConfidenceGp => 1,
            Self::QualityDiversityGp => 2,
            Self::LiteRaceControl => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvolutionBudgetV1 {
    pub population: u32,
    pub evaluated_generations: u32,
    pub search_splits: u32,
    pub elites: u32,
    pub tournament_size: u32,
    pub parent_pool: u32,
    pub mutations: u32,
    pub crossovers: u32,
    pub provisional_archive: u32,
    pub cv7_finalists: u32,
    pub cv_folds: u32,
    pub predicted_compute_cap: u64,
}

impl EvolutionBudgetV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.population != 80
            || self.evaluated_generations != 5
            || self.search_splits != 5
            || self.elites != 3
            || self.tournament_size != 3
            || self.parent_pool != 40
            || self.mutations != 58
            || self.crossovers != 19
            || self.elites + self.mutations + self.crossovers != self.population
            || self.provisional_archive != 20
            || self.cv7_finalists != 10
            || self.cv_folds != 7
            || self.predicted_compute_cap == 0
        {
            return Err(
                "campaign budget must use the frozen 80/5/3/58/19/20/10/CV7 schedule".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub catalog_freeze_sha256: String,
    pub checkpoint_freeze_sha256: String,
    pub evaluator_freeze_sha256: String,
    pub observation_snapshot_sha256: String,
    pub generated_registry_sha256: String,
    pub split_seed_sha256: String,
    pub bootstrap_seed_sha256: String,
    pub budgets: BTreeMap<CampaignArmV1, EvolutionBudgetV1>,
    pub production_pointer_writes: bool,
    pub production_learning_ingest: bool,
}

impl CampaignFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCampaignFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        nonempty("campaign_id", &self.campaign_id)?;
        for (name, digest) in [
            ("catalog freeze", self.catalog_freeze_sha256.as_str()),
            ("checkpoint freeze", self.checkpoint_freeze_sha256.as_str()),
            ("evaluator freeze", self.evaluator_freeze_sha256.as_str()),
            (
                "observation snapshot",
                self.observation_snapshot_sha256.as_str(),
            ),
            (
                "generated registry",
                self.generated_registry_sha256.as_str(),
            ),
            ("split seed", self.split_seed_sha256.as_str()),
            ("bootstrap seed", self.bootstrap_seed_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        let required: BTreeSet<_> = CampaignArmV1::EXPERIMENTS
            .into_iter()
            .chain([CampaignArmV1::LiteRaceControl])
            .collect();
        if self.budgets.keys().copied().collect::<BTreeSet<_>>() != required {
            return Err("campaign must freeze three experiments and one matched control".into());
        }
        for budget in self.budgets.values() {
            budget.validate()?;
        }
        let caps: BTreeSet<_> = self
            .budgets
            .values()
            .map(|budget| budget.predicted_compute_cap)
            .collect();
        if caps.len() != 1 {
            return Err("pilot arms and control must have a matched predicted-compute cap".into());
        }
        if self.production_pointer_writes || self.production_learning_ingest {
            return Err(
                "experimental campaigns cannot mutate production pointers or learning".into(),
            );
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
pub struct PilotDatasetFreezeV1 {
    pub schema: String,
    pub catalog_freeze_sha256: String,
    pub selected: BTreeMap<GpTaskV1, Vec<String>>,
    pub alternates: BTreeMap<GpTaskV1, Vec<String>>,
    pub first_arm_started: bool,
}

impl PilotDatasetFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPilotDatasetFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("pilot catalog freeze", &self.catalog_freeze_sha256)?;
        for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
            let selected = self
                .selected
                .get(&task)
                .ok_or_else(|| format!("pilot lacks {} selection", task.as_str()))?;
            if selected.len() != 10 {
                return Err(format!("pilot requires 10 {} datasets", task.as_str()));
            }
            let alternates = self
                .alternates
                .get(&task)
                .ok_or_else(|| format!("pilot alternates must be frozen for {}", task.as_str()))?;
            if alternates.is_empty() {
                return Err(format!(
                    "pilot requires at least one frozen {} alternate",
                    task.as_str()
                ));
            }
            let union: BTreeSet<_> = selected.iter().chain(alternates).collect();
            if union.len() != selected.len() + alternates.len() {
                return Err("pilot selections and alternates must be unique and disjoint".into());
            }
            for digest in union {
                validate_sha256("pilot dataset", digest)?;
            }
        }
        Ok(())
    }

    pub fn replacement_allowed(&self) -> bool {
        !self.first_arm_started
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeV1 {
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub task: GpTaskV1,
    pub head_family: String,
    pub transform_count: u32,
    pub complexity: u64,
}

impl GenomeV1 {
    pub fn from_canonical(task: GpTaskV1, source: &str) -> Result<Self, String> {
        let parsed = GenomeIr::parse(source).map_err(|error| format!("parse genome: {error:#}"))?;
        let canonical = parsed.canonical();
        if canonical != source {
            return Err("genome is not canonical".into());
        }
        let GenomeIr::Pipeline(pipeline) = parsed else {
            return Err("generalized GP accepts only pipeline genomes".into());
        };
        let head = pipeline
            .stages
            .last()
            .ok_or_else(|| "pipeline has no task head".to_string())?
            .op
            .clone();
        if !task_head_valid(task, &head)
            || pipeline.stages[..pipeline.stages.len().saturating_sub(1)]
                .iter()
                .any(|stage| is_head_like(&stage.op))
        {
            return Err(format!("task-incompatible or misplaced head {head}"));
        }
        let transform_count = pipeline.stages.len().saturating_sub(1) as u32;
        if transform_count > 6 {
            return Err("generalized GP genomes permit at most six transforms".into());
        }
        Ok(Self {
            semantic_sha256: raw_sha256(source.as_bytes()),
            canonical_genome: source.to_string(),
            task,
            head_family: head,
            transform_count,
            complexity: pipeline.stages.len() as u64,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        let rebuilt = Self::from_canonical(self.task, &self.canonical_genome)?;
        if &rebuilt != self {
            return Err("genome metadata does not match its canonical semantics".into());
        }
        Ok(())
    }

    pub fn niche(&self) -> String {
        let bucket = match self.transform_count {
            0..=2 => "0-2",
            3..=4 => "3-4",
            _ => "5-6",
        };
        format!("{}:{bucket}", self.head_family)
    }
}

fn is_head_like(op: &str) -> bool {
    feat_math::heads::is_head_op(op)
        || matches!(
            op,
            "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls"
        )
}

pub fn task_head_valid(task: GpTaskV1, head: &str) -> bool {
    match task {
        GpTaskV1::Binary => matches!(
            head,
            "logit_ridge" | "logistic" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
        ),
        GpTaskV1::Regression => {
            head != "logit_ridge"
                && head != "logistic"
                && (feat_math::heads::is_head_op(head)
                    || matches!(head, "tab_boost" | "extra_trees" | "gam_boost" | "pls"))
        }
    }
}

pub fn raw_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenomeOriginV1 {
    ModelProposal,
    GlobalGene,
    DiversitySeed,
    Elite,
    Mutation,
    Crossover,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeAncestryV1 {
    pub origin: GenomeOriginV1,
    pub source_rank: Option<u32>,
    pub parents: Vec<String>,
    pub seed: u64,
}

impl GenomeAncestryV1 {
    pub fn validate(&self) -> Result<(), String> {
        for parent in &self.parents {
            validate_sha256("ancestry parent", parent)?;
        }
        match self.origin {
            GenomeOriginV1::ModelProposal
            | GenomeOriginV1::GlobalGene
            | GenomeOriginV1::DiversitySeed
                if self.source_rank.is_some() && self.parents.is_empty() =>
            {
                Ok(())
            }
            GenomeOriginV1::Elite | GenomeOriginV1::Mutation if self.parents.len() == 1 => Ok(()),
            GenomeOriginV1::Crossover if self.parents.len() == 2 => Ok(()),
            _ => Err("ancestry origin, source rank, and parent arity disagree".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationMemberV1 {
    pub slot: u32,
    pub genome: GenomeV1,
    pub ancestry: GenomeAncestryV1,
}

impl PopulationMemberV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.genome.validate()?;
        self.ancestry.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProposalFreezeV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub checkpoint_sha256: String,
    pub inference_request_sha256: String,
    pub inference_receipt_sha256: String,
    pub inference_ordinal: u32,
    pub proposals: Vec<GenomeV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceLedgerEntryV1 {
    pub dataset_sha256: String,
    pub proposal_freeze_sha256: String,
    pub inference_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceLedgerV1 {
    pub schema: String,
    pub entries: Vec<InferenceLedgerEntryV1>,
}

impl InferenceLedgerV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpInferenceLedger.v1";

    pub fn empty() -> Self {
        Self {
            schema: Self::SCHEMA.into(),
            entries: Vec::new(),
        }
    }

    /// Records the one permitted model inference. An identical retry is
    /// idempotent; any second result for the dataset is rejected.
    pub fn record(&mut self, proposal: &ProposalFreezeV1) -> Result<String, String> {
        proposal.validate()?;
        let proposal_sha256 = proposal.identity()?;
        if let Some(existing) = self
            .entries
            .iter()
            .find(|entry| entry.dataset_sha256 == proposal.dataset_sha256)
        {
            if existing.proposal_freeze_sha256 == proposal_sha256
                && existing.inference_receipt_sha256 == proposal.inference_receipt_sha256
            {
                return Ok(proposal_sha256);
            }
            return Err("one-inference rule forbids a second dataset inference".into());
        }
        self.entries.push(InferenceLedgerEntryV1 {
            dataset_sha256: proposal.dataset_sha256.clone(),
            proposal_freeze_sha256: proposal_sha256.clone(),
            inference_receipt_sha256: proposal.inference_receipt_sha256.clone(),
        });
        self.entries
            .sort_by(|left, right| left.dataset_sha256.cmp(&right.dataset_sha256));
        self.validate()?;
        Ok(proposal_sha256)
    }

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        let mut datasets = BTreeSet::new();
        for entry in &self.entries {
            validate_sha256("inference dataset", &entry.dataset_sha256)?;
            validate_sha256("proposal freeze", &entry.proposal_freeze_sha256)?;
            validate_sha256("inference receipt", &entry.inference_receipt_sha256)?;
            if !datasets.insert(entry.dataset_sha256.as_str()) {
                return Err("inference ledger contains multiple entries for one dataset".into());
            }
        }
        if !self
            .entries
            .windows(2)
            .all(|pair| pair[0].dataset_sha256 < pair[1].dataset_sha256)
        {
            return Err("inference ledger entries must be sorted".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl ProposalFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpProposalFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("proposal dataset", self.dataset_sha256.as_str()),
            ("proposal checkpoint", self.checkpoint_sha256.as_str()),
            ("inference request", self.inference_request_sha256.as_str()),
            ("inference receipt", self.inference_receipt_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.inference_ordinal != 1 || self.proposals.len() != 64 {
            return Err("each dataset requires exactly one inference yielding 64 proposals".into());
        }
        let tasks: BTreeSet<_> = self.proposals.iter().map(|value| value.task).collect();
        let semantics: BTreeSet<_> = self
            .proposals
            .iter()
            .map(|value| value.semantic_sha256.as_str())
            .collect();
        if tasks.len() != 1 || semantics.len() != 64 {
            return Err("model proposals must be unique and task-separated".into());
        }
        for proposal in &self.proposals {
            proposal.validate()?;
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
pub struct GenerationZeroV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub proposal_freeze_sha256: String,
    pub population: Vec<PopulationMemberV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalGeneObservationV1 {
    pub observation_snapshot_sha256: String,
    pub dataset_sha256: String,
    pub source_family: String,
    pub genome: GenomeV1,
    pub within_dataset_percentile: f64,
    pub strict_win: bool,
}

impl GlobalGeneObservationV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256(
            "global observation snapshot",
            &self.observation_snapshot_sha256,
        )?;
        validate_sha256("global observation dataset", &self.dataset_sha256)?;
        nonempty("global observation source family", &self.source_family)?;
        self.genome.validate()?;
        if !self.within_dataset_percentile.is_finite()
            || !(0.0..=1.0).contains(&self.within_dataset_percentile)
        {
            return Err("within-dataset percentile must be in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalGeneRankV1 {
    pub genome: GenomeV1,
    pub support_shrunk_lower_bound: f64,
    pub worst_percentile: f64,
    pub dataset_support: u32,
    pub source_family_support: u32,
    pub strict_wins: u32,
}

impl GenerationZeroV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpGenerationZero.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("generation-zero dataset", &self.dataset_sha256)?;
        validate_sha256("proposal freeze", &self.proposal_freeze_sha256)?;
        if self.population.len() != 80 {
            return Err("generation zero must contain exactly 80 members".into());
        }
        let mut semantics = BTreeSet::new();
        let mut origins = BTreeMap::<GenomeOriginV1, usize>::new();
        for (slot, member) in self.population.iter().enumerate() {
            if member.slot != slot as u32 {
                return Err("generation-zero slots must be contiguous".into());
            }
            member.validate()?;
            if !semantics.insert(member.genome.semantic_sha256.as_str()) {
                return Err("generation zero contains a semantic duplicate".into());
            }
            *origins.entry(member.ancestry.origin).or_default() += 1;
        }
        if origins.get(&GenomeOriginV1::ModelProposal) != Some(&64)
            || origins.get(&GenomeOriginV1::GlobalGene) != Some(&10)
            || origins.get(&GenomeOriginV1::DiversitySeed) != Some(&6)
        {
            return Err("generation zero must preserve the exact 64/10/6 composition".into());
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
pub struct SplitRowV1 {
    pub stable_row_id: String,
    pub atomic_group: String,
    pub regression_target: Option<f64>,
    pub binary_target: Option<u8>,
}

impl SplitRowV1 {
    pub fn validate(&self, task: GpTaskV1) -> Result<(), String> {
        nonempty("stable row ID", &self.stable_row_id)?;
        nonempty("atomic group", &self.atomic_group)?;
        match task {
            GpTaskV1::Regression
                if self.regression_target.is_some_and(f64::is_finite)
                    && self.binary_target.is_none() =>
            {
                Ok(())
            }
            GpTaskV1::Binary
                if matches!(self.binary_target, Some(0 | 1))
                    && self.regression_target.is_none() =>
            {
                Ok(())
            }
            _ => Err("split row target does not match its task".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSplitV1 {
    pub split_index: u32,
    pub seed: u64,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_groups: Vec<String>,
    pub balance_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotatingSplitPlanV1 {
    pub schema: String,
    pub task: GpTaskV1,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub stable_row_ids: Vec<String>,
    pub row_groups: BTreeMap<String, String>,
    pub splits: Vec<RowSplitV1>,
}

impl RotatingSplitPlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpRotatingSplitPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.validate_k(Self::SCHEMA, 5, true)
    }

    pub(crate) fn validate_k(
        &self,
        expected_schema: &str,
        expected: usize,
        distinct_seeds: bool,
    ) -> Result<(), String> {
        schema(&self.schema, expected_schema)?;
        validate_sha256("split dataset", &self.dataset_sha256)?;
        validate_sha256("split training", &self.training_sha256)?;
        let all: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all.len() != self.stable_row_ids.len() || all.len() < expected {
            return Err("split plan stable rows are duplicate or insufficient".into());
        }
        if self.row_groups.len() != all.len()
            || self
                .row_groups
                .iter()
                .any(|(row, group)| !all.contains(row) || group.is_empty())
        {
            return Err("split plan must bind every stable row to an atomic group".into());
        }
        if self.splits.len() != expected {
            return Err(format!("split plan requires exactly {expected} partitions"));
        }
        let mut seeds = BTreeSet::new();
        let mut validations = BTreeSet::new();
        for (index, split) in self.splits.iter().enumerate() {
            if split.split_index != index as u32 {
                return Err("split indices must be contiguous".into());
            }
            seeds.insert(split.seed);
            let train: BTreeSet<_> = split.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = split.validation_row_ids.iter().collect();
            if train.len() != split.train_row_ids.len()
                || validation.len() != split.validation_row_ids.len()
                || validation.is_empty()
                || !train.is_disjoint(&validation)
                || train.union(&validation).copied().collect::<BTreeSet<_>>() != all
            {
                return Err(format!(
                    "split {index} is not a complete disjoint partition"
                ));
            }
            let validation_groups: BTreeSet<_> = split.validation_groups.iter().collect();
            for row in &split.validation_row_ids {
                let group = self.row_groups.get(row).unwrap();
                if !validation_groups.contains(group) {
                    return Err(format!("split {index} omits validation atomic group"));
                }
            }
            for row in &split.train_row_ids {
                if validation_groups.contains(self.row_groups.get(row).unwrap()) {
                    return Err(format!("split {index} crosses an atomic group"));
                }
            }
            if self.task == GpTaskV1::Binary
                && (split.balance_counts.get("class_0").copied().unwrap_or(0) == 0
                    || split.balance_counts.get("class_1").copied().unwrap_or(0) == 0)
            {
                return Err(format!(
                    "binary split {index} lacks both validation classes"
                ));
            }
            validations.insert(split.validation_row_ids.clone());
        }
        if distinct_seeds && seeds.len() != expected {
            return Err("rotating split seeds must be distinct".into());
        }
        if validations.len() != expected {
            return Err("rotating validation partitions must be distinct".into());
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
pub struct Cv7PlanV1(pub RotatingSplitPlanV1);

impl Cv7PlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCv7Plan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate_k(Self::SCHEMA, 7, false)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionCacheKeyV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub split_sha256: String,
    pub candidate_semantic_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl PredictionCacheKeyV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPredictionKey.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("prediction dataset", self.dataset_sha256.as_str()),
            ("prediction training", self.training_sha256.as_str()),
            ("prediction split", self.split_sha256.as_str()),
            (
                "prediction candidate",
                self.candidate_semantic_sha256.as_str(),
            ),
            ("prediction evaluator", self.evaluator_sha256.as_str()),
            ("prediction math policy", self.math_policy_sha256.as_str()),
            (
                "prediction backend policy",
                self.backend_policy_sha256.as_str(),
            ),
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeObservationV1 {
    pub cpu_seconds: f64,
    pub winner_latency_seconds: f64,
    pub timed_out: bool,
}

impl RuntimeObservationV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.cpu_seconds.is_finite()
            && self.cpu_seconds >= 0.0
            && self.winner_latency_seconds.is_finite()
            && self.winner_latency_seconds >= 0.0
        {
            Ok(())
        } else {
            Err("runtime values must be finite and nonnegative".into())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSplitEvaluationV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub generation: u32,
    pub split_index: u32,
    pub split_sha256: String,
    pub candidate: GenomeV1,
    pub prediction_object_sha256: String,
    pub candidate_skill: f64,
    pub anchor_median_skill: f64,
    pub paired_uplift: f64,
    /// RMSE for regression and log-loss for binary; lower is better.
    pub secondary_metric: f64,
    pub valid: bool,
    pub invalid_reason: Option<String>,
    pub runtime: RuntimeObservationV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorSkillV1 {
    pub semantic_sha256: String,
    pub skill: f64,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorBaselineV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub anchors: Vec<AnchorSkillV1>,
    pub median_skill: f64,
}

impl AnchorBaselineV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpAnchorBaseline.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("anchor dataset", &self.dataset_sha256)?;
        validate_sha256("anchor split", &self.split_sha256)?;
        if self.anchors.len() != 10 || self.anchors.iter().any(|anchor| !anchor.valid) {
            return Err("paired normalization requires ten valid frozen global anchors".into());
        }
        let mut semantics = BTreeSet::new();
        let mut skills = Vec::with_capacity(10);
        for anchor in &self.anchors {
            validate_sha256("anchor semantic", &anchor.semantic_sha256)?;
            finite("anchor skill", anchor.skill)?;
            if !semantics.insert(anchor.semantic_sha256.as_str()) {
                return Err("anchor baseline contains a duplicate".into());
            }
            skills.push(anchor.skill);
        }
        skills.sort_by(f64::total_cmp);
        let expected = 0.5 * (skills[4] + skills[5]);
        if !self.median_skill.is_finite() || (expected - self.median_skill).abs() > 1e-12 {
            return Err("anchor median does not match the ten frozen anchor skills".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl CandidateSplitEvaluationV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCandidateSplitEvaluation.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("evaluation dataset", &self.dataset_sha256)?;
        validate_sha256("evaluation split", &self.split_sha256)?;
        validate_sha256("prediction object", &self.prediction_object_sha256)?;
        self.candidate.validate()?;
        self.runtime.validate()?;
        for (name, value) in [
            ("candidate skill", self.candidate_skill),
            ("anchor median skill", self.anchor_median_skill),
            ("paired uplift", self.paired_uplift),
            ("secondary metric", self.secondary_metric),
        ] {
            finite(name, value)?;
        }
        if (self.candidate_skill - self.anchor_median_skill - self.paired_uplift).abs() > 1e-12 {
            return Err("paired uplift is inconsistent with candidate and anchor skill".into());
        }
        if self.candidate.task == GpTaskV1::Regression
            && !(-1.0..=1.0).contains(&self.candidate_skill)
        {
            return Err("regression search skill must be clipped to [-1,1]".into());
        }
        match (self.valid, self.invalid_reason.as_deref()) {
            (true, None) => Ok(()),
            (false, Some(reason)) if !reason.trim().is_empty() => Ok(()),
            _ => Err("candidate validity and invalid reason disagree".into()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankedCandidateV1 {
    pub candidate: GenomeV1,
    pub primary_score: f64,
    pub mean_uplift: f64,
    pub worst_uplift: f64,
    pub support: u32,
    pub secondary_metric: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationCheckpointV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub generation: u32,
    pub split_sha256: String,
    pub population: Vec<PopulationMemberV1>,
    pub ranked: Vec<RankedCandidateV1>,
    pub anchor_baseline_sha256: String,
    pub evaluations: Vec<CandidateSplitEvaluationV1>,
    pub previous_checkpoint_sha256: Option<String>,
}

impl GenerationCheckpointV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpGenerationCheckpoint.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("generation dataset", &self.dataset_sha256)?;
        validate_sha256("generation split", &self.split_sha256)?;
        validate_sha256("generation anchor baseline", &self.anchor_baseline_sha256)?;
        if self.generation == 0 && self.previous_checkpoint_sha256.is_some()
            || self.generation > 0 && self.previous_checkpoint_sha256.is_none()
        {
            return Err("generation checkpoint chain is invalid".into());
        }
        if let Some(previous) = &self.previous_checkpoint_sha256 {
            validate_sha256("previous generation", previous)?;
        }
        if self.population.len() != 80 || self.ranked.len() != 80 {
            return Err(
                "generation checkpoint requires an 80-member population and ranking".into(),
            );
        }
        if self.evaluations.len() != 80 {
            return Err("generation checkpoint requires one fresh evaluation per member".into());
        }
        let population: BTreeSet<_> = self
            .population
            .iter()
            .map(|member| member.genome.semantic_sha256.as_str())
            .collect();
        let ranked: BTreeSet<_> = self
            .ranked
            .iter()
            .map(|member| member.candidate.semantic_sha256.as_str())
            .collect();
        if population.len() != 80 || population != ranked {
            return Err("generation ranking must cover exactly its population".into());
        }
        for (slot, member) in self.population.iter().enumerate() {
            if member.slot != slot as u32 {
                return Err("generation slots must be contiguous".into());
            }
            member.validate()?;
        }
        let mut evaluated = BTreeSet::new();
        for evaluation in &self.evaluations {
            evaluation.validate()?;
            if evaluation.arm != self.arm
                || evaluation.dataset_sha256 != self.dataset_sha256
                || evaluation.generation != self.generation
                || evaluation.split_sha256 != self.split_sha256
                || !evaluated.insert(evaluation.candidate.semantic_sha256.as_str())
            {
                return Err(
                    "generation contains stale, duplicate, or cross-arm evaluation evidence".into(),
                );
            }
        }
        if evaluated != population {
            return Err("every member, including all elites, must be freshly evaluated".into());
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
pub struct ArchiveMemberV1 {
    pub candidate: GenomeV1,
    pub split_evaluations: Vec<CandidateSplitEvaluationV1>,
    pub confirmed_rank: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveCheckpointV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub split_plan_sha256: String,
    pub members: Vec<ArchiveMemberV1>,
}

impl ArchiveCheckpointV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpArchiveCheckpoint.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("archive dataset", &self.dataset_sha256)?;
        validate_sha256("archive split plan", &self.split_plan_sha256)?;
        if self.members.len() != 20 {
            return Err("confirmed archive must contain exactly 20 genomes".into());
        }
        let mut semantics = BTreeSet::new();
        for (rank, member) in self.members.iter().enumerate() {
            member.candidate.validate()?;
            if member.confirmed_rank != rank as u32
                || !semantics.insert(member.candidate.semantic_sha256.as_str())
                || member.split_evaluations.len() != 5
            {
                return Err("archive ranks, identities, or split coverage are invalid".into());
            }
            let splits: BTreeSet<_> = member
                .split_evaluations
                .iter()
                .map(|evaluation| evaluation.split_index)
                .collect();
            if splits != BTreeSet::from([0, 1, 2, 3, 4]) {
                return Err("archive member lacks complete five-split confirmation".into());
            }
            for evaluation in &member.split_evaluations {
                evaluation.validate()?;
                if evaluation.candidate.semantic_sha256 != member.candidate.semantic_sha256 {
                    return Err("archive evidence names another candidate".into());
                }
            }
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
pub struct Cv7CandidateEvidenceV1 {
    pub candidate: GenomeV1,
    pub fold_primary_scores: Vec<f64>,
    pub pooled_oof_primary: f64,
    pub secondary_metric: f64,
    pub runtime_seconds: f64,
    pub valid: bool,
}

impl Cv7CandidateEvidenceV1 {
    pub fn robust_objective(&self) -> f64 {
        let fold_metric: Vec<_> = self
            .fold_primary_scores
            .iter()
            .map(|value| match self.candidate.task {
                GpTaskV1::Regression => *value,
                GpTaskV1::Binary => 2.0 * value - 1.0,
            })
            .collect();
        let pooled = match self.candidate.task {
            GpTaskV1::Regression => self.pooled_oof_primary,
            GpTaskV1::Binary => 2.0 * self.pooled_oof_primary - 1.0,
        };
        pooled - 0.25 * sample_sd(&fold_metric)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.candidate.validate()?;
        if self.fold_primary_scores.len() != 7
            || self
                .fold_primary_scores
                .iter()
                .any(|value| !value.is_finite())
            || !self.pooled_oof_primary.is_finite()
            || !self.secondary_metric.is_finite()
            || !self.runtime_seconds.is_finite()
            || self.runtime_seconds < 0.0
        {
            return Err("CV7 evidence is incomplete or nonfinite".into());
        }
        if self.candidate.task == GpTaskV1::Binary
            && (self
                .fold_primary_scores
                .iter()
                .any(|value| !(0.0..=1.0).contains(value))
                || !(0.0..=1.0).contains(&self.pooled_oof_primary))
        {
            return Err("binary CV7 AUC values must be in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cv7SelectionFreezeV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub archive_sha256: String,
    pub cv7_plan_sha256: String,
    pub finalists: Vec<Cv7CandidateEvidenceV1>,
    pub winner_semantic_sha256: String,
}

impl Cv7SelectionFreezeV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCv7SelectionFreeze.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("CV7 dataset", self.dataset_sha256.as_str()),
            ("CV7 archive", self.archive_sha256.as_str()),
            ("CV7 plan", self.cv7_plan_sha256.as_str()),
            ("CV7 winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.finalists.len() != 10 {
            return Err("CV7 selection must freeze exactly ten finalists".into());
        }
        let mut seen = BTreeSet::new();
        for finalist in &self.finalists {
            finalist.validate()?;
            if !seen.insert(finalist.candidate.semantic_sha256.as_str()) {
                return Err("CV7 finalists contain a duplicate".into());
            }
        }
        if !seen.contains(self.winner_semantic_sha256.as_str()) {
            return Err("CV7 winner is not a finalist".into());
        }
        Ok(())
    }

    pub fn winner(&self) -> Result<&Cv7CandidateEvidenceV1, String> {
        self.validate()?;
        self.finalists
            .iter()
            .find(|candidate| candidate.candidate.semantic_sha256 == self.winner_semantic_sha256)
            .ok_or_else(|| "CV7 winner is absent".into())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedTestResultV1 {
    pub schema: String,
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub selection_freeze_sha256: String,
    pub winner_semantic_sha256: String,
    pub task: GpTaskV1,
    pub primary_metric: f64,
    pub secondary_metric: f64,
    pub runtime: RuntimeObservationV1,
    pub valid: bool,
}

impl SealedTestResultV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpSealedTestResult.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("sealed dataset", self.dataset_sha256.as_str()),
            ("selection freeze", self.selection_freeze_sha256.as_str()),
            ("sealed winner", self.winner_semantic_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        finite("sealed primary", self.primary_metric)?;
        finite("sealed secondary", self.secondary_metric)?;
        self.runtime.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilotDatasetResultV1 {
    pub arm: CampaignArmV1,
    pub dataset_sha256: String,
    pub task: GpTaskV1,
    pub sealed_skill: f64,
    pub cpu_seconds: f64,
    pub winner_latency_seconds: f64,
    pub generation_zero_validity: f64,
    pub finalist_validity: f64,
    pub overall_validity: f64,
    pub sealed_winner: bool,
}

impl PilotDatasetResultV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("pilot result dataset", &self.dataset_sha256)?;
        for (name, value) in [
            ("sealed skill", self.sealed_skill),
            ("CPU seconds", self.cpu_seconds),
            ("winner latency", self.winner_latency_seconds),
            ("generation-zero validity", self.generation_zero_validity),
            ("finalist validity", self.finalist_validity),
            ("overall validity", self.overall_validity),
        ] {
            finite(name, value)?;
        }
        if self.cpu_seconds < 0.0
            || self.winner_latency_seconds < 0.0
            || [
                self.generation_zero_validity,
                self.finalist_validity,
                self.overall_validity,
            ]
            .iter()
            .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err("pilot runtime or validity is outside its domain".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilotArmSummaryV1 {
    pub arm: CampaignArmV1,
    pub mean_regression_delta: f64,
    pub mean_binary_delta: f64,
    pub equal_task_mean_delta: f64,
    pub one_sided_95_lower_bound: f64,
    pub predicted_cpu_seconds: f64,
    pub validity_gate: bool,
    pub sealed_winner_gate: bool,
    pub runtime_gate: bool,
    pub advances: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PilotDecisionV1 {
    Advance { arm: CampaignArmV1 },
    NoImprovement { receipt_sha256: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoImprovementV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub reason_codes: Vec<String>,
}

impl NoImprovementV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpNoImprovement.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("no-improvement campaign", &self.campaign_freeze_sha256)?;
        if self.reason_codes.is_empty()
            || self
                .reason_codes
                .iter()
                .any(|reason| reason.trim().is_empty())
        {
            return Err("no_improvement requires typed reason codes".into());
        }
        if !self.reason_codes.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err("no_improvement reason codes must be sorted and unique".into());
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
pub struct PilotComparisonV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub bootstrap_repetitions: u32,
    pub arms: Vec<PilotArmSummaryV1>,
    pub decision: PilotDecisionV1,
}

impl PilotComparisonV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPilotComparison.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("pilot comparison campaign", &self.campaign_freeze_sha256)?;
        if self.bootstrap_repetitions != 10_000 || self.arms.len() != 3 {
            return Err("pilot comparison requires 10,000 bootstraps over three arms".into());
        }
        let arms: BTreeSet<_> = self.arms.iter().map(|arm| arm.arm).collect();
        if arms != CampaignArmV1::EXPERIMENTS.into_iter().collect() {
            return Err("pilot comparison contains the wrong experiment arms".into());
        }
        if let PilotDecisionV1::Advance { arm } = self.decision {
            if !arm.eligible()
                || !self
                    .arms
                    .iter()
                    .any(|summary| summary.arm == arm && summary.advances)
            {
                return Err("pilot decision advances an ineligible or gated arm".into());
            }
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
pub struct LaunchGateV1 {
    pub schema: String,
    pub real_u1_evaluator_flow: bool,
    pub fresh_supply_evidence: bool,
    pub protected_wave_count: u32,
    pub reviewed_scq_acknowledged: bool,
    pub free_cores: u32,
    pub liveness_age_seconds: u64,
    pub max_liveness_age_seconds: u64,
    pub storage_free_fraction: f64,
    pub min_storage_free_fraction: f64,
}

impl LaunchGateV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpLaunchGate.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        if !self.storage_free_fraction.is_finite()
            || !self.min_storage_free_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.storage_free_fraction)
            || !(0.0..=1.0).contains(&self.min_storage_free_fraction)
            || self.max_liveness_age_seconds == 0
        {
            return Err("launch gate thresholds are invalid".into());
        }
        Ok(())
    }

    pub fn launch_allowed(&self) -> Result<bool, String> {
        self.validate()?;
        Ok(self.real_u1_evaluator_flow
            && self.fresh_supply_evidence
            && self.protected_wave_count >= 2
            && self.reviewed_scq_acknowledged
            && self.free_cores > 0
            && self.liveness_age_seconds <= self.max_liveness_age_seconds
            && self.storage_free_fraction >= self.min_storage_free_fraction)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrancheV1 {
    pub index: u32,
    pub partition: CatalogPartitionV1,
    pub dataset_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranchePlanV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub selected_arm: CampaignArmV1,
    pub partition: CatalogPartitionV1,
    pub pilot_comparison_sha256: String,
    pub development_completion_sha256: Option<String>,
    pub reused_pilot_dataset_sha256s: Vec<String>,
    pub tranches: Vec<TrancheV1>,
}

impl TranchePlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpTranchePlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("tranche campaign", &self.campaign_freeze_sha256)?;
        validate_sha256("pilot comparison", &self.pilot_comparison_sha256)?;
        if !self.selected_arm.eligible() || self.tranches.is_empty() {
            return Err("tranche plan requires an eligible experiment and nonempty work".into());
        }
        match (
            self.partition,
            self.development_completion_sha256.as_deref(),
        ) {
            (CatalogPartitionV1::Development, None) => {}
            (CatalogPartitionV1::Validation, Some(digest)) => {
                validate_sha256("development completion", digest)?
            }
            (CatalogPartitionV1::Development, Some(_)) => {
                return Err("development tranches cannot claim prior development completion".into())
            }
            (CatalogPartitionV1::Validation, None) => {
                return Err("validation audit cannot start before development completion".into())
            }
        }
        let mut datasets = BTreeSet::new();
        for (index, tranche) in self.tranches.iter().enumerate() {
            tranche.validate()?;
            if tranche.index != index as u32 || tranche.partition != self.partition {
                return Err("tranche ordering or partition changed".into());
            }
            for dataset in &tranche.dataset_sha256s {
                if !datasets.insert(dataset.as_str()) {
                    return Err("dataset occurs in more than one tranche".into());
                }
            }
        }
        let mut reused = BTreeSet::new();
        for dataset in &self.reused_pilot_dataset_sha256s {
            validate_sha256("reused pilot dataset", dataset)?;
            if !reused.insert(dataset.as_str()) || !datasets.contains(dataset.as_str()) {
                return Err("reused pilot evidence must name a unique planned dataset".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

impl TrancheV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.dataset_sha256s.is_empty() || self.dataset_sha256s.len() > 128 {
            return Err("tranches must contain 1 through 128 datasets".into());
        }
        let mut seen = BTreeSet::new();
        for digest in &self.dataset_sha256s {
            validate_sha256("tranche dataset", digest)?;
            if !seen.insert(digest) {
                return Err("tranche contains a duplicate dataset".into());
            }
        }
        Ok(())
    }
}

pub fn ensure_selection_before_test(
    selection: &Cv7SelectionFreezeV1,
    expected_dataset: &str,
    expected_candidate: &str,
) -> Result<String, String> {
    selection.validate()?;
    if selection.dataset_sha256 != expected_dataset
        || selection.winner_semantic_sha256 != expected_candidate
    {
        return Err("sealed-test request does not match its selection freeze".into());
    }
    selection.identity()
}

pub fn validate_prediction_binding(
    key: &PredictionCacheKeyV1,
    prediction: &PredictionObjectV1,
) -> Result<(), String> {
    key.validate()?;
    prediction.validate()?;
    if prediction.training_view_sha256 != key.training_sha256
        || prediction.fold_plan_sha256 != key.split_sha256
        || prediction.candidate_semantic_sha256 != key.candidate_semantic_sha256
        || prediction.evaluator_sha256 != key.evaluator_sha256
        || prediction.math_policy_sha256 != key.math_policy_sha256
        || prediction.backend_policy_sha256 != key.backend_policy_sha256
    {
        return Err("prediction object does not match its reusable cache key".into());
    }
    Ok(())
}

pub fn artifact_exists(path: &str) -> bool {
    Path::new(path).is_file()
}
