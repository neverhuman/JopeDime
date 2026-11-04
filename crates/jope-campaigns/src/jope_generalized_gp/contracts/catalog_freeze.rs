use super::*;

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
