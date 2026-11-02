use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceRole {
    ResearchDevelopment,
    Judge,
    ProtectedFamily,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContrastObservation {
    pub pair_id: String,
    pub dataset_sha256: String,
    pub cut_sha256: String,
    pub context_sha256: String,
    pub family_id: String,
    pub seed: u64,
    pub role: EvidenceRole,
    pub baseline_score: f64,
    pub macro_score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreregisteredContrast {
    pub schema: String,
    pub preregistration_sha256: String,
    pub motif_id: String,
    pub old_profile_id: String,
    pub old_profile_revision: u32,
    pub new_profile_id: String,
    pub new_profile_revision: u32,
    pub direction: MetricDirection,
    /// Minimum useful mean gain in the direction named above, fixed before outcomes.
    pub minimum_mean_gain: f64,
    pub observations: Vec<ContrastObservation>,
}

#[derive(Serialize)]
struct ContrastDesign<'a> {
    pub(super) schema: &'a str,
    pub(super) motif_id: &'a str,
    pub(super) old_profile_id: &'a str,
    pub(super) old_profile_revision: u32,
    pub(super) new_profile_id: &'a str,
    pub(super) new_profile_revision: u32,
    pub(super) direction: MetricDirection,
    pub(super) minimum_mean_gain: f64,
    pub(super) pairs: Vec<PlannedPair<'a>>,
}

#[derive(Serialize)]
struct PlannedPair<'a> {
    pub(super) pair_id: &'a str,
    pub(super) dataset_sha256: &'a str,
    pub(super) cut_sha256: &'a str,
    pub(super) context_sha256: &'a str,
    pub(super) family_id: &'a str,
    pub(super) seed: u64,
    pub(super) role: EvidenceRole,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EvidenceBoundary {
    pub(super) archive_manifest_sha256: String,
    pub(super) judge_datasets: BTreeSet<String>,
    pub(super) protected_families: BTreeSet<String>,
    pub(super) boundary_sha256: String,
}

impl EvidenceBoundary {
    /// Bind the held-out boundary to an immutable archive manifest. Empty boundary inventories
    /// are refused; callers cannot silently disable the leakage check with a default value.
    pub fn new(
        archive_manifest_sha256: String,
        judge_datasets: BTreeSet<String>,
        protected_families: BTreeSet<String>,
    ) -> Result<Self> {
        if !is_hex64(&archive_manifest_sha256)
            || judge_datasets.is_empty()
            || protected_families.is_empty()
            || judge_datasets.iter().any(|value| !is_hex64(value))
            || protected_families.iter().any(String::is_empty)
        {
            return Err(AdmissionError("evidence_boundary_invalid"));
        }
        #[derive(Serialize)]
        struct Binding<'a> {
            archive_manifest_sha256: &'a str,
            judge_datasets: &'a BTreeSet<String>,
            protected_families: &'a BTreeSet<String>,
        }
        let binding = Binding {
            archive_manifest_sha256: &archive_manifest_sha256,
            judge_datasets: &judge_datasets,
            protected_families: &protected_families,
        };
        let bytes = serde_json::to_vec(&binding)
            .map_err(|_| AdmissionError("evidence_boundary_invalid"))?;
        let boundary_sha256 = sha256(&bytes);
        Ok(Self {
            archive_manifest_sha256,
            judge_datasets,
            protected_families,
            boundary_sha256,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ValidatedContrast {
    pub(super) contrast_sha256: String,
    pub(super) evidence_boundary_sha256: String,
    pub(super) motif_id: String,
    pub(super) new_profile_id: String,
    pub(super) new_profile_revision: u32,
    pub(super) mean_gain: f64,
    pub(super) minimum_mean_gain: f64,
    pub(super) datasets: Vec<String>,
    pub(super) families: Vec<String>,
    pub(super) observation_count: u32,
}

impl ValidatedContrast {
    pub fn mean_gain(&self) -> f64 {
        self.mean_gain
    }

    pub fn minimum_mean_gain(&self) -> f64 {
        self.minimum_mean_gain
    }
}

impl PreregisteredContrast {
    /// Hash the design fields and planned pair identities, excluding observed scores. A
    /// preregistration receipt must retain this digest before outcomes are collected.
    pub fn computed_preregistration_sha256(&self) -> Result<String> {
        let design = ContrastDesign {
            schema: &self.schema,
            motif_id: &self.motif_id,
            old_profile_id: &self.old_profile_id,
            old_profile_revision: self.old_profile_revision,
            new_profile_id: &self.new_profile_id,
            new_profile_revision: self.new_profile_revision,
            direction: self.direction,
            minimum_mean_gain: self.minimum_mean_gain,
            pairs: self
                .observations
                .iter()
                .map(|row| PlannedPair {
                    pair_id: &row.pair_id,
                    dataset_sha256: &row.dataset_sha256,
                    cut_sha256: &row.cut_sha256,
                    context_sha256: &row.context_sha256,
                    family_id: &row.family_id,
                    seed: row.seed,
                    role: row.role,
                })
                .collect(),
        };
        let bytes =
            serde_json::to_vec(&design).map_err(|_| AdmissionError("contrast_identity_invalid"))?;
        Ok(sha256(&bytes))
    }

    pub fn validate(
        &self,
        candidate: &MotifCandidate,
        migration: &VocabularyMigration,
        boundary: &EvidenceBoundary,
    ) -> Result<ValidatedContrast> {
        if self.schema != CONTRAST_SCHEMA
            || !is_hex64(&self.preregistration_sha256)
            || self.motif_id != candidate.motif_id
            || self.old_profile_id != migration.old_profile_id
            || self.old_profile_revision != migration.old_profile_revision
            || self.new_profile_id != migration.new_profile_id
            || self.new_profile_revision != migration.new_profile_revision
            || self.preregistration_sha256 != self.computed_preregistration_sha256()?
            || !self.minimum_mean_gain.is_finite()
            || self.minimum_mean_gain <= 0.0
            || self.observations.len() < 2
            || !is_hex64(&boundary.archive_manifest_sha256)
            || !is_hex64(&boundary.boundary_sha256)
            || boundary.judge_datasets.is_empty()
            || boundary.protected_families.is_empty()
        {
            return Err(AdmissionError("contrast_identity_invalid"));
        }
        let mut pair_ids = BTreeSet::new();
        let mut datasets = BTreeSet::new();
        let mut families = BTreeSet::new();
        let mut family_gains: BTreeMap<&str, (f64, u32)> = BTreeMap::new();
        let mut total_gain = 0.0;
        for observation in &self.observations {
            if observation.pair_id.is_empty()
                || !pair_ids.insert(observation.pair_id.as_str())
                || !is_hex64(&observation.dataset_sha256)
                || !is_hex64(&observation.cut_sha256)
                || !is_hex64(&observation.context_sha256)
                || observation.family_id.is_empty()
                || observation.role != EvidenceRole::ResearchDevelopment
                || boundary
                    .judge_datasets
                    .contains(&observation.dataset_sha256)
                || boundary.protected_families.contains(&observation.family_id)
                || !observation.baseline_score.is_finite()
                || !observation.macro_score.is_finite()
            {
                return Err(AdmissionError("contrast_input_refused"));
            }
            let gain = match self.direction {
                MetricDirection::LowerIsBetter => {
                    observation.baseline_score - observation.macro_score
                }
                MetricDirection::HigherIsBetter => {
                    observation.macro_score - observation.baseline_score
                }
            };
            total_gain += gain;
            let entry = family_gains
                .entry(observation.family_id.as_str())
                .or_insert((0.0, 0));
            entry.0 += gain;
            entry.1 += 1;
            datasets.insert(observation.dataset_sha256.clone());
            families.insert(observation.family_id.clone());
        }
        let mean_gain = total_gain / self.observations.len() as f64;
        if datasets.len() < 2
            || families.len() < 2
            || !mean_gain.is_finite()
            || mean_gain <= self.minimum_mean_gain
            || family_gains
                .values()
                .any(|(sum, count)| *count == 0 || *sum / f64::from(*count) <= 0.0)
        {
            return Err(AdmissionError("contrast_not_useful"));
        }
        let bytes =
            serde_json::to_vec(self).map_err(|_| AdmissionError("contrast_identity_invalid"))?;
        Ok(ValidatedContrast {
            contrast_sha256: sha256(&bytes),
            evidence_boundary_sha256: boundary.boundary_sha256.clone(),
            motif_id: self.motif_id.clone(),
            new_profile_id: self.new_profile_id.clone(),
            new_profile_revision: self.new_profile_revision,
            mean_gain,
            minimum_mean_gain: self.minimum_mean_gain,
            datasets: datasets.into_iter().collect(),
            families: families.into_iter().collect(),
            observation_count: u32::try_from(self.observations.len())
                .map_err(|_| AdmissionError("contrast_input_refused"))?,
        })
    }
}
