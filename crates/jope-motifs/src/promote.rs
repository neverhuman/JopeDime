//! Additive qualification for one motif in a new research vocabulary.
//!
//! This module does not mutate `MotifTable`: its V0 verifier intentionally admits only
//! quarantined entries. A caller must bind the candidate to a verified table and pass a
//! source-verified validation report before an admission artifact can be produced.

use crate::fragment::{self, stage_texts};
use crate::source::QualifiedSet;
use crate::table::{Lifecycle, MotifTable, Promotion};
use crate::validate::{self, ValidateOptions, Validation, VALIDATION_SCHEMA};
use crate::NAME_PREFIX;
use feat_math::dsl::{parse_dsl, DslExpr, Stage};
use feat_math::GenomeIr;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const ADMISSION_SCHEMA: &str = "jope.motif-admission/v1";
pub const CONTRAST_SCHEMA: &str = "jope.motif-contrast/v1";
pub const VOCABULARY_SCHEMA: &str = "jope.motif-vocabulary-migration/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionError(pub &'static str);

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for AdmissionError {}

type Result<T> = std::result::Result<T, AdmissionError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MotifCandidate {
    table_sha256: String,
    motif_id: String,
    motif_name: String,
    expansion: String,
    index: u32,
    support_families: u32,
}

impl MotifCandidate {
    /// Get a candidate only from an internally consistent, still-quarantined table.
    pub fn from_table(table: &MotifTable, id: &str) -> Result<Self> {
        table
            .verify()
            .map_err(|_| AdmissionError("candidate_table_invalid"))?;
        let (index, motif) = table
            .body
            .motifs
            .iter()
            .enumerate()
            .find(|(_, motif)| motif.id == id)
            .ok_or(AdmissionError("candidate_not_in_table"))?;
        if motif.lifecycle != Lifecycle::Quarantined
            || motif.promotion != Promotion::required_not_run()
        {
            return Err(AdmissionError("candidate_not_quarantined"));
        }
        Ok(Self {
            table_sha256: table.table_sha256.clone(),
            motif_id: motif.id.clone(),
            motif_name: motif.name.clone(),
            expansion: motif.expansion.clone(),
            index: u32::try_from(index).map_err(|_| AdmissionError("candidate_index_invalid"))?,
            support_families: motif.support.families.unwrap_or(0),
        })
    }

    // Used by this module's focused integration target to exercise the pure gate without
    // manufacturing an authenticated source table. Production callers use `from_table`.
    #[cfg(test)]
    pub(crate) fn draft_for_gate(
        table_sha256: String,
        motif_id: String,
        motif_name: String,
        expansion: String,
        index: u32,
    ) -> Result<Self> {
        if !is_hex64(&table_sha256)
            || motif_id != motif_id_fn(&expansion)
            || motif_name != motif_name_fn(&motif_id)
            || expansion.contains(NAME_PREFIX)
        {
            return Err(AdmissionError("candidate_identity_invalid"));
        }
        Ok(Self {
            table_sha256,
            motif_id,
            motif_name,
            expansion,
            index,
            support_families: 2,
        })
    }

    pub fn table_sha256(&self) -> &str {
        &self.table_sha256
    }

    pub fn motif_id(&self) -> &str {
        &self.motif_id
    }

    pub fn motif_name(&self) -> &str {
        &self.motif_name
    }

    pub fn expansion(&self) -> &str {
        &self.expansion
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpansionLimits {
    /// Maximum expanded transform stages; the table format itself caps this at six.
    pub max_expansion_stages: usize,
    /// Maximum nested motif references inside an expansion. V1 requires zero.
    pub max_nested_macros: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpansionWitness {
    table_sha256: String,
    motif_id: String,
    recipe_sha256: String,
    stage_offset: u32,
    rewritten_sha256: String,
    expanded_sha256: String,
    expanded_stages: u32,
    nested_macros: u32,
}

impl ExpansionWitness {
    /// Reconstruct the macro rewrite and expansion independently, then demand byte-exact host
    /// equality. Finite-probe similarity is deliberately not an input to this proof.
    pub fn verify_host(
        candidate: &MotifCandidate,
        host: &str,
        stage_offset: usize,
        limits: ExpansionLimits,
    ) -> Result<Self> {
        if limits.max_expansion_stages == 0
            || limits.max_expansion_stages > 6
            || limits.max_nested_macros != 0
            || !is_hex64(&candidate.table_sha256)
            || candidate.motif_id != motif_id_fn(&candidate.expansion)
            || candidate.motif_name != motif_name_fn(&candidate.motif_id)
        {
            return Err(AdmissionError("expansion_limits_invalid"));
        }
        let expansion = fragment::expansion_stages(&candidate.expansion)
            .map_err(|_| AdmissionError("expansion_not_canonical"))?;
        if expansion.is_empty() || expansion.len() > limits.max_expansion_stages {
            return Err(AdmissionError("expansion_stage_limit"));
        }
        if candidate.expansion.contains(NAME_PREFIX) {
            return Err(AdmissionError("nested_macro_refused"));
        }
        let host_ir = GenomeIr::parse(host).map_err(|_| AdmissionError("host_not_canonical"))?;
        if host_ir.canonical() != host || host.contains(NAME_PREFIX) {
            return Err(AdmissionError("host_not_canonical"));
        }
        let parsed = parse_dsl(host).map_err(|_| AdmissionError("host_not_canonical"))?;
        let host_texts = stage_texts(&parsed.stages);
        let end = stage_offset
            .checked_add(expansion.len())
            .ok_or(AdmissionError("expansion_offset_invalid"))?;
        if host_texts.get(stage_offset..end) != Some(stage_texts(&expansion).as_slice()) {
            return Err(AdmissionError("expansion_occurrence_missing"));
        }

        let mut rewritten_stages = parsed.stages.clone();
        rewritten_stages.splice(
            stage_offset..end,
            [Stage::new(candidate.motif_name.clone())],
        );
        let rewritten = DslExpr {
            stages: rewritten_stages,
        }
        .to_string();
        let rewritten_parsed =
            parse_dsl(&rewritten).map_err(|_| AdmissionError("rewrite_invalid"))?;
        if rewritten_parsed
            .stages
            .iter()
            .filter(|stage| stage.op == candidate.motif_name)
            .count()
            != 1
        {
            return Err(AdmissionError("rewrite_invalid"));
        }
        let mut expanded_stages = Vec::new();
        let mut nested_macros = 0_u32;
        for stage in rewritten_parsed.stages {
            if stage.op == candidate.motif_name {
                nested_macros = nested_macros.saturating_add(1);
                expanded_stages.extend(expansion.iter().cloned());
            } else if stage.op.starts_with(NAME_PREFIX) {
                return Err(AdmissionError("nested_macro_refused"));
            } else {
                expanded_stages.push(stage);
            }
        }
        if nested_macros != 1 {
            return Err(AdmissionError("rewrite_invalid"));
        }
        let expanded = DslExpr {
            stages: expanded_stages,
        }
        .to_string();
        let expanded_ir =
            GenomeIr::parse(&expanded).map_err(|_| AdmissionError("expansion_not_canonical"))?;
        if expanded_ir.canonical() != host || expanded.contains(NAME_PREFIX) {
            return Err(AdmissionError("expansion_not_exact"));
        }
        let recipe_sha256 = sha256(host.as_bytes());
        Ok(Self {
            table_sha256: candidate.table_sha256.clone(),
            motif_id: candidate.motif_id.clone(),
            recipe_sha256,
            stage_offset: u32::try_from(stage_offset)
                .map_err(|_| AdmissionError("expansion_offset_invalid"))?,
            rewritten_sha256: sha256(rewritten.as_bytes()),
            expanded_sha256: sha256(expanded.as_bytes()),
            expanded_stages: u32::try_from(expansion.len())
                .map_err(|_| AdmissionError("expansion_stage_limit"))?,
            nested_macros,
        })
    }

    pub fn expanded_stages(&self) -> u32 {
        self.expanded_stages
    }

    pub fn nested_macros(&self) -> u32 {
        self.nested_macros
    }
}

/// Similarity on a finite set of probes. This diagnostic cannot be converted to an admission
/// witness and never establishes semantic equivalence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProbeSimilarity {
    pub probe_count: u32,
    pub max_absolute_error: f64,
    pub within_tolerance: bool,
}

impl ProbeSimilarity {
    pub fn compare(left: &[f64], right: &[f64], tolerance: f64) -> Result<Self> {
        if left.is_empty()
            || left.len() != right.len()
            || !tolerance.is_finite()
            || tolerance < 0.0
            || left.iter().chain(right).any(|value| !value.is_finite())
        {
            return Err(AdmissionError("probe_input_invalid"));
        }
        let max_absolute_error = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if !max_absolute_error.is_finite() {
            return Err(AdmissionError("probe_input_invalid"));
        }
        Ok(Self {
            probe_count: u32::try_from(left.len())
                .map_err(|_| AdmissionError("probe_input_invalid"))?,
            max_absolute_error,
            within_tolerance: max_absolute_error <= tolerance,
        })
    }
}

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
    schema: &'a str,
    motif_id: &'a str,
    old_profile_id: &'a str,
    old_profile_revision: u32,
    new_profile_id: &'a str,
    new_profile_revision: u32,
    direction: MetricDirection,
    minimum_mean_gain: f64,
    pairs: Vec<PlannedPair<'a>>,
}

#[derive(Serialize)]
struct PlannedPair<'a> {
    pair_id: &'a str,
    dataset_sha256: &'a str,
    cut_sha256: &'a str,
    context_sha256: &'a str,
    family_id: &'a str,
    seed: u64,
    role: EvidenceRole,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EvidenceBoundary {
    archive_manifest_sha256: String,
    judge_datasets: BTreeSet<String>,
    protected_families: BTreeSet<String>,
    boundary_sha256: String,
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
    contrast_sha256: String,
    evidence_boundary_sha256: String,
    motif_id: String,
    new_profile_id: String,
    new_profile_revision: u32,
    mean_gain: f64,
    minimum_mean_gain: f64,
    datasets: Vec<String>,
    families: Vec<String>,
    observation_count: u32,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VocabularyMigration {
    schema: String,
    old_profile_id: String,
    old_profile_revision: u32,
    new_profile_id: String,
    new_profile_revision: u32,
    motif_id: String,
    macro_name: String,
    expansion_sha256: String,
    /// Old policies continue to use the expanded canonical text; only the new version admits
    /// the macro token.
    pub old_policy_expands_macro: bool,
}

impl VocabularyMigration {
    pub fn new(
        candidate: &MotifCandidate,
        old_profile_id: impl Into<String>,
        old_profile_revision: u32,
        new_profile_id: impl Into<String>,
        new_profile_revision: u32,
    ) -> Result<Self> {
        let old_profile_id = old_profile_id.into();
        let new_profile_id = new_profile_id.into();
        if old_profile_id.is_empty()
            || new_profile_id.is_empty()
            || old_profile_revision == 0
            || new_profile_revision == 0
            || (old_profile_id == new_profile_id && new_profile_revision <= old_profile_revision)
        {
            return Err(AdmissionError("vocabulary_version_invalid"));
        }
        Ok(Self {
            schema: VOCABULARY_SCHEMA.to_owned(),
            old_profile_id,
            old_profile_revision,
            new_profile_id,
            new_profile_revision,
            motif_id: candidate.motif_id.clone(),
            macro_name: candidate.motif_name.clone(),
            expansion_sha256: sha256(candidate.expansion.as_bytes()),
            old_policy_expands_macro: true,
        })
    }

    pub fn enables_macro(&self, profile_id: &str, revision: u32) -> bool {
        self.schema == VOCABULARY_SCHEMA
            && profile_id == self.new_profile_id
            && revision == self.new_profile_revision
    }

    pub fn old_profile(&self) -> (&str, u32) {
        (&self.old_profile_id, self.old_profile_revision)
    }

    pub fn new_profile(&self) -> (&str, u32) {
        (&self.new_profile_id, self.new_profile_revision)
    }

    pub fn old_policy_expands_macro(&self) -> bool {
        self.old_policy_expands_macro
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MotifAdmission {
    pub schema: String,
    pub table_sha256: String,
    pub motif_id: String,
    pub expansion_witnesses: Vec<ExpansionWitness>,
    pub contrast: ValidatedContrast,
    pub vocabulary: VocabularyMigration,
    pub validation_sha256: String,
}

impl MotifAdmission {
    /// Validate and admit a candidate using a fresh source read and the package's real motif
    /// validator. A report supplied by a caller is never sufficient by itself.
    pub fn qualify(
        table: &MotifTable,
        set: &QualifiedSet,
        id: &str,
        contrast: &PreregisteredContrast,
        boundary: &EvidenceBoundary,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        let candidate = MotifCandidate::from_table(table, id)?;
        if vocabulary.motif_id != candidate.motif_id
            || vocabulary.expansion_sha256 != sha256(candidate.expansion.as_bytes())
        {
            return Err(AdmissionError("vocabulary_version_invalid"));
        }
        let motif = table
            .body
            .motifs
            .get(candidate.index as usize)
            .ok_or(AdmissionError("candidate_not_in_table"))?;
        let limits = ExpansionLimits {
            max_expansion_stages: usize::try_from(table.body.parameters.max_stages)
                .map_err(|_| AdmissionError("expansion_limits_invalid"))?,
            max_nested_macros: 0,
        };
        let mut witnesses = Vec::with_capacity(motif.provenance.len());
        for occurrence in &motif.provenance {
            let host = table
                .body
                .hosts
                .get(&occurrence.recipe_sha256)
                .ok_or(AdmissionError("expansion_occurrence_missing"))?;
            witnesses.push(ExpansionWitness::verify_host(
                &candidate,
                host,
                usize::try_from(occurrence.stage_offset)
                    .map_err(|_| AdmissionError("expansion_offset_invalid"))?,
                limits,
            )?);
        }
        let mut validation = validate::validate(table, &ValidateOptions::default())
            .map_err(|_| AdmissionError("validation_failed"))?;
        if validate::verify_source(&mut validation, table, set).is_some() {
            return Err(AdmissionError("source_not_verified"));
        }
        let contrast = contrast.validate(&candidate, &vocabulary, boundary)?;
        Self::finish(&candidate, witnesses, &validation, contrast, vocabulary)
    }

    /// Shared final predicate; the live entry point above obtains these inputs itself. The
    /// test-only route exercises this gate without claiming an admitted source or campaign.
    fn finish(
        candidate: &MotifCandidate,
        witnesses: Vec<ExpansionWitness>,
        validation: &Validation,
        contrast: ValidatedContrast,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        let result = validation
            .motifs
            .iter()
            .find(|result| result.index == candidate.index)
            .ok_or(AdmissionError("validation_not_exercised"))?;
        if validation.schema != VALIDATION_SCHEMA
            || validation.table_sha256 != candidate.table_sha256
            || !validation.passed()
            || !validation.source_verified
            || validation.source_digest_equal != Some(true)
            || !result.passed()
            || witnesses.is_empty()
            || candidate.support_families < 2
            || witnesses.iter().any(|witness| {
                witness.table_sha256 != candidate.table_sha256
                    || witness.motif_id != candidate.motif_id
                    || witness.expanded_sha256 != witness.recipe_sha256
                    || witness.expanded_stages == 0
                    || witness.nested_macros != 1
            })
            || contrast.motif_id != candidate.motif_id
            || contrast.new_profile_id != vocabulary.new_profile_id
            || contrast.new_profile_revision != vocabulary.new_profile_revision
            || vocabulary.schema != VOCABULARY_SCHEMA
            || vocabulary.motif_id != candidate.motif_id
            || vocabulary.expansion_sha256 != sha256(candidate.expansion.as_bytes())
            || !vocabulary.old_policy_expands_macro
        {
            return Err(AdmissionError("admission_evidence_incomplete"));
        }
        let validation_sha256 = sha256(
            &serde_json::to_vec(validation).map_err(|_| AdmissionError("validation_invalid"))?,
        );
        Ok(Self {
            schema: ADMISSION_SCHEMA.to_owned(),
            table_sha256: candidate.table_sha256.clone(),
            motif_id: candidate.motif_id.clone(),
            expansion_witnesses: witnesses,
            contrast,
            vocabulary,
            validation_sha256,
        })
    }

    #[cfg(test)]
    pub(crate) fn qualify_after_verified_source(
        candidate: &MotifCandidate,
        witnesses: Vec<ExpansionWitness>,
        validation: &Validation,
        contrast: ValidatedContrast,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        Self::finish(candidate, witnesses, validation, contrast, vocabulary)
    }
}

fn motif_id_fn(expansion: &str) -> String {
    let mut bytes = b"jope-motif-expansion/v1\0".to_vec();
    bytes.extend_from_slice(expansion.as_bytes());
    sha256(&bytes)
}

fn motif_name_fn(id: &str) -> String {
    format!("{NAME_PREFIX}{}", id.get(..16).unwrap_or(id))
}

fn is_hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
