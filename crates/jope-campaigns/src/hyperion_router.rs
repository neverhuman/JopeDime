//! Train-only authority for the actual Hyperion two-stream GP router.
//!
//! This module is deliberately additive. `hyperion_gate` and the V1 budget
//! campaign remain replay authorities for their historical receipts; neither
//! is reinterpreted after the old twenty-dataset cohort was observed.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::search_v3::{canonical_sha256, validate_sha256};

// Public V2 contracts live in a task-neutral module.  Re-export them here so
// existing Hyperion callers can adopt the per-family API without changing
// their module import path.
pub use crate::router_contracts_v2::{
    build_validation_folds_v2, mase_v2, predictor_columns_without_time,
    validate_family_ensemble_v2, FamilyRouteDecisionV2, FeatureViewV2, MemberInputPlanV2,
    ModelArtifactV4, PredictionBatchV2, PredictorRouteV2, RunManifestV6, SeedRouteMetricV2,
    StrategyScreenReceiptV3, ValidationFoldV2, ValidationSchemeV2,
};

pub const EXPLORATORY_CAMPAIGN_ID: &str =
    "hyperion-two-stream-current20-train-adaptive-exploratory-v2";
pub const CONFIRMATORY_CAMPAIGN_ID: &str = "hyperion-two-stream-fresh20-confirmatory-v1";
pub const ROUTER_SEEDS: [u64; 2] = [303, 404];
pub const OUTER_FOLDS: usize = 7;
pub const INNER_FOLDS: usize = 2;
pub const SEARCH_EVALUATIONS_PER_STREAM: u32 = 2_000;
pub const FIXED_BANK_SIZE: usize = 12;
pub const DEVELOPMENT_DATASET_COUNT: usize = 22;
pub const ROUTER_DATASET_COUNT: usize = 20;
pub const MINIMUM_FIT_ROWS: usize = 32;
pub const MINIMUM_HOLD_ROWS: usize = 2;
pub const MINIMUM_HOLD_GROUPS: usize = 2;
pub const MINIMUM_ATOMIC_GROUPS: usize = 14;
pub const GROUPED_BOOTSTRAP_REPETITIONS: usize = 4_096;
pub const FIXED_CANDIDATE_TIMEOUT_SECONDS: u32 = 120;

pub const CURRENT_EXPLORATORY_DATASET_IDS: [&str; ROUTER_DATASET_COUNT] = [
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1cb97b1d7314d43d",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "542e75bc0cfdd1bf",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e",
    "b253fbfae8460dc1",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "ee56a1015e344c11",
    "fe3cc716a05d036f",
];

fn raw_sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn require_finite(name: &str, values: &[f64]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains a nonfinite value"));
    }
    Ok(())
}

fn matrix_shape(name: &str, rows: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if rows.is_empty() || rows[0].is_empty() {
        return Err(format!("{name} must be a nonempty matrix"));
    }
    let width = rows[0].len();
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != width || row.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} row {row_index} is ragged or nonfinite"));
        }
    }
    Ok((rows.len(), width))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewKindV2 {
    Raw,
    HyperionCellOnly,
    HyperionCellAppend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellEvidenceScopeV2 {
    HistoricalFullTrainDiagnostic,
    FoldLocalOuterFit,
}

/// V2 binds the real `train3.vs` bytes even for the matched raw arm. This is
/// what makes three separately executed views a single matched experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureViewContractV2 {
    pub schema: String,
    pub view: FeatureViewKindV2,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub encoder_identity_sha256: String,
    pub rows: usize,
    pub raw_width: usize,
    pub cell_width: usize,
    pub output_width: usize,
    pub cell_scope: CellEvidenceScopeV2,
    pub test_material_present: bool,
}

impl FeatureViewContractV2 {
    pub const SCHEMA: &'static str = "JopeFeatureView.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.test_material_present {
            return Err("V2 feature view schema drifted or contains test material".into());
        }
        for (name, value) in [
            ("feature-view train", self.train_sha256.as_str()),
            ("feature-view real cell", self.real_cell_sha256.as_str()),
            (
                "feature-view encoder",
                self.encoder_identity_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        if self.rows == 0
            || self.raw_width == 0
            || self.cell_width != self.raw_width
            || self.output_width
                != match self.view {
                    FeatureViewKindV2::Raw | FeatureViewKindV2::HyperionCellOnly => self.raw_width,
                    FeatureViewKindV2::HyperionCellAppend => self.raw_width * 2,
                }
        {
            return Err("V2 feature-view dimensions are invalid".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn can_support_activation(&self) -> bool {
        self.cell_scope == CellEvidenceScopeV2::FoldLocalOuterFit
    }
}

pub fn assemble_feature_view_v2(
    contract: &FeatureViewContractV2,
    raw: &[Vec<f64>],
    real_cell: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    contract.validate()?;
    let raw_shape = matrix_shape("raw features", raw)?;
    let cell_shape = matrix_shape("real Hyperion cell features", real_cell)?;
    if raw_shape != (contract.rows, contract.raw_width)
        || cell_shape != (contract.rows, contract.cell_width)
    {
        return Err("feature-view matrix shape differs from its V2 contract".into());
    }
    Ok(match contract.view {
        FeatureViewKindV2::Raw => raw.to_vec(),
        FeatureViewKindV2::HyperionCellOnly => real_cell.to_vec(),
        FeatureViewKindV2::HyperionCellAppend => raw
            .iter()
            .zip(real_cell)
            .map(|(raw_row, cell_row)| {
                let mut row = Vec::with_capacity(contract.output_width);
                row.extend_from_slice(raw_row);
                row.extend_from_slice(cell_row);
                row
            })
            .collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentGenomeObservationV1 {
    pub development_dataset_sha256: String,
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub leave_one_dataset_percentile_micros: u32,
    pub production_compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateV1 {
    pub rank: u32,
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub median_leave_one_dataset_percentile_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateBankV1 {
    pub schema: String,
    pub development_dataset_sha256s: Vec<String>,
    pub candidates: Vec<FixedCandidateV1>,
}

impl FixedCandidateBankV1 {
    pub const SCHEMA: &'static str = "HyperionFixedCandidateBank.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.development_dataset_sha256s.len() != DEVELOPMENT_DATASET_COUNT
            || self.candidates.len() != FIXED_BANK_SIZE
        {
            return Err("fixed candidate bank schema or cardinality drifted".into());
        }
        let development = self
            .development_dataset_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        if development.len() != DEVELOPMENT_DATASET_COUNT {
            return Err("fixed candidate bank development cohort is duplicated".into());
        }
        for digest in development {
            validate_sha256("development dataset", digest)?;
        }
        let mut semantics = BTreeSet::new();
        for (index, candidate) in self.candidates.iter().enumerate() {
            validate_sha256("candidate semantic", &candidate.semantic_sha256)?;
            if candidate.rank as usize != index + 1
                || candidate.canonical_genome.trim().is_empty()
                || candidate.median_leave_one_dataset_percentile_micros > 1_000_000
                || !semantics.insert(candidate.semantic_sha256.as_str())
            {
                return Err("fixed candidate bank entry is invalid or duplicated".into());
            }
            if index > 0 {
                let prior = &self.candidates[index - 1];
                if prior.median_leave_one_dataset_percentile_micros
                    < candidate.median_leave_one_dataset_percentile_micros
                    || (prior.median_leave_one_dataset_percentile_micros
                        == candidate.median_leave_one_dataset_percentile_micros
                        && prior.semantic_sha256 >= candidate.semantic_sha256)
                {
                    return Err(
                        "fixed candidate bank is not in its deterministic rank order".into(),
                    );
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

fn integer_median(values: &mut [u32]) -> u32 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((values[middle - 1] as u64 + values[middle] as u64) / 2) as u32
    } else {
        values[middle]
    }
}

/// Build the fixed bank from complete leave-one-development-dataset evidence.
/// Any missing development score excludes that genome rather than imputing it.
pub fn build_fixed_candidate_bank(
    development_dataset_sha256s: &[String],
    observations: &[DevelopmentGenomeObservationV1],
) -> Result<FixedCandidateBankV1, String> {
    let development = development_dataset_sha256s
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if development.len() != DEVELOPMENT_DATASET_COUNT {
        return Err("candidate bank requires exactly 22 unique development datasets".into());
    }
    for digest in &development {
        validate_sha256("development dataset", digest)?;
    }
    let mut grouped: BTreeMap<String, (String, BTreeMap<String, u32>)> = BTreeMap::new();
    for observation in observations {
        if !observation.production_compatible {
            continue;
        }
        validate_sha256("candidate semantic", &observation.semantic_sha256)?;
        validate_sha256(
            "candidate development dataset",
            &observation.development_dataset_sha256,
        )?;
        if !development.contains(&observation.development_dataset_sha256)
            || observation.leave_one_dataset_percentile_micros > 1_000_000
            || observation.canonical_genome.trim().is_empty()
        {
            return Err("candidate observation is foreign or invalid".into());
        }
        let (canonical, scores) = grouped
            .entry(observation.semantic_sha256.clone())
            .or_insert_with(|| (observation.canonical_genome.clone(), BTreeMap::new()));
        if canonical != &observation.canonical_genome
            || scores
                .insert(
                    observation.development_dataset_sha256.clone(),
                    observation.leave_one_dataset_percentile_micros,
                )
                .is_some()
        {
            return Err("candidate semantic has conflicting or duplicate observations".into());
        }
    }
    let mut eligible = grouped
        .into_iter()
        .filter_map(|(semantic, (canonical, scores))| {
            if scores.len() != DEVELOPMENT_DATASET_COUNT {
                return None;
            }
            let mut values = scores.into_values().collect::<Vec<_>>();
            Some((integer_median(&mut values), semantic, canonical))
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if eligible.len() < FIXED_BANK_SIZE {
        return Err(
            "fewer than twelve production-compatible genomes have complete evidence".into(),
        );
    }
    let candidates = eligible
        .into_iter()
        .take(FIXED_BANK_SIZE)
        .enumerate()
        .map(
            |(index, (median, semantic_sha256, canonical_genome))| FixedCandidateV1 {
                rank: (index + 1) as u32,
                canonical_genome,
                semantic_sha256,
                median_leave_one_dataset_percentile_micros: median,
            },
        )
        .collect();
    let bank = FixedCandidateBankV1 {
        schema: FixedCandidateBankV1::SCHEMA.into(),
        development_dataset_sha256s: development.into_iter().collect(),
        candidates,
    };
    bank.validate()?;
    Ok(bank)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedOuterFoldV2 {
    pub fold_index: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub fit_groups: Vec<String>,
    pub hold_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedOuterPlanV2 {
    pub schema: String,
    pub seed: u64,
    pub row_count: usize,
    pub atomic_group_count: usize,
    pub row_atomic_groups: Vec<String>,
    pub folds: Vec<GroupedOuterFoldV2>,
}

impl GroupedOuterPlanV2 {
    pub const SCHEMA: &'static str = "HyperionGroupedOuterPlan.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.row_count != self.row_atomic_groups.len()
            || self.row_count == 0
            || self.folds.len() != OUTER_FOLDS
            || self.row_atomic_groups.iter().any(|group| group.is_empty())
        {
            return Err("grouped outer plan schema, seed, or shape drifted".into());
        }
        let all_groups = self
            .row_atomic_groups
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if all_groups.len() != self.atomic_group_count
            || self.atomic_group_count < MINIMUM_ATOMIC_GROUPS
        {
            return Err("grouped outer plan has fewer than fourteen atomic groups".into());
        }
        let mut coverage = vec![0u8; self.row_count];
        for (index, fold) in self.folds.iter().enumerate() {
            if fold.fold_index as usize != index
                || fold.fit_rows.len() < MINIMUM_FIT_ROWS
                || fold.hold_rows.len() < MINIMUM_HOLD_ROWS
                || fold.hold_groups.len() < MINIMUM_HOLD_GROUPS
            {
                return Err(format!(
                    "outer fold {index} violates the minimum structural guards"
                ));
            }
            let fit_rows = fold.fit_rows.iter().copied().collect::<BTreeSet<_>>();
            let hold_rows = fold.hold_rows.iter().copied().collect::<BTreeSet<_>>();
            if fit_rows.len() != fold.fit_rows.len()
                || hold_rows.len() != fold.hold_rows.len()
                || !fit_rows.is_disjoint(&hold_rows)
                || fit_rows.len() + hold_rows.len() != self.row_count
                || fit_rows.union(&hold_rows).copied().ne(0..self.row_count)
            {
                return Err(format!(
                    "outer fold {index} does not partition every row once"
                ));
            }
            let expected_hold_groups = hold_rows
                .iter()
                .map(|row| self.row_atomic_groups[*row].clone())
                .collect::<BTreeSet<_>>();
            let expected_fit_groups = fit_rows
                .iter()
                .map(|row| self.row_atomic_groups[*row].clone())
                .collect::<BTreeSet<_>>();
            let actual_hold_groups = fold.hold_groups.iter().cloned().collect::<BTreeSet<_>>();
            let actual_fit_groups = fold.fit_groups.iter().cloned().collect::<BTreeSet<_>>();
            if expected_hold_groups != actual_hold_groups
                || expected_fit_groups != actual_fit_groups
                || !actual_hold_groups.is_disjoint(&actual_fit_groups)
                || actual_hold_groups.len() != fold.hold_groups.len()
                || actual_fit_groups.len() != fold.fit_groups.len()
            {
                return Err(format!(
                    "outer fold {index} leaks or misstates atomic groups"
                ));
            }
            for row in hold_rows {
                coverage[row] = coverage[row].saturating_add(1);
            }
        }
        if coverage.iter().any(|count| *count != 1) {
            return Err(
                "outer fold bank does not provide exactly one hold prediction per row".into(),
            );
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn seeded_group_key(seed: u64, group: &str) -> String {
    raw_sha256(format!("hyperion-grouped-cv7-v2|{seed}|{group}").as_bytes())
}

pub fn build_grouped_outer_plan_v2(
    row_atomic_groups: &[String],
    seed: u64,
) -> Result<GroupedOuterPlanV2, String> {
    if !ROUTER_SEEDS.contains(&seed) {
        return Err("outer plan seed is outside the frozen 303/404 bank".into());
    }
    if row_atomic_groups.is_empty() || row_atomic_groups.iter().any(|group| group.is_empty()) {
        return Err("row atomic groups are empty".into());
    }
    let mut group_rows = BTreeMap::<String, Vec<usize>>::new();
    for (row, group) in row_atomic_groups.iter().enumerate() {
        group_rows.entry(group.clone()).or_default().push(row);
    }
    if group_rows.len() < MINIMUM_ATOMIC_GROUPS {
        return Err("Hyperion routing requires at least fourteen atomic groups".into());
    }
    let mut ordered = group_rows.keys().cloned().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        seeded_group_key(seed, a)
            .cmp(&seeded_group_key(seed, b))
            .then(a.cmp(b))
    });
    let mut fold_groups = vec![Vec::<String>::new(); OUTER_FOLDS];
    let mut fold_rows = vec![0usize; OUTER_FOLDS];
    for (index, group) in ordered.into_iter().enumerate() {
        let fold = if index < OUTER_FOLDS * MINIMUM_HOLD_GROUPS {
            index % OUTER_FOLDS
        } else {
            (0..OUTER_FOLDS)
                .min_by_key(|fold| (fold_rows[*fold], fold_groups[*fold].len(), *fold))
                .expect("seven folds")
        };
        fold_rows[fold] += group_rows[&group].len();
        fold_groups[fold].push(group);
    }
    let all_rows = (0..row_atomic_groups.len()).collect::<BTreeSet<_>>();
    let all_groups = group_rows.keys().cloned().collect::<BTreeSet<_>>();
    let folds = fold_groups
        .into_iter()
        .enumerate()
        .map(|(fold_index, groups)| {
            let hold_groups = groups.into_iter().collect::<BTreeSet<_>>();
            let hold_rows = hold_groups
                .iter()
                .flat_map(|group| group_rows[group].iter().copied())
                .collect::<BTreeSet<_>>();
            let fit_rows = all_rows.difference(&hold_rows).copied().collect::<Vec<_>>();
            GroupedOuterFoldV2 {
                fold_index: fold_index as u32,
                fit_rows,
                hold_rows: hold_rows.into_iter().collect(),
                fit_groups: all_groups.difference(&hold_groups).cloned().collect(),
                hold_groups: hold_groups.into_iter().collect(),
            }
        })
        .collect();
    let plan = GroupedOuterPlanV2 {
        schema: GroupedOuterPlanV2::SCHEMA.into(),
        seed,
        row_count: row_atomic_groups.len(),
        atomic_group_count: group_rows.len(),
        row_atomic_groups: row_atomic_groups.to_vec(),
        folds,
    };
    plan.validate()?;
    Ok(plan)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldBankV2 {
    pub schema: String,
    pub plans: Vec<GroupedOuterPlanV2>,
}

impl GroupedFoldBankV2 {
    pub const SCHEMA: &'static str = "HyperionGroupedFoldBank.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.plans.len() != ROUTER_SEEDS.len() {
            return Err("grouped fold bank schema or plan cardinality drifted".into());
        }
        for (index, plan) in self.plans.iter().enumerate() {
            plan.validate()?;
            if plan.seed != ROUTER_SEEDS[index] {
                return Err("grouped fold bank seed order drifted".into());
            }
            if index > 0
                && (plan.row_count != self.plans[0].row_count
                    || plan.row_atomic_groups != self.plans[0].row_atomic_groups)
            {
                return Err(
                    "grouped fold bank plans do not describe the same rows and groups".into(),
                );
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn build_grouped_fold_bank_v2(
    row_atomic_groups: &[String],
) -> Result<GroupedFoldBankV2, String> {
    let bank = GroupedFoldBankV2 {
        schema: GroupedFoldBankV2::SCHEMA.into(),
        plans: ROUTER_SEEDS
            .iter()
            .map(|seed| build_grouped_outer_plan_v2(row_atomic_groups, *seed))
            .collect::<Result<Vec<_>, _>>()?,
    };
    bank.validate()?;
    Ok(bank)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyScreenPurposeV3 {
    FixedBankDiagnostic,
    NestedGpStream,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyScreenConfigV3 {
    pub schema: String,
    pub campaign_id: String,
    pub purpose: StrategyScreenPurposeV3,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub feature_view: FeatureViewContractV2,
    pub grouped_fold_bank_sha256: String,
    pub candidate_bank_sha256: String,
    pub configuration_sha256: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub seed: u64,
    pub evaluation_budget: u32,
    pub candidate_timeout_seconds: u32,
    pub test_material_present: bool,
}

impl StrategyScreenConfigV3 {
    pub const SCHEMA: &'static str = "JopeStrategyScreenConfig.v3";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.test_material_present
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
        {
            return Err(
                "strategy-screen V3 authority, seed, timeout, or test boundary drifted".into(),
            );
        }
        self.feature_view.validate()?;
        if self.train_sha256 != self.feature_view.train_sha256
            || self.real_cell_sha256 != self.feature_view.real_cell_sha256
        {
            return Err(
                "strategy-screen V3 feature view is not bound to the same train and cell bytes"
                    .into(),
            );
        }
        for (name, value) in [
            ("strategy dataset", self.dataset_sha256.as_str()),
            ("strategy train", self.train_sha256.as_str()),
            ("strategy real cell", self.real_cell_sha256.as_str()),
            (
                "strategy grouped fold bank",
                self.grouped_fold_bank_sha256.as_str(),
            ),
            (
                "strategy candidate bank",
                self.candidate_bank_sha256.as_str(),
            ),
            ("strategy configuration", self.configuration_sha256.as_str()),
            ("strategy source tree", self.source_tree_sha256.as_str()),
            ("strategy binary", self.binary_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        match self.purpose {
            StrategyScreenPurposeV3::FixedBankDiagnostic
                if self.evaluation_budget != (FIXED_BANK_SIZE * OUTER_FOLDS) as u32 =>
            {
                Err("fixed-bank V3 budget must be exactly twelve genomes by seven folds".into())
            }
            StrategyScreenPurposeV3::NestedGpStream
                if self.evaluation_budget != SEARCH_EVALUATIONS_PER_STREAM =>
            {
                Err("nested GP V3 budget must be exactly 2,000 evaluations".into())
            }
            _ => Ok(()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyScreenResultV3 {
    pub schema: String,
    pub config_sha256: String,
    pub campaign_id: String,
    pub purpose: StrategyScreenPurposeV3,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub feature_view_identity_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub candidate_bank_sha256: String,
    pub configuration_sha256: String,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub seed: u64,
    pub evaluation_budget: u32,
    pub evaluations_completed: u32,
    pub pooled_oof_rmse: f64,
    pub complete: bool,
    pub timed_out_candidates: u32,
    pub test_material_present: bool,
}

impl StrategyScreenResultV3 {
    pub const SCHEMA: &'static str = "JopeStrategyScreenResult.v3";

    pub fn validate(&self, config: &StrategyScreenConfigV3) -> Result<(), String> {
        config.validate()?;
        if self.schema != Self::SCHEMA
            || self.config_sha256 != config.identity()?
            || self.campaign_id != config.campaign_id
            || self.purpose != config.purpose
            || self.dataset_sha256 != config.dataset_sha256
            || self.train_sha256 != config.train_sha256
            || self.real_cell_sha256 != config.real_cell_sha256
            || self.feature_view_identity_sha256 != config.feature_view.identity()?
            || self.grouped_fold_bank_sha256 != config.grouped_fold_bank_sha256
            || self.candidate_bank_sha256 != config.candidate_bank_sha256
            || self.configuration_sha256 != config.configuration_sha256
            || self.source_tree_sha256 != config.source_tree_sha256
            || self.binary_sha256 != config.binary_sha256
            || self.seed != config.seed
            || self.evaluation_budget != config.evaluation_budget
            || self.evaluations_completed != config.evaluation_budget
            || !self.complete
            || self.timed_out_candidates != 0
            || self.test_material_present
            || !self.pooled_oof_rmse.is_finite()
            || self.pooled_oof_rmse < 0.0
        {
            return Err("strategy-screen V3 result is incomplete or not internally bound".into());
        }
        Ok(())
    }

    pub fn identity(&self, config: &StrategyScreenConfigV3) -> Result<String, String> {
        self.validate(config)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldCellEncodingReceiptV2 {
    pub schema: String,
    pub seed: u64,
    pub fold_index: u32,
    pub outer_plan_sha256: String,
    pub train_sha256: String,
    pub encoder_identity_sha256: String,
    pub fit_rows_sha256: String,
    pub hold_rows_sha256: String,
    pub fit_cell_sha256: String,
    pub hold_cell_sha256: String,
    pub fit_row_count: usize,
    pub hold_row_count: usize,
    pub feature_width: usize,
    pub cuda_authenticated: bool,
    pub isolated_gpu: bool,
    pub dummy_hold_targets: bool,
    pub hold_labels_reached_encoder: bool,
    pub test_path_resolved: bool,
    pub finite_output: bool,
}

impl FoldCellEncodingReceiptV2 {
    pub const SCHEMA: &'static str = "HyperionFoldCellEncodingReceipt.v2";

    pub fn validate(
        &self,
        plan: &GroupedOuterPlanV2,
        train_sha256: &str,
        encoder_identity_sha256: &str,
        feature_width: usize,
    ) -> Result<(), String> {
        plan.validate()?;
        let fold = plan
            .folds
            .get(self.fold_index as usize)
            .ok_or_else(|| "fold cell receipt index is outside its outer plan".to_string())?;
        if self.schema != Self::SCHEMA
            || self.seed != plan.seed
            || self.outer_plan_sha256 != plan.identity()?
            || self.train_sha256 != train_sha256
            || self.encoder_identity_sha256 != encoder_identity_sha256
            || self.fit_rows_sha256 != canonical_sha256(&fold.fit_rows)?
            || self.hold_rows_sha256 != canonical_sha256(&fold.hold_rows)?
            || self.fit_row_count != fold.fit_rows.len()
            || self.hold_row_count != fold.hold_rows.len()
            || self.feature_width != feature_width
            || feature_width == 0
            || !self.cuda_authenticated
            || !self.isolated_gpu
            || !self.dummy_hold_targets
            || self.hold_labels_reached_encoder
            || self.test_path_resolved
            || !self.finite_output
        {
            return Err(
                "fold-local cell receipt failed CUDA, isolation, shape, or leakage proof".into(),
            );
        }
        for (name, value) in [
            ("fold train", self.train_sha256.as_str()),
            ("fold encoder", self.encoder_identity_sha256.as_str()),
            ("fold fit rows", self.fit_rows_sha256.as_str()),
            ("fold hold rows", self.hold_rows_sha256.as_str()),
            ("fold fit cell", self.fit_cell_sha256.as_str()),
            ("fold hold cell", self.hold_cell_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldLocalCellCoverageV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub encoder_identity_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub feature_width: usize,
    pub authorized_full_train_cell_sha256: String,
    pub fresh_full_train_cell_sha256: String,
    pub full_train_byte_equal: bool,
    pub receipts: Vec<FoldCellEncodingReceiptV2>,
    pub structurally_eligible: bool,
    pub failure_reason: Option<String>,
}

impl FoldLocalCellCoverageV2 {
    pub const SCHEMA: &'static str = "HyperionFoldLocalCellCoverage.v2";

    pub fn validate(&self, fold_bank: &GroupedFoldBankV2) -> Result<(), String> {
        fold_bank.validate()?;
        if self.schema != Self::SCHEMA
            || self.grouped_fold_bank_sha256 != fold_bank.identity()?
            || self.feature_width == 0
        {
            return Err("fold-local coverage schema, bank, or width drifted".into());
        }
        for (name, value) in [
            ("coverage dataset", self.dataset_sha256.as_str()),
            ("coverage train", self.train_sha256.as_str()),
            ("coverage encoder", self.encoder_identity_sha256.as_str()),
            (
                "coverage authorized full cell",
                self.authorized_full_train_cell_sha256.as_str(),
            ),
            (
                "coverage fresh full cell",
                self.fresh_full_train_cell_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        if !self.structurally_eligible {
            if self
                .failure_reason
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
                || !self.receipts.is_empty()
                || self.full_train_byte_equal
            {
                return Err("ineligible coverage must fail closed with no fold receipts".into());
            }
            return Ok(());
        }
        if self.failure_reason.is_some()
            || !self.full_train_byte_equal
            || self.fresh_full_train_cell_sha256 != self.authorized_full_train_cell_sha256
            || self.receipts.len() != ROUTER_SEEDS.len() * OUTER_FOLDS
        {
            return Err(
                "eligible coverage is incomplete or does not replay train3.vs byte-for-byte".into(),
            );
        }
        let mut keys = BTreeSet::new();
        for receipt in &self.receipts {
            if !keys.insert((receipt.seed, receipt.fold_index)) {
                return Err("fold-local coverage duplicates a seed/fold receipt".into());
            }
            let plan = fold_bank
                .plans
                .iter()
                .find(|plan| plan.seed == receipt.seed)
                .ok_or_else(|| "fold receipt seed is outside the grouped bank".to_string())?;
            receipt.validate(
                plan,
                &self.train_sha256,
                &self.encoder_identity_sha256,
                self.feature_width,
            )?;
        }
        Ok(())
    }

    pub fn identity(&self, fold_bank: &GroupedFoldBankV2) -> Result<String, String> {
        self.validate(fold_bank)?;
        canonical_sha256(self)
    }

    pub fn activation_ready(&self, fold_bank: &GroupedFoldBankV2) -> bool {
        self.validate(fold_bank).is_ok() && self.structurally_eligible
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedBankSeedResultV2 {
    pub seed: u64,
    pub raw_result_sha256: String,
    pub cell_only_result_sha256: String,
    pub append_result_sha256: String,
    pub raw_pooled_rmse: f64,
    pub cell_only_pooled_rmse: f64,
    pub append_pooled_rmse: f64,
    pub append_fold_relative_regressions: Vec<f64>,
    pub complete: bool,
}

impl FixedBankSeedResultV2 {
    pub fn validate(&self) -> Result<(), String> {
        if !ROUTER_SEEDS.contains(&self.seed)
            || !self.complete
            || self.append_fold_relative_regressions.len() != OUTER_FOLDS
            || ![
                self.raw_pooled_rmse,
                self.cell_only_pooled_rmse,
                self.append_pooled_rmse,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            || self
                .append_fold_relative_regressions
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("fixed-bank seed result is incomplete or nonfinite".into());
        }
        for digest in [
            &self.raw_result_sha256,
            &self.cell_only_result_sha256,
            &self.append_result_sha256,
        ] {
            validate_sha256("fixed-bank strategy result", digest)?;
        }
        Ok(())
    }

    pub fn append_relative_improvement(&self) -> f64 {
        if self.raw_pooled_rmse > 0.0 {
            (self.raw_pooled_rmse - self.append_pooled_rmse) / self.raw_pooled_rmse
        } else if self.append_pooled_rmse == 0.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    }

    pub fn worst_append_fold_regression(&self) -> f64 {
        self.append_fold_relative_regressions
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedBankAdmissionV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub candidate_bank_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub seeds: Vec<FixedBankSeedResultV2>,
    pub append_admitted: bool,
    pub activation_authority: bool,
}

impl FixedBankAdmissionV2 {
    pub const SCHEMA: &'static str = "HyperionFixedBankAdmission.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.activation_authority
            || self.seeds.len() != ROUTER_SEEDS.len()
        {
            return Err("fixed-bank screen must remain a two-seed diagnostic".into());
        }
        for (name, value) in [
            ("fixed-bank dataset", self.dataset_sha256.as_str()),
            ("fixed-bank train", self.train_sha256.as_str()),
            ("fixed-bank real cell", self.real_cell_sha256.as_str()),
            (
                "fixed-bank candidate bank",
                self.candidate_bank_sha256.as_str(),
            ),
            (
                "fixed-bank fold bank",
                self.grouped_fold_bank_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        for (index, seed) in self.seeds.iter().enumerate() {
            seed.validate()?;
            if seed.seed != ROUTER_SEEDS[index] {
                return Err("fixed-bank seed order drifted".into());
            }
        }
        let expected_admission = self.seeds.iter().all(|seed| {
            seed.append_relative_improvement() >= 0.0 && seed.worst_append_fold_regression() <= 0.10
        });
        if self.append_admitted != expected_admission {
            return Err(
                "fixed-bank append admission does not match its frozen two-seed gate".into(),
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
pub struct GpStreamReceiptV2 {
    pub schema: String,
    pub seed: u64,
    pub outer_fold: u32,
    pub view: FeatureViewKindV2,
    pub evaluation_budget: u32,
    pub evaluations_completed: u32,
    pub inner_folds: u32,
    pub pooled_oof_rmse: bool,
    pub vocabulary: String,
    pub winner_se_band: f64,
    pub candidate_timeout_seconds: u32,
    pub search_identity_sha256: String,
    pub ranking_sha256: String,
    pub candidate_archive_sha256: String,
    pub complete: bool,
    pub failure_count: u32,
    pub outer_hold_labels_reached_search: bool,
    pub test_material_present: bool,
}

impl GpStreamReceiptV2 {
    pub const SCHEMA: &'static str = "HyperionGpStreamReceipt.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.outer_fold as usize >= OUTER_FOLDS
            || self.evaluation_budget != SEARCH_EVALUATIONS_PER_STREAM
            || self.evaluations_completed != SEARCH_EVALUATIONS_PER_STREAM
            || self.inner_folds != INNER_FOLDS as u32
            || !self.pooled_oof_rmse
            || self.vocabulary != "expanded"
            || (self.winner_se_band - 0.5).abs() > f64::EPSILON
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
            || !self.complete
            || self.failure_count != 0
            || self.outer_hold_labels_reached_search
            || self.test_material_present
        {
            return Err(
                "GP stream drifted from the independent exact-2,000 nested contract".into(),
            );
        }
        for digest in [
            &self.search_identity_sha256,
            &self.ranking_sha256,
            &self.candidate_archive_sha256,
        ] {
            validate_sha256("GP stream", digest)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedTwoStreamSearchV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub grouped_fold_bank_sha256: String,
    pub fold_cell_coverage_sha256: String,
    pub fixed_bank_admission_sha256: String,
    pub append_admitted: bool,
    pub streams: Vec<GpStreamReceiptV2>,
    pub finalist_union_sha256: String,
    pub archives_combined_only_after_search: bool,
    pub test_material_present: bool,
}

impl NestedTwoStreamSearchV2 {
    pub const SCHEMA: &'static str = "HyperionNestedTwoStreamSearch.v2";

    pub fn validate(&self, admission: &FixedBankAdmissionV2) -> Result<(), String> {
        admission.validate()?;
        if self.schema != Self::SCHEMA
            || self.dataset_sha256 != admission.dataset_sha256
            || self.grouped_fold_bank_sha256 != admission.grouped_fold_bank_sha256
            || self.fixed_bank_admission_sha256 != admission.identity()?
            || self.append_admitted != admission.append_admitted
            || !self.archives_combined_only_after_search
            || self.test_material_present
        {
            return Err("nested two-stream search is foreign or crossed the test boundary".into());
        }
        validate_sha256("nested search dataset", &self.dataset_sha256)?;
        validate_sha256("nested search coverage", &self.fold_cell_coverage_sha256)?;
        validate_sha256("nested finalist union", &self.finalist_union_sha256)?;
        let expected_views = if self.append_admitted { 3 } else { 2 };
        if self.streams.len() != ROUTER_SEEDS.len() * OUTER_FOLDS * expected_views {
            return Err("nested search stream coverage is incomplete".into());
        }
        let mut keys = BTreeSet::new();
        let mut search_ids = BTreeSet::new();
        let mut ranking_ids = BTreeSet::new();
        let mut archive_ids = BTreeSet::new();
        for stream in &self.streams {
            stream.validate()?;
            if stream.view == FeatureViewKindV2::HyperionCellAppend && !self.append_admitted {
                return Err("append stream ran without diagnostic admission".into());
            }
            if !keys.insert((stream.seed, stream.outer_fold, stream.view))
                || !search_ids.insert(stream.search_identity_sha256.as_str())
                || !ranking_ids.insert(stream.ranking_sha256.as_str())
                || !archive_ids.insert(stream.candidate_archive_sha256.as_str())
            {
                return Err("GP views shared a search, ranking, archive, or seed/fold slot".into());
            }
        }
        for seed in ROUTER_SEEDS {
            for fold in 0..OUTER_FOLDS as u32 {
                for view in [FeatureViewKindV2::Raw, FeatureViewKindV2::HyperionCellOnly] {
                    if !keys.contains(&(seed, fold, view)) {
                        return Err("raw or cell-only nested stream is missing".into());
                    }
                }
                if self.append_admitted
                    && !keys.contains(&(seed, fold, FeatureViewKindV2::HyperionCellAppend))
                {
                    return Err("admitted append nested stream is missing".into());
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self, admission: &FixedBankAdmissionV2) -> Result<String, String> {
        self.validate(admission)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendFitV2 {
    pub alpha: f64,
    pub denominator: f64,
    pub stable: bool,
}

/// Fit the raw-biased blend from inner OOF predictions only.
pub fn fit_raw_biased_blend_v2(
    truth: &[f64],
    raw_oof: &[f64],
    cell_oof: &[f64],
) -> Result<BlendFitV2, String> {
    if truth.is_empty() || truth.len() != raw_oof.len() || truth.len() != cell_oof.len() {
        return Err("blend OOF vectors are empty or have different lengths".into());
    }
    require_finite("blend truth", truth)?;
    require_finite("blend raw predictions", raw_oof)?;
    require_finite("blend cell predictions", cell_oof)?;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for ((truth, raw), cell) in truth.iter().zip(raw_oof).zip(cell_oof) {
        let direction = cell - raw;
        numerator += direction * (truth - raw);
        denominator += direction * direction;
    }
    let stability_floor = 1e-12 * truth.len() as f64;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= stability_floor {
        return Ok(BlendFitV2 {
            alpha: 0.0,
            denominator,
            stable: false,
        });
    }
    Ok(BlendFitV2 {
        alpha: (numerator / denominator).clamp(0.0, 0.5),
        denominator,
        stable: true,
    })
}

pub fn apply_raw_biased_blend_v2(
    raw_predictions: &[f64],
    cell_predictions: &[f64],
    fit: BlendFitV2,
) -> Result<Vec<f64>, String> {
    if raw_predictions.len() != cell_predictions.len()
        || raw_predictions.is_empty()
        || !fit.alpha.is_finite()
        || !(0.0..=0.5).contains(&fit.alpha)
    {
        return Err("blend prediction inputs or coefficient are invalid".into());
    }
    require_finite("blend raw predictions", raw_predictions)?;
    require_finite("blend cell predictions", cell_predictions)?;
    Ok(raw_predictions
        .iter()
        .zip(cell_predictions)
        .map(|(raw, cell)| raw + fit.alpha * (cell - raw))
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteFamilyV2 {
    Raw,
    CellOnly,
    Append,
    RawCellBlend,
}

impl RouteFamilyV2 {
    fn preference(self) -> u8 {
        match self {
            Self::Raw => 0,
            Self::CellOnly => 1,
            Self::Append => 2,
            Self::RawCellBlend => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationThresholdsV2 {
    pub minimum_seed_relative_improvement_micros: u32,
    pub minimum_seed_normalized_improvement_micros: u32,
    pub minimum_fold_wins: u32,
    pub grouped_bootstrap_confidence_micros: u32,
    pub maximum_fold_regression_micros: u32,
    pub minimum_two_seed_median_improvement_micros: u32,
    pub bootstrap_repetitions: u32,
}

impl Default for ActivationThresholdsV2 {
    fn default() -> Self {
        Self {
            minimum_seed_relative_improvement_micros: 5_000,
            minimum_seed_normalized_improvement_micros: 5_000,
            minimum_fold_wins: 5,
            grouped_bootstrap_confidence_micros: 950_000,
            maximum_fold_regression_micros: 20_000,
            minimum_two_seed_median_improvement_micros: 10_000,
            bootstrap_repetitions: GROUPED_BOOTSTRAP_REPETITIONS as u32,
        }
    }
}

fn rmse_slice(truth: &[f64], predictions: &[f64]) -> Result<f64, String> {
    if truth.is_empty() || truth.len() != predictions.len() {
        return Err("RMSE vectors are empty or have different lengths".into());
    }
    require_finite("RMSE truth", truth)?;
    require_finite("RMSE predictions", predictions)?;
    Ok((truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum::<f64>()
        / truth.len() as f64)
        .sqrt())
}

fn rmse_rows(truth: &[f64], predictions: &[f64], rows: &[usize]) -> Result<f64, String> {
    if rows.is_empty() || truth.len() != predictions.len() {
        return Err("fold RMSE rows or vectors are invalid".into());
    }
    let mut sum = 0.0;
    for row in rows {
        let actual = *truth
            .get(*row)
            .ok_or_else(|| "fold RMSE row is out of range".to_string())?;
        let predicted = predictions[*row];
        if !actual.is_finite() || !predicted.is_finite() {
            return Err("fold RMSE contains a nonfinite value".into());
        }
        sum += (actual - predicted).powi(2);
    }
    Ok((sum / rows.len() as f64).sqrt())
}

fn population_standard_deviation(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("target standard deviation requires values".into());
    }
    require_finite("training targets", values)?;
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
        return Err("training target standard deviation is zero or unstable".into());
    }
    Ok(standard_deviation)
}

fn deterministic_draw_index(tag: &str, repetition: usize, draw: usize, len: usize) -> usize {
    let digest = Sha256::digest(format!("{tag}|{repetition}|{draw}").as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(bytes) as usize) % len
}

fn grouped_bootstrap_lower_bound(
    seed: u64,
    route: RouteFamilyV2,
    truth: &[f64],
    raw_predictions: &[f64],
    route_predictions: &[f64],
    row_atomic_groups: &[String],
) -> Result<f64, String> {
    if truth.len() != raw_predictions.len()
        || truth.len() != route_predictions.len()
        || truth.len() != row_atomic_groups.len()
        || truth.is_empty()
    {
        return Err("grouped bootstrap vectors have different lengths".into());
    }
    require_finite("grouped bootstrap truth", truth)?;
    require_finite("grouped bootstrap raw", raw_predictions)?;
    require_finite("grouped bootstrap route", route_predictions)?;
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (row, group) in row_atomic_groups.iter().enumerate() {
        if group.is_empty() {
            return Err("grouped bootstrap contains an empty group ID".into());
        }
        groups.entry(group.clone()).or_default().push(row);
    }
    if groups.len() < MINIMUM_ATOMIC_GROUPS {
        return Err("grouped bootstrap requires at least fourteen atomic groups".into());
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    let tag = format!("hyperion-router-grouped-bootstrap-v2|{seed}|{route:?}");
    let mut improvements = Vec::with_capacity(GROUPED_BOOTSTRAP_REPETITIONS);
    for repetition in 0..GROUPED_BOOTSTRAP_REPETITIONS {
        let mut raw_squared_error = 0.0;
        let mut route_squared_error = 0.0;
        let mut row_count = 0usize;
        for draw in 0..groups.len() {
            let group = &groups[deterministic_draw_index(&tag, repetition, draw, groups.len())];
            for row in group {
                raw_squared_error += (truth[*row] - raw_predictions[*row]).powi(2);
                route_squared_error += (truth[*row] - route_predictions[*row]).powi(2);
                row_count += 1;
            }
        }
        let raw_rmse = (raw_squared_error / row_count as f64).sqrt();
        let route_rmse = (route_squared_error / row_count as f64).sqrt();
        improvements.push(raw_rmse - route_rmse);
    }
    improvements.sort_by(f64::total_cmp);
    let lower_index = (GROUPED_BOOTSTRAP_REPETITIONS * 5) / 100;
    Ok(improvements[lower_index])
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedActivationEvidenceV2 {
    pub schema: String,
    pub seed: u64,
    pub route: RouteFamilyV2,
    pub raw_pooled_rmse: f64,
    pub route_pooled_rmse: f64,
    pub training_target_standard_deviation: f64,
    pub relative_rmse_improvement: f64,
    pub normalized_rmse_improvement: f64,
    pub raw_fold_rmse: Vec<f64>,
    pub route_fold_rmse: Vec<f64>,
    pub fold_wins: u32,
    pub grouped_bootstrap_lower_bound: f64,
    pub worst_fold_regression: f64,
    pub grouped_bootstrap_repetitions: u32,
}

impl SeedActivationEvidenceV2 {
    pub const SCHEMA: &'static str = "HyperionSeedActivationEvidence.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.route == RouteFamilyV2::Raw
            || self.raw_fold_rmse.len() != OUTER_FOLDS
            || self.route_fold_rmse.len() != OUTER_FOLDS
            || self.grouped_bootstrap_repetitions != GROUPED_BOOTSTRAP_REPETITIONS as u32
            || self.fold_wins > OUTER_FOLDS as u32
        {
            return Err("seed activation evidence schema, route, or fold coverage drifted".into());
        }
        let finite = [
            self.raw_pooled_rmse,
            self.route_pooled_rmse,
            self.training_target_standard_deviation,
            self.relative_rmse_improvement,
            self.normalized_rmse_improvement,
            self.grouped_bootstrap_lower_bound,
            self.worst_fold_regression,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite
            || self.raw_pooled_rmse < 0.0
            || self.route_pooled_rmse < 0.0
            || self.training_target_standard_deviation <= 0.0
            || self
                .raw_fold_rmse
                .iter()
                .chain(&self.route_fold_rmse)
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("seed activation evidence contains invalid metrics".into());
        }
        let expected_relative = if self.raw_pooled_rmse > 0.0 {
            (self.raw_pooled_rmse - self.route_pooled_rmse) / self.raw_pooled_rmse
        } else {
            0.0
        };
        let expected_normalized = (self.raw_pooled_rmse - self.route_pooled_rmse)
            / self.training_target_standard_deviation;
        let expected_wins = self
            .raw_fold_rmse
            .iter()
            .zip(&self.route_fold_rmse)
            .filter(|(raw, route)| route < raw)
            .count() as u32;
        let expected_worst = self
            .raw_fold_rmse
            .iter()
            .zip(&self.route_fold_rmse)
            .map(|(raw, route)| {
                if *raw > 0.0 {
                    (route - raw) / raw
                } else if *route == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if (self.relative_rmse_improvement - expected_relative).abs() > 1e-12
            || (self.normalized_rmse_improvement - expected_normalized).abs() > 1e-12
            || self.fold_wins != expected_wins
            || (self.worst_fold_regression - expected_worst).abs() > 1e-12
        {
            return Err(
                "seed activation evidence does not reconstruct from its RMSE values".into(),
            );
        }
        Ok(())
    }

    pub fn passes(&self, thresholds: ActivationThresholdsV2) -> bool {
        self.validate().is_ok()
            && thresholds == ActivationThresholdsV2::default()
            && self.relative_rmse_improvement
                >= thresholds.minimum_seed_relative_improvement_micros as f64 / 1_000_000.0
            && self.normalized_rmse_improvement
                >= thresholds.minimum_seed_normalized_improvement_micros as f64 / 1_000_000.0
            && self.fold_wins >= thresholds.minimum_fold_wins
            && self.grouped_bootstrap_lower_bound > 0.0
            && self.worst_fold_regression
                <= thresholds.maximum_fold_regression_micros as f64 / 1_000_000.0
    }
}

pub fn evaluate_seed_activation_v2(
    seed: u64,
    route: RouteFamilyV2,
    truth: &[f64],
    raw_predictions: &[f64],
    route_predictions: &[f64],
    plan: &GroupedOuterPlanV2,
) -> Result<SeedActivationEvidenceV2, String> {
    plan.validate()?;
    if plan.seed != seed || route == RouteFamilyV2::Raw {
        return Err("activation evidence seed or route is invalid".into());
    }
    if truth.len() != plan.row_count
        || raw_predictions.len() != plan.row_count
        || route_predictions.len() != plan.row_count
    {
        return Err("activation predictions do not cover the grouped outer plan".into());
    }
    let raw_pooled_rmse = rmse_slice(truth, raw_predictions)?;
    let route_pooled_rmse = rmse_slice(truth, route_predictions)?;
    let target_sd = population_standard_deviation(truth)?;
    let raw_fold_rmse = plan
        .folds
        .iter()
        .map(|fold| rmse_rows(truth, raw_predictions, &fold.hold_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let route_fold_rmse = plan
        .folds
        .iter()
        .map(|fold| rmse_rows(truth, route_predictions, &fold.hold_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let fold_wins = raw_fold_rmse
        .iter()
        .zip(&route_fold_rmse)
        .filter(|(raw, route)| route < raw)
        .count() as u32;
    let worst_fold_regression = raw_fold_rmse
        .iter()
        .zip(&route_fold_rmse)
        .map(|(raw, route)| {
            if *raw > 0.0 {
                (route - raw) / raw
            } else if *route == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let evidence = SeedActivationEvidenceV2 {
        schema: SeedActivationEvidenceV2::SCHEMA.into(),
        seed,
        route,
        raw_pooled_rmse,
        route_pooled_rmse,
        training_target_standard_deviation: target_sd,
        relative_rmse_improvement: if raw_pooled_rmse > 0.0 {
            (raw_pooled_rmse - route_pooled_rmse) / raw_pooled_rmse
        } else {
            0.0
        },
        normalized_rmse_improvement: (raw_pooled_rmse - route_pooled_rmse) / target_sd,
        raw_fold_rmse,
        route_fold_rmse,
        fold_wins,
        grouped_bootstrap_lower_bound: grouped_bootstrap_lower_bound(
            seed,
            route,
            truth,
            raw_predictions,
            route_predictions,
            &plan.row_atomic_groups,
        )?,
        worst_fold_regression,
        grouped_bootstrap_repetitions: GROUPED_BOOTSTRAP_REPETITIONS as u32,
    };
    evidence.validate()?;
    Ok(evidence)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCandidateV2 {
    pub route: RouteFamilyV2,
    pub seeds: Vec<SeedActivationEvidenceV2>,
    pub common_fold_mean_rmse: f64,
    pub common_fold_standard_error: f64,
    pub blend_coefficients: Vec<f64>,
}

impl RouteCandidateV2 {
    pub fn validate(&self, thresholds: ActivationThresholdsV2) -> Result<(), String> {
        if self.route == RouteFamilyV2::Raw
            || self.seeds.len() != ROUTER_SEEDS.len()
            || !self.common_fold_mean_rmse.is_finite()
            || self.common_fold_mean_rmse < 0.0
            || !self.common_fold_standard_error.is_finite()
            || self.common_fold_standard_error < 0.0
        {
            return Err("nonraw route candidate is malformed".into());
        }
        for (index, evidence) in self.seeds.iter().enumerate() {
            evidence.validate()?;
            if evidence.seed != ROUTER_SEEDS[index] || evidence.route != self.route {
                return Err("route candidate seeds disagree on route family".into());
            }
        }
        if self.route == RouteFamilyV2::RawCellBlend {
            if self.blend_coefficients.len() != ROUTER_SEEDS.len()
                || self
                    .blend_coefficients
                    .iter()
                    .any(|alpha| !alpha.is_finite() || !(0.0..=0.5).contains(alpha))
            {
                return Err("blend route lacks two clipped train-only coefficients".into());
            }
        } else if !self.blend_coefficients.is_empty() {
            return Err("non-blend route carries blend coefficients".into());
        }
        if thresholds != ActivationThresholdsV2::default() {
            return Err("activation thresholds cannot be weakened or replaced".into());
        }
        Ok(())
    }

    pub fn robustly_activates(&self, thresholds: ActivationThresholdsV2) -> bool {
        if self.validate(thresholds).is_err()
            || !self
                .seeds
                .iter()
                .all(|evidence| evidence.passes(thresholds))
        {
            return false;
        }
        let median = (self.seeds[0].relative_rmse_improvement
            + self.seeds[1].relative_rmse_improvement)
            * 0.5;
        median >= thresholds.minimum_two_seed_median_improvement_micros as f64 / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterDecisionV2 {
    pub schema: String,
    pub route: RouteFamilyV2,
    pub reason: String,
    pub thresholds: ActivationThresholdsV2,
    pub selected_candidate_sha256: Option<String>,
}

impl RouterDecisionV2 {
    pub const SCHEMA: &'static str = "HyperionRouterDecision.v2";

    pub fn identity(&self) -> Result<String, String> {
        if self.schema != Self::SCHEMA
            || self.reason.trim().is_empty()
            || self.thresholds != ActivationThresholdsV2::default()
            || (self.route == RouteFamilyV2::Raw) != self.selected_candidate_sha256.is_none()
        {
            return Err("router decision is malformed or uses nonfrozen thresholds".into());
        }
        if let Some(digest) = &self.selected_candidate_sha256 {
            validate_sha256("selected route candidate", digest)?;
        }
        canonical_sha256(self)
    }
}

/// Apply every gate first, then the common-fold 0.5-SE conservative rule.
/// Preference inside the acceptance band is raw, cell-only, append, blend.
pub fn choose_route_v2(
    raw_common_fold_mean_rmse: f64,
    candidates: &[RouteCandidateV2],
    upstream_failure: Option<&str>,
) -> Result<RouterDecisionV2, String> {
    let thresholds = ActivationThresholdsV2::default();
    if !raw_common_fold_mean_rmse.is_finite() || raw_common_fold_mean_rmse < 0.0 {
        return Err("matched raw common-fold RMSE is invalid".into());
    }
    if let Some(failure) = upstream_failure {
        if failure.trim().is_empty() {
            return Err("upstream failure reason is empty".into());
        }
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: format!("predeclared raw fallback: {failure}"),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        candidate.validate(thresholds)?;
        if !seen.insert(candidate.route) {
            return Err("route candidate family is duplicated".into());
        }
    }
    let mut passing = candidates
        .iter()
        .filter(|candidate| candidate.robustly_activates(thresholds))
        .collect::<Vec<_>>();
    if passing.is_empty() {
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: "no nonraw route passed both-seed activation gates".into(),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    passing.sort_by(|a, b| {
        a.common_fold_mean_rmse
            .total_cmp(&b.common_fold_mean_rmse)
            .then_with(|| a.route.preference().cmp(&b.route.preference()))
    });
    let best = passing[0];
    let acceptance_limit = best.common_fold_mean_rmse + 0.5 * best.common_fold_standard_error;
    if raw_common_fold_mean_rmse <= acceptance_limit {
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: "matched raw is inside the frozen 0.5-SE acceptance band".into(),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    let selected = passing
        .into_iter()
        .filter(|candidate| candidate.common_fold_mean_rmse <= acceptance_limit)
        .min_by_key(|candidate| candidate.route.preference())
        .expect("best candidate is inside its own acceptance band");
    Ok(RouterDecisionV2 {
        schema: RouterDecisionV2::SCHEMA.into(),
        route: selected.route,
        reason: "nonraw route passed both seeds and won the frozen 0.5-SE preference rule".into(),
        thresholds,
        selected_candidate_sha256: Some(canonical_sha256(selected)?),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedAblationV2 {
    pub old_production_raw_rmse: f64,
    pub matched_raw_gp_rmse: f64,
    pub routed_rmse: f64,
    pub gp_lift: f64,
    pub incremental_hyperion_lift: f64,
    pub total_lift: f64,
}

impl SeedAblationV2 {
    pub fn new(
        old_production_raw_rmse: f64,
        matched_raw_gp_rmse: f64,
        routed_rmse: f64,
    ) -> Result<Self, String> {
        if ![old_production_raw_rmse, matched_raw_gp_rmse, routed_rmse]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err("ablation RMSE values must be finite and positive".into());
        }
        Ok(Self {
            old_production_raw_rmse,
            matched_raw_gp_rmse,
            routed_rmse,
            gp_lift: (old_production_raw_rmse - matched_raw_gp_rmse) / old_production_raw_rmse,
            incremental_hyperion_lift: (matched_raw_gp_rmse - routed_rmse) / matched_raw_gp_rmse,
            total_lift: (old_production_raw_rmse - routed_rmse) / old_production_raw_rmse,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterPolicyFreezeV2 {
    pub schema: String,
    pub campaign_id: String,
    pub training_adaptation_dataset_sha256s: Vec<String>,
    pub fixed_candidate_bank_sha256: String,
    pub grouped_fold_protocol_sha256: String,
    pub encoder_identity_sha256: String,
    pub search_configuration_sha256: String,
    pub source_tree_sha256: String,
    pub search_binary_sha256: String,
    pub comparator_snapshot_sha256: String,
    pub evaluator_sha256: String,
    pub finalizer_binary_sha256: String,
    pub activation_thresholds: ActivationThresholdsV2,
    pub outer_seeds: Vec<u64>,
    pub outer_folds: u32,
    pub inner_folds: u32,
    pub evaluations_per_stream: u32,
    pub candidate_timeout_seconds: u32,
    pub test_inventory_resolved: bool,
    pub historical_confirmation_identity_reused: bool,
}

impl RouterPolicyFreezeV2 {
    pub const SCHEMA: &'static str = "HyperionRouterPolicyFreeze.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.campaign_id != EXPLORATORY_CAMPAIGN_ID
            || self.training_adaptation_dataset_sha256s.len() != ROUTER_DATASET_COUNT
            || self.activation_thresholds != ActivationThresholdsV2::default()
            || self.outer_seeds != ROUTER_SEEDS
            || self.outer_folds != OUTER_FOLDS as u32
            || self.inner_folds != INNER_FOLDS as u32
            || self.evaluations_per_stream != SEARCH_EVALUATIONS_PER_STREAM
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
            || self.test_inventory_resolved
            || self.historical_confirmation_identity_reused
        {
            return Err("router policy reused history, resolved test material, or drifted from frozen rules".into());
        }
        let mut unique = BTreeSet::new();
        for (index, dataset) in self.training_adaptation_dataset_sha256s.iter().enumerate() {
            validate_sha256("policy adaptation dataset", dataset)?;
            if !dataset.starts_with(CURRENT_EXPLORATORY_DATASET_IDS[index])
                || !unique.insert(dataset.as_str())
            {
                return Err(
                    "router policy adaptation cohort is foreign, reordered, or duplicated".into(),
                );
            }
        }
        for (name, value) in [
            (
                "policy candidate bank",
                self.fixed_candidate_bank_sha256.as_str(),
            ),
            (
                "policy grouped folds",
                self.grouped_fold_protocol_sha256.as_str(),
            ),
            ("policy encoder", self.encoder_identity_sha256.as_str()),
            (
                "policy search configuration",
                self.search_configuration_sha256.as_str(),
            ),
            ("policy source tree", self.source_tree_sha256.as_str()),
            ("policy search binary", self.search_binary_sha256.as_str()),
            (
                "policy comparator",
                self.comparator_snapshot_sha256.as_str(),
            ),
            ("policy evaluator", self.evaluator_sha256.as_str()),
            ("policy finalizer", self.finalizer_binary_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
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
pub struct RouterSeedSelectionV2 {
    pub schema: String,
    pub campaign_id: String,
    pub policy_freeze_sha256: String,
    pub dataset_sha256: String,
    pub seed: u64,
    pub route: RouteFamilyV2,
    pub matched_raw_predictor_sha256: String,
    pub selected_predictor_sha256: String,
    pub blend_alpha: Option<f64>,
    pub router_decision_sha256: String,
    pub nested_search_sha256: String,
    pub fold_cell_coverage_sha256: String,
    pub fixed_bank_admission_sha256: String,
    pub selection_finalized: bool,
    pub test_inventory_resolved: bool,
    pub test_material_present: bool,
}

impl RouterSeedSelectionV2 {
    pub const SCHEMA: &'static str = "HyperionRouterSeedSelection.v2";

    pub fn validate(&self, policy_sha256: &str) -> Result<(), String> {
        validate_sha256("selection policy", policy_sha256)?;
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || self.policy_freeze_sha256 != policy_sha256
            || !ROUTER_SEEDS.contains(&self.seed)
            || !self.selection_finalized
            || self.test_inventory_resolved
            || self.test_material_present
        {
            return Err("router selection is not a complete train-only freeze".into());
        }
        for (name, value) in [
            ("selection dataset", self.dataset_sha256.as_str()),
            (
                "selection matched raw predictor",
                self.matched_raw_predictor_sha256.as_str(),
            ),
            (
                "selection selected predictor",
                self.selected_predictor_sha256.as_str(),
            ),
            ("selection decision", self.router_decision_sha256.as_str()),
            (
                "selection nested search",
                self.nested_search_sha256.as_str(),
            ),
            (
                "selection cell coverage",
                self.fold_cell_coverage_sha256.as_str(),
            ),
            (
                "selection fixed-bank admission",
                self.fixed_bank_admission_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        match (self.route, self.blend_alpha) {
            (RouteFamilyV2::Raw, None)
                if self.selected_predictor_sha256 == self.matched_raw_predictor_sha256 =>
            {
                Ok(())
            }
            (RouteFamilyV2::RawCellBlend, Some(alpha))
                if alpha.is_finite() && (0.0..=0.5).contains(&alpha) =>
            {
                Ok(())
            }
            (RouteFamilyV2::CellOnly | RouteFamilyV2::Append, None) => Ok(()),
            _ => Err("selection predictor, route, and blend coefficient disagree".into()),
        }
    }

    pub fn identity(&self, policy_sha256: &str) -> Result<String, String> {
        self.validate(policy_sha256)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterSelectionInventoryV2 {
    pub schema: String,
    pub campaign_id: String,
    pub policy_freeze_sha256: String,
    pub dataset_sha256s: Vec<String>,
    pub selections: Vec<RouterSeedSelectionV2>,
    pub all_selections_precede_test_access: bool,
    pub test_access_count_at_freeze: u32,
}

impl RouterSelectionInventoryV2 {
    pub const SCHEMA: &'static str = "HyperionRouterSelectionInventory.v2";

    pub fn validate(&self, policy: &RouterPolicyFreezeV2) -> Result<(), String> {
        policy.validate()?;
        let policy_sha256 = policy.identity()?;
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || self.policy_freeze_sha256 != policy_sha256
            || self.dataset_sha256s.len() != ROUTER_DATASET_COUNT
            || self.selections.len() != ROUTER_DATASET_COUNT * ROUTER_SEEDS.len()
            || !self.all_selections_precede_test_access
            || self.test_access_count_at_freeze != 0
        {
            return Err(
                "selection inventory is partial, foreign, or was frozen after test access".into(),
            );
        }
        let datasets = self.dataset_sha256s.iter().collect::<BTreeSet<_>>();
        if datasets.len() != ROUTER_DATASET_COUNT {
            return Err("selection inventory dataset hashes are duplicated".into());
        }
        for dataset in &self.dataset_sha256s {
            validate_sha256("selection inventory dataset", dataset)?;
        }
        if self.campaign_id == EXPLORATORY_CAMPAIGN_ID {
            for (index, dataset) in self.dataset_sha256s.iter().enumerate() {
                if !dataset.starts_with(CURRENT_EXPLORATORY_DATASET_IDS[index]) {
                    return Err(
                        "exploratory selection inventory differs from the current twenty".into(),
                    );
                }
            }
        }
        for (index, selection) in self.selections.iter().enumerate() {
            selection.validate(&policy_sha256)?;
            let expected_dataset = &self.dataset_sha256s[index / ROUTER_SEEDS.len()];
            let expected_seed = ROUTER_SEEDS[index % ROUTER_SEEDS.len()];
            if selection.campaign_id != self.campaign_id
                || &selection.dataset_sha256 != expected_dataset
                || selection.seed != expected_seed
            {
                return Err("selection inventory order or 40-slot coverage drifted".into());
            }
            if index % ROUTER_SEEDS.len() == 1
                && self.selections[index - 1].route != selection.route
            {
                return Err("the two seeds selected different route families".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self, policy: &RouterPolicyFreezeV2) -> Result<String, String> {
        self.validate(policy)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmationApplicationFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub frozen_router_policy_sha256: String,
    pub fresh_cohort_freeze_sha256: String,
    pub comparator_snapshot_sha256: String,
    pub encoder_identity_sha256: String,
    pub search_binary_sha256: String,
    pub evaluator_sha256: String,
    pub finalizer_binary_sha256: String,
    pub thresholds_unchanged: bool,
    pub budgets_unchanged: bool,
    pub finalizer_unchanged: bool,
    pub test_inventory_resolved: bool,
}

impl ConfirmationApplicationFreezeV1 {
    pub const SCHEMA: &'static str = "HyperionConfirmationApplicationFreeze.v1";

    pub fn validate(&self, policy: &RouterPolicyFreezeV2) -> Result<(), String> {
        policy.validate()?;
        if self.schema != Self::SCHEMA
            || self.campaign_id != CONFIRMATORY_CAMPAIGN_ID
            || self.frozen_router_policy_sha256 != policy.identity()?
            || self.comparator_snapshot_sha256 != policy.comparator_snapshot_sha256
            || self.encoder_identity_sha256 != policy.encoder_identity_sha256
            || self.search_binary_sha256 != policy.search_binary_sha256
            || self.evaluator_sha256 != policy.evaluator_sha256
            || self.finalizer_binary_sha256 != policy.finalizer_binary_sha256
            || !self.thresholds_unchanged
            || !self.budgets_unchanged
            || !self.finalizer_unchanged
            || self.test_inventory_resolved
        {
            return Err("fresh confirmation does not apply the frozen router unchanged".into());
        }
        validate_sha256("fresh cohort freeze", &self.fresh_cohort_freeze_sha256)
    }

    pub fn identity(&self, policy: &RouterPolicyFreezeV2) -> Result<String, String> {
        self.validate(policy)?;
        canonical_sha256(self)
    }
}

/// A single-open boundary. Labels remain private until a route has been
/// chosen exclusively from encoding proof and the frozen selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SealedFeatureStageV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub feature_count: usize,
    pub feature_rows: Vec<Vec<f64>>,
    labels: Vec<f64>,
    pub test_access_ordinal: u32,
}

impl SealedFeatureStageV2 {
    pub const SCHEMA: &'static str = "HyperionSealedFeatureStage.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_access_ordinal != 1
            || self.feature_count == 0
            || self.feature_rows.is_empty()
            || self.labels.len() != self.feature_rows.len()
            || self.feature_rows.iter().any(|row| {
                row.len() != self.feature_count || row.iter().any(|value| !value.is_finite())
            })
            || self.labels.iter().any(|value| !value.is_finite())
        {
            return Err("sealed V2 feature stage has invalid shape or access boundary".into());
        }
        validate_sha256("sealed V2 dataset", &self.dataset_sha256)?;
        validate_sha256("sealed V2 test", &self.test_sha256)
    }

    pub fn hyperion_dummy_target_rows(&self) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;
        Ok(self
            .feature_rows
            .iter()
            .map(|features| {
                let mut row = features.clone();
                row.push(0.0);
                row
            })
            .collect())
    }

    fn labels_after_route_lock(&self) -> Result<&[f64], String> {
        self.validate()?;
        Ok(&self.labels)
    }
}

pub fn stage_sealed_test_once_v2(
    dataset_sha256: &str,
    test_sha256: &str,
    bytes: &[u8],
    feature_count: usize,
) -> Result<SealedFeatureStageV2, String> {
    validate_sha256("sealed V2 dataset", dataset_sha256)?;
    validate_sha256("sealed V2 test", test_sha256)?;
    if bytes.is_empty() || feature_count == 0 || raw_sha256(bytes) != test_sha256 {
        return Err("sealed test bytes are empty or differ from their authorized hash".into());
    }
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("sealed test is not UTF-8: {error}"))?;
    let mut feature_rows = Vec::new();
    let mut labels = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>();
        let mut values = match parsed {
            Ok(values) => values,
            Err(_) if line_index == 0 => continue,
            Err(error) => return Err(format!("sealed test row {line_index} is invalid: {error}")),
        };
        if values.len() != feature_count + 1 {
            return Err(format!(
                "sealed test row {line_index} has {} columns, expected {}",
                values.len(),
                feature_count + 1
            ));
        }
        let label = values.pop().expect("feature and target count checked");
        if !label.is_finite() || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("sealed test row {line_index} is nonfinite"));
        }
        feature_rows.push(values);
        labels.push(label);
    }
    let stage = SealedFeatureStageV2 {
        schema: SealedFeatureStageV2::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        test_sha256: test_sha256.into(),
        feature_count,
        feature_rows,
        labels,
        test_access_ordinal: 1,
    };
    stage.validate()?;
    Ok(stage)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizerEncodingProofV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub selection_sha256: String,
    pub requested_route: RouteFamilyV2,
    pub requested_predictor_sha256: String,
    pub authorized_full_train_cell_sha256: String,
    pub fresh_full_train_cell_sha256: String,
    pub authorized_test_cell_sha256: Option<String>,
    pub fresh_test_cell_sha256: String,
    pub encoder_identity_sha256: String,
    pub cuda_authenticated: bool,
    pub isolated_gpu: bool,
    pub dummy_test_targets: bool,
    pub labels_reached_hyperion: bool,
    pub train_shape_valid: bool,
    pub test_shape_valid: bool,
    pub row_order_valid: bool,
    pub finite_output: bool,
    pub test_row_count: usize,
    pub feature_width: usize,
    pub raw_latency_seconds: f64,
    pub routed_latency_seconds: f64,
}

impl FinalizerEncodingProofV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizerEncodingProof.v2";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizerRouteDecisionV2 {
    pub schema: String,
    pub requested_route: RouteFamilyV2,
    pub chosen_route: RouteFamilyV2,
    pub chosen_predictor_sha256: String,
    pub decision_preceded_label_metrics: bool,
    pub fallback_reasons: Vec<String>,
}

impl FinalizerRouteDecisionV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizerRouteDecision.v2";
}

/// Validate every encoder/provenance condition without reading `stage.labels`.
/// A bad proof is data, not an exception: it deterministically chooses raw.
pub fn lock_finalizer_route_before_labels_v2(
    stage: &SealedFeatureStageV2,
    selection: &RouterSeedSelectionV2,
    policy: &RouterPolicyFreezeV2,
    proof: &FinalizerEncodingProofV2,
) -> Result<FinalizerRouteDecisionV2, String> {
    stage.validate()?;
    policy.validate()?;
    let policy_sha256 = policy.identity()?;
    selection.validate(&policy_sha256)?;
    let selection_sha256 = selection.identity(&policy_sha256)?;
    let mut failures = Vec::new();
    if proof.schema != FinalizerEncodingProofV2::SCHEMA {
        failures.push("encoding proof schema drift".into());
    }
    if proof.dataset_sha256 != stage.dataset_sha256
        || proof.dataset_sha256 != selection.dataset_sha256
    {
        failures.push("dataset identity mismatch".into());
    }
    if proof.selection_sha256 != selection_sha256
        || proof.requested_route != selection.route
        || proof.requested_predictor_sha256 != selection.selected_predictor_sha256
    {
        failures.push("route identity mismatch".into());
    }
    if proof.encoder_identity_sha256 != policy.encoder_identity_sha256 {
        failures.push("encoder identity mismatch".into());
    }
    if validate_sha256(
        "authorized train cell",
        &proof.authorized_full_train_cell_sha256,
    )
    .is_err()
        || proof.fresh_full_train_cell_sha256 != proof.authorized_full_train_cell_sha256
    {
        failures.push("fresh full-train cell output is not byte-identical to train3.vs".into());
    }
    if validate_sha256("fresh test cell", &proof.fresh_test_cell_sha256).is_err() {
        failures.push("fresh test cell hash is invalid".into());
    }
    if let Some(expected) = &proof.authorized_test_cell_sha256 {
        if validate_sha256("authorized test cell", expected).is_err()
            || &proof.fresh_test_cell_sha256 != expected
        {
            failures.push("fresh test cell output differs from its authorized pre-cache".into());
        }
    }
    if !proof.cuda_authenticated || !proof.isolated_gpu {
        failures.push("CUDA or isolated-GPU proof failed".into());
    }
    if !proof.dummy_test_targets || proof.labels_reached_hyperion {
        failures.push("test labels reached or could reach Hyperion".into());
    }
    if !proof.train_shape_valid
        || !proof.test_shape_valid
        || !proof.row_order_valid
        || !proof.finite_output
        || proof.test_row_count != stage.feature_rows.len()
        || proof.feature_width != stage.feature_count
    {
        failures.push("cell shape, order, finiteness, or row count failed".into());
    }
    if !proof.raw_latency_seconds.is_finite()
        || !proof.routed_latency_seconds.is_finite()
        || proof.raw_latency_seconds < 0.0
        || proof.routed_latency_seconds < 0.0
    {
        failures.push("latency proof is invalid".into());
    }
    let (chosen_route, chosen_predictor_sha256) = if failures.is_empty() {
        (selection.route, selection.selected_predictor_sha256.clone())
    } else {
        (
            RouteFamilyV2::Raw,
            selection.matched_raw_predictor_sha256.clone(),
        )
    };
    Ok(FinalizerRouteDecisionV2 {
        schema: FinalizerRouteDecisionV2::SCHEMA.into(),
        requested_route: selection.route,
        chosen_route,
        chosen_predictor_sha256,
        decision_preceded_label_metrics: true,
        fallback_reasons: failures,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizedSeedScoreV2 {
    pub schema: String,
    pub seed: u64,
    pub route_decision_sha256: String,
    pub old_production_raw_rmse: f64,
    pub matched_raw_gp_rmse: f64,
    pub routed_rmse: f64,
    pub ablation: SeedAblationV2,
    pub labels_reached_hyperion: bool,
    pub labels_reached_prediction_fit: bool,
    pub common_staged_data_and_evaluator: bool,
}

impl FinalizedSeedScoreV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizedSeedScore.v2";
}

pub fn score_after_finalizer_lock_v2(
    stage: &SealedFeatureStageV2,
    seed: u64,
    decision: &FinalizerRouteDecisionV2,
    old_raw_predictions: &[f64],
    matched_raw_predictions: &[f64],
    requested_route_predictions: &[f64],
) -> Result<FinalizedSeedScoreV2, String> {
    stage.validate()?;
    if !ROUTER_SEEDS.contains(&seed)
        || decision.schema != FinalizerRouteDecisionV2::SCHEMA
        || !decision.decision_preceded_label_metrics
        || (decision.chosen_route == RouteFamilyV2::Raw) != !decision.fallback_reasons.is_empty()
            && decision.requested_route != RouteFamilyV2::Raw
    {
        return Err("finalizer score was not preceded by a valid route lock".into());
    }
    let labels = stage.labels_after_route_lock()?;
    for (name, predictions) in [
        ("old raw", old_raw_predictions),
        ("matched raw", matched_raw_predictions),
        ("requested route", requested_route_predictions),
    ] {
        if predictions.len() != labels.len() {
            return Err(format!(
                "{name} predictions do not cover the staged test rows"
            ));
        }
        require_finite(name, predictions)?;
    }
    let routed_predictions = if decision.chosen_route == RouteFamilyV2::Raw {
        matched_raw_predictions
    } else {
        requested_route_predictions
    };
    let old_production_raw_rmse = rmse_slice(labels, old_raw_predictions)?;
    let matched_raw_gp_rmse = rmse_slice(labels, matched_raw_predictions)?;
    let routed_rmse = rmse_slice(labels, routed_predictions)?;
    let ablation = SeedAblationV2::new(old_production_raw_rmse, matched_raw_gp_rmse, routed_rmse)?;
    Ok(FinalizedSeedScoreV2 {
        schema: FinalizedSeedScoreV2::SCHEMA.into(),
        seed,
        route_decision_sha256: canonical_sha256(decision)?,
        old_production_raw_rmse,
        matched_raw_gp_rmse,
        routed_rmse,
        ablation,
        labels_reached_hyperion: false,
        labels_reached_prediction_fit: false,
        common_staged_data_and_evaluator: true,
    })
}
