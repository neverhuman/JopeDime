//! Four-task scientific and operational contracts.
//!
//! These types deliberately distinguish absence from numeric zero and keep
//! within-task metrics separate.  They are shared by ingestion, rating, the
//! read-only TUI, and promotion tooling so none of those consumers can invent
//! a task-specific interpretation locally.

use crate::selfplay_shadow::{Sha256Digest, Task};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const FOUR_TASK_TELEMETRY_SCHEMA: &str = "JopeFourTaskTelemetry.v1";
pub const PROPOSAL_SELECTION_POLICY_SCHEMA: &str = "JopeProposalSelectionPolicy.v1";
pub const ROBUST_SEED_BANK_V2_SCHEMA: &str = "JopeRobustSeedBank.v2";
pub const EVOLUTION_TRAJECTORY_V2_SCHEMA: &str = "JopeEvolutionTrajectory.v2";
pub const HUMAN_GENOME_SEED_V1_SCHEMA: &str = "JopeHumanGenomeSeed.v1";

/// Display order is a contract and intentionally differs from the historical
/// rotation order: optional supervised multiclass precedes time series.
pub const FOUR_TASK_ORDER: [Task; 4] = [
    Task::Regression,
    Task::Binary,
    Task::Multiclass,
    Task::Timeseries,
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeErrorAggregation {
    FamilyBalancedDistribution,
    PerContextOnly,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskMetricContract {
    pub task: Task,
    pub primary_metric: &'static str,
    pub primary_display: &'static str,
    pub native_error: &'static str,
    pub native_error_aggregation: NativeErrorAggregation,
}

pub fn task_metric_contract(task: Task) -> TaskMetricContract {
    let semantics = task.semantics();
    match task {
        Task::Regression => TaskMetricContract {
            task,
            primary_metric: semantics.primary_metric.as_str(),
            primary_display: "R²",
            native_error: semantics.native_error_metric.as_str(),
            native_error_aggregation: NativeErrorAggregation::PerContextOnly,
        },
        Task::Binary => TaskMetricContract {
            task,
            primary_metric: semantics.primary_metric.as_str(),
            primary_display: "AUC",
            native_error: semantics.native_error_metric.as_str(),
            native_error_aggregation: NativeErrorAggregation::FamilyBalancedDistribution,
        },
        Task::Multiclass => TaskMetricContract {
            task,
            primary_metric: semantics.primary_metric.as_str(),
            primary_display: "macro-F1",
            native_error: semantics.native_error_metric.as_str(),
            native_error_aggregation: NativeErrorAggregation::FamilyBalancedDistribution,
        },
        Task::Timeseries => TaskMetricContract {
            task,
            primary_metric: semantics.primary_metric.as_str(),
            primary_display: "MASE",
            native_error: semantics.native_error_metric.as_str(),
            native_error_aggregation: NativeErrorAggregation::FamilyBalancedDistribution,
        },
    }
}

/// Shared rating/learning utility. Invalidity is handled by the caller and is
/// always -1; finite task scores are bounded before any subtraction.
pub fn bounded_task_utility(task: Task, score: f64) -> Result<f64> {
    Ok(crate::task::normalize_reward(task, score)?.get())
}

pub fn operational_loss(task: Task, score: f64) -> Result<f64> {
    crate::task::operational_loss(task, score)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Distribution {
    pub count: u64,
    pub minimum: f64,
    pub p05: f64,
    pub median: f64,
    pub mean: f64,
    pub p95: f64,
    pub maximum: f64,
}

impl Distribution {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.count > 0, "distribution is empty");
        ensure!(
            [
                self.minimum,
                self.p05,
                self.median,
                self.mean,
                self.p95,
                self.maximum,
            ]
            .into_iter()
            .all(f64::is_finite),
            "distribution contains a non-finite value"
        );
        ensure!(
            self.minimum <= self.p05
                && self.p05 <= self.median
                && self.median <= self.p95
                && self.p95 <= self.maximum,
            "distribution quantiles are not ordered"
        );
        Ok(())
    }

    pub fn from_values(mut values: Vec<f64>) -> Result<Self> {
        ensure!(
            !values.is_empty() && values.iter().copied().all(f64::is_finite),
            "distribution values are empty or non-finite"
        );
        values.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            let index = ((values.len() - 1) as f64 * fraction).round() as usize;
            values[index]
        };
        Ok(Self {
            count: values.len() as u64,
            minimum: values[0],
            p05: percentile(0.05),
            median: percentile(0.50),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p95: percentile(0.95),
            maximum: *values.last().expect("nonempty values"),
        })
    }
}

macro_rules! typed_value {
    ($name:ident, $value:ty) => {
        #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub value: Option<$value>,
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub na_reason: Option<String>,
        }

        impl $name {
            pub fn value(value: $value) -> Self {
                Self {
                    value: Some(value),
                    na_reason: None,
                }
            }

            pub fn na(reason: impl Into<String>) -> Self {
                Self {
                    value: None,
                    na_reason: Some(reason.into()),
                }
            }

            pub fn validate(&self) -> Result<()> {
                ensure!(
                    self.value.is_some()
                        ^ self
                            .na_reason
                            .as_deref()
                            .is_some_and(|reason| !reason.trim().is_empty()),
                    "typed value must contain exactly one value or N/A reason"
                );
                Ok(())
            }
        }
    };
}

typed_value!(TypedCount, u64);
typed_value!(TypedNumber, f64);
typed_value!(TypedText, String);
typed_value!(TypedDistribution, Distribution);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RatingTelemetry {
    pub snapshot_sha256: Sha256Digest,
    pub player: String,
    pub rating: f64,
    pub interval_95: [f64; 2],
    pub delta_from_anchor: f64,
    pub promotion_eligible: bool,
}

impl RatingTelemetry {
    fn validate(&self) -> Result<()> {
        ensure!(!self.player.trim().is_empty(), "rating player is empty");
        ensure!(
            [
                self.rating,
                self.interval_95[0],
                self.interval_95[1],
                self.delta_from_anchor,
            ]
            .into_iter()
            .all(f64::is_finite),
            "rating telemetry contains a non-finite value"
        );
        ensure!(
            self.interval_95[0] <= self.interval_95[1],
            "rating interval is reversed"
        );
        ensure!(
            !self.promotion_eligible,
            "adaptive/diagnostic rating was marked promotion eligible"
        );
        Ok(())
    }
}

typed_value!(TypedRating, RatingTelemetry);

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryCounts {
    pub catalog_entries: TypedCount,
    pub source_families: TypedCount,
    pub encoded: TypedCount,
    pub active: TypedCount,
    pub observed: TypedCount,
    pub queued: TypedCount,
    pub running: TypedCount,
    pub completed: TypedCount,
    pub failed: TypedCount,
    pub quarantined: TypedCount,
}

impl TelemetryCounts {
    fn validate(&self) -> Result<()> {
        for value in [
            &self.catalog_entries,
            &self.source_families,
            &self.encoded,
            &self.active,
            &self.observed,
            &self.queued,
            &self.running,
            &self.completed,
            &self.failed,
            &self.quarantined,
        ] {
            value.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryMetrics {
    pub primary_metric: String,
    pub primary_performance: TypedDistribution,
    pub native_errors: BTreeMap<String, TypedDistribution>,
    pub bounded_utility: TypedDistribution,
    pub operational_loss: TypedDistribution,
}

impl TelemetryMetrics {
    fn validate(&self, task: Task) -> Result<()> {
        ensure!(
            self.primary_metric == task_metric_contract(task).primary_metric,
            "telemetry primary metric drift"
        );
        self.primary_performance.validate()?;
        self.bounded_utility.validate()?;
        self.operational_loss.validate()?;
        if let Some(value) = &self.primary_performance.value {
            value.validate()?;
        }
        if let Some(value) = &self.bounded_utility.value {
            value.validate()?;
        }
        if let Some(value) = &self.operational_loss.value {
            value.validate()?;
        }
        for value in self.native_errors.values() {
            value.validate()?;
            if let Some(distribution) = &value.value {
                distribution.validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryEligibility {
    pub evidence_eligible: bool,
    pub training_eligible: bool,
    pub promotion_eligible: bool,
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerValidationTelemetry {
    pub metrics: BTreeMap<String, TypedNumber>,
    pub age_seconds: TypedCount,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBinding {
    pub path: String,
    pub schema: String,
    pub sha256: Sha256Digest,
    pub timestamp_ms: u128,
    pub freshness_budget_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourTaskTelemetryRow {
    pub task: Task,
    pub counts: TelemetryCounts,
    pub latest_observation_at_ms: TypedCount,
    pub valid_observations_per_minute: TypedNumber,
    pub evaluator_failures_per_minute: TypedNumber,
    pub freshness_reason: String,
    pub metrics: TelemetryMetrics,
    pub accepted_checkpoint: TypedText,
    pub bootstrap_anchor_state: String,
    pub latest_shadow: TypedText,
    pub active_candidate: TypedText,
    pub eligibility: TelemetryEligibility,
    pub adaptive_rating: TypedRating,
    pub diagnostic_rating: TypedRating,
    pub sealed_promotion: TypedText,
    pub trainer_validation: TrainerValidationTelemetry,
    pub sources: Vec<SourceBinding>,
}

impl FourTaskTelemetryRow {
    fn validate(&self) -> Result<()> {
        self.counts.validate()?;
        self.latest_observation_at_ms.validate()?;
        self.valid_observations_per_minute.validate()?;
        self.evaluator_failures_per_minute.validate()?;
        ensure!(
            !self.freshness_reason.trim().is_empty(),
            "telemetry freshness reason is empty"
        );
        self.metrics.validate(self.task)?;
        self.accepted_checkpoint.validate()?;
        self.latest_shadow.validate()?;
        self.active_candidate.validate()?;
        self.adaptive_rating.validate()?;
        self.diagnostic_rating.validate()?;
        if let Some(rating) = &self.adaptive_rating.value {
            rating.validate()?;
        }
        if let Some(rating) = &self.diagnostic_rating.value {
            rating.validate()?;
        }
        self.sealed_promotion.validate()?;
        self.trainer_validation.age_seconds.validate()?;
        for metric in self.trainer_validation.metrics.values() {
            metric.validate()?;
            ensure!(
                metric.value.is_none_or(f64::is_finite),
                "trainer metric is non-finite"
            );
        }
        ensure!(
            self.valid_observations_per_minute
                .value
                .is_none_or(f64::is_finite)
                && self
                    .evaluator_failures_per_minute
                    .value
                    .is_none_or(f64::is_finite),
            "telemetry rate is non-finite"
        );
        ensure!(
            !self.bootstrap_anchor_state.trim().is_empty(),
            "bootstrap anchor state is empty"
        );
        let mut source_paths = BTreeSet::new();
        for source in &self.sources {
            ensure!(
                !source.path.trim().is_empty()
                    && !source.schema.trim().is_empty()
                    && source.freshness_budget_seconds > 0
                    && source_paths.insert(source.path.as_str()),
                "telemetry source binding is empty or duplicated"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeFourTaskTelemetryV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub corpus_watermark: TypedCount,
    pub rows: Vec<FourTaskTelemetryRow>,
}

impl JopeFourTaskTelemetryV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == FOUR_TASK_TELEMETRY_SCHEMA,
            "four-task telemetry schema is not v1"
        );
        ensure!(
            self.generated_at_ms > 0,
            "telemetry generation time is absent"
        );
        self.corpus_watermark.validate()?;
        ensure!(
            self.rows.len() == FOUR_TASK_ORDER.len()
                && self.rows.iter().map(|row| row.task).eq(FOUR_TASK_ORDER),
            "four-task telemetry must contain exactly four canonical rows"
        );
        for row in &self.rows {
            row.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedRiskPolicy {
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_sha256: Option<Sha256Digest>,
    pub fallback: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionCoefficients {
    pub reward_dispersion: f64,
    pub instability: f64,
    pub static_risk: f64,
    pub instability_hard_threshold: f64,
    pub instability_hard_penalty: f64,
    pub complexity: f64,
    pub novelty: f64,
    pub latency: f64,
    pub calibrated_win: f64,
    pub calibrated_delta: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeProposalSelectionPolicyV1 {
    pub schema: String,
    pub task_head_routing: BTreeMap<Task, String>,
    pub learned_risk: LearnedRiskPolicy,
    pub coefficients: SelectionCoefficients,
    pub tie_breaking: Vec<String>,
    pub diversity_policy: String,
    pub proposal_budget: usize,
    pub invalid_slot_utility: f64,
}

impl JopeProposalSelectionPolicyV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PROPOSAL_SELECTION_POLICY_SCHEMA,
            "proposal selection policy schema is not v1"
        );
        let expected = BTreeMap::from([
            (Task::Regression, "regression".to_owned()),
            (Task::Binary, "binary".to_owned()),
            (Task::Multiclass, "binary".to_owned()),
            (Task::Timeseries, "regression".to_owned()),
        ]);
        ensure!(
            self.task_head_routing == expected,
            "proposal task/head routing drift"
        );
        ensure!(
            self.proposal_budget == 8 && self.invalid_slot_utility == -1.0,
            "proposal budget or invalid-slot utility drift"
        );
        ensure!(
            !self.learned_risk.enabled
                && self.learned_risk.calibration_sha256.is_none()
                && self.learned_risk.fallback == "empirical_reward_dispersion",
            "learned risk is enabled without family-disjoint calibration"
        );
        let coefficients = &self.coefficients;
        ensure!(
            [
                coefficients.reward_dispersion,
                coefficients.instability,
                coefficients.static_risk,
                coefficients.instability_hard_threshold,
                coefficients.instability_hard_penalty,
                coefficients.complexity,
                coefficients.novelty,
                coefficients.latency,
                coefficients.calibrated_win,
                coefficients.calibrated_delta,
            ]
            .into_iter()
            .all(f64::is_finite),
            "proposal policy has a non-finite coefficient"
        );
        ensure!(
            self.tie_breaking
                == [
                    "selection_score_desc",
                    "terminal_head_diversity",
                    "semantic_sha256_asc",
                ]
                .map(str::to_owned),
            "proposal tie-breaking drift"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginClassV2 {
    GpOrigin,
    HumanGpSeed,
    DlOrigin,
    DlDescendant,
    Mixed,
    Unknown,
}

impl OriginClassV2 {
    pub fn zero_dl(self) -> bool {
        matches!(self, Self::GpOrigin | Self::HumanGpSeed)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobustSeedRowV2 {
    pub task: Task,
    pub semantic_sha256: Sha256Digest,
    pub canonical_genome: String,
    pub origin_class: OriginClassV2,
    pub ancestry_evidence_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub exporter_sha256: Sha256Digest,
    pub evidence_watermark: u64,
    pub source_family_support: u64,
    pub within_family_mean_utility: String,
    pub within_family_median_utility: String,
    pub zero_dl_comparator_eligible: bool,
    pub human_seed_eligible: bool,
    pub dl_origin: bool,
    pub dl_descendant: bool,
}

impl RobustSeedRowV2 {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.canonical_genome.trim().is_empty()
                && self.semantic_sha256 == Sha256Digest::of_bytes(self.canonical_genome.as_bytes()),
            "robust seed semantic identity drift"
        );
        ensure!(
            self.source_family_support > 0,
            "robust seed lacks source-family support"
        );
        for value in [
            self.within_family_mean_utility.as_str(),
            self.within_family_median_utility.as_str(),
        ] {
            let parsed = value.parse::<f64>()?;
            ensure!(parsed.is_finite(), "robust seed merit is non-finite");
        }
        ensure!(
            self.zero_dl_comparator_eligible == self.origin_class.zero_dl()
                && self.human_seed_eligible == (self.origin_class == OriginClassV2::HumanGpSeed)
                && self.dl_origin == (self.origin_class == OriginClassV2::DlOrigin)
                && self.dl_descendant == (self.origin_class == OriginClassV2::DlDescendant),
            "robust seed ancestry flags disagree with origin_class"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeRobustSeedBankV2 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub prior_bank_sha256: Option<Sha256Digest>,
    pub tasks: BTreeMap<Task, Vec<RobustSeedRowV2>>,
}

impl JopeRobustSeedBankV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ROBUST_SEED_BANK_V2_SCHEMA,
            "robust seed bank schema is not v2"
        );
        ensure!(
            self.tasks.keys().copied().collect::<BTreeSet<_>>()
                == FOUR_TASK_ORDER.into_iter().collect(),
            "robust seed bank is not task-complete"
        );
        let mut semantics = BTreeSet::new();
        for (task, rows) in &self.tasks {
            for row in rows {
                ensure!(row.task == *task, "robust seed task attribution drift");
                row.validate()?;
                ensure!(
                    semantics.insert((*task, &row.semantic_sha256)),
                    "robust seed bank repeats a task semantic"
                );
            }
        }
        Ok(())
    }
}

pub fn validate_additive_seed_refresh(
    previous: &JopeRobustSeedBankV2,
    next: &JopeRobustSeedBankV2,
) -> Result<()> {
    previous.validate()?;
    next.validate()?;
    for task in FOUR_TASK_ORDER {
        let before = previous.tasks[&task]
            .iter()
            .map(|row| &row.semantic_sha256)
            .collect::<BTreeSet<_>>();
        let after = next.tasks[&task]
            .iter()
            .map(|row| &row.semantic_sha256)
            .collect::<BTreeSet<_>>();
        ensure!(
            before.is_subset(&after),
            "robust seed refresh erased {task} rows"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionOperatorV2 {
    Initial,
    Mutation,
    Crossover,
    DiversityImmigrant,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryDispositionV2 {
    Censored,
    Completed,
    Selected,
    Elite,
    Terminal,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DescendantUtilitiesV2 {
    pub horizon_1: Option<f64>,
    pub horizon_2: Option<f64>,
    pub horizon_4: Option<f64>,
    pub horizon_8: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeEvolutionTrajectoryV2 {
    pub schema: String,
    pub task: Task,
    pub dataset_hash: String,
    pub source_family_sha256: Sha256Digest,
    pub generation: u32,
    pub population_slot: u32,
    pub canonical_genome: String,
    pub semantic_sha256: Sha256Digest,
    pub parent_semantic_sha256: Vec<Sha256Digest>,
    pub operator: EvolutionOperatorV2,
    pub deterministic_seed: u64,
    pub origin_class: OriginClassV2,
    pub proposal_scores: BTreeMap<String, f64>,
    pub proposal_selection_policy_sha256: Sha256Digest,
    pub evaluator_score: Option<f64>,
    pub native_errors: BTreeMap<String, f64>,
    pub valid: bool,
    pub failure_class: Option<String>,
    pub runtime_ms: u64,
    pub resource_observations: BTreeMap<String, String>,
    pub descendant_utilities: DescendantUtilitiesV2,
    pub disposition: TrajectoryDispositionV2,
    pub promotion_eligible: bool,
    pub test_adaptive: bool,
}

impl JopeEvolutionTrajectoryV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EVOLUTION_TRAJECTORY_V2_SCHEMA,
            "evolution trajectory schema is not v2"
        );
        ensure!(
            !self.dataset_hash.trim().is_empty()
                && !self.canonical_genome.trim().is_empty()
                && self.semantic_sha256 == Sha256Digest::of_bytes(self.canonical_genome.as_bytes()),
            "trajectory identity is incomplete"
        );
        let expected_parents = match self.operator {
            EvolutionOperatorV2::Initial | EvolutionOperatorV2::DiversityImmigrant => 0,
            EvolutionOperatorV2::Mutation => 1,
            EvolutionOperatorV2::Crossover => 2,
        };
        ensure!(
            self.parent_semantic_sha256.len() == expected_parents,
            "trajectory parent arity disagrees with its operator"
        );
        ensure!(
            self.proposal_scores.values().copied().all(f64::is_finite)
                && self.native_errors.values().copied().all(f64::is_finite)
                && self.evaluator_score.is_none_or(f64::is_finite),
            "trajectory contains a non-finite score"
        );
        let descendants = [
            self.descendant_utilities.horizon_1,
            self.descendant_utilities.horizon_2,
            self.descendant_utilities.horizon_4,
            self.descendant_utilities.horizon_8,
        ];
        ensure!(
            descendants.into_iter().flatten().all(f64::is_finite),
            "trajectory contains a non-finite descendant target"
        );
        if self.disposition == TrajectoryDispositionV2::Censored {
            ensure!(
                descendants.into_iter().all(|value| value.is_none()),
                "censored trajectory carries descendant targets"
            );
        }
        ensure!(
            !self.promotion_eligible && self.test_adaptive,
            "adaptive trajectory was marked as promotion evidence"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HumanSeedParityRowV1 {
    pub context_sha256: Sha256Digest,
    pub class_labels_identical: bool,
    pub maximum_probability_error: String,
    pub repeated_run_identical: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeHumanGenomeSeedV1 {
    pub schema: String,
    pub task: Task,
    pub method_name: String,
    pub implementation_sha256: Sha256Digest,
    pub binary_sha256: Sha256Digest,
    pub parameters: Value,
    pub preprocessing_contract_sha256: Sha256Digest,
    pub randomness_contract_sha256: Sha256Digest,
    pub prediction_semantics_sha256: Sha256Digest,
    pub probability_semantics_sha256: Sha256Digest,
    pub canonical_genome: String,
    pub semantic_sha256: Sha256Digest,
    pub parity_tolerance: String,
    pub parity_rows: Vec<HumanSeedParityRowV1>,
    pub parity_receipt_sha256: Sha256Digest,
    pub origin_class: OriginClassV2,
}

impl JopeHumanGenomeSeedV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == HUMAN_GENOME_SEED_V1_SCHEMA,
            "human genome seed schema is not v1"
        );
        ensure!(
            self.origin_class == OriginClassV2::HumanGpSeed
                && !self.method_name.trim().is_empty()
                && !self.canonical_genome.trim().is_empty()
                && self.semantic_sha256 == Sha256Digest::of_bytes(self.canonical_genome.as_bytes()),
            "human seed identity or ancestry is invalid"
        );
        let tolerance = self.parity_tolerance.parse::<f64>()?;
        ensure!(
            tolerance.is_finite() && tolerance >= 0.0 && !self.parity_rows.is_empty(),
            "human seed parity contract is empty or invalid"
        );
        for row in &self.parity_rows {
            let error = row.maximum_probability_error.parse::<f64>()?;
            ensure!(
                row.class_labels_identical
                    && row.repeated_run_identical
                    && error.is_finite()
                    && error <= tolerance,
                "human seed parity failed"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_mapping_bounds_regression_and_orients_timeseries() {
        assert_eq!(bounded_task_utility(Task::Regression, -9.0).unwrap(), -1.0);
        assert_eq!(bounded_task_utility(Task::Regression, 9.0).unwrap(), 1.0);
        assert_eq!(operational_loss(Task::Regression, -9.0).unwrap(), 2.0);
        assert_eq!(bounded_task_utility(Task::Timeseries, -0.25).unwrap(), 0.75);
        assert_eq!(operational_loss(Task::Timeseries, -0.25).unwrap(), 0.25);
        assert_eq!(
            task_metric_contract(Task::Multiclass).primary_metric,
            "macro_f1"
        );
    }

    #[test]
    fn typed_absence_cannot_be_numeric_zero() {
        TypedCount::na("no observations").validate().unwrap();
        TypedCount::value(0).validate().unwrap();
        assert!(TypedCount {
            value: Some(0),
            na_reason: Some("missing".to_owned()),
        }
        .validate()
        .is_err());
        assert!(TypedCount {
            value: None,
            na_reason: None,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn bundled_proposal_policy_matches_the_rust_contract() {
        let value = include_str!("../manifests/selfplay/proposal-selection-policy.v1.json");
        let policy: JopeProposalSelectionPolicyV1 = serde_json::from_str(value).unwrap();
        policy.validate().unwrap();
    }
}
