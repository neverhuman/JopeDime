//! Preregistered benchmark, optional-task, and lift authorities for Encoder V2.

use crate::selfplay_data::canonical_json_sha256;
use crate::task::Task;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BENCHMARK_PLAN_SCHEMA: &str = "JopeEncoderBenchmarkPlan.v2";
pub const BENCHMARK_REPORT_SCHEMA: &str = "JopeEncoderBenchmarkReport.v2";
pub const OPTIONAL_READINESS_SCHEMA: &str = "JopeOptionalTaskReadiness.v2";
pub const OPTIONAL_EVIDENCE_SCHEMA: &str = "JopeOptionalTaskEvidence.v1";
pub const LIFT_EXPERIMENT_SCHEMA: &str = "JopeEncoderLiftExperiment.v2";
pub const LIFT_REPORT_SCHEMA: &str = "JopeEncoderLiftReport.v2";
pub const COHORT_SEED: u64 = 20_260_806;

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn validate_status(generated: u64, budget: u64, inputs: &[String]) -> Result<()> {
    ensure!(generated > 0, "status generation time is absent");
    ensure!(budget > 0, "status freshness budget is zero");
    ensure!(!inputs.is_empty(), "status input digests are absent");
    ensure!(
        inputs.windows(2).all(|pair| pair[0] < pair[1]),
        "status input digests are not unique and sorted"
    );
    for digest in inputs {
        require_sha256(digest, "status input digest")?;
    }
    Ok(())
}

fn valid_task(task: &str) -> bool {
    Task::parse(task).is_some()
}

fn task_rank(task: &str) -> Option<u8> {
    let task = Task::parse(task)?;
    Task::CANONICAL_ORDER
        .iter()
        .position(|candidate| *candidate == task)
        .map(|index| index as u8)
}

fn benchmark_dataset_precedes(left: &BenchmarkDataset, right: &BenchmarkDataset) -> bool {
    (
        task_rank(&left.task),
        &left.source_family,
        left.row_log2_bin,
        left.column_log2_bin,
        left.byte_size_log2_bin,
        &left.train_sha256,
        &left.input_identity_sha256,
    ) < (
        task_rank(&right.task),
        &right.source_family,
        right.row_log2_bin,
        right.column_log2_bin,
        right.byte_size_log2_bin,
        &right.train_sha256,
        &right.input_identity_sha256,
    )
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDataset {
    pub task: String,
    pub source_family: String,
    pub row_log2_bin: u8,
    pub column_log2_bin: u8,
    pub byte_size_log2_bin: u8,
    pub input_identity_sha256: String,
    pub train_sha256: String,
}

impl BenchmarkDataset {
    fn validate(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "benchmark dataset task is invalid");
        ensure!(
            !self.source_family.is_empty(),
            "benchmark source family is absent"
        );
        require_sha256(&self.input_identity_sha256, "benchmark input identity")?;
        require_sha256(&self.train_sha256, "benchmark train SHA-256")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderBenchmarkPlanV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub cohort_seed: u64,
    pub release_cohort: Vec<BenchmarkDataset>,
    pub tuning_cohort: Vec<BenchmarkDataset>,
    pub baseline_capsule_sha256: String,
    pub candidate_encoder_identity_sha256: String,
    pub hardware_capability_receipt_sha256: String,
    pub cache_reset_protocol_sha256: String,
    pub proof_chunk_size: u32,
    pub proof_datasets_per_task: u32,
    pub replay_chunk_size: u32,
    pub replay_datasets_per_task: u32,
    pub backlog_chunk_max: u32,
    pub untimed_warmups: u32,
    pub timed_repetitions: u32,
    pub cold_and_warm_lanes: bool,
    pub blockers: Vec<String>,
}

impl EncoderBenchmarkPlanV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == BENCHMARK_PLAN_SCHEMA,
            "invalid benchmark-plan schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        ensure!(
            self.cohort_seed == COHORT_SEED,
            "benchmark cohort seed drift"
        );
        for (value, field) in [
            (&self.baseline_capsule_sha256, "baseline capsule"),
            (
                &self.candidate_encoder_identity_sha256,
                "candidate encoder identity",
            ),
            (
                &self.hardware_capability_receipt_sha256,
                "hardware capability receipt",
            ),
            (&self.cache_reset_protocol_sha256, "cache reset protocol"),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(
            self.release_cohort.len() == 128,
            "release cohort is not 128 inputs"
        );
        ensure!(
            self.tuning_cohort.len() == 58,
            "tuning cohort is not 58 inputs"
        );
        let mut identities = BTreeSet::new();
        for dataset in self.release_cohort.iter().chain(self.tuning_cohort.iter()) {
            dataset.validate()?;
            ensure!(
                identities.insert(dataset.input_identity_sha256.as_str()),
                "benchmark tuning/release cohorts overlap"
            );
        }
        for task in ["regression", "binary", "multiclass", "timeseries"] {
            ensure!(
                self.release_cohort
                    .iter()
                    .filter(|row| row.task == task)
                    .count()
                    == 32,
                "release cohort does not contain 32 {task} inputs"
            );
        }
        for (task, count) in [
            ("regression", 16usize),
            ("binary", 16),
            ("multiclass", 16),
            ("timeseries", 10),
        ] {
            ensure!(
                self.tuning_cohort
                    .iter()
                    .filter(|row| row.task == task)
                    .count()
                    == count,
                "tuning cohort has the wrong {task} count"
            );
        }
        ensure!(
            self.release_cohort
                .windows(2)
                .all(|pair| benchmark_dataset_precedes(&pair[0], &pair[1]))
                && self
                    .tuning_cohort
                    .windows(2)
                    .all(|pair| benchmark_dataset_precedes(&pair[0], &pair[1])),
            "benchmark cohorts are not in frozen stratification order"
        );
        ensure!(
            self.proof_chunk_size == 32
                && self.proof_datasets_per_task == 8
                && self.replay_chunk_size == 128
                && self.replay_datasets_per_task == 32
                && self.backlog_chunk_max == 256,
            "benchmark chunk ladder drift"
        );
        ensure!(
            self.untimed_warmups == 1 && self.timed_repetitions >= 3,
            "benchmark repetitions are insufficient"
        );
        ensure!(
            self.cold_and_warm_lanes,
            "benchmark omits cold or warm lane"
        );
        ensure!(self.blockers.is_empty(), "benchmark plan has blockers");
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLaneMetrics {
    pub samples: u64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_per_gpu_slot: f64,
    pub cpu_hours: f64,
    pub load_time_ms: f64,
    pub preprocess_time_ms: f64,
    pub kernel_time_ms: f64,
    pub verification_publication_ms: f64,
    pub cache_lookup_ms: f64,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub failures: u64,
}

impl BenchmarkLaneMetrics {
    fn validate(&self) -> Result<()> {
        ensure!(self.samples > 0, "benchmark lane has no samples");
        for value in [
            self.p50_latency_ms,
            self.p90_latency_ms,
            self.p95_latency_ms,
            self.p99_latency_ms,
            self.throughput_per_gpu_slot,
            self.cpu_hours,
            self.load_time_ms,
            self.preprocess_time_ms,
            self.kernel_time_ms,
            self.verification_publication_ms,
            self.cache_lookup_ms,
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "benchmark metric is invalid"
            );
        }
        ensure!(
            self.p50_latency_ms <= self.p90_latency_ms
                && self.p90_latency_ms <= self.p95_latency_ms
                && self.p95_latency_ms <= self.p99_latency_ms,
            "benchmark latency percentiles are not monotonic"
        );
        ensure!(
            self.throughput_per_gpu_slot > 0.0,
            "benchmark throughput is zero"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderBenchmarkReportV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub benchmark_plan_sha256: String,
    pub baseline: BenchmarkLaneMetrics,
    pub candidate: BenchmarkLaneMetrics,
    pub cache_hit: BenchmarkLaneMetrics,
    pub paired_bootstrap_throughput_ratio_lower_95: f64,
    pub control_plane_overhead_fraction: f64,
    pub output_drift_count: u64,
    pub fallback_count: u64,
    pub backend_mismatch_count: u64,
    pub collision_count: u64,
    pub oom_as_success_count: u64,
    pub useful_work_green: bool,
    pub protected_reserves_green: bool,
    pub companion_cpu_credits_green: bool,
    pub release_speed_gate_passed: bool,
    pub blockers: Vec<String>,
}

impl EncoderBenchmarkReportV2 {
    pub fn computed_release_gate(&self) -> Result<bool> {
        self.baseline.validate()?;
        self.candidate.validate()?;
        self.cache_hit.validate()?;
        ensure!(
            self.paired_bootstrap_throughput_ratio_lower_95.is_finite(),
            "benchmark bootstrap bound is invalid"
        );
        ensure!(
            self.control_plane_overhead_fraction.is_finite()
                && self.control_plane_overhead_fraction >= 0.0,
            "control-plane overhead is invalid"
        );
        let throughput_ratio =
            self.candidate.throughput_per_gpu_slot / self.baseline.throughput_per_gpu_slot;
        let cache_budget = 250.0_f64.min(self.candidate.p95_latency_ms / 20.0);
        Ok(throughput_ratio >= 3.0
            && self.paired_bootstrap_throughput_ratio_lower_95 >= 2.5
            && self.candidate.p95_latency_ms <= self.baseline.p95_latency_ms * 0.5
            && self.cache_hit.p95_latency_ms <= cache_budget
            && self.control_plane_overhead_fraction < 0.05
            && self.output_drift_count == 0
            && self.fallback_count == 0
            && self.backend_mismatch_count == 0
            && self.collision_count == 0
            && self.oom_as_success_count == 0
            && self.useful_work_green
            && self.protected_reserves_green
            && self.companion_cpu_credits_green)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == BENCHMARK_REPORT_SCHEMA,
            "invalid benchmark-report schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.benchmark_plan_sha256, "benchmark plan SHA-256")?;
        let computed = self.computed_release_gate()?;
        ensure!(
            self.release_speed_gate_passed == computed,
            "benchmark release-gate claim does not match measurements"
        );
        if computed {
            ensure!(self.blockers.is_empty(), "passing benchmark has blockers");
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "failed benchmark lacks explicit blockers"
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskRow {
    pub task: String,
    pub state: String,
    pub distinct_dataset_identities: u64,
    pub minimum_required: u64,
    pub minimum_rolling_origins_per_dataset: Option<u32>,
    pub valid_v2_cache_bindings: u64,
    pub valid_context_transform_bindings: u64,
    pub evaluator_retained: bool,
    pub evaluator_sha256: String,
    pub task_valid_head_count: u64,
    pub robust_bank_sha256: String,
    pub fresh_unbiased_shadow_evidence: bool,
    pub causal_reduction_verified: Option<bool>,
    pub no_future_leakage: Option<bool>,
    pub primary_metric: String,
    pub native_error_metric: String,
    pub training_allowed: bool,
    pub blockers: Vec<String>,
}

impl OptionalTaskRow {
    pub fn computed_training_allowed(&self) -> Result<bool> {
        ensure!(
            matches!(
                self.state.as_str(),
                "absent" | "collecting" | "replay_pending" | "shadow_ready" | "blocked"
            ),
            "optional-task state is invalid"
        );
        require_sha256(&self.evaluator_sha256, "optional evaluator SHA-256")?;
        require_sha256(&self.robust_bank_sha256, "optional robust-bank SHA-256")?;
        let common = self.state == "shadow_ready"
            && self.distinct_dataset_identities >= self.minimum_required
            && self.valid_v2_cache_bindings >= self.distinct_dataset_identities
            && self.valid_context_transform_bindings >= self.distinct_dataset_identities
            && self.evaluator_retained
            && self.task_valid_head_count > 0
            && self.fresh_unbiased_shadow_evidence
            && self.blockers.is_empty();
        match self.task.as_str() {
            "multiclass" => Ok(common
                && self.minimum_required == 500
                && self.minimum_rolling_origins_per_dataset.is_none()
                && self.causal_reduction_verified.is_none()
                && self.no_future_leakage.is_none()
                && self.primary_metric == "macro_f1"
                && self.native_error_metric == "one_minus_macro_f1"),
            "timeseries" => Ok(common
                && self.distinct_dataset_identities == 42
                && self.minimum_required == 42
                && self
                    .minimum_rolling_origins_per_dataset
                    .is_some_and(|origins| origins >= 3)
                && self.causal_reduction_verified == Some(true)
                && self.no_future_leakage == Some(true)
                && self.primary_metric == "negative_mase"
                && self.native_error_metric == "mase"),
            _ => Err(anyhow::anyhow!("readiness row is not an optional task")),
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.training_allowed == self.computed_training_allowed()?,
            "optional-task training claim does not match its evidence"
        );
        if self.training_allowed {
            ensure!(
                self.blockers.is_empty(),
                "trainable optional task has blockers"
            );
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "unready optional task lacks typed blockers"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskReadinessV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub tasks: Vec<OptionalTaskRow>,
}

impl OptionalTaskReadinessV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OPTIONAL_READINESS_SCHEMA,
            "invalid optional readiness schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        ensure!(
            self.tasks.len() == 2,
            "optional readiness must have exactly two rows"
        );
        ensure!(
            self.tasks[0].task == "multiclass" && self.tasks[1].task == "timeseries",
            "optional readiness rows are missing or out of order"
        );
        for row in &self.tasks {
            row.validate()?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskEvidenceRow {
    pub task: String,
    pub cache_coverage_sha256: Option<String>,
    pub context_identity_sha256: Option<String>,
    pub evaluator_retention_sha256: Option<String>,
    pub comparator_lock_sha256: Option<String>,
    pub unbiased_shadow_sha256: Option<String>,
    pub typed_absence_reason: Option<String>,
}

impl OptionalTaskEvidenceRow {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.task.as_str(), "multiclass" | "timeseries"),
            "invalid task"
        );
        let present = [
            &self.cache_coverage_sha256,
            &self.context_identity_sha256,
            &self.evaluator_retention_sha256,
            &self.comparator_lock_sha256,
            &self.unbiased_shadow_sha256,
        ];
        for digest in present.iter().filter_map(|value| value.as_deref()) {
            require_sha256(digest, "optional-task evidence digest")?;
        }
        if present.iter().all(|value| value.is_some()) {
            ensure!(
                self.typed_absence_reason.is_none(),
                "complete evidence claims absence"
            );
        } else {
            ensure!(
                self.typed_absence_reason
                    .as_deref()
                    .is_some_and(|reason| !reason.is_empty()),
                "incomplete optional evidence lacks typed absence"
            );
        }
        Ok(())
    }
}

fn lift_panel_dataset_precedes(left: &LiftPanelDataset, right: &LiftPanelDataset) -> bool {
    (
        task_rank(&left.task),
        &left.source_family,
        &left.input_identity_sha256,
        &left.test_sha256,
        left.outer_fold,
    ) < (
        task_rank(&right.task),
        &right.source_family,
        &right.input_identity_sha256,
        &right.test_sha256,
        right.outer_fold,
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskEvidenceV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub readiness_sha256: String,
    pub tasks: Vec<OptionalTaskEvidenceRow>,
}

impl OptionalTaskEvidenceV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OPTIONAL_EVIDENCE_SCHEMA,
            "invalid optional evidence schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.readiness_sha256, "optional readiness SHA-256")?;
        ensure!(
            self.tasks.len() == 2
                && self.tasks[0].task == "multiclass"
                && self.tasks[1].task == "timeseries",
            "optional evidence rows are incomplete or out of order"
        );
        for row in &self.tasks {
            row.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingWeight {
    pub numerator: u8,
    pub denominator: u8,
}

pub fn eight_slot_sampling_weights(
    multiclass_ready: bool,
    timeseries_ready: bool,
) -> BTreeMap<&'static str, SamplingWeight> {
    let missing = u8::from(!multiclass_ready) + u8::from(!timeseries_ready);
    let denominator = 16;
    let core = 6 + missing;
    BTreeMap::from([
        (
            "regression",
            SamplingWeight {
                numerator: core,
                denominator,
            },
        ),
        (
            "binary",
            SamplingWeight {
                numerator: core,
                denominator,
            },
        ),
        (
            "multiclass",
            SamplingWeight {
                numerator: if multiclass_ready { 2 } else { 0 },
                denominator,
            },
        ),
        (
            "timeseries",
            SamplingWeight {
                numerator: if timeseries_ready { 2 } else { 0 },
                denominator,
            },
        ),
    ])
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiftPanelDataset {
    pub task: String,
    pub source_family: String,
    pub input_identity_sha256: String,
    pub test_sha256: String,
    pub outer_fold: u8,
}

impl LiftPanelDataset {
    fn validate(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "lift-panel task is invalid");
        ensure!(
            !self.source_family.is_empty(),
            "lift-panel family is absent"
        );
        require_sha256(&self.input_identity_sha256, "lift input identity")?;
        require_sha256(&self.test_sha256, "lift test SHA-256")?;
        ensure!(self.outer_fold < 5, "lift outer fold is invalid");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderLiftExperimentV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub v1_capsule_sha256: String,
    pub v2_encoder_identity_sha256: String,
    pub panel: Vec<LiftPanelDataset>,
    pub common_contexts_sha256: String,
    pub common_candidates_sha256: String,
    pub common_seeds_sha256: String,
    pub common_evaluators_sha256: String,
    pub failure_policy_sha256: String,
    pub standardization_fit_policy: String,
    pub eight_slot_compute: bool,
    pub family_cluster_bootstrap_replicates: u32,
    pub multiplicity_correction: String,
    pub bounded_metric_noninferiority_margin: String,
    pub native_error_noninferiority_ratio: String,
    pub blockers: Vec<String>,
}

impl EncoderLiftExperimentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LIFT_EXPERIMENT_SCHEMA,
            "invalid lift experiment schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        for (value, field) in [
            (&self.v1_capsule_sha256, "V1 capsule"),
            (&self.v2_encoder_identity_sha256, "V2 encoder identity"),
            (&self.common_contexts_sha256, "common contexts"),
            (&self.common_candidates_sha256, "common candidates"),
            (&self.common_seeds_sha256, "common seeds"),
            (&self.common_evaluators_sha256, "common evaluators"),
            (&self.failure_policy_sha256, "failure policy"),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(self.panel.len() == 426, "lift panel is not 426 datasets");
        let mut identities = BTreeSet::new();
        let mut family_folds = BTreeMap::new();
        for dataset in &self.panel {
            dataset.validate()?;
            ensure!(
                identities.insert(dataset.input_identity_sha256.as_str()),
                "lift panel repeats an input identity"
            );
            if let Some(existing) = family_folds.insert(&dataset.source_family, dataset.outer_fold)
            {
                ensure!(
                    existing == dataset.outer_fold,
                    "source family appears in more than one held-out fold"
                );
            }
        }
        for (task, count) in [
            ("regression", 128usize),
            ("binary", 128),
            ("multiclass", 128),
            ("timeseries", 42),
        ] {
            ensure!(
                self.panel.iter().filter(|row| row.task == task).count() == count,
                "lift panel has the wrong {task} count"
            );
        }
        let folds = self
            .panel
            .iter()
            .map(|row| row.outer_fold)
            .collect::<BTreeSet<_>>();
        ensure!(
            folds == BTreeSet::from([0, 1, 2, 3, 4]),
            "lift panel omits a fold"
        );
        ensure!(
            self.panel
                .windows(2)
                .all(|pair| lift_panel_dataset_precedes(&pair[0], &pair[1])),
            "lift panel is not in frozen order"
        );
        ensure!(
            self.standardization_fit_policy == "outer_fold_training_families_only",
            "lift standardization policy permits held-out leakage"
        );
        ensure!(
            self.eight_slot_compute,
            "lift compute is not matched at eight slots"
        );
        ensure!(
            self.family_cluster_bootstrap_replicates == 10_000,
            "lift bootstrap replicate count drift"
        );
        ensure!(
            self.multiplicity_correction == "holm",
            "lift correction is not Holm"
        );
        ensure!(
            self.bounded_metric_noninferiority_margin == "-0.005"
                && self.native_error_noninferiority_ratio == "1.01",
            "lift noninferiority authority drift"
        );
        ensure!(self.blockers.is_empty(), "lift experiment has blockers");
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiftTaskResult {
    pub task: String,
    pub corrected_primary_delta_lower_95: Option<f64>,
    pub native_error_ratio_upper_95: Option<f64>,
    pub strict_win_rate_lower_95: f64,
    pub p95_latency_ratio: f64,
    pub failure_rate_delta: f64,
    pub bounded_metric_worst_delta: Option<f64>,
    pub native_error_worst_ratio: Option<f64>,
    pub directional_agreement: bool,
    pub typed_absence: Option<String>,
}

impl LiftTaskResult {
    fn validate_numbers(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "lift result task is invalid");
        for value in [
            Some(self.strict_win_rate_lower_95),
            Some(self.p95_latency_ratio),
            Some(self.failure_rate_delta),
            self.corrected_primary_delta_lower_95,
            self.native_error_ratio_upper_95,
            self.bounded_metric_worst_delta,
            self.native_error_worst_ratio,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(value.is_finite(), "lift task metric is nonfinite");
        }
        Ok(())
    }

    fn noninferior(&self) -> Result<bool> {
        self.validate_numbers()?;
        match self.task.as_str() {
            "binary" | "multiclass" => Ok(self
                .corrected_primary_delta_lower_95
                .is_some_and(|lower| lower > -0.005)),
            "regression" | "timeseries" => Ok(self
                .native_error_ratio_upper_95
                .is_some_and(|upper| upper < 1.01)),
            _ => unreachable!(),
        }
    }

    fn catastrophic(&self) -> bool {
        self.bounded_metric_worst_delta
            .is_some_and(|delta| delta < -0.02)
            || self
                .native_error_worst_ratio
                .is_some_and(|ratio| ratio > 1.05)
    }

    fn positive_quality(&self) -> bool {
        self.corrected_primary_delta_lower_95
            .is_some_and(|lower| lower > 0.005)
            || self
                .native_error_ratio_upper_95
                .is_some_and(|ratio| ratio <= 0.99)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderLiftReportV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub experiment_sha256: String,
    pub benchmark_report_sha256: String,
    pub speed_gate_passed: bool,
    pub tasks: Vec<LiftTaskResult>,
    pub leakage_detected: bool,
    pub identity_collision_detected: bool,
    pub reproducible: bool,
    pub advancement_allowed: bool,
    pub blockers: Vec<String>,
}

impl EncoderLiftReportV2 {
    pub fn computed_advancement(&self) -> Result<bool> {
        ensure!(
            self.tasks.len() == 4
                && self.tasks[0].task == "regression"
                && self.tasks[1].task == "binary"
                && self.tasks[2].task == "multiclass"
                && self.tasks[3].task == "timeseries",
            "lift report does not contain four ordered task rows"
        );
        for task in &self.tasks {
            task.validate_numbers()?;
        }
        let all_noninferior = self
            .tasks
            .iter()
            .map(LiftTaskResult::noninferior)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .all(|value| value);
        let positive_quality = self.tasks.iter().any(LiftTaskResult::positive_quality);
        let common = self.tasks.iter().all(|task| {
            task.strict_win_rate_lower_95 > 0.5
                && task.p95_latency_ratio <= 1.25
                && task.failure_rate_delta <= 0.0025
                && task.directional_agreement
                && task.typed_absence.is_none()
                && !task.catastrophic()
        });
        Ok(all_noninferior
            && (positive_quality || self.speed_gate_passed)
            && common
            && !self.leakage_detected
            && !self.identity_collision_detected
            && self.reproducible)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LIFT_REPORT_SCHEMA,
            "invalid lift-report schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.experiment_sha256, "lift experiment SHA-256")?;
        require_sha256(&self.benchmark_report_sha256, "benchmark report SHA-256")?;
        let computed = self.computed_advancement()?;
        ensure!(
            self.advancement_allowed == computed,
            "lift advancement claim does not match preregistered gates"
        );
        if computed {
            ensure!(
                self.blockers.is_empty(),
                "advancing lift report has blockers"
            );
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "blocked lift report lacks explicit blockers"
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

pub fn require_plan_report_binding(
    plan: &EncoderBenchmarkPlanV2,
    report: &EncoderBenchmarkReportV2,
) -> Result<()> {
    ensure!(
        report.benchmark_plan_sha256 == plan.digest()?,
        "benchmark report does not bind the frozen plan"
    );
    report.validate()
}

pub fn require_experiment_report_binding(
    experiment: &EncoderLiftExperimentV2,
    report: &EncoderLiftReportV2,
) -> Result<()> {
    ensure!(
        report.experiment_sha256 == experiment.digest()?,
        "lift report does not bind the preregistered experiment"
    );
    report.validate().context("validate lift report")
}
