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

mod benchmark;
mod lift;
mod optional;

pub use benchmark::{
    BenchmarkDataset, BenchmarkLaneMetrics, EncoderBenchmarkPlanV2, EncoderBenchmarkReportV2,
};
pub use lift::{
    eight_slot_sampling_weights, EncoderLiftExperimentV2, EncoderLiftReportV2, LiftPanelDataset,
    LiftTaskResult, SamplingWeight,
};
pub use optional::{
    OptionalTaskEvidenceRow, OptionalTaskEvidenceV1, OptionalTaskReadinessV2, OptionalTaskRow,
};

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
