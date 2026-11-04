use crate::selfplay_data::canonical_json_sha256;
use crate::task::Task;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const TASK_PREFLIGHT_OUTCOME_SCHEMA: &str = "JopeTaskPreflightOutcome.v1";
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPreflightDisposition {
    Accepted,
    PermanentRejection,
    BoundedScreening,
    RetryableFailure,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPreflightReason {
    TimeseriesReleaseBlocked,
    MulticlassReleaseBlocked,
    MulticlassMetadataMissing,
    FeatureMaskWidthMismatch,
    LegacyRowTooShort,
    GenomeMalformed,
    NoncanonicalGenome,
    UnsupportedGenomeShape,
    UnsupportedOperator,
    QuarantinedOperator,
    TargetTransformNotFirst,
    TargetTransformUnsupportedForTask,
    MissingTerminalHead,
    NonterminalHead,
    TaskIncompatibleHead,
    RegressionTimeoutScreeningRequired,
    RegressionTimeoutScreeningFailed,
    RegistryUnreadable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPreflightWorkV1 {
    pub manifest_count: u64,
    pub candidate_evaluation_count: u64,
    pub predicted_cpu_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPreflightOutcomeV1 {
    pub schema: String,
    pub task: Task,
    pub disposition: TaskPreflightDisposition,
    pub reasons: Vec<TaskPreflightReason>,
    pub requested_work: TaskPreflightWorkV1,
    pub estimated_compute_avoided: TaskPreflightWorkV1,
    pub bounded_screening_budget_seconds: Option<u64>,
    pub operator_registry_sha256: String,
}

impl TaskPreflightOutcomeV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_PREFLIGHT_OUTCOME_SCHEMA,
            "task-preflight outcome schema drift"
        );
        ensure!(
            is_sha256(&self.operator_registry_sha256),
            "task-preflight registry digest is invalid"
        );
        ensure!(
            self.requested_work.manifest_count == 1
                && self.requested_work.candidate_evaluation_count > 0
                && self.requested_work.predicted_cpu_seconds > 0,
            "task-preflight requested work is invalid"
        );
        ensure!(
            self.reasons.windows(2).all(|pair| pair[0] < pair[1]),
            "task-preflight reasons are duplicated or unsorted"
        );
        ensure!(
            self.bounded_screening_budget_seconds
                .is_none_or(|seconds| seconds > 0 && self.task == Task::Regression),
            "bounded screening proof is invalid"
        );
        match self.disposition {
            TaskPreflightDisposition::Accepted => {
                ensure!(self.reasons.is_empty(), "accepted preflight has reasons");
                ensure!(
                    self.estimated_compute_avoided.candidate_evaluation_count == 0
                        && self.estimated_compute_avoided.predicted_cpu_seconds == 0,
                    "accepted preflight claims avoided compute"
                );
            }
            TaskPreflightDisposition::PermanentRejection => {
                ensure!(
                    !self.reasons.is_empty(),
                    "permanent preflight rejection lacks reasons"
                );
                ensure!(
                    self.estimated_compute_avoided == self.requested_work,
                    "permanent preflight rejection does not account for all avoided work"
                );
            }
            TaskPreflightDisposition::BoundedScreening
            | TaskPreflightDisposition::RetryableFailure => {
                ensure!(
                    !self.reasons.is_empty(),
                    "nonterminal preflight outcome lacks reasons"
                );
                ensure!(
                    self.estimated_compute_avoided.candidate_evaluation_count == 0
                        && self.estimated_compute_avoided.predicted_cpu_seconds == 0,
                    "nonterminal preflight outcome claims avoided compute"
                );
                ensure!(
                    self.bounded_screening_budget_seconds.is_none(),
                    "unfinished preflight claims a bounded screening proof"
                );
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }

    pub fn retryable_registry_failure(
        task: Task,
        candidate_evaluation_count: u64,
        predicted_cpu_seconds: u64,
        operator_registry_sha256: String,
    ) -> Result<Self> {
        let value = Self {
            schema: TASK_PREFLIGHT_OUTCOME_SCHEMA.to_owned(),
            task,
            disposition: TaskPreflightDisposition::RetryableFailure,
            reasons: vec![TaskPreflightReason::RegistryUnreadable],
            requested_work: TaskPreflightWorkV1 {
                manifest_count: 1,
                candidate_evaluation_count,
                predicted_cpu_seconds,
            },
            estimated_compute_avoided: TaskPreflightWorkV1 {
                manifest_count: 0,
                candidate_evaluation_count: 0,
                predicted_cpu_seconds: 0,
            },
            bounded_screening_budget_seconds: None,
            operator_registry_sha256,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn failed_regression_screening(
        accepted_or_screening: &Self,
        budget_seconds: u64,
    ) -> Result<Self> {
        ensure!(
            accepted_or_screening.task == Task::Regression && budget_seconds > 0,
            "bounded screening failure is not regression"
        );
        let mut reasons = accepted_or_screening.reasons.clone();
        reasons.retain(|reason| *reason != TaskPreflightReason::RegressionTimeoutScreeningRequired);
        reasons.push(TaskPreflightReason::RegressionTimeoutScreeningFailed);
        reasons.sort();
        reasons.dedup();
        let value = Self {
            schema: TASK_PREFLIGHT_OUTCOME_SCHEMA.to_owned(),
            task: accepted_or_screening.task,
            disposition: TaskPreflightDisposition::PermanentRejection,
            reasons,
            requested_work: accepted_or_screening.requested_work.clone(),
            estimated_compute_avoided: accepted_or_screening.requested_work.clone(),
            bounded_screening_budget_seconds: Some(budget_seconds),
            operator_registry_sha256: accepted_or_screening.operator_registry_sha256.clone(),
        };
        value.validate()?;
        Ok(value)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
