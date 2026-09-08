//! Task-aware admission preflight for evaluator manifests.
//!
//! The generated operator registry is the operator/task authority.  This
//! module deliberately does not carry a second handwritten operator list.

use crate::selfplay_data::canonical_json_sha256;
use crate::task::Task;
use anyhow::{ensure, Context, Result};
use feat_math::genome::{GenomeIr, PortfolioIr};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const TASK_PREFLIGHT_OUTCOME_SCHEMA: &str = "JopeTaskPreflightOutcome.v1";
pub const TASK_HEAD_POLICY_SCHEMA: &str = "JopeTaskHeadPolicy.v1";
pub const TASK_HEAD_POLICY_JSON: &str =
    include_str!("../manifests/epoch6/JopeTaskHeadPolicy.v1.json");
pub const TASK_HEAD_POLICY_SHA256: &str =
    "fa52320db524d316ef41d878a8c7dd2930cb1f8d989f05dda6a2c4e29a0927eb";

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

#[derive(Clone, Debug)]
pub struct TaskPreflightRequest<'a> {
    pub task: Task,
    pub canonical_genomes: &'a [String],
    pub feature_width: Option<u32>,
    pub mask_width: Option<u32>,
    pub observed_minimum_row_width: Option<u32>,
    pub required_minimum_row_width: Option<u32>,
    pub multiclass_metadata_complete: bool,
    pub release_a_only: bool,
    /// Registry cost units above this value require a separately bounded
    /// screening proof before admission. Zero disables the risk classifier.
    pub regression_screening_cost_threshold: u64,
    /// Present only after the evaluator completed within this wall-clock
    /// budget and every emitted candidate receipt was valid.
    pub bounded_screening_budget_seconds: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeneratedRegistryDocument {
    registry_version: String,
    registry_hash: String,
    maximum_output_columns: u32,
    operators: Vec<GeneratedOperator>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeneratedOperator {
    id: u32,
    name: String,
    role: String,
    #[serde(default)]
    target_side: bool,
    #[serde(default)]
    regression: bool,
    #[serde(default)]
    estimated_cost: u64,
    lifecycle: GeneratedOperatorLifecycle,
}

#[derive(Clone, Debug, Deserialize)]
struct GeneratedOperatorLifecycle {
    parseable: bool,
    #[serde(default)]
    quarantined: bool,
    #[serde(default)]
    deprecated: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedTaskHeadPolicyDocument {
    schema: String,
    heads: Vec<GeneratedTaskHeadPolicy>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedTaskHeadPolicy {
    operator: String,
    tasks: Vec<Task>,
    estimated_cost: u64,
}

#[derive(Clone, Debug)]
pub struct GeneratedOperatorRegistry {
    sha256: String,
    operators: BTreeMap<String, GeneratedOperator>,
    heads: BTreeMap<String, GeneratedTaskHeadPolicy>,
}

impl GeneratedOperatorRegistry {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let document: GeneratedRegistryDocument =
            serde_json::from_slice(bytes).context("decode generated operator registry")?;
        ensure!(
            !document.registry_version.trim().is_empty()
                && is_sha256(&document.registry_hash)
                && document.maximum_output_columns > 0
                && !document.operators.is_empty(),
            "generated operator registry header is invalid"
        );
        let mut ids = BTreeSet::new();
        let mut operators = BTreeMap::new();
        for operator in document.operators {
            ensure!(
                ids.insert(operator.id)
                    && !operator.name.trim().is_empty()
                    && !operator.role.trim().is_empty()
                    && operators.insert(operator.name.clone(), operator).is_none(),
                "generated operator registry repeats an operator identity"
            );
        }
        let head_document: GeneratedTaskHeadPolicyDocument =
            serde_json::from_str(TASK_HEAD_POLICY_JSON)
                .context("decode generated task/head policy")?;
        ensure!(
            hex::encode(Sha256::digest(TASK_HEAD_POLICY_JSON.as_bytes()))
                == TASK_HEAD_POLICY_SHA256,
            "generated task/head policy digest drift"
        );
        ensure!(
            head_document.schema == TASK_HEAD_POLICY_SCHEMA && !head_document.heads.is_empty(),
            "generated task/head policy header is invalid"
        );
        let mut heads = BTreeMap::new();
        for head in head_document.heads {
            ensure!(
                !head.operator.trim().is_empty()
                    && !head.tasks.is_empty()
                    && head.estimated_cost > 0
                    && head.tasks.iter().copied().collect::<BTreeSet<_>>().len()
                        == head.tasks.len()
                    && heads.insert(head.operator.clone(), head).is_none(),
                "generated task/head policy repeats or invalidates a head"
            );
        }
        for operator in operators
            .values()
            .filter(|operator| operator.role == "head")
        {
            ensure!(
                heads.contains_key(&operator.name),
                "generated operator head {} lacks task policy",
                operator.name
            );
        }
        Ok(Self {
            sha256: hex::encode(Sha256::digest(bytes)),
            operators,
            heads,
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    fn get(&self, name: &str) -> Option<&GeneratedOperator> {
        self.operators.get(name)
    }

    fn head(&self, name: &str) -> Option<&GeneratedTaskHeadPolicy> {
        self.heads.get(name)
    }
}

pub fn preflight_task_manifest(
    registry: &GeneratedOperatorRegistry,
    request: &TaskPreflightRequest<'_>,
) -> Result<TaskPreflightOutcomeV1> {
    ensure!(
        !request.canonical_genomes.is_empty(),
        "task preflight has no candidates"
    );
    let mut reasons = BTreeSet::new();
    if request.task == Task::Timeseries {
        reasons.insert(TaskPreflightReason::TimeseriesReleaseBlocked);
    }
    if request.release_a_only && request.task == Task::Multiclass {
        reasons.insert(TaskPreflightReason::MulticlassReleaseBlocked);
    }
    if request.task == Task::Multiclass && !request.multiclass_metadata_complete {
        reasons.insert(TaskPreflightReason::MulticlassMetadataMissing);
    }
    if matches!((request.feature_width, request.mask_width), (Some(feature), Some(mask)) if feature != mask)
    {
        reasons.insert(TaskPreflightReason::FeatureMaskWidthMismatch);
    }
    if matches!(
        (
            request.observed_minimum_row_width,
            request.required_minimum_row_width
        ),
        (Some(observed), Some(required)) if observed < required
    ) {
        reasons.insert(TaskPreflightReason::LegacyRowTooShort);
    }

    let mut predicted_cpu_seconds = 0_u64;
    let mut maximum_candidate_cost = 0_u64;
    for source in request.canonical_genomes {
        let genome = match GenomeIr::parse(source) {
            Ok(genome) => genome,
            Err(_) => {
                reasons.insert(TaskPreflightReason::GenomeMalformed);
                continue;
            }
        };
        if genome.canonical() != *source {
            reasons.insert(TaskPreflightReason::NoncanonicalGenome);
            continue;
        }
        let mut candidate_cost = 0_u64;
        match &genome {
            GenomeIr::Pipeline(pipeline) => preflight_stages(
                registry,
                request.task,
                &pipeline.stages,
                &mut candidate_cost,
                &mut reasons,
            ),
            GenomeIr::Portfolio(portfolio) => preflight_portfolio(
                registry,
                request.task,
                portfolio,
                &mut candidate_cost,
                &mut reasons,
            ),
            GenomeIr::Symbolic { .. } if request.task == Task::Regression => {}
            GenomeIr::Symbolic { .. } => {
                reasons.insert(TaskPreflightReason::UnsupportedGenomeShape);
            }
        }
        maximum_candidate_cost = maximum_candidate_cost.max(candidate_cost);
        predicted_cpu_seconds = predicted_cpu_seconds.saturating_add(candidate_cost.max(1));
    }
    predicted_cpu_seconds = predicted_cpu_seconds.max(request.canonical_genomes.len() as u64);

    let screening_required = request.task == Task::Regression
        && request.regression_screening_cost_threshold > 0
        && maximum_candidate_cost >= request.regression_screening_cost_threshold
        && request.bounded_screening_budget_seconds.is_none();
    if screening_required {
        reasons.insert(TaskPreflightReason::RegressionTimeoutScreeningRequired);
    }
    let permanent = reasons.iter().any(|reason| {
        !matches!(
            reason,
            TaskPreflightReason::RegressionTimeoutScreeningRequired
                | TaskPreflightReason::RegistryUnreadable
        )
    });
    let disposition = if permanent {
        TaskPreflightDisposition::PermanentRejection
    } else if screening_required {
        TaskPreflightDisposition::BoundedScreening
    } else {
        TaskPreflightDisposition::Accepted
    };
    let requested_work = TaskPreflightWorkV1 {
        manifest_count: 1,
        candidate_evaluation_count: request.canonical_genomes.len() as u64,
        predicted_cpu_seconds,
    };
    let estimated_compute_avoided = if disposition == TaskPreflightDisposition::PermanentRejection {
        requested_work.clone()
    } else {
        TaskPreflightWorkV1 {
            manifest_count: 0,
            candidate_evaluation_count: 0,
            predicted_cpu_seconds: 0,
        }
    };
    let outcome = TaskPreflightOutcomeV1 {
        schema: TASK_PREFLIGHT_OUTCOME_SCHEMA.to_owned(),
        task: request.task,
        disposition,
        reasons: reasons.into_iter().collect(),
        requested_work,
        estimated_compute_avoided,
        bounded_screening_budget_seconds: request.bounded_screening_budget_seconds,
        operator_registry_sha256: registry.sha256.clone(),
    };
    outcome.validate()?;
    Ok(outcome)
}

fn preflight_portfolio(
    registry: &GeneratedOperatorRegistry,
    task: Task,
    portfolio: &PortfolioIr,
    candidate_cost: &mut u64,
    reasons: &mut BTreeSet<TaskPreflightReason>,
) {
    if task != Task::Regression {
        reasons.insert(TaskPreflightReason::UnsupportedGenomeShape);
    }
    for member in &portfolio.members {
        preflight_stages(registry, task, &member.stages, candidate_cost, reasons);
    }
}

fn preflight_stages(
    registry: &GeneratedOperatorRegistry,
    task: Task,
    stages: &[feat_math::dsl::Stage],
    candidate_cost: &mut u64,
    reasons: &mut BTreeSet<TaskPreflightReason>,
) {
    if stages.is_empty() {
        reasons.insert(TaskPreflightReason::MissingTerminalHead);
        return;
    }
    let mut head_indexes = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        let operator = registry.get(&stage.op);
        let head = registry.head(&stage.op);
        if operator.is_none() && head.is_none() {
            reasons.insert(TaskPreflightReason::UnsupportedOperator);
            continue;
        }
        *candidate_cost = candidate_cost.saturating_add(
            head.map(|row| row.estimated_cost)
                .or_else(|| operator.map(|row| row.estimated_cost))
                .unwrap_or(1)
                .max(1),
        );
        if let Some(operator) = operator {
            if !operator.lifecycle.parseable
                || operator.lifecycle.quarantined
                || operator.lifecycle.deprecated
            {
                reasons.insert(TaskPreflightReason::QuarantinedOperator);
            }
        }
        if head.is_some() {
            head_indexes.push(index);
        }
        if operator.is_some_and(|operator| operator.target_side || operator.role == "target") {
            if index != 0 {
                reasons.insert(TaskPreflightReason::TargetTransformNotFirst);
            }
            if task != Task::Regression || !operator.is_some_and(|row| row.regression) {
                reasons.insert(TaskPreflightReason::TargetTransformUnsupportedForTask);
            }
        }
    }
    let terminal = stages.len() - 1;
    if head_indexes.is_empty() || head_indexes.last().copied() != Some(terminal) {
        reasons.insert(TaskPreflightReason::MissingTerminalHead);
    }
    if head_indexes.iter().any(|index| *index != terminal) {
        reasons.insert(TaskPreflightReason::NonterminalHead);
    }
    if !registry
        .head(&stages[terminal].op)
        .is_some_and(|head| head.tasks.contains(&task))
    {
        reasons.insert(TaskPreflightReason::TaskIncompatibleHead);
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> GeneratedOperatorRegistry {
        GeneratedOperatorRegistry::from_bytes(
            br#"{
              "registry_version":"generated-v1",
              "registry_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "maximum_output_columns":512,
              "operators":[
                {"id":0,"name":"target","role":"target","target_side":true,"regression":true,"estimated_cost":2,"lifecycle":{"parseable":true}},
                {"id":1,"name":"scale","role":"transform","regression":true,"binary_classification":true,"multiclass":true,"estimated_cost":2,"lifecycle":{"parseable":true}},
                {"id":2,"name":"ridge","role":"head","regression":true,"estimated_cost":3,"lifecycle":{"parseable":true}},
                {"id":3,"name":"logistic","role":"head","binary_classification":true,"estimated_cost":3,"lifecycle":{"parseable":true}},
                {"id":4,"name":"multiclass_ridge","role":"head","multiclass":true,"estimated_cost":3,"lifecycle":{"parseable":true}}
              ]
            }"#,
        )
        .unwrap()
    }

    fn request<'a>(task: Task, genomes: &'a [String]) -> TaskPreflightRequest<'a> {
        TaskPreflightRequest {
            task,
            canonical_genomes: genomes,
            feature_width: Some(8),
            mask_width: Some(8),
            observed_minimum_row_width: Some(9),
            required_minimum_row_width: Some(9),
            multiclass_metadata_complete: true,
            release_a_only: true,
            regression_screening_cost_threshold: 100,
            bounded_screening_budget_seconds: None,
        }
    }

    #[test]
    fn accepts_task_compatible_registry_pipeline() {
        let genomes = vec!["(pipe X (scale) (logistic))".to_owned()];
        let outcome =
            preflight_task_manifest(&registry(), &request(Task::Binary, &genomes)).unwrap();
        assert_eq!(outcome.disposition, TaskPreflightDisposition::Accepted);
        assert!(outcome.reasons.is_empty());
    }

    #[test]
    fn rejects_known_deterministic_failure_classes_with_avoided_work() {
        let genomes = vec!["(pipe X (logistic) (scale) (ridge))".to_owned()];
        let mut request = request(Task::Binary, &genomes);
        request.feature_width = Some(8);
        request.mask_width = Some(7);
        let outcome = preflight_task_manifest(&registry(), &request).unwrap();
        assert_eq!(
            outcome.disposition,
            TaskPreflightDisposition::PermanentRejection
        );
        assert!(outcome
            .reasons
            .contains(&TaskPreflightReason::FeatureMaskWidthMismatch));
        assert!(outcome
            .reasons
            .contains(&TaskPreflightReason::NonterminalHead));
        assert_eq!(outcome.estimated_compute_avoided, outcome.requested_work);
    }

    #[test]
    fn release_a_blocks_optional_tasks_and_regression_screening_is_typed() {
        let multiclass = vec!["(pipe X (multiclass_ridge))".to_owned()];
        let outcome =
            preflight_task_manifest(&registry(), &request(Task::Multiclass, &multiclass)).unwrap();
        assert!(outcome
            .reasons
            .contains(&TaskPreflightReason::MulticlassReleaseBlocked));

        let regression = vec!["(pipe X (target kind=log1p) (scale) (ridge))".to_owned()];
        let mut request = request(Task::Regression, &regression);
        request.regression_screening_cost_threshold = 1;
        let outcome = preflight_task_manifest(&registry(), &request).unwrap();
        assert_eq!(
            outcome.disposition,
            TaskPreflightDisposition::BoundedScreening
        );
        assert_eq!(
            outcome.reasons,
            [TaskPreflightReason::RegressionTimeoutScreeningRequired]
        );
        request.bounded_screening_budget_seconds = Some(30);
        let accepted = preflight_task_manifest(&registry(), &request).unwrap();
        assert_eq!(accepted.disposition, TaskPreflightDisposition::Accepted);
        assert_eq!(accepted.bounded_screening_budget_seconds, Some(30));

        let failed = TaskPreflightOutcomeV1::failed_regression_screening(&outcome, 30).unwrap();
        assert_eq!(
            failed.reasons,
            [TaskPreflightReason::RegressionTimeoutScreeningFailed]
        );
        assert_eq!(failed.bounded_screening_budget_seconds, Some(30));
    }

    #[test]
    fn target_transform_order_and_timeseries_are_fail_closed() {
        let genomes = vec!["(pipe X (scale) (target kind=log1p) (ridge))".to_owned()];
        let outcome =
            preflight_task_manifest(&registry(), &request(Task::Regression, &genomes)).unwrap();
        assert!(outcome
            .reasons
            .contains(&TaskPreflightReason::TargetTransformNotFirst));

        let genomes = vec!["(pipe X (scale) (ridge))".to_owned()];
        let outcome =
            preflight_task_manifest(&registry(), &request(Task::Timeseries, &genomes)).unwrap();
        assert!(outcome
            .reasons
            .contains(&TaskPreflightReason::TimeseriesReleaseBlocked));
    }
}
