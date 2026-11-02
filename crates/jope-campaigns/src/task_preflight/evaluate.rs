use super::*;
use crate::task::Task;
use anyhow::{ensure, Result};
use feat_math::genome::{GenomeIr, PortfolioIr};
use std::collections::BTreeSet;

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
