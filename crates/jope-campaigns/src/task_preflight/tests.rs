use super::*;
use crate::task::Task;

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
    let outcome = preflight_task_manifest(&registry(), &request(Task::Binary, &genomes)).unwrap();
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
