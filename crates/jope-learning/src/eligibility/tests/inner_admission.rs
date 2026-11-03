use super::*;
pub(super) fn inner_fixture(
    f: &Fixture,
) -> (
    FreezePlan,
    InnerPartition,
    FreezeReceipt,
    jope_pipeline::search_plan::SelectionPlan,
) {
    let partition = derive_inner_partition(
        &f.state,
        &f.plan.panel_sha256,
        "fixture",
        0,
        7,
        &"b".repeat(64),
        &"a".repeat(64),
    )
    .unwrap();
    let mut plan = f.plan.clone();
    plan.comparison.cut_id = partition.identity().unwrap();
    let freeze = freeze_inner_optimization(&f.state, &plan, &partition).unwrap();
    let old: FoldPredictions = blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    let mut outer: EvaluationRequest = blob(&f.state, &old.evaluation_request_sha256).unwrap();
    outer.stage = EvaluationStage::Screen;
    outer.fitting_rows_sha256 = partition.fitting_rows_sha256().unwrap();
    outer.scoring_rows_sha256 = canonical_sha256(&partition.hold_rows).unwrap();
    outer.fitting_rows = partition.fitting_rows.len();
    outer.scoring_rows = partition.hold_rows.len();
    let selection = jope_pipeline::search_plan::SelectionPlan::new(
        outer,
        partition.fitting_rows.clone(),
        partition.hold_rows.clone(),
        partition.inner_validation_rows.clone(),
        partition.fitting_groups.clone(),
        partition.hold_groups.clone(),
    )
    .unwrap();
    (plan, partition, freeze, selection)
}

#[test]
fn inner_partition_excludes_outer_hold_and_pins_comparison_not_candidate() {
    let f = fixture(Task::Regression, 5);
    let (plan, partition, freeze, _) = inner_fixture(&f);
    let policy = load_freeze(&f.state, &freeze).unwrap();
    let targets: Vec<TargetRow> = blob(&f.state, &policy.targets_sha256).unwrap();
    assert_eq!(
        targets.iter().map(|r| &r.id).collect::<Vec<_>>(),
        partition.fitting_rows.iter().collect::<Vec<_>>()
    );
    assert!(targets.iter().all(|r| !partition.hold_rows.contains(&r.id)));
    assert_eq!(targets.len(), 8);
    assert_eq!(
        partition
            .inner_validation_rows
            .iter()
            .map(Vec::len)
            .sum::<usize>(),
        8
    );
    assert_eq!(
        f.state.read_blob(&freeze.freeze_sha256).unwrap().unwrap(),
        fs::read(freeze_path(&f.state, &freeze.comparison_sha256).unwrap()).unwrap()
    );
    assert_eq!(
        blob::<Vec<String>>(&f.state, &partition.fitting_rows_sha256().unwrap()).unwrap(),
        partition.fitting_rows
    );
    let mut changed = partition.clone();
    changed.evaluation_seed += 1;
    assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
    changed = partition.clone();
    changed.runtime_inventory_sha256 = "e".repeat(64);
    assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
    changed = partition.clone();
    changed.executor_program_sha256 = "f".repeat(64);
    assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
    assert!(serde_json::to_value(&partition)
        .unwrap()
        .get("candidate_sha256")
        .is_none());
    // User-visible test labels have been burned on the original partition,
    // even though teaching has a distinct inner comparison identity.
    fs::remove_file(
        burn_path(
            &f.state,
            &plan.comparison.dataset_hash,
            &partition.original_split_sha256,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(require_unburned_test_partition(
        &f.state,
        &plan.comparison.dataset_hash,
        &partition.original_split_sha256
    )
    .is_err());
}

#[test]
fn inner_admission_refuses_retrofitted_unpinned_experiment() {
    let f = fixture(Task::Regression, 5);
    let (_, _, freeze, _) = inner_fixture(&f);
    let policy = load_freeze(&f.state, &freeze).unwrap();
    let experiment: Experiment = blob(&f.state, &f.evidence.experiment_blob_sha256).unwrap();
    let window: ExecutionWindow = blob(&f.state, &f.evidence.window_sha256).unwrap();
    let mut evidence = f.evidence.clone();
    evidence.freeze = freeze;
    evidence.inner = Some(InnerEvidence {
        selection_sha256: "a".repeat(64),
        frozen_recipe_sha256: "a".repeat(64),
        search_receipt_sha256: "a".repeat(64),
        runtime_sha256: "a".repeat(64),
        arm_window_sha256: "a".repeat(64),
        memory_policy_sha256: "a".repeat(64),
        candidate_freeze_sha256: "a".repeat(64),
    });
    assert!(
        inner::bind(&f.state, &policy, &evidence, &experiment, &window, false)
            .err()
            .unwrap()
            .to_string()
            .contains("pre-admission")
    );
}

#[test]
fn inner_fit_binding_rejects_seed_deadline_memory_and_late_freeze_changes() {
    let f = fixture(Task::Regression, 5);
    let (_, _, _, selection) = inner_fixture(&f);
    let experiment: Experiment = blob(&f.state, &f.evidence.experiment_blob_sha256).unwrap();
    let old: FoldPredictions = blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    let result: EvaluationResult = blob(&f.state, &old.execution_receipt_sha256).unwrap();
    let mut worker: WorkerOutcome = blob(&f.state, &result.worker_receipt_sha256).unwrap();
    let mut job: WorkerJob = blob(&f.state, &old.worker_job_sha256).unwrap();
    let arm = jope_pipeline::execution::ArmBudget {
        id_sha256: "1".repeat(64),
        seconds: 30,
    };
    let candidate = "2".repeat(64);
    let request = selection
        .selection_requests(&f.record.genome.program_sha256, &candidate)
        .unwrap()
        .remove(0);
    job.request = request.clone();
    job.arm = Some(arm.clone());
    let binding = inner::BoundInner {
        selection,
        candidate_sha256: candidate,
        arm,
        arm_start: 1000,
        arm_deadline: 5000,
        frozen_at: 4500,
        program_sha256: f.record.genome.program_sha256.clone(),
        memory: jope_pipeline::execution::fit_memory_allocation(&experiment).unwrap(),
        runtime: json!({"program":job.command.program, "argument_prefix":job.command.args,
                "environment":job.command.environment,"inputs":job.command.inputs}),
    };
    worker.consumption.gate_released_unix_ms = Some(1100);
    worker.consumption.observed_exit_unix_ms = Some(4000);
    worker.consumption.fit_memory_limit_bytes = Some(binding.memory.per_fit_bytes);
    let verify = |r: &EvaluationRequest, w: &WorkerOutcome, deadline| {
        inner::validate_execution(
            &binding,
            r,
            &job,
            w,
            1000,
            deadline,
            &f.record.genome.program_sha256,
        )
    };
    verify(&request, &worker, 4100).unwrap();
    let mut stale = request.clone();
    stale.seed ^= 1;
    assert!(verify(&stale, &worker, 4100).is_err());
    assert!(verify(&request, &worker, 6000).is_err());
    let mut bad = worker.clone();
    bad.consumption.fit_memory_limit_bytes = None;
    assert!(verify(&request, &bad, 4100).is_err());
    bad = worker.clone();
    bad.consumption.observed_exit_unix_ms = Some(4600);
    assert!(verify(&request, &bad, 4800).is_err());
    bad = worker;
    bad.consumption.observed_exit_unix_ms = Some(4101);
    assert!(verify(&request, &bad, 4100).is_err());
}
