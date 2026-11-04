use super::*;

#[test]
fn composite_rollback_rejects_runtime_unit_staleness_and_inventory_drift() {
    let capsule = rollback();
    capsule.validate_at(1_000).unwrap();
    assert!(capsule.digest_at(1_000).is_ok());

    let mut mismatch = capsule.clone();
    mismatch.node_receipts[0].unit_overlay_sha256 = digest(240);
    assert!(mismatch.validate_at(1_000).is_err());
    assert!(capsule.validate_at(302_000).is_err());

    let mut no_evaluator = capsule.clone();
    no_evaluator.units[0].evaluator = false;
    no_evaluator.unit_overlay_sha256 = no_evaluator.computed_unit_overlay_sha256().unwrap();
    assert!(no_evaluator.validate_at(1_000).is_err());

    let mut observed = capsule.units.clone();
    observed[0]
        .drop_ins
        .push(artifact("drop-ins/unknown.conf", 241));
    assert!(capsule.verify_effective_units(&observed).is_err());
}

#[test]
fn optional_task_floors_and_sampler_ratios_are_task_specific() {
    let readiness = OptionalTaskReadinessV2 {
        schema: OPTIONAL_READINESS_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(244)],
        tasks: vec![
            readiness_row("multiclass", true),
            readiness_row("timeseries", true),
        ],
    };
    readiness.validate().unwrap();
    let all = eight_slot_sampling_weights(true, true);
    assert_eq!(all["regression"].numerator, 6);
    assert_eq!(all["multiclass"].numerator, 2);
    let none = eight_slot_sampling_weights(false, false);
    assert_eq!(none["regression"].numerator, 8);
    assert_eq!(none["multiclass"].numerator, 0);

    let mut generic_floor = readiness;
    generic_floor.tasks[1].distinct_dataset_identities = 500;
    generic_floor.tasks[1].valid_v2_cache_bindings = 500;
    generic_floor.tasks[1].valid_context_transform_bindings = 500;
    assert!(generic_floor.validate().is_err());
}

#[test]
fn speed_gate_has_no_sub_three_x_exception() {
    let mut report = EncoderBenchmarkReportV2 {
        schema: BENCHMARK_REPORT_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(245)],
        benchmark_plan_sha256: digest(246),
        baseline: lane(100.0, 100.0),
        candidate: lane(300.0, 50.0),
        cache_hit: lane(10_000.0, 2.0),
        paired_bootstrap_throughput_ratio_lower_95: 2.5,
        control_plane_overhead_fraction: 0.04,
        output_drift_count: 0,
        fallback_count: 0,
        backend_mismatch_count: 0,
        collision_count: 0,
        oom_as_success_count: 0,
        useful_work_green: true,
        protected_reserves_green: true,
        companion_cpu_credits_green: true,
        release_speed_gate_passed: true,
        blockers: Vec::new(),
    };
    report.validate().unwrap();
    report.candidate.throughput_per_gpu_slot = 299.9;
    report.release_speed_gate_passed = false;
    report.blockers = vec!["throughput_below_3x".to_owned()];
    report.validate().unwrap();
}

#[test]
fn lift_requires_noninferiority_speed_or_quality_and_all_common_gates() {
    let task = |name: &str| LiftTaskResult {
        task: name.to_owned(),
        corrected_primary_delta_lower_95: matches!(name, "binary" | "multiclass").then_some(0.006),
        native_error_ratio_upper_95: matches!(name, "regression" | "timeseries").then_some(0.99),
        strict_win_rate_lower_95: 0.51,
        p95_latency_ratio: 1.0,
        failure_rate_delta: 0.0,
        bounded_metric_worst_delta: Some(0.0),
        native_error_worst_ratio: Some(1.0),
        directional_agreement: true,
        typed_absence: None,
    };
    let mut report = EncoderLiftReportV2 {
        schema: LIFT_REPORT_SCHEMA.to_owned(),
        generated_at_unix_ms: 1,
        freshness_budget_seconds: 300,
        input_digests: vec![digest(247)],
        experiment_sha256: digest(248),
        benchmark_report_sha256: digest(249),
        speed_gate_passed: true,
        tasks: ["regression", "binary", "multiclass", "timeseries"]
            .into_iter()
            .map(task)
            .collect(),
        leakage_detected: false,
        identity_collision_detected: false,
        reproducible: true,
        advancement_allowed: true,
        blockers: Vec::new(),
    };
    report.validate().unwrap();
    report.tasks[2].bounded_metric_worst_delta = Some(-0.021);
    report.advancement_allowed = false;
    report.blockers = vec!["catastrophic_multiclass_loss".to_owned()];
    report.validate().unwrap();
}

#[test]
fn capacity_advances_only_after_both_proof_thresholds_and_preempts_for_recovery() {
    assert_eq!(
        capacity().decide_at(1).unwrap(),
        CapacityDecision::Advance { workers: 2 }
    );
    let mut early = capacity();
    early.green_minutes_at_rung = 29;
    assert_eq!(
        early.decide_at(1).unwrap(),
        CapacityDecision::Hold { workers: 1 }
    );
    let mut saturated = capacity();
    saturated.cpu_psi = 0.251;
    assert_eq!(
        saturated.decide_at(1).unwrap(),
        CapacityDecision::GracefulPreempt {
            reason: "resource_pressure"
        }
    );
    let mut runway = capacity();
    runway.useful_work_at_target = false;
    assert_eq!(
        runway.decide_at(1).unwrap(),
        CapacityDecision::GracefulPreempt {
            reason: "recovery_runway"
        }
    );
    assert!(capacity().decide_at(300_002).is_err());
}
