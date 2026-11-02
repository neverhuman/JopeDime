#[test]
fn improved_requires_three_complete_disjoint_passes_and_pointer_cas() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let checkpoint = "a".repeat(64);
    for run in 0..3 {
        let panel = format!("{:064x}", 50_000 + run);
        atomic_json(
            &root
                .join("training/task-promotion-evaluations/accepted")
                .join(format!("run{run}.v3.json")),
            &json!({
                "schema": "TaskPromotionEvaluation.v3",
                "task": "regression",
                "accepted_parent_checkpoint_sha256": "c".repeat(64),
                "candidate_checkpoint_sha256": checkpoint,
                "panel_consumption_sha256": panel,
                "run_index": run,
                "metrics": {
                    "contexts": 128,
                    "confidence": {
                        "resamples": 10000,
                        "seed": run,
                        "candidate_vs_accepted_lower": 0.01,
                        "candidate_vs_gp_lower": 0.01,
                        "conditional_advantage_lower": 0.01
                    },
                    "maximum_slice_regression": 0.0,
                    "validity": 1.0,
                    "replay_verification": 1.0,
                    "failure_rate_increase": 0.0,
                    "latency_ratio": 1.0,
                    "strict_wins": 1,
                    "equal_budget_wins": 1
                },
                "accepted": true,
                "rejection_reasons": [],
                "evaluated_at_ms": 10 + run
            }),
        )
        .unwrap();
        let observations = (0..128)
            .map(|index| {
                let identity = run * 1_000 + index;
                json!({
                    "context_sha256": format!("{identity:064x}"),
                    "source_family_sha256": format!("{:064x}", 20_000 + identity / 2),
                    "candidate_score": 0.9,
                    "accepted_score": 0.8,
                    "gp_score": 0.8,
                    "candidate_conditional_advantage": 0.02,
                    "accepted_conditional_advantage": 0.01,
                    "valid": true,
                    "replay_verified": true,
                    "candidate_failed": false,
                    "accepted_failed": false,
                    "candidate_latency_ms": 100,
                    "accepted_latency_ms": 100,
                    "strict_win": index == 0,
                    "equal_budget_win": index == 1,
                    "slices": {
                        "origin": "real",
                        "size": "small",
                        "missingness": "none",
                        "imbalance": "balanced",
                        "target_shape": "balanced"
                    }
                })
            })
            .collect::<Vec<_>>();
        atomic_json(
            &root
                .join("training/task-promotion-evaluations/work")
                .join(format!("run{run}.observations.v3.json")),
            &json!({
                "schema": "TaskPromotionObservationSet.v3",
                "panel_consumption_sha256": panel,
                "observations": observations
            }),
        )
        .unwrap();
    }
    atomic_json(
        &root
            .join("training/task-promotions")
            .join(format!("regression-{checkpoint}.v1.json")),
        &json!({
            "schema": "TaskCheckpointPromotion.v2",
            "task": "regression",
            "parent_checkpoint_sha256": "c".repeat(64),
            "child_checkpoint_sha256": checkpoint,
            "external_pass_receipt_ids": ["run0", "run1", "run2"],
            "independent_single_use_panel_acceptances": 3
        }),
    )
    .unwrap();
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "tasks": {"regression": {"checkpoint_sha256": checkpoint}}
        }),
    )
    .unwrap();
    assert_eq!(publish_terminal_learning_reports(root).unwrap(), 1);
    let report = read_json(
        &root
            .join("ops/learning-reports/regression")
            .join(format!("{checkpoint}.v2.json")),
    )
    .unwrap();
    assert_eq!(report["disposition"], "improved");
    assert_eq!(
        report["promotion_disposition"]["pointer_compare_and_swap_succeeded"],
        true
    );
    assert_eq!(
        report["metrics"]["observation_run_coverage"],
        json!([0, 1, 2])
    );
}
