fn leader(model: &str, mase: f64, rmse: f64) -> Value {
    json!({
        "task_type": "timeseries",
        "model_name": model,
        "status": "completed",
        "metrics": {"mase": mase, "rmse": rmse}
    })
}

#[test]
fn official_leader_uses_raw_mase_then_rmse() {
    let dir = tempdir().unwrap();
    let reports = dir.path().join("reports");
    fs::create_dir_all(&reports).unwrap();
    fs::write(
        reports.join("a.json"),
        serde_json::to_vec(&leader("a", 0.8, 10.0)).unwrap(),
    )
    .unwrap();
    fs::write(
        reports.join("b.json"),
        serde_json::to_vec(&leader("b", 0.7, 20.0)).unwrap(),
    )
    .unwrap();
    fs::write(
        reports.join("c.json"),
        serde_json::to_vec(&leader("c", 0.7, 5.0)).unwrap(),
    )
    .unwrap();
    let accepted = external_reports(dir.path(), "dataset").unwrap();
    let best = official_leader(&accepted, dir.path()).unwrap();
    assert_eq!(accepted.len(), 3);
    assert_eq!(best.model, "c");
    assert_eq!(best.mase, 0.7);
}

#[test]
fn four_node_shards_include_xbabe4_and_spec_is_rust_only() {
    let node = StagedNode {
        node: "xbabe4".to_string(),
        hostname: "xbabe4".to_string(),
        bundle_root: "/remote/bundle".to_string(),
        runner_path: "/remote/bundle/jope-dime-regressor".to_string(),
        runner_sha256: "a".repeat(64),
        corpus_eval_path: "/remote/bundle/corpus_eval".to_string(),
        corpus_eval_sha256: "b".repeat(64),
        datasets: Vec::new(),
    };
    let spec = scq_spec(
        &node,
        "job",
        "/remote/manifest.json",
        "/remote/output",
        LEGACY_PROJECT,
    )
    .unwrap();
    assert!(spec.contains("xbabe4"));
    assert!(spec.contains("[[dependencies.smoke]]"));
    assert!(spec.contains("command = \"/usr/bin/test\""));
    assert!(spec.contains("args = [\"-f\", \"/remote/bundle/nodes/xbabe4.ready\"]"));
    assert!(!spec.contains("/etc/hostname"));
    assert!(!spec.contains("command = \"/bin/sh\""));
    assert!(!spec.contains("\nnode = "));
    assert!(spec.contains("timeseries-test-adaptive-batch-worker"));
    assert!(spec.contains("corpus_eval"));
    assert!(!spec.to_ascii_lowercase().contains("python"));
}

#[test]
fn shard_partition_is_stable() {
    assert_eq!(NODES, ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]);
    assert_eq!(SHARD_SIZE, 15);
    assert_eq!(POPULATION, SHARDS * SHARD_SIZE);
}

#[test]
fn wrong_node_marker_failure_is_capacity_not_staging_failure() {
    let wrong_node = anyhow::anyhow!(
        "TimeSeries42 SCQ check failed (exit status: 1): error[dependency-unsatisfied]: dependencies unsatisfied on all eligible nodes\n\
dependency diagnostics:\n  xbabe2: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1\n  xbabe3: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1"
    );
    assert!(is_scq_capacity_error_for_node(&wrong_node, "xbabe1"));

    let target_node = anyhow::anyhow!(
        "TimeSeries42 SCQ check failed (exit status: 1): error[dependency-unsatisfied]: dependencies unsatisfied on all eligible nodes\n\
dependency diagnostics:\n  xbabe1: smoke TimeSeries42 staged node xbabe1: smoke test exited with exit 1"
    );
    assert!(!is_scq_capacity_error_for_node(&target_node, "xbabe1"));
}

#[test]
fn queued_scq_timeout_is_retryable_capacity_status() {
    let now = Instant::now();
    let timeout = std::time::Duration::from_secs(SCQ_QUEUED_STALL_TIMEOUT_SECONDS);
    let queued_since = now - timeout - std::time::Duration::from_secs(1);
    assert_eq!(
        scq_wait_decision("queued", false, Some(queued_since), now, timeout),
        ScqWaitDecision::CancelQueuedStall
    );
    assert_eq!(
        scq_wait_decision("running", true, Some(queued_since), now, timeout),
        ScqWaitDecision::Wait
    );
    assert_eq!(
        scq_wait_decision("queued", true, Some(queued_since), now, timeout),
        ScqWaitDecision::Wait
    );
    let error = retryable_scq_stall_error("job", "stayed queued without last_start");
    assert!(is_scq_capacity_error_for_node(&error, "xbabe1"));
}

#[test]
fn outdated_cancelled_jobs_are_retryable_not_terminal_failures() {
    let error = retryable_scq_stall_error(
        "job",
        "was already cancelled and has no local shard receipt",
    );
    assert!(is_scq_capacity_error(&error));
}

#[test]
fn shard_retry_attempt_ids_stay_unique_across_nodes_and_cycles() {
    let manifest_sha256 = "a".repeat(64);
    let mut ids = BTreeSet::new();
    for attempt in 0..8 {
        let node = NODES[attempt % NODES.len()];
        ids.insert(expected_job_id(
            "run",
            "dataset",
            2,
            1,
            node,
            attempt,
            &manifest_sha256,
        ));
    }
    assert_eq!(ids.len(), 8);
}

#[test]
fn xbabe2_is_treated_as_local_without_ssh_alias() {
    assert!(is_local_scq_node("xbabe2", "controller-alias"));
    assert!(is_local_scq_node("controller-alias", "controller-alias"));
    assert!(!is_local_scq_node("xbabe1", "controller-alias"));
}

#[test]
fn receipt_reads_mase_from_negative_score() {
    let receipt = EvalReceipt {
        task: "timeseries".to_string(),
        dataset_hash: "d".to_string(),
        semantic_sha256: "s".to_string(),
        canonical_genome: "(pipe X (ridge))".to_string(),
        metric: "negative_mase".to_string(),
        score: Some(-0.42),
        secondary_metric: "mase".to_string(),
        secondary_score: Some(0.41),
        valid: true,
        failure: None,
        runtime_seconds: 0.0,
        predictions_sha256: None,
        rmse: Some(1.5),
        protocol_sha256: None,
    };
    let score = receipt_mase(&receipt).unwrap();
    assert_eq!(score.mase, 0.41);
    assert_eq!(score.rmse, 1.5);
}

fn test_spec(official_mase: f64) -> DatasetSpec {
    DatasetSpec {
        schema: DATASET_SCHEMA.to_string(),
        dataset_hash: "dataset".to_string(),
        display_name: "Dataset".to_string(),
        category: "test".to_string(),
        train_path: "/tmp/train.csv".to_string(),
        train_sha256: "a".repeat(64),
        test_path: "/tmp/test.csv".to_string(),
        test_sha256: "b".repeat(64),
        meta_path: "/tmp/meta.json".to_string(),
        meta_sha256: "c".repeat(64),
        train_rows: 10,
        test_rows: 5,
        csv_features: 1,
        reduction_features: 4,
        horizon: 1,
        seasonality: 1,
        frequency: "D".to_string(),
        target_is_final_column: true,
        known_future_covariates: Vec::new(),
        past_only_covariates: Vec::new(),
        source_real_data: true,
        synthetic_origin: false,
        reduction_version: String::new(),
        accepted_external_reports: Vec::new(),
        comparator_lock_sha256: String::new(),
        official_leader: OfficialLeader {
            model: "official".to_string(),
            mase: official_mase,
            rmse: 1.0,
            report_path: "/tmp/report.json".to_string(),
            report_sha256: "d".repeat(64),
        },
    }
}

fn test_member(semantic: &str, evolved_descendant: bool) -> Member {
    Member {
        canonical_genome: "(pipe X (ridge))".to_string(),
        semantic_sha256: semantic.to_string(),
        complexity: 1,
        rust_source_len: 10,
        evolved_descendant,
        ancestry: AncestryRecord {
            operation: "test".to_string(),
            parents: Vec::new(),
            retry_count: 0,
        },
    }
}
