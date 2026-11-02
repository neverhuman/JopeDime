#[test]
fn forecast3_worker_manifest_binds_campaign_binary_and_comparator() {
    let mut spec = test_spec(1.0);
    spec.comparator_lock_sha256 = "e".repeat(64);
    let member = test_member(&"f".repeat(64), false);
    let seed = "1".repeat(64);
    let release = "2".repeat(64);
    let campaign = "3".repeat(64);
    let protocol = "4".repeat(64);
    let manifest = corpus_manifest(
        &spec,
        &[member],
        ("train.csv", "test.csv", "meta.json"),
        label_authority(0, &seed, &release, &campaign),
        Some(&protocol),
        Some((FORECAST3_RUN_ID, &campaign, &release)),
    )
    .unwrap();
    assert_eq!(manifest.run_id.as_deref(), Some(FORECAST3_RUN_ID));
    assert_eq!(manifest.campaign_sha256.as_deref(), Some(campaign.as_str()));
    assert_eq!(
        manifest.release_binary_sha256.as_deref(),
        Some(release.as_str())
    );
    assert_eq!(
        manifest.comparator_lock_sha256.as_deref(),
        Some(spec.comparator_lock_sha256.as_str())
    );
}

fn test_slot(generation: usize, slot: usize, mase: f64, evolved_descendant: bool) -> LogicalSlot {
    LogicalSlot {
        generation,
        slot,
        member: test_member(&format!("{generation:02}{slot:062}"), evolved_descendant),
        score: Some(TimeSeriesScore { mase, rmse: 1.0 }),
        valid: true,
        failure: None,
        receipt_file: "/tmp/receipt.jsonl".to_string(),
        receipt_sha256: "e".repeat(64),
        prediction_sha256: Some("f".repeat(64)),
        runtime_seconds: 0.1,
    }
}

fn base_result(strict_winner: bool) -> DatasetResult {
    DatasetResult {
        schema: RESULT_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: "base".to_string(),
        base_run_id: None,
        base_status: None,
        base_strict_winner: None,
        base_best_jope_mase: None,
        continuation_status: None,
        best_evidence_run_id: None,
        best_receipt_file: None,
        best_receipt_sha256: None,
        best_prediction_sha256: None,
        campaign_sha256: "a".repeat(64),
        release_binary_sha256: "b".repeat(64),
        dataset_hash: "dataset".to_string(),
        status: if strict_winner { "win" } else { "loss" }.to_string(),
        official_leader_model: "official".to_string(),
        official_leader_mase: 0.5,
        generation_zero_best_mase: Some(0.4),
        best_jope_mase: Some(if strict_winner { 0.35 } else { 0.45 }),
        best_jope_rmse: Some(1.0),
        mase_delta_vs_official: Some(if strict_winner { 0.15 } else { 0.05 }),
        strict_winner,
        best_beats_official: true,
        best_semantic_sha256: Some("s".to_string()),
        best_canonical_genome: Some("(pipe X (ridge))".to_string()),
        best_generation: Some(if strict_winner { 1 } else { 0 }),
        best_slot: Some(0),
        seed_source_family: None,
        logical_evaluations: 120,
        valid_evaluations: 120,
        completed_generations: 2,
        completed_at_unix_ms: 1,
    }
}

#[test]
fn gen30_stop_rule_requires_generation_two_descendant_for_base_losses() {
    let spec = test_spec(0.5);
    let base = base_result(false);
    let generation_zero_best = Some(0.4);
    let mut slots = vec![test_slot(0, 0, 0.39, false)];
    assert!(!continuation_should_stop(
        &base,
        &slots,
        &spec,
        generation_zero_best,
        GEN30_START_GENERATION,
    ));
    slots.push(test_slot(GEN30_START_GENERATION, 1, 0.45, true));
    assert!(!continuation_should_stop(
        &base,
        &slots,
        &spec,
        generation_zero_best,
        GEN30_START_GENERATION,
    ));
    slots.push(test_slot(GEN30_START_GENERATION, 2, 0.35, true));
    assert!(continuation_should_stop(
        &base,
        &slots,
        &spec,
        generation_zero_best,
        GEN30_START_GENERATION,
    ));
}

#[test]
fn gen30_base_wins_chase_until_terminal_generation() {
    let spec = test_spec(0.5);
    let base = base_result(true);
    let slots = vec![test_slot(1, 0, 0.35, true)];
    assert!(!continuation_should_stop(
        &base,
        &slots,
        &spec,
        Some(0.4),
        GEN30_START_GENERATION,
    ));
    assert!(continuation_should_stop(
        &base,
        &slots,
        &spec,
        Some(0.4),
        GEN30_GENERATIONS - 1,
    ));
}

#[test]
fn forecast3_campaign_can_win_at_generation_zero_and_runs_at_most_twenty_generations() {
    let config = base_campaign_config(FORECAST3_RUN_ID);
    assert_eq!(config.generations, FORECAST3_GENERATIONS);
    assert_eq!(config.minimum_generations, 0);

    let spec = test_spec(0.5);
    let seed = test_slot(0, 0, 0.49, false);
    assert!(slot_is_base_campaign_winner(&seed, &spec, Some(0.49), true,));
    assert!(!slot_is_base_campaign_winner(
        &seed,
        &spec,
        Some(0.49),
        false,
    ));
}

#[test]
fn forecast3_non_winners_are_exhausted_after_the_terminal_generation() {
    let loss = TimeSeriesScore {
        mase: 2.0,
        rmse: 3.0,
    };
    let tie = TimeSeriesScore {
        mase: 1.0,
        rmse: 3.0,
    };
    assert_eq!(
        base_campaign_status(FORECAST3_RUN_ID, false, Some(&loss), 1.0),
        "exhausted"
    );
    assert_eq!(
        base_campaign_status(FORECAST3_RUN_ID, false, Some(&tie), 1.0),
        "exhausted"
    );
    assert_eq!(
        base_campaign_status("legacy", false, Some(&loss), 1.0),
        "loss"
    );
    assert_eq!(
        base_campaign_status("legacy", false, Some(&tie), 1.0),
        "tie"
    );
}

#[test]
fn forecast3_win_rule_targets_external_leader_not_generation_zero_seed() {
    let spec = test_spec(0.5);
    let child = test_slot(1, 0, 0.45, true);
    assert!(slot_is_base_campaign_winner(&child, &spec, Some(0.4), true,));
    assert!(!slot_is_base_campaign_winner(
        &child,
        &spec,
        Some(0.4),
        false,
    ));
}

#[test]
fn mase_ranking_minimizes_before_rmse_and_complexity() {
    let mut low_mase = test_slot(2, 0, 0.2, true);
    low_mase.score.as_mut().unwrap().rmse = 100.0;
    let mut high_mase = test_slot(2, 1, 0.3, true);
    high_mase.score.as_mut().unwrap().rmse = 0.1;
    assert_eq!(rank_slots(&low_mase, &high_mase), Ordering::Less);

    let mut low_rmse = test_slot(2, 2, 0.2, true);
    low_rmse.score.as_mut().unwrap().rmse = 1.0;
    assert_eq!(rank_slots(&low_rmse, &low_mase), Ordering::Less);
}

fn write_generation_fixture(
    root: &Path,
    run_id: &str,
    campaign_sha256: &str,
    spec: &DatasetSpec,
    generation: usize,
) {
    let generation_dir = generation_root(&dataset_root(root, &spec.dataset_hash), generation);
    fs::create_dir_all(generation_dir.join("receipts")).unwrap();
    let members = (0..POPULATION)
        .map(|slot| {
            test_member(
                &format!("{generation:02}{slot:062}"),
                generation >= GEN30_START_GENERATION,
            )
        })
        .collect::<Vec<_>>();
    let mut shards = Vec::new();
    for (shard, node) in NODES[..SHARDS].iter().enumerate() {
        let mut receipt_bytes = Vec::new();
        for offset in 0..SHARD_SIZE {
            let slot = shard * SHARD_SIZE + offset;
            let row = json!({
                "task": "timeseries",
                "dataset_hash": spec.dataset_hash,
                "semantic_sha256": members[slot].semantic_sha256,
                "canonical_genome": members[slot].canonical_genome,
                "metric": "negative_mase",
                "score": -0.5,
                "secondary_metric": "mase",
                "secondary_score": 0.5,
                "valid": true,
                "failure": null,
                "runtime_seconds": 0.1,
                "predictions_sha256": "p",
                "rmse": 1.0
            });
            receipt_bytes.extend_from_slice(serde_json::to_string(&row).unwrap().as_bytes());
            receipt_bytes.push(b'\n');
        }
        let receipt_file = generation_dir
            .join("receipts")
            .join(format!("shard-{shard:02}.receipts.jsonl"));
        fs::write(&receipt_file, &receipt_bytes).unwrap();
        let receipt_sha256 = sha256_bytes(&receipt_bytes);
        shards.push(ShardRecord {
            generation,
            shard,
            node: node.to_string(),
            attempt: 0,
            job_id: format!("job-{shard}"),
            corpus_manifest_file: "/tmp/manifest.json".to_string(),
            corpus_manifest_sha256: "a".repeat(64),
            scq_spec_file: "/tmp/spec.toml".to_string(),
            scq_spec_sha256: "b".repeat(64),
            receipt_file: receipt_file.to_string_lossy().into_owned(),
            receipt_sha256,
            valid_candidates: SHARD_SIZE,
            total_candidates: SHARD_SIZE,
        });
    }
    let manifest = GenerationManifest {
        schema: GENERATION_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        campaign_sha256: campaign_sha256.to_string(),
        release_binary_sha256: "c".repeat(64),
        dataset_hash: spec.dataset_hash.clone(),
        generation,
        members,
        shards,
        best_semantic_sha256: None,
        best_mase: Some(0.5),
        valid_candidates: POPULATION,
        completed_at_unix_ms: 1,
    };
    fs::write(
        generation_dir.join("generation.v1.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn base_generation_loader_rejects_outdated_artifacts() {
    let dir = tempdir().unwrap();
    let spec = test_spec(0.5);
    let campaign_sha256 = "a".repeat(64);
    write_generation_fixture(dir.path(), "other-run", &campaign_sha256, &spec, 0);
    let error = load_generation_slots(dir.path(), "base-run", &campaign_sha256, &spec, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("outdated or invalid"));
}

#[test]
fn timeseries_dataset_report_is_benchmark_invalid_and_mase_authoritative() {
    let spec = test_spec(0.5);
    let mut result = base_result(false);
    result.run_id = "gen30".to_string();
    result.base_run_id = Some("base".to_string());
    result.continuation_status = Some("gen30_strict_win".to_string());
    result.status = "win".to_string();
    result.strict_winner = true;
    result.best_jope_mase = Some(0.3);
    result.best_jope_rmse = Some(2.0);
    result.mase_delta_vs_official = Some(0.2);
    result.best_receipt_file = Some("/tmp/best.receipts.jsonl".to_string());
    result.best_receipt_sha256 = Some("e".repeat(64));
    let report = timeseries_report_value(
        &spec,
        &result,
        Path::new("/tmp/terminal-summary.v1.json"),
        &"f".repeat(64),
    );
    assert_eq!(
        report["report_schema_version"],
        JOPE_TIMESERIES_REPORT_SCHEMA
    );
    assert_eq!(report["algorithm"], "jope-dime-regressor");
    assert_eq!(report["task_type"], "timeseries");
    assert_eq!(report["benchmark_valid"], false);
    assert_eq!(report["test_adaptive"], true);
    assert_eq!(report["source_path"], "/tmp/best.receipts.jsonl");
    assert_eq!(report["metrics"]["mase"], 0.3);
    assert_eq!(report["metrics"]["negative_mase"], -0.3);
    assert_eq!(
        report["comparison"]["continuation_status"],
        "gen30_strict_win"
    );
    assert_eq!(report["provenance"]["base_run_id"], "base");
}
