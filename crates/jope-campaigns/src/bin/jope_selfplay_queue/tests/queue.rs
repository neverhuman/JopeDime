use super::*;
use tempfile::tempdir;

#[test]
fn queue_policy_has_a_dynamic_slot_horizon_and_held_is_not_a_queue_state() {
    let targets = dynamic_queue_targets(&[20.0, 30.0, 40.0], 200).unwrap();
    assert_eq!(targets.target, 2_048);
    assert_eq!(targets.hard_cap, 4_096);
    assert_eq!(dynamic_queue_targets(&[20.0], 2_000).unwrap().target, 2_500);
    assert!(QUEUE_STATES.contains(&"queued"));
    assert!(QUEUE_STATES.contains(&"starting"));
    assert!(!QUEUE_STATES.contains(&"held"));
    assert!(!QUEUE_STATES.contains(&"running"));
}

#[test]
fn held_release_only_fills_the_gap_under_target() {
    // The 2026-08-01 pileup: 996 held, queue empty, target 2048 — release
    // a full batch, never more than the batch cap.
    assert_eq!(held_release_budget(0, 2_048, 996), HELD_RELEASE_BATCH_MAX);
    // Queue already at/over target: a busy cluster releases nothing, so
    // the overload guard's decision is respected while pressure is real.
    assert_eq!(held_release_budget(2_048, 2_048, 996), 0);
    assert_eq!(held_release_budget(3_000, 2_048, 996), 0);
    // Fewer held than gap: release exactly what exists.
    assert_eq!(held_release_budget(2_040, 2_048, 3), 3);
    // Gap smaller than batch cap: release exactly the gap.
    assert_eq!(held_release_budget(2_000, 2_048, 996), 48);
    // A zero/negative target (fail-closed cycles) releases nothing.
    assert_eq!(held_release_budget(0, 0, 996), 0);
    assert_eq!(held_release_budget(0, -5, 996), 0);
}

#[test]
fn held_multiclass_jobs_are_outside_the_release_contract() {
    let job = SchedulerJob {
        id: "jope-mc272-protected".to_owned(),
        project: "jope-multiclass-test-a".to_owned(),
        state: "held".to_owned(),
        failure_class: None,
        queue_ts: None,
    };

    assert!(!scheduler_job_is_adoptable(&job));
}

#[test]
fn failure_report_preserves_the_queue_status_envelope() {
    let value = serde_json::to_value(failure_report(true, "boom")).unwrap();
    assert_eq!(value["schema"], "JopeSelfplayQueueKeeper.v3");
    assert_eq!(value["admission_allowed"], false);
    assert_eq!(value["degraded"], true);
    assert_eq!(value["error"], "boom");
    for field in [
        "queued_target",
        "queued_hard_cap",
        "queued_starting",
        "dispatching",
        "admission_waves",
        "unclaimed_current_checkpoint_manifests",
    ] {
        assert!(value.get(field).is_some(), "missing {field}");
    }
}

#[test]
fn backlog_initialization_does_not_rebuild_the_active_attempt_index() {
    let connection = Connection::open_in_memory().unwrap();
    initialize_backlog(&connection).unwrap();
    let before: i64 = connection
        .query_row("PRAGMA schema_version", [], |row| row.get(0))
        .unwrap();
    initialize_backlog(&connection).unwrap();
    let after: i64 = connection
        .query_row("PRAGMA schema_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(after, before);
    let sql: String = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name='keeper_one_active_attempt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(sql.contains("deferred_horizon"));
}

#[test]
fn manifest_validation_is_bounded_to_the_next_admission_batch() {
    assert_eq!(bounded_validation_budget(1_408, 512), 576);
    assert_eq!(bounded_validation_budget(40, 512), 40);
    assert_eq!(bounded_validation_budget(100, 1), 2);
    assert_eq!(bounded_validation_budget(100, 0), 0);
    assert!(evaluator_failure_is_permanent(&anyhow::anyhow!(
        "Rust evaluator rejected manifest"
    )));
    assert!(!evaluator_failure_is_permanent(&anyhow::anyhow!(
        "run Rust evaluator: resource temporarily unavailable"
    )));
}

#[test]
fn admission_waves_are_oldest_first_and_exclude_claimed_manifests() {
    let root = tempdir().unwrap();
    let first = root
        .path()
        .join("waves/strict-wave-000001/manifests/first.json");
    let second = root
        .path()
        .join("waves/strict-wave-000002/manifests/second.json");
    fs::create_dir_all(first.parent().unwrap()).unwrap();
    fs::create_dir_all(second.parent().unwrap()).unwrap();
    fs::write(&first, br#"{"candidates":[{},{}]}"#).unwrap();
    fs::write(&second, br#"{"candidates":[{},{},{}]}"#).unwrap();
    let waves = vec![
        MicroWave {
            wave: "strict-wave-000001".into(),
            summary_sha256: "a".repeat(64),
        },
        MicroWave {
            wave: "strict-wave-000002".into(),
            summary_sha256: "b".repeat(64),
        },
    ];
    let mut claims = HashMap::new();
    claims.insert(
        "claimed".into(),
        Claim {
            manifest_sha256: "c".repeat(64),
            manifest: first,
            job_id: "jte-claimed".into(),
            spec: root.path().join("claimed.toml"),
            state: "submitted".into(),
            attempts: 1,
            owner: Some("owner".into()),
        },
    );
    let admission =
        current_admission_waves(root.path(), &waves, &claims, &BTreeSet::new()).unwrap();
    assert_eq!(admission.len(), 1);
    assert_eq!(admission[0].wave, "strict-wave-000002");
    assert_eq!(admission[0].unclaimed_manifests, 1);
    assert_eq!(admission[0].candidate_evaluation_count, 3);
    let rejected = BTreeSet::from([second]);
    assert!(
        current_admission_waves(root.path(), &waves, &claims, &rejected)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn newest_two_complete_waves_are_never_admitted() {
    let wave = |index: usize, unclaimed: usize, complete: bool| AdmissionWave {
        wave: format!("strict-wave-{index:06}"),
        summary_sha256: format!("{index:064x}"),
        manifest_count: 10,
        unclaimed_manifests: unclaimed,
        candidate_evaluation_count: (unclaimed * 50) as u64,
        complete,
    };
    let (admission, reserve) = partition_protected_reserve(vec![
        wave(1, 4, false),
        wave(2, 10, true),
        wave(3, 10, true),
        wave(4, 10, true),
    ]);
    assert_eq!(
        admission
            .iter()
            .map(|wave| wave.wave.as_str())
            .collect::<Vec<_>>(),
        ["strict-wave-000001", "strict-wave-000002"]
    );
    assert_eq!(
        reserve
            .iter()
            .map(|wave| wave.wave.as_str())
            .collect::<Vec<_>>(),
        ["strict-wave-000003", "strict-wave-000004"]
    );
    assert!(reserve.iter().all(|wave| wave.complete));
}

#[test]
fn filters_stale_epoch_old_wave_and_protected_lane() {
    let root = tempdir().unwrap();
    let path = root.path().join("manifest.json");
    let current_epoch = format!("checkpoint-{}", checkpoint());
    atomic_json(&path, &manifest_value("strict-wave-000009", &current_epoch)).unwrap();
    assert!(validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_ok());
    assert!(validate_manifest(
        &path,
        &checkpoint(),
        "checkpoint-stale",
        "strict-wave-000009"
    )
    .is_err());
    assert!(validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000008").is_err());
    let mut protected = manifest_value("strict-wave-000009", &current_epoch);
    protected["evaluation_lane"] = json!("sealed_audit");
    atomic_json(&path, &protected).unwrap();
    assert!(validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_err());
}

#[test]
fn restart_submitting_claim_is_idempotently_adoptable() {
    let root = tempdir().unwrap();
    let connection = Connection::open(root.path().join("backlog.sqlite")).unwrap();
    initialize_backlog(&connection).unwrap();
    let genome = "(pipe X (ridge))";
    let manifest = ValidManifest {
        path: root.path().join("manifest.json"),
        sha256: "a".repeat(64),
        generated_at_ms: 1,
        pairs: vec![PairKey {
            task: "regression".into(),
            dataset_hash: "dataset".into(),
            semantic_sha256: sha256_bytes(genome.as_bytes()),
            protocol_sha256: "b".repeat(64),
        }],
        value: json!({}),
    };
    for _ in 0..2 {
        let tx = connection.unchecked_transaction().unwrap();
        append_keeper_attempts(&tx, &manifest, "wave", "jte-a", "submitting", 1).unwrap();
        tx.commit().unwrap();
    }
    let attempts: i64 = connection
        .query_row("SELECT COUNT(*) FROM keeper_attempts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(attempts, 1);
    let live = SchedulerJob {
        id: "jte-a".into(),
        project: PROJECT.into(),
        state: "queued".into(),
        failure_class: None,
        queue_ts: None,
    };
    assert_eq!(live.project, PROJECT);
    assert!(ACTIVE_JOB_STATES.contains(&live.state.as_str()));
    assert!(scheduler_job_is_adoptable(&live));
    let completed = SchedulerJob {
        state: "completed".into(),
        ..live
    };
    assert!(scheduler_job_is_adoptable(&completed));
}

#[test]
fn append_keeper_attempts_treats_sql_shaped_pair_keys_as_values() {
    let connection = Connection::open_in_memory().unwrap();
    initialize_backlog(&connection).unwrap();
    let hostile = "' OR 1=1; DROP TABLE keeper_attempts; --";
    let manifest = ValidManifest {
        path: PathBuf::from("/manifest.json"),
        sha256: "a".repeat(64),
        generated_at_ms: 1,
        pairs: vec![PairKey {
            task: hostile.to_owned(),
            dataset_hash: hostile.to_owned(),
            semantic_sha256: hostile.to_owned(),
            protocol_sha256: hostile.to_owned(),
        }],
        value: json!({}),
    };

    let first = connection.unchecked_transaction().unwrap();
    append_keeper_attempts(&first, &manifest, "owner", "old-job", "completed", 1).unwrap();
    first.commit().unwrap();

    let retry = connection.unchecked_transaction().unwrap();
    append_keeper_attempts(&retry, &manifest, "owner", "new-job", "submitting", 0).unwrap();
    retry.commit().unwrap();

    let inserted: (String, String, String, String, i64) = connection
        .query_row(
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number \
                 FROM keeper_attempts WHERE job_id=?",
            ["new-job"],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        inserted,
        (
            hostile.to_owned(),
            hostile.to_owned(),
            hostile.to_owned(),
            hostile.to_owned(),
            2
        )
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                ["keeper_attempts"],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}
