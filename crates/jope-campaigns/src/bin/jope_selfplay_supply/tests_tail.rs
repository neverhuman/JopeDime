use super::*;

#[test]
fn proposer_capacity_pressure_is_a_normal_backoff() {
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: dependencies cannot be verified on any eligible \
             node; eligible_node launch-ready-capacity: no eligible launch-ready node currently"
    ));
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: candidate node: smoke proposal GPU capacity: \
             smoke test exited with exit 6"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: unrelated smoke test exited with exit 6"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "no eligible launch-ready node currently"
    ));
}

#[test]
fn frozen_catalog_snapshot_survives_independent_live_advancement() {
    let directory = tempdir().unwrap();
    let live = directory.path().join("current.json");
    let snapshot = directory.path().join("frozen.json");
    let first = json!({
        "catalog_sha256": "a".repeat(64),
        "catalog": "/catalog-a.json",
        "data": "/catalog-a.f32",
    });
    std::fs::write(&live, serde_json::to_vec(&first).unwrap()).unwrap();
    assert_eq!(
        frozen_catalog_pointer(&live, &snapshot, &"a".repeat(64)).unwrap(),
        first
    );

    let advanced = json!({
        "catalog_sha256": "b".repeat(64),
        "catalog": "/catalog-b.json",
        "data": "/catalog-b.f32",
    });
    std::fs::write(&live, serde_json::to_vec(&advanced).unwrap()).unwrap();
    assert_eq!(
        frozen_catalog_pointer(&live, &snapshot, &"a".repeat(64)).unwrap(),
        first
    );
}

#[test]
fn keeper_handshake_allows_only_one_unacknowledged_wave() {
    let directory = tempdir().unwrap();
    let ready = vec![
        ReadyWave {
            wave: "strict-wave-000001".to_owned(),
            summary_sha256: "a".repeat(64),
            manifests: 384,
            unclaimed_manifests: 300,
        },
        ReadyWave {
            wave: "strict-wave-000002".to_owned(),
            summary_sha256: "b".repeat(64),
            manifests: 384,
            unclaimed_manifests: 384,
        },
    ];
    let status = QueueStatus {
        schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
        generated_at_ms: now_ms(),
        admission_allowed: true,
        degraded: false,
        error: None,
        fail_closed_reason: None,
        active_checkpoint_sha256: Some("c".repeat(64)),
        checkpoint_epoch: Some(checkpoint_epoch(&"c".repeat(64)).unwrap()),
        queued_target: 2_048,
        queued_hard_cap: 4_096,
        queued: 0,
        starting: 0,
        queued_starting: 0,
        dispatching: 0,
        admission_waves: vec![KeeperAdmissionWave {
            wave: ready[0].wave.clone(),
            summary_sha256: ready[0].summary_sha256.clone(),
            manifest_count: 384,
            unclaimed_manifests: ready[0].unclaimed_manifests,
            candidate_evaluation_count: 25_600,
            complete: false,
        }],
        protected_reserve_waves: Vec::new(),
        complete_current_checkpoint_waves: 0,
        partial_current_checkpoint_waves: 1,
        complete_reserve_waves: 0,
        reserve_manifests: 0,
        admission_manifests: 300,
        validated_ready: 300,
        useful_work: 300,
        candidate_evaluation_work: 25_600,
        unclaimed_current_checkpoint_manifests: 300,
    };
    assert!(validate_keeper_admission(
        &status,
        &ready,
        directory.path(),
        Some("strict-wave-000002"),
        8
    )
    .is_ok());
    assert!(validate_keeper_admission(&status, &ready, directory.path(), None, 8).is_err());
}

#[test]
fn keeper_handshake_accepts_monotonic_claim_progress_after_its_snapshot() {
    let directory = tempdir().unwrap();
    let ready = vec![ReadyWave {
        wave: "strict-wave-000001".to_owned(),
        summary_sha256: "a".repeat(64),
        manifests: 384,
        unclaimed_manifests: 299,
    }];
    let status = QueueStatus {
        schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
        generated_at_ms: now_ms(),
        admission_allowed: true,
        degraded: false,
        error: None,
        fail_closed_reason: None,
        active_checkpoint_sha256: Some("c".repeat(64)),
        checkpoint_epoch: Some(checkpoint_epoch(&"c".repeat(64)).unwrap()),
        queued_target: 2_048,
        queued_hard_cap: 4_096,
        queued: 0,
        starting: 0,
        queued_starting: 0,
        dispatching: 0,
        admission_waves: vec![KeeperAdmissionWave {
            wave: ready[0].wave.clone(),
            summary_sha256: ready[0].summary_sha256.clone(),
            manifest_count: 384,
            unclaimed_manifests: 300,
            candidate_evaluation_count: 25_600,
            complete: false,
        }],
        protected_reserve_waves: Vec::new(),
        complete_current_checkpoint_waves: 0,
        partial_current_checkpoint_waves: 1,
        complete_reserve_waves: 0,
        reserve_manifests: 0,
        admission_manifests: 300,
        validated_ready: 300,
        useful_work: 300,
        candidate_evaluation_work: 25_600,
        unclaimed_current_checkpoint_manifests: 300,
    };
    assert!(validate_keeper_admission(&status, &ready, directory.path(), None, 8).is_ok());

    let impossible = vec![ReadyWave {
        unclaimed_manifests: 301,
        ..ready[0].clone()
    }];
    assert!(validate_keeper_admission(&status, &impossible, directory.path(), None, 8).is_err());
}

#[test]
fn keeper_acknowledgment_must_be_newer_than_the_build() {
    let directory = tempdir().unwrap();
    let checkpoint = "c".repeat(64);
    let wave = "strict-wave-000002";
    let pending = PendingAck {
        wave: wave.to_owned(),
        since_ms: 10_000,
    };
    let mut status = QueueStatus {
        schema: "JopeSelfplayQueueKeeper.v3".to_owned(),
        generated_at_ms: pending.since_ms,
        admission_allowed: true,
        degraded: false,
        error: None,
        fail_closed_reason: None,
        active_checkpoint_sha256: Some(checkpoint.clone()),
        checkpoint_epoch: Some(checkpoint_epoch(&checkpoint).unwrap()),
        queued_target: 2_048,
        queued_hard_cap: 4_096,
        queued: 0,
        starting: 0,
        queued_starting: 0,
        dispatching: 0,
        admission_waves: Vec::new(),
        protected_reserve_waves: vec![KeeperAdmissionWave {
            wave: wave.to_owned(),
            summary_sha256: "a".repeat(64),
            manifest_count: 384,
            unclaimed_manifests: 384,
            candidate_evaluation_count: 32_768,
            complete: true,
        }],
        complete_current_checkpoint_waves: 1,
        partial_current_checkpoint_waves: 0,
        complete_reserve_waves: 1,
        reserve_manifests: 384,
        admission_manifests: 0,
        validated_ready: 384,
        useful_work: 384,
        candidate_evaluation_work: 32_768,
        unclaimed_current_checkpoint_manifests: 384,
    };
    let inputs = CheckpointInputs {
        checkpoint_sha256: checkpoint.clone(),
        checkpoint_pointer: PathBuf::new(),
        checkpoint_artifact: PathBuf::new(),
        proposals_pointer: PathBuf::new(),
    };
    assert!(!keeper_acknowledged(
        &audit_args(directory.path()),
        &status,
        &inputs,
        &checkpoint_epoch(&checkpoint).unwrap(),
        &BTreeSet::new(),
        &pending,
    )
    .unwrap());
    status.generated_at_ms += 1;
    assert!(keeper_acknowledged(
        &audit_args(directory.path()),
        &status,
        &inputs,
        &checkpoint_epoch(&checkpoint).unwrap(),
        &BTreeSet::new(),
        &pending,
    )
    .unwrap());
}

#[test]
fn rejected_manifests_are_not_counted_as_unclaimed_supply() {
    let directory = tempdir().unwrap();
    let backlog = directory.path().join("backlog.sqlite");
    let rejected_hash = "d".repeat(64);
    let connection = Connection::open(&backlog).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE claims(manifest_sha256 TEXT);
                 CREATE TABLE rejected_manifests(
                   manifest TEXT PRIMARY KEY,
                   manifest_sha256 TEXT,
                   reason TEXT NOT NULL,
                   updated_at_ms INTEGER NOT NULL
                 ) WITHOUT ROWID;",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO rejected_manifests VALUES('/absent',?1,'invalid',1)",
            [&rejected_hash],
        )
        .unwrap();
    drop(connection);
    let unavailable = claimed_manifest_hashes(&backlog).unwrap();
    assert!(unavailable.contains(&rejected_hash));

    let missing_table = directory.path().join("old-backlog.sqlite");
    Connection::open(&missing_table)
        .unwrap()
        .execute_batch("CREATE TABLE claims(manifest_sha256 TEXT);")
        .unwrap();
    assert!(claimed_manifest_hashes(&missing_table).is_err());
}

#[test]
fn next_wave_identity_includes_unresolved_pending_cohort() {
    let directory = tempdir().unwrap();
    let waves = directory.path();
    fs::create_dir(waves.join("strict-wave-000004")).unwrap();
    fs::create_dir(waves.join("strict-wave-invalid")).unwrap();
    fs::create_dir(waves.join("pending")).unwrap();
    File::create(waves.join("pending").join("strict-wave-000006.cohort.json")).unwrap();
    assert_eq!(next_wave_id(waves).unwrap(), "strict-wave-000007");
    assert_eq!(wave_to_build(waves).unwrap(), "strict-wave-000006");
}

#[test]
fn operator_quarantine_skips_only_hash_bound_pending_wave() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let waves = root.join("waves");
    fs::create_dir_all(waves.join("pending")).unwrap();
    fs::create_dir_all(waves.join("quarantine")).unwrap();
    fs::create_dir_all(root.join("ops/cohort-supply-state-v2")).unwrap();
    let wave = "strict-wave-000006";
    let cohort = waves.join("pending").join(format!("{wave}.cohort.json"));
    let state = supply_state_path(root, wave);
    fs::write(&cohort, b"frozen-cohort").unwrap();
    fs::write(&state, b"frozen-state").unwrap();
    atomic_json(
        &waves
            .join("quarantine")
            .join(format!("{wave}.receipt.json")),
        &json!({
            "schema": "CohortSupplyQuarantineReceipt.v1",
            "wave": wave,
            "cohort_sha256": sha256_file(&cohort).unwrap(),
            "supply_state_sha256": sha256_file(&state).unwrap(),
            "operator_authorized": true,
            "reason": "proposal validity below immutable gate",
        }),
    )
    .unwrap();
    assert_eq!(wave_to_build(&waves).unwrap(), "strict-wave-000007");
}

#[test]
fn stale_checkpoint_wave_is_not_launch_ready() {
    let directory = tempdir().unwrap();
    let wave = "strict-wave-000001";
    let wave_directory = directory.path().join(wave);
    fs::create_dir_all(wave_directory.join("manifests")).unwrap();
    fs::write(wave_directory.join("manifests/000.json"), b"{}").unwrap();
    let current = "a".repeat(64);
    let stale = "b".repeat(64);
    atomic_json(
        &wave_directory.join("summary.json"),
        &json!({
            "schema": WAVE_SCHEMA,
            "immutable": true,
            "wave_kind": MICRO_WAVE_KIND,
            "wave": wave,
            "checkpoint_sha256": stale,
            "checkpoint_epoch": checkpoint_epoch(&"b".repeat(64)).unwrap(),
            "manifest_count": 1,
            "manifests": [{
                "relative_path": "manifests/000.json",
                "sha256": "c".repeat(64)
            }]
        }),
    )
    .unwrap();
    let ready = validate_ready_wave(
        wave,
        &wave_directory,
        &current,
        &checkpoint_epoch(&current).unwrap(),
        &BTreeSet::new(),
    )
    .unwrap();
    assert!(ready.is_none());
}

#[test]
fn legacy_summary_is_ignored_before_current_micro_wave_validation() {
    let directory = tempdir().unwrap();
    let wave = "strict-wave-000010";
    let checkpoint = "a".repeat(64);
    let wave_directory = directory.path().join(wave);
    fs::create_dir_all(&wave_directory).unwrap();
    atomic_json(
        &wave_directory.join("summary.json"),
        &json!({
            "schema": "JopeTestOnlyStrictWave.v1",
            "wave": wave,
            "checkpoint_sha256": checkpoint
        }),
    )
    .unwrap();
    assert!(validate_ready_wave(
        wave,
        &wave_directory,
        &"a".repeat(64),
        &checkpoint_epoch(&"a".repeat(64)).unwrap(),
        &BTreeSet::new(),
    )
    .unwrap()
    .is_none());

    let mut current_claim: Value =
        serde_json::from_slice(&fs::read(wave_directory.join("summary.json")).unwrap()).unwrap();
    current_claim["schema"] = json!(WAVE_SCHEMA);
    atomic_json(&wave_directory.join("summary.json"), &current_claim).unwrap();
    assert!(validate_ready_wave(
        wave,
        &wave_directory,
        &"a".repeat(64),
        &checkpoint_epoch(&"a".repeat(64)).unwrap(),
        &BTreeSet::new(),
    )
    .is_err());
}

#[test]
fn malformed_manifest_path_fails_closed() {
    let directory = tempdir().unwrap();
    let wave = "strict-wave-000001";
    let checkpoint = "a".repeat(64);
    let wave_directory = directory.path().join(wave);
    fs::create_dir_all(&wave_directory).unwrap();
    atomic_json(
        &wave_directory.join("summary.json"),
        &json!({
            "schema": WAVE_SCHEMA,
            "immutable": true,
            "wave_kind": MICRO_WAVE_KIND,
            "wave": wave,
            "checkpoint_sha256": checkpoint,
            "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
            "manifest_count": 1,
            "manifests": [{
                "relative_path": "../escape.json",
                "sha256": "c".repeat(64)
            }]
        }),
    )
    .unwrap();
    assert!(validate_ready_wave(
        wave,
        &wave_directory,
        &"a".repeat(64),
        &checkpoint_epoch(&"a".repeat(64)).unwrap(),
        &BTreeSet::new(),
    )
    .is_err());
}

#[test]
fn audit_cycle_counts_only_current_checkpoint_immutable_supply() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("ops")).unwrap();
    fs::create_dir_all(root.join("proposals")).unwrap();
    let checkpoint = "a".repeat(64);
    atomic_json(
        &root.join("proposals/current.json"),
        &json!({"checkpoint_sha256": checkpoint}),
    )
    .unwrap();
    let wave = "strict-wave-000001";
    let wave_directory = root.join("waves").join(wave);
    fs::create_dir_all(wave_directory.join("manifests")).unwrap();
    fs::write(wave_directory.join("manifests/000.json"), b"{}").unwrap();
    atomic_json(
        &wave_directory.join("summary.json"),
        &json!({
            "schema": WAVE_SCHEMA,
            "immutable": true,
            "wave_kind": MICRO_WAVE_KIND,
            "wave": wave,
            "checkpoint_sha256": "a".repeat(64),
            "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
            "manifest_count": 1,
            "manifests": [{
                "relative_path": "manifests/000.json",
                "sha256": "c".repeat(64)
            }]
        }),
    )
    .unwrap();
    let summary_sha256 = sha256_file(&wave_directory.join("summary.json")).unwrap();
    atomic_json(
        &root.join("ops/queue-keeper-v3.json"),
        &json!({
            "schema": "JopeSelfplayQueueKeeper.v3",
            "generated_at_ms": now_ms(),
            "admission_allowed": false,
            "active_checkpoint_sha256": checkpoint,
            "checkpoint_epoch": checkpoint_epoch(&"a".repeat(64)).unwrap(),
            "queued_target": 2048,
            "queued_hard_cap": 4096,
            "queued": 900,
            "starting": 0,
            "queued_starting": 900,
            "dispatching": 50,
            "admission_waves": [{
                "wave": wave,
                "summary_sha256": summary_sha256,
                "manifest_count": 1,
                "unclaimed_manifests": 1,
                "candidate_evaluation_count": 50,
                "complete": true
            }],
            "protected_reserve_waves": [],
            "complete_current_checkpoint_waves": 1,
            "partial_current_checkpoint_waves": 0,
            "complete_reserve_waves": 0,
            "reserve_manifests": 0,
            "admission_manifests": 1,
            "validated_ready": 1,
            "useful_work": 951,
            "candidate_evaluation_work": 47_550,
            "unclaimed_current_checkpoint_manifests": 1
        }),
    )
    .unwrap();

    let report = cycle(&audit_args(root)).unwrap();
    assert_eq!(report.queue_depth, 950);
    assert!(report.below_floor);
    assert_eq!(report.ready_manifests, 1);
    assert_eq!(report.projected_jobs_before_build, 951);
    assert_eq!(report.builds_planned, 0);
    assert!(!report.admission_allowed);
    assert_eq!(
        report.hold_reason.as_deref(),
        Some("queue admission is held")
    );

    atomic_json(
        &root.join("proposals/current.json"),
        &json!({"checkpoint_sha256": "b".repeat(64)}),
    )
    .unwrap();
    assert!(
        cycle(&audit_args(root)).is_ok(),
        "a queue hold must not cascade into unrelated pointer validation"
    );
}

#[test]
fn in_schema_queue_error_is_a_healthy_supply_hold() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("ops")).unwrap();
    atomic_json(
        &root.join("ops/queue-keeper-v3.json"),
        &json!({
            "schema": "JopeSelfplayQueueKeeper.v3",
            "generated_at_ms": now_ms(),
            "admission_allowed": false,
            "degraded": true,
            "error": "native attempt reservation conflict",
            "fail_closed_reason": "queue cycle failed closed",
            "active_checkpoint_sha256": Value::Null,
            "checkpoint_epoch": Value::Null,
            "queued_target": 0,
            "queued_hard_cap": 0,
            "queued": 0,
            "starting": 0,
            "queued_starting": 0,
            "dispatching": 0,
            "admission_waves": [],
            "protected_reserve_waves": [],
            "complete_current_checkpoint_waves": 0,
            "partial_current_checkpoint_waves": 0,
            "complete_reserve_waves": 0,
            "reserve_manifests": 0,
            "admission_manifests": 0,
            "validated_ready": 0,
            "useful_work": 0,
            "candidate_evaluation_work": 0,
            "unclaimed_current_checkpoint_manifests": 0
        }),
    )
    .unwrap();
    let report = cycle(&audit_args(root)).unwrap();
    assert!(!report.admission_allowed);
    assert_eq!(report.builds_planned, 0);
    assert_eq!(
        report.hold_reason.as_deref(),
        Some("native attempt reservation conflict")
    );
}
