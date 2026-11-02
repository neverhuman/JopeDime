use super::*;

#[test]
fn protected_nontrainer_work_is_not_part_of_duplicate_guard() {
    let protected = job(
        "ratchet-shadow-protected",
        "jope",
        "running",
        Some("xbabe3"),
        1,
    );
    assert!(parse_trainer_job(&protected).is_none());
    let proposal = job("jope-selfplay-propose-abcdef", PROJECT, "queued", None, 1);
    assert!(parse_trainer_job(&proposal).is_none());
    let trainer = job(
        "jope-selfplay-train-000100000-c-r000001",
        PROJECT,
        "running",
        Some("xbabe2"),
        1,
    );
    let parsed = parse_trainer_job(&trainer).unwrap().unwrap();
    validate_active_trainer(&parsed).unwrap();
}

#[test]
fn malformed_or_misplaced_trainers_fail_closed() {
    let malformed = job(
        "jope-selfplay-train-not-a-watermark",
        PROJECT,
        "running",
        Some("xbabe2"),
        1,
    );
    assert!(parse_trainer_job(&malformed).unwrap().is_err());
    let misplaced = job(
        "jope-selfplay-train-000100000-c-r000001",
        PROJECT,
        "running",
        Some("xbabe3"),
        1,
    );
    let parsed = parse_trainer_job(&misplaced).unwrap().unwrap();
    assert!(validate_active_trainer(&parsed).is_err());
}

#[test]
fn duplicate_active_trainers_fail_closed_without_touching_other_work() {
    let trainers = [
        job(
            "jope-selfplay-train-000100000-c-r000001",
            PROJECT,
            "running",
            Some("xbabe2"),
            1,
        ),
        job(
            "jope-selfplay-train-000100000-c-r000002",
            PROJECT,
            "queued",
            None,
            1,
        ),
    ]
    .iter()
    .map(|row| parse_trainer_job(row).unwrap().unwrap())
    .collect::<Vec<_>>();
    assert!(active_trainers(&trainers).is_err());
}

#[test]
fn priority_must_be_exactly_ten() {
    assert!(priority_is_ready(10.0));
    assert!(!priority_is_ready(1.0));
    assert!(!priority_is_ready(f64::NAN));
}

#[test]
fn durable_intent_round_trips_and_prevents_identity_drift() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("state.json");
    let state = DurableState {
        schema: "JopeContinuousTrainerWatchdogState.v1".to_owned(),
        max_observed_labels: 250_000,
        max_observed_snapshot_watermark: 250_000,
        last_evidence_watermark: 200_000,
        next_sequence: 9,
        intent: Some(SubmissionIntent {
            job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
            watermark: 250_000,
            sequence: 8,
            checkpoint_sha256: "a".repeat(64),
            spec: directory.path().join("trainer.toml"),
            spec_sha256: "b".repeat(64),
            mode: "coalesced_50k_labels".to_owned(),
            seed: 20_260_734,
            created_at_ms: 1,
            last_error: None,
            checkpoint_mismatch_observations: 0,
        }),
        updated_at_ms: 2,
    };
    save_state(&path, &state).unwrap();
    let restored = load_state(&path).unwrap();
    assert_eq!(restored.next_sequence, 9);
    assert_eq!(restored.max_observed_snapshot_watermark, 250_000);
    assert_eq!(restored.intent.unwrap().watermark, 250_000);
}

#[test]
fn durable_unsubmitted_intent_survives_newer_corpus_but_not_checkpoint() {
    let directory = tempdir().unwrap();
    let intent = SubmissionIntent {
        job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
        watermark: 250_000,
        sequence: 8,
        checkpoint_sha256: "a".repeat(64),
        spec: directory.path().join("trainer.toml"),
        spec_sha256: "b".repeat(64),
        mode: "coalesced_50k_labels".to_owned(),
        seed: 20_260_734,
        created_at_ms: 1,
        last_error: None,
        checkpoint_mismatch_observations: 0,
    };
    let snapshot = TrainingSnapshot {
        watermark: 300_000,
        manifest: PathBuf::new(),
        packed_manifest: PathBuf::new(),
        packed_catalog: PathBuf::new(),
        packed_files: Vec::new(),
    };
    let checkpoint = Checkpoint {
        path: PathBuf::new(),
        sha256: "a".repeat(64),
        ..Checkpoint::default()
    };
    validate_intent(&intent, &snapshot, &checkpoint).unwrap();
    let changed_checkpoint = Checkpoint {
        path: PathBuf::new(),
        sha256: "c".repeat(64),
        ..Checkpoint::default()
    };
    assert!(validate_intent(&intent, &snapshot, &changed_checkpoint).is_err());
}

#[test]
fn stale_unsubmitted_intent_requires_three_checkpoint_observations() {
    let directory = tempdir().unwrap();
    let mut intent = SubmissionIntent {
        job_id: "jope-selfplay-train-000250000-c-r000008".to_owned(),
        watermark: 250_000,
        sequence: 8,
        checkpoint_sha256: "a".repeat(64),
        spec: directory.path().join("trainer.toml"),
        spec_sha256: "b".repeat(64),
        mode: "coalesced_50k_labels".to_owned(),
        seed: 20_260_734,
        created_at_ms: 1,
        last_error: None,
        checkpoint_mismatch_observations: 0,
    };
    assert_eq!(
        observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
        IntentCheckpointDisposition::Confirming(1)
    );
    assert_eq!(
        observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
        IntentCheckpointDisposition::Confirming(2)
    );
    assert_eq!(
        observe_intent_checkpoint(&mut intent, &"c".repeat(64)),
        IntentCheckpointDisposition::Supersede
    );
    assert_eq!(
        observe_intent_checkpoint(&mut intent, &"a".repeat(64)),
        IntentCheckpointDisposition::Current
    );
    assert_eq!(intent.checkpoint_mismatch_observations, 0);
}

#[test]
fn immutable_spec_write_is_idempotent_and_rejects_drift() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("trainer.toml");
    write_immutable(&path, b"one\n").unwrap();
    write_immutable(&path, b"one\n").unwrap();
    assert!(write_immutable(&path, b"two\n").is_err());
}

#[test]
fn temporary_capacity_shortage_queues_but_real_dependency_failure_does_not() {
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: dependencies cannot be verified\n\
             no eligible launch-ready node currently"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: /usr/bin/python3 is absent"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "no eligible launch-ready node currently"
    ));
}

#[test]
fn stale_spec_contract_is_identified_for_immutable_replacement() {
    let directory = tempdir().unwrap();
    let legacy = directory.path().join("legacy.toml");
    fs::write(
        &legacy,
        "args = [\"--packed-source-train\", \"{input:train}\"]\n",
    )
    .unwrap();
    let lean = directory.path().join("lean.toml");
    fs::write(
        &lean,
        format!(
            "args = [\"--packed-manifest\", \"{{input:packed_manifest}}\"]\n\
                 JOPE_TRAINER_SPEC_CONTRACT = \"{TRAINER_SPEC_CONTRACT}\"\n"
        ),
    )
    .unwrap();
    assert!(intent_uses_stale_contract(&SubmissionIntent {
        spec: legacy,
        ..SubmissionIntent::default()
    }));
    assert!(!intent_uses_stale_contract(&SubmissionIntent {
        spec: lean,
        ..SubmissionIntent::default()
    }));
}

#[test]
fn rendered_contract_has_one_gpu_xbabe2_and_python_only_as_payload() {
    let text = format!(
        "project = \"{PROJECT}\"\n\
             command = \"/usr/bin/python3\"\n\
             gpus_per_task = 1\n\
             args = [\"-qx\", \"xbabe2\", \"/etc/hostname\"]\n"
    );
    assert!(text.contains("project = \"jope-selfplay-train\""));
    assert!(text.contains("gpus_per_task = 1"));
    assert!(text.contains("\"xbabe2\""));
    assert!(text.contains("command = \"/usr/bin/python3\""));
}

/// Placement must be constrained by `[[dependencies.smoke]]` and NOTHING else.
///
/// Two wrong answers were tried first, both in production:
///   1. `[dependencies] hostname = [...]` -- a serde NO-OP. `hostname` is not
///      a field of scq's DependencySpec, so serde drops it, `dependencies`
///      becomes empty and `prepare_dependencies` CLEARS
///      `dependency_allowed_nodes`. Placement was never constrained: scq put
///      the trainer on xbabe3, the launch RPC died with "Broken pipe" 92
///      times, and each failure counted against xbabe3's launch budget until
///      scq excluded the whole node -- black-holing 432 slots, 41% of the
///      cluster, 9 times in one hour.
///   2. `[resource] node = "xbabe2"` -- rejected outright: "node is
///      controller-only; normal jobs must be node-agnostic". This took the
///      trainer to ZERO running jobs.
///
/// `smoke` IS in the schema and IS enforced at placement. Probe the
/// capability (the live flywheel root, which exists only where the trainer
/// can actually read its corpus) rather than the hostname, so a second
/// qualifying node auto-qualifies instead of being hard-coded out.
#[test]
fn spec_constrains_placement_via_smoke_dependency_only() {
    let code = TRAINER.render_placement_block();
    assert!(
        code.contains("[[dependencies.smoke]]\n"),
        "smoke dependency missing -- placement would be UNCONSTRAINED and can \
             black-hole a remote node"
    );
    assert!(
        !code.contains("node = \"xbabe2\""),
        "[resource] node pin present -- scq rejects it as controller-only and \
             the trainer will not launch at all"
    );
    assert!(
        !code.contains("hostname"),
        "the hostname dependency is a serde no-op and must not come back"
    );
}

#[test]
fn packed_filenames_cannot_escape_the_staging_directory() {
    ensure_safe_filename("scores.npy").unwrap();
    let traversal = Path::new("..").join("scores.npy");
    assert!(ensure_safe_filename(traversal.to_str().unwrap()).is_err());
    assert!(ensure_safe_filename("nested/scores.npy").is_err());
}

#[test]
fn training_and_label_pointers_are_content_verified() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let train = root.join("train.jsonl");
    let validation = root.join("validation.jsonl");
    let packed_dir = root.join("packed");
    fs::create_dir(&packed_dir).unwrap();
    let scores = packed_dir.join("scores.npy");
    let catalog = packed_dir.join("catalog.f32le");
    fs::write(&train, b"train\n").unwrap();
    fs::write(&validation, b"validation\n").unwrap();
    fs::write(&scores, b"scores\n").unwrap();
    fs::write(&catalog, b"catalog\n").unwrap();

    let manifest = root.join("corpus-manifest.json");
    atomic_json(
        &manifest,
        &json!({
            "schema": "JopeTestOnlyTrainingCorpus.v1",
            "source_label_watermark": 100_000,
            "source_family_disjoint": true,
            "strict_filter": {"test_adaptive": true, "benchmark_valid": false},
            "train": train,
            "train_sha256": sha256_file(&train).unwrap(),
            "validation": validation,
            "validation_sha256": sha256_file(&validation).unwrap(),
        }),
    )
    .unwrap();
    let packed_manifest = packed_dir.join("manifest.json");
    atomic_json(
        &packed_manifest,
        &json!({
            "source_manifest_sha256": sha256_file(&manifest).unwrap(),
            "train_sha256": sha256_file(&train).unwrap(),
            "validation_sha256": sha256_file(&validation).unwrap(),
            "catalog_data": catalog,
            "catalog_data_sha256": sha256_file(&catalog).unwrap(),
            "files": {
                "scores.npy": {
                    "sha256": sha256_file(&scores).unwrap(),
                }
            }
        }),
    )
    .unwrap();
    let pointer = root.join("current-corpus.json");
    atomic_json(
        &pointer,
        &json!({
            "watermark": 100_000,
            "manifest": manifest,
            "manifest_sha256": sha256_file(&manifest).unwrap(),
            "packed_manifest": packed_manifest,
            "packed_manifest_sha256": sha256_file(&packed_manifest).unwrap(),
        }),
    )
    .unwrap();
    assert_eq!(
        load_training_snapshot(&pointer, 120_000).unwrap().watermark,
        100_000
    );
    fs::write(&scores, b"drift\n").unwrap();
    assert!(load_training_snapshot(&pointer, 120_000).is_err());

    let labels = root.join("labels.json");
    atomic_json(&labels, &json!({"exact_rows_total": 120_000})).unwrap();
    assert_eq!(load_label_watermark(&labels).unwrap(), 120_000);
}
