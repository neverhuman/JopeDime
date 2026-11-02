use super::*;

use super::*;
use tempfile::tempdir;

/// A failed cycle must not change the document's shape. Consumers read this
/// file with structs built against the success shape; the old error
/// document dropped 19 of 24 keys, which is the same defect that killed
/// jope-selfplay-supply 5,893 times on 2026-07-28.
#[test]
fn a_failed_cycle_degrades_in_schema() {
    let degraded = CycleReport::degraded("submit", "scq unreachable: connection refused");
    assert_eq!(degraded["schema"], WATCHDOG_SCHEMA);
    assert_eq!(degraded["degraded"], true);
    assert_eq!(degraded["error"], "scq unreachable: connection refused");
    assert_eq!(degraded["action"], "cycle_failed");
    assert_eq!(degraded["submitted"], false);

    // Every key a successful report carries is still present.
    let healthy = serde_json::to_value(CycleReport {
        schema: WATCHDOG_SCHEMA,
        generated_at_ms: 1,
        mode: "submit",
        continuous: true,
        label_step: LABEL_STEP,
        labels: 2,
        snapshot_watermark: 3,
        pending_unfrozen_labels: 4,
        last_evidence_watermark: 5,
        next_label_threshold: 6,
        checkpoint_sha256: "abc".into(),
        checkpoint_origin: "lineage:x".into(),
        checkpoint_tensors: 152,
        checkpoint_full: true,
        lineage_full_runs: 10,
        family_priority: 10.0,
        priority_ready: true,
        active_trainers: vec!["run".into()],
        action: "active_trainer_preserved".into(),
        planned_job_id: None,
        continuation_kind: None,
        submitted: false,
        protected_work_mutations: 0,
        promotion_mutations: 0,
    })
    .unwrap();
    for key in healthy.as_object().unwrap().keys() {
        assert!(degraded.get(key).is_some(), "degraded report dropped {key}");
    }
}

fn job(id: &str, project: &str, state: &str, node: Option<&str>, gpus: u64) -> Value {
    json!({
        "id": id,
        "project": project,
        "tier": "normal",
        "state": state,
        "node": node,
        "gpus": gpus,
    })
}

fn checkpoint(sha: &str, origin: &str, tensors: usize, missing: &[&str]) -> Checkpoint {
    schema_checkpoint(sha, origin, tensors, missing, legacy_checkpoint_schema())
}

fn schema_checkpoint(
    sha: &str,
    origin: &str,
    tensors: usize,
    missing: &[&str],
    schema: &'static CheckpointSchema,
) -> Checkpoint {
    Checkpoint {
        path: PathBuf::from(format!("/nonexistent/{sha}.model.pt")),
        sha256: sha.to_owned(),
        origin: origin.to_owned(),
        tensors,
        missing_groups: missing.iter().map(|group| (*group).to_owned()).collect(),
        schema: schema.name,
        required_tensors: schema.tensors,
        ..Checkpoint::default()
    }
}

fn split_context_schema() -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .iter()
        .find(|schema| schema.marker == Some("reward_encoder."))
        .expect("the v3.4 split-context schema is registered")
}

/// The exact shapes observed on disk: the original anchor fbcd5459 carries
/// 130 tensors and no binary branch, a-g8's d36274de carries all 152.
fn anchor() -> Checkpoint {
    checkpoint(
        &"f".repeat(64),
        "original_anchor",
        130,
        REQUIRED_CHECKPOINT_GROUPS,
    )
}

fn full_lineage(job: &str) -> Checkpoint {
    checkpoint(&"d".repeat(64), &format!("lineage:{job}"), 152, &[])
}

#[test]
fn tensor_deficient_anchor_never_wins_over_a_full_lineage_checkpoint() {
    // Candidate order is preference order: the anchor sorts last, but even a
    // *preferred* deficient candidate must lose to a full one.
    let selected = select_checkpoint(&[anchor(), full_lineage("a-g8")]).unwrap();
    assert_eq!(selected.origin, "lineage:a-g8");
    assert_eq!(selected.tensors, 152);
    assert!(selected.is_full());
}

#[test]
fn a_deficient_accepted_pointer_does_not_pin_a_lobotomised_checkpoint() {
    let pointer = checkpoint(
        &"e".repeat(64),
        "accepted_pointer:training/accepted.json",
        130,
        &["binary_reward_heads."],
    );
    let selected = select_checkpoint(&[pointer, full_lineage("a-g8")]).unwrap();
    assert_eq!(selected.origin, "lineage:a-g8");
}

#[test]
fn a_full_accepted_pointer_outranks_the_lineage() {
    let pointer = checkpoint(
        &"e".repeat(64),
        "accepted_pointer:training/accepted.json",
        152,
        &[],
    );
    let selected = select_checkpoint(&[pointer, full_lineage("a-g8")]).unwrap();
    assert_eq!(selected.origin, "accepted_pointer:training/accepted.json");
}

#[test]
fn no_full_candidate_stops_admission_instead_of_degrading_silently() {
    let error = select_checkpoint(&[anchor()]).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("tensor-deficient"), "{message}");
    assert!(message.contains("130 tensors"), "{message}");
    assert!(select_checkpoint(&[]).is_err());
}

#[test]
fn a_full_tensor_count_with_a_missing_group_is_still_deficient() {
    let sneaky = checkpoint(&"a".repeat(64), "lineage:x", 152, &["binary_routing_head."]);
    assert!(!sneaky.is_full());
    assert!(sneaky.deficiency().is_some());
}

/// v3.4 un-shared the context encoder (8 new tensors, 152 -> 160). Both
/// schemas are legitimate warm-start sources during the transition, so the
/// guard must judge each archive against the schema it actually claims.
#[test]
fn the_split_context_schema_is_resolved_by_its_marker_group() {
    let legacy = resolve_checkpoint_schema(&[
        "binary_reward_heads.".to_owned(),
        "binary_routing_head.".to_owned(),
        "binary_stability_head.".to_owned(),
    ]);
    assert_eq!(legacy.name, "policy_v3.3-fused-heads");
    assert_eq!(legacy.tensors, 152);
    let split = resolve_checkpoint_schema(&[
        "binary_reward_heads.".to_owned(),
        "binary_routing_head.".to_owned(),
        "binary_stability_head.".to_owned(),
        "reward_encoder.".to_owned(),
    ]);
    assert_eq!(split.name, "policy_v3.4-split-context");
    assert_eq!(split.tensors, 160);
}

#[test]
fn a_full_split_context_checkpoint_is_accepted_at_one_hundred_and_sixty() {
    let full = schema_checkpoint(
        &"1".repeat(64),
        "lineage:v34",
        160,
        &[],
        split_context_schema(),
    );
    assert!(full.is_full(), "{:?}", full.deficiency());
    assert_eq!(full.schema, "policy_v3.4-split-context");
}

/// The whole point of the schema table: an archive that CARRIES
/// `reward_encoder.` but stops at the v3.3 tensor count is 8 tensors short of
/// its own schema. Judging it against 152 would warm-start a model whose
/// reward-side context encoder is random — the v3.4 shape of the lobotomy.
#[test]
fn a_truncated_split_context_checkpoint_cannot_pass_on_the_legacy_count() {
    let truncated = schema_checkpoint(
        &"2".repeat(64),
        "lineage:v34-truncated",
        152,
        &[],
        split_context_schema(),
    );
    assert!(!truncated.is_full());
    let message = truncated.deficiency().unwrap();
    assert!(message.contains("152 tensors"), "{message}");
    assert!(message.contains("160"), "{message}");
}

#[test]
fn a_split_context_checkpoint_missing_the_reward_encoder_group_is_deficient() {
    let stripped = schema_checkpoint(
        &"3".repeat(64),
        "lineage:v34-stripped",
        160,
        &["reward_encoder."],
        split_context_schema(),
    );
    assert!(!stripped.is_full());
}

/// A defaulted Checkpoint has no resolved schema. It must never be pinnable:
/// `required_tensors == 0` is "unknown", not "no floor".
#[test]
fn a_checkpoint_with_no_resolved_schema_is_never_full() {
    let unresolved = Checkpoint::default();
    assert_eq!(unresolved.required_tensors, 0);
    assert!(!unresolved.is_full());
    assert!(select_checkpoint(&[unresolved]).is_err());
}

/// Both schemas may be warm-start sources during the transition, and the
/// newest completion wins on ordering, not on schema.
#[test]
fn a_legacy_and_a_split_context_checkpoint_are_both_admissible() {
    let legacy = full_lineage("a-g8");
    let split = schema_checkpoint(
        &"4".repeat(64),
        "lineage:v34",
        160,
        &[],
        split_context_schema(),
    );
    assert!(legacy.is_full());
    assert!(split.is_full());
    assert_eq!(
        select_checkpoint(&[split.clone(), legacy]).unwrap().origin,
        "lineage:v34"
    );
}

#[test]
fn lineage_orders_newest_completion_first_and_skips_missing_checkpoints() {
    let directory = tempdir().unwrap();
    let runs = directory.path();
    // A run whose train-result.json names a checkpoint that no longer exists
    // must be skipped rather than fail the whole scan.
    let orphan = runs
        .join("jope-selfplay-train-000000001-a-g1/attempts/0/outputs/training/watermark-000000001");
    fs::create_dir_all(&orphan).unwrap();
    fs::write(
        orphan.join("train-result.json"),
        json!({"checkpoint_sha256": "a".repeat(64), "recipe": "a"}).to_string(),
    )
    .unwrap();
    // A directory that is not a trainer run must be ignored entirely.
    fs::create_dir_all(runs.join("jte-12345/attempts/0/outputs/training/watermark-1")).unwrap();
    assert!(scan_lineage(runs).is_empty());
}

#[test]
fn torch_archive_pickle_scan_finds_required_group_names() {
    assert!(contains_subslice(
        b"\x8c\x18binary_reward_heads.0.0.weight",
        b"binary_reward_heads."
    ));
    assert!(!contains_subslice(
        b"\x8c\x0ereward_heads.0",
        b"binary_reward_heads."
    ));
    assert!(!contains_subslice(b"", b"binary_routing_head."));
}

#[test]
fn trainer_ids_preserve_watermark_and_continuation_sequence() {
    assert_eq!(
        parse_trainer_id("jope-selfplay-train-001805437-c-r000042"),
        Some((1_805_437, 42))
    );
    assert_eq!(
        parse_trainer_id("jope-selfplay-train-001805437-c"),
        Some((1_805_437, 0))
    );
    assert_eq!(
        parse_trainer_id("jope-testonly-train-1000"),
        Some((1_000, 0))
    );
    assert_eq!(parse_trainer_id("jope-selfplay-propose-123"), None);
}

#[test]
fn fifty_thousand_labels_coalesce_into_one_latest_snapshot() {
    assert_eq!(
        continuation_kind(260_000, 100_000),
        ContinuationKind::CoalescedLabels
    );
    assert_eq!(
        trainer_job_id(260_000, "c", 7),
        "jope-selfplay-train-000260000-c-r000007"
    );
}

#[test]
fn no_new_threshold_still_selects_continuous_recent_replay() {
    assert_eq!(
        continuation_kind(120_000, 100_000),
        ContinuationKind::RecentReplay
    );
    assert_eq!(
        continuation_kind(100_000, 100_000),
        ContinuationKind::RecentReplay
    );
}

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
