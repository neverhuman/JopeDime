use super::*;

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
