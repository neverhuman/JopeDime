use super::*;

fn digest(character: char) -> Sha256Digest {
    Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
}

fn observation(
    task: Task,
    parent: char,
    semantic: char,
    receipt: char,
    kind: EvidenceKind,
) -> CausalEvidenceObservation {
    CausalEvidenceObservation {
        key: CausalEvidenceKey {
            task,
            accepted_parent_checkpoint_sha256: digest(parent),
            wave: 190,
            manifest_sha256: digest('a'),
            candidate_semantic_sha256: digest(semantic),
            receipt_sha256: digest(receipt),
        },
        context_identity: "context".to_owned(),
        kind,
        successfully_ingested: true,
        valid_proposal: true,
        zero_cross_context_backfill: true,
        completed_causal_parent_wave: true,
        valid_original_lineage: true,
        dl_ancestry_kind: DlAncestryKind::None,
    }
}

#[test]
fn admission_recovers_exact_hold_release_symmetry() {
    let mut admission = GpuLaneAdmissionV1::new(
        GpuLaneRole::AlternatingTrainer,
        "/tmp/spec".into(),
        digest('a'),
        "trainer".to_owned(),
        1,
    )
    .unwrap();
    admission.stage = GpuAdmissionStage::QueuePaused;
    admission.record_held("eval-1".to_owned(), 2).unwrap();
    admission.record_held("eval-2".to_owned(), 3).unwrap();
    admission.stage = GpuAdmissionStage::Submitted;
    admission.recover();
    assert_eq!(
        admission.recovery_disposition,
        RecoveryDisposition::ConfirmSubmissionThenRelease
    );
    admission.record_released("eval-1", 4).unwrap();
    admission.recover();
    assert_eq!(
        admission.recovery_disposition,
        RecoveryDisposition::ResumeExactRelease
    );
    assert_eq!(admission.unreleased_evaluator_ids(), vec!["eval-2"]);
    assert!(admission.record_released("never-held", 5).is_err());
    admission.record_released("eval-2", 6).unwrap();
    admission.stage = GpuAdmissionStage::Complete;
    admission.validate().unwrap();
}

#[test]
fn lane_resources_are_node_specific_and_fixed() {
    for (role, node) in [
        (GpuLaneRole::CohortProposer, "xbabe1"),
        (GpuLaneRole::AlternatingTrainer, "xbabe2"),
        (GpuLaneRole::BreadthEncoder, "xbabe3"),
    ] {
        let row = GpuLaneAdmissionV1::new(role, "/tmp/spec".into(), digest('a'), "job".into(), 1)
            .unwrap();
        assert_eq!(row.target_node, node);
        assert_eq!(row.resource_reservation.gpus, 1);
        assert_eq!(row.resource_reservation.companion_cpu_cores, 4);
    }
}

#[test]
fn evidence_deduplicates_and_excludes_shadow_and_wave_189() {
    let accepted = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
    let direct = observation(Task::Binary, 'b', 'd', 'e', EvidenceKind::NativeDlDirect);
    let comparator = observation(
        Task::Binary,
        'b',
        'f',
        'a',
        EvidenceKind::MatchedGpComparator,
    );
    let mut wave_189 = direct.clone();
    wave_189.key.wave = 189;
    wave_189.key.candidate_semantic_sha256 = digest('1');
    let mut shadow = direct.clone();
    shadow.kind = EvidenceKind::ShadowDlDirect;
    shadow.key.candidate_semantic_sha256 = digest('2');
    let ledger = build_causal_evidence_ledger(
        &accepted,
        vec![direct.clone(), direct, comparator, wave_189, shadow],
        1,
    )
    .unwrap();
    let binary = &ledger.tasks[&Task::Binary];
    assert_eq!(binary.completed_native_direct, 1);
    assert_eq!(binary.completed_matched_gp, 1);
    assert_eq!(binary.exclusions.duplicate, 1);
    assert_eq!(binary.exclusions.wave_189, 1);
    assert_eq!(binary.exclusions.shadow, 1);
}

#[test]
fn task_parent_selection_and_shadow_lineage_reset() {
    let first = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
    let mut state = ShadowAlternationState::default();
    assert_eq!(state.next(&first), Task::Regression);
    state
        .last_shadow_by_task
        .insert(Task::Regression, digest('d'));
    assert_eq!(state.next(&first), Task::Binary);
    assert_eq!(state.warm_start(Task::Regression), Some(&digest('d')));
    let changed = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('e'))]);
    assert_eq!(state.next(&changed), Task::Regression);
    assert_eq!(state.warm_start(Task::Regression), Some(&digest('e')));
}

#[test]
fn alternation_with_only_legacy_tasks_reproduces_the_old_sequence() {
    // S1.2 gate: with only regression+binary eligible (no accepted anchors
    // for the new tasks), eight successive calls reproduce the exact
    // pre-widening R,B,R,B,R,B,R,B alternation.
    let accepted = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
    let mut state = ShadowAlternationState::default();
    let picks: Vec<Task> = (0..8).map(|_| state.next(&accepted)).collect();
    assert_eq!(
        picks,
        vec![
            Task::Regression,
            Task::Binary,
            Task::Regression,
            Task::Binary,
            Task::Regression,
            Task::Binary,
            Task::Regression,
            Task::Binary,
        ]
    );
}

#[test]
fn alternation_honours_next_task_override_and_legacy_state() {
    let accepted = BTreeMap::from([(Task::Binary, digest('b')), (Task::Regression, digest('c'))]);
    // Legacy state file: next_task persisted, no rotation cursor.
    let mut state = ShadowAlternationState {
        next_task: Some(Task::Binary),
        ..ShadowAlternationState::default()
    };
    assert_eq!(state.next(&accepted), Task::Binary);
    assert_eq!(state.next(&accepted), Task::Regression);
    // Terminal-failure retry: the controller re-pins next_task.
    state.next_task = Some(Task::Regression);
    assert_eq!(state.next(&accepted), Task::Regression);
    assert_eq!(state.next(&accepted), Task::Binary);
}

#[test]
fn alternation_rotates_new_tasks_once_they_have_accepted_anchors() {
    let accepted = BTreeMap::from([
        (Task::Binary, digest('b')),
        (Task::Regression, digest('c')),
        (Task::Timeseries, digest('d')),
        (Task::Multiclass, digest('e')),
    ]);
    let mut state = ShadowAlternationState::default();
    let picks: Vec<Task> = (0..8).map(|_| state.next(&accepted)).collect();
    assert_eq!(picks, Task::ROTATION_CYCLE.to_vec());
    // The cycle wraps deterministically.
    assert_eq!(state.next(&accepted), Task::Regression);
}

#[test]
fn production_intent_cannot_use_shadow_lineage() {
    let intent = TrainerIntentV2 {
        schema: TRAINER_INTENT_SCHEMA.to_owned(),
        task: Task::Regression,
        mode: TrainerMode::Production,
        accepted_parent_checkpoint_sha256: digest('a'),
        shadow_parent_checkpoint_sha256: Some(digest('d')),
        warm_start_checkpoint_sha256: digest('d'),
        corpus_identity_sha256: digest('c'),
        corpus_watermark: 1,
        seed: 1,
        immutable_job_identity: "job".to_owned(),
        immutable_spec_sha256: digest('d'),
        created_at_ms: 1,
        runtime_bundle_path: None,
        runtime_bundle_sha256: None,
        catalog_binding_path: None,
        catalog_binding_sha256: None,
        recipe: None,
    };
    assert!(intent.validate().is_err());
}

#[test]
fn shadow_canary_shape_and_dl_origin_cap_are_strict() {
    let summary = ShadowBattleSummaryV1 {
        schema: SHADOW_BATTLE_SUMMARY_SCHEMA.to_owned(),
        task: Task::Regression,
        accepted_parent_checkpoint_sha256: digest('a'),
        shadow_checkpoint_sha256: digest('f'),
        contexts: canonical_shadow_canary_contexts(),
        shadow_proposals_per_context: 8,
        matched_gp_comparators_per_context: 8,
        metrics: ShadowBattleMetrics {
            paired_validity: 1.0,
            paired_novelty: 0.5,
            paired_win_rate: 0.5,
            paired_stability: 0.5,
            paired_efficiency: 0.5,
        },
        gp_seed_eligible_winners: Vec::new(),
        promotion_eligible: false,
        created_at_ms: 1,
    };
    summary.validate().unwrap();
    assert!(can_admit_dl_origin_seed(6, 1, 1));
    assert!(!can_admit_dl_origin_seed(6, 2, 1));
}

#[test]
fn promotion_requires_three_independent_task_acceptances() {
    let mut streaks = TaskCanaryStreaks::default();
    assert!(!streaks.record(Task::Binary, true));
    assert!(!streaks.record(Task::Binary, true));
    assert!(streaks.record(Task::Binary, true));
    assert_eq!(streaks.streaks.get(&Task::Regression), None);
    streaks.reset_after_promotion(Task::Binary);
    assert_eq!(streaks.streaks[&Task::Binary], 0);
}
