use jope_campaigns::selfplay_shadow::*;
use std::collections::BTreeMap;
use tempfile::tempdir;

fn digest(character: char) -> Sha256Digest {
    Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
}

fn front(task: Task, checkpoint: char) -> AcceptedFront {
    AcceptedFront {
        task,
        generation: 0,
        checkpoint_sha256: digest(checkpoint),
        parent_checkpoint_sha256: None,
        promotion_evidence_sha256: digest('e'),
        promoted_at_ms: 1,
    }
}

fn context() -> DatasetContext {
    DatasetContext {
        task: Task::Binary,
        dataset_id: "binary-dataset".into(),
        dataset_content_sha256: digest('1'),
        source_family: "held-family".into(),
        context_sha256: digest('2'),
    }
}

fn candidate(origin: Origin, family: &str, index: usize) -> PolicyCandidate {
    let program = format!("{origin:?}-{family}-{index}");
    PolicyCandidate {
        task: Task::Binary,
        dataset_id: "binary-dataset".into(),
        dataset_content_sha256: digest('1'),
        context_sha256: digest('2'),
        semantic_sha256: Sha256Digest::of_bytes(program.as_bytes()),
        canonical_program: program,
        source_family: family.into(),
        origin,
        parentage: Parentage::default(),
        producer_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('a')),
        behavior_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('a')),
        decoder_log_probability: Some(-1.0),
        proposal_rank: index as u32,
        seed: index as u64,
        signals: SelectionSignals {
            conditional_reward: 1.0 / (index + 1) as f64,
            stability: 0.9,
            incumbent_lift_probability: 0.75,
            diversity: 0.5,
            raw_uncertainty: index as f64,
        },
    }
}

#[test]
fn accepted_parent_to_actions_corpus_train_gate_and_task_acceptance() {
    let temp = tempdir().unwrap();
    let paths = ShadowPaths::new(temp.path().to_path_buf()).unwrap();
    let accepted =
        AcceptedByTaskV2::new(front(Task::Binary, 'a'), front(Task::Regression, 'b')).unwrap();
    initialize_shadow(&paths, &accepted, 1).unwrap();

    let pool = ContextProposalPool {
        context: context(),
        dl_direct: (0..8)
            .map(|index| candidate(Origin::DlDirect, "matched", index))
            .collect(),
        gp_comparator: (0..8)
            .map(|index| candidate(Origin::GpComparator, "matched", index + 100))
            .collect(),
        gp_exploration: (0..256)
            .map(|index| candidate(Origin::GpExploration, "explore", index + 1_000))
            .collect(),
    };
    let lane = build_context_lane(
        pool,
        8,
        &digest('a'),
        &digest('c'),
        &digest('d'),
        &EvaluationIndex::default(),
    )
    .unwrap();
    assert_eq!(lane.actual_d, 8);
    assert_eq!(lane.actions.len(), 256);
    assert!(lane.zero_backfill);

    let direct = lane
        .actions
        .iter()
        .find(|action| action.candidate.origin == Origin::DlDirect)
        .unwrap();
    let result = EvaluationResultV2 {
        schema: EVALUATION_RESULT_SCHEMA.into(),
        evaluation_key: direct.evaluation_key.clone(),
        result_sha256: digest('f'),
        metadata: PackedTrainingMetadata::from(&direct.candidate),
        outcome: EvaluationOutcome {
            valid: true,
            strict_win: true,
            reward: Some(1.0),
            stability: Some(0.9),
            failure_class: None,
        },
        authoritative: true,
        completed_at_ms: 2,
    };
    append_evaluation_result(&paths, &result).unwrap();
    assert_eq!(load_evaluation_index(&paths).unwrap().completed.len(), 1);

    let shard = stage_corpus_shard(&paths, b"{\"evaluation\":\"f\"}\n").unwrap();
    let pointer = advance_corpus_pointer(&paths, None, &shard, 1, 3).unwrap();
    assert_eq!(pointer.rows, 1);
    assert!(training_is_eligible(ParentEvidence {
        completed_parent_waves: 1,
        authoritative_dl_direct_outcomes: 2_048,
    }));
    validate_target_aux_mix(32, 32).unwrap();
    let metrics = TargetCheckpointMetrics {
        pairwise_accuracy: 0.7,
        genome_only_pairwise_accuracy: 0.55,
        conditional_advantage: 0.03,
        strict_win_calibration: 0.9,
        top_budget_regret: 0.1,
        failure_accuracy: 0.99,
        validity: 0.995,
        uniqueness: 0.99,
        collapse_guard_triggered: false,
    };
    assert_eq!(metrics.selection_score(), Some(0.76));

    TournamentSpec {
        task: Task::Binary,
        candidate_checkpoint_sha256: digest('3'),
        parent_checkpoint_sha256: digest('a'),
        cohort_sha256: digest('4'),
        corpus_sha256: pointer.shard_sha256.clone(),
        evaluator_sha256: digest('c'),
        protocol_sha256: digest('d'),
        proposal_sha256: digest('5'),
        context_sha256: digest('2'),
        seeds: vec![7, 11],
        budget_per_dataset: 256,
        family_disjoint: true,
        absent_from_candidate_corpus: true,
        absent_from_sealed_audit: true,
    }
    .validate()
    .unwrap();
    assert_eq!(
        promotion_decision(PromotionEvidence {
            datasets: 256,
            validity: 0.995,
            all_native: true,
            all_contextualized: true,
            direct_strict_wins: 1,
            conditional_advantage: 0.03,
            context_ablation_lower_bound: 0.01,
            latency_ratio: 1.1,
            family_utility_lower_bound: 0.01,
            family_failure_delta_upper_bound: 0.0,
        })
        .unwrap(),
        PromotionDecision::Accept
    );

    for (index, stage) in [
        LineageStage::AcceptedParent,
        LineageStage::PolicyActions,
        LineageStage::Evaluations,
        LineageStage::CorpusShards,
        LineageStage::Candidate,
        LineageStage::Tournament,
        LineageStage::AcceptedChild,
        LineageStage::NextActions,
    ]
    .into_iter()
    .enumerate()
    {
        append_lineage_transition(
            &paths,
            Task::Binary,
            0,
            digest('a'),
            stage,
            Sha256Digest::of_bytes(format!("artifact-{index}").as_bytes()),
            index as u64 + 1,
        )
        .unwrap();
    }
    let regression_before = accepted.regression;
    let advanced = compare_and_swap_accepted_file(
        &paths,
        Task::Binary,
        &digest('a'),
        digest('3'),
        digest('6'),
        10,
    )
    .unwrap();
    assert_eq!(advanced.binary.generation, 1);
    assert_eq!(advanced.regression, regression_before);
}

#[test]
fn queue_state_survives_crashes_without_duplicate_evaluator_runs() {
    let key = EvaluationKeyV2::new(
        Task::Regression,
        digest('1'),
        digest('2'),
        digest('3'),
        digest('4'),
    );
    let mut state = QueueReservationBook::default();
    state.reserve(key.clone(), "wave").unwrap();
    let mut state: QueueReservationBook =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    state.validate().unwrap();
    state.submit(&key, "wave", 1, "evaluator-1").unwrap();
    let mut state: QueueReservationBook =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    state.submit(&key, "wave", 1, "evaluator-1").unwrap();
    assert!(state.submit(&key, "wave", 1, "evaluator-2").is_err());
    state.finalize(&key, "wave", 1, digest('5')).unwrap();
    let mut state: QueueReservationBook =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    state.finalize(&key, "wave", 1, digest('5')).unwrap();
    assert_eq!(state.retry(&key, "retry").unwrap(), 2);
}

#[test]
fn every_frozen_run_identity_drift_fails_closed() {
    let expected = FrozenRunIdentity {
        checkpoint_sha256: digest('1'),
        corpus_sha256: digest('2'),
        catalog_sha256: digest('3'),
        evaluator_sha256: digest('4'),
        proposal_sha256: digest('5'),
        context_sha256: digest('6'),
    };
    expected.validate_observed(&expected).unwrap();
    for field in 0..6 {
        let mut changed = expected.clone();
        match field {
            0 => changed.checkpoint_sha256 = digest('a'),
            1 => changed.corpus_sha256 = digest('b'),
            2 => changed.catalog_sha256 = digest('c'),
            3 => changed.evaluator_sha256 = digest('d'),
            4 => changed.proposal_sha256 = digest('e'),
            5 => changed.context_sha256 = digest('f'),
            _ => unreachable!(),
        }
        assert!(expected.validate_observed(&changed).is_err());
    }
}

#[test]
fn selector_replay_must_beat_first_unique_on_both_tasks() {
    let replay = SelectorReplay {
        binary: SelectorReplayTask {
            reward_ranked_strict_wins_per_evaluation: 0.03,
            first_unique_strict_wins_per_evaluation: 0.02,
            emitted_validity: 0.995,
        },
        regression: SelectorReplayTask {
            reward_ranked_strict_wins_per_evaluation: 0.04,
            first_unique_strict_wins_per_evaluation: 0.03,
            emitted_validity: 0.999,
        },
    };
    assert!(replay.passes());
    assert_eq!(
        choose_shared_training_task(
            &AcceptedByTaskV2::new(front(Task::Binary, 'a'), front(Task::Regression, 'b')).unwrap(),
            &BTreeMap::from([
                (
                    Task::Binary,
                    ParentEvidence {
                        completed_parent_waves: 1,
                        authoritative_dl_direct_outcomes: 2_048,
                    }
                ),
                (
                    Task::Regression,
                    ParentEvidence {
                        completed_parent_waves: 0,
                        authoritative_dl_direct_outcomes: 9_999,
                    }
                ),
            ])
        ),
        Some(Task::Binary)
    );
}
