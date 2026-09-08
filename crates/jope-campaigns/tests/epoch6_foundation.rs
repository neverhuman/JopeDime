use jope_dime_regressor::epoch6::*;
use jope_dime_regressor::selfplay_shadow::Task;
use jope_dime_regressor::task::TASK_SEMANTICS_SHA256;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;

fn digest(byte: char) -> String {
    byte.to_string().repeat(64)
}

fn profile(task: Task, maturity: LaneMaturity) -> TaskLaneProfileV1 {
    TaskLaneProfileV1 {
        schema: TASK_LANE_PROFILE_SCHEMA.to_owned(),
        task,
        maturity,
        roles: match maturity {
            LaneMaturity::Bootstrap => vec![LaneRole::GrammarImmigrant, LaneRole::GpSeed],
            LaneMaturity::Mature => vec![LaneRole::StableAnchor, LaneRole::LeastObserved],
        },
    }
}

fn replay() -> ReplayProtocolV2 {
    ReplayProtocolV2 {
        schema: REPLAY_PROTOCOL_SCHEMA.to_owned(),
        task_order: Task::CANONICAL_ORDER.to_vec(),
        task_lane_sizes: Task::CANONICAL_ORDER
            .into_iter()
            .map(|task| (task, 32))
            .collect(),
        selector_versions: Task::CANONICAL_ORDER
            .into_iter()
            .map(|task| (task, "selector-v2".to_owned()))
            .collect(),
        frozen_lane_profiles: BTreeMap::from([
            (
                Task::Regression,
                profile(Task::Regression, LaneMaturity::Mature),
            ),
            (Task::Binary, profile(Task::Binary, LaneMaturity::Mature)),
            (
                Task::Multiclass,
                profile(Task::Multiclass, LaneMaturity::Bootstrap),
            ),
            (
                Task::Timeseries,
                profile(Task::Timeseries, LaneMaturity::Bootstrap),
            ),
        ]),
        candidate_slots: CandidateSlotArithmeticV2 {
            slots_per_dataset: 60,
            common_winner_slots: vec![0, 1],
            checkpoint_proposal_slots: vec![2, 3],
        },
        gp_seed_slots: vec![4, 5],
        operator_vocabulary: vec![
            OperatorModulusV2 {
                operator: "crossover".to_owned(),
                modulus: 17,
            },
            OperatorModulusV2 {
                operator: "mutation".to_owned(),
                modulus: 31,
            },
        ],
        dl_slot_policy: DlSlotPolicyV2 {
            maximum_slots: 12,
            replacement_only: true,
            candidate_evaluation_calls: 0,
        },
        proposal_sampling_policy: "without-replacement-v2".to_owned(),
        rng_algorithm: RngAlgorithmV2::Sha256Counter,
        evaluator_protocol_sha256: digest('a'),
        task_semantics_sha256: TASK_SEMANTICS_SHA256.to_owned(),
        encoding_protocol_sha256: digest('b'),
    }
}

#[test]
fn replay_identity_excludes_operational_timing_and_domain_separates_every_item() {
    let protocol = replay();
    protocol.validate().unwrap();
    let first = protocol
        .derive_item_seed(7, Task::Binary, &digest('c'), "proposal", 2, "mutation", 0)
        .unwrap();
    let replayed = protocol
        .derive_item_seed(7, Task::Binary, &digest('c'), "proposal", 2, "mutation", 0)
        .unwrap();
    let next = protocol
        .derive_item_seed(7, Task::Binary, &digest('c'), "proposal", 2, "mutation", 1)
        .unwrap();
    assert_eq!(first, replayed);
    assert_ne!(first, next);

    let mut drift = protocol;
    drift.task_lane_sizes.remove(&Task::Timeseries);
    assert!(drift.validate().is_err());
}

#[test]
fn activation_is_explicit_and_wave_commits_use_actual_counts() {
    let activation = LaneActivationSetV1 {
        schema: LANE_ACTIVATION_SCHEMA.to_owned(),
        activations: vec![
            TaskActivationV1 {
                task: Task::Regression,
                activation_epoch: "release-a".to_owned(),
                activation_wave: 0,
            },
            TaskActivationV1 {
                task: Task::Binary,
                activation_epoch: "release-a".to_owned(),
                activation_wave: 0,
            },
        ],
    };
    activation.validate().unwrap();
    assert!(activation.is_active(Task::Binary, "release-a", 8));
    assert!(!activation.is_active(Task::Multiclass, "release-a", 8));

    let mut commit = WaveCommitV2 {
        schema: WAVE_COMMIT_SCHEMA.to_owned(),
        epoch: "release-a".to_owned(),
        wave: 8,
        cohort_plan_sha256: digest('d'),
        task_counts: BTreeMap::from([(Task::Regression, 64), (Task::Binary, 64)]),
        manifest_count: 128,
        manifest_set_sha256: digest('e'),
    };
    commit.validate().unwrap();
    commit.manifest_count = 192;
    assert!(commit.validate().is_err());
}

#[test]
fn artifact_cas_is_immutable_reverified_and_materialized_read_only() {
    let root = tempfile::tempdir().unwrap();
    let cas = ArtifactCas::new(root.path().join("cas"));
    let reference = cas
        .publish_bytes(ArtifactKind::Evaluator, &digest('f'), b"evaluator-a")
        .unwrap();
    assert_eq!(
        reference,
        cas.publish_bytes(ArtifactKind::Evaluator, &digest('f'), b"evaluator-a")
            .unwrap()
    );
    cas.verify(&reference).unwrap();
    let destination = root.path().join("job/evaluator");
    cas.materialize(&reference, &destination).unwrap();
    assert_eq!(fs::read(&destination).unwrap(), b"evaluator-a");
    assert_eq!(
        fs::metadata(destination).unwrap().permissions().mode() & 0o777,
        0o444
    );

    let object = root
        .path()
        .join("cas/objects/sha256")
        .join(&reference.object_sha256[..2])
        .join(&reference.object_sha256);
    fs::set_permissions(&object, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(&object, b"corrupt").unwrap();
    assert!(cas.verify(&reference).is_err());
}

#[test]
fn gc_and_supply_state_machines_fail_closed() {
    let roots = ArtifactGcRootsV1 {
        schema: ARTIFACT_GC_ROOTS_SCHEMA.to_owned(),
        roots: ArtifactGcRootClass::ALL
            .into_iter()
            .map(|class| ArtifactGcRootV1 {
                class,
                readable: true,
                artifacts: vec![],
            })
            .collect(),
    };
    assert!(roots.validate_for_sweep().unwrap().is_empty());
    let mut unreadable = roots;
    unreadable.roots[3].readable = false;
    assert!(unreadable.validate_for_sweep().is_err());

    let mut intent = SupplyIntentV2 {
        schema: SUPPLY_INTENT_SCHEMA.to_owned(),
        checkpoint_sha256: digest('1'),
        cohort_sha256: digest('2'),
        catalog_sha256: digest('3'),
        replay_protocol_sha256: digest('4'),
        task_activation_sha256: digest('5'),
        proposal_identity_sha256: digest('6'),
        seed_namespace_sha256: digest('7'),
        expected_manifest_count: 192,
        admission_budget: 256,
        state: SupplyIntentStateV2::Planned,
    };
    let identity = intent.identity().unwrap();
    intent.transition(SupplyIntentStateV2::Proposing).unwrap();
    intent
        .transition(SupplyIntentStateV2::FailedRetryable)
        .unwrap();
    assert_eq!(intent.identity().unwrap(), identity);
    intent.transition(SupplyIntentStateV2::Proposing).unwrap();
    intent.transition(SupplyIntentStateV2::Built).unwrap();
    intent.transition(SupplyIntentStateV2::Published).unwrap();
    intent.transition(SupplyIntentStateV2::KeeperAcked).unwrap();
    assert!(intent.transition(SupplyIntentStateV2::Published).is_err());
}

fn edge(id: &str, producer: &str, consumer: &str) -> PipelineEdgeV1 {
    PipelineEdgeV1 {
        schema: PIPELINE_EDGE_SCHEMA.to_owned(),
        edge_id: id.to_owned(),
        producer: producer.to_owned(),
        consumer: consumer.to_owned(),
        accepted_schemas: vec!["Intent.v1".to_owned()],
        claim_ttl_seconds: 60,
        runnable_slo_seconds: 30,
        progress_slo_seconds: 20,
        terminal_result_schemas: vec!["Result.v1".to_owned()],
    }
}

fn heartbeat(edge: &PipelineEdgeV1, now: u64) -> ConsumerHeartbeatV1 {
    ConsumerHeartbeatV1 {
        schema: CONSUMER_HEARTBEAT_SCHEMA.to_owned(),
        edge_id: edge.edge_id.clone(),
        consumer: edge.consumer.clone(),
        state: PipelineStageStateV2::Idle,
        process_heartbeat_unix_ms: now,
        last_state_transition_unix_ms: now,
        last_semantic_progress_unix_ms: now,
        oldest_runnable_dependency_unix_ms: None,
        oldest_unresolved_dependency_age_ms: None,
        claimed_intent_sha256: None,
        claim_expires_unix_ms: None,
    }
}

#[test]
fn liveness_tracks_semantic_progress_and_preflights_the_complete_route() {
    let now = 30_000;
    let first = edge(
        "freeze-to-executor",
        "selection-freeze",
        "promotion-executor",
    );
    let second = edge(
        "executor-to-panel",
        "promotion-executor",
        "three-panel-outcome",
    );
    let status = JopePipelineStatusV2 {
        schema: PIPELINE_STATUS_SCHEMA.to_owned(),
        generated_at_unix_ms: now,
        valid_until_unix_ms: now + 1_000,
        protected_complete_wave_count: 2,
        manifest_count: 128,
        candidate_evaluation_count: 32_768,
        predicted_cpu_seconds: 65_536,
        consumers: vec![heartbeat(&first, now), heartbeat(&second, now)],
        edges: vec![first, second],
    };
    assert_eq!(
        status
            .preflight_route("selection-freeze", "three-panel-outcome", now)
            .unwrap(),
        ["freeze-to-executor", "executor-to-panel"]
    );

    let mut stalled = status;
    stalled.consumers[0].state = PipelineStageStateV2::Running;
    stalled.consumers[0].claimed_intent_sha256 = Some(digest('8'));
    stalled.consumers[0].claim_expires_unix_ms = Some(now + 1_000);
    stalled.consumers[0].last_semantic_progress_unix_ms = 0;
    // A fresh process heartbeat does not conceal stale semantic work.
    assert!(stalled.validate_at(now).is_err());
}

#[test]
fn pipeline_status_fails_closed_on_staleness_and_dependency_age_drift() {
    let now = 30_000;
    let edge = edge("supply-to-queue", "supply", "queue");
    let mut consumer = heartbeat(&edge, now);
    consumer.state = PipelineStageStateV2::Waiting {
        dependency: "immutable evaluator".to_owned(),
        deadline_unix_ms: now + 500,
    };
    consumer.oldest_runnable_dependency_unix_ms = Some(now - 250);
    consumer.oldest_unresolved_dependency_age_ms = Some(250);
    let status = JopePipelineStatusV2 {
        schema: PIPELINE_STATUS_SCHEMA.to_owned(),
        generated_at_unix_ms: now,
        valid_until_unix_ms: now + 1_000,
        protected_complete_wave_count: 2,
        manifest_count: 128,
        candidate_evaluation_count: 32_768,
        predicted_cpu_seconds: 65_536,
        edges: vec![edge],
        consumers: vec![consumer],
    };
    status.validate_at(now + 500).unwrap();
    assert!(status.validate_at(now + 1_001).is_err());

    let mut drifted = status;
    drifted.consumers[0].oldest_unresolved_dependency_age_ms = Some(249);
    assert!(drifted.validate_at(now + 500).is_err());
}
