use std::collections::BTreeMap;
use std::fs;

use jope_dime_regressor::epoch6::*;
use jope_dime_regressor::epoch6_wave::*;
use jope_dime_regressor::selfplay_shadow::Task;
use jope_dime_regressor::task::TASK_SEMANTICS_SHA256;

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

fn activation() -> LaneActivationSetV1 {
    LaneActivationSetV1 {
        schema: LANE_ACTIVATION_SCHEMA.to_owned(),
        activations: Task::CANONICAL_ORDER
            .into_iter()
            .map(|task| TaskActivationV1 {
                task,
                activation_epoch: "epoch-6-proof".to_owned(),
                activation_wave: 0,
            })
            .collect(),
    }
}

#[test]
fn actual_128_192_256_routes_replay_under_retries_and_shuffled_completion() {
    let temp = tempfile::tempdir().unwrap();
    let protocol = replay();
    let activation = activation();
    for (wave, count) in [128_u32, 192, 256].into_iter().enumerate() {
        let forward: Vec<_> = (0..count).collect();
        let reverse: Vec<_> = (0..count).rev().collect();
        let first = execute_synthetic_wave(
            &temp.path().join(format!("wave-{count}-forward")),
            &protocol,
            &activation,
            "epoch-6-proof",
            wave as u64,
            count,
            &forward,
        )
        .unwrap();
        let retried = execute_synthetic_wave(
            &temp.path().join(format!("wave-{count}-reverse")),
            &protocol,
            &activation,
            "epoch-6-proof",
            wave as u64,
            count,
            &reverse,
        )
        .unwrap();
        assert_eq!(
            first.semantic_root().unwrap(),
            retried.semantic_root().unwrap()
        );
        assert_eq!(first.manifest_count, count);
        assert_eq!(first.task_counts.values().copied().sum::<u32>(), count);
        assert!(first
            .task_counts
            .values()
            .all(|task_count| *task_count == count / 4));
        assert_eq!(
            fs::read_dir(temp.path().join(format!("wave-{count}-forward/jobs")))
                .unwrap()
                .count(),
            count as usize
        );
    }
}

#[test]
fn supply_history_preserves_identity_and_rejects_skipped_transitions() {
    let temp = tempfile::tempdir().unwrap();
    let mut intent = SupplyIntentV2 {
        schema: SUPPLY_INTENT_SCHEMA.to_owned(),
        checkpoint_sha256: digest('1'),
        cohort_sha256: digest('2'),
        catalog_sha256: digest('3'),
        replay_protocol_sha256: digest('4'),
        task_activation_sha256: digest('5'),
        proposal_identity_sha256: digest('6'),
        seed_namespace_sha256: digest('7'),
        expected_manifest_count: 128,
        admission_budget: 128,
        state: SupplyIntentStateV2::Planned,
    };
    let identity = intent.identity().unwrap();
    let history = SupplyIntentHistoryV2::create(temp.path(), &intent).unwrap();
    assert!(history
        .transition(&mut intent, SupplyIntentStateV2::Published)
        .is_err());
    assert_eq!(intent.state, SupplyIntentStateV2::Planned);
    history
        .transition(&mut intent, SupplyIntentStateV2::Proposing)
        .unwrap();
    history
        .transition(&mut intent, SupplyIntentStateV2::FailedRetryable)
        .unwrap();
    history
        .transition(&mut intent, SupplyIntentStateV2::Proposing)
        .unwrap();
    assert_eq!(intent.identity().unwrap(), identity);
    assert_eq!(history.read().unwrap().len(), 4);
}
