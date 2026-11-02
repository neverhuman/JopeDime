use super::*;

fn digest(value: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(value.as_bytes())
}

fn fixture(order_reversed: bool) -> TaskRatingBuildInputV1 {
    let mut players = vec![
        RatingPlayerV1 {
            player: "anchor".to_owned(),
            checkpoint_sha256: digest("anchor-checkpoint"),
            ancestry_sha256: digest("anchor-ancestry"),
        },
        RatingPlayerV1 {
            player: "candidate".to_owned(),
            checkpoint_sha256: digest("candidate-checkpoint"),
            ancestry_sha256: digest("candidate-ancestry"),
        },
    ];
    let mut contexts = (0..2)
        .map(|index| RatingContextV1 {
            context: format!("context-{index}"),
            context_sha256: digest(&format!("context-{index}")),
            source_family_sha256: digest(&format!("family-{index}")),
            near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
            evaluator_sha256: digest("evaluator"),
            protocol_sha256: digest("protocol"),
            runtime_sha256: digest("runtime"),
            bundle_sha256: digest("bundle"),
        })
        .collect::<Vec<_>>();
    let mut observations = Vec::new();
    for context in &contexts {
        for slot in 0..RATING_SLOTS {
            observations.push(RatingSlotObservationV1 {
                player: "anchor".to_owned(),
                context_sha256: context.context_sha256.clone(),
                slot,
                score: Some(0.4),
                valid: true,
                failed: false,
                latency_ms: Some(10),
            });
            if slot != 7 {
                observations.push(RatingSlotObservationV1 {
                    player: "candidate".to_owned(),
                    context_sha256: context.context_sha256.clone(),
                    slot,
                    score: Some(0.8),
                    valid: true,
                    failed: false,
                    latency_ms: Some(12),
                });
            }
        }
    }
    if order_reversed {
        players.reverse();
        contexts.reverse();
        observations.reverse();
    }
    TaskRatingBuildInputV1 {
        schema: TASK_RATING_BUILD_INPUT_SCHEMA.to_owned(),
        task: Task::Multiclass,
        rating_kind: RatingKindV1::AdaptiveSearch,
        rating_epoch: "epoch-1".to_owned(),
        fixed_anchor_player: "anchor".to_owned(),
        display_player: "candidate".to_owned(),
        proposal_selection_policy_sha256: digest("policy"),
        players,
        contexts,
        observations,
        bootstrap_seed: 7,
        bootstrap_resamples: RATING_BOOTSTRAP_RESAMPLES,
        generated_at_ms: 10,
    }
}

#[test]
fn rating_is_permutation_invariant_and_pads_missing_slots() {
    let first = build_task_rating_snapshot(fixture(false)).unwrap();
    let second = build_task_rating_snapshot(fixture(true)).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.players[0].player, "anchor");
    assert_eq!(first.players[0].rating, 1500.0);
    let candidate = first
        .players
        .iter()
        .find(|player| player.player == "candidate")
        .unwrap();
    assert!(candidate.rating > 1500.0);
    assert_eq!(candidate.validity, 14.0 / 16.0);
    assert_eq!(candidate.failure_rate, 2.0 / 16.0);
    assert!(first.policy_contexts.iter().all(|row| row.slots.len() == 8));
}

#[test]
fn timeseries_utility_and_tie_tolerance_are_directionally_correct() {
    assert_eq!(bounded_task_utility(Task::Timeseries, -0.25).unwrap(), 0.75);
    assert!(tie(1.0, 1.0 + 5e-13));
    assert!(!tie(1.0, 1.0 + 2e-12));
}

#[test]
fn solver_matches_the_archived_sotarena_bradley_terry_fixture() {
    let comparisons = (0..3)
        .flat_map(|_| {
            [
                Comparison {
                    left: 0,
                    right: 1,
                    outcome: 1.0,
                },
                Comparison {
                    left: 0,
                    right: 2,
                    outcome: 1.0,
                },
                Comparison {
                    left: 1,
                    right: 2,
                    outcome: 1.0,
                },
            ]
        })
        .collect::<Vec<_>>();
    let (ratings, convergence) = fit_bradley_terry(3, &comparisons).unwrap();
    assert!(convergence.converged);
    let shift = INITIAL_RATING - ratings[1];
    let anchored = ratings
        .into_iter()
        .map(|rating| rating + shift)
        .collect::<Vec<_>>();
    let expected = [1602.296273771328, 1500.0, 1397.703726228672];
    for (actual, expected) in anchored.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
}
