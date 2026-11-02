use super::*;

#[test]
fn search_artifact_root_is_replay_stable_under_shuffled_completion() {
    let mut first = search_result();
    let mut replay = first.clone();
    replay.archive.reverse();
    replay.finalists.reverse();
    for population in &mut replay.populations {
        population.semantic_sha256s.reverse();
    }
    for member in &mut replay.archive {
        member.split_evidence.reverse();
    }
    first.normalize_completion_order();
    replay.normalize_completion_order();
    first.validate().unwrap();
    replay.validate().unwrap();
    assert_eq!(first, replay);
    assert_eq!(first.identity().unwrap(), replay.identity().unwrap());

    let mut leaked = first.clone();
    leaked.test_data_accessed = true;
    assert!(leaked.validate().unwrap_err().contains("sealed-test"));
    let mut incomplete = first;
    incomplete.archive[0].split_evidence.pop();
    assert!(incomplete.validate().unwrap_err().contains("eight-split"));
}

#[test]
fn sealed_score_requires_a_bound_selection_and_exactly_one_test_access() {
    let winner_predictor = FrozenPredictorV1::single(genome(0));
    let winner = winner_predictor.semantic_sha256().unwrap();
    let runner_up_predictor = FrozenPredictorV1::single(genome(1));
    let selection = SelectionFreezeV1 {
        schema: SelectionFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("sealed-campaign"),
        arm: Best5ArmV1::HeadFamilyMapElites,
        dataset_sha256: digest("sealed-dataset"),
        choices: vec![
            SeedChoiceV1 {
                master_seed: 101,
                search_result_sha256: digest("sealed-search-101"),
                winner_semantic_sha256: winner.clone(),
                winner_predictor: winner_predictor.clone(),
                pooled_oof_r2: 0.5,
                fold_r2_sd: 0.1,
                rmse: 1.0,
                runtime_seconds: 2.0,
                complexity: 3,
            },
            SeedChoiceV1 {
                master_seed: 202,
                search_result_sha256: digest("sealed-search-202"),
                winner_semantic_sha256: runner_up_predictor.semantic_sha256().unwrap(),
                winner_predictor: runner_up_predictor,
                pooled_oof_r2: 0.4,
                fold_r2_sd: 0.1,
                rmse: 1.1,
                runtime_seconds: 2.0,
                complexity: 3,
            },
        ],
        selected_seed: 101,
        winner_semantic_sha256: winner.clone(),
        winner_predictor: winner_predictor.clone(),
    };
    selection.validate().unwrap();
    let score = SealedScoreV1 {
        schema: SealedScoreV1::SCHEMA.into(),
        campaign_freeze_sha256: selection.campaign_freeze_sha256.clone(),
        selection_freeze_sha256: selection.identity().unwrap(),
        arm: selection.arm,
        dataset_sha256: selection.dataset_sha256.clone(),
        test_sha256: digest("sealed-test"),
        winner_semantic_sha256: winner.clone(),
        predictor: winner_predictor.clone(),
        test_access_ordinal: 1,
        r2: 0.5,
        rmse: 1.0,
        latency_seconds: 0.1,
        validity: 1.0,
        ancestry: AncestryV1 {
            predictor: winner_predictor,
            founder_semantic_sha256: winner,
            parent_semantic_sha256s: Vec::new(),
            model_seeded: true,
        },
    };
    score.validate(&selection).unwrap();
    let mut early_or_repeated = score.clone();
    early_or_repeated.test_access_ordinal = 0;
    assert!(early_or_repeated
        .validate(&selection)
        .unwrap_err()
        .contains("exactly once"));
    let mut unbound = score;
    unbound.selection_freeze_sha256 = digest("different-selection");
    assert!(unbound
        .validate(&selection)
        .unwrap_err()
        .contains("does not bind"));
}

#[test]
fn one_inference_and_generation_zero_preserve_exact_64_10_6() {
    let proposal = ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: full_dataset_digest(PILOT_DATASET_IDS[0]),
        checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.into(),
        inference_request_sha256: digest("request"),
        inference_receipt_sha256: digest("receipt"),
        inference_ordinal: 1,
        proposals: (0..64).map(genome).collect(),
    };
    let mut anchors = vec![RankedAnchorV1 {
        genome: proposal.proposals[0].clone(),
        support_shrunk_percentile_lcb: 1.0,
        worst_percentile: 1.0,
        dataset_support: 3,
        source_family_support: 2,
        strict_wins: 1,
    }];
    anchors.extend((64..75).map(|index| RankedAnchorV1 {
        genome: genome(index),
        support_shrunk_percentile_lcb: 0.9,
        worst_percentile: 0.8,
        dataset_support: 3,
        source_family_support: 2,
        strict_wins: 1,
    }));
    let mut seeds = vec![proposal.proposals[1].clone(), anchors[1].genome.clone()];
    seeds.extend((75..82).map(genome));
    let zero = build_generation_zero(&proposal, &anchors, &seeds).unwrap();
    let origins = zero
        .population
        .iter()
        .fold(BTreeMap::new(), |mut values, member| {
            *values.entry(member.ancestry.origin).or_insert(0usize) += 1;
            values
        });
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::ModelProposal],
        64
    );
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::GlobalGene],
        10
    );
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::DiversitySeed],
        6
    );
    assert_eq!(
        zero.population
            .iter()
            .map(|member| &member.genome.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        80
    );

    let mut malformed = proposal;
    malformed.inference_ordinal = 2;
    assert!(malformed.validate().is_err());
}

#[test]
fn anchor_ranking_excludes_target_dataset_and_source_family() {
    let snapshot = digest("snapshot");
    let target = digest("target");
    let strong = genome(0);
    let weak = genome(1);
    let mut observations = Vec::new();
    for (index, family) in ["family-b", "family-c", "family-b"].into_iter().enumerate() {
        observations.push(AnchorObservationV1 {
            observation_snapshot_sha256: snapshot.clone(),
            dataset_sha256: digest(&format!("source-{index}")),
            source_family: family.into(),
            genome: strong.clone(),
            within_dataset_percentile: 0.8 + index as f64 / 100.0,
            strict_win: true,
        });
    }
    for index in 0..4 {
        observations.push(AnchorObservationV1 {
            observation_snapshot_sha256: snapshot.clone(),
            dataset_sha256: if index == 3 {
                target.clone()
            } else {
                digest(&format!("leak-{index}"))
            },
            source_family: "target-family".into(),
            genome: weak.clone(),
            within_dataset_percentile: 1.0,
            strict_win: true,
        });
    }
    let ranked =
        rank_anchor_observations(&target, "target-family", &snapshot, &observations).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].genome.semantic_sha256, strong.semantic_sha256);
    assert_eq!(ranked[0].dataset_support, 3);
    assert_eq!(ranked[0].source_family_support, 2);
}

#[test]
fn search_splits_are_deterministic_atomic_and_seed_separated() {
    let rows = target_rows(70);
    let dataset = digest("split dataset");
    let training = digest("split training");
    let first = build_search_split_plan(&dataset, &training, 101, &rows).unwrap();
    let replay = build_search_split_plan(&dataset, &training, 101, &rows).unwrap();
    let other = build_search_split_plan(&dataset, &training, 202, &rows).unwrap();
    assert_eq!(first, replay);
    assert_ne!(first, other);
    assert_eq!(first.splits.len(), 8);
    assert!(first
        .splits
        .iter()
        .all(|split| split.validation_row_ids.len() == 14));
    let cv7 = build_cv7_plan(&dataset, &training, 101, &rows).unwrap();
    cv7.validate().unwrap();
    assert_eq!(cv7.0.splits.len(), 7);
}

#[test]
fn lineage_ucb_uses_paired_improvement_and_exact_schedule() {
    let credits = (0..20)
        .map(|index| LineageCreditV1 {
            founder_semantic_sha256: digest(&format!("founder-{index}")),
            paired_improvements: if index == 0 {
                vec![0.8, 0.7]
            } else {
                vec![0.0, 0.0]
            },
        })
        .collect::<Vec<_>>();
    let allocation = allocate_lineage_offspring(&credits).unwrap();
    assert_eq!(
        allocation.iter().map(|value| value.mutations).sum::<u32>(),
        58
    );
    assert_eq!(
        allocation.iter().map(|value| value.crossovers).sum::<u32>(),
        19
    );
    assert!(allocation.iter().all(|value| value.mutations >= 1));
    assert!(allocation[0].mutations + allocation[0].crossovers > 1);
    assert_eq!(allocation, allocate_lineage_offspring(&credits).unwrap());
}

#[test]
fn control_uses_semantic_three_fifty_eight_nineteen_schedule() {
    let ranked = (0..80).map(genome).collect::<Vec<_>>();
    let first = build_control_offspring(&ranked, 1, 101).unwrap();
    let replay = build_control_offspring(&ranked, 1, 101).unwrap();
    assert_eq!(first, replay);
    assert_eq!(first.len(), 80);
    let origins = first.iter().fold(BTreeMap::new(), |mut counts, member| {
        *counts.entry(member.ancestry.origin).or_insert(0usize) += 1;
        counts
    });
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::Elite],
        3
    );
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::Mutation],
        58
    );
    assert_eq!(
        origins[&jope_dime_regressor::jope_generalized_gp::GenomeOriginV1::Crossover],
        19
    );
}

#[test]
fn map_elites_protects_niches_caps_families_and_crosses_families() {
    let candidates = (0..80)
        .map(|index| MapEliteCandidateV1 {
            semantic_sha256: digest(&format!("map-{index}")),
            head_superfamily: (index % 4) as u8,
            transform_count: (index % 7) as u8,
            score: 1.0 - index as f64 / 100.0,
        })
        .collect::<Vec<_>>();
    let schedule = build_map_elites_schedule(&candidates).unwrap();
    assert_eq!(
        schedule.protected_semantic_sha256s.len() + schedule.slots.len(),
        80
    );
    let crossovers = schedule
        .slots
        .iter()
        .filter(|slot| slot.operation == MapOperationV1::Crossover)
        .collect::<Vec<_>>();
    for (index, crossover) in crossovers.iter().enumerate() {
        assert_eq!(crossover.cross_head_superfamilies, (index + 1) % 4 == 0);
    }
    let ordinary = schedule.slots.iter().fold([0usize; 4], |mut values, slot| {
        values[slot.parent_niche.0 as usize] += 1;
        values
    });
    assert!(ordinary.iter().all(|count| *count <= 20));
}

#[test]
fn surrogate_selection_is_deterministic_and_never_promotes_predictions_to_truth() {
    let generated = (0..320)
        .map(|index| SurrogateCandidateV1 {
            semantic_sha256: digest(&format!("generated-{index}")),
            descriptor: descriptor(index),
            true_uplift: None,
        })
        .collect::<Vec<_>>();
    let historical = (0..64)
        .map(|index| SurrogateCandidateV1 {
            semantic_sha256: digest(&format!("historical-{index}")),
            descriptor: descriptor(index + 3),
            true_uplift: Some(index as f64 / 100.0),
        })
        .collect::<Vec<_>>();
    let elites = generated[..3]
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<Vec<_>>();
    let first = build_surrogate_selection(&generated, &historical, &elites, 101).unwrap();
    let replay = build_surrogate_selection(&generated, &historical, &elites, 101).unwrap();
    assert_eq!(first, replay);
    assert_eq!(first.elite_semantic_sha256s.len(), 3);
    assert_eq!(first.ucb.len(), 60);
    assert_eq!(first.novelty.len(), 17);
    assert!(generated
        .iter()
        .all(|candidate| candidate.true_uplift.is_none()));
}

#[test]
fn residual_targets_are_oof_and_outer_validation_never_self_trained() {
    let rows = target_rows(35);
    let train = rows[..25]
        .iter()
        .map(|row| row.stable_row_id.clone())
        .collect::<Vec<_>>();
    let validation = rows[25..]
        .iter()
        .map(|row| row.stable_row_id.clone())
        .collect::<Vec<_>>();
    let targets = rows
        .iter()
        .map(|row| (row.stable_row_id.clone(), row.target))
        .collect::<BTreeMap<_, _>>();
    let residuals =
        cross_fitted_residual_targets(&rows, &train, &validation, 101, |fit, predict| {
            assert!(fit.iter().all(|row| !predict.contains(row)));
            let center = fit.iter().map(|row| targets[row]).sum::<f64>() / fit.len() as f64;
            Ok(vec![center; predict.len()])
        })
        .unwrap();
    assert_eq!(residuals.inner_folds, 5);
    assert_eq!(residuals.oof_training.len(), 25);
    assert_eq!(residuals.validation.len(), 10);
    assert!(residuals
        .oof_training
        .iter()
        .chain(&residuals.validation)
        .all(|row| row.fitted_without_row));
}

#[test]
fn sparse_portfolio_is_nonnegative_convex_and_at_most_four() {
    let actual = (0..28).map(|index| index as f64 / 10.0).collect::<Vec<_>>();
    let lanes = [
        PortfolioLaneV1::ConfidenceLcb,
        PortfolioLaneV1::CumulativeMean,
        PortfolioLaneV1::WorstSplit,
        PortfolioLaneV1::SemanticNovelty,
    ];
    let finalists = (0..8)
        .map(|index| PortfolioMemberV1 {
            semantic_sha256: digest(&format!("portfolio-{index}")),
            lane: lanes[index / 2],
            oof_predictions: actual
                .iter()
                .enumerate()
                .map(|(row, value)| value + ((row + index) % 3) as f64 * 0.01)
                .collect(),
        })
        .collect::<Vec<_>>();
    let fit = fit_sparse_portfolio(&actual, &finalists).unwrap();
    fit.validate().unwrap();
    assert!(fit.weights.iter().all(|weight| *weight >= 0.0));
    assert!((fit.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!(fit.weights.len() <= 4);
}
