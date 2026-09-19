use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::PermissionsExt;

use feat_math::GenomeIr;
use jope_campaigns::jope_dlgp_best5::*;
use jope_campaigns::jope_generalized_gp::{
    raw_sha256, ArtifactStoreV1, GenomeV1, GpTaskV1, ProposalFreezeV1,
};

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn full_dataset_digest(id: &str) -> String {
    format!("{id}{}", &digest(id)[16..])
}

fn genome(index: usize) -> GenomeV1 {
    let head = match index % 4 {
        0 => format!("(ridge lambda_frac={})", (index + 1) as f64 / 100_000.0),
        1 => format!(
            "(elasticnet alpha={} l1_ratio=0.5)",
            (index + 1) as f64 / 100_000.0
        ),
        2 => format!("(krr m={})", 20 + index),
        _ => format!("(gpr kernel=linear m={} steps=10)", 20 + index),
    };
    let source = format!("(pipe X (standardize) {head})");
    let canonical = GenomeIr::parse(&source).unwrap().canonical();
    GenomeV1::from_canonical(GpTaskV1::Regression, &canonical).unwrap()
}

fn frozen_file(label: &str, sha256: String) -> FrozenFileV1 {
    FrozenFileV1 {
        path: format!("/frozen/{label}"),
        sha256,
        bytes: 1,
    }
}

fn campaign() -> CampaignFreezeV1 {
    let regression_heads = vec!["elasticnet", "gpr", "krr", "ridge"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let input: serde_json::Value =
        serde_json::from_str(include_str!("../manifests/input-manifest.v1.json")).unwrap();
    let mut datasets = input["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let id = row["dataset_hash"].as_str().unwrap();
            FrozenDatasetV1 {
                dataset_id: id.into(),
                dataset_sha256: full_dataset_digest(id),
                train_relative_path: format!("data/{id}/train.csv"),
                test_relative_path: format!("data/{id}/test.csv"),
                metadata_relative_path: format!("splits/{id}/metadata.json"),
                train_sha256: row["train_sha256"].as_str().unwrap().into(),
                test_sha256: row["test_sha256"].as_str().unwrap().into(),
                metadata_sha256: row["meta_sha256"].as_str().unwrap().into(),
                train_rows: row["train_rows"].as_u64().unwrap() as u32,
                feature_count: row["features"].as_u64().unwrap() as u32,
                source_family: format!("family-{}", index % 7),
                stage: if index < 20 {
                    DatasetStageV1::Pilot
                } else {
                    DatasetStageV1::Confirmation
                },
                current_sotarena_win: index < 6,
            }
        })
        .collect::<Vec<_>>();
    // Freeze order is the input-manifest order, not a later score ordering.
    datasets.shrink_to_fit();
    CampaignFreezeV1 {
        schema: CampaignFreezeV1::SCHEMA.into(),
        campaign_id: CAMPAIGN_ID.into(),
        input_manifest_sha256: FROZEN_INPUT_MANIFEST_SHA256.into(),
        panel_sha256: FROZEN_PANEL_SHA256.into(),
        observation_snapshot_sha256: digest("observations"),
        generated_registry_sha256: digest("registry"),
        evaluator_sha256: digest("evaluator"),
        math_policy_sha256: digest("math"),
        backend_policy_sha256: digest("backend"),
        six_win_manifest_sha256: digest("six wins"),
        proposal_generator_sha256: digest("proposal generator"),
        proposal_runtime_sha256: digest("proposal runtime"),
        sotarena_root: "/frozen/sotarena".into(),
        input_manifest_artifact: frozen_file(
            "input-manifest.json",
            FROZEN_INPUT_MANIFEST_SHA256.into(),
        ),
        panel_artifact: frozen_file("panel.txt", FROZEN_PANEL_SHA256.into()),
        observation_snapshot_artifact: frozen_file("observations.json", digest("observations")),
        six_win_manifest_artifact: frozen_file("six-wins.json", digest("six wins")),
        registry_artifact: frozen_file("registry.json", digest("registry")),
        evaluator_artifact: frozen_file("evaluator", digest("evaluator")),
        math_policy_artifact: frozen_file("math.json", digest("math")),
        backend_policy_artifact: frozen_file("backend.json", digest("backend")),
        proposal_generator_artifact: frozen_file("generator", digest("proposal generator")),
        proposal_runtime_artifact: frozen_file("runtime", digest("proposal runtime")),
        checkpoint: FrozenCheckpointV1 {
            artifact_path: "/checkpoint/must-exist-at-launch".into(),
            artifact_sha256: FROZEN_CHECKPOINT_SHA256.into(),
            artifact_bytes: 100,
            regression_head_schema_sha256: jope_campaigns::search_v3::canonical_sha256(
                &regression_heads,
            )
            .unwrap(),
            regression_heads,
            inference_protocol_sha256: digest("inference"),
        },
        datasets,
        search_seeds: SEARCH_SEEDS.to_vec(),
        bootstrap_seed: BOOTSTRAP_SEED,
        budgets: Best5ArmV1::ALL
            .into_iter()
            .map(|arm| (arm, ArmBudgetV1::frozen()))
            .collect(),
        scq: ScqPolicyV1::frozen(),
        production_pointer_writes: false,
        production_learning_ingest: false,
        legacy_strategy_state_reused: false,
    }
}

#[test]
fn campaign_freezes_exact_20_plus_22_and_retains_nine_small_datasets() {
    let campaign = campaign();
    campaign.validate().unwrap();
    assert_eq!(campaign.datasets.len(), 42);
    assert_eq!(
        campaign
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Pilot && dataset.train_rows < 300)
            .count(),
        9
    );
    let mut hash_drift = campaign.clone();
    hash_drift.datasets[0].train_sha256 = digest("substituted training bytes");
    assert!(hash_drift
        .validate()
        .unwrap_err()
        .contains("input-manifest identity or hash drift"));
    let mut filtered = campaign.clone();
    filtered
        .datasets
        .retain(|dataset| dataset.train_rows >= 300);
    assert!(filtered.validate().is_err());
    let mut substituted = campaign;
    substituted.checkpoint.artifact_sha256 = digest("newer checkpoint");
    assert!(substituted
        .validate()
        .unwrap_err()
        .contains("substitution is forbidden"));
}

#[test]
fn budget_ledger_charges_cache_hits_and_rejects_incomplete_coverage() {
    let control = BudgetLedgerV1 {
        logical_operations: 950,
        predicted_cpu_seconds: 7_200,
        mandatory_anchor_evaluations: 80,
        completed_anchor_evaluations: 80,
        mandatory_search_evaluations: 640,
        completed_search_evaluations: 640,
        mandatory_archive_evaluations: 160,
        completed_archive_evaluations: 160,
        mandatory_cv7_evaluations: 70,
        completed_cv7_evaluations: 70,
        arm_specific_operations: 0,
    };
    control.validate(Best5ArmV1::ConfidenceGpControl).unwrap();
    let mut lineage = control.clone();
    lineage.logical_operations = 1_090;
    lineage.arm_specific_operations = 140;
    lineage
        .validate(Best5ArmV1::EvolvabilityUcbLineages)
        .unwrap();
    let mut incomplete = control;
    incomplete.completed_cv7_evaluations = 69;
    incomplete.logical_operations = 949;
    assert!(incomplete
        .validate(Best5ArmV1::ConfidenceGpControl)
        .unwrap_err()
        .contains("incomplete_mandatory_coverage"));
}

fn proposal_for(dataset_sha256: &str) -> ProposalFreezeV1 {
    ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.into(),
        inference_request_sha256: digest(&format!("request-{dataset_sha256}")),
        inference_receipt_sha256: digest(&format!("receipt-{dataset_sha256}")),
        inference_ordinal: 1,
        proposals: (0..64).map(genome).collect(),
    }
}

#[test]
fn proposal_batch_normalizes_shuffled_completion_and_covers_all_42_once() {
    let campaign = campaign();
    let mut batch = ProposalBatchFreezeV1 {
        schema: ProposalBatchFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        proposals: campaign
            .datasets
            .iter()
            .rev()
            .map(|dataset| proposal_for(&dataset.dataset_sha256))
            .collect(),
    };
    assert!(batch.validate(&campaign).is_err());
    batch.normalize_completion_order(&campaign);
    batch.validate(&campaign).unwrap();
    assert_eq!(batch.proposals.len(), 42);
    assert!(batch
        .proposals
        .iter()
        .all(|proposal| proposal.inference_ordinal == 1 && proposal.proposals.len() == 64));
}

fn search_result() -> SearchResultV1 {
    let archive = (0..20)
        .map(|index| {
            let candidate = genome(index);
            let predictor = FrozenPredictorV1::single(candidate.clone());
            let ancestry = AncestryV1 {
                predictor: predictor.clone(),
                founder_semantic_sha256: candidate.semantic_sha256.clone(),
                parent_semantic_sha256s: Vec::new(),
                model_seeded: true,
            };
            ArchiveMemberV1 {
                split_evidence: (0..8)
                    .map(|split_index| CandidateEvidenceV1 {
                        predictor: predictor.clone(),
                        ancestry: ancestry.clone(),
                        prediction_receipt_sha256: digest(&format!(
                            "archive-receipt-{index}-{split_index}"
                        )),
                        split_index,
                        r2_clipped: 0.2,
                        anchor_median_r2_clipped: 0.1,
                        paired_uplift: 0.1,
                        rmse: 1.0,
                        runtime_seconds: 1.0,
                        valid: true,
                        failure: None,
                    })
                    .collect(),
                predictor,
                ancestry,
            }
        })
        .collect::<Vec<_>>();
    let finalist_semantics = archive
        .iter()
        .take(10)
        .map(|member| member.predictor.semantic_sha256().unwrap())
        .collect::<Vec<_>>();
    let winner_semantic_sha256 = finalist_semantics.iter().min().unwrap().clone();
    let finalists = finalist_semantics
        .into_iter()
        .map(|semantic_sha256| {
            let predictor = archive
                .iter()
                .find(|member| member.predictor.semantic_sha256().unwrap() == semantic_sha256)
                .unwrap()
                .predictor
                .clone();
            Cv7EvidenceV1 {
                semantic_sha256: semantic_sha256.clone(),
                predictor: predictor.clone(),
                fold_prediction_receipt_sha256s: (0..7)
                    .map(|fold| digest(&format!("cv7-receipt-{semantic_sha256}-{fold}")))
                    .collect(),
                fold_r2: vec![0.2; 7],
                pooled_oof_r2: 0.2,
                rmse: 1.0,
                runtime_seconds: 1.0,
                complexity: predictor.complexity(),
                valid: true,
            }
        })
        .collect();
    SearchResultV1 {
        schema: SearchResultV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("search-campaign"),
        proposal_freeze_sha256: digest("search-proposals"),
        arm: Best5ArmV1::ConfidenceGpControl,
        dataset_sha256: digest("search-dataset"),
        master_seed: 101,
        split_plan_sha256: digest("search-splits"),
        populations: (0..8)
            .map(|generation| GenerationPopulationV1 {
                generation,
                semantic_sha256s: (0..80)
                    .rev()
                    .map(|slot| digest(&format!("population-{generation}-{slot}")))
                    .collect(),
            })
            .collect(),
        archive,
        cv7_plan_sha256: digest("search-cv7"),
        finalists,
        winner_semantic_sha256,
        ledger: BudgetLedgerV1 {
            logical_operations: 950,
            predicted_cpu_seconds: 7_200,
            mandatory_anchor_evaluations: 80,
            completed_anchor_evaluations: 80,
            mandatory_search_evaluations: 640,
            completed_search_evaluations: 640,
            mandatory_archive_evaluations: 160,
            completed_archive_evaluations: 160,
            mandatory_cv7_evaluations: 70,
            completed_cv7_evaluations: 70,
            arm_specific_operations: 0,
        },
        test_data_accessed: false,
        ranking_state_shared: false,
    }
}

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
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::ModelProposal],
        64
    );
    assert_eq!(
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::GlobalGene],
        10
    );
    assert_eq!(
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::DiversitySeed],
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

fn target_rows(count: usize) -> Vec<TargetRowV1> {
    (0..count)
        .map(|index| TargetRowV1 {
            stable_row_id: format!("row-{index:03}"),
            atomic_group: format!("group-{index:03}"),
            target: (index as f64 / 5.0).sin() + index as f64 / 100.0,
        })
        .collect()
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
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::Elite],
        3
    );
    assert_eq!(
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::Mutation],
        58
    );
    assert_eq!(
        origins[&jope_campaigns::jope_generalized_gp::GenomeOriginV1::Crossover],
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

fn descriptor(index: usize) -> DescriptorV1 {
    let mut counts = [0u8; 8];
    counts[index % 8] = (index % 6) as u8;
    DescriptorV1 {
        head_superfamily: (index % 4) as u8,
        transform_type_counts: counts,
        transform_count_normalized: (index % 7) as f64 / 6.0,
        complexity_normalized: (index % 20) as f64 / 19.0,
        normalized_hyperparameters: [(index % 11) as f64 / 10.0, (index % 13) as f64 / 12.0],
    }
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

#[test]
fn paired_log_rmse_comparison_applies_pilot_gates_and_tiebreaks() {
    let datasets = (0..20)
        .map(|index| digest(&format!("comparison-{index}")))
        .collect::<Vec<_>>();
    let mut scores = Vec::new();
    for dataset in &datasets {
        scores.push(ComparisonScoreV1 {
            arm: Best5ArmV1::ConfidenceGpControl,
            dataset_sha256: dataset.clone(),
            rmse: 1.0,
            latency_seconds: 1.0,
            finalist_validity: 1.0,
            overall_validity: 1.0,
            strict_benchmark_win: false,
            preserved_existing_win: true,
            predicted_cpu_seconds: 10.0,
        });
        for arm in Best5ArmV1::EXPERIMENTS {
            scores.push(ComparisonScoreV1 {
                arm,
                dataset_sha256: dataset.clone(),
                rmse: 0.9,
                latency_seconds: 1.0,
                finalist_validity: 1.0,
                overall_validity: 1.0,
                strict_benchmark_win: false,
                preserved_existing_win: true,
                predicted_cpu_seconds: 10.0,
            });
        }
    }
    let comparison = compare_campaign(
        &digest("campaign comparison"),
        DatasetStageV1::Pilot,
        &datasets,
        &scores,
    )
    .unwrap();
    assert_eq!(
        comparison.decision,
        CampaignDecisionV1::PilotWinner {
            arm: Best5ArmV1::EvolvabilityUcbLineages
        }
    );
    assert!(comparison.arms.iter().all(|arm| arm.advances));
}

#[test]
fn scq_plan_has_protected_wave_and_enforces_exact_two_percent_disk_floor() {
    let campaign = campaign();
    let blocked = LaunchEvidenceV1 {
        schema: LaunchEvidenceV1::SCHEMA.into(),
        scq_doctor_success: true,
        evaluator_evidence_age_seconds: 1,
        supply_evidence_age_seconds: 1,
        max_evidence_age_seconds: 60,
        reviewed_action_acknowledged: true,
        observed_project_priority_weight: 10,
        eligible_worker_disk_free_fractions: vec![0.02, 0.019_999],
    };
    let plan = build_scq_plan(&campaign, &blocked, false, None).unwrap();
    assert!(!plan.launch_gate_passed);
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::ProtectedPilot)
            .count(),
        24
    );
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::RemainingPilot)
            .count(),
        216
    );
    assert_eq!(
        plan.jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::PilotSealedScore)
            .count(),
        0
    );

    let mut ready = blocked;
    ready.eligible_worker_disk_free_fractions = vec![0.02, 0.049];
    assert!(ready.disk_warning().unwrap());
    let released = build_scq_plan(
        &campaign,
        &ready,
        true,
        Some(Best5ArmV1::HeadFamilyMapElites),
    )
    .unwrap();
    assert!(released.launch_gate_passed);
    verify_submission_weight(&released, 10).unwrap();
    assert!(verify_submission_weight(&released, 1)
        .unwrap_err()
        .contains("submission aborted"));
    assert_eq!(
        released
            .jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::PilotSealedScore)
            .count(),
        20
    );
    assert_eq!(
        released
            .jobs
            .iter()
            .filter(|job| job.stage == ScqStageV1::ConfirmationSearch)
            .count(),
        88
    );
}

#[test]
fn residual_and_portfolio_predictors_round_trip_and_reconstruct() {
    let residual = FrozenPredictorV1::ResidualComposite {
        base: genome(0),
        residual_pipeline: genome(1),
        eta: 0.25,
    };
    let residual_values = residual
        .reconstruct(&[vec![1.0, 2.0], vec![4.0, -4.0]])
        .unwrap();
    assert_eq!(residual_values, vec![2.0, 1.0]);
    let encoded = serde_json::to_vec(&residual).unwrap();
    let decoded: FrozenPredictorV1 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, residual);
    assert_eq!(decoded.semantic_sha256(), residual.semantic_sha256());

    let portfolio = FrozenPredictorV1::SparsePortfolio {
        pipelines: vec![genome(2), genome(3), genome(4)],
        weights: vec![0.2, 0.3, 0.5],
    };
    assert_eq!(
        portfolio
            .reconstruct(&[vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]])
            .unwrap(),
        vec![2.3, 4.6]
    );
    let mut invalid = portfolio;
    if let FrozenPredictorV1::SparsePortfolio { weights, .. } = &mut invalid {
        weights[0] = -0.1;
    }
    assert!(invalid.validate().is_err());
}

#[test]
fn training_staging_rebuilds_stable_rows_without_resolving_test() {
    let mut campaign = campaign();
    campaign.sotarena_root = format!("{}/sotarena", env!("CARGO_MANIFEST_DIR"));
    let dataset = &mut campaign.datasets[0];
    dataset.train_relative_path = format!(
        "data/tabular-regression-v1/{}/train.csv",
        dataset.dataset_id
    );
    dataset.metadata_relative_path = format!(
        "splits/tabular-regression-v1/{}/metadata.json",
        dataset.dataset_id
    );
    dataset.test_relative_path = "intentionally/not/resolved/test.csv".into();
    let dataset_id = dataset.dataset_id.clone();
    let train_rows = dataset.train_rows;
    let first = build_training_table(&campaign, &dataset_id).unwrap();
    let replay = build_training_table(&campaign, &dataset_id).unwrap();
    assert_eq!(first.target_rows, replay.target_rows);
    assert_eq!(first.target_rows.len(), train_rows as usize);
    assert_eq!(
        first
            .target_rows
            .iter()
            .map(|row| &row.stable_row_id)
            .collect::<BTreeSet<_>>()
            .len(),
        first.target_rows.len()
    );
    assert!(!first.training_artifact.path.contains("test.csv"));
    assert!(!first.metadata_artifact.path.contains("test.csv"));
}

#[test]
fn dataset_level_sealed_claim_slot_rejects_competing_selection_sets() {
    let directory = tempfile::tempdir().unwrap();
    let store = ArtifactStoreV1::new(directory.path());
    let campaign = digest("claim-campaign");
    let dataset = digest("claim-dataset");
    let slot = sealed_claim_slot_identity(&campaign, &dataset).unwrap();
    let first = SealedClaimV1 {
        schema: SealedClaimV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.clone(),
        dataset_sha256: dataset.clone(),
        selection_freeze_sha256s: vec![digest("selection-a"), digest("selection-b")],
        access_ordinal: 1,
    };
    let second = SealedClaimV1 {
        selection_freeze_sha256s: vec![digest("selection-a"), digest("selection-c")],
        ..first.clone()
    };
    store.publish(&slot, &first).unwrap();
    assert!(store.publish(&slot, &first).is_ok());
    assert!(store
        .publish(&slot, &second)
        .unwrap_err()
        .contains("collision"));
}

#[test]
fn public_finalizer_bundle_shape_rejects_bare_search_result() {
    let bare = serde_json::to_vec(&search_result()).unwrap();
    assert!(serde_json::from_slice::<SearchBundleV1>(&bare).is_err());
}

#[test]
fn scq_renderer_declares_immutable_inputs_and_outputs() {
    let job = ScqJobV1 {
        identity: "best5-render-test".into(),
        stage: ScqStageV1::ProposalAll42,
        dataset_id: None,
        arm: None,
        master_seed: None,
        gpus: 1,
        dependencies: Vec::new(),
    };
    let rendered = render_scq_spec_with_io(
        &job,
        &["worker".into(), "propose-all".into()],
        "/work",
        &[ScqInputV1 {
            name: "worker".into(),
            source: "/immutable/worker".into(),
            target: "worker".into(),
            sha256: digest("worker"),
            executable: true,
        }],
        &[ScqOutputV1 {
            name: "artifacts".into(),
            source: "worker-output".into(),
            directory: true,
        }],
    );
    assert!(rendered.contains("[[input]]"));
    assert!(rendered.contains("sha256 ="));
    assert!(rendered.contains("executable = true"));
    assert!(rendered.contains("[[output]]"));
    assert!(rendered.contains("ram_mb = 8192"));
    assert!(rendered.contains("JOPE_MIN_WORKER_DISK_FREE_FRACTION_MICROS = \"20000\""));
    assert!(rendered.contains("JOPE_EXPECTED_PROJECT_PRIORITY_WEIGHT = \"10\""));
}

#[test]
fn emit_plan_checks_every_immutable_spec_and_never_submits() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("scq.log");
    let fake_scq = directory.path().join("scq");
    fs::write(
        &fake_scq,
        format!(
            "#!/bin/sh\n[ \"$1\" = check ] || exit 91\nprintf '%s\\n' \"$2\" >> '{}'\n",
            log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_scq, fs::Permissions::from_mode(0o755)).unwrap();
    let campaign = campaign();
    let evidence = LaunchEvidenceV1 {
        schema: LaunchEvidenceV1::SCHEMA.into(),
        scq_doctor_success: false,
        evaluator_evidence_age_seconds: 1,
        supply_evidence_age_seconds: 1,
        max_evidence_age_seconds: 60,
        reviewed_action_acknowledged: false,
        observed_project_priority_weight: 10,
        eligible_worker_disk_free_fractions: vec![0.09],
    };
    let plan = build_scq_plan(&campaign, &evidence, false, None).unwrap();
    let input = |name: &str| ScqInputV1 {
        name: name.into(),
        source: format!("/immutable/{name}"),
        target: name.into(),
        sha256: digest(name),
        executable: name == "worker",
    };
    let checked = emit_and_check_plan(
        &plan,
        input("worker"),
        input("campaign"),
        &directory.path().join("specs"),
        "/work",
        &fake_scq,
        |job| {
            Ok((job.stage != ScqStageV1::ProposalAll42)
                .then(|| input("proposal_batch"))
                .into_iter()
                .collect())
        },
    )
    .unwrap();
    assert_eq!(checked.len(), plan.jobs.len());
    assert_eq!(
        fs::read_to_string(log).unwrap().lines().count(),
        plan.jobs.len()
    );
    assert!(checked.iter().all(|path| {
        let spec = fs::read_to_string(path).unwrap();
        spec.contains("[[input]]") && spec.contains("[[output]]")
    }));
}

fn fake_evaluator(path: &std::path::Path) -> FrozenFileV1 {
    let source = r#"#!/usr/bin/python3
import argparse, hashlib, json, pathlib, struct
p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--threads')
p.add_argument('--timeout-seconds')
a = p.parse_args()
manifest_bytes = pathlib.Path(a.manifest).read_bytes()
m = json.loads(manifest_bytes)
out = pathlib.Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
train = pathlib.Path(m['train_csv']).read_text().splitlines()
test = pathlib.Path(m['test_csv']).read_text().splitlines()
center = sum(float(row.split(',')[-1]) for row in train) / len(train)
rows = []
for candidate in m['candidates']:
    raw = b''.join(struct.pack('<d', center) for _ in test)
    name = candidate['semantic_sha256'] + '.bin'
    (out / name).write_bytes(raw)
    rows.append({
        'schema': 'JopeFlywheelCorpusEvalReceipt.v1',
        'task': m['task'],
        'dataset_hash': m['dataset_hash'],
        'semantic_sha256': candidate['semantic_sha256'],
        'predictions_sha256': hashlib.sha256(raw).hexdigest(),
        'predictions_path': name,
        'prediction_count': len(test),
        'valid': True,
        'failure': None,
        'failure_message': None,
        'runtime_seconds': 0.0,
        'train_sha256': m['train_sha256'],
        'test_sha256': m['test_sha256'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    })
(out / (m['dataset_hash'] + '.receipts.jsonl')).write_text(
    ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
)
(out / (m['dataset_hash'] + '.done')).write_text(json.dumps({
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()
}, separators=(',', ':')))
"#;
    fs::write(path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    let bytes = fs::read(path).unwrap();
    FrozenFileV1 {
        path: path.to_string_lossy().into_owned(),
        sha256: raw_sha256(&bytes),
        bytes: bytes.len() as u64,
    }
}

#[test]
fn all_six_arms_complete_eight_generation_synthetic_worker_with_exact_ledgers() {
    let directory = tempfile::tempdir().unwrap();
    let mut campaign = campaign();
    campaign.sotarena_root = format!("{}/sotarena", env!("CARGO_MANIFEST_DIR"));
    let dataset_id = campaign.datasets[0].dataset_id.clone();
    campaign.datasets[0].train_relative_path =
        format!("data/tabular-regression-v1/{dataset_id}/train.csv");
    campaign.datasets[0].metadata_relative_path =
        format!("splits/tabular-regression-v1/{dataset_id}/metadata.json");
    campaign.datasets[0].test_relative_path =
        format!("data/tabular-regression-v1/{dataset_id}/test.csv");
    let evaluator = fake_evaluator(&directory.path().join("corpus-eval"));
    campaign.evaluator_sha256 = evaluator.sha256.clone();
    campaign.evaluator_artifact = evaluator;

    let protocol_id = digest("synthetic-observation-protocol");
    let observations = (64..80)
        .flat_map(|index| {
            let protocol_id = protocol_id.clone();
            (0..3).map(move |support| AnchorObservationV1 {
                observation_snapshot_sha256: protocol_id.clone(),
                dataset_sha256: digest(&format!("anchor-source-{index}-{support}")),
                source_family: format!("anchor-family-{}", support % 2),
                genome: genome(index),
                within_dataset_percentile: 0.9 - index as f64 / 10_000.0,
                strict_win: true,
            })
        })
        .collect::<Vec<_>>();
    let observation_path = directory.path().join("observations.json");
    fs::write(
        &observation_path,
        serde_json::to_vec(&serde_json::json!({
            "snapshot_sha256": protocol_id,
            "observations": observations,
            "diversity_seeds": [],
        }))
        .unwrap(),
    )
    .unwrap();
    let final_bytes = fs::read(&observation_path).unwrap();
    campaign.observation_snapshot_sha256 = digest("synthetic-observation-protocol");
    campaign.observation_snapshot_artifact.path = observation_path.to_string_lossy().into_owned();
    campaign.observation_snapshot_artifact.sha256 = raw_sha256(&final_bytes);
    campaign.observation_snapshot_artifact.bytes = final_bytes.len() as u64;

    let proposal_batch = ProposalBatchFreezeV1 {
        schema: ProposalBatchFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        proposals: campaign
            .datasets
            .iter()
            .map(|dataset| proposal_for(&dataset.dataset_sha256))
            .collect(),
    };
    proposal_batch.validate(&campaign).unwrap();
    let artifact_root = directory.path().join("artifacts");
    let work_root = directory.path().join("work");
    let store = ArtifactStoreV1::new(&artifact_root);
    store
        .publish(
            &proposal_batch.identity(&campaign).unwrap(),
            &proposal_batch,
        )
        .unwrap();
    let mut control_bundle = None;
    let mut selections = Vec::new();
    for arm in Best5ArmV1::ALL {
        let bundle = run_search_one(
            &campaign,
            &proposal_batch,
            &dataset_id,
            arm,
            101,
            &artifact_root,
            &work_root,
        )
        .unwrap();
        assert_eq!(bundle.result.populations.len(), 8);
        assert!(bundle
            .result
            .populations
            .iter()
            .all(|population| population.semantic_sha256s.len() == 80));
        bundle.result.ledger.validate(arm).unwrap();
        assert_eq!(
            bundle.prediction_receipts.len(),
            bundle.result.ledger.logical_operations as usize
        );
        if arm == Best5ArmV1::CrossFittedResidualDlgp {
            assert!(matches!(
                bundle
                    .result
                    .finalists
                    .iter()
                    .find(|finalist| {
                        finalist.semantic_sha256 == bundle.result.winner_semantic_sha256
                    })
                    .unwrap()
                    .predictor,
                FrozenPredictorV1::ResidualComposite { .. }
            ));
        }
        if arm == Best5ArmV1::SparseOofPortfolioGp {
            assert!(bundle.result.finalists.iter().any(|finalist| matches!(
                finalist.predictor,
                FrozenPredictorV1::SparsePortfolio { .. }
            )));
        }
        if arm == Best5ArmV1::ConfidenceGpControl {
            verify_search_bundle(&campaign, &proposal_batch, &bundle, &artifact_root).unwrap();
            control_bundle = Some(bundle);
        } else {
            let winner = bundle
                .result
                .finalists
                .iter()
                .find(|candidate| candidate.semantic_sha256 == bundle.result.winner_semantic_sha256)
                .unwrap();
            let choices = SEARCH_SEEDS
                .into_iter()
                .map(|master_seed| SeedChoiceV1 {
                    master_seed,
                    search_result_sha256: bundle.result.identity().unwrap(),
                    winner_semantic_sha256: winner.semantic_sha256.clone(),
                    winner_predictor: winner.predictor.clone(),
                    pooled_oof_r2: winner.pooled_oof_r2,
                    fold_r2_sd: 0.0,
                    rmse: winner.rmse,
                    runtime_seconds: winner.runtime_seconds,
                    complexity: winner.complexity,
                })
                .collect::<Vec<_>>();
            selections.push(SelectionFreezeV1 {
                schema: SelectionFreezeV1::SCHEMA.into(),
                campaign_freeze_sha256: campaign.identity().unwrap(),
                arm,
                dataset_sha256: campaign.datasets[0].dataset_sha256.clone(),
                selected_seed: 101,
                winner_semantic_sha256: winner.semantic_sha256.clone(),
                winner_predictor: winner.predictor.clone(),
                choices,
            });
        }
    }

    let control_bundle = control_bundle.unwrap();
    let control_winner = control_bundle
        .result
        .finalists
        .iter()
        .find(|candidate| candidate.semantic_sha256 == control_bundle.result.winner_semantic_sha256)
        .unwrap();
    selections.push(SelectionFreezeV1 {
        schema: SelectionFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        arm: Best5ArmV1::ConfidenceGpControl,
        dataset_sha256: campaign.datasets[0].dataset_sha256.clone(),
        choices: SEARCH_SEEDS
            .into_iter()
            .map(|master_seed| SeedChoiceV1 {
                master_seed,
                search_result_sha256: control_bundle.result.identity().unwrap(),
                winner_semantic_sha256: control_winner.semantic_sha256.clone(),
                winner_predictor: control_winner.predictor.clone(),
                pooled_oof_r2: control_winner.pooled_oof_r2,
                fold_r2_sd: 0.0,
                rmse: control_winner.rmse,
                runtime_seconds: control_winner.runtime_seconds,
                complexity: control_winner.complexity,
            })
            .collect(),
        selected_seed: 101,
        winner_semantic_sha256: control_winner.semantic_sha256.clone(),
        winner_predictor: control_winner.predictor.clone(),
    });
    let sealed = run_sealed_one(&campaign, selections, &artifact_root, &work_root).unwrap();
    verify_sealed_bundle(&campaign, &sealed, &artifact_root).unwrap();
    assert_eq!(sealed.scores.len(), 6);

    let warm_replay = run_search_one(
        &campaign,
        &proposal_batch,
        &dataset_id,
        Best5ArmV1::ConfidenceGpControl,
        101,
        &artifact_root,
        &work_root,
    )
    .unwrap();
    assert_eq!(
        control_bundle.identity().unwrap(),
        warm_replay.identity().unwrap()
    );

    let mut missing_receipt = control_bundle.clone();
    missing_receipt.prediction_receipts.pop();
    assert!(missing_receipt.validate().is_err());

    let mut altered_metric = control_bundle.clone();
    altered_metric.result.archive[0].split_evidence[0].rmse += 1.0;
    assert!(
        verify_search_bundle(&campaign, &proposal_batch, &altered_metric, &artifact_root).is_err()
    );

    let mut foreign = control_bundle;
    let foreign_campaign = digest("foreign-campaign");
    foreign.staging.campaign_freeze_sha256 = foreign_campaign.clone();
    foreign.result.campaign_freeze_sha256 = foreign_campaign;
    assert!(verify_search_bundle(&campaign, &proposal_batch, &foreign, &artifact_root).is_err());
}
