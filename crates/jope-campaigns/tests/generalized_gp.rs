use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use feat_math::GenomeIr;
use jope_dime_regressor::jope_generalized_gp::*;
use jope_dime_regressor::search_v3::PredictionObjectV1;
use tempfile::TempDir;

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn genome(task: GpTaskV1, index: usize) -> GenomeV1 {
    let head = match task {
        GpTaskV1::Regression => "ridge",
        GpTaskV1::Binary => "logit_ridge",
    };
    let source = format!(
        "(pipe X (standardize) ({head} lambda_frac={}))",
        (index + 1) as f64 / 10_000.0
    );
    let canonical = GenomeIr::parse(&source).unwrap().canonical();
    GenomeV1::from_canonical(task, &canonical).unwrap()
}

fn rows(task: GpTaskV1, count: usize) -> Vec<SplitRowV1> {
    (0..count)
        .map(|index| SplitRowV1 {
            stable_row_id: format!("row-{index:03}"),
            atomic_group: format!("group-{:03}", index / 2),
            regression_target: (task == GpTaskV1::Regression).then_some((index as f64 / 5.0).sin()),
            binary_target: (task == GpTaskV1::Binary).then_some((index % 2) as u8),
        })
        .collect()
}

fn proposal(task: GpTaskV1) -> ProposalFreezeV1 {
    ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: digest(&format!("dataset-{task:?}")),
        checkpoint_sha256: digest(&format!("checkpoint-{task:?}")),
        inference_request_sha256: digest(&format!("request-{task:?}")),
        inference_receipt_sha256: digest(&format!("receipt-{task:?}")),
        inference_ordinal: 1,
        proposals: (0..64).map(|index| genome(task, index)).collect(),
    }
}

fn generation_zero(task: GpTaskV1) -> GenerationZeroV1 {
    let proposal = proposal(task);
    let global = (64..74)
        .map(|index| GlobalGeneRankV1 {
            genome: genome(task, index),
            support_shrunk_lower_bound: 0.8,
            worst_percentile: 0.7,
            dataset_support: 3,
            source_family_support: 2,
            strict_wins: 1,
        })
        .collect::<Vec<_>>();
    let diversity = (74..80)
        .map(|index| genome(task, index))
        .collect::<Vec<_>>();
    build_generation_zero(&proposal, &global, &diversity).unwrap()
}

fn evaluation(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    generation: u32,
    split_index: u32,
    split_sha256: &str,
    candidate: &GenomeV1,
    uplift: f64,
) -> CandidateSplitEvaluationV1 {
    CandidateSplitEvaluationV1 {
        schema: CandidateSplitEvaluationV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        generation,
        split_index,
        split_sha256: split_sha256.into(),
        candidate: candidate.clone(),
        prediction_object_sha256: digest(&format!(
            "prediction|{}|{generation}|{split_index}",
            candidate.semantic_sha256
        )),
        candidate_skill: 0.1 + uplift,
        anchor_median_skill: 0.1,
        paired_uplift: uplift,
        secondary_metric: 1.0 - uplift,
        valid: true,
        invalid_reason: None,
        runtime: RuntimeObservationV1 {
            cpu_seconds: 0.1,
            winner_latency_seconds: 0.1,
            timed_out: false,
        },
    }
}

#[test]
fn real_regression_benchmark_has_no_300_row_cut() {
    let mut datasets = Vec::new();
    for index in 0..42 {
        datasets.push(CatalogDatasetV1 {
            dataset_sha256: digest(&format!("sotarena-{index}")),
            task: GpTaskV1::Regression,
            partition: CatalogPartitionV1::Development,
            origin: DatasetOriginV1::Real,
            source_family: format!("family-{}", index % 5),
            row_count: if index < 20 { 20 + index } else { 500 + index },
            feature_count: 2 + index,
            missing_fraction: index as f64 / 100.0,
            target_distribution: index as f64 / 10.0,
            sotarena_dataset: true,
            current_strict_win: index < 6,
        });
    }
    datasets.push(CatalogDatasetV1 {
        dataset_sha256: digest("synthetic"),
        task: GpTaskV1::Regression,
        partition: CatalogPartitionV1::Development,
        origin: DatasetOriginV1::Synthetic,
        source_family: "generator".into(),
        row_count: 10_000,
        feature_count: 10,
        missing_fraction: 0.0,
        target_distribution: 0.0,
        sotarena_dataset: false,
        current_strict_win: false,
    });
    let freeze = freeze_catalog(digest("observations"), datasets).unwrap();
    assert_eq!(freeze.datasets.len(), 42);
    assert_eq!(
        freeze
            .datasets
            .iter()
            .filter(|row| row.row_count < 300)
            .count(),
        20
    );
    assert_eq!(
        freeze
            .datasets
            .iter()
            .filter(|row| row.current_strict_win)
            .count(),
        6
    );
    assert_eq!(freeze.counts.development_regression, 42);
}

#[test]
fn pilot_and_tranche_freezes_are_deterministic_and_phase_ordered() {
    let mut datasets = Vec::new();
    for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
        for index in 0..12 {
            datasets.push(CatalogDatasetV1 {
                dataset_sha256: digest(&format!("coverage-{task:?}-{index}")),
                task,
                partition: CatalogPartitionV1::Development,
                origin: DatasetOriginV1::Real,
                source_family: format!("family-{}", index % 3),
                row_count: 10 + index,
                feature_count: 1 + index,
                missing_fraction: index as f64 / 20.0,
                target_distribution: index as f64 / 11.0,
                sotarena_dataset: index < 2,
                current_strict_win: false,
            });
        }
        datasets.push(CatalogDatasetV1 {
            dataset_sha256: digest(&format!("validation-{task:?}")),
            task,
            partition: CatalogPartitionV1::Validation,
            origin: DatasetOriginV1::Real,
            source_family: "validation-family".into(),
            row_count: 8,
            feature_count: 2,
            missing_fraction: 0.0,
            target_distribution: 0.5,
            sotarena_dataset: false,
            current_strict_win: false,
        });
    }
    let catalog = freeze_catalog(digest("pilot observations"), datasets).unwrap();
    let first = select_max_min_pilot(&catalog, 2).unwrap();
    let second = select_max_min_pilot(&catalog, 2).unwrap();
    assert_eq!(first, second);
    assert!(first.replacement_allowed());
    assert_eq!(first.selected[&GpTaskV1::Regression].len(), 10);
    assert_eq!(first.alternates[&GpTaskV1::Binary].len(), 2);

    let validation_without_development = build_tranche_plan(
        &catalog,
        &digest("tranche campaign"),
        CampaignArmV1::RotatingMean,
        CatalogPartitionV1::Validation,
        &digest("pilot comparison"),
        None,
        Vec::new(),
    );
    assert!(validation_without_development
        .unwrap_err()
        .contains("before development completion"));
    let validation = build_tranche_plan(
        &catalog,
        &digest("tranche campaign"),
        CampaignArmV1::RotatingMean,
        CatalogPartitionV1::Validation,
        &digest("pilot comparison"),
        Some(digest("development complete")),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(validation.tranches.len(), 1);
    assert_eq!(validation.tranches[0].dataset_sha256s.len(), 2);
}

#[test]
fn one_inference_and_exact_generation_zero_are_enforced() {
    let proposal = proposal(GpTaskV1::Regression);
    proposal.validate().unwrap();
    let mut ledger = InferenceLedgerV1::empty();
    let first = ledger.record(&proposal).unwrap();
    assert_eq!(ledger.record(&proposal).unwrap(), first);
    let mut second = proposal.clone();
    second.inference_receipt_sha256 = digest("different inference");
    assert!(ledger
        .record(&second)
        .unwrap_err()
        .contains("one-inference"));

    let mut global = vec![GlobalGeneRankV1 {
        genome: proposal.proposals[0].clone(),
        support_shrunk_lower_bound: 1.0,
        worst_percentile: 1.0,
        dataset_support: 3,
        source_family_support: 2,
        strict_wins: 1,
    }];
    global.extend((64..74).map(|index| GlobalGeneRankV1 {
        genome: genome(GpTaskV1::Regression, index),
        support_shrunk_lower_bound: 0.9,
        worst_percentile: 0.8,
        dataset_support: 3,
        source_family_support: 2,
        strict_wins: 1,
    }));
    let mut diversity = vec![proposal.proposals[1].clone()];
    diversity.extend((74..80).map(|index| genome(GpTaskV1::Regression, index)));
    let zero = build_generation_zero(&proposal, &global, &diversity).unwrap();
    let origins = zero
        .population
        .iter()
        .fold(BTreeMap::new(), |mut map, member| {
            *map.entry(member.ancestry.origin).or_insert(0usize) += 1;
            map
        });
    assert_eq!(origins[&GenomeOriginV1::ModelProposal], 64);
    assert_eq!(origins[&GenomeOriginV1::GlobalGene], 10);
    assert_eq!(origins[&GenomeOriginV1::DiversitySeed], 6);
    assert_eq!(
        zero.population
            .iter()
            .map(|member| &member.genome.semantic_sha256)
            .collect::<BTreeSet<_>>()
            .len(),
        80
    );

    let mut wrong_task = proposal.clone();
    wrong_task.proposals[0] = genome(GpTaskV1::Binary, 0);
    assert!(wrong_task.validate().is_err());
}

#[test]
fn global_ranking_excludes_target_and_requires_cross_family_support() {
    let task = GpTaskV1::Regression;
    let target = digest("target");
    let snapshot = digest("snapshot");
    let strong = genome(task, 0);
    let weak = genome(task, 1);
    let mut observations = Vec::new();
    for index in 0..4 {
        observations.push(GlobalGeneObservationV1 {
            observation_snapshot_sha256: snapshot.clone(),
            dataset_sha256: if index == 3 {
                target.clone()
            } else {
                digest(&format!("source-{index}"))
            },
            source_family: format!("family-{}", index % 2),
            genome: strong.clone(),
            within_dataset_percentile: 0.9,
            strict_win: true,
        });
    }
    for index in 0..3 {
        observations.push(GlobalGeneObservationV1 {
            observation_snapshot_sha256: snapshot.clone(),
            dataset_sha256: digest(&format!("weak-{index}")),
            source_family: "one-family".into(),
            genome: weak.clone(),
            within_dataset_percentile: 1.0,
            strict_win: true,
        });
    }
    let ranked = build_global_gene_ranking(task, &target, &snapshot, &observations).unwrap();
    assert_eq!(ranked.len(), 1);
    assert_eq!(ranked[0].genome.semantic_sha256, strong.semantic_sha256);
    assert_eq!(ranked[0].dataset_support, 3);
}

#[test]
fn rotating_splits_are_deterministic_grouped_and_binary_stratified() {
    let task = GpTaskV1::Binary;
    let rows = rows(task, 70);
    let dataset = digest("binary dataset");
    let training = digest("binary training");
    let seed = digest("split seed");
    let first = build_rotating_split_plan(task, &dataset, &training, &seed, &rows).unwrap();
    let second = build_rotating_split_plan(task, &dataset, &training, &seed, &rows).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.splits.len(), 5);
    assert_eq!(
        first
            .splits
            .iter()
            .map(|split| split.validation_row_ids.clone())
            .collect::<BTreeSet<_>>()
            .len(),
        5
    );
    for split in &first.splits {
        assert!(split.balance_counts["class_0"] > 0);
        assert!(split.balance_counts["class_1"] > 0);
        assert_eq!(split.validation_row_ids.len(), 14);
        for group in &split.validation_groups {
            let members: Vec<_> = rows
                .iter()
                .filter(|row| &row.atomic_group == group)
                .map(|row| &row.stable_row_id)
                .collect();
            assert!(members
                .iter()
                .all(|row| split.validation_row_ids.contains(row)));
        }
    }
    let cv7 = build_cv7_plan(task, &dataset, &training, &digest("cv seed"), &rows).unwrap();
    cv7.validate().unwrap();
    assert!(build_cv7_plan(task, &dataset, &training, &digest("cv seed"), &rows[..6]).is_err());
}

#[test]
fn confidence_penalizes_one_split_wonders() {
    assert!(confidence_score(&[0.4; 5]).unwrap() > confidence_score(&[0.9]).unwrap());
    let candidate_a = genome(GpTaskV1::Regression, 0);
    let candidate_b = genome(GpTaskV1::Regression, 1);
    let dataset = digest("rank dataset");
    let split = digest("rank split");
    let one = evaluation(
        CampaignArmV1::ConfidenceGp,
        &dataset,
        0,
        0,
        &split,
        &candidate_a,
        0.9,
    );
    let stable: Vec<_> = (0..5)
        .map(|index| {
            evaluation(
                CampaignArmV1::ConfidenceGp,
                &dataset,
                index,
                index,
                &digest(&format!("split-{index}")),
                &candidate_b,
                0.4,
            )
        })
        .collect();
    let mut values = vec![one];
    values.extend(stable);
    let ranked = rank_population(CampaignArmV1::ConfidenceGp, &values).unwrap();
    assert_eq!(
        ranked[0].candidate.semantic_sha256,
        candidate_b.semantic_sha256
    );
}

#[test]
fn offspring_schedule_and_fresh_elite_evaluation_are_exact() {
    let zero = generation_zero(GpTaskV1::Regression);
    let dataset = zero.dataset_sha256.clone();
    let split = digest("generation split");
    let evaluations: Vec<_> = zero
        .population
        .iter()
        .enumerate()
        .map(|(index, member)| {
            evaluation(
                CampaignArmV1::RotatingMean,
                &dataset,
                0,
                0,
                &split,
                &member.genome,
                0.8 - index as f64 / 100.0,
            )
        })
        .collect();
    let ranked = rank_population(CampaignArmV1::RotatingMean, &evaluations).unwrap();
    let next = build_next_generation(
        CampaignArmV1::RotatingMean,
        1,
        &digest("evolution seed"),
        &ranked,
    )
    .unwrap();
    let mut shuffled = evaluations.clone();
    shuffled.reverse();
    let replay_ranked = rank_population(CampaignArmV1::RotatingMean, &shuffled).unwrap();
    let replay = build_next_generation(
        CampaignArmV1::RotatingMean,
        1,
        &digest("evolution seed"),
        &replay_ranked,
    )
    .unwrap();
    assert_eq!(next, replay);
    assert_eq!(next.len(), 80);
    assert_eq!(
        next.iter()
            .filter(|member| member.ancestry.origin == GenomeOriginV1::Elite)
            .count(),
        3
    );
    assert_eq!(
        next.iter()
            .filter(|member| member.ancestry.origin == GenomeOriginV1::Mutation)
            .count(),
        58
    );
    assert_eq!(
        next.iter()
            .filter(|member| member.ancestry.origin == GenomeOriginV1::Crossover)
            .count(),
        19
    );
    let checkpoint = GenerationCheckpointV1 {
        schema: GenerationCheckpointV1::SCHEMA.into(),
        arm: CampaignArmV1::RotatingMean,
        dataset_sha256: dataset,
        generation: 0,
        split_sha256: split,
        population: zero.population,
        ranked,
        anchor_baseline_sha256: digest("anchors"),
        evaluations,
        previous_checkpoint_sha256: None,
    };
    checkpoint.validate().unwrap();
    let mut stale = checkpoint;
    stale.evaluations[0].generation = 1;
    assert!(stale.validate().unwrap_err().contains("stale"));
}

fn varied_regression_genome(index: usize) -> GenomeV1 {
    let transform = match (index / 6) % 3 {
        0 => "(standardize)".to_string(),
        1 => "(standardize) (robust_scale) (row_stats append=true)".to_string(),
        _ => "(standardize) (robust_scale) (row_stats append=true) (rank append=true) (cheb degree=3)"
            .to_string(),
    };
    let head = match index % 6 {
        0 => format!("(ridge lambda_frac={})", (index + 1) as f64 / 10_000.0),
        1 => format!(
            "(elasticnet alpha={} l1_ratio=0.5)",
            (index + 1) as f64 / 10_000.0
        ),
        2 => format!("(gpr kernel=linear m={} steps=10)", 50 + index),
        3 => format!("(krr m={})", 50 + index),
        4 => format!("(gbm depth={} rounds=20)", 2 + index % 5),
        _ => format!("(histgbm2 leaves={} rounds=20)", 3 + index),
    };
    let parsed = GenomeIr::parse(&format!("(pipe X {transform} {head})")).unwrap();
    GenomeV1::from_canonical(GpTaskV1::Regression, &parsed.canonical()).unwrap()
}

#[test]
fn quality_diversity_caps_ordinary_niche_occupancy() {
    let dataset = digest("qd dataset");
    let split = digest("qd split");
    let evaluations: Vec<_> = (0..80)
        .map(|index| {
            evaluation(
                CampaignArmV1::QualityDiversityGp,
                &dataset,
                0,
                0,
                &split,
                &varied_regression_genome(index),
                0.8 - index as f64 / 100.0,
            )
        })
        .collect();
    let ranked = rank_population(CampaignArmV1::QualityDiversityGp, &evaluations).unwrap();
    let next = build_next_generation(
        CampaignArmV1::QualityDiversityGp,
        1,
        &digest("qd seed"),
        &ranked,
    )
    .unwrap();
    let occupancy = next
        .iter()
        .skip(3)
        .fold(BTreeMap::new(), |mut map, member| {
            *map.entry(member.genome.niche()).or_insert(0usize) += 1;
            map
        });
    assert_eq!(occupancy.values().sum::<usize>(), 77);
    assert!(occupancy.values().all(|count| *count <= 20));
}

#[test]
fn archive_backfill_and_cv7_freeze_are_complete() {
    let zero = generation_zero(GpTaskV1::Regression);
    let dataset = zero.dataset_sha256.clone();
    let arm = CampaignArmV1::ConfidenceGp;
    let split_hashes: Vec<_> = (0..5)
        .map(|index| digest(&format!("archive split {index}")))
        .collect();
    let split_zero: Vec<_> = zero
        .population
        .iter()
        .enumerate()
        .map(|(index, member)| {
            evaluation(
                arm,
                &dataset,
                0,
                0,
                &split_hashes[0],
                &member.genome,
                0.5 - index as f64 / 1000.0,
            )
        })
        .collect();
    let ranked = rank_population(arm, &split_zero).unwrap();
    let history = vec![ranked; 5];
    let mut backfills = 0usize;
    let archive = build_provisional_archive(
        arm,
        &dataset,
        &digest("split plan"),
        &history,
        &split_zero,
        |candidate, split_index| {
            backfills += 1;
            Ok(evaluation(
                arm,
                &dataset,
                split_index,
                split_index,
                &split_hashes[split_index as usize],
                candidate,
                0.4 - split_index as f64 / 100.0,
            ))
        },
    )
    .unwrap();
    assert_eq!(backfills, 20 * 4);
    archive.validate().unwrap();
    let finalists = archive
        .members
        .iter()
        .take(10)
        .enumerate()
        .map(|(index, member)| Cv7CandidateEvidenceV1 {
            candidate: member.candidate.clone(),
            fold_primary_scores: vec![0.8 - index as f64 / 100.0; 7],
            pooled_oof_primary: 0.8 - index as f64 / 100.0,
            secondary_metric: 0.2 + index as f64 / 100.0,
            runtime_seconds: 1.0,
            valid: true,
        })
        .collect();
    let selection = select_cv7_winner(
        arm,
        &dataset,
        &archive.identity().unwrap(),
        &digest("cv7 plan"),
        finalists,
    )
    .unwrap();
    assert_eq!(selection.finalists.len(), 10);
    assert_eq!(
        selection.winner_semantic_sha256,
        selection.finalists[0].candidate.semantic_sha256
    );
    let mut invalid = selection.clone();
    invalid.finalists[0].fold_primary_scores.pop();
    assert!(invalid.validate().is_err());
}

fn pilot_results(delta: f64) -> Vec<PilotDatasetResultV1> {
    let mut results = Vec::new();
    for task in [GpTaskV1::Regression, GpTaskV1::Binary] {
        for index in 0..10 {
            let dataset = digest(&format!("pilot-{task:?}-{index}"));
            results.push(PilotDatasetResultV1 {
                arm: CampaignArmV1::LiteRaceControl,
                dataset_sha256: dataset.clone(),
                task,
                sealed_skill: 0.5,
                cpu_seconds: 10.0,
                winner_latency_seconds: 1.0,
                generation_zero_validity: 1.0,
                finalist_validity: 1.0,
                overall_validity: 1.0,
                sealed_winner: true,
            });
            for (arm, bonus, cpu) in [
                (CampaignArmV1::RotatingMean, 0.0, 10.0),
                (CampaignArmV1::ConfidenceGp, 0.001, 11.0),
                (CampaignArmV1::QualityDiversityGp, 0.002, 9.0),
            ] {
                results.push(PilotDatasetResultV1 {
                    arm,
                    dataset_sha256: dataset.clone(),
                    task,
                    sealed_skill: 0.5 + delta + bonus,
                    cpu_seconds: cpu,
                    winner_latency_seconds: 1.0,
                    generation_zero_validity: 1.0,
                    finalist_validity: 1.0,
                    overall_validity: 1.0,
                    sealed_winner: true,
                });
            }
        }
    }
    results
}

#[test]
fn pilot_bootstrap_gates_and_typed_no_improvement_are_deterministic() {
    let campaign = digest("campaign");
    let seed = digest("bootstrap");
    let first = compare_pilot(&campaign, &seed, &pilot_results(0.02)).unwrap();
    let second = compare_pilot(&campaign, &seed, &pilot_results(0.02)).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.decision,
        PilotDecisionV1::Advance {
            arm: CampaignArmV1::QualityDiversityGp
        }
    );
    let stopped = compare_pilot(&campaign, &seed, &pilot_results(-0.01)).unwrap();
    assert!(matches!(
        stopped.decision,
        PilotDecisionV1::NoImprovement { .. }
    ));
}

#[test]
fn cas_retries_are_idempotent_and_collisions_or_corruption_fail() {
    let temp = TempDir::new().unwrap();
    let store = ArtifactStoreV1::new(temp.path());
    let identity = digest("logical");
    let first = store.publish(&identity, &vec![1u32, 2, 3]).unwrap();
    assert_eq!(store.publish(&identity, &vec![1u32, 2, 3]).unwrap(), first);
    assert!(store
        .publish(&identity, &vec![4u32])
        .unwrap_err()
        .contains("collision"));
    let loaded: Vec<u32> = store.load(&identity).unwrap();
    assert_eq!(loaded, vec![1, 2, 3]);
    let object = temp
        .path()
        .join("objects/sha256")
        .join(&first[..2])
        .join(format!("{first}.json"));
    fs::write(object, b"corrupt").unwrap();
    assert!(store
        .load::<Vec<u32>>(&identity)
        .unwrap_err()
        .contains("corrupt"));
}

#[test]
fn prediction_cache_is_arm_free_and_rejects_mismatched_evidence() {
    let temp = TempDir::new().unwrap();
    let candidate = genome(GpTaskV1::Regression, 0);
    let key = PredictionCacheKeyV1 {
        schema: PredictionCacheKeyV1::SCHEMA.into(),
        dataset_sha256: digest("cache dataset"),
        training_sha256: digest("cache training"),
        split_sha256: digest("cache split"),
        candidate_semantic_sha256: candidate.semantic_sha256.clone(),
        evaluator_sha256: digest("cache evaluator"),
        math_policy_sha256: digest("cache math"),
        backend_policy_sha256: digest("cache backend"),
    };
    let prediction = PredictionObjectV1 {
        schema: PredictionObjectV1::SCHEMA.into(),
        training_view_sha256: key.training_sha256.clone(),
        fold_plan_sha256: key.split_sha256.clone(),
        fold_index: 0,
        candidate_semantic_sha256: key.candidate_semantic_sha256.clone(),
        evaluator_sha256: key.evaluator_sha256.clone(),
        fit_sha256: digest("cache fit"),
        math_policy_sha256: key.math_policy_sha256.clone(),
        backend_policy_sha256: key.backend_policy_sha256.clone(),
        validation_row_ids: vec!["r0".into(), "r1".into()],
        prediction_shape: vec![2],
        prediction_f64_bits: vec![0.1f64.to_bits(), 0.2f64.to_bits()],
        valid: true,
        invalid_reason: None,
    };
    let cache = PredictionCacheV1::new(temp.path());
    cache.publish(&key, &prediction).unwrap();
    assert_eq!(cache.load(&key).unwrap(), Some(prediction.clone()));
    let mut stale = prediction;
    stale.fold_plan_sha256 = digest("another split");
    assert!(cache
        .publish(&key, &stale)
        .unwrap_err()
        .contains("does not match"));
}

#[test]
fn operational_adapter_preserves_legacy_shards_and_seals_test_access() {
    let evaluator_path = env!("CARGO_BIN_EXE_corpus_eval");
    let evaluator_bytes = fs::read(evaluator_path).unwrap();
    let evaluator = EvaluatorFreezeV1 {
        schema: EvaluatorFreezeV1::SCHEMA.into(),
        artifact: FrozenArtifactV1 {
            path: evaluator_path.into(),
            sha256: raw_sha256(&evaluator_bytes),
            byte_length: evaluator_bytes.len() as u64,
        },
        evaluator_protocol_sha256: digest("evaluator protocol"),
        math_policy_sha256: digest("math policy"),
        backend_policy_sha256: digest("backend policy"),
    };
    let lines: Vec<_> = (0..20)
        .map(|index| format!("{},{}", index as f64 / 10.0, index as f64 / 5.0))
        .collect();
    let table_rows: Vec<_> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| DatasetRowV1 {
            split: SplitRowV1 {
                stable_row_id: format!("r{index}"),
                atomic_group: format!("g{index}"),
                regression_target: Some(index as f64 / 5.0),
                binary_target: None,
            },
            csv_line: line.clone(),
        })
        .collect();
    let mut bytes = lines.join("\n").into_bytes();
    bytes.push(b'\n');
    let dataset = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: digest("adapter dataset"),
        training_sha256: raw_sha256(&bytes),
        rows: table_rows,
    };
    let split = RowSplitV1 {
        split_index: 0,
        seed: 1,
        train_row_ids: (0..16).map(|index| format!("r{index}")).collect(),
        validation_row_ids: (16..20).map(|index| format!("r{index}")).collect(),
        validation_groups: (16..20).map(|index| format!("g{index}")).collect(),
        balance_counts: BTreeMap::new(),
    };
    let temp = TempDir::new().unwrap();
    let adapter = CorpusEvalAdapter::new(evaluator, temp.path(), 30).unwrap();
    let outcome = adapter
        .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
        .unwrap();
    assert_eq!(outcome.len(), 1);
    assert!(outcome[0].prediction.valid);
    assert_eq!(
        outcome[0].prediction.validation_row_ids,
        split.validation_row_ids
    );
    assert_eq!(
        adapter
            .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
            .unwrap(),
        outcome
    );

    let prefix = fs::read_dir(temp.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let invocation = fs::read_dir(prefix)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let done = fs::read_dir(invocation.join("output"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "done")
        })
        .unwrap();
    fs::write(
        done,
        format!("{{\"manifest_sha256\":\"{}\"}}", "0".repeat(64)),
    )
    .unwrap();
    assert!(adapter
        .evaluate_search_split(&dataset, &split, &[genome(GpTaskV1::Regression, 0)])
        .unwrap_err()
        .contains("stale"));

    let finalists = (0..10)
        .map(|index| Cv7CandidateEvidenceV1 {
            candidate: genome(GpTaskV1::Regression, index),
            fold_primary_scores: vec![0.5; 7],
            pooled_oof_primary: 0.5,
            secondary_metric: 1.0,
            runtime_seconds: 1.0,
            valid: true,
        })
        .collect();
    let wrong_selection = select_cv7_winner(
        CampaignArmV1::RotatingMean,
        &digest("wrong dataset"),
        &digest("archive"),
        &digest("cv"),
        finalists,
    )
    .unwrap();
    let sealed = SealedTestTableV1 {
        dataset_sha256: dataset.dataset_sha256.clone(),
        test_csv_path: temp
            .path()
            .join("must-not-be-read.csv")
            .display()
            .to_string(),
        test_sha256: digest("test"),
        test_row_ids: vec!["test-0".into()],
    };
    let error = adapter
        .evaluate_sealed_test(&dataset, &sealed, &wrong_selection)
        .unwrap_err();
    assert!(error.contains("selection freeze"));
    assert!(!error.contains("read sealed"));
}

#[test]
fn malformed_heads_missing_artifacts_and_incomplete_plans_fail_closed() {
    let binary_with_regression_head =
        GenomeV1::from_canonical(GpTaskV1::Binary, "(pipe X (standardize) (ridge))");
    assert!(binary_with_regression_head
        .unwrap_err()
        .contains("task-incompatible"));

    let missing = FrozenArtifactV1 {
        path: "/definitely/not/a/generalized-gp-checkpoint".into(),
        sha256: digest("missing"),
        byte_length: 1,
    };
    assert!(missing
        .verify()
        .unwrap_err()
        .contains("read frozen artifact"));

    let task = GpTaskV1::Regression;
    let source_rows = rows(task, 70);
    let mut plan = build_rotating_split_plan(
        task,
        &digest("fault dataset"),
        &digest("fault training"),
        &digest("fault seed"),
        &source_rows,
    )
    .unwrap();
    plan.splits[0].train_row_ids.pop();
    assert!(plan.validate().unwrap_err().contains("complete disjoint"));

    // Timeouts are typed invalid evidence, not crashes or synthetic scores.
    let timeout = CandidateSplitEvaluationV1 {
        valid: false,
        invalid_reason: Some("timeout".into()),
        runtime: RuntimeObservationV1 {
            cpu_seconds: 30.0,
            winner_latency_seconds: 30.0,
            timed_out: true,
        },
        ..evaluation(
            CampaignArmV1::RotatingMean,
            &digest("timeout dataset"),
            0,
            0,
            &digest("timeout split"),
            &genome(task, 0),
            0.0,
        )
    };
    timeout.validate().unwrap();
}

#[test]
fn launch_gate_blocks_the_current_zero_core_and_93_percent_disk_state() {
    let gate = LaunchGateV1 {
        schema: LaunchGateV1::SCHEMA.into(),
        real_u1_evaluator_flow: false,
        fresh_supply_evidence: false,
        protected_wave_count: 0,
        reviewed_scq_acknowledged: false,
        free_cores: 0,
        liveness_age_seconds: 10_000,
        max_liveness_age_seconds: 300,
        storage_free_fraction: 0.07,
        min_storage_free_fraction: 0.10,
    };
    assert!(!gate.launch_allowed().unwrap());
    let mut ready = gate;
    ready.real_u1_evaluator_flow = true;
    ready.fresh_supply_evidence = true;
    ready.protected_wave_count = 2;
    ready.reviewed_scq_acknowledged = true;
    ready.free_cores = 8;
    ready.liveness_age_seconds = 10;
    ready.storage_free_fraction = 0.20;
    assert!(ready.launch_allowed().unwrap());
}
