use super::*;

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
