use super::*;

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
