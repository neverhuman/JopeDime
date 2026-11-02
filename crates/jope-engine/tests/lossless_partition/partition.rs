use super::*;

#[test]
fn partition_v1_cannot_form_three_ranks_and_averages_unequal_local_means() {
    // v1 requires global_batch % (world × microbatch) == 0 and a world in
    // {1, 2, 4, 8}: no three-rank partition of the 2048-row batch exists.
    for micro in [64, 128, 256] {
        assert!(training_partition(3, micro, 2048).is_err());
    }
    // Its normalization averages microbatch-local means. With unequal shards
    // (7 comparisons over three ranks: 3, 2, 2) that is a different objective
    // from the single-process one.
    let b = batch(7, 0);
    let reference = v1(&b, &leaves(&b), 0, 56).0.double_value(&[]);
    let l = leaves(&b);
    let averaged = [(0, 24), (24, 16), (40, 16)]
        .iter()
        .map(|&(first, rows)| v1(&b, &l, first, rows).0.double_value(&[]))
        .sum::<f64>()
        / 3.0;
    assert!(
        (averaged - reference).abs() > 1e-3,
        "unequal local means happened to agree: {averaged} vs {reference}"
    );
}

#[test]
fn assignment_is_lossless_deterministic_and_keeps_comparisons_whole() {
    for groups in 0..=40 {
        for world in 1..=9 {
            for micro in 1..=5 {
                let shards = partition::assign(groups, world, micro).unwrap();
                assert_eq!(shards, partition::assign(groups, world, micro).unwrap());
                assert_eq!(shards.len(), world);
                let mut next = 0;
                for (rank, shard) in shards.iter().enumerate() {
                    assert_eq!(shard.rank, rank);
                    assert_eq!(shard.first_group, next, "contiguous, in rank order");
                    assert_eq!(
                        shard.groups,
                        groups / world + usize::from(rank < groups % world)
                    );
                    assert_eq!(shard.microbatches.iter().sum::<usize>(), shard.groups);
                    assert!(shard.microbatches.iter().all(|&m| (1..=micro).contains(&m)));
                    next += shard.groups;
                }
                assert_eq!(next, groups, "no comparison dropped or repeated");
            }
        }
    }
    // The three-rank partition of the 2048-row batch with 256-row microbatches.
    let plan = Plan::new(3, 2048, 256).unwrap();
    let shards = plan.shards();
    assert_eq!(
        shards.iter().map(|s| s.groups).collect::<Vec<_>>(),
        [86, 85, 85]
    );
    assert_eq!(shards[0].microbatches, [32, 32, 22]);
    assert_eq!(shards[1].microbatches, [32, 32, 21]);
    assert_eq!(shards[2].first_group, 171);
    let record = plan.record();
    assert_eq!(record["schema"], partition::SCHEMA);
    assert_eq!(record["world"], 3);
    assert_eq!(record["normalization"], partition::NORMALIZATION);
    assert_eq!(
        record["numerical_policy"]["loss_rtol"],
        partition::LOSS_RTOL
    );
    assert_eq!(
        record["numerical_policy"]["gradient_rtol_of_max_abs"],
        partition::GRADIENT_RTOL
    );
    assert_eq!(record, Plan::new(3, 2048, 256).unwrap().record());
    // Malformed plans are refused.
    for (world, global, micro) in [
        (0, 2048, 256),
        (3, 0, 256),
        (3, 2044, 256),
        (3, 2048, 0),
        (3, 2048, 12),
    ] {
        assert!(
            Plan::new(world, global, micro).is_err(),
            "{world} {global} {micro}"
        );
    }
}

#[test]
fn every_partition_matches_the_single_process_reference() {
    // Uneven groups, nondivisible batches, masks, ties, boosted pairs and a
    // rank whose first microbatches are all ties (zero local pair weight).
    for (label, b) in [
        ("7 comparisons, 2 tied", batch(7, 2)),
        ("11 comparisons", batch(11, 0)),
    ] {
        let reference = reference(&b);
        let groups = b.rows.len() / 8;
        for (world, micro) in [
            (1, 1),
            (1, 2),
            (1, 64),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (5, 1),
            (7, 1),
            (8, 1),
            (9, 2),
            (12, 1),
        ] {
            let plan = Plan::new(world, groups * 8, micro * 8).unwrap();
            let run = run_plan(&b, &plan, Transport::Sum, 1.0);
            assert_within_policy(
                &format!("{label}: world {world}, microbatch {micro} comparisons"),
                &run,
                &reference,
            );
        }
    }
}

#[test]
fn a_mean_transport_needs_the_world_scale() {
    let b = batch(11, 0);
    let reference = reference(&b);
    let plan = Plan::new(3, 88, 16).unwrap();
    assert_eq!(plan.local_scale(Transport::Sum), 1.0);
    assert_eq!(plan.local_scale(Transport::Mean), 3.0);
    let scaled = run_plan(
        &b,
        &plan,
        Transport::Mean,
        plan.local_scale(Transport::Mean),
    );
    assert_within_policy("mean transport, scaled by world", &scaled, &reference);
    // Negative control: without the scale a mean transport trains on L / W.
    let unscaled = run_plan(&b, &plan, Transport::Mean, 1.0);
    assert!(!within_policy(
        "mean transport, unscaled",
        &unscaled,
        &reference
    ));
    assert!((unscaled.loss * 3.0 - reference.loss).abs() < 1e-4);
}

#[test]
fn globally_empty_terms_are_zero_and_empty_ranks_contribute_zero_gradients() {
    // Every pair ties: both Bradley–Terry terms are globally empty.
    let b = batch(5, 5);
    let global = denominators(&b.rows).unwrap();
    assert_eq!(global.pair_weight, 0.0);
    assert_eq!(global.empty_terms(), ["ranking_bt", "contextfree_bt"]);
    let reference = reference(&b);
    assert_eq!(reference.terms[1], 0.0);
    assert_eq!(reference.terms[4], 0.0);
    assert!(reference.grads[4].iter().all(|&g| g == 0.0));
    for (world, micro) in [(1, 5), (3, 1), (5, 1), (7, 1)] {
        let plan = Plan::new(world, 40, micro * 8).unwrap();
        let run = run_plan(&b, &plan, Transport::Sum, 1.0);
        assert_eq!(run.terms[1], 0.0, "a globally empty term has value 0");
        assert_eq!(run.terms[4], 0.0);
        assert!(run.grads[4].iter().all(|&g| g == 0.0), "and gradient 0");
        assert_within_policy(&format!("all ties, world {world}"), &run, &reference);
    }
    // More ranks than comparisons: ranks 7 and 8 are empty, explicitly.
    let b = batch(7, 2);
    let plan = Plan::new(9, 56, 8).unwrap();
    assert_eq!(
        plan.record()["shards"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["groups"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    );
    for rank in [7, 8] {
        assert!(plan.slots(rank, SEED, 0).unwrap().is_empty());
        let run = run_rank(&b, &plan, rank, 1.0);
        assert_eq!(run.loss, 0.0);
        assert_eq!(run.terms, [0.0; 5]);
        for (name, g) in OUTPUTS.iter().zip(&run.grads) {
            assert!(
                g.iter().all(|&x| x == 0.0),
                "empty rank {rank} has a nonzero {name} gradient"
            );
        }
    }
    assert!(empty_rank_objective(std::iter::empty()).is_err());
}

#[test]
fn term_denominators_are_global_and_must_cover_the_microbatch() {
    let b = batch(7, 2);
    let global = denominators(&b.rows).unwrap();
    assert_eq!(global.listwise_groups, 7);
    assert_eq!(global.runtime_rows, 56);
    // Tied comparisons 0 and 1 carry row weight but no pair weight.
    let tied = denominators(&b.rows[..16]).unwrap();
    assert_eq!(tied.pair_weight, 0.0);
    let fractions = tied.fractions_of(&global).unwrap();
    assert!(fractions[0] > 0.0 && fractions[2] == 2.0 / 7.0 && fractions[3] == 16.0 / 56.0);
    assert_eq!((fractions[1], fractions[4]), (0.0, 0.0));
    // Term-specific shares: every comparison has the same percentiles, so its
    // row-weight share is 1/7, but its pair-weight share is not 1/5 of the
    // five untied comparisons, because |Δv| shrinks with the comparison index.
    let untied = denominators(&b.rows[16..24]).unwrap();
    let f = untied.fractions_of(&global).unwrap();
    assert!((f[0] - 1.0 / 7.0).abs() < 1e-12, "{}", f[0]);
    assert!(
        f[1] - 1.0 / 5.0 > 1e-2,
        "pair share {} is a weight share",
        f[1]
    );
    assert_eq!(f[1], f[4], "the ablation term shares the pair denominator");
    // A microbatch that the global denominators do not cover is refused.
    let smaller = denominators(&b.rows[..24]).unwrap();
    let whole = denominators(&b.rows).unwrap();
    assert!(whole.fractions_of(&smaller).is_err());
    let l = leaves(&b);
    let error = objective_of(&b, &l, 24, 32, Some(&smaller)).unwrap_err();
    assert!(error.to_string().contains("exceeds the global"), "{error}");
    // Pair weight where the global batch has none is refused.
    let all_tied = denominators(&batch(7, 7).rows).unwrap();
    assert!(untied.fractions_of(&all_tied).is_err());
    // Incomplete comparisons never reach a denominator or an objective.
    assert!(denominators(&b.rows[..7]).is_err());
    assert!(objective_of(&b, &l, 0, 12, Some(&global)).is_err());
}
