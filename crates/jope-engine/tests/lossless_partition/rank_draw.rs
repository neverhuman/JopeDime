use super::*;

#[test]
fn every_rank_count_draws_the_same_global_batch_with_rank_local_rng() {
    let (index, families) = index();
    for local_step in 0..3u64 {
        let plans = [(1, 16), (3, 16), (4, 24), (5, 8), (13, 8)]
            .map(|(world, micro)| Plan::new(world, 88, micro).unwrap());
        let expected = sample(
            &index,
            &families,
            plans[0].global_serial(local_step).unwrap() as usize,
            88,
            SEED,
        );
        for plan in &plans {
            let mut ids = Vec::new();
            let mut seeds = std::collections::BTreeSet::new();
            for rank in 0..plan.world() {
                for slot in plan.slots(rank, SEED, local_step).unwrap() {
                    assert_eq!(slot.rows, slot.comparisons * 8);
                    assert_eq!(
                        slot.dropout_seed,
                        consumption::dropout_seed(SEED, local_step, rank, slot.microbatch)
                    );
                    assert!(seeds.insert(slot.dropout_seed), "rank-local RNG streams");
                    ids.extend(sample(
                        &index,
                        &families,
                        slot.serial as usize,
                        slot.rows,
                        SEED,
                    ));
                }
            }
            assert_eq!(ids, expected, "world {} changed the data", plan.world());
        }
    }
    // The v1 cursor cannot express an uneven three-rank shard.
    assert!(consumption::sampler_serial(0, 88, 3, 1, 0, 16).is_err());
}
