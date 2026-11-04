use super::*;
use rand_chacha::rand_core::SeedableRng;

fn rng(seed: u64) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(seed)
}

fn slate(costs: &[f64]) -> Vec<f64> {
    [vec![-1e100; REVEALED], costs.to_vec()].concat()
}

#[test]
fn normal_selection_excludes_revealed_and_preserves_rng() {
    let costs = slate(&[8., 7., 6., 5., 4., 3., 2., 1.]);
    let mut actual = rng(7);
    assert_eq!(select_costs(&costs, false, &mut actual), Ok([7, 6, 5, 4]));
    assert_eq!(actual.next_u64(), rng(7).next_u64());
}

#[test]
fn ties_and_signed_zero_keep_proposal_order() {
    let costs = slate(&[1., -0., 0., -0., 0., 1., 1., 1.]);
    assert_eq!(select_costs(&costs, false, &mut rng(0)), Ok([1, 2, 3, 4]));
    assert_eq!(
        select_costs(&slate(&[0.; 8]), false, &mut rng(0)),
        Ok([0, 1, 2, 3])
    );
}

#[test]
fn exploration_has_two_best_and_two_distinct_random_choices() {
    let costs = slate(&(0..16).map(|v| -(v as f64)).collect::<Vec<_>>());
    let mut sampled = BTreeSet::<usize>::new();
    for seed in 0..100 {
        let mut actual = rng(seed);
        let chosen = select_costs(&costs, true, &mut actual).unwrap();
        assert_eq!(chosen[..2], [15, 14]);
        assert_eq!(chosen.into_iter().collect::<BTreeSet<_>>().len(), 4);
        assert!(chosen[2..].iter().all(|index| *index < 14));
        sampled.extend(chosen[2..].iter().copied());
        let mut replay = rng(seed);
        assert_eq!(select_costs(&costs, true, &mut replay).unwrap(), chosen);
        assert_eq!(actual.next_u64(), replay.next_u64());
    }
    assert_eq!(sampled, (0..14).collect());
}

#[test]
fn selection_refuses_shapes_and_all_nonfinite_positions_before_rng_use() {
    for len in [0, 4, 8, 11, 13, 19, 21] {
        let mut actual = rng(12);
        assert_eq!(
            select_costs(&vec![0.; len], false, &mut actual),
            Err(AdmissionError::InvalidSlate)
        );
        assert_eq!(actual.next_u64(), rng(12).next_u64());
    }
    for explore in [false, true] {
        let len = if explore { 20 } else { 12 };
        for index in 0..len {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut costs = vec![0.; len];
                costs[index] = invalid;
                let mut actual = rng(12);
                assert_eq!(
                    select_costs(&costs, explore, &mut actual),
                    Err(AdmissionError::NonFiniteCost)
                );
                assert_eq!(actual.next_u64(), rng(12).next_u64());
            }
        }
    }
    assert_eq!(
        select_costs(&[0.; 12], true, &mut rng(0)),
        Err(AdmissionError::InvalidSlate)
    );
    assert_eq!(
        select_costs(&[0.; 20], false, &mut rng(0)),
        Err(AdmissionError::InvalidSlate)
    );
}

fn measurements() -> Vec<MeasuredAlgorithm<'static>> {
    ["a", "b", "c", "d", "e", "f"]
        .into_iter()
        .enumerate()
        .map(|(i, id)| MeasuredAlgorithm {
            id,
            surrogate_fitness: 5. - i as f64,
            gate: i % 2 == 0,
        })
        .collect()
}

#[test]
fn reveal_choice_is_explicit_and_keeps_original_measurements_paired() {
    let measured = measurements();
    assert_eq!(
        select_revealed(&measured, RevealedRule::StorageOrder),
        Ok([0, 1, 2, 3])
    );
    let sorted = select_revealed(&measured, RevealedRule::SurrogateFitness).unwrap();
    assert_eq!(sorted, [5, 4, 3, 2]);
    assert_eq!(sorted.map(|i| measured[i].gate), [false, true, false, true]);
    assert_eq!(
        sorted.map(|i| measured[i].surrogate_fitness),
        [0., 1., 2., 3.]
    );
    let mut tied = measured;
    tied.iter_mut().for_each(|v| v.surrogate_fitness = 0.);
    assert_eq!(
        select_revealed(&tied, RevealedRule::SurrogateFitness),
        Ok([0, 1, 2, 3])
    );
}

#[test]
fn reveal_refuses_short_large_duplicate_empty_and_nonfinite_pools() {
    let ids: Vec<_> = (0..ELITE_CAPACITY)
        .map(|i| format!("algorithm-{i}"))
        .collect();
    let full: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| MeasuredAlgorithm {
            id,
            surrogate_fitness: i as f64,
            gate: true,
        })
        .collect();
    assert_eq!(
        select_revealed(&full, RevealedRule::SurrogateFitness),
        Ok([0, 1, 2, 3])
    );
    for rule in [RevealedRule::StorageOrder, RevealedRule::SurrogateFitness] {
        assert_eq!(
            select_revealed(&[], rule),
            Err(AdmissionError::InvalidRevealed)
        );
        let mut measured = measurements();
        assert_eq!(
            select_revealed(&measured[..3], rule),
            Err(AdmissionError::InvalidRevealed)
        );
        measured[5].id = measured[0].id;
        assert_eq!(
            select_revealed(&measured, rule),
            Err(AdmissionError::InvalidRevealed)
        );
        measured[5].id = "";
        assert_eq!(
            select_revealed(&measured, rule),
            Err(AdmissionError::InvalidRevealed)
        );
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut measured = measurements();
            measured[5].surrogate_fitness = invalid;
            assert_eq!(
                select_revealed(&measured, rule),
                Err(AdmissionError::InvalidRevealed)
            );
        }
        assert_eq!(
            select_revealed(&vec![measurements()[0]; 49], rule),
            Err(AdmissionError::InvalidRevealed)
        );
    }
}
