//! Unit tests proving all five `DV-02` invariants for `CpuAdapter::predict_costs_with_history_mask`.

use super::tests::{synthetic_adapter, synthetic_fixture};
use super::{HistoryMode, HISTORY_CONTRACT_V1_SCHEMA, PROG_FEATURES, PROJ_FEATURES, REVEALED};

fn bits_slice(xs: &[f32]) -> Vec<u32> {
    xs.iter().copied().map(f32::to_bits).collect()
}

#[test]
fn dv02_invariant_1_absent_slot_poisoning_invariance() {
    let model = synthetic_adapter();
    let base_f = synthetic_fixture(8);
    let empty_mask = [false; 8];
    let baseline_empty = model
        .predict_costs_with_history_mask(base_f.inputs(2, 8), &empty_mask)
        .unwrap();

    // Mutate absent slots heavily: fitness 100 vs 9999 vs -50, gates 0 vs 1, collided/invalid indices.
    let mut poisoned = synthetic_fixture(8);
    poisoned.revealed_fitness = vec![9999.0, -50.0, 100.0, 1e6, -999.0, 42.0, 9999.0, -50.0];
    poisoned.revealed_gates = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    poisoned.revealed_indices = vec![0, 0, 11, -5, 7, 7, 99, 0];
    let poisoned_empty = model
        .predict_costs_with_history_mask(poisoned.inputs(2, 8), &empty_mask)
        .unwrap();

    assert_eq!(
        bits_slice(&baseline_empty.total_cost),
        bits_slice(&poisoned_empty.total_cost)
    );
    assert_eq!(
        bits_slice(&baseline_empty.anchor_cost),
        bits_slice(&poisoned_empty.anchor_cost)
    );
    assert_eq!(
        baseline_empty
            .proposal_kl_to_anchor
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        poisoned_empty
            .proposal_kl_to_anchor
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    // Partial presence mask `[true, false, true, false]` across batch=2:
    let partial_mask = [true, false, true, false, false, true, false, true];
    let baseline_partial = model
        .predict_costs_with_history_mask(base_f.inputs(2, 8), &partial_mask)
        .unwrap();

    let mut partial_poisoned = synthetic_fixture(8);
    // Mutate ONLY the absent slots (indices 1, 3 in batch 0 and 4, 6 in batch 1)
    for &absent in &[1usize, 3, 4, 6] {
        partial_poisoned.revealed_fitness[absent] = 9999.0;
        partial_poisoned.revealed_gates[absent] = 1.0 - partial_poisoned.revealed_gates[absent];
        partial_poisoned.revealed_indices[absent] = partial_poisoned.revealed_indices[0];
    }
    let poisoned_partial = model
        .predict_costs_with_history_mask(partial_poisoned.inputs(2, 8), &partial_mask)
        .unwrap();

    assert_eq!(
        bits_slice(&baseline_partial.total_cost),
        bits_slice(&poisoned_partial.total_cost)
    );
    assert_eq!(
        bits_slice(&baseline_partial.anchor_cost),
        bits_slice(&poisoned_partial.anchor_cost)
    );
}

#[test]
fn dv02_invariant_2_candidate_permutation_equivariance_in_true_empty_mode() {
    let model = synthetic_adapter();
    let orig = synthetic_fixture(8);
    let n = 12usize;
    let empty_mask = [false; 8];
    let orig_pred = model
        .predict_costs_with_history_mask(orig.inputs(2, 8), &empty_mask)
        .unwrap();

    let perm: [usize; 12] = [7, 2, 11, 0, 5, 9, 1, 10, 4, 8, 3, 6];
    let mut permuted = synthetic_fixture(8);
    for b in 0..2 {
        for (new_pos, &old_pos) in perm.iter().enumerate() {
            permuted.prog_412
                [(b * n + new_pos) * PROG_FEATURES..(b * n + new_pos + 1) * PROG_FEATURES]
                .copy_from_slice(
                    &orig.prog_412
                        [(b * n + old_pos) * PROG_FEATURES..(b * n + old_pos + 1) * PROG_FEATURES],
                );
            permuted.proj_16
                [(b * n + new_pos) * PROJ_FEATURES..(b * n + new_pos + 1) * PROJ_FEATURES]
                .copy_from_slice(
                    &orig.proj_16
                        [(b * n + old_pos) * PROJ_FEATURES..(b * n + old_pos + 1) * PROJ_FEATURES],
                );
        }
    }
    let perm_pred = model
        .predict_costs_with_history_mask(permuted.inputs(2, 8), &empty_mask)
        .unwrap();

    for b in 0..2 {
        for (new_pos, &old_pos) in perm.iter().enumerate() {
            let diff_total = (perm_pred.total_cost[b * n + new_pos]
                - orig_pred.total_cost[b * n + old_pos])
                .abs();
            let diff_anchor = (perm_pred.anchor_cost[b * n + new_pos]
                - orig_pred.anchor_cost[b * n + old_pos])
                .abs();
            assert!(
                diff_total < 1e-5,
                "candidate-permutation total_cost drift {diff_total} at b={b}, old={old_pos}"
            );
            assert!(
                diff_anchor < 1e-5,
                "candidate-permutation anchor_cost drift {diff_anchor} at b={b}, old={old_pos}"
            );
        }
    }
}

#[test]
fn dv02_invariant_3_finite_and_deterministic_across_empty_and_partial_masks() {
    assert_eq!(
        HISTORY_CONTRACT_V1_SCHEMA,
        "jope.ranker-history-contract/v1"
    );
    assert_ne!(HistoryMode::LegacyF12, HistoryMode::MaskedPresenceV1);
    let model = synthetic_adapter();
    let f = synthetic_fixture(16);
    for mask in [
        [false; 8],
        [true, false, false, false, false, false, false, true],
        [false, true, true, false, true, true, true, false],
    ] {
        let first = model
            .predict_costs_with_history_mask(f.inputs(2, 16), &mask)
            .unwrap();
        let second = model
            .predict_costs_with_history_mask(f.inputs(2, 16), &mask)
            .unwrap();
        assert!(first.total_cost.iter().all(|v| v.is_finite()));
        assert!(first.anchor_cost.iter().all(|v| v.is_finite()));
        assert!(first.proposal_kl_to_anchor.iter().all(|v| v.is_finite()));
        assert_eq!(
            bits_slice(&first.total_cost),
            bits_slice(&second.total_cost)
        );
        assert_eq!(
            bits_slice(&first.anchor_cost),
            bits_slice(&second.anchor_cost)
        );
    }
}

#[test]
fn dv02_invariant_4_legacy_four_present_zero_ulp_parity() {
    let model = synthetic_adapter();
    for proposals in [8, 16] {
        let f = synthetic_fixture(proposals);
        let legacy = model.predict_costs(f.inputs(2, proposals)).unwrap();
        let masked_all = model
            .predict_costs_with_history_mask(f.inputs(2, proposals), &[true; 8])
            .unwrap();
        assert_eq!(
            bits_slice(&masked_all.total_cost),
            bits_slice(&legacy.total_cost)
        );
        assert_eq!(
            bits_slice(&masked_all.anchor_cost),
            bits_slice(&legacy.anchor_cost)
        );
        assert_eq!(
            masked_all
                .proposal_kl_to_anchor
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            legacy
                .proposal_kl_to_anchor
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(masked_all.exceeds_kl_limit, legacy.exceeds_kl_limit);
    }
}

#[test]
fn dv02_invariant_5_real_failure_observation_differs_from_absent_history() {
    let model = synthetic_adapter();
    let mut f = synthetic_fixture(8);
    f.revealed_indices[0] = 0;
    f.revealed_fitness[0] = 100.0;
    f.revealed_gates[0] = 0.0;

    let absent = model
        .predict_costs_with_history_mask(f.inputs(2, 8), &[false; 8])
        .unwrap();
    let failed_slot = model
        .predict_costs_with_history_mask(
            f.inputs(2, 8),
            &[true, false, false, false, false, false, false, false],
        )
        .unwrap();

    let max_total_diff = absent
        .total_cost
        .iter()
        .take(12)
        .zip(failed_slot.total_cost.iter().take(12))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max_anchor_diff = absent
        .anchor_cost
        .iter()
        .take(12)
        .zip(failed_slot.anchor_cost.iter().take(12))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_total_diff > 1e-2 && max_anchor_diff > 1e-2,
        "failed observation must differ from absence (total={max_total_diff}, anchor={max_anchor_diff})"
    );

    // Reject malformed mask length and non-finite present fitness.
    assert!(model
        .predict_costs_with_history_mask(f.inputs(2, 8), &[true; REVEALED])
        .is_err());
    let mut bad_present = synthetic_fixture(8);
    bad_present.revealed_fitness[0] = f32::NAN;
    assert!(model
        .predict_costs_with_history_mask(
            bad_present.inputs(2, 8),
            &[true, false, false, false, false, false, false, false]
        )
        .is_err());
}
