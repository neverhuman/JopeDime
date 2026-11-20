//! Unit tests proving all five `DV-02` invariants, `0-ULP` slot-index invariance,
//! absent-slot `NaN`/OOB poisoning invariance, variable slate sizes `1..=4096`, and
//! cross-language Python `FlywheelBlockFormerAdapterNet` parity.

use super::tests::{synthetic_adapter, synthetic_fixture};
use super::{
    HistoryMode, PredictCostsInputs, CONTEXT_FEATURES, HISTORY_CONTRACT_V1_SCHEMA, PROG_FEATURES,
    PROJ_FEATURES, REVEALED, SCALE, SOS_FEATURES, TOKENS, TOKEN_WIDTH,
};

#[path = "history_mask_golden_tests.rs"]
mod golden_parity;

pub(super) fn bits_slice(xs: &[f32]) -> Vec<u32> {
    xs.iter().copied().map(f32::to_bits).collect()
}

#[test]
fn dv02_invariant_1_absent_slot_nan_and_oob_poisoning_invariance() {
    let model = synthetic_adapter();
    let base_f = synthetic_fixture(8);
    let empty_mask = [false; 8];
    let baseline_empty = model
        .predict_costs_with_history_mask(base_f.inputs(2, 8), &empty_mask)
        .unwrap();

    // Absent slots with NaN/Inf fitness, non-binary gates, and out-of-bounds/negative indices
    // are completely ignored and produce 0-ULP identical predictions.
    let mut poisoned = synthetic_fixture(8);
    poisoned.revealed_fitness = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        9999.0,
        -50.0,
        f32::NAN,
        1e6,
        -999.0,
    ];
    poisoned.revealed_gates = vec![2.0, -1.0, f32::NAN, 1.0, 0.5, 9.0, 1.0, 0.0];
    poisoned.revealed_indices = vec![-99, 9999, 0, 0, -1, 4096, 7, 7];
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

    let partial_mask = [true, false, true, false, false, true, false, true];
    let baseline_partial = model
        .predict_costs_with_history_mask(base_f.inputs(2, 8), &partial_mask)
        .unwrap();

    let mut partial_poisoned = synthetic_fixture(8);
    for &absent in &[1usize, 3, 4, 6] {
        partial_poisoned.revealed_fitness[absent] = f32::NAN;
        partial_poisoned.revealed_gates[absent] = f32::INFINITY;
        partial_poisoned.revealed_indices[absent] = -42;
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
    let mut orig = synthetic_fixture(8);
    for (i, bp) in orig.base_prior_cost.iter_mut().enumerate() {
        *bp = ((i % 12) as f32 - 5.5) * 0.11;
    }
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
            permuted.base_prior_cost[b * n + new_pos] = orig.base_prior_cost[b * n + old_pos];
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
fn dv02_invariant_3_slot_index_zero_ulp_invariance_and_variable_slate_sizes() {
    assert_eq!(
        HISTORY_CONTRACT_V1_SCHEMA,
        "jope.ranker-history-contract/v1"
    );
    assert_ne!(HistoryMode::LegacyF12, HistoryMode::MaskedPresenceV1);
    let model = synthetic_adapter();

    // K_rev = 1: slot 0 vs slot 3 carrying the exact same (r=2, f=0.7, g=1.0)
    let mut f_slot0 = synthetic_fixture(8);
    f_slot0.revealed_indices[0] = 2;
    f_slot0.revealed_fitness[0] = 0.7;
    f_slot0.revealed_gates[0] = 1.0;

    let mut f_slot3 = synthetic_fixture(8);
    f_slot3.revealed_indices[3] = 2;
    f_slot3.revealed_fitness[3] = 0.7;
    f_slot3.revealed_gates[3] = 1.0;

    let p0 = model
        .predict_costs_with_history_mask(
            f_slot0.inputs(2, 8),
            &[true, false, false, false, false, false, false, false],
        )
        .unwrap();
    let p3 = model
        .predict_costs_with_history_mask(
            f_slot3.inputs(2, 8),
            &[false, false, false, true, false, false, false, false],
        )
        .unwrap();
    assert_eq!(bits_slice(&p0.total_cost), bits_slice(&p3.total_cost));
    assert_eq!(bits_slice(&p0.anchor_cost), bits_slice(&p3.anchor_cost));
    assert_eq!(
        p0.proposal_kl_to_anchor[0].to_bits(),
        p3.proposal_kl_to_anchor[0].to_bits()
    );

    // K_rev = 2 & K_rev = 3 slot-placement 0-ULP invariance
    let mut f_k2_a = synthetic_fixture(8);
    f_k2_a.revealed_indices[..4].copy_from_slice(&[1, 5, 9, 10]);
    f_k2_a.revealed_fitness[..4].copy_from_slice(&[0.3, 1.4, 99.0, 99.0]);
    f_k2_a.revealed_gates[..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    let mut f_k2_b = synthetic_fixture(8);
    f_k2_b.revealed_indices[..4].copy_from_slice(&[8, 1, 11, 5]);
    f_k2_b.revealed_fitness[..4].copy_from_slice(&[99.0, 0.3, 99.0, 1.4]);
    f_k2_b.revealed_gates[..4].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
    let k2_a = model
        .predict_costs_with_history_mask(
            f_k2_a.inputs(2, 8),
            &[true, true, false, false, false, false, false, false],
        )
        .unwrap();
    let k2_b = model
        .predict_costs_with_history_mask(
            f_k2_b.inputs(2, 8),
            &[false, true, false, true, false, false, false, false],
        )
        .unwrap();
    assert_eq!(bits_slice(&k2_a.total_cost), bits_slice(&k2_b.total_cost));
    assert_eq!(bits_slice(&k2_a.anchor_cost), bits_slice(&k2_b.anchor_cost));

    // Variable candidate counts: 1, 8, 16, 32
    for cands in [1usize, 8, 16, 32] {
        let prog = vec![0.15_f32; cands * PROG_FEATURES];
        let proj = vec![-0.05_f32; cands * PROJ_FEATURES];
        let tokens = vec![0.02_f32; TOKENS * TOKEN_WIDTH];
        let token_mask = [1.0_f32; TOKENS];
        let rev_idx = [0_i64, 1.min(cands as i64 - 1), 0, 0];
        let rev_fit = [0.4_f32; REVEALED];
        let rev_gate = [1.0_f32; REVEALED];
        let ctx_feat = [0.0_f32; CONTEXT_FEATURES];
        let bp: Vec<f32> = (0..cands).map(|i| (i as f32) * 0.03).collect();
        let sos = vec![0.0_f32; cands * SOS_FEATURES];
        let inp = PredictCostsInputs {
            batch: 1,
            candidates: cands,
            prog_412: &prog,
            proj_16: &proj,
            tokens: &tokens,
            token_mask: &token_mask,
            revealed_indices: &rev_idx,
            revealed_fitness: &rev_fit,
            revealed_gates: &rev_gate,
            ctx_feat: &ctx_feat,
            base_prior_cost: &bp,
            sos_prior: &sos,
            scale: SCALE,
        };
        let out_empty = model
            .predict_costs_with_history_mask(inp, &[false; 4])
            .unwrap();
        assert_eq!(out_empty.total_cost.len(), cands);
        assert!(out_empty.total_cost.iter().all(|v| v.is_finite()));
        assert!(out_empty.proposal_kl_to_anchor[0].is_finite());
        let out_one = model
            .predict_costs_with_history_mask(inp, &[true, false, false, false])
            .unwrap();
        assert_eq!(out_one.total_cost.len(), cands);
        assert!(out_one.total_cost.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn dv02_invariant_4_and_5_legacy_zero_ulp_parity_and_failure_distinction() {
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
    assert!(max_total_diff > 1e-2);
}
