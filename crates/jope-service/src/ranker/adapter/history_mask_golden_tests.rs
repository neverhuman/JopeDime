//! Cross-language golden parity tests against Python `FlywheelBlockFormerAdapterNet`
//! and `FlywheelGameNet` (`phase12_multimodal_model.py` + `model.py`) across `K_rev = 0..=4`
//! and zero / nonzero `base_prior_cost`.

use super::super::tests::{synthetic_adapter, synthetic_fixture};
use super::super::{PredictCostsInputs, REVEALED, SCALE};

const PY_TOTAL_K0_BP_ZERO: [f32; 12] = [
    0.31958225, 0.35259116, 0.35403183, 0.35458195, 0.3546554, 0.30820704, 0.32574195, 0.3133742,
    0.27853554, 0.3111931, 0.30619174, 0.30800793,
];
const PY_ANCHOR_K0_BP_ZERO: [f32; 12] = [
    0.3197923, 0.35205486, 0.35424805, 0.35518646, 0.35492873, 0.30801216, 0.32517758, 0.3131689,
    0.27847952, 0.3116, 0.30584776, 0.3081983,
];

const PY_TOTAL_K0_BP_NONZERO: [f32; 12] = [
    -0.5092279,
    -0.32444972,
    -0.17271315,
    -0.022284463,
    0.12915678,
    0.23291354,
    0.40119815,
    0.5385572,
    0.6549984,
    0.8388085,
    0.9812256,
    1.1361538,
];
const PY_ANCHOR_K0_BP_NONZERO: [f32; 12] = [
    -0.5090178,
    -0.324986,
    -0.17249693,
    -0.021679943,
    0.12943009,
    0.23271868,
    0.4006338,
    0.5383519,
    0.6549424,
    0.8392154,
    0.98088163,
    1.1363442,
];

const PY_TOTAL_K1_BP_NONZERO: [f32; 12] = [
    1.909681, 2.0987735, 2.5285661, 2.3885286, 2.5735154, 2.7544374, 2.8606567, 2.9901748,
    3.1170726, 3.3647823, 3.3993852, 3.5977132,
];
const PY_ANCHOR_K1_BP_NONZERO: [f32; 12] = [
    1.6864879, 1.8703644, 2.0227332, 2.173757, 2.3249354, 2.4283423, 2.5963247, 2.7340221,
    2.8506064, 3.034822, 3.1763947, 3.331691,
];

const PY_TOTAL_K2_BP_ZERO: [f32; 12] = [
    1.0251666, 1.7315397, 2.3009071, 1.7465713, 1.7184479, 1.6422702, 1.569637, 1.4740252,
    1.4287885, 1.4460074, 1.0427523, 1.7202024,
];
const PY_ANCHOR_K2_BP_ZERO: [f32; 12] = [
    0.9002252, 1.5753874, 1.9331825, 1.6071639, 1.5528752, 1.42849, 1.3912187, 1.3173084,
    1.2440429, 1.2256249, 0.90616935, 1.5384562,
];

const PY_TOTAL_K3_BP_NONZERO: [f32; 12] = [
    21.545504, 162.88954, 98.561905, 24.31354, 49.53748, 61.970127, 65.71323, 65.905266, 64.58759,
    61.429996, 24.378082, 161.63164,
];
const PY_ANCHOR_K3_BP_NONZERO: [f32; 12] = [
    20.958591, 161.48022, 97.74604, 23.027628, 48.524937, 61.16569, 64.670685, 65.054436,
    63.650227, 60.53927, 23.521982, 160.7147,
];

const PY_TOTAL_K4_BP_ZERO: [f32; 12] = [
    20.067091, 140.06183, 30.392803, 22.168129, 39.2294, 47.149967, 49.980057, 50.580578,
    49.907997, 48.055317, 21.24088, 135.91689,
];
const PY_ANCHOR_K4_BP_ZERO: [f32; 12] = [
    19.653345, 138.85426, 29.888481, 21.03328, 38.416054, 46.53604, 49.13792, 49.90944, 49.16059,
    47.37238, 20.553736, 135.19713,
];

fn assert_close_12(label: &str, got: &[f32], expected: &[f32; 12]) {
    let max_diff = got
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "{label} max abs diff {max_diff:.9e} exceeds 1e-5: got={got:?} expected={expected:?}"
    );
}

#[test]
fn python_flywheel_adapter_golden_parity_across_k0_to_k4_and_base_prior() {
    let model = synthetic_adapter();
    let f = synthetic_fixture(8);
    let bp_zero = [0.0_f32; 12];
    let bp_nz: [f32; 12] = std::array::from_fn(|c| (c as f32 - 5.5) * 0.15);

    let run_case =
        |bp: &[f32; 12], r_idx: [i64; 4], r_fit: [f32; 4], r_gate: [f32; 4], present: [bool; 4]| {
            let inp = PredictCostsInputs {
                batch: 1,
                candidates: 12,
                prog_412: &f.prog_412[..12 * 412],
                proj_16: &f.proj_16[..12 * 16],
                tokens: &f.tokens[..320],
                token_mask: &f.token_mask[..10],
                revealed_indices: &r_idx,
                revealed_fitness: &r_fit,
                revealed_gates: &r_gate,
                ctx_feat: &f.ctx_feat[..584],
                base_prior_cost: bp,
                sos_prior: &f.sos_prior[..24],
                scale: SCALE,
            };
            model
                .predict_costs_with_history_mask(inp, &present)
                .unwrap()
        };

    let c0_zero = run_case(
        &bp_zero,
        [-1, 99, 0, 0],
        [f32::NAN; 4],
        [2.0; 4],
        [false; 4],
    );
    assert_close_12(
        "k0_bp_zero.total",
        &c0_zero.total_cost,
        &PY_TOTAL_K0_BP_ZERO,
    );
    assert_close_12(
        "k0_bp_zero.anchor",
        &c0_zero.anchor_cost,
        &PY_ANCHOR_K0_BP_ZERO,
    );

    let c0_nz = run_case(&bp_nz, [0; 4], [100.0; 4], [0.0; 4], [false; 4]);
    assert_close_12(
        "k0_bp_nonzero.total",
        &c0_nz.total_cost,
        &PY_TOTAL_K0_BP_NONZERO,
    );
    assert_close_12(
        "k0_bp_nonzero.anchor",
        &c0_nz.anchor_cost,
        &PY_ANCHOR_K0_BP_NONZERO,
    );

    // K_rev = 1 placed at physical slot 2 (`[false, false, true, false]`) matches compacted Python `[[2]]`!
    let c1_nz = run_case(
        &bp_nz,
        [-1, -1, 2, -1],
        [f32::NAN, f32::NAN, 0.7, f32::NAN],
        [0.0, 0.0, 1.0, 0.0],
        [false, false, true, false],
    );
    assert_close_12(
        "k1_bp_nonzero.total",
        &c1_nz.total_cost,
        &PY_TOTAL_K1_BP_NONZERO,
    );
    assert_close_12(
        "k1_bp_nonzero.anchor",
        &c1_nz.anchor_cost,
        &PY_ANCHOR_K1_BP_NONZERO,
    );

    // K_rev = 2 placed at slots 0 and 2 (`[true, false, true, false]`) matches Python `[[0, 2]]`!
    let c2_zero = run_case(
        &bp_zero,
        [0, -1, 2, -1],
        [0.2, f32::NAN, 0.7, f32::NAN],
        [1.0, 0.0, 1.0, 0.0],
        [true, false, true, false],
    );
    assert_close_12(
        "k2_bp_zero.total",
        &c2_zero.total_cost,
        &PY_TOTAL_K2_BP_ZERO,
    );
    assert_close_12(
        "k2_bp_zero.anchor",
        &c2_zero.anchor_cost,
        &PY_ANCHOR_K2_BP_ZERO,
    );

    // K_rev = 3 placed at slots 0, 1, 3 (`[true, true, false, true]`) matches Python `[[0, 1, 3]]`!
    let c3_nz = run_case(
        &bp_nz,
        [0, 1, -1, 3],
        [0.2, 100.0, f32::NAN, 3.7],
        [1.0, 0.0, 0.0, 0.0],
        [true, true, false, true],
    );
    assert_close_12(
        "k3_bp_nonzero.total",
        &c3_nz.total_cost,
        &PY_TOTAL_K3_BP_NONZERO,
    );
    assert_close_12(
        "k3_bp_nonzero.anchor",
        &c3_nz.anchor_cost,
        &PY_ANCHOR_K3_BP_NONZERO,
    );

    let c4_zero = run_case(
        &bp_zero,
        [0, 1, 2, 3],
        [0.2, 100.0, 0.7, 3.7],
        [1.0, 0.0, 1.0, 0.0],
        [true; 4],
    );
    assert_close_12(
        "k4_bp_zero.total",
        &c4_zero.total_cost,
        &PY_TOTAL_K4_BP_ZERO,
    );
    assert_close_12(
        "k4_bp_zero.anchor",
        &c4_zero.anchor_cost,
        &PY_ANCHOR_K4_BP_ZERO,
    );

    // Reject non-finite base_prior_cost and out-of-range candidate counts (0 and 4097)
    let mut bad_bp = bp_zero;
    bad_bp[3] = f32::NAN;
    let bad_inp = PredictCostsInputs {
        batch: 1,
        candidates: 12,
        prog_412: &f.prog_412[..12 * 412],
        proj_16: &f.proj_16[..12 * 16],
        tokens: &f.tokens[..320],
        token_mask: &f.token_mask[..10],
        revealed_indices: &[0, 1, 2, 3],
        revealed_fitness: &[0.2; REVEALED],
        revealed_gates: &[1.0; REVEALED],
        ctx_feat: &f.ctx_feat[..584],
        base_prior_cost: &bad_bp,
        sos_prior: &f.sos_prior[..24],
        scale: SCALE,
    };
    assert!(model
        .predict_costs_with_history_mask(bad_inp, &[false; 4])
        .is_err());
    let zero_cands = PredictCostsInputs {
        candidates: 0,
        prog_412: &[],
        proj_16: &[],
        base_prior_cost: &[],
        sos_prior: &[],
        ..bad_inp
    };
    assert!(model
        .predict_costs_with_history_mask(zero_cands, &[false; 4])
        .is_err());
}
