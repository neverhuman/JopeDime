#[cfg(test)]
use super::{
    flat, synthetic_adapter, synthetic_fixture, CpuAdapter, PredictCostsInputs, ANCHOR_12,
    ANCHOR_20, SCALE, TOTAL_12, TOTAL_20,
};
#[cfg(test)]
use candle_core_0_11_0::{Device, Tensor};
#[cfg(test)]
use std::collections::BTreeMap;

#[test]
fn cpu_norm_preserves_translation_of_wide_offset_rows() {
    let device = Device::Cpu;
    let width = 558;
    let adapter = CpuAdapter {
        parameters: BTreeMap::from([
            (
                "wide.weight".into(),
                Tensor::new(vec![1.25f32; width], &device).unwrap(),
            ),
            (
                "wide.bias".into(),
                Tensor::new(vec![-0.5f32; width], &device).unwrap(),
            ),
        ]),
        device: device.clone(),
    };
    // Quarter increments remain exactly representable after this offset.
    let row: Vec<f32> = (0..width).map(|i| (i % 17) as f32 * 0.25).collect();
    let shifted: Vec<f32> = row.iter().map(|x| x + 1_048_576.).collect();
    let input = Tensor::from_vec([row, shifted].concat(), (2, width), &device).unwrap();
    let output = flat(&adapter.norm(&input, "wide").unwrap()).unwrap();
    for (a, b) in output[..width].iter().zip(&output[width..]) {
        assert!(
            (a - b).abs() <= 1e-6,
            "LayerNorm changed under an exact translation: {a} vs {b}"
        );
    }
    let constant = Tensor::new(vec![1_048_576f32; width], &device).unwrap();
    assert!(flat(&adapter.norm(&constant, "wide").unwrap())
        .unwrap()
        .iter()
        .all(|x| *x == -0.5));
}

#[test]
fn full_cpu_graph_matches_nonzero_python_controls_and_admission_on_both_slates() {
    use rand_chacha::{rand_core::SeedableRng, ChaCha12Rng};
    let model = synthetic_adapter();
    let mut decisive = 0;
    for (n, expected, expected_anchor) in [
        (12, TOTAL_12.as_slice(), ANCHOR_12.as_slice()),
        (20, TOTAL_20.as_slice(), ANCHOR_20.as_slice()),
    ] {
        let fixture = synthetic_fixture(n - 4);
        let result = model.predict_costs(fixture.inputs(2, n - 4)).unwrap();
        for (name, actual, oracle) in [
            ("total", result.total_cost.as_slice(), expected),
            ("anchor", result.anchor_cost.as_slice(), expected_anchor),
        ] {
            for (i, (a, e)) in actual.iter().zip(oracle).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
                    "{name}[{i}] actual={a} expected={e}"
                );
                // Stronger synthetic control detects omitted small branches even
                // when the large uncapped W1 failure cost loosens M2's rtol.
                assert!(
                    (a - e).abs() < 8e-5,
                    "{name}[{i}] synthetic error {}",
                    (a - e).abs()
                );
            }
        }
        assert!(result
            .total_cost
            .iter()
            .zip(&result.anchor_cost)
            .any(|(a, b)| (a - b).abs() > 0.1));
        for b in 0..2 {
            let mut order: Vec<usize> = (4..n).collect();
            order.sort_by(|left, right| {
                expected[b * n + *left].total_cmp(&expected[b * n + *right])
            });
            let ranked = if n == 12 { 4 } else { 2 };
            let lo = expected[b * n + order[ranked - 1]];
            let hi = expected[b * n + order[ranked]];
            if hi - lo > 2.0 * (1e-5 + 1e-4 * lo.abs()) {
                decisive += 1;
                let actual: Vec<f64> = result.total_cost[b * n..b * n + n]
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let oracle: Vec<f64> = expected[b * n..b * n + n]
                    .iter()
                    .map(|v| f64::from(*v))
                    .collect();
                let a = crate::ranker::admission::select_costs(
                    &actual,
                    n == 20,
                    &mut ChaCha12Rng::seed_from_u64(71),
                )
                .unwrap();
                let e = crate::ranker::admission::select_costs(
                    &oracle,
                    n == 20,
                    &mut ChaCha12Rng::seed_from_u64(71),
                )
                .unwrap();
                assert_eq!(a, e);
            }
        }
    }
    assert!(
        decisive > 0,
        "synthetic control must exercise a decision outside the tie band"
    );
}

#[test]
fn batch_rows_share_only_their_own_conditioning() {
    let model = synthetic_adapter();
    for proposals in [8, 16] {
        let f = synthetic_fixture(proposals);
        let n = 4 + proposals;
        let all = f.inputs(2, proposals);
        let batched = model.predict_costs(all).unwrap();
        for b in 0..2 {
            let one = PredictCostsInputs {
                batch: 1,
                candidates: n,
                prog_412: &f.prog_412[b * n * 412..(b + 1) * n * 412],
                proj_16: &f.proj_16[b * n * 16..(b + 1) * n * 16],
                tokens: &f.tokens[b * 320..b * 320 + 320],
                token_mask: &f.token_mask[b * 10..b * 10 + 10],
                revealed_indices: &f.revealed_indices[b * 4..b * 4 + 4],
                revealed_fitness: &f.revealed_fitness[b * 4..b * 4 + 4],
                revealed_gates: &f.revealed_gates[b * 4..b * 4 + 4],
                ctx_feat: &f.ctx_feat[b * 584..b * 584 + 584],
                base_prior_cost: &f.base_prior_cost[b * n..b * n + n],
                sos_prior: &f.sos_prior[b * n * 2..(b + 1) * n * 2],
                scale: SCALE,
            };
            assert_eq!(
                model.predict_costs(one).unwrap().total_cost,
                batched.total_cost[b * n..b * n + n]
            );
        }
    }
}

#[test]
fn masked_tokens_are_inert_but_token_zero_is_always_present() {
    let model = synthetic_adapter();
    let mut f = synthetic_fixture(8);
    let original = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
    for b in 0..2 {
        for m in 1..10 {
            if f.token_mask[b * 10 + m] == 0.0 {
                for v in &mut f.tokens[(b * 10 + m) * 32..(b * 10 + m + 1) * 32] {
                    *v += 100.0;
                }
            }
        }
    }
    assert_eq!(
        model.predict_costs(f.inputs(2, 8)).unwrap().total_cost,
        original
    );
    f.token_mask[0] = 1.0;
    assert_eq!(
        model.predict_costs(f.inputs(2, 8)).unwrap().total_cost,
        original
    );
    f.token_mask.fill(0.0);
    let first = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
    for k in 0..32 {
        f.tokens[k] += k as f32 * 0.3;
    }
    let changed = model.predict_costs(f.inputs(2, 8)).unwrap().total_cost;
    assert!(first.iter().zip(changed).any(|(a, b)| (a - b).abs() > 1e-6));
}
