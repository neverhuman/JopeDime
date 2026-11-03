#[test]
fn refuses_nonfinite_inputs_noncanonical_constants_masks_and_reveals() {
    let model = super::synthetic_adapter();
    for case in 0..16 {
        let mut f = super::synthetic_fixture(8);
        match case {
            0 => f.prog_412[0] = f32::NAN,
            1 => f.proj_16[0] = f32::INFINITY,
            2 => f.tokens[0] = f32::NAN,
            3 => f.revealed_fitness[0] = f32::INFINITY,
            4 => f.token_mask[0] = 0.5,
            5 => f.revealed_gates[0] = 2.0,
            6 => f.ctx_feat[0] = 1.0,
            7 => f.base_prior_cost[0] = 1.0,
            8 => f.sos_prior[0] = 1.0,
            9 => f.revealed_indices[0] = -1,
            10 => f.revealed_indices[0] = 12,
            11 => f.revealed_indices[0] = f.revealed_indices[1],
            _ => {}
        }
        let mut input = f.inputs(2, 8);
        match case {
            12 => input.scale = 1.0,
            13 => input.batch = 0,
            14 => input.candidates = 19,
            15 => input.scale = f32::NAN,
            _ => {}
        }
        assert!(
            model.predict_costs(input).is_err(),
            "accepted invalid input case {case}"
        );
    }
}

#[test]
fn loader_refuses_missing_misshaped_nonfinite_and_bad_scaler_parameters() {
    for case in 0..5 {
        let bytes = super::synthetic_weights(|p| match case {
            0 => {
                p.remove("anchor.alpha_attn");
            }
            1 => p.get_mut("modality_norm.weight").unwrap().0 = vec![1, 32],
            2 => p.get_mut("prog_mean").unwrap().1[0] = f32::NAN,
            3 => p.get_mut("prog_std").unwrap().1[0] = 0.0,
            4 => p.get_mut("prog_std").unwrap().1[0] = -1.0,
            _ => unreachable!(),
        });
        assert!(
            super::CpuAdapter::from_bytes(&bytes).is_err(),
            "accepted invalid parameter case {case}"
        );
    }
    let mut bytes = super::synthetic_weights(|_| {});
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec())
        .unwrap()
        .replace("F32", "I32");
    bytes[8..8 + header_len].copy_from_slice(header.as_bytes());
    assert!(super::CpuAdapter::from_bytes(&bytes).is_err());
    assert!(super::CpuAdapter::from_bytes(&u64::MAX.to_le_bytes()).is_err());
}

#[test]
fn kl_readout_has_a_known_zero_and_flags_large_proposal_shift() {
    assert_eq!(super::proposal_kl(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    let shifted = super::proposal_kl(&[0.0, 10.0], &[10.0, 0.0]);
    assert!((shifted - 9.9990920426).abs() < 1e-8);
    assert!(shifted > 0.35);
    assert!(super::proposal_kl(&[f32::MAX, -f32::MAX], &[-f32::MAX, f32::MAX]).is_finite());
}

#[test]
#[ignore = "requires RES-07 real serving-slate fixtures for >=50 contexts; synthetic controls are not M2"]
fn real_serving_slate_cost_and_admitted_set_m2_parity() {
    use rand_chacha::{rand_core::SeedableRng, ChaCha12Rng};
    let fixtures = super::real_cpu_parity::Fixtures::open();
    let adapter = super::CpuAdapter::from_bytes(&fixtures.weights()).unwrap();
    let mut stats = super::real_cpu_parity::Stats::default();
    let (mut decisive, mut ties, mut admission_failures) = (0, 0, 0);
    let mut max_kl = 0.0_f64;
    for cid in &fixtures.contexts {
        for width in [12, 20] {
            let f = fixtures.load(&format!("slates/{cid}-{width}.safetensors"));
            let prog = f.f32("input.prog_412");
            let proj = f.f32("input.proj_16");
            let tokens = f.f32("input.mm_tokens_10x32");
            let mask = f.f32("input.mm_mask_10");
            let indices = f.i64("input.revealed_indices");
            let fitness = f.f32("input.revealed_raw_f");
            let gates = f.f32("input.revealed_gate");
            let ctx = f.f32("input.ctx_feat");
            let prior = f.f32("input.base_prior_cost");
            let sos = f.f32("input.sos_prior");
            let scale = f.f32("input.scale");
            let actual = adapter
                .predict_costs(super::PredictCostsInputs {
                    batch: 1,
                    candidates: width,
                    prog_412: &prog,
                    proj_16: &proj,
                    tokens: &tokens,
                    token_mask: &mask,
                    revealed_indices: &indices,
                    revealed_fitness: &fitness,
                    revealed_gates: &gates,
                    ctx_feat: &ctx,
                    base_prior_cost: &prior,
                    sos_prior: &sos,
                    scale: scale[0],
                })
                .unwrap();
            let expected = f.numbers("output.total_cost");
            stats.compare(&actual.total_cost, &expected);
            max_kl = max_kl.max(actual.proposal_kl_to_anchor[0]);
            let mut order: Vec<_> = (4..width).collect();
            order.sort_by(|&a, &b| expected[a].total_cmp(&expected[b]).then_with(|| a.cmp(&b)));
            let cutoff = if width == 12 { 4 } else { 2 };
            let (a, b) = (expected[order[cutoff - 1]], expected[order[cutoff]]);
            let band = 2.0 * (1e-5 + 1e-4 * a.abs().max(b.abs()));
            if b - a > band {
                decisive += 1;
                let mut r1 = ChaCha12Rng::from_seed([42; 32]);
                let mut r2 = r1.clone();
                let observed: Vec<_> = actual.total_cost.iter().map(|v| f64::from(*v)).collect();
                let selected =
                    crate::ranker::admission::select_costs(&observed, width == 20, &mut r1)
                        .unwrap();
                let reference =
                    crate::ranker::admission::select_costs(&expected, width == 20, &mut r2)
                        .unwrap();
                let mut a = selected;
                let mut b = reference;
                a.sort_unstable();
                b.sort_unstable();
                admission_failures += usize::from(a != b);
            } else {
                ties += 1;
            }
        }
    }
    stats.failures += admission_failures;
    stats.finish("RNK-08", serde_json::json!({"contexts":50,"slates":100,"decisive_admission_slates":decisive,
            "tie_band_slates":ties,"admitted_set_mismatches":admission_failures,"max_proposal_KL":max_kl,"KL_above_0_35":max_kl>0.35,
            "randomness":"same persisted ChaCha12 seed for both cost vectors; not NumPy RNG qualification"}));
}
