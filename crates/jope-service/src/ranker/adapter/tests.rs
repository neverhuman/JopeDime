use super::*;

struct Fixture {
    prog_412: Vec<f32>,
    proj_16: Vec<f32>,
    tokens: Vec<f32>,
    token_mask: Vec<f32>,
    revealed_indices: Vec<i64>,
    revealed_fitness: Vec<f32>,
    revealed_gates: Vec<f32>,
    ctx_feat: Vec<f32>,
    base_prior_cost: Vec<f32>,
    sos_prior: Vec<f32>,
}

impl Fixture {
    fn serving(batch: usize, proposals: usize) -> Self {
        let candidates = REVEALED + proposals;
        let bn = batch * candidates;
        Self {
            prog_412: vec![0.0; bn * PROG_FEATURES],
            proj_16: vec![0.0; bn * PROJ_FEATURES],
            tokens: vec![0.0; batch * TOKENS * TOKEN_WIDTH],
            token_mask: vec![0.0; batch * TOKENS],
            revealed_indices: (0..batch).flat_map(|_| 0..REVEALED as i64).collect(),
            revealed_fitness: vec![0.0; batch * REVEALED],
            revealed_gates: vec![0.0; batch * REVEALED],
            ctx_feat: vec![0.0; batch * CONTEXT_FEATURES],
            base_prior_cost: vec![0.0; bn],
            sos_prior: vec![0.0; bn * SOS_FEATURES],
        }
    }

    fn inputs(&self, batch: usize, proposals: usize) -> PredictCostsInputs<'_> {
        PredictCostsInputs {
            batch,
            candidates: REVEALED + proposals,
            prog_412: &self.prog_412,
            proj_16: &self.proj_16,
            tokens: &self.tokens,
            token_mask: &self.token_mask,
            revealed_indices: &self.revealed_indices,
            revealed_fitness: &self.revealed_fitness,
            revealed_gates: &self.revealed_gates,
            ctx_feat: &self.ctx_feat,
            base_prior_cost: &self.base_prior_cost,
            sos_prior: &self.sos_prior,
            scale: SCALE,
        }
    }
}

#[test]
fn accepts_both_serving_slate_shapes() {
    for proposals in [8, 16] {
        let fixture = Fixture::serving(2, proposals);
        assert_eq!(fixture.inputs(2, proposals).validate_shapes(), Ok(()));
    }
}

#[test]
fn reports_a_mismatch_for_every_flattened_f12_input_shape() {
    for proposals in [8, 16] {
        let fixture = Fixture::serving(2, proposals);
        let valid = fixture.inputs(2, proposals);

        macro_rules! assert_one_short {
            ($field:ident, $name:literal) => {{
                let actual = fixture.$field.len() - 1;
                let malformed = PredictCostsInputs {
                    $field: &fixture.$field[..actual],
                    ..valid
                };
                let error = malformed.validate_shapes().unwrap_err();
                assert_eq!(error.field, $name);
                assert_eq!(error.expected, fixture.$field.len());
                assert_eq!(error.actual, actual);
            }};
        }

        assert_one_short!(prog_412, "prog_412");
        assert_one_short!(proj_16, "proj_16");
        assert_one_short!(tokens, "tokens");
        assert_one_short!(token_mask, "token_mask");
        assert_one_short!(revealed_indices, "revealed_indices");
        assert_one_short!(revealed_fitness, "revealed_fitness");
        assert_one_short!(revealed_gates, "revealed_gates");
        assert_one_short!(ctx_feat, "ctx_feat");
        assert_one_short!(base_prior_cost, "base_prior_cost");
        assert_one_short!(sos_prior, "sos_prior");
    }
}

const TOTAL_12: [f32; 24] = [
    f32::from_bits(0x41a08967),
    f32::from_bits(0x430c0fd4),
    f32::from_bits(0x41f32476),
    f32::from_bits(0x41b15854),
    f32::from_bits(0x421ceae8),
    f32::from_bits(0x423c9991),
    f32::from_bits(0x4247eb94),
    f32::from_bits(0x424a5283),
    f32::from_bits(0x4247a1ca),
    f32::from_bits(0x424038a5),
    f32::from_bits(0x41a9ed52),
    f32::from_bits(0x4307eab9),
    f32::from_bits(0x410b3f37),
    f32::from_bits(0x41e6367b),
    f32::from_bits(0x416f0610),
    f32::from_bits(0x40b02f44),
    f32::from_bits(0x414a6145),
    f32::from_bits(0x414d8686),
    f32::from_bits(0x4156ecc2),
    f32::from_bits(0x415eb38b),
    f32::from_bits(0x41693c78),
    f32::from_bits(0x4159b47c),
    f32::from_bits(0x41bd0c4b),
    f32::from_bits(0x4189a479),
];
const ANCHOR_12: [f32; 24] = [
    f32::from_bits(0x419d3a0d),
    f32::from_bits(0x430adab1),
    f32::from_bits(0x41ef1b9c),
    f32::from_bits(0x41a84428),
    f32::from_bits(0x4219aa0a),
    f32::from_bits(0x423a24e8),
    f32::from_bits(0x42448d3b),
    f32::from_bits(0x4247a344),
    f32::from_bits(0x4244a472),
    f32::from_bits(0x423d7d51),
    f32::from_bits(0x41a46e0d),
    f32::from_bits(0x43073277),
    f32::from_bits(0x40ee8d77),
    f32::from_bits(0x41bc7590),
    f32::from_bits(0x41440c3f),
    f32::from_bits(0x408628ac),
    f32::from_bits(0x411e5f20),
    f32::from_bits(0x412968a1),
    f32::from_bits(0x4135dad8),
    f32::from_bits(0x413a7222),
    f32::from_bits(0x413f80fe),
    f32::from_bits(0x4137fb46),
    f32::from_bits(0x41aacc23),
    f32::from_bits(0x416b7a6c),
];
const TOTAL_20: [f32; 40] = [
    f32::from_bits(0x41a08662),
    f32::from_bits(0x430c0fd5),
    f32::from_bits(0x41f322c3),
    f32::from_bits(0x41b157d6),
    f32::from_bits(0x421cea8c),
    f32::from_bits(0x423c98bb),
    f32::from_bits(0x4247e979),
    f32::from_bits(0x424a5356),
    f32::from_bits(0x4247a252),
    f32::from_bits(0x42403907),
    f32::from_bits(0x41a9ed57),
    f32::from_bits(0x4307eab3),
    f32::from_bits(0x422e639c),
    f32::from_bits(0x421f1229),
    f32::from_bits(0x423238e4),
    f32::from_bits(0x423cad7b),
    f32::from_bits(0x4246953b),
    f32::from_bits(0x42494abc),
    f32::from_bits(0x424279db),
    f32::from_bits(0x422cda12),
    f32::from_bits(0x410b3a3a),
    f32::from_bits(0x41e6317b),
    f32::from_bits(0x416f04a0),
    f32::from_bits(0x40b034f2),
    f32::from_bits(0x414a5f4e),
    f32::from_bits(0x414d8e48),
    f32::from_bits(0x4156eb94),
    f32::from_bits(0x415ebd96),
    f32::from_bits(0x41693a09),
    f32::from_bits(0x4159c2e3),
    f32::from_bits(0x41bd1119),
    f32::from_bits(0x4189a3bd),
    f32::from_bits(0x413ea8cc),
    f32::from_bits(0x40e92d5d),
    f32::from_bits(0x4145be65),
    f32::from_bits(0x415236a2),
    f32::from_bits(0x415c9654),
    f32::from_bits(0x416224b4),
    f32::from_bits(0x4162cbac),
    f32::from_bits(0x4156b367),
];
const ANCHOR_20: [f32; 40] = [
    f32::from_bits(0x419d3a0d),
    f32::from_bits(0x430adab1),
    f32::from_bits(0x41ef1b9c),
    f32::from_bits(0x41a84428),
    f32::from_bits(0x4219aa0a),
    f32::from_bits(0x423a24e8),
    f32::from_bits(0x42448d3b),
    f32::from_bits(0x4247a344),
    f32::from_bits(0x4244a472),
    f32::from_bits(0x423d7d51),
    f32::from_bits(0x41a46e0d),
    f32::from_bits(0x43073277),
    f32::from_bits(0x422bd528),
    f32::from_bits(0x421b32ed),
    f32::from_bits(0x422f9062),
    f32::from_bits(0x42399916),
    f32::from_bits(0x424390d7),
    f32::from_bits(0x42469419),
    f32::from_bits(0x423f8c80),
    f32::from_bits(0x422a4998),
    f32::from_bits(0x40ee8d77),
    f32::from_bits(0x41bc7590),
    f32::from_bits(0x41440c3f),
    f32::from_bits(0x408628ac),
    f32::from_bits(0x411e5f20),
    f32::from_bits(0x412968a1),
    f32::from_bits(0x4135dad8),
    f32::from_bits(0x413a7222),
    f32::from_bits(0x413f80fe),
    f32::from_bits(0x4137fb46),
    f32::from_bits(0x41aacc23),
    f32::from_bits(0x416b7a6c),
    f32::from_bits(0x4120996b),
    f32::from_bits(0x4095f12d),
    f32::from_bits(0x411eb10a),
    f32::from_bits(0x41303688),
    f32::from_bits(0x4138d3a9),
    f32::from_bits(0x413bdfc8),
    f32::from_bits(0x413f43dc),
    f32::from_bits(0x41389f45),
];

type FixtureParameters = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn synthetic_weights(change: impl FnOnce(&mut FixtureParameters)) -> Vec<u8> {
    let mut parameters: FixtureParameters = parameter_specs()
        .into_iter()
        .map(|(name, shape)| {
            let count = shape.iter().product::<usize>();
            let salt = name.bytes().map(usize::from).sum::<usize>() % 97;
            let values = (0..count)
                .map(|i| {
                    let value = if name == "prog_std" {
                        1.0 + (i % 7) as f32 * 0.05
                    } else if name == "prog_mean" {
                        ((i % 17) as i32 - 8) as f32 * 0.01
                    } else if shape.is_empty() {
                        match name.as_str() {
                            "anchor.alpha_prior" => 0.08,
                            "anchor.alpha_struct_kernel" => 0.15,
                            "anchor.alpha_attn" | "anchor.gate_barrier_weight" => 0.05,
                            "mm_prior_scale" => 0.31,
                            "null_kernel_w0_raw" => -0.5,
                            "sem_kernel_scale_raw" => 1.2,
                            "emb_kernel_scale_raw" => 0.8,
                            "traj_kernel_blend" => 0.25,
                            "traj_attn_scale" => 0.27,
                            _ => panic!("unknown synthetic scalar"),
                        }
                    } else if name.ends_with(".weight") && shape.len() == 1 {
                        1.0 + (((i + salt) % 7) as i32 - 3) as f32 * 0.01
                    } else {
                        (((i * 37 + salt) % 101) as i32 - 50) as f32 * 0.002
                    };
                    if name == "mm_prior_out.2.weight" {
                        value * 50.0
                    } else {
                        value
                    }
                })
                .collect();
            (name, (shape, values))
        })
        .collect();
    change(&mut parameters);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, (shape, values)) in parameters {
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            format!("adapter_10tok.{name}"),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

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

fn synthetic_adapter() -> CpuAdapter {
    CpuAdapter::from_bytes(&synthetic_weights(|_| {})).unwrap()
}

fn synthetic_fixture(proposals: usize) -> Fixture {
    let mut f = Fixture::serving(2, proposals);
    let n = 4 + proposals;
    for b in 0..2 {
        for c in 0..n {
            for k in 0..412 {
                f.prog_412[(b * n + c) * 412 + k] =
                    (((b * 31 + c * 17 + k * 7) % 41) as i32 - 20) as f32 * 0.12;
            }
            for k in 0..16 {
                f.proj_16[(b * n + c) * 16 + k] =
                    (((b * 13 + c * 3 + k * 11) % 29) as i32 - 14) as f32 * 0.07;
            }
        }
        for m in 0..10 {
            f.token_mask[b * 10 + m] = if (b + m) % 3 == 0 { 0.0 } else { 1.0 };
            for k in 0..32 {
                f.tokens[(b * 10 + m) * 32 + k] =
                    (((b * 7 + m * 3 + k) % 23) as i32 - 11) as f32 * 0.09;
            }
        }
    }
    f.revealed_indices = vec![0, 1, 2, 3, 3, 1, 0, 2];
    f.revealed_fitness = vec![0.2, 100.0, 0.7, 3.7, 0.4, 12.5, 1.1, 5.0];
    f.revealed_gates = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    f
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
                let a = super::super::admission::select_costs(
                    &actual,
                    n == 20,
                    &mut ChaCha12Rng::seed_from_u64(71),
                )
                .unwrap();
                let e = super::super::admission::select_costs(
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

#[test]
fn refuses_nonfinite_inputs_noncanonical_constants_masks_and_reveals() {
    let model = synthetic_adapter();
    for case in 0..16 {
        let mut f = synthetic_fixture(8);
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
        let bytes = synthetic_weights(|p| match case {
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
            CpuAdapter::from_bytes(&bytes).is_err(),
            "accepted invalid parameter case {case}"
        );
    }
    let mut bytes = synthetic_weights(|_| {});
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec())
        .unwrap()
        .replace("F32", "I32");
    bytes[8..8 + header_len].copy_from_slice(header.as_bytes());
    assert!(CpuAdapter::from_bytes(&bytes).is_err());
    assert!(CpuAdapter::from_bytes(&u64::MAX.to_le_bytes()).is_err());
}

#[test]
fn kl_readout_has_a_known_zero_and_flags_large_proposal_shift() {
    assert_eq!(proposal_kl(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
    let shifted = proposal_kl(&[0.0, 10.0], &[10.0, 0.0]);
    assert!((shifted - 9.9990920426).abs() < 1e-8);
    assert!(shifted > 0.35);
    assert!(proposal_kl(&[f32::MAX, -f32::MAX], &[-f32::MAX, f32::MAX]).is_finite());
}

#[test]
#[ignore = "requires RES-07 real serving-slate fixtures for >=50 contexts; synthetic controls are not M2"]
fn real_serving_slate_cost_and_admitted_set_m2_parity() {
    use rand_chacha::{rand_core::SeedableRng, ChaCha12Rng};
    let fixtures = super::real_cpu_parity::Fixtures::open();
    let adapter = CpuAdapter::from_bytes(&fixtures.weights()).unwrap();
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
                .predict_costs(PredictCostsInputs {
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
