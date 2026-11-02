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

#[path = "tests/cpu_graph.rs"]
mod cpu_graph;
#[path = "tests/shape.rs"]
mod shape;
#[path = "tests/validation.rs"]
mod validation;
