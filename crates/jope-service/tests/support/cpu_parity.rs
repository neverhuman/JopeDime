//! Strict external, hash-pinned real CPU fixtures, shared by two unit-test modules.
#![allow(dead_code)] // Each consumer uses a different subset of this test-only helper.
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Component, Path, PathBuf},
};

const BUNDLE: &str = "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c";
const JOPE_SHA: &str = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";

pub struct Fixtures {
    root: PathBuf,
    pub index: Value,
    pub contexts: Vec<String>,
    pub weights: PathBuf,
}
pub struct TensorFile(Vec<u8>);

fn hashed(path: &Path, expected: &str) -> Vec<u8> {
    assert!(!fs::symlink_metadata(path)
        .expect("required real fixture")
        .file_type()
        .is_symlink());
    assert!(
        fs::metadata(path).unwrap().len() <= 64 * 1024 * 1024,
        "fixture byte limit"
    );
    let bytes = fs::read(path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        expected,
        "fixture identity: {}",
        path.display()
    );
    bytes
}

impl Fixtures {
    pub fn open() -> Self {
        let root = PathBuf::from(
            env::var_os("JOPE_GOLDEN_DIR")
                .expect("JOPE_GOLDEN_DIR required; missing real data is a failure"),
        );
        let pin =
            env::var("JOPE_GOLDEN_INDEX_SHA256").expect("explicit trusted index SHA required");
        let index: Value = serde_json::from_slice(&hashed(&root.join("index.json"), &pin)).unwrap();
        assert_eq!(index["schema"], "jope.golden-fixtures/v1");
        assert_eq!(index["bundle_digest"], BUNDLE);
        assert_eq!(index["contexts"], 50);
        let contexts: Vec<_> = index["files"]
            .as_object()
            .unwrap()
            .keys()
            .filter_map(|name| {
                name.strip_prefix("contexts/")?
                    .strip_suffix(".safetensors")
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(contexts.len(), 50);
        assert!(contexts.iter().all(|c| c.len() == 64
            && c.bytes()
                .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())));
        let weights = PathBuf::from(
            env::var_os("JOPE_PARITY_BUNDLE_DIR").expect("real frozen RES06 bundle required"),
        )
        .join("jope.safetensors");
        hashed(&weights, JOPE_SHA);
        Self {
            root: root.canonicalize().unwrap(),
            index,
            contexts,
            weights,
        }
    }
    pub fn weights(&self) -> Vec<u8> {
        hashed(&self.weights, JOPE_SHA)
    }
    pub fn load(&self, name: &str) -> TensorFile {
        assert!(Path::new(name)
            .components()
            .all(|p| matches!(p, Component::Normal(_))));
        let path = self.root.join(name);
        assert!(
            path.canonicalize().unwrap().starts_with(&self.root),
            "fixture path escapes root"
        );
        let spec = &self.index["files"][name];
        let bytes = hashed(
            &path,
            spec["sha256"]
                .as_str()
                .expect("fixture absent from pinned inventory"),
        );
        assert_eq!(bytes.len() as u64, spec["bytes"].as_u64().unwrap());
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let schema = spec["tensors"].as_object().unwrap();
        assert_eq!(tensors.len(), schema.len());
        for (key, expected) in schema {
            let t = tensors.tensor(key).unwrap();
            assert_eq!(serde_json::to_value(t.shape()).unwrap(), expected["shape"]);
            assert_eq!(
                format!("{:?}", t.dtype()),
                expected["dtype"].as_str().unwrap()
            );
        }
        TensorFile(bytes)
    }
}
impl TensorFile {
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let ts = SafeTensors::deserialize(&self.0).unwrap();
        let t = ts.tensor(name).unwrap();
        assert_eq!(t.dtype(), Dtype::F32);
        t.data()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    pub fn i64(&self, name: &str) -> Vec<i64> {
        let ts = SafeTensors::deserialize(&self.0).unwrap();
        let t = ts.tensor(name).unwrap();
        assert_eq!(t.dtype(), Dtype::I64);
        t.data()
            .chunks_exact(8)
            .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }
    pub fn numbers(&self, name: &str) -> Vec<f64> {
        let ts = SafeTensors::deserialize(&self.0).unwrap();
        let t = ts.tensor(name).unwrap();
        match t.dtype() {
            Dtype::F32 => t
                .data()
                .chunks_exact(4)
                .map(|v| f64::from(f32::from_le_bytes(v.try_into().unwrap())))
                .collect(),
            Dtype::F64 => t
                .data()
                .chunks_exact(8)
                .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
                .collect(),
            _ => panic!("expected floating-point oracle: {name}"),
        }
    }
}
#[derive(Default)]
pub struct Stats {
    pub values: usize,
    pub failures: usize,
    pub max_abs: f64,
    pub max_tolerance_ratio: f64,
}
impl Stats {
    pub fn compare(&mut self, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            let error = (f64::from(a) - b).abs();
            let tolerance = 1e-5 + 1e-4 * b.abs();
            self.values += 1;
            self.failures += usize::from(error > tolerance);
            self.max_abs = self.max_abs.max(error);
            self.max_tolerance_ratio = self.max_tolerance_ratio.max(error / tolerance);
        }
    }
    pub fn finish(&self, name: &str, extra: Value) {
        let receipt = json!({"consumer":name,"device":"cpu","atol":1e-5,"rtol":1e-4,"values":self.values,"failed_values":self.failures,
            "max_absolute_error":self.max_abs,"max_tolerance_ratio":self.max_tolerance_ratio,"passed":self.failures==0,"details":extra});
        println!("{receipt}");
        let root = PathBuf::from(
            env::var_os("JOPE_PARITY_REPORT_DIR")
                .expect("external parity report directory required"),
        );
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        assert_eq!(
            self.failures, 0,
            "real CPU oracle differs; receipt retained"
        );
    }
}

// RNK-10 composition consumes the same hash-pinned real packets as RNK-08.
// Counterfactual perturbations below are controls, not measured quality results.
#[test]
#[ignore = "requires hash-pinned RES-07 CPU packets and frozen RES06 weights"]
fn real_round_admission_composition() {
    round_admission_fixture(false);
}

// Measurement execution is separate from the M5 verdict. The release provider
// requires positive 4+8 controls from this report, even when this case succeeds.
#[test]
#[ignore = "requires hash-pinned RES-07 CPU packets and frozen RES06 weights"]
fn m5_serving_causal_measurements() {
    round_admission_fixture(true);
}

fn round_admission_fixture(causal_measurements_only: bool) {
    use crate::discover::r#loop::{Plan, RoundAdmission, RoundContext, RoundMeasurement};
    use crate::ranker::admission::{
        select_costs, CpuRoundAdmission, RevealedRule, RoundInputProvider, RoundInputs,
        RoundProgram,
    };
    use crate::ranker::stage1b::TokenProjection;
    use rand_chacha::{
        rand_core::{RngCore, SeedableRng},
        ChaCha12Rng,
    };
    use std::cell::Cell;

    struct Provider<'a> {
        dsls: &'a [String],
        prog: Vec<f32>,
        proj: Vec<f32>,
        tokens: Vec<f32>,
        mask: Vec<f32>,
        gates: Vec<f32>,
        mode: &'a Cell<u8>,
    }
    impl RoundInputProvider for Provider<'_> {
        fn observe(
            &self,
            _: &RoundContext<'_>,
            proposals: &[String],
        ) -> anyhow::Result<RoundInputs> {
            let program = |i: usize| RoundProgram {
                dsl: self.dsls[i].clone(),
                features412: self.prog[i * 412..(i + 1) * 412].try_into().unwrap(),
                projection16: self.proj[i * 16..(i + 1) * 16].try_into().unwrap(),
            };
            let mut result = RoundInputs {
                pool: (0..4).map(program).collect(),
                pool_gates: self
                    .gates
                    .iter()
                    .map(|&g| {
                        assert!(g == 0.0 || g == 1.0);
                        g == 1.0
                    })
                    .collect(),
                proposals: (4..4 + proposals.len()).map(program).collect(),
                conditioning: TokenProjection {
                    tokens: std::array::from_fn(|i| {
                        self.tokens[i * 32..(i + 1) * 32].try_into().unwrap()
                    }),
                    mask: self.mask.as_slice().try_into().unwrap(),
                },
            };
            match self.mode.get() {
                1 => {
                    result.conditioning.tokens = [[0.0; 32]; 10];
                    result.conditioning.mask = [0.0; 10];
                }
                2 => result.proposals.swap(0, 1),
                3 => result.pool.swap(0, 1),
                4 => {
                    result.pool_gates.pop();
                }
                5..=7 => {
                    // These slots depend on landmarks; preserve DOPE-only slots
                    // and every mask bit so this is not a missing-input control.
                    for slot in [0, 1, 7, 9] {
                        for value in &mut result.conditioning.tokens[slot] {
                            *value = [-6.0, 0.0, 6.0][usize::from(self.mode.get() - 5)];
                        }
                    }
                }
                _ => {}
            }
            Ok(result)
        }
    }
    let fixtures = Fixtures::open();
    let spec = &fixtures.index["files"]["programs.jsonl"];
    let bytes = hashed(
        &fixtures.root.join("programs.jsonl"),
        spec["sha256"].as_str().unwrap(),
    );
    let dsls: Vec<String> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["dsl"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .take(20)
        .collect();
    assert_eq!(dsls.len(), 20);
    let plan = Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: "a".repeat(64),
        bundle_digest: BUNDLE.strip_prefix("sha256:").unwrap().into(),
        binary_version: "RNK10-real-CPU-replay".into(),
        matrix_path: "FIT-only-not-opened-by-replay".into(),
        matrix_sha256: "b".repeat(64),
        target: "target".into(),
        features: vec![],
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort: jope_api::Effort::Fast,
        evaluations: 24,
        wall_seconds: 60,
        evaluation_wall_ms: 15_000,
        evaluation_rss_bytes: 2 * 1024 * 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let mut stats = Stats::default();
    #[cfg(feature = "research")]
    let (mut recorded_slates, mut recording_refusals) = (0, 0);
    let (mut decisive, mut ties, mut mismatches) = (0, 0, 0);
    let (mut landmark_changes, mut fitness_changes, mut disconnected_changes) = (0, 0, 0);
    let (
        mut serving_slates,
        mut serving_landmark_changes,
        mut serving_fitness_changes,
        mut serving_disconnected_changes,
    ) = (0, 0, 0, 0);
    let mut max_kl = 0.0_f64;
    let weights = fixtures.weights();
    for cid in &fixtures.contexts {
        for width in [12, 20] {
            let f = fixtures.load(&format!("slates/{cid}-{width}.safetensors"));
            assert_eq!(f.i64("input.revealed_indices"), vec![0, 1, 2, 3]);
            let fitness = f.f32("input.revealed_raw_f");
            let mode = Cell::new(0);
            let provider = Provider {
                dsls: &dsls,
                prog: f.f32("input.prog_412"),
                proj: f.f32("input.proj_16"),
                tokens: f.f32("input.mm_tokens_10x32"),
                mask: f.f32("input.mm_mask_10"),
                gates: f.f32("input.revealed_gate"),
                mode: &mode,
            };
            let admission = CpuRoundAdmission::for_cpu_replay(
                super::CpuAdapter::from_bytes(&weights).unwrap(),
                plan.clone(),
                provider,
                RevealedRule::StorageOrder,
            );
            let ids: Vec<_> = (0..4).map(|i| format!("alg_{i:08}")).collect();
            let mut context = RoundContext {
                plan: &plan,
                round: 1,
                measured: (0..4)
                    .map(|i| RoundMeasurement {
                        algorithm_id: &ids[i],
                        dsl: &dsls[i],
                        surrogate_fitness: f64::from(fitness[i]),
                    })
                    .collect(),
            };
            let proposals = dsls[4..width].to_vec();
            let rng = ChaCha12Rng::from_seed([42; 32]);
            let mut actual_rng = rng.clone();
            let actual = admission
                .evaluate_round(&context, &proposals, width == 20, &mut actual_rng)
                .unwrap();
            let expected = f.numbers("output.total_cost");
            stats.compare(&actual.costs.total_cost, &expected);
            max_kl = max_kl.max(actual.costs.proposal_kl_to_anchor[0]);
            let mut reference_rng = rng.clone();
            let indices = select_costs(&expected, width == 20, &mut reference_rng).unwrap();
            let reference: Vec<_> = indices.iter().map(|&i| proposals[i].clone()).collect();
            assert_eq!(
                actual_rng.next_u64(),
                reference_rng.next_u64(),
                "same round RNG consumption"
            );
            let mut order: Vec<_> = expected[4..].to_vec();
            order.sort_by(f64::total_cmp);
            let cutoff = if width == 12 { 4 } else { 2 };
            let (a, b) = (order[cutoff - 1], order[cutoff]);
            if b - a > 2.0 * (1e-5 + 1e-4 * a.abs().max(b.abs())) {
                decisive += 1;
                mismatches += usize::from(actual.admitted != reference);
            } else {
                ties += 1;
            }
            disconnected_changes += usize::from(actual.admitted != proposals[..4]);
            if width == 12 {
                serving_slates += 1;
                serving_disconnected_changes += usize::from(actual.admitted != proposals[..4]);
                for intervention in 5..=7 {
                    mode.set(intervention);
                    let landmark_only = admission
                        .evaluate_round(&context, &proposals, false, &mut rng.clone())
                        .unwrap();
                    if landmark_only.admitted != actual.admitted {
                        serving_landmark_changes += 1;
                        break;
                    }
                }
            }
            mode.set(1);
            let changed = admission
                .evaluate_round(&context, &proposals, width == 20, &mut rng.clone())
                .unwrap();
            landmark_changes += usize::from(changed.admitted != actual.admitted);
            mode.set(0);
            for (i, m) in context.measured.iter_mut().enumerate() {
                m.surrogate_fitness = f64::from(fitness[3 - i]);
            }
            let changed = admission
                .evaluate_round(&context, &proposals, width == 20, &mut rng.clone())
                .unwrap();
            fitness_changes += usize::from(changed.admitted != actual.admitted);
            if width == 12 {
                serving_fitness_changes += usize::from(changed.admitted != actual.admitted);
            }
            for (i, m) in context.measured.iter_mut().enumerate() {
                m.surrogate_fitness = f64::from(fitness[i]);
            }
            assert_eq!(
                admission
                    .admit_round(&context, proposals.clone(), width == 20, &mut rng.clone())
                    .unwrap(),
                actual.admitted
            );
            #[cfg(feature = "research")]
            {
                use crate::discover::r#loop::RecordedAdmissionContext;
                use crate::research::{
                    ObservedSlate, RecordedPolicyKind, RecordedProgram, RecordedReveal,
                };
                let prog = f.f32("input.prog_412");
                let proj = f.f32("input.proj_16");
                let gates = f.f32("input.revealed_gate");
                let tokens = f.f32("input.mm_tokens_10x32");
                let mask = f.f32("input.mm_mask_10");
                let program = |i: usize| RecordedProgram {
                    program_sha256: format!("{:x}", Sha256::digest(dsls[i].as_bytes())),
                    // Record metadata is a software control, not an Engine evidence grant.
                    operator_family: "fixture_observation".into(),
                    feat412: prog[i * 412..(i + 1) * 412].to_vec(),
                    proj16: proj[i * 16..(i + 1) * 16]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect(),
                };
                let observed = ObservedSlate {
                    programs: (4..width).map(program).collect(),
                    revealed: (0..4)
                        .map(|i| RecordedReveal {
                            program: program(i),
                            surrogate: f64::from(fitness[i]),
                            gate: gates[i] == 1.0,
                        })
                        .collect(),
                };
                let conditioning = TokenProjection {
                    tokens: std::array::from_fn(|i| {
                        tokens[i * 32..(i + 1) * 32].try_into().unwrap()
                    }),
                    mask: mask.as_slice().try_into().unwrap(),
                };
                let mut recorded_rng = rng.clone();
                let policy = admission
                    .admit_recorded_round(
                        &context,
                        proposals.clone(),
                        width == 20,
                        &mut recorded_rng,
                        &RecordedAdmissionContext {
                            slate: &observed,
                            conditioning: TokenProjection {
                                tokens: conditioning.tokens,
                                mask: conditioning.mask,
                            },
                        },
                    )
                    .unwrap();
                assert_eq!(policy.kind, RecordedPolicyKind::Learned);
                assert_eq!(policy.explore, width == 20);
                assert_eq!(policy.admitted, actual.admitted_indices);
                let actual_costs: Vec<_> = actual
                    .costs
                    .total_cost
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                assert_eq!(policy.slate_costs.as_ref().unwrap(), &actual_costs);
                let mut expected_rng = rng.clone();
                select_costs(&actual_costs, width == 20, &mut expected_rng).unwrap();
                assert_eq!(recorded_rng.next_u64(), expected_rng.next_u64());
                recorded_slates += 1;
                for bad in 0..15 {
                    let mut changed = ObservedSlate {
                        programs: observed.programs.clone(),
                        revealed: observed.revealed.clone(),
                    };
                    let mut changed_conditioning = TokenProjection {
                        tokens: conditioning.tokens,
                        mask: conditioning.mask,
                    };
                    match bad {
                        0 => changed.programs[0].program_sha256 = "c".repeat(64),
                        1 => changed.programs[0].feat412[0] += 1.0,
                        2 => changed.programs[0].proj16[0] += 1.0,
                        3 => changed.revealed.swap(0, 1),
                        // One f64 ULP can survive as the same graph f32: raw recorded
                        // fitness still must equal the committed measurement exactly.
                        4 => {
                            changed.revealed[0].surrogate =
                                f64::from_bits(changed.revealed[0].surrogate.to_bits() ^ 1)
                        }
                        5 => changed.revealed[0].gate = !changed.revealed[0].gate,
                        6 => changed_conditioning.tokens[0][0] += 1.0,
                        7 => changed_conditioning.mask[0] = 1.0 - changed_conditioning.mask[0],
                        8 => {
                            changed.revealed.pop();
                        }
                        9 => {
                            changed.programs.pop();
                        }
                        10 => changed.revealed[0].program.feat412[0] += 1.0,
                        11 => changed.revealed[0].program.proj16[0] += 1.0,
                        12 => changed.revealed[0].program.program_sha256 = "d".repeat(64),
                        13 => changed.programs[0].proj16[0] = f64::INFINITY,
                        _ => changed_conditioning.mask[0] = f32::NAN,
                    }
                    let mut refused_rng = rng.clone();
                    assert!(
                        admission
                            .admit_recorded_round(
                                &context,
                                proposals.clone(),
                                width == 20,
                                &mut refused_rng,
                                &RecordedAdmissionContext {
                                    slate: &changed,
                                    conditioning: changed_conditioning
                                },
                            )
                            .is_err(),
                        "accepted different recorded graph inputs, case {bad}"
                    );
                    assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
                    recording_refusals += 1;
                }
            }
            // Alignment and identity failures must not advance the caller's RNG.
            for bad in [2, 3, 4] {
                mode.set(bad);
                let mut refused_rng = rng.clone();
                assert!(admission
                    .evaluate_round(&context, &proposals, width == 20, &mut refused_rng)
                    .is_err());
                assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
            }
            mode.set(0);
            for field in 0..6 {
                let mut wrong = plan.clone();
                match field {
                    0 => wrong.problem_digest = "c".repeat(64),
                    1 => wrong.bundle_digest = "d".repeat(64),
                    2 => wrong.matrix_sha256 = "e".repeat(64),
                    3 => wrong.binary_version.push('x'),
                    4 => wrong.objective = "mae".into(),
                    _ => wrong.seed += 1,
                }
                let wrong_context = RoundContext {
                    plan: &wrong,
                    round: context.round,
                    measured: context
                        .measured
                        .iter()
                        .map(|m| RoundMeasurement {
                            algorithm_id: m.algorithm_id,
                            dsl: m.dsl,
                            surrogate_fitness: m.surrogate_fitness,
                        })
                        .collect(),
                };
                let mut refused_rng = rng.clone();
                assert!(admission
                    .evaluate_round(&wrong_context, &proposals, width == 20, &mut refused_rng)
                    .is_err());
                assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
            }
        }
    }
    if causal_measurements_only {
        use std::io::Write;
        assert_eq!(serving_slates, 50, "M5 requires every serving 4+8 fixture");
        assert_eq!(stats.values, 1600, "M5 requires both complete slate widths");
        let passed = serving_landmark_changes > 0
            && serving_fitness_changes > 0
            && serving_disconnected_changes > 0;
        let report = json!({
            "schema":"jope.m5-causal/v1","consumer":"M5-causal","device":"cpu",
            "passed":passed,"values":stats.values,"failed_values":stats.failures,
            "details":{"contexts":50,"slates":100,"admitted_order_mismatches":mismatches,
                "quality_gate_granted":false,"Q1_ready":false,
                "serving_4_plus_8":{"slates":serving_slates,
                    "landmark_only_changed_admission":serving_landmark_changes,
                    "fitness_changed_admission":serving_fitness_changes,
                    "disconnected_heuristic_mismatches":serving_disconnected_changes,
                    "landmark_slots":[0,1,7,9],"interventions":[-6.0,0.0,6.0],
                    "masks_and_dope_only_slots_preserved":true}}
        });
        let root = PathBuf::from(
            env::var_os("JOPE_PARITY_REPORT_DIR").expect("M5 report directory required"),
        );
        fs::create_dir_all(&root).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("M5-causal.json"))
            .unwrap()
            .write_all(&serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
        println!("{report}");
        return;
    }
    assert!(decisive > 0, "no decisive reference admission comparisons");
    assert!(landmark_changes > 0, "disconnected landmark control");
    assert!(fitness_changes > 0, "disconnected measured-fitness control");
    assert!(
        disconnected_changes > 0,
        "heuristic disconnected control must fail"
    );
    #[cfg(feature = "research")]
    let recorded_summary = json!({"executed":true,"slates":recorded_slates,"input_mismatch_refusals":recording_refusals,"same_costs_indices_and_RNG":true});
    #[cfg(not(feature = "research"))]
    let recorded_summary = json!({"executed":false,"reason":"research feature required"});
    stats.failures += mismatches;
    stats.finish("RNK-10-composition", json!({
        "contexts":50,"slates":100,"recorded_admission":recorded_summary,"decisive_admission_slates":decisive,"tie_band_slates":ties,
        "admitted_order_mismatches":mismatches,"landmark_perturbation_changed_admission":landmark_changes,
        "fitness_perturbation_changed_admission":fitness_changes,"disconnected_heuristic_mismatches":disconnected_changes,
        "serving_4_plus_8":{"slates":serving_slates,"landmark_only_changed_admission":serving_landmark_changes,
            "fitness_changed_admission":serving_fitness_changes,"disconnected_heuristic_mismatches":serving_disconnected_changes,
            "landmark_slots":[0,1,7,9],"masks_and_dope_only_slots_preserved":true},
        "max_proposal_KL":max_kl,"quality_gate_granted":false,"Q1_ready":false,
        "reveal_rule":"explicit StorageOrder matching captured first four; not an OD13 grant",
        "identity_scope":"packet and weights hash-pinned; dummy Plan used only to exercise binding refusals"
    }));
}
