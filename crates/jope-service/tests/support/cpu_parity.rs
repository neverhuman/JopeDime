//! Strict external, hash-pinned real CPU fixtures, shared by two unit-test modules.
#![allow(dead_code)] // Each consumer uses a different subset of this test-only helper.
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Component, Path, PathBuf},
};

#[path = "cpu_parity/round_admission.rs"]
mod round_admission;
pub(super) use round_admission::round_admission_fixture;

use crate::discover::r#loop::RoundContext;
use crate::ranker::admission::{RoundInputProvider, RoundInputs, RoundProgram};
use crate::ranker::stage1b::TokenProjection;
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
    fn observe(&self, _: &RoundContext<'_>, proposals: &[String]) -> anyhow::Result<RoundInputs> {
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
                // Add a zero-mean channel pattern so token LayerNorm does
                // not erase the landmark-only intervention.
                intervene_on_landmark_tokens(
                    &mut result.conditioning.tokens,
                    [-6.0, 0.0, 6.0][usize::from(self.mode.get() - 5)],
                );
            }
            _ => {}
        }
        Ok(result)
    }
}

const BUNDLE: &str = "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c";
const JOPE_SHA: &str = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";
const R882_JOPE_SHA: &str = "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c";
const LANDMARK_SLOTS: [usize; 4] = [0, 1, 7, 9];

fn intervene_on_landmark_tokens(tokens: &mut [[f32; 32]; 10], delta: f32) {
    for slot in LANDMARK_SLOTS {
        for (channel, value) in tokens[slot].iter_mut().enumerate() {
            let direction = if channel % 2 == 0 { -1.0 } else { 1.0 };
            *value += delta * direction;
        }
    }
}

fn layer_norm_channels(values: &[f32; 32]) -> [f32; 32] {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let centered = values.map(|value| value - mean);
    let variance = centered.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let denom = (variance + 1e-5).sqrt();
    centered.map(|value| value / denom)
}

fn output_oracle_matches(
    reference_bundle: &str,
    reference_jope_sha: &str,
    actual_bundle: &str,
    actual_jope_sha: &str,
) -> bool {
    reference_bundle == actual_bundle && reference_jope_sha == actual_jope_sha
}

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
        let signed_r882 = env::var_os("JOPE_RNK10_SIGNED_BUNDLE_DIR");
        let weights_root = signed_r882
            .clone()
            .or_else(|| env::var_os("JOPE_PARITY_BUNDLE_DIR"))
            .expect("real frozen or signed release bundle required");
        let weights = PathBuf::from(weights_root).join("jope.safetensors");
        if signed_r882.is_none() {
            hashed(&weights, JOPE_SHA);
        }
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

#[test]
fn output_oracle_requires_matching_bundle_and_jope_weights() {
    assert!(output_oracle_matches(BUNDLE, JOPE_SHA, BUNDLE, JOPE_SHA));
    assert!(!output_oracle_matches(
        BUNDLE,
        JOPE_SHA,
        super::super::cuda::release::R882,
        R882_JOPE_SHA,
    ));
    assert!(!output_oracle_matches(
        BUNDLE,
        JOPE_SHA,
        BUNDLE,
        R882_JOPE_SHA
    ));
}

#[test]
fn landmark_intervention_survives_token_layer_norm() {
    let constant_minus = layer_norm_channels(&[-6.0; 32]);
    let constant_zero = layer_norm_channels(&[0.0; 32]);
    let constant_plus = layer_norm_channels(&[6.0; 32]);
    assert_eq!(constant_minus, constant_zero);
    assert_eq!(constant_zero, constant_plus);

    let original = std::array::from_fn(|channel| channel as f32 / 16.0 - 1.0);
    let baseline = layer_norm_channels(&original);
    for delta in [-6.0, 6.0] {
        let mut tokens = [[0.0; 32]; 10];
        for slot in LANDMARK_SLOTS {
            tokens[slot] = original;
        }
        intervene_on_landmark_tokens(&mut tokens, delta);
        for slot in LANDMARK_SLOTS {
            assert_ne!(
                layer_norm_channels(&tokens[slot]),
                baseline,
                "landmark delta {delta} must survive token normalization"
            );
        }
    }
}
