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
