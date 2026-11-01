//! Native projection/token replay for any separately hash-pinned RES-07 packet.
//! Compile beside the maintained source using its Cargo-built dependency rlibs.
#![allow(dead_code)]
#[path = "../../../crates/jope-service/src/ranker/proj16.rs"]
mod projection;
#[path = "../../../crates/jope-service/src/ranker/stage1b.rs"]
mod stage1b;

use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn payload(root: &Path, index: &Value, name: &str) -> Vec<u8> {
    assert!(!name.starts_with('/') && !name.split('/').any(|p| p == ".."));
    let path = root.join(name);
    assert!(fs::symlink_metadata(&path).unwrap().is_file());
    let bytes = fs::read(path).unwrap();
    assert_eq!(
        hash(&bytes),
        index["files"][name]["sha256"].as_str().unwrap()
    );
    assert_eq!(
        bytes.len() as u64,
        index["files"][name]["bytes"].as_u64().unwrap()
    );
    bytes
}
fn f32s(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Vec<f32> {
    let t = tensors.tensor(name).unwrap();
    assert_eq!(t.dtype(), Dtype::F32);
    assert_eq!(t.shape(), shape);
    t.data()
        .chunks_exact(4)
        .map(|v| {
            let f = f32::from_le_bytes(v.try_into().unwrap());
            assert!(f.is_finite());
            f
        })
        .collect()
}
fn slot(t: &SafeTensors<'_>, name: &str, width: usize) -> stage1b::SvdSlot {
    stage1b::SvdSlot {
        mean: f32s(t, &format!("{name}.mean"), &[width]),
        std: f32s(t, &format!("{name}.std"), &[width]),
        active: f32s(t, &format!("{name}.active"), &[width]),
        projection: f32s(t, &format!("{name}.projection"), &[width, 32]),
    }
}
struct Captured(Vec<f32>);
impl stage1b::TokenProjector for Captured {
    fn project(&self, _: &[f32]) -> Vec<f32> {
        self.0.clone()
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "packet root and independently trusted index SHA required"
    );
    let root = Path::new(&args[1]);
    let raw = fs::read(root.join("index.json")).unwrap();
    assert_eq!(hash(&raw), args[2]);
    let index: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(index["schema"], "jope.golden-fixtures/v1");
    assert_eq!(index["authority"], "rulings_t1900.md R5");
    assert_eq!(index["excluded_unlogged"], 2688);
    let count = index["reconstructed_programs"].as_u64().unwrap() as usize;
    let bytes = payload(root, &index, "programs.safetensors");
    let tensor = SafeTensors::deserialize(&bytes).unwrap();
    let features = f32s(&tensor, "feat412", &[count, 412]);
    let oracle = tensor.tensor("proj16").unwrap();
    assert_eq!(oracle.dtype(), Dtype::F64);
    assert_eq!(oracle.shape(), [count, 16]);
    let mut max_projection = 0.0_f64;
    for row in 0..count {
        let input: [f64; 412] = std::array::from_fn(|i| f64::from(features[row * 412 + i]));
        for (i, actual) in projection::proj16(&input).into_iter().enumerate() {
            let offset = (row * 16 + i) * 8;
            let expected =
                f64::from_le_bytes(oracle.data()[offset..offset + 8].try_into().unwrap());
            assert!(expected.is_finite());
            max_projection = max_projection.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() <= 1e-12,
                "projection row {row} column {i}"
            );
        }
    }
    let bytes = payload(root, &index, "rnk07/stage1b-parameters.safetensors");
    let t = SafeTensors::deserialize(&bytes).unwrap();
    let parameters = stage1b::Stage1bParameters {
        lm80_mean: f32s(&t, "lm80.mean", &[80]).try_into().unwrap(),
        lm80_std: f32s(&t, "lm80.std", &[80]).try_into().unwrap(),
        lm48_mean: f32s(&t, "lm48.mean", &[48]).try_into().unwrap(),
        lm48_std: f32s(&t, "lm48.std", &[48]).try_into().unwrap(),
        slot0: slot(&t, "slot0", 184),
        slot1: slot(&t, "slot1", 48),
        slot2: slot(&t, "slot2", 629),
        slot4: slot(&t, "slot4", 629),
        slot5: slot(&t, "slot5", 629),
        slot7: slot(&t, "slot7", 184),
        slot9: slot(&t, "slot9", 64),
    };
    parameters.validate().unwrap();
    let mut contexts = 0;
    let mut max_token = 0.0_f32;
    for name in index["files"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|n| n.starts_with("contexts/"))
    {
        let bytes = payload(root, &index, name);
        let t = SafeTensors::deserialize(&bytes).unwrap();
        let lm = f32s(&t, "lm48", &[1, 48]);
        let input = f32s(&t, "v256", &[1, 256]);
        let dope = f32s(&t, "dope629", &[1, 629]);
        let tokens = f32s(&t, "tokens", &[1, 10, 32]);
        let mask = f32s(&t, "mask", &[1, 10]);
        let actual = stage1b::project(&Captured(lm), &input, &dope, &parameters).unwrap();
        for (a, b) in actual.tokens.iter().flatten().zip(tokens) {
            max_token = max_token.max((a - b).abs());
            assert!(
                (a - b).abs() <= 1e-5 + 1e-4 * b.abs(),
                "Stage1B {name}: {a} != {b}"
            );
        }
        assert_eq!(actual.mask.to_vec(), mask);
        contexts += 1;
    }
    assert_eq!(contexts, 50);
    println!(
        "{}",
        serde_json::json!({"source":"native proj16.rs and stage1b.rs",
        "index_sha256":args[2],"programs":count,"projection_values":count*16,
        "projection_max_absolute_error":max_projection,"contexts":contexts,"token_values":contexts*320,
        "token_max_absolute_error":max_token,"mask_mismatches":0,"failed_values":0})
    );
}
