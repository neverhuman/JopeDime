//! Explicit artifact qualification; defaults never report absent candidate coverage as a pass.
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};
use starforge::dope_encoder::{DopeCheckpointIdentity, DopeEncoder};

const REFERENCE: &str = include_str!("../fixtures/dope_dropout_candidate_reference.json");
const INPUT: &str = include_str!("../fixtures/dope_cuda_reference.json");

fn reference(device: &Device) -> anyhow::Result<DopeEncoder> {
    let expected: serde_json::Value = serde_json::from_str(REFERENCE)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(INPUT)),
        expected["input_sha256"]
    );
    let path = std::env::var_os("DOPE_CANDIDATE_WEIGHTS")
        .expect("explicit candidate qualification requires DOPE_CANDIDATE_WEIGHTS");
    let bytes = std::fs::read(path)?;
    let identity = DopeCheckpointIdentity::unqualified_candidate(
        expected["weights_sha256"].as_str().unwrap().into(),
        expected["training_receipt_sha256"].as_str().unwrap().into(),
    )?;
    let encoder = DopeEncoder::from_pinned_bytes(&bytes, device, identity)?;
    let input: serde_json::Value = serde_json::from_str(INPUT)?;
    let vec: Vec<Vec<f32>> = serde_json::from_value(input["vec"].clone())?;
    let cols: Vec<Vec<Vec<f32>>> = serde_json::from_value(input["columns"].clone())?;
    let expected: Vec<Vec<f32>> = serde_json::from_value(expected["expected629"].clone())?;
    assert_eq!(expected.len(), 4, "reference must cover every input row");
    assert!(expected.iter().all(|row| row.len() == 629));
    let descriptor = Tensor::from_vec(vec.into_iter().flatten().collect(), (4, 256), device)?;
    let columns = Tensor::from_vec(
        cols.into_iter().flatten().flatten().collect(),
        (4, 64, 17),
        device,
    )?;
    let model = dope_core::mlc_v3::MlcV3Model::from_safetensors_bytes_on_device(&bytes, device)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let full = pool.install(|| model.encode_features_629_tensor(&descriptor, &columns, "high"))?;
    let projected = encoder.encode_tensors(&descriptor, &columns)?;
    assert_eq!(full.dims(), [4, 629]);
    assert_eq!(projected.dims(), [4, 512]);
    assert!(full.device().same_device(device) && projected.device().same_device(device));
    let full = full.to_vec2::<f32>()?;
    let projected = projected.to_vec2::<f32>()?;
    let mut error = 0f32;
    for ((actual, projected), expected) in full.iter().zip(projected).zip(expected) {
        assert_eq!(actual[..512], projected);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            error = error.max((actual - expected).abs());
        }
    }
    eprintln!(
        "candidate full629/first512 vs Python CPU: device={:?} max_abs={error:.9e}",
        device.location()
    );
    assert!(error < 3e-5, "trained candidate numerical parity: {error}");
    Ok(encoder)
}

#[test]
#[ignore = "requires the pinned trained-candidate artifact and governed JOPE bundle"]
fn candidate_cpu_parity_and_three_task_consumers() -> anyhow::Result<()> {
    let encoder = reference(&Device::Cpu)?;
    let root = super::models_root::try_bundle_root().map_err(anyhow::Error::msg)?;
    super::run_consumers(encoder, &root)
}

#[cfg(feature = "dope-cuda")]
#[test]
#[ignore = "requires the pinned trained-candidate artifact, governed JOPE bundle and an admitted GPU window"]
fn candidate_cuda_parity() -> anyhow::Result<()> {
    let encoder = reference(&DopeEncoder::cuda(0)?)?;
    let root = super::models_root::try_bundle_root().map_err(anyhow::Error::msg)?;
    super::run_consumers(encoder, &root)
}
