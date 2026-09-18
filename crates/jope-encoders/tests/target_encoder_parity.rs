//! Mandatory component parity for the fused label-to-512 target encoder.
//!
//! This starts from labels and checks every exposed Rust intermediate. It does
//! not prove end-to-end Prime or Lime proposal/prediction parity.

use anyhow::{Context, Result};
use candle_core::Device;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use starforge::bundle::LoadedModelBundle;
use starforge::target_encoder::FusedTargetEncoder;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const TARGET_SHA256: &str = "be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f";
const TARGET_BYTES: u64 = 78_615_176;
const INVENTORY_SHA256: &str = "89a25ebf1fbc49fc30354b63cffd94822807a6a98ab6723de91a329711c80baf";
const PARITY_FIXTURE_SHA256: &str =
    "e371f3e34e0c493c39daacae1c5d22ee8932caf1503c68979f826192acce718d";

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    oracle: OracleIdentity,
    artifact_sha256: String,
    feature_names_sha256: String,
    tensor_inventory_sha256: String,
    cases: Vec<ParityCase>,
    max_abs_tolerance: f32,
}

#[derive(Debug, Deserialize)]
struct OracleIdentity {
    implementation_sha256: String,
    source_revision: String,
    source_uri: String,
}

#[derive(Debug, Deserialize)]
struct ParityCase {
    input: FixtureInput,
    raw_signature: Vec<f32>,
    neural_best: Vec<f32>,
    neural_final: Vec<f32>,
    classical_component: Vec<f32>,
    expected_vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct FixtureInput {
    dataset_hash: String,
    csv_sha256: String,
    task: String,
    original_length: usize,
    effective_count: usize,
    replacement_fraction: f32,
    seed: u64,
    values: Vec<f32>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parity_metrics(left: &[f32], right: &[f32]) -> (f32, f32, usize) {
    assert_eq!(left.len(), right.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut exact = 0usize;
    for (&left, &right) in left.iter().zip(right) {
        assert!(
            left.is_finite() && right.is_finite(),
            "parity values must be finite"
        );
        let delta = (left - right).abs();
        max_abs = max_abs.max(delta);
        // A unit floor gives a stable mixed absolute/relative metric near zero.
        max_rel = max_rel.max(delta / left.abs().max(right.abs()).max(1.0));
        exact += usize::from(left.to_bits() == right.to_bits());
    }
    (max_abs, max_rel, exact)
}

fn assert_parity(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    let (max_abs, max_rel, exact) = parity_metrics(actual, expected);
    eprintln!(
        "target Rust-vs-Python {label}: max_abs={max_abs:.9e} max_rel={max_rel:.9e} exact_f32={exact}/{}",
        actual.len()
    );
    assert!(
        max_abs <= tolerance && max_rel <= tolerance,
        "{label} exceeded tolerance: max_abs={max_abs:.9e}, max_rel={max_rel:.9e}"
    );
}

fn assert_exact_f32(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "{label} must match the frozen Python oracle bit-for-bit"
    );
}

#[test]
fn real_binary_and_regression_fused_target_parity() -> Result<()> {
    let root = repo_root();
    let bundle = LoadedModelBundle::load(&root)?;
    let encoder = &bundle.manifest.encoders["target"];
    assert_eq!(encoder.sha256, TARGET_SHA256);
    assert_eq!(
        encoder.source_revision,
        "sha256:5715a3fd860ad503bfd2ff737cedc5109ee3ffe9653cc7873072084753164cd8"
    );
    let artifact = bundle
        .manifest
        .files
        .iter()
        .find(|file| file.path == encoder.artifact)
        .context("target artifact is not in the bundle manifest")?;
    assert_eq!(artifact.bytes, TARGET_BYTES);
    assert_eq!(artifact.sha256, TARGET_SHA256);
    assert_eq!(artifact.tensor_count, 171);
    assert_eq!(
        artifact.tensor_inventory_sha256.as_deref(),
        Some(INVENTORY_SHA256)
    );

    let bytes = bundle.read_verified(&encoder.artifact)?;
    assert_eq!(bytes.len() as u64, TARGET_BYTES);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), TARGET_SHA256);
    let tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?;
    assert_eq!(tensors.len(), 171);
    assert_eq!(
        tensors
            .keys()
            .filter(|name| name.starts_with("neural_best."))
            .count(),
        80
    );
    assert_eq!(
        tensors
            .keys()
            .filter(|name| name.starts_with("neural_final."))
            .count(),
        80
    );

    let metadata: serde_json::Value = serde_json::from_slice(
        &bundle.read_verified("artifacts/model-bundle/target_encoder_fused_v1.meta.json")?,
    )?;
    assert_eq!(metadata["schema"], "jain.target-encoder/v1");
    assert_eq!(metadata["artifact_sha256"], TARGET_SHA256);
    assert_eq!(metadata["tensor_inventory_sha256"], INVENTORY_SHA256);
    assert_eq!(metadata["seed"], 17);
    assert_eq!(metadata["vector_dim"], 512);
    assert_eq!(
        metadata["input_preprocessing"]["binary"],
        "threshold_ge_0.5"
    );
    assert_eq!(
        metadata["input_preprocessing"]["regression"],
        "clip_closed_interval_0_1"
    );
    assert_eq!(metadata["block_layout"].as_array().map(Vec::len), Some(3));
    assert_eq!(metadata["license"]["schema"], "jain.model-license/v1");
    assert_eq!(
        metadata["license"]["policy_eligibility"]["open_only"],
        false
    );
    let fused_encoder = FusedTargetEncoder::load(&bundle)?;

    let fixture_bytes = std::fs::read(
        root.join("crates/starforge/tests/fixtures/target_encoder_fused_v1_parity.json"),
    )?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&fixture_bytes)),
        PARITY_FIXTURE_SHA256
    );
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes)?;
    assert_eq!(fixture.schema, "jain.target-encoder-parity/v1");
    assert_eq!(fixture.oracle.implementation_sha256, encoder.recipe_sha256);
    assert_eq!(fixture.oracle.source_revision, encoder.source_revision);
    assert_eq!(fixture.oracle.source_uri, encoder.source_uri);
    assert_eq!(fixture.artifact_sha256, TARGET_SHA256);
    assert_eq!(
        fixture.feature_names_sha256,
        "ba969d2c5c9a50691187aa9191c8b99a0259f0fddecd9a612cd1406f9920d036"
    );
    assert_eq!(fixture.tensor_inventory_sha256, INVENTORY_SHA256);
    assert_eq!(fixture.max_abs_tolerance, 1e-5);
    assert_eq!(fixture.cases.len(), 2);

    let tasks = fixture
        .cases
        .iter()
        .map(|case| case.input.task.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(tasks, BTreeSet::from(["binary", "regression"]));

    for case in fixture.cases {
        assert_eq!(case.raw_signature.len(), 2344);
        assert_eq!(case.neural_best.len(), 64);
        assert_eq!(case.neural_final.len(), 64);
        assert_eq!(case.classical_component.len(), 384);
        assert_eq!(case.expected_vector.len(), 512);
        assert_eq!(case.input.seed, 17);
        assert_eq!(case.input.original_length, case.input.values.len());
        assert_eq!(case.input.effective_count, 32);
        assert_eq!(case.input.replacement_fraction, 0.0);
        assert!(case.input.values.len() >= 40);
        match case.input.task.as_str() {
            "binary" => {
                assert_eq!(case.input.dataset_hash, "ada8511705948a07");
                assert_eq!(
                    case.input.csv_sha256,
                    "540a80a0b0bd6ebe13854309d4c87887859a8d00357a74c371eb35ce1f664e22"
                );
                assert!(case.input.values.iter().any(|value| *value < 0.5));
                assert!(case.input.values.iter().any(|value| *value >= 0.5));
            }
            "regression" => {
                assert_eq!(case.input.dataset_hash, "00709d315efacf0c");
                assert_eq!(
                    case.input.csv_sha256,
                    "654c5754fd41fa38447a388efaf1b08f5d30b89e2862972fde94d531a53bd15d"
                );
                assert!(case
                    .input
                    .values
                    .iter()
                    .any(|value| 0.0 < *value && *value < 1.0));
            }
            task => panic!("unexpected task {task}"),
        }

        let actual = fused_encoder.encode(&case.input.values, case.input.task.parse()?)?;
        assert_parity(
            &format!("{} raw signature", case.input.task),
            &actual.raw_signature,
            &case.raw_signature,
            fixture.max_abs_tolerance,
        );
        assert_exact_f32(
            &format!("{} raw signature", case.input.task),
            &actual.raw_signature,
            &case.raw_signature,
        );
        assert_parity(
            &format!("{} best neural component", case.input.task),
            &actual.neural_best,
            &case.neural_best,
            fixture.max_abs_tolerance,
        );
        assert_parity(
            &format!("{} final neural component", case.input.task),
            &actual.neural_final,
            &case.neural_final,
            fixture.max_abs_tolerance,
        );
        assert_parity(
            &format!("{} classical component", case.input.task),
            &actual.classical_component,
            &case.classical_component,
            fixture.max_abs_tolerance,
        );
        assert_parity(
            &format!("{} fused vector", case.input.task),
            &actual.vector,
            &case.expected_vector,
            fixture.max_abs_tolerance,
        );
    }
    Ok(())
}
