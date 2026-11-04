//! Mandatory offline differential proof for the shared Prime/Lime
//! preprocessing, foundation, and target-encoding components.
//!
//! The committed fixture was generated once by the exact, hash-bound historical
//! Python reference stack. This test binds both the fixture bytes and the
//! deterministically generated request, so sealed CI needs no Python runtime,
//! mutable sibling checkout, or controller-local cache.

use anyhow::{bail, Context, Result};

#[path = "support/models_root.rs"]
mod models_root;
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use starforge::bundle::LoadedModelBundle;
use starforge::foundation::{
    jope_external_selected_indices, prepare_jope_foundation_rows, preprocess_jope_external_train,
    FoundationEncodeOptions, JopeExternalPreprocessedV1, StarlightFoundationEncoder,
    JOPE_FOUNDATION_CONTEXT_ROWS, JOPE_FOUNDATION_QUERY_ROWS, JOPE_FOUNDATION_SEED,
};
use starforge::target_encoder::{FusedTargetEncoder, FusedTargetEncoding, TargetTask};
use std::collections::BTreeMap;

const TARGET_BUNDLE: &str = "target-maker/fused-monster-smoke-real-20260705/final.bundle";
const MAX_ABS: f64 = 1e-5;
const MAX_REL: f64 = 1e-5;
const FOUNDATION_HIDDEN_MAX_ABS: f64 = 1e-4;
const FOUNDATION_HIDDEN_MAX_REL: f64 = 1e-4;
const FOUNDATION_POOLED_MAX_ABS: f64 = 1e-4;
const FOUNDATION_POOLED_MAX_REL: f64 = 1e-4;
const FROZEN_ORACLE_JSON: &[u8] = include_bytes!("fixtures/novel_input_oracle.v1.json");
const FROZEN_ORACLE_LOCK_JSON: &[u8] = include_bytes!("fixtures/novel_input_oracle.v1.lock.json");
const FROZEN_ORACLE_SHA256: &str =
    "67f3a66d619639aa793e07871bf2a4ab15ac55cb7b8ad7be3914546dd148515e";
const FROZEN_ORACLE_LOCK_SHA256: &str =
    "b52d1ab50d1637885cc02e62c8ca31afb887992b30ea979d4d4044ed5d29bbf4";
const FROZEN_ORACLE_REQUEST_SHA256: &str =
    "3edf7f695068e0ad82454e47a7e50fe8eb5c2be871e244b046b21943bafc7211";
const FROZEN_ORACLE_GENERATOR_SHA256: &str =
    "c507488b1e1164b38d6f23e9439e6134fb61f6e43ba4f718ba3ec22e61c7fb0c";
const CHOICE_REGRESSION_SHA256: &[(usize, &str)] = &[
    (
        1001,
        "28df0d52bcc8110ef580eec29d422df6bf019ddbb91d99e544724aa1c35409d7",
    ),
    (
        1500,
        "85112e69647f4ecdbfba1c9f131c6c16dbc48ff25f58bcec2e807416869db2a1",
    ),
    (
        10_000,
        "e9aff16073b1edd90ee8787bc52a17395e2a19ffee937752358570fbdbfbf125",
    ),
    (
        50_000,
        "050f0780fa9ecf10faca5d16609331434c1a91bc2e3d33522044732679f725e4",
    ),
];

#[derive(Serialize)]
struct OracleRequest {
    target_bundle: &'static str,
    choice_populations: Vec<usize>,
    external_cases: Vec<ExternalInput>,
    target_cases: Vec<TargetInput>,
}

#[derive(Serialize)]
struct ExternalInput {
    name: String,
    rows_bits: Vec<Vec<u64>>,
}

#[derive(Serialize)]
struct TargetInput {
    name: String,
    task: String,
    values_bits: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleOutput {
    versions: Versions,
    modules: BTreeMap<String, String>,
    choices: BTreeMap<String, Vec<usize>>,
    external_cases: Vec<ExternalOracle>,
    target_cases: Vec<TargetOracle>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Versions {
    python: String,
    numpy: String,
    torch: String,
    sklearn: String,
    joblib: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalOracle {
    name: String,
    scaler_low: Vec<f64>,
    scaler_span: Vec<f64>,
    selected_indices: Vec<usize>,
    encoding_rows: Vec<Vec<f64>>,
    all_encoding_rows: Vec<Vec<f64>>,
    feature_low: Vec<f64>,
    feature_span: Vec<f64>,
    context_indices: Vec<usize>,
    query_indices: Vec<usize>,
    context_x: Vec<Vec<f32>>,
    context_y: Vec<f32>,
    query_x: Vec<Vec<f32>>,
    query_hidden: Vec<Vec<f32>>,
    mean: Vec<f32>,
    std: Vec<f32>,
    p10: Vec<f32>,
    p90: Vec<f32>,
    vector: Vec<f32>,
    target: TargetOracle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetOracle {
    name: String,
    task: String,
    error: Option<String>,
    #[serde(default)]
    preprocessed_values: Vec<f32>,
    #[serde(default)]
    raw_signature: Vec<f32>,
    #[serde(default)]
    neural_best: Vec<f32>,
    #[serde(default)]
    neural_final: Vec<f32>,
    #[serde(default)]
    classical_component: Vec<f32>,
    #[serde(default)]
    vector: Vec<f32>,
}

fn generated_rows(count: usize) -> Vec<Vec<f64>> {
    let mut rows = (0..count)
        .map(|index| {
            vec![
                7.5,
                (index % 5) as f64 - 2.0,
                ((index * 37) % 101) as f64 / 7.0 - 5.0,
                index as f64 * 0.03125 - 10.0,
                ((index * 19) % 67) as f64 * 0.125 - 3.0,
                (index % 29) as f64 * 0.25 - 3.5,
            ]
        })
        .collect::<Vec<_>>();
    rows[0][2] = f64::NAN;
    rows[1][5] = f64::INFINITY;
    rows[2][3] = f64::NEG_INFINITY;
    rows[3][2] = -100.0;
    rows[4][2] = 200.0;
    rows
}

fn binary_edge_values() -> Vec<f32> {
    let mut values = (0..47)
        .map(|index| match index % 6 {
            0 => 0.0,
            1 => 1.0,
            2 => 0.49,
            3 => 0.5,
            4 => -3.0,
            _ => 4.0,
        })
        .collect::<Vec<_>>();
    values[0] = f32::NAN;
    values[1] = f32::INFINITY;
    values[2] = f32::NEG_INFINITY;
    values
}

fn frozen_oracle(request: &OracleRequest) -> Result<OracleOutput> {
    let actual_fixture = format!("{:x}", Sha256::digest(FROZEN_ORACLE_JSON));
    if actual_fixture != FROZEN_ORACLE_SHA256 {
        bail!(
            "frozen differential fixture identity changed: {actual_fixture}, expected {FROZEN_ORACLE_SHA256}"
        );
    }
    if FROZEN_ORACLE_JSON
        .windows(b"/home/".len())
        .any(|window| window == b"/home/")
    {
        bail!("frozen differential fixture contains a host-local path");
    }
    let actual_lock = format!("{:x}", Sha256::digest(FROZEN_ORACLE_LOCK_JSON));
    if actual_lock != FROZEN_ORACLE_LOCK_SHA256 {
        bail!(
            "frozen differential provenance lock changed: {actual_lock}, expected {FROZEN_ORACLE_LOCK_SHA256}"
        );
    }
    if FROZEN_ORACLE_LOCK_JSON
        .windows(b"/home/".len())
        .any(|window| window == b"/home/")
    {
        bail!("frozen differential provenance lock contains a host-local path");
    }
    let request_json = serde_json::to_vec(request)?;
    let actual_request = format!("{:x}", Sha256::digest(&request_json));
    if actual_request != FROZEN_ORACLE_REQUEST_SHA256 {
        bail!(
            "frozen differential request identity changed: {actual_request}, expected {FROZEN_ORACLE_REQUEST_SHA256}"
        );
    }
    let oracle: OracleOutput =
        serde_json::from_slice(FROZEN_ORACLE_JSON).context("decode frozen differential fixture")?;
    let lock: serde_json::Value = serde_json::from_slice(FROZEN_ORACLE_LOCK_JSON)
        .context("decode frozen differential provenance lock")?;
    let expected_lock_keys = [
        "fixture",
        "generator",
        "references",
        "request_sha256",
        "schema",
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    let actual_lock_keys = lock
        .as_object()
        .context("frozen differential provenance lock must be an object")?
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    if actual_lock_keys != expected_lock_keys
        || lock["schema"] != "jain.starforge-novel-input-oracle-lock/v1"
        || lock["fixture"]["sha256"] != FROZEN_ORACLE_SHA256
        || lock["request_sha256"] != FROZEN_ORACLE_REQUEST_SHA256
        || lock["generator"]["corrected_sha256"] != FROZEN_ORACLE_GENERATOR_SHA256
        || lock["references"].as_object().map(|refs| refs.len()) != Some(21)
    {
        bail!("frozen differential provenance lock is inconsistent");
    }
    Ok(oracle)
}

fn flat_f64(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn flat_f32(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32], abs: f64, rel: f64) -> Result<()> {
    let actual = actual
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let expected = expected
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    compare_f64(label, &actual, &expected, abs, rel)
}

fn compare_f64(label: &str, actual: &[f64], expected: &[f64], abs: f64, rel: f64) -> Result<()> {
    if actual.len() != expected.len() {
        bail!(
            "{label} length mismatch: Rust={}, Python={}",
            actual.len(),
            expected.len()
        );
    }
    let mut first = None;
    let mut max_abs = (0usize, 0.0f64);
    let mut max_rel = (0usize, 0.0f64);
    for (index, (&rust, &python)) in actual.iter().zip(expected).enumerate() {
        if !rust.is_finite() || !python.is_finite() {
            bail!("{label} contains non-finite value at {index}: Rust={rust}, Python={python}");
        }
        let delta = (rust - python).abs();
        let relative = delta / rust.abs().max(python.abs()).max(1.0);
        if delta > max_abs.1 {
            max_abs = (index, delta);
        }
        if relative > max_rel.1 {
            max_rel = (index, relative);
        }
        if first.is_none() && delta > abs + rel * rust.abs().max(python.abs()) {
            first = Some((index, rust, python, delta, relative));
        }
    }
    eprintln!(
        "novel differential {label}: max_abs={:.9e}@{} max_rel={:.9e}@{}",
        max_abs.1, max_abs.0, max_rel.1, max_rel.0
    );
    if let Some((index, rust, python, delta, relative)) = first {
        bail!(
            "{label} first mismatch at {index}: Rust={rust:.9e}, Python={python:.9e}, abs={delta:.9e}, rel={relative:.9e}; max_abs={:.9e}@{}, max_rel={:.9e}@{}",
            max_abs.1,
            max_abs.0,
            max_rel.1,
            max_rel.0
        );
    }
    Ok(())
}

fn compare_target(name: &str, actual: &FusedTargetEncoding, expected: &TargetOracle) -> Result<()> {
    if let Some(error) = &expected.error {
        bail!("Python unexpectedly rejected target case {name}: {error}");
    }
    compare(
        &format!("{name}.preprocessed"),
        &actual.preprocessed_values,
        &expected.preprocessed_values,
        0.0,
        0.0,
    )?;
    compare(
        &format!("{name}.raw_signature"),
        &actual.raw_signature,
        &expected.raw_signature,
        MAX_ABS,
        MAX_REL,
    )?;
    compare(
        &format!("{name}.neural_best"),
        &actual.neural_best,
        &expected.neural_best,
        MAX_ABS,
        MAX_REL,
    )?;
    compare(
        &format!("{name}.neural_final"),
        &actual.neural_final,
        &expected.neural_final,
        MAX_ABS,
        MAX_REL,
    )?;
    compare(
        &format!("{name}.classical"),
        &actual.classical_component,
        &expected.classical_component,
        MAX_ABS,
        MAX_REL,
    )?;
    compare(
        &format!("{name}.vector"),
        &actual.vector,
        &expected.vector,
        MAX_ABS,
        MAX_REL,
    )
}

#[path = "novel_input_differential/runtime_differential.rs"]
mod runtime_differential;
