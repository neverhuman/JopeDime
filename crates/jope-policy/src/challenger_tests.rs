//! Native challenger binding (plan §4.5, DEMO-3) on the tiny CPU policy: the identity pins, the
//! challenger receipt parse, bound loading and tampering, refusal by every accepted-only path,
//! the challenger proposal, and the producer transport of a challenger request. These run in
//! the lib test target, where `cfg(test)` enables the test helpers, so the hosted `rust-test`
//! lane runs them. Scratch lives under the workspace `target/tmp`, never `/tmp`.
use crate::config::{ChallengerIdentity, CheckpointBinding, PolicyConfig, EXPORT_RECEIPT_SCHEMA};
use crate::context::PolicyContext;
use crate::decode::{self, SamplingOptions};
use crate::grammar::{bind_vocabulary, BoundVocabulary};
use crate::model::{expected_inventory, PolicyModel};
use crate::producer;
use crate::ranking::{score_pool, Bindings, Options};
use crate::seeds::{
    self, population_bytes, validate_for_search, ProposalSeeds, SeedBinding, ORIGIN_NEURAL,
    ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM,
};
use crate::{
    context_sha256, sha256_hex, ACCEPTED_CONFIG_SHA256, ACCEPTED_MODEL_SHA256,
    ACCEPTED_PARAMETER_SHA256, ACCEPTED_STATE_SHA256, ACCEPTED_STEP, CHECKPOINT_VOCABULARY_SHA256,
    PRODUCT_VOCABULARY_SHA256, REJECTED_MODEL_SHA256, REJECTED_STATE_SHA256,
};
use candle_core::Device;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `policy.safetensors` sha256 of the accepted step-60000 export.
const ACCEPTED_WEIGHTS: &str = "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e";
const VERIFIED: [&str; 4] = [
    "model_sha256",
    "config_sha256",
    "autocast_cache_lifetime",
    "parameter_sha256",
];

fn digest(tag: &str) -> String {
    sha256_hex(tag.as_bytes())
}

/// `tests/jope_dime.rs` `tiny_config`.
fn tiny() -> PolicyConfig {
    let mut config = PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config
}

/// `tests/jope_dime.rs` `tiny_tensors`: deterministic tiny-config tensors; `pad_row_shift` is
/// added to the PAD row (row 0) of `token_emb.weight` only.
fn tensors(pad_row_shift: f32) -> HashMap<String, candle_core::Tensor> {
    let config = tiny();
    expected_inventory(&config)
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel = shape.iter().product::<usize>();
            let pad_row = if name == "token_emb.weight" {
                config.width
            } else {
                0
            };
            let values = (0..numel)
                .map(|j| {
                    let v = ((i * 131 + j) as f32 * 0.37).sin() * 0.2;
                    if j < pad_row {
                        v + pad_row_shift * (1.0 + j as f32 * 0.1)
                    } else {
                        v
                    }
                })
                .collect::<Vec<f32>>();
            let t = candle_core::Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
            (name, t)
        })
        .collect()
}

fn context() -> PolicyContext {
    PolicyContext::from_frozen_values((0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect())
        .unwrap()
}

fn vocabulary() -> BoundVocabulary {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256).unwrap()
}

fn run_binding(ctx: &[f32]) -> SeedBinding {
    SeedBinding::new(
        &digest("prepared.json"),
        0,
        &digest("cut-0/train.csv"),
        &context_sha256(ctx),
    )
    .unwrap()
}

/// A fresh per-test directory under the workspace `target/tmp`.
fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/tmp")
        .join(format!(
            "jope-policy-challenger-{}-{tag}",
            std::process::id()
        ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn to_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}

fn assert_refused<T>(result: anyhow::Result<T>, needle: &str) {
    match result {
        Ok(_) => panic!("expected a refusal containing {needle:?}"),
        Err(err) => assert!(format!("{err:#}").contains(needle), "{err:#}"),
    }
}

/// A receipt shaped like `tests/jope_dime.rs` `accepted_receipt`: the accepted identity around
/// labelled tensor hashes (a receipt only; no file).
fn accepted_receipt() -> Value {
    let config = PolicyConfig::regression(401);
    let tensors = expected_inventory(&config)
        .into_iter()
        .map(|(name, shape)| {
            let numel = shape.iter().product::<usize>();
            json!({"name": name, "dtype": "F32", "shape": shape, "numel": numel,
                   "bytes": numel * 4, "sha256": digest(&name)})
        })
        .collect::<Vec<_>>();
    json!({
        "schema": EXPORT_RECEIPT_SCHEMA,
        "checkpoint": {
            "accepted_identity": true,
            "carried_bindings": {
                "corpus_sha256": digest("corpus"),
                "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
            },
            "config": config,
            "config_sha256": ACCEPTED_CONFIG_SHA256,
            "expected_state_sha256": ACCEPTED_STATE_SHA256,
            "model_sha256": ACCEPTED_MODEL_SHA256,
            "parameter_sha256": ACCEPTED_PARAMETER_SHA256,
            "parameter_sha256_recomputed": ACCEPTED_PARAMETER_SHA256,
            "state_sha256": ACCEPTED_STATE_SHA256,
            "step": ACCEPTED_STEP,
            "verified_bindings": VERIFIED,
        },
        "policy": {
            "output": {"sha256": ACCEPTED_WEIGHTS},
            "tensor_count": tensors.len(),
            "tensors": tensors,
        },
    })
}

/// Write tiny challenger weights to `dir/policy.safetensors` and return them with a receipt
/// written for a non-accepted checkpoint (state `digest(state)`, step 61,600) and the identity
/// pinned from the same independent values: the file's sha256 and the loaded parameter hash.
fn challenger_export(dir: &Path, shift: f32, state: &str) -> (PathBuf, Value, ChallengerIdentity) {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("policy.safetensors");
    candle_core::safetensors::save(&tensors(shift), &path).unwrap();
    let loaded = PolicyModel::load_safetensors(&path, tiny(), Device::Cpu).unwrap();
    assert!(loaded.bound_identity().is_none());
    let parameters = loaded.parameter_sha256().unwrap();
    let inventory = serde_json::to_value(loaded.tensor_inventory().unwrap()).unwrap();
    let count = inventory.as_array().unwrap().len();
    let weights = sha256_hex(&std::fs::read(&path).unwrap());
    let config = tiny();
    let config_sha256 = config.sha256().unwrap();
    let state = digest(state);
    let receipt = json!({
        "schema": EXPORT_RECEIPT_SCHEMA,
        "checkpoint": {
            "accepted_identity": false,
            "carried_bindings": {
                "corpus_sha256": digest("corpus"),
                "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
            },
            "config": config,
            "config_sha256": config_sha256,
            "expected_state_sha256": state,
            "model_sha256": digest("challenger model"),
            "parameter_sha256": parameters,
            "parameter_sha256_recomputed": parameters,
            "state_sha256": state,
            "step": 61600,
            "verified_bindings": VERIFIED,
        },
        "policy": {
            "output": {"sha256": weights},
            "tensor_count": count,
            "tensors": inventory,
        },
    });
    let identity = ChallengerIdentity {
        state_sha256: state,
        parameter_sha256: parameters,
        config_sha256,
        policy_safetensors_sha256: weights,
    };
    (path, receipt, identity)
}

fn challenger_binding(
    receipt: &Value,
    identity: &ChallengerIdentity,
) -> anyhow::Result<CheckpointBinding> {
    CheckpointBinding::from_challenger_receipt_bytes(&to_bytes(receipt), identity)
}

#[path = "challenger_tests/identity.rs"]
mod identity;
#[path = "challenger_tests/policy.rs"]
mod policy;
#[path = "challenger_tests/produce.rs"]
mod produce;
#[path = "challenger_tests/propose.rs"]
mod propose;
#[path = "challenger_tests/receipt.rs"]
mod receipt;
#[path = "challenger_tests/weights.rs"]
mod weights;
