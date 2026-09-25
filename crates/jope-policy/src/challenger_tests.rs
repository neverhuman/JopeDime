//! Native challenger binding (plan §4.5, DEMO-3) on the tiny CPU policy: the identity pins, the
//! challenger receipt parse, bound loading and tampering, refusal by every accepted-only path,
//! the challenger proposal, and the producer transport of a challenger request. These run in
//! the lib test target, where `cfg(test)` enables the test helpers, so the hosted `rust-test`
//! lane runs them. Scratch lives under the workspace `target/tmp`, never `/tmp`.
use crate::config::{ChallengerIdentity, CheckpointBinding, PolicyConfig, EXPORT_RECEIPT_SCHEMA};
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

fn context() -> Vec<f32> {
    (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect()
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
        .join(format!("jope-policy-challenger-{}-{tag}", std::process::id()));
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
            "output": {"sha256": digest("policy")},
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

#[test]
fn challenger_identity_refuses_accepted_rejected_and_malformed_pins() {
    let valid = ChallengerIdentity {
        state_sha256: digest("challenger state"),
        parameter_sha256: digest("challenger parameters"),
        config_sha256: tiny().sha256().unwrap(),
        policy_safetensors_sha256: digest("challenger weights"),
    };
    valid.validate().unwrap();
    let with = |edit: &dyn Fn(&mut ChallengerIdentity)| {
        let mut identity = valid.clone();
        edit(&mut identity);
        identity.validate()
    };
    let accepted_pins: [&dyn Fn(&mut ChallengerIdentity); 3] = [
        &|i: &mut ChallengerIdentity| i.state_sha256 = ACCEPTED_STATE_SHA256.into(),
        &|i: &mut ChallengerIdentity| i.parameter_sha256 = ACCEPTED_PARAMETER_SHA256.into(),
        &|i: &mut ChallengerIdentity| i.policy_safetensors_sha256 = ACCEPTED_WEIGHTS.into(),
    ];
    for edit in accepted_pins {
        assert_refused(with(edit), "challenger names the accepted identity");
    }
    assert_refused(
        with(&|i: &mut ChallengerIdentity| i.state_sha256 = REJECTED_STATE_SHA256.into()),
        "rejected step-69000",
    );
    // Every pin must be exactly 64 lowercase hex characters.
    let set = |identity: &mut ChallengerIdentity, field: usize, value: String| match field {
        0 => identity.state_sha256 = value,
        1 => identity.parameter_sha256 = value,
        2 => identity.config_sha256 = value,
        _ => identity.policy_safetensors_sha256 = value,
    };
    let pins = [
        valid.state_sha256.clone(),
        valid.parameter_sha256.clone(),
        valid.config_sha256.clone(),
        valid.policy_safetensors_sha256.clone(),
    ];
    for (field, pin) in pins.iter().enumerate() {
        for malformed in [pin.to_ascii_uppercase(), pin[..63].to_string()] {
            let mut identity = valid.clone();
            set(&mut identity, field, malformed);
            assert_refused(identity.validate(), "lowercase sha256");
        }
    }
    // The architecture is not jope-policy's decision: the accepted config is allowed here.
    with(&|i: &mut ChallengerIdentity| i.config_sha256 = ACCEPTED_CONFIG_SHA256.into()).unwrap();
    // Unknown keys are refused; the pins round-trip.
    let value = serde_json::to_value(&valid).unwrap();
    assert_eq!(
        serde_json::from_value::<ChallengerIdentity>(value.clone()).unwrap(),
        valid
    );
    let mut extra = value;
    extra["extra"] = json!(1);
    assert!(serde_json::from_value::<ChallengerIdentity>(extra).is_err());
}

#[test]
fn challenger_receipt_binds_every_pin() {
    let dir = scratch("receipt");
    let (_, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    std::fs::remove_dir_all(&dir).unwrap();
    let binding = challenger_binding(&receipt, &identity).unwrap();
    assert_eq!(binding.state_sha256(), identity.state_sha256);
    assert_eq!(binding.step(), 61600);
    binding.require_challenger_identity(&identity).unwrap();
    let other = digest("other");
    // A binding checked against an identity that differs only in state.
    assert_refused(
        binding.require_challenger_identity(&ChallengerIdentity {
            state_sha256: other.clone(),
            ..identity.clone()
        }),
        "checkpoint is not the pinned challenger",
    );
    let cases: [(&dyn Fn(&mut Value), &str); 16] = [
        (
            &|r: &mut Value| r["checkpoint"]["accepted_identity"] = json!(true),
            "does not carry the pinned challenger identity",
        ),
        // Expected state differs from state.
        (
            &|r: &mut Value| r["checkpoint"]["expected_state_sha256"] = json!(other),
            "does not carry the pinned challenger identity",
        ),
        // State differs from expected state.
        (
            &|r: &mut Value| r["checkpoint"]["state_sha256"] = json!(other),
            "does not carry the pinned challenger identity",
        ),
        // A self-consistent receipt for another state: only the identity pin refuses it.
        (
            &|r: &mut Value| {
                r["checkpoint"]["state_sha256"] = json!(other);
                r["checkpoint"]["expected_state_sha256"] = json!(other);
            },
            "does not carry the pinned challenger identity",
        ),
        (
            &|r: &mut Value| {
                r["checkpoint"]["parameter_sha256"] = json!(other);
                r["checkpoint"]["parameter_sha256_recomputed"] = json!(other);
            },
            "not the pinned challenger parameters",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["parameter_sha256_recomputed"] = json!(other),
            "parameter hash was not reproduced",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["config_sha256"] = json!(other),
            "not the pinned challenger config",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["config"]["width"] = json!(32),
            "not the pinned challenger config",
        ),
        (
            &|r: &mut Value| r["policy"]["output"]["sha256"] = json!(other),
            "names other challenger weights",
        ),
        (
            &|r: &mut Value| {
                r["checkpoint"]["carried_bindings"]["vocabulary_sha256"] =
                    json!(PRODUCT_VOCABULARY_SHA256);
            },
            "is not the step-60000 vocabulary",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["model_sha256"] = json!(ACCEPTED_MODEL_SHA256),
            "not the pinned challenger",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["model_sha256"] = json!(REJECTED_MODEL_SHA256),
            "rejected step-69000",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["step"] = json!(ACCEPTED_STEP),
            "does not continue",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["step"] = json!(1),
            "does not continue",
        ),
        (
            &|r: &mut Value| r["checkpoint"]["step"] = json!(69000),
            "rejected step-69000",
        ),
        (
            &|r: &mut Value| {
                r["checkpoint"]["verified_bindings"] =
                    json!(["model_sha256", "config_sha256", "autocast_cache_lifetime"]);
            },
            "did not verify parameter_sha256",
        ),
    ];
    for (edit, needle) in cases {
        let mut changed = receipt.clone();
        edit(&mut changed);
        assert_refused(challenger_binding(&changed, &identity), needle);
    }
    // Parser separation: the accepted parser never admits a challenger receipt, and the
    // challenger parser never admits the accepted one.
    assert_refused(
        CheckpointBinding::from_receipt_bytes(&to_bytes(&receipt)),
        "does not carry a verified accepted identity",
    );
    assert!(CheckpointBinding::for_proposal(&to_bytes(&receipt), None).is_err());
    let accepted = to_bytes(&accepted_receipt());
    CheckpointBinding::for_proposal(&accepted, None).unwrap();
    assert!(CheckpointBinding::for_proposal(&accepted, Some(&identity)).is_err());
    let proposed = CheckpointBinding::for_proposal(&to_bytes(&receipt), Some(&identity)).unwrap();
    assert_eq!(proposed.state_sha256(), identity.state_sha256);
}

#[test]
fn challenger_weights_load_bound_and_tampering_is_refused() {
    let dir = scratch("load");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let model = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    let bound = model.bound_identity().unwrap();
    assert_eq!(bound.state_sha256(), identity.state_sha256);
    assert_eq!(bound.parameter_sha256(), identity.parameter_sha256);
    assert_eq!(
        bound.policy_safetensors_sha256(),
        identity.policy_safetensors_sha256
    );
    // One flipped bit in the last tensor's data.
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let tampered = sha256_hex(&bytes);
    // (a) The original binding refuses the file.
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "differs from its export receipt",
    );
    // (b) Receipt and identity re-pinned to the new file: the per-tensor inventory refuses it.
    let mut repinned = receipt.clone();
    repinned["policy"]["output"]["sha256"] = json!(tampered);
    let repinned_identity = ChallengerIdentity {
        policy_safetensors_sha256: tampered.clone(),
        ..identity.clone()
    };
    let binding = challenger_binding(&repinned, &repinned_identity).unwrap();
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "differ from the export receipt inventory",
    );
    // (c) The inventory re-pinned as well: the recomputed parameter hash refuses it.
    let inventory = PolicyModel::load_safetensors(&path, tiny(), Device::Cpu)
        .unwrap()
        .tensor_inventory()
        .unwrap();
    repinned["policy"]["tensors"] = serde_json::to_value(inventory).unwrap();
    let binding = challenger_binding(&repinned, &repinned_identity).unwrap();
    assert_refused(
        PolicyModel::load_bound(&path, &binding, Device::Cpu),
        "recomputed parameter hash differs",
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn challenger_never_satisfies_the_accepted_policy_checks() {
    const NOT_ACCEPTED: &str = "not the accepted step-60000 state";
    let dir = scratch("accepted-checks");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vocabulary = vocabulary();
    let ctx = context();
    let run = run_binding(&ctx);
    let proposal_seeds = ProposalSeeds::derive(3);
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    assert_refused(crate::require_accepted_policy(&m), NOT_ACCEPTED);
    assert_refused(
        decode::pure_pool(&m, vocabulary.vocab(), &ctx, "regression", 1, 2, &options),
        NOT_ACCEPTED,
    );
    assert_refused(
        Bindings::for_accepted_policy(&m, &context_sha256(&ctx), &vocabulary),
        NOT_ACCEPTED,
    );
    assert!(seeds::propose(
        &m,
        &binding,
        &vocabulary,
        &ctx,
        &run,
        &proposal_seeds,
        &options
    )
    .is_err());
    let accepted = CheckpointBinding::from_receipt_bytes(&to_bytes(&accepted_receipt())).unwrap();
    assert!(seeds::propose(
        &m,
        &accepted,
        &vocabulary,
        &ctx,
        &run,
        &proposal_seeds,
        &options
    )
    .is_err());
    assert!(binding.require_accepted_identity().is_err());
    // The test helpers refuse any bound model.
    let unbound = CheckpointBinding::unbound_for_testing(tiny(), CHECKPOINT_VOCABULARY_SHA256);
    assert_refused(
        crate::testing::propose_unbound(
            &m,
            &unbound,
            &vocabulary,
            &ctx,
            &run,
            &proposal_seeds,
            &options,
        ),
        "unbound test path",
    );
    assert_refused(
        crate::testing::pure_pool_unbound(
            &m,
            vocabulary.vocab(),
            &ctx,
            "regression",
            1,
            2,
            &options,
        ),
        "unbound test path",
    );
    // Scoring: never under the accepted state or another state, only under its own.
    let programs = vec!["(pipe X (ridge))".to_string()];
    let one = Options {
        batch_size: 1,
        keep: 1,
    };
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    assert!(score_pool(&m, vocabulary.vocab(), &ctx, &programs, claimed, &one).is_err());
    let elsewhere = Bindings::new(
        &digest("other state"),
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    assert_refused(
        score_pool(&m, vocabulary.vocab(), &ctx, &programs, elsewhere, &one),
        "scored under",
    );
    let own = Bindings::new(
        &identity.state_sha256,
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    let ranking = score_pool(&m, vocabulary.vocab(), &ctx, &programs, own, &one).unwrap();
    assert_eq!(
        ranking.bindings().checkpoint_sha256(),
        identity.state_sha256
    );
}

#[test]
fn propose_challenger_stamps_the_challenger_state() {
    let dir = scratch("propose");
    let (path, receipt, identity) = challenger_export(&dir.join("first"), 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    let (second_path, second_receipt, second_identity) =
        challenger_export(&dir.join("second"), 0.5, "second challenger state");
    let second_binding = challenger_binding(&second_receipt, &second_identity).unwrap();
    let second = PolicyModel::load_bound(&second_path, &second_binding, Device::Cpu).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vocabulary = vocabulary();
    let ctx = context();
    let run = run_binding(&ctx);
    let proposal_seeds = ProposalSeeds::derive(3);
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let attempt =
        |model: &PolicyModel, checkpoint: &CheckpointBinding, pinned: &ChallengerIdentity| {
            seeds::propose_challenger(
                model,
                checkpoint,
                pinned,
                &vocabulary,
                &ctx,
                &run,
                &proposal_seeds,
                &options,
            )
        };
    let proposal = attempt(&m, &binding, &identity).unwrap();
    let population = proposal.population();
    assert_eq!(population.checkpoint_sha256(), identity.state_sha256);
    assert_eq!(
        proposal.ranking().bindings().checkpoint_sha256(),
        identity.state_sha256
    );
    assert_eq!(population.candidates().len(), 80);
    assert!(population.candidates()[..64]
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL || c.origin() == ORIGIN_NEURAL_FALLBACK));
    assert!(population.candidates()[64..]
        .iter()
        .all(|c| c.origin() == ORIGIN_RANDOM));
    let admitted = validate_for_search(
        &population_bytes(&proposal).unwrap(),
        run.prepared_sha256(),
        run.cut(),
        run.training_sha256(),
    )
    .unwrap();
    assert_eq!(admitted.checkpoint_sha256(), identity.state_sha256);
    // (a) Untrained weights, (b) another pinned parameter hash, (c) another pinned state,
    // (d) a model bound to another challenger, and (e) the accepted receipt are all refused.
    let random = PolicyModel::random_weights(tiny(), 9).unwrap();
    assert_refused(attempt(&random, &binding, &identity), "load_bound");
    let other_parameters = ChallengerIdentity {
        parameter_sha256: digest("other parameters"),
        ..identity.clone()
    };
    assert_refused(
        attempt(&m, &binding, &other_parameters),
        "not the pinned challenger parameters",
    );
    let other_state = ChallengerIdentity {
        state_sha256: digest("other state"),
        ..identity.clone()
    };
    assert_refused(
        attempt(&m, &binding, &other_state),
        "checkpoint is not the pinned challenger",
    );
    assert_refused(
        attempt(&second, &binding, &identity),
        "bound to a different checkpoint",
    );
    let accepted = CheckpointBinding::from_receipt_bytes(&to_bytes(&accepted_receipt())).unwrap();
    assert!(attempt(&m, &accepted, &identity).is_err());
}

#[test]
fn produce_sends_a_challenger_request_through_propose_challenger() {
    let dir = scratch("produce");
    let (path, receipt, identity) = challenger_export(&dir, 0.0, "challenger state");
    let binding = challenger_binding(&receipt, &identity).unwrap();
    let m = PolicyModel::load_bound(&path, &binding, Device::Cpu).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let vocabulary = vocabulary();
    let ctx = context();
    let proposal_seeds = ProposalSeeds::derive(3);
    let request = producer::Request {
        worker_source_sha256: digest("producer source"),
        prepared_sha256: digest("prepared.json"),
        training_sha256: digest("cut-0/train.csv"),
        cut: 0,
        context_sha256: context_sha256(&ctx),
        checkpoint_sha256: identity.state_sha256.clone(),
        policy_safetensors_sha256: identity.policy_safetensors_sha256.clone(),
        vocabulary_sha256: CHECKPOINT_VOCABULARY_SHA256.into(),
        sampling_seed: proposal_seeds.sampling,
        fallback_seed: proposal_seeds.fallback,
        exploration_seed: proposal_seeds.exploration,
        sampling_rounds: 1,
        sampling_batch: 4,
        sampling_temperature: 1.0,
        challenger: Some(identity.clone()),
    };
    // `seeds::propose` refuses this model, so success means the request reached
    // `propose_challenger`.
    let bytes = producer::produce(&request, &m, &binding, &vocabulary, &ctx).unwrap();
    let file = producer::read_claims(&bytes, &request).unwrap();
    let population = file.population();
    assert_eq!(population.checkpoint_sha256(), identity.state_sha256);
    assert_eq!(population.candidates().len(), 80);
    // The same inputs without the challenger field are refused.
    assert_refused(
        producer::produce(
            &producer::Request {
                challenger: None,
                ..request.clone()
            },
            &m,
            &binding,
            &vocabulary,
            &ctx,
        ),
        "unbound policy safetensors",
    );
}
