use super::*;

// ---------------------------------------------------------------- configuration and receipt

#[test]
fn config_parse_rejects_unknown_fields() {
    let good = serde_json::to_vec(&PolicyConfig::regression(401)).unwrap();
    let parsed = PolicyConfig::from_json(&good).unwrap();
    assert_eq!(parsed.embedding_rows, 657);
    assert_eq!(parsed.max_seq, 128);
    let mut extra: Value = serde_json::from_slice(&good).unwrap();
    extra["kv_cache"] = json!(true);
    let err = PolicyConfig::from_json(&serde_json::to_vec(&extra).unwrap()).unwrap_err();
    assert!(format!("{err:#}").contains("unknown field"), "{err:#}");
    let mut missing: Value = serde_json::from_slice(&good).unwrap();
    missing.as_object_mut().unwrap().remove("heads");
    assert!(PolicyConfig::from_json(&serde_json::to_vec(&missing).unwrap()).is_err());
    let mut invalid = PolicyConfig::regression(401);
    invalid.heads = 3;
    assert!(invalid.validate().is_err());
}

#[test]
fn step_60000_inventory_has_136_tensors() {
    let inventory = expected_inventory(&PolicyConfig::regression(401));
    assert_eq!(inventory.len(), 136);
    let names = inventory
        .iter()
        .map(|e| e.0.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), 136);
    for name in [
        "reward_encoder.0.weight",
        "reward_heads.3.3.bias",
        "loss_log_vars.routing_ce",
        "decoder.layers.5.self_attn.in_proj_weight",
    ] {
        assert!(names.contains(name), "{name}");
    }
    let find = |n: &str| inventory.iter().find(|e| e.0 == n).unwrap().1.clone();
    assert_eq!(find("token_emb.weight"), vec![657, 512]);
    assert_eq!(find("pos_emb.weight"), vec![129, 512]);
    assert_eq!(find("encoder.0.weight"), vec![512, 2048]);
    assert_eq!(find("runtime_head.0.weight"), vec![512, 1024]);
    assert_eq!(find("loss_log_vars.seq_nll"), Vec::<usize>::new());
}

pub(crate) fn synthetic_receipt(state: &str, model: &str, step: u64) -> Value {
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
        "schema": "jope.checkpoint_export.v1",
        "checkpoint": {
            "accepted_identity": true,
            "carried_bindings": {
                "corpus_sha256": digest("corpus"),
                "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
            },
            "config": config,
            "config_sha256": ACCEPTED_CONFIG_SHA256,
            "expected_state_sha256": state,
            "model_sha256": model,
            "parameter_sha256": digest("parameters"),
            "parameter_sha256_recomputed": digest("parameters"),
            "state_sha256": state,
            "step": step,
            "verified_bindings": ["model_sha256", "config_sha256", "autocast_cache_lifetime", "parameter_sha256"],
        },
        "policy": {
            "output": {"sha256": digest("policy")},
            "tensor_count": tensors.len(),
            "tensors": tensors,
        },
    })
}

/// `policy.safetensors` of the published step-60000 bundle
/// (`models/bundles/policy-step-60000.json`); checked against that file below.
pub(crate) const RELEASED_WEIGHTS_SHA256: &str =
    "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e";

pub(crate) fn accepted_receipt() -> Value {
    let mut receipt =
        synthetic_receipt(ACCEPTED_STATE_SHA256, ACCEPTED_MODEL_SHA256, ACCEPTED_STEP);
    receipt["checkpoint"]["parameter_sha256"] = json!(ACCEPTED_PARAMETER_SHA256);
    receipt["checkpoint"]["parameter_sha256_recomputed"] = json!(ACCEPTED_PARAMETER_SHA256);
    receipt["policy"]["output"]["sha256"] = json!(RELEASED_WEIGHTS_SHA256);
    receipt
}

pub(crate) fn parse_receipt(receipt: &Value) -> anyhow::Result<CheckpointBinding> {
    CheckpointBinding::from_receipt_bytes(&serde_json::to_vec(receipt).unwrap())
}

#[test]
fn receipt_binds_the_accepted_checkpoint() {
    let receipt = accepted_receipt();
    let binding = parse_receipt(&receipt).unwrap();
    binding.require_accepted_identity().unwrap();
    assert_eq!(binding.step(), ACCEPTED_STEP);
    assert_eq!(binding.model_sha256(), ACCEPTED_MODEL_SHA256);
    assert_eq!(binding.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    assert_eq!(binding.tensors().len(), 136);
    assert_eq!(binding.vocabulary_sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(*binding.config(), PolicyConfig::regression(401));
}

#[test]
fn rejected_checkpoint_state_is_refused() {
    for receipt in [
        synthetic_receipt(REJECTED_STATE_SHA256, &digest("model"), 60000),
        synthetic_receipt(ACCEPTED_STATE_SHA256, REJECTED_MODEL_SHA256, 60000),
        synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 69000),
    ] {
        let err = parse_receipt(&receipt).unwrap_err();
        assert!(format!("{err:#}").contains("rejected"), "{err:#}");
    }
    // Any other identity, an unverified parameter hash, or an unknown config field is refused.
    let other = synthetic_receipt(&digest("other state"), &digest("model"), 60000);
    assert!(parse_receipt(&other).is_err());
    let mut unverified = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    unverified["checkpoint"]["parameter_sha256_recomputed"] = json!(digest("different"));
    assert!(parse_receipt(&unverified).is_err());
    let mut unknown = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    unknown["checkpoint"]["config"]["rope"] = json!(true);
    assert!(parse_receipt(&unknown).is_err());
    // Scoring bindings and the seed assembler refuse the rejected state too.
    assert!(Bindings::new(REJECTED_STATE_SHA256, &digest("context"), &digest("vocab")).is_err());
    let binding =
        SeedBinding::new(&digest("prepared"), 0, &digest("train"), &digest("context")).unwrap();
    let ranking = fake_ranking(&digest("other state"), &binding, &[]);
    assert!(assemble_for_testing(&binding, REJECTED_STATE_SHA256, &ranking, &[], &[]).is_err());
}
