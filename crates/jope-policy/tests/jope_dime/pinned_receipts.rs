use super::*;

// ---------------------------------------------------------------- W2 G-08: pinned receipt fields

/// Every object the accepted receipt carries: where the G-08 cases add one unpinned key.
pub(crate) const ACCEPTED_OBJECTS: &[&[&str]] = &[
    &[],
    &["runtime"],
    &["checkpoint"],
    &["checkpoint", "carried_bindings"],
    &["checkpoint", "config"],
    &["policy"],
    &["policy", "source"],
    &["policy", "output"],
    &["policy", "output", "round_trip"],
    &["context_transforms"],
    &["context_transforms", "source"],
    &["context_transforms", "output"],
    &["context_transforms", "output", "round_trip"],
    &["context_transforms", "assembly"],
];

/// Every object a native challenger receipt carries (`jope-parallel export-policy`).
pub(crate) const CHALLENGER_OBJECTS: &[&[&str]] = &[
    &[],
    &["runtime"],
    &["checkpoint"],
    &["checkpoint", "carried_bindings"],
    &["checkpoint", "config"],
    &["policy"],
    &["policy", "source"],
    &["policy", "output"],
    &["policy", "output", "round_trip"],
    &["relative_change"],
    &["relative_change", "parent"],
    &["relative_change", "total"],
];

pub(crate) const ACCEPTED_FIELD_REFUSAL: &str = "outside the pinned accepted receipt field set";
pub(crate) const CHALLENGER_FIELD_REFUSAL: &str = "outside the pinned challenger receipt field set";

pub(crate) fn at<'a>(value: &'a mut Value, path: &[&str]) -> &'a mut Value {
    path.iter().fold(value, |v, key| &mut v[*key])
}

pub(crate) fn round_trip() -> Value {
    json!({"readers": ["tch::Tensor::read_safetensors", "independent header parse"],
           "compared": ["name set", "dtype", "shape", "raw bytes"], "bit_exact": true})
}

/// Every field the accepted receipt carries (jain-datasets `tools/ot-export` 0.1.0,
/// `step-60000.receipt.json`), around the synthetic accepted identity.
pub(crate) fn full_accepted_receipt() -> Value {
    let mut receipt = accepted_receipt();
    receipt["runtime"] = json!({"tool": "ot-export", "tool_version": "0.1.0", "tch": "0.26.0",
        "libtorch": "/sdk/libtorch", "libtorch_build_version": "2.13.0+cu130", "device": "cpu",
        "cuda_visible_devices": ""});
    let checkpoint = &mut receipt["checkpoint"];
    checkpoint["directory"] = json!("/runs/step-60000");
    checkpoint["state_bytes"] = json!(403795);
    checkpoint["refused_identities"] = json!([REJECTED_STATE_SHA256]);
    checkpoint["refused_steps"] = json!([69000]);
    let policy = &mut receipt["policy"];
    policy["source"] = json!({"path": "/runs/step-60000/model.ot",
        "sha256": ACCEPTED_MODEL_SHA256, "file_bytes": 111218463});
    policy["output"]["path"] = json!("/out/step-60000/policy.safetensors");
    policy["output"]["file_bytes"] = json!(111199088);
    policy["output"]["round_trip"] = round_trip();
    policy["total_tensor_bytes"] = json!(111185976);
    policy["inventory_sha256"] = json!(digest("inventory"));
    receipt["context_transforms"] = json!({
        "source": {"path": "/encodings/transforms.ot", "sha256": digest("transforms.ot"),
                   "file_bytes": 8413808},
        "output": {"path": "/out/step-60000/context_transforms.safetensors",
                   "sha256": digest("transforms"), "file_bytes": 8413808,
                   "round_trip": round_trip()},
        "tensor_count": 1,
        "total_tensor_bytes": 4096,
        "inventory_sha256": digest("transforms inventory"),
        "tensors": [{"name": "projection_mean", "dtype": "F32", "shape": [1024], "numel": 1024,
                     "bytes": 4096, "sha256": digest("projection_mean")}],
        "assembly": {"path": null, "sha256": digest("assembly"),
                     "binding": "pinned pair", "clip_abs": null},
    });
    receipt
}

/// W2 G-08, owner decision 2026-09-26 (`docs/internal/core-ip/work-orders/OWNER-DECISIONS.md`
/// D13): the accepted receipt's field set is pinned. A receipt carrying every pinned field binds
/// the accepted identity; the same receipt with one more field anywhere is refused, and so are
/// the native challenger exporter's own fields.
#[test]
fn an_accepted_receipt_carrying_an_unpinned_field_is_refused() {
    let receipt = full_accepted_receipt();
    let bytes = serde_json::to_vec(&receipt).unwrap();
    CheckpointBinding::for_proposal(&bytes, None).unwrap();
    parse_receipt(&receipt)
        .unwrap()
        .require_accepted_identity()
        .unwrap();
    let mut cases = ACCEPTED_OBJECTS
        .iter()
        .map(|path| {
            let mut extra = receipt.clone();
            at(&mut extra, path)["unpinned"] = json!(true);
            extra
        })
        .collect::<Vec<_>>();
    // A tensor entry, and an object inside an array the pinned set holds only scalars in.
    let mut entry = receipt.clone();
    entry["context_transforms"]["tensors"][0]["role"] = json!("decode");
    let mut nested = receipt.clone();
    nested["checkpoint"]["verified_bindings"] = json!([{"model_sha256": true}]);
    // The native challenger exporter's own fields are not part of the accepted receipt.
    let mut architecture = receipt.clone();
    architecture["policy"]["architecture"] = json!("neural::Policy::new names and shapes");
    let mut relative = receipt.clone();
    relative["relative_change"] = Value::Null;
    cases.extend([entry, nested, architecture, relative]);
    for case in cases {
        let bytes = serde_json::to_vec(&case).unwrap();
        assert_refused(
            CheckpointBinding::from_receipt_bytes(&bytes),
            ACCEPTED_FIELD_REFUSAL,
        );
        assert_refused(
            CheckpointBinding::for_proposal(&bytes, None),
            ACCEPTED_FIELD_REFUSAL,
        );
    }
    // The rejected checkpoint is still refused first, whatever else the receipt carries.
    let mut rejected = synthetic_receipt(REJECTED_STATE_SHA256, &digest("model"), 60000);
    rejected["unpinned"] = json!(true);
    assert_refused(parse_receipt(&rejected), "rejected");
}

/// The challenger path is pinned to the field set this repository's native challenger exporter
/// (`jope-parallel export-policy`) writes; one more field, or the accepted exporter's
/// `context_transforms`, is refused there.
#[test]
fn a_challenger_receipt_carrying_an_unpinned_field_is_refused() {
    let identity = ChallengerIdentity {
        state_sha256: digest("challenger state"),
        parameter_sha256: digest("challenger parameters"),
        config_sha256: ACCEPTED_CONFIG_SHA256.into(),
        policy_safetensors_sha256: digest("challenger weights"),
    };
    let mut receipt = full_accepted_receipt();
    receipt
        .as_object_mut()
        .unwrap()
        .remove("context_transforms");
    receipt["runtime"]["tool"] = json!("jope-parallel export-policy");
    let checkpoint = &mut receipt["checkpoint"];
    checkpoint["accepted_identity"] = json!(false);
    checkpoint["state_sha256"] = json!(identity.state_sha256);
    checkpoint["expected_state_sha256"] = json!(identity.state_sha256);
    checkpoint["model_sha256"] = json!(digest("challenger model"));
    checkpoint["step"] = json!(61600);
    checkpoint["parameter_sha256"] = json!(identity.parameter_sha256);
    checkpoint["parameter_sha256_recomputed"] = json!(identity.parameter_sha256);
    receipt["policy"]["output"]["sha256"] = json!(identity.policy_safetensors_sha256);
    receipt["policy"]["architecture"] = json!("neural::Policy::new names and shapes; F32; finite");
    receipt["relative_change"] = json!({
        "parent": {"directory": "/runs/step-60000", "state_sha256": ACCEPTED_STATE_SHA256,
                   "step": 60000, "model_sha256": ACCEPTED_MODEL_SHA256,
                   "parameter_sha256": ACCEPTED_PARAMETER_SHA256},
        "norm": "l2 over float32 values, f64 accumulation in row-major order",
        "tensors": [{"name": "ctx_proj.bias", "parent_norm": 1.0, "change_norm": 0.5,
                     "relative_change": 0.5, "changed_values": 3}],
        "total": {"parent_norm": 1.0, "change_norm": 0.5, "relative_change": 0.5},
    });
    let parse = |r: &Value| {
        CheckpointBinding::from_challenger_receipt_bytes(&serde_json::to_vec(r).unwrap(), &identity)
    };
    parse(&receipt).unwrap();
    CheckpointBinding::for_proposal(&serde_json::to_vec(&receipt).unwrap(), Some(&identity))
        .unwrap();
    // Without a parent the exporter writes `relative_change: null`.
    let mut unparented = receipt.clone();
    unparented["relative_change"] = Value::Null;
    parse(&unparented).unwrap();
    let mut cases = CHALLENGER_OBJECTS
        .iter()
        .map(|path| {
            let mut extra = receipt.clone();
            at(&mut extra, path)["unpinned"] = json!(true);
            extra
        })
        .collect::<Vec<_>>();
    let mut row = receipt.clone();
    row["relative_change"]["tensors"][0]["unpinned"] = json!(true);
    let mut transforms = receipt.clone();
    transforms["context_transforms"] = full_accepted_receipt()["context_transforms"].clone();
    cases.extend([row, transforms]);
    for case in cases {
        assert_refused(parse(&case), CHALLENGER_FIELD_REFUSAL);
    }
    // The accepted parser refuses it (it names no accepted identity and carries challenger fields).
    assert!(parse_receipt(&receipt).is_err());
}

pub(crate) const DERIVED_REFUSAL: &str = "a derived re-export carries no acceptance authority";

/// The accepted checkpoint re-exported by this repository's exporter, exactly as
/// `jope-parallel export-policy` writes it (the ASM-demo anchor's shape).
pub(crate) fn derived_reexport_receipt() -> Value {
    let mut receipt = full_accepted_receipt();
    receipt
        .as_object_mut()
        .unwrap()
        .remove("context_transforms");
    receipt["runtime"]["tool"] = json!(jope_policy::config::REEXPORT_TOOL);
    receipt["policy"]["architecture"] = json!("neural::Policy::new names and shapes; F32; finite");
    receipt["relative_change"] = Value::Null;
    receipt
}

/// D13, the ASM-demo anchor's disposition: a re-export of the accepted checkpoint by this
/// repository's exporter is refused wherever acceptance authority is needed (accepted parser,
/// proposal path, challenger path), and is consumed only as a non-authoritative derived
/// re-export that must reproduce the accepted receipt tensor for tensor.
#[test]
fn a_derived_reexport_reproduces_the_accepted_receipt_without_authority() {
    let accepted = parse_receipt(&full_accepted_receipt()).unwrap();
    let reexport = derived_reexport_receipt();
    let bytes = serde_json::to_vec(&reexport).unwrap();
    // Refused wherever an acceptance authority is required, with a plain reason.
    assert_refused(
        CheckpointBinding::from_receipt_bytes(&bytes),
        DERIVED_REFUSAL,
    );
    assert_refused(
        CheckpointBinding::for_proposal(&bytes, None),
        DERIVED_REFUSAL,
    );
    let identity = ChallengerIdentity {
        state_sha256: ACCEPTED_STATE_SHA256.into(),
        parameter_sha256: ACCEPTED_PARAMETER_SHA256.into(),
        config_sha256: ACCEPTED_CONFIG_SHA256.into(),
        policy_safetensors_sha256: RELEASED_WEIGHTS_SHA256.into(),
    };
    assert!(CheckpointBinding::for_proposal(&bytes, Some(&identity)).is_err());
    // Consumed only as derived evidence: it reproduces the accepted export.
    let derived = accepted.derived_reexport(&bytes).unwrap();
    assert_eq!(derived.receipt_sha256, sha256_hex(&bytes));
    assert_eq!(derived.policy_safetensors_sha256, RELEASED_WEIGHTS_SHA256);
    assert!(derived.byte_identical);
    // Different re-exported bytes are recorded, not refused (anchor rule A2).
    let mut rewritten = reexport.clone();
    rewritten["policy"]["output"]["sha256"] = json!(digest("re-serialized weights"));
    let derived = accepted
        .derived_reexport(&serde_json::to_vec(&rewritten).unwrap())
        .unwrap();
    assert!(!derived.byte_identical);
    // Anything that does not reproduce the accepted export is refused (anchor rule A1).
    type Edit = fn(&mut Value);
    let cases: [(&str, Edit); 7] = [
        ("another tensor", |r| {
            r["policy"]["tensors"][0]["sha256"] = json!(digest("other tensor"))
        }),
        ("another state", |r| {
            r["checkpoint"]["state_sha256"] = json!(digest("other state"))
        }),
        ("unreproduced parameters", |r| {
            r["checkpoint"]["parameter_sha256_recomputed"] = json!(digest("other"))
        }),
        ("another vocabulary", |r| {
            r["checkpoint"]["carried_bindings"]["vocabulary_sha256"] =
                json!(PRODUCT_VOCABULARY_SHA256)
        }),
        ("another exporter", |r| {
            r["runtime"]["tool"] = json!("ot-export")
        }),
        ("an unpinned field", |r| {
            r["policy"]["unpinned"] = json!(true)
        }),
        ("the accepted exporter's transforms", |r| {
            r["context_transforms"] = full_accepted_receipt()["context_transforms"].clone()
        }),
    ];
    for (case, edit) in cases {
        let mut forged = reexport.clone();
        edit(&mut forged);
        assert!(
            accepted
                .derived_reexport(&serde_json::to_vec(&forged).unwrap())
                .is_err(),
            "{case} accepted as a derived re-export"
        );
    }
    // Only the accepted binding can vouch for a re-export.
    let challenger = CheckpointBinding::unbound_for_testing(
        PolicyConfig::regression(401),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    assert!(challenger.derived_reexport(&bytes).is_err());
}

/// W2 G-08 on the real receipts (owner decision 2026-09-26, `OWNER-DECISIONS.md` D13).
/// `JOPE_DIME_EXPORT_RECEIPT` is the accepted `step-60000.receipt.json` (`tools/ot-export`): it
/// binds the accepted identity and the released weights under the pinned field set, and one added
/// field in any of its objects is refused. `JOPE_DIME_REEXPORT_RECEIPT` (the ASM-demo anchor, a
/// re-export of the accepted checkpoint by `jope-parallel export-policy`) is refused as the
/// accepted receipt and verified only as a non-authoritative derived re-export.
#[test]
#[ignore = "needs JOPE_DIME_EXPORT_RECEIPT (the accepted step-60000.receipt.json)"]
fn the_real_accepted_receipt_carries_only_pinned_fields() {
    let bytes = std::fs::read(env_path("JOPE_DIME_EXPORT_RECEIPT")).unwrap();
    println!("accepted receipt sha256 {}", sha256_hex(&bytes));
    let binding = CheckpointBinding::for_proposal(&bytes, None).unwrap();
    assert_eq!(binding.state_sha256(), ACCEPTED_STATE_SHA256);
    assert_eq!(binding.policy_safetensors_sha256(), RELEASED_WEIGHTS_SHA256);
    let receipt: Value = serde_json::from_slice(&bytes).unwrap();
    for path in ACCEPTED_OBJECTS {
        assert!(at(&mut receipt.clone(), path).is_object(), "{path:?}");
        let mut extra = receipt.clone();
        at(&mut extra, path)["unpinned"] = json!(true);
        assert_refused(
            CheckpointBinding::for_proposal(&serde_json::to_vec(&extra).unwrap(), None),
            ACCEPTED_FIELD_REFUSAL,
        );
    }
    if let Some(path) = std::env::var_os("JOPE_DIME_REEXPORT_RECEIPT") {
        let bytes = std::fs::read(path).unwrap();
        println!("re-export receipt sha256 {}", sha256_hex(&bytes));
        assert_refused(
            CheckpointBinding::for_proposal(&bytes, None),
            DERIVED_REFUSAL,
        );
        let derived = binding.derived_reexport(&bytes).unwrap();
        println!(
            "derived re-export: weights {} byte_identical={} (no acceptance authority)",
            derived.policy_safetensors_sha256, derived.byte_identical
        );
    }
}
