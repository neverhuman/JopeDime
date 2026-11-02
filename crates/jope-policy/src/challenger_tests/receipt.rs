use super::*;

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
    type ReceiptEdit<'a> = &'a dyn Fn(&mut Value);
    let cases: [(ReceiptEdit<'_>, &str); 16] = [
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
