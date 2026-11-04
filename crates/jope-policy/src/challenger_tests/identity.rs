use super::*;

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
