use super::*;

#[test]
fn the_flag_is_jope_seal_gate_enabled_and_defaults_off() {
    assert_eq!(FLAG, "JOPE_SEAL_GATE_ENABLED");
    assert_eq!(Gate::default(), Gate::Off);
    assert_eq!(Gate::from_value(None), Gate::Off);
    assert!(!Gate::from_value(None).is_on());
}

#[test]
fn this_test_environment_keeps_the_gate_off() {
    // A tripwire: the gate must never be switched on where these tests run.
    assert!(
        !Gate::from_env().is_on(),
        "{FLAG} is on in the test environment"
    );
}

#[test]
fn only_the_exact_value_1_turns_the_gate_on() {
    assert_eq!(Gate::from_value(Some(OsStr::new("1"))), Gate::On);
    for off in ["", "0"] {
        assert_eq!(
            Gate::from_value(Some(OsStr::new(off))),
            Gate::Off,
            "{off:?}"
        );
    }
    for value in [
        "true", "TRUE", "yes", "on", " 1", "1 ", "01", "2", "-1", "1\n",
    ] {
        let gate = Gate::from_value(Some(OsStr::new(value)));
        assert_eq!(gate, Gate::Unrecognized(value.to_owned()), "{value:?}");
        assert!(!gate.is_on(), "{value:?}");
    }
    let not_utf8 = Gate::from_value(Some(OsStr::from_bytes(b"1\xff")));
    assert!(matches!(not_utf8, Gate::Unrecognized(_)) && !not_utf8.is_on());
}

// ---- seal (G23 / J17) --------------------------------------------------------------------------

#[test]
fn seal_is_refused_by_default_before_the_proposal_is_read() {
    for input in [committed_proposal(), b"not json".to_vec(), Vec::new()] {
        let refusal = request_seal(&Gate::from_env(), &input).unwrap_err();
        assert_eq!(
            refusal,
            Refusal::GateOff {
                action: Action::Seal,
                gate: Gate::Off
            }
        );
    }
    let refusal = request_seal(&Gate::Unrecognized("true".to_owned()), b"not json").unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::GateOff {
            action: Action::Seal,
            ..
        }
    ));
    assert!(refusal.to_string().contains("JOPE_SEAL_GATE_ENABLED"));
}

#[test]
fn the_committed_proposal_is_sound_but_its_alpha_budget_is_undecided() {
    let raw = committed_proposal();
    let report = roster::check(&raw).unwrap();
    // Shown with --nocapture: the report a reviewer can bind to the proposal's SHA-256.
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    assert_eq!(report.proposal_sha256, sha256_hex(&raw));
    assert!(report.defects.is_empty(), "{:?}", report.defects);
    assert_eq!(report.owner_decisions.len(), 1);
    assert!(report.owner_decisions[0].contains("alpha budget (U13)"));
    assert!(report.owner_decisions[0].contains("unspent_jepson_decision"));
    assert!(report
        .limitations
        .iter()
        .any(|item| item.contains("pretraining closure")));
    assert!(!report.adoptable());
    assert!(!report.adopted && !report.seal_created);
    assert!(report.families["sealed_final"] > 0);

    // Even with the gate on, the undecided budget keeps it from being sealed.
    let refusal = request_seal(&Gate::On, &raw).unwrap_err();
    match refusal {
        Refusal::NotReady { action, reasons } => {
            assert_eq!(action, Action::Seal);
            assert!(reasons.iter().any(|item| item.contains("alpha budget")));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_adoptable_proposal_is_still_refused_for_the_pending_g23_ruling() {
    let raw = bytes(&proposal(json!(0.05)));
    let report = roster::check(&raw).unwrap();
    assert!(report.adoptable(), "{:?}", report.reasons());
    assert_eq!(report.families["sealed_final"], 3);
    let refusal = request_seal(&Gate::On, &raw).unwrap_err();
    assert_eq!(
        refusal,
        Refusal::RulingPending {
            action: Action::Seal
        }
    );
    assert!(refusal.to_string().contains("G23"));
    assert!(refusal.to_string().contains("this build has no seal path"));
}
