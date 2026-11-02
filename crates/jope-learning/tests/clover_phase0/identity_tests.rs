use super::*;

#[test]
fn identity_is_canonical_complete_and_model_free() {
    let id = identity();
    let wire = serde_json::to_value(&id).unwrap();
    let parsed: ExperimentIdentity = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(id.sha256().unwrap(), parsed.sha256().unwrap());
    assert_eq!(
        id.sha256().unwrap(),
        sha256_hex(&canonical_json(&id).unwrap())
    );
    for field in wire.as_object().unwrap().keys() {
        let mut missing = wire.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<ExperimentIdentity>(missing).is_err(),
            "missing {field}"
        );
    }
    let mut extra = wire;
    extra["lane"] = json!("CONTROL");
    assert!(serde_json::from_value::<ExperimentIdentity>(extra).is_err());
    for field in ["search_state_sha256", "rng_state_sha256"] {
        let mut missing = serde_json::to_value(&id).unwrap();
        missing["initial_state"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(serde_json::from_value::<ExperimentIdentity>(missing).is_err());
    }
}

#[test]
fn every_comparison_dimension_changes_identity_and_invalid_pins_fail() {
    let changes: Vec<fn(&mut ExperimentIdentity)> = vec![
        |i| i.source.commit = "b".repeat(40),
        |i| i.source.closure_sha256 = digest("other source"),
        |i| {
            i.initial_state = InitialState::Checkpoint {
                parameters_sha256: digest("params"),
                optimizer_sha256: digest("optimizer"),
                rng_state_sha256: digest("rng"),
            }
        },
        |i| i.evaluator_sha256 = digest("other eval"),
        |i| i.parity_contract_sha256 = digest("other parity"),
        |i| i.support_sha256 = digest("expanded support"),
        |i| i.objective_sha256 = digest("other objective"),
        |i| i.data_roster_sha256 = digest("other data"),
        |i| i.seed = 1,
        |i| i.budget.attempts = 3,
        |i| {
            i.guards
                .insert("hold_recovery".into(), digest("other guard"))
                .map(|_| ())
                .unwrap()
        },
    ];
    let original = identity().sha256().unwrap();
    for change in changes {
        let mut id = identity();
        change(&mut id);
        assert_ne!(id.sha256().unwrap(), original);
    }
    let invalid: Vec<fn(&mut ExperimentIdentity)> = vec![
        |i| i.source.commit.truncate(8),
        |i| i.source.closure_sha256.clear(),
        |i| i.evaluator_sha256 = "unbound".into(),
        |i| i.support_sha256.clear(),
        |i| i.guards.clear(),
        |i| i.budget.attempts = 0,
        |i| i.budget.daily_bytes += 1,
        |i| i.budget.daily_puts += 1,
        |i| {
            i.initial_state = InitialState::Checkpoint {
                parameters_sha256: digest("params"),
                optimizer_sha256: String::new(),
                rng_state_sha256: digest("rng"),
            }
        },
    ];
    for change in invalid {
        let mut id = identity();
        change(&mut id);
        assert!(id.sha256().is_err());
    }
}

#[test]
fn conflicting_lane_labels_are_rejected_after_restart() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    drop(RunStore::create(&path, "CONTROL", identity()).unwrap());
    RunStore::open(&path, "CONTROL", identity()).unwrap();
    let mut other = identity();
    other.seed = 1;
    assert!(RunStore::open(&path, "CONTROL", other).is_err());
    RunStore::open(&path, "ordinary-gp", identity()).unwrap();
    let mut invalid = identity();
    invalid.evaluator_sha256.clear();
    let absent = tmp.path().join("invalid.sqlite");
    assert!(RunStore::open(&absent, "bad", invalid).is_err());
    assert!(!absent.exists());
}
