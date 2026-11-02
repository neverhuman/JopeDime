use super::*;

pub(super) fn synthetic_evidence(macro_lift: f64) -> Value {
    synthetic_evidence_with(macro_lift, Some(0.06), Some(0.02))
}

/// Same shape as the cases in `tests/fixtures/conditionality_gate_parity.json`:
/// `None` means the key is absent from that side of the evidence entirely.
fn synthetic_evidence_with(
    macro_lift: f64,
    candidate_conditional: Option<f64>,
    baseline_conditional: Option<f64>,
) -> Value {
    let mut evidence = json!({
        "baseline": {
            "checkpoint_sha256": "b".repeat(64),
            "macro_pairwise_accuracy": 0.5,
            "regression_pairwise_accuracy": 0.5,
            "binary_pairwise_accuracy": 0.5,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100.0,
        },
        "candidate": {
            "checkpoint_sha256": "a".repeat(64),
            "macro_pairwise_accuracy": 0.5 + macro_lift,
            "regression_pairwise_accuracy": 0.5 + macro_lift,
            "binary_pairwise_accuracy": 0.5 + macro_lift,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 110.0,
        },
        "proposal_validity": 1.0,
        "verified_checkpoint_origin_strict_wins": 3,
        "candidate_equal_budget_result": 0.6,
        "baseline_equal_budget_result": 0.5,
        "replay_verification": {
            "predictions": true,
            "scores": true,
            "evaluator_identity": true,
            "dataset_hashes": true,
            "candidate_origin": true,
        },
    });
    if let Some(value) = candidate_conditional {
        evidence["candidate"]["val_conditional_advantage"] = json!(value);
    }
    if let Some(value) = baseline_conditional {
        evidence["baseline"]["val_conditional_advantage"] = json!(value);
    }
    evidence
}

pub(super) fn gate_reasons(result: &Value, key: &str) -> Vec<String> {
    result[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|reason| reason.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn gate_thresholds() {
    let internal = evaluate_gate(&synthetic_evidence(0.005), None).unwrap();
    assert_eq!(internal["pass"], json!(true));
    assert_eq!(internal["promotion_tier"], json!("internal"));
    let milestone = evaluate_gate(&synthetic_evidence(0.02), None).unwrap();
    assert_eq!(milestone["promotion_tier"], json!("milestone"));
    let rejected = evaluate_gate(&synthetic_evidence(0.001), None).unwrap();
    assert_eq!(rejected["pass"], json!(false));
    assert_eq!(rejected["promotion_tier"], json!("rejected"));
    assert!(rejected["reasons"]
        .as_array()
        .unwrap()
        .contains(&json!("macro_pairwise_lift_below_point_two")));
}

/// A checkpoint that does not condition on the dataset embedding may never
/// be promoted, however large its macro lift is.
#[test]
fn conditionality_is_a_hard_promotion_gate() {
    let dead = evaluate_gate(
        &synthetic_evidence_with(0.05, Some(-0.06), Some(0.02)),
        None,
    )
    .unwrap();
    assert_eq!(dead["pass"], json!(false));
    assert_eq!(dead["promotion_tier"], json!("rejected"));
    assert!(
        gate_reasons(&dead, "reasons").contains(&"candidate_policy_is_not_conditional".to_owned())
    );
    assert!(gate_reasons(&dead, "milestone_reasons")
        .contains(&"candidate_policy_is_not_conditional".to_owned()));
    assert_eq!(dead["candidate_conditional_advantage"], json!(-0.06));
    assert_eq!(dead["conditional_advantage_lift"], json!(-0.06 - 0.02));

    // Exactly zero advantage is still not a conditional policy.
    let zero = evaluate_gate(&synthetic_evidence_with(0.05, Some(0.0), Some(-0.01)), None).unwrap();
    assert_eq!(zero["pass"], json!(false));
    assert!(
        gate_reasons(&zero, "reasons").contains(&"candidate_policy_is_not_conditional".to_owned())
    );

    // Unmeasured conditionality is fail-closed, never treated as zero.
    let missing = evaluate_gate(&synthetic_evidence_with(0.05, None, None), None).unwrap();
    assert_eq!(missing["pass"], json!(false));
    assert_eq!(missing["candidate_conditional_advantage"], Value::Null);
    assert_eq!(missing["conditional_advantage_lift"], Value::Null);
    let reasons = gate_reasons(&missing, "reasons");
    assert!(reasons.contains(&"candidate_conditional_advantage_evidence_missing".to_owned()));
    assert!(reasons.contains(&"baseline_conditional_advantage_evidence_missing".to_owned()));
    assert!(!reasons.contains(&"candidate_policy_is_not_conditional".to_owned()));

    // Positive, but under the internal margin.
    let thin = evaluate_gate(
        &synthetic_evidence_with(0.05, Some(0.004), Some(-0.02)),
        None,
    )
    .unwrap();
    assert_eq!(thin["pass"], json!(false));
    assert!(gate_reasons(&thin, "reasons")
        .contains(&"conditional_advantage_below_half_point".to_owned()));

    // Beats the baseline by half a point but not by two: internal only.
    let internal =
        evaluate_gate(&synthetic_evidence_with(0.05, Some(0.03), Some(0.02)), None).unwrap();
    assert_eq!(internal["promotion_tier"], json!("internal"));
    assert!(gate_reasons(&internal, "milestone_reasons")
        .contains(&"conditional_advantage_lift_below_two_points".to_owned()));

    // Conditional on both margins: milestone stays reachable.
    let milestone =
        evaluate_gate(&synthetic_evidence_with(0.05, Some(0.06), Some(0.02)), None).unwrap();
    assert_eq!(milestone["promotion_tier"], json!("milestone"));
}

/// Byte-for-byte cross-check against the Python gate: the fixture records
/// what `scripts/test_exact_accept.evaluate_gate` produces for each case and
/// `tests/test_durable_selfplay.py` asserts the same file from that side.
#[test]
fn conditionality_gate_parity_with_python() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/conditionality_gate_parity.json");
    let fixture = read_json(&fixture_path).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty(), "parity fixture has no cases");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let produced = py_dumps(
            &evaluate_gate(&case["evidence"], None).unwrap(),
            PyStyle::Compact,
        )
        .unwrap();
        assert_eq!(
            produced,
            case["expected_gate_compact_json"].as_str().unwrap(),
            "gate output diverged from the Python implementation for case {name}"
        );
    }
}
