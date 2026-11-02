use super::*;

#[test]
fn an_l6_regression_against_the_frozen_reference_is_refused() {
    let manifest = bytes(&manifest());
    let mut document = results(&manifest, rows(clear_lift));
    // L6 is a loss: above frozen v1 0.8973 is worse.
    document["safety"] = json!({"challenger": 0.8990, "reference": 0.8973});
    let report = promotion::decide(&manifest, &bytes(&document)).unwrap();
    assert_eq!(report.verdict, Verdict::Regression);
    assert!(report.safety_regression > 0.0);
    assert!(report
        .reasons
        .iter()
        .any(|item| item.contains("lower is better")));
    // Below the reference is not a regression.
    document["safety"] = json!({"challenger": 0.8960, "reference": 0.8973});
    let report = promotion::decide(&manifest, &bytes(&document)).unwrap();
    assert_eq!(report.verdict, Verdict::PromoteEligible);
}

#[test]
fn holds_failures_and_missing_attempt_accounting_hold_the_gate() {
    let manifest = bytes(&manifest());
    let mut held = results(&manifest, rows(clear_lift));
    held["holds"] = json!(["custody HOLD 2026-10-13"]);
    let report = promotion::decide(&manifest, &bytes(&held)).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);

    let mut failing = rows(clear_lift);
    for row in &mut failing {
        row["failures"] = json!(1);
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, failing))).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);
    assert!((report.failure_rate.unwrap() - 0.1).abs() < 1e-12);

    let mut unaccounted = rows(clear_lift);
    for row in &mut unaccounted {
        row["attempts"] = json!(0);
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, unaccounted))).unwrap();
    assert_eq!(report.verdict, Verdict::Hold);
    assert!(report.failure_rate.is_none());
}

#[test]
fn unknown_outcomes_drop_their_family_and_are_never_imputed() {
    let manifest = bytes(&manifest());
    let mut partial = rows(clear_lift);
    // Five families lose one context each: seven complete families remain, fewer than eight.
    for row in partial.iter_mut().step_by(2).take(5) {
        row["challenger"] = Value::Null;
    }
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, partial))).unwrap();
    assert_eq!(report.contexts_unknown, 5);
    assert_eq!(report.families_complete, FAMILIES - 5);
    assert_eq!(report.families_incomplete.len(), 5);
    assert_eq!(report.verdict, Verdict::Inconclusive);
    assert!(report.lower.is_none() && report.upper.is_none());

    // A declared family with no rows at all is incomplete too.
    let some: Vec<Value> = rows(clear_lift).into_iter().skip(2).collect();
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, some))).unwrap();
    assert_eq!(report.families_incomplete, vec!["fam-01".to_owned()]);
    assert_eq!(report.verdict, Verdict::Inconclusive);
}

#[test]
fn incomplete_families_cap_the_verdict_unless_predeclared() {
    // Eleven families show a clear lift; the twelfth has one unknown context. Dropping it must
    // not let the remaining families promote: selection on missingness.
    let mut one_unknown = rows(clear_lift);
    one_unknown[0]["incumbent"] = Value::Null;
    let strict = bytes(&manifest());
    let report =
        promotion::decide(&strict, &bytes(&results(&strict, one_unknown.clone()))).unwrap();
    assert_eq!(report.max_incomplete_families, 0);
    assert_eq!(report.families_incomplete, vec!["fam-01".to_owned()]);
    assert!(
        report.lower.unwrap() > report.practical_margin,
        "the bounds alone would promote"
    );
    assert_eq!(report.verdict, Verdict::Inconclusive);
    assert!(report
        .reasons
        .iter()
        .any(|item| item.contains("selection on missingness")));
    let refusal = request_promotion(
        &Gate::On,
        &strict,
        &bytes(&results(&strict, one_unknown.clone())),
    );
    assert!(matches!(refusal, Err(Refusal::NotReady { .. })));

    // A lift below the margin is not decided either while a family is missing.
    let mut tiny = rows(|family, _| if family % 2 == 0 { 0.0005 } else { 0.0 });
    tiny[0]["challenger"] = Value::Null;
    let report = promotion::decide(&strict, &bytes(&results(&strict, tiny))).unwrap();
    assert_eq!(report.verdict, Verdict::Inconclusive);

    // A manifest that predeclares one tolerated incomplete family decides the look.
    let mut tolerant = manifest();
    tolerant["max_incomplete_families"] = json!(1);
    let tolerant = bytes(&tolerant);
    let report = promotion::decide(&tolerant, &bytes(&results(&tolerant, one_unknown))).unwrap();
    assert_eq!(report.max_incomplete_families, 1);
    assert_eq!(report.verdict, Verdict::PromoteEligible);
}

#[test]
fn invalid_evidence_is_refused_not_judged() {
    let manifest_value = manifest();
    let manifest = bytes(&manifest_value);
    let base = results(&manifest, rows(clear_lift));
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "outer-oracle",
            Box::new(|r| r["rows"][3]["selection"] = json!("oracle")),
        ),
        (
            "exact manifest bytes",
            Box::new(|r| r["manifest_sha256"] = json!(sha(9))),
        ),
        (
            "not declared",
            Box::new(|r| r["rows"][0]["family_id"] = json!("fam-99")),
        ),
        (
            "repeated",
            Box::new(|r| r["rows"][1]["context_id"] = json!("ctx-01-0")),
        ),
        (
            "more failures than attempts",
            Box::new(|r| r["rows"][0]["failures"] = json!(11)),
        ),
        ("look 3 is not declared", Box::new(|r| r["look"] = json!(3))),
        ("prior looks", Box::new(|r| r["look"] = json!(2))),
        (
            "stopped at look 1",
            Box::new(|r| {
                r["look"] = json!(2);
                r["prior_looks"] = json!([{"index": 1, "verdict": "no_practical_lift"}]);
            }),
        ),
        (
            "results schema",
            Box::new(|r| r["schema"] = json!("jope.other/v1")),
        ),
        ("unknown field", Box::new(|r| r["oracle_rows"] = json!([]))),
    ];
    for (expected, mutate) in cases {
        let mut document = base.clone();
        mutate(&mut document);
        let error = promotion::decide(&manifest, &bytes(&document)).unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
        let refusal = request_promotion(&Gate::On, &manifest, &bytes(&document)).unwrap_err();
        assert!(matches!(
            refusal,
            Refusal::Invalid {
                action: Action::Promote,
                ..
            }
        ));
    }
    // A second look after an inconclusive first look is in schedule.
    let mut second = base.clone();
    second["look"] = json!(2);
    second["prior_looks"] = json!([{"index": 1, "verdict": "inconclusive"}]);
    let report = promotion::decide(&manifest, &bytes(&second)).unwrap();
    assert!((report.alpha_spent - 0.05).abs() < 1e-12);
    assert!((report.alpha_look - 0.04).abs() < 1e-12);
}
