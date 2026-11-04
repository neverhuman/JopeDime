use super::*;

#[test]
fn promotion_is_refused_by_default_before_the_evidence_is_read() {
    let (manifest, results) = eligible();
    for (m, r) in [(manifest, results), (b"{".to_vec(), Vec::new())] {
        let refusal = request_promotion(&Gate::from_env(), &m, &r).unwrap_err();
        assert_eq!(
            refusal,
            Refusal::GateOff {
                action: Action::Promote,
                gate: Gate::Off
            }
        );
    }
    let refusal = request_promotion(&Gate::Unrecognized("yes".to_owned()), b"{", b"{").unwrap_err();
    assert!(matches!(
        refusal,
        Refusal::GateOff {
            action: Action::Promote,
            ..
        }
    ));
}

#[test]
fn an_eligible_challenger_is_still_refused_for_the_pending_g36_ruling() {
    let (manifest, results) = eligible();
    let report = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::PromoteEligible,
        "{:?}",
        report.reasons
    );
    assert!(report.lower.unwrap() > report.practical_margin);
    assert_eq!(report.families_complete, FAMILIES);
    assert_eq!((report.wins, report.ties, report.losses), (FAMILIES, 0, 0));
    assert!(!report.promoted);
    assert_eq!(report.generalization_claim, promotion::GENERALIZATION_CLAIM);
    assert_eq!(report.manifest_sha256, sha256_hex(&manifest));
    assert_eq!(report.results_sha256, sha256_hex(&results));

    let refusal = request_promotion(&Gate::On, &manifest, &results).unwrap_err();
    assert_eq!(
        refusal,
        Refusal::RulingPending {
            action: Action::Promote
        }
    );
    assert!(refusal.to_string().contains("G36"));
    assert!(refusal
        .to_string()
        .contains("this build has no promote path"));
}

#[test]
fn an_inconclusive_arena_is_a_verdict_not_an_error() {
    let manifest = bytes(&manifest());
    let noisy = rows(|family, _| if family % 2 == 0 { 0.03 } else { -0.02 });
    let results = bytes(&results(&manifest, noisy));
    let report = promotion::decide(&manifest, &results).unwrap();
    assert_eq!(
        report.verdict,
        Verdict::Inconclusive,
        "{:?}",
        report.reasons
    );
    assert!(report.lower.unwrap() <= report.practical_margin);
    assert!(report.upper.unwrap() >= report.practical_margin);
    let refusal = request_promotion(&Gate::On, &manifest, &results).unwrap_err();
    match refusal {
        Refusal::NotReady { reasons, .. } => assert_eq!(reasons[0], "verdict inconclusive"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_lift_below_the_practical_margin_is_no_practical_lift() {
    let manifest = bytes(&manifest());
    let tiny = rows(|family, _| if family % 2 == 0 { 0.0005 } else { 0.0 });
    let report = promotion::decide(&manifest, &bytes(&results(&manifest, tiny))).unwrap();
    assert!(
        report.lower.unwrap() > 0.0,
        "a positive lower bound alone is not enough"
    );
    assert_eq!(
        report.verdict,
        Verdict::NoPracticalLift,
        "{:?}",
        report.reasons
    );
}

#[test]
fn the_effect_direction_follows_the_metric() {
    let mut higher = manifest();
    higher["metric"]["lower_is_better"] = json!(false);
    let higher = bytes(&higher);
    let report = promotion::decide(&higher, &bytes(&results(&higher, rows(clear_lift)))).unwrap();
    // The challenger's metric is lower; on a higher-is-better metric that is a loss.
    assert_eq!((report.wins, report.losses), (0, FAMILIES));
    assert_eq!(report.verdict, Verdict::NoPracticalLift);
}
