use super::*;

#[test]
fn hybrid_receipts_require_every_matched_dimension() {
    let baseline = HybridMatchedRunV1 {
        dataset_sha256: digest("dataset"),
        context_sha256: digest("context"),
        seed: 7,
        slot: 3,
        evaluator_sha256: digest("evaluator"),
        ancestry_sha256: digest("ancestry"),
        compute_units: 100,
        score: 0.4,
    };
    let mut hybrid = baseline.clone();
    hybrid.score = 0.6;
    let receipt = HybridContributionReceiptV1::new(baseline.clone(), hybrid.clone()).unwrap();
    assert_eq!(
        receipt.incremental_gain.to_bits(),
        (0.6_f64 - 0.4).to_bits()
    );
    for mutation in 0..7 {
        let mut mismatched = hybrid.clone();
        match mutation {
            0 => mismatched.dataset_sha256 = digest("other-dataset"),
            1 => mismatched.context_sha256 = digest("other-context"),
            2 => mismatched.seed += 1,
            3 => mismatched.slot += 1,
            4 => mismatched.evaluator_sha256 = digest("other-evaluator"),
            5 => mismatched.ancestry_sha256 = digest("other-ancestry"),
            6 => mismatched.compute_units += 1,
            _ => unreachable!(),
        }
        assert!(HybridContributionReceiptV1::new(baseline.clone(), mismatched).is_err());
    }
}

#[test]
fn equal_evidence_is_required_for_every_comparison() {
    let plan = candidate_plan(0);
    let contract = contract(&plan);
    let mut evidence = complete_evidence(&contract, &plan);
    evidence[59]
        .evaluations
        .insert(1, evaluation(&contract, &plan.candidates[59], 1));
    assert!(PromotionScheduleV1::build(&plan, &evidence, 5)
        .unwrap_err()
        .contains("expected"));
    let complete: BTreeSet<_> = (0..5).collect();
    assert_eq!(complete.len(), 5);
}
