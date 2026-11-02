#[test]
fn single_parent_cannot_claim_an_oof_estimate() {
    let (pools, cf, pf) = fixture(1);
    assert!(ContextualSurrogateQHat::fit(&[&pools[0]], &cf, &pf, 2, 2.0, 0.1).is_err());
}
#[test]
fn supported_oof_has_finite_predictions_and_independently_recomputed_loss() {
    let (pools, cf, pf) = fixture(18);
    let model =
        ContextualSurrogateQHat::fit(&pools.iter().collect::<Vec<_>>(), &cf, &pf, 2, 2.0, 0.1)
            .unwrap();
    record_evidence(
        "supported-oof",
        &serde_json::json!({"estimate":model.inner_cv_estimate(),"folds":model.inner_fold_fits(),"multipliers":model.selected_multipliers()}),
    );
    let counts = model.inner_cv_estimate().counts();
    assert_eq!(counts.parents, 18);
    assert_eq!(counts.attempted_units, 54);
    assert_eq!(counts.scored_units, 54);
    assert_eq!(counts.failed_units + counts.excluded_units, 0);
    let mut squared = 0.0;
    let mut scored = 0;
    for trace in model.inner_fold_fits() {
        assert!(!trace.held_parents.is_empty());
        assert!(trace
            .held_parents
            .iter()
            .all(|p| !trace.training_parents.contains(p)));
        assert_eq!(trace.training_input_sha256.len(), 64);
        assert_eq!(trace.training_features_sha256.len(), 64);
        assert_eq!(trace.split_sha256.len(), 64);
        let held: Vec<_> = pools
            .iter()
            .filter(|p| trace.held_parents.contains(&p.parent_id))
            .collect();
        for (pool, bits) in held.iter().zip(&trace.selected_prediction_bits) {
            let pred: Vec<_> = bits
                .iter()
                .map(|b| f64::from_bits(u64::from_str_radix(b, 16).unwrap()))
                .collect();
            assert_eq!(pred.len(), pool.members.len());
            assert!(pred.iter().all(|p| p.is_finite()));
            let target = rank_targets(pool);
            let pm = pred.iter().sum::<f64>() / pred.len() as f64;
            let tm = target.iter().sum::<f64>() / target.len() as f64;
            for (p, t) in pred.iter().zip(target) {
                squared += ((p - pm) - (t - tm)).powi(2);
                scored += 1;
            }
        }
    }
    assert_eq!(scored, counts.scored_units);
    assert!((squared / scored as f64 - model.inner_cv_loss().unwrap()).abs() < 1e-14);
    assert!(model
        .scores(&pools[0].view())
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}
#[test]
fn colliding_parent_folds_are_unavailable() {
    let (mut pools, cf, pf) = fixture(6);
    let parents: Vec<_> = (0..1000)
        .map(|i| format!("collision-{i}"))
        .filter(|p| fold_of("jope.j9-inner-ridge-folds/v1", p, 5) == 0)
        .take(6)
        .collect();
    for (pool, parent) in pools.iter_mut().zip(parents) {
        pool.parent_id = parent;
    }
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::EmptyUsableFolds);
}
#[test]
fn empty_member_folds_are_unavailable() {
    let (mut pools, cf, pf) = fixture(6);
    for pool in &mut pools {
        pool.members.clear();
    }
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::EmptyUsableFolds);
}
