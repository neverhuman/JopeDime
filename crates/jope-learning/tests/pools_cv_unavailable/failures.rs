#[test]
fn missing_component_fit_is_not_silently_replaced_by_in_sample_ridge() {
    let (pools, cf, mut pf) = fixture(12);
    pf.rows.remove(&pools[0].members[0].program_sha256);
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::ComponentFit);
}
#[test]
fn malformed_features_are_typed_shape_failures() {
    let (pools, cf, mut pf) = fixture(12);
    pf.rows.values_mut().next().unwrap().push(0.0);
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::ShapeMismatch);
    // An empty feature map must refuse before dimension arithmetic or allocation.
    pf.rows.clear();
    pf.dim = usize::MAX;
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::ShapeMismatch);
}
#[test]
fn nonfinite_features_never_enter_a_score() {
    let (pools, cf, mut pf) = fixture(12);
    pf.rows.values_mut().next().unwrap()[0] = f64::NAN;
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::NonfiniteSystem);
}
#[test]
fn no_finite_penalty_candidate_is_unavailable_without_a_zero_loss() {
    let (pools, cf, pf) = fixture(12);
    let estimate = expect_unavailable(
        &pools,
        &cf,
        &pf,
        1.0e308,
        CvUnavailableReason::NoFiniteCandidate,
    );
    assert!(estimate.counts().failed_candidates > 0);
    assert!(!estimate.counts().failure_details.is_empty());
}
#[test]
fn duplicated_design_is_supported_when_regularization_makes_it_solvable() {
    let (pools, cf, mut pf) = fixture(12);
    for row in pf.rows.values_mut() {
        row[0] = 1.0;
    }
    let model =
        ContextualSurrogateQHat::fit(&pools.iter().collect::<Vec<_>>(), &cf, &pf, 2, 2.0, 0.1)
            .unwrap();
    assert!(model.inner_cv_loss().unwrap().is_finite());
}
#[test]
fn cut_zero_and_unknown_context_vectors_are_refused_but_target_free_rows_are_accepted() {
    let (pools, mut cf, pf) = fixture(12);
    cf.rows.values_mut().next().unwrap()[0] = 0.75;
    expect_unavailable(
        &pools,
        &cf,
        &pf,
        0.1,
        CvUnavailableReason::UnboundContextProvenance,
    );
    let text = cf
        .rows
        .iter()
        .map(|(id, vector)| serde_json::json!({"id":id,"vector":vector}).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let unbound = Features::from_jsonl(&text).unwrap();
    expect_unavailable(
        &pools,
        &unbound,
        &pf,
        0.1,
        CvUnavailableReason::UnboundContextProvenance,
    );
    let valid = Features::context_id_descriptors(&pools, 1).unwrap();
    assert!(ContextualSurrogateQHat::fit(
        &pools.iter().collect::<Vec<_>>(),
        &valid,
        &pf,
        2,
        2.0,
        0.1
    )
    .is_ok());
}
#[test]
fn held_label_poison_preserves_all_fold_fitted_components_and_prescore_predictions() {
    let (pools, cf, pf) = fixture(18);
    let mut prior = ExternalProgramPrior::default();
    for pool in &pools {
        for member in &pool.members {
            let value = member.q.unwrap();
            prior.parent_stats.insert(
                (pool.parent_id.clone(), member.program_sha256.clone()),
                (value, 1),
            );
            let e = prior
                .tot_stats
                .entry(member.program_sha256.clone())
                .or_default();
            e.0 += value;
            e.1 += 1;
        }
    }
    let before = ContextualSurrogateQHat::fit_with_prior(
        &pools.iter().collect::<Vec<_>>(),
        &cf,
        &pf,
        2,
        2.0,
        0.1,
        Some(&prior),
    )
    .unwrap();
    let fold = before.inner_fold_fits()[0].fold;
    let mut poisoned = pools.clone();
    let mut poisoned_prior = prior.clone();
    for pool in &mut poisoned {
        if fold_of("jope.j9-inner-ridge-folds/v1", &pool.parent_id, 5) == fold {
            for member in &mut pool.members {
                let old = member.q.unwrap();
                member.q = Some(1.0 - old);
                let key = (pool.parent_id.clone(), member.program_sha256.clone());
                poisoned_prior.parent_stats.get_mut(&key).unwrap().0 = 1.0 - old;
                poisoned_prior
                    .tot_stats
                    .get_mut(&member.program_sha256)
                    .unwrap()
                    .0 += 1.0 - 2.0 * old;
            }
        }
    }
    let after = ContextualSurrogateQHat::fit_with_prior(
        &poisoned.iter().collect::<Vec<_>>(),
        &cf,
        &pf,
        2,
        2.0,
        0.1,
        Some(&poisoned_prior),
    )
    .unwrap();
    record_evidence(
        "held-label-poison",
        &serde_json::json!({"held_fold":fold,"before":before.inner_fold_fits(),"after":after.inner_fold_fits()}),
    );
    let a = before
        .inner_fold_fits()
        .iter()
        .find(|f| f.fold == fold)
        .unwrap();
    let b = after
        .inner_fold_fits()
        .iter()
        .find(|f| f.fold == fold)
        .unwrap();
    assert_eq!(a.training_input_sha256, b.training_input_sha256);
    assert_eq!(
        a.state_sha256, b.state_sha256,
        "prior, SoS, neighbours, base and projections"
    );
    assert_eq!(a.held_design_sha256, b.held_design_sha256);
    assert_eq!(
        a.prediction_sha256, b.prediction_sha256,
        "each fixed candidate before held scoring"
    );
    // Active leaky control: other closures fit the poisoned labels and their states must move.
    assert!(before
        .inner_fold_fits()
        .iter()
        .zip(after.inner_fold_fits())
        .any(|(a, b)| a.fold != fold && a.state_sha256 != b.state_sha256));
    assert_ne!(
        before.scores(&pools[0].view()).unwrap(),
        after.scores(&pools[0].view()).unwrap(),
        "full fit negative control"
    );
}
