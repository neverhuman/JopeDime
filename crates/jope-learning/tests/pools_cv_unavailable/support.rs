#[test]
fn unattributed_external_prior_is_unavailable() {
    let (pools, cf, pf) = fixture(12);
    let prior = ExternalProgramPrior {
        tot_stats: BTreeMap::from([(pools[0].members[0].program_sha256.clone(), (1.0, 1))]),
        parent_stats: BTreeMap::new(),
    };
    let error = ContextualSurrogateQHat::fit_with_prior(
        &pools.iter().collect::<Vec<_>>(),
        &cf,
        &pf,
        2,
        2.0,
        0.1,
        Some(&prior),
    )
    .err()
    .unwrap();
    assert!(matches!(
        &error.downcast_ref::<CvUnavailable>().unwrap().0,
        CvEstimate::Unavailable {
            reason: CvUnavailableReason::ExternalPriorProvenance,
            ..
        }
    ));
}
#[test]
fn partial_outer_support_is_removed_from_every_comparison_with_reconciled_counts() {
    use jope_learning::pools::evaluate::{evaluate, Config, Inputs};
    use jope_learning::pools::pool::{freeze, FreezeRule, MeasurementRow, Reference};
    let (mut pools, _, pf) = fixture(8);
    for (i, pool) in pools.iter_mut().enumerate() {
        let desired = if i < 6 { 0 } else { i - 5 };
        pool.parent_id = (0..1000)
            .map(|j| format!("outer-{i}-{j}"))
            .find(|p| fold_of("outer-fixture", p, 3) == desired)
            .unwrap();
    }
    let rows: Vec<_> = pools
        .iter()
        .flat_map(|p| {
            p.members.iter().map(|m| MeasurementRow {
                context_id: p.context_id.clone(),
                parent_id: p.parent_id.clone(),
                role: p.role.clone(),
                program_sha256: m.program_sha256.clone(),
                status: m.status,
                q: m.q,
                incumbent_q: p.incumbent_q,
                protocol_sha256: p.protocol_sha256.clone(),
                tokens: m.tokens.clone(),
            })
        })
        .collect();
    let rule = FreezeRule {
        min_pool: 3,
        max_pool: None,
        salt: "fixture".into(),
        min_distinct_measured: 2,
        source_grade: "synthetic_fixture".into(),
    };
    let (pools, manifest) = freeze(&rows, &rule, &"cd".repeat(32)).unwrap();
    let cf = Features::context_id_descriptors(&pools, 1).unwrap();
    let config = Config {
        reference: Reference::PoolFloor,
        budget: 1,
        folds: 3,
        fold_salt: "outer-fixture".into(),
        prior_weight: 2.0,
        knn_k: 2,
        ridge_lambda: 0.1,
        bootstrap_draws: 100,
        bootstrap_seed: 7,
        alpha: 0.05,
    };
    let receipt = evaluate(
        &pools,
        &manifest,
        &Inputs {
            context_features: Some(&cf),
            program_features: Some(&pf),
            external: vec![],
            unavailable: vec![],
        },
        &config,
    )
    .unwrap();
    assert!(receipt
        .surrogate_cv
        .iter()
        .any(|f| f.estimate.loss().is_some()));
    assert!(receipt
        .surrogate_cv
        .iter()
        .any(|f| f.estimate.loss().is_none()));
    assert!(!receipt.summaries.contains_key("contextual_surrogate_qhat"));
    assert!(receipt
        .contexts
        .iter()
        .all(|c| !c.metrics.contains_key("contextual_surrogate_qhat")));
    assert!(receipt.comparisons.is_empty());
    record_evidence("partial-cohort-refusal", &receipt);
    let coverage = &receipt.selector_coverage["contextual_surrogate_qhat"];
    assert_eq!(coverage.attempted, 8);
    assert_eq!(coverage.scored, 0);
    assert_eq!(coverage.failed, 6);
    assert_eq!(coverage.excluded, 2);
    assert_eq!(
        coverage.attempted,
        coverage.scored + coverage.failed + coverage.excluded
    );
    assert!(receipt.summaries.values().all(|s| s.contexts == 8));
    assert!(receipt.shadow_only && !receipt.qualified_claim && !receipt.l6_claim);
}
#[test]
fn measured_zero_loss_is_distinct_from_unavailable() {
    let (mut pools, cf, mut pf) = fixture(12);
    for pool in &mut pools {
        for m in &mut pool.members {
            m.q = Some(0.5);
        }
    }
    for row in pf.rows.values_mut() {
        row[0] = 1.0;
    }
    let model =
        ContextualSurrogateQHat::fit(&pools.iter().collect::<Vec<_>>(), &cf, &pf, 2, 2.0, 0.1)
            .unwrap();
    assert_eq!(model.inner_cv_loss(), Some(0.0));
    record_evidence("measured-zero", model.inner_cv_estimate());
    let json = serde_json::to_value(model.inner_cv_estimate()).unwrap();
    assert_eq!(json["status"], "supported");
    assert_eq!(json["metric"]["f64_bits"], "0000000000000000");
    assert!(json.get("reason").is_none());
}
#[test]
fn wholly_failed_and_single_member_pools_have_no_supported_ranking_loss() {
    let (mut pools, cf, pf) = fixture(12);
    for pool in &mut pools {
        for member in &mut pool.members {
            member.status = Status::ProgramFailed;
            member.q = None;
        }
    }
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::EmptyUsableFolds);
    let (mut pools, cf, pf) = fixture(12);
    for pool in &mut pools {
        pool.members.truncate(1);
    }
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::EmptyUsableFolds);
}
#[test]
fn finite_inputs_that_overflow_normal_equations_are_typed_nonfinite_failure() {
    let (pools, cf, mut pf) = fixture(12);
    for row in pf.rows.values_mut() {
        row[0] = 1.0e308;
    }
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::NonfiniteSystem);
}
#[test]
fn numerically_singular_fit_is_typed_instead_of_silently_dropped() {
    let (pools, cf, mut pf) = fixture(12);
    pf.dim = 2;
    for row in pf.rows.values_mut() {
        *row = vec![1.0e8, 1.0e8];
    }
    // This is a numerical failure of the actual regularized component, independently verified
    // before checking propagation. The ordinary duplicated-design case above remains supported.
    let error = jope_learning::pools::selectors::RidgeProgram::fit(
        &pools.iter().collect::<Vec<_>>(),
        &pf,
        0.1,
    )
    .err()
    .expect("fixture must really fail fitting");
    assert!(format!("{error:#}").contains("not positive definite"));
    expect_unavailable(&pools, &cf, &pf, 0.1, CvUnavailableReason::SingularSystem);
}
