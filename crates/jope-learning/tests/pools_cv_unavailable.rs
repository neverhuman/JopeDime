use jope_learning::pools::evaluate::fold_of;
use jope_learning::pools::pool::{Member, Pool, Status, POOL_SCHEMA};
use jope_learning::pools::selectors::{
    rank_targets, ContextualSurrogateQHat, CvEstimate, CvUnavailable, CvUnavailableReason,
    ExternalProgramPrior, Features, Selector,
};
use std::collections::BTreeMap;

fn fixture(n: usize) -> (Vec<Pool>, Features, Features) {
    let pools: Vec<_> = (0..n)
        .map(|i| Pool {
            schema: POOL_SCHEMA.into(),
            context_id: format!("{i:064x}"),
            parent_id: format!("parent-{i}"),
            role: "train".into(),
            protocol_sha256: "ab".repeat(32),
            incumbent_q: Some(0.0),
            members: (0..3)
                .map(|j| Member {
                    program_sha256: format!("{j:064x}"),
                    status: Status::Complete,
                    q: Some((i % 2 == j % 2) as u8 as f64),
                    tokens: Some(vec![j as u32]),
                })
                .collect(),
        })
        .collect();
    let cf = Features::context_id_descriptors(&pools, 1).unwrap();
    let pf = Features {
        dim: 1,
        rows: BTreeMap::from_iter((0..3).map(|j| (format!("{j:064x}"), vec![j as f64]))),
    };
    (pools, cf, pf)
}

#[test]
fn single_parent_cannot_claim_an_oof_estimate() {
    let (pools, cf, pf) = fixture(1);
    assert!(ContextualSurrogateQHat::fit(&[&pools[0]], &cf, &pf, 2, 2.0, 0.1).is_err());
}

fn record_evidence(name: &str, value: &impl serde::Serialize) {
    if let Some(path) = std::env::var_os("I03_EVIDENCE_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join(format!("{name}.json")),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .unwrap();
    }
}

fn expect_unavailable(
    pools: &[Pool],
    cf: &Features,
    pf: &Features,
    lambda: f64,
    reason: CvUnavailableReason,
) -> CvEstimate {
    let error = match ContextualSurrogateQHat::fit(
        &pools.iter().collect::<Vec<_>>(),
        cf,
        pf,
        2,
        2.0,
        lambda,
    ) {
        Ok(_) => panic!("expected unavailable {reason:?}"),
        Err(error) => error,
    };
    let estimate = error
        .downcast_ref::<CvUnavailable>()
        .expect("typed unavailable")
        .0
        .clone();
    let CvEstimate::Unavailable {
        reason: actual,
        counts,
        ..
    } = &estimate
    else {
        panic!("expected missing metric")
    };
    assert_eq!(*actual, reason, "{error}");
    assert_eq!(
        counts.attempted_units,
        counts.scored_units + counts.failed_units + counts.excluded_units
    );
    let serialized = serde_json::to_value(&estimate).unwrap();
    assert!(serialized.get("metric").is_none());
    assert!(estimate.loss().is_none());
    record_evidence(
        &format!(
            "unavailable-{reason:?}-{}",
            jope_learning::canon::canonical_sha256(&estimate).unwrap()
        ),
        &estimate,
    );
    estimate
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
