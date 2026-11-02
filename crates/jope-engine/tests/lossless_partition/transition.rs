use super::*;

#[test]
fn a_rank_count_change_is_an_explicit_transition() {
    let eight = Plan::new(8, 2048, 256).unwrap().record();
    let three = Plan::new(3, 2048, 256).unwrap().record();
    let from_eight = parent(eight.clone(), partition::objective_identity());
    // Exact resume refuses a rank-count change.
    let error = check(Mode::ExactResume, &from_eight, 3, &three, None).unwrap_err();
    assert!(
        error.to_string().contains("exact resume requires"),
        "{error}"
    );
    // Optimizer carry must declare it, exactly.
    let error = check(Mode::NewCorpusTransition, &from_eight, 3, &three, None).unwrap_err();
    assert!(error.to_string().contains("undeclared"), "{error}");
    let declared = json!({"optimizer":"carry","declared_changes":{"execution_partition":three}});
    let record = check(
        Mode::NewCorpusTransition,
        &from_eight,
        3,
        &three,
        Some(&declared),
    )
    .unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "changed");
    assert_eq!(record["execution_partition"]["declared"], three);
    assert_eq!(record["objective"]["continuity"], "identical");
    assert_eq!(record["exact_resume_claim"], false);
    let four = Plan::new(4, 2048, 256).unwrap().record();
    let wrong = json!({"optimizer":"carry","declared_changes":{"execution_partition":four}});
    assert!(check(
        Mode::NewCorpusTransition,
        &from_eight,
        3,
        &three,
        Some(&wrong)
    )
    .is_err());
    // A weight warm start carries nothing and declares nothing.
    let record = check(Mode::WeightWarmStart, &from_eight, 3, &three, None).unwrap();
    assert_eq!(record["optimizer"], "reset");
    assert!(check(
        Mode::WeightWarmStart,
        &from_eight,
        3,
        &three,
        Some(&declared)
    )
    .is_err());
    // The same three-rank partition resumes exactly.
    let from_three = parent(three.clone(), partition::objective_identity());
    let record = check(Mode::ExactResume, &from_three, 3, &three, None).unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "identical");
    // A partition record for another topology than the request is refused.
    assert!(check(Mode::ExactResume, &from_three, 3, &four, None).is_err());
    // v1 → v2 changes the objective's reductions: never an optimizer carry.
    let v1_parent = parent(
        training_partition(8, 256, 2048).unwrap(),
        objective::identity(),
    );
    let error = check(
        Mode::NewCorpusTransition,
        &v1_parent,
        3,
        &three,
        Some(&declared),
    )
    .unwrap_err();
    assert!(error.to_string().contains("same objective"), "{error}");
    let record = check(Mode::WeightWarmStart, &v1_parent, 3, &three, None).unwrap();
    assert_eq!(record["objective"]["continuity"], "changed");
    // The v1 check cannot even describe three ranks.
    let config = config();
    let v1_digest = objective::identity_sha256();
    assert!(check_parent(
        Mode::WeightWarmStart,
        &v1_parent,
        &config,
        &Current {
            corpus_sha256: CORPUS,
            vocabulary_sha256: VOCABULARY,
            world: 3,
            microbatch: 256,
            global_batch: 2048,
            seed: SEED,
            sampler_id: SAMPLER,
            config: &config,
            objective_sha256: &v1_digest,
        },
        None,
    )
    .is_err());
}
