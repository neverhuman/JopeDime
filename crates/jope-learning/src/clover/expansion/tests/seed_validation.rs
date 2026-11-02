use super::*;

#[test]
fn immutable_fixture_packet_contains_only_a_training_family() {
    let (roster, cut, policy, retrieval) = fixture();
    if let Some(path) = std::env::var_os("CLOVER_GB06_FIXTURE_PACKET") {
        let packet = json!({"schema":"clover.gp-fixture-inputs/v1","qualification":"synthetic_fixture_only",
            "roster":roster,"role_manifest":serde_json::from_slice::<Value>(&roles()).unwrap(),
            "policy":policy,"retrieval":retrieval,
            "cut":{"index":cut.index,"train_x":cut.train_x,"train_y":cut.train_y,
                "hold_x":cut.hold_x,"hold_y":cut.hold_y,"hold_indices":cut.hold_indices}});
        std::fs::write(path, serde_json::to_vec_pretty(&packet).unwrap()).unwrap();
    }
    assert_eq!(roster.contexts.len(), 1);
    assert_eq!(roster.contexts[0].role, "train");
    assert_eq!(policy.snapshot.context_width, 2051);
    roster.verify_roles(&roles()).unwrap();
}

#[test]
fn roster_rejects_validation_sealed_duplicates_and_changed_labels() {
    let (mut roster, mut cut, _, _) = fixture();
    let digest = roster.digest().unwrap();
    roster.verify_roles(&roles()).unwrap();
    assert!(roster.verify_roles(b"changed role manifest").is_err());
    roster.contexts[0].parent_family = "forbidden-parent".into();
    assert!(roster.verify_roles(&roles()).is_err());
    roster.contexts[0].parent_family = "synthetic-linear-parent".into();
    for role in ["val", "validation", "meta_test", "sealed", ""] {
        roster.contexts[0].role = role.into();
        assert!(roster.digest().is_err());
    }
    roster.contexts[0].role = "train".into();
    assert_eq!(roster.digest().unwrap(), digest);
    cut.train_y[0] += 1.;
    assert!(roster
        .bound_context(&roster.contexts[0].context_sha256, &cut)
        .is_err());
    roster.contexts.push(roster.contexts[0].clone());
    assert!(roster.validate().is_err());
}

#[test]
fn policy_bridge_rejects_legacy_actor_bad_likelihood_and_context_mismatch() {
    let (roster, _, mut policy, _) = fixture();
    let c = &roster.contexts[0];
    assert!(policy.pipeline_seeds(c).is_ok());
    policy.snapshot.context_width = 1536;
    assert!(policy.pipeline_seeds(c).is_err());
    policy.snapshot.context_width = 2051;
    for likelihood in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.01] {
        policy.proposals[0].behavior_logprob = likelihood;
        assert!(policy.pipeline_seeds(c).is_err());
    }
    policy.proposals[0].behavior_logprob = -3.;
    policy.encoder_input_sha256 = pin("another-signature");
    assert!(policy.pipeline_seeds(c).is_err());
}

#[test]
fn pipeline_seed_ingestion_keeps_draws_but_deduplicates_fitted_seeds() {
    let (roster, _, mut policy, _) = fixture();
    let mut duplicate = policy.proposals[0].clone();
    duplicate.draw_seed += 1;
    policy.proposals.push(duplicate);
    let file = policy.pipeline_seeds(&roster.contexts[0]).unwrap();
    assert_eq!(policy.proposals.len(), 2);
    file.check().unwrap();
    assert_eq!(file.genomes.len(), 1);
    let readback: seeds::SeedsFile =
        serde_json::from_slice(&serde_json::to_vec(&file).unwrap()).unwrap();
    assert_eq!(readback, file);
    policy.proposals[1].genome = "invalid unvalidated tail".into();
    assert!(policy.pipeline_seeds(&roster.contexts[0]).is_err());
}

#[test]
fn actual_engine_deserializes_current_actor_seeds_without_parent_steering() {
    let (roster, _, policy, _) = fixture();
    let independent = jope_parallel::gp::metadata_initial_population(20260907).unwrap();
    let prepared = pin("native-prepared-json");
    let csv = pin("native-training-csv");
    let json = policy
        .engine_population(&roster.contexts[0], &prepared, &csv, &independent)
        .unwrap();
    let population: jope_parallel::gp::SeedPopulation = serde_json::from_value(json).unwrap();
    assert_eq!(population.candidates.len(), 80);
    assert_eq!(population.training_sha256, csv);
    assert_eq!(
        population.checkpoint_sha256,
        policy.snapshot.checkpoint_sha256
    );
    assert_eq!(population.candidates[0].origin, "neural_policy");
    assert!(population
        .candidates
        .iter()
        .all(|p| p.policy_logprob.is_none()));
    assert!(policy
        .engine_population(&roster.contexts[0], &prepared, &csv, &[])
        .is_err());
}

#[test]
fn retrieval_ancestry_is_train_only() {
    let (roster, _, _, mut retrieval) = fixture();
    assert!(retrieval
        .pipeline_seeds(&roster, &roster.contexts[0].dataset_sha256)
        .is_ok());
    retrieval.seeds[0].source_context_sha256 = pin("validation-context");
    assert!(retrieval
        .pipeline_seeds(&roster, &roster.contexts[0].dataset_sha256)
        .is_err());
}

#[test]
fn new_string_with_same_training_probe_behavior_is_not_behavioral_novelty() {
    let catalog = vec![search::BOOTSTRAP_GENOME.to_owned()];
    let probe = pin("legal-training-probe");
    let reference = novelty(
        &catalog[0],
        &catalog,
        Some((&probe, &[0., 1.])),
        &BTreeSet::new(),
    )
    .unwrap();
    let behaviors = BTreeSet::from([reference["behavior_sha256"].as_str().unwrap().to_owned()]);
    let result = novelty(
        "(pipe X (standardize append=false) (ridge))",
        &catalog,
        Some((&probe, &[-0., 1.])),
        &behaviors,
    )
    .unwrap();
    assert_eq!(result["syntactically_novel"], true);
    assert_eq!(result["novel_on_training_probe"], false);
    let missing = novelty(
        "(pipe X (standardize append=false) (ridge))",
        &catalog,
        None,
        &behaviors,
    )
    .unwrap();
    assert!(missing["novel_on_training_probe"].is_null());
    assert_eq!(missing["qualification"], "unqualified");
    assert!(novelty(
        &catalog[0],
        &catalog,
        Some((&probe, &[f64::NAN])),
        &behaviors
    )
    .is_err());
}
