#[test]
fn hyperion_development_promotion_requires_frontier_and_robust_wins() {
    let mut evidence = HyperionDevelopmentEvidenceV1 {
        schema: HyperionDevelopmentEvidenceV1::SCHEMA.into(),
        campaign_id: BUDGET_CAMPAIGN_ID.into(),
        seed_evidence: DEVELOPMENT_DATASET_IDS
            .iter()
            .flat_map(|dataset| {
                SEARCH_SEEDS
                    .into_iter()
                    .map(move |seed| HyperionDevelopmentSeedEvidenceV1 {
                        dataset_id: (*dataset).into(),
                        seed,
                        raw_rmse: 1.02,
                        gated_rmse: 1.0,
                        comparator_rmse: 1.01,
                        selected_view: FeatureViewKindV1::HyperionCell,
                        raw_receipt_sha256: digest("dev-raw"),
                        gated_receipt_sha256: digest("dev-gated"),
                    })
            })
            .collect(),
        protected_screen_trials: 10,
        protected_screen_wins: 10,
        expanded_frontier_complete: true,
        existing_robust_wins: 3,
        retained_existing_robust_wins: 3,
        robust_wins: 22,
        median_log_improvement: (1.02_f64).ln(),
        passed: true,
    };
    evidence.validate().unwrap();
    evidence.seed_evidence[0].gated_rmse = 2.0;
    assert!(evidence.validate().is_err());
}

#[test]
fn hyperion_confirmation_is_globally_bound_and_deterministic() {
    let policy = hyperion_policy();
    let policy_sha256 = policy.identity().unwrap();
    let selections = HyperionSelectionInventoryV1 {
        schema: HyperionSelectionInventoryV1::SCHEMA.into(),
        policy_freeze_sha256: policy_sha256.clone(),
        selections: CONFIRMATION_DATASET_IDS
            .iter()
            .flat_map(|dataset| {
                CONFIRMATION_SEEDS
                    .into_iter()
                    .map(move |seed| HyperionSeedSelectionV1 {
                        dataset_id: (*dataset).into(),
                        dataset_sha256: full_dataset_digest(dataset),
                        seed,
                        raw_control_commit_sha256: digest("raw-control"),
                        gate_decision_sha256: digest("gate"),
                        cache_receipt_sha256: digest("cache"),
                        selected_view: FeatureViewKindV1::Raw,
                        selected_predictor_sha256: digest("raw-predictor"),
                        raw_fallback_predictor_sha256: digest("raw-predictor"),
                        fallback_reason: Some("fixture raw fallback".into()),
                    })
            })
            .collect(),
    };
    selections.validate(&policy).unwrap();
    let selections_sha256 = selections.identity(&policy).unwrap();
    let ratio = 1.2_f64;
    let datasets = CONFIRMATION_DATASET_IDS
        .iter()
        .map(|dataset| HyperionConfirmationDatasetV1 {
            dataset_id: (*dataset).into(),
            dataset_sha256: full_dataset_digest(dataset),
            raw_robust_win: false,
            gated_robust_win: true,
            seeds: CONFIRMATION_SEEDS
                .into_iter()
                .map(|seed| HyperionSeedConfirmationV1 {
                    seed,
                    raw_rmse: ratio,
                    gated_rmse: 1.0,
                    raw_latency_seconds: 1.0,
                    gated_latency_seconds: 1.0,
                    gated_validity: 1.0,
                    raw_receipt_sha256: digest("raw-receipt"),
                    gated_receipt_sha256: digest("gated-receipt"),
                    raw_fallback: true,
                })
                .collect(),
            paired_delta: ratio.ln(),
            sealed_finalizer_receipt_sha256: digest("sealed-finalizer"),
            test_access_ordinal: 1,
        })
        .collect::<Vec<_>>();
    let delta = ratio.ln();
    let report = HyperionConfirmationReportV1 {
        schema: HyperionConfirmationReportV1::SCHEMA.into(),
        policy_freeze_sha256: policy_sha256,
        selection_inventory_sha256: selections_sha256,
        datasets,
        raw_robust_wins: 0,
        gated_robust_wins: 20,
        positive_delta_datasets: 20,
        median_delta: delta,
        bootstrap_lower_bound: delta,
        aggregate_validity: 1.0,
        claim_increased_win_rate: true,
        complete_immutable_receipts: true,
        labels_reached_hyperion: false,
        labels_reached_prediction_fit: false,
    };
    report.validate(&policy, &selections).unwrap();

    let mut leaked = report.clone();
    leaked.labels_reached_hyperion = true;
    assert!(leaked.validate(&policy, &selections).is_err());
}

#[test]
fn hyperion_cache_registry_is_exactly_the_frozen_42_dataset_projection() {
    assert_eq!(HYPERION_TOTAL_ENCODER_EXECUTIONS, 546);
    let ids = DEVELOPMENT_DATASET_IDS
        .iter()
        .chain(CONFIRMATION_DATASET_IDS.iter())
        .map(|id| (*id).to_owned())
        .collect::<Vec<_>>();
    validate_hyperion_campaign_registry_ids(&ids).unwrap();
    let mut altered = ids;
    altered.swap(0, 1);
    assert!(validate_hyperion_campaign_registry_ids(&altered).is_err());
}

#[test]
fn sealed_hyperion_stage_exposes_only_dummy_targets_before_finalizer_labels() {
    let dataset_sha256 = digest("sealed-dataset");
    let test_sha256 = digest("sealed-test");
    let stage = stage_hyperion_sealed_features(
        &dataset_sha256,
        &test_sha256,
        b"feature_a,feature_b,target\n1.0,2.0,99.0\n3.0,4.0,101.0\n",
        2,
    )
    .unwrap();
    assert_eq!(stage.test_access_ordinal, 1);
    assert_eq!(
        stage.hyperion_dummy_target_rows().unwrap(),
        vec![vec![1.0, 2.0, 0.0], vec![3.0, 4.0, 0.0]]
    );
    assert_eq!(stage.finalizer_labels().unwrap(), &[99.0, 101.0]);
    assert!(choose_predeclared_raw_fallback(false, true));
    assert!(choose_predeclared_raw_fallback(true, false));
    assert!(!choose_predeclared_raw_fallback(true, true));
    let frozen_bytes = b"full-train-sidecars";
    let frozen_sha256 = digest("full-train-sidecars");
    assert_eq!(
        verify_full_train_replay(&frozen_sha256, frozen_bytes).unwrap(),
        frozen_sha256
    );
    assert!(verify_full_train_replay(&frozen_sha256, b"changed").is_err());

    let mut altered = stage.clone();
    altered.test_access_ordinal = 2;
    assert!(altered.validate().is_err());
    assert!(stage_hyperion_sealed_features(
        &dataset_sha256,
        &test_sha256,
        b"1.0,2.0,3.0\nmalformed\n",
        2,
    )
    .is_err());
}
