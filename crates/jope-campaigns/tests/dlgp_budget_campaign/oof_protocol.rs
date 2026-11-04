fn fold(index: u32) -> OuterFoldBaseReceiptV1 {
    let train = vec![format!("row-{index}-a"), format!("row-{index}-b")];
    let validation = vec![format!("row-{index}-v")];
    let groups = BTreeMap::from([
        (train[0].clone(), format!("group-{index}-a")),
        (train[1].clone(), format!("group-{index}-b")),
        (validation[0].clone(), format!("group-{index}-v")),
    ]);
    OuterFoldBaseReceiptV1 {
        outer_fold: index,
        outer_training_row_ids: train,
        outer_validation_row_ids: validation,
        row_atomic_groups: groups,
        component: tabfm(),
        inner_oof_prediction_sha256: digest(&format!("inner-{index}")),
        outer_validation_prediction_sha256: digest(&format!("outer-{index}")),
    }
}

#[test]
fn nested_oof_rejects_leakage_and_preselection_full_train_predictions() {
    let mut bundle = NestedOofBundleV1 {
        schema: NestedOofBundleV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("campaign"),
        dataset_sha256: digest("dataset"),
        license_track: LicenseTrackV1::ResearchOnly,
        cross_fit_protocol_sha256: digest("cross-fit"),
        folds: (0..7).map(fold).collect(),
        selection_freeze_sha256: None,
        full_train_base_prediction_sha256: None,
    };
    bundle.validate().unwrap();
    bundle.full_train_base_prediction_sha256 = Some(digest("premature-full-train"));
    assert!(bundle.validate().unwrap_err().contains("selection freeze"));
    bundle.full_train_base_prediction_sha256 = None;
    let first = &mut bundle.folds[0];
    let validation_group = first.row_atomic_groups[&first.outer_validation_row_ids[0]].clone();
    first
        .row_atomic_groups
        .insert(first.outer_training_row_ids[0].clone(), validation_group);
    assert!(bundle.validate().unwrap_err().contains("atomic group"));
}

#[test]
fn campaign_cohorts_are_exact_and_search_json_cannot_carry_test_material() {
    let input: serde_json::Value =
        serde_json::from_str(include_str!("../../manifests/input-manifest.v1.json")).unwrap();
    let by_id = input["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["dataset_hash"].as_str().unwrap(), row))
        .collect::<BTreeMap<_, _>>();
    let bindings = |ids: &[&str]| {
        ids.iter()
            .map(|id| SearchDatasetBindingV1 {
                dataset_id: (*id).into(),
                dataset_sha256: full_dataset_digest(id),
                train_sha256: by_id[id]["train_sha256"].as_str().unwrap().into(),
                metadata_sha256: by_id[id]["meta_sha256"].as_str().unwrap().into(),
            })
            .collect::<Vec<_>>()
    };
    let campaign = BudgetFirstCampaignFreezeV1 {
        schema: BudgetFirstCampaignFreezeV1::SCHEMA.into(),
        campaign_id: BUDGET_CAMPAIGN_ID.into(),
        development: bindings(&DEVELOPMENT_DATASET_IDS),
        confirmation: bindings(&CONFIRMATION_DATASET_IDS),
        comparator_snapshot_sha256: digest("comparators"),
        code_revision: "frozen-revision".into(),
        source_tree_sha256: digest("source-tree"),
        search_binary_sha256: digest("binary"),
        evaluator_sha256: digest("evaluator"),
        proposal_checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.into(),
        proposal_inference_protocol_sha256: digest("proposal-inference"),
        metric_sha256: digest("rmse"),
        budget_rungs: BUDGET_RUNGS.to_vec(),
        development_seeds: SEARCH_SEEDS.to_vec(),
        confirmation_seeds: CONFIRMATION_SEEDS.to_vec(),
        test_access_policy: TestAccessPolicyV1::SearchTrainMetadataOnlySealedFinalizerOwnsTest,
        license_tracks: BTreeSet::from([
            LicenseTrackV1::ProductionCompatible,
            LicenseTrackV1::ResearchOnly,
        ]),
    };
    campaign.validate().unwrap();
    let serialized = serde_json::to_value(&campaign).unwrap();
    validate_search_artifact_json(&serialized).unwrap();

    let leaked = serde_json::json!({
        "test_csv_read": false,
        "test_sha256": digest("sealed-test"),
    });
    assert!(validate_search_artifact_json(&leaked).is_err());
    let read = serde_json::json!({"test_csv_read": true});
    assert!(validate_search_artifact_json(&read).is_err());
}

#[test]
fn robust_win_is_strict_for_both_seeds_and_zero_target_is_unbeatable() {
    let mut score = SealedDatasetTrackScoreV1 {
        dataset_sha256: digest("dataset"),
        license_track: LicenseTrackV1::ProductionCompatible,
        comparator_rmse: 1.0,
        seed_rmse: BTreeMap::from([(101, 0.9), (202, 0.99)]),
    };
    assert!(score.robust_win(SEARCH_SEEDS).unwrap());
    score.seed_rmse.insert(202, 1.0);
    assert!(!score.robust_win(SEARCH_SEEDS).unwrap());
    score.comparator_rmse = 0.0;
    assert!(!score.robust_win(SEARCH_SEEDS).unwrap());
}

#[test]
fn checked_in_protocol_matches_compiled_cohorts_and_budget() {
    let protocol: serde_json::Value = serde_json::from_str(include_str!(
        "../../manifests/dlgp-budget-first/protocol.v1.json"
    ))
    .unwrap();
    let strings = |path: &serde_json::Value| {
        path.as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        strings(&protocol["cohorts"]["development"]),
        DEVELOPMENT_DATASET_IDS.map(str::to_owned)
    );
    assert_eq!(
        strings(&protocol["cohorts"]["confirmation"]),
        CONFIRMATION_DATASET_IDS.map(str::to_owned)
    );
    assert_eq!(
        protocol["budget_ladder"]["rungs"],
        serde_json::json!(BUDGET_RUNGS)
    );
    assert_eq!(protocol["best5_screen"]["jobs"], 24);
    assert_eq!(
        protocol["round2_evidence"]["payload_sha256"],
        "62d250dd6d9600b526ec9513eaa44bfd01b007b060feaf6622d02daec4cd92f1"
    );
}

fn hyperion_policy() -> HyperionPolicyFreezeV1 {
    let hash = |label| digest(label);
    HyperionPolicyFreezeV1 {
        schema: HyperionPolicyFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: hash("campaign"),
        registry_sha256: hash("registry"),
        coverage_sha256: hash("coverage"),
        encoder_identity_sha256: hash("encoder"),
        encoder_runtime_authority_sha256: hash("runtime"),
        raw_protocol_sha256: hash("raw-protocol"),
        development_evidence_sha256: hash("development"),
        comparator_snapshot_sha256: hash("comparators"),
        finalizer_binary_sha256: hash("finalizer"),
        confirmation_dataset_ids: CONFIRMATION_DATASET_IDS
            .iter()
            .map(|id| (*id).into())
            .collect(),
        confirmation_seeds: CONFIRMATION_SEEDS.to_vec(),
        gate_thresholds: GateThresholdsV1::default(),
        decision_rule: HyperionDecisionRuleV1::default(),
    }
}
