use std::collections::{BTreeMap, BTreeSet};

use jope_campaigns::hyperion_gate::{FeatureViewKindV1, GateThresholdsV1};
use jope_campaigns::jope_dlgp_best5::*;
use jope_campaigns::jope_generalized_gp::raw_sha256;

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn full_dataset_digest(id: &str) -> String {
    format!("{id}{}", &digest(id)[16..])
}

fn archive(offset: usize) -> Vec<String> {
    let mut values = (0..20)
        .map(|index| digest(&format!("archive-{}", index + offset)))
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn checkpoint(
    evaluations: u32,
    prefix_prediction_roots: BTreeMap<u32, String>,
    cv7_rmse: Vec<f64>,
    archive: Vec<String>,
) -> BudgetCheckpointV1 {
    BudgetCheckpointV1 {
        evaluations,
        prefix_prediction_roots,
        population_sha256: digest(&format!("population-{evaluations}")),
        selection_sha256: digest(&format!("selection-{evaluations}")),
        receipt_root_sha256: digest(&format!("receipts-{evaluations}")),
        prediction_root_sha256: digest(&format!("predictions-{evaluations}")),
        cv7_rmse,
        archive_top20_semantic_sha256s: archive,
        logical_operations: evaluations as u64,
        cache_hits: 0,
        cpu_seconds: evaluations as f64,
        gpu_seconds: 0.0,
    }
}

fn ladder(continue_to_10k: bool) -> BudgetLadderBundleV1 {
    let roots = BTreeMap::from([(1_000, digest("predictions-1000"))]);
    let one = checkpoint(1_000, roots, vec![1.1; 7], archive(0));
    let roots = BTreeMap::from([
        (1_000, digest("predictions-1000")),
        (2_000, digest("predictions-2000")),
    ]);
    let two = checkpoint(2_000, roots, vec![1.0; 7], archive(0));
    let five_values = if continue_to_10k {
        vec![0.998, 0.998, 0.998, 0.998, 1.0, 1.0, 1.0]
    } else {
        vec![1.01; 7]
    };
    let roots = BTreeMap::from([
        (1_000, digest("predictions-1000")),
        (2_000, digest("predictions-2000")),
        (5_000, digest("predictions-5000")),
    ]);
    let five = checkpoint(5_000, roots, five_values, archive(0));
    let mut checkpoints = vec![one, two.clone(), five.clone()];
    if continue_to_10k {
        let roots = BTreeMap::from([
            (1_000, digest("predictions-1000")),
            (2_000, digest("predictions-2000")),
            (5_000, digest("predictions-5000")),
            (10_000, digest("predictions-10000")),
        ]);
        checkpoints.push(checkpoint(10_000, roots, vec![0.99; 7], archive(0)));
    }
    BudgetLadderBundleV1 {
        schema: BudgetLadderBundleV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("campaign"),
        dataset_sha256: full_dataset_digest(STAGE1_DATASET_IDS[0]),
        master_seed: 101,
        license_track: LicenseTrackV1::ProductionCompatible,
        metric_sha256: digest("metric"),
        evaluator_sha256: digest("evaluator"),
        split_sha256: digest("split"),
        test_csv_read: false,
        checkpoints,
        promotion: BudgetPromotionDecisionV1 {
            source_2k_checkpoint_sha256: two.identity().unwrap(),
            source_5k_checkpoint_sha256: five.identity().unwrap(),
            continue_to_10k,
            improved_folds_at_least_point_one_percent: if continue_to_10k { 4 } else { 0 },
            archive_jaccard_micros: 1_000_000,
            cv7_mean_ratio_micros: if continue_to_10k { 998_857 } else { 1_010_000 },
        },
    }
}

fn license() -> LicenseIdentityV1 {
    LicenseIdentityV1 {
        spdx_expression: "research-only".into(),
        license_text_sha256: digest("license"),
        upstream_source: "https://example.invalid/upstream".into(),
        upstream_revision: "commit-1".into(),
    }
}

fn tabfm() -> FrozenPredictorComponentV1 {
    FrozenPredictorComponentV1::ExternalModel {
        family: ExternalModelFamilyV1::TabFm,
        artifact_sha256: digest("tabfm-model"),
        artifact_bytes: 42,
        inference_protocol_sha256: digest("tabfm-inference"),
        license: license(),
        complexity: 100,
    }
}

#[test]
fn budget_ladder_is_nested_resumable_and_rule_frozen() {
    let bundle = ladder(true);
    bundle.validate().unwrap();
    let replay = bundle.clone();
    verify_prefix_equality(&bundle, &replay, 1_000).unwrap();

    let mut bad_prefix = replay;
    bad_prefix.checkpoints[3]
        .prefix_prediction_roots
        .insert(1_000, digest("changed-prefix"));
    assert!(bad_prefix.validate().is_err());

    let stopped = ladder(false);
    stopped.validate().unwrap();
    let mut illicit_ten_k = stopped;
    illicit_ten_k
        .checkpoints
        .push(bundle.checkpoints[3].clone());
    assert!(illicit_ten_k.validate().is_err());
}

#[test]
fn component_predictors_bind_inference_and_license_tracks() {
    let research = FrozenPredictorV1::ComponentResidualComposite {
        base: tabfm(),
        residual: FrozenPredictorComponentV1::ExternalModel {
            family: ExternalModelFamilyV1::GpControl,
            artifact_sha256: digest("gp-control"),
            artifact_bytes: 1,
            inference_protocol_sha256: digest("gp-control-inference"),
            license: LicenseIdentityV1 {
                spdx_expression: "Apache-2.0".into(),
                ..license()
            },
            complexity: 3,
        },
        eta: 0.5,
        cross_fit_protocol_sha256: digest("cross-fit"),
        license_track: LicenseTrackV1::ResearchOnly,
    };
    research.validate().unwrap();
    assert_eq!(
        research
            .reconstruct(&[vec![1.0, 2.0], vec![2.0, -2.0]])
            .unwrap(),
        vec![2.0, 1.0]
    );
    let decoded: FrozenPredictorV1 =
        serde_json::from_slice(&serde_json::to_vec(&research).unwrap()).unwrap();
    assert_eq!(decoded.semantic_sha256(), research.semantic_sha256());

    let mut foreign = research;
    if let FrozenPredictorV1::ComponentResidualComposite { license_track, .. } = &mut foreign {
        *license_track = LicenseTrackV1::ProductionCompatible;
    }
    assert!(foreign.validate().unwrap_err().contains("research-only"));

    let roots = TrackArtifactNamespacesV1 {
        production_compatible_root: "/artifacts/production".into(),
        research_only_root: "/artifacts/research".into(),
    };
    roots
        .validate_artifact(
            LicenseTrackV1::ProductionCompatible,
            "/artifacts/production/result.json",
        )
        .unwrap();
    assert!(roots
        .validate_artifact(
            LicenseTrackV1::ProductionCompatible,
            "/artifacts/research/result.json",
        )
        .is_err());
}

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
        serde_json::from_str(include_str!("../manifests/input-manifest.v1.json")).unwrap();
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
        "../manifests/dlgp-budget-first/protocol.v1.json"
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
