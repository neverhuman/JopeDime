use std::collections::BTreeMap;
use std::fs;

use jope_dime_regressor::hyperion_gate::*;
use jope_dime_regressor::jope_generalized_gp::{EvaluatorFreezeV1, FrozenArtifactV1};
use jope_dime_regressor::search_v3::canonical_sha256;
use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn fixture_folds(
    count: usize,
    arms: &[FeatureViewKindV1],
    errors: impl Fn(usize, FeatureViewKindV1) -> f64,
) -> Vec<PairedFoldEvidenceV1> {
    (0..count)
        .map(|fold| {
            let actual = (0..8).map(|value| value as f64).collect::<Vec<_>>();
            let rows = actual.len();
            let raw_predictions = actual.iter().map(|value| value + 10.0).collect();
            let arm_predictions = arms
                .iter()
                .copied()
                .map(|arm| {
                    (
                        arm,
                        actual
                            .iter()
                            .map(|value| value + errors(fold, arm))
                            .collect(),
                    )
                })
                .collect();
            PairedFoldEvidenceV1 {
                fold_index: fold as u32,
                row_ids: (0..rows).map(|row| format!("row-{fold}-{row}")).collect(),
                atomic_groups: (0..rows).map(|row| format!("group-{fold}-{row}")).collect(),
                actual,
                raw_predictions,
                arm_predictions,
            }
        })
        .collect()
}

#[test]
fn exact_feature_layouts_and_fold_train_preprocessing_are_enforced() {
    let raw_width = 3;
    assert_eq!(
        JopeFeatureViewV1::new(FeatureViewKindV1::Raw, raw_width)
            .unwrap()
            .output_features,
        3
    );
    assert_eq!(
        JopeFeatureViewV1::new(FeatureViewKindV1::HyperionCell, raw_width)
            .unwrap()
            .output_features,
        6
    );
    assert_eq!(
        JopeFeatureViewV1::new(FeatureViewKindV1::HyperionRow128, raw_width)
            .unwrap()
            .output_features,
        131
    );
    assert_eq!(
        JopeFeatureViewV1::new(FeatureViewKindV1::HyperionBoth, raw_width)
            .unwrap()
            .output_features,
        134
    );

    let fit_cell = vec![vec![0.0, 10.0, 5.0], vec![2.0, 20.0, 7.0]];
    let fit_row = vec![vec![0.0; 128], vec![2.0; 128]];
    let fitted = FittedFeatureBlocksV1::fit(&fit_cell, &fit_row, 3).unwrap();
    let hold_cell = fitted.transform_cell(&[vec![1.0, 1000.0, 6.0]]).unwrap();
    let hold_row = fitted.transform_row(&[vec![3.0; 128]]).unwrap();
    assert_eq!(hold_cell[0], vec![0.5, 1.0, 0.5]);
    assert!(hold_row[0].iter().all(|value| (*value - 2.0).abs() < 1e-12));

    let raw = vec![vec![1.0, 2.0, 3.0]];
    let both = assemble_feature_view(FeatureViewKindV1::HyperionBoth, &raw, &hold_cell, &hold_row)
        .unwrap();
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].len(), 2 * raw_width + 128);
}

#[test]
fn cell_is_not_vetoed_by_width_while_dense_row_views_remain_guarded() {
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::Raw, 570, 8).is_ok());
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::HyperionCell, 3, 32).is_ok());
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::HyperionRow128, 3, 32).is_err());
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::HyperionCell, 570, 100).is_ok());
    assert!(
        validate_feature_view_feasibility(FeatureViewKindV1::HyperionBoth, 1_040, 127).is_err()
    );
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::HyperionBoth, 1_040, 128).is_ok());
    assert!(validate_feature_view_feasibility(FeatureViewKindV1::HyperionCell, 3, 31).is_err());
}

#[test]
fn row_ineligibility_does_not_suppress_cell_screening() {
    let groups = (0..120)
        .map(|row| format!("group-{row}"))
        .collect::<Vec<_>>();
    let cv5 = build_grouped_fold_plan(GateStageV1::Cv5Screen, &groups).unwrap();
    let cv7 = build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &groups).unwrap();
    let (eligible, excluded) =
        independently_feasible_feature_views(&cv5, &cv7, &FeatureViewKindV1::EXPANDED, 570);
    assert_eq!(eligible, vec![FeatureViewKindV1::HyperionCell]);
    assert!(excluded.contains_key(&FeatureViewKindV1::HyperionRow128));
    assert!(excluded.contains_key(&FeatureViewKindV1::HyperionBoth));

    let evidence = fixture_folds(5, &eligible, |_, _| 8.0);
    let (results, advanced) =
        screen_cv5_arms(&evidence, &eligible, &GateThresholdsV1::default()).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(advanced, Some(FeatureViewKindV1::HyperionCell));
}

#[test]
fn stable_gain_advances_one_arm_and_confirms_on_cv7() {
    let cv5 = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |_, arm| match arm {
        FeatureViewKindV1::HyperionCell => 8.0,
        FeatureViewKindV1::HyperionRow128 => 9.8,
        FeatureViewKindV1::HyperionBoth => 11.0,
        FeatureViewKindV1::Raw => unreachable!(),
    });
    let (results, advanced) = screen_cv5(&cv5, &GateThresholdsV1::default()).unwrap();
    assert_eq!(advanced, Some(FeatureViewKindV1::HyperionCell));
    assert_eq!(results.iter().filter(|value| value.passed).count(), 1);

    let cv7 = fixture_folds(7, &[FeatureViewKindV1::HyperionCell], |_, _| 8.0);
    let confirmation = confirm_cv7(
        &cv7,
        FeatureViewKindV1::HyperionCell,
        &GateThresholdsV1::default(),
    )
    .unwrap();
    assert!(confirmation.passed);
    assert_eq!(confirmation.fold_wins, 7);
    assert!(confirmation.bootstrap_lower_bound_log_improvement > 0.0);
}

#[test]
fn noisy_spike_inconsistent_arm_and_catastrophic_fold_fail_closed() {
    let one_spike = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |fold, arm| {
        if arm == FeatureViewKindV1::HyperionCell && fold == 0 {
            1.0
        } else {
            10.1
        }
    });
    assert_eq!(
        screen_cv5(&one_spike, &GateThresholdsV1::default())
            .unwrap()
            .1,
        None
    );

    let inconsistent = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |fold, arm| {
        let winner = FeatureViewKindV1::EXPANDED[fold % 3];
        if arm == winner {
            8.0
        } else {
            8.5
        }
    });
    assert_eq!(
        screen_cv5(&inconsistent, &GateThresholdsV1::default())
            .unwrap()
            .1,
        None
    );

    let small_gain = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |_, _| 9.96);
    assert_eq!(
        screen_cv5(&small_gain, &GateThresholdsV1::default())
            .unwrap()
            .1,
        None
    );

    let catastrophic = fixture_folds(7, &[FeatureViewKindV1::HyperionCell], |fold, _| {
        if fold == 6 {
            13.0
        } else {
            7.0
        }
    });
    let confirmation = confirm_cv7(
        &catastrophic,
        FeatureViewKindV1::HyperionCell,
        &GateThresholdsV1::default(),
    )
    .unwrap();
    assert!(!confirmation.passed);
    assert!(confirmation.worst_fold_regression > 0.02);
}

#[test]
fn cv5_success_is_provisional_and_cannot_substitute_for_cv7() {
    let cv5 = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |_, arm| {
        if arm == FeatureViewKindV1::HyperionCell {
            8.0
        } else {
            11.0
        }
    });
    let (_, advanced) = screen_cv5(&cv5, &GateThresholdsV1::default()).unwrap();
    assert_eq!(advanced, Some(FeatureViewKindV1::HyperionCell));
    assert!(confirm_cv7(
        &cv5,
        FeatureViewKindV1::HyperionCell,
        &GateThresholdsV1::default()
    )
    .is_err());
}

#[test]
fn grouped_plans_are_distinct_deterministic_and_atomic() {
    let groups = (0..70)
        .map(|row| format!("group-{}", row / 2))
        .collect::<Vec<_>>();
    let cv5 = build_grouped_fold_plan(GateStageV1::Cv5Screen, &groups).unwrap();
    let replay = build_grouped_fold_plan(GateStageV1::Cv5Screen, &groups).unwrap();
    let cv7 = build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &groups).unwrap();
    assert_eq!(cv5, replay);
    assert_ne!(cv5.seed, cv7.seed);
    assert_eq!(cv5.folds.len(), 5);
    assert_eq!(cv7.folds.len(), 7);
    for plan in [&cv5, &cv7] {
        let mut group_fold = BTreeMap::new();
        for fold in &plan.folds {
            for row in &fold.hold_rows {
                if let Some(prior) = group_fold.get(&groups[*row]) {
                    assert_eq!(*prior, fold.fold_index);
                } else {
                    group_fold.insert(groups[*row].clone(), fold.fold_index);
                }
            }
        }
    }
}

#[test]
fn grouped_plans_reject_tiny_cv_and_require_two_groups_per_fold() {
    let tiny = (0..9).map(|row| format!("group-{row}")).collect::<Vec<_>>();
    assert!(build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &tiny).is_err());

    let feasible = (0..40)
        .map(|row| format!("group-{row}"))
        .collect::<Vec<_>>();
    let plan = build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &feasible).unwrap();
    assert!(plan.folds.iter().all(|fold| {
        fold.fit_rows.len() >= MINIMUM_FOLD_FIT_ROWS
            && fold.hold_rows.len() >= MINIMUM_FOLD_HOLD_ROWS
            && fold.hold_groups.len() >= MINIMUM_FOLD_HOLD_GROUPS
    }));
}

#[test]
fn duplicated_rows_cross_fold_groups_and_insufficient_groups_are_rejected() {
    let mut folds = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |_, _| 8.0);
    folds[1].row_ids[0] = folds[0].row_ids[0].clone();
    assert!(screen_cv5(&folds, &GateThresholdsV1::default()).is_err());

    let mut folds = fixture_folds(5, &FeatureViewKindV1::EXPANDED, |_, _| 8.0);
    folds[1].atomic_groups[0] = folds[0].atomic_groups[0].clone();
    assert!(screen_cv5(&folds, &GateThresholdsV1::default()).is_err());

    let groups = vec!["a".to_string(); 20];
    assert!(build_grouped_fold_plan(GateStageV1::Cv5Screen, &groups).is_err());
}

#[test]
fn registry_and_cli_authority_are_path_safe_and_train_only() {
    assert!(validate_opaque_id("opaque_dataset-01").is_ok());
    assert!(validate_opaque_id("../hold.csv").is_err());
    assert!(safe_registry_path(std::path::Path::new("/authority"), "d/train.csv").is_ok());
    assert!(safe_registry_path(std::path::Path::new("/authority"), "../sealed.csv").is_err());

    let malicious = serde_json::json!({
        "schema": JopeHyperionTrainOnlyRegistryV1::SCHEMA,
        "encoder": {},
        "evaluator": {},
        "datasets": [],
        "test_path": "/forbidden"
    });
    assert!(serde_json::from_value::<JopeHyperionTrainOnlyRegistryV1>(malicious).is_err());

    let legacy = serde_json::json!({
        "schema": JopeHyperionTrainOnlyRegistryV1::SCHEMA,
        "encoder": {},
        "evaluator": {},
        "datasets": []
    });
    assert!(serde_json::from_value::<JopeHyperionTrainOnlyRegistryV2>(legacy).is_err());
}

#[test]
fn precomputed_cell_cache_is_authoritative_when_optional_row_cache_is_bad() {
    let temporary = tempfile::tempdir().unwrap();
    let dataset_root = temporary.path().join("fixture");
    fs::create_dir_all(&dataset_root).unwrap();
    let train = (0..40)
        .map(|row| format!("{},{},{}\n", row, row + 1, row as f64 / 10.0))
        .collect::<String>();
    let cell = (0..40)
        .map(|row| format!("{},{}\n", row as f64 / 40.0, (40 - row) as f64 / 40.0))
        .collect::<String>();
    fs::write(dataset_root.join("train.csv"), &train).unwrap();
    fs::write(dataset_root.join("train3.vs"), &cell).unwrap();
    fs::write(dataset_root.join("hyperion3train.vs"), "1,2\n").unwrap();

    let artifact = |name: &str| FrozenArtifactV1 {
        path: format!("/authority/{name}"),
        sha256: sha256(name.as_bytes()),
        byte_length: 1,
    };
    let registry = JopeHyperionTrainOnlyRegistryV2 {
        schema: JopeHyperionTrainOnlyRegistryV2::SCHEMA.into(),
        cache_root: "cache".into(),
        encoder: HyperionEncoderIdentityV1 {
            binary: artifact("encoder"),
            weights: artifact("weights"),
            manifest: artifact("manifest"),
            seed: 7,
            sweeps: 3,
            scales: vec![4, 7, 11, 15],
            batch_size: 32,
            device_type: "cuda".into(),
            device_ordinal: 0,
            runtime_versions: BTreeMap::from([
                ("rustc".into(), "fixture".into()),
                ("cuda_driver".into(), "fixture".into()),
                ("gpu_name".into(), "fixture".into()),
                ("gpu_uuid".into(), "GPU-fixture".into()),
            ]),
            source_hashes: BTreeMap::from([("source".into(), sha256(b"source"))]),
        },
        evaluator: EvaluatorFreezeV1 {
            schema: EvaluatorFreezeV1::SCHEMA.into(),
            artifact: artifact("evaluator"),
            evaluator_protocol_sha256: sha256(b"protocol"),
            math_policy_sha256: sha256(b"math"),
            backend_policy_sha256: sha256(b"backend"),
        },
        datasets: vec![TrainOnlyDatasetV1 {
            dataset_id: "fixture".into(),
            dataset_sha256: sha256(b"dataset"),
            train_csv_path: "fixture/train.csv".into(),
            train_sha256: sha256(train.as_bytes()),
            feature_count: 2,
            stable_row_ids: (0..40).map(|row| format!("row-{row}")).collect(),
            atomic_groups: (0..40).map(|row| format!("group-{row}")).collect(),
        }],
    };
    let receipt =
        inspect_precomputed_train_features(&registry, temporary.path(), "fixture").unwrap();
    assert_eq!(receipt.cell.sha256, sha256(cell.as_bytes()));
    assert_eq!(
        receipt.available_views(),
        vec![FeatureViewKindV1::HyperionCell]
    );
    assert!(receipt.row128.is_none());
    assert!(receipt.row128_unavailable_reason.is_some());
    assert!(!receipt.test_artifacts_read);

    let registry_path = temporary.path().join("registry.v2.json");
    fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();
    let coverage = freeze_precomputed_train_feature_coverage(&registry_path).unwrap();
    assert_eq!(coverage.datasets, vec![receipt]);
    assert!(!coverage.test_artifacts_read);
    assert_eq!(
        verify_precomputed_train_feature_coverage(&registry_path).unwrap(),
        coverage
    );

    fs::write(dataset_root.join("train3.vs"), "1,2,3\n").unwrap();
    assert!(inspect_precomputed_train_features(&registry, temporary.path(), "fixture").is_err());
    assert!(verify_precomputed_train_feature_coverage(&registry_path).is_err());
}

#[test]
fn captured_output_requires_an_exact_cuda_device_report() {
    assert!(encoder_output_proves_device(
        "starting\ndevice=cuda:0\ncomplete\n",
        "",
        "cuda:0"
    ));
    assert!(encoder_output_proves_device(
        "",
        "device=cuda:3\n",
        "cuda:3"
    ));
    for forged in [
        "device=cpu",
        "device=cpu device=cuda:0",
        "claimed_device=cuda:0",
        "device=cuda",
        "device=cuda:1",
        "device=cuda:0,",
    ] {
        assert!(!encoder_output_proves_device(forged, "", "cuda:0"));
    }

    let stdout = "device=cuda:0\n".to_string();
    let proof = EncoderExecutionProofV2 {
        encoder_identity_sha256: sha256(b"encoder"),
        gpu_uuid: "GPU-fixture".into(),
        cuda_visible_devices: "GPU-fixture".into(),
        argv_sha256: canonical_sha256(&hyperion_argv_schema_v2()).unwrap(),
        environment_sha256: canonical_sha256(&hyperion_environment_v2()).unwrap(),
        observed_device: "cuda:0".into(),
        stdout_sha256: sha256(stdout.as_bytes()),
        stderr_sha256: sha256(b""),
        stdout,
        stderr: String::new(),
        elapsed_milliseconds: 7,
    };
    proof.validate(0).unwrap();
    assert!(proof.validate(1).is_err());

    let mut altered = proof;
    altered.stdout.push_str("altered");
    assert!(altered.validate(0).is_err());
}

#[test]
fn portable_identity_and_physical_execution_proof_are_separate() {
    let artifact = |name: &str| FrozenArtifactV1 {
        path: format!("/authority/{name}"),
        sha256: sha256(name.as_bytes()),
        byte_length: 1,
    };
    let identity = HyperionEncoderIdentityV2 {
        schema: HYPERION_ENCODER_IDENTITY_V2_SCHEMA.into(),
        binary: artifact("encoder"),
        weights: artifact("weights"),
        manifest: artifact("manifest"),
        seed: 7,
        sweeps: 2,
        scales: vec![1, 4],
        batch_size: 32,
        device_type: "cuda".into(),
        runtime_class: "a100-cuda12".into(),
        runtime_versions: BTreeMap::from([
            ("rustc".into(), "rustc 1.86".into()),
            ("cuda_driver".into(), "550.54".into()),
            ("gpu_name".into(), "NVIDIA A100".into()),
        ]),
        source_hashes: BTreeMap::from([("source_commit".into(), sha256(b"commit"))]),
        argv_schema: hyperion_argv_schema_v2(),
        environment: hyperion_environment_v2(),
    };
    let identity_sha256 = identity.identity().unwrap();
    let stdout = "device=cuda:0\n";
    let proof = HyperionEncoderExecutionProofV2 {
        schema: HYPERION_ENCODER_PROOF_V2_SCHEMA.into(),
        encoder_identity_sha256: identity_sha256,
        node: "xbabe1".into(),
        gpu_uuid: "GPU-123".into(),
        gpu_name: "NVIDIA A100".into(),
        cuda_driver: "550.54".into(),
        cuda_visible_devices: "GPU-123".into(),
        observed_device: "cuda:0".into(),
        argv_sha256: canonical_sha256(&identity.argv_schema).unwrap(),
        environment_sha256: canonical_sha256(&identity.environment).unwrap(),
        stdout: stdout.into(),
        stdout_sha256: sha256(stdout.as_bytes()),
        stderr: String::new(),
        stderr_sha256: sha256(b""),
    };
    proof.validate(&identity).unwrap();

    let mut moved = proof;
    moved.node = "another-node".into();
    assert!(moved.validate(&identity).is_ok());
    moved.cuda_visible_devices = "GPU-456".into();
    assert!(moved.validate(&identity).is_err());
}
