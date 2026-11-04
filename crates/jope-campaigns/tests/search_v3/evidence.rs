use super::*;

#[test]
fn candidate_plan_and_four_task_contracts_are_exhaustive() {
    let plan = candidate_plan(0);
    plan.validate().unwrap();
    let mut too_short = plan.clone();
    too_short.candidates.pop();
    assert!(too_short.validate().is_err());
    assert_eq!(SearchTaskV1::ALL.len(), 4);
    assert!(SearchTaskV1::Regression.production_qualifiable());
    assert!(SearchTaskV1::Binary.production_qualifiable());
    assert!(!SearchTaskV1::Multiclass.production_qualifiable());
    assert!(!SearchTaskV1::Timeseries.production_qualifiable());

    let mut regression = contract(&plan);
    regression.validate().unwrap();
    regression.task = SearchTaskV1::Binary;
    assert!(regression.validate().is_err());
    regression.fitness = RegisteredFitnessV1::BinaryAuc {
        lift_depths: vec![0.01, 0.1],
        max_log_loss: 1.0,
    };
    regression.validate().unwrap();
    regression.task = SearchTaskV1::Multiclass;
    regression.fitness = RegisteredFitnessV1::MulticlassMacroF1 {
        classes: 3,
        min_class_support: 1,
        max_log_loss: 2.0,
    };
    regression.validate().unwrap();
    regression.task = SearchTaskV1::Timeseries;
    regression.budget.declared_fold_count = 3;
    regression.fitness = RegisteredFitnessV1::NegativeMase { max_mase: 2.0 };
    regression.validate().unwrap();

    let mut value = serde_json::to_value(contract(&plan)).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), true.into());
    assert!(serde_json::from_value::<SearchRunContractV3>(value).is_err());
}

#[test]
fn tabular_folds_keep_groups_and_duplicates_atomic_and_validate_class_support() {
    let regression_rows: Vec<_> = (0..50)
        .map(|row| TabularRowV2 {
            stable_row_id: format!("row-{row:03}"),
            group_id: (row < 4).then(|| format!("group-{}", row / 2)),
            duplicate_signature: if row == 10 || row == 11 {
                "duplicate-pair".into()
            } else {
                format!("signature-{row}")
            },
            regression_target: Some(row as f64),
            class_label: None,
        })
        .collect();
    let input = TabularFoldInputV2 {
        task: SearchTaskV1::Regression,
        training_view_sha256: digest("regression-view"),
        rows: regression_rows,
    };
    let plan = build_tabular_fold_plan(&input).unwrap();
    plan.validate().unwrap();
    assert_eq!(plan.folds.len(), 5);
    assert_eq!(
        plan.row_group_sha256s["row-010"],
        plan.row_group_sha256s["row-011"]
    );
    assert_eq!(
        plan.identity().unwrap(),
        build_tabular_fold_plan(&input).unwrap().identity().unwrap()
    );

    let binary = TabularFoldInputV2 {
        task: SearchTaskV1::Binary,
        training_view_sha256: digest("binary-view"),
        rows: (0..30)
            .map(|row| TabularRowV2 {
                stable_row_id: format!("b-{row}"),
                group_id: None,
                duplicate_signature: format!("b-sig-{row}"),
                regression_target: None,
                class_label: Some(row % 2),
            })
            .collect(),
    };
    let binary_plan = build_tabular_fold_plan(&binary).unwrap();
    assert!(binary_plan
        .folds
        .iter()
        .all(|fold| fold.balance_counts.contains_key("class_0")
            && fold.balance_counts.contains_key("class_1")));

    let mut impossible = binary;
    for row in impossible
        .rows
        .iter_mut()
        .filter(|row| row.class_label == Some(1))
    {
        row.group_id = Some(format!("collapsed-{}", row.stable_row_id.as_bytes()[2] % 4));
    }
    assert!(build_tabular_fold_plan(&impossible)
        .unwrap_err()
        .contains("atomic groups"));

    let multiclass = TabularFoldInputV2 {
        task: SearchTaskV1::Multiclass,
        training_view_sha256: digest("multi-view"),
        rows: (0..45)
            .map(|row| TabularRowV2 {
                stable_row_id: format!("m-{row}"),
                group_id: None,
                duplicate_signature: format!("m-sig-{row}"),
                regression_target: None,
                class_label: Some(row % 3),
            })
            .collect(),
    };
    build_tabular_fold_plan(&multiclass)
        .unwrap()
        .validate()
        .unwrap();
}

#[test]
fn timeseries_origins_enforce_order_purge_embargo_and_no_future_features() {
    let input = TimeseriesFoldInputV2 {
        training_view_sha256: digest("timeseries-view"),
        rows: (0..30)
            .map(|row| TimeseriesRowV2 {
                stable_row_id: format!("t-{row:02}"),
                time_key: row,
                feature_max_time_key: row,
            })
            .collect(),
        origins: vec![
            TimeseriesOriginV2 {
                train_end_index: 8,
                validation_len: 2,
                purge_rows: 1,
                embargo_rows: 1,
                required_seasonal_history: 4,
            },
            TimeseriesOriginV2 {
                train_end_index: 12,
                validation_len: 2,
                purge_rows: 1,
                embargo_rows: 1,
                required_seasonal_history: 4,
            },
            TimeseriesOriginV2 {
                train_end_index: 16,
                validation_len: 2,
                purge_rows: 1,
                embargo_rows: 1,
                required_seasonal_history: 4,
            },
        ],
    };
    let plan = build_timeseries_fold_plan(&input).unwrap();
    plan.validate().unwrap();
    assert_eq!(plan.folds[0].validation_row_ids, vec!["t-10", "t-11"]);
    let mut future = input.clone();
    future.rows[5].feature_max_time_key = 6;
    assert!(build_timeseries_fold_plan(&future)
        .unwrap_err()
        .contains("future-derived"));
    let mut unordered = input;
    unordered.rows.swap(3, 4);
    assert!(build_timeseries_fold_plan(&unordered).is_err());
}

#[test]
fn prediction_and_scoring_identities_separate_fit_truth_and_objective() {
    let plan = candidate_plan(0);
    let contract = contract(&plan);
    let record = evaluation(&contract, &plan.candidates[0], 0);
    let prediction_identity = record.prediction.identity().unwrap();
    let tape_identity = record.scoring_tape.identity().unwrap();

    let alternate_fitness = RegisteredFitnessV1::RobustR2 { max_rmse: 0.5 };
    let (alternate_key, _) = score_prediction(
        &record.prediction,
        &record.scoring_tape,
        &alternate_fitness,
        1,
    )
    .unwrap();
    assert_eq!(record.prediction.identity().unwrap(), prediction_identity);
    assert_eq!(record.scoring_tape.identity().unwrap(), tape_identity);
    assert_ne!(alternate_key, record.score_key);

    let mut changed_truth = record.scoring_tape.clone();
    changed_truth.truth_f64_bits[0] = 0.25_f64.to_bits();
    assert_eq!(record.prediction.identity().unwrap(), prediction_identity);
    assert_ne!(changed_truth.identity().unwrap(), tape_identity);

    let mut changed_fit = record.prediction.clone();
    changed_fit.fit_sha256 = digest("different-fit");
    assert_ne!(changed_fit.identity().unwrap(), prediction_identity);
    let mut changed_backend = record.prediction.clone();
    changed_backend.backend_policy_sha256 = digest("different-backend");
    assert_ne!(changed_backend.identity().unwrap(), prediction_identity);

    let mut invalid = record.prediction;
    invalid.valid = false;
    invalid.invalid_reason = Some("nonfinite evaluator output".into());
    invalid.prediction_f64_bits[0] = f64::NAN.to_bits();
    let invalid_tape = ScoringTapeV2 {
        prediction_object_sha256: invalid.identity().unwrap(),
        ..record.scoring_tape
    };
    let (_, invalid_fitness) =
        score_prediction(&invalid, &invalid_tape, &contract.fitness, 1).unwrap();
    assert!(!invalid_fitness.evidence_valid);
    assert_eq!(
        invalid_fitness.compare_for_selection(&record.fitness),
        std::cmp::Ordering::Greater
    );
}
