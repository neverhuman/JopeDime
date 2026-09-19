use std::collections::BTreeSet;
use std::fs;

use jope_campaigns::search_v3::*;

fn digest(value: impl AsRef<str>) -> String {
    canonical_sha256(&value.as_ref()).unwrap()
}

fn candidate_plan(generation: u32) -> SearchCandidatePlanV1 {
    SearchCandidatePlanV1 {
        schema: SearchCandidatePlanV1::SCHEMA.into(),
        generation,
        candidates: (0..60)
            .map(|slot| SearchCandidateIdentityV1 {
                slot,
                semantic_sha256: digest(format!("generation-{generation}-candidate-{slot}")),
                ancestry_sha256: digest(format!("generation-{generation}-ancestry-{slot}")),
                producer: match slot % 3 {
                    0 => CandidateProducerV1::Gp,
                    1 => CandidateProducerV1::Dl,
                    _ => CandidateProducerV1::Hybrid,
                },
                seed: u64::from(generation) << 32 | u64::from(slot),
                complexity: u64::from(slot) + 1,
                compute_units: u64::from(slot) + 1,
            })
            .collect(),
    }
}

fn contract(plan: &SearchCandidatePlanV1) -> SearchRunContractV3 {
    SearchRunContractV3 {
        schema: SearchRunContractV3::SCHEMA.into(),
        task: SearchTaskV1::Regression,
        task_semantics_sha256: digest("task-semantics"),
        training_view_sha256: digest("training-view"),
        fold_plan_sha256: digest("fold-plan"),
        candidate_plan_root_sha256: plan.identity().unwrap(),
        replay_protocol_sha256: digest("replay"),
        evaluator_sha256: digest("evaluator"),
        dsl_sha256: digest("dsl"),
        dependency_lockfile_sha256: digest("lockfile"),
        math_policy_sha256: digest("math"),
        backend_policy_sha256: digest("backend"),
        fitness: RegisteredFitnessV1::RobustR2 { max_rmse: 100.0 },
        conditioning: ConditioningPolicyV1 {
            policy_sha256: digest("conditioning"),
            validation_truth_access: false,
            held_out_release_access: false,
        },
        budget: SearchBudgetV1 {
            generations: 2,
            candidates_per_generation: 60,
            declared_fold_count: 5,
            stage_two_candidates: 30,
            finalist_candidates: 8,
        },
        future_export: FutureExportRequirementsV1 {
            source_required: true,
            predict_protocol_sha256: digest("predict"),
            benchmark_protocol_sha256: digest("benchmark"),
        },
    }
}

fn evaluation(
    contract: &SearchRunContractV3,
    candidate: &SearchCandidateIdentityV1,
    fold: u32,
) -> EvaluationRecordV2 {
    let truth = [0.0_f64, 1.0, 2.0, 3.0];
    let error = f64::from(candidate.slot) * 0.01 + f64::from(fold) * 0.001;
    let prediction = PredictionObjectV1 {
        schema: PredictionObjectV1::SCHEMA.into(),
        training_view_sha256: contract.training_view_sha256.clone(),
        fold_plan_sha256: contract.fold_plan_sha256.clone(),
        fold_index: fold,
        candidate_semantic_sha256: candidate.semantic_sha256.clone(),
        evaluator_sha256: contract.evaluator_sha256.clone(),
        fit_sha256: digest(format!("fit-{}-{fold}", candidate.semantic_sha256)),
        math_policy_sha256: contract.math_policy_sha256.clone(),
        backend_policy_sha256: contract.backend_policy_sha256.clone(),
        validation_row_ids: (0..4).map(|row| format!("fold-{fold}-row-{row}")).collect(),
        prediction_shape: vec![4],
        prediction_f64_bits: truth
            .iter()
            .map(|value| (value + error).to_bits())
            .collect(),
        valid: true,
        invalid_reason: None,
    };
    let tape = ScoringTapeV2 {
        schema: ScoringTapeV2::SCHEMA.into(),
        prediction_object_sha256: prediction.identity().unwrap(),
        validation_row_ids: prediction.validation_row_ids.clone(),
        truth_shape: vec![4],
        truth_f64_bits: truth.iter().map(|value| value.to_bits()).collect(),
        auxiliary_scale_f64_bits: None,
    };
    let (score_key, fitness) =
        score_prediction(&prediction, &tape, &contract.fitness, candidate.complexity).unwrap();
    EvaluationRecordV2 {
        schema: EvaluationRecordV2::SCHEMA.into(),
        candidate_slot: candidate.slot,
        fold_index: fold,
        prediction,
        scoring_tape: tape,
        score_key,
        fitness,
    }
}

fn complete_evidence(
    contract: &SearchRunContractV3,
    plan: &SearchCandidatePlanV1,
) -> Vec<CandidateEvidenceV1> {
    plan.candidates
        .iter()
        .map(|candidate| {
            let folds: Vec<u32> = if candidate.slot < 8 {
                (0..5).collect()
            } else if candidate.slot < 30 {
                vec![0, 1]
            } else {
                vec![0]
            };
            CandidateEvidenceV1 {
                candidate: candidate.clone(),
                evaluations: folds
                    .into_iter()
                    .map(|fold| (fold, evaluation(contract, candidate, fold)))
                    .collect(),
            }
        })
        .collect()
}

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

#[test]
fn every_registered_fitness_has_typed_guards_and_native_semantics() {
    fn object(
        task: &str,
        shape: Vec<usize>,
        predictions: Vec<f64>,
        truth: Vec<f64>,
        scale: Option<f64>,
    ) -> (PredictionObjectV1, ScoringTapeV2) {
        let prediction = PredictionObjectV1 {
            schema: PredictionObjectV1::SCHEMA.into(),
            training_view_sha256: digest(format!("{task}-view")),
            fold_plan_sha256: digest(format!("{task}-folds")),
            fold_index: 0,
            candidate_semantic_sha256: digest(format!("{task}-candidate")),
            evaluator_sha256: digest("evaluator"),
            fit_sha256: digest(format!("{task}-fit")),
            math_policy_sha256: digest("math"),
            backend_policy_sha256: digest("backend"),
            validation_row_ids: (0..truth.len()).map(|row| format!("row-{row}")).collect(),
            prediction_shape: shape,
            prediction_f64_bits: predictions.into_iter().map(f64::to_bits).collect(),
            valid: true,
            invalid_reason: None,
        };
        let tape = ScoringTapeV2 {
            schema: ScoringTapeV2::SCHEMA.into(),
            prediction_object_sha256: prediction.identity().unwrap(),
            validation_row_ids: prediction.validation_row_ids.clone(),
            truth_shape: vec![truth.len()],
            truth_f64_bits: truth.into_iter().map(f64::to_bits).collect(),
            auxiliary_scale_f64_bits: scale.map(f64::to_bits),
        };
        (prediction, tape)
    }

    let (binary_prediction, binary_tape) = object(
        "binary",
        vec![4],
        vec![0.01, 0.99, 0.1, 0.9],
        vec![0.0, 1.0, 0.0, 1.0],
        None,
    );
    let (_, binary) = score_prediction(
        &binary_prediction,
        &binary_tape,
        &RegisteredFitnessV1::BinaryAuc {
            lift_depths: vec![0.25, 0.5],
            max_log_loss: 1.0,
        },
        4,
    )
    .unwrap();
    assert_eq!(binary.primary.task, SearchTaskV1::Binary);
    assert_eq!(binary.primary.value, 1.0);
    assert!(binary.constraints_satisfied);

    let multiclass_truth = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let multiclass_predictions = multiclass_truth
        .iter()
        .flat_map(|truth| {
            let truth = *truth as usize;
            (0..3).map(move |class| if class == truth { 0.9 } else { 0.05 })
        })
        .collect();
    let (multiclass_prediction, multiclass_tape) = object(
        "multiclass",
        vec![6, 3],
        multiclass_predictions,
        multiclass_truth,
        None,
    );
    let (_, multiclass) = score_prediction(
        &multiclass_prediction,
        &multiclass_tape,
        &RegisteredFitnessV1::MulticlassMacroF1 {
            classes: 3,
            min_class_support: 2,
            max_log_loss: 1.0,
        },
        6,
    )
    .unwrap();
    assert_eq!(multiclass.primary.task, SearchTaskV1::Multiclass);
    assert_eq!(multiclass.primary.value, 1.0);
    assert!(multiclass.constraints_satisfied);

    let (timeseries_prediction, timeseries_tape) = object(
        "timeseries",
        vec![4],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, 2.0, 3.0],
        Some(2.0),
    );
    let (_, timeseries) = score_prediction(
        &timeseries_prediction,
        &timeseries_tape,
        &RegisteredFitnessV1::NegativeMase { max_mase: 1.0 },
        3,
    )
    .unwrap();
    assert_eq!(timeseries.primary.task, SearchTaskV1::Timeseries);
    assert_eq!(timeseries.primary.value, -0.5);
    assert!(timeseries.constraints_satisfied);
}

#[test]
fn common_evidence_schedule_is_114_and_completion_order_cannot_change_roots() {
    let plan = candidate_plan(0);
    let contract = contract(&plan);
    let evidence = complete_evidence(&contract, &plan);
    let state = build_generation_state(
        contract.clone(),
        plan.clone(),
        evidence.clone(),
        None,
        &ConfirmedArchiveV1::empty(),
    )
    .unwrap();
    assert_eq!(state.promotion_schedule.fold_evaluation_count, 114);
    assert_eq!(
        state.promotion_schedule.promoted_thirty_slots,
        (0..30).collect::<Vec<_>>()
    );
    assert_eq!(
        state.promotion_schedule.finalist_eight_slots,
        (0..8).collect::<Vec<_>>()
    );
    assert_eq!(state.archive.candidates.len(), 8);
    let expected = state.identity().unwrap();
    let mut roots = vec![expected.clone(); 3];

    for schedule in 0..100usize {
        let stride = [1usize, 7, 11, 13, 17, 19, 23, 29, 31, 37][schedule % 10];
        let offset = schedule % 60;
        let shuffled: Vec<_> = (0..60)
            .map(|index| evidence[(index * stride + offset) % 60].clone())
            .collect();
        let replay = build_generation_state(
            contract.clone(),
            plan.clone(),
            shuffled,
            None,
            &ConfirmedArchiveV1::empty(),
        )
        .unwrap();
        let replay_root = replay.identity().unwrap();
        assert_eq!(replay_root, expected);
        roots.push(replay_root);
    }
    let exhaustive = plan
        .candidates
        .iter()
        .map(|candidate| evaluation(&contract, candidate, 0).fitness)
        .collect();
    let report = qualify_against_exhaustive(&state, exhaustive, &roots).unwrap();
    assert!(report.qualified);
    assert_eq!(report.top_five_recall, 1.0);
    assert!((report.fold_work_reduction - 0.62).abs() < 1e-12);
}

#[test]
fn archive_never_forgets_and_checkpoint_recovers_only_complete_generations() {
    let first_plan = candidate_plan(0);
    let first_contract = contract(&first_plan);
    let first_state = build_generation_state(
        first_contract.clone(),
        first_plan.clone(),
        complete_evidence(&first_contract, &first_plan),
        None,
        &ConfirmedArchiveV1::empty(),
    )
    .unwrap();
    let first_digest = first_state.identity().unwrap();

    let second_plan = candidate_plan(1);
    let second_contract = contract(&second_plan);
    let second_state = build_generation_state(
        second_contract.clone(),
        second_plan.clone(),
        complete_evidence(&second_contract, &second_plan),
        Some(first_digest.clone()),
        &first_state.archive,
    )
    .unwrap();
    assert_eq!(second_state.archive.candidates.len(), 16);
    assert!(first_state.archive.candidates.iter().all(|earlier| {
        second_state.archive.candidates.iter().any(|candidate| {
            candidate.candidate.semantic_sha256 == earlier.candidate.semantic_sha256
        })
    }));

    let temp = tempfile::tempdir().unwrap();
    let store = CheckpointStoreV2::new(temp.path());
    assert_eq!(store.publish(&first_state).unwrap(), first_digest);
    let second_digest = store.publish(&second_state).unwrap();
    let structural = store.verify(VerificationMode::Structural).unwrap();
    assert_eq!(structural.latest_generation, 1);
    assert_eq!(
        store
            .verify(VerificationMode::Witness)
            .unwrap()
            .generations_checked,
        2
    );
    assert_eq!(
        store
            .verify(VerificationMode::Full)
            .unwrap()
            .evaluations_checked,
        228
    );

    fs::write(temp.path().join("checkpoint.current"), b"corrupt\n").unwrap();
    assert_eq!(store.recover_latest().unwrap().0, second_digest);
    store.repair_pointer_to_latest_complete().unwrap();

    fs::write(
        temp.path().join("ledgers/generation-00000001.json"),
        b"corrupt",
    )
    .unwrap();
    assert_eq!(store.recover_latest().unwrap().0, first_digest);
    assert_eq!(
        store.repair_pointer_to_latest_complete().unwrap(),
        first_digest
    );
    assert_eq!(
        store
            .verify(VerificationMode::Structural)
            .unwrap()
            .latest_generation,
        0
    );
}

#[test]
fn hybrid_receipts_require_every_matched_dimension() {
    let baseline = HybridMatchedRunV1 {
        dataset_sha256: digest("dataset"),
        context_sha256: digest("context"),
        seed: 7,
        slot: 3,
        evaluator_sha256: digest("evaluator"),
        ancestry_sha256: digest("ancestry"),
        compute_units: 100,
        score: 0.4,
    };
    let mut hybrid = baseline.clone();
    hybrid.score = 0.6;
    let receipt = HybridContributionReceiptV1::new(baseline.clone(), hybrid.clone()).unwrap();
    assert_eq!(
        receipt.incremental_gain.to_bits(),
        (0.6_f64 - 0.4).to_bits()
    );
    for mutation in 0..7 {
        let mut mismatched = hybrid.clone();
        match mutation {
            0 => mismatched.dataset_sha256 = digest("other-dataset"),
            1 => mismatched.context_sha256 = digest("other-context"),
            2 => mismatched.seed += 1,
            3 => mismatched.slot += 1,
            4 => mismatched.evaluator_sha256 = digest("other-evaluator"),
            5 => mismatched.ancestry_sha256 = digest("other-ancestry"),
            6 => mismatched.compute_units += 1,
            _ => unreachable!(),
        }
        assert!(HybridContributionReceiptV1::new(baseline.clone(), mismatched).is_err());
    }
}

#[test]
fn equal_evidence_is_required_for_every_comparison() {
    let plan = candidate_plan(0);
    let contract = contract(&plan);
    let mut evidence = complete_evidence(&contract, &plan);
    evidence[59]
        .evaluations
        .insert(1, evaluation(&contract, &plan.candidates[59], 1));
    assert!(PromotionScheduleV1::build(&plan, &evidence, 5)
        .unwrap_err()
        .contains("expected"));
    let complete: BTreeSet<_> = (0..5).collect();
    assert_eq!(complete.len(), 5);
}
