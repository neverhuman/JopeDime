use super::*;

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
