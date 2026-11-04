use super::*;
#[test]
fn global_companions_require_qualification_and_context_exposure_proofs() {
    use crate::replay::admission::*;
    use jope_records::wire_contract::CanonicalPayload;
    let f = fixture(Task::Regression, 5);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"kind":"v2-fixture"}),
    )
    .unwrap();
    let retain = |value: serde_json::Value| {
        let canonical =
            CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&value).unwrap()).unwrap();
        f.state.put_blob(canonical.as_bytes()).unwrap()
    };
    let qualification = f
        .state
        .put_blob(&canonical_json(&qualified).unwrap())
        .unwrap();
    let mut subjects = BTreeSet::from([
        qualification.clone(),
        qualified.context_handoff_sha256.clone().unwrap(),
    ]);
    let mut pending = vec![qualified.context_receipt_sha256.clone().unwrap()];
    while let Some(sha) = pending.pop() {
        if !subjects.insert(sha.clone()) {
            continue;
        }
        let receipt: crate::encoder_receipt::EncoderReceipt = blob(&f.state, &sha).unwrap();
        subjects.insert(receipt.fitted_state_sha256);
        subjects.insert(receipt.output_sha256);
        pending.extend(receipt.inputs.parent_receipts);
    }
    let mut proofs = BTreeMap::new();
    for subject in &subjects {
        let receipt = ExposureReceipt {
            schema: EXPOSURE_SCHEMA.into(),
            subject_sha256: subject.clone(),
            family: "fixture".into(),
            meta_partition: "train_search".into(),
            original_role: "train".into(),
            within_task_view: "training".into(),
            fidelity: "measured_cpu_cv".into(),
            producer: "fixture".into(),
            policy_epoch: 1,
            observed_at_ms: 100,
            episode: None,
            purposes: BTreeSet::from([Purpose::GlobalPolicyTraining]),
            prior_exposure: Some(vec![]),
            parents: vec![],
            observation: Observation::Proposed,
        };
        proofs.insert(
            subject.clone(),
            retain(serde_json::to_value(receipt).unwrap()),
        );
    }
    let policy = AdmissionPolicy {
        schema: POLICY_SCHEMA.into(),
        issuer: "fixture-controller".into(),
        epoch: 1,
        effective_from_ms: 0,
        effective_until_ms: 1000,
        evaluation_boundary_sha256: f.state.put_blob(b"future evaluation boundary").unwrap(),
        purposes: BTreeSet::from([Purpose::GlobalPolicyTraining]),
        receipts: proofs.values().cloned().collect(),
        supersedes: None,
    };
    let pin = PolicyPin {
        schema: ADOPTION_SCHEMA.into(),
        policy_sha256: retain(serde_json::to_value(policy).unwrap()),
        issuer: "fixture-controller".into(),
        epoch: 1,
    };
    let adoption =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&pin).unwrap()).unwrap();
    let path = f.state.root().join(ADOPTION_PATH);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, adoption.as_bytes()).unwrap();
    let verifier = AdmissionVerifier::from_adopted_policy(&f.state).unwrap();
    assert_eq!(
        qualify_for(&f.state, &f.record, &f.evidence, &verifier, &proofs, 500)
            .unwrap()
            .0,
        qualified
    );
    assert_eq!(
        freeze_teaching_for(
            &f.state,
            &verifier,
            Purpose::GlobalPolicyTraining,
            &proofs,
            500
        )
        .unwrap()
        .len(),
        1
    );
    let mut missing = proofs.clone();
    missing.remove(qualified.context_handoff_sha256.as_ref().unwrap());
    assert!(freeze_teaching_for(
        &f.state,
        &verifier,
        Purpose::GlobalPolicyTraining,
        &missing,
        500
    )
    .unwrap_err()
    .to_string()
    .contains("context lacks global exposure proof"));
    assert!(revalidate_replay_for(
        &f.state,
        &[qualified],
        &verifier,
        Purpose::GlobalPolicyTraining,
        &BTreeMap::new(),
        500
    )
    .is_err());
    // Retain this new v2 proof fixture with the lane evidence instead of losing the
    // authenticated context/qualification closure when the test returns.
    let retained = f._directory.keep();
    println!("retained v2 teaching fixture: {}", retained.display());
}

#[test]
fn all_tasks_require_measured_aggregate_and_four_of_five_fold_gains() {
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        for gains in [0, 3, 4, 5] {
            let f = fixture(task, gains);
            let result = qualify(&f.state, &f.record, &f.evidence);
            if gains < 4 {
                assert!(result.is_err(), "no-signal/three-fold witness admitted");
            } else {
                let qualified = result.unwrap();
                assert_eq!(qualified.improved_folds, gains);
                assert!(qualified.candidate_loss < qualified.baseline_loss);
                hwm::ingest(
                    &f.state,
                    None,
                    vec![Candidate {
                        record: f.record.clone(),
                        blob: b"synthetic candidate".to_vec(),
                        chain_from_source: false,
                    }],
                    json!({"kind":"synthetic"}),
                )
                .unwrap();
                let manifest = freeze_teaching(&f.state).unwrap();
                assert_eq!(manifest.entries.len(), 1);
                assert_eq!(manifest.entries[0], qualified);
            }
        }
    }
}

#[test]
fn replay_requires_accepted_history_and_recomputes_retained_measurements() {
    let f = fixture(Task::Regression, 5);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert!(revalidate_replay(&f.state, std::slice::from_ref(&qualified)).is_err());
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"kind":"synthetic"}),
    )
    .unwrap();
    assert_eq!(
        revalidate_replay(&f.state, std::slice::from_ref(&qualified))
            .unwrap()
            .entries,
        vec![qualified.clone()]
    );
    let mut changed = qualified;
    changed.candidate_loss += 0.1;
    assert!(revalidate_replay(&f.state, &[changed]).is_err());
}

#[test]
fn freeze_is_independent_and_cannot_be_recreated_from_a_candidate() {
    let f = fixture(Task::Regression, 5);
    assert_eq!(
        freeze_optimization(&f.state, &f.plan).unwrap(),
        f.evidence.freeze
    );
    let mut changed = f.plan.clone();
    changed.evaluation_seed = 100;
    assert!(freeze_optimization(&f.state, &changed).is_err());
    let mut fake = f.evidence.clone();
    fake.freeze.freeze_sha256 = canonical_sha256(&f.record).unwrap();
    assert!(qualify(&f.state, &f.record, &fake).is_err());
    let mut claimed = f.record.clone();
    claimed.score.value = 0.5;
    assert!(qualify(&f.state, &claimed, &f.evidence).is_err());
}

#[test]
fn stale_identity_missing_fold_and_reordered_rows_cannot_teach() {
    let f = fixture(Task::Regression, 5);
    let mut stale = f.record.clone();
    stale.identity.vocabulary_sha256 = "f".repeat(64);
    stale.identity_sha256 = stale.identity.sha256().unwrap();
    assert!(qualify(&f.state, &stale, &f.evidence).is_err());
    let mut legacy = f.record.clone();
    legacy.identity.comparison = None;
    legacy.identity_sha256 = legacy.identity.sha256().unwrap();
    assert!(qualify(&f.state, &legacy, &f.evidence).is_err());
    let mut incomplete = f.evidence.clone();
    incomplete.fold_predictions_sha256.pop();
    assert!(qualify(&f.state, &f.record, &incomplete).is_err());
    let mut reordered: FoldPredictions =
        blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    reordered.row_ids.reverse();
    let mut evidence = f.evidence.clone();
    evidence.fold_predictions_sha256[0] = f
        .state
        .put_blob(&canonical_json(&reordered).unwrap())
        .unwrap();
    assert!(qualify(&f.state, &f.record, &evidence).is_err());
}

#[test]
fn structural_receipt_without_actual_predictions_or_complete_execution_is_ineligible() {
    let f = fixture(Task::Regression, 5);
    let mut prediction: FoldPredictions =
        blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    prediction.predictions_sha256 = "f".repeat(64);
    let mut evidence = f.evidence.clone();
    evidence.fold_predictions_sha256[0] = f
        .state
        .put_blob(&canonical_json(&prediction).unwrap())
        .unwrap();
    assert!(qualify(&f.state, &f.record, &evidence).is_err());
    let mut result: EvaluationResult =
        blob(&f.state, &prediction.execution_receipt_sha256).unwrap();
    result.status = EvaluationStatus::Failed;
    result.primary = None;
    result.scored_rows = 0;
    result.reason = Some("synthetic failure".into());
    prediction.execution_receipt_sha256 =
        f.state.put_blob(&canonical_json(&result).unwrap()).unwrap();
    evidence.fold_predictions_sha256[0] = f
        .state
        .put_blob(&canonical_json(&prediction).unwrap())
        .unwrap();
    assert!(qualify(&f.state, &f.record, &evidence).is_err());
}

#[test]
fn unknown_accounting_and_unbound_fitted_outputs_cannot_be_substituted() {
    let f = fixture(Task::Regression, 5);
    let mut prediction: FoldPredictions =
        blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    let original: EvaluationResult = blob(&f.state, &prediction.execution_receipt_sha256).unwrap();
    let mut tampered = original.clone();
    tampered.consumption.cpu_seconds = Some(0.0);
    prediction.execution_receipt_sha256 = f
        .state
        .put_blob(&canonical_json(&tampered).unwrap())
        .unwrap();
    let mut evidence = f.evidence.clone();
    evidence.fold_predictions_sha256[0] = f
        .state
        .put_blob(&canonical_json(&prediction).unwrap())
        .unwrap();
    assert!(qualify(&f.state, &f.record, &evidence).is_err());

    let mut unrelated = original;
    prediction.fitted_artifact_sha256 = f
        .state
        .put_blob(b"unrelated synthetic fitted state")
        .unwrap();
    unrelated.fitted_state_sha256 = Some(prediction.fitted_artifact_sha256.clone());
    prediction.execution_receipt_sha256 = f
        .state
        .put_blob(&canonical_json(&unrelated).unwrap())
        .unwrap();
    evidence.fold_predictions_sha256[0] = f
        .state
        .put_blob(&canonical_json(&prediction).unwrap())
        .unwrap();
    let error = qualify(&f.state, &f.record, &evidence).unwrap_err();
    assert!(
        error.to_string().contains("expected value")
            || error.to_string().contains("native fitted recipe")
            || error
                .to_string()
                .contains("absent from the bound worker outputs")
    );
}

#[test]
fn burns_rejections_guard_margin_and_unqualified_incumbents_are_durable() {
    let f = fixture(Task::Binary, 0);
    assert!(require_unburned_test_partition(
        &f.state,
        &f.record.identity.dataset_hash,
        &f.record.identity.cut_id
    )
    .is_err());
    assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
    let ledger = Ledger::open(f.state.root(), "eligibility");
    ledger.require_intact().unwrap();
    let decisions = ledger.events().unwrap();
    assert!(decisions.iter().any(|event| event["outcome"] == "rejected"
        && event["measurement"]["improved_folds"] == 0
        && event["guard"]["aggregation"] == "mean_fold_loss"));
    let report = hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"source":"synthetic"}),
    )
    .unwrap();
    assert_eq!(report.rejected, 1);
    assert!(f
        .state
        .read_pointer(&PointerKey::of(&f.record))
        .unwrap()
        .is_none());
    let policy = GuardPolicy::current();
    assert!(!policy.improves(0.5 - 1e-8, 0.5));
    assert!(policy.improves(0.49, 0.5));
}

#[test]
fn old_execution_cannot_be_attached_after_freeze_and_pruning_keeps_measured_evidence() {
    let f = fixture(Task::Regression, 5);
    let mut evidence = f.evidence.clone();
    let mut experiment: Experiment = blob(&f.state, &evidence.experiment_blob_sha256).unwrap();
    experiment.checkpoints.remove("teaching_freeze");
    evidence.experiment_blob_sha256 = f
        .state
        .put_blob(&canonical_json(&experiment).unwrap())
        .unwrap();
    assert!(qualify(&f.state, &f.record, &evidence).is_err());
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    // Simulate mutable journal output pruning after verified bytes entered content-addressed custody.
    for fold in 0..5 {
        fs::remove_dir_all(f.state.root().join(format!("fixture-worker-{fold}"))).unwrap();
    }
    require_qualified(&f.state, &f.record, &Decisions::load(&f.state).unwrap()).unwrap();
    assert!(qualified.guard.admits(&qualified));
}

#[test]
fn tampered_prediction_blob_fails_closed() {
    let f = fixture(Task::Regression, 5);
    let prediction: FoldPredictions =
        blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
    fs::write(
        f.state.blob_path(&prediction.predictions_sha256),
        b"{\"kind\":\"values\",\"data\":[null,null]}",
    )
    .unwrap();
    assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
    assert_eq!(effective_label_role("test", true).unwrap(), "development");
    assert_eq!(effective_label_role("test", false).unwrap(), "test");
    assert!(effective_label_role("sealed", true).is_err());
}
