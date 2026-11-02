use super::*;

#[test]
fn finalization_audits_each_real_partition_and_declares_the_refit_and_score_scopes() {
    for kind in [Refit::Fit, Refit::FitHoldout] {
        let mut f = fixture(false);
        f.request.refit = kind;
        let native = Native::default();
        let selected = search(&f, &native);
        assert!(native
            .calls
            .lock()
            .unwrap()
            .iter()
            .all(|(_, _, path)| path == &f.plan.matrix_path));
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let receipt = finalizer.finalize(&f.request, &native).unwrap();
        assert_eq!(receipt.scores.cv, Some(selected.value));
        assert_eq!(
            receipt.completion_reason,
            CompletionReason::BudgetEvaluations
        );
        assert_eq!(receipt.budget_bound_by.as_deref(), Some("evaluations"));
        assert_eq!(receipt.budget_used.evaluations.0, 1);
        assert_eq!(receipt.nondeterminism.count.0, 0);
        assert!(receipt.scores.holdout.unwrap().is_finite());
        assert!(receipt.scores.test.unwrap().is_finite());
        let exported = ModelObject::read(&receipt.model_path, HANDOFF_MAX).unwrap();
        exported.verify_executable().unwrap();
        assert_eq!(
            exported.manifest().metadata["refit"],
            serde_json::to_value(kind).unwrap()
        );
        assert_eq!(
            exported.manifest().metadata["training_rows"],
            if kind == Refit::Fit { 40 } else { 50 }
        );
        assert_eq!(
            receipt.model_digest == receipt.holdout_model_digest,
            kind == Refit::Fit
        );
        let entries = receipt.audit["entries"].as_array().unwrap();
        assert_eq!(
            entries
                .iter()
                .filter(|e| e["phase"] == "test_score")
                .count(),
            1
        );
        for entry in entries {
            let phase = entry["phase"].as_str().unwrap();
            for row in entry["rows"].as_array().unwrap() {
                let id = row["row"].as_u64().unwrap();
                if f.request.holdout.rows.iter().any(|r| r.row == id) {
                    assert!(matches!(phase, "holdout_score" | "finalization_refit"));
                }
                if f.request
                    .test
                    .as_ref()
                    .unwrap()
                    .rows
                    .iter()
                    .any(|r| r.row == id)
                {
                    assert_eq!(phase, "test_score");
                }
            }
        }
        let snapshot = finalizer
            .journal()
            .snapshot(&f.plan.run_id)
            .unwrap()
            .unwrap();
        assert_eq!(snapshot.state, RunState::Completed);
        assert_eq!(snapshot.phase, RunPhase::Finalizing);
        assert_eq!(
            snapshot.best.unwrap().algorithm_id.to_string(),
            selected.candidate.algorithm_id
        );
        let events = finalizer
            .journal()
            .page(&f.plan.run_id, 0, 256, &[])
            .unwrap()
            .unwrap();
        assert_eq!(
            events
                .events
                .iter()
                .filter(|e| matches!(e.event, EventBody::RunFinished(_)))
                .count(),
            1
        );
    }
}

#[test]
fn completed_replay_and_recovery_do_not_read_or_score_protected_partitions_again() {
    let f = fixture(false);
    let native = Native::default();
    search(&f, &native);
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    let first = finalizer.finalize(&f.request, &native).unwrap();
    let calls = native.calls.lock().unwrap().len();
    fs::rename(
        &f.request.holdout.path,
        f.directory.path().join("holdout-unavailable"),
    )
    .unwrap();
    fs::rename(
        &f.request.test.as_ref().unwrap().path,
        f.directory.path().join("test-unavailable"),
    )
    .unwrap();
    drop(finalizer);
    drop(f.store);
    let recovered = Arc::new(Store::open(f.directory.path().join("store")).unwrap());
    let finalizer = Finalizer::new(recovered);
    let replay = finalizer.finalize(&f.request, &native).unwrap();
    assert_eq!(first.model_digest, replay.model_digest);
    assert_eq!(calls, native.calls.lock().unwrap().len());
    let mut changed = f.request.clone();
    changed.refit = Refit::Fit;
    assert!(finalizer.finalize(&changed, &native).is_err());
    assert_eq!(calls, native.calls.lock().unwrap().len());

    // Run provenance belongs to the run-scoped receipt, not the canonical
    // predictor identity: the same problem and fit must have one digest.
    let other = fixture(false);
    let other_native = Native::default();
    search(&other, &other_native);
    let independent = Finalizer::new(Arc::clone(&other.store))
        .finalize(&other.request, &other_native)
        .unwrap();
    assert_ne!(first.run_id, independent.run_id);
    assert_eq!(first.run_id, f.request.run_id);
    assert_eq!(independent.run_id, other.request.run_id);
    assert_eq!(first.model_digest, independent.model_digest);
    assert_eq!(first.model_artifact, independent.model_artifact);
    assert_eq!(first.holdout_model_digest, independent.holdout_model_digest);
}

#[test]
fn failed_bounded_fit_preserves_audit_and_never_retries_or_reads_holdout_or_test() {
    let f = fixture(false);
    search(&f, &Native::default());
    let failing = Native {
        failure: Some(WorkerFailure::Timeout),
        ..Native::default()
    };
    fs::rename(
        &f.request.holdout.path,
        f.directory.path().join("holdout-unavailable"),
    )
    .unwrap();
    fs::rename(
        &f.request.test.as_ref().unwrap().path,
        f.directory.path().join("test-unavailable"),
    )
    .unwrap();
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    assert!(finalizer.finalize(&f.request, &failing).is_err());
    let record = f.store.get(&claim_key(&f.plan.run_id)).unwrap().unwrap();
    assert_eq!(record.data["audit"]["entries"].as_array().unwrap().len(), 1);
    assert_eq!(
        record.data["audit"]["entries"][0]["phase"],
        "finalization_fit"
    );
    assert_eq!(
        finalizer
            .journal()
            .snapshot(&f.plan.run_id)
            .unwrap()
            .unwrap()
            .state,
        RunState::Failed
    );
    assert!(finalizer.finalize(&f.request, &Native::default()).is_err());
    assert!(!f.request.model_root.exists());
}

#[test]
fn bad_rosters_refuse_before_claim_and_missing_values_refuse_without_format_drift() {
    let f = fixture(false);
    let native = Native::default();
    search(&f, &native);
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    let mut bad = f.request.clone();
    bad.holdout.rows[0] = bad.fit.rows[0].clone();
    assert!(finalizer.finalize(&bad, &native).is_err());
    assert!(f.store.get(&claim_key(&f.plan.run_id)).unwrap().is_none());
    let f = fixture(true);
    let native = Native::default();
    search(&f, &native);
    let calls = native.calls.lock().unwrap().len();
    let error = Finalizer::new(Arc::clone(&f.store))
        .finalize(&f.request, &native)
        .unwrap_err();
    assert!(error.to_string().contains("complete finite numeric"));
    assert_eq!(calls, native.calls.lock().unwrap().len());
}

#[test]
fn unspecified_test_is_absent_and_an_interrupted_claim_never_becomes_a_zero_score() {
    let mut f = fixture(false);
    f.request.test = None;
    let native = Native::default();
    search(&f, &native);
    let receipt = Finalizer::new(Arc::clone(&f.store))
        .finalize(&f.request, &native)
        .unwrap();
    assert_eq!(receipt.scores.test, None);
    assert_eq!(receipt.test_scope, None);
    let f = fixture(false);
    search(&f, &native);
    let run = f
        .store
        .get(&RecordKey::Run(f.plan.run_id.clone()))
        .unwrap()
        .unwrap();
    let identity = digest(
        &serde_json::to_vec(&(
            &f.request,
            &run.data["checkpoint"]["plan"],
            &run.data["checkpoint"]["champion"],
        ))
        .unwrap(),
    );
    f.store
        .transaction(|tx| {
            tx.put(
                &claim_key(&f.plan.run_id),
                &Record::metadata(json!({"identity":identity,"state":"started"}), vec![]),
            )
        })
        .unwrap();
    let before = native.calls.lock().unwrap().len();
    assert!(Finalizer::new(Arc::clone(&f.store))
        .finalize(&f.request, &native)
        .unwrap_err()
        .to_string()
        .contains("unknown"));
    assert_eq!(before, native.calls.lock().unwrap().len());
    // An operator's explicit Continue is a new authorized attempt, while the
    // interrupted attempt remains durably unknown rather than a zero score.
    let revision = Finalizer::new(Arc::clone(&f.store))
        .journal()
        .snapshot(&f.plan.run_id)
        .unwrap()
        .unwrap()
        .revision
        .0;
    let finalizer = Finalizer::new(Arc::clone(&f.store));
    let resumed = finalizer
        .finalize_after_explicit_continue(&f.request, &native)
        .unwrap();
    assert!(resumed.scores.holdout.unwrap().is_finite());
    let history = f
        .store
        .get(&RecordKey::Command {
            principal: "discover.finalization.interrupted.v1".into(),
            idempotency_key: format!("{}:{revision}", f.plan.run_id),
        })
        .unwrap()
        .unwrap();
    assert_eq!(history.data["outcome"], "unknown");
    assert_eq!(history.data["prior"]["data"]["state"], "started");
    let calls = native.calls.lock().unwrap().len();
    assert_eq!(
        finalizer
            .finalize(&f.request, &native)
            .unwrap()
            .model_digest,
        resumed.model_digest
    );
    assert_eq!(calls, native.calls.lock().unwrap().len());
}
