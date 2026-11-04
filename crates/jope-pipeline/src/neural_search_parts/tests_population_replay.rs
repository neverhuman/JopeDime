#[test]
fn original_expired_window_admits_no_fits_and_does_not_reset() -> Result<()> {
    let plan = comparison(80);
    let arm = plan.arm(true)?;
    let mut backend = FixtureBackend::default();
    let deadline = backend.begin_arm(&arm)?;
    backend.time = deadline + 1;
    assert_eq!(backend.begin_arm(&arm)?, deadline);
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let result = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        synthetic_initial(&arm),
        None,
    )?;
    assert!(backend.dispatched.is_empty());
    assert_eq!(result.observations.len(), 80);
    assert!(result
        .observations
        .iter()
        .all(|o| o.feedback.status == EvaluationStatus::BudgetExhausted));
    assert!(result.winner().is_none());
    Ok(())
}

#[test]
fn unknown_physical_fit_is_not_a_win_or_an_evolution_parent() -> Result<()> {
    let plan = comparison(84);
    let mut backend = FixtureBackend {
        unknown_counter: true,
        ..Default::default()
    };
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let report = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(report.observations.len(), 80);
    assert_eq!(report.stop_reason, "no_measured_parent");
    assert!(report.winner().is_none());
    assert!(report
        .observations
        .iter()
        .all(|o| o.feedback.physical_fit_starts.is_none()));
    Ok(())
}

#[test]
fn resume_rechecks_cached_artifacts_and_keeps_original_window() -> Result<()> {
    let plan = comparison(84);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let first = run_generic(&mut backend, &plan, &mut variation)?;
    backend.time = first.deadline_unix_ms + 5000;
    let second = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(backend.dispatched.len(), 84);
    assert_eq!(
        serde_json::to_value(&first)?,
        serde_json::to_value(&second)?
    );
    backend
        .cache
        .values_mut()
        .next()
        .unwrap()
        .fitted_artifact_sha256 = Some(sha256_hex(b"changed"));
    assert!(run_generic(&mut backend, &plan, &mut variation).is_err());
    Ok(())
}

#[test]
fn partial_resume_replays_scheduled_generation_without_new_deadline() -> Result<()> {
    let plan = comparison(84);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let first = run_generic(&mut backend, &plan, &mut variation)?;
    backend
        .journal
        .remove(&format!("{}/search-receipt", first.arm.id_sha256));
    backend.time = first.deadline_unix_ms + 1;
    let resumed = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(resumed.observations.len(), 84);
    assert_eq!(resumed.deadline_unix_ms, first.deadline_unix_ms);
    assert_eq!(backend.dispatched.len(), 84);
    Ok(())
}

#[test]
fn nonfinite_scores_never_rank() {
    let candidate = synthetic_initial(&comparison(80).arm(true).unwrap()).remove(0);
    let mut f = Feedback {
        candidate_sha256: candidate.id_sha256,
        status: EvaluationStatus::Completed,
        loss: Some(f64::NAN),
        physical_fit_starts: Some(1),
        worker_receipt_sha256: Some(sha256_hex(b"worker")),
        replay_evidence_sha256: None,
        fitted_artifact_sha256: Some(sha256_hex(b"artifact")),
        reason: None,
        oof: None,
        retryable_inventory: false,
    };
    f.enforce_measurement();
    assert_eq!(f.status, EvaluationStatus::Failed);
    assert!(f.loss.is_none());
}
#[test]
fn pair_plan_is_frozen_before_clocks_and_rejects_changed_budget_or_sampling() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend::default();
    let variation = FixtureVariation {
        started: backend.started.clone(),
    };
    freeze_pair(&mut backend, &plan, &variation)?;
    assert!(!backend.started.get());
    assert_eq!(backend.journal.len(), 3);
    let mut changed = comparison(84);
    assert!(freeze_pair(&mut backend, &changed, &variation).is_err());
    changed = comparison(80);
    match &mut changed.proposal {
        ProposalPlan::Accepted(value) => value.sampling_seed += 1,
        #[cfg(feature = "qualified-context")]
        ProposalPlan::Experimental(_) => unreachable!(),
    }
    assert!(freeze_pair(&mut backend, &changed, &variation).is_err());
    Ok(())
}
#[test]
fn unavailable_nonattempt_does_not_poison_resume() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend {
        admission_unavailable: true,
        ..Default::default()
    };
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let pending = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(pending.stop_reason, "admission_pending_original_window");
    assert!(pending.winner().is_none());
    assert!(!backend
        .journal
        .keys()
        .any(|k| k.ends_with("search-receipt") || k.ends_with("observed")));
    backend.time += 100;
    let resumed = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(resumed.deadline_unix_ms, pending.deadline_unix_ms);
    assert_eq!(backend.dispatched.len(), 80);
    assert!(resumed.winner().is_some());
    Ok(())
}
#[test]
fn nonattempt_at_original_arm_cutoff_finalizes_without_new_deadline() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend {
        admission_unavailable: true,
        expire_during_admission: true,
        ..Default::default()
    };
    let original_start = backend.time;
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let receipt = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(
        receipt.deadline_unix_ms,
        original_start + plan.seconds * 1000
    );
    assert_eq!(receipt.stop_reason, "original_deadline");
    assert_eq!(receipt.observations.len(), 80);
    assert!(receipt
        .observations
        .iter()
        .all(|o| o.feedback.status == EvaluationStatus::Unavailable
            && o.feedback.retryable_inventory));
    assert!(backend.dispatched.is_empty());
    assert!(backend
        .journal
        .contains_key(&format!("{}/search-receipt", receipt.arm.id_sha256)));
    assert!(backend.journal.contains_key(&format!(
        "{}/generation-0000-observed",
        receipt.arm.id_sha256
    )));
    assert!(freeze_winner(&mut backend, &plan, &receipt)?.is_none());
    backend.time += 100;
    let resumed = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(
        serde_json::to_value(&receipt)?,
        serde_json::to_value(&resumed)?
    );
    assert!(backend.dispatched.is_empty());
    assert_eq!(backend.deadlines.len(), 1);
    Ok(())
}

#[test]
fn expired_nonattempt_preserves_completed_winner_and_resume_integrity() -> Result<()> {
    let plan = comparison(80);
    let arm = plan.arm(true)?;
    let initial = synthetic_initial(&arm);
    let mut backend = FixtureBackend::default();
    let deadline = backend.begin_arm(&arm)?;
    // Synthetic retained callback outcomes from the same original window.
    backend.evaluate(&arm, &plan.selection, &initial[..79])?;
    backend.admission_unavailable = true;
    backend.expire_during_admission = true;
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let receipt = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial.clone(),
        None,
    )?;
    assert_eq!(receipt.stop_reason, "original_deadline");
    assert_eq!(
        receipt
            .observations
            .iter()
            .filter(|o| o.feedback.status == EvaluationStatus::Completed)
            .count(),
        79
    );
    assert_eq!(backend.dispatched.len(), 79);
    let frozen = freeze_winner(&mut backend, &plan, &receipt)?.expect("retained completed winner");
    assert_eq!(frozen.candidate.origin, Origin::NeuralPolicy);
    backend.time += 100;
    let resumed = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial.clone(),
        None,
    )?;
    assert_eq!(
        serde_json::to_value(&receipt)?,
        serde_json::to_value(&resumed)?
    );
    assert_eq!(backend.dispatched.len(), 79);
    assert_eq!(backend.deadlines[&arm.id_sha256], deadline);
    // A later cache inconsistency cannot turn the original nonattempt into
    // a measurement hidden by the expiry exception.
    let pending = initial.last().unwrap();
    let mut forged = backend.cache[&frozen.candidate.id_sha256].clone();
    forged.candidate_sha256 = pending.id_sha256.clone();
    backend.cache.insert(pending.id_sha256.clone(), forged);
    assert!(run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial.clone(),
        None
    )
    .is_err());
    backend.cache.remove(&pending.id_sha256);
    // The expiry exception must never exempt a completed predictor.
    backend
        .cache
        .get_mut(&frozen.candidate.id_sha256)
        .unwrap()
        .loss = Some(999.);
    assert!(run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm,
        deadline,
        initial,
        None
    )
    .is_err());
    Ok(())
}

#[test]
fn partial_fold_replay_authenticates_every_measured_outcome() -> Result<()> {
    use crate::execution::{Artifact, ObservedConsumption, EXECUTION_SCHEMA};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    std::fs::create_dir_all(&root)?;
    let scratch = tempfile::tempdir_in(root.canonicalize()?)?;
    let output = scratch.path().join("fitted.json");
    std::fs::write(&output, b"retained fitted state")?;
    let artifact = Artifact {
        path: output.clone(),
        sha256: sha256_hex(b"retained fitted state"),
    };
    let mut folds = (0..5)
        .map(|fold| WorkerOutcome {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: sha256_hex(format!("job {fold}").as_bytes()),
            request_sha256: sha256_hex(format!("request {fold}").as_bytes()),
            command_status: EvaluationStatus::Completed,
            reason: None,
            exit_code: Some(0),
            signal: None,
            outputs: vec![artifact.clone()],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                gate_released_unix_ms: Some(10),
                observed_exit_unix_ms: Some(20),
                cpu_seconds: Some(0.01),
                wall_seconds: Some(0.01),
                accounting_reason: "synthetic measured fixture".into(),
                ..Default::default()
            },
        })
        .collect::<Vec<_>>();
    folds[4].command_status = EvaluationStatus::Cancelled;
    folds[4].exit_code = None;
    folds[4].outputs.clear();
    folds[4].consumption = ObservedConsumption {
        accounting_reason: "launch failed before gate release; gate/setup CPU unavailable".into(),
        ..Default::default()
    };
    let digest = fold_replay_evidence(&folds)?;
    let mut resumed = folds.clone();
    resumed[4].command_status = EvaluationStatus::BudgetExhausted;
    resumed[4].reason = Some("original cutoff".into());
    resumed[4].consumption = ObservedConsumption {
        cpu_seconds: Some(0.),
        wall_seconds: Some(0.),
        accounting_reason: "not_started: gate never released".into(),
        ..Default::default()
    };
    assert_eq!(digest, fold_replay_evidence(&resumed)?);
    // A partial candidate can retain unknown physical starts while its four
    // measured folds must match exactly across cancellation and expiry.
    let arm = comparison(80).arm(true)?;
    let candidate = synthetic_initial(&arm).remove(0);
    let mut feedback = Feedback::exhausted(&candidate);
    feedback.status = EvaluationStatus::Cancelled;
    feedback.physical_fit_starts = None;
    feedback.retryable_inventory = true;
    feedback.replay_evidence_sha256 = Some(digest.clone());
    let saved = vec![Observation {
        candidate,
        feedback: feedback.clone(),
    }];
    feedback.status = EvaluationStatus::BudgetExhausted;
    feedback.worker_receipt_sha256 = Some(sha256_hex(b"new raw inventory"));
    verify_observed_replay(&saved, &[feedback.clone()], true)?;
    for index in 0..4 {
        let mut changed = resumed.clone();
        changed[index].consumption.cpu_seconds = Some(123.);
        feedback.replay_evidence_sha256 = Some(fold_replay_evidence(&changed)?);
        assert!(verify_observed_replay(&saved, &[feedback.clone()], true).is_err());
    }
    feedback.replay_evidence_sha256 = None;
    assert!(verify_observed_replay(&saved, &[feedback], true).is_err());
    let mut swapped = resumed.clone();
    swapped.swap(0, 1);
    assert_ne!(digest, fold_replay_evidence(&swapped)?);
    assert!(fold_replay_evidence(&resumed[..4]).is_err());
    let mut duplicate = resumed.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(fold_replay_evidence(&duplicate).is_err());
    // Unknown controller interruption, logs, or any measurement cannot be
    // normalized solely because the candidate status says unavailable.
    for variant in 0..6 {
        let mut changed = folds.clone();
        match variant {
            0 => changed[4].consumption.accounting_reason = "unknown interrupted attempt".into(),
            1 => changed[4].stdout = Some(artifact.clone()),
            2 => changed[4].consumption.observed_exit_unix_ms = Some(20),
            3 => changed[4].consumption.cpu_seconds = Some(1.),
            4 => changed[4].consumption.wall_seconds = Some(1.),
            _ => changed[4].consumption.cgroup_peak_memory_bytes = Some(1),
        }
        assert_ne!(digest, fold_replay_evidence(&changed)?);
    }
    std::fs::write(output, b"changed fitted state")?;
    assert!(fold_replay_evidence(&resumed).is_err());
    Ok(())
}
