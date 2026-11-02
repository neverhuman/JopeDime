#[test]
fn cancellation_at_cutoff_preserves_observed_generation_on_resume() -> Result<()> {
    interrupted_observed_generation_replays(true)
}

#[test]
fn crash_after_observed_generation_preserves_completed_fits_on_resume() -> Result<()> {
    interrupted_observed_generation_replays(false)
}

fn interrupted_observed_generation_replays(cancelled: bool) -> Result<()> {
    let plan = comparison(80);
    let arm = plan.arm(true)?;
    let initial = synthetic_initial(&arm);
    let mut backend = FixtureBackend::default();
    let deadline = backend.begin_arm(&arm)?;
    backend.evaluate(&arm, &plan.selection, &initial[..79])?;
    backend.admission_unavailable = true;
    backend.expire_during_admission = true;
    backend.cancelled = cancelled;
    backend.interrupt_after_observed = !cancelled;
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let first = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial.clone(),
        None,
    );
    if cancelled {
        assert!(first.is_ok());
    } else {
        assert!(first.is_err());
        assert!(!backend
            .journal
            .contains_key(&format!("{}/search-receipt", arm.id_sha256)));
    }
    let observed_key = format!("{}/generation-0000-observed", arm.id_sha256);
    let original_observed = backend.journal[&observed_key].clone();
    backend.cancelled = false;
    backend.time += 100;
    if !cancelled {
        let completed_id = initial[0].id_sha256.clone();
        let pending_id = initial.last().unwrap().id_sha256.clone();
        let original = backend.cache[&completed_id].clone();
        let mut partial = Feedback::exhausted(initial.last().unwrap());
        partial.physical_fit_starts = None;
        partial.worker_receipt_sha256 = Some(sha256_hex(b"changed partial fold"));
        backend.cache.insert(pending_id.clone(), partial);
        assert!(run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm.clone(),
            deadline,
            initial.clone(),
            None,
        )
        .is_err());
        let mut forged = original.clone();
        forged.candidate_sha256 = pending_id.clone();
        backend.cache.insert(pending_id.clone(), forged);
        assert!(run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm.clone(),
            deadline,
            initial.clone(),
            None,
        )
        .is_err());
        backend.cache.remove(&pending_id);
        backend.cache.get_mut(&completed_id).unwrap().loss = Some(999.);
        assert!(run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm.clone(),
            deadline,
            initial.clone(),
            None,
        )
        .is_err());
        backend.cache.insert(completed_id, original);
        assert_eq!(backend.journal[&observed_key], original_observed);
    }
    let resumed = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial,
        None,
    )?;
    assert_eq!(resumed.stop_reason, "original_deadline");
    assert_eq!(resumed.deadline_unix_ms, deadline);
    assert_eq!(backend.deadlines.len(), 1);
    assert_eq!(backend.dispatched.len(), 79);
    assert_eq!(backend.journal[&observed_key], original_observed);
    assert_eq!(
        resumed
            .observations
            .iter()
            .filter(|o| o.feedback.status == EvaluationStatus::Completed)
            .count(),
        79
    );
    assert!(freeze_winner(&mut backend, &plan, &resumed)?.is_some());
    assert!(backend
        .journal
        .contains_key(&format!("{}/search-receipt", arm.id_sha256)));
    Ok(())
}

#[test]
fn canonical_alias_slot_keeps_origin_without_duplicate_fit() -> Result<()> {
    let plan = comparison(80);
    let arm = plan.arm(true)?;
    let mut backend = FixtureBackend::default();
    let deadline = backend.begin_arm(&arm)?;
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let mut initial = synthetic_initial(&arm);
    initial[1] = Candidate::new(
        &arm,
        " (pipe X (ridge   lambda=1)) ".into(),
        0,
        1,
        Origin::NeuralPolicy,
        vec![],
        None,
    )?;
    assert_eq!(initial[0].program_sha256, initial[1].program_sha256);
    let receipt = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm,
        deadline,
        initial,
        None,
    )?;
    assert_eq!(receipt.observations.len(), 80);
    assert_eq!(backend.dispatched.len(), 79);
    assert_eq!(
        receipt.observations[1].candidate.origin,
        Origin::NeuralPolicy
    );
    assert_eq!(
        receipt.observations[1].feedback.physical_fit_starts,
        Some(0)
    );
    assert!(receipt.observations[1]
        .feedback
        .reason
        .as_ref()
        .unwrap()
        .starts_with("canonical_duplicate_of:"));
    Ok(())
}
#[test]
fn outer_score_cannot_rank_and_refit_requires_frozen_oof() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let receipt = run_generic(&mut backend, &plan, &mut variation)?;
    let frozen = freeze_winner(&mut backend, &plan, &receipt)?.unwrap();
    assert_eq!(
        frozen.refit_request.fitting_rows_sha256,
        plan.selection.outer().fitting_rows_sha256
    );
    assert_eq!(
        frozen.refit_request.scoring_rows_sha256,
        plan.selection.outer().scoring_rows_sha256
    );
    assert_eq!(
        frozen.refit_request.recipe_sha256,
        frozen.candidate.program_sha256
    );
    let winner = receipt.winner().unwrap();
    let mut bad = winner.feedback.clone();
    bad.oof.as_mut().unwrap().scored_rows_sha256 =
        plan.selection.outer().scoring_rows_sha256.clone();
    bad.enforce_selection(&plan.selection, &winner.candidate);
    assert_eq!(bad.status, EvaluationStatus::Failed);
    assert!(bad.loss.is_none());
    Ok(())
}
#[test]
fn unavailable_neural_context_still_runs_generic_and_preserves_early_clock() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let (arm, deadline) = begin_pair(&mut backend, &plan, &variation)?;
    backend.time += 1000;
    let result = run_pair(&mut backend, &plan, &mut variation, None)?;
    assert!(result.neural.is_none());
    assert!(result.neural_error.is_some());
    assert_eq!(backend.deadlines[&arm.id_sha256], deadline);
    let generic = result.generic.unwrap();
    assert_eq!(generic.observations.len(), 80);
    assert_eq!(generic.arm.seconds, arm.seconds);
    Ok(())
}

fn generic_frozen() -> Result<(PairInputPlan, FrozenRecipe)> {
    // Generation zero only: offspring tie slot 0 on the fixture loss, and the
    // generic sampler never draws an offspring recipe, so an offspring seed
    // could not exercise the generic pre-seeding.
    let plan = comparison(80);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let receipt = run_generic(&mut backend, &plan, &mut variation)?;
    let frozen = freeze_winner(&mut backend, &plan, &receipt)?.context("generic winner")?;
    Ok((plan, frozen))
}
fn repinned_seed(
    value: &serde_json::Value,
    selection: &SelectionPlan,
    neural: bool,
) -> Result<IncumbentSeed> {
    let bytes = serde_json::to_vec(value)?;
    IncumbentSeed::from_frozen_recipe(&bytes, &sha256_hex(&bytes), selection, neural)
}
fn generic_incumbent() -> Result<IncumbentSeed> {
    let (plan, frozen) = generic_frozen()?;
    let bytes = serde_json::to_vec(&frozen)?;
    IncumbentSeed::from_frozen_recipe(&bytes, &sha256_hex(&bytes), &plan.selection, false)
}
fn with_initial(receipt: &SearchReceipt, slot: usize, candidate: Candidate) -> SearchReceipt {
    let mut changed = copy_receipt(receipt);
    changed.observations[slot] = Observation {
        feedback: Feedback::exhausted(&candidate),
        candidate,
    };
    changed
}

fn challenger() -> jope_policy::config::ChallengerIdentity {
    jope_policy::config::ChallengerIdentity {
        state_sha256: sha256_hex(b"challenger state"),
        parameter_sha256: sha256_hex(b"challenger parameters"),
        config_sha256: jope_policy::ACCEPTED_CONFIG_SHA256.into(),
        policy_safetensors_sha256: sha256_hex(b"challenger weights"),
    }
}

#[test]
fn challenger_proposal_changes_arms_and_refuses_incumbents() -> Result<()> {
    let p = comparison(80);
    assert!(!String::from_utf8(serde_json::to_vec(&p)?)?.contains("\"challenger\""));
    let c = challenger();
    let mut q = comparison(80);
    match &mut q.proposal {
        ProposalPlan::Accepted(value) => {
            value.checkpoint_sha256 = c.state_sha256.clone();
            value.policy_safetensors_sha256 = c.policy_safetensors_sha256.clone();
            value.challenger = Some(c.clone());
        }
        #[cfg(feature = "qualified-context")]
        ProposalPlan::Experimental(_) => unreachable!(),
    }
    for neural in [true, false] {
        assert_ne!(q.arm(neural)?.id_sha256, p.arm(neural)?.id_sha256);
    }
    // The untagged plan still decodes as the accepted variant, carrying the identity.
    let decoded: PairInputPlan = serde_json::from_slice(&serde_json::to_vec(&q)?)?;
    assert_eq!(decoded.proposal.accepted()?.challenger, Some(c.clone()));
    assert_eq!(
        decoded.proposal.checkpoint_sha256(),
        c.state_sha256.as_str()
    );
    assert_eq!(serde_json::to_vec(&decoded)?, serde_json::to_vec(&q)?);
    // A challenger comparison seeds no incumbents.
    q.incumbents = Some(ArmIncumbents {
        neural: vec![generic_incumbent()?],
        generic: vec![],
    });
    let error = q.arm(true).unwrap_err();
    assert!(
        format!("{error:#}").contains("incumbent seeding requires the accepted proposal plan"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn absent_incumbents_keep_pair_bytes() -> Result<()> {
    let plan = comparison(80);
    let bytes = serde_json::to_vec(&plan)?;
    assert!(!String::from_utf8_lossy(&bytes).contains("\"incumbents\""));
    let decoded: PairInputPlan = serde_json::from_slice(&bytes)?;
    assert_eq!(serde_json::to_vec(&decoded)?, bytes);
    for neural in [true, false] {
        assert_eq!(decoded.arm(neural)?.id_sha256, plan.arm(neural)?.id_sha256);
    }
    Ok(())
}

#[test]
fn incumbent_blob_is_verified_for_digest_arm_and_rows() -> Result<()> {
    let (plan, frozen) = generic_frozen()?;
    let bytes = serde_json::to_vec(&frozen)?;
    let pin = sha256_hex(&bytes);
    let seed = IncumbentSeed::from_frozen_recipe(&bytes, &pin, &plan.selection, false)?;
    assert_eq!(seed.original_proposal, frozen.candidate.genome);
    assert_eq!(seed.program_sha256, frozen.candidate.program_sha256);
    assert_eq!(seed.frozen_recipe_sha256, pin);
    let admits = |bytes: &[u8], pin: &str, selection: &SelectionPlan, neural: bool| {
        IncumbentSeed::from_frozen_recipe(bytes, pin, selection, neural).is_ok()
    };
    // Bytes that do not hash to the pin.
    assert!(!admits(
        &bytes,
        &sha256_hex(b"other"),
        &plan.selection,
        false
    ));
    // A changed proposal, re-pinned, no longer derives its recorded recipe.
    let original = serde_json::to_value(&frozen)?;
    let mut tampered = original.clone();
    tampered["candidate"]["genome"] = serde_json::json!("(pipe X (ridge lambda=999))");
    let error = repinned_seed(&tampered, &plan.selection, false).unwrap_err();
    assert!(format!("{error:#}").contains("recipe derivation"));
    // The checkpoint pin names the arm, in both directions.
    assert!(!admits(&bytes, &pin, &plan.selection, true));
    let mut neural = original.clone();
    neural["refit_request"]["checkpoint_sha256"] =
        serde_json::json!(jope_policy::ACCEPTED_STATE_SHA256);
    repinned_seed(&neural, &plan.selection, true)?;
    assert!(repinned_seed(&neural, &plan.selection, false).is_err());
    // Another dataset or another evaluation seed.
    for (key, value) in [
        (
            "dataset_sha256",
            serde_json::json!(sha256_hex(b"other dataset")),
        ),
        ("seed", serde_json::json!(plan.selection.outer().seed + 1)),
    ] {
        let mut selection = serde_json::to_value(&plan.selection)?;
        selection["outer"][key] = value;
        let other: SelectionPlan = serde_json::from_value(selection)?;
        other.validate()?;
        assert!(!admits(&bytes, &pin, &other, false));
    }
    // A failed train-side selection.
    let mut failed = original;
    failed["selection_feedback"]["status"] = serde_json::json!("failed");
    assert!(repinned_seed(&failed, &plan.selection, false).is_err());
    Ok(())
}
