/// W4-T03/T07: a generation-zero policy slot may name the proposal-ledger record of its
/// draw without changing its identity; no other candidate may name one, and a candidate
/// without one (an older ledger) keeps its meaning.
#[test]
fn only_policy_slots_name_their_proposal_records() -> Result<()> {
    let plan = comparison(84);
    let original = metadata_receipt(&plan, true)?;
    let mut joined = copy_receipt(&original);
    for (slot, o) in joined.observations.iter_mut().enumerate() {
        o.candidate.proposal_record_sha256 = Some(sha256_hex(format!("record {slot}").as_bytes()));
    }
    // The record is not part of the identity; older candidates without one keep theirs.
    assert_eq!(
        joined.observations[0].candidate.id_sha256,
        original.observations[0].candidate.id_sha256
    );
    original.validate_population(&plan)?;
    joined.validate_population(&plan)?;
    joined.observations[0]
        .candidate
        .validate_initial(&joined.arm)?;
    let refused = |receipt: &SearchReceipt, plan: &PairInputPlan| {
        let error = format!("{:#}", receipt.validate_population(plan).unwrap_err());
        assert!(
            error.contains("outside a generation-zero policy slot"),
            "{error}"
        );
    };
    // An offspring claiming a proposal record.
    let mut offspring = copy_receipt(&joined);
    append_metadata_child(&mut offspring, &plan, 1, 0, Origin::Mutation, &[0])?;
    offspring.validate_population(&plan)?;
    offspring.observations[80].candidate.proposal_record_sha256 = Some(sha256_hex(b"record"));
    refused(&offspring, &plan);
    // A malformed record on a policy slot.
    let mut malformed = copy_receipt(&joined);
    malformed.observations[0].candidate.proposal_record_sha256 = Some("not a digest".into());
    refused(&malformed, &plan);
    assert!(malformed.observations[0]
        .candidate
        .validate_initial(&malformed.arm)
        .is_err());
    // A generic candidate claiming one.
    let mut generic = metadata_receipt(&plan, false)?;
    generic.validate_population(&plan)?;
    generic.observations[0].candidate.proposal_record_sha256 = Some(sha256_hex(b"record"));
    refused(&generic, &plan);
    Ok(())
}

// Metadata-only fixtures: callback counters/hashes are explicitly synthetic,
// not receipts from models or fitted workers.
fn metadata_receipt(plan: &PairInputPlan, neural: bool) -> Result<SearchReceipt> {
    let arm = plan.arm(neural)?;
    let candidates = if neural {
        synthetic_initial(&arm)
    } else {
        (0..INITIAL_SIZE)
            .map(|slot| {
                Candidate::new(
                    &arm,
                    format!("(pipe X (ridge lambda={}))", slot + 1),
                    0,
                    slot,
                    Origin::Generic,
                    vec![],
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };
    let mut backend = FixtureBackend::default();
    let deadline = backend.begin_arm(&arm)?;
    let feedback = backend.evaluate(&arm, &plan.selection, &candidates)?;
    Ok(SearchReceipt {
        schema: PROTOCOL.into(),
        comparison_sha256: sha256_hex(&serde_json::to_vec(plan)?),
        selection_sha256: plan.selection.identity()?,
        arm,
        deadline_unix_ms: deadline,
        proposal: None,
        observations: candidates
            .into_iter()
            .zip(feedback)
            .map(|(candidate, feedback)| Observation {
                candidate,
                feedback,
            })
            .collect(),
        stop_reason: "synthetic_metadata_fixture".into(),
    })
}
fn append_metadata_child(
    receipt: &mut SearchReceipt,
    plan: &PairInputPlan,
    generation: usize,
    slot: usize,
    origin: Origin,
    parents: &[usize],
) -> Result<()> {
    let candidate = Candidate::new(
        &receipt.arm,
        format!(
            "(pipe X (standardize) (ridge lambda={}))",
            receipt.observations.len() + 1000
        ),
        generation,
        slot,
        origin,
        parents
            .iter()
            .map(|i| receipt.observations[*i].candidate.id_sha256.clone())
            .collect(),
        None,
    )?;
    let mut backend = FixtureBackend::default();
    backend.begin_arm(&receipt.arm)?;
    let feedback = backend
        .evaluate(
            &receipt.arm,
            &plan.selection,
            std::slice::from_ref(&candidate),
        )?
        .remove(0);
    receipt.observations.push(Observation {
        candidate,
        feedback,
    });
    Ok(())
}
fn copy_receipt(receipt: &SearchReceipt) -> SearchReceipt {
    serde_json::from_value(serde_json::to_value(receipt).unwrap()).unwrap()
}
fn legacy_candidate_checks_pass(receipt: &SearchReceipt) -> bool {
    receipt.observations.iter().all(|o| {
        o.candidate.recipe.validate().is_ok()
            && o.candidate.program_sha256 == o.candidate.recipe.executable_sha256
            && o.feedback.candidate_sha256 == o.candidate.id_sha256
    })
}

#[test]
fn population_replay_rejects_forged_identity_and_duplicate_or_missing_slots() -> Result<()> {
    let plan = comparison(84);
    let original = metadata_receipt(&plan, true)?;
    original.validate_population(&plan)?;
    let mut forged = copy_receipt(&original);
    forged.observations[0].candidate.id_sha256 = sha256_hex(b"forged candidate");
    forged.observations[0].feedback.candidate_sha256 =
        forged.observations[0].candidate.id_sha256.clone();
    assert!(legacy_candidate_checks_pass(&forged)); // concrete old report gap
    assert!(forged.validate_population(&plan).is_err());
    let mut repeated = copy_receipt(&original);
    repeated.observations[1] = repeated.observations[0].clone();
    assert!(legacy_candidate_checks_pass(&repeated));
    assert!(repeated.validate_population(&plan).is_err());
    let mut missing = copy_receipt(&original);
    missing.observations.remove(79);
    assert!(legacy_candidate_checks_pass(&missing));
    assert!(missing.validate_population(&plan).is_err());
    let mut wrong_slot = copy_receipt(&original);
    wrong_slot.observations[1].candidate.initial_slot = Some(0);
    assert!(legacy_candidate_checks_pass(&wrong_slot));
    assert!(wrong_slot.validate_population(&plan).is_err());
    let mut wrong_genome = copy_receipt(&original);
    wrong_genome.observations[0].candidate.genome = "(pipe X (ridge lambda=999))".into();
    assert!(legacy_candidate_checks_pass(&wrong_genome));
    assert!(wrong_genome.validate_population(&plan).is_err());
    let mut wrong_origin = copy_receipt(&original);
    let candidate = Candidate::new(
        &original.arm,
        original.observations[0].candidate.genome.clone(),
        0,
        0,
        Origin::RandomExploration,
        vec![],
        None,
    )?;
    wrong_origin.observations[0] = Observation {
        feedback: Feedback::exhausted(&candidate),
        candidate,
    };
    assert!(legacy_candidate_checks_pass(&wrong_origin));
    assert!(wrong_origin.validate_population(&plan).is_err());
    Ok(())
}

#[test]
fn offspring_from_exploration_or_fallback_does_not_prove_neural_evolution() -> Result<()> {
    let plan = comparison(84);
    let mut receipt = metadata_receipt(&plan, true)?;
    append_metadata_child(&mut receipt, &plan, 1, 0, Origin::Mutation, &[64])?;
    append_metadata_child(&mut receipt, &plan, 1, 1, Origin::Crossover, &[60, 65])?;
    append_metadata_child(&mut receipt, &plan, 2, 0, Origin::Mutation, &[80])?;
    receipt.validate_population(&plan)?;
    assert!(receipt
        .observations
        .iter()
        .any(|o| o.candidate.origin == Origin::NeuralPolicy
            && o.feedback.status == EvaluationStatus::Completed));
    let legacy_evolved = receipt
        .observations
        .iter()
        .any(|o| o.candidate.generation > 0 && o.feedback.status == EvaluationStatus::Completed);
    assert!(legacy_evolved); // old combined flag falsely claimed neural evolution
    assert!(!receipt.has_fitted_neural_descendant());
    append_metadata_child(&mut receipt, &plan, 2, 1, Origin::Crossover, &[81, 0])?;
    receipt.validate_population(&plan)?;
    assert!(receipt.has_fitted_neural_descendant());
    Ok(())
}

#[test]
fn population_replay_requires_completed_parents_from_earlier_generations() -> Result<()> {
    let plan = comparison(84);
    let mut receipt = metadata_receipt(&plan, true)?;
    append_metadata_child(&mut receipt, &plan, 1, 0, Origin::Mutation, &[0])?;
    receipt.validate_population(&plan)?;
    assert!(receipt.has_fitted_neural_descendant());
    let mut failed_parent = copy_receipt(&receipt);
    failed_parent.observations[0].feedback =
        Feedback::exhausted(&failed_parent.observations[0].candidate);
    assert!(legacy_candidate_checks_pass(&failed_parent));
    assert!(failed_parent.validate_population(&plan).is_err());
    assert!(!failed_parent.has_fitted_neural_descendant());
    let mut same_generation = copy_receipt(&receipt);
    append_metadata_child(&mut same_generation, &plan, 1, 1, Origin::Mutation, &[80])?;
    assert!(legacy_candidate_checks_pass(&same_generation));
    assert!(same_generation.validate_population(&plan).is_err());
    let mut wrong_arity = metadata_receipt(&plan, true)?;
    append_metadata_child(&mut wrong_arity, &plan, 1, 0, Origin::Mutation, &[0, 1])?;
    assert!(legacy_candidate_checks_pass(&wrong_arity));
    assert!(wrong_arity.validate_population(&plan).is_err());
    let mut skipped_generation = metadata_receipt(&plan, true)?;
    append_metadata_child(&mut skipped_generation, &plan, 2, 0, Origin::Mutation, &[0])?;
    assert!(skipped_generation.validate_population(&plan).is_err());
    Ok(())
}

#[test]
fn population_replay_preserves_neural_duplicates_but_rejects_generic_duplicates() -> Result<()> {
    let plan = comparison(80);
    let mut neural = metadata_receipt(&plan, true)?;
    let duplicate = Candidate::new(
        &neural.arm,
        neural.observations[0].candidate.genome.clone(),
        0,
        1,
        Origin::NeuralPolicy,
        vec![],
        None,
    )?;
    neural.observations[1] = Observation {
        feedback: Feedback::exhausted(&duplicate),
        candidate: duplicate,
    };
    neural.validate_population(&plan)?;
    let mut generic = metadata_receipt(&plan, false)?;
    let duplicate = Candidate::new(
        &generic.arm,
        generic.observations[0].candidate.genome.clone(),
        0,
        1,
        Origin::Generic,
        vec![],
        None,
    )?;
    generic.observations[1] = Observation {
        feedback: Feedback::exhausted(&duplicate),
        candidate: duplicate,
    };
    assert!(legacy_candidate_checks_pass(&generic));
    assert!(generic.validate_population(&plan).is_err());
    generic = metadata_receipt(&plan, false)?;
    generic.arm.seconds += 1;
    assert!(generic.validate_population(&plan).is_err());
    Ok(())
}

#[test]
fn resume_rejects_forged_offspring_metadata_before_cached_fit_replay() -> Result<()> {
    let plan = comparison(84);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let mut saved = run_generic(&mut backend, &plan, &mut variation)?;
    let original_deadline = saved.deadline_unix_ms;
    saved.observations[80].candidate.parents = vec![sha256_hex(b"absent parent")];
    // The old completed-receipt path rechecked cached feedback by unchanged
    // candidate ID, but never reconstructed the child's changed parents.
    assert!(legacy_candidate_checks_pass(&saved));
    backend.journal.insert(
        format!("{}/search-receipt", saved.arm.id_sha256),
        serde_json::to_value(saved)?,
    );
    assert!(run_generic(&mut backend, &plan, &mut variation).is_err());
    assert_eq!(backend.dispatched.len(), 84);
    assert_eq!(
        backend.deadlines[&plan.arm(false)?.id_sha256],
        original_deadline
    );
    Ok(())
}

#[test]
fn synthetic_neural_slots_reach_callbacks_and_evolution_with_origins_intact() -> Result<()> {
    let plan = comparison(84);
    let arm = plan.arm(true)?;
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let deadline = backend.begin_arm(&arm)?;
    let report = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        synthetic_initial(&arm),
        None,
    )?;
    assert_eq!(backend.dispatched.len(), 84);
    assert_eq!(
        backend
            .dispatched
            .iter()
            .filter(|c| c.generation == 0)
            .count(),
        80
    );
    assert_eq!(
        backend
            .dispatched
            .iter()
            .filter(|c| c.origin == Origin::NeuralPolicy)
            .count(),
        60
    );
    assert_eq!(
        backend
            .dispatched
            .iter()
            .filter(|c| c.origin == Origin::NeuralRejectionFallback)
            .count(),
        4
    );
    assert_eq!(
        backend
            .dispatched
            .iter()
            .filter(|c| c.origin == Origin::RandomExploration)
            .count(),
        16
    );
    assert_eq!(report.neural_physical_fit_starts(), Some(300));
    for child in &backend.dispatched[80..] {
        assert_eq!(child.generation, 1);
        assert!(child.initial_slot.is_none());
        assert!(child
            .parents
            .iter()
            .all(|p| backend.dispatched[..60].iter().any(|c| c.id_sha256 == *p)));
    }
    assert!(report.winner().is_some());
    Ok(())
}

#[test]
fn generic_has_80_slots_and_identical_limits_clock_precedes_generation() -> Result<()> {
    let plan = comparison(80);
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let generic = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(generic.observations.len(), 80);
    assert!(generic
        .observations
        .iter()
        .all(|o| o.candidate.origin == Origin::Generic));
    assert_eq!(generic.arm.seconds, plan.arm(true)?.seconds);
    assert_eq!(generic.deadline_unix_ms, 121000);
    assert_ne!(generic.arm.id_sha256, plan.arm(true)?.id_sha256);
    assert!(comparison(79).arm(false).is_err());
    Ok(())
}
