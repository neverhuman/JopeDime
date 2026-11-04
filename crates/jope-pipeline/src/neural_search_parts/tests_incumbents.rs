#[test]
fn generic_incumbent_takes_final_slot_without_duplicate() -> Result<()> {
    let seed = generic_incumbent()?;
    // Precondition: unseeded, the first generic draw is the incumbent itself.
    let mut fresh = FixtureBackend::default();
    let mut fresh_variation = FixtureVariation {
        started: fresh.started.clone(),
    };
    let unseeded = run_generic(&mut fresh, &comparison(84), &mut fresh_variation)?;
    assert_eq!(
        unseeded.observations[0].candidate.program_sha256,
        seed.program_sha256
    );
    let mut plan = comparison(84);
    plan.incumbents = Some(ArmIncumbents {
        neural: vec![],
        generic: vec![seed.clone()],
    });
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let receipt = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(receipt.observations.len(), 84);
    assert_eq!(
        receipt
            .observations
            .iter()
            .filter(|o| o.candidate.generation == 0)
            .count(),
        80
    );
    let incumbent = &receipt.observations[79].candidate;
    assert_eq!(incumbent.origin, Origin::Incumbent);
    assert_eq!(incumbent.genome, seed.original_proposal);
    assert_eq!(incumbent.initial_slot, Some(79));
    assert!(incumbent.rank_log_weight.is_none());
    let initial = &receipt.observations[..79];
    assert!(initial
        .iter()
        .all(|o| o.candidate.origin == Origin::Generic));
    assert!(!initial
        .iter()
        .any(|o| o.candidate.program_sha256 == seed.program_sha256));
    receipt.validate_population(&plan)?;
    let resumed = run_generic(&mut backend, &plan, &mut variation)?;
    assert_eq!(
        serde_json::to_value(&receipt)?,
        serde_json::to_value(&resumed)?
    );
    assert_eq!(backend.dispatched.len(), 84);
    Ok(())
}

#[test]
fn neural_incumbent_replaces_exploration_tail_only_in_its_own_arm() -> Result<()> {
    // A generic-arm blob is enough here: seating and replay never reread it.
    let seed = generic_incumbent()?;
    let mut plan = comparison(84);
    plan.incumbents = Some(ArmIncumbents {
        neural: vec![seed.clone()],
        generic: vec![],
    });
    let arm = plan.arm(true)?;
    let mut initial = synthetic_initial(&arm);
    seat_incumbents(&arm, &plan, &mut initial)?;
    assert_eq!(initial[79].origin, Origin::Incumbent);
    assert_eq!(initial[79].genome, seed.original_proposal);
    assert_eq!(
        serde_json::to_value(&initial[..79])?,
        serde_json::to_value(&synthetic_initial(&arm)[..79])?
    );
    let mut backend = FixtureBackend::default();
    let mut variation = FixtureVariation {
        started: backend.started.clone(),
    };
    let deadline = backend.begin_arm(&arm)?;
    let receipt = run_population(
        &mut backend,
        &plan,
        &mut variation,
        arm.clone(),
        deadline,
        initial,
        None,
    )?;
    receipt.validate_population(&plan)?;
    let dispatched_initial = backend
        .dispatched
        .iter()
        .filter(|c| c.generation == 0)
        .count();
    assert!(dispatched_initial <= INITIAL_SIZE);
    let mut generation_zero = copy_receipt(&receipt);
    generation_zero.observations.truncate(INITIAL_SIZE);
    generation_zero.validate_population(&plan)?;
    let incumbent = |slot: usize, genome: &str, rank: Option<f64>| {
        Candidate::new(
            &arm,
            genome.into(),
            0,
            slot,
            Origin::Incumbent,
            vec![],
            rank,
        )
    };
    // An undesignated slot, a changed proposal and a policy rank are refused.
    for (slot, candidate) in [
        (0, incumbent(0, &seed.original_proposal, None)?),
        (79, incumbent(79, "(pipe X (ridge lambda=4242))", None)?),
        (79, incumbent(79, &seed.original_proposal, Some(0.))?),
    ] {
        assert!(with_initial(&generation_zero, slot, candidate)
            .validate_population(&plan)
            .is_err());
    }
    // The generic arm of this plan has no seeds.
    let generic = metadata_receipt(&plan, false)?;
    let foreign = Candidate::new(
        &generic.arm,
        seed.original_proposal.clone(),
        0,
        79,
        Origin::Incumbent,
        vec![],
        None,
    )?;
    assert!(with_initial(&generic, 79, foreign)
        .validate_population(&plan)
        .is_err());
    // A plan without incumbents admits none in either arm.
    let unseeded = comparison(84);
    let plain = metadata_receipt(&unseeded, true)?;
    let foreign = Candidate::new(
        &plain.arm,
        seed.original_proposal.clone(),
        0,
        79,
        Origin::Incumbent,
        vec![],
        None,
    )?;
    assert!(with_initial(&plain, 79, foreign)
        .validate_population(&unseeded)
        .is_err());
    // A policy slot is never displaced.
    let mut policy_tail = synthetic_initial(&arm);
    policy_tail[79] = Candidate::new(
        &arm,
        policy_tail[79].genome.clone(),
        0,
        79,
        Origin::NeuralPolicy,
        vec![],
        None,
    )?;
    assert!(seat_incumbents(&arm, &plan, &mut policy_tail).is_err());
    // Count and per-arm uniqueness bounds. Distinct recipes isolate the count bound.
    let distinct = |count: usize| {
        (1..=count)
            .map(|i| -> Result<IncumbentSeed> {
                let proposal = format!("(pipe X (ridge lambda={i}))");
                let program_sha256 = SynthesizedRecipe::new(proposal.clone())?.executable_sha256;
                Ok(IncumbentSeed {
                    frozen_recipe_sha256: sha256_hex(proposal.as_bytes()),
                    original_proposal: proposal,
                    program_sha256,
                })
            })
            .collect::<Result<Vec<_>>>()
    };
    let refusal = |incumbents: ArmIncumbents| -> String {
        let mut bounded = comparison(84);
        bounded.incumbents = Some(incumbents);
        bounded
            .arm(true)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
    };
    let mut at_bound = comparison(84);
    at_bound.incumbents = Some(ArmIncumbents {
        neural: distinct(MAX_INCUMBENTS)?,
        generic: distinct(MAX_INCUMBENTS)?,
    });
    at_bound.arm(true)?;
    for incumbents in [
        ArmIncumbents {
            neural: distinct(MAX_INCUMBENTS + 1)?,
            generic: vec![],
        },
        ArmIncumbents {
            neural: vec![],
            generic: distinct(MAX_INCUMBENTS + 1)?,
        },
        ArmIncumbents {
            neural: vec![],
            generic: vec![],
        },
    ] {
        assert!(refusal(incumbents).contains("count outside approved bounds"));
    }
    assert!(refusal(ArmIncumbents {
        neural: vec![seed.clone(), seed.clone()],
        generic: vec![],
    })
    .contains("repeated within an arm"));
    // An incumbent root never establishes neural-policy lineage.
    let mut descendant = copy_receipt(&generation_zero);
    append_metadata_child(&mut descendant, &plan, 1, 0, Origin::Mutation, &[79])?;
    descendant.validate_population(&plan)?;
    assert!(!descendant.has_fitted_neural_descendant());
    Ok(())
}

#[test]
fn winner_lineage_marks_policy_roots_and_descendants_only() -> Result<()> {
    let plan = comparison(84);
    let mut receipt = metadata_receipt(&plan, true)?;
    let id = |receipt: &SearchReceipt, index: usize| {
        receipt.observations[index].candidate.id_sha256.clone()
    };
    assert!(receipt.neural_policy_lineage(&id(&receipt, 0)));
    assert!(!receipt.neural_policy_lineage(&id(&receipt, 62)));
    assert!(!receipt.neural_policy_lineage(&id(&receipt, 64)));
    append_metadata_child(&mut receipt, &plan, 1, 0, Origin::Mutation, &[64])?;
    append_metadata_child(&mut receipt, &plan, 1, 1, Origin::Crossover, &[60, 65])?;
    append_metadata_child(&mut receipt, &plan, 2, 0, Origin::Mutation, &[80])?;
    assert!(!receipt.neural_policy_lineage(&id(&receipt, 80)));
    assert!(!receipt.has_fitted_neural_descendant());
    append_metadata_child(&mut receipt, &plan, 2, 1, Origin::Crossover, &[81, 0])?;
    assert!(receipt.neural_policy_lineage(&id(&receipt, 83)));
    assert!(receipt.has_fitted_neural_descendant());
    Ok(())
}

/// W2-T02 G-03 / W2-T07: a comparison attaches only encoding outputs to its preregistered
/// plan: the context digest and, for a qualified CPU context, the controller-issued
/// admission digest the worker must verify against.
#[test]
fn comparison_attaches_only_encoding_outputs_including_the_issued_admission() -> Result<()> {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    let scratch = target.join("tmp");
    std::fs::create_dir_all(&scratch)?;
    let temp = tempfile::Builder::new()
        .prefix("comparison-admission-fixture-")
        .tempdir_in(scratch)?;
    let path = std::fs::canonicalize(temp.path())?.join("receipt.json");
    std::fs::write(&path, b"synthetic production receipt")?;
    let receipt = crate::execution::Artifact {
        sha256: sha256_hex(b"synthetic production receipt"),
        path,
    };
    let plan = comparison(80);
    let accepted = plan.proposal.accepted()?.clone();
    let with = |request: jope_policy::producer::Request| Comparison {
        input_plan: plan.clone(),
        producer_request: request,
        context_receipt_sha256: sha256_hex(b"context receipt"),
        context_cost: ContextCost {
            cached: false,
            production_receipt: receipt.clone(),
            original_wall_seconds: Some(1.),
            original_cpu_seconds: Some(1.),
            original_gpu_seconds: Some(0.),
            amortization_uses: Some(1),
        },
    };
    let context = sha256_hex(b"context");
    let pins = sha256_hex(b"controller-issued admission pins");
    with(accepted.request(context.clone())).validate()?;
    let issued = accepted.qualified_request(context.clone(), pins.clone());
    with(issued.clone()).validate()?;
    assert_eq!(
        issued.context_admission_sha256.as_deref(),
        Some(pins.as_str())
    );
    // Anything else that differs from the plan is still refused, admission or not.
    let mut changed = issued.clone();
    changed.sampling_seed += 1;
    assert!(with(changed).validate().is_err());
    let mut malformed = issued;
    malformed.context_admission_sha256 = Some(pins.to_ascii_uppercase());
    assert!(with(malformed).validate().is_err());
    Ok(())
}
