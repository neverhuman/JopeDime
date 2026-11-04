/// Read-only validation shared by execution and recovery reporting. No model
/// is loaded and no command/window is admitted here.
pub fn verify_policy_inputs(
    comparison: &Comparison,
    worker: &PolicyWorker,
) -> Result<super::search_context::VerifiedRegressionContext> {
    comparison.validate()?;
    ensure!(
        worker.request == comparison.producer_request
            && worker.expected_program_sha256 == comparison.installation.policy_program_sha256
            && worker.source_manifest.sha256 == comparison.installation.source_manifest_sha256
            && worker.checkpoint_receipt.sha256
                == comparison.installation.checkpoint_receipt_sha256
            && worker.context.receipt().sha256 == comparison.context_receipt_sha256,
        "policy worker differs from frozen paired installation/request"
    );
    worker.release_manifest.verify()?;
    ensure!(
        worker.release_manifest.sha256 == comparison.installation.release_manifest_sha256,
        "installation release receipt changed"
    );
    let context = worker.context.load_for_comparison(comparison)?;
    context.matches(comparison)?;
    worker.request.validate()?;
    ensure!(
        worker.request.prepared_sha256 == comparison.prepared_sha256
            && worker.request.training_sha256 == comparison.training_sha256
            && worker.request.cut == comparison.cut
            && worker.request.context_sha256 == context.sha256()
            && worker.request.vocabulary_sha256 == comparison.vocabulary_sha256,
        "policy request differs from comparison/context"
    );
    // The worker verifies a qualified CPU context against the admission this request issues;
    // it must be exactly the admission the controller verified here (none for a native context).
    ensure!(
        worker.request.context_admission_sha256.as_deref() == context.admission_pins_sha256(),
        "policy request does not issue the controller-verified context admission"
    );
    ensure!(
        worker.job.command.program.sha256 == worker.expected_program_sha256
            && jope_policy::is_sha256_hex(&worker.expected_program_sha256),
        "policy executable differs from installation pin"
    );
    worker.checkpoint_receipt.verify()?;
    // The accepted identity, or exactly the native challenger the request pins.
    let checkpoint = jope_policy::config::CheckpointBinding::for_proposal(
        &std::fs::read(&worker.checkpoint_receipt.path)?,
        worker.request.challenger.as_ref(),
    )?;
    ensure!(
        checkpoint.state_sha256() == worker.request.checkpoint_sha256
            && checkpoint.policy_safetensors_sha256() == worker.request.policy_safetensors_sha256,
        "policy checkpoint receipt changed"
    );
    worker.source_manifest.verify()?;
    ensure!(
        worker.request.worker_source_sha256 == worker.source_manifest.sha256,
        "policy source manifest changed"
    );
    worker.request_artifact.verify()?;
    let actual_request: jope_policy::producer::Request =
        serde_json::from_slice(&std::fs::read(&worker.request_artifact.path)?)?;
    ensure!(
        actual_request == worker.request,
        "policy request artifact differs"
    );
    for input in [
        &worker.source_manifest,
        &worker.request_artifact,
        &worker.checkpoint_receipt,
    ]
    .into_iter()
    .chain(worker.context.inputs())
    {
        ensure!(
            worker
                .job
                .command
                .inputs
                .iter()
                .any(|i| i.path == input.path && i.sha256 == input.sha256),
            "policy input missing from executable closure"
        );
    }
    ensure!(
        worker
            .job
            .command
            .inputs
            .iter()
            .any(|i| i.sha256 == worker.request.policy_safetensors_sha256)
            && worker
                .job
                .command
                .inputs
                .iter()
                .any(|i| i.sha256 == worker.request.vocabulary_sha256),
        "policy weights or vocabulary missing from closure"
    );
    ensure!(
        worker.job.outputs.contains(&worker.output_name),
        "policy output was not declared"
    );
    Ok(context)
}

fn incumbent_candidate(arm: &ArmBudget, seed: &IncumbentSeed, slot: usize) -> Result<Candidate> {
    let candidate = Candidate::new(
        arm,
        seed.original_proposal.clone(),
        0,
        slot,
        Origin::Incumbent,
        vec![],
        None,
    )?;
    ensure!(
        candidate.program_sha256 == seed.program_sha256,
        "incumbent recipe changed"
    );
    Ok(candidate)
}
/// Seat the neural arm's incumbents in the final slots. They may displace only
/// exploration slots; all 64 policy slots and the producer's slots stay bound.
fn seat_incumbents(
    arm: &ArmBudget,
    plan: &PairInputPlan,
    candidates: &mut [Candidate],
) -> Result<()> {
    let seeds = plan.arm_incumbents(true);
    ensure!(
        candidates.len() == INITIAL_SIZE,
        "initial slot count is not 80"
    );
    let first = INITIAL_SIZE - seeds.len();
    for (offset, seed) in seeds.iter().enumerate() {
        let slot = first + offset;
        ensure!(
            candidates[slot].origin == Origin::RandomExploration,
            "incumbent may displace only an exploration slot"
        );
        candidates[slot] = incumbent_candidate(arm, seed, slot)?;
    }
    Ok(())
}

fn run_neural<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &Comparison,
    variation: &mut V,
    worker: &PolicyWorker,
) -> Result<NeuralRun> {
    comparison.validate()?;
    let arm = comparison.arm(true)?;
    let deadline = backend.begin_arm(&arm)?;
    let context = verify_policy_inputs(comparison, worker)?;
    let mut job = worker.job.clone();
    job.arm = Some(arm.clone());
    let expected_job = job.identity()?;
    let expected_request = job.request.identity()?;
    let outcome = backend.produce(job)?;
    ensure!(
        outcome.job_sha256 == expected_job && outcome.request_sha256 == expected_request,
        "policy execution identity differs"
    );
    let nonattempt = retryable_inventory(&outcome)
        && matches!(
            outcome.command_status,
            EvaluationStatus::Cancelled | EvaluationStatus::Unavailable
        );
    let record = serde_json::to_value(&outcome)?;
    let name = if nonattempt {
        format!(
            "proposal-inventory-{}",
            sha256_hex(&serde_json::to_vec(&record)?)
        )
    } else {
        "proposal-execution".into()
    };
    backend.persist(&arm, &name, &record)?;
    if outcome.command_status != EvaluationStatus::Completed {
        return Ok(NeuralRun {
            proposal_execution: outcome,
            search: None,
        });
    }
    outcome.verify_outputs()?;
    let output_path = worker.job.directory.join(&worker.output_name);
    let artifact = outcome
        .outputs
        .iter()
        .find(|o| o.path == output_path)
        .context("policy producer artifact absent")?;
    artifact.verify()?;
    let producer =
        jope_policy::producer::read_claims(&std::fs::read(&artifact.path)?, &worker.request)?;
    let population = producer.population();
    ensure!(
        population.candidates().len() == INITIAL_SIZE,
        "policy must provide80 initial slots"
    );
    // The reader reconciled the ledger with this population and request; an artifact without
    // one carries no candidate identity or denominators and is not admitted to the search.
    let proposal_ledger_sha256 = producer
        .ledger()
        .context("policy producer artifact lacks its proposal ledger")?
        .sha256()?;
    // W4-T03: the candidate ledger records every draw's terminal outcome, and each policy slot
    // names the record of the draw it came from (W4-T07 join).
    let proposal_draws = producer.draw_outcomes()?;
    let mut candidates = population
        .candidates()
        .iter()
        .enumerate()
        .map(|(slot, seed)| {
            let origin = match (slot < NEURAL_SLOTS, seed.origin()) {
                (true, ORIGIN_NEURAL) => Origin::NeuralPolicy,
                (true, ORIGIN_NEURAL_FALLBACK) => Origin::NeuralRejectionFallback,
                (false, ORIGIN_RANDOM) => Origin::RandomExploration,
                _ => anyhow::bail!("incorrect origin in proposal slot {slot}"),
            };
            let mut candidate = Candidate::new(
                &arm,
                seed.genome().into(),
                0,
                slot,
                origin,
                vec![],
                seed.rank_log_weight(),
            )?;
            candidate.proposal_record_sha256 = seed.proposal_record_sha256().map(Into::into);
            Ok(candidate)
        })
        .collect::<Result<Vec<_>>>()?;
    seat_incumbents(&arm, comparison, &mut candidates)?;
    let evidence = AcceptedProposalEvidence {
        request: worker.request.clone(),
        context_cost: comparison.context_cost.clone(),
        producer_job_sha256: outcome.job_sha256.clone(),
        producer_output_sha256: artifact.sha256.clone(),
        population_sha256: producer.population_sha256().into(),
        context_sha256: context.sha256().into(),
        checkpoint_sha256: population.checkpoint_sha256().into(),
        ranking_backend: producer.ranking_backend().into(),
        native_cuda_parity_qualified: false,
        proposal_ledger_sha256: Some(proposal_ledger_sha256),
        proposal_draws,
    };
    let search = run_population(
        backend,
        comparison,
        variation,
        arm,
        deadline,
        candidates,
        Some(ProposalEvidence::Accepted(Box::new(evidence))),
    )?;
    Ok(NeuralRun {
        proposal_execution: outcome,
        search: Some(search),
    })
}

fn run_generic<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &mut V,
) -> Result<SearchReceipt> {
    let arm = comparison.arm(false)?;
    let deadline = backend.begin_arm(&arm)?;
    if let Some(saved) = backend.load(&arm, "initial")? {
        let candidates = serde_json::from_value(
            saved
                .get(1)
                .context("saved generic initial candidates missing")?
                .clone(),
        )?;
        return run_population(
            backend, comparison, variation, arm, deadline, candidates, None,
        );
    }
    let incumbents = comparison.arm_incumbents(false);
    let draws = INITIAL_SIZE - incumbents.len();
    let mut candidates = Vec::with_capacity(INITIAL_SIZE);
    // Incumbent programs are pre-seeded so no generic draw duplicates them.
    let mut seen: BTreeSet<String> = incumbents
        .iter()
        .map(|s| s.program_sha256.clone())
        .collect();
    for attempt in 0..INITIAL_SIZE * 128 {
        if backend.now_ms() >= deadline {
            anyhow::bail!("generic generation exhausted original arm before initial population");
        }
        let Some(genome) = variation.generic(stream_seed(comparison.random_seed, 0, attempt))?
        else {
            continue;
        };
        let canonical = SynthesizedRecipe::new(genome.clone())?.executable_sha256;
        if seen.insert(canonical) {
            candidates.push(Candidate::new(
                &arm,
                genome,
                0,
                candidates.len(),
                Origin::Generic,
                vec![],
                None,
            )?);
        }
        if candidates.len() == draws {
            break;
        }
    }
    ensure!(
        candidates.len() == draws,
        "generic grammar cannot supply 80 distinct initial candidates"
    );
    for seed in incumbents {
        candidates.push(incumbent_candidate(&arm, seed, candidates.len())?);
    }
    run_population(
        backend, comparison, variation, arm, deadline, candidates, None,
    )
}
