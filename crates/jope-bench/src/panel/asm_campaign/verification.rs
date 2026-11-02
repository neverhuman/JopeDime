use super::*;

pub(super) fn verify_search(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    search: &DatasetSearch,
) -> Result<()> {
    ensure!(
        search.dataset == dataset.dataset.id
            && search.task == dataset.dataset.task
            && !search.final_test_targets_used,
        "wrong search dataset or label closure"
    );
    let cut = dataset.cut(0)?;
    let (input, expected) = optimize::build_pair_input(plan, config, dataset)?;
    ensure!(
        digest(&expected)? == digest(&search.pair)?,
        "paired plan differs from exact frozen training cells, labels or full configuration"
    );
    let input_file = artifact(
        journal
            .join("asm-context-inputs")
            .join(format!("{}.json", expected.encoding.input_sha256)),
    )?;
    ensure!(
        input_file.sha256 == expected.encoding.input_sha256
            && fs::read(&input_file.path)? == serde_json::to_vec(&input)?,
        "context worker input differs from exact frozen training rows"
    );
    let adapter = FitAdapter::prepare(
        dataset,
        &cut,
        search.pair.selection.clone(),
        plan.runtimes["rust"].clone(),
        journal,
        false,
    )?;
    for (neural, receipt, frozen) in [
        (
            true,
            search
                .receipt
                .neural
                .as_ref()
                .and_then(|r| r.search.as_ref()),
            search.neural.as_ref(),
        ),
        (
            false,
            search.receipt.generic.as_ref(),
            search.generic.as_ref(),
        ),
    ] {
        if let Some(receipt) = receipt {
            verify_receipt(&adapter, journal, &search.pair, receipt, &plan.experiment)?;
            ensure!(
                receipt.arm.id_sha256 == search.pair.arm(neural)?.id_sha256,
                "arm assignment changed"
            );
            if neural {
                config.verify_worker_builds()?;
                if matches!(
                    search.pair.proposal,
                    neural_search::ProposalPlan::Experimental(_)
                ) {
                    let vocabulary = jope_policy::grammar::bind_vocabulary(
                        &fs::read(&config.vocabulary.path)?,
                        jope_policy::CHECKPOINT_VOCABULARY_SHA256,
                    )?;
                    let mut variation = optimize::TaskVariation {
                        vocabulary: &vocabulary,
                        task: dataset.dataset.task,
                    };
                    super::experimental_proposal::verify_report(
                        config,
                        &input,
                        search,
                        receipt,
                        journal,
                        &mut variation,
                    )?;
                } else {
                    verify_producer(journal, search, receipt)?;
                }
            }
            if let Some(frozen) = frozen {
                let winner = receipt
                    .winner()
                    .context("frozen winner lacks measurement")?;
                ensure!(
                    digest(&frozen.candidate)? == digest(&winner.candidate)?
                        && digest(&frozen.selection_feedback)? == digest(&winner.feedback)?
                        && frozen.search_receipt_sha256 == digest(receipt)?,
                    "frozen winner differs from minimum train-side OOF loss"
                );
                let persisted: Value = serde_json::from_slice(&fs::read(
                    journal
                        .join("asm-search")
                        .join(&receipt.arm.id_sha256)
                        .join("frozen-recipe.json"),
                )?)?;
                ensure!(
                    persisted == serde_json::to_value(frozen)?,
                    "frozen selection evidence changed"
                );
            } else {
                ensure!(
                    receipt.winner().is_none() || optimize::pending(receipt),
                    "omitted measured frozen winner"
                );
            }
        } else {
            ensure!(frozen.is_none(), "winner without arm execution");
        }
    }
    ensure!(
        config.teaching.is_none()
            == (search.teaching_admission.is_none() && search.teaching_error.is_none())
            && !(search.teaching_admission.is_some() && search.teaching_error.is_some()),
        "teaching configuration is missing an admission or explicit failure"
    );
    if let Some(file) = &search.teaching_admission {
        super::teaching::verify_admission(plan, config, dataset, search, journal, file)?;
    }
    Ok(())
}
fn verify_receipt(
    adapter: &FitAdapter,
    journal: &Path,
    plan: &neural_search::PairInputPlan,
    receipt: &SearchReceipt,
    experiment: &jope_pipeline::experiment::Experiment,
) -> Result<()> {
    receipt.validate_population(plan)?;
    ensure!(
        receipt.deadline_unix_ms
            == super::receipt_timing::arm_deadline(
                journal,
                &receipt.arm,
                &plan.selection.outer().experiment_sha256
            )?,
        "search receipt extended or changed original clipped deadline"
    );
    let path = journal
        .join("asm-search")
        .join(&receipt.arm.id_sha256)
        .join("search-receipt.json");
    if path.exists() {
        let persisted: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            persisted == serde_json::to_value(receipt)?,
            "search receipt differs from original completion"
        );
    } else {
        ensure!(
            optimize::pending(receipt)
                && (arms_expired(plan, journal)? || original_cutoff_reached(experiment, journal)?),
            "missing final receipt before original arm deadlines"
        );
    }
    for observation in &receipt.observations {
        if observation.feedback.status != EvaluationStatus::Completed {
            continue;
        }
        let requests = plan.selection.selection_requests(
            &observation.candidate.program_sha256,
            &observation.candidate.id_sha256,
        )?;
        let mut outcomes = Vec::new();
        for request in requests {
            let mut job = adapter.job(&observation.candidate, &request)?;
            job.arm = Some(receipt.arm.clone());
            let file = artifact(
                journal
                    .join("finished")
                    .join(format!("{}.json", job.identity()?)),
            )?;
            outcomes.push(serde_json::from_slice::<WorkerOutcome>(&fs::read(
                file.path,
            )?)?);
        }
        let actual = adapter.score(&observation.candidate, &outcomes, &plan.selection)?;
        ensure!(
            digest(&actual)? == digest(&observation.feedback)?,
            "retained OOF fitness differs from fitted predictor replay"
        );
    }
    Ok(())
}
fn verify_producer(journal: &Path, search: &DatasetSearch, receipt: &SearchReceipt) -> Result<()> {
    let directory = journal
        .join("asm-contexts")
        .join(search.pair.arm(true)?.id_sha256);
    let (comparison, worker): (neural_search::Comparison, neural_search::PolicyWorker) =
        serde_json::from_slice(&fs::read(directory.join("policy-admission.json"))?)?;
    ensure!(
        digest(&comparison.input_plan)? == digest(&search.pair)?,
        "proposal admission changed pair inputs"
    );
    neural_search::verify_policy_inputs(&comparison, &worker)?;
    let outcome = &search
        .receipt
        .neural
        .as_ref()
        .context("neural execution absent")?
        .proposal_execution;
    let mut job = worker.job.clone();
    job.arm = Some(search.pair.arm(true)?);
    ensure!(
        outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.reason.is_none()
            && outcome.job_sha256 == job.identity()?,
        "policy proposal did not complete pinned command"
    );
    let original: WorkerOutcome = serde_json::from_slice(&fs::read(
        journal
            .join("finished")
            .join(format!("{}.json", job.identity()?)),
    )?)?;
    ensure!(
        digest(outcome)? == digest(&original)?,
        "proposal execution receipt changed"
    );
    super::receipt_timing::completed_fit(journal, &job, outcome)?;
    outcome.verify_outputs()?;
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == job.directory.join(&worker.output_name))
        .context("proposal output absent")?;
    let producer = jope_policy::producer::read_claims(&fs::read(&output.path)?, &worker.request)?;
    let evidence = receipt
        .proposal
        .as_ref()
        .context("neural provenance absent")?
        .accepted()?;
    // Evidence that names a proposal ledger binds exactly that reconciled ledger; evidence
    // recorded before the ledger existed must come from an artifact that has none.
    let ledger_sha256 = producer.ledger().map(|l| l.sha256()).transpose()?;
    ensure!(
        evidence.producer_job_sha256 == outcome.job_sha256
            && evidence.producer_output_sha256 == output.sha256
            && evidence.population_sha256 == producer.population_sha256()
            && evidence.proposal_ledger_sha256 == ledger_sha256,
        "proposal provenance changed"
    );
    // W4-T03: draw outcomes the candidate ledger records are exactly the reconciled ledger's
    // (evidence recorded before the candidate ledger carried them has none).
    if let Some(draws) = &evidence.proposal_draws {
        ensure!(
            Some(draws) == producer.draw_outcomes()?.as_ref(),
            "proposal draw outcomes differ from the producer's ledger"
        );
    }
    for observation in receipt
        .observations
        .iter()
        .filter(|o| o.candidate.generation == 0)
    {
        let slot = observation
            .candidate
            .initial_slot
            .context("generation-zero slot absent")?;
        let seed = producer
            .population()
            .candidates()
            .get(slot)
            .context("proposal slot out of range")?;
        let first_incumbent = neural_search::INITIAL_SIZE - search.pair.arm_incumbents(true).len();
        if slot >= first_incumbent {
            ensure!(
                seed.origin() == jope_policy::seeds::ORIGIN_RANDOM,
                "incumbent displaced a policy proposal slot"
            );
            continue; // validate_population already bound this slot to its pinned incumbent
        }
        let origin = match seed.origin() {
            jope_policy::seeds::ORIGIN_NEURAL => Origin::NeuralPolicy,
            jope_policy::seeds::ORIGIN_NEURAL_FALLBACK => Origin::NeuralRejectionFallback,
            jope_policy::seeds::ORIGIN_RANDOM => Origin::RandomExploration,
            _ => anyhow::bail!("untrusted proposal origin"),
        };
        ensure!(
            observation.candidate.genome == seed.genome()
                && observation.candidate.origin == origin
                && observation.candidate.rank_log_weight == seed.rank_log_weight()
                && observation.candidate.proposal_record_sha256.as_deref()
                    == seed.proposal_record_sha256(),
            "generation-zero origin/genome/rank differs from actual producer output"
        );
    }
    Ok(())
}

fn original_cutoff_reached(
    experiment: &jope_pipeline::experiment::Experiment,
    journal: &Path,
) -> Result<bool> {
    let window: jope_pipeline::experiment::ExecutionWindow =
        serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
    window.validate(experiment)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as u64;
    Ok(now
        >= window
            .deadline_unix_ms
            .saturating_sub(experiment.resources.reporting_seconds * 1000))
}
