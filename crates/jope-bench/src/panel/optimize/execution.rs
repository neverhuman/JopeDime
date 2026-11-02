use super::*;

/// Used by the shared-wave controller, and by standalone optimize on one frozen
/// dataset. PoolAccess is either the actual pool or its bounded message proxy.
pub fn search_dataset<P: PoolAccess>(
    campaign: &CampaignPlan,
    configuration: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    pool: &mut P,
    cancel: &AtomicBool,
) -> Result<DatasetSearch> {
    let cut = dataset.cut(0)?;
    let (input, plan) = build_pair_input(campaign, configuration, dataset)?;
    let input_file = immutable(
        &journal
            .join("asm-context-inputs")
            .join(format!("{}.json", plan.encoding.input_sha256)),
        &input,
    )?;
    let vocabulary = jope_policy::grammar::bind_vocabulary(
        &fs::read(&configuration.vocabulary.path)?,
        jope_policy::CHECKPOINT_VOCABULARY_SHA256,
    )?;
    let mut variation = TaskVariation {
        vocabulary: &vocabulary,
        task: dataset.dataset.task,
    };
    let adapter = FitAdapter::prepare(
        dataset,
        &cut,
        plan.selection.clone(),
        campaign.runtimes["rust"].clone(),
        journal,
        true,
    )?;
    let mut backend = PoolBackend {
        pool,
        cancelled: cancel,
        journal: journal.join("asm-search"),
        make_job: |candidate: &neural_search::Candidate, request: &EvaluationRequest| {
            adapter.job(candidate, request)
        },
        score: |candidate: &neural_search::Candidate,
                outcomes: &[WorkerOutcome],
                selection: &jope_pipeline::search_plan::SelectionPlan| {
            adapter.score(candidate, outcomes, selection)
        },
    };
    neural_search::begin_pair(&mut backend, &plan, &variation)?;
    let (pair, unavailable, neural_retryable_inventory) = if matches!(
        plan.proposal,
        neural_search::ProposalPlan::Experimental(_)
    ) {
        let neural_input = super::experimental_proposal::prepare(
            &mut backend,
            &plan,
            &mut variation,
            configuration,
            &input,
            &input_file,
            &campaign.experiment,
            journal,
        );
        let unavailable = neural_input.as_ref().err().map(|e| format!("{e:#}"));
        let retryable = neural_input.is_err()
            && configuration.qualification.is_some()
            && configuration.encoder.unavailable.is_none()
            && !terminal_encoding_failure(journal, &plan)?;
        let pair = neural_search::run_experimental_pair(
            &mut backend,
            &plan,
            &mut variation,
            neural_input
                .as_ref()
                .ok()
                .map(|(worker, context)| (worker, context)),
        )?;
        (pair, unavailable, retryable)
    } else {
        let neural_input = if dataset.dataset.task != Task::Regression {
            Err(anyhow::anyhow!(
                "accepted policy is regression-only; no configured qualified experimental classifier producer"
            ))
        } else {
            prepare_neural(
                &mut backend,
                &plan,
                &variation,
                configuration,
                input.accepted()?,
                &input_file,
                &campaign.experiment,
                journal,
            )
        };
        let unavailable = neural_input.as_ref().err().map(|e| format!("{e:#}"));
        let retryable = neural_input.is_err()
            && dataset.dataset.task == Task::Regression
            && configuration.qualification.is_some()
            && configuration.policy.unavailable.is_none()
            && configuration.encoder.unavailable.is_none()
            && !terminal_encoding_failure(journal, &plan)?;
        let pair = neural_search::run_pair(
            &mut backend,
            &plan,
            &mut variation,
            neural_input
                .as_ref()
                .ok()
                .map(|(comparison, worker)| (comparison, worker)),
        )?;
        (pair, unavailable, retryable)
    };
    let neural = pair
        .neural
        .as_ref()
        .and_then(|n| n.search.as_ref())
        .filter(|r| !pending(r))
        .map(|r| neural_search::freeze_winner(&mut backend, &plan, r))
        .transpose()?
        .flatten();
    let generic = pair
        .generic
        .as_ref()
        .filter(|r| !pending(r))
        .map(|r| neural_search::freeze_winner(&mut backend, &plan, r))
        .transpose()?
        .flatten();
    let mut result = DatasetSearch {
        dataset: dataset.dataset.id.clone(),
        task: dataset.dataset.task,
        pair: plan,
        receipt: pair,
        neural,
        generic,
        neural_unavailable: unavailable,
        neural_retryable_inventory,
        final_test_targets_used: false,
        teaching_admission: None,
        teaching_error: None,
    };
    match super::teaching::admit_dataset(campaign, configuration, dataset, &result, journal) {
        Ok(value) => result.teaching_admission = value,
        Err(error) => result.teaching_error = Some(format!("{error:#}")),
    }
    let value = serde_json::to_value(&result)?;
    immutable(
        &journal
            .join("asm-inventory")
            .join(&dataset.dataset.id)
            .join(format!("{}.json", digest(&value)?)),
        &value,
    )?;
    Ok(result)
}

pub(crate) fn diagnostic(
    plan: &PairInputPlan,
    purpose: &str,
    pin: &str,
) -> Result<EvaluationRequest> {
    neural_search::diagnostic_request(plan, purpose, pin)
}
pub(crate) fn proof(
    experiment: &Experiment,
    journal: &Path,
    job: WorkerJob,
) -> Result<ExecutionProof> {
    let id = job.identity()?;
    Ok(ExecutionProof {
        outcome: artifact(journal.join("finished").join(format!("{id}.json")))?,
        started: artifact(journal.join("started").join(format!("{id}.json")))?,
        window: artifact(journal.join("window.json"))?,
        arm_window: job
            .arm
            .as_ref()
            .map(|a| artifact(journal.join("arms").join(format!("{}.json", a.id_sha256))))
            .transpose()?,
        experiment: experiment.clone(),
        job,
    })
}
#[allow(clippy::too_many_arguments)] // Keep independently pinned execution inputs explicit.
fn prepare_neural<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    config: &Configuration,
    input: &TrainingInput,
    input_file: &Artifact,
    experiment: &Experiment,
    journal: &Path,
) -> Result<(Comparison, PolicyWorker)> {
    config.qualification.as_ref().context(
        "CPU encoder qualification absent; accepted native-component parity remains unmeasured",
    )?;
    ensure!(
        config.encoder.unavailable.is_none() && config.policy.unavailable.is_none(),
        "planned encoder or policy runtime unavailable"
    );
    for file in config.dependencies() {
        file.verify()?;
    }
    let worker_builds = config.verify_worker_builds()?;
    let (proposal_receipt, proposal_weights) = proposal_files(config);
    pinned_checkpoint(config)?;
    let (files, outcome) = prepare_regression_context(
        backend,
        plan,
        variation,
        config,
        input,
        input_file,
        experiment,
        journal,
        &worker_builds,
    )?;
    let context_directory = journal.join("asm-contexts").join(plan.arm(true)?.id_sha256);
    let context_sha256 = files.vector.sha256.clone();
    // The worker admits the qualified bundle only against pins this request issues: exactly the
    // admission prepare_regression_context built from the controller's own inputs and verified.
    let bundle_bytes = fs::read(&files.bundle.path)?;
    ensure!(
        jope_pipeline::sha256_hex(&bundle_bytes) == files.bundle.sha256,
        "qualified context bundle changed after admission"
    );
    let bundle: QualifiedContextBundle = serde_json::from_slice(&bundle_bytes)?;
    let producer_request = plan
        .proposal
        .accepted()?
        .qualified_request(context_sha256, bundle.pins.sha256()?);
    let request_file = immutable(
        &context_directory.join("proposal-request.json"),
        &producer_request,
    )?;
    let request = diagnostic(
        plan,
        "asm.policy-proposal/v1",
        &producer_request.identity()?,
    )?;
    let directory = journal.join("jobs").join(request.identity()?);
    let named = [
        ("--request", &request_file),
        ("--source-manifest", &config.source_manifest),
        ("--context", &files.vector),
        ("--context-receipt", &files.artifact),
        ("--context-bundle", &files.bundle),
        ("--transforms", &config.frozen_transforms),
        ("--checkpoint-receipt", proposal_receipt),
        ("--policy", proposal_weights),
        ("--vocabulary", &config.vocabulary),
    ];
    let mut args = Vec::new();
    let mut inputs = config.policy.inputs.clone();
    inputs.extend(worker_builds);
    inputs.push(config.release_manifest.clone());
    for (name, file) in named {
        args.extend([name.into(), file.path.display().to_string()]);
        inputs.push(file.clone());
    }
    args.extend([
        "--output".into(),
        directory.join("output.json").display().to_string(),
    ]);
    let comparison = Comparison {
        input_plan: plan.clone(),
        producer_request: producer_request.clone(),
        context_receipt_sha256: files.artifact.sha256.clone(),
        context_cost: ContextCost {
            cached: false,
            production_receipt: files.context_execution.outcome.clone(),
            original_wall_seconds: outcome.consumption.wall_seconds,
            original_cpu_seconds: outcome.consumption.cpu_seconds,
            original_gpu_seconds: Some(0.),
            amortization_uses: Some(1),
        },
    };
    let worker = PolicyWorker {
        job: WorkerJob {
            request,
            command: WorkerCommand {
                program: config.policy.program.clone(),
                args,
                inputs,
                environment: config.policy.environment.clone(),
            },
            directory,
            outputs: vec!["output.json".into()],
            arm: Some(plan.arm(true)?),
        },
        expected_program_sha256: config.policy.program.sha256.clone(),
        release_manifest: config.release_manifest.clone(),
        source_manifest: config.source_manifest.clone(),
        request_artifact: request_file,
        checkpoint_receipt: proposal_receipt.clone(),
        request: producer_request,
        context: ContextFiles::QualifiedCpu(Box::new(files)),
        output_name: "output.json".into(),
    };
    immutable(
        &context_directory.join("policy-admission.json"),
        &(&comparison, &worker),
    )?;
    Ok((comparison, worker))
}

fn terminal_encoding_failure(journal: &Path, plan: &PairInputPlan) -> Result<bool> {
    let directory = journal.join("asm-search").join(plan.arm(true)?.id_sha256);
    if !directory.exists() {
        return Ok(false);
    }
    for file in fs::read_dir(directory)? {
        let file = file?;
        if !file
            .file_name()
            .to_string_lossy()
            .starts_with("encoding-inventory-")
        {
            continue;
        }
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(file.path())?)?;
        if outcome.command_status != EvaluationStatus::Completed
            && (outcome.exit_code.is_some() || outcome.consumption.gate_released_unix_ms.is_some())
        {
            let retained = journal
                .join("finished")
                .join(format!("{}.json", outcome.job_sha256));
            if retained.exists() {
                let actual: WorkerOutcome = serde_json::from_slice(&fs::read(retained)?)?;
                ensure!(
                    digest(&actual)? == digest(&outcome)?,
                    "encoding failure receipt changed"
                );
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub fn pending(receipt: &neural_search::SearchReceipt) -> bool {
    matches!(
        receipt.stop_reason.as_str(),
        "admission_pending_original_window" | "cancelled_original_window"
    )
}
