/// Fresh encoding runs through the same bounded child API and original neural
/// deadline. The output may later construct Comparison, never a new input plan.
pub fn encode_for_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    mut job: WorkerJob,
) -> Result<WorkerOutcome> {
    let (arm, _) = begin_pair(backend, plan, variation)?;
    ensure!(
        job.command.program.sha256 == plan.encoding.program_sha256
            && job.request.dataset_sha256 == plan.selection.outer().dataset_sha256
            && job.request.fitting_rows_sha256 == plan.selection.outer().fitting_rows_sha256,
        "encoding worker differs from exact training-side plan"
    );
    for digest in [
        &plan.encoding.input_sha256,
        &plan.encoding.source_manifest_sha256,
        &plan.encoding.encoder_assets_manifest_sha256,
    ] {
        ensure!(
            job.command.inputs.iter().any(|a| &a.sha256 == digest),
            "encoding input missing from pinned worker closure"
        );
    }
    job.arm = Some(arm.clone());
    let job_sha = job.identity()?;
    let result = backend.produce(job)?;
    ensure!(
        result.job_sha256 == job_sha,
        "encoding execution identity differs"
    );
    let value = serde_json::to_value(&result)?;
    backend.persist(
        &arm,
        &format!(
            "encoding-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        ),
        &value,
    )?;
    Ok(result)
}
/// `None` records unavailable context/proposal input and still evaluates generic.
/// Generic results are retained when the neural producer fails or times out.
pub fn run_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &mut V,
    neural_input: Option<(&Comparison, &PolicyWorker)>,
) -> Result<PairedReceipt> {
    begin_pair(backend, plan, variation)?;
    let (neural, neural_error) = if let Some((comparison, worker)) = neural_input {
        let result = (|| {
            ensure!(
                serde_json::to_value(plan)? == serde_json::to_value(&comparison.input_plan)?,
                "neural outputs attached to different input plan"
            );
            run_neural(backend, comparison, variation, worker)
        })();
        match result {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        }
    } else {
        (
            None,
            Some("verified context or proposal input unavailable".into()),
        )
    };
    let (generic, generic_error) = match run_generic(backend, plan, variation) {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(format!("{e:#}"))),
    };
    let receipt = PairedReceipt {
        comparison_sha256: sha256_hex(&serde_json::to_vec(plan)?),
        neural,
        neural_error,
        generic,
        generic_error,
    };
    let value = serde_json::to_value(&receipt)?;
    backend.persist(
        &plan.arm(true)?,
        &format!(
            "pair-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        ),
        &value,
    )?;
    Ok(receipt)
}

/// Experimental typed admission reuses the exact population/OOF/evolution path.
/// Context production and Python proposal execution spend the original neural
/// allowance; the generic arm keeps the same120second maximum and80 slots.
#[cfg(feature = "qualified-context")]
pub fn run_experimental_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &mut V,
    neural_input: Option<(
        &crate::experimental_producer::Worker,
        &crate::experimental_producer::VerifiedProposalContext,
    )>,
) -> Result<PairedReceipt> {
    begin_pair(backend, plan, variation)?;
    ensure!(
        matches!(plan.proposal, ProposalPlan::Experimental(_)),
        "experimental arm requires its separate input plan"
    );
    let (neural, neural_error) = match neural_input {
        Some((worker, context)) => {
            match run_experimental(backend, plan, variation, worker, context) {
                Ok(value) => (Some(value), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            }
        }
        None => (
            None,
            Some("qualified experimental context or Python proposal input unavailable".into()),
        ),
    };
    let (generic, generic_error) = match run_generic(backend, plan, variation) {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    let receipt = PairedReceipt {
        comparison_sha256: sha256_hex(&serde_json::to_vec(plan)?),
        neural,
        neural_error,
        generic,
        generic_error,
    };
    let value = serde_json::to_value(&receipt)?;
    backend.persist(
        &plan.arm(true)?,
        &format!(
            "pair-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        ),
        &value,
    )?;
    Ok(receipt)
}

#[cfg(feature = "qualified-context")]
fn run_experimental<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &mut V,
    worker: &crate::experimental_producer::Worker,
    context: &crate::experimental_producer::VerifiedProposalContext,
) -> Result<NeuralRun> {
    use crate::{cpu_context_bridge::ExecutionProof, execution::Artifact, experimental_producer};
    plan.validate()?;
    let ProposalPlan::Experimental(request_plan) = &plan.proposal else {
        anyhow::bail!("accepted plan cannot enter experimental producer")
    };
    let arm = plan.arm(true)?;
    let deadline = backend.begin_arm(&arm)?;
    ensure!(
        worker
            .job
            .arm
            .as_ref()
            .is_some_and(|a| a.id_sha256 == arm.id_sha256 && a.seconds == arm.seconds),
        "Python worker lacks original paired arm"
    );
    experimental_producer::verify_inputs(request_plan, worker, context, variation)?;
    let outcome = backend.produce(worker.job.clone())?;
    ensure!(
        outcome.job_sha256 == worker.job.identity()?
            && outcome.request_sha256 == worker.job.request.identity()?,
        "experimental process identity differs"
    );
    let value = serde_json::to_value(&outcome)?;
    let name = if retryable_inventory(&outcome) {
        format!(
            "proposal-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        )
    } else {
        "proposal-execution".into()
    };
    backend.persist(&arm, &name, &value)?;
    if outcome.command_status != EvaluationStatus::Completed {
        return Ok(NeuralRun {
            proposal_execution: outcome,
            search: None,
        });
    }
    // Reconstruct original pool locations; no caller-provided completion boolean.
    let journal = worker
        .job
        .directory
        .parent()
        .and_then(Path::parent)
        .context("proposal job outside journal/jobs")?;
    let artifact = |path: PathBuf| -> Result<Artifact> {
        let a = Artifact {
            sha256: crate::sha256_file(&path)?,
            path,
        };
        a.verify()?;
        Ok(a)
    };
    let proof = ExecutionProof {
        job: worker.job.clone(),
        outcome: artifact(
            journal
                .join("finished")
                .join(format!("{}.json", worker.job.identity()?)),
        )?,
        started: artifact(
            journal
                .join("started")
                .join(format!("{}.json", worker.job.identity()?)),
        )?,
        experiment: serde_json::from_slice(&std::fs::read(journal.join("experiment.json"))?)?,
        window: artifact(journal.join("window.json"))?,
        arm_window: Some(artifact(
            journal.join("arms").join(format!("{}.json", arm.id_sha256)),
        )?),
    };
    let admitted = experimental_producer::admit(plan, worker, context, &proof, variation)?;
    let candidates = admitted
        .candidates()
        .iter()
        .map(|seed| {
            let origin = match seed.origin.as_str() {
                "neural_policy" => Origin::NeuralPolicy,
                "neural_rejection_fallback" => Origin::NeuralRejectionFallback,
                "random_exploration" => Origin::RandomExploration,
                _ => anyhow::bail!("invalid admitted origin"),
            };
            let candidate = Candidate::new(
                &arm,
                seed.recipe.original_proposal.clone(),
                0,
                seed.slot,
                origin,
                vec![],
                seed.rank_log_weight,
            )?;
            ensure!(
                candidate.program_sha256 == seed.recipe.executable_sha256,
                "admitted recipe canonicalization changed"
            );
            Ok(candidate)
        })
        .collect::<Result<Vec<_>>>()?;
    let search = run_population(
        backend,
        plan,
        variation,
        arm,
        deadline,
        candidates,
        Some(ProposalEvidence::Experimental(Box::new(
            admitted.evidence().clone(),
        ))),
    )?;
    Ok(NeuralRun {
        proposal_execution: outcome,
        search: Some(search),
    })
}

/// Identity of a non-predictor component/proposal job inside a frozen pair.
pub fn diagnostic_request(
    plan: &PairInputPlan,
    purpose: &str,
    pin: &str,
) -> Result<EvaluationRequest> {
    let mut request = plan.selection.outer().clone();
    request.stage = crate::experiment::EvaluationStage::Diagnostic;
    request.recipe_sha256 = sha256_hex(&serde_json::to_vec(&(purpose, pin))?);
    request.seed = crate::search_plan::candidate_seed(request.seed, &request.recipe_sha256)?;
    request.scoring_rows = 1;
    request.scoring_rows_sha256 = sha256_hex(&serde_json::to_vec(&(
        purpose,
        "output-coordinate-not-a-predictor-score",
    ))?);
    request.validate()?;
    Ok(request)
}

/// Private fitted-recipe identity frozen before the controller opens the outer
/// development scoring labels. It is selection evidence, not an outer win.
#[derive(Debug, Serialize, Deserialize)]
pub struct FrozenRecipe {
    pub candidate: Candidate,
    pub selection_feedback: Feedback,
    pub search_receipt_sha256: String,
    pub refit_request: EvaluationRequest,
}
pub fn freeze_winner<B: Backend>(
    backend: &mut B,
    comparison: &PairInputPlan,
    receipt: &SearchReceipt,
) -> Result<Option<FrozenRecipe>> {
    comparison.validate()?;
    ensure!(
        receipt.arm.id_sha256 == comparison.arm(true)?.id_sha256
            || receipt.arm.id_sha256 == comparison.arm(false)?.id_sha256,
        "winner arm differs from preregistered pair"
    );
    ensure!(
        receipt.comparison_sha256 == sha256_hex(&serde_json::to_vec(comparison)?)
            && receipt.selection_sha256 == comparison.selection.identity()?,
        "winner belongs to different comparison"
    );
    ensure!(
        !matches!(
            receipt.stop_reason.as_str(),
            "admission_pending_original_window" | "cancelled_original_window"
        ),
        "cannot freeze winner while admissions remain retryable"
    );
    let Some(winner) = receipt.winner() else {
        return Ok(None);
    };
    let mut feedback = winner.feedback.clone();
    feedback.enforce_selection(&comparison.selection, &winner.candidate);
    ensure!(
        feedback.status == EvaluationStatus::Completed,
        "winner lacks complete train-side evidence"
    );
    let mut refit_request = comparison.selection.refit_request(
        &winner.candidate.program_sha256,
        &winner.candidate.id_sha256,
    )?;
    refit_request.checkpoint_sha256 = receipt
        .proposal
        .as_ref()
        .map(|p| p.checkpoint_sha256().to_owned());
    refit_request.validate()?;
    let frozen = FrozenRecipe {
        candidate: winner.candidate.clone(),
        selection_feedback: feedback,
        search_receipt_sha256: sha256_hex(&serde_json::to_vec(receipt)?),
        refit_request,
    };
    backend.persist(
        &receipt.arm,
        "frozen-recipe",
        &serde_json::to_value(&frozen)?,
    )?;
    Ok(Some(frozen))
}

fn evaluate_unique<B: Backend>(
    backend: &mut B,
    arm: &ArmBudget,
    selection: &SelectionPlan,
    candidates: &[Candidate],
) -> Result<Vec<Feedback>> {
    let mut first = BTreeMap::new();
    let mut unique = vec![];
    let mut duplicate = vec![];
    for candidate in candidates {
        if let Some(previous) = first.get(&candidate.program_sha256) {
            duplicate.push(Feedback {
                candidate_sha256: candidate.id_sha256.clone(),
                status: EvaluationStatus::Failed,
                loss: None,
                physical_fit_starts: Some(0),
                worker_receipt_sha256: None,
                replay_evidence_sha256: None,
                fitted_artifact_sha256: None,
                reason: Some(format!("canonical_duplicate_of:{previous}")),
                oof: None,
                retryable_inventory: false,
            });
        } else {
            first.insert(
                candidate.program_sha256.clone(),
                candidate.id_sha256.clone(),
            );
            unique.push(candidate.clone());
        }
    }
    let mut feedback = backend.evaluate(arm, selection, &unique)?;
    for item in &mut feedback {
        let candidate = unique
            .iter()
            .find(|c| c.id_sha256 == item.candidate_sha256)
            .context("unexpected candidate feedback")?;
        item.enforce_selection(selection, candidate);
    }
    feedback.extend(duplicate);
    Ok(feedback)
}
