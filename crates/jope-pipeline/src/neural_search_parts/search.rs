fn run_population<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &mut V,
    arm: ArmBudget,
    deadline: u64,
    mut candidates: Vec<Candidate>,
    proposal: Option<ProposalEvidence>,
) -> Result<SearchReceipt> {
    ensure!(
        candidates.len() == INITIAL_SIZE,
        "initial population is not 80"
    );
    let mut seen: BTreeSet<_> = candidates
        .iter()
        .map(|c| c.program_sha256.clone())
        .collect();
    let mut receipt = SearchReceipt {
        schema: PROTOCOL.into(),
        comparison_sha256: sha256_hex(&serde_json::to_vec(comparison)?),
        selection_sha256: comparison.selection.identity()?,
        arm,
        deadline_unix_ms: deadline,
        proposal,
        observations: vec![],
        stop_reason: String::new(),
    };
    backend.persist(
        &receipt.arm,
        "initial",
        &serde_json::to_value((&receipt.proposal, &candidates, deadline))?,
    )?;
    if let Some(value) = backend.load(&receipt.arm, "search-receipt")? {
        let saved: SearchReceipt = serde_json::from_value(value)?;
        saved.validate_population(comparison)?;
        ensure!(
            saved.schema == PROTOCOL
                && saved.comparison_sha256 == receipt.comparison_sha256
                && saved.selection_sha256 == receipt.selection_sha256
                && saved.arm.id_sha256 == receipt.arm.id_sha256
                && saved.deadline_unix_ms == deadline
                && serde_json::to_value(&saved.proposal)?
                    == serde_json::to_value(&receipt.proposal)?,
            "saved search binding changed"
        );
        ensure!(
            saved.observations.len() >= INITIAL_SIZE
                && serde_json::to_value(
                    saved.observations[..INITIAL_SIZE]
                        .iter()
                        .map(|o| &o.candidate)
                        .collect::<Vec<_>>()
                )? == serde_json::to_value(&candidates)?,
            "saved initial population changed"
        );
        // Recheck the execution cache and fitted outputs; no deadline reset and
        // no receipt-only claim of completed work.
        for group in saved
            .observations
            .chunk_by(|a, b| a.candidate.generation == b.candidate.generation)
        {
            let batch = group
                .iter()
                .map(|o| o.candidate.clone())
                .collect::<Vec<_>>();
            let feedback = evaluate_unique(backend, &receipt.arm, &comparison.selection, &batch)?;
            verify_observed_replay(
                group,
                &feedback,
                saved.stop_reason == "original_deadline" && backend.now_ms() >= deadline,
            )?;
        }
        return Ok(saved);
    }
    for generation in 0..1000 {
        backend.persist(
            &receipt.arm,
            &format!("generation-{generation:04}-scheduled"),
            &serde_json::to_value(&candidates)?,
        )?;
        let saved_observations = backend
            .load(
                &receipt.arm,
                &format!("generation-{generation:04}-observed"),
            )?
            .map(serde_json::from_value::<Vec<Observation>>)
            .transpose()?;
        if let Some(saved) = &saved_observations {
            ensure!(
                serde_json::to_value(saved.iter().map(|o| &o.candidate).collect::<Vec<_>>())?
                    == serde_json::to_value(&candidates)?,
                "saved observed population changed"
            );
        }
        // The pool returns intact cached successes even after the deadline; new
        // jobs receive explicit budget results. Never turn cached fits into zero.
        let feedback = evaluate_unique(backend, &receipt.arm, &comparison.selection, &candidates)?;
        if let Some(saved) = &saved_observations {
            // A controller may stop after the observed journal is durable but
            // before the final search receipt. Replay that same evidence too.
            verify_observed_replay(saved, &feedback, backend.now_ms() >= deadline)?;
        }
        let mut by_id = BTreeMap::new();
        for item in feedback {
            ensure!(
                by_id.insert(item.candidate_sha256.clone(), item).is_none(),
                "duplicate fit feedback"
            );
        }
        ensure!(
            by_id.len() == candidates.len(),
            "missing or extra fit feedback"
        );
        let mut observations = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let feedback = by_id
                .remove(&candidate.id_sha256)
                .context("fit feedback bound to different candidate")?;
            observations.push(Observation {
                candidate,
                feedback,
            });
        }
        if let Some(saved) = saved_observations {
            observations = saved;
        }
        // A nonattempt is resumable only before this arm's original cutoff.
        // Preserve its exact inventory, but do not strand all completed fits
        // waiting for the other arm or global campaign window to expire.
        if backend.now_ms() < deadline
            && observations.iter().any(|o| o.feedback.retryable_inventory)
        {
            let inventory = serde_json::to_value(&observations)?;
            backend.persist(
                &receipt.arm,
                &format!(
                    "generation-{generation:04}-inventory-{}",
                    sha256_hex(&serde_json::to_vec(&inventory)?)
                ),
                &inventory,
            )?;
            receipt.observations.extend(observations);
            receipt.validate_population(comparison)?;
            receipt.stop_reason = "admission_pending_original_window".into();
            return Ok(receipt); // no immutable observed/final receipt: resume may admit later
        }
        backend.persist(
            &receipt.arm,
            &format!("generation-{generation:04}-observed"),
            &serde_json::to_value(&observations)?,
        )?;
        receipt.observations.extend(observations);
        if let Some(saved) = backend.load(
            &receipt.arm,
            &format!("generation-{:04}-scheduled", generation + 1),
        )? {
            candidates = serde_json::from_value(saved)?;
            ensure!(
                !candidates.is_empty()
                    && candidates.len() <= INITIAL_SIZE - ELITES
                    && receipt.observations.len() + candidates.len() <= comparison.max_evaluations,
                "saved offspring count changed"
            );
            for (slot, child) in candidates.iter().enumerate() {
                ensure!(
                    child.generation == generation + 1
                        && matches!(child.origin, Origin::Mutation | Origin::Crossover)
                        && !child.parents.is_empty(),
                    "saved offspring origin changed"
                );
                ensure!(
                    child
                        .parents
                        .iter()
                        .all(|id| receipt
                            .observations
                            .iter()
                            .any(|o| o.candidate.id_sha256 == *id
                                && o.feedback.status == EvaluationStatus::Completed)),
                    "saved child lacks measured parent"
                );
                let expected = Candidate::new(
                    &receipt.arm,
                    child.genome.clone(),
                    generation + 1,
                    slot,
                    child.origin.clone(),
                    child.parents.clone(),
                    None,
                )?;
                ensure!(
                    serde_json::to_value(child)? == serde_json::to_value(expected)?
                        && seen.insert(child.program_sha256.clone()),
                    "saved offspring identity changed"
                );
            }
            continue;
        }
        if backend.now_ms() >= deadline {
            receipt.stop_reason = "original_deadline".into();
            break;
        }
        if backend.cancelled() {
            receipt.validate_population(comparison)?;
            receipt.stop_reason = "cancelled_original_window".into();
            return Ok(receipt); // terminal attempts stay, but future generations may resume
        }
        if receipt.observations.len() >= comparison.max_evaluations {
            receipt.stop_reason = "evaluation_limit".into();
            break;
        }
        let mut parents = receipt
            .observations
            .iter()
            .filter(|o| o.feedback.status == EvaluationStatus::Completed)
            .collect::<Vec<_>>();
        parents.sort_by(|a, b| {
            a.feedback
                .loss
                .unwrap()
                .total_cmp(&b.feedback.loss.unwrap())
                .then_with(|| a.candidate.program_sha256.cmp(&b.candidate.program_sha256))
        });
        parents.truncate(ELITES);
        if parents.is_empty() {
            receipt.stop_reason = "no_measured_parent".into();
            break;
        }
        let desired =
            (INITIAL_SIZE - ELITES).min(comparison.max_evaluations - receipt.observations.len());
        candidates = Vec::with_capacity(desired);
        for attempt in 0..desired * 128 {
            if backend.now_ms() >= deadline {
                break;
            }
            let seed = stream_seed(comparison.random_seed, generation + 1, attempt);
            let left = &parents[attempt % parents.len()].candidate;
            let right = &parents[(attempt + 1) % parents.len()].candidate;
            let (genome, origin, ancestors) = if attempt % 2 == 0 {
                (
                    variation.mutate(&left.genome, seed),
                    Origin::Mutation,
                    vec![left.id_sha256.clone()],
                )
            } else {
                (
                    variation.crossover(&left.genome, &right.genome, seed),
                    Origin::Crossover,
                    vec![left.id_sha256.clone(), right.id_sha256.clone()],
                )
            };
            let genome = match genome {
                Ok(value) => value,
                Err(error) => {
                    backend.persist(&receipt.arm,&format!("generation-{}-variation-rejected-{attempt:06}",generation+1),&serde_json::json!({"seed":seed,"origin":origin,"parents":ancestors,"error":format!("{error:#}")}))?;
                    continue;
                }
            };
            let canonical = SynthesizedRecipe::new(genome.clone())?.executable_sha256;
            if seen.insert(canonical.clone()) {
                backend.persist(&receipt.arm,&format!("generation-{}-variation-accepted-{:06}",generation+1,candidates.len()),&serde_json::json!({"seed":seed,"attempt":attempt,"variation":comparison.variation_identity,"origin":origin,"parents":ancestors,"canonical_executable_sha256":canonical,"genome":genome}))?;
                candidates.push(Candidate::new(
                    &receipt.arm,
                    genome,
                    generation + 1,
                    candidates.len(),
                    origin,
                    ancestors,
                    None,
                )?);
            }
            if candidates.len() == desired {
                break;
            }
        }
        if candidates.is_empty() {
            receipt.stop_reason = if backend.now_ms() >= deadline {
                "original_deadline"
            } else {
                "search_space_exhausted"
            }
            .into();
            break;
        }
    }
    receipt.validate_population(comparison)?;
    backend.persist(
        &receipt.arm,
        "search-receipt",
        &serde_json::to_value(&receipt)?,
    )?;
    Ok(receipt)
}

/// Preserve original nonattempt inventory at expiry, while authenticating every
/// measured result. Both final receipts and crash-interrupted observed journals
/// use the same rule; neither may conceal a later fit or changed fitted output.
fn verify_observed_replay(
    saved: &[Observation],
    feedback: &[Feedback],
    expired: bool,
) -> Result<()> {
    let mut current = BTreeMap::new();
    for item in feedback {
        ensure!(
            current.insert(&item.candidate_sha256, item).is_none(),
            "duplicate saved execution feedback"
        );
    }
    ensure!(
        current.len() == saved.len(),
        "saved execution cache changed"
    );
    for observation in saved {
        let replay = current
            .remove(&observation.candidate.id_sha256)
            .context("saved worker outcome missing")?;
        if expired
            && observation.feedback.retryable_inventory
            && observation.feedback.status != EvaluationStatus::Completed
        {
            let same_folds = observation
                .feedback
                .replay_evidence_sha256
                .as_ref()
                .is_some_and(|digest| {
                    jope_policy::is_sha256_hex(digest)
                        && replay.replay_evidence_sha256.as_ref() == Some(digest)
                });
            // Preparation failures and older known-zero inventory have no fold
            // digest. Unknown partial attempts never qualify for that fallback.
            let known_nonattempt = observation.feedback.replay_evidence_sha256.is_none()
                && observation.feedback.physical_fit_starts == Some(0)
                && replay.physical_fit_starts == Some(0);
            ensure!(
                replay.status != EvaluationStatus::Completed
                    && replay.loss.is_none()
                    && replay.physical_fit_starts.is_none_or(|n| n == 0)
                    && replay.fitted_artifact_sha256.is_none()
                    && replay.oof.is_none()
                    && (same_folds
                        || known_nonattempt
                        || serde_json::to_value(replay)?
                            == serde_json::to_value(&observation.feedback)?),
                "expired inventory gained or changed fitted evidence"
            );
        } else {
            ensure!(
                serde_json::to_value(replay)? == serde_json::to_value(&observation.feedback)?,
                "saved fitted evidence changed"
            );
        }
    }
    Ok(())
}

fn stream_seed(base: u64, generation: usize, attempt: usize) -> u64 {
    let digest = sha256_hex(format!("{PROTOCOL}/{base}/{generation}/{attempt}").as_bytes());
    u64::from_str_radix(&digest[..16], 16).expect("sha256 hex")
}

/// Concrete bridge to the one shared worker pool. The job builder owns the
/// fit-only input closure and must declare every output. The scorer verifies
/// fitted output bytes and performs development scoring outside the worker.
/// Pool access is message-based in the panel coordinator; only its supervisor
/// thread owns ExecutionPool. This trait does not authorize direct engine fits.
pub trait PoolAccess {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64>;
    fn run(&mut self, jobs: Vec<WorkerJob>, cancelled: &AtomicBool) -> Result<Vec<WorkerOutcome>>;
}
impl PoolAccess for ExecutionPool {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        ExecutionPool::begin_arm(self, arm)
    }
    fn run(&mut self, jobs: Vec<WorkerJob>, cancelled: &AtomicBool) -> Result<Vec<WorkerOutcome>> {
        ExecutionPool::run(self, jobs, cancelled)
    }
}
pub struct PoolBackend<'a, J, S, P = ExecutionPool> {
    pub pool: &'a mut P,
    pub cancelled: &'a AtomicBool,
    pub journal: PathBuf,
    pub make_job: J,
    pub score: S,
}
