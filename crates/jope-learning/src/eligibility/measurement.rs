use super::*;

pub(super) fn measured(
    state: &State,
    record: &HwmRecord,
    evidence: &OofEvidence,
    live: bool,
) -> Result<Qualification> {
    record.validate()?;
    ensure!(
        !record.sealed && !record.benchmark_valid && record.replayable,
        "sealed or unbound record cannot teach"
    );
    let selected_task = task(&record.identity)?;
    let expected_metric = match selected_task {
        Task::Regression => Metric::Rmse,
        Task::Binary => Metric::Auc,
        Task::Multiclass => Metric::Logloss,
        _ => unreachable!(),
    };
    ensure!(
        record.score.metric == expected_metric,
        "metric differs from measured task protocol"
    );
    ensure!(
        record.score.units
            == if expected_metric == Metric::Rmse {
                "original_target_units"
            } else {
                "dimensionless"
            },
        "score units differ from measured task protocol"
    );
    let policy = load_freeze(state, &evidence.freeze)?;
    ensure!(
        policy.guard == GuardPolicy::current(),
        "unknown teaching guard"
    );
    // W12-T05: nothing in the candidate's closure may be revoked evidence, and neither may any
    // ancestor of it (the imported closure already contains every descendant of a revoked node).
    let revocations = crate::revocation::Revocations::load(state)?;
    let mut closure = vec![
        record.provenance.record_sha256.as_str(),
        evidence.freeze.freeze_sha256.as_str(),
        evidence.experiment_blob_sha256.as_str(),
        evidence.window_sha256.as_str(),
        policy.targets_sha256.as_str(),
        policy.panel_sha256.as_str(),
    ];
    closure.extend(evidence.fold_predictions_sha256.iter().map(String::as_str));
    if let Some(inner) = &evidence.inner {
        closure.extend([
            inner.selection_sha256.as_str(),
            inner.frozen_recipe_sha256.as_str(),
            inner.search_receipt_sha256.as_str(),
            inner.runtime_sha256.as_str(),
            inner.arm_window_sha256.as_str(),
            inner.memory_policy_sha256.as_str(),
            inner.candidate_freeze_sha256.as_str(),
        ]);
    }
    revocations.refuse("the teaching candidate", closure)?;
    // The teacher's experiment and original window, held to the freeze below. The teaching
    // context must have been produced inside exactly this execution (PR #114 review round 1), so
    // they are read first; the context check still runs before the inner binding, the folds and
    // any blob retention.
    let experiment: Experiment = blob(state, &evidence.experiment_blob_sha256)?;
    let window: ExecutionWindow = blob(state, &evidence.window_sha256)?;
    let (context_receipt_sha256, context_handoff_sha256) = context::verify(
        state,
        &policy,
        record,
        evidence,
        &revocations,
        &context::Teaching {
            experiment: &experiment,
            window: &window,
        },
    )?;
    experiment.validate()?;
    window.validate(&experiment)?;
    let inner_binding = inner::bind(state, &policy, evidence, &experiment, &window, live)?;
    ensure!(
        (inner_binding.is_some()
            || (experiment.checkpoints.get("teaching_freeze")
                == Some(&evidence.freeze.freeze_sha256)
                && experiment.checkpoints.get("teaching_vocabulary")
                    == Some(&record.identity.vocabulary_sha256)))
            && experiment.panel_sha256 == policy.panel_sha256
            && experiment.protocol == record.identity.protocol
            && experiment.seed == policy.evaluation_seed
            && experiment.numerical_policy == NUMERICAL_POLICY
            && window.started_unix_ms >= policy.created_at_unix * 1000,
        "experiment does not commit to the preoptimization freeze"
    );
    ensure!(
        record.identity == policy.comparison
            && record.identity_sha256 == evidence.freeze.comparison_sha256,
        "stale comparison identity"
    );
    ensure!(
        evidence.fold_predictions_sha256.len() == 5 && policy.folds.len() == 5,
        "complete five-fold OOF required"
    );
    let classes = &record.identity.comparison.as_ref().unwrap().class_order;
    let rows: Vec<TargetRow> = blob(state, &policy.targets_sha256)?;
    let mut fold_losses = Vec::new();
    let mut fold_rows_scored = Vec::new();
    let mut pooled = Vec::new();
    let mut improved_folds = 0;
    // Parsed only when a fold's fitted artifact names its fitting input (W3-T04).
    let mut development: Option<DevelopmentData> = None;
    let mut bound_folds = 0;
    for (fold, frozen) in policy.folds.iter().enumerate() {
        ensure!(frozen.fold as usize == fold, "frozen fold order changed");
        let (train, hold) = fold_rows(&rows, fold as u8);
        let candidate: FoldPredictions = blob(state, &evidence.fold_predictions_sha256[fold])?;
        ensure!(
            candidate.schema == PREDICTIONS
                && candidate.fold as usize == fold
                && candidate.comparison_sha256 == record.identity_sha256
                && candidate.program_sha256 == record.genome.program_sha256
                && candidate.fit_started_unix >= policy.created_at_unix,
            "prediction provenance differs from pre-optimization freeze"
        );
        ensure!(
            candidate.training_rows_sha256 == frozen.training_rows_sha256
                && candidate.training_rows_sha256 == rows_sha(&train)?
                && candidate.scoring_rows_sha256 == frozen.scoring_rows_sha256
                && candidate.scoring_rows_sha256 == rows_sha(&hold)?
                && candidate.row_ids == hold.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            "OOF row identity or order changed"
        );
        require_blob(state, &candidate.fitted_artifact_sha256)?;
        let fitted: serde_json::Value = blob(state, &candidate.fitted_artifact_sha256)?;
        let schema = fitted.get("schema").and_then(|v| v.as_str());
        ensure!(
            (((schema == Some(RUST_FITTED) || schema == Some(RUST_FITTED_V2))
                && fitted.pointer("/method/dsl").and_then(|v| v.as_str())
                    == Some(&record.genome.dsl))
                || (inner_binding.is_some()
                    && schema == Some("jope.panel-fitted-synthesized/v1")
                    && fitted
                        .pointer("/recipe/executable_canonical")
                        .and_then(|v| v.as_str())
                        == Some(&record.genome.dsl)))
                && fitted
                    .pointer("/model/genome_sha256")
                    .and_then(|v| v.as_str())
                    == Some(&record.genome.program_sha256),
            "native fitted recipe differs from teaching candidate"
        );
        // W3-T04: a v2 artifact names the exact fitting input it was fitted on; on the train-side
        // OOF route that must be this fold's authenticated training partition. A v1 artifact
        // names none and is unbound, as every artifact before v2.
        let named = fitted.get("fitting_input_sha256");
        ensure!(
            (schema == Some(RUST_FITTED_V2)) == named.is_some(),
            "FittedInputMismatch: fold {fold}'s fitted artifact must name its fitting input \
             exactly when it is {RUST_FITTED_V2}"
        );
        if let (Some(named), None) = (named, &inner_binding) {
            let expected =
                fold_fitting_input(state, &policy.dataset, &mut development, fold as u8)?;
            ensure!(
                named.as_str() == Some(expected.as_str()),
                "FittedInputMismatch: fold {fold}'s fitted artifact was fitted on another input \
                 than the fold's authenticated training partition"
            );
            bound_folds += 1;
        }
        let request: EvaluationRequest = blob(state, &candidate.evaluation_request_sha256)?;
        let result: EvaluationResult = blob(state, &candidate.execution_receipt_sha256)?;
        revocations.refuse(
            &format!("fold {fold} of the teaching candidate"),
            [
                candidate.predictions_sha256.as_str(),
                candidate.fitted_artifact_sha256.as_str(),
                candidate.evaluation_request_sha256.as_str(),
                candidate.worker_job_sha256.as_str(),
                candidate.execution_receipt_sha256.as_str(),
                candidate.started_receipt_sha256.as_str(),
                result.worker_receipt_sha256.as_str(),
                frozen.baseline_predictions_sha256.as_str(),
            ],
        )?;
        request.validate()?;
        result.validate(&request)?;
        ensure!(
            result.reused_from_request_sha256.is_none(),
            "reused predictions are not a new OOF teaching observation"
        );
        let worker: WorkerOutcome = blob(state, &result.worker_receipt_sha256)?;
        let job: WorkerJob = blob(state, &candidate.worker_job_sha256)?;
        if live {
            worker.verify_outputs()?;
        }
        ensure!(
            worker.schema == EXECUTION_SCHEMA && !worker.outputs.is_empty(),
            "invalid worker receipt"
        );
        for artifact in worker
            .outputs
            .iter()
            .chain(worker.stdout.iter())
            .chain(worker.stderr.iter())
        {
            retain_artifact(state, artifact, live)?;
        }
        let started: serde_json::Value = blob(state, &candidate.started_receipt_sha256)?;
        let started_job: WorkerJob =
            serde_json::from_value(started.get("job").context("started job missing")?.clone())?;
        let started_ms = started
            .get("started_unix_ms")
            .and_then(|v| v.as_u64())
            .context("start time missing")?;
        let deadline_ms = started
            .get("fit_deadline_unix_ms")
            .and_then(|v| v.as_u64())
            .context("fit deadline missing")?;
        ensure!(
            started_job.identity()? == job.identity()?
                && started.get("job_sha256").and_then(|v| v.as_str()) == Some(&worker.job_sha256)
                && started_ms >= window.started_unix_ms
                && candidate.fit_started_unix == started_ms / 1000
                && deadline_ms > started_ms
                && deadline_ms <= window.deadline_unix_ms
                && deadline_ms - started_ms <= experiment.resources.per_fit_seconds * 1000,
            "worker started receipt differs from original admission"
        );
        ensure!(
            worker.request_sha256 == request.identity()?
                && worker.job_sha256 == job.identity()?
                && job.request.identity()? == request.identity()?
                && worker.command_status == EvaluationStatus::Completed
                && worker.exit_code == Some(0)
                && worker.signal.is_none()
                && worker.reason.is_none(),
            "OOF lacks a clean bound worker execution"
        );
        retain_artifact(state, &job.command.program, live)?;
        ensure!(
            !job.command.inputs.is_empty(),
            "worker input closure is missing"
        );
        for input in &job.command.inputs {
            retain_artifact(state, input, live)?;
        }
        ensure!(
            worker
                .outputs
                .iter()
                .map(|artifact| artifact.path.clone())
                .collect::<BTreeSet<_>>()
                == job
                    .outputs
                    .iter()
                    .map(|path| job.directory.join(path))
                    .collect::<BTreeSet<_>>()
                && worker.outputs.len() == job.outputs.len(),
            "worker output inventory changed"
        );
        ensure!(
            result.consumption.wall_seconds == worker.consumption.wall_seconds
                && result.consumption.cpu_seconds == worker.consumption.cpu_seconds
                && result.consumption.peak_rss_bytes == worker.consumption.peak_rss_bytes
                && result.consumption.cgroup_peak_memory_bytes
                    == worker.consumption.cgroup_peak_memory_bytes
                && result.consumption.sampled_peak_rss_bytes
                    == worker.consumption.sampled_peak_rss_bytes
                && result.consumption.accounting_reason == worker.consumption.accounting_reason,
            "resource receipt changed or unknown consumption was replaced"
        );
        let context = record.identity.comparison.as_ref().unwrap();
        if let Some(binding) = &inner_binding {
            inner::validate_execution(
                binding,
                &request,
                &job,
                &worker,
                started_ms,
                deadline_ms,
                &record.genome.program_sha256,
            )?;
        }
        ensure!(
            request.experiment_sha256 == experiment.identity()?
                && request.dataset_sha256 == record.identity.dataset_hash
                && (inner_binding.is_some() || request.split_sha256 == record.identity.cut_id)
                && request.fitting_rows_sha256 == frozen.training_rows_sha256
                && request.scoring_rows_sha256 == frozen.scoring_rows_sha256
                && request.fitting_rows == train.len()
                && request.scoring_rows == hold.len()
                && request.recipe_sha256 == record.genome.program_sha256
                && request.evaluator_sha256 == record.identity.evaluator_sha256
                && request.checkpoint_sha256 == context.checkpoint_sha256
                && request.task == selected_task
                && request.classes == *classes
                && request.fold as usize == fold
                && request.stage == EvaluationStage::TrainOof
                && (inner_binding.is_some() || request.seed == policy.evaluation_seed)
                && request.numerical_policy == context.numerical_policy,
            "execution request differs from frozen train-side OOF comparison"
        );
        ensure!(
            result.status == EvaluationStatus::Completed
                && result.predictions_sha256.as_ref() == Some(&candidate.predictions_sha256)
                && result.fitted_state_sha256.as_ref() == Some(&candidate.fitted_artifact_sha256),
            "OOF execution is incomplete or names different artifacts"
        );
        let predictions: Predictions = blob(state, &candidate.predictions_sha256)?;
        verify_measured_outputs(
            state,
            &worker,
            &candidate,
            &predictions,
            inner_binding.is_some(),
        )?;
        if inner_binding.is_some() {
            ensure!(
                result.secondary.get("physical_fit_starts") == Some(&1.),
                "inner physical fit count changed"
            );
        }
        let baseline_predictions: Predictions = blob(state, &frozen.baseline_predictions_sha256)?;
        ensure!(
            baseline_predictions == baseline(selected_task, classes, &train, hold.len())?,
            "baseline differs from train-only trivial predictor"
        );
        let targets = hold
            .iter()
            .map(|row| row.target.clone())
            .collect::<Vec<_>>();
        let c = score_predictions(selected_task, classes, &targets, &predictions)?;
        let b = score_predictions(selected_task, classes, &targets, &baseline_predictions)?;
        ensure!(
            result.primary == Some(c.value),
            "execution score differs from stored predictions"
        );
        improved_folds += usize::from(policy.guard.improves(c.loss, b.loss));
        fold_losses.push((c.loss, b.loss));
        fold_rows_scored.push(hold.len());
        if let (Predictions::Values(c), Predictions::Values(b)) =
            (&predictions, &baseline_predictions)
        {
            for ((row, c), b) in hold.iter().zip(c).zip(b) {
                pooled.push((row.target.parse::<f64>()?, *c, *b));
            }
        }
    }
    let candidate_loss = fold_losses
        .iter()
        .map(|(candidate, _)| candidate)
        .sum::<f64>()
        / 5.0;
    let baseline_loss = fold_losses
        .iter()
        .map(|(_, baseline)| baseline)
        .sum::<f64>()
        / 5.0;
    let candidate_value = if selected_task == Task::Binary {
        1.0 - candidate_loss
    } else {
        candidate_loss
    };
    ensure!(
        candidate_value == record.score.value,
        "claimed HWM score differs from measured mean-fold predictions"
    );
    let summary = diagnostics(&fold_losses, fold_rows_scored, &pooled)?;
    let fitted_inputs_bound = (bound_folds == policy.folds.len()).then_some(true);
    Ok(Qualification {
        schema: QUALIFICATION.into(),
        comparison: record.identity.clone(),
        freeze: evidence.freeze.clone(),
        record_sha256: record.provenance.record_sha256.clone(),
        program_sha256: record.genome.program_sha256.clone(),
        targets_sha256: policy.targets_sha256,
        baseline_predictions_sha256: policy
            .folds
            .iter()
            .map(|fold| fold.baseline_predictions_sha256.clone())
            .collect(),
        fold_predictions_sha256: evidence.fold_predictions_sha256.clone(),
        guard: policy.guard,
        label_role: policy.label_role,
        original_label_role: policy.original_label_role,
        label_burn_sha256: policy.label_burn_sha256,
        candidate_loss,
        baseline_loss,
        improved_folds,
        fold_losses,
        context_fitting_rows_sha256:policy.inner_partition.as_ref().map(|p|p.fitting_rows_sha256()).transpose()?,
        context_outer_fold:policy.inner_partition.as_ref().map(|p|p.outer_fold),
        context_receipt_sha256: Some(context_receipt_sha256),
        context_handoff_sha256: Some(context_handoff_sha256),
        fitted_inputs_bound,
        diagnostics: Some(summary),
        measurement:
            "recomputed-from-pinned-targets-and-predictions; analytic-baseline-worker-fits=0; execution receipts retained separately"
                .into(),
    })
}
