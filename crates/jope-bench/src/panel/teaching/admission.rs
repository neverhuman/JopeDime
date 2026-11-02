use super::*;

pub(super) fn retain(state: &State, path: &Path) -> Result<String> {
    state.put_blob(&fs::read(path)?)
}
/// W3-T02: the dataset's teaching context, as the W2 producer run in `directory` (the neural
/// arm's `prepare_regression_context`) handed it off to teacher admission, retained with that
/// run's pool receipts where teacher admission looks for them. `None` when the run emitted no
/// handoff (the neural arm produced no context): the submission then carries no context and
/// teacher admission refuses (`MissingEncoderProvenance`). Teacher admission authenticates the
/// handoff against the preregistered producer and re-verifies the run from these receipts.
pub(super) fn handed_off_context(
    state: &State,
    freeze: &FreezeReceipt,
    directory: &Path,
) -> Result<Option<ContextEvidence>> {
    let handoff = directory.join(super::optimize::TEACHER_HANDOFF);
    if !handoff.try_exists()? {
        return Ok(None);
    }
    let files: CpuContextFiles =
        serde_json::from_slice(&fs::read(directory.join("context-files.json"))?)?;
    let run = &files.context_execution;
    let run = ContextRun {
        handoff: fs::read(&handoff)?,
        job: canonical_json(&run.job)?,
        outcome: run.outcome.read_verified()?,
        experiment: canonical_json(&run.experiment)?,
        window: run.window.read_verified()?,
        artifact: files.artifact.read_verified()?,
        context: files.vector.read_verified()?,
    };
    eligibility::retain_handed_off_context(state, freeze, &run).map(Some)
}
#[derive(Serialize, Deserialize)]
pub(super) struct Admission {
    pub(super) method: String,
    pub(super) candidate_sha256: String,
    pub(super) record: Option<HwmRecord>,
    pub(super) qualified: bool,
    pub(super) reason: Option<String>,
    pub(super) qualification: Option<eligibility::Qualification>,
}
#[derive(Serialize, Deserialize)]
pub(super) struct Report {
    pub(super) schema: String,
    pub(super) experiment_sha256: String,
    pub(super) dataset: String,
    pub(super) registry_sha256: String,
    pub(super) original_label_role: String,
    pub(super) effective_label_role: String,
    pub(super) baseline_worker_fits: u64,
    pub(super) entries: Vec<Admission>,
    pub(super) registration_unavailable: Option<String>,
}
#[allow(clippy::too_many_arguments)] // One explicit controller-owned admission closure.
pub(super) fn admit_one(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    state: &State,
    registration: &Registration,
    search: &DatasetSearch,
    method: &str,
    frozen: &FrozenRecipe,
    receipt: &SearchReceipt,
) -> Result<Admission> {
    let Registration::Frozen {
        comparison,
        partition,
        freeze,
        context,
    } = registration
    else {
        anyhow::bail!("teaching partition unavailable")
    };
    ensure!(
        plan.experiment
            .checkpoints
            .get(&format!("teaching_freeze/{}", freeze.comparison_sha256))
            == Some(&freeze.freeze_sha256),
        "original experiment lacks teaching freeze pin"
    );
    let partition_now = eligibility::derive_inner_partition(
        state,
        &plan.experiment.panel_sha256,
        &dataset.dataset.id,
        0,
        plan.experiment.seed,
        &plan.runtimes["rust"].program.sha256,
        &canonical_sha256(&plan.runtimes["rust"])?,
    )?;
    ensure!(
        partition_now.identity()? == partition.identity()?
            && comparison.vocabulary_sha256
                == config
                    .experimental
                    .get(dataset.dataset.task.as_str())
                    .map_or(&config.vocabulary, |c| &c.vocabulary)
                    .sha256,
        "teaching comparison changed"
    );
    frozen.candidate.recipe.validate()?;
    let receipt_sha = state.put_blob(&serde_json::to_vec(receipt)?)?;
    ensure!(
        receipt_sha == frozen.search_receipt_sha256,
        "frozen winner receipt differs"
    );
    let frozen_sha = state.put_blob(&serde_json::to_vec(frozen)?)?;
    let marker = eligibility::register_inner_candidate(state, freeze, &frozen_sha, &receipt_sha)?;
    let selection_sha = state.put_blob(&serde_json::to_vec(&search.pair.selection)?)?;
    let inner = InnerEvidence {
        selection_sha256: selection_sha,
        frozen_recipe_sha256: frozen_sha.clone(),
        search_receipt_sha256: receipt_sha,
        runtime_sha256: state.put_blob(&canonical_json(&plan.runtimes["rust"])?)?,
        arm_window_sha256: retain(
            state,
            &journal
                .join("arms")
                .join(format!("{}.json", receipt.arm.id_sha256)),
        )?,
        memory_policy_sha256: retain(state, &journal.join("fit-memory-policy.json"))?,
        candidate_freeze_sha256: marker,
    };
    let adapter = FitAdapter::prepare(
        dataset,
        &dataset.cut(0)?,
        search.pair.selection.clone(),
        plan.runtimes["rust"].clone(),
        journal,
        false,
    )?;
    let mut folds = Vec::new();
    let mut losses = Vec::new();
    let mut wall = Some(0f64);
    for request in search.pair.selection.selection_requests(
        &frozen.candidate.program_sha256,
        &frozen.candidate.id_sha256,
    )? {
        let mut job = adapter.job(&frozen.candidate, &request)?;
        job.arm = Some(receipt.arm.clone());
        let job_sha = job.identity()?;
        let worker_path = journal.join("finished").join(format!("{job_sha}.json"));
        let worker: WorkerOutcome = serde_json::from_slice(&fs::read(&worker_path)?)?;
        super::receipt_timing::completed_fit(journal, &job, &worker)?;
        let output =
            optimize_fit::checked_output(&job, &worker, &frozen.candidate.recipe, request.seed)?;
        let score = super::score::score_predictions(
            request.task,
            &request.classes,
            &adapter.folds[request.fold as usize].labels,
            &output.predictions,
        )?;
        losses.push(score.loss);
        wall = wall
            .zip(worker.consumption.wall_seconds)
            .map(|(a, b)| a + b);
        let predictions_sha = state.put_blob(&canonical_json(&output.predictions)?)?;
        let fitted_sha = state.put_blob(&output.artifact.bytes()?)?;
        ensure!(
            fitted_sha == output.artifact_sha256,
            "fitted model changed during retention"
        );
        let worker_sha = retain(state, &worker_path)?;
        let result = EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            request_sha256: request.identity()?,
            status: EvaluationStatus::Completed,
            reason: None,
            primary: Some(score.value),
            secondary: BTreeMap::from([(
                "physical_fit_starts".into(),
                output.physical_fit_starts as f64,
            )]),
            scored_rows: request.scoring_rows,
            predictions_sha256: Some(predictions_sha.clone()),
            fitted_state_sha256: Some(fitted_sha.clone()),
            consumption: Consumption {
                wall_seconds: worker.consumption.wall_seconds,
                cpu_seconds: worker.consumption.cpu_seconds,
                gpu_seconds: None,
                peak_rss_bytes: worker.consumption.peak_rss_bytes,
                cgroup_peak_memory_bytes: worker.consumption.cgroup_peak_memory_bytes,
                sampled_peak_rss_bytes: worker.consumption.sampled_peak_rss_bytes,
                accounting_reason: worker.consumption.accounting_reason.clone(),
            },
            worker_receipt_sha256: worker_sha,
            reused_from_request_sha256: None,
        };
        result.validate(&request)?;
        let started_path = journal.join("started").join(format!("{job_sha}.json"));
        let started: Value = serde_json::from_slice(&fs::read(&started_path)?)?;
        let candidate = FoldPredictions {
            schema: "jope.train-oof-predictions/v1".into(),
            comparison_sha256: comparison.sha256()?,
            program_sha256: frozen.candidate.program_sha256.clone(),
            fold: request.fold,
            training_rows_sha256: request.fitting_rows_sha256.clone(),
            scoring_rows_sha256: request.scoring_rows_sha256.clone(),
            row_ids: search
                .pair
                .selection
                .inner_rows(request.fold as usize)?
                .1
                .to_vec(),
            predictions_sha256: predictions_sha,
            evaluation_request_sha256: state.put_blob(&serde_json::to_vec(&request)?)?,
            worker_job_sha256: state.put_blob(&serde_json::to_vec(&job)?)?,
            fitted_artifact_sha256: fitted_sha,
            execution_receipt_sha256: state.put_blob(&serde_json::to_vec(&result)?)?,
            started_receipt_sha256: retain(state, &started_path)?,
            fit_started_unix: started["started_unix_ms"]
                .as_u64()
                .context("actual fit start absent")?
                / 1000,
        };
        folds.push(state.put_blob(&canonical_json(&candidate)?)?);
    }
    ensure!(losses.len() == 5, "incomplete fivefold teaching candidate");
    let loss = losses.iter().sum::<f64>() / 5.;
    let metric = match dataset.dataset.task {
        jope_datasets::catalog::Task::Regression => Metric::Rmse,
        jope_datasets::catalog::Task::Binary => Metric::Auc,
        jope_datasets::catalog::Task::Multiclass => Metric::Logloss,
        _ => anyhow::bail!("unsupported teaching task"),
    };
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity: comparison.clone(),
        identity_sha256: comparison.sha256()?,
        legacy_identity_sha256: None,
        genome: Genome {
            dsl: frozen.candidate.recipe.executable_canonical.clone(),
            program_sha256: frozen.candidate.program_sha256.clone(),
            semantic_sha256: None,
        },
        score: Score {
            metric,
            value: if metric == Metric::Auc {
                1. - loss
            } else {
                loss
            },
            direction: metric.direction(),
            units: if metric == Metric::Rmse {
                "original_target_units"
            } else {
                "dimensionless"
            }
            .into(),
        },
        cost: Cost {
            runtime_seconds: wall,
            search_seconds: None,
        },
        evaluator: Evaluator {
            sha256: comparison.evaluator_sha256.clone(),
            version: super::EVALUATOR.into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: true,
        provenance: Provenance {
            source: "asm-inner-oof".into(),
            run_id: format!(
                "{}/{}/{}",
                plan.experiment.identity()?,
                receipt.arm.id_sha256,
                frozen.candidate.id_sha256
            ),
            git_sha: plan.experiment.source_sha.clone(),
            bundle_id: search.pair.selection.identity()?,
            record_sha256: frozen_sha.clone(),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: jope_learning::ledger::now_unix(),
    };
    record.validate()?;
    // W3-T02: the dataset's context as the preregistered W2 producer handed it off. Without a
    // preregistered producer (no context is used) the submission carries none.
    let context = match context {
        Some(TeachingContext::Producer { .. }) => handed_off_context(
            state,
            freeze,
            &journal
                .join("asm-contexts")
                .join(search.pair.arm(true)?.id_sha256),
        )?,
        Some(TeachingContext::NoContext { .. }) | None => None,
    };
    let evidence = OofEvidence {
        freeze: freeze.clone(),
        experiment_blob_sha256: state.put_blob(&serde_json::to_vec(&plan.experiment)?)?,
        window_sha256: retain(state, &journal.join("window.json"))?,
        fold_predictions_sha256: folds,
        inner: Some(inner),
        context,
    };
    let result = eligibility::qualify(state, &record, &evidence);
    let (qualified, reason, qualification) = match result {
        Ok(value) => {
            hwm::ingest(
                state,
                None,
                vec![hwm::Candidate {
                    record: record.clone(),
                    blob: fs::read(state.blob_path(&frozen_sha))?,
                    chain_from_source: false,
                }],
                json!({"kind":"asm-inner-oof","experiment_sha256":plan.experiment.identity()?}),
            )?;
            (true, None, Some(value))
        }
        Err(e) => (false, Some(format!("{e:#}")), None),
    };
    Ok(Admission {
        method: method.into(),
        candidate_sha256: frozen.candidate.id_sha256.clone(),
        record: Some(record),
        qualified,
        reason,
        qualification,
    })
}
