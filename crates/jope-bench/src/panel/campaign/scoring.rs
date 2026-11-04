use super::*;

pub(crate) fn checked_predictions(
    fit: &PendingFit,
    outcome: &WorkerOutcome,
) -> Result<(Predictions, String)> {
    #[cfg(feature = "asm")]
    if matches!(fit.method, Method::Synthesized { .. }) {
        let journal = fit
            .job
            .directory
            .parent()
            .and_then(|p| p.parent())
            .context("fit outside journal")?;
        super::super::receipt_timing::completed_fit(journal, &fit.job, outcome)?;
    }
    outcome.verify_outputs()?;
    ensure!(
        outcome.job_sha256 == fit.job.identity()?
            && outcome.request_sha256 == fit.request.identity()?,
        "stale worker receipt"
    );
    let expected_paths: BTreeSet<_> = fit
        .job
        .outputs
        .iter()
        .map(|p| fit.job.directory.join(p))
        .collect();
    ensure!(
        outcome.outputs.len() == expected_paths.len()
            && outcome
                .outputs
                .iter()
                .map(|a| a.path.clone())
                .collect::<BTreeSet<_>>()
                == expected_paths,
        "worker output inventory changed"
    );
    let bytes = fs::read(fit.job.directory.join("output.json"))?;
    match &fit.method {
        Method::Synthesized { recipe, .. } => {
            let result: super::super::synthesized::SynthesizedWorkerOutput =
                serde_json::from_slice(&bytes)?;
            ensure!(
                result.schema == "jope.panel-synthesized-worker-output/v1"
                    && result.input_sha256 == fit.input_sha256
                    && result.executable_recipe_sha256 == recipe.executable_sha256
                    && result.physical_fit_starts == 1
                    && result.artifact.scheduling_seed == fit.request.seed
                    && result.artifact.task == fit.request.task
                    && result.artifact.classes == fit.request.classes
                    && digest(&result.artifact.recipe)? == digest(recipe)?,
                "generated fitted envelope changed"
            );
            result.artifact.validate()?;
            ensure!(
                result.artifact_sha256 == jope_pipeline::sha256_hex(&result.artifact.bytes()?),
                "generated fitted artifact checksum mismatch"
            );
            Ok((result.predictions, result.artifact_sha256))
        }
        Method::Rust { recipe } => {
            let result: RustWorkerOutput = serde_json::from_slice(&bytes)?;
            ensure!(
                result.schema == "jope.panel-rust-worker-output/v1"
                    && result.input_sha256 == fit.input_sha256,
                "worker input or schema mismatch"
            );
            result.artifact.validate()?;
            ensure!(
                &result.artifact.method == recipe
                    && result.artifact.task == fit.request.task
                    && result.artifact.classes == fit.request.classes,
                "fitted envelope changed"
            );
            ensure!(
                result.artifact_sha256 == jope_pipeline::sha256_hex(&result.artifact.bytes()?),
                "complete fitted artifact checksum mismatch"
            );
            Ok((result.predictions, result.artifact_sha256))
        }
        Method::External { .. } => {
            let result: Value = serde_json::from_slice(&bytes)?;
            ensure!(
                result["contract"] == super::super::methods::ADAPTER_CONTRACT
                    && result["input_sha256"] == fit.input_sha256
                    && result["engine_version"] == fit.expected_engine_version,
                "adapter contract, input or version changed"
            );
            let fitted = &result["fitted"];
            let artifact_path = fit.job.directory.join("output.json.fitted.pkl");
            let artifact = outcome
                .outputs
                .iter()
                .find(|a| a.path == artifact_path)
                .context("missing external fitted artifact")?;
            ensure!(
                fitted["path"] == "output.json.fitted.pkl"
                    && fitted["sha256"] == artifact.sha256
                    && fitted["reload_verified"] == false
                    && fitted["format"] == "jope.panel-fitted-python/v1"
                    && fitted["preprocessing_policy"] == super::super::methods::INPUT_POLICY
                    && fitted["bytes"].as_u64() == Some(fs::metadata(&artifact_path)?.len()),
                "external fitted artifact metadata mismatch"
            );
            Ok((
                serde_json::from_value(result["predictions"].clone())?,
                artifact.sha256.clone(),
            ))
        }
    }
}

pub fn score_fit(
    fit: &PendingFit,
    outcome: &WorkerOutcome,
    labels: &[String],
) -> Result<Observation> {
    ensure!(
        outcome.schema == "jope.worker-execution/v2"
            && outcome.job_sha256 == fit.job.identity()?
            && outcome.request_sha256 == fit.request.identity()?,
        "stale worker outcome"
    );
    let mut result = EvaluationResult {
        schema: RESULT_SCHEMA.into(),
        request_sha256: fit.request.identity()?,
        status: outcome.command_status,
        reason: outcome.reason.clone(),
        primary: None,
        secondary: BTreeMap::new(),
        scored_rows: 0,
        predictions_sha256: None,
        fitted_state_sha256: None,
        worker_receipt_sha256: digest(outcome)?,
        reused_from_request_sha256: None,
        consumption: Consumption {
            wall_seconds: outcome.consumption.wall_seconds,
            cpu_seconds: outcome.consumption.cpu_seconds,
            gpu_seconds: None,
            peak_rss_bytes: outcome.consumption.peak_rss_bytes,
            cgroup_peak_memory_bytes: outcome.consumption.cgroup_peak_memory_bytes,
            sampled_peak_rss_bytes: outcome.consumption.sampled_peak_rss_bytes,
            accounting_reason: outcome.consumption.accounting_reason.clone(),
        },
    };
    if outcome.command_status == EvaluationStatus::Completed {
        let scored = (|| -> Result<_> {
            let (predictions, artifact) = checked_predictions(fit, outcome)?;
            let score = score::score_predictions(
                fit.request.task,
                &fit.request.classes,
                labels,
                &predictions,
            )?;
            Ok((score.value, digest(&predictions)?, artifact))
        })();
        match scored {
            Ok((score, predictions, artifact)) => {
                result.primary = Some(score);
                result.scored_rows = labels.len();
                result.predictions_sha256 = Some(predictions);
                result.fitted_state_sha256 = Some(artifact);
            }
            Err(error) => {
                result.status = EvaluationStatus::Failed;
                result.reason = Some(format!("controller rejected output: {error:#}"));
            }
        }
    }
    result.validate(&fit.request)?;
    Ok(Observation {
        method: fit.method.id().into(),
        request: fit.request.clone(),
        result,
    })
}

/// Fold zero is the same physical fit as screening. Reuse its authenticated
/// predictor and observations, while retaining the original resource receipt.
pub(crate) fn reuse_fold_zero(
    screen: &Observation,
    confirm: EvaluationRequest,
) -> Result<Observation> {
    let mut original = confirm.clone();
    original.stage = EvaluationStage::Screen;
    ensure!(
        original.identity()? == screen.request.identity()?,
        "reuse changes the original cut"
    );
    let mut result = screen.result.clone();
    result.reused_from_request_sha256 = Some(screen.request.identity()?);
    result.request_sha256 = confirm.identity()?;
    result.consumption = Consumption {
        wall_seconds: Some(0.),
        cpu_seconds: Some(0.),
        accounting_reason: "reused screen predictor; physical cost retained on original receipt"
            .into(),
        ..Default::default()
    };
    result.validate(&confirm)?;
    Ok(Observation {
        method: screen.method.clone(),
        request: confirm,
        result,
    })
}

pub(crate) fn score_batch(
    fits: &[PendingFit],
    outcomes: Vec<WorkerOutcome>,
    data: &[DevelopmentData],
    journal: &Path,
) -> Result<Vec<Observation>> {
    let mut by_request = BTreeMap::new();
    for outcome in outcomes {
        ensure!(
            by_request
                .insert(outcome.request_sha256.clone(), outcome)
                .is_none(),
            "duplicate worker result"
        );
    }
    ensure!(
        by_request.len() == fits.len(),
        "incomplete worker inventory"
    );
    let mut observations = Vec::new();
    for fit in fits {
        let dataset = data
            .iter()
            .find(|d| d.dataset.source.sha256 == fit.request.dataset_sha256)
            .context("unknown worker dataset")?;
        let cut = dataset.cut(fit.request.fold)?;
        let outcome = by_request
            .get(&fit.request.identity()?)
            .context("worker outcome absent")?;
        let observation = score_fit(fit, outcome, &cut.hold.labels)?;
        immutable(
            &journal
                .join("observations")
                .join(format!("{}.json", fit.request.identity()?)),
            &observation,
        )?;
        observations.push(observation);
    }
    Ok(observations)
}
