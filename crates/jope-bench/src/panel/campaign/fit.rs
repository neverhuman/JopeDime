use super::*;

pub(crate) fn digest(value: &impl Serialize) -> Result<String> {
    Ok(jope_pipeline::sha256_hex(&serde_json::to_vec(value)?))
}

/// Create once or compare exact JSON value on resume. Never replace evidence.
pub(crate) fn immutable(path: &Path, value: &impl Serialize) -> Result<()> {
    if path.try_exists()? {
        let previous: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            previous == serde_json::to_value(value)?,
            "stale evidence: {}",
            path.display()
        );
        Ok(())
    } else {
        super::super::worker::write_private(path, value)
    }
}

pub(crate) fn request(
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    experiment: &Experiment,
    stage: EvaluationStage,
) -> Result<EvaluationRequest> {
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: experiment.identity()?,
        dataset_sha256: dataset.dataset.source.sha256.clone(),
        split_sha256: dataset.dataset.split.sha256.clone(),
        fitting_rows_sha256: cut.train.rows_sha256.clone(),
        scoring_rows_sha256: cut.hold.rows_sha256.clone(),
        fitting_rows: cut.train.rows.len(),
        scoring_rows: cut.hold.rows.len(),
        recipe_sha256: method.identity()?,
        evaluator_sha256: jope_pipeline::sha256_hex(super::super::EVALUATOR.as_bytes()),
        checkpoint_sha256: match method {
            Method::Synthesized {
                checkpoint_sha256, ..
            } => checkpoint_sha256.clone(),
            _ => None,
        },
        task: dataset.dataset.task,
        metric: dataset.dataset.metric,
        classes: dataset.dataset.classes.clone(),
        fold: cut.fold,
        stage,
        seed: experiment.seed,
        numerical_policy: experiment.numerical_policy.clone(),
    };
    request.validate()?;
    Ok(request)
}

/// Holdout labels never enter the subprocess input or argv. Original nulls,
/// category strings and training labels are retained at the Python boundary.
pub fn prepare_fit(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    stage: EvaluationStage,
    journal: &Path,
) -> Result<PendingFit> {
    prepare_fit_impl(plan, dataset, cut, method, stage, journal, true)
}

pub(crate) fn prepare_fit_impl(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    stage: EvaluationStage,
    journal: &Path,
    persist: bool,
) -> Result<PendingFit> {
    let request = request(dataset, cut, method, &plan.experiment, stage)?;
    let id = request.identity()?;
    let input = journal.join("inputs").join(format!("{id}.json"));
    let input_value = match method {
        Method::Synthesized { recipe, .. } => {
            serde_json::to_value(super::super::synthesized::SynthesizedFitRequest {
                task: dataset.dataset.task,
                schema: dataset.schema.clone(),
                classes: dataset.dataset.classes.clone(),
                train: cut.train.table.clone(),
                targets: cut.train.targets.clone(),
                predict: cut.hold.table.clone(),
                recipe: recipe.clone(),
                scheduling_seed: request.seed,
            })?
        }
        Method::Rust { recipe } => {
            let (_, input) =
                super::super::evaluation_input(dataset, cut, recipe, &plan.experiment, stage)?;
            serde_json::to_value(input)?
        }
        Method::External { .. } => json!({
            "contract":super::super::methods::ADAPTER_CONTRACT, "task":dataset.dataset.task,
            "columns":dataset.dataset.features, "classes":dataset.dataset.classes,
            "train":{"features":cut.train.rows.iter().map(|r| &r.values).collect::<Vec<_>>(),
                "targets":cut.train.labels},
            "predict":{"features":cut.hold.rows.iter().map(|r| &r.values).collect::<Vec<_>>()}
        }),
    };
    if persist {
        immutable(&input, &input_value)?;
    } else {
        let retained: Value = serde_json::from_slice(&fs::read(&input)?)?;
        ensure!(
            retained == input_value,
            "retained fit input changed original panel rows"
        );
    }
    let input_sha256 = jope_pipeline::sha256_hex(&fs::read(&input)?);
    let runtime = &plan.runtimes[method.runtime()];
    let directory = journal.join("jobs").join(id);
    let output = directory.join("output.json");
    let mut args = runtime.argument_prefix.clone();
    let mut outputs = vec![PathBuf::from("output.json")];
    match method {
        Method::Rust { .. } | Method::Synthesized { .. } => args.extend(
            [
                if matches!(method, Method::Synthesized { .. }) {
                    "synthesized-fit-worker"
                } else {
                    "fit-worker"
                },
                "--input",
            ]
            .map(str::to_owned)
            .into_iter()
            .chain([
                input.display().to_string(),
                "--output".into(),
                output.display().to_string(),
            ]),
        ),
        Method::External { .. } => {
            args.extend([
                "--panel-v2".into(),
                input.display().to_string(),
                output.display().to_string(),
            ]);
            outputs.push("output.json.fitted.pkl".into());
        }
    }
    let mut inputs = runtime.inputs.clone();
    inputs.push(Artifact {
        path: input,
        sha256: input_sha256.clone(),
    });
    let job = WorkerJob {
        request: request.clone(),
        command: WorkerCommand {
            // A declared missing dependency has an impossible-to-satisfy private
            // sentinel; it can never accidentally run a now-present executable.
            program: if runtime.unavailable.is_some() {
                Artifact {
                    path: journal.join("unavailable").join(method.runtime()),
                    sha256: "0".repeat(64),
                }
            } else {
                runtime.program.clone()
            },
            args,
            inputs,
            environment: runtime.environment.clone(),
        },
        directory,
        outputs,
        arm: None,
    };
    job.identity()?;
    Ok(PendingFit {
        method: method.clone(),
        request,
        job,
        input_sha256,
        expected_engine_version: runtime.version.clone(),
    })
}
