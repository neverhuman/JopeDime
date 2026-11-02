use super::*;

/// Join retained attempts before aggregating cost. A started command without a
/// finished receipt remains an unknown attempt, never an implicit zero or fit.
struct CommandInventory {
    attempts: Vec<(jope_pipeline::execution::WorkerJob, Option<WorkerOutcome>)>,
    /// No started command exists to authenticate these against an experiment.
    /// They remain explicit non-launched inventory and cannot add predictors.
    unstarted_terminal: Vec<WorkerOutcome>,
}
fn command_inventory(journal: &Path, experiment_sha256: &str) -> Result<CommandInventory> {
    let mut started = BTreeMap::new();
    for entry in fs::read_dir(journal.join("started"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|s| s != "json") {
            continue;
        }
        ensure!(entry.file_type()?.is_file(), "nonphysical started receipt");
        let record: Value = serde_json::from_slice(&fs::read(entry.path())?)?;
        let job: jope_pipeline::execution::WorkerJob =
            serde_json::from_value(record["job"].clone())?;
        let id = job.identity()?;
        ensure!(
            entry.file_name() == format!("{id}.json").as_str()
                && record["job_sha256"] == id
                && job.request.experiment_sha256 == experiment_sha256
                && job.directory == journal.join("jobs").join(job.request.identity()?),
            "started cost receipt changed experiment/job/directory"
        );
        ensure!(
            started.insert(id, job).is_none(),
            "duplicate started cost receipt"
        );
    }
    let mut finished = BTreeMap::new();
    for entry in fs::read_dir(journal.join("finished"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|s| s != "json") {
            continue;
        }
        ensure!(entry.file_type()?.is_file(), "nonphysical finished receipt");
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(entry.path())?)?;
        ensure!(
            entry.file_name() == format!("{}.json", outcome.job_sha256).as_str()
                && jope_pipeline::experiment::sha256(&outcome.job_sha256)
                && jope_pipeline::experiment::sha256(&outcome.request_sha256),
            "misnamed or invalid finished cost receipt"
        );
        outcome.verify_outputs()?;
        ensure!(
            [
                outcome.consumption.cpu_seconds,
                outcome.consumption.wall_seconds
            ]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && v >= 0.),
            "invalid worker accounting"
        );
        if let Some(job) = started.get(&outcome.job_sha256) {
            ensure!(
                job.request.identity()? == outcome.request_sha256,
                "finished cost receipt changed request"
            );
        } else {
            ensure!(
                outcome.command_status != EvaluationStatus::Completed
                    && outcome.consumption.gate_released_unix_ms.is_none()
                    && outcome.exit_code.is_none()
                    && outcome.signal.is_none()
                    && outcome.outputs.is_empty()
                    && outcome.consumption.cpu_seconds.is_none_or(|v| v == 0.)
                    && outcome.consumption.wall_seconds.is_none_or(|v| v == 0.),
                "launched command lost started cost receipt"
            );
        }
        ensure!(
            finished
                .insert(outcome.job_sha256.clone(), outcome)
                .is_none(),
            "duplicate finished cost receipt"
        );
    }
    let attempts = started
        .into_iter()
        .map(|(id, job)| (job, finished.remove(&id)))
        .collect();
    Ok(CommandInventory {
        attempts,
        unstarted_terminal: finished.into_values().collect(),
    })
}

/// Include incomplete OOF candidates: their successful constituent fits still
/// cost resources and produce predictors even if five-fold selection failed.
/// Proposal/context commands are accounted separately from fitted predictors.
pub fn accounting(plan: &CampaignPlan, journal: &Path) -> Result<Value> {
    accounting_for_experiment(&plan.experiment.identity()?, journal)
}
pub(super) fn accounting_for_experiment(experiment_sha256: &str, journal: &Path) -> Result<Value> {
    use std::collections::BTreeSet;
    let inventory = command_inventory(journal, experiment_sha256)?;
    let missing_finished = inventory
        .attempts
        .iter()
        .filter(|(_, outcome)| outcome.is_none())
        .count();
    let unstarted_terminal = inventory.unstarted_terminal.iter().map(|outcome|
        json!({"job_sha256":outcome.job_sha256,"request_sha256":outcome.request_sha256,
            "status":outcome.command_status,"reason":outcome.reason,
            "cpu_seconds":outcome.consumption.cpu_seconds,"wall_seconds":outcome.consumption.wall_seconds,
            "basis":"no started receipt; inventory only, not an authenticated launched command or fitted predictor"})
    ).collect::<Vec<_>>();
    let mut cpu = 0.;
    let mut wall = 0.;
    let mut unknown_cpu = 0usize;
    let mut unknown_wall = 0usize;
    let mut commands = 0usize;
    let mut fitting_outputs = 0usize;
    let mut predictors = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    let mut diagnostic_commands = 0usize;
    let mut fitted_export_commands = 0usize;
    for (job, outcome) in inventory.attempts {
        commands += 1;
        if job.request.stage == EvaluationStage::Diagnostic {
            if job
                .command
                .args
                .iter()
                .any(|arg| arg == "fitted-export-worker")
            {
                fitted_export_commands += 1;
            } else {
                diagnostic_commands += 1;
            }
        }
        let Some(outcome) = outcome else {
            unknown_cpu += 1;
            unknown_wall += 1;
            continue;
        };
        match outcome.consumption.cpu_seconds {
            Some(v) => cpu += v,
            None => unknown_cpu += 1,
        }
        match outcome.consumption.wall_seconds {
            Some(v) => wall += v,
            None => unknown_wall += 1,
        }
        if job.request.stage == EvaluationStage::Diagnostic {
            continue;
        }
        if outcome.command_status != EvaluationStatus::Completed {
            continue;
        }
        let output = outcome
            .outputs
            .iter()
            .find(|a| a.path == job.directory.join("output.json"))
            .context("fitted output absent")?;
        let output: Value = serde_json::from_slice(&fs::read(&output.path)?)?;
        let fitted = match output.get("schema").and_then(Value::as_str) {
            Some("jope.panel-synthesized-worker-output/v1") => {
                let value: super::super::synthesized::SynthesizedWorkerOutput =
                    serde_json::from_value(output)?;
                value.artifact.validate()?;
                ensure!(
                    value.artifact_sha256 == jope_pipeline::sha256_hex(&value.artifact.bytes()?),
                    "generated cost artifact changed"
                );
                Some(value.artifact_sha256)
            }
            Some("jope.panel-rust-worker-output/v1") => {
                let value: super::super::worker::RustWorkerOutput = serde_json::from_value(output)?;
                value.artifact.validate()?;
                ensure!(
                    value.artifact_sha256 == jope_pipeline::sha256_hex(&value.artifact.bytes()?),
                    "native cost artifact changed"
                );
                Some(value.artifact_sha256)
            }
            _ => {
                if output.get("contract").and_then(Value::as_str)
                    == Some(super::super::methods::ADAPTER_CONTRACT)
                {
                    let hash = output["fitted"]["sha256"]
                        .as_str()
                        .context("external fitted checksum absent")?;
                    ensure!(
                        output["fitted"]["reload_verified"] == false
                            && outcome
                                .outputs
                                .iter()
                                .any(|a| a.path == job.directory.join("output.json.fitted.pkl")
                                    && a.sha256 == hash),
                        "external cost artifact absent"
                    );
                    Some(hash.to_owned())
                } else {
                    None
                }
            }
        };
        if let Some(hash) = fitted {
            ensure!(
                jope_pipeline::experiment::sha256(&hash),
                "invalid fitted output identity"
            );
            predictors.insert(hash);
            recipes.insert(job.request.recipe_sha256);
            fitting_outputs += 1;
        }
    }
    Ok(
        json!({"worker_command_receipts":commands,"context_or_proposal_commands":diagnostic_commands,"fitted_export_commands":fitted_export_commands,"completed_fitted_output_commands":fitting_outputs,
        "unique_fitted_predictors":predictors.len(),"unique_reusable_recipes":recipes.len(),"known_worker_cpu_seconds":cpu,"unknown_cpu_attempts":unknown_cpu,
        "known_sum_worker_wall_seconds":wall,"unknown_wall_attempts":unknown_wall,
        "started_commands_missing_finished_receipts":missing_finished,
        "unstarted_terminal_inventory":unstarted_terminal,
        "cpu_seconds_per_unique_predictor_known":(!predictors.is_empty()).then(||cpu/predictors.len() as f64),
        "cpu_seconds_per_unique_recipe_known":(!recipes.is_empty()).then(||cpu/recipes.len() as f64),
        "basis":"union of retained started and finished commands; started-only attempts have unknown CPU/wall and no fitted-output claim; unstarted terminal inventory is separate; unique complete fitted artifact bytes; no behavioral-equivalence claim",
        "controller_cpu_seconds":null,"gpu_seconds":null,"one_hour_ten_million_token_scenario":{"supplied_seconds":3600,"supplied_tokens":10000000,"token_usage_measured":null,"token_cost_comparison":"not_comparable_without_actual_token_and_price_receipts"}}),
    )
}
