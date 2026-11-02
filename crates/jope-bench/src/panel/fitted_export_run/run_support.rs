use super::*;

/// W9-T05: seal, install (once) and record the package of every qualified backend. An
/// existing installation must load under the re-derived identity; an existing record must
/// equal the re-derived one.
pub(super) fn record_packages(
    directory: &Path,
    plan: &RunPlan,
    prepared: &PreparedExport,
) -> Result<Vec<PackageRecord>> {
    let mut records = Vec::new();
    for entry in &prepared.entries {
        let Some((manifest, files)) = sealed_package(plan, prepared, entry)? else {
            continue;
        };
        let backend = entry.backend();
        let identity = inference::package::identity(&manifest)?;
        let installed = package_directory(directory, backend);
        if !installed.try_exists()? {
            fs::create_dir_all(directory.join(PACKAGES))?;
            ensure!(
                inference::package::install(&installed, &manifest, &files)? == identity,
                "installed package identity differs from its seal"
            );
        }
        inference::package::load(&installed, &identity)
            .context("installed package differs from its authenticated seal")?;
        let record = expected_record(prepared, backend, &manifest)?;
        campaign::immutable(&record_path(directory, backend), &record)?;
        records.push(record);
    }
    Ok(records)
}

/// W9-T05: required/optional outcome of the recorded packages
/// (`inference::package::assess`); every emitted backend not required is optional.
pub(super) fn assess_records(
    directory: &Path,
    plan: &RunPlan,
    prepared: &PreparedExport,
) -> Result<Value> {
    let request = inference::package::PackageRequest {
        required: plan.required.clone(),
        optional: prepared
            .entries
            .iter()
            .filter(|entry| matches!(entry, Entry::Prepared { .. }))
            .map(Entry::backend)
            .filter(|backend| !plan.required.contains(backend))
            .collect(),
    };
    let mut outcomes = Vec::new();
    for backend in request.required.iter().chain(&request.optional) {
        let path = record_path(directory, *backend);
        let outcome = if path.try_exists()? {
            let record: PackageRecord = read(&path)?;
            inference::package::load(&package_directory(directory, *backend), &record.identity)
                .map_err(|error| format!("{error:#}"))
        } else {
            Err("not qualified: no package".to_owned())
        };
        outcomes.push((*backend, outcome));
    }
    inference::package::assess(&request, &outcomes)
}

/// Only public model code and unlabeled prediction inputs reach the bounded child.
/// Native expected predictions, campaign data paths and scoring labels stay out.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerInput {
    schema: String,
    files: BTreeMap<String, String>,
    commands: Vec<Vec<String>>,
    runtimes: BTreeMap<String, Runtime>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandReceipt {
    arguments: Vec<String>,
    program: Artifact,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    wall_seconds: f64,
    exit_code: Option<i32>,
    signal: Option<i32>,
    stdout: Artifact,
    stderr: Artifact,
}
fn equal(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}
pub(super) fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        fs::metadata(path)?.len() <= 256 * 1024 * 1024,
        "export receipt exceeds 256 MiB"
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub(super) fn campaign(prepared: &PreparedExport) -> Result<(CampaignPlan, PathBuf)> {
    let plan: CampaignPlan = read(&prepared.campaign_plan.path)?;
    let journal = prepared
        .original_outcome
        .path
        .parent()
        .and_then(Path::parent)
        .context("original journal missing")?
        .to_path_buf();
    ensure!(
        prepared.original_outcome.path
            == journal
                .join("finished")
                .join(format!("{}.json", prepared.original_job.identity()?)),
        "original journal changed"
    );
    Ok((plan, journal))
}
pub(super) fn verify_plan(plan: &RunPlan, prepared: &PreparedExport) -> Result<()> {
    ensure!(
        plan.schema == "jope.fitted-export-run/v1"
            && plan.prepared_sha256 == prepared.identity()?,
        "export run binding changed"
    );
    plan.prepared.verify()?;
    plan.tools.verify()?;
    ensure!(
        equal(&read::<PreparedExport>(&plan.prepared.path)?, prepared)?
            && equal(
                &read::<BTreeMap<String, Runtime>>(&plan.tools.path)?,
                &plan.runtimes
            )?,
        "export input inventory changed"
    );
    Ok(())
}
fn needed(commands: &[Vec<String>]) -> impl Iterator<Item = &str> {
    commands
        .iter()
        .filter_map(|args| args.first().map(String::as_str))
        .filter(|name| *name != "./qualification")
}
fn unavailable(plan: &RunPlan, commands: &[Vec<String>]) -> Option<String> {
    for name in needed(commands) {
        let Some(runtime) = plan.runtimes.get(name) else {
            return Some(format!("tool {name} has no pinned inventory"));
        };
        if let Some(reason) = &runtime.unavailable {
            return Some(format!("{name}: {reason}"));
        }
        if runtime.version.trim().is_empty() || !runtime.argument_prefix.is_empty() {
            return Some(format!(
                "{name}: missing version or unsupported argument prefix"
            ));
        }
        for item in std::iter::once(&runtime.program)
            .chain(&runtime.inputs)
            .chain(&runtime.package_files)
        {
            if let Err(error) = item.verify() {
                return Some(format!("{name}: {error:#}"));
            }
        }
    }
    None
}
pub(super) fn derived(
    plan: &RunPlan,
    prepared: &PreparedExport,
    entry: &Entry,
) -> Result<(WorkerInput, WorkerJob)> {
    let Entry::Prepared { qualification, .. } = entry else {
        anyhow::bail!("unsupported export has no execution job")
    };
    let (campaign, journal) = campaign(prepared)?;
    let mut runtimes = BTreeMap::new();
    for name in needed(&qualification.commands) {
        runtimes.insert(
            name.to_owned(),
            plan.runtimes
                .get(name)
                .context("missing tool inventory")?
                .clone(),
        );
    }
    let input = WorkerInput {
        schema: "jope.fitted-export-worker/v1".into(),
        files: qualification.files.clone(),
        commands: qualification.commands.clone(),
        runtimes,
    };
    let mut request = prepared.observation.request.clone();
    request.stage = EvaluationStage::Diagnostic;
    request.recipe_sha256 = qualification.identity()?;
    request.evaluator_sha256 = campaign::digest(&(&plan.tools, &input))?;
    request.numerical_policy = qualification.backend.numerical_policy();
    let id = request.identity()?;
    let input_path = journal.join("export-inputs").join(format!("{id}.json"));
    // write_private uses pretty JSON with a final newline.
    let mut bytes = serde_json::to_vec_pretty(&input)?;
    bytes.push(b'\n');
    let mut inputs = vec![
        Artifact {
            path: input_path.clone(),
            sha256: super::methods::digest(&bytes),
        },
        plan.tools.clone(),
    ];
    for runtime in input.runtimes.values() {
        inputs.push(runtime.program.clone());
        inputs.extend(runtime.inputs.clone());
        inputs.extend(runtime.package_files.clone());
    }
    inputs.sort_by(|a, b| a.path.cmp(&b.path));
    inputs.dedup_by(|a, b| a.path == b.path && a.sha256 == b.sha256);
    let mut args = campaign.runtimes["rust"].argument_prefix.clone();
    args.extend([
        "fitted-export-worker".into(),
        "--input".into(),
        input_path.display().to_string(),
        "--output".into(),
        "predictions.json".into(),
    ]);
    let mut outputs: Vec<PathBuf> = vec!["predictions.json".into(), "commands.json".into()];
    outputs.extend(input.files.keys().map(PathBuf::from));
    for index in 0..input.commands.len() {
        outputs.extend([
            format!("command-{index}.stdout").into(),
            format!("command-{index}.stderr").into(),
        ]);
    }
    match qualification.backend {
        inference::Backend::Rust => outputs.push("qualification".into()),
        inference::Backend::Cpp17 | inference::Backend::Cpp20 => outputs.push("model.so".into()),
        _ => {}
    }
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program: prepared.program.clone(),
            args,
            inputs,
            environment: BTreeMap::from([
                ("PATH".into(), "/usr/bin:/bin".into()),
                ("LC_ALL".into(), "C.UTF-8".into()),
            ]),
        },
        directory: journal.join("jobs").join(id),
        outputs,
        arm: None,
    };
    job.identity()?;
    for runtime in input.runtimes.values() {
        let mut environment_guard = job.clone();
        environment_guard.command.environment = runtime.environment.clone();
        environment_guard.identity()?;
    }
    Ok((input, job))
}
pub(super) fn input_path(job: &WorkerJob) -> Result<&Path> {
    job.command
        .inputs
        .iter()
        .find(|item| {
            item.path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|n| n == "export-inputs")
        })
        .map(|item| item.path.as_path())
        .context("worker payload missing")
}
fn original_receipt(journal: &Path, job: &WorkerJob) -> Result<Option<WorkerOutcome>> {
    let path = journal
        .join("finished")
        .join(format!("{}.json", job.identity()?));
    if !path.try_exists()? {
        return Ok(None);
    }
    let outcome: WorkerOutcome = read(&path)?;
    ensure!(
        outcome.job_sha256 == job.identity()?
            && outcome.request_sha256 == job.request.identity()?,
        "qualification execution identity changed"
    );
    outcome.verify_outputs()?;
    Ok(Some(outcome))
}
pub(super) fn backend_report(
    plan: &RunPlan,
    prepared: &PreparedExport,
    entry: &Entry,
) -> Result<Value> {
    let Entry::Prepared {
        backend,
        qualification,
        ..
    } = entry
    else {
        let Entry::Unsupported { backend, reason } = entry else {
            unreachable!()
        };
        return Ok(json!({"backend":backend,"status":"unsupported","reason":reason}));
    };
    let dependency_failure = unavailable(plan, &qualification.commands);
    if needed(&qualification.commands).any(|name| !plan.runtimes.contains_key(name)) {
        return Ok(
            json!({"backend":backend,"status":"unavailable","reason":dependency_failure,"cost":null}),
        );
    }
    let (input, job) = derived(plan, prepared, entry)?;
    let (_, journal) = campaign(prepared)?;
    let id = job.identity()?;
    let Some(outcome) = original_receipt(&journal, &job)? else {
        let started_path = journal.join("started").join(format!("{id}.json"));
        let started = started_path.try_exists()?;
        if started {
            let value: Value = read(&started_path)?;
            ensure!(
                value["job_sha256"] == id && value["job"] == serde_json::to_value(&job)?,
                "incomplete qualification lost started identity"
            );
        }
        return Ok(
            json!({"backend":backend,"status":if started {"unknown_started_without_finished"} else if dependency_failure.is_some() {"unavailable"} else {"unattempted"},"reason":dependency_failure,"job_sha256":id,"cost":null}),
        );
    };
    ensure!(
        equal(&read::<WorkerInput>(input_path(&job)?)?, &input)?,
        "qualification payload changed"
    );
    let started = journal.join("started").join(format!("{id}.json"));
    if started.try_exists()? {
        let value: Value = read(&started)?;
        ensure!(
            value["job_sha256"] == id && value["job"] == serde_json::to_value(&job)?,
            "started qualification changed"
        );
    } else {
        ensure!(
            outcome.command_status != EvaluationStatus::Completed
                && outcome.consumption.gate_released_unix_ms.is_none()
                && outcome.outputs.is_empty()
                && outcome.exit_code.is_none()
                && outcome.signal.is_none()
                && outcome.consumption.cpu_seconds.is_none_or(|v| v == 0.)
                && outcome.consumption.wall_seconds.is_none_or(|v| v == 0.),
            "launched qualification lost started receipt"
        );
    }
    ensure!(
        [
            outcome.consumption.cpu_seconds,
            outcome.consumption.wall_seconds
        ]
        .into_iter()
        .flatten()
        .all(|v| v.is_finite() && v >= 0.),
        "invalid qualification execution cost"
    );
    let mut result = json!({"backend":backend,"status":"failed","execution_status":outcome.command_status,"reason":outcome.reason,"job_sha256":id,"request_sha256":job.request.identity()?,"outcome":fitted_export::artifact(&journal.join("finished").join(format!("{id}.json")))?,"cost":outcome.consumption});
    if outcome.command_status != EvaluationStatus::Completed {
        return Ok(result);
    }
    ensure!(
        dependency_failure.is_none(),
        "completed qualification has stale tool dependencies: {dependency_failure:?}"
    );
    super::receipt_timing::completed_fit(&journal, &job, &outcome)?;
    let dependencies =
        jope_pipeline::execution::inventory_dependencies(std::slice::from_ref(&job))?;
    ensure!(
        dependencies.len() == 1 && dependencies[0].available,
        "qualification input closure changed"
    );
    let commands: Vec<CommandReceipt> = read(&job.directory.join("commands.json"))?;
    ensure!(
        commands.len() == qualification.commands.len(),
        "incomplete compiler/predictor commands"
    );
    for (index, (actual, expected)) in commands.iter().zip(&qualification.commands).enumerate() {
        ensure!(
            actual.stdout.path == job.directory.join(format!("command-{index}.stdout"))
                && actual.stderr.path == job.directory.join(format!("command-{index}.stderr")),
            "compiler/predictor log location changed"
        );
        ensure!(
            &actual.arguments == expected
                && actual.exit_code == Some(0)
                && actual.signal.is_none()
                && actual.finished_unix_ms >= actual.started_unix_ms
                && actual.wall_seconds.is_finite()
                && actual.wall_seconds >= 0.,
            "compiler/predictor receipt changed"
        );
        let program = if expected[0] == "./qualification" {
            fitted_export::artifact(&job.directory.join("qualification"))?
        } else {
            input.runtimes[&expected[0]].program.clone()
        };
        ensure!(
            equal(&actual.program, &program)?,
            "executed compiler/predictor differs"
        );
        actual.program.verify()?;
        actual.stdout.verify()?;
        actual.stderr.verify()?;
        ensure!(
            actual.started_unix_ms
                >= outcome
                    .consumption
                    .gate_released_unix_ms
                    .context("qualification gate missing")?
                && actual.finished_unix_ms
                    <= outcome
                        .consumption
                        .observed_exit_unix_ms
                        .context("qualification exit missing")?,
            "compiler/predictor escaped bounded worker"
        );
    }
    for (name, source) in &input.files {
        ensure!(
            fs::read(job.directory.join(name))? == source.as_bytes(),
            "executed public model or unlabeled input changed"
        );
    }
    let output = fs::read(job.directory.join("predictions.json"))?;
    ensure!(
        super::methods::digest(&output) == commands.last().context("zero commands")?.stdout.sha256,
        "prediction output differs from executed program stdout"
    );
    match inference::verify_qualification(qualification, &qualification.identity()?, &output) {
        Ok(parity) => {
            result["status"] = json!("qualified");
            result["parity"] = serde_json::to_value(parity)?;
            result["bounded_execution_verified"] = json!(true);
        }
        Err(error) => {
            result["status"] = json!("failed_parity");
            result["reason"] = json!(format!("{error:#}"));
        }
    }
    Ok(result)
}
