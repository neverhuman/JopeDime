use super::*;

pub fn recover(directory: &Path, prepared: &PreparedExport) -> Result<Value> {
    let path = directory.join("run.json");
    let mut reports = Vec::new();
    if path.try_exists()? {
        let plan: RunPlan = read(&path)?;
        verify_plan(&plan, prepared)?;
        let mut previous_settled = true;
        for entry in &prepared.entries {
            let mut report = backend_report(&plan, prepared, entry)?;
            if report["status"] == "qualified" {
                ensure!(
                    previous_settled,
                    "qualification order lacks prior backend evidence"
                );
            }
            if !previous_settled && report["status"] == "unattempted" {
                report["reason"] =
                    json!("prior backend is not qualified or explicitly unsupported");
            }
            previous_settled &=
                report["status"] == "qualified" || report["status"] == "unsupported";
            reports.push(report);
        }
    } else {
        reports = prepared.entries.iter().map(|entry|match entry {Entry::Unsupported{backend,reason}=>json!({"backend":backend,"status":"unsupported","reason":reason}),Entry::Prepared{backend,qualification,..}=>json!({"backend":backend,"status":"unattempted","qualification_sha256":qualification.identity().ok(),"cost":null})}).collect();
    }
    let (plan, journal) = campaign(prepared)?;
    Ok(
        json!({"all_backends_qualified":reports.iter().all(|r|r["status"]=="qualified"),"backends":reports,"experiment_sha256":plan.experiment.identity()?,"original_window":read::<Value>(&journal.join("window.json"))?,"budget":"original campaign execution cutoff; no additional allowance","cost_scope":if plan.scope=="asm-panel" {"qualification costs are a subset of this campaign's worker command accounting; do not add twice"} else {"qualification costs are additional to comparator fit-observation accounting"},"preparation_cost":{"wall_seconds":prepared.preparation_wall_seconds,"cpu_seconds":prepared.preparation_cpu_seconds,"scope":"preparation through public materialization and verification is additional to worker counters; final receipt publication excluded","receipt_publication_cpu_seconds":null},"controller_cpu_seconds":null,"fits":0}),
    )
}
pub(super) fn may_submit(status: &Value) -> bool {
    status == "unattempted" || status == "unknown_started_without_finished"
}

pub fn run(args: QualifyArgs) -> Result<Value> {
    let prepared_artifact = fitted_export::artifact(&args.prepared)?;
    let prepared: PreparedExport = read(&prepared_artifact.path)?;
    ensure!(
        prepared.identity()? == args.prepared_sha256,
        "prepared export identity changed"
    );
    prepared.verify()?;
    let tools = fitted_export::artifact(&args.tools)?;
    ensure!(
        tools.sha256 == args.tools_sha256,
        "tool inventory checksum changed"
    );
    let mut required = args
        .require
        .iter()
        .map(|name| backend_named(name))
        .collect::<Result<Vec<_>>>()?;
    required.sort_by_key(|backend| fitted_export::ORDER.iter().position(|b| b == backend));
    required.dedup();
    let plan = RunPlan {
        schema: "jope.fitted-export-run/v1".into(),
        prepared: prepared_artifact.clone(),
        prepared_sha256: args.prepared_sha256.clone(),
        runtimes: read(&tools.path)?,
        tools,
        required,
    };
    verify_plan(&plan, &prepared)?;
    let (campaign, journal) = campaign(&prepared)?;
    let directory = fitted_export::location(&journal, &prepared.observation)?;
    ensure!(
        prepared_artifact.path == directory.join("prepared.json"),
        "export is outside the original campaign"
    );
    ensure!(
        equal(
            &fitted_export::artifact(&std::env::current_exe()?)?,
            &prepared.program
        )?,
        "running CLI differs from original fitted source"
    );
    campaign::immutable(&directory.join("run.json"), &plan)?;
    if prepared
        .entries
        .iter()
        .all(|entry| matches!(entry, Entry::Unsupported { .. }))
    {
        return Ok(
            json!({"ok":true,"execution_started":false,"report":recover(&directory,&prepared)?}),
        );
    }
    if let Some(unit) = args.supervisor_unit {
        let mut pool = ExecutionPool::open(
            &campaign.experiment,
            &journal,
            jope_pipeline::supervisor::for_unit(&unit)?,
        )?;
        let cancellation = jope_pipeline::supervisor::cancellation()?;
        prepared.verify_executable_plans()?;
        for entry in &prepared.entries {
            if matches!(entry, Entry::Unsupported { .. }) {
                continue;
            }
            let report = backend_report(&plan, &prepared, entry)?;
            if report["status"] == "qualified" {
                continue;
            }
            if !may_submit(&report["status"]) {
                break;
            }
            let (input, job) = derived(&plan, &prepared, entry)?;
            fs::create_dir_all(journal.join("export-inputs"))?;
            campaign::immutable(input_path(&job)?, &input)?;
            pool.run(vec![job], &cancellation)?;
            if backend_report(&plan, &prepared, entry)?["status"] != "qualified" {
                break;
            }
        }
        // W9-T05: every qualified backend is sealed, installed and recorded from its
        // authenticated receipts; required and optional outcomes are assessed, never released.
        let records = record_packages(&directory, &plan, &prepared)?;
        let assessment = assess_records(&directory, &plan, &prepared)?;
        let report = recover(&directory, &prepared)?;
        return Ok(
            json!({"ok":report["backends"].as_array().context("missing backends")?.iter().all(|r|r["status"]=="qualified" || r["status"]=="unsupported"),"report":report,"packages":records,"package_assessment":assessment}),
        );
    }
    let mut arguments = campaign.runtimes["rust"].argument_prefix.clone();
    arguments.extend([
        "qualify-export".into(),
        "--prepared".into(),
        prepared_artifact.path.display().to_string(),
        "--prepared-sha256".into(),
        args.prepared_sha256,
        "--tools".into(),
        plan.tools.path.display().to_string(),
        "--tools-sha256".into(),
        args.tools_sha256,
    ]);
    let launch = jope_pipeline::supervisor::launch(
        &campaign.experiment,
        &journal,
        &prepared.program,
        &arguments,
    )?;
    let report = recover(&directory, &prepared)?;
    let complete = report["backends"]
        .as_array()
        .context("missing backends")?
        .iter()
        .all(|r| r["status"] == "qualified" || r["status"] == "unsupported");
    Ok(json!({"ok":launch.exit_code==Some(0) && complete,"launch":launch,"report":report}))
}
pub(super) fn normal(name: &str) -> bool {
    let path = Path::new(name);
    !name.is_empty()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        && path.components().count() == 1
}
fn now_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
/// This endpoint is always scheduled by ExecutionPool: its descendants inherit
/// the same worker cgroup, one-thread environment, memory cap and cancellation.
pub fn worker(input: &Path, output: &Path) -> Result<()> {
    ensure!(
        output == Path::new("predictions.json"),
        "unexpected qualification output"
    );
    let input: WorkerInput = read(input)?;
    ensure!(
        input.schema == "jope.fitted-export-worker/v1" && !input.commands.is_empty(),
        "invalid qualification worker input"
    );
    for (name, source) in &input.files {
        ensure!(
            normal(name)
                && ![
                    "commands.json",
                    "predictions.json",
                    "stdout.log",
                    "stderr.log"
                ]
                .contains(&&name[..]),
            "unsafe generated filename"
        );
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(name)?;
        file.write_all(source.as_bytes())?;
    }
    let directory = std::env::current_dir()?;
    let mut receipts = Vec::new();
    for (index, args) in input.commands.iter().enumerate() {
        ensure!(!args.is_empty(), "empty generated command");
        let mut command;
        let program;
        if args[0] == "./qualification" {
            ensure!(
                index > 0 && args.len() == 1,
                "unexpected generated executable"
            );
            program = fitted_export::artifact(&directory.join("qualification"))?;
            command = Command::new(&program.path);
        } else {
            let runtime = input
                .runtimes
                .get(&args[0])
                .context("unregistered generated command")?;
            ensure!(
                runtime.unavailable.is_none() && runtime.argument_prefix.is_empty(),
                "unavailable generated command"
            );
            for item in std::iter::once(&runtime.program)
                .chain(&runtime.inputs)
                .chain(&runtime.package_files)
            {
                item.verify()?;
            }
            program = runtime.program.clone();
            command = Command::new(&program.path);
            command.envs(&runtime.environment);
        }
        let stdout = directory.join(format!("command-{index}.stdout"));
        let stderr = directory.join(format!("command-{index}.stderr"));
        command
            .args(&args[1..])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&stdout)?)
            .stderr(fs::File::create(&stderr)?);
        let started_unix_ms = now_ms()?;
        let clock = Instant::now();
        let status = command.status()?;
        let finished_unix_ms = now_ms()?;
        use std::os::unix::process::ExitStatusExt;
        receipts.push(CommandReceipt {
            arguments: args.clone(),
            program,
            started_unix_ms,
            finished_unix_ms,
            wall_seconds: clock.elapsed().as_secs_f64(),
            exit_code: status.code(),
            signal: status.signal(),
            stdout: fitted_export::artifact(&stdout)?,
            stderr: fitted_export::artifact(&stderr)?,
        });
        // Incremental failure evidence survives an unsuccessful later command.
        fs::write("commands.json", serde_json::to_vec_pretty(&receipts)?)?;
        ensure!(
            status.success(),
            "generated command {index} failed: {status}"
        );
        if index + 1 == input.commands.len() {
            fs::copy(stdout, output)?;
        }
    }
    Ok(())
}
