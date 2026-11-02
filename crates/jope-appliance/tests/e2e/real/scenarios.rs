use super::*;
use super::{receipts::*, setup::*, transport::*};

pub(super) fn shared_run_surfaces() -> Result<()> {
    let bytes = input()?;
    let (server, spec) = Server::start(&bytes)?;
    let output = batch_command(server.home.path(), &spec).output()?;
    retain_execution(&server.engine, &output, "shared")?;
    let completed = snapshot(output)?;
    let id = &completed.run_id;
    let manifest = server.client.model_export(
        id,
        &jope_api::ExportRequest {
            algorithm_id: completed
                .best
                .as_ref()
                .context("missing champion")?
                .algorithm_id
                .clone(),
            kind: jope_api::ExportKind::Bundle,
            languages: vec![jope_api::Language::Python, jope_api::Language::Rust],
            require_all_languages: true,
            refit: jope_api::Refit::FitHoldout,
        },
        Some("int01-normal-fh-export"),
    )?;
    ensure!(
        manifest.refit == jope_api::Refit::FitHoldout && manifest.languages.len() == 2,
        "normal run did not finalize requested FitHoldout model"
    );
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        fs::create_dir_all(&root)?;
        fs::write(
            PathBuf::from(root).join("normal-fit-holdout-export.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    ndjson_receipt(server.home.path(), &spec, &completed, None)?;
    ndjson_receipt(
        server.home.path(),
        &spec,
        &completed,
        Some(completed.event_watermark),
    )?;
    let mut captures = vec![capture(&server.client, Surface::BatchCli, id)?];
    ensure!(
        captures[0].projection.champion == completed.best,
        "batch receipt differs from journal"
    );
    replay(&server.client, &spec, id)?;
    captures.push(capture(&server.client, Surface::LoopbackHttp, id)?);
    let inproc = server.inprocess()?;
    replay(&inproc, &spec, id)?;
    captures.push(capture(&inproc, Surface::InProcess, id)?);
    let mut rpc = Rpc::open(server.home.path())?;
    let mut args = serde_json::to_value(&spec)?;
    args["idempotency_key"] = json!(KEY);
    let started: RunSnapshot = serde_json::from_value(rpc.tool("run_start", args)?)?;
    ensure!(&started.run_id == id, "MCP retry created another run");
    let snapshot: RunSnapshot =
        serde_json::from_value(rpc.tool("run_status", json!({"run_id":id}))?)?;
    let page = rpc.tool("run_events", json!({"run_id":id,"after":"0","limit":256}))?;
    let next_after: DecimalU64 = serde_json::from_value(page["next_after"].clone())?;
    ensure!(
        next_after == snapshot.event_watermark,
        "MCP event page truncated"
    );
    captures.push(
        SurfaceCapture::from_observation(
            Surface::Mcp,
            snapshot,
            serde_json::from_value(page["events"].clone())?,
        )
        .map_err(anyhow::Error::msg)?,
    );
    let csv = server.home.path().join("original-label.csv");
    fs::write(&csv, &bytes)?;
    let transcript = terminal(server.home.path(), &csv, id, &captures[0].projection, KEY)?;
    ensure!(
        transcript.contains(&format!("run {id} started")),
        "PTY omitted actual run result"
    );
    captures.push(capture(&server.client, Surface::Terminal, id)?);
    let inventory = server.client.runs()?;
    ensure!(
        inventory.runs.len() == 1
            && &inventory.runs[0].run_id == id
            && inventory.next_after.is_none(),
        "surface retries created duplicate runs"
    );
    assert_equivalent(&captures).map_err(anyhow::Error::msg)?;
    let mut independent = vec![captures[0].clone()];
    independent.push(start_independent(
        &server.client,
        &spec,
        "int01-independent-http",
        Surface::LoopbackHttp,
    )?);
    independent.push(start_independent(
        &inproc,
        &spec,
        "int01-independent-inproc",
        Surface::InProcess,
    )?);
    let mut args = serde_json::to_value(&spec)?;
    args["idempotency_key"] = json!("int01-independent-mcp");
    let started: RunSnapshot = serde_json::from_value(rpc.tool("run_start", args)?)?;
    wait_complete(&server.client, &started.run_id)?;
    let snapshot =
        serde_json::from_value(rpc.tool("run_status", json!({"run_id":started.run_id}))?)?;
    let page = rpc.tool(
        "run_events",
        json!({"run_id":started.run_id,"after":"0","limit":256}),
    )?;
    independent.push(
        SurfaceCapture::from_observation(
            Surface::Mcp,
            snapshot,
            serde_json::from_value(page["events"].clone())?,
        )
        .map_err(anyhow::Error::msg)?,
    );
    let first_run_terminal = terminal(
        server.home.path(),
        &csv,
        id,
        &captures[0].projection,
        "int01-independent-terminal",
    )?;
    let inventory = server.client.runs()?;
    let existing: Vec<_> = independent
        .iter()
        .map(|c| &c.projection.events[0].run_id)
        .collect();
    let new_runs: Vec<_> = inventory
        .runs
        .iter()
        .filter(|r| !existing.contains(&&r.run_id))
        .collect();
    ensure!(
        new_runs.len() == 1 && inventory.runs.len() == 5 && inventory.next_after.is_none(),
        "independent surfaces did not create exactly five runs"
    );
    let terminal_id = &new_runs[0].run_id;
    ensure!(
        first_run_terminal.contains(&format!("run {terminal_id} started")),
        "terminal did not initiate its independent run"
    );
    independent.push(capture(&server.client, Surface::Terminal, terminal_id)?);
    for capture in &independent {
        let value = serde_json::to_value(capture.projection.events.last().unwrap())?;
        ensure!(
            value["body"]["nondeterminism"]["count"] == "0",
            "independent S1 run hit a limit exception"
        );
    }
    assert_independent_equivalent(&independent).map_err(anyhow::Error::msg)?;
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::write(root.join("independent-terminal.txt"), first_run_terminal)?;
        let raw: Vec<_> = independent.iter().map(|c| json!({"surface":c.surface.name(),"champion":c.projection.champion,"scores":c.projection.scores,"events":c.projection.events,"r2_projection":canonical_events(&c.projection.events)})).collect();
        fs::write(
            root.join("independent-five-surfaces.json"),
            serde_json::to_vec_pretty(&raw)?,
        )?;
    }
    ensure!(
        captures[0]
            .projection
            .events
            .iter()
            .any(|e| matches!(e.event, EventBody::ExperimentFinished(_))),
        "no native evaluations recorded"
    );
    {
        let root = std::env::var_os("JOPE_INT01_RECEIPTS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01-receipts")
            });
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("shared-snapshot.json"),
            serde_json::to_vec_pretty(&completed)?,
        )?;
        fs::write(
            root.join("shared-events.json"),
            serde_json::to_vec_pretty(&captures[0].projection.events)?,
        )?;
        fs::write(root.join("terminal.txt"), transcript)?;
        fs::write(root.join("input.csv"), &bytes)?;
        let runtime = directory()?;
        let runtime_name = runtime
            .path()
            .file_name()
            .context("AG1 runtime name")?
            .to_string_lossy();
        let evidence = root.join(format!("golden-mcp-{runtime_name}"));
        let app = server.home.path().join("jopedime");
        let version: Value = serde_json::from_slice(
            &Command::new(&app)
                .args(["--version", "--json"])
                .output()?
                .stdout,
        )?;
        super::golden_mcp::run(
            &app,
            version["source_sha40"].as_str().context("source missing")?,
            &csv,
            &jope_runtime::sha256_hex(&bytes),
            &evidence,
            &runtime.path().join("ag1-home"),
        )?;
    }
    Ok(())
}

pub(super) fn batch_fallback() -> Result<()> {
    let home = directory()?;
    let bytes = input()?;
    let (engine, spec) = initialize(home.path(), &bytes)?;
    drop(engine);
    let csv = home.path().join("original-label.csv");
    fs::write(&csv, &bytes)?;
    let template = batch_command(home.path(), &spec);
    let mut args: Vec<_> = template.get_args().map(std::ffi::OsStr::to_owned).collect();
    let at = args
        .iter()
        .position(|arg| arg == "--dataset")
        .context("dataset argument missing")?;
    args.drain(at..at + 2);
    args.insert(1, csv.as_os_str().to_owned());
    let out = home.path().join("csv-export");
    let output = Command::new(home.path().join("jopedime"))
        .env("JOPEDIME_HOME", home.path())
        .env_remove("JOPEDIME_ENDPOINT")
        .args(args)
        .arg("--out")
        .arg(&out)
        .output()?;
    let engine = InProcessEngine::open_with_worker(home.path(), home.path().join("jopedime"))?;
    retain_execution(&engine, &output, "batch")?;
    ensure!(
        out.join("src/python/predictor.py").is_file()
            && out.join("src/rust/predictor.rs").is_file(),
        "CSV batch did not export both requested languages"
    );

    let completed = snapshot(output)?;
    ensure!(
        !home.path().join("daemon.json").exists(),
        "batch fallback left a daemon"
    );
    let row = engine
        .store()
        .get(&RecordKey::Run(completed.run_id.clone()))?
        .context("committed run missing")?;
    let stored: RunSnapshot = serde_json::from_value(row.data["snapshot"].clone())?;
    ensure!(
        stored == completed,
        "batch did not return committed snapshot"
    );
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("batch-snapshot.json"),
            serde_json::to_vec_pretty(&completed)?,
        )?;
    }
    Ok(())
}
