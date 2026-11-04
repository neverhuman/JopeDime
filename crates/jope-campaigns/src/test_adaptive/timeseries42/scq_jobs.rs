fn expected_job_id(
    run_id: &str,
    dataset: &str,
    generation: usize,
    shard: usize,
    node: &str,
    attempt: usize,
    manifest_sha256: &str,
) -> String {
    let binding =
        format!("{run_id}:{dataset}:{generation}:{shard}:{node}:{attempt}:{manifest_sha256}");
    format!("jope-ts42-{}", &sha256_bytes(binding.as_bytes())[..24])
}

fn scq_spec(
    node: &StagedNode,
    job_id: &str,
    remote_manifest: &str,
    remote_output: &str,
    project: &str,
) -> Result<String> {
    let args = [
        "RAYON_NUM_THREADS=1".to_string(),
        "RUST_NUM_THREADS=1".to_string(),
        "OMP_NUM_THREADS=1".to_string(),
        "OPENBLAS_NUM_THREADS=1".to_string(),
        "MKL_NUM_THREADS=1".to_string(),
        "VECLIB_MAXIMUM_THREADS=1".to_string(),
        "NUMEXPR_NUM_THREADS=1".to_string(),
        node.runner_path.clone(),
        "timeseries-test-adaptive-batch-worker".to_string(),
        "--manifest".to_string(),
        remote_manifest.to_string(),
        "--output-dir".to_string(),
        remote_output.to_string(),
        "--corpus-eval".to_string(),
        node.corpus_eval_path.clone(),
        "--timeout-seconds".to_string(),
        CANDIDATE_TIMEOUT_SECONDS.to_string(),
    ];
    let rendered_args = args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    let node_probe_args = [
        "-f".to_string(),
        node_marker_path(&node.bundle_root, &node.hostname),
    ];
    let rendered_node_probe_args = node_probe_args
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!(
        "[job]\nid = {job_id:?}\nproject = {project:?}\ncommand = \"/usr/bin/env\"\nargs = [{rendered_args}]\nworkdir = {workdir:?}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {SHARD_SIZE}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = {probe_name:?}\ncommand = \"/usr/bin/test\"\nargs = [{rendered_node_probe_args}]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 24\nmax_restarts = 0\n",
        workdir = node.bundle_root,
        probe_name = format!("TimeSeries42 staged node {}", node.node),
    ))
}

fn scq_json(args: &[&str]) -> Result<Value> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("--json").args(args);
            command
        },
        "TimeSeries42 SCQ JSON command",
    )?;
    serde_json::from_slice(&output.stdout).context("parse TimeSeries42 SCQ JSON")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScqWaitDecision {
    Terminal,
    CancelQueuedStall,
    Wait,
}

fn scq_job_has_last_start(job: &serde_json::Map<String, Value>) -> bool {
    job.get("last_start").is_some_and(|value| !value.is_null())
}

fn scq_wait_decision(
    state: &str,
    has_last_start: bool,
    queued_since: Option<Instant>,
    now: Instant,
    timeout: std::time::Duration,
) -> ScqWaitDecision {
    if matches!(state, "completed" | "failed" | "cancelled") {
        return ScqWaitDecision::Terminal;
    }
    if state == "queued"
        && !has_last_start
        && queued_since.is_some_and(|started| now.duration_since(started) >= timeout)
    {
        return ScqWaitDecision::CancelQueuedStall;
    }
    ScqWaitDecision::Wait
}

fn retryable_scq_stall_error(job_id: &str, reason: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "TimeSeries42 SCQ retryable stall ({SCQ_RETRYABLE_STALL_MARKER}): job {job_id} {reason}; launch-ready-capacity retry"
    )
}

fn cancel_scq_job(id: &str) -> Result<()> {
    scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("cancel").arg(id);
            command
        },
        "cancel TimeSeries42 stalled SCQ shard",
    )?;
    Ok(())
}

fn wait_for_job(id: &str) -> Result<Value> {
    let timeout = std::time::Duration::from_secs(SCQ_QUEUED_STALL_TIMEOUT_SECONDS);
    let mut queued_since = None;
    loop {
        let show = scq_json(&["show", id])?;
        let job = scq_batch::scq_job(&show, id)?;
        let state = job
            .get("state")
            .and_then(Value::as_str)
            .context("SCQ job has no state")?;
        let has_last_start = scq_job_has_last_start(job);
        if state == "queued" && !has_last_start {
            queued_since.get_or_insert_with(Instant::now);
        } else {
            queued_since = None;
        }
        match scq_wait_decision(state, has_last_start, queued_since, Instant::now(), timeout) {
            ScqWaitDecision::Terminal => return Ok(show),
            ScqWaitDecision::CancelQueuedStall => {
                cancel_scq_job(id)?;
                return Err(retryable_scq_stall_error(
                    id,
                    &format!(
                        "stayed queued without last_start for at least {SCQ_QUEUED_STALL_TIMEOUT_SECONDS}s"
                    ),
                ));
            }
            ScqWaitDecision::Wait => {}
        }
        thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn is_scq_capacity_error(error: &anyhow::Error) -> bool {
    let text = format!("{error:#}");
    text.contains(SCQ_RETRYABLE_STALL_MARKER)
        || text.contains("launch-ready-capacity")
        || text.contains("no eligible launch-ready node currently has the requested resources")
        || text.contains("Wait for capacity or reduce the resource request")
}

fn is_wrong_node_marker_capacity_error(error: &anyhow::Error, target_node: &str) -> bool {
    let text = format!("{error:#}");
    if !text.contains("dependencies unsatisfied on all eligible nodes")
        || !text.contains("smoke TimeSeries42 staged node")
    {
        return false;
    }
    let target_diagnostic =
        format!("{target_node}: smoke TimeSeries42 staged node {target_node}: smoke test exited");
    !text
        .lines()
        .map(str::trim_start)
        .any(|line| line.starts_with(&target_diagnostic))
}

fn is_scq_capacity_error_for_node(error: &anyhow::Error, target_node: &str) -> bool {
    is_scq_capacity_error(error) || is_wrong_node_marker_capacity_error(error, target_node)
}

fn scq_check(spec: &Path, target_node: &str) -> Result<()> {
    let result = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("check").arg(spec);
            command
        },
        "TimeSeries42 SCQ check",
    );
    match result {
        Ok(_) => Ok(()),
        Err(error) if is_scq_capacity_error_for_node(&error, target_node) => Err(error),
        Err(error) => Err(error),
    }
}

fn scq_job_exists(job_id: &str) -> bool {
    scq_json(&["show", job_id])
        .ok()
        .and_then(|show| scq_batch::scq_job(&show, job_id).ok().map(|_| ()))
        .is_some()
}

fn submit_batch(spec_paths: &[PathBuf]) -> Result<()> {
    if spec_paths.is_empty() {
        return Ok(());
    }
    scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("submit-batch");
            for spec_path in spec_paths {
                command.arg(spec_path);
            }
            command
        },
        "submit TimeSeries42 generation shard batch",
    )?;
    Ok(())
}

fn preflight_representative_scq_checks(
    root: &Path,
    staging: &StagingReceipt,
    run_id: &str,
) -> Result<()> {
    let project = if is_forecast3_gen20(run_id) {
        FORECAST3_PROJECT
    } else {
        LEGACY_PROJECT
    };
    let mut passed = 0usize;
    for node in &staging.nodes {
        let job_id = format!(
            "jope-ts42-preflight-{}-{}",
            node.node,
            &staging.release_binary_sha256[..12]
        );
        let spec = scq_spec(
            node,
            &job_id,
            &format!("{}/preflight/manifest.json", node.bundle_root),
            &format!("{}/preflight/{job_id}/output", node.bundle_root),
            project,
        )?;
        let spec_sha256 = sha256_bytes(spec.as_bytes());
        let spec_path = root.join("preflight").join("scq-checks").join(format!(
            "{}-{}.toml",
            node.node,
            &spec_sha256[..16]
        ));
        immutable_or_equal(&spec_path, spec.as_bytes())?;
        match scq_check(&spec_path, &node.node) {
            Ok(()) => passed += 1,
            Err(error)
                if node.node == "xbabe4" || is_scq_capacity_error_for_node(&error, &node.node) =>
            {
                eprintln!(
                    "TimeSeries42 representative SCQ check did not pass on {}; shard retries will route around it when needed: {error:#}",
                    node.node
                );
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("TimeSeries42 representative SCQ check on {}", node.node)
                });
            }
        }
    }
    if passed == 0 {
        eprintln!(
            "TimeSeries42 representative SCQ checks found no currently launch-ready staged node; per-shard retry will keep probing capacity"
        );
    }
    Ok(())
}
