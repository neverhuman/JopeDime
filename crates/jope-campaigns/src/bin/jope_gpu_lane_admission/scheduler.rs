fn validate_and_submit(
    args: &Args,
    state: &GpuLaneAdmissionV1,
    jobs: &[SchedulerJob],
) -> Result<bool> {
    if jobs
        .iter()
        .any(|job| job.id == state.immutable_job_identity)
    {
        return Ok(true);
    }
    ensure!(
        sha256_file(&state.immutable_spec_path)? == state.immutable_spec_sha256.as_str(),
        "immutable GPU spec hash drift"
    );
    let spec = fs::read_to_string(&state.immutable_spec_path)?;
    ensure!(
        spec.contains("gpus_per_task = 1") && spec.contains("cpus_reserved = 4"),
        "GPU spec does not reserve one GPU and four companion cores"
    );
    let output = Command::new(&args.scq)
        .arg("check")
        .arg(&state.immutable_spec_path)
        .output()?;
    if !output.status.success() {
        let message = command_message(&output);
        if capacity_backoff(&message) {
            return Ok(false);
        }
        bail!("dependency verification rejected immutable spec: {message}");
    }
    let output = Command::new(&args.scq)
        .arg("submit")
        .arg(&state.immutable_spec_path)
        .output()?;
    let message = command_message(&output);
    if output.status.success() || message.to_ascii_lowercase().contains("already exists") {
        return Ok(true);
    }
    if capacity_backoff(&message) {
        return Ok(false);
    }
    bail!("SCQ rejected immutable GPU submission: {message}")
}

fn command_message(output: &std::process::Output) -> String {
    String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned()
}

fn capacity_backoff(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity"))
}

fn scheduler_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    Ok(serde_json::from_value(
        value
            .get("Jobs")
            .and_then(|value| value.get("jobs"))
            .cloned()
            .context("SCQ jobs response is missing Jobs.jobs")?,
    )?)
}

fn node_capacity(value: &Value) -> Result<BTreeMap<String, NodeCapacity>> {
    let rows: Vec<NodeCapacity> = serde_json::from_value(
        value
            .get("Nodes")
            .and_then(|value| value.get("nodes"))
            .cloned()
            .context("SCQ nodes response is missing Nodes.nodes")?,
    )?;
    Ok(rows
        .into_iter()
        .map(|row| (row.node.clone(), row))
        .collect())
}

fn save_state(root: &Path, state: &GpuLaneAdmissionV1) -> Result<()> {
    state.validate()?;
    atomic_json(&state_path(root, state.role), &serde_json::to_value(state)?)
}

fn save_sentinel(root: &Path, sentinel: &PauseSentinel) -> Result<()> {
    sentinel.validate()?;
    atomic_json(&sentinel_path(root), &serde_json::to_value(sentinel)?)
}

fn archive_request(root: &Path, state: &GpuLaneAdmissionV1) -> Result<()> {
    let source = request_path(root, state.role);
    if !source.is_file() {
        return Ok(());
    }
    let directory = root.join("ops/gpu-lane-admission/completed");
    fs::create_dir_all(&directory)?;
    let target = directory.join(format!(
        "{}-{}-{}.json",
        state.role.as_str(),
        state.requested_at_ms,
        state.immutable_job_identity
    ));
    fs::rename(source, target)?;
    Ok(())
}

fn archive_sentinel(root: &Path, sentinel: &PauseSentinel) -> Result<()> {
    let source = sentinel_path(root);
    if !source.is_file() {
        return Ok(());
    }
    let directory = root.join("ops/gpu-lane-admission/completed");
    fs::create_dir_all(&directory)?;
    fs::rename(
        source,
        directory.join(format!("pause-{}.json", sentinel.created_at_ms)),
    )?;
    Ok(())
}

fn archived_exact_release(root: &Path, state: &GpuLaneAdmissionV1) -> Result<Option<Vec<String>>> {
    let directory = root.join("ops/gpu-lane-admission/completed");
    if !directory.is_dir() {
        return Ok(None);
    }
    let expected = state
        .held_evaluator_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut paths = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("pause-") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths.reverse();
    for path in paths {
        let sentinel: PauseSentinel = serde_json::from_value(read_json(&path)?)?;
        sentinel.validate()?;
        let held = sentinel
            .held_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let released = sentinel
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if sentinel.roles.contains(&state.role) && held == expected && released == expected {
            return Ok(Some(sentinel.released_evaluator_ids));
        }
    }
    Ok(None)
}

fn recover_orphan_sentinel(args: &Args, jobs: &[SchedulerJob]) -> Result<()> {
    let mut sentinel: PauseSentinel =
        serde_json::from_value(read_json(&sentinel_path(&args.root))?)?;
    recover_inflight(args, &mut sentinel, jobs)?;
    release_exact_evaluators(args, &mut sentinel)?;
    archive_sentinel(&args.root, &sentinel)
}

fn publish_status(
    args: &Args,
    states: &BTreeMap<GpuLaneRole, GpuLaneAdmissionV1>,
    nodes: &BTreeMap<String, NodeCapacity>,
    pause_active: bool,
) -> Result<()> {
    let alarms = states
        .values()
        .filter(|state| state.idle_alarm_due(now_ms()))
        .map(|state| {
            json!({
                "role": state.role,
                "job_id": state.immutable_job_identity,
                "idle_ms": now_ms().saturating_sub(state.eligible_idle_since_ms),
                "stage": state.stage,
                "last_capacity_backoff": state.last_capacity_backoff,
            })
        })
        .collect::<Vec<_>>();
    let status = json!({
        "schema": "GpuLaneAdmissionStatus.v1",
        "generated_at_ms": now_ms(),
        "pause_active": pause_active,
        "evaluator_holds_enabled": false,
        "gpu_companion_core_reserve_required": 4,
        "arbitration_order": ["cohort_proposer_first_when_runway_low", "independent_ready_roles_concurrent", "no_cross_node_suppression"],
        "retry_interval_ms": GPU_ADMISSION_RETRY_MS,
        "states": states,
        "nodes": nodes,
        "alarms": alarms,
    });
    atomic_json(
        &args.root.join("ops/gpu-lane-admission/status.v1.json"),
        &status,
    )?;
    atomic_json(
        &args.root.join("ops/gpu-lane-admission/alarm.v1.json"),
        &json!({
            "schema": "GpuLaneIdleAlarm.v1",
            "generated_at_ms": now_ms(),
            "active": !alarms.is_empty(),
            "alarms": alarms,
        }),
    )
}

fn append_event<T: Serialize>(root: &Path, kind: &str, detail: &T) -> Result<()> {
    let path = root.join("ops/gpu-lane-admission/events.v1.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "schema": "GpuLaneAdmissionEvent.v1",
            "kind": kind,
            "created_at_ms": now_ms(),
            "detail": detail,
        }),
    )?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}
