use super::*;

// ---------------------------------------------------------------------------
// cycle
// ---------------------------------------------------------------------------

pub(crate) fn read_state(path: &Path) -> Result<Value> {
    match fs::symlink_metadata(path) {
        Ok(_) => read_json(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(json!({"schema": STATE_SCHEMA}))
        }
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

pub(crate) fn attempt_for(state: &Value, key: &str) -> u64 {
    state
        .get("attempts")
        .and_then(|attempts| attempts.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1)
}

pub(crate) fn record_attempt(state: &mut Value, key: &str, attempt: u64) {
    let object = state.as_object_mut().expect("state is an object");
    let attempts = object
        .entry("attempts")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(map) = attempts.as_object_mut() {
        map.insert(key.to_owned(), json!(attempt));
    }
}

/// Lanes this cycle drives, in priority order.
///
/// The combined whole-catalog lane is ALWAYS present and ALWAYS first. It is the
/// lane that writes `proposals/current.json`, which is the pointer every
/// consumer actually reads (`selfplay_runtime.rs:107`, `supply.rs:96`,
/// `freshness.rs:409`, `alarms.rs:1166`, `trainer_watchdog.rs:1171`) and the
/// only lane whose output the wave/queue path can publish today. Dropping it
/// while the branches disagree would freeze the DL seed set at whatever catalog
/// it was last bound to: this unit exists to "regenerate gen-0 DL seeds whenever
/// the live catalog grows", and every dataset added after the first per-task
/// promotion would silently never receive one. The per-task lanes are ADDITIVE
/// and queue behind it.
pub(crate) fn ordered_lanes(combined: Lane, task_lanes: Option<Vec<Lane>>) -> Vec<Lane> {
    let mut lanes = vec![combined];
    lanes.extend(task_lanes.unwrap_or_default());
    lanes
}

pub(crate) fn cycle(args: &Args) -> Result<Value> {
    let root = args.root.as_path();
    let events = root.join("ops/propose-events.v1.jsonl");
    let state_path = root.join("ops/propose-driver-v1.json");

    let live = read_live_catalog(root)?;
    let binding = read_proposal_binding(root)?;

    let checkpoint =
        PathBuf::from(checkpoints_str()).join(format!("{}.model.pt", binding.checkpoint_sha256));
    if !checkpoint.is_file() {
        bail!(
            "accepted checkpoint artifact is absent: {}",
            checkpoint.display()
        );
    }
    let combined = Lane {
        task: None,
        checkpoint_sha256: binding.checkpoint_sha256.clone(),
        checkpoint,
        bound_catalog_sha256: binding.bound_catalog_sha256.clone(),
    };
    // A malformed per-task pointer, or a branch checkpoint whose artifact has
    // been reaped, degrades to the historical combined lane instead of aborting
    // the cycle. A fault in an additive lane must never take down the lane the
    // GP consumes.
    let (task_lanes, degraded) = match read_task_lanes(root, &binding) {
        Ok(lanes) => (lanes, None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    let forked = task_lanes.is_some();
    let lanes = ordered_lanes(combined, task_lanes);
    if !forked {
        let mut outcome = run_lane(
            args,
            &live,
            &binding,
            &lanes[0],
            &events,
            &state_path,
            false,
        )?;
        if let Some(reason) = degraded {
            outcome["task_lanes_degraded"] = json!(reason);
            append_event(
                &events,
                &json!({
                    "schema": EVENT_SCHEMA,
                    "at_ms": now_ms() as u64,
                    "action": "task_lanes_degraded",
                    "reason": outcome["task_lanes_degraded"],
                }),
            )?;
        }
        return Ok(outcome);
    }
    // Forked. One GPU generation per cycle across all branches: two multi-hour
    // propose jobs at once double the draw the trainer competes with, and the
    // branches are independent, so serialising them costs latency and nothing
    // else. Installs are pure filesystem work and are never deferred. The
    // combined lane runs first and so holds first claim on that generation.
    let mut submitted = false;
    let mut per_lane = Map::new();
    for lane in &lanes {
        let result = run_lane(args, &live, &binding, lane, &events, &state_path, submitted)?;
        if result["action"] == json!("submitted") || result["action"] == json!("would_submit") {
            submitted = true;
        }
        per_lane.insert(lane.label(), result);
    }
    let combined_written = per_lane
        .get("combined")
        .is_some_and(|result| result["action"] == json!("installed"));
    let outcome = json!({
        "schema": EVENT_SCHEMA,
        "at_ms": now_ms() as u64,
        "action": "per_task",
        "live_catalog_sha256": live.catalog_sha256,
        "combined_checkpoint_sha256": binding.checkpoint_sha256,
        "task_checkpoints": lanes
            .iter()
            .map(|lane| (lane.label(), json!(lane.checkpoint_sha256)))
            .collect::<Map<String, Value>>(),
        // Whether proposals/current.json was rebound this cycle. The combined
        // lane keeps running while forked, so this is no longer always false.
        "combined_pointer_written": combined_written,
        "lanes": Value::Object(per_lane),
    });
    append_event(&events, &outcome)?;
    Ok(outcome)
}

/// One lane's cycle. Byte-for-byte the historical behaviour when
/// `lane.task == None`; the per-task lanes differ only in job id, the `--task`
/// argument, the publish directory, and which pointer is rewritten.
pub(crate) fn run_lane(
    args: &Args,
    live: &LiveCatalog,
    binding: &ProposalBinding,
    lane: &Lane,
    events: &Path,
    state_path: &Path,
    submission_used: bool,
) -> Result<Value> {
    let root = args.root.as_path();
    let bound = lane.bound_catalog_sha256.clone();
    let mut outcome = json!({
        "schema": EVENT_SCHEMA,
        "at_ms": now_ms() as u64,
        "task": lane.task,
        "checkpoint_sha256": lane.checkpoint_sha256,
        "live_catalog_sha256": live.catalog_sha256,
        "bound_catalog_sha256": bound,
    });

    if bound.as_deref() == Some(live.catalog_sha256.as_str()) {
        outcome["action"] = json!("up_to_date");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }

    let prefix = lane.prefix(&live.catalog_sha256);
    // Any propose job for this lane's (task, checkpoint) blocks a new
    // submission, even one aimed at an older catalog: two multi-hour GPU
    // generations at once would starve the trainer and the second would
    // supersede the first anyway.
    let family = lane.family();
    let jobs = job_states(&args.scq, &family)?;
    outcome["jobs"] = json!(jobs
        .iter()
        .map(|(id, state)| json!({"id": id, "state": state}))
        .collect::<Vec<_>>());

    if let Some((id, state)) = jobs.iter().find(|(_, state)| is_active(state)) {
        outcome["action"] = json!("waiting");
        outcome["job_id"] = json!(id);
        outcome["job_state"] = json!(state);
        append_event(events, &outcome)?;
        return Ok(outcome);
    }

    // Prefer a generation aimed at the live catalog, but a completed run for an
    // intermediate catalog still beats what is bound now: catalogs only grow,
    // so any newer proposal set strictly widens coverage. Installing it keeps
    // progress monotone when generations (hours) are slower than tranche
    // activations (hourly).
    let candidates = order_install_candidates(&jobs, &prefix);
    if let Some(id) = candidates.first().copied() {
        if args.dry_run {
            outcome["action"] = json!("would_install");
            outcome["job_id"] = json!(id);
            append_event(events, &outcome)?;
            return Ok(outcome);
        }
        if let Some(installed) = install(
            &args.scq,
            root,
            id,
            &lane.checkpoint_sha256,
            bound.as_deref(),
            lane.task.as_deref(),
        )? {
            let pointer = match lane.task {
                Some(_) => rebind_task(root, lane, &installed)?,
                None => rebind(binding, &installed)?,
            };
            outcome["action"] = json!("installed");
            outcome["job_id"] = json!(id);
            outcome["installed_catalog_sha256"] = json!(installed.catalog_sha256);
            outcome["catalog_still_ahead"] = json!(installed.catalog_sha256 != live.catalog_sha256);
            outcome["datasets"] = json!(installed.datasets);
            outcome["counts"] = installed.counts.clone();
            outcome["validity"] = installed.validity.clone();
            outcome["proposals"] = json!(installed.proposals.to_string_lossy());
            outcome["proposals_sha256"] = json!(installed.proposals_sha256);
            append_event(events, &outcome)?;
            let mut state = read_state(state_path)?;
            state[format!("last_install:{}", lane.label())] = outcome.clone();
            if lane.task.is_none() {
                state["last_install"] = outcome.clone();
                state["pointer"] = pointer;
            } else {
                state[format!("pointer:{}", lane.label())] = pointer;
            }
            state["schema"] = json!(STATE_SCHEMA);
            atomic_json(state_path, &state)?;
            return Ok(outcome);
        }
    }

    // Nothing active and nothing installable: submit (or retry) a generation.
    let mut state = read_state(state_path)?;
    let terminal_failures = jobs
        .iter()
        .filter(|(id, state)| {
            id.starts_with(&prefix) && matches!(state.as_str(), "failed" | "cancelled")
        })
        .count() as u64;
    let attempt_key = match &lane.task {
        Some(task) => format!("{task}:{}", live.catalog_sha256),
        None => live.catalog_sha256.clone(),
    };
    let attempt = attempt_for(&state, &attempt_key).max(1 + terminal_failures);
    if attempt > MAX_ATTEMPTS {
        outcome["action"] = json!("blocked");
        outcome["reason"] = json!("attempt budget exhausted; operator review required");
        outcome["attempt"] = json!(attempt);
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    if submission_used {
        // Another branch already claimed this cycle's GPU generation.
        outcome["action"] = json!("deferred");
        outcome["reason"] = json!("one propose generation per cycle; another branch submitted");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    // The generator aborts at startup unless the context blob matches its
    // pointer. Checking here costs one hash and saves a multi-hour GPU
    // reservation that could only ever fail.
    let data_sha256 = sha256_file(&live.data)
        .with_context(|| format!("hash context blob {}", live.data.display()))?;
    if data_sha256 != live.data_sha256 {
        bail!(
            "live context blob {} hashes {data_sha256}, pointer declares {}",
            live.data.display(),
            live.data_sha256
        );
    }
    let spec = render_spec(
        root,
        &lane.checkpoint,
        &lane.checkpoint_sha256,
        live,
        attempt,
        lane.task.as_deref(),
    )?;
    outcome["spec"] = json!(spec.path.to_string_lossy());
    outcome["job_id"] = json!(spec.job_id);
    outcome["attempt"] = json!(attempt);
    if args.dry_run {
        outcome["action"] = json!("would_submit");
        append_event(events, &outcome)?;
        return Ok(outcome);
    }
    check_and_submit(&args.scq, &spec.path)?;
    outcome["action"] = json!("submitted");
    append_event(events, &outcome)?;
    record_attempt(&mut state, &attempt_key, attempt);
    state["schema"] = json!(STATE_SCHEMA);
    state[format!("last_submit:{}", lane.label())] = outcome.clone();
    if lane.task.is_none() {
        state["last_submit"] = outcome.clone();
    }
    atomic_json(state_path, &state)?;
    Ok(outcome)
}
