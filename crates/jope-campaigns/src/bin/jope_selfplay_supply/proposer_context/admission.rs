use super::*;

pub(crate) fn run_scq(args: &Args, arguments: &[&str]) -> Result<String> {
    let output = Command::new(&args.scq)
        .args(arguments)
        .output()
        .with_context(|| format!("spawn {} {:?}", args.scq.display(), arguments))?;
    if !output.status.success() {
        bail!(
            "scq {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Is this scq rejection "come back later", or a real configuration error?
///
/// scq probes dependencies only on nodes that ALREADY have the requested
/// GPU/cores/RAM free, so a perfectly healthy cluster rejects a submit whenever
/// every capable node happens to be busy. That must queue, not `bail!` -- a bail
/// fails the supply cycle and opens the circuit breaker.
///
/// This used to name ONE probe (`smoke proposal GPU capacity`) literally. That
/// made it fragile in exactly the way `spec_policy::PROBE_FLYWHEEL_ROOT_XBABE2_ONLY`
/// documents: when the proposer's probe changed, a busy-node rejection stopped
/// matching and produced **44 consecutive `cycle_failed` records**. Now that the
/// proposer probes `jope-prime policy source present` instead, a busy-node
/// rejection reads `... smoke jope-prime policy source present: smoke test exited
/// with exit 1`, which the old literal would also have missed.
///
/// So match the SHAPE of a probe failure rather than any probe's name. scq renders
/// one as `<node>: smoke <probe name>: smoke test exited with exit <n>`, so the
/// discriminator is the delimiter **`": smoke test exited"`** -- a named probe
/// that ran and failed. Bare prose that merely happens to contain the words is
/// not a probe result and stays fatal, which is what
/// `proposer_capacity_pressure_is_a_normal_backoff` has always asserted.
///
/// A genuinely mis-provisioned node (e.g. xbabe0, which has a free GPU but no
/// runtime bundle) lands here too -- correctly, because the right response is to
/// keep queueing for a node that can run the job, not to fail the supply loop.
pub(crate) fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            || (message.contains("smoke ") && message.contains(": smoke test exited")))
}

pub(crate) fn register_proposer_admission(args: &Args, spec: &Path) -> Result<()> {
    let job_id = fs::read_to_string(spec)?
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("id = ")
                .and_then(|value| serde_json::from_str::<String>(value).ok())
        })
        .context("cohort proposer spec lacks job identity")?;
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let controller = executable
        .parent()
        .context("supply executable has no bin directory")?
        .join("jope_gpu_lane_admission");
    let output = Command::new(&controller)
        .args([
            "--root",
            args.root.to_string_lossy().as_ref(),
            "--register-role",
            "cohort-proposer",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            &job_id,
        ])
        .output()
        .with_context(|| {
            format!(
                "register proposer admission through {}",
                controller.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "cohort proposer GPU admission request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

/// Capacity-only failures are handed to the crash-safe GPU admission owner.
/// The supply state remains at the submit stage until that owner has placed
/// the immutable spec.
pub(crate) fn scq_check_submit(args: &Args, spec: &Path) -> Result<bool> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} check {}", args.scq.display(), spec.display()))?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && temporarily_lacks_launch_capacity(&check_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    if !check.status.success() {
        bail!(
            "SCQ check rejected cohort proposer: {}",
            check_message.trim()
        );
    }

    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()
        .with_context(|| format!("spawn {} submit {}", args.scq.display(), spec.display()))?;
    let submit_message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if submit.status.success()
        || submit_message
            .to_ascii_lowercase()
            .contains("already exists")
    {
        return Ok(true);
    }
    if temporarily_lacks_launch_capacity(&submit_message) {
        register_proposer_admission(args, spec)?;
        return Ok(false);
    }
    bail!(
        "SCQ submit rejected cohort proposer: {}",
        submit_message.trim()
    )
}

pub(crate) fn proposer_job_state(args: &Args, job_id: &str) -> Result<Option<String>> {
    let value: Value = serde_json::from_str(&run_scq(args, &["--json", "ls"])?)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find_map(|job| {
                (job.get("id").and_then(Value::as_str) == Some(job_id)).then(|| {
                    job.get("state")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_owned()
                })
            })
        }))
}

pub(crate) fn recursive_find(directory: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, found);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            found.push(path);
        }
    }
}

pub(crate) fn fetch_cohort_proposals(
    args: &Args,
    state: &PersistedSupplyState,
) -> Result<(PathBuf, String)> {
    let job_id = state
        .proposal_job_id
        .as_deref()
        .context("proposal job identity is absent")?;
    let incoming = args.root.join("proposals/_incoming").join(job_id);
    if incoming.exists() {
        fs::remove_dir_all(&incoming)?;
    }
    fs::create_dir_all(&incoming)?;
    run_scq(
        args,
        &["fetch", job_id, "--output", &incoming.to_string_lossy()],
    )?;
    let mut proposals = Vec::new();
    let mut summaries = Vec::new();
    recursive_find(&incoming, "dl-proposals.jsonl", &mut proposals);
    recursive_find(&incoming, "dl-proposals.summary.json", &mut summaries);
    let proposal_source = proposals
        .into_iter()
        .next()
        .context("cohort proposer output lacks proposals")?;
    let summary_source = summaries
        .into_iter()
        .next()
        .context("cohort proposer output lacks summary")?;
    let summary: Value = serde_json::from_slice(&fs::read(&summary_source)?)?;
    let proposal_sha256 = sha256_file(&proposal_source)?;
    if summary
        .get("model_checkpoint_sha256")
        .and_then(Value::as_str)
        != Some(state.checkpoint_sha256.as_str())
        || summary.get("catalog_sha256").and_then(Value::as_str)
            != Some(state.catalog_sha256.as_str())
        || summary.get("cohort_sha256").and_then(Value::as_str)
            != Some(state.cohort_sha256.as_str())
        || summary.get("output_sha256").and_then(Value::as_str) != Some(proposal_sha256.as_str())
        // Any legal micro-wave shape (tasks.len() * 64). The cohort identity is
        // already pinned by cohort_sha256 above, so this is a sanity check on
        // shape, not the provenance authority. NOTE for activation: a 192-wave
        // at max_candidates_per_manifest=25 emits 2,112 manifests, which breaks
        // `queued_target + manifests_per_wave <= queued_hard_cap` -- raising the
        // manifest cap (validated 25..=100) is part of flipping activation on.
        || !matches!(
            summary.get("datasets").and_then(Value::as_u64),
            Some(128 | 192 | 256)
        )
        || summary
            .get("zero_cross_context_backfill")
            .and_then(Value::as_bool)
            != Some(true)
    {
        bail!("cohort proposer output provenance drift");
    }
    let validity = summary
        .get("proposal_validity")
        .and_then(Value::as_f64)
        .context("cohort proposer summary lacks validity")?;
    if validity < 0.99 {
        bail!("cohort proposal validity {validity:.6} is below 0.99");
    }
    let cohort_value: Value = serde_json::from_slice(&fs::read(&state.cohort)?)?;
    let mut task_counts = BTreeMap::<String, usize>::new();
    for entry in cohort_value
        .get("entries")
        .and_then(Value::as_array)
        .context("frozen cohort lacks entries")?
    {
        let task = entry
            .get("task")
            .and_then(Value::as_str)
            .context("frozen cohort entry lacks task")?;
        *task_counts.entry(task.to_owned()).or_default() += 1;
    }
    if task_counts.values().any(|count| *count != 64) {
        bail!("cohort proposal task scope is not 64 contexts per task");
    }
    let destination = args
        .root
        .join("proposals/by-cohort")
        .join(&state.cohort_sha256);
    fs::create_dir_all(&destination)?;
    let proposals_path = destination.join(format!("{proposal_sha256}.jsonl"));
    if !proposals_path.is_file() {
        fs::copy(&proposal_source, &proposals_path)?;
    }
    if sha256_file(&proposals_path)? != proposal_sha256 {
        bail!("published cohort proposal content drift");
    }
    let summary_sha256 = sha256_file(&summary_source)?;
    let summary_path = destination.join(format!("{summary_sha256}.summary.json"));
    if !summary_path.is_file() {
        fs::copy(&summary_source, &summary_path)?;
    }
    // Every task the cohort carries gets a DL allocation at the historical
    // 8-slot rung. The literal {"binary": 8, "regression": 8} that used to sit
    // in the pointer stranded wave 552 for hours: the wave builder asks
    // `normalized_dl_slots` for each DECLARED task, and a three-task cohort
    // met a two-key map -- "proposal pointer lacks multiclass DL allocation",
    // retried forever against the same immutable pointer.
    let dl_slots_by_task = task_counts
        .keys()
        .map(|task| (task.clone(), json!(8)))
        .collect::<serde_json::Map<_, _>>();
    let task_scopes = task_counts
        .into_iter()
        .map(|(task, context_count)| {
            json!({
                "task": task,
                "accepted_checkpoint_sha256": state.checkpoint_sha256,
                "context_count": context_count,
            })
        })
        .collect::<Vec<_>>();
    let pointer = json!({
        "schema": "CohortProposalPointer.v1",
        "immutable": true,
        "task_scopes": task_scopes,
        "cohort": state.cohort,
        "cohort_sha256": state.cohort_sha256,
        "catalog_sha256": state.catalog_sha256,
        "checkpoint_sha256": state.checkpoint_sha256,
        "decoder_configuration_sha256": state.decoder_configuration_sha256,
        "seed": state.seed,
        "proposal_job_id": job_id,
        "proposals": proposals_path,
        "proposals_sha256": proposal_sha256,
        "summary": summary_path,
        "summary_sha256": summary_sha256,
        "dl_slots_by_task": dl_slots_by_task,
        "cross_context_backfill_rows": 0,
    });
    let mut bytes = serde_json::to_vec(&pointer)?;
    bytes.push(b'\n');
    let pointer_sha256 = sha256_bytes(&bytes);
    let pointer_path = destination.join(format!("{pointer_sha256}.pointer.json"));
    if pointer_path.is_file() {
        if fs::read(&pointer_path)? != bytes {
            bail!("immutable cohort proposal pointer drift");
        }
    } else {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pointer_path)?;
        file.write_all(&bytes)?;
        file.sync_data()?;
    }
    Ok((pointer_path, pointer_sha256))
}
