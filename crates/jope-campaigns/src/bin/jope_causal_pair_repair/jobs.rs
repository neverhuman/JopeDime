use super::*;

pub(super) fn repair_succeeded(
    root: &Path,
    entry: &RepairEntry,
    receipt_sha256: &str,
) -> Result<bool> {
    let connection = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let candidate = entry
        .candidate_sha256
        .as_ref()
        .context("repair state lacks candidate identity")?;
    let success: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM causal_evaluation_receipts_v2
            WHERE pair_id=?1 AND lane=?2 AND candidate_sha256=?3
              AND receipt_sha256=?4 AND success=1 AND proposal_valid=1
         )",
        (
            entry.pair_id.as_str(),
            lane_name(entry.missing_lane),
            candidate.as_str(),
            receipt_sha256,
        ),
        |row| row.get(0),
    )?;
    Ok(success)
}

pub(super) fn ingest_completed(args: &Args, entry: &RepairEntry) -> Result<bool> {
    let job_id = entry
        .job_id
        .as_deref()
        .context("repair state lacks job id")?;
    let manifest = entry
        .repair_manifest_path
        .as_ref()
        .context("repair state lacks manifest path")?;
    let output = args
        .root
        .join("ops/causal-pair-repairs/receipts")
        .join(job_id);
    let mut markers = Vec::new();
    recursive_find(&output, "SHARD_OK", &mut markers);
    if markers.is_empty() {
        fetch(args, job_id, &output)?;
        recursive_find(&output, "SHARD_OK", &mut markers);
    }
    ensure!(markers.len() == 1, "repair fetch has no unique SHARD_OK");
    let receipt_root = markers[0]
        .parent()
        .context("repair SHARD_OK has no parent")?;
    let dataset = read_json(manifest)?
        .get("dataset_hash")
        .and_then(Value::as_str)
        .context("repair manifest lacks dataset")?
        .to_owned();
    let receipt_path = receipt_root.join(format!("{dataset}.receipts.jsonl"));
    let receipt_sha256 = sha256_file(&receipt_path)?;
    ingest_receipt_with_manifest(
        &args.root.join("corpus/strict-test-exact.sqlite"),
        job_id,
        receipt_root,
        manifest,
    )?;
    repair_succeeded(&args.root, entry, &receipt_sha256)
}

pub(super) fn record_failure(entry: &mut RepairEntry, message: String, max_attempts: usize) {
    entry.identical_failures = if entry.last_error.as_deref() == Some(message.as_str()) {
        entry.identical_failures.saturating_add(1)
    } else {
        1
    };
    entry.last_error = Some(message);
    entry.job_id = None;
    entry.spec_path = None;
    entry.status = if entry.attempts >= max_attempts {
        "permanent_failure"
    } else {
        "pending"
    }
    .to_owned();
    let exponent = entry.identical_failures.saturating_sub(1).min(7) as u32;
    let delay_seconds = 2_u128.pow(exponent).min(120);
    entry.retry_after_ms = now_ms().saturating_add(delay_seconds * 1_000);
    entry.updated_at_ms = now_ms();
}

pub(super) fn note_control_error(entry: &mut RepairEntry, error: anyhow::Error) {
    let message = format!("{error:#}");
    entry.identical_failures = if entry.last_error.as_deref() == Some(message.as_str()) {
        entry.identical_failures.saturating_add(1)
    } else {
        1
    };
    entry.last_error = Some(message);
    entry.status = "blocked".to_owned();
    let exponent = entry.identical_failures.saturating_sub(1).min(7) as u32;
    let delay_seconds = 2_u128.pow(exponent).min(120);
    entry.retry_after_ms = now_ms().saturating_add(delay_seconds * 1_000);
    entry.updated_at_ms = now_ms();
}

pub(super) fn complete_jobs(
    args: &Args,
    state: &mut RepairState,
    jobs: &BTreeMap<String, String>,
) -> Result<()> {
    let keys = state
        .entries
        .iter()
        .filter(|(_, entry)| entry.status == "submitted")
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    for key in keys {
        let entry = state.entries.get_mut(&key).expect("entry key exists");
        let job_id = entry
            .job_id
            .clone()
            .context("submitted repair lacks job id")?;
        match jobs.get(&job_id).map(String::as_str) {
            Some("completed") => match ingest_completed(args, entry) {
                Ok(true) => {
                    entry.status = "completed".to_owned();
                    entry.last_error = None;
                    entry.identical_failures = 0;
                    entry.retry_after_ms = 0;
                    entry.updated_at_ms = now_ms();
                }
                Ok(false) => record_failure(
                    entry,
                    "evaluator returned an unsuccessful exact member".to_owned(),
                    args.max_attempts,
                ),
                Err(error) => note_control_error(entry, error),
            },
            Some(state) if ACTIVE_STATES.contains(&state) => {}
            Some(state @ ("failed" | "cancelled")) => record_failure(
                entry,
                format!("SCQ terminal state {state}"),
                args.max_attempts,
            ),
            Some(state_name) => note_control_error(
                entry,
                anyhow::anyhow!("SCQ returned unknown repair state {state_name}"),
            ),
            None => {
                let spec = entry
                    .spec_path
                    .as_ref()
                    .context("submitted repair lacks spec")?;
                if let Err(error) = submit_spec(args, spec) {
                    note_control_error(entry, error);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn task_priority(root: &Path) -> Task {
    let ledger =
        read_json(&root.join("training/causal-evidence-ledger.v2.json")).unwrap_or(Value::Null);
    let score = |task: Task| {
        let row = &ledger["tasks"][task.as_str()];
        let pairs = row
            .get("complete_pairs")
            .and_then(Value::as_u64)
            .unwrap_or_default() as f64;
        let validity = row
            .get("proposal_validity")
            .and_then(Value::as_f64)
            .unwrap_or_default();
        (pairs / 2_048.0).min(1.0) + validity
    };
    if score(Task::Regression) > score(Task::Binary) {
        Task::Regression
    } else {
        Task::Binary
    }
}

pub(super) fn allocate_requests(
    requests: &[PairRepairRequestV1],
    limit: usize,
    priority: Task,
) -> Vec<PairRepairRequestV1> {
    let mut by_task = BTreeMap::<Task, Vec<PairRepairRequestV1>>::new();
    for request in requests {
        by_task
            .entry(request.task)
            .or_default()
            .push(request.clone());
    }
    for rows in by_task.values_mut() {
        rows.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    }
    let priority_quota = (limit * 3).div_ceil(5);
    // The repair lane still splits its budget across the two legacy tasks;
    // `task_priority` can only return Regression or Binary until new-task
    // repair activates with its own allocation policy.
    let other = if priority == Task::Binary {
        Task::Regression
    } else {
        Task::Binary
    };
    let mut output = Vec::new();
    for (task, quota) in [(priority, priority_quota), (other, limit - priority_quota)] {
        output.extend(
            by_task
                .get_mut(&task)
                .into_iter()
                .flat_map(|rows| rows.drain(..rows.len().min(quota))),
        );
    }
    if output.len() < limit {
        for task in [priority, other] {
            let remaining = limit - output.len();
            output.extend(
                by_task
                    .get_mut(&task)
                    .into_iter()
                    .flat_map(|rows| rows.drain(..rows.len().min(remaining))),
            );
        }
    }
    output
}

pub(super) fn prepare_and_submit(
    args: &Args,
    request: &PairRepairRequestV1,
    entry: &mut RepairEntry,
    runner: &Path,
    evaluators: &BTreeMap<String, PathBuf>,
    jobs: &BTreeMap<String, String>,
) -> Result<()> {
    let identity = pair_identity(&args.root, request)?;
    let target = target_candidate(&args.root, request, &identity)?;
    let evaluator = evaluator_binding(&args.root, &identity, &target, evaluators)?;
    let manifest = repair_manifest(&args.root, request, &target, &evaluator)?;
    let attempt = entry.attempts.saturating_add(1);
    let (job_id, spec) = render_spec(
        args, runner, request, &target, &evaluator, &manifest, attempt,
    )?;
    if !jobs.contains_key(&job_id) {
        submit_spec(args, &spec)?;
    }
    entry.status = "submitted".to_owned();
    entry.attempts = attempt;
    entry.identical_failures = 0;
    entry.last_error = None;
    entry.retry_after_ms = 0;
    entry.job_id = Some(job_id);
    entry.spec_path = Some(spec);
    entry.repair_manifest_path = Some(manifest);
    entry.original_manifest_path = Some(target.manifest_path);
    entry.original_manifest_sha256 = Some(Sha256Digest::parse(target.manifest_sha256.clone())?);
    entry.original_evaluator_sha256 = Some(Sha256Digest::parse(evaluator.sha256)?);
    entry.candidate_sha256 = Some(Sha256Digest::parse(target.candidate_sha256)?);
    entry.updated_at_ms = now_ms();
    Ok(())
}
