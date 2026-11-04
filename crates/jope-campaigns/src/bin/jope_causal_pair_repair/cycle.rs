use super::*;

pub(super) fn cycle(args: &Args) -> Result<Value> {
    let queue = load_queue(&args.root)?;
    if args.verify_only {
        let evaluators = bundle_evaluators(&args.root)?;
        let selected = allocate_requests(
            &queue
                .repairs
                .iter()
                .filter(|request| !request.permanent_failure)
                .cloned()
                .collect::<Vec<_>>(),
            args.submissions_per_cycle,
            task_priority(&args.root),
        );
        let mut verified = Vec::new();
        let mut failures = Vec::new();
        for request in selected {
            let result = (|| -> Result<Value> {
                let identity = pair_identity(&args.root, &request)?;
                let target = target_candidate(&args.root, &request, &identity)?;
                let evaluator = evaluator_binding(&args.root, &identity, &target, &evaluators)?;
                Ok(json!({
                    "pair_id": request.pair_id,
                    "lane": request.missing_lane,
                    "task": request.task,
                    "target_manifest_sha256": target.manifest_sha256,
                    "candidate_sha256": target.candidate_sha256,
                    "evaluator_sha256": evaluator.sha256,
                }))
            })();
            match result {
                Ok(row) => verified.push(row),
                Err(error) => failures.push(json!({
                    "pair_id": request.pair_id,
                    "lane": request.missing_lane,
                    "task": request.task,
                    "error": format!("{error:#}"),
                })),
            }
        }
        let audit = json!({
            "schema": "CausalPairRepairExecutor.v1",
            "generated_at_ms": now_ms(),
            "verify_only": true,
            "queue_depth": queue.repairs.len(),
            "verified": verified,
            "failures": failures,
            "permanent_failures_are_imputed": false,
        });
        atomic_json(
            &args
                .root
                .join("ops/causal-pair-repair-verification.v1.json"),
            &audit,
        )?;
        return Ok(audit);
    }
    let mut state = load_state(&args.root)?;
    for request in &queue.repairs {
        let key = entry_key(&request.pair_id, request.missing_lane);
        let entry = state
            .entries
            .entry(key)
            .or_insert_with(|| RepairEntry::new(request));
        if request.permanent_failure && entry.status != "completed" {
            entry.status = "permanent_failure".to_owned();
            entry.last_error = Some("ledger marked the pair member irreparable".to_owned());
            entry.updated_at_ms = now_ms();
        }
    }

    let jobs = scq_jobs(args)?;
    complete_jobs(args, &mut state, &jobs)?;
    let inflight = state
        .entries
        .values()
        .filter(|entry| entry.status == "submitted")
        .count();
    let capacity = args.max_inflight.saturating_sub(inflight);
    let limit = capacity.min(args.submissions_per_cycle);
    let active_queue_keys = queue
        .repairs
        .iter()
        .map(|request| entry_key(&request.pair_id, request.missing_lane))
        .collect::<BTreeSet<_>>();
    let eligible = queue
        .repairs
        .iter()
        .filter(|request| {
            let key = entry_key(&request.pair_id, request.missing_lane);
            state.entries.get(&key).is_some_and(|entry| {
                matches!(entry.status.as_str(), "pending" | "blocked")
                    && entry.attempts < args.max_attempts
                    && entry.retry_after_ms <= now_ms()
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected = allocate_requests(&eligible, limit, task_priority(&args.root));
    let runner = runtime_runner(args)?;
    let evaluators = bundle_evaluators(&args.root)?;
    for request in selected {
        let key = entry_key(&request.pair_id, request.missing_lane);
        let entry = state.entries.get_mut(&key).expect("queue entry exists");
        if let Err(error) = prepare_and_submit(args, &request, entry, &runner, &evaluators, &jobs) {
            note_control_error(entry, error);
        }
    }
    save_state(&args.root, &mut state)?;

    let counts =
        state
            .entries
            .values()
            .fold(BTreeMap::<String, usize>::new(), |mut counts, entry| {
                *counts.entry(entry.status.clone()).or_default() += 1;
                counts
            });
    let critical = state.entries.values().filter(|entry| {
        active_queue_keys.contains(&entry_key(&entry.pair_id, entry.missing_lane))
            && entry.identical_failures >= 3
            && entry.status != "permanent_failure"
    });
    let critical = critical
        .map(|entry| {
            json!({
                "pair_id": entry.pair_id,
                "lane": entry.missing_lane,
                "error": entry.last_error,
            })
        })
        .collect::<Vec<_>>();
    let audit = json!({
        "schema": "CausalPairRepairExecutor.v1",
        "generated_at_ms": now_ms(),
        "queue_depth": queue.repairs.len(),
        "state_counts": counts,
        "inflight": state.entries.values().filter(|entry| entry.status == "submitted").count(),
        "task_priority": task_priority(&args.root),
        "critical_identical_failures": critical,
        "permanent_failures_are_imputed": false,
    });
    atomic_json(
        &args.root.join("ops/causal-pair-repair-executor.v1.json"),
        &audit,
    )?;
    Ok(audit)
}
