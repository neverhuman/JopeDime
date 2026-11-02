use super::*;

pub(super) fn cycle(args: &Args) -> Result<JopeDlgpBudgetCampaignStatusV1> {
    let now = now_ms();
    let protocol: Value = read_json(&args.protocol)?;
    ensure!(
        protocol.get("schema").and_then(Value::as_str) == Some("JopeDlgpBudgetFirstProtocol.v1"),
        "budget-first protocol schema drift"
    );
    let protocol_sha256 = sha256_file(&args.protocol)?;
    let campaign_root = args.campaign_root();
    let plans = load_plans(&campaign_root, &protocol_sha256)?;
    let progress = load_progress(&campaign_root, &plans, &protocol_sha256)?;
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?);

    let mut blockers = Vec::new();
    let stage0_path = campaign_root.join("gates/stage0-reproduction.v1.json");
    let stage0 = if stage0_path.is_file() {
        Some(serde_json::from_value::<Stage0Receipt>(read_json(
            &stage0_path,
        )?)?)
    } else {
        None
    };
    let reproduction_passed = stage0.as_ref().is_some_and(|receipt| {
        receipt.schema == "JopeDlgpBudgetStage0Reproduction.v1"
            && receipt.protocol_sha256 == protocol_sha256
            && receipt.round2_payload_sha256
                == protocol
                    .pointer("/round2_evidence/payload_sha256")
                    .and_then(Value::as_str)
                    .unwrap_or("")
            && receipt.control_prefix_count == 6
            && !receipt.test_csv_read
            && receipt.reproduced
            && receipt.generated_at_ms > 0
    });
    if !reproduction_passed {
        blockers.push("Stage 0 reproduction and six 1k control prefixes are required".to_owned());
    }
    let gate_path = campaign_root.join("gates/launch-gates.v1.json");
    let gates = if gate_path.is_file() {
        Some(serde_json::from_value::<LaunchGates>(read_json(
            &gate_path,
        )?)?)
    } else {
        None
    };
    if reproduction_passed {
        if let Some(gates) = &gates {
            blockers.extend(gates.blockers(&protocol_sha256, now));
        } else {
            blockers.push("launch gate receipt is absent".to_owned());
        }
    }

    let mut plan_attempts = BTreeMap::new();
    let mut known_jobs = BTreeSet::new();
    for plan in plans.values() {
        let payload_sha256 = plan.scientific_payload()?.identity_sha256()?;
        let attempts =
            infrastructure_attempts(&args.root, DimeRoleV1::BudgetCampaign, &payload_sha256)?;
        known_jobs.insert(plan.job_identity.clone());
        for attempt in &attempts {
            known_jobs.insert(attempt.job_identity.clone());
            if let Some(next) = &attempt.next_job_identity {
                known_jobs.insert(next.clone());
            }
        }
        plan_attempts.insert(plan.trajectory_identity.clone(), attempts);
    }
    let foreign = jobs
        .iter()
        .filter(|job| {
            (job.project.starts_with(PROJECT) || job.project.contains("strategy-screen"))
                && !known_jobs.contains(&job.id)
        })
        .collect::<Vec<_>>();
    if !foreign.is_empty() {
        let quarantine = json!({
            "schema": "JopeDlgpForeignJobQuarantine.v1",
            "generated_at_ms": now,
            "protocol_sha256": protocol_sha256,
            "jobs": foreign.iter().map(|job| json!({
                "id": job.id,
                "project": job.project,
                "state": job.state,
                "disposition": "quarantined_not_adopted_not_cancelled",
            })).collect::<Vec<_>>(),
        });
        atomic_json(
            &campaign_root.join("quarantine/foreign-jobs-current.v1.json"),
            &quarantine,
        )?;
        blockers.push(format!(
            "{} strategy-screen jobs are foreign to the frozen campaign",
            foreign.len()
        ));
    }

    let evaluations_used = progress
        .values()
        .map(|value| value.completed_evaluations)
        .sum::<u64>();
    ensure!(
        evaluations_used <= STAGE1_MAX_SEARCH_EVALUATIONS,
        "partial and completed receipts exceed the 200,000-evaluation ceiling"
    );
    let mut rung_counts = BTreeMap::from([
        ("1000".to_owned(), 0),
        ("2000".to_owned(), 0),
        ("5000".to_owned(), 0),
        ("10000".to_owned(), 0),
    ]);
    let mut promotion_decisions = BTreeSet::new();
    for value in progress.values() {
        for rung in &value.completed_rungs {
            *rung_counts.entry(rung.to_string()).or_default() += 1;
        }
        promotion_decisions.extend(value.promotion_decisions.iter().cloned());
    }
    let launch_gates_clear = blockers.is_empty();
    let mut drain_suppressed = false;
    let mut active_jobs = Vec::new();
    let mut launch_candidates = Vec::new();
    let mut terminal_candidates = Vec::new();
    for plan in plans.values() {
        let completed = progress
            .get(&plan.trajectory_identity)
            .map_or(0, |value| value.completed_evaluations);
        if completed >= 10_000 {
            continue;
        }
        let payload = plan.scientific_payload()?;
        let payload_sha256 = payload.identity_sha256()?;
        let attempts = plan_attempts
            .get(&plan.trajectory_identity)
            .context("campaign attempt index missing")?;
        let (attempt_number, job_identity, retry_not_before_ms) =
            if let Some(last) = attempts.last() {
                let Some(next) = last.next_job_identity.clone() else {
                    continue;
                };
                (last.attempt_number + 1, next, last.retry_not_before_ms)
            } else {
                (1, plan.job_identity.clone(), 0)
            };
        let launch =
            read_launch_receipt(&campaign_root, &plan.trajectory_identity, attempt_number)?;
        let scheduler = jobs.iter().find(|job| job.id == job_identity);
        if let Some(job) = scheduler.filter(|job| {
            matches!(
                job.state.as_str(),
                "queued" | "starting" | "dispatching" | "running" | "held"
            )
        }) {
            active_jobs.push(DlgpActiveJobV1 {
                job_identity: job_identity.clone(),
                trajectory_identity: plan.trajectory_identity.clone(),
                attempt_number,
                state: job.state.clone(),
                progress_evaluations: completed,
                reason: "frozen trajectory; infrastructure identity is mutable only".to_owned(),
            });
            continue;
        }
        if let Some(job) = scheduler {
            if let Some(class) = terminal_class(&job.state) {
                if job.state == "cancelled"
                    && matching_drain_intent(&args.root, DimeRoleV1::BudgetCampaign, &payload, now)?
                        .is_some()
                {
                    drain_suppressed = true;
                    blockers.push(format!(
                        "trajectory {} intentionally drained by an exact DIME receipt",
                        plan.trajectory_identity
                    ));
                    continue;
                }
                terminal_candidates.push((
                    plan.trajectory_identity.clone(),
                    attempt_number,
                    job_identity,
                    job.state.clone(),
                    class,
                    job.state == "cancelled",
                    sha256_bytes(&serde_json::to_vec(job)?),
                    payload,
                ));
                continue;
            }
        }
        if let Some(launch) = launch {
            if now.saturating_sub(launch.submitted_at_ms) >= 60_000 {
                terminal_candidates.push((
                    plan.trajectory_identity.clone(),
                    attempt_number,
                    job_identity,
                    "vanished".to_owned(),
                    DimeTerminalClassV1::InfrastructureFailure,
                    false,
                    sha256_bytes(&serde_json::to_vec(&launch)?),
                    payload,
                ));
            } else {
                active_jobs.push(DlgpActiveJobV1 {
                    job_identity,
                    trajectory_identity: plan.trajectory_identity.clone(),
                    attempt_number,
                    state: "launching".to_owned(),
                    progress_evaluations: completed,
                    reason: "submitted; awaiting scheduler visibility".to_owned(),
                });
            }
            continue;
        }
        if now < retry_not_before_ms {
            active_jobs.push(DlgpActiveJobV1 {
                job_identity,
                trajectory_identity: plan.trajectory_identity.clone(),
                attempt_number,
                state: "backoff".to_owned(),
                progress_evaluations: completed,
                reason: format!("retry not before {retry_not_before_ms}"),
            });
        } else {
            launch_candidates.push((
                plan.trajectory_identity.clone(),
                attempt_number,
                job_identity,
                payload_sha256,
            ));
        }
    }

    if args.submit && launch_gates_clear {
        if let Some((
            trajectory,
            attempt_number,
            job_identity,
            state,
            class,
            foreign,
            evidence,
            payload,
        )) = terminal_candidates.into_iter().next()
        {
            let attempts = plan_attempts
                .get(&trajectory)
                .context("terminal campaign attempt index missing")?;
            let next_job_identity = plans
                .get(&trajectory)
                .map(|plan| next_attempt_identity(plan, attempt_number + 1))
                .context("terminal campaign plan missing")?;
            let receipt = DimeInfrastructureAttemptV1::new_terminal(
                DimeRoleV1::BudgetCampaign,
                payload,
                attempt_number,
                job_identity,
                Some(next_job_identity),
                attempts.last().map(|value| value.content_sha256.clone()),
                now,
                now.saturating_add(retry_backoff_ms(attempt_number)),
                DimeTerminalEvidenceV1 {
                    terminal_state: state.clone(),
                    class,
                    evidence_sha256: evidence,
                    observed_at_ms: now,
                    foreign_cancellation: foreign,
                },
            )?;
            append_infrastructure_attempt(&args.root, &receipt)?;
            if foreign {
                blockers.push(format!(
                    "foreign cancellation recovered for {trajectory}; scientific payload unchanged"
                ));
            }
        } else if let Some((trajectory, attempt_number, job_identity, payload_sha256)) =
            launch_candidates.into_iter().next()
        {
            ensure!(
                evaluations_used < STAGE1_MAX_SEARCH_EVALUATIONS,
                "evaluation ceiling leaves no launch budget"
            );
            let plan = plans
                .get(&trajectory)
                .context("campaign launch plan missing")?;
            submit_attempt(
                args,
                &campaign_root,
                plan,
                &payload_sha256,
                attempt_number,
                &job_identity,
            )?;
        }
    }
    let prior_foreign_cancellations = plan_attempts
        .values()
        .flatten()
        .filter(|attempt| attempt.terminal.foreign_cancellation)
        .count();
    if prior_foreign_cancellations >= 2 {
        blockers.push(format!(
            "warning: {prior_foreign_cancellations} repeated foreign campaign cancellations; recovery preserves scientific payloads"
        ));
    }

    let wins_path = campaign_root.join("receipts/robust-wins.v1.json");
    let (robust_wins, seed_agreement) = match fs::symlink_metadata(&wins_path) {
        Ok(_) => {
            let wins = read_json(&wins_path)?;
            let robust_wins = wins
                .get("robust_wins")
                .and_then(Value::as_u64)
                .context("robust wins receipt is missing its count")?;
            let seed_agreement = wins
                .get("seed_agreement")
                .and_then(Value::as_str)
                .context("robust wins receipt is missing seed agreement")?
                .to_owned();
            (robust_wins, seed_agreement)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            blockers.push("robust wins receipt unavailable; count remains unknown".to_owned());
            (0, "unavailable".to_owned())
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect robust wins receipt {}", wins_path.display()));
        }
    };
    let last_advanced_at_ms = progress
        .values()
        .map(|value| value.updated_at_ms)
        .max()
        .or_else(|| stage0.as_ref().map(|receipt| receipt.generated_at_ms))
        .unwrap_or(0);
    let complete = plans.len() == 20
        && progress.len() == plans.len()
        && progress
            .values()
            .all(|value| value.completed_evaluations >= 5_000);
    let gate_state = if complete {
        DlgpCampaignGateStateV1::Complete
    } else if !reproduction_passed {
        DlgpCampaignGateStateV1::ReproductionRequired
    } else if launch_gates_clear && !drain_suppressed {
        DlgpCampaignGateStateV1::Runnable
    } else {
        DlgpCampaignGateStateV1::LaunchGatesBlocked
    };
    let track_state = |track: &str| {
        if plans.values().any(|plan| plan.license_track == track) {
            "active"
        } else {
            "not_started"
        }
    };
    let status = JopeDlgpBudgetCampaignStatusV1 {
        schema: JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA.to_owned(),
        generated_at_ms: now,
        last_advanced_at_ms,
        protocol_sha256,
        stage: if complete {
            "stage1_complete"
        } else if reproduction_passed {
            "stage1"
        } else {
            "stage0_reproduction"
        }
        .to_owned(),
        gate_state,
        trajectory_count: plans.len() as u64,
        completed_trajectory_count: progress
            .values()
            .filter(|value| value.completed_evaluations >= 5_000)
            .count() as u64,
        rung_counts,
        evaluations_used,
        evaluations_cap: STAGE1_MAX_SEARCH_EVALUATIONS,
        active_jobs,
        robust_wins,
        robust_wins_target: 7,
        robust_wins_denominator: 22,
        seed_agreement,
        license_track: "separate".to_owned(),
        production_compatible_state: track_state("production_compatible").to_owned(),
        research_only_state: track_state("research_only").to_owned(),
        test_opening_state: "sealed_until_selection_freeze".to_owned(),
        promotion_decisions: promotion_decisions.into_iter().collect(),
        blockers,
    };
    status.validate()?;
    Ok(status)
}
