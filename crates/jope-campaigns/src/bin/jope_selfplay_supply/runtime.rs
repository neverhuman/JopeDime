use super::*;

pub(super) fn load_supply_failure_streak(path: &Path) -> Result<(Option<String>, u32)> {
    if !path.is_file() {
        return Ok((None, 0));
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    if value.get("schema").and_then(Value::as_str) != Some("SupplyCircuitBreaker.v1")
        || value.get("open").and_then(Value::as_bool) != Some(true)
    {
        return Ok((None, 0));
    }
    let error = value
        .get("error")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let count = value
        .get("identical_failures")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        .min(u32::MAX as u64) as u32;
    Ok((error, count))
}

pub(super) fn validate_args(args: &Args) -> Result<()> {
    if args.queue_floor == 0
        || args.queue_floor > args.queued_target
        || args.queued_target > args.queued_hard_cap
    {
        bail!("queue thresholds must satisfy 0 < floor <= target <= hard cap");
    }
    if args.ready_reserve_min == 0 || args.ready_reserve_min > args.queued_target {
        bail!("ready reserve must satisfy 0 < reserve <= scheduler target");
    }
    if args.useful_work_min < MIN_USEFUL_WORK {
        bail!("useful work minimum must be at least {MIN_USEFUL_WORK}");
    }
    if args.candidate_work_min < MIN_CANDIDATE_EVALUATION_WORK
        || args.candidate_reserve_min < MIN_CANDIDATE_RESERVE
        || args.candidate_reserve_min > args.candidate_work_min
    {
        bail!(
            "candidate work must preserve at least eight waves of runway and two protected waves"
        );
    }
    if args.min_complete_reserve_waves < MIN_COMPLETE_RESERVE_WAVES
        || args.min_complete_reserve_waves > args.max_overlap_waves
    {
        bail!("complete reserve waves must be within 2..=max-overlap-waves");
    }
    if !(1..=64).contains(&args.max_overlap_waves) {
        bail!("--max-overlap-waves must be in 1..=64");
    }
    if args.max_builds_per_cycle != 1 {
        bail!("--max-builds-per-cycle must be exactly 1 for keeper acknowledgment");
    }
    if !(25..=100).contains(&args.max_candidates_per_manifest) {
        bail!("--max-candidates-per-manifest must be in 25..=100");
    }
    Ok(())
}

pub(super) fn performance_measurements(args: &Args) -> Result<PerformanceMeasurements> {
    let performance = args.root.join("ops/supply-performance.v1.json");
    let value: Value =
        serde_json::from_slice(&fs::read(&performance).context("supply performance unavailable")?)
            .context("parse supply performance")?;
    if value.get("schema").and_then(Value::as_str) != Some("SupplyPerformance.v1") {
        bail!("supply performance schema drift");
    }
    let generated_at_ms = value
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .map(u128::from)
        .context("supply performance timestamp is absent")?;
    if now_ms().saturating_sub(generated_at_ms)
        > u128::from(args.performance_max_age_seconds) * 1_000
    {
        bail!("supply performance is stale");
    }
    let p95_jobs_per_minute = value
        .get("p95_jobs_per_minute")
        .and_then(Value::as_f64)
        .or_else(|| {
            value
                .get("evaluator_burn_per_second_p95")
                .and_then(Value::as_f64)
                .map(|per_second| per_second * 60.0)
        })
        .context("p95 jobs-per-minute measurement is absent")?;
    let p95_candidate_evaluations_per_minute = value
        .get("p95_candidate_evaluations_per_minute")
        .and_then(Value::as_f64)
        .context("p95 candidate-evaluations-per-minute measurement is absent")?;
    let proposer_to_ready_minutes = value
        .get("proposer_to_ready_minutes_p95")
        .and_then(Value::as_f64)
        .or_else(|| {
            value
                .get("proposer_to_ready_seconds_p95")
                .and_then(Value::as_f64)
                .map(|seconds| seconds / 60.0)
        })
        .context("p95 proposer-to-ready measurement is absent")?;
    if !p95_jobs_per_minute.is_finite()
        || !p95_candidate_evaluations_per_minute.is_finite()
        || !proposer_to_ready_minutes.is_finite()
        || p95_jobs_per_minute <= 0.0
        || p95_candidate_evaluations_per_minute <= 0.0
        || proposer_to_ready_minutes <= 0.0
        || value
            .get("proposer_samples")
            .and_then(Value::as_u64)
            .unwrap_or_default()
            == 0
    {
        bail!("supply performance measurements are missing or invalid");
    }
    let burn_runway = (120.0 * p95_candidate_evaluations_per_minute).ceil() as usize;
    let lead_time_runway =
        (1.5 * proposer_to_ready_minutes * p95_candidate_evaluations_per_minute).ceil() as usize;
    Ok(PerformanceMeasurements {
        generated_at_ms,
        p95_jobs_per_minute,
        p95_candidate_evaluations_per_minute,
        proposer_to_ready_minutes,
        candidate_evaluation_work_target: args
            .candidate_work_min
            .max(burn_runway)
            .max(lead_time_runway),
    })
}

pub(super) fn cycle(args: &Args) -> Result<CycleReport> {
    let status = load_queue_status(args)?;
    let measured = performance_measurements(args);
    if !status.admission_allowed {
        return Ok(held_report(args, &status, measured.ok(), None));
    }
    let performance = match measured {
        Ok(value) => value,
        Err(error) => {
            return Ok(held_report(
                args,
                &status,
                None,
                Some(format!("performance_fail_closed:{error:#}")),
            ))
        }
    };
    let inputs = checkpoint_inputs(args, &status)?;
    let epoch = checkpoint_epoch(&inputs.checkpoint_sha256)?;
    if status.checkpoint_epoch.as_deref() != Some(epoch.as_str()) {
        bail!("queue keeper checkpoint epoch differs from its checkpoint identity");
    }
    let claimed = claimed_manifest_hashes(&args.root.join("ops/backlog.sqlite"))?;
    let waves = ready_waves(
        &args.root.join("waves"),
        &inputs.checkpoint_sha256,
        &epoch,
        &claimed,
        args.max_overlap_waves + 1,
    )?;
    if waves.len() > args.max_overlap_waves {
        bail!("more than the bounded number of current-checkpoint waves remain unclaimed");
    }
    let mut pending_ack = load_pending_ack(&args.root, &inputs.checkpoint_sha256)?;
    if let Some(pending) = &pending_ack {
        if keeper_acknowledged(args, &status, &inputs, &epoch, &claimed, pending)? {
            mark_keeper_acknowledged(&args.root, &pending.wave, &inputs.checkpoint_sha256)?;
            pending_ack = None;
        }
    }
    validate_keeper_admission(
        &status,
        &waves,
        &args.root.join("waves"),
        pending_ack.as_ref().map(|pending| pending.wave.as_str()),
        args.max_overlap_waves,
    )?;
    let queue_depth = status
        .queued_starting
        .checked_add(status.dispatching)
        .context("queue depth overflow")?;
    let dynamic_queue_target = usize::try_from(status.queued_target)
        .context("keeper queue target is negative or overflows usize")?;
    ensure!(
        (2_048..=4_096).contains(&dynamic_queue_target)
            && status.queued_hard_cap == args.queued_hard_cap as i64,
        "keeper queue target/hard-cap differs from the canonical dynamic policy"
    );
    let dynamic_ready_reserve = dynamic_queue_target
        .checked_add(2 * manifests_per_wave(args.max_candidates_per_manifest))
        .context("dynamic ready-reserve target overflow")?;
    if queue_depth > args.queued_hard_cap {
        bail!("current queued+dispatching depth exceeds the configured hard cap");
    }
    let ready_before = status.validated_ready;
    let candidate_evaluation_work = usize::try_from(status.candidate_evaluation_work)
        .context("queue candidate-evaluation work overflows usize")?;
    let protected_candidate_evaluations = status
        .protected_reserve_waves
        .iter()
        .map(|wave| wave.candidate_evaluation_count)
        .try_fold(0_u64, u64::checked_add)
        .context("protected candidate-evaluation work overflow")?;
    let protected_candidate_evaluations = usize::try_from(protected_candidate_evaluations)
        .context("protected candidate-evaluation work overflows usize")?;
    let decision = supply_decision(
        queue_depth,
        candidate_evaluation_work,
        protected_candidate_evaluations,
        waves.len(),
        status.complete_reserve_waves,
        dynamic_queue_target,
        dynamic_queue_target,
        performance.candidate_evaluation_work_target,
        args.queued_hard_cap,
        args.candidate_reserve_min,
        args.min_complete_reserve_waves,
        args.max_overlap_waves,
        args.max_builds_per_cycle,
        args.max_candidates_per_manifest,
    );
    let mut completed = Vec::new();
    let current_checkpoint_manifests = queue_depth.saturating_add(ready_before);
    let manifest_reserve_builds = dynamic_ready_reserve
        .saturating_sub(current_checkpoint_manifests)
        .div_ceil(manifests_per_wave(args.max_candidates_per_manifest))
        .min(args.max_overlap_waves.saturating_sub(waves.len()));
    let manifest_target_feasible = dynamic_ready_reserve
        <= args.queued_hard_cap.saturating_add(
            args.max_overlap_waves
                .saturating_mul(manifests_per_wave(args.max_candidates_per_manifest)),
        );
    let builds_planned =
        if pending_ack.is_none() && decision.target_feasible && manifest_target_feasible {
            decision.builds_planned.max(manifest_reserve_builds).min(1)
        } else {
            0
        };
    if args.build && builds_planned == 1 {
        ensure_native_wave_builder(&args.wave_builder)?;
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        let built = build_next_wave(args, &inputs)?;
        assert_checkpoint_unchanged(args, &inputs.checkpoint_sha256)?;
        if let Some(built) = built {
            pending_ack = Some(PendingAck {
                wave: built.wave.clone(),
                since_ms: now_ms(),
            });
            completed.push(built);
        }
    }
    let runway_compliant = decision.target_feasible
        && manifest_target_feasible
        && current_checkpoint_manifests >= dynamic_ready_reserve
        && decision.candidate_evaluation_work >= decision.candidate_evaluation_work_target
        && protected_candidate_evaluations >= args.candidate_reserve_min
        && status.complete_reserve_waves >= args.min_complete_reserve_waves;
    let report = CycleReport {
        schema: "JopeMicroWaveSupplyStatus.v2",
        generated_at_ms: now_ms(),
        mode: if args.build { "build" } else { "audit" },
        active_checkpoint_sha256: inputs.checkpoint_sha256,
        checkpoint_epoch: epoch,
        queue_floor: dynamic_queue_target,
        queued_target: dynamic_queue_target,
        queued_hard_cap: args.queued_hard_cap,
        ready_reserve_min: dynamic_ready_reserve,
        useful_work_min: args.useful_work_min,
        useful_work_target: args.useful_work_min,
        useful_work_target_formula: "legacy_job_count_only_not_used_for_capacity_decisions",
        p95_jobs_per_minute: Some(performance.p95_jobs_per_minute),
        p95_candidate_evaluations_per_minute: Some(
            performance.p95_candidate_evaluations_per_minute,
        ),
        proposer_to_ready_minutes: Some(performance.proposer_to_ready_minutes),
        performance_generated_at_ms: Some(performance.generated_at_ms),
        performance_fresh: true,
        keeper_queued_target: status.queued_target,
        keeper_queued_hard_cap: status.queued_hard_cap,
        queued: status.queued,
        starting: status.starting,
        queued_starting: status.queued_starting,
        dispatching: status.dispatching,
        queue_depth,
        below_floor: decision.below_floor,
        overloaded: decision.overloaded,
        current_ready_waves: waves,
        ready_manifests: ready_before,
        validated_ready: status.validated_ready,
        reserve_manifests: status.reserve_manifests,
        admission_manifests: status.admission_manifests,
        complete_current_checkpoint_waves: status.complete_current_checkpoint_waves,
        partial_current_checkpoint_waves: status.partial_current_checkpoint_waves,
        complete_reserve_waves: status.complete_reserve_waves,
        minimum_complete_reserve_waves: args.min_complete_reserve_waves,
        protected_reserve_waves: status.protected_reserve_waves.clone(),
        useful_work: status.useful_work,
        candidate_work_min: args.candidate_work_min,
        candidate_reserve_min: args.candidate_reserve_min,
        candidate_evaluation_work: decision.candidate_evaluation_work,
        candidate_evaluation_work_target: decision.candidate_evaluation_work_target,
        protected_candidate_evaluations,
        target_feasible: decision.target_feasible && manifest_target_feasible,
        critical_infeasibility_reason: if !decision.target_feasible {
            Some(format!(
                "eight_overlap_waves_cannot_meet_candidate_work_target:{}",
                performance.candidate_evaluation_work_target
            ))
        } else if !manifest_target_feasible {
            Some(format!(
                "overlap_bound_cannot_meet_dynamic_manifest_reserve:{}",
                dynamic_ready_reserve
            ))
        } else {
            None
        },
        projected_jobs_before_build: status.useful_work,
        // A built wave is deliberately not counted until a newer keeper
        // status acknowledges it in the bounded admission set.
        projected_jobs_after_build: status.useful_work,
        max_overlap_waves: args.max_overlap_waves,
        builds_planned,
        builds_completed: completed,
        awaiting_keeper_ack: pending_ack.as_ref().map(|pending| pending.wave.clone()),
        awaiting_keeper_ack_since_ms: pending_ack.as_ref().map(|pending| pending.since_ms),
        keeper_status_generated_at_ms: status.generated_at_ms,
        admission_allowed: status.admission_allowed && runway_compliant,
        hold_reason: (!runway_compliant).then(|| {
            if !decision.target_feasible {
                "useful_work_target_infeasible_with_eight_overlap_waves".to_owned()
            } else if !manifest_target_feasible {
                "dynamic_manifest_reserve_infeasible_with_overlap_bound".to_owned()
            } else if current_checkpoint_manifests < dynamic_ready_reserve {
                "current_checkpoint_manifests_below_dynamic_target_plus_two_waves".to_owned()
            } else if status.complete_reserve_waves < args.min_complete_reserve_waves {
                "complete_reserve_waves_below_minimum".to_owned()
            } else if protected_candidate_evaluations < args.candidate_reserve_min {
                "candidate_evaluation_reserve_below_minimum".to_owned()
            } else {
                "candidate_evaluation_work_below_dynamic_target".to_owned()
            }
        }),
    };
    if !report.builds_completed.is_empty() {
        append_event(
            &args.root.join("ops/micro-wave-supply-events.v1.jsonl"),
            &serde_json::to_value(&report)?,
        )?;
    }
    Ok(report)
}

pub(super) fn held_report(
    args: &Args,
    status: &QueueStatus,
    performance: Option<PerformanceMeasurements>,
    explicit_reason: Option<String>,
) -> CycleReport {
    let active = status.active_checkpoint_sha256.clone().unwrap_or_default();
    let epoch = status.checkpoint_epoch.clone().unwrap_or_default();
    let queue_depth = status.queued_starting.saturating_add(status.dispatching);
    let dynamic_queue_target = usize::try_from(status.queued_target)
        .ok()
        .filter(|target| (2_048..=4_096).contains(target))
        .unwrap_or(args.queued_target);
    let dynamic_ready_reserve = dynamic_queue_target
        .saturating_add(2 * manifests_per_wave(args.max_candidates_per_manifest));
    CycleReport {
        schema: "JopeMicroWaveSupplyStatus.v2",
        generated_at_ms: now_ms(),
        mode: if args.build { "build" } else { "audit" },
        active_checkpoint_sha256: active,
        checkpoint_epoch: epoch,
        queue_floor: dynamic_queue_target,
        queued_target: dynamic_queue_target,
        queued_hard_cap: args.queued_hard_cap,
        ready_reserve_min: dynamic_ready_reserve,
        useful_work_min: args.useful_work_min,
        useful_work_target: args.useful_work_min,
        useful_work_target_formula: "legacy_job_count_only_not_used_for_capacity_decisions",
        p95_jobs_per_minute: performance.map(|value| value.p95_jobs_per_minute),
        p95_candidate_evaluations_per_minute: performance
            .map(|value| value.p95_candidate_evaluations_per_minute),
        proposer_to_ready_minutes: performance.map(|value| value.proposer_to_ready_minutes),
        performance_generated_at_ms: performance.map(|value| value.generated_at_ms),
        performance_fresh: performance.is_some(),
        keeper_queued_target: status.queued_target,
        keeper_queued_hard_cap: status.queued_hard_cap,
        queued: status.queued,
        starting: status.starting,
        queued_starting: status.queued_starting,
        dispatching: status.dispatching,
        queue_depth,
        below_floor: queue_depth < dynamic_queue_target,
        overloaded: queue_depth > dynamic_queue_target,
        current_ready_waves: Vec::new(),
        ready_manifests: status.validated_ready,
        validated_ready: status.validated_ready,
        reserve_manifests: status.reserve_manifests,
        admission_manifests: status.admission_manifests,
        complete_current_checkpoint_waves: status.complete_current_checkpoint_waves,
        partial_current_checkpoint_waves: status.partial_current_checkpoint_waves,
        complete_reserve_waves: status.complete_reserve_waves,
        minimum_complete_reserve_waves: args.min_complete_reserve_waves,
        protected_reserve_waves: status.protected_reserve_waves.clone(),
        useful_work: status.useful_work,
        candidate_work_min: args.candidate_work_min,
        candidate_reserve_min: args.candidate_reserve_min,
        candidate_evaluation_work: status.candidate_evaluation_work as usize,
        candidate_evaluation_work_target: performance
            .map(|value| value.candidate_evaluation_work_target)
            .unwrap_or(args.candidate_work_min),
        protected_candidate_evaluations: status
            .protected_reserve_waves
            .iter()
            .map(|wave| wave.candidate_evaluation_count as usize)
            .sum(),
        target_feasible: false,
        critical_infeasibility_reason: explicit_reason.clone(),
        projected_jobs_before_build: queue_depth.saturating_add(status.validated_ready),
        projected_jobs_after_build: queue_depth.saturating_add(status.validated_ready),
        max_overlap_waves: args.max_overlap_waves,
        builds_planned: 0,
        builds_completed: Vec::new(),
        awaiting_keeper_ack: None,
        awaiting_keeper_ack_since_ms: None,
        keeper_status_generated_at_ms: status.generated_at_ms,
        admission_allowed: false,
        hold_reason: explicit_reason.or_else(|| {
            status
                .error
                .clone()
                .or_else(|| status.fail_closed_reason.clone())
                .or_else(|| {
                    status
                        .degraded
                        .then(|| "queue keeper is degraded".to_owned())
                })
                .or_else(|| Some("queue admission is held".to_owned()))
        }),
    }
}

pub(super) fn load_queue_status(args: &Args) -> Result<QueueStatus> {
    let path = args.queue_status();
    let status: QueueStatus = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse queue status {}", path.display()))?;
    if status.schema != "JopeSelfplayQueueKeeper.v3" {
        bail!("queue keeper status schema is not v3");
    }
    let maximum_age_ms = u128::from(args.queue_status_max_age_seconds) * 1_000;
    if now_ms().saturating_sub(status.generated_at_ms) > maximum_age_ms {
        bail!("queue keeper status is stale");
    }
    if status.queued.saturating_add(status.starting) != status.queued_starting {
        bail!("queue keeper queued/starting aggregate drift");
    }
    if status
        .reserve_manifests
        .saturating_add(status.admission_manifests)
        != status.unclaimed_current_checkpoint_manifests
    {
        bail!("queue keeper reserve/admission manifest aggregate drift");
    }
    if status.complete_reserve_waves != status.protected_reserve_waves.len() {
        bail!("queue keeper complete reserve-wave aggregate drift");
    }
    let useful_work = status
        .queued
        .saturating_add(status.starting)
        .saturating_add(status.dispatching)
        .saturating_add(status.validated_ready);
    if useful_work != status.useful_work {
        bail!("queue keeper useful-work aggregate drift");
    }
    if (status.useful_work == 0) != (status.candidate_evaluation_work == 0)
        || status.candidate_evaluation_work < status.useful_work as u64
    {
        bail!("queue keeper candidate-evaluation work aggregate drift");
    }
    if status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .any(|wave| wave.unclaimed_manifests == 0 || wave.candidate_evaluation_count == 0)
    {
        bail!("queue keeper wave lacks candidate-evaluation work");
    }
    let admission = status
        .admission_waves
        .iter()
        .map(|wave| wave.wave.as_str())
        .collect::<BTreeSet<_>>();
    if status
        .protected_reserve_waves
        .iter()
        .any(|wave| !wave.complete || admission.contains(wave.wave.as_str()))
    {
        bail!("queue keeper reserve/admission wave separation drift");
    }
    Ok(status)
}

pub(super) fn load_pending_ack(root: &Path, checkpoint_sha256: &str) -> Result<Option<PendingAck>> {
    let path = root.join("ops/micro-wave-supply-v2.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse prior supply status {}", path.display()))?;
    if value.get("schema").and_then(Value::as_str) != Some("JopeMicroWaveSupplyStatus.v2")
        || value
            .get("active_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(checkpoint_sha256)
    {
        return Ok(None);
    }
    let Some(wave) = value.get("awaiting_keeper_ack").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !valid_wave_id(wave) {
        bail!("prior supply status has an invalid acknowledgment wave");
    }
    let since_ms = value
        .get("awaiting_keeper_ack_since_ms")
        .and_then(Value::as_u64)
        .context("prior supply status lacks acknowledgment timestamp")?;
    Ok(Some(PendingAck {
        wave: wave.to_owned(),
        since_ms: u128::from(since_ms),
    }))
}

pub(super) fn keeper_acknowledged(
    args: &Args,
    status: &QueueStatus,
    inputs: &CheckpointInputs,
    epoch: &str,
    claimed: &BTreeSet<String>,
    pending: &PendingAck,
) -> Result<bool> {
    if status.generated_at_ms <= pending.since_ms {
        return Ok(false);
    }
    if status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .any(|wave| wave.wave == pending.wave)
    {
        return Ok(true);
    }
    let directory = args.root.join("waves").join(&pending.wave);
    let Some(wave) = validate_ready_wave(
        &pending.wave,
        &directory,
        &inputs.checkpoint_sha256,
        epoch,
        claimed,
    )?
    else {
        bail!("keeper did not acknowledge the built current-checkpoint wave");
    };
    Ok(wave.unclaimed_manifests == 0)
}

pub(super) fn validate_keeper_admission(
    status: &QueueStatus,
    ready: &[ReadyWave],
    waves_root: &Path,
    pending_wave: Option<&str>,
    max_overlap_waves: usize,
) -> Result<()> {
    if status
        .admission_waves
        .len()
        .saturating_add(status.protected_reserve_waves.len())
        > max_overlap_waves
    {
        bail!("keeper admission wave set exceeds the bounded overlap");
    }
    if status
        .admission_waves
        .windows(2)
        .any(|pair| pair[0].wave >= pair[1].wave)
    {
        bail!("keeper admission waves are not deterministic oldest-first");
    }
    let reported_unclaimed = status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .try_fold(0_usize, |total, wave| {
            total.checked_add(wave.unclaimed_manifests)
        })
        .context("keeper unclaimed manifest count overflow")?;
    if reported_unclaimed != status.unclaimed_current_checkpoint_manifests {
        bail!("keeper admission wave counts do not match its aggregate");
    }
    let ready_by_wave = ready
        .iter()
        .map(|wave| (wave.wave.as_str(), wave))
        .collect::<BTreeMap<_, _>>();
    let mut admitted = BTreeSet::new();
    for keeper_wave in status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
    {
        if !admitted.insert(keeper_wave.wave.as_str()) {
            bail!("keeper admission wave is duplicated");
        }
        if let Some(local) = ready_by_wave.get(keeper_wave.wave.as_str()) {
            if keeper_wave.summary_sha256 != local.summary_sha256
                || local.unclaimed_manifests > keeper_wave.unclaimed_manifests
            {
                bail!("keeper admission wave provenance or count drift");
            }
        } else {
            if !valid_wave_id(&keeper_wave.wave) {
                bail!("keeper admission wave identity is invalid");
            }
            let summary = waves_root.join(&keeper_wave.wave).join("summary.json");
            if !summary.is_file() || sha256_file(&summary)? != keeper_wave.summary_sha256 {
                bail!("keeper admitted a wave with no matching local immutable summary");
            }
        }
    }
    if status
        .protected_reserve_waves
        .iter()
        .any(|wave| !wave.complete || wave.manifest_count != wave.unclaimed_manifests)
    {
        bail!("keeper protected a partial reserve wave");
    }
    for wave in ready {
        if !admitted.contains(wave.wave.as_str()) && pending_wave != Some(wave.wave.as_str()) {
            bail!("current-checkpoint unclaimed wave is outside keeper admission");
        }
    }
    Ok(())
}

pub(super) fn checkpoint_inputs(args: &Args, status: &QueueStatus) -> Result<CheckpointInputs> {
    let active = status
        .active_checkpoint_sha256
        .as_deref()
        .context("queue keeper has no active checkpoint")?;
    validate_sha256(active)?;
    let checkpoint_pointer = if let Some(path) = &args.checkpoint_pointer {
        path.clone()
    } else {
        ["training/learning.json", "training/accepted.json"]
            .into_iter()
            .map(|relative| args.root.join(relative))
            .find(|path| path.is_file())
            .unwrap_or_else(|| args.proposals_pointer())
    };
    if pointer_checkpoint(&checkpoint_pointer)? != active {
        bail!("authoritative checkpoint pointer differs from queue keeper status");
    }
    let checkpoint_value: Value = serde_json::from_slice(&fs::read(&checkpoint_pointer)?)?;
    let declared_checkpoint = checkpoint_value
        .get("checkpoint")
        .or_else(|| checkpoint_value.pointer("/legacy_adopted/checkpoint"))
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let checkpoint_artifact = [
        declared_checkpoint,
        Some(
            args.root
                .join("training/immutable/checkpoints")
                .join(format!("{active}.model.pt")),
        ),
        Some(paths::legacy_checkpoints_root().join(format!("{active}.model.pt"))),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.is_file() && sha256_file(path).ok().as_deref() == Some(active))
    .context("active checkpoint artifact is absent or content-invalid")?;
    let proposals_pointer = args.proposals_pointer();
    if pointer_checkpoint(&proposals_pointer)? != active {
        bail!("proposal pointer is not from the active checkpoint");
    }
    Ok(CheckpointInputs {
        checkpoint_sha256: active.to_owned(),
        checkpoint_pointer,
        checkpoint_artifact,
        proposals_pointer,
    })
}

pub(super) fn assert_checkpoint_unchanged(args: &Args, expected: &str) -> Result<()> {
    let status = load_queue_status(args)?;
    let inputs = checkpoint_inputs(args, &status)?;
    if inputs.checkpoint_sha256 != expected {
        bail!("checkpoint advanced while a micro-wave was being built");
    }
    Ok(())
}

pub(super) fn pointer_checkpoint(path: &Path) -> Result<String> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("parse checkpoint pointer {}", path.display()))?;
    let digest = value
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .context("checkpoint pointer lacks checkpoint_sha256")?;
    validate_sha256(digest)?;
    Ok(digest.to_owned())
}

pub(super) fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid SHA-256 identity");
    }
    Ok(())
}

pub(super) fn claimed_manifest_hashes(path: &Path) -> Result<BTreeSet<String>> {
    if !path.is_file() {
        return Ok(BTreeSet::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let mut statement = connection
        .prepare("SELECT DISTINCT manifest_sha256 FROM claims WHERE manifest_sha256!=''")?;
    let mut hashes = statement
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<BTreeSet<_>>>()?;
    drop(statement);
    let has_rejections: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='rejected_manifests'
         )",
        [],
        |row| row.get(0),
    )?;
    if !has_rejections {
        bail!("rejection-aware keeper backlog lacks rejected_manifests");
    }
    let mut rejected =
        connection.prepare("SELECT manifest,manifest_sha256 FROM rejected_manifests")?;
    for row in rejected.query_map([], |row| {
        Ok((
            PathBuf::from(row.get::<_, String>(0)?),
            row.get::<_, Option<String>>(1)?,
        ))
    })? {
        let (manifest, recorded) = row?;
        let digest = match recorded.filter(|value| !value.is_empty()) {
            Some(digest) => digest,
            None if manifest.is_file() => sha256_file(&manifest)?,
            None => bail!(
                "rejected manifest lacks both a content hash and readable file: {}",
                manifest.display()
            ),
        };
        validate_sha256(&digest)?;
        hashes.insert(digest);
    }
    Ok(hashes)
}
