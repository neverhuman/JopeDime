use super::*;

pub(crate) fn cycle(args: &Args) -> Result<CycleReport> {
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

pub(crate) fn held_report(
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
