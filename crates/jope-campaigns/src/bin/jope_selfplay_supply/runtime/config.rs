use super::*;

pub(crate) fn load_supply_failure_streak(path: &Path) -> Result<(Option<String>, u32)> {
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

pub(crate) fn validate_args(args: &Args) -> Result<()> {
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

pub(crate) fn performance_measurements(args: &Args) -> Result<PerformanceMeasurements> {
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
