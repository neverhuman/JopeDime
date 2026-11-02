use super::*;

pub fn node_slot_target(effective_cores: i64, multiplier: f64) -> PolicyResult<i64> {
    if effective_cores < 0 || !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(error("invalid effective cores or CPU-job slot multiplier"));
    }
    let target = effective_cores as f64 * multiplier;
    if !target.is_finite() || target > i64::MAX as f64 {
        return Err(error("CPU-job slot target overflows"));
    }
    Ok(target.floor() as i64)
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeCapacity {
    pub node: String,
    pub physical_cores: i64,
    pub effective_cores: i64,
    pub cpu_job_slot_multiplier: f64,
    pub slot_target: i64,
    pub running_slots: i64,
}

pub fn node_capacity(
    node: &str,
    physical_cores: i64,
    effective_cores: i64,
    running_slots: i64,
    multiplier: f64,
) -> PolicyResult<NodeCapacity> {
    if physical_cores < effective_cores
        || effective_cores < 0
        || running_slots < 0
        || node.is_empty()
    {
        return Err(error("invalid node capacity counters"));
    }
    Ok(NodeCapacity {
        node: node.to_owned(),
        physical_cores,
        effective_cores,
        cpu_job_slot_multiplier: multiplier,
        slot_target: node_slot_target(effective_cores, multiplier)?,
        running_slots,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PressureSample {
    pub cpu_psi: f64,
    pub memory_psi: f64,
    pub io_psi: f64,
    pub available_memory_fraction: Option<f64>,
    pub failure_rate: f64,
    pub p95_runtime_seconds: f64,
    pub throughput_per_slot: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PressureBaseline {
    pub failure_rate: f64,
    pub p95_runtime_seconds: f64,
    pub throughput_per_slot: f64,
}

pub fn unsafe_sample(
    sample: &PressureSample,
    baseline: &PressureBaseline,
) -> PolicyResult<Vec<&'static str>> {
    for (name, value) in [
        ("cpu_psi", sample.cpu_psi),
        ("memory_psi", sample.memory_psi),
        ("io_psi", sample.io_psi),
        ("failure_rate", sample.failure_rate),
        ("p95_runtime_seconds", sample.p95_runtime_seconds),
        ("throughput_per_slot", sample.throughput_per_slot),
        ("baseline failure rate", baseline.failure_rate),
        ("baseline p95 runtime seconds", baseline.p95_runtime_seconds),
        ("baseline throughput per slot", baseline.throughput_per_slot),
    ] {
        validate_finite(value, name)?;
    }
    if sample
        .available_memory_fraction
        .is_some_and(|value| !value.is_finite())
    {
        return Err(error("available memory fraction must be finite"));
    }
    let mut reasons = Vec::new();
    if sample.cpu_psi >= 0.25 {
        reasons.push("cpu_psi");
    }
    if sample.memory_psi >= 0.05 {
        reasons.push("memory_psi");
    }
    if sample.io_psi >= 0.10 {
        reasons.push("io_psi");
    }
    if sample.available_memory_fraction.unwrap_or(1.0) < 0.15 {
        reasons.push("available_memory");
    }
    if sample.failure_rate - baseline.failure_rate > 0.0025 {
        reasons.push("failure_rate_regression");
    }
    let runtime_regressed = baseline.p95_runtime_seconds > 0.0
        && sample.p95_runtime_seconds > 1.5 * baseline.p95_runtime_seconds;
    let throughput_regressed = baseline.throughput_per_slot > 0.0
        && sample.throughput_per_slot <= 0.8 * baseline.throughput_per_slot;
    if runtime_regressed && throughput_regressed {
        reasons.push("runtime_and_throughput_regression");
    }
    Ok(reasons)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RampStage {
    Baseline,
    Ramp30,
    Target40,
    RolledBack,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RampState {
    pub stage: RampStage,
    pub effective_cores: i64,
    pub multiplier: f64,
    pub slot_target: i64,
    pub stage_started_at_ms: Option<i64>,
    pub rollback_at_ms: Option<i64>,
    pub rollback_reasons: Vec<String>,
    pub baseline: PressureBaseline,
}

impl RampState {
    pub fn baseline(effective_cores: i64, baseline: PressureBaseline) -> PolicyResult<Self> {
        Ok(Self {
            stage: RampStage::Baseline,
            effective_cores,
            multiplier: 1.0,
            slot_target: node_slot_target(effective_cores, 1.0)?,
            stage_started_at_ms: None,
            rollback_at_ms: None,
            rollback_reasons: Vec::new(),
            baseline,
        })
    }
}

pub fn ramp_decision(
    state: &RampState,
    samples: &[PressureSample],
    now_at_ms: i64,
) -> PolicyResult<RampState> {
    let recent = samples.iter().rev().take(3).collect::<Vec<_>>();
    let unsafe_values = recent
        .iter()
        .map(|sample| unsafe_sample(sample, &state.baseline))
        .collect::<PolicyResult<Vec<_>>>()?;
    if unsafe_values.len() == 3 && unsafe_values.iter().all(|reasons| !reasons.is_empty()) {
        let mut reasons = unsafe_values
            .into_iter()
            .flatten()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        reasons.sort();
        reasons.dedup();
        let mut result = state.clone();
        result.stage = RampStage::RolledBack;
        result.multiplier = 1.0;
        result.slot_target = state.effective_cores;
        result.rollback_at_ms = Some(now_at_ms);
        result.rollback_reasons = reasons;
        return Ok(result);
    }
    let mut result = state.clone();
    match state.stage {
        RampStage::Baseline => {
            result.stage = RampStage::Ramp30;
            result.multiplier = 1.5;
            result.slot_target = RAMP_MID_SLOTS;
            result.stage_started_at_ms = Some(now_at_ms);
        }
        RampStage::Ramp30 => {
            let stage_started = state.stage_started_at_ms.unwrap_or(now_at_ms);
            if now_at_ms - stage_started >= RAMP_MID_DURATION_MS {
                result.stage = RampStage::Target40;
                result.multiplier = 2.0;
                result.slot_target = RAMP_FINAL_SLOTS;
                result.stage_started_at_ms = Some(now_at_ms);
            }
        }
        RampStage::Target40 | RampStage::RolledBack => {}
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BurnSample {
    pub cluster_cpu_fraction: f64,
    pub pressure: PressureSample,
    pub baseline: PressureBaseline,
}

pub fn xbabe2_burn_eligible(samples: &[BurnSample], fresh_queue_depth: i64) -> PolicyResult<bool> {
    if samples.len() < 15 || fresh_queue_depth <= 0 {
        return Ok(false);
    }
    let recent = &samples[samples.len() - 15..];
    if recent
        .iter()
        .any(|sample| !sample.cluster_cpu_fraction.is_finite())
    {
        return Err(error("cluster CPU fraction must be finite"));
    }
    if recent
        .iter()
        .map(|sample| sample.cluster_cpu_fraction)
        .sum::<f64>()
        / 15.0
        >= 0.75
    {
        return Ok(false);
    }
    for sample in recent {
        if !unsafe_sample(&sample.pressure, &sample.baseline)?.is_empty() {
            return Ok(false);
        }
    }
    Ok(true)
}
