use super::*;

pub(crate) fn build_sample(
    sampled_at_ms: u128,
    nodes: &[NodeTelemetry],
    metrics: ExternalMetrics,
) -> Result<StoredSample> {
    let xbabe4 = nodes
        .iter()
        .find(|row| row.node == "xbabe4")
        .context("xbabe4 telemetry is absent")?;
    let total_cores: i64 = nodes.iter().map(|row| row.physical_cores).sum();
    let busy = nodes
        .iter()
        .map(|row| row.cpu_busy_fraction * row.physical_cores as f64)
        .sum::<f64>();
    let mut cluster = zero_pressure();
    cluster.cpu_psi = nodes
        .iter()
        .map(|row| row.pressure.cpu_psi)
        .fold(0.0, f64::max);
    cluster.memory_psi = nodes
        .iter()
        .map(|row| row.pressure.memory_psi)
        .fold(0.0, f64::max);
    cluster.io_psi = nodes
        .iter()
        .map(|row| row.pressure.io_psi)
        .fold(0.0, f64::max);
    cluster.available_memory_fraction = nodes
        .iter()
        .filter_map(|row| row.pressure.available_memory_fraction)
        .min_by(f64::total_cmp);
    let set_metrics = |pressure: &mut StoredPressure| {
        pressure.failure_rate = metrics.failure_rate;
        pressure.p95_runtime_seconds = metrics.p95_runtime_seconds;
        pressure.throughput_per_slot = metrics.throughput_per_slot;
    };
    let mut x4_pressure = xbabe4.pressure.clone();
    set_metrics(&mut x4_pressure);
    set_metrics(&mut cluster);
    Ok(StoredSample {
        sampled_at_ms,
        cluster_cpu_fraction: if total_cores == 0 {
            0.0
        } else {
            busy / total_cores as f64
        },
        xbabe4: x4_pressure,
        cluster,
    })
}

pub(crate) fn load_history(path: &Path) -> Result<Vec<StoredSample>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    serde_json::from_value(read_json(path)?)
        .with_context(|| format!("parse capacity telemetry {}", path.display()))
}

pub(crate) fn sample_due(history: &[StoredSample], now: u128) -> bool {
    history
        .last()
        .is_none_or(|sample| now.saturating_sub(sample.sampled_at_ms) >= SAMPLE_MIN_GAP_MS)
}

pub(crate) fn trailing_minute_samples(history: &[StoredSample]) -> Vec<StoredSample> {
    if history.is_empty() {
        return Vec::new();
    }
    let mut start = history.len() - 1;
    while start > 0 {
        let gap = history[start]
            .sampled_at_ms
            .saturating_sub(history[start - 1].sampled_at_ms);
        if !(SAMPLE_MIN_GAP_MS..=SAMPLE_MAX_GAP_MS).contains(&gap) {
            break;
        }
        start -= 1;
    }
    history[start..].to_vec()
}
