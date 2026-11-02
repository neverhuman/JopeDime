use super::*;

pub(super) fn parse_policy(text: &str) -> Result<StaticPolicy> {
    let mut poll_ms = None;
    let mut telemetry_ms = None;
    let mut current_node: Option<String> = None;
    let mut multipliers = BTreeMap::new();
    let mut seen_nodes = BTreeSet::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[node]]" {
            current_node = Some(String::new());
            continue;
        }
        let Some((key, raw_value)) = line.split_once('=') else {
            bail!("unsupported policy line: {line}");
        };
        let key = key.trim();
        let value = raw_value.trim().trim_matches('"');
        if let Some(node) = current_node.as_mut() {
            match key {
                "name" => {
                    if node.is_empty() {
                        *node = value.to_owned();
                        if !seen_nodes.insert(node.clone()) {
                            bail!("duplicate node policy for {node}");
                        }
                        multipliers.insert(node.clone(), 1.0);
                    } else {
                        bail!("duplicate node name in policy");
                    }
                }
                "cpu_job_slot_multiplier" => {
                    if node.is_empty() {
                        bail!("node multiplier appears before node name");
                    }
                    let multiplier = value
                        .parse::<f64>()
                        .with_context(|| format!("parse multiplier for {node}"))?;
                    multipliers.insert(node.clone(), multiplier);
                }
                _ => {}
            }
        } else {
            match key {
                "poll_ms" => poll_ms = Some(value.parse()?),
                "telemetry_ms" => telemetry_ms = Some(value.parse()?),
                _ => {}
            }
        }
    }
    if current_node.as_deref() == Some("") {
        bail!("node policy is missing its name");
    }
    Ok(StaticPolicy {
        poll_ms: poll_ms.context("policy lacks poll_ms")?,
        telemetry_ms: telemetry_ms.context("policy lacks telemetry_ms")?,
        multipliers,
    })
}

pub(super) fn validate_policy(policy: &StaticPolicy) -> Result<()> {
    if policy.poll_ms != 2_000 || policy.telemetry_ms != 1_000 {
        bail!("self-play policy must use a 2s scheduler tick and 1s telemetry");
    }
    for node in ["xbabe1", "xbabe2", "xbabe3"] {
        if policy.multipliers.get(node).copied() != Some(1.0) {
            bail!("{node} must start at multiplier 1.0");
        }
    }
    if policy.multipliers.get("xbabe4").copied() != Some(2.0) {
        bail!("xbabe4 policy target must be multiplier 2.0");
    }
    if policy
        .multipliers
        .values()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!("invalid CPU slot multiplier");
    }
    Ok(())
}

pub(super) fn validate_metrics(metrics: &ExternalMetrics) -> Result<()> {
    for (name, value) in [
        ("failure_rate", metrics.failure_rate),
        ("p95_runtime_seconds", metrics.p95_runtime_seconds),
        ("throughput_per_slot", metrics.throughput_per_slot),
        ("baseline_failure_rate", metrics.baseline_failure_rate),
        (
            "baseline_p95_runtime_seconds",
            metrics.baseline_p95_runtime_seconds,
        ),
        (
            "baseline_throughput_per_slot",
            metrics.baseline_throughput_per_slot,
        ),
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("{name} must be finite and nonnegative");
        }
    }
    Ok(())
}
