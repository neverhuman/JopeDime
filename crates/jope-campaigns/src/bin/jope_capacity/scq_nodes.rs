use super::*;

pub(crate) type ScqNodes = (Vec<NodeTelemetry>, BTreeMap<String, Vec<String>>, bool);

pub(crate) fn parse_scq_nodes(value: &Value) -> Result<ScqNodes> {
    let rows = value
        .get("Nodes")
        .and_then(|row| row.get("nodes"))
        .and_then(Value::as_array)
        .context("SCQ nodes response lacks Nodes.nodes")?;
    let mut nodes = Vec::new();
    let mut scheduler_supports = true;
    for row in rows {
        let node = row
            .get("node")
            .and_then(Value::as_str)
            .context("SCQ node lacks name")?
            .to_owned();
        let physical = integer(row, "cores")?;
        let effective = integer(row, "effective_cores")?;
        let running = integer(row, "running_jobs")?;
        if physical < effective || effective < 0 || running < 0 {
            bail!("invalid capacity counters for {node}");
        }
        let total_memory = number(row, "mem_total_gb")?;
        let available_memory = number(row, "mem_avail_gb")?;
        if total_memory <= 0.0 || !(0.0..=total_memory).contains(&available_memory) {
            bail!("invalid memory telemetry for {node}");
        }
        let scheduler_multiplier = row.get("cpu_job_slot_multiplier").and_then(Value::as_f64);
        let scheduler_slot_target = row.get("slot_target").and_then(Value::as_i64);
        scheduler_supports &= scheduler_multiplier.is_some() && scheduler_slot_target.is_some();
        if scheduler_multiplier.is_some_and(|value| !value.is_finite() || value <= 0.0)
            || scheduler_slot_target.is_some_and(|value| value < 0)
        {
            bail!("invalid scheduler multiplier telemetry for {node}");
        }
        nodes.push(NodeTelemetry {
            node,
            physical_cores: physical,
            effective_cores: effective,
            running_slots: running,
            scheduler_multiplier,
            scheduler_slot_target,
            cpu_busy_fraction: number(row, "cpu_busy_pct")? / 100.0,
            pressure: StoredPressure {
                cpu_psi: number(row, "psi_cpu10")?,
                memory_psi: number(row, "psi_mem10")?,
                io_psi: number(row, "psi_io10")?,
                available_memory_fraction: Some(available_memory / total_memory),
                ..zero_pressure()
            },
        });
    }
    let expected = BTreeSet::from(["xbabe1", "xbabe2", "xbabe3", "xbabe4"]);
    let actual = nodes.iter().map(|row| row.node.as_str()).collect();
    if expected != actual {
        bail!("SCQ node inventory must contain exactly xbabe1..xbabe4");
    }
    let mut gpu_allocations = BTreeMap::<String, Vec<String>>::new();
    for row in value
        .get("Nodes")
        .and_then(|row| row.get("gpus"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(node) = row.get("node").and_then(Value::as_str) else {
            continue;
        };
        let status = row
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if status != "free" {
            gpu_allocations
                .entry(node.to_owned())
                .or_default()
                .push(status.to_owned());
        }
    }
    Ok((nodes, gpu_allocations, scheduler_supports))
}

pub(crate) fn zero_pressure() -> StoredPressure {
    StoredPressure {
        cpu_psi: 0.0,
        memory_psi: 0.0,
        io_psi: 0.0,
        available_memory_fraction: None,
        failure_rate: 0.0,
        p95_runtime_seconds: 0.0,
        throughput_per_slot: 0.0,
    }
}

pub(crate) fn integer(value: &Value, key: &str) -> Result<i64> {
    value
        .get(key)
        .and_then(|field| {
            field
                .as_i64()
                .or_else(|| field.as_u64().map(|row| row as i64))
        })
        .with_context(|| format!("SCQ node lacks integer {key}"))
}

pub(crate) fn number(value: &Value, key: &str) -> Result<f64> {
    let result = value
        .get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("SCQ node lacks numeric {key}"))?;
    if !result.is_finite() {
        bail!("SCQ node {key} is not finite");
    }
    Ok(result)
}
