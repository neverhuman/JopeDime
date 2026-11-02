use super::*;

pub(super) fn parse_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    let rows = value
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .and_then(Value::as_array)
        .context("SCQ ls JSON lacks Jobs.jobs")?;
    rows.iter()
        .map(|row| {
            Ok(SchedulerJob {
                id: row
                    .get("id")
                    .and_then(Value::as_str)
                    .context("SCQ job lacks id")?
                    .to_owned(),
                project: row
                    .get("project")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                state: row
                    .get("state")
                    .and_then(Value::as_str)
                    .context("SCQ job lacks state")?
                    .to_owned(),
                failure_class: row
                    .get("failure_class")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                queue_ts: row
                    .get("queue_ts")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect()
}

pub(super) fn parse_nodes(value: &Value) -> Result<Vec<NodeSlots>> {
    let rows = value
        .get("Nodes")
        .and_then(|value| value.get("nodes"))
        .and_then(Value::as_array)
        .context("SCQ nodes JSON lacks Nodes.nodes")?;
    rows.iter()
        .map(|row| {
            let node = row
                .get("node")
                .and_then(Value::as_str)
                .context("SCQ node lacks identity")?;
            let effective = integer_field(row, "effective_cores").max(0);
            let multiplier = row
                .get("cpu_job_slot_multiplier")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            if !multiplier.is_finite() || !(0.1..=4.0).contains(&multiplier) {
                bail!("SCQ node has an invalid CPU slot multiplier");
            }
            let computed = (effective as f64 * multiplier).floor() as i64;
            let target = row
                .get("job_slot_target")
                .or_else(|| row.get("slot_target"))
                .and_then(Value::as_i64)
                .unwrap_or(computed);
            Ok(NodeSlots {
                node: node.to_owned(),
                physical_cores: integer_field(row, "cores").max(0),
                effective_cores: effective,
                cpu_job_slot_multiplier: multiplier,
                slot_target: target.max(0),
                running_slots: row
                    .get("running_job_slots")
                    .and_then(|field| {
                        field
                            .as_i64()
                            .or_else(|| field.as_u64().map(|value| value as i64))
                    })
                    .unwrap_or_else(|| integer_field(row, "running_jobs"))
                    .max(0),
            })
        })
        .collect()
}

pub(super) fn integer_field(value: &Value, key: &str) -> i64 {
    value
        .get(key)
        .and_then(|field| field.as_i64().or_else(|| field.as_u64().map(|v| v as i64)))
        .unwrap_or(0)
}
