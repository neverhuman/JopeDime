use super::*;

pub(super) fn parse_nodes(scq: &Value) -> Vec<NodeRow> {
    let mut rows = scq
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            let node = string_at(value, &["node"])?;
            let up = bool_at(value, &["up"]).unwrap_or(false);
            let effective_cores = u64_at(value, &["effective_cores"])
                .unwrap_or_else(|| u64_at(value, &["cores"]).unwrap_or(0));
            let multiplier = f64_at(value, &["cpu_job_slot_multiplier"]).unwrap_or(1.0);
            let slot_target = u64_at(value, &["job_slot_target"])
                .or_else(|| u64_at(value, &["slot_target"]))
                .unwrap_or_else(|| (effective_cores as f64 * multiplier).floor() as u64);
            Some(NodeRow {
                node: node.to_owned(),
                state: if up { "up" } else { "down" }.to_owned(),
                health: string_at(value, &["health"])
                    .unwrap_or("unknown")
                    .to_owned(),
                physical_cores: u64_at(value, &["cores"]).unwrap_or(0),
                effective_cores,
                multiplier,
                slot_target,
                running_slots: u64_at(value, &["running_job_slots"])
                    .or_else(|| u64_at(value, &["running_jobs"]))
                    .unwrap_or(0),
                cpu_busy_pct: f64_at(value, &["cpu_busy_pct"]),
                psi_cpu10: f64_at(value, &["psi_cpu10"]),
                psi_mem10: f64_at(value, &["psi_mem10"]),
                psi_io10: f64_at(value, &["psi_io10"]),
                memory_available_gb: u64_at(value, &["mem_avail_gb"]).unwrap_or(0),
                memory_total_gb: u64_at(value, &["mem_total_gb"]).unwrap_or(0),
                disk_use_pct: f64_at(value, &["disk_use_pct"]),
            })
        })
        .collect::<Vec<_>>();
    let present = rows
        .iter()
        .map(|row| row.node.as_str())
        .collect::<BTreeSet<_>>();
    let missing = EXPECTED_NODES
        .iter()
        .filter(|node| !present.contains(**node))
        .map(|node| NodeRow {
            node: (*node).to_owned(),
            state: "absent".to_owned(),
            health: "no SCQ status".to_owned(),
            ..NodeRow::default()
        })
        .collect::<Vec<_>>();
    rows.extend(missing);
    rows.sort_by(|left, right| left.node.cmp(&right.node));
    rows
}

pub(super) fn aggregate_cluster(scq: &Value, nodes: &[NodeRow]) -> ClusterSummary {
    let active = nodes
        .iter()
        .filter(|node| node.state == "up")
        .collect::<Vec<_>>();
    let weighted_cores = active
        .iter()
        .filter(|node| node.cpu_busy_pct.is_some())
        .map(|node| node.physical_cores)
        .sum::<u64>();
    let cpu_busy_pct = if weighted_cores == 0 {
        f64_at(scq, &["counters", "cpu_busy_pct"]).unwrap_or(0.0)
    } else {
        active
            .iter()
            .map(|node| node.cpu_busy_pct.unwrap_or(0.0) * node.physical_cores as f64)
            .sum::<f64>()
            / weighted_cores as f64
    };
    ClusterSummary {
        cpu_busy_pct,
        psi_cpu10: maximum(active.iter().filter_map(|node| node.psi_cpu10)),
        psi_mem10: maximum(active.iter().filter_map(|node| node.psi_mem10)),
        psi_io10: maximum(active.iter().filter_map(|node| node.psi_io10)),
        physical_cores: active.iter().map(|node| node.physical_cores).sum(),
        effective_cores: active.iter().map(|node| node.effective_cores).sum(),
        slot_target: active.iter().map(|node| node.slot_target).sum(),
        running_slots: active.iter().map(|node| node.running_slots).sum(),
        memory_available_gb: active.iter().map(|node| node.memory_available_gb).sum(),
        nodes_up: active.len(),
        nodes_total: nodes.iter().filter(|node| node.state != "absent").count(),
        jobs_failed: u64_at(scq, &["counters", "jobs_failed"]).unwrap_or(0),
        throughput_per_minute: u64_at(scq, &["counters", "throughput_per_min"]).unwrap_or(0),
    }
}

pub(super) fn parse_families(scq: &Value) -> Vec<FamilyRow> {
    let mut jobs_by_family: BTreeMap<String, FamilyRow> = BTreeMap::new();
    if let Some(jobs) = scq.get("jobs").and_then(Value::as_array) {
        for job in jobs {
            let family = string_at(job, &["project"]).unwrap_or("unknown");
            let state = string_at(job, &["state"]).unwrap_or("unknown");
            let row = jobs_by_family
                .entry(family.to_owned())
                .or_insert_with(|| FamilyRow {
                    family: family.to_owned(),
                    ..FamilyRow::default()
                });
            match state {
                "queued" => {
                    row.queued += 1;
                    if let Some(timestamp) = string_at(job, &["queue_ts"]) {
                        if row
                            .oldest_queue_ts
                            .as_deref()
                            .is_none_or(|prior| timestamp < prior)
                        {
                            row.oldest_queue_ts = Some(timestamp.to_owned());
                        }
                    }
                }
                "starting" => row.starting += 1,
                "dispatching" => row.dispatching += 1,
                "running" => row.running += 1,
                "held" => row.held += 1,
                _ => {}
            }
        }
    }
    if let Some(families) = scq.get("families").and_then(Value::as_array) {
        for family in families {
            let Some(name) = string_at(family, &["family"]) else {
                continue;
            };
            let row = jobs_by_family
                .entry(name.to_owned())
                .or_insert_with(|| FamilyRow {
                    family: name.to_owned(),
                    ..FamilyRow::default()
                });
            row.weight = f64_at(family, &["effective_weight"]).unwrap_or(1.0);
            row.queued = u64_at(family, &["queued_jobs"]).unwrap_or(row.queued);
            row.running = u64_at(family, &["running_jobs"]).unwrap_or(row.running);
            row.held = u64_at(family, &["held_jobs"]).unwrap_or(row.held);
        }
    }
    let mut rows = jobs_by_family.into_values().collect::<Vec<_>>();
    rows.retain(|row| {
        row.queued + row.starting + row.dispatching + row.running + row.held > 0
            || row.weight >= 10.0
            || row.family.starts_with("jope")
    });
    rows.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| active_jobs(right).cmp(&active_jobs(left)))
            .then_with(|| left.family.cmp(&right.family))
    });
    rows
}
