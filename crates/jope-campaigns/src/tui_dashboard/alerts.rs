use super::*;

pub(crate) fn collect_alerts(raw: &RawSources, scq: &Value, dashboard: &Dashboard) -> Vec<String> {
    let mut alerts = BTreeSet::new();
    for source in raw.source_states.iter().filter(|source| !source.healthy) {
        alerts.insert(format!(
            "source {}: {}",
            source.name,
            short(&source.note, 80)
        ));
    }
    add_stale_source_alert(
        &mut alerts,
        "status",
        raw.status.as_ref(),
        "updated_at_ms",
        raw.collected_at_ms,
        120_000,
    );
    add_stale_source_alert(
        &mut alerts,
        "queue-v3",
        raw.queue.as_ref(),
        "generated_at_ms",
        raw.collected_at_ms,
        15_000,
    );
    add_stale_source_alert(
        &mut alerts,
        "supply",
        raw.supply.as_ref(),
        "generated_at_ms",
        raw.collected_at_ms,
        30_000,
    );
    if raw.collection_latency_ms >= 300 {
        alerts.insert(format!(
            "refresh latency {}ms exceeds 300ms target",
            raw.collection_latency_ms
        ));
    }
    if dashboard.cluster.cpu_busy_pct >= CPU_HARD_GUARD_PCT {
        alerts.insert(format!(
            "cluster CPU {:.1}% reached 95% hard guard",
            dashboard.cluster.cpu_busy_pct
        ));
    } else if dashboard.cluster.cpu_busy_pct >= CPU_TARGET_PCT {
        alerts.insert(format!(
            "cluster CPU {:.1}% exceeds 80% target",
            dashboard.cluster.cpu_busy_pct
        ));
    }
    for node in &dashboard.nodes {
        if node
            .cpu_busy_pct
            .is_some_and(|cpu| cpu >= CPU_HARD_GUARD_PCT)
        {
            alerts.insert(format!(
                "{} CPU hard guard at {:.1}%",
                node.node,
                node.cpu_busy_pct.unwrap()
            ));
        }
        if node.node == "xbabe5" && node.state != "up" {
            alerts.insert(format!("xbabe5 {}", node.state));
        }
    }
    if dashboard.queue.hard_cap > 0 && dashboard.queue.useful_depth > dashboard.queue.hard_cap {
        alerts.insert(format!(
            "fresh queue {} exceeds hard cap {}",
            dashboard.queue.useful_depth, dashboard.queue.hard_cap
        ));
    }
    if dashboard.queue.maximum_age_seconds > 1_800.0 {
        alerts.insert(format!(
            "queued work age {:.0}s exceeds 30m",
            dashboard.queue.maximum_age_seconds
        ));
    }
    if dashboard.queue.stale_epochs > 0 || dashboard.queue.stale_held > 0 {
        alerts.insert(format!(
            "stale epochs {} / held {}",
            dashboard.queue.stale_epochs, dashboard.queue.stale_held
        ));
    }
    if dashboard.queue.admission_allowed == Some(false) {
        alerts.insert("fresh admission is blocked".to_owned());
    }
    if let Some(values) = scq.get("alerts").and_then(Value::as_array) {
        for value in values.iter().filter_map(Value::as_str) {
            alerts.insert(format!("SCQ: {value}"));
        }
    }
    if let Some(values) = raw
        .status
        .as_ref()
        .and_then(|value| value.get("alerts"))
        .and_then(Value::as_array)
    {
        for value in values.iter().filter_map(Value::as_str) {
            alerts.insert(format!("JOPE: {value}"));
        }
    }
    alerts.into_iter().collect()
}

pub(crate) fn add_stale_source_alert(
    alerts: &mut BTreeSet<String>,
    name: &str,
    value: Option<&Value>,
    timestamp_key: &str,
    now: u128,
    maximum_age_ms: u128,
) {
    let Some(timestamp) = value
        .and_then(|value| u64_at(value, &[timestamp_key]))
        .map(u128::from)
    else {
        return;
    };
    let age = now.saturating_sub(timestamp);
    if age > maximum_age_ms {
        alerts.insert(format!("{name} status stale by {}s", age / 1_000));
    }
}

/// The one condition an operator must never be able to miss: the scheduler is
/// blocked or throttled and our work is not flowing. Requested 2026-08-01
/// after a cpu-overload-guard episode stranded 996 held eval jobs for ~3 h
/// with only a quiet queue panel to show for it. Returns the most severe
/// active condition, or None when work flows normally.
pub(crate) fn blocked_banner(dashboard: &Dashboard) -> Option<String> {
    let queue = &dashboard.queue;
    let below_floor_without_ready = queue.target > 0
        && queue.useful_depth < queue.target
        && dashboard.wave.ready_manifests == 0;
    if dashboard.supply_circuit_open && below_floor_without_ready {
        return Some(format!(
            "DIME BLOCKED — SUPPLY CIRCUIT OPEN ({}) + queue {}/{} with ZERO ready work",
            short(&dashboard.supply_circuit_error, 72),
            queue.useful_depth,
            queue.target
        ));
    }
    if dashboard.supply_circuit_open {
        return Some(format!(
            "DIME BLOCKED — SUPPLY CIRCUIT OPEN: {}",
            short(&dashboard.supply_circuit_error, 100)
        ));
    }
    if dashboard.stranded_trainer || dashboard.stranded_proposer {
        return Some(format!(
            "DIME BLOCKED — stranded frozen intent: {}{}",
            if dashboard.stranded_trainer {
                "TRAINER"
            } else {
                ""
            },
            if dashboard.stranded_proposer {
                " PROPOSER"
            } else {
                ""
            }
        ));
    }
    if dashboard.foreign_cancellations >= 2 {
        return Some(format!(
            "DIME BLOCKED — {} repeated FOREIGN proposer cancellations (scientific retries unchanged)",
            dashboard.foreign_cancellations
        ));
    }
    if dashboard.priority_drift {
        return Some(
            "DIME BLOCKED — project priority drift; launch remains fail-closed".to_owned(),
        );
    }
    if dashboard.stale_critical_source {
        return Some(
            "DIME BLOCKED — one or more critical canonical status sources are stale or absent"
                .to_owned(),
        );
    }
    if below_floor_without_ready {
        return Some(format!(
            "DIME STARVED — queue {}/{} below floor with ZERO ready frozen work",
            queue.useful_depth, queue.target
        ));
    }
    // Sticky guard-holds: lots held, almost nothing of ours running.
    if queue.held >= 32 && queue.running < queue.held / 4 {
        return Some(format!(
            "SCQ THROTTLED — {} jobs HELD, only {} running (guard-held work is NOT draining)",
            queue.held, queue.running
        ));
    }
    if queue.admission_allowed == Some(false) {
        return Some("SCQ BLOCKED — fresh admission is blocked by the queue keeper".to_owned());
    }
    let waiting = queue.queued + queue.starting + queue.dispatching;
    if queue.running == 0 && waiting > 0 {
        return Some(format!(
            "SCQ STALLED — {waiting} jobs waiting, ZERO running"
        ));
    }
    let dime_waiting = dashboard
        .families
        .iter()
        .filter(|family| family.family.starts_with("jope"))
        .map(|family| family.queued + family.starting + family.dispatching)
        .sum::<u64>();
    let dime_running = dashboard
        .families
        .iter()
        .filter(|family| family.family.starts_with("jope"))
        .map(|family| family.running)
        .sum::<u64>();
    if dime_waiting > 0 && dime_running == 0 {
        return Some(format!(
            "DIME STARVED — {dime_waiting} runnable DIME jobs with ZERO running"
        ));
    }
    if dime_waiting > 0 {
        if let Some(node) = dashboard
            .nodes
            .iter()
            .find(|node| node.state == "up" && node.cpu_busy_pct.is_some_and(|value| value < 25.0))
        {
            return Some(format!(
                "DIME STARVED — runnable backlog while {} remains at {:.1}% CPU",
                node.node,
                node.cpu_busy_pct.unwrap_or(0.0)
            ));
        }
    }
    // Critical starvation alarms from the alarm files (the collector inserts
    // them into `alerts` as "ALARM[severity] name: ..." rows). These two are
    // the flywheel-is-not-flowing signatures the 2026-08-01 incidents proved
    // can otherwise hide below the fold of the alerts panel.
    if let Some(alert) = dashboard.alerts.iter().find(|alert| {
        alert.starts_with("ALARM[critical] proposer_gpu_starved")
            || alert.starts_with("ALARM[critical] scq_jobs_held_starving")
    }) {
        return Some(alert.clone());
    }
    // The scheduler's own alert lines (surfaced into alerts as "SCQ: ...").
    dashboard
        .alerts
        .iter()
        .find(|alert| alert.starts_with("SCQ: "))
        .map(|alert| format!("{alert} — scheduler is degraded"))
}

pub(crate) fn why_idle_chain(dashboard: &Dashboard) -> String {
    if let Some(blocker) = blocked_banner(dashboard) {
        return format!("WHY IDLE → {blocker}");
    }
    if dashboard.campaign.gate_state != "runnable"
        && dashboard.campaign.gate_state != "complete"
        && dashboard.queue.useful_depth == 0
        && dashboard.wave.ready_manifests == 0
    {
        return format!(
            "WHY IDLE → NO ELIGIBLE FROZEN WORK → campaign {}",
            dashboard.campaign.gate_state
        );
    }
    if dashboard.gpus.iter().any(|gpu| gpu.owner_job.is_none())
        && dashboard.trainers.is_empty()
        && dashboard.campaign.active_jobs == 0
    {
        return "WHY IDLE → NO ELIGIBLE FROZEN WORK".to_owned();
    }
    "WHY IDLE → resources are owned or scientifically authorized work is running".to_owned()
}

pub(crate) fn cockpit_state(dashboard: &Dashboard) -> &'static str {
    if dashboard.supply_circuit_open
        || dashboard.stranded_trainer
        || dashboard.stranded_proposer
        || dashboard.foreign_cancellations >= 2
        || dashboard.priority_drift
        || dashboard.stale_critical_source
    {
        return "BLOCKED";
    }
    let dime_waiting = dashboard
        .families
        .iter()
        .filter(|family| family.family.starts_with("jope"))
        .map(|family| family.queued + family.starting + family.dispatching)
        .sum::<u64>();
    if (dashboard.queue.target > 0
        && dashboard.queue.useful_depth < dashboard.queue.target
        && dashboard.wave.ready_manifests == 0)
        || (dime_waiting > 0 && dashboard.queue.running == 0)
    {
        return "STARVED";
    }
    let advanced_recently = dashboard.campaign.last_advanced_at_ms > 0
        && dashboard
            .collected_at_ms
            .saturating_sub(dashboard.campaign.last_advanced_at_ms)
            <= 120_000;
    if advanced_recently || dashboard.cluster.throughput_per_minute > 0 {
        return "CLIMBING";
    }
    if dashboard.queue.running > 0
        || dashboard.campaign.active_jobs > 0
        || dashboard.gpus.iter().any(|gpu| gpu.owner_job.is_some())
    {
        return "RUNNING";
    }
    if dashboard.campaign.gate_state != "runnable" && dashboard.campaign.gate_state != "complete" {
        return "GATED";
    }
    "RUNNING"
}
