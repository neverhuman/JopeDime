use super::*;

pub(crate) fn build_dashboard(raw: &RawSources, slow: &SlowView) -> Dashboard {
    let mut dashboard = Dashboard {
        collected_at_ms: raw.collected_at_ms,
        collection_latency_ms: raw.collection_latency_ms,
        sources_healthy: raw
            .source_states
            .iter()
            .filter(|state| state.healthy)
            .count()
            + slow
                .source_states
                .iter()
                .filter(|state| state.healthy)
                .count(),
        sources_total: raw.source_states.len() + slow.source_states.len(),
        gpus: slow.gpus.clone(),
        trainers: slow.trainers.clone(),
        watchdog: slow.watchdog.clone(),
        accepted_tasks: parse_accepted_task_fronts(raw.accepted_by_task.as_ref()),
        four_task_telemetry: raw.four_task_telemetry.clone(),
        gameplay: slow.gameplay.clone(),
        gameplay_stats: slow.gameplay_stats.clone(),
        wave_progress: slow.wave_progress.clone(),
        daemon_ages: slow.daemon_ages.clone(),
        generations: slow.generations.clone(),
        seeding: slow.seeding.clone(),
        dl_history: slow.dl_history.clone(),
        funnel: slow.funnel.clone(),
        feed_fetched_at_ms: slow.feed_fetched_at_ms,
        ..Dashboard::default()
    };
    let scq = raw
        .scq
        .as_ref()
        .and_then(|value| value.get("Snapshot"))
        .unwrap_or(&Value::Null);
    dashboard.scq_epoch = u64_at(scq, &["epoch"]);
    dashboard.scq_ts = string_at(scq, &["ts"]).unwrap_or("-").to_owned();
    dashboard.nodes = parse_nodes(scq);
    dashboard.cluster = aggregate_cluster(scq, &dashboard.nodes);
    dashboard.families = parse_families(scq);
    dashboard.queue = parse_queue(raw, scq, &dashboard.families);
    dashboard.campaign = parse_campaign(raw.campaign.as_ref());
    enrich_gpu_admission(&mut dashboard.gpus, raw.gpu_admission.as_ref());
    dashboard.supply_circuit_open = raw
        .supply_breaker
        .as_ref()
        .and_then(|value| bool_at(value, &["open"]))
        .unwrap_or(false);
    dashboard.supply_circuit_error = raw
        .supply_breaker
        .as_ref()
        .and_then(|value| string_at(value, &["error"]))
        .unwrap_or("-")
        .to_owned();
    let proposer_foreign_cancellations = raw
        .proposer_state
        .as_ref()
        .and_then(|value| u64_at(value, &["foreign_cancellations"]))
        .unwrap_or(0);
    let campaign_foreign_cancellations = if dashboard
        .campaign
        .blockers
        .iter()
        .any(|blocker| blocker.contains("repeated foreign campaign cancellations"))
    {
        2
    } else {
        0
    };
    dashboard.foreign_cancellations =
        proposer_foreign_cancellations.max(campaign_foreign_cancellations);
    dashboard.learning = parse_learning(raw, scq);
    if let Some(watchdog) = dashboard.watchdog.as_ref() {
        dashboard.learning.labels = watchdog.labels;
        dashboard.learning.frozen_watermark = watchdog.snapshot_watermark;
        if watchdog.checkpoint_sha256 != "-" {
            dashboard.learning.checkpoint = watchdog.checkpoint_sha256.clone();
        }
        if let Some(trainer) = dashboard.trainers.iter().find(|trainer| trainer.is_trainer) {
            dashboard.learning.trainer = trainer.job_id.clone();
            dashboard.learning.trainer_state = if trainer.phase == "-" {
                "running".to_owned()
            } else {
                trainer.phase.clone()
            };
        } else if let Some(trainer) = watchdog.active_trainers.first() {
            dashboard.learning.trainer = trainer.clone();
            dashboard.learning.trainer_state = watchdog.action.clone();
        }
    }
    dashboard.wave = parse_wave(raw);
    let active_job = |job_id: &str| {
        scq.get("jobs")
            .and_then(Value::as_array)
            .is_some_and(|jobs| {
                jobs.iter().any(|job| {
                    string_at(job, &["id"]) == Some(job_id)
                        && string_at(job, &["state"]).is_some_and(|state| {
                            matches!(
                                state,
                                "queued"
                                    | "starting"
                                    | "dispatching"
                                    | "running"
                                    | "held"
                                    | "preempting"
                            )
                        })
                })
            })
    };
    dashboard.stranded_trainer = raw
        .trainer_controller
        .as_ref()
        .and_then(|value| string_at(value, &["action"]))
        .is_some_and(|action| {
            action.contains("pending")
                || action.contains("failed")
                || action.contains("backoff")
                || action.contains("circuit_open")
        })
        && dashboard.learning.trainer != "-"
        && !active_job(&dashboard.learning.trainer);
    dashboard.stranded_proposer = raw.proposer_state.as_ref().is_some_and(|value| {
        string_at(value, &["stage"]) == Some("submit_cohort_proposer")
            && string_at(value, &["proposal_job_id"]).is_some_and(|job| !active_job(job))
            && !bool_at(value, &["retry_pending_submission"]).unwrap_or(false)
    });
    dashboard.priority_drift = dashboard
        .campaign
        .blockers
        .iter()
        .any(|blocker| blocker.contains("priority"));
    dashboard.stale_critical_source = raw.source_states.iter().any(|source| {
        !source.healthy
            && matches!(
                source.name.as_str(),
                "scq"
                    | "queue-v3"
                    | "supply"
                    | "supply-breaker"
                    | "trainer-controller"
                    | "utilization-supervisor"
                    | "budget-campaign"
            )
    });
    dashboard.events = parse_events(scq, &raw.local_events);
    dashboard.failures = parse_failures(scq);
    dashboard.alerts = collect_alerts(raw, scq, &dashboard);
    // Alarm-FILE rows go first, ahead of the alerts this TUI derives itself.
    // render_alerts() only draws the first five, and it was drawing five
    // derived rows while a critical `cluster_idle` row sat below the fold —
    // which is the same failure mode as not raising the alarm at all.
    // ALARM_SOURCES already orders utilization ahead of health.
    for (index, alarm) in slow.health_alarms.iter().enumerate() {
        dashboard.alerts.insert(index, alarm.clone());
    }
    let now = raw.collected_at_ms;
    if slow.newest_receipt_at_ms > 0 && now.saturating_sub(slow.newest_receipt_at_ms) > 300_000 {
        dashboard.alerts.push(format!(
            "gameplay feed stalled: newest receipt {}s old",
            now.saturating_sub(slow.newest_receipt_at_ms) / 1_000
        ));
    }
    for trainer in &slow.trainers {
        if trainer.progress_at_ms > 0 && now.saturating_sub(trainer.progress_at_ms) > 600_000 {
            dashboard.alerts.push(format!(
                "{} progress {}s stale while running",
                short(&trainer.job_id, 32),
                now.saturating_sub(trainer.progress_at_ms) / 1_000
            ));
        }
    }
    dashboard.why_idle = why_idle_chain(&dashboard);
    dashboard.top_state = cockpit_state(&dashboard).to_owned();
    dashboard
}
