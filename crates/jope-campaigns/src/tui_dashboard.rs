use super::*;

pub(super) fn build_dashboard(raw: &RawSources, slow: &SlowView) -> Dashboard {
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

pub(super) fn parse_campaign(value: Option<&Value>) -> CampaignSummary {
    let value = value.unwrap_or(&Value::Null);
    let rung_counts = value.get("rung_counts").unwrap_or(&Value::Null);
    let active_progress = value
        .get("active_jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| jobs.first())
        .and_then(|job| u64_at(job, &["progress_evaluations"]));
    let current_rung = active_progress
        .map(|progress| match progress {
            0..=999 => "1000",
            1_000..=1_999 => "2000",
            2_000..=4_999 => "5000",
            _ => "10000",
        })
        .or_else(|| {
            ["10000", "5000", "2000", "1000"]
                .into_iter()
                .find(|rung| u64_at(rung_counts, &[*rung]).unwrap_or(0) > 0)
        })
        .unwrap_or("-")
        .to_owned();
    CampaignSummary {
        stage: string_at(value, &["stage"])
            .unwrap_or("unavailable")
            .to_owned(),
        gate_state: string_at(value, &["gate_state"])
            .unwrap_or("launch_gates_blocked")
            .to_owned(),
        robust_wins: u64_at(value, &["robust_wins"]).unwrap_or(0),
        robust_target: u64_at(value, &["robust_wins_target"]).unwrap_or(7),
        robust_denominator: u64_at(value, &["robust_wins_denominator"]).unwrap_or(22),
        trajectories_complete: u64_at(value, &["completed_trajectory_count"]).unwrap_or(0),
        trajectories_total: u64_at(value, &["trajectory_count"]).unwrap_or(0),
        evaluations_used: u64_at(value, &["evaluations_used"]).unwrap_or(0),
        evaluations_cap: u64_at(value, &["evaluations_cap"]).unwrap_or(200_000),
        current_rung,
        seed_agreement: string_at(value, &["seed_agreement"])
            .unwrap_or("unavailable")
            .to_owned(),
        active_jobs: value
            .get("active_jobs")
            .and_then(Value::as_array)
            .map_or(0, |jobs| jobs.len() as u64),
        license_track: string_at(value, &["license_track"])
            .unwrap_or("separate")
            .to_owned(),
        test_opening_state: string_at(value, &["test_opening_state"])
            .unwrap_or("sealed")
            .to_owned(),
        last_advanced_at_ms: u64_at(value, &["last_advanced_at_ms"])
            .map(u128::from)
            .unwrap_or(0),
        blockers: value
            .get("blockers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
    }
}

pub(super) fn infrastructure_attempt_number(job_identity: &str) -> u64 {
    job_identity
        .rsplit_once("-infra")
        .and_then(|(_, suffix)| suffix.parse::<u64>().ok())
        .unwrap_or(1)
}

pub(super) fn enrich_gpu_admission(gpus: &mut [GpuRow], value: Option<&Value>) {
    let Some(states) = value
        .and_then(|value| value.get("states"))
        .and_then(Value::as_object)
    else {
        return;
    };
    for (role, state) in states {
        let Some(node) = string_at(state, &["target_node"]) else {
            continue;
        };
        let job_identity = string_at(state, &["immutable_job_identity"]).unwrap_or("-");
        let stage = string_at(state, &["stage"]).unwrap_or("unknown");
        let reason = string_at(state, &["dependency_failure"])
            .or_else(|| string_at(state, &["last_capacity_backoff"]))
            .unwrap_or(stage);
        if let Some(gpu) = gpus.iter_mut().find(|gpu| gpu.node == node) {
            gpu.role = role.clone();
            gpu.attempt = infrastructure_attempt_number(job_identity);
            gpu.admission_reason = reason.to_owned();
        }
    }
}

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

pub(super) fn parse_queue(raw: &RawSources, scq: &Value, families: &[FamilyRow]) -> QueueSummary {
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let freshness = raw.freshness.as_ref().unwrap_or(&Value::Null);
    let counters = scq.get("counters").unwrap_or(&Value::Null);
    QueueSummary {
        queued: u64_at(counters, &["jobs_queued"])
            .unwrap_or_else(|| families.iter().map(|row| row.queued).sum()),
        starting: families.iter().map(|row| row.starting).sum(),
        dispatching: u64_at(queue, &["dispatching"])
            .unwrap_or_else(|| families.iter().map(|row| row.dispatching).sum()),
        running: u64_at(counters, &["jobs_running"])
            .or_else(|| u64_at(queue, &["running"]))
            .unwrap_or_else(|| families.iter().map(|row| row.running).sum()),
        held: u64_at(counters, &["jobs_held"])
            .or_else(|| u64_at(queue, &["held"]))
            .unwrap_or_else(|| families.iter().map(|row| row.held).sum()),
        useful_depth: u64_at(queue, &["queued_starting"]).unwrap_or(0),
        target: u64_at(queue, &["queued_target"]).unwrap_or(0),
        hard_cap: u64_at(queue, &["queued_hard_cap"]).unwrap_or(0),
        maximum_age_seconds: f64_at(queue, &["queue_maximum_age_seconds"]).unwrap_or(0.0),
        epoch: string_at(queue, &["checkpoint_epoch"])
            .or_else(|| string_at(freshness, &["checkpoint_epoch"]))
            .unwrap_or("-")
            .to_owned(),
        stale_epochs: u64_at(freshness, &["stale_epoch_count"]).unwrap_or(0),
        stale_held: u64_at(freshness, &["stale_held"]).unwrap_or(0),
        admission_allowed: bool_at(queue, &["admission_allowed"]),
        mode: string_at(queue, &["mode"]).unwrap_or("degraded").to_owned(),
    }
}

pub(super) fn parse_learning(raw: &RawSources, scq: &Value) -> LearningSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let trainer = status.get("trainer").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let fallback = raw.trainer_fallback.as_ref().unwrap_or(&Value::Null);
    let proposal = raw.proposal_summary.as_ref().unwrap_or(&Value::Null);
    let flywheel = raw.flywheel.as_ref().unwrap_or(&Value::Null);
    let trainer_id = trainer
        .get("active_trainers")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_str)
        .or_else(|| string_at(trainer, &["trainer"]))
        .or_else(|| string_at(fallback, &["job_id"]))
        .unwrap_or("-");
    let trainer_state = scq
        .get("jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| {
            jobs.iter().find(|job| {
                string_at(job, &["id"]).is_some_and(|candidate| candidate == trainer_id)
            })
        })
        .and_then(|job| string_at(job, &["state"]))
        .unwrap_or("not in SCQ");
    let promotion = status.get("promotion").unwrap_or(&Value::Null);
    LearningSummary {
        checkpoint: string_at(
            raw.queue.as_ref().unwrap_or(&Value::Null),
            &["active_checkpoint_sha256"],
        )
        .or_else(|| {
            string_at(
                raw.freshness.as_ref().unwrap_or(&Value::Null),
                &["checkpoint_sha256"],
            )
        })
        .or_else(|| string_at(proposal, &["model_checkpoint_sha256"]))
        .unwrap_or("-")
        .to_owned(),
        trainer: trainer_id.to_owned(),
        trainer_state: trainer_state.to_owned(),
        labels: u64_at(trainer, &["labels"])
            .or_else(|| u64_at(status, &["adaptive_fitness_watermark"]))
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        frozen_watermark: u64_at(flywheel, &["last_frozen_watermark"]).unwrap_or(0),
        internal_promotion: promotion_label(promotion.get("internal_learning")),
        milestone_promotion: promotion_label(promotion.get("milestone")),
        proposal_regression: u64_at(proposal, &["counts", "regression"]).unwrap_or(0),
        proposal_binary: u64_at(proposal, &["counts", "binary"]).unwrap_or(0),
        proposal_validity: f64_at(proposal, &["proposal_validity"]),
        consecutive_rejections: u64_at(status, &["evolution", "consecutive_rejections"])
            .or_else(|| u64_at(flywheel, &["consecutive_rejections"]))
            .unwrap_or(0),
    }
}

pub(super) fn parse_wave(raw: &RawSources) -> WaveSummary {
    let status = raw.status.as_ref().unwrap_or(&Value::Null);
    let queue = raw.queue.as_ref().unwrap_or(&Value::Null);
    let supply = raw.supply.as_ref().unwrap_or(&Value::Null);
    let catalog = status.get("catalog").unwrap_or(&Value::Null);
    let sealed = status.get("sealed_audit").unwrap_or(&Value::Null);
    let ingestion = status.get("ingestion").unwrap_or(&Value::Null);
    let pairs = status.get("pair_registry").unwrap_or(&Value::Null);
    WaveSummary {
        active_wave: string_at(queue, &["newest_micro_wave"])
            .or_else(|| string_at(status, &["active_micro_cohort", "wave"]))
            .unwrap_or("-")
            .to_owned(),
        admission_waves: queue
            .get("admission_waves")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        unclaimed_manifests: u64_at(queue, &["unclaimed_current_checkpoint_manifests"])
            .or_else(|| u64_at(queue, &["manifests_unclaimed"]))
            .unwrap_or(0),
        ready_manifests: u64_at(supply, &["ready_manifests"]).unwrap_or(0),
        awaiting_keeper_ack: string_at(supply, &["awaiting_keeper_ack"])
            .unwrap_or("-")
            .to_owned(),
        adaptive_regression: u64_at(catalog, &["task_counts", "regression"]).unwrap_or(0),
        adaptive_binary: u64_at(catalog, &["task_counts", "binary"]).unwrap_or(0),
        sealed_isolated: bool_at(sealed, &["isolated_from_training"]),
        sealed_scored: u64_at(sealed, &["states", "scored"]).unwrap_or(0),
        corpus_rows: u64_at(status, &["adaptive_fitness_watermark"])
            .or_else(|| u64_at(ingestion, &["exact_rows_total"]))
            .unwrap_or(0),
        strict_wins: u64_at(ingestion, &["strict_wins"]).unwrap_or(0),
        pair_count: u64_at(pairs, &["pairs"]).unwrap_or(0),
        duplicate_pairs_prevented: u64_at(pairs, &["duplicates_prevented"]).unwrap_or(0),
        evidence_bytes: raw.evidence_bytes.unwrap_or(0),
    }
}

pub(super) fn parse_events(scq: &Value, local: &[Value]) -> Vec<String> {
    let mut events = scq
        .get("events")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .rev()
        .take(8)
        .map(format_event)
        .collect::<Vec<_>>();
    events.extend(local.iter().take(6).map(format_event));
    events.truncate(12);
    events
}

pub(super) fn parse_failures(scq: &Value) -> Vec<String> {
    scq.get("jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|job| string_at(job, &["state"]) == Some("failed"))
        .rev()
        .take(8)
        .map(|job| {
            format!(
                "{} {} {}",
                short(string_at(job, &["id"]).unwrap_or("-"), 24),
                string_at(job, &["failure_class"]).unwrap_or("failed"),
                short(string_at(job, &["message"]).unwrap_or(""), 48)
            )
        })
        .collect()
}

pub(super) fn collect_alerts(raw: &RawSources, scq: &Value, dashboard: &Dashboard) -> Vec<String> {
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

pub(super) fn add_stale_source_alert(
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
pub(super) fn blocked_banner(dashboard: &Dashboard) -> Option<String> {
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

pub(super) fn why_idle_chain(dashboard: &Dashboard) -> String {
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

pub(super) fn cockpit_state(dashboard: &Dashboard) -> &'static str {
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

#[path = "../tui_render.rs"]
mod tui_render;
