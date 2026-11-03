use super::*;

/// Headless verification path. Drives the exact collectors the TUI renders from
/// and prints what the DL charts would plot, so "the plot has data" is a
/// checkable claim rather than a screenshot.
pub(super) fn dump_dl_json(args: &Args) -> Result<()> {
    let raw = collect_sources(args);
    let mut slow = SlowState::new();
    collect_slow_sources(args, &raw, &mut slow);
    let dashboard = tui_dashboard::build_dashboard(&raw, &slow.view());
    let series = continuous_series(&dashboard.dl_history);
    let generations = dashboard
        .dl_history
        .runs
        .iter()
        .map(|run| {
            json!({
                "run": run.run_label,
                "short": run.short_label,
                "live": run.live,
                "epochs": run.points.len(),
                "updated_at_ms": run.updated_at_ms as u64,
                "first_val_loss": run.points.first().map(|point| point.val_loss),
                "last_val_loss": run.points.last().map(|point| point.val_loss),
                "last_val_pairwise": run.points.last().and_then(|point| point.val_pairwise),
                "last_conditional_advantage":
                    run.points.last().and_then(|point| point.val_conditional_advantage),
            })
        })
        .collect::<Vec<_>>();
    let live_trainers = dashboard
        .trainers
        .iter()
        .filter(|trainer| trainer.is_trainer)
        .map(|trainer| {
            let early_stopped = trainer.early_stopped;
            json!({
                "job_id": trainer.job_id,
                "node": trainer.node,
                "phase": trainer.phase,
                "epoch": trainer.epoch,
                "epochs": trainer.epochs,
                "pct": trainer.pct,
                "running_loss": trainer.running_loss,
                "history_points": trainer.history.len(),
                "early_stopped": early_stopped,
                "progress_at_ms": trainer.progress_at_ms as u64,
            })
        })
        .collect::<Vec<_>>();
    let funnel = dashboard
        .funnel
        .rows
        .iter()
        .map(|row| {
            json!({
                "task": row.task,
                "registry": row.registry,
                "encoded": row.encoded,
                "active": row.active,
                "in_play": row.in_play,
                "queued": row.queued,
                "encoding": row.encoding,
                "sealed": row.sealed,
                "locked": row.locked(),
            })
        })
        .collect::<Vec<_>>();
    let active_total: u64 = dashboard.funnel.rows.iter().map(|row| row.active).sum();
    let payload = json!({
        "schema": "JopeTuiDlSeries.v1",
        "collected_at_ms": dashboard.collected_at_ms as u64,
        "generations": generations,
        "live_trainers": live_trainers,
        "gpus": dashboard.gpus.iter().map(|gpu| {
            let temp_c = gpu.temp_c;
            let power_limit_w = gpu.power_limit_w;
            let throttled = gpu.throttled;
            json!({
                "node": gpu.node,
                "owner_job": gpu.owner_job,
                "util_pct": gpu.util_pct,
                "mem_used_mb": gpu.mem_used_mb,
                "mem_total_mb": gpu.mem_total_mb,
                "temp_c": temp_c,
                "power_w": gpu.power_w,
                "power_limit_w": power_limit_w,
                "throttled": throttled,
            })
        }).collect::<Vec<_>>(),
        "accepted_task_fronts": dashboard.accepted_tasks.iter().map(|front| json!({
            "task": front.task,
            "checkpoint_sha256": front.checkpoint_sha256,
            "generation": front.generation,
            "updated_at_ms": front.updated_at_ms as u64,
        })).collect::<Vec<_>>(),
        "watchdog": dashboard.watchdog.as_ref().map(|watchdog| {
            let last_evidence_watermark = watchdog.last_evidence_watermark;
            json!({
                "action": watchdog.action,
                "checkpoint_sha256": watchdog.checkpoint_sha256,
                "snapshot_watermark": watchdog.snapshot_watermark,
                "labels": watchdog.labels,
                "pending_unfrozen_labels": watchdog.pending_unfrozen_labels,
                "last_evidence_watermark": last_evidence_watermark,
                "active_trainers": watchdog.active_trainers,
                "generated_at_ms": watchdog.generated_at_ms as u64,
            })
        }),
        "continuous": {
            "total_epochs": series.total_epochs,
            "marks": series
                .marks
                .iter()
                .map(|(x, label, live)| json!({"x": x, "label": label, "live": live}))
                .collect::<Vec<_>>(),
            "train": series.train,
            "val": series.val,
            "pairwise": series.pairwise,
            "ablation": series.ablation,
        },
        "generations_panel": dashboard
            .generations
            .iter()
            .map(|(label, pairwise)| json!({"label": label, "val_pairwise": pairwise}))
            .collect::<Vec<_>>(),
        "funnel": funnel,
        "funnel_source": dashboard.funnel.source,
        "catalog_active_total": active_total,
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(())
}

pub(super) fn dump_unified_json(args: &Args) -> Result<()> {
    let raw = collect_sources(args);
    let mut slow = SlowState::new();
    collect_slow_sources(args, &raw, &mut slow);
    let dashboard = tui_dashboard::build_dashboard(&raw, &slow.view());
    let payload = json!({
        "schema": "JopeDimeUnifiedCockpit.v1",
        "collected_at_ms": dashboard.collected_at_ms as u64,
        "state": dashboard.top_state,
        "blocking_banner": tui_dashboard::blocked_banner(&dashboard),
        "why_idle": dashboard.why_idle,
        "last_advanced_at_ms": dashboard.campaign.last_advanced_at_ms as u64,
        "campaign": {
            "robust_wins": dashboard.campaign.robust_wins,
            "robust_target": dashboard.campaign.robust_target,
            "robust_denominator": dashboard.campaign.robust_denominator,
            "stage": dashboard.campaign.stage,
            "gate_state": dashboard.campaign.gate_state,
            "current_rung": dashboard.campaign.current_rung,
            "trajectories": [dashboard.campaign.trajectories_complete, dashboard.campaign.trajectories_total],
            "evaluations": [dashboard.campaign.evaluations_used, dashboard.campaign.evaluations_cap],
            "active_jobs": dashboard.campaign.active_jobs,
            "seed_agreement": dashboard.campaign.seed_agreement,
            "license_track": dashboard.campaign.license_track,
            "test_opening_state": dashboard.campaign.test_opening_state,
            "blockers": dashboard.campaign.blockers,
        },
        "flywheel": {
            "queue": {
                "target": dashboard.queue.target,
                "depth": dashboard.queue.useful_depth,
                "ready_reserve": dashboard.wave.ready_manifests,
                "running": dashboard.queue.running,
                "throughput_per_minute": dashboard.cluster.throughput_per_minute,
            },
            "supply_circuit_open": dashboard.supply_circuit_open,
            "foreign_cancellations": dashboard.foreign_cancellations,
            "trainer": {
                "job": dashboard.learning.trainer,
                "state": dashboard.learning.trainer_state,
                "labels": dashboard.learning.labels,
                "frozen_watermark": dashboard.learning.frozen_watermark,
                "checkpoint": dashboard.learning.checkpoint,
            },
            "nodes": dashboard.nodes.iter().map(|node| json!({
                "node": node.node,
                "allocated_slots": node.running_slots,
                "slot_target": node.slot_target,
                "cpu_busy_pct": node.cpu_busy_pct,
            })).collect::<Vec<_>>(),
            "gpus": dashboard.gpus.iter().map(|gpu| json!({
                "node": gpu.node,
                "role": if gpu.role.is_empty() { Value::Null } else { json!(gpu.role) },
                "owner": gpu.owner_job,
                "attempt": (gpu.attempt > 0).then_some(gpu.attempt),
                "util_pct": gpu.util_pct,
                "temp_c": gpu.temp_c,
                "power_w": gpu.power_w,
                "power_limit_w": gpu.power_limit_w,
                "throttled": gpu.throttled,
                "status": gpu.status,
                "reason": if gpu.admission_reason.is_empty() {
                    gpu.owner_job.is_none().then_some(dashboard.why_idle.as_str())
                } else {
                    Some(gpu.admission_reason.as_str())
                },
            })).collect::<Vec<_>>(),
        },
        "sources": [dashboard.sources_healthy, dashboard.sources_total],
    });
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}

pub(super) fn run_terminal(shared: &Arc<RwLock<Dashboard>>, refresh_ms: u64) -> Result<()> {
    let _session = TerminalSession::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    let cadence = Duration::from_millis(refresh_ms);

    let mut ui = UiState::default();
    loop {
        let snapshot = match shared.read() {
            Ok(value) => value.clone(),
            Err(_poisoned) => {
                let poisoned_fallback: Dashboard = Dashboard::default();
                poisoned_fallback
            }
        };
        terminal.draw(|frame| tui_render::render(frame, &snapshot, refresh_ms, ui))?;
        if event::poll(cadence)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if matches!(key.code, KeyCode::Esc | KeyCode::Char('q'))
                        || (key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL))
                    {
                        break;
                    }
                    // Arm selection is independent of the tab switch below so
                    // `n`/arrows do not fall through and reset the tab.
                    match key.code {
                        KeyCode::Char('n') | KeyCode::Right => {
                            ui.arm = ui.arm.wrapping_add(1);
                        }
                        KeyCode::Left => {
                            // Wrap without underflow; the renderer takes it modulo
                            // the live arm count, so a large value is harmless.
                            ui.arm = ui.arm.checked_sub(1).unwrap_or(usize::MAX);
                        }
                        _ => {}
                    }
                    ui.tab = match key.code {
                        KeyCode::Char('1') => Tab::Overview,
                        KeyCode::Char('2') => Tab::GpuDl,
                        KeyCode::Char('3') => Tab::Gameplay,
                        KeyCode::Char('4') | KeyCode::Char('d') => Tab::Dl,
                        KeyCode::Tab => match ui.tab {
                            Tab::Overview => Tab::GpuDl,
                            Tab::GpuDl => Tab::Gameplay,
                            Tab::Gameplay => Tab::Dl,
                            Tab::Dl => Tab::Overview,
                        },
                        KeyCode::BackTab => match ui.tab {
                            Tab::Overview => Tab::Dl,
                            Tab::GpuDl => Tab::Overview,
                            Tab::Gameplay => Tab::GpuDl,
                            Tab::Dl => Tab::Gameplay,
                        },
                        _ => ui.tab,
                    };
                }
            }
        }
    }
    Ok(())
}
