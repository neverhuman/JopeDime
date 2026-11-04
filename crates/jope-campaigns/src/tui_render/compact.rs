use super::*;

pub(crate) fn render_compact_gpu_dl(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let now = dashboard.collected_at_ms;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("DIME GPU/DL • {}", dashboard.top_state),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  sources {}/{}  {}",
            dashboard.sources_healthy,
            dashboard.sources_total,
            clock_string(now)
        )),
    ])];
    lines.push(Line::from(dashboard.why_idle.clone()));

    if dashboard.accepted_tasks.is_empty() {
        lines.push(Line::from("fronts: accepted-by-task unavailable"));
    } else {
        lines.extend(dashboard.accepted_tasks.iter().map(|front| {
            Line::from(vec![
                Span::styled(
                    format!("{:<10}", front.task),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "accepted {}  generation {}  age {}s",
                    short_hash(&front.checkpoint_sha256),
                    front.generation,
                    now.saturating_sub(front.updated_at_ms) / 1_000
                )),
            ])
        }));
    }

    lines.push(
        dashboard
            .watchdog
            .as_ref()
            .map(|watchdog| {
                Line::from(format!(
                    "watchdog {} • corpus {} • labels {} • gap {} • ckpt {}",
                    watchdog.action,
                    human_count(watchdog.snapshot_watermark),
                    human_count(watchdog.labels),
                    human_count(watchdog.pending_unfrozen_labels),
                    short_hash(&watchdog.checkpoint_sha256)
                ))
            })
            .unwrap_or_else(|| Line::from("watchdog unavailable")),
    );

    lines.push(Line::from(Span::styled(
        "GPUs",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.gpus.is_empty() {
        lines.push(Line::from("  no GPU telemetry"));
    } else {
        lines.extend(dashboard.gpus.iter().map(|gpu| {
            let active = gpu.owner_job.is_some();
            let role = if gpu.role.is_empty() {
                String::new()
            } else {
                format!("{}#{} ", gpu.role, gpu.attempt)
            };
            let reason = if gpu.admission_reason.is_empty() {
                gpu.owner_job
                    .as_deref()
                    .unwrap_or(dashboard.why_idle.as_str())
            } else {
                gpu.admission_reason.as_str()
            };
            Line::from(vec![
                Span::styled(
                    format!(
                        "  {} {:>3.0}% {:>5}/{:<5} MiB {:>3.0}W ",
                        gpu.node, gpu.util_pct, gpu.mem_used_mb, gpu.mem_total_mb, gpu.power_w
                    ),
                    Style::default().fg(if active {
                        Color::Green
                    } else {
                        Color::DarkGray
                    }),
                ),
                Span::raw(short(
                    &format!(
                        "{role}{} — {reason}",
                        gpu.owner_job.as_deref().unwrap_or("idle")
                    ),
                    area.width.saturating_sub(42) as usize,
                )),
            ])
        }));
    }

    lines.push(Line::from(Span::styled(
        "ACTIVE GPU WORK",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.trainers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(Color::Red),
        )));
    } else {
        for trainer in &dashboard.trainers {
            let heartbeat_age = now.saturating_sub(trainer.progress_at_ms) / 1_000;
            let role = if trainer.is_trainer { "TRAIN" } else { "GPU" };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {role:<5} "),
                    Style::default()
                        .fg(if trainer.is_trainer {
                            Color::Green
                        } else {
                            Color::Cyan
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "{} @ {}",
                    short(&trainer.job_id, area.width.saturating_sub(22) as usize),
                    trainer.node
                )),
            ]));
            if trainer.is_trainer {
                lines.push(Line::from(format!(
                    "        {} epoch {}/{} batch {}/{} ({:.1}%) loss {} hb {}s",
                    trainer.phase,
                    trainer.epoch,
                    trainer.epochs,
                    trainer.batch,
                    trainer.total_batches,
                    trainer.pct,
                    trainer
                        .running_loss
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    heartbeat_age,
                )));
            }
        }
    }

    lines.push(Line::from(format!(
        "1 overview • 2 gpu/dl • 4 charts • q quit • {}ms refresh / {}ms collect",
        refresh_ms, dashboard.collection_latency_ms
    )));
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .title(" DL CONTROL PLANE ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

pub(crate) fn render_compact(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let nodes = dashboard
        .nodes
        .iter()
        .map(|node| {
            format!(
                "{}:{} CPU {} slots {}/{}",
                node.node,
                node.state,
                node.cpu_busy_pct
                    .map(|value| format!("{value:.0}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                node.running_slots,
                node.slot_target
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let mut lines = vec![
        Line::from(format!(
            "DIME UNIFIED • {} • read-only",
            dashboard.top_state
        )),
        Line::from(format!(
            "campaign robust {}/{} of {} • {} rung {} • evals {}/{} • seeds {}",
            dashboard.campaign.robust_wins,
            dashboard.campaign.robust_target,
            dashboard.campaign.robust_denominator,
            dashboard.campaign.stage,
            dashboard.campaign.current_rung,
            dashboard.campaign.evaluations_used,
            dashboard.campaign.evaluations_cap,
            dashboard.campaign.seed_agreement,
        )),
        Line::from(dashboard.why_idle.clone()),
        Line::from(format!(
            "CPU {:.1}% (target 80 / guard 95) PSI {}/{}/{}",
            dashboard.cluster.cpu_busy_pct,
            dashboard.cluster.psi_cpu10,
            dashboard.cluster.psi_mem10,
            dashboard.cluster.psi_io10
        )),
        Line::from(nodes),
        Line::from(format!(
            "queue fresh {}/{} cap {} age {:.0}s epoch {}",
            dashboard.queue.useful_depth,
            dashboard.queue.target,
            dashboard.queue.hard_cap,
            dashboard.queue.maximum_age_seconds,
            short_epoch(&dashboard.queue.epoch)
        )),
        Line::from(format!(
            "checkpoint {} trainer {} [{}] labels {}",
            short_hash(&dashboard.learning.checkpoint),
            short(&dashboard.learning.trainer, 30),
            dashboard.learning.trainer_state,
            human_count(dashboard.learning.labels)
        )),
        Line::from(format!(
            "wave {} adaptive R{}/B{} sealed {}",
            dashboard.wave.active_wave,
            dashboard.wave.adaptive_regression,
            dashboard.wave.adaptive_binary,
            yes_no(dashboard.wave.sealed_isolated)
        )),
    ];
    for task in FOUR_TASK_ORDER {
        let row = dashboard
            .four_task_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.rows.iter().find(|row| row.task == task));
        lines.push(Line::from(row.map_or_else(
            || {
                format!(
                    "{:>2} N/A — four-task telemetry unavailable",
                    funnel_tag(task.as_str())
                )
            },
            |row| {
                format!(
                    "{:>2} fam {} primary {} error {} flow {}",
                    funnel_tag(task.as_str()),
                    telemetry_count(&row.counts.source_families),
                    telemetry_median(&row.metrics.primary_performance),
                    row.metrics
                        .native_errors
                        .get(task_metric_contract(task).native_error)
                        .map(telemetry_median)
                        .unwrap_or_else(|| "N/A".to_owned()),
                    row.valid_observations_per_minute
                        .value
                        .map(|value| format!("{value:.0}/m"))
                        .unwrap_or_else(|| "N/A".to_owned()),
                )
            },
        )));
    }
    lines.extend([
        Line::from(format!(
            "alerts: {}",
            if dashboard.alerts.is_empty() {
                "none".to_owned()
            } else {
                dashboard.alerts.join(" | ")
            }
        )),
        Line::from(format!(
            "1 overview • 2 gpu/dl • 4 charts • q/esc quit • {}ms refresh • {}ms collect",
            refresh_ms, dashboard.collection_latency_ms
        )),
    ]);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL)),
        area,
    );
}
