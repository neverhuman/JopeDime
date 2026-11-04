use super::*;

pub(crate) fn render_gpu_tab(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let trainer_rows = (dashboard.trainers.len().max(1) * 9) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3 + dashboard.gpus.len().max(1) as u16),
            Constraint::Length(4),
            Constraint::Min(trainer_rows.min(27)),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);

    let header = Row::new([
        "node",
        "gpu",
        "status/owner",
        "util",
        "mem MiB",
        "temp",
        "power W",
        "thr",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let gpu_rows = dashboard.gpus.iter().map(|gpu| {
        let color = if gpu.temp_c >= 80 {
            Color::Red
        } else if gpu.util_pct >= 1.0 {
            Color::Green
        } else {
            Color::DarkGray
        };
        let ownership = if gpu.role.is_empty() {
            gpu.owner_job.as_deref().unwrap_or(&gpu.status).to_owned()
        } else {
            format!(
                "{}#{} {} [{}]",
                gpu.role,
                gpu.attempt,
                gpu.owner_job.as_deref().unwrap_or("idle"),
                gpu.admission_reason
            )
        };
        Row::new(vec![
            Cell::from(gpu.node.clone()),
            Cell::from(format!("{}?", gpu.model_hint)),
            Cell::from(short(&ownership, 40)),
            Cell::from(format!("{:.0}%", gpu.util_pct)),
            Cell::from(format!("{}/{}", gpu.mem_used_mb, gpu.mem_total_mb)),
            Cell::from(format!("{}C", gpu.temp_c)),
            Cell::from(format!("{:.0}/{:.0}", gpu.power_w, gpu.power_limit_w)),
            Cell::from(if gpu.throttled { "YES" } else { "-" }),
        ])
        .style(Style::default().fg(color))
    });
    frame.render_widget(
        Table::new(
            gpu_rows,
            [
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Min(30),
                Constraint::Length(6),
                Constraint::Length(13),
                Constraint::Length(5),
                Constraint::Length(9),
                Constraint::Length(4),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" GPUs (all nodes) ")
                .borders(Borders::ALL),
        ),
        rows[1],
    );

    let watchdog = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "watchdog: labels {} • watermark {} • next {} • pending {}",
                human_count(watchdog.labels),
                human_count(watchdog.last_evidence_watermark),
                human_count(watchdog.next_label_threshold),
                human_count(watchdog.pending_unfrozen_labels)
            )
        })
        .unwrap_or_else(|| "watchdog: no status".to_owned());
    let fronts = if dashboard.accepted_tasks.is_empty() {
        "fronts: unavailable".to_owned()
    } else {
        format!(
            "fronts: {}",
            dashboard
                .accepted_tasks
                .iter()
                .map(|front| format!(
                    "{}={} (g{})",
                    front.task,
                    short_hash(&front.checkpoint_sha256),
                    front.generation
                ))
                .collect::<Vec<_>>()
                .join(" • ")
        )
    };
    frame.render_widget(
        Paragraph::new(vec![Line::from(watchdog), Line::from(fronts)]).block(
            Block::default()
                .title(" Trainer watchdog / task fronts ")
                .borders(Borders::ALL),
        ),
        rows[2],
    );

    if dashboard.trainers.is_empty() {
        frame.render_widget(
            Paragraph::new("no active GPU jobs").block(
                Block::default()
                    .title(" Deep learning (live) ")
                    .borders(Borders::ALL),
            ),
            rows[3],
        );
    } else {
        // Detail blocks want 9 rows each; with parallel arms that overflows the
        // pane. Fit as many as the space allows and account for the rest in a
        // trailing note rather than drawing past the edge.
        let available = rows[3].height as usize;
        let per_block = if dashboard.trainers.len() > 2 { 6 } else { 9 };
        let shown = dashboard.trainers.len().min(available / per_block.max(1));
        let overflow = dashboard.trainers.len().saturating_sub(shown);
        let mut constraints = (0..shown)
            .map(|_| Constraint::Length(per_block as u16))
            .collect::<Vec<_>>();
        if overflow > 0 {
            constraints.push(Constraint::Length(1));
        }
        if constraints.is_empty() {
            constraints.push(Constraint::Min(1));
        }
        let blocks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(rows[3]);
        for (index, trainer) in dashboard.trainers.iter().take(shown).enumerate() {
            if index >= blocks.len() {
                break;
            }
            render_trainer_detail(frame, dashboard, trainer, blocks[index]);
        }
        if overflow > 0 {
            if let Some(area) = blocks.get(shown) {
                frame.render_widget(
                    Paragraph::new(format!(
                        "+{overflow} more GPU job(s) — press 4 for per-arm charts"
                    ))
                    .style(Style::default().fg(Color::DarkGray)),
                    *area,
                );
            }
        }
    }
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[4]);
}

pub(crate) fn render_trainer_detail(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    trainer: &TrainerDetail,
    area: Rect,
) {
    let title = format!(
        " {} @ {}{} ",
        short(&trainer.job_id, 44),
        trainer.node,
        stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
    );
    let block = Block::default().title(title).borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .split(inner);
    let epoch_ratio = if trainer.epochs > 0 {
        (trainer.epoch as f64 / trainer.epochs as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let validating = trainer.phase.starts_with("validation");
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(if validating {
                Color::Magenta
            } else {
                Color::Cyan
            }))
            .ratio(epoch_ratio)
            .label(phase_label(trainer)),
        parts[0],
    );
    let val_history = trainer
        .history
        .iter()
        .map(|point| point.val_loss)
        .collect::<Vec<_>>();
    let trend = match trainer.history.len() {
        len if len >= 2 => {
            let last = trainer.history[len - 1].val_loss;
            let prior = trainer.history[len - 2].val_loss;
            if last < prior {
                "↓"
            } else if last > prior {
                "↑"
            } else {
                "→"
            }
        }
        _ => "-",
    };
    let early_stop = if trainer.early_stopped
        || (trainer.patience > 0 && trainer.bad_passes >= trainer.patience)
    {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
        Span::styled(
            format!(" bad {}/{} ", trainer.bad_passes, trainer.patience),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    } else {
        Span::raw(format!(
            "bad {}/{}",
            trainer.bad_passes,
            if trainer.patience > 0 {
                trainer.patience.to_string()
            } else {
                "?".to_owned()
            }
        ))
    };
    let last_point = trainer.history.last();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "loss {} {}  val {} {}  gap {}",
                trainer
                    .running_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                trend,
                trainer
                    .val_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                sparkline(&val_history),
                trainer
                    .overfit_gap
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_owned()),
            )),
            Line::from(vec![
                early_stop,
                Span::raw(format!(
                    "  pairwise {}  binary {}",
                    last_point
                        .and_then(|point| point.val_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    last_point
                        .and_then(|point| point.val_binary_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                )),
            ]),
            Line::from(format!(
                "watermark {}  elapsed {:.0}m{}",
                human_count(trainer.label_watermark),
                trainer.elapsed_sec / 60.0,
                if trainer.is_trainer {
                    ""
                } else {
                    "  (generic GPU job)"
                }
            )),
        ]),
        parts[1],
    );
}
