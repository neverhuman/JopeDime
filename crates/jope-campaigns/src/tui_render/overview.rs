use super::*;

pub(crate) fn render_overview(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Min(10),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_global(frame, dashboard, rows[1]);
    frame.render_widget(compact_funnel_line(dashboard), rows[2]);
    render_four_task_scoreboard(frame, dashboard, rows[3]);
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[4]);
    render_nodes(frame, dashboard, main[0]);
    render_queue(frame, dashboard, main[1]);
    let learning = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[5]);
    render_learning(frame, dashboard, learning[0]);
    render_wave(frame, dashboard, learning[1]);
    let tail = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[6]);
    render_alerts(frame, dashboard, tail[0]);
    render_events(frame, dashboard, tail[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[7]);
}

pub(crate) fn stale_suffix(now_ms_value: u128, fetched_at_ms: u128) -> String {
    if fetched_at_ms == 0 {
        return " [NO DATA]".to_owned();
    }
    let age = now_ms_value.saturating_sub(fetched_at_ms);
    if age > STALE_MS {
        format!(" [STALE {}s]", age / 1_000)
    } else {
        String::new()
    }
}

pub(crate) fn sparkline(values: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return "-".to_owned();
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    finite
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * 7.0).round() as usize;
            BARS[bucket.min(7)]
        })
        .collect()
}

pub(crate) fn origin_label(origin: Origin) -> (&'static str, Color) {
    match origin {
        Origin::DlProposal => ("dl", Color::Magenta),
        Origin::GpOffspring => ("gp", Color::Cyan),
        Origin::LocalElite => ("elite", Color::Green),
        Origin::CommonWinner => ("robust", Color::Blue),
        Origin::Unknown => ("?", Color::DarkGray),
    }
}

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

pub(crate) fn render_gameplay_tab(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_four_task_scoreboard(frame, dashboard, rows[1]);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[2]);
    if let Some(progress) = &dashboard.wave_progress {
        let ratio = if progress.manifest_count > 0 {
            (progress.receipts_landed as f64 / progress.manifest_count as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let claims = if progress.claims_by_state.is_empty() {
            String::new()
        } else {
            format!(
                " • claims {}",
                progress
                    .claims_by_state
                    .iter()
                    .map(|(state, count)| format!("{state}:{count}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .title(format!(" {} (receipts since TUI start) ", progress.wave))
                        .borders(Borders::ALL),
                )
                .gauge_style(Style::default().fg(Color::Cyan))
                .ratio(ratio)
                .label(format!(
                    "{}/{} manifests • {} datasets{}",
                    progress.receipts_landed, progress.manifest_count, progress.datasets, claims
                )),
            top[0],
        );
    } else {
        frame.render_widget(
            Paragraph::new("no wave summary")
                .block(Block::default().title(" Wave ").borders(Borders::ALL)),
            top[0],
        );
    }
    let stats = &dashboard.gameplay_stats;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "manifests seen {} • DL top-1 {} / GP top-1 {} (window {})",
                stats.manifests_seen, stats.dl_top1, stats.gp_top1, GAMEPLAY_STATS_WINDOW
            )),
            Line::from(format!(
                "DL slot share {:.1}% • Δ incumbent N/A (typed context authority unavailable)",
                stats.dl_slot_share_pct
            )),
        ])
        .block(
            Block::default()
                .title(" DL vs GP (rolling) ")
                .borders(Borders::ALL),
        ),
        top[1],
    );

    let wins = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::StrictWin)
        .take(5)
        .map(|event| {
            let (label, _) = origin_label(event.origin);
            ListItem::new(format!(
                "{} {} {} Δ{} [{}]",
                event.task,
                short(&event.dataset_hash, 16),
                event
                    .score
                    .map(|value| format!("{value:.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                event
                    .delta_vs_incumbent
                    .map(|value| format!("{value:+.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                label
            ))
            .style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect::<Vec<_>>();
    let wins = if wins.is_empty() {
        vec![ListItem::new("no strict wins observed since TUI start")
            .style(Style::default().fg(Color::DarkGray))]
    } else {
        wins
    };
    frame.render_widget(
        List::new(wins).block(
            Block::default()
                .title(" Strict wins (live) ")
                .borders(Borders::ALL),
        ),
        rows[3],
    );

    let feed_title = format!(
        " Evaluation feed{} ",
        stale_suffix(dashboard.collected_at_ms, dashboard.feed_fetched_at_ms)
    );
    let header = Row::new([
        "time",
        "T",
        "dataset",
        "score",
        "Δ incumbent",
        "origin",
        "node",
        "wave",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let feed_rows = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::Eval)
        .take(rows[4].height.saturating_sub(3) as usize)
        .map(|event| {
            let (label, color) = origin_label(event.origin);
            let delta_cell = event
                .delta_vs_incumbent
                .map(|value| format!("{value:+.5}"))
                .unwrap_or_else(|| "-".to_owned());
            let style = if !event.valid {
                Style::default().fg(Color::DarkGray)
            } else if event.delta_vs_incumbent.is_some_and(|value| value > 0.0) {
                Style::default().fg(Color::Green)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(clock_string(event.at_ms)),
                Cell::from(match event.task.as_str() {
                    "regression" => "R",
                    "binary" => "B",
                    "timeseries" => "T",
                    "multiclass" | "multi_label" => "M",
                    _ => "?",
                }),
                Cell::from(short(&event.dataset_hash, 18)),
                Cell::from(
                    event
                        .score
                        .map(|value| format!("{value:.5}"))
                        .unwrap_or_else(|| "invalid".to_owned()),
                ),
                Cell::from(delta_cell),
                Cell::from(Span::styled(label, Style::default().fg(color))),
                Cell::from(event.node.clone()),
                Cell::from(short(
                    event
                        .wave
                        .strip_prefix("strict-wave-")
                        .unwrap_or(&event.wave),
                    8,
                )),
            ])
            .style(style)
        });
    frame.render_widget(
        Table::new(
            feed_rows,
            [
                Constraint::Length(9),
                Constraint::Length(2),
                Constraint::Length(19),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(8),
            ],
        )
        .header(header)
        .block(Block::default().title(feed_title).borders(Borders::ALL)),
        rows[4],
    );
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[5]);
}

pub(crate) fn phase_label(trainer: &TrainerDetail) -> String {
    if trainer.phase.starts_with("validation") {
        format!(
            "epoch {}/{} • VALIDATION {}/{} • training batches done",
            trainer.epoch, trainer.epochs, trainer.validation_batch, trainer.validation_batches
        )
    } else {
        format!(
            "epoch {}/{} • batch {}/{} ({:.0}%) • {}",
            trainer.epoch,
            trainer.epochs,
            trainer.batch,
            trainer.total_batches,
            trainer.pct,
            trainer.phase
        )
    }
}

pub(crate) fn trend_arrow(prior: f64, last: f64) -> &'static str {
    if last < prior {
        "↓"
    } else if last > prior {
        "↑"
    } else {
        "→"
    }
}

// The per-epoch chart series (train/val loss, pairwise, ablation) are produced
// by `continuous_series` from the cross-generation `DlHistory`; the old
// single-run extractors keyed on one trainer's history were what emptied the
// chart on job turnover and have been removed.

/// Training arms in a stable display order: real trainers first (so a proposal
/// or encoder GPU job never steals arm 0), each keeping its collector ordering.

pub(crate) fn training_arms(dashboard: &Dashboard) -> Vec<&TrainerDetail> {
    let mut arms: Vec<&TrainerDetail> =
        dashboard.trainers.iter().filter(|t| t.is_trainer).collect();
    arms.extend(dashboard.trainers.iter().filter(|t| !t.is_trainer));
    arms
}

/// Newest metric row for an arm, or None while its metrics file is still empty.
/// Callers must render "warming up" rather than borrowing another arm's numbers.

pub(crate) fn arm_latest(trainer: &TrainerDetail) -> Option<&TrainerMetricPoint> {
    trainer.history.last()
}

pub(crate) fn gpu_for_job<'a>(dashboard: &'a Dashboard, job_id: &str) -> Option<&'a GpuRow> {
    dashboard
        .gpus
        .iter()
        .find(|gpu| gpu.owner_job.as_deref() == Some(job_id))
}
