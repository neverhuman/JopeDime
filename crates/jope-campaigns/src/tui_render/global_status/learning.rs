use super::*;

pub(crate) fn render_dl_status(frame: &mut Frame<'_>, trainer: &TrainerDetail, area: Rect) {
    let history = &trainer.history;
    let gap_trend = if history.len() >= 2 {
        trend_arrow(
            history[history.len() - 2].overfit_gap,
            history[history.len() - 1].overfit_gap,
        )
    } else {
        "-"
    };
    let validating = trainer.phase.starts_with("validation");
    let epoch_minutes = if trainer.epoch > 0 {
        trainer.elapsed_sec / 60.0 / trainer.epoch as f64
    } else {
        0.0
    };
    let early = if trainer.patience > 0 && trainer.bad_passes >= trainer.patience {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else {
        Span::styled(
            format!(
                " early-stop {}/{} ",
                trainer.bad_passes,
                trainer.patience.max(1)
            ),
            if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
                Style::default().fg(Color::Black).bg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            },
        )
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![Span::styled(
                phase_label(trainer),
                Style::default().fg(if validating { Color::Magenta } else { Color::Cyan }),
            )]),
            Line::from(vec![
                early,
                Span::raw(format!(
                    "  overfit-gap {} {}  •  ~{epoch_minutes:.0} min/epoch  •  elapsed {:.0}m",
                    decimal(trainer.overfit_gap),
                    gap_trend,
                    trainer.elapsed_sec / 60.0,
                )),
            ]),
            // Conditional advantage is the headline: at or below zero the model
            // is not beating the genome-only control, so a healthy-looking
            // pairwise number means nothing.
            match arm_latest(trainer).map(|point| point.val_conditional_advantage) {
                Some(Some(value)) => Line::from(vec![
                    Span::raw("conditional advantage "),
                    Span::styled(
                        format!("{value:+.4}"),
                        if value > 0.0 {
                            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                        },
                    ),
                    Span::styled(
                        if value > 0.0 {
                            "  (beating the genome-only control)"
                        } else {
                            "  (NOT using the dataset embedding)"
                        },
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Some(None) => Line::from(Span::styled(
                    "conditional advantage n/a — metric absent from this run",
                    Style::default().fg(Color::DarkGray),
                )),
                None => Line::from(Span::styled(
                    "conditional advantage — warming up, no metrics yet",
                    Style::default().fg(Color::DarkGray),
                )),
            },
            Line::from(format!(
                "watermark {}  •  progress heartbeat is LIVE during validation — a full sweep takes minutes",
                human_count(trainer.label_watermark)
            )),
        ])
        .block(Block::default().title(" TRAINER STATE ").borders(Borders::ALL)),
        area,
    );
}

pub(crate) fn render_dl_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    // Two lines per task (numbers + stacked bar) plus borders. Hard-coding 4
    // truncated the BINARY row off the bottom, so the panel only ever showed
    // regression — both task totals have to be visible.
    let funnel_height = (dashboard.funnel.rows.len().max(1) * 2 + 2).min(10) as u16;
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(7),
            Constraint::Length(3),
            Constraint::Length(funnel_height),
            Constraint::Length(9),
            Constraint::Length(3),
            Constraint::Length(GENERATIONS_MAX as u16 + SEEDING_CURVE_ROWS as u16 + 3),
        ])
        .split(area);
    let alert_items = if dashboard.alerts.is_empty() {
        vec![ListItem::new(Span::styled(
            "all clear",
            Style::default().fg(Color::Green),
        ))]
    } else {
        dashboard
            .alerts
            .iter()
            .map(|alert| {
                let color = if alert.contains("critical") || alert.contains("CRITICAL") {
                    Color::Red
                } else {
                    Color::Yellow
                };
                ListItem::new(Span::styled(alert.clone(), Style::default().fg(color)))
            })
            .collect()
    };
    frame.render_widget(
        List::new(alert_items).block(
            Block::default()
                .title(" GLOBAL ERRORS / ALARMS ")
                .borders(Borders::ALL),
        ),
        parts[0],
    );
    let heartbeat_lines = dashboard
        .daemon_ages
        .iter()
        .map(|(name, age, expected)| {
            let (text, color) = if *age == u64::MAX {
                ("MISSING".to_owned(), Color::Red)
            } else if *age > expected * 2 {
                (format!("{age}s (expect ~{expected}s)"), Color::Red)
            } else if *age > *expected {
                (format!("{age}s (expect ~{expected}s)"), Color::Yellow)
            } else {
                (format!("{age}s"), Color::Green)
            };
            Line::from(vec![
                Span::raw(format!("{name:<24} ")),
                Span::styled(text, Style::default().fg(color)),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(heartbeat_lines).block(
            Block::default()
                .title(" DAEMON HEARTBEATS ")
                .borders(Borders::ALL),
        ),
        parts[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "failed {}  •  {} jobs/min  •  cluster CPU {:.0}% (target {:.0}%)",
            human_count(dashboard.cluster.jobs_failed),
            dashboard.cluster.throughput_per_minute,
            dashboard.cluster.cpu_busy_pct,
            CPU_TARGET_PCT
        ))
        .block(Block::default().title(" SCQ ").borders(Borders::ALL)),
        parts[2],
    );
    let funnel_lines = if dashboard.funnel.rows.is_empty() {
        vec![Line::from("collecting…")]
    } else {
        let bar_width = (parts[3].width.saturating_sub(4) as usize).clamp(10, 60);
        dashboard
            .funnel
            .rows
            .iter()
            .flat_map(|row| [funnel_line(row), funnel_bar(row, bar_width)])
            .collect()
    };
    frame.render_widget(
        Paragraph::new(funnel_lines).block(
            Block::default()
                .title(format!(
                    " DATASET UNLOCK ({}){} ",
                    dashboard.funnel.source,
                    stale_suffix(now_ms(), dashboard.funnel.fetched_at_ms)
                ))
                .borders(Borders::ALL),
        ),
        parts[3],
    );
    let trainer = dashboard
        .trainers
        .iter()
        .find(|trainer| trainer.is_trainer)
        .or_else(|| dashboard.trainers.first());
    let metric_lines = trainer
        .map(|trainer| {
            let history = &trainer.history;
            let last = history.last();
            let prior = history
                .len()
                .checked_sub(2)
                .and_then(|index| history.get(index));
            let row = |label: &str, get: &dyn Fn(&TrainerMetricPoint) -> Option<f64>| {
                let latest = last.and_then(get);
                let arrow = match (prior.and_then(get), latest) {
                    (Some(previous), Some(current)) => trend_arrow(previous, current),
                    _ => "-",
                };
                Line::from(format!("{label:<22} {}  {arrow}", decimal(latest)))
            };
            vec![
                row("val_pairwise_win", &|point| point.val_pairwise),
                row("val_regression_win", &|point| point.val_regression_pairwise),
                row("val_binary_win", &|point| point.val_binary_pairwise),
                row("reward_cv_corr", &|point| point.val_reward_cv),
                row("top1_regret (↓ good)", &|point| point.val_top1_regret),
                row("overfit_gap (↓ good)", &|point| Some(point.overfit_gap)),
                row("val_loss (↓ good)", &|point| Some(point.val_loss)),
            ]
        })
        .unwrap_or_else(|| vec![Line::from("no trainer history yet")]);
    frame.render_widget(
        Paragraph::new(metric_lines).block(
            Block::default()
                .title(" VAL METRICS (vs prior epoch) ")
                .borders(Borders::ALL),
        ),
        parts[4],
    );
    let labels_line = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "labels {}  •  pending {}  •  next trainer at {}",
                human_count(watchdog.labels),
                human_count(watchdog.pending_unfrozen_labels),
                human_count(watchdog.next_label_threshold)
            )
        })
        .unwrap_or_else(|| "watchdog status unavailable".to_owned());
    frame.render_widget(
        Paragraph::new(labels_line).block(
            Block::default()
                .title(" LABEL SUPPLY ")
                .borders(Borders::ALL),
        ),
        parts[5],
    );
    let percent = |value: Option<f64>| {
        value
            .map(|share| format!("{share:.1}%"))
            .unwrap_or_else(|| "-".to_owned())
    };
    let mut generation_lines = vec![Line::from(Span::styled(
        format!(
            "{:<30}  {:>7}  {:>7}  {:>7}",
            "run / seed checkpoint", "val-pw", "DL-win", "DL-obs"
        ),
        Style::default().fg(Color::DarkGray),
    ))];
    if dashboard.generations.is_empty() {
        generation_lines.push(Line::from("collecting…"));
    } else {
        generation_lines.extend(dashboard.generations.iter().map(|(label, pairwise)| {
            Line::from(format!(
                "{:<30}  {:>7}  {:>7}  {:>7}",
                short(label, 30),
                decimal(*pairwise),
                "-",
                "-"
            ))
        }));
    }
    // DL-share column source: the newest seeding-curve checkpoint generations.
    let skip = dashboard
        .seeding
        .points
        .len()
        .saturating_sub(SEEDING_CURVE_ROWS);
    for point in dashboard.seeding.points.iter().skip(skip) {
        let label = format!(
            "ckpt {} ({} wins)",
            point.checkpoint_short, point.strict_wins_total
        );
        let observed = if point.observations_caught_up {
            percent(point.dl_obs_share_pct)
        } else {
            format!("~{}", percent(point.dl_obs_share_pct))
        };
        generation_lines.push(Line::from(vec![
            Span::raw(format!("{:<30}  {:>7}  ", short(&label, 30), "-")),
            Span::styled(
                format!("{:>7}", percent(point.dl_win_share_pct)),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(format!("  {observed:>7}")),
        ]));
    }
    frame.render_widget(
        Paragraph::new(generation_lines)
            .block(
                Block::default()
                    .title(format!(
                        " GENERATIONS (val-pw • DL-share, old→new){} ",
                        stale_suffix(now_ms(), dashboard.seeding.fetched_at_ms)
                    ))
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        parts[6],
    );
}
