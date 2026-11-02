use super::*;

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
