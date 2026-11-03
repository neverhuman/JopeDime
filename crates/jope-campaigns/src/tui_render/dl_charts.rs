use super::*;

pub(crate) fn render_dl_tab(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
    arm: usize,
) {
    let arms = training_arms(dashboard);
    // The comparison strip is sized to the live arm count so a single-arm run
    // keeps the charts as tall as before.
    let strip_height = if arms.len() > 1 {
        arms.len() as u16 + 2
    } else {
        0
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(strip_height),
            Constraint::Min(16),
            Constraint::Length(1),
        ])
        .split(area);
    if strip_height > 0 {
        render_arm_strip(frame, dashboard, &arms, arm, rows[0]);
    }
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(11),
            Constraint::Length(1),
            Constraint::Length(8),
            Constraint::Length(6),
        ])
        .split(columns[0]);
    // The charts are driven by the on-disk generation history, not by whichever
    // job happens to be running, so they keep rendering across trainer
    // changeovers and while a successor is still staging.
    let series = continuous_series(&dashboard.dl_history);
    let live = arms.get(selected_arm(arm, arms.len())).copied();
    render_loss_chart(frame, dashboard, &series, live, left[0]);
    frame.render_widget(
        Paragraph::new(generation_ruler(&series, left[1].width)),
        left[1],
    );
    render_pairwise_chart(frame, &series, left[2]);
    match live {
        Some(trainer) => render_dl_status(frame, trainer, left[3]),
        None => frame.render_widget(
            Paragraph::new(
                "no live trainer — the watchdog submits the next one when labels accumulate",
            )
            .style(Style::default().fg(Color::Yellow))
            .block(Block::default().title(" LIVE RUN ").borders(Borders::ALL)),
            left[3],
        ),
    }
    render_dl_global(frame, dashboard, columns[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[2]);
}

/// Wrap the raw key-press counter onto the live arm count. Kept separate so the
/// tests can pin the wrap behaviour when arms appear or exit mid-run.
pub(crate) fn selected_arm(arm: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        arm % count
    }
}

/// One line per arm: the at-a-glance comparison across every GPU. Shown only
/// when more than one arm is live.
pub(crate) fn render_arm_strip(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    arms: &[&TrainerDetail],
    arm: usize,
    area: Rect,
) {
    let selected = selected_arm(arm, arms.len());
    let best = arms
        .iter()
        .filter_map(|t| arm_latest(t).and_then(|p| p.val_conditional_advantage))
        .fold(f64::NEG_INFINITY, f64::max);
    let lines = arms
        .iter()
        .enumerate()
        .map(|(index, trainer)| {
            let marker = if index == selected { "▶" } else { " " };
            let node = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!("{} {}", trainer.node, gpu.model_hint))
                .unwrap_or_else(|| trainer.node.clone());
            let Some(point) = arm_latest(trainer) else {
                return Line::from(vec![
                    Span::raw(format!(
                        "{marker} {:<22} {:<14} ",
                        short(&trainer.job_id, 22),
                        node
                    )),
                    Span::styled(
                        "warming up — no metrics yet",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
            };
            let advantage = point.val_conditional_advantage;
            let advantage_style = match advantage {
                Some(value) if value >= best && best > f64::NEG_INFINITY => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                Some(value) if value > 0.0 => Style::default().fg(Color::Green),
                Some(_) => Style::default().fg(Color::Red),
                None => Style::default().fg(Color::DarkGray),
            };
            Line::from(vec![
                Span::raw(format!(
                    "{marker} {:<22} {:<14} ep {:>3}/{:<3} train {:>6.3} val {:>7.3} pw {:>5.3} ",
                    short(&trainer.job_id, 22),
                    node,
                    trainer.epoch,
                    trainer.epochs,
                    point.train_loss,
                    point.val_loss,
                    point.val_pairwise.unwrap_or(f64::NAN),
                )),
                Span::styled(
                    match advantage {
                        Some(value) => format!("cond {value:+.4}"),
                        None => "cond    n/a".to_owned(),
                    },
                    advantage_style,
                ),
                Span::raw(format!(
                    "  stop {}/{}",
                    point.bad_passes.max(trainer.bad_passes),
                    trainer.patience
                )),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(format!(
                    " TRAINING ARMS ({}) — n/←/→ to select ",
                    arms.len()
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}

/// Cross-generation loss chart. Takes the flattened history rather than a live
/// trainer so it renders whether or not a job is currently running.
pub(crate) fn render_loss_chart(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    series: &DlSeries,
    trainer: Option<&TrainerDetail>,
    area: Rect,
) {
    let (train, val) = (&series.train, &series.val);
    let all = train.iter().chain(val.iter()).map(|(_, y)| *y);
    let y_max = all.clone().fold(f64::MIN, f64::max);
    let y_min = all.fold(f64::MAX, f64::min);
    let (y_min, y_max) = if y_min > y_max {
        (0.0, 1.0)
    } else {
        ((y_min * 0.9).max(0.0), (y_max * 1.1).max(y_min + 0.1))
    };
    let x_min = train
        .first()
        .or(val.first())
        .map(|(x, _)| *x)
        .unwrap_or(0.0);
    let x_max = train
        .last()
        .or(val.last())
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    // Unnamed datasets stay out of the legend, so the boundaries read as plain
    // rules between generations.
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_min), (*x - 0.5, y_max)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name(format!(
                "train {}",
                decimal(
                    trainer
                        .and_then(|trainer| trainer.running_loss)
                        .or_else(|| train.last().map(|(_, y)| *y))
                )
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(train),
    );
    datasets.push(
        Dataset::default()
            .name(format!("val {}", decimal(val.last().map(|(_, y)| *y))))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(val),
    );
    let arms = training_arms(dashboard);
    let live_label = match trainer {
        Some(trainer) => {
            let arm_label = arms
                .iter()
                .position(|candidate| candidate.job_id == trainer.job_id)
                .filter(|_| arms.len() > 1)
                .map(|index| format!(" [arm {}/{}]", index + 1, arms.len()))
                .unwrap_or_default();
            let gpu_label = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!(" {}", gpu.model_hint))
                .unwrap_or_default();
            format!(
                "live {} @ {}{}{}{}",
                short(&trainer.job_id, 34),
                trainer.node,
                gpu_label,
                arm_label,
                stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
            )
        }
        None => "no live trainer".to_owned(),
    };
    let title = format!(
        " HISTORICAL LOSS • {}{} ",
        live_label,
        stale_suffix(
            dashboard.collected_at_ms,
            dashboard.dl_history.fetched_at_ms
        )
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(Block::default().title(title).borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("epoch")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max])
                    .labels(vec![
                        format!("{x_min:.0}"),
                        format!("{:.0}", (x_min + x_max) / 2.0),
                        format!("{x_max:.0}"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        format!("{y_min:.2}"),
                        format!("{:.2}", (y_min + y_max) / 2.0),
                        format!("{y_max:.2}"),
                    ]),
            ),
        area,
    );
}

/// Generation ruler drawn directly under the loss chart: one tick per trainer
/// generation, positioned proportionally, so the boundaries are readable rather
/// than merely visible. The live generation is highlighted.
pub(crate) fn generation_ruler(series: &DlSeries, width: u16) -> Line<'static> {
    if series.marks.is_empty() || width < 8 {
        return Line::from(Span::styled(
            "no trainer generations on disk yet",
            Style::default().fg(Color::DarkGray),
        ));
    }
    // Match the chart's plotting area: 1 col of left border + 1 of right.
    let inner = width.saturating_sub(2) as usize;
    let span_epochs = series.total_epochs.max(1.0);
    let mut spans = vec![Span::raw(" ")];
    let mut cursor = 0usize;
    for (x, label, live) in &series.marks {
        let column = (((*x - 1.0) / span_epochs) * inner as f64).round() as usize;
        let text = format!("├{label}");
        if column + text.chars().count() > inner {
            continue;
        }
        if column < cursor {
            continue;
        }
        spans.push(Span::raw(" ".repeat(column - cursor)));
        let style = if *live {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::styled(text.clone(), style));
        cursor = column + text.chars().count();
    }
    Line::from(spans)
}

/// Cross-generation pairwise-win chart (the "should climb" curve).
pub(crate) fn render_pairwise_chart(frame: &mut Frame<'_>, series: &DlSeries, area: Rect) {
    let points = series.pairwise.clone();
    let ablation = series.ablation.clone();
    let x_min = points.first().map(|(x, _)| *x).unwrap_or(0.0);
    let x_max = points
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    let y_low = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MAX, f64::min)
        .min(0.48);
    let y_high = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MIN, f64::max)
        .max(0.55);
    let (y_low, y_high) = if y_low > y_high {
        (0.4, 0.8)
    } else {
        ((y_low - 0.02).max(0.0), (y_high + 0.02).min(1.0))
    };
    let chance = vec![(x_min, 0.5), (x_max, 0.5)];
    let latest = points.last().map(|(_, y)| *y);
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_low), (*x - 0.5, y_high)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name("chance 0.5")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(&chance),
    );
    datasets.push(
        Dataset::default()
            .name(format!(
                "ablation(control) {}",
                decimal(ablation.last().map(|(_, y)| *y))
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&ablation),
    );
    datasets.push(
        Dataset::default()
            .name(format!("pairwise {}", decimal(latest)))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&points),
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::default()
                    .title(" HISTORICAL VAL PAIRWISE-WIN — all generations ")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_low, y_high])
                    .labels(vec![
                        format!("{y_low:.2}"),
                        "0.50".to_owned(),
                        format!("{y_high:.2}"),
                    ]),
            ),
        area,
    );
}
