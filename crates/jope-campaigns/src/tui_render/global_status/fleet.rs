use super::super::*;

pub(crate) fn render_header(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let state_color = match dashboard.top_state.as_str() {
        "BLOCKED" | "STARVED" => Color::Red,
        "GATED" => Color::Yellow,
        "CLIMBING" => Color::Green,
        _ => Color::Cyan,
    };
    let health = if dashboard.alerts.is_empty() {
        Span::styled(
            " HEALTHY ",
            Style::default().fg(Color::Black).bg(Color::Green),
        )
    } else {
        Span::styled(
            format!(" {} ALERTS ", dashboard.alerts.len()),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    };
    let title = Line::from(vec![
        Span::styled(
            " DIME UNIFIED COCKPIT ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" {} ", dashboard.top_state),
            Style::default()
                .fg(Color::White)
                .bg(state_color)
                .add_modifier(Modifier::BOLD),
        ),
        health,
        Span::raw(format!(
            " SCQ {} src {}/{} {} • {}",
            dashboard
                .scq_epoch
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            dashboard.sources_healthy,
            dashboard.sources_total,
            clock_string(dashboard.collected_at_ms),
            short(&dashboard.why_idle, 72),
        )),
    ]);
    frame.render_widget(
        Paragraph::new(title).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

pub(crate) fn render_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(38),
            Constraint::Percentage(31),
            Constraint::Percentage(31),
        ])
        .split(area);
    let cpu = dashboard.cluster.cpu_busy_pct.clamp(0.0, 100.0);
    frame.render_widget(
        Gauge::default()
            .block(
                Block::default()
                    .title(" Global CPU • target 80% / guard 95% ")
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(cpu_color(cpu)))
            .ratio(cpu / 100.0)
            .label(format!("{cpu:.1}%")),
        columns[0],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "robust {}/{} of {}  {} / rung {}\nevals {}/{}  traj {}/{}  seeds {}",
            dashboard.campaign.robust_wins,
            dashboard.campaign.robust_target,
            dashboard.campaign.robust_denominator,
            dashboard.campaign.stage,
            dashboard.campaign.current_rung,
            dashboard.campaign.evaluations_used,
            dashboard.campaign.evaluations_cap,
            dashboard.campaign.trajectories_complete,
            dashboard.campaign.trajectories_total,
            dashboard.campaign.seed_agreement,
        ))
        .block(
            Block::default()
                .title(format!(
                    " ≥7/22 BUDGET CAMPAIGN • {} • test {} ",
                    dashboard.campaign.license_track, dashboard.campaign.test_opening_state
                ))
                .borders(Borders::ALL),
        ),
        columns[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "nodes {}/{}  cores {}/{}\nslots {}/{} RAM {}G flow {}/m",
            dashboard.cluster.nodes_up,
            dashboard.cluster.nodes_total,
            dashboard.cluster.effective_cores,
            dashboard.cluster.physical_cores,
            dashboard.cluster.running_slots,
            dashboard.cluster.slot_target,
            dashboard.cluster.memory_available_gb,
            dashboard.cluster.throughput_per_minute,
        ))
        .block(Block::default().title(" Capacity ").borders(Borders::ALL)),
        columns[2],
    );
}

pub(crate) fn render_nodes(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let header = Row::new([
        "node",
        "state",
        "cores",
        "mult",
        "slots",
        "CPU",
        "PSI c/m/i",
        "RAM",
        "disk",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let rows = dashboard.nodes.iter().map(|node| {
        let style = match node.state.as_str() {
            "up" => Style::default().fg(cpu_color(node.cpu_busy_pct.unwrap_or(0.0))),
            "down" => Style::default().fg(Color::Red),
            _ => Style::default().fg(Color::DarkGray),
        };
        Row::new(vec![
            Cell::from(node.node.clone()),
            Cell::from(format!("{}/{}", node.state, short(&node.health, 8))),
            Cell::from(format!("{}/{}", node.effective_cores, node.physical_cores)),
            Cell::from(format!("{:.1}x", node.multiplier)),
            Cell::from(format!("{}/{}", node.running_slots, node.slot_target)),
            Cell::from(
                node.cpu_busy_pct
                    .map(|value| format!("{value:.1}%"))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
            Cell::from(format!(
                "{}/{}/{}",
                decimal(node.psi_cpu10),
                decimal(node.psi_mem10),
                decimal(node.psi_io10)
            )),
            Cell::from(format!(
                "{}/{}G",
                node.memory_available_gb, node.memory_total_gb
            )),
            Cell::from(
                node.disk_use_pct
                    .map(|value| format!("{value:.0}%"))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
        ])
        .style(style)
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(9),
                Constraint::Length(15),
                Constraint::Length(9),
                Constraint::Length(5),
                Constraint::Length(10),
                Constraint::Length(7),
                Constraint::Length(14),
                Constraint::Length(11),
                Constraint::Length(6),
            ],
        )
        .header(header)
        .block(Block::default().title(" SCQ nodes ").borders(Borders::ALL)),
        area,
    );
}

pub(crate) fn render_queue(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(4)])
        .split(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "fresh {}/{} cap {}  age {:.0}s  mode {}",
                dashboard.queue.useful_depth,
                dashboard.queue.target,
                dashboard.queue.hard_cap,
                dashboard.queue.maximum_age_seconds,
                dashboard.queue.mode
            )),
            Line::from(format!(
                "Q/S/D/R/H {}/{}/{}/{}/{}  epoch {}",
                dashboard.queue.queued,
                dashboard.queue.starting,
                dashboard.queue.dispatching,
                dashboard.queue.running,
                dashboard.queue.held,
                short_epoch(&dashboard.queue.epoch)
            )),
        ])
        .block(
            Block::default()
                .title(" Queue horizon / epoch ")
                .borders(Borders::ALL),
        ),
        parts[0],
    );
    let header = Row::new(["family", "prio", "Q", "S", "D", "R", "H", "oldest"])
        .style(Style::default().fg(Color::Cyan));
    let rows = dashboard.families.iter().take(8).map(|family| {
        let style = if family.weight >= 10.0 {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else if family.family.starts_with("jope-selfplay-eval") {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        Row::new(vec![
            Cell::from(short(&family.family, 21)),
            Cell::from(format!("{:.1}", family.weight)),
            Cell::from(family.queued.to_string()),
            Cell::from(family.starting.to_string()),
            Cell::from(family.dispatching.to_string()),
            Cell::from(family.running.to_string()),
            Cell::from(family.held.to_string()),
            Cell::from(
                family
                    .oldest_queue_ts
                    .as_deref()
                    .map(|value| short(value, 12))
                    .unwrap_or_else(|| "-".to_owned()),
            ),
        ])
        .style(style)
    });
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Min(16),
                Constraint::Length(5),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(12),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" Priority / active states ")
                .borders(Borders::ALL),
        ),
        parts[1],
    );
}

pub(crate) fn render_learning(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let learning = &dashboard.learning;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!("checkpoint {}", short_hash(&learning.checkpoint))),
            Line::from(format!(
                "trainer {} [{}]",
                short(&learning.trainer, 38),
                learning.trainer_state
            )),
            Line::from(format!(
                "labels {}  frozen {}  rejects {}",
                human_count(learning.labels),
                human_count(learning.frozen_watermark),
                learning.consecutive_rejections
            )),
            Line::from(format!(
                "promotion internal={} milestone={}",
                learning.internal_promotion, learning.milestone_promotion
            )),
            Line::from(format!(
                "DL proposals R {} / B {}  validity {}",
                human_count(learning.proposal_regression),
                human_count(learning.proposal_binary),
                learning
                    .proposal_validity
                    .map(|value| format!("{:.1}%", value * 100.0))
                    .unwrap_or_else(|| "-".to_owned())
            )),
        ])
        .block(
            Block::default()
                .title(" Active checkpoint / DL trainer / promotion ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

/// The per-wave cohort is a rotating sample, not the population — the label
/// must make that impossible to misread.
pub(crate) fn cohort_label(funnel_active_total: u64, wave: &WaveSummary) -> String {
    let population = if funnel_active_total > 0 {
        funnel_active_total
    } else {
        wave.adaptive_regression + wave.adaptive_binary
    };
    format!(
        "gameplay cohort: 128 datasets/wave (64R/64B, 256 genomes) rotating over {} active",
        thousands(population)
    )
}

pub(crate) fn render_wave(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let wave = &dashboard.wave;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "wave {}  admissions {}  unclaimed {}",
                wave.active_wave, wave.admission_waves, wave.unclaimed_manifests
            )),
            Line::from(format!(
                "supply ready {}  keeper ack {}",
                wave.ready_manifests, wave.awaiting_keeper_ack
            )),
            Line::from(cohort_label(
                dashboard.funnel.rows.iter().map(|row| row.active).sum(),
                &dashboard.wave,
            )),
            Line::from(format!(
                "adaptive catalog R {} / B {}  corpus labels {}",
                wave.adaptive_regression,
                wave.adaptive_binary,
                human_count(wave.corpus_rows)
            )),
            Line::from(format!(
                "sealed isolated={} scored={}  pairs={} dedup={}",
                yes_no(wave.sealed_isolated),
                wave.sealed_scored,
                human_count(wave.pair_count),
                human_count(wave.duplicate_pairs_prevented)
            )),
            Line::from(format!(
                "strict wins {}  gene evidence {}",
                wave.strict_wins,
                human_bytes(wave.evidence_bytes)
            )),
        ])
        .block(
            Block::default()
                .title(" Micro-waves / adaptive gameplay / sealed lane ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

pub(crate) fn render_alerts(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let mut lines = dashboard
        .alerts
        .iter()
        .take(5)
        .map(|alert| ListItem::new(short(alert, 80)).style(Style::default().fg(Color::Yellow)))
        .collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push(ListItem::new("no active alerts").style(Style::default().fg(Color::Green)));
    }
    for failure in dashboard.failures.iter().take(2) {
        lines.push(ListItem::new(failure.clone()).style(Style::default().fg(Color::Red)));
    }
    frame.render_widget(
        List::new(lines).block(
            Block::default()
                .title(format!(
                    " Alerts / recent failures ({}) ",
                    dashboard.cluster.jobs_failed
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}

pub(crate) fn render_events(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    let lines = dashboard
        .events
        .iter()
        .take(area.height.saturating_sub(2) as usize)
        .map(|event| ListItem::new(short(event, 110)))
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(lines).block(Block::default().title(" Event tail ").borders(Borders::ALL)),
        area,
    );
}
