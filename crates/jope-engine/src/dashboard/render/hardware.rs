use super::*;

pub(crate) fn render(f: &mut Frame, body: Rect, h: &Hardware, history: &History, stale: bool) {
    let parts = Layout::vertical([
        Constraint::Length(11),
        Constraint::Min(1),
        Constraint::Length(5),
    ])
    .split(body);
    let rows = (0..8).map(|i| {
        if let Some(g) = h.gpus.iter().find(|g| g.index == i) {
            Row::new(vec![
                format!("A100 / {i}"),
                format!("{:5.1}%", g.utilization),
                format!(
                    "{:.1} / {:.0} GiB",
                    g.memory_mb / 1024.0,
                    g.memory_total_mb / 1024.0
                ),
                format!("{:.0} W", g.watts),
                format!("{:.0} °C", g.temperature),
                if stale { "STALE".into() } else { "live".into() },
            ])
            .style(Style::default().fg(if stale {
                GOLD
            } else if g.utilization >= 85. {
                GREEN
            } else if g.utilization >= 20. {
                BLUE
            } else {
                MUTED
            }))
        } else {
            Row::new(vec![
                format!("A100 / {i}"),
                "—".into(),
                "—".into(),
                "—".into(),
                "—".into(),
                "unavailable".into(),
            ])
        }
    });
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(20),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Min(8),
            ],
        )
        .header(
            Row::new(["Device", "Activity", "VRAM", "Power", "Temp", "Telemetry"])
                .style(Style::default().fg(Color::Cyan)),
        )
        .block(panel(" All eight GPUs ")),
        parts[0],
    );
    let bands = Layout::horizontal([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(parts[1]);
    let plots = Layout::vertical([Constraint::Ratio(1, 8); 8]).split(bands[0]);
    for i in 0..8 {
        f.render_widget(
            Sparkline::default()
                .data(history.gpu[i].iter().copied().collect::<Vec<_>>())
                .max(100)
                .style(Style::default().fg(Color::Cyan))
                .block(Block::default().title(format!("GPU {i} · 120s"))),
            plots[i],
        );
    }
    let per_row = ((bands[1].width.saturating_sub(2)) as usize / 9).max(1);
    let lines = h
        .cpu
        .chunks(per_row)
        .enumerate()
        .map(|(r, chunk)| {
            Line::from(
                chunk
                    .iter()
                    .enumerate()
                    .map(|(c, &usage)| {
                        Span::styled(
                            format!("{:03}:{:3.0}% ", r * per_row + c, usage),
                            Style::default()
                                .bg(if usage > 85.0 {
                                    Color::Rgb(24, 91, 80)
                                } else if usage > 30.0 {
                                    Color::Rgb(51, 65, 78)
                                } else {
                                    Color::Rgb(17, 30, 43)
                                })
                                .fg(if usage > 85. {
                                    GREEN
                                } else if usage > 30. {
                                    GOLD
                                } else {
                                    MUTED
                                }),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    f.render_widget(
        Paragraph::new(lines).block(panel(" Every logical CPU ")),
        bands[1],
    );
    let numa = if h.numa.is_empty() {
        "NUMA activity unavailable".into()
    } else {
        h.numa
            .iter()
            .map(|n| format!("NUMA {}: {:.1}% [{}]", n.node, n.activity, n.cpus))
            .collect::<Vec<_>>()
            .join(" · ")
    };
    f.render_widget(Paragraph::new(format!("RAM {:.1} / {:.1} GiB · available {:.1} GiB · disk free {:.0} GiB\n{}\nDisk read/write {} / {} MiB/s · load {}",h.ram_used_gb,h.ram_total_gb,h.available_gb,h.disk_free_gb,numa,h.disk_read_mib_s.map(|v|format!("{v:.1}")).unwrap_or_else(||"—".into()),h.disk_write_mib_s.map(|v|format!("{v:.1}")).unwrap_or_else(||"—".into()),h.load.trim())).block(panel(" Host resources ")),parts[2]);
}
