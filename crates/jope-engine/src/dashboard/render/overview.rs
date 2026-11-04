use super::*;

pub(crate) fn render(
    f: &mut Frame,
    body: Rect,
    h: &Hardware,
    run: &Value,
    gp: &Value,
    neural: &Value,
    stale: bool,
) {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(
            7 + u16::from(run["encoding_pipeline"]["phase"].is_string())
                + u16::from(run["campaign_service"]["phase"].is_string())
                + u16::from(run["selected_checkpoint"]["checkpoint"].is_string()),
        ),
        Constraint::Length((workflow_rows(run, gp, neural).len() + 3) as u16),
        Constraint::Min(1),
        Constraint::Length(4),
    ])
    .split(body);
    let cards = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(rows[0]);
    let gpu = h.gpus.iter().map(|g| g.utilization).sum::<f64>() / h.gpus.len().max(1) as f64;
    for (area, title, value, color, available) in [
        (
            cards[0],
            " GPU activity ",
            gpu,
            Color::Cyan,
            h.gpus.len() == 8,
        ),
        (
            cards[1],
            " CPU activity ",
            h.cpu_total,
            Color::Green,
            !h.cpu.is_empty(),
        ),
        (
            cards[2],
            " Memory used ",
            100.0 * h.ram_used_gb / h.ram_total_gb.max(1.0),
            Color::Magenta,
            h.ram_total_gb > 0.,
        ),
    ] {
        f.render_widget(
            Gauge::default()
                .block(panel(title))
                .ratio((value / 100.0).clamp(0.0, 1.0))
                .label(if stale || !available {
                    "unavailable".into()
                } else {
                    format!("{value:.1}%")
                })
                .gauge_style(Style::default().fg(if stale || !available {
                    Color::DarkGray
                } else {
                    color
                })),
            area,
        );
    }
    f.render_widget(
        Paragraph::new(milestone_lines(run, gp, neural))
            .block(panel(" Progress toward a complete learning cycle ")),
        rows[1],
    );
    f.render_widget(
        Table::new(
            workflow_rows(run, gp, neural)
                .into_iter()
                .map(TaskRow::table_row),
            [
                Constraint::Length(24),
                Constraint::Length(10),
                Constraint::Length(29),
                Constraint::Min(15),
            ],
        )
        .header(
            Row::new(["WORKFLOW", "STATE", "ACTUAL PROGRESS", "NEXT / EVIDENCE"])
                .style(Style::default().fg(MUTED)),
        )
        .column_spacing(1)
        .block(panel(" Task board · all independent work ")),
        rows[2],
    );
    let lanes = Layout::horizontal([Constraint::Ratio(1, 2); 2]).split(rows[3]);
    f.render_widget(
        Paragraph::new(teacher_lines(run))
            .wrap(Wrap { trim: true })
            .block(panel(" Teacher strength and family coverage ")),
        lanes[0],
    );
    f.render_widget(
        Paragraph::new(comparison_lines(run))
            .wrap(Wrap { trim: true })
            .block(panel(" Policy learning · completed comparisons ")),
        lanes[1],
    );
    f.render_widget(
        Paragraph::new(format!(
            "{}\nRAM available {:.0} GiB · disk free {:.0} GiB · 4 CPUs reserved for host services",
            gpu_reason(run, neural, h),
            h.available_gb,
            h.disk_free_gb
        ))
        .wrap(Wrap { trim: true })
        .block(panel(" Resource assignment / next work ")),
        rows[4],
    );
}
