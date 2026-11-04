use super::*;

pub(crate) fn render(f: &mut Frame, body: Rect, run: &Value, gp: &Value, history: &History) {
    let parts = Layout::vertical([
        Constraint::Length(5),
        Constraint::Min(8),
        Constraint::Length(14),
    ])
    .split(body);
    f.render_widget(Paragraph::new(format!("{} activity · {} active jobs · {} queued\n{} completed · {} failed · {} committed evaluations\n{} eval/s · {}",gp["current_activity"].as_str().unwrap_or("unknown"),decimal_count(gp,"active"),number(gp,"queued"),number(gp,"completed_views"),number(gp,"failed_views"),number(gp,"completed_evaluations"),decimal(gp,"evaluations_per_second"),age(gp))).block(panel(" GP search jobs ")),parts[0]);
    let rows = gp["active_rows"]
        .as_array()
        .into_iter()
        .flatten()
        .skip(history.worker_offset)
        .map(|v| {
            Row::new(vec![
                number(v, "slot").to_string(),
                v["dataset"]
                    .as_str()
                    .map(|id| run["dataset_names"][id].as_str().unwrap_or(id))
                    .unwrap_or("—")
                    .to_string(),
                number(v, "cut").to_string(),
                number(v, "evaluations").to_string(),
                v["generation"]
                    .as_u64()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".into()),
                rmse(v, "best_rmse"),
                decimal(v, "elapsed_seconds"),
            ])
        });
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(7),
                Constraint::Min(30),
                Constraint::Length(7),
                Constraint::Length(13),
                Constraint::Length(9),
                Constraint::Length(18),
                Constraint::Min(12),
            ],
        )
        .header(
            Row::new([
                "Worker",
                "Dataset",
                "Cut",
                "Evaluations",
                "Gen",
                "Best RMSE orig",
                "Seconds",
            ])
            .style(Style::default().fg(Color::Green)),
        )
        .block(panel(" Adaptive GP teacher searches · ↑ ↓ scroll workers ")),
        parts[1],
    );
    let evidence = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(parts[2]);
    f.render_widget(
        Paragraph::new(teacher_evolution_lines(run))
            .wrap(Wrap { trim: true })
            .block(panel(" Best teacher progress · no optimum claim ")),
        evidence[0],
    );
    let mut lines = comparison_lines(run);
    if let Some(report) = run["reports"].as_array().and_then(|v| v.last()) {
        let g = &report["generation"]["neural"];
        lines.push(label(
            "Neural proposal validity",
            format!(
                "{} / {} · {} unique · {} duplicates",
                number(g, "valid"),
                number(g, "attempted"),
                number(g, "unique_neural"),
                number(g, "duplicates")
            ),
        ));
    }
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel(" Matched evaluations · NN / generic / shuffled ")),
        evidence[1],
    );
}
