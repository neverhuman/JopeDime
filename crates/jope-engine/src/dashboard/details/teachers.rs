use super::*;

pub(super) fn teacher_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let boot = &run["bootstrap"];
    let mut lines = vec![
        Line::styled(
            "BEST GP + TRAINING ENCODING → POLICY → NEXT GP",
            Style::default().fg(GREEN).bold(),
        ),
        Line::raw("Improve the teacher on each cut, then learn its program choices."),
        Line::raw("A stronger teacher and a stronger policy are measured separately."),
        Line::raw(""),
    ];
    for (key, label) in [
        ("meta_train", "Training families"),
        ("meta_validation", "Validation families"),
        ("meta_test", "Test families · sealed"),
    ] {
        let v = &boot["partitions"][key];
        lines.push(self::label(
            label,
            format!(
                "{} / {} with completed GP views",
                number(v, "families_with_completed_view"),
                number(v, "selected_families")
            ),
        ));
    }
    lines.push(label(
        "Coverage receipt",
        if boot["compact_receipt_ready"] == true {
            "coverage ready · saved receipt".into()
        } else {
            status_age(boot)
        },
    ));
    lines.push(label(
        "Best-teacher coverage",
        teacher["cuts_with_teacher"]
            .as_u64()
            .map(|v| format!("{v} unique cuts"))
            .unwrap_or_else(|| "awaiting compact teacher receipt".into()),
    ));
    lines.push(label(
        "New best programs",
        teacher["new_bests"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unavailable".into()),
    ));
    lines.push(label(
        "Teacher telemetry",
        status_age_with_limit(teacher, 15),
    ));
    lines
}
pub(super) fn teacher_evolution_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let mut lines = vec![
        Line::styled(
            "BEST FOUND · EXACT CUT / EVALUATOR IDENTITY",
            Style::default().fg(GREEN).bold(),
        ),
        label(
            "Teacher coverage",
            teacher["cuts_with_teacher"]
                .as_u64()
                .map(|n| {
                    format!(
                        "{n} unique cuts · {} identities",
                        number(teacher, "evaluator_cut_identities")
                    )
                })
                .unwrap_or_else(|| "awaiting durable champion receipt".into()),
        ),
        label(
            "New best / improving cuts",
            format!(
                "{} / {}",
                decimal_count(teacher, "new_bests"),
                decimal_count(teacher, "matched_cut_improvements")
            ),
        ),
        label(
            "Champion reuse",
            teacher["champion_reuse"]["status"]
                .as_str()
                .unwrap_or("unavailable"),
        ),
    ];
    for item in teacher["top_improvements"]
        .as_array()
        .into_iter()
        .flatten()
        .take(2)
    {
        let family = item["family"]
            .as_str()
            .or_else(|| item["dataset"].as_str())
            .unwrap_or("cut");
        lines.push(Line::raw(format!(
            "{} · cut {}",
            family,
            number(item, "cut")
        )));
        lines.push(Line::styled(
            format!(
                "  RMSE {} → {} · {} evaluations",
                rmse(item, "old_rmse"),
                rmse(item, "new_rmse"),
                number(item, "evaluations")
            ),
            Style::default().fg(GREEN),
        ));
        let points = item["trajectory"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v["rmse"].as_f64())
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        if points.len() > 1 {
            let low = points.iter().copied().fold(f64::INFINITY, f64::min);
            let high = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let glyphs = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            let curve = points
                .iter()
                .map(|v| {
                    glyphs[(((v - low) / (high - low).max(1e-20) * 7.).round() as usize).min(7)]
                })
                .collect::<String>();
            lines.push(Line::styled(
                format!(
                    "  Best-found RMSE {curve} · {} ordered updates",
                    points.len()
                ),
                Style::default().fg(VIOLET),
            ));
        }
    }
    lines.push(Line::styled(
        "Best observed programs, without a global-optimum claim.",
        Style::default().fg(MUTED),
    ));
    lines
}
