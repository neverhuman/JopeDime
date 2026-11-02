use super::*;

pub(super) fn decimal_count(v: &Value, key: &str) -> String {
    v[key]
        .as_u64()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}
pub(super) fn learning_chart(f: &mut Frame, area: Rect, run: &Value) {
    let samples = run["learning_curve"].as_array();
    let train = samples
        .into_iter()
        .flatten()
        .filter_map(|v| Some((v["step"].as_f64()?, v["sequence_nll"].as_f64()?)))
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    let mut validation = samples
        .into_iter()
        .flatten()
        .filter_map(|v| {
            Some((
                v["validation"]["evaluated_step"].as_f64()?,
                v["validation"]["sequence_nll"].as_f64()?,
            ))
        })
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    validation.dedup_by(|a, b| a.0 == b.0);
    if train.len() < 2 {
        f.render_widget(
            Paragraph::new("Learning curve unavailable until two optimizer receipts are saved.")
                .wrap(Wrap { trim: true })
                .block(panel(" Sequence NLL · saved optimizer history ")),
            area,
        );
        return;
    }
    let xmax = train.last().unwrap().0.max(1.);
    let ymin = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let ymax = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((ymax - ymin) * 0.1).max(0.02);
    let mut datasets = vec![Dataset::default()
        .name("train")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(BLUE))
        .data(&train)];
    if !validation.is_empty() {
        datasets.push(
            Dataset::default()
                .name("validation (bound step)")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(GOLD))
                .data(&validation),
        );
    }
    f.render_widget(
        Chart::new(datasets)
            .block(panel(" Sequence NLL · saved optimizer history "))
            .x_axis(
                Axis::default()
                    .title("optimizer step")
                    .bounds([0., xmax])
                    .labels(["0".to_string(), format!("{xmax:.0}")])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([(ymin - padding).max(0.), ymax + padding])
                    .labels([format!("{ymin:.2}"), format!("{ymax:.2}")])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}
pub(super) fn neural_lines(v: &Value) -> Vec<Line<'static>> {
    if v["encoding"].as_bool() == Some(true) {
        return vec![
            Line::styled(
                phase(v).to_string(),
                Style::default().fg(Color::Cyan).bold(),
            ),
            Line::raw(""),
            label("GPU workers", "8"),
            label(
                "Cut encodings saved",
                format!("{} / {}", number(v, "completed"), number(v, "total")),
            ),
            label(
                "CPU allocation",
                if v["persistent"] == true {
                    "120 CPUs · 4 host CPUs reserved"
                } else {
                    "16 CPUs"
                },
            ),
            label("Storage", "pooled vectors + hash receipts"),
            label("Cut data", "source pointers; memory views"),
            label("Telemetry", status_age(v)),
            Line::raw("Frozen pretrained encoders. No final-test targets enter training context."),
        ];
    }
    vec![
        Line::styled(
            phase(v).to_string(),
            Style::default().fg(Color::Cyan).bold(),
        ),
        Line::raw(""),
        label("Telemetry", status_age(v)),
        label("Synchronized ranks", format!("{} / 8", number(v, "world"))),
        label(
            "Optimizer steps",
            format!("{} / {}", number(v, "step"), number(v, "steps")),
        ),
        label(
            "Global / micro batch",
            format!(
                "{} / {}",
                number(v, "global_batch"),
                number(v, "microbatch")
            ),
        ),
        label("Loss", decimal(v, "loss")),
        label(
            "NLL / pairwise rank",
            format!(
                "{} / {}",
                decimal(v, "sequence_nll"),
                decimal(v, "ranking_bt")
            ),
        ),
        label(
            "Step / reduction (s)",
            format!(
                "{} / {}",
                decimal(v, "step_seconds"),
                decimal(v, "communication_seconds")
            ),
        ),
        label(
            "Examples / second",
            if v["end_to_end_examples_per_second"].is_number() {
                format!(
                    "{} end-to-end",
                    decimal(v, "end_to_end_examples_per_second")
                )
            } else {
                format!("{} · step timing", decimal(v, "examples_per_second"))
            },
        ),
        label(
            "Corpus eligibility",
            if v["corpus_eligible"] == true {
                "qualified"
            } else {
                "qualification only"
            },
        ),
    ]
}
