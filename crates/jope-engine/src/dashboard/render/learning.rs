use super::*;

pub(crate) fn render_history(f: &mut Frame, body: Rect, history: &History, run: &Value) {
    crate::learning_history::render(
        f,
        body,
        &history.learning,
        &history.learning_navigation,
        run,
    );
}

pub(crate) fn render_policy(f: &mut Frame, body: Rect, run: &Value, neural: &Value) {
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    let parts = Layout::vertical([
        Constraint::Length(18),
        Constraint::Length(3),
        Constraint::Min(7),
        Constraint::Length(5),
    ])
    .split(body);
    let top = Layout::horizontal([Constraint::Percentage(49), Constraint::Percentage(51)])
        .split(parts[0]);
    let mut lines = neural_lines(policy);
    let (status, detail) = policy_task_state(run, policy);
    lines[0] = Line::raw(format!("Policy optimization: {status}"));
    lines.insert(1, Line::raw(detail));
    lines.push(label(
        "Trained / val families",
        format!(
            "{} / {}",
            number(policy, "training_families"),
            number(&policy["validation"], "source_families")
        ),
    ));
    lines.push(label(
        "Initialization",
        policy["initialization"]["kind"]
            .as_str()
            .unwrap_or("unavailable"),
    ));
    lines.push(label(
        "Validation step",
        policy["validation"]["evaluated_step"]
            .as_u64()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unbound legacy step · no curve plotted".into()),
    ));
    lines.push(label(
        "Elapsed",
        policy["elapsed_seconds"]
            .as_f64()
            .map(|v| duration(v as u64))
            .unwrap_or_else(|| "—".into()),
    ));
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel(" Policy optimizer · BF16 / FP32 AdamW ")),
        top[0],
    );
    learning_chart(f, top[1], run);
    let done = number(policy, "step");
    let total = number(policy, "steps");
    f.render_widget(
        Gauge::default()
            .ratio((done as f64 / total.max(1) as f64).clamp(0., 1.))
            .label(format!("{done} / {total} steps · {}", status_age(policy)))
            .gauge_style(Style::default().fg(VIOLET))
            .block(panel(if completed(policy) {
                " Saved policy block "
            } else {
                " Current training block "
            })),
        parts[1],
    );
    let rows = run["checkpoints"]
        .as_array()
        .into_iter()
        .flatten()
        .rev()
        .map(|v| {
            let m = &v["metrics"];
            let origin = m["initialization"]["kind"].as_str().unwrap_or("—");
            Row::new(vec![
                v["name"].as_str().unwrap_or("checkpoint").to_owned(),
                if v["complete"] == true {
                    "saved".into()
                } else {
                    "active".into()
                },
                origin.into(),
                number(v, "step").to_string(),
                number(m, "training_families").to_string(),
                decimal(m, "sequence_nll"),
                decimal(&m["validation"], "sequence_nll"),
                decimal(&m["validation"], "ranking_bt"),
            ])
        });
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Min(29),
                Constraint::Length(8),
                Constraint::Length(12),
                Constraint::Length(7),
                Constraint::Length(9),
                Constraint::Length(11),
                Constraint::Length(11),
                Constraint::Length(11),
            ],
        )
        .header(
            Row::new([
                "MODEL / TRAINING BLOCK",
                "STATE",
                "LINEAGE",
                "STEP",
                "FAMILIES",
                "TRAIN NLL",
                "VAL NLL",
                "VAL RANK",
            ])
            .style(Style::default().fg(VIOLET)),
        )
        .block(panel(" Saved checkpoint history · durable model lineage ")),
        parts[2],
    );
    f.render_widget(Paragraph::new(vec![label("Saved checkpoint",run["selected_checkpoint"]["checkpoint"].as_str().or_else(||policy["checkpoint"].as_str()).or_else(||run["checkpoint"]["checkpoint"].as_str()).unwrap_or("awaiting complete policy receipt")),
                Line::styled("Adaptive teaching rewards optimize each cut's held-out targets; final tests stay sealed.",Style::default().fg(GOLD)),
                Line::raw("Lower imitation loss is learning evidence; matched GP comparisons measure policy value.")]).wrap(Wrap{trim:true}).block(panel(" Learning meaning and durable state ")),parts[3]);
}
