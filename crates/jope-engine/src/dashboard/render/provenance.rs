use super::*;

pub(crate) fn render(
    f: &mut Frame,
    body: Rect,
    h: &Hardware,
    run: &Value,
    gp: &Value,
    neural: &Value,
    events: &str,
) {
    let mut lines = vec![
        label("Campaign", goal_phase(run, neural, gp)),
        label(
            "Pilot launched",
            if run["pilot_started"] == true {
                "yes"
            } else {
                "no"
            },
        ),
        label(
            "Context coverage",
            format!(
                "{} saved views; {} source families",
                number(&run["contexts"], "completed"),
                number(&run["contexts"], "families")
            ),
        ),
        label(
            "Prepared workload",
            format!(
                "{} distinct cut problems from {} datasets",
                number(&run["preparation"], "views"),
                number(&run["preparation"], "datasets")
            ),
        ),
        label(
            "Cut storage",
            "source pointers + exact indices + preprocessing recipes",
        ),
        label(
            "Final tests",
            "sealed; teaching held-out scores are adaptive",
        ),
        label(
            "Family preparation",
            format!(
                "train {} / 48 · validation {} / 8 · test {} / 8 eligible families",
                number(
                    &run["family_readiness"]["families_with_verified_preparation_candidates"],
                    "meta_train"
                ),
                number(
                    &run["family_readiness"]["families_with_verified_preparation_candidates"],
                    "meta_validation"
                ),
                number(
                    &run["family_readiness"]["families_with_verified_preparation_candidates"],
                    "meta_test"
                )
            ),
        ),
        label(
            "Encoding pipeline",
            format!(
                "{} · {}",
                phase(&run["encoding_pipeline"]),
                status_age(&run["encoding_pipeline"])
            ),
        ),
        label(
            "Next dependency",
            run["encoding_pipeline"]["reason"]
                .as_str()
                .unwrap_or("No queued encoding work receipt"),
        ),
        label(
            "Qualification",
            "complete real-data inventory and final evaluation remain pilot gates",
        ),
        Line::raw(
            run["cycle"]["reason"]
                .as_str()
                .unwrap_or(
                    "All displayed progress comes from saved manifests and live worker receipts.",
                )
                .to_string(),
        ),
        Line::raw(""),
    ];
    lines.extend(
        events
            .lines()
            .rev()
            .take(30)
            .map(|s| Line::raw(s.to_string())),
    );
    if run["binary_qualification"].is_object() {
        let binary = &run["binary_qualification"];
        let detail = if run["binary_admitted_contexts"].is_number() {
            binary_replay_detail(run)
        } else {
            format!(
                "{} / {} saved · {}",
                decimal_count(binary, "completed"),
                decimal_count(binary, "total"),
                phase(binary)
            )
        };
        lines.insert(0, label("Binary contexts", detail));
        lines.insert(1,label("Binary input scope","training rows only · original source references · frozen regression projection qualification"));
        lines.insert(
            2,
            label(
                "Binary limitation",
                "Sonar acquisition grouping unresolved; duplicate groups preserved",
            ),
        );
    }
    lines.extend(
        h.errors
            .iter()
            .map(|s| Line::styled(s.clone(), Style::default().fg(Color::Yellow))),
    );
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel(" Provenance and recent events ")),
        body,
    );
}
