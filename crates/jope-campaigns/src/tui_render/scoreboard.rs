use super::*;

pub(crate) fn footer_line(dashboard: &Dashboard, refresh_ms: u64) -> Paragraph<'static> {
    Paragraph::new(format!(
        "1:overview 2:gpu/dl 3:gameplay 4:DL-charts n/←/→:arm tab:cycle • q/esc quit • read-only • refresh {}ms • collect {}ms",
        refresh_ms, dashboard.collection_latency_ms
    ))
    .style(Style::default().fg(Color::DarkGray))
}

pub(crate) fn telemetry_count(value: &jope_dime_regressor::four_task::TypedCount) -> String {
    value
        .value
        .map(human_count)
        .unwrap_or_else(|| "N/A".to_owned())
}

pub(crate) fn telemetry_median(
    value: &jope_dime_regressor::four_task::TypedDistribution,
) -> String {
    value
        .value
        .as_ref()
        .map(|distribution| format!("{:.4}", distribution.median))
        .unwrap_or_else(|| "N/A".to_owned())
}

pub(crate) fn four_task_scoreboard_rows(
    telemetry: Option<&JopeFourTaskTelemetryV1>,
    now: u128,
) -> Vec<Row<'static>> {
    let stale = telemetry.is_none_or(|value| now.saturating_sub(value.generated_at_ms) > 45_000);
    FOUR_TASK_ORDER
        .into_iter()
        .map(|task| {
            let Some(row) =
                telemetry.and_then(|telemetry| telemetry.rows.iter().find(|row| row.task == task))
            else {
                return Row::new(vec![
                    Cell::from(match task.as_str() {
                        "regression" => "R",
                        "binary" => "B",
                        "multiclass" => "MC",
                        "timeseries" => "TS",
                        _ => "?",
                    }),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A"),
                    Cell::from("N/A — telemetry absent"),
                ])
                .style(Style::default().fg(Color::DarkGray));
            };
            four_task_scoreboard_row(row, now, stale)
        })
        .collect()
}

pub(crate) fn four_task_scoreboard_row(
    row: &FourTaskTelemetryRow,
    now: u128,
    telemetry_stale: bool,
) -> Row<'static> {
    let completed = row.counts.completed.value;
    let failed = row.counts.failed.value;
    let valid = completed
        .zip(failed)
        .map(|(total, failed)| total.saturating_sub(failed));
    let validity = completed.zip(failed).and_then(|(total, failed)| {
        (total > 0).then_some((total.saturating_sub(failed)) as f64 / total as f64)
    });
    let error_name = task_metric_contract(row.task).native_error;
    let native_error = row
        .metrics
        .native_errors
        .get(error_name)
        .map(telemetry_median)
        .unwrap_or_else(|| "N/A".to_owned());
    let search_elo = row.adaptive_rating.value.as_ref().map_or_else(
        || "N/A".to_owned(),
        |rating| {
            format!(
                "{:.0} [{:.0},{:.0}]",
                rating.rating, rating.interval_95[0], rating.interval_95[1]
            )
        },
    );
    let anchor_delta = row.diagnostic_rating.value.as_ref().map_or_else(
        || "N/A".to_owned(),
        |rating| format!("{:+.0}", rating.delta_from_anchor),
    );
    let accepted = row
        .accepted_checkpoint
        .value
        .as_deref()
        .map_or_else(|| "N/A".to_owned(), |checkpoint| short(checkpoint, 9));
    let flow = row.latest_observation_at_ms.value.map_or_else(
        || "N/A — no observations".to_owned(),
        |latest| {
            let age = now.saturating_sub(latest as u128) / 1_000;
            let rate = row.valid_observations_per_minute.value.unwrap_or(0.0);
            format!("{rate:.0}/m {age}s")
        },
    );
    let cells = vec![
        Cell::from(match row.task.as_str() {
            "regression" => "R",
            "binary" => "B",
            "multiclass" => "MC",
            "timeseries" => "TS",
            _ => "?",
        }),
        Cell::from(telemetry_count(&row.counts.source_families)),
        Cell::from(completed.zip(failed).map_or_else(
            || "N/A".to_owned(),
            |(_, failed)| {
                format!(
                    "{}/{}",
                    human_count(valid.unwrap_or(0)),
                    human_count(failed)
                )
            },
        )),
        Cell::from(validity.map_or_else(
            || "N/A".to_owned(),
            |value| format!("{:.1}%", value * 100.0),
        )),
        Cell::from(telemetry_median(&row.metrics.primary_performance)),
        Cell::from(native_error),
        Cell::from(search_elo),
        Cell::from(anchor_delta),
        Cell::from(accepted),
        Cell::from(flow),
    ];
    Row::new(cells).style(if telemetry_stale {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    })
}

pub(crate) fn render_four_task_scoreboard(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    area: Rect,
) {
    let age = dashboard.four_task_telemetry.as_ref().map_or_else(
        || "NO DATA".to_owned(),
        |telemetry| {
            let age = dashboard
                .collected_at_ms
                .saturating_sub(telemetry.generated_at_ms)
                / 1_000;
            if age > 45 {
                format!("STALE {age}s — LAST-KNOWN ONLY")
            } else {
                format!("fresh {age}s")
            }
        },
    );
    let header = Row::new([
        "TASK",
        "FAM",
        "OBS/FAIL",
        "VALID",
        "PRIMARY",
        "ERROR",
        "SEARCH ELO [95%]",
        "Δ ANCHOR",
        "ACCEPTED",
        "FLOW/AGE",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_widget(
        Table::new(
            four_task_scoreboard_rows(
                dashboard.four_task_telemetry.as_ref(),
                dashboard.collected_at_ms,
            ),
            [
                Constraint::Length(5),
                Constraint::Length(7),
                Constraint::Length(14),
                Constraint::Length(7),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(21),
                Constraint::Length(10),
                Constraint::Length(11),
                Constraint::Min(18),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(format!(
                    " FOUR-TASK • ADAPTIVE/TEST-USED/NOT PROMOTION • ΔANCHOR DIAGNOSTIC/REUSABLE/PROMOTION-INELIGIBLE • {age} "
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}
