use super::*;

pub(super) const FUNNEL_HISTORY_CAP: usize = 200;

/// Canonical funnel display order. Every one of these tasks always gets a row:
/// tasks the data sources predate fall back to a corpus census so a pending
/// corpus is visible as "waiting" instead of silently absent.
pub(super) const FUNNEL_TASKS: [&str; 4] = ["regression", "binary", "timeseries", "multiclass"];

pub(super) fn funnel_tag(task: &str) -> &'static str {
    match task {
        "regression" => "REG",
        "binary" => "BIN",
        "timeseries" => "TS",
        "multiclass" | "multi_label" => "MC",
        _ => "?",
    }
}

pub(super) fn funnel_tag_compact(task: &str) -> &'static str {
    match task {
        "regression" => "reg",
        "binary" => "bin",
        "timeseries" => "ts",
        "multiclass" | "multi_label" => "mc",
        _ => "?",
    }
}

/// Ensure every funnel task has a row and impose canonical display order.
/// Missing tasks use only the typed catalog/dataset-state telemetry fallback.
/// Filesystem census roots are deliberately not a scientific authority.
pub(super) fn complete_funnel_rows(
    rows: &mut Vec<FunnelRow>,
    fallback: &BTreeMap<String, FunnelRow>,
) {
    for task in FUNNEL_TASKS {
        if !rows.iter().any(|row| row.task == task) {
            rows.push(fallback.get(task).cloned().unwrap_or_else(|| FunnelRow {
                task: task.to_owned(),
                ..Default::default()
            }));
        }
    }
    let position = |task: &str| {
        FUNNEL_TASKS
            .iter()
            .position(|known| *known == task)
            .unwrap_or(FUNNEL_TASKS.len())
    };
    rows.sort_by(|a, b| {
        position(&a.task)
            .cmp(&position(&b.task))
            .then(a.task.cmp(&b.task))
    });
}

pub(super) fn funnel_rows_from_telemetry(root: &Path) -> BTreeMap<String, FunnelRow> {
    let path = root.join("ops/four-task-telemetry.v1.json");
    let Ok(bytes) = fs::read(path) else {
        return BTreeMap::new();
    };
    let Ok(telemetry) = serde_json::from_slice::<JopeFourTaskTelemetryV1>(&bytes) else {
        return BTreeMap::new();
    };
    if telemetry.validate().is_err() {
        return BTreeMap::new();
    }
    telemetry
        .rows
        .into_iter()
        .map(|row| {
            let registry = row.counts.catalog_entries.value.unwrap_or(0);
            let encoded = row.counts.encoded.value.unwrap_or(0);
            let active = row.counts.active.value.unwrap_or(0);
            (
                row.task.as_str().to_owned(),
                FunnelRow {
                    task: row.task.as_str().to_owned(),
                    registry,
                    encoded,
                    active,
                    in_play: row.counts.observed.value.unwrap_or(0),
                    queued: encoded.saturating_sub(active).min(registry),
                    ..FunnelRow::default()
                },
            )
        })
        .collect()
}

/// Freshest source: the expansion driver's per-dataset lifecycle index.
pub(super) fn funnel_rows_from_index(path: &Path) -> Result<Vec<FunnelRow>> {
    use rusqlite::OpenFlags;
    let connection = rusqlite::Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_millis(250))?;
    connection.pragma_update(None, "query_only", "ON")?;
    // Full pipeline query; older index files without sealed/encode_job_state
    // columns fall back to the legacy stages with the new ones zeroed.
    let extended = connection.prepare(
        "SELECT task, COUNT(*), SUM(sidecars_complete), SUM(catalog_active), \
         SUM(CASE WHEN catalog_active=1 AND obs_count>0 THEN 1 ELSE 0 END), \
         SUM(CASE WHEN sidecars_complete=1 AND catalog_active=0 AND sealed=0 \
             THEN 1 ELSE 0 END), \
         SUM(CASE WHEN encode_job_state IS NOT NULL AND sidecars_complete=0 \
             THEN 1 ELSE 0 END), \
         SUM(sealed) \
         FROM datasets GROUP BY task ORDER BY task DESC",
    );
    let mut out = Vec::new();
    match extended {
        Ok(mut statement) => {
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                out.push(FunnelRow {
                    task: row.get(0)?,
                    registry: row.get::<_, i64>(1)?.max(0) as u64,
                    encoded: row.get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u64,
                    active: row.get::<_, Option<i64>>(3)?.unwrap_or(0).max(0) as u64,
                    in_play: row.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u64,
                    queued: row.get::<_, Option<i64>>(5)?.unwrap_or(0).max(0) as u64,
                    encoding: row.get::<_, Option<i64>>(6)?.unwrap_or(0).max(0) as u64,
                    sealed: row.get::<_, Option<i64>>(7)?.unwrap_or(0).max(0) as u64,
                    ..Default::default()
                });
            }
        }
        Err(_) => {
            let mut statement = connection.prepare(
                "SELECT task, COUNT(*), SUM(sidecars_complete), SUM(catalog_active), \
                 SUM(CASE WHEN catalog_active=1 AND obs_count>0 THEN 1 ELSE 0 END) \
                 FROM datasets GROUP BY task ORDER BY task DESC",
            )?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                out.push(FunnelRow {
                    task: row.get(0)?,
                    registry: row.get::<_, i64>(1)?.max(0) as u64,
                    encoded: row.get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u64,
                    active: row.get::<_, Option<i64>>(3)?.unwrap_or(0).max(0) as u64,
                    in_play: row.get::<_, Option<i64>>(4)?.unwrap_or(0).max(0) as u64,
                    ..Default::default()
                });
            }
        }
    }
    Ok(out)
}

/// Fallback: the expansion driver's embedded stage summary (refreshed each cycle).
pub(super) fn funnel_rows_from_driver(value: &Value) -> Vec<FunnelRow> {
    let summary = value.get("stage_summary");
    FUNNEL_TASKS
        .iter()
        .filter_map(|task| {
            let stage = summary.and_then(|section| section.get(*task))?;
            Some(FunnelRow {
                task: (*task).to_owned(),
                registry: u64_at(stage, &["registry"]).unwrap_or(0),
                encoded: u64_at(stage, &["sidecars_complete"]).unwrap_or(0),
                active: u64_at(stage, &["catalog_active"]).unwrap_or(0),
                in_play: u64_at(stage, &["observed"]).unwrap_or(0),
                queued: u64_at(stage, &["queued"]).unwrap_or(0),
                encoding: u64_at(stage, &["encoding"]).unwrap_or(0),
                sealed: u64_at(stage, &["sealed"]).unwrap_or(0),
                ..Default::default()
            })
        })
        .collect()
}

/// Climb indicator: compare against the oldest sample inside the last hour.
pub(super) fn apply_funnel_deltas(rows: &mut [FunnelRow], history: &mut FunnelHistory, now: u128) {
    let sample = rows
        .iter()
        .map(|row| (row.task.clone(), row.active, row.in_play))
        .collect::<Vec<_>>();
    history.push_back((now, sample));
    while history.len() > FUNNEL_HISTORY_CAP {
        history.pop_front();
    }
    let cutoff = now.saturating_sub(3_600_000);
    let baseline = history
        .iter()
        .find(|(at, _)| *at >= cutoff)
        .or_else(|| history.front());
    if let Some((_, base)) = baseline {
        for row in rows.iter_mut() {
            if let Some((_, active, in_play)) = base.iter().find(|(task, _, _)| *task == row.task) {
                row.delta_active_1h = row.active as i64 - *active as i64;
                row.delta_in_play_1h = row.in_play as i64 - *in_play as i64;
            }
        }
    }
}

pub(super) fn refresh_unlock_funnel(root: &Path, state: &mut SlowState) {
    let (mut rows, source) = match funnel_rows_from_index(&root.join("ops/dataset-state.sqlite")) {
        Ok(rows) if !rows.is_empty() => (rows, "index"),
        _ => {
            let driver = fs::read_to_string(root.join("ops/expansion-driver-v1.json"))
                .ok()
                .and_then(|text| serde_json::from_str::<Value>(&text).ok());
            match driver.as_ref().map(funnel_rows_from_driver) {
                Some(rows) if !rows.is_empty() => (rows, "driver"),
                _ => {
                    let rows = fs::read_to_string(root.join("catalog/current.json"))
                        .ok()
                        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
                        .and_then(|pointer| {
                            let path = pointer.get("catalog").and_then(Value::as_str)?.to_owned();
                            let catalog: Value =
                                serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
                            let mut rows = vec![
                                FunnelRow {
                                    task: "regression".to_owned(),
                                    active: u64_at(&catalog, &["regression_count"]).unwrap_or(0),
                                    ..Default::default()
                                },
                                FunnelRow {
                                    task: "binary".to_owned(),
                                    active: u64_at(&catalog, &["binary_count"]).unwrap_or(0),
                                    ..Default::default()
                                },
                            ];
                            // Newer tasks only when the catalog knows them: an
                            // absent key must fall through to the corpus census
                            // instead of rendering a misleading zero registry.
                            for task in ["timeseries", "multiclass"] {
                                let key = format!("{task}_count");
                                if let Some(active) = u64_at(&catalog, &[key.as_str()]) {
                                    rows.push(FunnelRow {
                                        task: task.to_owned(),
                                        active,
                                        ..Default::default()
                                    });
                                }
                            }
                            Some(rows)
                        })
                        .unwrap_or_default();
                    (rows, "catalog")
                }
            }
        }
    };
    if rows.is_empty() {
        return; // keep the previous snapshot rather than blanking the panel
    }
    complete_funnel_rows(&mut rows, &funnel_rows_from_telemetry(root));
    let now = now_ms();
    apply_funnel_deltas(&mut rows, &mut state.funnel_history, now);
    state.funnel = UnlockFunnel {
        rows,
        source: source.to_owned(),
        fetched_at_ms: now,
    };
}

pub(super) const STAGE_ACTIVE: Color = Color::Green;
pub(super) const STAGE_IN_PLAY: Color = Color::LightGreen;
pub(super) const STAGE_QUEUED: Color = Color::Yellow;
pub(super) const STAGE_ENCODING: Color = Color::Cyan;
pub(super) const STAGE_LOCKED: Color = Color::DarkGray;

/// Colored per-task pipeline line (shared by tabs 1 and 4).
pub(super) fn funnel_line(row: &FunnelRow) -> Line<'static> {
    let tag = funnel_tag(&row.task);
    let mut spans = vec![Span::styled(
        format!("{tag} {} ", thousands(row.registry)),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    spans.push(Span::styled(
        format!("● active {} ", thousands(row.active)),
        Style::default().fg(STAGE_ACTIVE),
    ));
    spans.push(Span::styled(
        format!("(in-play {}) ", thousands(row.in_play)),
        Style::default()
            .fg(STAGE_IN_PLAY)
            .add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        format!("● queued +{} ", thousands(row.queued)),
        Style::default().fg(STAGE_QUEUED),
    ));
    spans.push(Span::styled(
        format!("● encoding {} ", thousands(row.encoding)),
        Style::default().fg(STAGE_ENCODING),
    ));
    spans.push(Span::styled(
        format!(
            "● locked {} (sealed {}) ",
            thousands(row.locked()),
            thousands(row.sealed)
        ),
        Style::default().fg(STAGE_LOCKED),
    ));
    spans.push(Span::styled(
        format!(
            "Δ1h active {:+} / in-play {:+}",
            row.delta_active_1h, row.delta_in_play_1h
        ),
        Style::default().fg(Color::Gray),
    ));
    Line::from(spans)
}

/// Stacked proportional bar for one task: active/queued/encoding/locked+sealed.
pub(super) fn funnel_bar(row: &FunnelRow, width: usize) -> Line<'static> {
    let total = row.registry.max(1) as f64;
    let cells = |count: u64| ((count as f64 / total) * width as f64).round() as usize;
    let mut spans = Vec::new();
    for (count, color) in [
        (row.active, STAGE_ACTIVE),
        (row.queued, STAGE_QUEUED),
        (row.encoding, STAGE_ENCODING),
        (row.locked() + row.sealed, STAGE_LOCKED),
    ] {
        let n = cells(count);
        if n > 0 {
            spans.push(Span::styled("█".repeat(n), Style::default().fg(color)));
        }
    }
    Line::from(spans)
}

pub(super) fn compact_funnel_line(dashboard: &Dashboard) -> Paragraph<'static> {
    if dashboard.funnel.rows.is_empty() {
        return Paragraph::new(Line::from(Span::styled(
            " DATASETS  collecting…",
            Style::default().fg(Color::Cyan),
        )));
    }
    let mut spans = vec![Span::styled(
        " DATASETS  ",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )];
    for (index, row) in dashboard.funnel.rows.iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw("  •  "));
        }
        let tag = funnel_tag_compact(&row.task);
        spans.push(Span::styled(
            format!("{tag} "),
            Style::default().add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(
                "{}/{} active ",
                thousands(row.active),
                thousands(row.registry)
            ),
            Style::default().fg(STAGE_ACTIVE),
        ));
        spans.push(Span::styled(
            format!("({} play) ", thousands(row.in_play)),
            Style::default()
                .fg(STAGE_IN_PLAY)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!("+{} queued ", thousands(row.queued)),
            Style::default().fg(STAGE_QUEUED),
        ));
        spans.push(Span::styled(
            format!("{} encoding ", thousands(row.encoding)),
            Style::default().fg(STAGE_ENCODING),
        ));
        spans.push(Span::styled(
            format!("{:+}Δ1h", row.delta_active_1h),
            Style::default().fg(Color::Gray),
        ));
    }
    Paragraph::new(Line::from(spans))
}
