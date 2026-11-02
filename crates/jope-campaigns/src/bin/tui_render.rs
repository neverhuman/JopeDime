use super::*;

pub(super) fn render(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, ui: UiState) {
    let mut area = frame.area();
    // The blocked/throttled banner pre-empts the first row of EVERY tab and
    // both compact layouts: if the flywheel is starved, no view is allowed to
    // look calm.
    if let Some(text) = blocked_banner(dashboard) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!("▲▲▲  {text}  ▲▲▲")).style(
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
            rows[0],
        );
        area = rows[1];
    }
    if area.width < 100 || area.height < 32 {
        match ui.tab {
            Tab::Overview | Tab::Gameplay => render_compact(frame, dashboard, refresh_ms, area),
            Tab::GpuDl | Tab::Dl => render_compact_gpu_dl(frame, dashboard, refresh_ms, area),
        }
        return;
    }
    match ui.tab {
        Tab::Overview => render_overview(frame, dashboard, refresh_ms, area),
        Tab::GpuDl => render_gpu_tab(frame, dashboard, refresh_ms, area),
        Tab::Gameplay => render_gameplay_tab(frame, dashboard, refresh_ms, area),
        Tab::Dl => render_dl_tab(frame, dashboard, refresh_ms, area, ui.arm),
    }
}

fn footer_line(dashboard: &Dashboard, refresh_ms: u64) -> Paragraph<'static> {
    Paragraph::new(format!(
        "1:overview 2:gpu/dl 3:gameplay 4:DL-charts n/←/→:arm tab:cycle • q/esc quit • read-only • refresh {}ms • collect {}ms",
        refresh_ms, dashboard.collection_latency_ms
    ))
    .style(Style::default().fg(Color::DarkGray))
}

fn telemetry_count(value: &jope_dime_regressor::four_task::TypedCount) -> String {
    value
        .value
        .map(human_count)
        .unwrap_or_else(|| "N/A".to_owned())
}

fn telemetry_median(value: &jope_dime_regressor::four_task::TypedDistribution) -> String {
    value
        .value
        .as_ref()
        .map(|distribution| format!("{:.4}", distribution.median))
        .unwrap_or_else(|| "N/A".to_owned())
}

pub(super) fn four_task_scoreboard_rows(
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

fn four_task_scoreboard_row(
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

pub(super) fn render_four_task_scoreboard(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_overview(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Min(10),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_global(frame, dashboard, rows[1]);
    frame.render_widget(compact_funnel_line(dashboard), rows[2]);
    render_four_task_scoreboard(frame, dashboard, rows[3]);
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[4]);
    render_nodes(frame, dashboard, main[0]);
    render_queue(frame, dashboard, main[1]);
    let learning = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[5]);
    render_learning(frame, dashboard, learning[0]);
    render_wave(frame, dashboard, learning[1]);
    let tail = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[6]);
    render_alerts(frame, dashboard, tail[0]);
    render_events(frame, dashboard, tail[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[7]);
}

fn stale_suffix(now_ms_value: u128, fetched_at_ms: u128) -> String {
    if fetched_at_ms == 0 {
        return " [NO DATA]".to_owned();
    }
    let age = now_ms_value.saturating_sub(fetched_at_ms);
    if age > STALE_MS {
        format!(" [STALE {}s]", age / 1_000)
    } else {
        String::new()
    }
}

pub(super) fn sparkline(values: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return "-".to_owned();
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    finite
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * 7.0).round() as usize;
            BARS[bucket.min(7)]
        })
        .collect()
}

fn origin_label(origin: Origin) -> (&'static str, Color) {
    match origin {
        Origin::DlProposal => ("dl", Color::Magenta),
        Origin::GpOffspring => ("gp", Color::Cyan),
        Origin::LocalElite => ("elite", Color::Green),
        Origin::CommonWinner => ("robust", Color::Blue),
        Origin::Unknown => ("?", Color::DarkGray),
    }
}

fn render_gpu_tab(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let trainer_rows = (dashboard.trainers.len().max(1) * 9) as u16;
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3 + dashboard.gpus.len().max(1) as u16),
            Constraint::Length(4),
            Constraint::Min(trainer_rows.min(27)),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);

    let header = Row::new([
        "node",
        "gpu",
        "status/owner",
        "util",
        "mem MiB",
        "temp",
        "power W",
        "thr",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let gpu_rows = dashboard.gpus.iter().map(|gpu| {
        let color = if gpu.temp_c >= 80 {
            Color::Red
        } else if gpu.util_pct >= 1.0 {
            Color::Green
        } else {
            Color::DarkGray
        };
        let ownership = if gpu.role.is_empty() {
            gpu.owner_job.as_deref().unwrap_or(&gpu.status).to_owned()
        } else {
            format!(
                "{}#{} {} [{}]",
                gpu.role,
                gpu.attempt,
                gpu.owner_job.as_deref().unwrap_or("idle"),
                gpu.admission_reason
            )
        };
        Row::new(vec![
            Cell::from(gpu.node.clone()),
            Cell::from(format!("{}?", gpu.model_hint)),
            Cell::from(short(&ownership, 40)),
            Cell::from(format!("{:.0}%", gpu.util_pct)),
            Cell::from(format!("{}/{}", gpu.mem_used_mb, gpu.mem_total_mb)),
            Cell::from(format!("{}C", gpu.temp_c)),
            Cell::from(format!("{:.0}/{:.0}", gpu.power_w, gpu.power_limit_w)),
            Cell::from(if gpu.throttled { "YES" } else { "-" }),
        ])
        .style(Style::default().fg(color))
    });
    frame.render_widget(
        Table::new(
            gpu_rows,
            [
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Min(30),
                Constraint::Length(6),
                Constraint::Length(13),
                Constraint::Length(5),
                Constraint::Length(9),
                Constraint::Length(4),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .title(" GPUs (all nodes) ")
                .borders(Borders::ALL),
        ),
        rows[1],
    );

    let watchdog = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "watchdog: labels {} • watermark {} • next {} • pending {}",
                human_count(watchdog.labels),
                human_count(watchdog.last_evidence_watermark),
                human_count(watchdog.next_label_threshold),
                human_count(watchdog.pending_unfrozen_labels)
            )
        })
        .unwrap_or_else(|| "watchdog: no status".to_owned());
    let fronts = if dashboard.accepted_tasks.is_empty() {
        "fronts: unavailable".to_owned()
    } else {
        format!(
            "fronts: {}",
            dashboard
                .accepted_tasks
                .iter()
                .map(|front| format!(
                    "{}={} (g{})",
                    front.task,
                    short_hash(&front.checkpoint_sha256),
                    front.generation
                ))
                .collect::<Vec<_>>()
                .join(" • ")
        )
    };
    frame.render_widget(
        Paragraph::new(vec![Line::from(watchdog), Line::from(fronts)]).block(
            Block::default()
                .title(" Trainer watchdog / task fronts ")
                .borders(Borders::ALL),
        ),
        rows[2],
    );

    if dashboard.trainers.is_empty() {
        frame.render_widget(
            Paragraph::new("no active GPU jobs").block(
                Block::default()
                    .title(" Deep learning (live) ")
                    .borders(Borders::ALL),
            ),
            rows[3],
        );
    } else {
        // Detail blocks want 9 rows each; with parallel arms that overflows the
        // pane. Fit as many as the space allows and account for the rest in a
        // trailing note rather than drawing past the edge.
        let available = rows[3].height as usize;
        let per_block = if dashboard.trainers.len() > 2 { 6 } else { 9 };
        let shown = dashboard.trainers.len().min(available / per_block.max(1));
        let overflow = dashboard.trainers.len().saturating_sub(shown);
        let mut constraints = (0..shown)
            .map(|_| Constraint::Length(per_block as u16))
            .collect::<Vec<_>>();
        if overflow > 0 {
            constraints.push(Constraint::Length(1));
        }
        if constraints.is_empty() {
            constraints.push(Constraint::Min(1));
        }
        let blocks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(rows[3]);
        for (index, trainer) in dashboard.trainers.iter().take(shown).enumerate() {
            if index >= blocks.len() {
                break;
            }
            render_trainer_detail(frame, dashboard, trainer, blocks[index]);
        }
        if overflow > 0 {
            if let Some(area) = blocks.get(shown) {
                frame.render_widget(
                    Paragraph::new(format!(
                        "+{overflow} more GPU job(s) — press 4 for per-arm charts"
                    ))
                    .style(Style::default().fg(Color::DarkGray)),
                    *area,
                );
            }
        }
    }
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[4]);
}

fn render_trainer_detail(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    trainer: &TrainerDetail,
    area: Rect,
) {
    let title = format!(
        " {} @ {}{} ",
        short(&trainer.job_id, 44),
        trainer.node,
        stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
    );
    let block = Block::default().title(title).borders(Borders::ALL);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(4)])
        .split(inner);
    let epoch_ratio = if trainer.epochs > 0 {
        (trainer.epoch as f64 / trainer.epochs as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let validating = trainer.phase.starts_with("validation");
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(if validating {
                Color::Magenta
            } else {
                Color::Cyan
            }))
            .ratio(epoch_ratio)
            .label(phase_label(trainer)),
        parts[0],
    );
    let val_history = trainer
        .history
        .iter()
        .map(|point| point.val_loss)
        .collect::<Vec<_>>();
    let trend = match trainer.history.len() {
        len if len >= 2 => {
            let last = trainer.history[len - 1].val_loss;
            let prior = trainer.history[len - 2].val_loss;
            if last < prior {
                "↓"
            } else if last > prior {
                "↑"
            } else {
                "→"
            }
        }
        _ => "-",
    };
    let early_stop = if trainer.early_stopped
        || (trainer.patience > 0 && trainer.bad_passes >= trainer.patience)
    {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
        Span::styled(
            format!(" bad {}/{} ", trainer.bad_passes, trainer.patience),
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    } else {
        Span::raw(format!(
            "bad {}/{}",
            trainer.bad_passes,
            if trainer.patience > 0 {
                trainer.patience.to_string()
            } else {
                "?".to_owned()
            }
        ))
    };
    let last_point = trainer.history.last();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "loss {} {}  val {} {}  gap {}",
                trainer
                    .running_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                trend,
                trainer
                    .val_loss
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "-".to_owned()),
                sparkline(&val_history),
                trainer
                    .overfit_gap
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_owned()),
            )),
            Line::from(vec![
                early_stop,
                Span::raw(format!(
                    "  pairwise {}  binary {}",
                    last_point
                        .and_then(|point| point.val_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    last_point
                        .and_then(|point| point.val_binary_pairwise)
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                )),
            ]),
            Line::from(format!(
                "watermark {}  elapsed {:.0}m{}",
                human_count(trainer.label_watermark),
                trainer.elapsed_sec / 60.0,
                if trainer.is_trainer {
                    ""
                } else {
                    "  (generic GPU job)"
                }
            )),
        ]),
        parts[1],
    );
}

fn render_gameplay_tab(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_four_task_scoreboard(frame, dashboard, rows[1]);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[2]);
    if let Some(progress) = &dashboard.wave_progress {
        let ratio = if progress.manifest_count > 0 {
            (progress.receipts_landed as f64 / progress.manifest_count as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let claims = if progress.claims_by_state.is_empty() {
            String::new()
        } else {
            format!(
                " • claims {}",
                progress
                    .claims_by_state
                    .iter()
                    .map(|(state, count)| format!("{state}:{count}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .title(format!(" {} (receipts since TUI start) ", progress.wave))
                        .borders(Borders::ALL),
                )
                .gauge_style(Style::default().fg(Color::Cyan))
                .ratio(ratio)
                .label(format!(
                    "{}/{} manifests • {} datasets{}",
                    progress.receipts_landed, progress.manifest_count, progress.datasets, claims
                )),
            top[0],
        );
    } else {
        frame.render_widget(
            Paragraph::new("no wave summary")
                .block(Block::default().title(" Wave ").borders(Borders::ALL)),
            top[0],
        );
    }
    let stats = &dashboard.gameplay_stats;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "manifests seen {} • DL top-1 {} / GP top-1 {} (window {})",
                stats.manifests_seen, stats.dl_top1, stats.gp_top1, GAMEPLAY_STATS_WINDOW
            )),
            Line::from(format!(
                "DL slot share {:.1}% • Δ incumbent N/A (typed context authority unavailable)",
                stats.dl_slot_share_pct
            )),
        ])
        .block(
            Block::default()
                .title(" DL vs GP (rolling) ")
                .borders(Borders::ALL),
        ),
        top[1],
    );

    let wins = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::StrictWin)
        .take(5)
        .map(|event| {
            let (label, _) = origin_label(event.origin);
            ListItem::new(format!(
                "{} {} {} Δ{} [{}]",
                event.task,
                short(&event.dataset_hash, 16),
                event
                    .score
                    .map(|value| format!("{value:.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                event
                    .delta_vs_incumbent
                    .map(|value| format!("{value:+.5}"))
                    .unwrap_or_else(|| "-".to_owned()),
                label
            ))
            .style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect::<Vec<_>>();
    let wins = if wins.is_empty() {
        vec![ListItem::new("no strict wins observed since TUI start")
            .style(Style::default().fg(Color::DarkGray))]
    } else {
        wins
    };
    frame.render_widget(
        List::new(wins).block(
            Block::default()
                .title(" Strict wins (live) ")
                .borders(Borders::ALL),
        ),
        rows[3],
    );

    let feed_title = format!(
        " Evaluation feed{} ",
        stale_suffix(dashboard.collected_at_ms, dashboard.feed_fetched_at_ms)
    );
    let header = Row::new([
        "time",
        "T",
        "dataset",
        "score",
        "Δ incumbent",
        "origin",
        "node",
        "wave",
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let feed_rows = dashboard
        .gameplay
        .iter()
        .filter(|event| event.kind == EventKind::Eval)
        .take(rows[4].height.saturating_sub(3) as usize)
        .map(|event| {
            let (label, color) = origin_label(event.origin);
            let delta_cell = event
                .delta_vs_incumbent
                .map(|value| format!("{value:+.5}"))
                .unwrap_or_else(|| "-".to_owned());
            let style = if !event.valid {
                Style::default().fg(Color::DarkGray)
            } else if event.delta_vs_incumbent.is_some_and(|value| value > 0.0) {
                Style::default().fg(Color::Green)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(clock_string(event.at_ms)),
                Cell::from(match event.task.as_str() {
                    "regression" => "R",
                    "binary" => "B",
                    "timeseries" => "T",
                    "multiclass" | "multi_label" => "M",
                    _ => "?",
                }),
                Cell::from(short(&event.dataset_hash, 18)),
                Cell::from(
                    event
                        .score
                        .map(|value| format!("{value:.5}"))
                        .unwrap_or_else(|| "invalid".to_owned()),
                ),
                Cell::from(delta_cell),
                Cell::from(Span::styled(label, Style::default().fg(color))),
                Cell::from(event.node.clone()),
                Cell::from(short(
                    event
                        .wave
                        .strip_prefix("strict-wave-")
                        .unwrap_or(&event.wave),
                    8,
                )),
            ])
            .style(style)
        });
    frame.render_widget(
        Table::new(
            feed_rows,
            [
                Constraint::Length(9),
                Constraint::Length(2),
                Constraint::Length(19),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(7),
                Constraint::Length(7),
                Constraint::Length(8),
            ],
        )
        .header(header)
        .block(Block::default().title(feed_title).borders(Borders::ALL)),
        rows[4],
    );
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[5]);
}

pub(super) fn phase_label(trainer: &TrainerDetail) -> String {
    if trainer.phase.starts_with("validation") {
        format!(
            "epoch {}/{} • VALIDATION {}/{} • training batches done",
            trainer.epoch, trainer.epochs, trainer.validation_batch, trainer.validation_batches
        )
    } else {
        format!(
            "epoch {}/{} • batch {}/{} ({:.0}%) • {}",
            trainer.epoch,
            trainer.epochs,
            trainer.batch,
            trainer.total_batches,
            trainer.pct,
            trainer.phase
        )
    }
}

pub(super) fn trend_arrow(prior: f64, last: f64) -> &'static str {
    if last < prior {
        "↓"
    } else if last > prior {
        "↑"
    } else {
        "→"
    }
}

// The per-epoch chart series (train/val loss, pairwise, ablation) are produced
// by `continuous_series` from the cross-generation `DlHistory`; the old
// single-run extractors keyed on one trainer's history were what emptied the
// chart on job turnover and have been removed.

/// Training arms in a stable display order: real trainers first (so a proposal
/// or encoder GPU job never steals arm 0), each keeping its collector ordering.
pub(super) fn training_arms(dashboard: &Dashboard) -> Vec<&TrainerDetail> {
    let mut arms: Vec<&TrainerDetail> =
        dashboard.trainers.iter().filter(|t| t.is_trainer).collect();
    arms.extend(dashboard.trainers.iter().filter(|t| !t.is_trainer));
    arms
}

/// Newest metric row for an arm, or None while its metrics file is still empty.
/// Callers must render "warming up" rather than borrowing another arm's numbers.
pub(super) fn arm_latest(trainer: &TrainerDetail) -> Option<&TrainerMetricPoint> {
    trainer.history.last()
}

fn gpu_for_job<'a>(dashboard: &'a Dashboard, job_id: &str) -> Option<&'a GpuRow> {
    dashboard
        .gpus
        .iter()
        .find(|gpu| gpu.owner_job.as_deref() == Some(job_id))
}

fn render_dl_tab(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
    arm: usize,
) {
    let arms = training_arms(dashboard);
    // The comparison strip is sized to the live arm count so a single-arm run
    // keeps the charts as tall as before.
    let strip_height = if arms.len() > 1 {
        arms.len() as u16 + 2
    } else {
        0
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(strip_height),
            Constraint::Min(16),
            Constraint::Length(1),
        ])
        .split(area);
    if strip_height > 0 {
        render_arm_strip(frame, dashboard, &arms, arm, rows[0]);
    }
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(11),
            Constraint::Length(1),
            Constraint::Length(8),
            Constraint::Length(6),
        ])
        .split(columns[0]);
    // The charts are driven by the on-disk generation history, not by whichever
    // job happens to be running, so they keep rendering across trainer
    // changeovers and while a successor is still staging.
    let series = continuous_series(&dashboard.dl_history);
    let live = arms.get(selected_arm(arm, arms.len())).copied();
    render_loss_chart(frame, dashboard, &series, live, left[0]);
    frame.render_widget(
        Paragraph::new(generation_ruler(&series, left[1].width)),
        left[1],
    );
    render_pairwise_chart(frame, &series, left[2]);
    match live {
        Some(trainer) => render_dl_status(frame, trainer, left[3]),
        None => frame.render_widget(
            Paragraph::new(
                "no live trainer — the watchdog submits the next one when labels accumulate",
            )
            .style(Style::default().fg(Color::Yellow))
            .block(Block::default().title(" LIVE RUN ").borders(Borders::ALL)),
            left[3],
        ),
    }
    render_dl_global(frame, dashboard, columns[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[2]);
}

/// Wrap the raw key-press counter onto the live arm count. Kept separate so the
/// tests can pin the wrap behaviour when arms appear or exit mid-run.
pub(super) fn selected_arm(arm: usize, count: usize) -> usize {
    if count == 0 {
        0
    } else {
        arm % count
    }
}

/// One line per arm: the at-a-glance comparison across every GPU. Shown only
/// when more than one arm is live.
fn render_arm_strip(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    arms: &[&TrainerDetail],
    arm: usize,
    area: Rect,
) {
    let selected = selected_arm(arm, arms.len());
    let best = arms
        .iter()
        .filter_map(|t| arm_latest(t).and_then(|p| p.val_conditional_advantage))
        .fold(f64::NEG_INFINITY, f64::max);
    let lines = arms
        .iter()
        .enumerate()
        .map(|(index, trainer)| {
            let marker = if index == selected { "▶" } else { " " };
            let node = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!("{} {}", trainer.node, gpu.model_hint))
                .unwrap_or_else(|| trainer.node.clone());
            let Some(point) = arm_latest(trainer) else {
                return Line::from(vec![
                    Span::raw(format!(
                        "{marker} {:<22} {:<14} ",
                        short(&trainer.job_id, 22),
                        node
                    )),
                    Span::styled(
                        "warming up — no metrics yet",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
            };
            let advantage = point.val_conditional_advantage;
            let advantage_style = match advantage {
                Some(value) if value >= best && best > f64::NEG_INFINITY => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                Some(value) if value > 0.0 => Style::default().fg(Color::Green),
                Some(_) => Style::default().fg(Color::Red),
                None => Style::default().fg(Color::DarkGray),
            };
            Line::from(vec![
                Span::raw(format!(
                    "{marker} {:<22} {:<14} ep {:>3}/{:<3} train {:>6.3} val {:>7.3} pw {:>5.3} ",
                    short(&trainer.job_id, 22),
                    node,
                    trainer.epoch,
                    trainer.epochs,
                    point.train_loss,
                    point.val_loss,
                    point.val_pairwise.unwrap_or(f64::NAN),
                )),
                Span::styled(
                    match advantage {
                        Some(value) => format!("cond {value:+.4}"),
                        None => "cond    n/a".to_owned(),
                    },
                    advantage_style,
                ),
                Span::raw(format!(
                    "  stop {}/{}",
                    point.bad_passes.max(trainer.bad_passes),
                    trainer.patience
                )),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(format!(
                    " TRAINING ARMS ({}) — n/←/→ to select ",
                    arms.len()
                ))
                .borders(Borders::ALL),
        ),
        area,
    );
}

/// Cross-generation loss chart. Takes the flattened history rather than a live
/// trainer so it renders whether or not a job is currently running.
fn render_loss_chart(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    series: &DlSeries,
    trainer: Option<&TrainerDetail>,
    area: Rect,
) {
    let (train, val) = (&series.train, &series.val);
    let all = train.iter().chain(val.iter()).map(|(_, y)| *y);
    let y_max = all.clone().fold(f64::MIN, f64::max);
    let y_min = all.fold(f64::MAX, f64::min);
    let (y_min, y_max) = if y_min > y_max {
        (0.0, 1.0)
    } else {
        ((y_min * 0.9).max(0.0), (y_max * 1.1).max(y_min + 0.1))
    };
    let x_min = train
        .first()
        .or(val.first())
        .map(|(x, _)| *x)
        .unwrap_or(0.0);
    let x_max = train
        .last()
        .or(val.last())
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    // Unnamed datasets stay out of the legend, so the boundaries read as plain
    // rules between generations.
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_min), (*x - 0.5, y_max)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name(format!(
                "train {}",
                decimal(
                    trainer
                        .and_then(|trainer| trainer.running_loss)
                        .or_else(|| train.last().map(|(_, y)| *y))
                )
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(train),
    );
    datasets.push(
        Dataset::default()
            .name(format!("val {}", decimal(val.last().map(|(_, y)| *y))))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Red))
            .data(val),
    );
    let arms = training_arms(dashboard);
    let live_label = match trainer {
        Some(trainer) => {
            let arm_label = arms
                .iter()
                .position(|candidate| candidate.job_id == trainer.job_id)
                .filter(|_| arms.len() > 1)
                .map(|index| format!(" [arm {}/{}]", index + 1, arms.len()))
                .unwrap_or_default();
            let gpu_label = gpu_for_job(dashboard, &trainer.job_id)
                .map(|gpu| format!(" {}", gpu.model_hint))
                .unwrap_or_default();
            format!(
                "live {} @ {}{}{}{}",
                short(&trainer.job_id, 34),
                trainer.node,
                gpu_label,
                arm_label,
                stale_suffix(dashboard.collected_at_ms, trainer.progress_at_ms)
            )
        }
        None => "no live trainer".to_owned(),
    };
    let title = format!(
        " HISTORICAL LOSS • {}{} ",
        live_label,
        stale_suffix(
            dashboard.collected_at_ms,
            dashboard.dl_history.fetched_at_ms
        )
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(Block::default().title(title).borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .title("epoch")
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max])
                    .labels(vec![
                        format!("{x_min:.0}"),
                        format!("{:.0}", (x_min + x_max) / 2.0),
                        format!("{x_max:.0}"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        format!("{y_min:.2}"),
                        format!("{:.2}", (y_min + y_max) / 2.0),
                        format!("{y_max:.2}"),
                    ]),
            ),
        area,
    );
}

/// Generation ruler drawn directly under the loss chart: one tick per trainer
/// generation, positioned proportionally, so the boundaries are readable rather
/// than merely visible. The live generation is highlighted.
pub(super) fn generation_ruler(series: &DlSeries, width: u16) -> Line<'static> {
    if series.marks.is_empty() || width < 8 {
        return Line::from(Span::styled(
            "no trainer generations on disk yet",
            Style::default().fg(Color::DarkGray),
        ));
    }
    // Match the chart's plotting area: 1 col of left border + 1 of right.
    let inner = width.saturating_sub(2) as usize;
    let span_epochs = series.total_epochs.max(1.0);
    let mut spans = vec![Span::raw(" ")];
    let mut cursor = 0usize;
    for (x, label, live) in &series.marks {
        let column = (((*x - 1.0) / span_epochs) * inner as f64).round() as usize;
        let text = format!("├{label}");
        if column + text.chars().count() > inner {
            continue;
        }
        if column < cursor {
            continue;
        }
        spans.push(Span::raw(" ".repeat(column - cursor)));
        let style = if *live {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::styled(text.clone(), style));
        cursor = column + text.chars().count();
    }
    Line::from(spans)
}

/// Cross-generation pairwise-win chart (the "should climb" curve).
fn render_pairwise_chart(frame: &mut Frame<'_>, series: &DlSeries, area: Rect) {
    let points = series.pairwise.clone();
    let ablation = series.ablation.clone();
    let x_min = points.first().map(|(x, _)| *x).unwrap_or(0.0);
    let x_max = points
        .last()
        .map(|(x, _)| *x)
        .unwrap_or(1.0)
        .max(x_min + 1.0);
    let y_low = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MAX, f64::min)
        .min(0.48);
    let y_high = points
        .iter()
        .chain(ablation.iter())
        .map(|(_, y)| *y)
        .fold(f64::MIN, f64::max)
        .max(0.55);
    let (y_low, y_high) = if y_low > y_high {
        (0.4, 0.8)
    } else {
        ((y_low - 0.02).max(0.0), (y_high + 0.02).min(1.0))
    };
    let chance = vec![(x_min, 0.5), (x_max, 0.5)];
    let latest = points.last().map(|(_, y)| *y);
    let boundaries = series
        .marks
        .iter()
        .skip(1)
        .map(|(x, _, _)| vec![(*x - 0.5, y_low), (*x - 0.5, y_high)])
        .collect::<Vec<_>>();
    let mut datasets = boundaries
        .iter()
        .map(|line| {
            Dataset::default()
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(line)
        })
        .collect::<Vec<_>>();
    datasets.push(
        Dataset::default()
            .name("chance 0.5")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::DarkGray))
            .data(&chance),
    );
    datasets.push(
        Dataset::default()
            .name(format!(
                "ablation(control) {}",
                decimal(ablation.last().map(|(_, y)| *y))
            ))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Magenta))
            .data(&ablation),
    );
    datasets.push(
        Dataset::default()
            .name(format!("pairwise {}", decimal(latest)))
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&points),
    );
    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::default()
                    .title(" HISTORICAL VAL PAIRWISE-WIN — all generations ")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([x_min, x_max]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::DarkGray))
                    .bounds([y_low, y_high])
                    .labels(vec![
                        format!("{y_low:.2}"),
                        "0.50".to_owned(),
                        format!("{y_high:.2}"),
                    ]),
            ),
        area,
    );
}

fn render_dl_status(frame: &mut Frame<'_>, trainer: &TrainerDetail, area: Rect) {
    let history = &trainer.history;
    let gap_trend = if history.len() >= 2 {
        trend_arrow(
            history[history.len() - 2].overfit_gap,
            history[history.len() - 1].overfit_gap,
        )
    } else {
        "-"
    };
    let validating = trainer.phase.starts_with("validation");
    let epoch_minutes = if trainer.epoch > 0 {
        trainer.elapsed_sec / 60.0 / trainer.epoch as f64
    } else {
        0.0
    };
    let early = if trainer.patience > 0 && trainer.bad_passes >= trainer.patience {
        Span::styled(
            " EARLY-STOPPED ",
            Style::default().fg(Color::Black).bg(Color::Red),
        )
    } else {
        Span::styled(
            format!(
                " early-stop {}/{} ",
                trainer.bad_passes,
                trainer.patience.max(1)
            ),
            if trainer.patience > 0 && trainer.bad_passes * 4 >= trainer.patience * 3 {
                Style::default().fg(Color::Black).bg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            },
        )
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![Span::styled(
                phase_label(trainer),
                Style::default().fg(if validating { Color::Magenta } else { Color::Cyan }),
            )]),
            Line::from(vec![
                early,
                Span::raw(format!(
                    "  overfit-gap {} {}  •  ~{epoch_minutes:.0} min/epoch  •  elapsed {:.0}m",
                    decimal(trainer.overfit_gap),
                    gap_trend,
                    trainer.elapsed_sec / 60.0,
                )),
            ]),
            // Conditional advantage is the headline: at or below zero the model
            // is not beating the genome-only control, so a healthy-looking
            // pairwise number means nothing.
            match arm_latest(trainer).map(|point| point.val_conditional_advantage) {
                Some(Some(value)) => Line::from(vec![
                    Span::raw("conditional advantage "),
                    Span::styled(
                        format!("{value:+.4}"),
                        if value > 0.0 {
                            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                        },
                    ),
                    Span::styled(
                        if value > 0.0 {
                            "  (beating the genome-only control)"
                        } else {
                            "  (NOT using the dataset embedding)"
                        },
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Some(None) => Line::from(Span::styled(
                    "conditional advantage n/a — metric absent from this run",
                    Style::default().fg(Color::DarkGray),
                )),
                None => Line::from(Span::styled(
                    "conditional advantage — warming up, no metrics yet",
                    Style::default().fg(Color::DarkGray),
                )),
            },
            Line::from(format!(
                "watermark {}  •  progress heartbeat is LIVE during validation — a full sweep takes minutes",
                human_count(trainer.label_watermark)
            )),
        ])
        .block(Block::default().title(" TRAINER STATE ").borders(Borders::ALL)),
        area,
    );
}

fn render_dl_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
    // Two lines per task (numbers + stacked bar) plus borders. Hard-coding 4
    // truncated the BINARY row off the bottom, so the panel only ever showed
    // regression — both task totals have to be visible.
    let funnel_height = (dashboard.funnel.rows.len().max(1) * 2 + 2).min(10) as u16;
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(7),
            Constraint::Length(3),
            Constraint::Length(funnel_height),
            Constraint::Length(9),
            Constraint::Length(3),
            Constraint::Length(GENERATIONS_MAX as u16 + SEEDING_CURVE_ROWS as u16 + 3),
        ])
        .split(area);
    let alert_items = if dashboard.alerts.is_empty() {
        vec![ListItem::new(Span::styled(
            "all clear",
            Style::default().fg(Color::Green),
        ))]
    } else {
        dashboard
            .alerts
            .iter()
            .map(|alert| {
                let color = if alert.contains("critical") || alert.contains("CRITICAL") {
                    Color::Red
                } else {
                    Color::Yellow
                };
                ListItem::new(Span::styled(alert.clone(), Style::default().fg(color)))
            })
            .collect()
    };
    frame.render_widget(
        List::new(alert_items).block(
            Block::default()
                .title(" GLOBAL ERRORS / ALARMS ")
                .borders(Borders::ALL),
        ),
        parts[0],
    );
    let heartbeat_lines = dashboard
        .daemon_ages
        .iter()
        .map(|(name, age, expected)| {
            let (text, color) = if *age == u64::MAX {
                ("MISSING".to_owned(), Color::Red)
            } else if *age > expected * 2 {
                (format!("{age}s (expect ~{expected}s)"), Color::Red)
            } else if *age > *expected {
                (format!("{age}s (expect ~{expected}s)"), Color::Yellow)
            } else {
                (format!("{age}s"), Color::Green)
            };
            Line::from(vec![
                Span::raw(format!("{name:<24} ")),
                Span::styled(text, Style::default().fg(color)),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(heartbeat_lines).block(
            Block::default()
                .title(" DAEMON HEARTBEATS ")
                .borders(Borders::ALL),
        ),
        parts[1],
    );
    frame.render_widget(
        Paragraph::new(format!(
            "failed {}  •  {} jobs/min  •  cluster CPU {:.0}% (target {:.0}%)",
            human_count(dashboard.cluster.jobs_failed),
            dashboard.cluster.throughput_per_minute,
            dashboard.cluster.cpu_busy_pct,
            CPU_TARGET_PCT
        ))
        .block(Block::default().title(" SCQ ").borders(Borders::ALL)),
        parts[2],
    );
    let funnel_lines = if dashboard.funnel.rows.is_empty() {
        vec![Line::from("collecting…")]
    } else {
        let bar_width = (parts[3].width.saturating_sub(4) as usize).clamp(10, 60);
        dashboard
            .funnel
            .rows
            .iter()
            .flat_map(|row| [funnel_line(row), funnel_bar(row, bar_width)])
            .collect()
    };
    frame.render_widget(
        Paragraph::new(funnel_lines).block(
            Block::default()
                .title(format!(
                    " DATASET UNLOCK ({}){} ",
                    dashboard.funnel.source,
                    stale_suffix(now_ms(), dashboard.funnel.fetched_at_ms)
                ))
                .borders(Borders::ALL),
        ),
        parts[3],
    );
    let trainer = dashboard
        .trainers
        .iter()
        .find(|trainer| trainer.is_trainer)
        .or_else(|| dashboard.trainers.first());
    let metric_lines = trainer
        .map(|trainer| {
            let history = &trainer.history;
            let last = history.last();
            let prior = history
                .len()
                .checked_sub(2)
                .and_then(|index| history.get(index));
            let row = |label: &str, get: &dyn Fn(&TrainerMetricPoint) -> Option<f64>| {
                let latest = last.and_then(get);
                let arrow = match (prior.and_then(get), latest) {
                    (Some(previous), Some(current)) => trend_arrow(previous, current),
                    _ => "-",
                };
                Line::from(format!("{label:<22} {}  {arrow}", decimal(latest)))
            };
            vec![
                row("val_pairwise_win", &|point| point.val_pairwise),
                row("val_regression_win", &|point| point.val_regression_pairwise),
                row("val_binary_win", &|point| point.val_binary_pairwise),
                row("reward_cv_corr", &|point| point.val_reward_cv),
                row("top1_regret (↓ good)", &|point| point.val_top1_regret),
                row("overfit_gap (↓ good)", &|point| Some(point.overfit_gap)),
                row("val_loss (↓ good)", &|point| Some(point.val_loss)),
            ]
        })
        .unwrap_or_else(|| vec![Line::from("no trainer history yet")]);
    frame.render_widget(
        Paragraph::new(metric_lines).block(
            Block::default()
                .title(" VAL METRICS (vs prior epoch) ")
                .borders(Borders::ALL),
        ),
        parts[4],
    );
    let labels_line = dashboard
        .watchdog
        .as_ref()
        .map(|watchdog| {
            format!(
                "labels {}  •  pending {}  •  next trainer at {}",
                human_count(watchdog.labels),
                human_count(watchdog.pending_unfrozen_labels),
                human_count(watchdog.next_label_threshold)
            )
        })
        .unwrap_or_else(|| "watchdog status unavailable".to_owned());
    frame.render_widget(
        Paragraph::new(labels_line).block(
            Block::default()
                .title(" LABEL SUPPLY ")
                .borders(Borders::ALL),
        ),
        parts[5],
    );
    let percent = |value: Option<f64>| {
        value
            .map(|share| format!("{share:.1}%"))
            .unwrap_or_else(|| "-".to_owned())
    };
    let mut generation_lines = vec![Line::from(Span::styled(
        format!(
            "{:<30}  {:>7}  {:>7}  {:>7}",
            "run / seed checkpoint", "val-pw", "DL-win", "DL-obs"
        ),
        Style::default().fg(Color::DarkGray),
    ))];
    if dashboard.generations.is_empty() {
        generation_lines.push(Line::from("collecting…"));
    } else {
        generation_lines.extend(dashboard.generations.iter().map(|(label, pairwise)| {
            Line::from(format!(
                "{:<30}  {:>7}  {:>7}  {:>7}",
                short(label, 30),
                decimal(*pairwise),
                "-",
                "-"
            ))
        }));
    }
    // DL-share column source: the newest seeding-curve checkpoint generations.
    let skip = dashboard
        .seeding
        .points
        .len()
        .saturating_sub(SEEDING_CURVE_ROWS);
    for point in dashboard.seeding.points.iter().skip(skip) {
        let label = format!(
            "ckpt {} ({} wins)",
            point.checkpoint_short, point.strict_wins_total
        );
        let observed = if point.observations_caught_up {
            percent(point.dl_obs_share_pct)
        } else {
            format!("~{}", percent(point.dl_obs_share_pct))
        };
        generation_lines.push(Line::from(vec![
            Span::raw(format!("{:<30}  {:>7}  ", short(&label, 30), "-")),
            Span::styled(
                format!("{:>7}", percent(point.dl_win_share_pct)),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(format!("  {observed:>7}")),
        ]));
    }
    frame.render_widget(
        Paragraph::new(generation_lines)
            .block(
                Block::default()
                    .title(format!(
                        " GENERATIONS (val-pw • DL-share, old→new){} ",
                        stale_suffix(now_ms(), dashboard.seeding.fetched_at_ms)
                    ))
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        parts[6],
    );
}

fn render_header(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_global(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_nodes(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_queue(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_learning(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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
pub(super) fn cohort_label(funnel_active_total: u64, wave: &WaveSummary) -> String {
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

fn render_wave(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

pub(super) fn render_alerts(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_events(frame: &mut Frame<'_>, dashboard: &Dashboard, area: Rect) {
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

fn render_compact_gpu_dl(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let now = dashboard.collected_at_ms;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("DIME GPU/DL • {}", dashboard.top_state),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "  sources {}/{}  {}",
            dashboard.sources_healthy,
            dashboard.sources_total,
            clock_string(now)
        )),
    ])];
    lines.push(Line::from(dashboard.why_idle.clone()));

    if dashboard.accepted_tasks.is_empty() {
        lines.push(Line::from("fronts: accepted-by-task unavailable"));
    } else {
        lines.extend(dashboard.accepted_tasks.iter().map(|front| {
            Line::from(vec![
                Span::styled(
                    format!("{:<10}", front.task),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "accepted {}  generation {}  age {}s",
                    short_hash(&front.checkpoint_sha256),
                    front.generation,
                    now.saturating_sub(front.updated_at_ms) / 1_000
                )),
            ])
        }));
    }

    lines.push(
        dashboard
            .watchdog
            .as_ref()
            .map(|watchdog| {
                Line::from(format!(
                    "watchdog {} • corpus {} • labels {} • gap {} • ckpt {}",
                    watchdog.action,
                    human_count(watchdog.snapshot_watermark),
                    human_count(watchdog.labels),
                    human_count(watchdog.pending_unfrozen_labels),
                    short_hash(&watchdog.checkpoint_sha256)
                ))
            })
            .unwrap_or_else(|| Line::from("watchdog unavailable")),
    );

    lines.push(Line::from(Span::styled(
        "GPUs",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.gpus.is_empty() {
        lines.push(Line::from("  no GPU telemetry"));
    } else {
        lines.extend(dashboard.gpus.iter().map(|gpu| {
            let active = gpu.owner_job.is_some();
            let role = if gpu.role.is_empty() {
                String::new()
            } else {
                format!("{}#{} ", gpu.role, gpu.attempt)
            };
            let reason = if gpu.admission_reason.is_empty() {
                gpu.owner_job
                    .as_deref()
                    .unwrap_or(dashboard.why_idle.as_str())
            } else {
                gpu.admission_reason.as_str()
            };
            Line::from(vec![
                Span::styled(
                    format!(
                        "  {} {:>3.0}% {:>5}/{:<5} MiB {:>3.0}W ",
                        gpu.node, gpu.util_pct, gpu.mem_used_mb, gpu.mem_total_mb, gpu.power_w
                    ),
                    Style::default().fg(if active {
                        Color::Green
                    } else {
                        Color::DarkGray
                    }),
                ),
                Span::raw(short(
                    &format!(
                        "{role}{} — {reason}",
                        gpu.owner_job.as_deref().unwrap_or("idle")
                    ),
                    area.width.saturating_sub(42) as usize,
                )),
            ])
        }));
    }

    lines.push(Line::from(Span::styled(
        "ACTIVE GPU WORK",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if dashboard.trainers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(Color::Red),
        )));
    } else {
        for trainer in &dashboard.trainers {
            let heartbeat_age = now.saturating_sub(trainer.progress_at_ms) / 1_000;
            let role = if trainer.is_trainer { "TRAIN" } else { "GPU" };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {role:<5} "),
                    Style::default()
                        .fg(if trainer.is_trainer {
                            Color::Green
                        } else {
                            Color::Cyan
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "{} @ {}",
                    short(&trainer.job_id, area.width.saturating_sub(22) as usize),
                    trainer.node
                )),
            ]));
            if trainer.is_trainer {
                lines.push(Line::from(format!(
                    "        {} epoch {}/{} batch {}/{} ({:.1}%) loss {} hb {}s",
                    trainer.phase,
                    trainer.epoch,
                    trainer.epochs,
                    trainer.batch,
                    trainer.total_batches,
                    trainer.pct,
                    trainer
                        .running_loss
                        .map(|value| format!("{value:.4}"))
                        .unwrap_or_else(|| "-".to_owned()),
                    heartbeat_age,
                )));
            }
        }
    }

    lines.push(Line::from(format!(
        "1 overview • 2 gpu/dl • 4 charts • q quit • {}ms refresh / {}ms collect",
        refresh_ms, dashboard.collection_latency_ms
    )));
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(
            Block::default()
                .title(" DL CONTROL PLANE ")
                .borders(Borders::ALL),
        ),
        area,
    );
}

fn render_compact(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, area: Rect) {
    let nodes = dashboard
        .nodes
        .iter()
        .map(|node| {
            format!(
                "{}:{} CPU {} slots {}/{}",
                node.node,
                node.state,
                node.cpu_busy_pct
                    .map(|value| format!("{value:.0}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                node.running_slots,
                node.slot_target
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let mut lines = vec![
        Line::from(format!(
            "DIME UNIFIED • {} • read-only",
            dashboard.top_state
        )),
        Line::from(format!(
            "campaign robust {}/{} of {} • {} rung {} • evals {}/{} • seeds {}",
            dashboard.campaign.robust_wins,
            dashboard.campaign.robust_target,
            dashboard.campaign.robust_denominator,
            dashboard.campaign.stage,
            dashboard.campaign.current_rung,
            dashboard.campaign.evaluations_used,
            dashboard.campaign.evaluations_cap,
            dashboard.campaign.seed_agreement,
        )),
        Line::from(dashboard.why_idle.clone()),
        Line::from(format!(
            "CPU {:.1}% (target 80 / guard 95) PSI {}/{}/{}",
            dashboard.cluster.cpu_busy_pct,
            dashboard.cluster.psi_cpu10,
            dashboard.cluster.psi_mem10,
            dashboard.cluster.psi_io10
        )),
        Line::from(nodes),
        Line::from(format!(
            "queue fresh {}/{} cap {} age {:.0}s epoch {}",
            dashboard.queue.useful_depth,
            dashboard.queue.target,
            dashboard.queue.hard_cap,
            dashboard.queue.maximum_age_seconds,
            short_epoch(&dashboard.queue.epoch)
        )),
        Line::from(format!(
            "checkpoint {} trainer {} [{}] labels {}",
            short_hash(&dashboard.learning.checkpoint),
            short(&dashboard.learning.trainer, 30),
            dashboard.learning.trainer_state,
            human_count(dashboard.learning.labels)
        )),
        Line::from(format!(
            "wave {} adaptive R{}/B{} sealed {}",
            dashboard.wave.active_wave,
            dashboard.wave.adaptive_regression,
            dashboard.wave.adaptive_binary,
            yes_no(dashboard.wave.sealed_isolated)
        )),
    ];
    for task in FOUR_TASK_ORDER {
        let row = dashboard
            .four_task_telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.rows.iter().find(|row| row.task == task));
        lines.push(Line::from(row.map_or_else(
            || {
                format!(
                    "{:>2} N/A — four-task telemetry unavailable",
                    funnel_tag(task.as_str())
                )
            },
            |row| {
                format!(
                    "{:>2} fam {} primary {} error {} flow {}",
                    funnel_tag(task.as_str()),
                    telemetry_count(&row.counts.source_families),
                    telemetry_median(&row.metrics.primary_performance),
                    row.metrics
                        .native_errors
                        .get(task_metric_contract(task).native_error)
                        .map(telemetry_median)
                        .unwrap_or_else(|| "N/A".to_owned()),
                    row.valid_observations_per_minute
                        .value
                        .map(|value| format!("{value:.0}/m"))
                        .unwrap_or_else(|| "N/A".to_owned()),
                )
            },
        )));
    }
    lines.extend([
        Line::from(format!(
            "alerts: {}",
            if dashboard.alerts.is_empty() {
                "none".to_owned()
            } else {
                dashboard.alerts.join(" | ")
            }
        )),
        Line::from(format!(
            "1 overview • 2 gpu/dl • 4 charts • q/esc quit • {}ms refresh • {}ms collect",
            refresh_ms, dashboard.collection_latency_ms
        )),
    ]);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL)),
        area,
    );
}
