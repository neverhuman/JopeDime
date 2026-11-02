use super::campaign::{goal_phase, gpu_reason, milestone_lines};
use super::details::{
    binary_replay_detail, comparison_lines, decimal_count, learning_chart, neural_lines,
    teacher_evolution_lines, teacher_lines,
};
use super::evidence::{
    age, completed, decimal, duration, metrics, number, panel, phase, rmse, state, status_age,
};
use super::layout::{compact_layout, render_compact, DashboardData};
use super::prelude::*;
use super::tasks::{label, policy_task_state, workflow_rows, TaskRow};

pub fn render(f: &mut Frame, view: usize, h: &Hardware, data: DashboardData<'_>) {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let observed_gp = crate::telemetry::observed_status(gp, now());
    let gp = &observed_gp;
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(9, 17, 28)).fg(INK)),
        f.area(),
    );
    if f.area().width < 48 || f.area().height < 15 {
        f.render_widget(
            Paragraph::new(
                "JOPE / PARALLEL\nResize to at least 48 × 15.\nq detaches; services keep running.",
            )
            .block(panel(" Terminal size ")),
            f.area(),
        );
        return;
    }
    if compact_layout(f.area()) {
        render_compact(
            f,
            view,
            h,
            crate::dashboard::DashboardData {
                gp,
                neural,
                run,
                events,
                history,
            },
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());
    let stale = h.timestamp == 0 || now().saturating_sub(h.timestamp) > 3;
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" JOPE / PARALLEL  ", Style::default().fg(BLUE).bold()),
            Span::raw(format!(
                "8 × A100  ·  {} CPUs  ·  {}",
                h.cpu.len(),
                goal_phase(run, neural, gp)
            )),
        ]))
        .block(panel(
            " LEARNING OPERATIONS · live receipts, saved evidence ",
        )),
        areas[0],
    );
    f.render_widget(
        Tabs::new([
            "1 Overview",
            "2 Hardware",
            "3 Deep learning",
            "4 GP search",
            "5 Data / Events",
        ])
        .select(view)
        .highlight_style(Style::default().fg(Color::Rgb(9, 17, 28)).bg(BLUE).bold())
        .style(Style::default().fg(MUTED))
        .block(panel(" Navigate ")),
        areas[1],
    );
    let body = areas[2];
    match view {
        0 => {
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
            let gpu =
                h.gpus.iter().map(|g| g.utilization).sum::<f64>() / h.gpus.len().max(1) as f64;
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
            f.render_widget(Paragraph::new(format!("{}\nRAM available {:.0} GiB · disk free {:.0} GiB · 4 CPUs reserved for host services",gpu_reason(run,neural,h),h.available_gb,h.disk_free_gb)).wrap(Wrap{trim:true}).block(panel(" Resource assignment / next work ")),rows[4]);
        }
        1 => {
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
            let bands =
                Layout::horizontal([Constraint::Percentage(36), Constraint::Percentage(64)])
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
        2 if !history.learning_details => {
            crate::learning_history::render(
                f,
                body,
                &history.learning,
                &history.learning_navigation,
                run,
            );
        }
        2 => {
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
        3 => {
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
            let evidence =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
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
        _ => {
            let mut lines = vec![
                label("Campaign", goal_phase(run,neural,gp)),
                label(
                    "Pilot launched",
                    if run["pilot_started"] == true {
                        "yes"
                    } else {
                        "no"
                    },
                ),
                label("Context coverage",format!("{} saved views; {} source families",number(&run["contexts"],"completed"),number(&run["contexts"],"families"))),
                label("Prepared workload",format!("{} distinct cut problems from {} datasets",number(&run["preparation"],"views"),number(&run["preparation"],"datasets"))),
                label("Cut storage","source pointers + exact indices + preprocessing recipes"),
                label("Final tests","sealed; teaching held-out scores are adaptive"),
                label("Family preparation", format!("train {} / 48 · validation {} / 8 · test {} / 8 eligible families", number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_train"), number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_validation"), number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_test"))),
                label("Encoding pipeline", format!("{} · {}", phase(&run["encoding_pipeline"]), status_age(&run["encoding_pipeline"]))),
                label("Next dependency",run["encoding_pipeline"]["reason"].as_str().unwrap_or("No queued encoding work receipt")),
                label("Qualification","complete real-data inventory and final evaluation remain pilot gates"),
                Line::raw(run["cycle"]["reason"].as_str().unwrap_or("All displayed progress comes from saved manifests and live worker receipts.").to_string()),
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
    }
    f.render_widget(
        Paragraph::new(if stale {
            " TELEMETRY STALE / UNAVAILABLE · q detaches; workers keep running"
        } else if h.gpus.len() != 8 || h.cpu.is_empty() {
            " PARTIAL TELEMETRY · inspect Hardware / Events · q detaches; workers keep running"
        } else {
            " LIVE · 1 Hz telemetry / 4 fps · ← → or 1–5 navigate · q detaches"
        })
        .style(Style::default().fg(if stale {
            Color::Yellow
        } else {
            Color::DarkGray
        })),
        areas[3],
    );
}
