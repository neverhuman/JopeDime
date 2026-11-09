use crate::model::*;
use ratatui::{prelude::*, widgets::*};

const BG: Color = Color::Rgb(9, 14, 27);
const PANEL: Color = Color::Rgb(14, 23, 41);
const EDGE: Color = Color::Rgb(39, 57, 80);
const TEXT: Color = Color::Rgb(217, 231, 245);
const DIM: Color = Color::Rgb(139, 160, 184);
const CYAN: Color = Color::Rgb(47, 221, 237);
const GREEN: Color = Color::Rgb(110, 239, 166);
const AMBER: Color = Color::Rgb(255, 201, 100);
const PINK: Color = Color::Rgb(245, 126, 194);
const RED: Color = Color::Rgb(255, 113, 126);

#[derive(Default)]
pub struct Ui {
    pub detail_scroll: u16,
    pub anchor: Option<String>,
    pub selected: usize,
    pub offset: usize,
    pub query: String,
    pub searching: bool,
    pub paused: bool,
    pub help: bool,
    pub failures: bool,
    pub quality: bool,
}
fn panel(title: impl Into<String>, c: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(EDGE))
        .title(Line::from(title.into()).style(Style::default().fg(c).bold()))
        .style(Style::default().bg(PANEL).fg(TEXT))
}
pub fn filtered<'a>(s: &'a Snapshot, u: &Ui) -> Vec<&'a View> {
    let q = u.query.to_lowercase();
    let mut rows: Vec<_> = s
        .views
        .iter()
        .filter(|v| {
            (!u.failures
                || !["trained", "measured", "running", "queued"].contains(&v.status.as_str()))
                && (q.is_empty()
                    || format!("{} {} {} {}", v.name, v.id, v.task, v.fold)
                        .to_lowercase()
                        .contains(&q))
        })
        .collect();
    rows.sort_by(|a, b| {
        b.updated_utc_us
            .cmp(&a.updated_utc_us)
            .then_with(|| a.id.cmp(&b.id))
    });
    rows
}
fn tile(f: &mut Frame, r: Rect, title: &str, value: String, sub: String, c: Color) {
    let p = Paragraph::new(vec![
        Line::from(value).style(Style::default().fg(c).bold()),
        Line::from(sub).style(Style::default().fg(DIM)),
    ])
    .block(panel(title, c));
    f.render_widget(p, r);
}
fn ms(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1} ms"))
        .unwrap_or_else(|| "unknown".into())
}
fn graph(f: &mut Frame, r: Rect, title: &str, data: &[u64], c: Color) {
    let last = data.last().copied().unwrap_or(0);
    let peak = data.iter().max().copied().unwrap_or(0);
    let shown = &data[data
        .len()
        .saturating_sub(r.width.saturating_sub(2) as usize)..];
    f.render_widget(
        Sparkline::default()
            .block(panel(format!(" {title}  {last}/s  ·  60s peak {peak} "), c))
            .data(shown)
            .style(Style::default().fg(c))
            .max(peak.max(1)),
        r,
    );
}
pub fn draw(f: &mut Frame, s: &Snapshot, u: &mut Ui, error: Option<&str>, now: i64) {
    let area = f.area();
    f.render_widget(
        Block::default().style(Style::default().bg(BG).fg(TEXT)),
        area,
    );
    if area.width < 72 || area.height < 24 {
        f.render_widget(Paragraph::new("JOPE / LAMBDA WATCH\nResize to at least 72 × 24.\nq quits; training is unaffected.").style(Style::default().fg(CYAN)),area);
        return;
    }
    let compact = area.height < 32;
    let main = Layout::vertical([
        Constraint::Length(if compact { 2 } else { 3 }),
        Constraint::Length(4),
        Constraint::Length(if compact { 3 } else { 5 }),
        Constraint::Min(5),
        Constraint::Length(if compact { 5 } else { 7 }),
        Constraint::Length(2),
    ])
    .margin(1)
    .split(area);
    let (health, live) = s.health(now);
    let state = if u.paused {
        "PAUSED VIEW"
    } else if error.is_some() {
        if s.schema.is_empty() {
            "WAITING"
        } else {
            "FEED ERROR"
        }
    } else {
        health
    };
    let hc = if error.is_some() {
        RED
    } else if live {
        GREEN
    } else {
        AMBER
    };
    let age = ((now - s.generated_utc_us).max(0) as f64) / 1e6;
    f.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(" JOPE ", Style::default().fg(BG).bg(CYAN).bold()),
                Span::styled("  LAMBDA WATCH  ", Style::default().fg(TEXT).bold()),
                Span::styled(format!(" {state} "), Style::default().fg(BG).bg(hc).bold()),
                Span::styled(
                    format!(
                        "  {}{}",
                        if state == s.mode.to_uppercase() {
                            String::new()
                        } else {
                            format!("{} · ", clean(&s.mode).to_uppercase())
                        },
                        clean(&s.campaign)
                    ),
                    Style::default().fg(DIM),
                ),
            ]),
            Line::from(format!(
                " {}  ·  feed {:.1}s  ·  event {:.1}s ago  ·  {}",
                clock(s.cursor_utc_us),
                age,
                (s.cursor_utc_us - s.last_event_utc_us).max(0) as f64 / 1e6,
                clean(&s.scope)
            ))
            .style(Style::default().fg(DIM)),
        ]),
        main[0],
    );
    let ts = Layout::horizontal([Constraint::Ratio(1, 4); 4]).split(main[1]);
    tile(
        f,
        ts[0],
        " IN FLIGHT ",
        count(s.counts.in_flight),
        clean(&s.activity_label),
        CYAN,
    );
    tile(
        f,
        ts[1],
        " DATASET EVENTS ",
        count(s.counts.jobs),
        format!(
            "{} failed · {} fits",
            s.counts.job_failures,
            count(s.counts.candidates)
        ),
        GREEN,
    );
    tile(
        f,
        ts[2],
        " LAMBDA OBSERVED ",
        format!("\u{24}{:.6}", s.budget.lambda_usd),
        format!("{} missing bills", s.counts.unknown_bills),
        PINK,
    );
    tile(
        f,
        ts[3],
        if s.budget.reserved_usd.is_some_and(|x| x > s.budget.cap_usd) {
            " CAP EXCEEDED "
        } else {
            " CAP REMAINING "
        },
        s.budget
            .reserved_usd
            .map(|x| format!("\u{24}{:.2}", (s.budget.cap_usd - x).max(0.0)))
            .unwrap_or_else(|| "UNKNOWN".into()),
        format!("\u{24}{:.0} cap · reservations", s.budget.cap_usd),
        if s.budget.reserved_usd.is_some_and(|x| x > s.budget.cap_usd) {
            RED
        } else {
            AMBER
        },
    );
    let gs = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(main[2]);
    graph(f, gs[0], "CALL COMPLETIONS", &s.rates.calls, CYAN);
    graph(f, gs[1], "DATASET COMPLETIONS", &s.rates.jobs, GREEN);
    graph(f, gs[2], "CANDIDATE FITS", &s.rates.candidates, PINK);
    let body = Layout::horizontal([
        Constraint::Percentage(if area.width < 110 { 100 } else { 70 }),
        Constraint::Percentage(if area.width < 110 { 0 } else { 30 }),
    ])
    .split(main[3]);
    let rows = filtered(s, u);
    if let Some(id) = &u.anchor {
        if let Some(index) = rows.iter().position(|v| &v.id == id) {
            u.selected = index;
        }
    }
    u.selected = u.selected.min(rows.len().saturating_sub(1));
    u.anchor = rows.get(u.selected).map(|v| v.id.clone());
    let visible = body[0].height.saturating_sub(4) as usize;
    if u.selected < u.offset {
        u.offset = u.selected;
    }
    if visible > 0 && u.selected >= u.offset + visible {
        u.offset = u.selected + 1 - visible;
    }
    let rendered = rows
        .iter()
        .enumerate()
        .skip(u.offset)
        .take(visible)
        .map(|(i, v)| {
            let c = match v.status.as_str() {
                "trained" | "measured" => GREEN,
                "running" => CYAN,
                "queued" => DIM,
                _ => RED,
            };
            let score = v
                .score
                .map(|x| format!("{x:.5}"))
                .unwrap_or_else(|| "—".into());
            Row::new(vec![
                clean(&v.name),
                clean(&v.fold),
                v.generation
                    .map(|n| {
                        v.target_generation
                            .map(|target| format!("{n}/{target}"))
                            .unwrap_or_else(|| format!("g{n}"))
                    })
                    .unwrap_or_else(|| "—".into()),
                clean(&v.metric),
                score,
                clean(&v.status),
            ])
            .style(if i == u.selected {
                Style::default().fg(TEXT).bg(Color::Rgb(29, 61, 82)).bold()
            } else {
                Style::default().fg(c)
            })
        });
    let title = format!(
        " DATASET / FOLD  ·  {} views  {} ",
        rows.len(),
        if u.failures { "[failures]" } else { "" }
    );
    let table = Table::new(
        rendered,
        [
            Constraint::Percentage(36),
            Constraint::Percentage(14),
            Constraint::Length(7),
            Constraint::Percentage(17),
            Constraint::Length(11),
            Constraint::Length(9),
        ],
    )
    .header(
        Row::new(["DATASET", "FOLD", "GEN", "KPI", "BEST", "STATE"])
            .style(Style::default().fg(DIM)),
    )
    .block(panel(title, CYAN))
    .column_spacing(1);
    f.render_widget(table, body[0]);
    let side = vec![
        Line::from(format!(
            "{} dispatched / {} returned",
            count(s.counts.dispatched),
            count(s.counts.completed)
        )),
        Line::from(format!(
            "{} failed · {} uncertain",
            s.counts.failed, s.counts.unknown
        ))
        .style(
            Style::default().fg(if s.counts.unknown + s.counts.failed > 0 {
                AMBER
            } else {
                GREEN
            }),
        ),
        Line::from(format!("{} verified S3 archives", count(s.counts.archived))),
        Line::from(format!("p50 {}  / p95 {}", ms(s.p50_ms), ms(s.p95_ms)))
            .style(Style::default().fg(CYAN)),
        Line::from(format!("S3 requests est. \u{24}{:.6}", s.budget.s3_usd)),
        Line::from(""),
        Line::from("MEMORY TIER  /  CALLS  /  PEAK").style(Style::default().fg(DIM)),
    ];
    let mut side = side;
    for t in &s.tiers {
        side.push(
            Line::from(format!(
                "{:>4} MiB  {:>7}  {:>4} MiB",
                t.mb, t.calls, t.peak_mb
            ))
            .style(Style::default().fg(if t.peak_mb > t.mb * 9 / 10 {
                RED
            } else {
                GREEN
            })),
        );
    }
    side.push(Line::from(""));
    side.push(Line::from(clean(&s.notice)).style(Style::default().fg(AMBER)));
    if let Some(e) = s.errors.last() {
        side.push(Line::from(format!("FEED: {}", clean(e))).style(Style::default().fg(RED)));
    }
    f.render_widget(
        Paragraph::new(side)
            .wrap(Wrap { trim: true })
            .block(panel(" FLIGHT RECORDER ", PINK)),
        body[1],
    );
    if let Some(v) = rows.get(u.selected) {
        let dr = Layout::horizontal([
            Constraint::Percentage(if area.width < 110 { 100 } else { 70 }),
            Constraint::Percentage(if area.width < 110 { 0 } else { 30 }),
        ])
        .split(main[4]);
        let mut lines = vec![
            Line::from(format!(
                "{} · {} · {} rows × {} features · {}",
                clean(&v.name),
                clean(&v.task),
                v.rows,
                v.features,
                clock(v.updated_utc_us)
            ))
            .style(Style::default().fg(CYAN)),
            Line::from(format!(
                "{} · {} · {}",
                clean(&v.context),
                clean(&v.direction),
                clean(&v.program_sha)
            ))
            .style(Style::default().fg(DIM)),
        ];
        if compact {
            lines.truncate(1);
        }
        if u.quality {
            lines.push(
                Line::from(format!(
                    "{} {:?}   baseline {:?}   history {:?}",
                    clean(&v.metric),
                    v.score,
                    v.baseline,
                    v.history
                ))
                .style(Style::default().fg(GREEN)),
            );
        } else {
            lines.push(Line::from(clean(&v.program)).style(Style::default().fg(GREEN)));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(panel(" SELECTED GP  ·  enter toggles tree / KPI ", GREEN)),
            dr[0],
        );
        let hist: Vec<u64> = if v.history.is_empty() {
            vec![]
        } else {
            let lo = v.history.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = v.history.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            v.history
                .iter()
                .map(|x| (((x - lo) / (hi - lo).max(1e-12)) * 100.0) as u64 + 1)
                .collect()
        };
        f.render_widget(
            Sparkline::default()
                .data(&hist)
                .style(Style::default().fg(AMBER))
                .block(panel(
                    format!(
                        " {} · {} · per-view scale ",
                        clean(&v.metric),
                        if v.direction == "min" {
                            "↓ better"
                        } else {
                            "↑ better"
                        }
                    ),
                    AMBER,
                )),
            dr[1],
        );
    } else {
        f.render_widget(
            Paragraph::new("No matching dataset/fold telemetry yet.")
                .block(panel(" SELECTED GP ", GREEN)),
            main[4],
        );
    }
    let footer = if u.searching {
        format!(" /{}▏  Enter applies · Esc clears", clean(&u.query))
    } else if u.help {
        " q quit  ↑/↓ select  PgUp/PgDn jump  / filter  f failures  Space freeze  Enter tree/KPI  ? help".into()
    } else {
        format!(
            " q quit  ↑/↓ select  / filter  f failures  Space freeze  ? help  │ {}",
            error.map(clean).unwrap_or_else(|| format!(
                "{} · seq {}",
                clean(&s.status),
                s.sequence
            ))
        )
    };
    f.render_widget(Paragraph::new(vec![Line::from(footer).style(Style::default().fg(if error.is_some(){RED}else{DIM})),Line::from(" Observed rates use 1s UTC buckets; arrival can lag execution. This viewer cannot launch or stop training.").style(Style::default().fg(DIM))]),main[5]);
    if u.quality {
        if let Some(v) = rows.get(u.selected) {
            let popup = area.inner(Margin {
                horizontal: 2,
                vertical: 1,
            });
            let lines = vec![
                Line::from(format!(
                    "{} · {} · {}",
                    clean(&v.name),
                    clean(&v.task),
                    clean(&v.fold)
                ))
                .style(Style::default().fg(CYAN).bold()),
                Line::from(format!(
                    "{}: {:?}   baseline {:?}   {} is better",
                    clean(&v.metric),
                    v.score,
                    v.baseline,
                    if v.direction == "min" {
                        "lower"
                    } else {
                        "higher"
                    }
                ))
                .style(Style::default().fg(GREEN)),
                Line::from(format!("View: {}", clean(&v.id))),
                Line::from(format!("Program SHA256: {}", clean(&v.program_sha))),
                Line::from(clean(&v.context)).style(Style::default().fg(AMBER)),
                Line::from(""),
                Line::from("EXECUTABLE GP PROGRAM").style(Style::default().fg(PINK).bold()),
                Line::from(clean(&v.program)).style(Style::default().fg(GREEN)),
                Line::from(""),
                Line::from(format!("KPI history (one view/attempt): {:?}", v.history)),
                Line::from(""),
                Line::from(format!(
                    "Updated {} · epoch {} µs",
                    clock(v.updated_utc_us),
                    v.updated_utc_us
                )),
                Line::from("↑/↓ or PgUp/PgDn scroll · Space freezes · Enter/Esc returns · q quits")
                    .style(Style::default().fg(DIM)),
            ];
            f.render_widget(Clear, popup);
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .scroll((u.detail_scroll, 0))
                    .block(panel(" GP DETAILS ", GREEN)),
                popup,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    #[test]
    fn handles_every_supported_size() {
        for (w, h) in [(40, 12), (72, 24), (100, 32), (160, 50)] {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            t.draw(|f| draw(f, &Snapshot::default(), &mut Ui::default(), None, 0))
                .unwrap();
        }
    }
    #[test]
    fn filters_without_pooling_metrics() {
        let mut s = Snapshot::default();
        s.views = vec![
            View {
                id: "a".into(),
                name: "iris".into(),
                ..Default::default()
            },
            View {
                id: "b".into(),
                name: "houses".into(),
                ..Default::default()
            },
        ];
        let u = Ui {
            query: "IRIS".into(),
            ..Default::default()
        };
        assert_eq!(filtered(&s, &u)[0].id, "a");
    }
}
