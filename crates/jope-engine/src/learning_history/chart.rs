use super::{Navigation, Run, Snapshot};
use ratatui::{
    layout::{Constraint, Layout, Margin, Rect},
    style::{Color, Style, Stylize},
    text::Line,
    widgets::{Block, List, ListItem, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

fn title(text: &str) -> Block<'_> {
    Block::bordered()
        .title(text)
        .title_style(Style::default().fg(Color::Cyan).bold())
}
fn short(path: &Path) -> String {
    let parts = path
        .components()
        .map(|p| p.as_os_str().to_string_lossy())
        .collect::<Vec<_>>();
    parts[parts.len().saturating_sub(
        if path.ends_with("auxiliary/train") || path.ends_with("auxiliary-pack/train") {
            3
        } else {
            2
        },
    )..]
        .join("/")
}
fn one_line(text: &str, width: u16) -> String {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= width as usize {
        return text.into();
    }
    let keep = (width as usize).saturating_sub(1) / 2;
    format!(
        "{}…{}",
        chars[..keep].iter().collect::<String>(),
        chars[chars.len() - keep..].iter().collect::<String>()
    )
}
/// Keep first/last and each horizontal bucket's extrema; never discard a spike
/// merely because the full recorded history exceeds the terminal width.
///
/// The input is not modified. Empty input returns an empty vector; `columns == 0`
/// still uses one bucket when reduction is needed. `bounds` controls bucket
/// placement and should describe the x range shown by the chart.
pub fn envelope(points: &[(f64, f64)], columns: usize, bounds: [f64; 2]) -> Vec<(f64, f64)> {
    if points.len() <= columns.saturating_mul(2) {
        return points.to_vec();
    }
    let mut bins = vec![None::<((f64, f64), (f64, f64))>; columns.max(1)];
    for &(x, y) in points {
        let index = (((x - bounds[0]) / (bounds[1] - bounds[0]).max(1.) * bins.len() as f64)
            as usize)
            .min(bins.len() - 1);
        match &mut bins[index] {
            Some((low, high)) => {
                if y < low.1 {
                    *low = (x, y);
                }
                if y > high.1 {
                    *high = (x, y);
                }
            }
            None => bins[index] = Some(((x, y), (x, y))),
        }
    }
    let mut result = bins
        .into_iter()
        .flatten()
        .flat_map(|(a, b)| [a, b])
        .collect::<Vec<_>>();
    if let Some(p) = points.first() {
        result.push(*p);
    }
    if let Some(p) = points.last() {
        result.push(*p);
    }
    result.sort_by(|a, b| a.0.total_cmp(&b.0));
    result.dedup();
    result
}
/// Smooth a series with a centered time-window mean.
///
/// The window spans ten percent of the observed x range. Inputs with fewer than
/// three points are returned unchanged; other inputs are sorted by x before
/// smoothing. The returned points therefore need not retain the input order.
pub fn moving_average(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let span = (sorted.last().unwrap().0 - sorted.first().unwrap().0).max(1e-9);
    let half = span * 0.05;
    let mut out = Vec::with_capacity(sorted.len());
    let mut lo = 0usize;
    let mut hi = 0usize;
    let mut sum = 0.0;
    for i in 0..sorted.len() {
        let x = sorted[i].0;
        while hi < sorted.len() && sorted[hi].0 <= x + half {
            sum += sorted[hi].1;
            hi += 1;
        }
        while lo < hi && sorted[lo].0 < x - half {
            sum -= sorted[lo].1;
            lo += 1;
        }
        let n = (hi - lo) as f64;
        if n > 0.0 {
            out.push((x, sum / n));
        }
    }
    out
}
/// One interpolated SMA sample per terminal column so Chart draws a continuous line.
pub fn column_line(points: &[(f64, f64)], columns: usize, bounds: [f64; 2]) -> Vec<(f64, f64)> {
    let smooth = moving_average(points);
    if smooth.is_empty() || columns == 0 {
        return smooth;
    }
    let n = columns.max(2);
    let span = (bounds[1] - bounds[0]).max(1e-9);
    let mut j = 0;
    (0..n)
        .map(|i| {
            let x = bounds[0] + span * i as f64 / (n - 1) as f64;
            while j + 1 < smooth.len() && smooth[j + 1].0 < x {
                j += 1;
            }
            let y = if j + 1 < smooth.len() {
                let (x0, y0) = smooth[j];
                let (x1, y1) = smooth[j + 1];
                if (x1 - x0).abs() > 1e-12 {
                    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                } else {
                    y0
                }
            } else {
                smooth[j].1
            };
            (x, y)
        })
        .collect()
}
/// Map a loss onto a chart row. Log scale is used when the trend spans decades so
/// late improvement is not flattened under an early spike.
pub fn trend_row(y: f64, low: f64, high: f64, height: u16, log: bool) -> u16 {
    if height == 0 {
        return 0;
    }
    let (a, b, v) = if log && low > 0.0 && high > 0.0 {
        (low.ln(), high.ln(), y.max(low).ln())
    } else {
        (low, high, y)
    };
    let t = ((v - a) / (b - a).max(1e-9)).clamp(0.0, 1.0);
    ((1.0 - t) * (height.saturating_sub(1) as f64)).round() as u16
}
fn paint_trend(buf: &mut ratatui::buffer::Buffer, graph: Rect, rows: &[u16], color: Color) {
    for (i, &row) in rows.iter().enumerate() {
        let x = graph.x + i as u16;
        if x >= graph.right() {
            break;
        }
        let next = rows.get(i + 1).copied().unwrap_or(row);
        let (top, bot) = if row <= next {
            (row, next)
        } else {
            (next, row)
        };
        for y_off in top..=bot {
            let y = graph.y + y_off;
            if y >= graph.bottom() {
                continue;
            }
            let ch = if top == bot {
                '─'
            } else if y_off == row && next > row {
                '╮'
            } else if y_off == row && next < row {
                '╯'
            } else if y_off == next && next > row {
                '╰'
            } else if y_off == next && next < row {
                '╭'
            } else {
                '│'
            };
            buf[(x, y)]
                .set_char(ch)
                .set_style(Style::default().fg(color));
        }
    }
}
fn chart(
    f: &mut Frame,
    area: Rect,
    name: &str,
    points: &[(f64, f64)],
    bounds: [f64; 2],
    axis: &str,
    color: Color,
) {
    if points.is_empty() {
        f.render_widget(
            Paragraph::new("No bound saved observations yet").block(title(name)),
            area,
        );
        return;
    }
    f.render_widget(title(name), area);
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width < 12 || inner.height < 4 {
        return;
    }
    let label_w = 7u16;
    let graph = Rect {
        x: inner.x + label_w,
        y: inner.y,
        width: inner.width.saturating_sub(label_w),
        height: inner.height.saturating_sub(2),
    };
    if graph.width < 4 || graph.height < 2 {
        return;
    }
    let smooth = column_line(points, graph.width.max(2) as usize, bounds);
    let low = smooth.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let high = smooth.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let high = if high > low { high } else { low + 0.001 };
    let log = low > 0.0 && high / low >= 3.0;
    let scatter = envelope(points, graph.width as usize, bounds);
    let buf = f.buffer_mut();
    buf.set_stringn(
        inner.x,
        graph.y,
        format!("{high:.3}"),
        label_w as usize,
        Style::default(),
    );
    buf.set_stringn(
        inner.x,
        graph.bottom().saturating_sub(1),
        format!("{low:.3}"),
        label_w as usize,
        Style::default(),
    );
    for &(x, y) in &scatter {
        if y < low || y > high {
            continue;
        }
        let col = (((x - bounds[0]) / (bounds[1] - bounds[0]).max(1e-9)
            * (graph.width.saturating_sub(1) as f64)) as u16)
            .min(graph.width.saturating_sub(1));
        let row = trend_row(y, low, high, graph.height, log);
        let cell = (graph.x + col, graph.y + row);
        if cell.0 < graph.right() && cell.1 < graph.bottom() {
            buf[cell]
                .set_char('·')
                .set_style(Style::default().fg(Color::DarkGray));
        }
    }
    let rows: Vec<u16> = smooth
        .iter()
        .map(|p| trend_row(p.1, low, high, graph.height, log))
        .collect();
    paint_trend(buf, graph, &rows, color);
    let axis_y = graph.bottom();
    if axis_y < inner.bottom() {
        buf.set_stringn(
            graph.x,
            axis_y,
            format!("{:.1}", bounds[0]),
            8,
            Style::default(),
        );
        let right = format!("{:.1}", bounds[1]);
        buf.set_stringn(
            graph.right().saturating_sub(right.len() as u16),
            axis_y,
            right,
            8,
            Style::default(),
        );
    }
    if axis_y + 1 < inner.bottom() || inner.bottom() > graph.bottom() {
        let y = inner.bottom().saturating_sub(1);
        buf.set_stringn(graph.x, y, axis, graph.width as usize, Style::default());
    }
}
/// The newest saved losses and the history notice, shared by the text summary in
/// [`render`] and the compact Deep-learning dashboard view, so small terminals keep the
/// latest validation loss. INVALID runs are excluded, as in the overview charts. Training
/// and validation each come from the record with the greatest timestamp across all
/// remaining runs, so an overlapping run that merely started earlier cannot hide a newer
/// validation. The notice is the first source error, otherwise the counts of unbound
/// validation rows and incomplete final lines.
pub fn latest_summary(snapshot: &Snapshot) -> ([String; 3], String) {
    let points = || {
        snapshot
            .runs
            .iter()
            .filter(|r| r.status != "INVALID")
            .flat_map(|r| r.points.iter())
    };
    let last = points().max_by_key(|p| p.timestamp);
    let last_val = points()
        .filter(|p| p.validation.is_some())
        .max_by_key(|p| p.timestamp)
        .and_then(|p| p.validation.as_ref());
    let value = |v: Option<f64>| {
        v.map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "unavailable".into())
    };
    let unbound: usize = snapshot.runs.iter().map(|r| r.unbound_validation).sum();
    let pending = snapshot.runs.iter().filter(|r| r.pending_line).count();
    let notice = snapshot.errors.first().cloned().unwrap_or_else(|| {
        format!("{unbound} unbound validation rows not plotted · {pending} incomplete final lines pending; saved logs remain unchanged")
    });
    (
        [
            format!("Training Sequence NLL: {}", value(last.map(|p| p.train))),
            format!(
                "Validation Sequence NLL: {}",
                value(last_val.map(|v| v.sequence_nll))
            ),
            format!(
                "Validation pairwise loss: {}",
                value(last_val.map(|v| v.ranking_bt))
            ),
        ],
        notice,
    )
}
pub fn render(
    f: &mut Frame,
    area: Rect,
    snapshot: &Snapshot,
    navigation: &Navigation,
    run: &Value,
) {
    let selected = navigation
        .selected
        .as_ref()
        .and_then(|p| snapshot.runs.iter().position(|r| &r.path == p));
    let count: usize = snapshot.runs.iter().map(|r| r.points.len()).sum();
    let missing: usize = snapshot.runs.iter().map(|r| r.malformed).sum();
    let ([training, validation_nll, pairwise], notice) = latest_summary(snapshot);
    let compact = area.height < 18;
    let header = if !snapshot.ready {
        "Loading FULL saved history in background…".into()
    } else if compact {
        format!(
            "Deep learning · {} runs · {count} records",
            snapshot.runs.len()
        )
    } else {
        format!("FULL saved history · {} runs · {count} optimizer records · {} missing sources · {missing} malformed records", snapshot.runs.len(), snapshot.errors.len())
    };
    let layout = Layout::vertical([
        Constraint::Length(if compact { 1 } else { 3 }),
        Constraint::Min(1),
        Constraint::Length(if compact { 2 } else { 6 }),
    ])
    .split(area);
    let header = Paragraph::new(header).wrap(Wrap { trim: true });
    f.render_widget(
        if compact {
            header
        } else {
            header.block(title(" Deep learning loss · restored after restart "))
        },
        layout[0],
    );
    let mut content = layout[1];
    if area.width >= 100 && area.height >= 24 {
        let split = Layout::horizontal([Constraint::Length(35), Constraint::Min(1)]).split(content);
        content = split[1];
        let visible = split[0].height.saturating_sub(2) as usize;
        let end = selected.map_or(snapshot.runs.len(), |i| {
            (i + 1).max(visible).min(snapshot.runs.len())
        });
        let start = end.saturating_sub(visible);
        let items = snapshot.runs[start..end]
            .iter()
            .enumerate()
            .map(|(i, r)| {
                ListItem::new(format!(
                    "{} {:>3} {} {}",
                    if selected == Some(start + i) {
                        ">"
                    } else {
                        " "
                    },
                    start + i + 1,
                    r.status,
                    short(&r.path)
                ))
                .style(Style::default().fg(if selected == Some(start + i) {
                    Color::Cyan
                } else {
                    Color::Gray
                }))
            })
            .collect::<Vec<_>>();
        f.render_widget(
            List::new(items).block(title(" Runs · ↑↓ / PgUp PgDn · Home End ")),
            split[0],
        );
    }
    // Charts always show every valid run. Selecting a row only highlights it and
    // changes the footer; it must not zoom the three global loss histories.
    let runs: Vec<&Run> = snapshot
        .runs
        .iter()
        .filter(|r| r.status != "INVALID")
        .collect();
    let first = runs
        .iter()
        .flat_map(|r| r.points.iter())
        .map(|p| p.timestamp)
        .min()
        .unwrap_or(0);
    let mut train = Vec::new();
    let mut validation = Vec::new();
    let mut ranking = Vec::new();
    let mut panels = BTreeSet::new();
    for r in &runs {
        for p in &r.points {
            let x = p.timestamp.saturating_sub(first) as f64 / 3600.;
            train.push((x, p.train));
            if let Some(v) = &p.validation {
                validation.push((x, v.sequence_nll));
                ranking.push((x, v.ranking_bt));
                panels.insert(v.panel_sha256.as_deref().unwrap_or(""));
            }
        }
    }
    let xmax = train.iter().map(|p| p.0).fold(1., f64::max);
    let bounds = [0., xmax];
    let axis = "hours since first saved training record";
    if content.height >= 18 {
        let plots = Layout::vertical([Constraint::Ratio(1, 3); 3]).split(content);
        chart(
            f,
            plots[0],
            " Training Sequence NLL · global SMA · lower is better ",
            &train,
            bounds,
            axis,
            Color::Cyan,
        );
        chart(
            f,
            plots[1],
            " Validation Sequence NLL · global SMA · lower is better ",
            &validation,
            bounds,
            axis,
            Color::Yellow,
        );
        chart(
            f,
            plots[2],
            " Validation pairwise loss · global SMA · lower is better ",
            &ranking,
            bounds,
            axis,
            Color::Magenta,
        );
    } else {
        let text=format!("{training}\n{validation_nll}\n{pairwise}\n{}\nEnlarge terminal for three full-history charts.",selected.map(|i|short(&snapshot.runs[i].path)).unwrap_or_else(||"All recorded campaign and auxiliary runs".into()));
        let widget = Paragraph::new(text).wrap(Wrap { trim: true });
        f.render_widget(
            if compact {
                widget
            } else {
                widget.block(title(" Loss history "))
            },
            content,
        );
    }
    let detail=selected.map(|i| {let r=&snapshot.runs[i];format!("{} · {} records · {} attempts",r.path.display(),r.points.len(),r.attempts)})
        .unwrap_or_else(||format!("All runs, including auxiliary branches · {} validation panels; panels/corpora may differ",panels.len()));
    let checkpoint = run["selected_checkpoint"]["checkpoint"]
        .as_str()
        .unwrap_or("unavailable");
    let lines = if compact {
        vec![
            Line::raw("↑↓ runs · Home/End · a full · d details"),
            Line::raw(detail),
        ]
    } else {
        vec![Line::raw("↑↓ select run · PgUp/PgDn page · Home oldest · End newest · a FULL history · d optimizer details"),
        Line::raw(one_line(&detail,area.width)),Line::raw(one_line(&format!("Saved checkpoint: {checkpoint}"),area.width)),
        Line::styled("Training fit is not generalization. Validation is adaptive; compare the same panel.",Style::default().fg(Color::Yellow)),
        Line::raw(one_line(&selected.map(|i|format!("Parent: {} · {} · restart boundaries retained",snapshot.runs[i].parent,snapshot.runs[i].status))
            .unwrap_or_else(||"Restart boundaries stay recorded. INVALID runs are excluded from the overview; select them to inspect.".into()),area.width)),
        Line::raw(one_line(&notice,area.width))]
    };
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), layout[2]);
}

#[cfg(test)]
mod chart_series_tests {
    use super::{envelope, moving_average};

    #[test]
    fn empty_chart_series_are_empty() {
        assert!(envelope(&[], 0, [0.0, 1.0]).is_empty());
        assert!(moving_average(&[]).is_empty());
    }

    #[test]
    fn moving_average_sorts_long_series_by_x() {
        let points = [(2.0, 2.0), (0.0, 0.0), (1.0, 1.0)];
        let smoothed = moving_average(&points);
        assert!(smoothed.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        assert_eq!(smoothed.first().map(|point| point.0), Some(0.0));
        assert_eq!(smoothed.last().map(|point| point.0), Some(2.0));
    }
}
