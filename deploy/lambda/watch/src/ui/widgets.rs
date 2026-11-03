use super::*;

pub(super) const BG: Color = Color::Rgb(9, 14, 27);
pub(super) const PANEL: Color = Color::Rgb(14, 23, 41);
pub(super) const EDGE: Color = Color::Rgb(39, 57, 80);
pub(super) const TEXT: Color = Color::Rgb(217, 231, 245);
pub(super) const DIM: Color = Color::Rgb(139, 160, 184);
pub(super) const CYAN: Color = Color::Rgb(47, 221, 237);
pub(super) const GREEN: Color = Color::Rgb(110, 239, 166);
pub(super) const AMBER: Color = Color::Rgb(255, 201, 100);
pub(super) const PINK: Color = Color::Rgb(245, 126, 194);
pub(super) const RED: Color = Color::Rgb(255, 113, 126);

pub(super) fn panel(title: impl Into<String>, c: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(EDGE))
        .title(Line::from(title.into()).style(Style::default().fg(c).bold()))
        .style(Style::default().bg(PANEL).fg(TEXT))
}
pub(super) fn tile(f: &mut Frame, r: Rect, title: &str, value: String, sub: String, c: Color) {
    let p = Paragraph::new(vec![
        Line::from(value).style(Style::default().fg(c).bold()),
        Line::from(sub).style(Style::default().fg(DIM)),
    ])
    .block(panel(title, c));
    f.render_widget(p, r);
}
pub(super) fn ms(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1} ms"))
        .unwrap_or_else
        (|| "unknown".into())
}
pub(super) fn graph(f: &mut Frame, r: Rect, title: &str, data: &[u64], c: Color) {
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
