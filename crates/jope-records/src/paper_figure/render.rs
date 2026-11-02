use super::{Figure, PLATEAU_FROM};
use std::fmt::Write as _;

const WIDTH: f64 = 800.0;
const HEIGHT: f64 = 324.0;
/// The baseline of the panel titles, under the takeaway line.
const PANEL_TITLE: f64 = 52.0;
const TOP: f64 = 76.0;
const BOTTOM: f64 = 260.0;
/// The figure's one-line takeaway, so the image keeps its point when it is shown alone.
const TAKEAWAY: &str =
    "Why it matters: the first generation stopped improving, and its outcome head ranked new programs at chance.";
const TAKEAWAY_SURROGATE: &str =
    "Why it matters: v1 plateaued with fresh-program q-hat at chance; shadow-pool lift is non-qualifying.";
const INK: &str = "#222222";
const AXIS: &str = "#555555";
const LINE: &str = "#1f4e79";
const SURR: &str = "#2e6b3e";
const SHADE: &str = "#e3efe6";

/// Linear map of `v` in `[a, b]` to `[p, q]`.
fn scale(v: f64, (a, b): (f64, f64), (p, q): (f64, f64)) -> f64 {
    p + (v - a) / (b - a) * (q - p)
}

fn label(svg: &mut String, x: f64, y: f64, anchor: &str, size: u32, body: &str) {
    let _ = writeln!(
        svg,
        r#"<text x="{x:.1}" y="{y:.1}" text-anchor="{anchor}" font-size="{size}" fill="{INK}">{body}</text>"#
    );
}

fn line(svg: &mut String, (x1, y1): (f64, f64), (x2, y2): (f64, f64), extra: &str) {
    let _ = writeln!(
        svg,
        r#"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="{AXIS}" stroke-width="1"{extra}/>"#
    );
}

/// Axes, y ticks and the y-axis title of one panel.
fn axes(svg: &mut String, xs: (f64, f64), ys: (f64, f64), ticks: &[f64], title: &str) {
    line(svg, (xs.0, TOP), (xs.0, BOTTOM), "");
    line(svg, (xs.0, BOTTOM), (xs.1, BOTTOM), "");
    for &t in ticks {
        let y = scale(t, ys, (BOTTOM, TOP));
        line(svg, (xs.0 - 4.0, y), (xs.0, y), "");
        label(svg, xs.0 - 7.0, y + 4.0, "end", 11, &format!("{t:.2}"));
    }
    let (x, y) = (xs.0 - 44.0, (TOP + BOTTOM) / 2.0);
    let _ = writeln!(
        svg,
        r#"<text x="{x:.1}" y="{y:.1}" text-anchor="middle" font-size="12" fill="{INK}" transform="rotate(-90 {x:.1} {y:.1})">{title}</text>"#
    );
}

/// The SVG for `f`, byte for byte the same for the same values.
pub fn render(f: &Figure) -> String {
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">"#
    );
    let _ = writeln!(
        svg,
        r#"<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>"#
    );
    let takeaway = if f.surrogate.is_some() {
        TAKEAWAY_SURROGATE
    } else {
        TAKEAWAY
    };
    let _ = writeln!(
        svg,
        r#"<text x="26.0" y="24.0" text-anchor="start" font-size="13" font-weight="bold" fill="{INK}">{takeaway}</text>"#
    );

    // (a) Full validation loss by closed epoch, the plateau shaded.
    let (xs, ys) = ((76.0, 470.0), (0.88, 1.02));
    let epochs = (0.0, f64::from(f.last_epoch));
    let x = |e: f64| scale(e, epochs, xs);
    let y = |l: f64| scale(l, ys, (BOTTOM, TOP));
    label(
        &mut svg,
        xs.0 - 50.0,
        PANEL_TITLE,
        "start",
        13,
        "(a) First generation: full validation loss by epoch",
    );
    let (s0, s1) = (x(f64::from(PLATEAU_FROM)), x(epochs.1));
    let _ = writeln!(
        svg,
        r#"<rect x="{s0:.1}" y="{TOP:.1}" width="{:.1}" height="{:.1}" fill="{SHADE}"/>"#,
        s1 - s0,
        BOTTOM - TOP
    );
    let mid = (s0 + s1) / 2.0;
    let plateau = f.epochs.iter().filter(|e| e.epoch >= PLATEAU_FROM).count();
    label(
        &mut svg,
        mid,
        TOP + 22.0,
        "middle",
        12,
        &format!("{plateau} epochs ({PLATEAU_FROM}–{}) within", f.last_epoch),
    );
    label(
        &mut svg,
        mid,
        TOP + 38.0,
        "middle",
        12,
        &format!("[{:.4}, {:.4}]", f.band.0, f.band.1),
    );
    axes(
        &mut svg,
        xs,
        ys,
        &[0.88, 0.92, 0.96, 1.00],
        "full validation loss",
    );
    for t in (0..=f.last_epoch).step_by(10) {
        let tx = x(f64::from(t));
        line(&mut svg, (tx, BOTTOM), (tx, BOTTOM + 4.0), "");
        label(&mut svg, tx, BOTTOM + 17.0, "middle", 11, &t.to_string());
    }
    label(
        &mut svg,
        (xs.0 + xs.1) / 2.0,
        BOTTOM + 36.0,
        "middle",
        12,
        "closed epoch",
    );
    let points: Vec<String> = f
        .epochs
        .iter()
        .map(|e| format!("{:.1},{:.1}", x(f64::from(e.epoch)), y(e.loss)))
        .collect();
    let _ = writeln!(
        svg,
        r#"<polyline points="{}" fill="none" stroke="{LINE}" stroke-width="1.5"/>"#,
        points.join(" ")
    );
    for e in &f.epochs {
        let _ = writeln!(
            svg,
            r#"<circle cx="{:.1}" cy="{:.1}" r="2.2" fill="{LINE}"/>"#,
            x(f64::from(e.epoch)),
            y(e.loss)
        );
    }

    // (b) Pairwise ranking: v1 outcome head on fresh programs vs. shadow-pool surrogate.
    if let Some(s) = &f.surrogate {
        let (xs, ys) = ((566.0, 776.0), (0.40, 0.72));
        let y = |a: f64| scale(a, ys, (BOTTOM, TOP));
        label(
            &mut svg,
            xs.0 - 52.0,
            PANEL_TITLE,
            "start",
            13,
            "(b) Ranking: fresh programs vs. shadow pool",
        );
        axes(
            &mut svg,
            xs,
            ys,
            &[0.40, 0.50, 0.60, 0.70],
            "pairwise ranking accuracy",
        );
        let chance = y(0.5);
        line(
            &mut svg,
            (xs.0, chance),
            (xs.1, chance),
            r#" stroke-dasharray="4 3""#,
        );
        label(&mut svg, xs.0 + 4.0, chance - 5.0, "start", 11, "chance");
        line(
            &mut svg,
            (660.0, TOP),
            (660.0, BOTTOM),
            r#" stroke-dasharray="2 3""#,
        );

        let (value, low, high) = f.ranking;
        let cx1 = 608.0;
        let _ = writeln!(
            svg,
            r#"<line x1="{cx1:.1}" y1="{:.1}" x2="{cx1:.1}" y2="{:.1}" stroke="{LINE}" stroke-width="1.5"/>"#,
            y(low),
            y(high)
        );
        for end in [low, high] {
            let _ = writeln!(
                svg,
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{LINE}" stroke-width="1.5"/>"#,
                cx1 - 6.0,
                y(end),
                cx1 + 6.0,
                y(end)
            );
        }
        let _ = writeln!(
            svg,
            r#"<circle cx="{cx1:.1}" cy="{:.1}" r="4" fill="{LINE}"/>"#,
            y(value)
        );
        label(
            &mut svg,
            cx1,
            y(high) - 18.0,
            "middle",
            11,
            &format!("{value:.3}"),
        );
        label(
            &mut svg,
            cx1,
            y(high) - 5.0,
            "middle",
            10,
            &format!("[{low:.3}, {high:.3}]"),
        );
        label(
            &mut svg,
            cx1,
            BOTTOM + 17.0,
            "middle",
            10,
            "v1 q-hat (fresh)",
        );
        label(
            &mut svg,
            cx1,
            BOTTOM + 31.0,
            "middle",
            9,
            &format!("({} lineages)", f.lineages),
        );

        let cx2 = 720.0;
        let _ = writeln!(
            svg,
            r#"<line x1="{cx2:.1}" y1="{:.1}" x2="{cx2:.1}" y2="{:.1}" stroke="{SURR}" stroke-width="1.5"/>"#,
            y(s.r0),
            y(s.r4)
        );
        let _ = writeln!(
            svg,
            r#"<circle cx="{cx2:.1}" cy="{:.1}" r="3.5" fill="white" stroke="{SURR}" stroke-width="1.5"/>"#,
            y(s.r0)
        );
        let _ = writeln!(
            svg,
            r#"<circle cx="{cx2:.1}" cy="{:.1}" r="4" fill="{SURR}"/>"#,
            y(s.r4)
        );
        label(
            &mut svg,
            cx2,
            y(s.r4) - 22.0,
            "middle",
            11,
            &format!("R0 {:.3} -> R4 {:.3}", s.r0, s.r4),
        );
        label(
            &mut svg,
            cx2,
            y(s.r4) - 9.0,
            "middle",
            10,
            &format!("lift +{:.3} [+{:.3}, +{:.3}]", s.lift, s.low, s.high),
        );
        label(
            &mut svg,
            cx2,
            BOTTOM + 17.0,
            "middle",
            10,
            "SoS surrogate (shadow)",
        );
        label(
            &mut svg,
            cx2,
            BOTTOM + 31.0,
            "middle",
            9,
            &format!("({} clusters, non-qualifying)", s.clusters),
        );
    } else {
        let (xs, ys) = ((612.0, 772.0), (0.40, 0.70));
        let y = |a: f64| scale(a, ys, (BOTTOM, TOP));
        label(
            &mut svg,
            xs.0 - 60.0,
            PANEL_TITLE,
            "start",
            13,
            "(b) Outcome head on fresh programs",
        );
        axes(
            &mut svg,
            xs,
            ys,
            &[0.40, 0.50, 0.60, 0.70],
            "pairwise ranking accuracy",
        );
        let chance = y(0.5);
        line(
            &mut svg,
            (xs.0, chance),
            (xs.1, chance),
            r#" stroke-dasharray="4 3""#,
        );
        label(&mut svg, xs.0 + 4.0, chance - 5.0, "start", 11, "chance");
        let (value, low, high) = f.ranking;
        let cx = (xs.0 + xs.1) / 2.0 - 16.0;
        let _ = writeln!(
            svg,
            r#"<line x1="{cx:.1}" y1="{:.1}" x2="{cx:.1}" y2="{:.1}" stroke="{LINE}" stroke-width="1.5"/>"#,
            y(low),
            y(high)
        );
        for end in [low, high] {
            let _ = writeln!(
                svg,
                r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{LINE}" stroke-width="1.5"/>"#,
                cx - 6.0,
                y(end),
                cx + 6.0,
                y(end)
            );
        }
        let _ = writeln!(
            svg,
            r#"<circle cx="{cx:.1}" cy="{:.1}" r="4" fill="{LINE}"/>"#,
            y(value)
        );
        // Beside the upper cap, clear of the chance line.
        label(
            &mut svg,
            cx + 12.0,
            y(high) + 4.0,
            "start",
            12,
            &format!("{value:.3}"),
        );
        label(
            &mut svg,
            cx + 12.0,
            y(high) + 19.0,
            "start",
            11,
            &format!("[{low:.3}, {high:.3}]"),
        );
        label(
            &mut svg,
            (xs.0 + xs.1) / 2.0,
            BOTTOM + 17.0,
            "middle",
            11,
            &format!("{} lineages", f.lineages),
        );
    }
    svg.push_str("</svg>\n");
    svg
}
