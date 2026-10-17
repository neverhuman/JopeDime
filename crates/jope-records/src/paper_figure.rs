//! Figure 1 of the white paper: the first generation's validation plateau, its outcome head at
//! chance, and the post-C183 S3 SoS-debiased contextual surrogate shadow progression.
//!
//! `docs/paper/figures/why-it-matters.svg` is rendered from
//! `docs/paper/figures/v1_full_validation.dat`, the per-epoch full validation pass that claim C27
//! pins, and from claims C76 and C190 in `docs/paper/claims.json`. `paper-trace` re-renders it and
//! fails when:
//!
//! 1. the data file's SHA-256 is not C27's pin, or its epochs are not consecutive;
//! 2. the plateau recomputed from the data (its band, its length and the epoch it starts at) is
//!    not C27's value;
//! 3. the epoch-2, best and last losses recomputed from the data are not C26's value;
//! 4. C76's value or lineage count, or C190's shadow-pool-v2 pairwise values, cannot be read;
//! 5. the committed SVG is not the rendering of the data and the ledger
//!    (`jope-records paper-trace --write-figure` rewrites it), or the write-up does not show it.

use std::fmt::Write as _;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

pub const DATA: &str = "docs/paper/figures/v1_full_validation.dat";
pub const FIGURE: &str = "docs/paper/figures/why-it-matters.svg";
/// The write-up's link to the figure, relative to `docs/paper/`.
pub const LINK: &str = "](figures/why-it-matters.svg)";
/// The first epoch of the plateau C27 states.
const PLATEAU_FROM: u32 = 10;

static RANKING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d\.\d+) \[(\d\.\d+), (\d\.\d+)\]$").expect("static pattern"));
static LINEAGES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"over (\d+) lineages").expect("static pattern"));
static SURROGATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"shadow-pool-v2 \(\d[\d,]* contexts, (\d+) clusters\) R0->R4 pairwise (\d\.\d+) -> (\d\.\d+) \(\+(\d\.\d+) \[\+(\d\.\d+), \+(\d\.\d+)\] GREATER\)",
    )
    .expect("static pattern")
});

#[derive(Debug, Clone, Copy)]
struct Epoch {
    epoch: u32,
    loss: f64,
}

#[derive(Debug, Clone)]
struct Surrogate {
    r0: f64,
    r4: f64,
    lift: f64,
    low: f64,
    high: f64,
    clusters: String,
}

/// What the figure draws: every epoch's loss, the plateau band, C76 ranking accuracy, and
/// C190 shadow-pool surrogate progression (non-qualifying).
#[derive(Debug)]
pub struct Figure {
    epochs: Vec<Epoch>,
    band: (f64, f64),
    last_epoch: u32,
    ranking: (f64, f64, f64),
    lineages: String,
    surrogate: Option<Surrogate>,
}

fn row(line: &str) -> Option<Epoch> {
    let mut cols = line.split_whitespace();
    Some(Epoch {
        epoch: cols.next()?.parse().ok()?,
        loss: cols.next()?.parse().ok()?,
    })
}

fn claim<'a>(doc: &'a Value, id: &str) -> Option<&'a Value> {
    doc.get("claims")?
        .as_list()?
        .iter()
        .find(|c| c.get("id").and_then(Value::as_str) == Some(id))
}

fn text<'a>(claim: Option<&'a Value>, key: &str) -> &'a str {
    claim
        .and_then(|c| c.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// The figure's values, with the problems rules 1 to 4 find.
pub fn figure(dat: &[u8], ledger_text: &str, problems: &mut Vec<String>) -> Result<Figure> {
    let data =
        std::str::from_utf8(dat).map_err(|e| Error::parse(Path::new(DATA), e.to_string()))?;
    let mut lines = data.lines();
    let header: Vec<&str> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .take(2)
        .collect();
    if header != ["epoch", "loss"] {
        return Err(Error::Shape(format!(
            "{DATA}: the header is not `epoch loss ...`"
        )));
    }
    let mut epochs = vec![];
    for (i, line) in lines.enumerate() {
        epochs.push(
            row(line).ok_or_else(|| {
                Error::Shape(format!("{DATA}:{}: not `<epoch> <loss> ...`", i + 2))
            })?,
        );
    }
    let (Some(&first), Some(&last)) = (epochs.first(), epochs.last()) else {
        return Err(Error::Shape(format!("{DATA}: no epochs")));
    };
    if !epochs.windows(2).all(|w| w[1].epoch == w[0].epoch + 1) {
        problems.push(format!("figure: {DATA}: the epochs are not consecutive"));
    }

    let doc = pyjson::parse_str(ledger_text)
        .map_err(|m| Error::parse(Path::new(crate::paper_trace::LEDGER), m))?;
    let (c26, c27, c76, c190) = (
        claim(&doc, "C26"),
        claim(&doc, "C27"),
        claim(&doc, "C76"),
        claim(&doc, "C190"),
    );

    let digest = py::sha256_hex(dat);
    if digest != text(c27, "sha256") {
        problems.push(format!(
            "figure: {DATA} hashes to {digest}, not C27's pin {:?}",
            text(c27, "sha256")
        ));
    }

    let plateau: Vec<f64> = epochs
        .iter()
        .filter(|e| e.epoch >= PLATEAU_FROM)
        .map(|e| e.loss)
        .collect();
    let lo = plateau.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = plateau.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let recomputed = format!("[{lo:.7}, {hi:.7}] over {} epochs", plateau.len());
    if recomputed != text(c27, "value") {
        problems.push(format!(
            "figure: the plateau from epoch {PLATEAU_FROM} in {DATA} is {recomputed:?}, not C27's value {:?}",
            text(c27, "value")
        ));
    }
    if epochs
        .iter()
        .any(|e| e.epoch == PLATEAU_FROM - 1 && (lo..=hi).contains(&e.loss))
    {
        problems.push(format!(
            "figure: epoch {} in {DATA} is already inside the band, so the plateau does not start at epoch {PLATEAU_FROM}",
            PLATEAU_FROM - 1
        ));
    }

    let best = epochs
        .iter()
        .copied()
        .fold(first, |b, e| if e.loss < b.loss { e } else { b });
    let recomputed = format!(
        "{:.10} -> {:.10} (ep {}) -> {:.10} (ep {})",
        first.loss, best.loss, best.epoch, last.loss, last.epoch
    );
    if recomputed != text(c26, "value") {
        problems.push(format!(
            "figure: the first, best and last losses in {DATA} are {recomputed:?}, not C26's value {:?}",
            text(c26, "value")
        ));
    }

    let ranking = RANKING
        .captures(text(c76, "value"))
        .and_then(|c| Some((c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?)));
    let lineages = LINEAGES
        .captures(text(c76, "statement"))
        .map(|c| c[1].to_owned());
    if ranking.is_none() || lineages.is_none() {
        problems
            .push("figure: C76 is not `<accuracy> [<low>, <high>]` over `<n> lineages`".to_owned());
    }
    let surrogate = if c190.is_some() {
        let parsed = SURROGATE.captures(text(c190, "value")).and_then(|c| {
            Some(Surrogate {
                clusters: c[1].to_owned(),
                r0: c[2].parse().ok()?,
                r4: c[3].parse().ok()?,
                lift: c[4].parse().ok()?,
                low: c[5].parse().ok()?,
                high: c[6].parse().ok()?,
            })
        });
        if parsed.is_none() {
            problems.push(
                "figure: C190 is not `shadow-pool-v2 (<contexts> contexts, <clusters> clusters) R0->R4 pairwise <r0> -> <r4> (+<lift> [+<low>, +<high>] GREATER)`"
                    .to_owned(),
            );
        }
        parsed
    } else {
        None
    };
    Ok(Figure {
        band: (lo, hi),
        last_epoch: last.epoch,
        ranking: ranking.unwrap_or((0.5, 0.5, 0.5)),
        lineages: lineages.unwrap_or_default(),
        surrogate,
        epochs,
    })
}

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
    "Why it matters: v1 plateaued with fresh-program q-hat at chance (0.506); shadow surrogate rose 0.634 to 0.680 (non-qualifying).";
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

/// The problems with the figure, its data, the claims it draws and the write-up's link to it;
/// with `write`, the SVG is rewritten first.
pub fn check(root: &Path, writeup: &str, ledger_text: &str, write: bool) -> Result<Vec<String>> {
    let mut problems = vec![];
    let f = figure(
        &py::read_bytes(&root.join(DATA))?,
        ledger_text,
        &mut problems,
    )?;
    if f.surrogate.is_none() {
        problems.push("figure: C190 is missing from claims.json".to_owned());
    }
    let svg = render(&f);
    let path = root.join(FIGURE);
    if write {
        std::fs::write(&path, &svg).map_err(|e| Error::io(&path, e))?;
    }
    if std::fs::read_to_string(&path).ok().as_deref() != Some(svg.as_str()) {
        problems.push(format!(
            "figure: {FIGURE} is not the rendering of {DATA} and the ledger; run `jope-records paper-trace --write-figure`"
        ));
    }
    if !writeup.contains(LINK) {
        problems.push(format!(
            "figure: {} does not show {FIGURE}",
            crate::paper_trace::WRITEUP
        ));
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAT: &str = "epoch loss ce\n2 1.25 0\n3 0.95 0\n4 0.9001 0\n5 0.9005 0\n6 0.9002 0\n";

    /// A ledger whose C26, C27 and C76 agree with `DAT` when the plateau starts at epoch 4.
    fn ledger(dat: &str, c27: &str, c76: &str) -> String {
        let pin = py::sha256_hex(dat.as_bytes());
        format!(
            r#"{{"claims": [
              {{"id": "C26", "value": "1.2500000000 -> 0.9001000000 (ep 4) -> 0.9002000000 (ep 6)"}},
              {{"id": "C27", "value": "{c27}", "sha256": "{pin}"}},
              {{"id": "C76", "statement": "accuracy over 32 lineages", "value": "{c76}"}}
            ]}}"#
        )
    }

    fn problems(dat: &str, ledger_text: &str) -> Vec<String> {
        let mut found = vec![];
        let f = figure(dat.as_bytes(), ledger_text, &mut found).expect("parses");
        // Synthetic data plateaus at epoch 4, not 10; drop the start-epoch rule's finding.
        found.retain(|p| !p.contains("plateau from epoch") && !p.contains("already inside"));
        assert!(render(&f).ends_with("</svg>\n"));
        found
    }

    #[test]
    fn the_figure_follows_its_data_and_its_claims() {
        let c76 = "0.5056656 [0.4412160, 0.5684545]";
        let ok = ledger(DAT, "x", c76);
        assert_eq!(problems(DAT, &ok), Vec::<String>::new());

        // Data edited after the pin: the hash and C26's last value no longer agree.
        let drifted = DAT.replace("6 0.9002", "6 0.9003");
        let found = problems(&drifted, &ok);
        assert!(
            found.iter().any(|p| p.contains("not C27's pin")),
            "{found:?}"
        );
        assert!(
            found.iter().any(|p| p.contains("not C26's value")),
            "{found:?}"
        );

        // An unreadable C76 value.
        let found = problems(DAT, &ledger(DAT, "x", "about half"));
        assert!(found.iter().any(|p| p.contains("C76 is not")), "{found:?}");

        // Valid and unreadable C190 values.
        let pin = py::sha256_hex(DAT.as_bytes());
        let with_c190 = format!(
            r#"{{"claims": [
              {{"id": "C26", "value": "1.2500000000 -> 0.9001000000 (ep 4) -> 0.9002000000 (ep 6)"}},
              {{"id": "C27", "value": "x", "sha256": "{pin}"}},
              {{"id": "C76", "statement": "accuracy over 32 lineages", "value": "{c76}"}},
              {{"id": "C190", "value": "8,908 S3 rows; shadow-pool-v2 (1,136 contexts, 255 clusters) R0->R4 pairwise 0.634318 -> 0.680342 (+0.046025 [+0.039928, +0.051692] GREATER)"}}
            ]}}"#
        );
        assert_eq!(problems(DAT, &with_c190), Vec::<String>::new());
        let mut found = vec![];
        let f = figure(DAT.as_bytes(), &with_c190, &mut found).expect("parses");
        let svg = render(&f);
        assert!(svg.contains("(b) Ranking: fresh programs vs. shadow pool"));
        assert!(svg.contains("shadow surrogate rose 0.634 to 0.680 (non-qualifying)"));
        assert!(svg.contains("v1 q-hat (fresh)"));
        assert!(svg.contains("R0 0.634 -> R4 0.680"));
        assert!(svg.contains("lift +0.046 [+0.040, +0.052]"));
        assert!(svg.contains("(255 clusters, non-qualifying)"));
        assert!(!svg.contains("(b) Fresh-program pairwise ranking accuracy"));
        // Only the vertical R0->R4 segment uses the surrogate stroke on <line>; no level caps around R4.
        assert_eq!(svg.matches(r##"stroke="#2e6b3e""##).count(), 2);

        let bad_c190 = with_c190.replace("R0->R4 pairwise", "broken");
        let found = problems(DAT, &bad_c190);
        assert!(found.iter().any(|p| p.contains("C190 is not")), "{found:?}");
    }

    #[test]
    fn the_plateau_is_recomputed_from_epoch_ten() {
        let mut dat = String::from("epoch loss\n");
        for e in 2..=12 {
            let loss = if e < 9 {
                1.0
            } else {
                0.9 + f64::from(e) * 1e-5
            };
            dat.push_str(&format!("{e} {loss}\n"));
        }
        let mut found = vec![];
        let led = ledger(
            &dat,
            "[0.9001000, 0.9001200] over 3 epochs",
            "0.5 [0.4, 0.6]",
        );
        figure(dat.as_bytes(), &led, &mut found).expect("parses");
        // Epoch 9 (0.90009) is below the band, so the plateau's start is accepted.
        assert!(!found.iter().any(|p| p.contains("plateau")), "{found:?}");
        let led = ledger(
            &dat,
            "[0.9001000, 0.9001300] over 3 epochs",
            "0.5 [0.4, 0.6]",
        );
        let mut found = vec![];
        figure(dat.as_bytes(), &led, &mut found).expect("parses");
        assert!(
            found.iter().any(|p| p.contains("not C27's value")),
            "{found:?}"
        );
    }

    #[test]
    fn the_rendering_is_deterministic_and_draws_every_epoch() {
        let mut found = vec![];
        let led = ledger(DAT, "x", "0.5056656 [0.4412160, 0.5684545]");
        let f = figure(DAT.as_bytes(), &led, &mut found).expect("parses");
        let svg = render(&f);
        assert_eq!(svg, render(&f));
        // One marker per epoch plus the ranking point.
        assert_eq!(svg.matches("<circle").count(), 5 + 1);
        assert!(svg.contains(">0.506<") && svg.contains(">[0.441, 0.568]<"));
        assert!(svg.contains(">32 lineages<"));
    }
}
