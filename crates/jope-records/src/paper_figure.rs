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

mod render;
#[cfg(test)]
mod tests;
pub use render::render;

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
