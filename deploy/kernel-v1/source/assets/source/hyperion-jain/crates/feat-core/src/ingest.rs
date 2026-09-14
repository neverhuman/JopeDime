//! Typed CSV ingestion. Unlike `data::load_csv` (which parses every field straight to f32, silently
//! turning string columns into all-NaN), `read_frame` keeps raw tokens and classifies each feature
//! column as Numeric or Categorical so the encoder (see `encode.rs`) can use categoricals instead of
//! dropping them. Targets are kept as tokens too and parsed/mapped by the task layer.
//!
//! This reader intentionally uses the `csv` crate over byte records. That gives us quoted fields,
//! CRLF, BOM handling, invalid UTF-8 recovery, and flexible ragged rows without Python assistance.
use anyhow::{bail, Context, Result};
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColKind {
    Numeric,
    Categorical,
}

/// How to treat the first row of a CSV when ingesting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderMode {
    /// Sniff: treat as headerless iff the first non-empty row is entirely numeric.
    Auto,
    /// Force header present (first row = column names).
    Present,
    /// Force headerless (synthesize `f0..` names; the last column is the target).
    Absent,
}

#[derive(Clone, Debug)]
pub struct RawColumn {
    pub name: String,
    pub kind: ColKind,
    pub tokens: Vec<String>, // column-major: one token per row
}

/// A typed table: feature columns (original order, targets removed) + target columns (in `targets` order).
#[derive(Clone, Debug)]
pub struct Frame {
    pub feats: Vec<RawColumn>,
    pub targets: Vec<RawColumn>,
    pub primary: usize, // index into `targets`
    pub n: usize,
    pub issues: Vec<String>,
}

pub fn is_missing_token(t: &str) -> bool {
    let t = t.trim();
    if t.is_empty() {
        return true;
    }
    matches!(
        t.to_ascii_lowercase().as_str(),
        "na" | "n/a" | "null" | "none" | "nan" | "?" | "#n/a"
    )
}

fn is_bool_token(t: &str) -> bool {
    matches!(
        t.trim().to_ascii_lowercase().as_str(),
        "true" | "false" | "yes" | "no"
    )
}

fn is_numeric_token(t: &str) -> bool {
    let t = t.trim();
    if is_missing_token(t) || is_bool_token(t) {
        return true;
    }
    if matches!(
        t.to_ascii_lowercase().as_str(),
        "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    ) {
        return true;
    }
    t.parse::<f64>().is_ok()
}

fn decode_lossy(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    s.trim().to_string()
}

fn sanitize_header(raw: &str, idx: usize) -> String {
    let mut s = raw.trim().trim_start_matches('\u{feff}').to_string();
    if s.is_empty() {
        s = format!("column_{}", idx + 1);
    }
    s
}

fn unique_headers(raw: &[String]) -> (Vec<String>, Vec<String>) {
    use std::collections::HashMap;
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::with_capacity(raw.len());
    let mut issues = Vec::new();
    for name in raw {
        let count = seen.entry(name.clone()).or_insert(0);
        if *count == 0 {
            out.push(name.clone());
        } else {
            let renamed = format!("{name}__dup{count}");
            issues.push(format!("duplicate header '{name}' renamed to '{renamed}'"));
            out.push(renamed);
        }
        *count += 1;
    }
    (out, issues)
}

/// Read column names. With `has_header=false`, synthesize headerless names (`f0..`, last = `target`)
/// from the first record's width instead of reading a header line.
pub fn read_header_with(path: &Path, has_header: bool) -> Result<Vec<String>> {
    if !has_header {
        return Ok(synth_headerless_names(count_columns(path)?));
    }
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let headers = rdr
        .byte_headers()
        .with_context(|| format!("reading CSV header from {}", path.display()))?;
    if headers.is_empty() {
        bail!("CSV header has no columns");
    }
    let raw: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, f)| sanitize_header(&decode_lossy(f), i))
        .collect();
    Ok(unique_headers(&raw).0)
}

/// Read the header row's column names (assumes a header is present).
pub fn read_header(path: &Path) -> Result<Vec<String>> {
    read_header_with(path, true)
}

/// Synthesize headerless column names: `f0..f{ncol-2}` then a final `target`.
///
/// The `f` prefix is load-bearing: `leakage.rs` flags any feature whose name starts with `target`,
/// so features must not be named `target*`. Naming the last column literally `target` lets
/// `targets::resolve_target_columns("auto")` pick it as the primary without pulling siblings
/// (sibling collection requires the `target_` prefix, which plain `target` does not match).
fn synth_headerless_names(ncol: usize) -> Vec<String> {
    let mut v: Vec<String> = (0..ncol.saturating_sub(1))
        .map(|i| format!("f{i}"))
        .collect();
    v.push("target".to_string());
    v
}

/// Column count from the first non-empty record (own reader, headerless). Used to synthesize names.
fn count_columns(path: &Path) -> Result<usize> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut rec = csv::ByteRecord::new();
    while rdr
        .read_byte_record(&mut rec)
        .with_context(|| format!("reading first row of {}", path.display()))?
    {
        if !(rec.is_empty() || rec.iter().all(|f| decode_lossy(f).is_empty())) {
            return Ok(rec.len());
        }
    }
    bail!("CSV has no data rows");
}

/// Sniff whether a CSV has a header row. Headerless iff EVERY field of the first non-empty record is
/// numeric (per `is_numeric_token`); i.e. a header is present iff any first-row field is non-numeric.
pub fn sniff_has_header(path: &Path) -> Result<bool> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut rec = csv::ByteRecord::new();
    while rdr
        .read_byte_record(&mut rec)
        .with_context(|| format!("reading first row of {}", path.display()))?
    {
        if rec.is_empty() || rec.iter().all(|f| decode_lossy(f).is_empty()) {
            continue;
        }
        let all_numeric = rec.iter().all(|f| is_numeric_token(&decode_lossy(f)));
        return Ok(!all_numeric);
    }
    bail!("CSV has no data rows");
}

/// Resolve a `HeaderMode` to a concrete `has_header` bool (sniffing the file for `Auto`).
pub fn resolve_header(path: &Path, mode: HeaderMode) -> Result<bool> {
    match mode {
        HeaderMode::Auto => sniff_has_header(path),
        HeaderMode::Present => Ok(true),
        HeaderMode::Absent => Ok(false),
    }
}

fn target_position(headers: &[String], targets: &[&str], target: &str) -> Result<usize> {
    headers.iter().position(|c| c == target).with_context(|| {
        format!(
            "target column '{target}' not in header; available targets requested: {:?}",
            targets
        )
    })
}

/// Read a header-CSV into a typed `Frame`. A feature column is Categorical iff any non-empty token
/// fails to parse as f32. Column order (features) matches the file with targets removed — so on an
/// all-numeric file the resulting matrix is identical to `load_csv`.
pub fn read_frame(path: &Path, targets: &[&str], primary: &str) -> Result<Frame> {
    read_frame_with(path, targets, primary, true)
}

/// Like `read_frame`, but `has_header=false` ingests a HEADERLESS CSV: the first row is data, and
/// columns are named `f0..f{c-2}` with a final `target` (see `synth_headerless_names`). Used by the
/// `--header no`/auto-sniff path so headerless corpora can be benchmarked without preprocessing.
pub fn read_frame_with(
    path: &Path,
    targets: &[&str],
    primary: &str,
    has_header: bool,
) -> Result<Frame> {
    if targets.is_empty() {
        bail!("at least one target column is required");
    }
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let (cols, mut issues) = if has_header {
        let headers = rdr
            .byte_headers()
            .with_context(|| format!("reading CSV header from {}", path.display()))?;
        if headers.is_empty() {
            bail!("CSV header has no columns");
        }
        let raw_cols: Vec<String> = headers
            .iter()
            .enumerate()
            .map(|(i, f)| sanitize_header(&decode_lossy(f), i))
            .collect();
        unique_headers(&raw_cols)
    } else {
        // Headerless: synthesize names from the first record's width. `byte_records()` on a
        // `has_headers(false)` reader yields row 0 as data, so no row is consumed as a header.
        (synth_headerless_names(count_columns(path)?), Vec::new())
    };
    let ncol = cols.len();
    let line_base = 1 + has_header as usize;

    // Target positions in the file, in the requested `targets` order.
    let tgt_pos: Vec<usize> = targets
        .iter()
        .map(|t| target_position(&cols, targets, t))
        .collect::<Result<_>>()?;
    let primary = targets
        .iter()
        .position(|t| t == &primary)
        .context("primary not in targets")?;
    let is_target: Vec<bool> = (0..ncol).map(|i| tgt_pos.contains(&i)).collect();
    let feat_pos: Vec<usize> = (0..ncol).filter(|&i| !is_target[i]).collect();

    let mut feat_tokens: Vec<Vec<String>> = vec![Vec::new(); feat_pos.len()];
    let mut tgt_tokens: Vec<Vec<String>> = vec![Vec::new(); tgt_pos.len()];
    let mut feat_numeric: Vec<bool> = vec![true; feat_pos.len()];
    let mut n = 0usize;
    for (lineno, rec) in rdr.byte_records().enumerate() {
        let rec = rec.with_context(|| format!("reading CSV row {}", lineno + line_base))?;
        if rec.is_empty() || rec.iter().all(|f| decode_lossy(f).is_empty()) {
            continue;
        }
        if rec.len() > ncol {
            issues.push(format!(
                "row {} has {} fields; extra fields beyond width {ncol} ignored",
                lineno + line_base,
                rec.len()
            ));
        }
        for (fi, &ci) in feat_pos.iter().enumerate() {
            let tok = rec.get(ci).map(decode_lossy).unwrap_or_default();
            if feat_numeric[fi] && !is_numeric_token(&tok) {
                feat_numeric[fi] = false;
            }
            feat_tokens[fi].push(tok);
        }
        for (ti, &ci) in tgt_pos.iter().enumerate() {
            tgt_tokens[ti].push(rec.get(ci).map(decode_lossy).unwrap_or_default());
        }
        n += 1;
    }
    if n == 0 {
        bail!("CSV has no data rows");
    }

    let feats = feat_pos
        .iter()
        .enumerate()
        .map(|(fi, &ci)| RawColumn {
            name: cols[ci].clone(),
            kind: if feat_numeric[fi] {
                ColKind::Numeric
            } else {
                ColKind::Categorical
            },
            tokens: std::mem::take(&mut feat_tokens[fi]),
        })
        .collect();
    let targets = tgt_pos
        .iter()
        .enumerate()
        .map(|(ti, &ci)| RawColumn {
            name: cols[ci].clone(),
            kind: ColKind::Numeric,
            tokens: std::mem::take(&mut tgt_tokens[ti]),
        })
        .collect();

    Ok(Frame {
        feats,
        targets,
        primary,
        n,
        issues,
    })
}
