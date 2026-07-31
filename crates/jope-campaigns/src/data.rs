use anyhow::{bail, Context, Result};
use feat_math::Rows;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub(crate) struct RegressionData {
    pub(crate) x: Rows,
    pub(crate) y: Vec<f64>,
    pub(crate) features: usize,
}

pub(crate) fn read_regression_csv(path: &Path) -> Result<RegressionData> {
    Ok(read_csv_core(path, IndicatorMode::None)?.0)
}

/// Parse an already-authorized and already-hashed CSV byte snapshot. V2 proof
/// commands use this entrypoint so the bytes they attest are exactly the bytes
/// they evaluate, with no path reopen between hashing and parsing.
pub(crate) fn read_regression_csv_bytes(bytes: &[u8], source: &str) -> Result<RegressionData> {
    let text =
        std::str::from_utf8(bytes).with_context(|| format!("read regression CSV {source}"))?;
    Ok(read_csv_text_core(text, source, IndicatorMode::None)?.0)
}

/// Which missingness-indicator treatment a CSV load applies. Indicator column
/// sets are a TRAIN-derived schema: the test file must be loaded with
/// `Apply(train_schema)` so both files agree on width and column meaning —
/// deriving per-file would silently misalign features (or panic downstream)
/// whenever train and test have different missing patterns.
#[allow(dead_code)] // wired by the binary-classification campaign lane
pub(crate) enum IndicatorMode<'a> {
    /// Historical behavior: non-finite features silently become 0.0.
    None,
    /// Train side: flag every column with >= 1 non-finite cell in THIS file.
    Derive,
    /// Test side: append one 0/1 indicator per train-flagged column, in
    /// schema order, regardless of this file's own missing pattern.
    Apply(&'a [usize]),
}

/// Train-side loader: returns the widened data plus the derived indicator
/// schema (ascending original-column indices) that the test loader must be
/// given.
#[allow(dead_code)] // wired by the binary-classification campaign lane
pub(crate) fn read_regression_csv_derive_indicators(
    path: &Path,
) -> Result<(RegressionData, Vec<usize>)> {
    read_csv_core(path, IndicatorMode::Derive)
}

/// Test-side loader: applies a train-derived indicator schema verbatim.
#[allow(dead_code)] // wired by the binary-classification campaign lane
pub(crate) fn read_regression_csv_apply_indicators(
    path: &Path,
    schema: &[usize],
) -> Result<RegressionData> {
    Ok(read_csv_core(path, IndicatorMode::Apply(schema))?.0)
}

/// Strict binary-classification loader: the shared CSV reader plus target
/// validation. Every target must be exactly 0.0 or 1.0 (bitwise — 0.9999 is
/// rejected, never rounded), both classes must be present, and the minority
/// class needs at least 2 rows so stratified CV is constructible. The task
/// kind itself is manifest-pinned; this validation is the data-side check.
#[allow(dead_code)] // wired by the binary-classification campaign lane
pub(crate) fn read_binary_csv(path: &Path) -> Result<RegressionData> {
    let data = read_regression_csv(path)?;
    let mut n_pos = 0usize;
    for (row, value) in data.y.iter().enumerate() {
        if *value == 1.0 {
            n_pos += 1;
        } else if *value != 0.0 {
            bail!(
                "{}: row {} target {} is not a 0/1 class label",
                path.display(),
                row + 1,
                value
            );
        }
    }
    let n_neg = data.y.len() - n_pos;
    if n_pos.min(n_neg) < 2 {
        bail!(
            "{}: binary target needs both classes with >= 2 rows each (positives: {}, negatives: {})",
            path.display(),
            n_pos,
            n_neg
        );
    }
    Ok(data)
}

/// Shared CSV core. Returns the (possibly indicator-widened) data plus the
/// derived indicator schema (empty unless `IndicatorMode::Derive`).
fn read_csv_core(path: &Path, mode: IndicatorMode<'_>) -> Result<(RegressionData, Vec<usize>)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read regression CSV {}", path.display()))?;
    read_csv_text_core(&text, &path.display().to_string(), mode)
}

fn read_csv_text_core(
    text: &str,
    source: &str,
    mode: IndicatorMode<'_>,
) -> Result<(RegressionData, Vec<usize>)> {
    let track_missing = !matches!(mode, IndicatorMode::None);
    let mut x = Vec::new();
    let mut y = Vec::new();
    // Per-row bool mask (O(1) membership; a per-row index list would make the
    // indicator append quadratic on wide sparse data).
    let mut missing: Vec<Vec<bool>> = Vec::new();
    let mut features = None;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut values = line
            .split(',')
            .map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("{source}:{} invalid numeric field", line_no + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        if values.len() < 2 {
            bail!(
                "{}:{} needs at least one feature and one target",
                source,
                line_no + 1
            );
        }
        let width = values.len() - 1;
        match features {
            Some(expected) if expected != width => bail!(
                "{}:{} expected {} features, found {}",
                source,
                line_no + 1,
                expected,
                width
            ),
            None => features = Some(width),
            _ => {}
        }
        let target = values.pop().expect("length checked");
        if !target.is_finite() {
            bail!("{source}:{} target is not finite", line_no + 1);
        }
        if track_missing {
            missing.push(values.iter().map(|v| !v.is_finite()).collect());
        }
        values.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = 0.0;
            }
        });
        x.push(values);
        y.push(target);
    }
    let mut features = features.context("empty CSV")?;
    let schema: Vec<usize> = match mode {
        IndicatorMode::None => Vec::new(),
        IndicatorMode::Derive => (0..features)
            .filter(|&j| missing.iter().any(|row| row[j]))
            .collect(),
        IndicatorMode::Apply(schema) => {
            if let Some(&bad) = schema.iter().find(|&&j| j >= features) {
                bail!(
                    "{}: indicator schema column {} exceeds width {}",
                    source,
                    bad,
                    features
                );
            }
            schema.to_vec()
        }
    };
    if !schema.is_empty() {
        for (row, miss) in x.iter_mut().zip(&missing) {
            row.extend(schema.iter().map(|&j| if miss[j] { 1.0 } else { 0.0 }));
        }
        features += schema.len();
    }
    Ok((RegressionData { x, y, features }, schema))
}
