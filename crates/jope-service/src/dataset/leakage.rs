//! Exact target leakage on a parsed table.
//!
//! A feature is flagged only when every row is a finite numeric pair and the
//! feature is the target or an exact affine function of it. A related column
//! such as a component of a score is not flagged.

use jope_api::{Warning, WarningState};
use jope_runtime::table::CsvTable;

/// How a feature repeats the selected target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetLeakKind {
    /// The feature equals the target on every row.
    Copy,
    /// `feature = slope * target + intercept`, slope not zero, residual at float noise.
    Affine,
}

/// One feature that repeats the selected target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetLeak {
    pub feature: String,
    pub target: String,
    pub kind: TargetLeakKind,
}

/// Features that are an exact function of `target`.
///
/// The target column itself is not a feature. A constant target, a constant
/// feature, a missing cell, or fewer than three rows produces no finding.
pub fn target_leakage(table: &CsvTable, target: &str) -> Vec<TargetLeak> {
    let Some(target_index) = table.headers.iter().position(|name| name == target) else {
        return Vec::new();
    };
    let mut leaks = Vec::new();
    for (index, name) in table.headers.iter().enumerate() {
        if index == target_index {
            continue;
        }
        let Some(pairs) = paired(table, index, target_index) else {
            continue;
        };
        let Some(kind) = relation(&pairs) else {
            continue;
        };
        leaks.push(TargetLeak {
            feature: name.clone(),
            target: target.to_owned(),
            kind,
        });
    }
    leaks
}

/// A raised `TARGET_LEAKAGE` warning for one finding.
pub fn warning(leak: &TargetLeak) -> Result<Warning, crate::warnings::WarningInputError> {
    let message = match leak.kind {
        TargetLeakKind::Copy => format!(
            "`{}` copies the target `{}` on every row.",
            leak.feature, leak.target
        ),
        TargetLeakKind::Affine => format!(
            "`{}` is an exact linear function of the target `{}`.",
            leak.feature, leak.target
        ),
    };
    crate::warnings::catalog("TARGET_LEAKAGE")
        .expect("TARGET_LEAKAGE is catalogued")
        .event(
            WarningState::Raised,
            None,
            None,
            message,
            format!(
                "Drop `{}` from the features, or ignore that column, before treating the score as honest.",
                leak.feature
            ),
        )
}

fn paired(table: &CsvTable, feature: usize, target: usize) -> Option<Vec<(f64, f64)>> {
    if table.rows.len() < 3 {
        return None;
    }
    let mut pairs = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        let feature = numeric(row.get(feature)?.as_deref()?)?;
        let target = numeric(row.get(target)?.as_deref()?)?;
        pairs.push((target, feature));
    }
    Some(pairs)
}

fn numeric(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

fn relation(pairs: &[(f64, f64)]) -> Option<TargetLeakKind> {
    if pairs.iter().all(|(target, feature)| feature == target) {
        return Some(TargetLeakKind::Copy);
    }
    let count = pairs.len() as f64;
    let mean_target = pairs.iter().map(|(target, _)| target).sum::<f64>() / count;
    let mean_feature = pairs.iter().map(|(_, feature)| feature).sum::<f64>() / count;
    let mut variance = 0.0;
    let mut covariance = 0.0;
    for (target, feature) in pairs {
        let centered = target - mean_target;
        variance += centered * centered;
        covariance += centered * (feature - mean_feature);
    }
    if variance <= 1e-18 {
        return None;
    }
    let slope = covariance / variance;
    if slope.abs() <= 1e-12 {
        return None;
    }
    let intercept = mean_feature - slope * mean_target;
    let scale = pairs
        .iter()
        .map(|(_, feature)| feature.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1e-8 * scale;
    pairs
        .iter()
        .all(|(target, feature)| (feature - (slope * target + intercept)).abs() <= tolerance)
        .then_some(TargetLeakKind::Affine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(headers: &[&str], rows: &[&[Option<&str>]]) -> CsvTable {
        CsvTable {
            headers: headers.iter().map(|header| (*header).to_owned()).collect(),
            rows: rows
                .iter()
                .map(|row| row.iter().map(|cell| cell.map(str::to_owned)).collect())
                .collect(),
            sha256: String::new(),
            bytes: 0,
        }
    }

    #[test]
    fn an_exact_copy_and_an_exact_affine_feature_are_flagged() {
        let parsed = table(
            &["units", "copy", "scaled", "noise", "total"],
            &[
                &[Some("10"), Some("1"), Some("3"), Some("1.2"), Some("1")],
                &[Some("20"), Some("2"), Some("5"), Some("9"), Some("2")],
                &[Some("30"), Some("4"), Some("9"), Some("4"), Some("4")],
            ],
        );
        let leaks = target_leakage(&parsed, "total");
        assert_eq!(
            leaks
                .iter()
                .map(|leak| (leak.feature.as_str(), leak.kind))
                .collect::<Vec<_>>(),
            vec![
                ("copy", TargetLeakKind::Copy),
                ("scaled", TargetLeakKind::Affine),
            ]
        );
    }

    #[test]
    fn a_related_column_a_constant_and_a_hole_are_not_flagged() {
        let parsed = table(
            &["part", "flat", "hole", "total"],
            &[
                &[Some("1"), Some("7"), Some("1"), Some("3")],
                &[Some("1"), Some("7"), None, Some("4")],
                &[Some("2"), Some("7"), Some("2"), Some("5")],
            ],
        );
        assert!(target_leakage(&parsed, "total").is_empty());
        assert!(target_leakage(&parsed, "missing").is_empty());
    }
}
