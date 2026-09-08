//! Pre-search leakage detectors. Run on the TRAIN features only, before any model search, to catch
//! the obvious ways a "lift" can be an illusion. Detectors are cheap (column-major, rayon) and return
//! severity-tagged flags; a `Blocker` should abort a headline claim under `--leakage-strict`.
use crate::data::Matrix;
use crate::hashing::row_hash;
use crate::metrics::auc;
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LeakageSeverity {
    Info,
    Warn,
    Blocker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakageKind {
    SuspiciousName,
    TargetNameSibling,
    SingleFeatureAucSpike,
    ConstantColumn,
    DuplicateRowsAcrossFolds,
}

#[derive(Clone, Debug)]
pub struct LeakageFlag {
    pub severity: LeakageSeverity,
    pub kind: LeakageKind,
    pub feature_idx: Option<usize>,
    pub feature_name: Option<String>,
    pub statistic: f64,
    pub message: String,
}

#[derive(Clone, Debug, Default)]
pub struct LeakageReport {
    pub flags: Vec<LeakageFlag>,
    pub max_single_feature_auc: f64,
    pub duplicate_rate: f64,
}

impl LeakageReport {
    pub fn worst(&self) -> LeakageSeverity {
        self.flags
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(LeakageSeverity::Info)
    }
    pub fn n_blockers(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| f.severity == LeakageSeverity::Blocker)
            .count()
    }
}

fn suspicious_name(name: &str) -> Option<(LeakageSeverity, LeakageKind)> {
    let l = name.to_ascii_lowercase();
    const BLOCK: [&str; 7] = [
        "target",
        "label",
        "outcome",
        "response",
        "future",
        "days_until",
        "days_to_event",
    ];
    const WARN: [&str; 9] = [
        "postop",
        "post_op",
        "after",
        "followup",
        "discharge",
        "complication",
        "readmit",
        "mortality",
        "result",
    ];
    if BLOCK.iter().any(|s| l.contains(s)) {
        let kind = if l.starts_with("target") {
            LeakageKind::TargetNameSibling
        } else {
            LeakageKind::SuspiciousName
        };
        Some((LeakageSeverity::Blocker, kind))
    } else if WARN.iter().any(|s| l.contains(s)) {
        Some((LeakageSeverity::Warn, LeakageKind::SuspiciousName))
    } else {
        None
    }
}

/// Scan the (train) feature matrix + names + target. `folds` (when provided) enables cross-fold
/// duplicate-row detection over the scope's partition.
pub fn scan(
    x: &Matrix,
    y: &[f32],
    names: &[String],
    folds: Option<&[(Vec<usize>, Vec<usize>)]>,
) -> LeakageReport {
    let p = x.c;
    let mut flags: Vec<LeakageFlag> = Vec::new();

    // 1. Suspicious / target-family names (belt-and-suspenders vs the load_csv exclusion).
    for (j, nm) in names.iter().enumerate() {
        if let Some((sev, kind)) = suspicious_name(nm) {
            flags.push(LeakageFlag {
                severity: sev,
                kind,
                feature_idx: Some(j),
                feature_name: Some(nm.clone()),
                statistic: 0.0,
                message: format!("feature '{nm}' has a target-like name"),
            });
        }
    }

    // 2. Single-feature |AUC| spike + 3. constant columns (parallel over columns).
    let per_col: Vec<(usize, f64, bool)> = (0..p)
        .into_par_iter()
        .map(|j| {
            let col = x.col(j);
            let a0 = auc(y, col);
            let a = a0.max(1.0 - a0);
            let first = col.iter().copied().find(|v| v.is_finite());
            let is_const = match first {
                None => true,
                Some(f0) => !col.iter().any(|&v| v.is_finite() && (v - f0).abs() > 1e-12),
            };
            (j, a, is_const)
        })
        .collect();
    let mut max_auc = 0.5f64;
    for (j, a, is_const) in &per_col {
        if *a > max_auc {
            max_auc = *a;
        }
        if *a >= 0.995 {
            flags.push(LeakageFlag {
                severity: LeakageSeverity::Blocker,
                kind: LeakageKind::SingleFeatureAucSpike,
                feature_idx: Some(*j),
                feature_name: names.get(*j).cloned(),
                statistic: *a,
                message: format!("single-feature AUC {a:.4} (near-perfect separator)"),
            });
        } else if *a >= 0.95 {
            flags.push(LeakageFlag {
                severity: LeakageSeverity::Warn,
                kind: LeakageKind::SingleFeatureAucSpike,
                feature_idx: Some(*j),
                feature_name: names.get(*j).cloned(),
                statistic: *a,
                message: format!("single-feature AUC {a:.4} (suspiciously high)"),
            });
        }
        if *is_const {
            flags.push(LeakageFlag {
                severity: LeakageSeverity::Info,
                kind: LeakageKind::ConstantColumn,
                feature_idx: Some(*j),
                feature_name: names.get(*j).cloned(),
                statistic: 0.0,
                message: "constant column".into(),
            });
        }
    }

    // 4. Duplicate rows across folds (a val row identical to an in-fold train row).
    let mut dup_rate = 0.0;
    if let Some(folds) = folds {
        let n = x.n;
        let hashes: Vec<u64> = (0..n).into_par_iter().map(|r| row_hash(x, r)).collect();
        let (mut dup, mut tot) = (0usize, 0usize);
        for (tr, va) in folds {
            let trset: HashSet<u64> = tr.iter().map(|&i| hashes[i]).collect();
            for &i in va {
                tot += 1;
                if trset.contains(&hashes[i]) {
                    dup += 1;
                }
            }
        }
        if tot > 0 {
            dup_rate = dup as f64 / tot as f64;
        }
        if dup_rate >= 0.10 {
            flags.push(LeakageFlag {
                severity: LeakageSeverity::Blocker,
                kind: LeakageKind::DuplicateRowsAcrossFolds,
                feature_idx: None,
                feature_name: None,
                statistic: dup_rate,
                message: format!(
                    "{:.1}% of validation rows duplicate an in-fold train row",
                    dup_rate * 100.0
                ),
            });
        } else if dup_rate >= 0.02 {
            flags.push(LeakageFlag {
                severity: LeakageSeverity::Warn,
                kind: LeakageKind::DuplicateRowsAcrossFolds,
                feature_idx: None,
                feature_name: None,
                statistic: dup_rate,
                message: format!("{:.1}% duplicate rows across folds", dup_rate * 100.0),
            });
        }
    }

    LeakageReport {
        flags,
        max_single_feature_auc: max_auc,
        duplicate_rate: dup_rate,
    }
}
