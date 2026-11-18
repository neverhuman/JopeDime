//! Deterministic 2-class binary categorical/boolean target conditioning and holdout probability metrics.
//!
//! Maps arbitrary two-class target columns (`yes`/`no`, `true`/`false`, `+`/`-`, `<=50K`/`>50K`,
//! `benign`/`malignant`, or any two distinct string labels) into `{0.0, 1.0}` for client-side
//! `JMAT v1` preparation, while failing closed with actionable diagnostics on constant (`1`-class)
//! or multiclass (`>= 3`-class) string targets.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Error returned when a non-numeric target column cannot be conditioned as a 2-class binary target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryTargetError {
    /// Zero non-missing target values were present.
    Empty,
    /// Only one distinct class was observed across all non-missing target cells.
    ConstantCategorical { only_label: String },
    /// Three or more distinct categorical classes were observed (multiclass).
    MulticlassUnsupported {
        class_count: usize,
        sample_classes: Vec<String>,
    },
}

impl fmt::Display for BinaryTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("target column has zero non-missing values"),
            Self::ConstantCategorical { only_label } => write!(
                f,
                "target column is constant (only 1 distinct class '{only_label}'); binary preprocessing requires 2 distinct classes"
            ),
            Self::MulticlassUnsupported {
                class_count,
                sample_classes,
            } => write!(
                f,
                "target column has {class_count} distinct categorical classes ([{}]); multiclass targets (>= 3 classes) are not yet enabled in Carbon v1.3.0 (supported: numeric regression and 2-class binary targets)",
                sample_classes.join(", ")
            ),
        }
    }
}

impl std::error::Error for BinaryTargetError {}

/// Learned mapping between a 2-class categorical/boolean target and `{0.0, 1.0}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryTargetMapping {
    /// Canonical display label mapped to `0.0` (negative class).
    pub negative_label: String,
    /// Canonical display label mapped to `1.0` (positive class).
    pub positive_label: String,
    /// Whether label matching is ASCII case-insensitive (used when input casing varies).
    #[serde(default)]
    pub case_insensitive: bool,
    /// Whether standard boolean synonyms (`true`/`yes`/`y`/`t` vs `false`/`no`/`n`/`f`) are accepted.
    #[serde(default)]
    pub boolean_synonyms: bool,
}

/// Client-side binary holdout metrics computed on `{0.0, 1.0}` targets vs continuous/probabilistic predictions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryHoldoutMetrics {
    /// Mean squared probability error `mean((y - clamp(p, 0, 1))^2)`.
    pub brier: f64,
    /// Binary cross-entropy with probabilities clipped to `[1e-7, 1 - 1e-7]`.
    pub logloss: f64,
    /// Tie-aware Mann-Whitney ROC-AUC (`None` if holdout contains only one class).
    pub roc_auc: Option<f64>,
    /// Classification accuracy at probability threshold `0.5`.
    pub accuracy: f64,
}

fn strict_boolean_polarity(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "false" | "f" | "no" | "n" => Some(false),
        "true" | "t" | "yes" | "y" => Some(true),
        _ => None,
    }
}

fn semantic_polarity_hint(raw: &str) -> Option<bool> {
    let lower = raw.trim().to_ascii_lowercase();
    if let Some(b) = strict_boolean_polarity(&lower) {
        return Some(b);
    }
    match lower.as_str() {
        "0" | "off" | "negative" | "neg" | "-" | "fail" | "failed" | "benign" | "inactive"
        | "absent" | "control" | "retain" | "retained" | "declined" | "denied" | "rejected"
        | "lost" | "<=50k" | "<=50k." | "<50k" => Some(false),
        "1" | "on" | "positive" | "pos" | "+" | "pass" | "passed" | "malignant" | "active"
        | "present" | "treatment" | "churn" | "churned" | "approved" | "accepted" | "won"
        | ">50k" | ">50k." | ">=50k" => Some(true),
        _ => None,
    }
}

fn order_binary_pair(label_a: String, label_b: String) -> (String, String) {
    match (
        semantic_polarity_hint(&label_a),
        semantic_polarity_hint(&label_b),
    ) {
        (Some(false), Some(true)) | (Some(false), None) | (None, Some(true)) => (label_a, label_b),
        (Some(true), Some(false)) | (Some(true), None) | (None, Some(false)) => (label_b, label_a),
        _ => {
            if label_a <= label_b {
                (label_a, label_b)
            } else {
                (label_b, label_a)
            }
        }
    }
}

impl BinaryTargetMapping {
    /// Infer a deterministic `{0.0, 1.0}` mapping from non-missing string cells of a target column.
    pub fn infer(non_missing_cells: &[&str]) -> Result<Self, BinaryTargetError> {
        let mut exact_counts: BTreeMap<String, usize> = BTreeMap::new();
        for &raw in non_missing_cells {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                *exact_counts.entry(trimmed.to_owned()).or_insert(0) += 1;
            }
        }
        if exact_counts.is_empty() {
            return Err(BinaryTargetError::Empty);
        }
        if exact_counts.len() == 1 {
            let only_label = exact_counts.keys().next().cloned().unwrap_or_default();
            return Err(BinaryTargetError::ConstantCategorical { only_label });
        }
        if exact_counts.len() == 2 {
            let mut it = exact_counts.into_keys();
            let a = it.next().unwrap_or_default();
            let b = it.next().unwrap_or_default();
            let is_bool_pair =
                strict_boolean_polarity(&a).is_some() && strict_boolean_polarity(&b).is_some();
            if is_bool_pair && strict_boolean_polarity(&a) == strict_boolean_polarity(&b) {
                return Err(BinaryTargetError::ConstantCategorical { only_label: a });
            }
            let (negative_label, positive_label) = order_binary_pair(a, b);
            return Ok(Self {
                negative_label,
                positive_label,
                case_insensitive: is_bool_pair,
                boolean_synonyms: is_bool_pair,
            });
        }

        // Check ASCII case-folded classes (e.g., mixed `"Yes"`, `"yes"`, `"NO"`, `"no"`).
        let mut folded: BTreeMap<String, (usize, String)> = BTreeMap::new();
        for (spelling, count) in &exact_counts {
            let key = spelling.to_ascii_lowercase();
            let entry = folded.entry(key).or_insert((0, spelling.clone()));
            if *count > entry.0 {
                entry.1 = spelling.clone();
            }
            entry.0 += *count;
        }
        if folded.len() == 1 {
            let only_label = folded
                .values()
                .next()
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            return Err(BinaryTargetError::ConstantCategorical { only_label });
        }
        if folded.len() == 2 {
            let mut it = folded.into_values().map(|(_, disp)| disp);
            let a = it.next().unwrap_or_default();
            let b = it.next().unwrap_or_default();
            let is_bool_pair =
                strict_boolean_polarity(&a).is_some() && strict_boolean_polarity(&b).is_some();
            if is_bool_pair && strict_boolean_polarity(&a) == strict_boolean_polarity(&b) {
                return Err(BinaryTargetError::ConstantCategorical { only_label: a });
            }
            let (negative_label, positive_label) = order_binary_pair(a, b);
            return Ok(Self {
                negative_label,
                positive_label,
                case_insensitive: true,
                boolean_synonyms: is_bool_pair,
            });
        }

        // Check if every value is a strict boolean synonym (`true`/`yes`/`y`/`t` vs `false`/`no`/`n`/`f`).
        if exact_counts
            .keys()
            .all(|k| strict_boolean_polarity(k).is_some())
        {
            let mut best_neg: Option<(usize, String)> = None;
            let mut best_pos: Option<(usize, String)> = None;
            for (spelling, count) in &exact_counts {
                match strict_boolean_polarity(spelling) {
                    Some(false) if best_neg.as_ref().is_none_or(|(c, _)| *count > *c) => {
                        best_neg = Some((*count, spelling.clone()));
                    }
                    Some(true) if best_pos.as_ref().is_none_or(|(c, _)| *count > *c) => {
                        best_pos = Some((*count, spelling.clone()));
                    }
                    _ => {}
                }
            }
            return match (best_neg, best_pos) {
                (Some((_, negative_label)), Some((_, positive_label))) => Ok(Self {
                    negative_label,
                    positive_label,
                    case_insensitive: true,
                    boolean_synonyms: true,
                }),
                (Some((_, only)), None) | (None, Some((_, only))) => {
                    Err(BinaryTargetError::ConstantCategorical { only_label: only })
                }
                (None, None) => Err(BinaryTargetError::Empty),
            };
        }

        let sample_classes: Vec<String> = exact_counts.keys().take(6).cloned().collect();
        Err(BinaryTargetError::MulticlassUnsupported {
            class_count: exact_counts.len(),
            sample_classes,
        })
    }

    /// Map a raw string cell to `0.0` (negative) or `1.0` (positive), or `None` if unrecognized.
    pub fn encode_label(&self, raw: &str) -> Option<f64> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        if self.case_insensitive {
            if trimmed.eq_ignore_ascii_case(&self.negative_label) {
                return Some(0.0);
            }
            if trimmed.eq_ignore_ascii_case(&self.positive_label) {
                return Some(1.0);
            }
        } else {
            if trimmed == self.negative_label {
                return Some(0.0);
            }
            if trimmed == self.positive_label {
                return Some(1.0);
            }
        }
        if self.boolean_synonyms {
            return match strict_boolean_polarity(trimmed) {
                Some(false) => Some(0.0),
                Some(true) => Some(1.0),
                None => None,
            };
        }
        None
    }

    /// Clamp a continuous score into `[0.0, 1.0]` and return `(probability, predicted_label)` at `threshold`.
    pub fn decode_prediction(&self, raw_score: f64, threshold: f64) -> (f64, &str) {
        let prob = if raw_score.is_finite() {
            raw_score.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let label = if prob >= threshold {
            self.positive_label.as_str()
        } else {
            self.negative_label.as_str()
        };
        (prob, label)
    }
}

/// Compute Brier score, clipped log-loss, tie-aware ROC-AUC, and `0.5`-threshold accuracy on `{0.0, 1.0}` targets.
pub fn compute_binary_holdout_metrics(
    y_true: &[f64],
    y_pred_raw: &[f64],
) -> Option<BinaryHoldoutMetrics> {
    if y_true.is_empty() || y_true.len() != y_pred_raw.len() {
        return None;
    }
    let n = y_true.len();
    let mut brier_sum = 0.0_f64;
    let mut logloss_sum = 0.0_f64;
    let mut correct = 0usize;
    let mut n_pos = 0usize;
    let mut n_neg = 0usize;
    let mut pairs: Vec<(f64, bool)> = Vec::with_capacity(n);

    for (&yt, &yp) in y_true.iter().zip(y_pred_raw.iter()) {
        if !yt.is_finite() || !yp.is_finite() {
            return None;
        }
        let is_pos = yt >= 0.5;
        let y_bin = if is_pos { 1.0 } else { 0.0 };
        if is_pos {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
        let prob = yp.clamp(0.0, 1.0);
        let diff = y_bin - prob;
        brier_sum += diff * diff;

        let p_clip = prob.clamp(1e-7, 1.0 - 1e-7);
        logloss_sum -= if is_pos {
            p_clip.ln()
        } else {
            (1.0 - p_clip).ln()
        };

        if (prob >= 0.5) == is_pos {
            correct += 1;
        }
        pairs.push((yp, is_pos));
    }

    let roc_auc = if n_pos > 0 && n_neg > 0 {
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut pos_rank_sum = 0.0_f64;
        let mut idx = 0usize;
        while idx < n {
            let mut end = idx + 1;
            while end < n && pairs[end].0 == pairs[idx].0 {
                end += 1;
            }
            // 1-based midrank of tied span `[idx..end)`
            let avg_rank = 0.5 * ((idx + 1 + end) as f64);
            let pos_in_tie = pairs[idx..end].iter().filter(|&&(_, p)| p).count();
            pos_rank_sum += (pos_in_tie as f64) * avg_rank;
            idx = end;
        }
        let u_stat = pos_rank_sum - ((n_pos * (n_pos + 1)) as f64) * 0.5;
        Some((u_stat / ((n_pos * n_neg) as f64)).clamp(0.0, 1.0))
    } else {
        None
    };

    let denom = n as f64;
    Some(BinaryHoldoutMetrics {
        brier: brier_sum / denom,
        logloss: logloss_sum / denom,
        roc_auc,
        accuracy: (correct as f64) / denom,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_polarity_for_boolean_and_arbitrary_binary_targets() {
        let yn = BinaryTargetMapping::infer(&["yes", "no", "Yes", "NO"]).expect("yes/no");
        assert_eq!(yn.encode_label("yes"), Some(1.0));
        assert_eq!(yn.encode_label("YES"), Some(1.0));
        assert_eq!(yn.encode_label("no"), Some(0.0));
        assert_eq!(yn.encode_label("false"), Some(0.0));

        let adult = BinaryTargetMapping::infer(&[">50K", "<=50K", "<=50K"]).expect("adult");
        assert_eq!(adult.negative_label, "<=50K");
        assert_eq!(adult.positive_label, ">50K");
        assert_eq!(adult.encode_label(">50K"), Some(1.0));

        let arb = BinaryTargetMapping::infer(&["orange", "apple"]).expect("arbitrary");
        assert_eq!(arb.negative_label, "apple");
        assert_eq!(arb.positive_label, "orange");
        assert_eq!(arb.decode_prediction(0.82, 0.5), (0.82, "orange"));
        assert_eq!(arb.decode_prediction(0.12, 0.5), (0.12, "apple"));
    }

    #[test]
    fn refuses_constant_and_multiclass_targets_cleanly() {
        assert!(matches!(
            BinaryTargetMapping::infer(&["yes", "YES", "true"]),
            Err(BinaryTargetError::ConstantCategorical { .. })
        ));
        assert!(matches!(
            BinaryTargetMapping::infer(&["red", "green", "blue"]),
            Err(BinaryTargetError::MulticlassUnsupported { class_count: 3, .. })
        ));
    }
}
