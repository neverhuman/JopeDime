//! Task inference — decide Binary / MultiClass / MultiLabel / Regression from the raw target column(s).
//!
//! This is the DATA-side task descriptor (what the labels look like). The model-side objective
//! (`backend::ObjectiveSpec`) is derived from it (see `ObjectiveSpec::from_task`, wired in a later step).
//! Keeping the two separate lets Phases 0-5 stay structurally task-agnostic: they consume a single
//! `TaskSpec` and a `task_score`/`model_predict` shim rather than hardcoding binary AUC.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    Binary,
    MultiClass,
    MultiLabel,
    Regression,
}

/// Resolved description of the learning task, inferred once from the training targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskSpec {
    pub kind: TaskKind,
    /// Number of target columns (1 except MultiLabel).
    pub n_targets: usize,
    /// Distinct classes (Binary=2, MultiClass=C, MultiLabel=2, Regression=0).
    pub n_classes: usize,
    /// Sorted distinct label values for single-target classification → maps raw value to class index.
    /// Empty for Regression / MultiLabel.
    pub class_values: Vec<f32>,
    /// Positive rate of the primary (classification); NaN for Regression. Feeds `infer_regime`.
    #[serde(with = "serde_prevalence")]
    pub prevalence: f64,
}

mod serde_prevalence {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.is_finite() {
            serializer.serialize_some(value)
        } else {
            serializer.serialize_none()
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::NAN))
    }
}

pub struct TaskInferConfig {
    /// An integer single target with `<= max_classes` distinct values is MultiClass; above → Regression.
    pub max_classes: usize,
}
impl Default for TaskInferConfig {
    fn default() -> Self {
        Self { max_classes: 20 }
    }
}

fn distinct_sorted(col: &[f32]) -> Vec<f32> {
    let mut v: Vec<f32> = col.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}
fn all_integer(vals: &[f32]) -> bool {
    vals.iter().all(|&x| (x - x.round()).abs() < 1e-6)
}
fn is_binary_01(col: &[f32]) -> bool {
    col.iter().all(|&x| !x.is_finite() || x == 0.0 || x == 1.0)
}

/// Infer the task from the (parsed, un-mapped) target columns. `primary` indexes the scored column.
pub fn infer_task(targets: &[Vec<f32>], primary: usize, cfg: &TaskInferConfig) -> TaskSpec {
    let k = targets.len();
    // MultiLabel: 2+ targets, each an independent binary {0,1} column.
    if k >= 2 && targets.iter().all(|c| is_binary_01(c)) {
        let prim = &targets[primary];
        let prev = prim.iter().filter(|&&x| x > 0.5).count() as f64 / prim.len().max(1) as f64;
        return TaskSpec {
            kind: TaskKind::MultiLabel,
            n_targets: k,
            n_classes: 2,
            class_values: vec![0.0, 1.0],
            prevalence: prev,
        };
    }
    // Single-target on the primary column.
    let col = &targets[primary];
    let d = distinct_sorted(col);
    let nd = d.len();
    let integer = all_integer(&d);
    if integer && nd <= 2 {
        // Binary (incl. degenerate 1-value). Prevalence = rate of the larger class value.
        let prev = if nd == 2 {
            let hi = d[1];
            col.iter().filter(|&&x| (x - hi).abs() < 1e-6).count() as f64 / col.len().max(1) as f64
        } else {
            f64::NAN
        };
        TaskSpec {
            kind: TaskKind::Binary,
            n_targets: k,
            n_classes: 2,
            class_values: d,
            prevalence: prev,
        }
    } else if integer && nd <= cfg.max_classes {
        TaskSpec {
            kind: TaskKind::MultiClass,
            n_targets: k,
            n_classes: nd,
            class_values: d,
            prevalence: f64::NAN,
        }
    } else {
        TaskSpec {
            kind: TaskKind::Regression,
            n_targets: k,
            n_classes: 0,
            class_values: vec![],
            prevalence: f64::NAN,
        }
    }
}

/// Task-aware prediction shim. The CatBoost shim applies per-dim sigmoid when `proba=true`, which is
/// correct for Binary/MultiLabel but wrong for MultiClass (needs softmax) and Regression (needs raw).
/// Returns rows×1 for Binary/Regression, rows×`n_classes` (row-major `r*k+c`) for MultiClass/MultiLabel.
pub fn model_predict(
    m: &dyn crate::backend::Model,
    x: &[f32],
    rows: usize,
    p: usize,
    kind: TaskKind,
    n_classes: usize,
) -> Vec<f32> {
    match kind {
        TaskKind::Binary | TaskKind::MultiLabel => m.predict(x, rows, p, true),
        TaskKind::Regression => m.predict(x, rows, p, false),
        TaskKind::MultiClass => {
            multiclass_softmax_to_full(m.predict(x, rows, p, false), rows, n_classes)
        }
    }
}

/// Softmax a raw multiclass score buffer into a full `rows × n_classes` probability matrix.
///
/// A model trained on a CV fold that did not observe every class (common with rare classes / small folds)
/// emits FEWER than `n_classes` columns. The old code fed that short buffer straight into `softmax_rows`
/// with stride `n_classes`, overrunning it (`range end index 693 out of range for slice of length 690`).
/// Here we softmax over the columns the model actually produced and place them into the leading slots,
/// leaving unobserved classes at ~0 probability — preserving the documented `rows × n_classes` contract and
/// making an overrun impossible.
fn multiclass_softmax_to_full(raw: Vec<f32>, rows: usize, n_classes: usize) -> Vec<f32> {
    if rows == 0 || n_classes == 0 {
        return raw;
    }
    let actual_k = raw.len() / rows;
    if actual_k == n_classes {
        return softmax_rows(raw, rows, n_classes);
    }
    let sm = softmax_rows(raw, rows, actual_k.max(1));
    let w = actual_k.min(n_classes);
    let mut full = vec![0f32; rows * n_classes];
    for r in 0..rows {
        for c in 0..w {
            full[r * n_classes + c] = sm[r * actual_k + c];
        }
    }
    full
}

fn softmax_rows(raw: Vec<f32>, rows: usize, k: usize) -> Vec<f32> {
    let mut out = raw;
    if k == 0 {
        return out;
    }
    for r in 0..rows {
        let start = r * k;
        if start + k > out.len() {
            break; // defensive: never index past the buffer, whatever the caller passed
        }
        let s = &mut out[start..start + k];
        let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in s.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in s.iter_mut() {
                *v /= sum;
            }
        }
    }
    out
}

impl TaskSpec {
    /// Map a raw label value to a class index (nearest class value). For {0,1} this equals `v > 0.5`,
    /// preserving today's binarization exactly; for {-1,1} or {2,5} it does the right thing.
    pub fn class_index(&self, v: f32) -> f32 {
        if self.class_values.is_empty() || !v.is_finite() {
            return 0.0;
        }
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (i, &cv) in self.class_values.iter().enumerate() {
            let dd = (v - cv).abs();
            if dd < bd {
                bd = dd;
                best = i;
            }
        }
        best as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression for the reported crash: a multiclass model that saw only a subset of classes in its CV
    // fold emits fewer than n_classes columns; producing the full rows×n_classes matrix must not overrun
    // (was: "range end index 693 out of range for slice of length 690" at softmax_rows).
    #[test]
    fn multiclass_short_output_does_not_overrun() {
        // 3 rows, model produced only 2 columns/row (len 6), but the task has 3 classes.
        let raw = vec![1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let out = multiclass_softmax_to_full(raw, 3, 3);
        assert_eq!(out.len(), 9); // rows * n_classes, no panic
        for r in 0..3 {
            assert!(out[r * 3 + 2].abs() < 1e-6); // unobserved 3rd class -> ~0
            let s: f32 = out[r * 3..r * 3 + 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-4); // observed classes softmax to 1
        }
    }

    #[test]
    fn multiclass_full_width_softmaxes_normally() {
        let out = multiclass_softmax_to_full(vec![1.0, 2.0, 3.0], 1, 3);
        assert_eq!(out.len(), 3);
        assert!((out.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(out[2] > out[0]); // monotonic in the logits
    }

    // softmax_rows itself must never overrun even if handed a mismatched stride.
    #[test]
    fn softmax_rows_never_overruns() {
        let out = softmax_rows(vec![1.0, 2.0, 3.0, 4.0], 10, 3); // claims 10 rows of 3 in a 4-len buffer
        assert_eq!(out.len(), 4); // returns without panic
    }
}
