//! Phase-1 proxy target: a per-row residual the GP searches to explain. Task-aware:
//! Binary/MultiLabel → `y − p̂` (out-of-fold proba); Regression → `y − ŷ` (raw); MultiClass → `1 − p_true`.
//! For Binary this is byte-identical to the original `y − oof_proba`.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::stratified_kfold;
use crate::task::{model_predict, TaskKind};

#[allow(clippy::too_many_arguments)]
pub fn oof_residual(
    x: &Matrix,
    y: &[f32],
    be: &dyn Backend,
    spec: &TrainSpec,
    folds: usize,
    seed: u64,
    task: TaskKind,
    n_classes: usize,
) -> Vec<f32> {
    let n = x.n;
    let p = x.c;
    let all: Vec<usize> = (0..n).collect();
    let parts = stratified_kfold(&all, y, folds, seed);
    let cols: Vec<u32> = (0..p as u32).collect();
    let mut resid = vec![0f32; n];
    for (tr, va) in &parts {
        let xtr = x.submatrix(tr, &cols);
        let ytr: Vec<f32> = tr.iter().map(|&i| y[i]).collect();
        let m = be.train_single(&xtr, tr.len(), p, &ytr, spec);
        let xva = x.submatrix(va, &cols);
        let pr = model_predict(&*m, &xva, va.len(), p, task, n_classes);
        match task {
            TaskKind::MultiClass => {
                for (j, &i) in va.iter().enumerate() {
                    let c = (y[i] as usize).min(n_classes.saturating_sub(1));
                    resid[i] = 1.0 - pr[j * n_classes + c];
                }
            }
            // Binary/MultiLabel: y − proba ; Regression: y − raw prediction (both rows×1).
            _ => {
                for (j, &i) in va.iter().enumerate() {
                    resid[i] = y[i] - pr[j];
                }
            }
        }
    }
    resid
}
