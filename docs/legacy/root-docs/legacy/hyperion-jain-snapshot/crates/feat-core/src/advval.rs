//! Adversarial validation: train a classifier to tell TRAIN rows from HOLDOUT rows using only the
//! features. AUC ≈ 0.5 ⇒ same distribution (CV will track the holdout); ≫ 0.5 ⇒ covariate shift (the
//! exact failure mode behind the original STS 2017-holdout). A trust diagnostic, not a hard gate.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::{auc, stratified_kfold};
use crate::task::{model_predict, TaskKind};

/// 3-fold out-of-fold AUC of a "is this row from the holdout?" classifier over the shared feature space.
pub fn adversarial_auc(
    train: &Matrix,
    hold: &Matrix,
    be: &dyn Backend,
    spec: &TrainSpec,
    seed: u64,
) -> f64 {
    assert_eq!(train.c, hold.c, "adversarial_auc: feature count mismatch");
    let (n_tr, n_ho, p) = (train.n, hold.n, train.c);
    let n = n_tr + n_ho;
    // Combined column-major matrix: train rows first, then holdout rows.
    let mut data = vec![0f32; n * p];
    for j in 0..p {
        let dst = &mut data[j * n..j * n + n];
        dst[..n_tr].copy_from_slice(train.col(j));
        dst[n_tr..].copy_from_slice(hold.col(j));
    }
    let x = Matrix::new(data, n, p);
    let y: Vec<f32> = (0..n).map(|i| if i < n_tr { 0.0 } else { 1.0 }).collect();
    let folds = stratified_kfold(&(0..n).collect::<Vec<_>>(), &y, 3, seed);
    let cols: Vec<u32> = (0..p as u32).collect();
    let mut oof = vec![0f32; n];
    for (tr, va) in &folds {
        let xtr = x.submatrix(tr, &cols);
        let ytr: Vec<f32> = tr.iter().map(|&i| y[i]).collect();
        let m = be.train_single(&xtr, tr.len(), p, &ytr, spec);
        let xva = x.submatrix(va, &cols);
        let pr = model_predict(&*m, &xva, va.len(), p, TaskKind::Binary, 2);
        for (j, &i) in va.iter().enumerate() {
            oof[i] = pr[j];
        }
    }
    let a = auc(&y, &oof);
    a.max(1.0 - a)
}
