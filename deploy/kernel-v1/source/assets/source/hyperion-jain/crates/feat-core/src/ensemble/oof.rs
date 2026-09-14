use super::{MemberPred, LCB_T};
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::task_score;
use crate::task::{model_predict, TaskKind};
use rayon::prelude::*;

/// Build a fold-pure OOF member: each train row is predicted by a model trained WITHOUT it (Invariant P);
/// the holdout is predicted by a single full-train refit (Invariant H). NaN-init + a finiteness assert
/// proves complete OOF coverage (Invariant C).
#[allow(clippy::too_many_arguments)]
pub fn build_oof_member(
    name: &str,
    be: &dyn Backend,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
) -> MemberPred {
    let n = y.len();
    let p = cols.len();
    // Folds are independent -> train them in parallel; each returns (val_rows, scalar_val_preds, fold_score).
    // `pr` from model_predict is rowsx1 for Binary/MultiLabel/Regression and rowsxK for MultiClass; the
    // scalar `oof` column (used by GES/blend) is pr[j] for scalar tasks, a placeholder for MultiClass
    // (which never goes through the blend path - see select_model).
    let fold_out: Vec<(Vec<usize>, Vec<f32>, f64)> = folds
        .par_iter()
        .map(|(tr, va)| {
            let xtr = x.submatrix(tr, cols);
            let ytr: Vec<f32> = tr.iter().map(|&i| y[i]).collect();
            // Guard: a constant fold-target crashes the tree backends ("only one unique value"). A rare
            // binary target can be all-one-class within a bootstrap fold; predict that constant instead.
            let f0 = ytr[0];
            if ytr.iter().all(|&v| v == f0) {
                let yva: Vec<f32> = va.iter().map(|&i| y[i]).collect();
                let oof_col = vec![f0; va.len()];
                let a = task_score(task, &yva, &oof_col, n_classes);
                return (va.clone(), oof_col, a);
            }
            let model = be.train_single(&xtr, tr.len(), p, &ytr, cv_spec);
            let xva = x.submatrix(va, cols);
            let pr = model_predict(&*model, &xva, va.len(), p, task, n_classes);
            let yva: Vec<f32> = va.iter().map(|&i| y[i]).collect();
            let a = task_score(task, &yva, &pr, n_classes);
            let scalar = pr.len() == va.len();
            let oof_col: Vec<f32> = (0..va.len())
                .map(|j| if scalar { pr[j] } else { pr[j * n_classes] })
                .collect();
            (va.clone(), oof_col, a)
        })
        .collect();
    let mut oof = vec![f32::NAN; n];
    let mut fold_auc = Vec::with_capacity(fold_out.len());
    for (va, pr, a) in &fold_out {
        for (j, &i) in va.iter().enumerate() {
            oof[i] = pr[j]; // write ONLY into held-out rows
        }
        fold_auc.push(*a);
    }
    debug_assert!(
        oof.iter().all(|v| v.is_finite()),
        "OOF coverage hole for member {name}"
    );
    // full-train refit -> holdout inference (holdout labels never read here)
    let y0 = y[0];
    let infer = if y.iter().all(|&v| v == y0) {
        vec![y0; xh.n] // constant target -> constant prediction (skip the crashing train)
    } else {
        let xfull = x.select_cols(cols);
        let full = be.train_single(&xfull, x.n, p, y, ho_spec);
        let xho = xh.select_cols(cols);
        model_predict(&*full, &xho, xh.n, p, task, n_classes)
    };

    let k = fold_auc.len().max(1) as f64;
    let mean = fold_auc.iter().sum::<f64>() / k;
    let var = fold_auc
        .iter()
        .map(|a| (a - mean) * (a - mean))
        .sum::<f64>()
        / k;
    let lcb = mean - LCB_T * var.sqrt() / k.sqrt();
    MemberPred {
        name: name.into(),
        oof,
        infer,
        fold_auc,
        mean_auc: mean,
        lcb_auc: lcb,
    }
}
