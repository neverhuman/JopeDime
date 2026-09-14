//! Phase 3 — single-target scorer (Logloss) for the EDA selector.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::task_score;
use crate::select::MaskScorer;
use crate::task::{model_predict, TaskKind};

pub struct SingleScorer<'a> {
    pub x: &'a Matrix,
    pub y: &'a [f32],
    pub xh: &'a Matrix,
    pub yh: &'a [f32],
    pub folds: &'a [(Vec<usize>, Vec<usize>)],
    pub be: &'a dyn Backend,
    pub cv_spec: TrainSpec,
    pub ho_spec: TrainSpec,
    pub task: TaskKind,
    pub n_classes: usize,
}

pub fn mean_std(a: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    if n < 1.0 {
        return (0.5, 0.0);
    }
    let m = a.iter().sum::<f64>() / n;
    let v = a.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, v.sqrt())
}

impl<'a> MaskScorer for SingleScorer<'a> {
    fn n_features(&self) -> usize {
        self.x.c
    }

    fn score_cv(&self, mask: &[u32]) -> (f64, f64) {
        self.score_cv_with_progress(mask, || false, |_, _| {})
            .unwrap_or((0.5, 0.0))
    }

    fn score_holdout(&self, mask: &[u32]) -> f64 {
        let p = mask.len();
        let xtr = self.x.select_cols(mask);
        let m = self
            .be
            .train_single(&xtr, self.x.n, p, self.y, &self.ho_spec);
        let xh = self.xh.select_cols(mask);
        let pr = model_predict(&*m, &xh, self.xh.n, p, self.task, self.n_classes);
        task_score(self.task, self.yh, &pr, self.n_classes)
    }
}

impl SingleScorer<'_> {
    pub fn score_cv_with_progress<C, F>(
        &self,
        mask: &[u32],
        should_cancel: C,
        mut on_fold: F,
    ) -> Option<(f64, f64)>
    where
        C: Fn() -> bool,
        F: FnMut(usize, usize),
    {
        let p = mask.len();
        let mut aucs = Vec::with_capacity(self.folds.len());
        for (fold_idx, (tr, va)) in self.folds.iter().enumerate() {
            if should_cancel() {
                return None;
            }
            let xtr = self.x.submatrix(tr, mask);
            let ytr: Vec<f32> = tr.iter().map(|&i| self.y[i]).collect();
            let m = self.be.train_single(&xtr, tr.len(), p, &ytr, &self.cv_spec);
            if should_cancel() {
                return None;
            }
            let xva = self.x.submatrix(va, mask);
            let pr = model_predict(&*m, &xva, va.len(), p, self.task, self.n_classes);
            let yva: Vec<f32> = va.iter().map(|&i| self.y[i]).collect();
            aucs.push(task_score(self.task, &yva, &pr, self.n_classes));
            on_fold(fold_idx + 1, self.folds.len());
        }
        Some(mean_std(&aucs))
    }
}
