//! Phase 4 — MIMO scorer (MultiLogloss over all targets, scoring only the primary column).
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::task_score;
use crate::phase3::mean_std;
use crate::select::MaskScorer;
use crate::task::TaskKind;

pub struct MimoScorer<'a> {
    pub x: &'a Matrix,
    pub y: &'a [f32],       // primary target, length n (for stratify/score)
    pub y_multi: &'a [f32], // column-major k*n (train)
    pub k: usize,
    pub primary: usize,
    pub xh: &'a Matrix,
    pub yh: &'a [f32], // primary holdout target
    pub folds: &'a [(Vec<usize>, Vec<usize>)],
    pub be: &'a dyn Backend,
    pub cv_spec: TrainSpec,
    pub ho_spec: TrainSpec,
    pub task: TaskKind,
    pub n_classes: usize,
}

impl<'a> MimoScorer<'a> {
    /// Build a column-major (k x |rows|) multi-target buffer for the given training rows.
    fn multi_rows(&self, rows: &[usize]) -> Vec<f32> {
        let n = self.x.n;
        let mut out = Vec::with_capacity(rows.len() * self.k);
        for t in 0..self.k {
            let src = &self.y_multi[t * n..(t + 1) * n];
            for &i in rows {
                out.push(src[i]);
            }
        }
        out
    }
    fn primary_col(&self, pred: &[f32], rows: usize) -> Vec<f32> {
        (0..rows).map(|i| pred[i * self.k + self.primary]).collect()
    }
}

impl<'a> MaskScorer for MimoScorer<'a> {
    fn n_features(&self) -> usize {
        self.x.c
    }

    fn score_cv(&self, mask: &[u32]) -> (f64, f64) {
        let p = mask.len();
        let mut aucs = Vec::with_capacity(self.folds.len());
        for (tr, va) in self.folds {
            let xtr = self.x.submatrix(tr, mask);
            let ytr = self.multi_rows(tr);
            let m = self
                .be
                .train_multi(&xtr, tr.len(), p, &ytr, self.k, &self.cv_spec);
            let xva = self.x.submatrix(va, mask);
            let pr = m.predict(&xva, va.len(), p, true);
            let col = self.primary_col(&pr, va.len());
            let yva: Vec<f32> = va.iter().map(|&i| self.y[i]).collect();
            aucs.push(task_score(self.task, &yva, &col, self.n_classes));
        }
        mean_std(&aucs)
    }

    fn score_holdout(&self, mask: &[u32]) -> f64 {
        let p = mask.len();
        let xtr = self.x.select_cols(mask);
        let m = self
            .be
            .train_multi(&xtr, self.x.n, p, self.y_multi, self.k, &self.ho_spec);
        let xh = self.xh.select_cols(mask);
        let pr = m.predict(&xh, self.xh.n, p, true);
        let col = self.primary_col(&pr, self.xh.n);
        task_score(self.task, self.yh, &col, self.n_classes)
    }
}
