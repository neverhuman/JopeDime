use super::{BackendKind, TrainSpec};
use std::path::Path;

/// A validation pool for fixed-fold early stopping (column-major, like the train matrix).
pub struct ValidSet<'a> {
    pub x: &'a [f32],
    pub m: usize,
    pub y: &'a [f32],
}

/// Result of an (optionally early-stopped) fit.
pub struct FitOutcome {
    pub model: Box<dyn Model>,
    pub best_iteration: usize,
}

/// A trained model.
pub trait Model {
    fn output_dim(&self) -> usize;

    /// Predict on column-major `x` (rows x p). Returns `rows*output_dim`, layout `[obj*dim + cls]`.
    fn predict(&self, x: &[f32], rows: usize, p: usize, proba: bool) -> Vec<f32>;

    /// PredictionValuesChange-style importance, length `p`.
    fn importance(&self, p: usize) -> Vec<f32>;

    /// Snapshot predict using only the first `ntree_end` boosting iterations. Default = full model.
    fn predict_ntree(
        &self,
        x: &[f32],
        rows: usize,
        p: usize,
        proba: bool,
        ntree_end: usize,
    ) -> Vec<f32> {
        let _ = ntree_end;
        self.predict(x, rows, p, proba)
    }

    /// Number of boosting iterations (0 => snapshots unsupported).
    fn tree_count(&self) -> usize {
        0
    }

    /// Persist native model state. Pure-Rust/mock models return an error unless they opt in.
    fn save_native(&self, path: &Path) -> Result<(), String> {
        let _ = path;
        Err("model does not support native persistence".into())
    }
}

/// A GBM training backend.
pub trait Backend: Send + Sync {
    fn train_single(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        s: &TrainSpec,
    ) -> Box<dyn Model>;

    fn train_multi(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y_multi: &[f32],
        k: usize,
        s: &TrainSpec,
    ) -> Box<dyn Model>;

    /// Fixed-fold AUC early stopping. Default: no early stopping.
    fn fit_single_es(
        &self,
        x: &[f32],
        n: usize,
        p: usize,
        y: &[f32],
        valid: Option<ValidSet<'_>>,
        s: &TrainSpec,
    ) -> FitOutcome {
        let _ = valid;
        FitOutcome {
            model: self.train_single(x, n, p, y, s),
            best_iteration: s.iters,
        }
    }

    /// Which backend this is (for the OOF-library member registry). Default Mock.
    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }
}
