use anyhow::Result;

/// One tiled patch's top-left corner in PERMUTED coordinates.
#[derive(Clone, Copy, Debug)]
pub(super) struct PatchCoord {
    pub(super) r: usize,
    pub(super) c: usize,
}

/// Outputs of one model batch over `batch` p×p patches.
pub struct PatchOutputs {
    /// `batch × emb_dim` row-major patch embeddings.
    pub emb: Vec<f32>,
    /// `batch × p × p` row-major per-cell confidence LOGITS (sigmoid is applied on averages).
    pub logits: Vec<f32>,
}

/// The model the sweep drives — candle-free trait so the sweep layer unit-tests with a fake and the
/// real candle `V7Model` plugs in behind the `hyperion-cpu` feature.
pub trait PatchModel: Send + Sync {
    /// `x`/`m` are `batch × p × p` row-major canvases (values already normalized to model range;
    /// mask 1.0 = real-or-window-pad cell — canvas == window, so no mask-0 region exists).
    fn forward_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs>;
    /// Embedding-only forward for callers that only need dataset vectors. The default preserves
    /// existing fake/test models by discarding logits from the full forward.
    fn forward_emb_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<Vec<f32>> {
        Ok(self.forward_batch(x, m, batch, p)?.emb)
    }
    /// Embedding width returned per patch (v7: 512; the sweep keeps dims `0..EMB_KEEP`).
    fn emb_dim(&self) -> usize;
}
