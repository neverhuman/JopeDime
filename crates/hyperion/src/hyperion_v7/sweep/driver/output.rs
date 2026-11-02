use super::*;

/// Running accumulators across all sweeps/scales (ORIGINAL coordinates). Each window's 128-d
/// embedding is scattered into EVERY original cell it covered (rows/cols un-shuffled on write-back)
/// and summed; the dense per-cell map (`cell_sum / cell_count`, then per-dim 0..1) is assembled in
/// `run_composite_sweep`.
pub struct SweepAccum {
    pub h: usize,
    pub w: usize,
    /// `h*w*EMB_KEEP` row-major running SUM of per-cell embeddings (dims `0..EMB_KEEP`).
    pub cell_sum: Vec<f32>,
    /// `h × w` row-major running mean of per-cell confidence LOGITS.
    pub cell_logit_mean: Vec<f32>,
    /// `h × w` count of covering patches per cell (shared by the embedding sum + logit mean).
    pub cell_count: Vec<u32>,
}

impl SweepAccum {
    pub(super) fn new(h: usize, w: usize, with_confidence: bool) -> Self {
        Self {
            h,
            w,
            cell_sum: vec![0.0; h * w * EMB_KEEP],
            cell_logit_mean: if with_confidence {
                vec![0.0; h * w]
            } else {
                Vec::new()
            },
            cell_count: vec![0; h * w],
        }
    }

    /// Streaming dataset vector (for the live UI): mean over covered cells of the per-cell MEAN
    /// embedding (unnormalized; the final normalized map is built in `run_composite_sweep`).
    pub fn dataset_vector(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; EMB_KEEP];
        let mut n = 0usize;
        for cell in 0..self.h * self.w {
            let cnt = self.cell_count[cell];
            if cnt > 0 {
                n += 1;
                let inv = 1.0 / cnt as f32;
                let base = cell * EMB_KEEP;
                for (o, &s) in out.iter_mut().zip(&self.cell_sum[base..base + EMB_KEEP]) {
                    *o += s * inv;
                }
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            out.iter_mut().for_each(|v| *v *= inv);
        }
        out
    }

    /// Sigmoid of the current running-mean logit for one cell (0.5 for never-written cells).
    #[inline]
    pub fn cell_confidence(&self, r: u32, c: u32) -> f32 {
        let cell = r as usize * self.w + c as usize;
        sigmoid(self.cell_logit_mean[cell])
    }
}

#[inline]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One streamed chunk: the sweep position plus the cells this chunk updated, with read access to
/// the live accumulators so the caller can assemble a frame (vector + running confidences).
pub struct ChunkReport<'a> {
    pub sweep: u32,
    pub sweeps: u32,
    /// Fraction of THIS sweep completed (patches flushed / total patches, across all scales).
    pub frac: f64,
    pub touched: &'a [(u32, u32)],
    pub accum: &'a SweepAccum,
}

pub struct SweepOutput {
    /// DENSE per-cell encoding map, `h*w*EMB_KEEP` row-major. Each ORIGINAL cell holds the mean of
    /// every window embedding that swept over it, then per-dim min-max normalized to \[0,1\] over the
    /// covered cells. Uncovered cells are 0. This is the primary encoder output.
    pub cellmap: Vec<f32>,
    /// Grid dims for `cellmap` (`h` rows, `w` cols); the embedding dim is `EMB_KEEP`.
    pub h: usize,
    pub w: usize,
    /// `h × EMB_KEEP` row-major per-ROW features = col-mean of `cellmap` over covered cells
    /// (the `hyperion::v*` feature columns; now derived from the cell-map).
    pub row_emb: Vec<f32>,
    /// `h × w` row-major final per-cell confidences = sigmoid(mean logit) ∈ \[0,1\]
    /// (0.5 for cells never written, which only happens on a cancelled partial run).
    pub confidence: Vec<f32>,
    /// The 1×EMB_KEEP dataset vector = mean over rows of `row_emb`.
    pub dataset_vec: Vec<f32>,
    pub cancelled: bool,
    pub advanced: bool,
}

/// The same normalized embedding outputs as the full composite sweep. No
/// confidence head is executed or reported; this is not the raw-vector shortcut.
pub struct EmbeddingSweepOutput {
    pub cellmap: Vec<f32>,
    pub h: usize,
    pub w: usize,
    pub row_emb: Vec<f32>,
    pub dataset_vec: Vec<f32>,
    pub cancelled: bool,
    pub advanced: bool,
}
