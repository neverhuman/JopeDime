//! Row-major u8 grid + quantization (vendored from the reference `preprocess.rs`, adapted to use
//! jain's TRAIN-FITTED quantile bounds so hold/scoring matrices replay train-only state instead of
//! re-fitting per-matrix percentiles).

use crate::data::Matrix;
use crate::hyperion::HyperionQuantileBounds;

/// A row-major u8 matrix (feat-core has no ndarray; this replaces the reference's `Array2<u8>`).
#[derive(Clone, Debug)]
pub struct U8Grid {
    pub data: Vec<u8>,
    pub h: usize,
    pub w: usize,
}

impl U8Grid {
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> u8 {
        self.data[r * self.w + c]
    }
}

/// Quantize a (column-major) feat-core `Matrix` into a row-major u8 grid using per-column
/// train-fitted 1/99-percentile bounds. Non-finite cells map to `pad_u8` (the model's neutral
/// mid value); the mask stays pure geometry, matching both references.
pub fn quantize_u8_with_bounds(
    x: &Matrix,
    bounds: &[HyperionQuantileBounds],
    pad_u8: u8,
) -> U8Grid {
    let (h, w) = (x.n, x.c);
    let mut data = vec![0u8; h * w];
    for c in 0..w {
        let col = x.col(c);
        let b = bounds.get(c).cloned().unwrap_or(HyperionQuantileBounds {
            low: 0.0,
            high: 1.0,
        });
        let mut span = b.high - b.low;
        if !(span.is_finite()) || span < 1e-9 {
            span = 1.0;
        }
        for r in 0..h {
            let v = col[r];
            data[r * w + c] = if v.is_finite() {
                let norm = (v - b.low) / span;
                (norm * 255.0).clamp(0.0, 255.0) as u8 // truncation == floor here (matches numpy)
            } else {
                pad_u8
            };
        }
    }
    U8Grid { data, h, w }
}

/// Map a quantized u8 cell to the model's input range: `((v/255)*2 − 1) * quant_scale`.
#[inline]
pub fn normalize_u8_to_model(v: u8, quant_scale: f32) -> f32 {
    ((v as f32 / 255.0) * 2.0 - 1.0) * quant_scale
}
