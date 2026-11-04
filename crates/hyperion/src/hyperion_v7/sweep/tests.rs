//! Candle-free sweep tests (ported + extended from the veox reference suite): the streaming
//! scatter must equal a brute-force materialized reference, be batch-size invariant, honor the
//! identity permutation on sweep 0, and produce confidences in [0,1] with a consistent vector.

use crate::hyperion_v7::config::SweepPlan;
use crate::hyperion_v7::frames::{build_frame, FRAME_MAX_CELLS, GRID_MAX_COLS, GRID_MAX_ROWS};
use crate::hyperion_v7::grid::{normalize_u8_to_model, U8Grid};
use crate::hyperion_v7::sweep::{
    run_composite_sweep, run_raw_dataset_vector_sweep, PatchModel, PatchOutputs,
};
use crate::hyperion_v7::EMB_KEEP;
use crate::progress::{NullSink, SweepFrame};
use anyhow::Result;

/// Deterministic fake: emb[d] = patch-mean * (d+1) scaled; logits echo the canvas values.
struct FakePatchModel;

impl PatchModel for FakePatchModel {
    fn forward_batch(&self, x: &[f32], _m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        let area = p * p;
        let mut emb = vec![0.0f32; batch * self.emb_dim()];
        let mut logits = vec![0.0f32; batch * area];
        for b in 0..batch {
            let patch = &x[b * area..(b + 1) * area];
            let mean: f32 = patch.iter().sum::<f32>() / area as f32;
            for d in 0..self.emb_dim() {
                emb[b * self.emb_dim() + d] = mean * (1.0 + d as f32 * 0.01);
            }
            logits[b * area..(b + 1) * area].copy_from_slice(patch);
        }
        Ok(PatchOutputs { emb, logits })
    }

    fn emb_dim(&self) -> usize {
        512
    }
}

fn grid(h: usize, w: usize) -> U8Grid {
    // Deterministic content covering the full u8 range.
    let data = (0..h * w).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    U8Grid { data, h, w }
}

fn run(
    xu8: &U8Grid,
    plan: &SweepPlan,
) -> (crate::hyperion_v7::sweep::SweepOutput, Vec<SweepFrame>) {
    let mut frames = Vec::new();
    let out = run_composite_sweep(
        &FakePatchModel,
        xu8,
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: 128,
            quant_scale: 3.0,
            max_patch: 24,
        },
        plan,
        &NullSink,
        |report| {
            frames.push(build_frame(&report, report.frac, 0.0));
        },
    )
    .unwrap();
    (out, frames)
}

/// Brute-force reference: re-tile with the same perms, scatter each patch embedding into every
/// covered cell (running SUM), build the normalized cell-map, then derive the per-row features
/// (col-mean of the cell-map) and per-cell confidences. Returns `(cellmap, row_emb, conf)`.
fn reference(xu8: &U8Grid, plan: &SweepPlan) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (h, w) = (xu8.h, xu8.w);
    let model = FakePatchModel;
    let mut cell_sum = vec![0.0f64; h * w * EMB_KEEP];
    let mut logit_sum = vec![0.0f64; h * w];
    let mut cell_n = vec![0u32; h * w];
    for k in 0..plan.sweeps as usize {
        let (perm_r, perm_c) = plan.fixed_perms.as_ref().unwrap()[k].clone();
        for &p in &plan.scales {
            let area = p * p;
            for r in (0..h.div_ceil(p) * p).step_by(p) {
                for c in (0..w.div_ceil(p) * p).step_by(p) {
                    let mut x = vec![0.0f32; area];
                    let m = vec![1.0f32; area];
                    for i in 0..p {
                        for j in 0..p {
                            let val = if r + i < h && c + j < w {
                                xu8.get(perm_r[r + i], perm_c[c + j])
                            } else {
                                128
                            };
                            x[i * p + j] = normalize_u8_to_model(val, 3.0);
                        }
                    }
                    let out = model.forward_batch(&x, &m, 1, p).unwrap();
                    if r >= h || c >= w {
                        continue;
                    }
                    let len_r = (r + p).min(h) - r;
                    let len_c = (c + p).min(w) - c;
                    for i in 0..len_r {
                        let ro = perm_r[r + i];
                        for j in 0..len_c {
                            let co = perm_c[c + j];
                            let cell = ro * w + co;
                            cell_n[cell] += 1;
                            logit_sum[cell] += out.logits[i * p + j] as f64;
                            let base = cell * EMB_KEEP;
                            for d in 0..EMB_KEEP {
                                cell_sum[base + d] += out.emb[d] as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    // cell-map = per-cell mean, then per-dim min-max over covered cells.
    let mut cellmap = vec![0.0f32; h * w * EMB_KEEP];
    for (cell, &n) in cell_n.iter().enumerate().take(h * w) {
        if n > 0 {
            let inv = 1.0 / n as f64;
            let base = cell * EMB_KEEP;
            for d in 0..EMB_KEEP {
                cellmap[base + d] = (cell_sum[base + d] * inv) as f32;
            }
        }
    }
    let mut lo = vec![f32::INFINITY; EMB_KEEP];
    let mut hi = vec![f32::NEG_INFINITY; EMB_KEEP];
    for (cell, &n) in cell_n.iter().enumerate().take(h * w) {
        if n > 0 {
            let base = cell * EMB_KEEP;
            for d in 0..EMB_KEEP {
                let v = cellmap[base + d];
                lo[d] = lo[d].min(v);
                hi[d] = hi[d].max(v);
            }
        }
    }
    for (cell, &n) in cell_n.iter().enumerate().take(h * w) {
        if n > 0 {
            let base = cell * EMB_KEEP;
            for d in 0..EMB_KEEP {
                let span = hi[d] - lo[d];
                cellmap[base + d] = if span < 1e-9 {
                    0.0
                } else {
                    (cellmap[base + d] - lo[d]) / span
                };
            }
        }
    }
    // row features = col-mean of the cell-map over covered cells.
    let mut row_emb = vec![0.0f32; h * EMB_KEEP];
    for rr in 0..h {
        let mut nc = 0usize;
        for cc in 0..w {
            let cell = rr * w + cc;
            if cell_n[cell] > 0 {
                nc += 1;
                let base = cell * EMB_KEEP;
                for d in 0..EMB_KEEP {
                    row_emb[rr * EMB_KEEP + d] += cellmap[base + d];
                }
            }
        }
        if nc > 0 {
            let inv = 1.0 / nc as f32;
            for d in 0..EMB_KEEP {
                row_emb[rr * EMB_KEEP + d] *= inv;
            }
        }
    }
    let conf: Vec<f32> = (0..h * w)
        .map(|i| {
            let mean = if cell_n[i] > 0 {
                (logit_sum[i] / cell_n[i] as f64) as f32
            } else {
                0.0
            };
            1.0 / (1.0 + (-mean).exp())
        })
        .collect();
    (cellmap, row_emb, conf)
}

fn identity_perms(h: usize, w: usize, sweeps: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..sweeps)
        .map(|k| {
            // Distinct deterministic perms per sweep: rotate by k (a valid permutation).
            let pr: Vec<usize> = (0..h).map(|i| (i + k) % h).collect();
            let pc: Vec<usize> = (0..w).map(|j| (j + 2 * k) % w).collect();
            (pr, pc)
        })
        .collect()
}

include!("tests/streaming.rs");
include!("tests/embedding.rs");
