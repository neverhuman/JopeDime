//! Candle-free sweep tests (ported + extended from the veox reference suite): the streaming
//! scatter must equal a brute-force materialized reference, be batch-size invariant, honor the
//! identity permutation on sweep 0, and produce confidences in [0,1] with a consistent vector.

use crate::hyperion_v7::config::SweepPlan;
use crate::hyperion_v7::frames::{build_frame, FRAME_MAX_CELLS, GRID_MAX_COLS, GRID_MAX_ROWS};
use crate::hyperion_v7::grid::{normalize_u8_to_model, U8Grid};
use crate::hyperion_v7::sweep::{run_composite_sweep, PatchModel, PatchOutputs};
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
        128,
        3.0,
        24,
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

#[test]
fn streaming_output_matches_materialized_weighted_mean() {
    let xu8 = grid(17, 7);
    let plan = SweepPlan {
        sweeps: 3,
        scales: vec![3, 5],
        batch: 4,
        seed: 42,
        target_chunks_per_sweep: 5,
        fixed_perms: Some(identity_perms(17, 7, 3)),
    };
    let (out, _) = run(&xu8, &plan);
    let (ref_cell, ref_emb, ref_conf) = reference(&xu8, &plan);
    assert_eq!(out.cellmap.len(), 17 * 7 * EMB_KEEP);
    assert_eq!((out.h, out.w), (17, 7));
    for (a, b) in out.cellmap.iter().zip(&ref_cell) {
        assert!((a - b).abs() < 1e-4, "cellmap mismatch: {a} vs {b}");
    }
    // cell-map is normalized to [0,1] on covered cells.
    assert!(out.cellmap.iter().all(|&v| (0.0..=1.0).contains(&v)));
    for (a, b) in out.row_emb.iter().zip(&ref_emb) {
        assert!((a - b).abs() < 1e-4, "row emb mismatch: {a} vs {b}");
    }
    for (a, b) in out.confidence.iter().zip(&ref_conf) {
        assert!((a - b).abs() < 1e-5, "confidence mismatch: {a} vs {b}");
    }
}

#[test]
fn batch_size_does_not_change_output() {
    let xu8 = grid(13, 6);
    let mk = |batch: usize, chunks: usize| SweepPlan {
        sweeps: 2,
        scales: vec![3, 7],
        batch,
        seed: 7,
        target_chunks_per_sweep: chunks,
        fixed_perms: Some(identity_perms(13, 6, 2)),
    };
    let (a, _) = run(&xu8, &mk(1, 1));
    let (b, _) = run(&xu8, &mk(7, 3));
    let (c, _) = run(&xu8, &mk(256, 8));
    for (x, y) in a.confidence.iter().zip(&b.confidence) {
        assert_eq!(x, y, "cell means must be exactly batch-invariant");
    }
    for (x, y) in b.confidence.iter().zip(&c.confidence) {
        assert_eq!(x, y);
    }
    for (x, y) in a.row_emb.iter().zip(&c.row_emb) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn sweep_zero_uses_identity_permutation() {
    // No fixed perms, seed set: sweep 0 must still be identity — the first patch's touched cells
    // are exactly rows 0..3 × cols 0..3 in i-major order.
    let xu8 = grid(9, 6);
    let plan = SweepPlan {
        sweeps: 2,
        scales: vec![3],
        batch: 1, // chunk per patch
        seed: 1234,
        target_chunks_per_sweep: 1000,
        fixed_perms: None,
    };
    let mut first_chunk: Option<Vec<(u32, u32)>> = None;
    run_composite_sweep(
        &FakePatchModel,
        &xu8,
        128,
        3.0,
        24,
        &plan,
        &NullSink,
        |report| {
            if first_chunk.is_none() {
                first_chunk = Some(report.touched.to_vec());
            }
        },
    )
    .unwrap();
    let expected: Vec<(u32, u32)> = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i as u32, j as u32)))
        .collect();
    assert_eq!(first_chunk.unwrap(), expected);
}

#[test]
fn confidence_in_unit_interval_and_vector_is_row_mean() {
    let xu8 = grid(11, 5);
    let plan = SweepPlan {
        sweeps: 2,
        scales: vec![3, 5],
        batch: 8,
        seed: 9,
        target_chunks_per_sweep: 4,
        fixed_perms: None,
    };
    let (out, frames) = run(&xu8, &plan);
    assert!(out.confidence.iter().all(|&v| (0.0..=1.0).contains(&v)));
    // dataset_vec == column means of row_emb over all rows (all rows have weight after a sweep).
    for d in 0..EMB_KEEP {
        let mean: f32 = (0..11).map(|r| out.row_emb[r * EMB_KEEP + d]).sum::<f32>() / 11.0;
        assert!((mean - out.dataset_vec[d]).abs() < 1e-4);
    }
    // The LAST frame's vector is the live (unnormalized) streaming preview — a well-formed
    // EMB_KEEP-length, all-finite vector. (It is NOT bit-equal to the final `dataset_vec`, which is
    // the col-mean of the per-dim 0..1 normalized cell-map; the preview is intentionally the raw
    // running fingerprint so the UI can render before the final normalization exists.)
    let last = frames.last().unwrap();
    assert_eq!(last.vector.len(), EMB_KEEP);
    assert!(last.vector.iter().all(|v| v.is_finite()));
    // Frames stream ≥ target chunks per sweep boundaries and frac ends at 1.0.
    assert!(frames.len() >= 2);
    assert!((frames.last().unwrap().frac - 1.0).abs() < 1e-9);
}

#[test]
fn invalid_plan_errors() {
    let xu8 = grid(6, 4);
    let bad = |f: fn(&mut SweepPlan)| {
        let mut plan = SweepPlan::default();
        f(&mut plan);
        run_composite_sweep(
            &FakePatchModel,
            &xu8,
            128,
            3.0,
            24,
            &plan,
            &NullSink,
            |_| {},
        )
        .err()
        .expect("plan must be rejected")
    };
    bad(|p| p.sweeps = 0);
    bad(|p| p.batch = 0);
    bad(|p| p.scales = vec![]);
    bad(|p| p.scales = vec![1]);
    bad(|p| p.scales = vec![30]); // > max_patch 24
}

#[test]
fn frames_identity_vs_binned_and_caps() {
    // Identity mode: small grid → literal original coordinates.
    let xu8 = grid(10, 4);
    let plan = SweepPlan {
        sweeps: 1,
        scales: vec![3],
        batch: 256,
        seed: 0,
        target_chunks_per_sweep: 1,
        fixed_perms: None,
    };
    let (_, frames) = run(&xu8, &plan);
    let f = frames.last().unwrap();
    assert_eq!((f.grid_rows, f.grid_cols), (10, 4));
    assert!(f
        .cells
        .iter()
        .all(|c| c.0 < 10 && c.1 < 4 && (0.0..=1.0).contains(&c.2)));
    assert!(f.cells.len() <= FRAME_MAX_CELLS);

    // Binned mode: rows exceed the cap → grid rows == cap, coordinates within the grid.
    let big = grid(300, 5);
    let plan2 = SweepPlan {
        sweeps: 1,
        scales: vec![5],
        batch: 256,
        seed: 0,
        target_chunks_per_sweep: 1,
        fixed_perms: None,
    };
    let (_, frames2) = run(&big, &plan2);
    let f2 = frames2.last().unwrap();
    assert_eq!(f2.grid_rows as usize, GRID_MAX_ROWS);
    assert_eq!(f2.grid_cols as usize, 5.min(GRID_MAX_COLS));
    assert!(f2
        .cells
        .iter()
        .all(|c| (c.0 as usize) < GRID_MAX_ROWS && c.1 < 5));
    assert!(f2.cells.len() <= FRAME_MAX_CELLS);
}
