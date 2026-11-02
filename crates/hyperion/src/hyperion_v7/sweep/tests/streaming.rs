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
fn raw_dataset_vector_matches_weighted_patch_mean() {
    let xu8 = grid(9, 7);
    let plan = SweepPlan {
        sweeps: 2,
        scales: vec![3, 5],
        batch: 4,
        seed: 17,
        target_chunks_per_sweep: 4,
        fixed_perms: Some(identity_perms(9, 7, 2)),
    };
    let got = run_raw_dataset_vector_sweep(&FakePatchModel, &xu8, 128, 3.0, 24, &plan).unwrap();

    let model = FakePatchModel;
    let mut expected = vec![0.0f32; EMB_KEEP];
    let mut total_weight = 0usize;
    for k in 0..plan.sweeps as usize {
        let (perm_r, perm_c) = plan.fixed_perms.as_ref().unwrap()[k].clone();
        for &p in &plan.scales {
            let area = p * p;
            for r in (0..xu8.h.div_ceil(p) * p).step_by(p) {
                for c in (0..xu8.w.div_ceil(p) * p).step_by(p) {
                    let mut x = vec![0.0f32; area];
                    let m = vec![1.0f32; area];
                    for i in 0..p {
                        for j in 0..p {
                            let val = if r + i < xu8.h && c + j < xu8.w {
                                xu8.get(perm_r[r + i], perm_c[c + j])
                            } else {
                                128
                            };
                            x[i * p + j] = normalize_u8_to_model(val, 3.0);
                        }
                    }
                    if r >= xu8.h || c >= xu8.w {
                        continue;
                    }
                    let out = model.forward_batch(&x, &m, 1, p).unwrap();
                    let weight = ((r + p).min(xu8.h) - r) * ((c + p).min(xu8.w) - c);
                    total_weight += weight;
                    for (dst, &src) in expected.iter_mut().zip(&out.emb[..EMB_KEEP]) {
                        *dst += src * weight as f32;
                    }
                }
            }
        }
    }
    let inv = 1.0 / total_weight as f32;
    expected.iter_mut().for_each(|v| *v *= inv);
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-5, "raw vector mismatch: {a} vs {b}");
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
        crate::hyperion_v7::sweep::PatchInput {
            pad_u8: 128,
            quant_scale: 3.0,
            max_patch: 24,
        },
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
            crate::hyperion_v7::sweep::PatchInput {
                pad_u8: 128,
                quant_scale: 3.0,
                max_patch: 24,
            },
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

type EmbeddingCall = (usize, usize, Vec<f32>, Vec<f32>);

/// Synthetic only: records exact arguments while deliberately forbidding the
/// confidence path in the embedding-only model.
#[derive(Default)]
struct EmbeddingTrace {
    calls: std::sync::Mutex<Vec<EmbeddingCall>>,
    forbid_confidence: bool,
}
impl PatchModel for EmbeddingTrace {
    fn forward_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<PatchOutputs> {
        anyhow::ensure!(!self.forbid_confidence, "confidence branch must not run");
        self.calls
            .lock()
            .unwrap()
            .push((batch, p, x.to_vec(), m.to_vec()));
        FakePatchModel.forward_batch(x, m, batch, p)
    }
    fn forward_emb_batch(&self, x: &[f32], m: &[f32], batch: usize, p: usize) -> Result<Vec<f32>> {
        self.calls
            .lock()
            .unwrap()
            .push((batch, p, x.to_vec(), m.to_vec()));
        // The fake's embeddings are deterministic; only the real V7 override is
        // the numerical-performance path. This fixture proves dispatch/order.
        Ok(FakePatchModel.forward_batch(x, m, batch, p)?.emb)
    }
    fn emb_dim(&self) -> usize {
        512
    }
}
