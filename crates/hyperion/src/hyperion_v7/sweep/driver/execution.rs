use super::*;

/// Reuse exact scale/patch/permutation/chunk order and scatter normalization,
/// computing only the embedding branch. Cooperative stopping remains per chunk.
pub fn run_composite_embedding_sweep(
    model: &dyn PatchModel,
    xu8: &U8Grid,
    pad_u8: u8,
    quant_scale: f32,
    max_patch: usize,
    plan: &SweepPlan,
    sink: &dyn Sink,
) -> Result<EmbeddingSweepOutput> {
    let (embedding, _) = run_sweep(
        model,
        xu8,
        pad_u8,
        quant_scale,
        max_patch,
        plan,
        sink,
        |_| {},
        false,
    )?;
    Ok(embedding)
}

/// Inputs kept together for run composite sweep; no serialized contract changes.
#[derive(Clone, Copy)]
pub struct PatchInput {
    pub pad_u8: u8,
    pub quant_scale: f32,
    pub max_patch: usize,
}

/// The composite randomized sweep. Sweep 0 = identity permutations (UI moving front); sweeps 1..K
/// shuffle rows and columns with `seeded(plan.seed ^ mix64(k))`. Each sweep tiles every scale in
/// `plan.scales`, batching patches (chunk-capped so ~`target_chunks_per_sweep` frames stream per
/// sweep) and scattering back to ORIGINAL coordinates. `on_chunk` fires after every flush;
/// `sink.is_cancelled()` and `sink.take_advance_request(0)` are honored per chunk with a
/// cooperative partial return.
pub fn run_composite_sweep(
    model: &dyn PatchModel,
    xu8: &U8Grid,
    input: PatchInput,
    plan: &SweepPlan,
    sink: &dyn Sink,
    on_chunk: impl FnMut(ChunkReport<'_>),
) -> Result<SweepOutput> {
    let PatchInput {
        pad_u8,
        quant_scale,
        max_patch,
    } = input;

    let (embedding, confidence) = run_sweep(
        model,
        xu8,
        pad_u8,
        quant_scale,
        max_patch,
        plan,
        sink,
        on_chunk,
        true,
    )?;
    Ok(SweepOutput {
        cellmap: embedding.cellmap,
        h: embedding.h,
        w: embedding.w,
        row_emb: embedding.row_emb,
        confidence: confidence.context("full sweep confidence missing")?,
        dataset_vec: embedding.dataset_vec,
        cancelled: embedding.cancelled,
        advanced: embedding.advanced,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_sweep(
    model: &dyn PatchModel,
    xu8: &U8Grid,
    pad_u8: u8,
    quant_scale: f32,
    max_patch: usize,
    plan: &SweepPlan,
    sink: &dyn Sink,
    mut on_chunk: impl FnMut(ChunkReport<'_>),
    with_confidence: bool,
) -> Result<(EmbeddingSweepOutput, Option<Vec<f32>>)> {
    validate_plan(plan, max_patch)?;
    let (h, w) = (xu8.h, xu8.w);
    let mut acc = SweepAccum::new(h, w, with_confidence);
    let mut cancelled = false;
    let mut advanced = false;

    // Total patches per sweep across all scales (constant across sweeps).
    let total_patches: usize = plan
        .scales
        .iter()
        .map(|&p| h.div_ceil(p) * w.div_ceil(p))
        .sum();
    let chunk_cap = plan
        .batch
        .min((total_patches.div_ceil(plan.target_chunks_per_sweep.max(1))).max(1));

    let mut touched: Vec<(u32, u32)> = Vec::new();

    'sweeps: for k in 0..plan.sweeps {
        let mut perm_r: Vec<usize> = (0..h).collect();
        let mut perm_c: Vec<usize> = (0..w).collect();
        if let Some(fixed) = &plan.fixed_perms {
            if (k as usize) < fixed.len() {
                validate_permutation(&fixed[k as usize].0, h, "row")?;
                validate_permutation(&fixed[k as usize].1, w, "column")?;
                perm_r = fixed[k as usize].0.clone();
                perm_c = fixed[k as usize].1.clone();
            }
        } else if k > 0 {
            // Sweep 0 keeps the identity order so the first pass reads as a moving front.
            let mut rng = seeded(plan.seed ^ mix64(k as u64));
            perm_r.shuffle(&mut rng);
            perm_c.shuffle(&mut rng);
        }

        let mut patches_done = 0usize;
        for &p in &plan.scales {
            let area = p * p;
            let h_p = h.div_ceil(p) * p;
            let w_p = w.div_ceil(p) * p;
            let mut x_flat = vec![0.0f32; chunk_cap * area];
            let mut m_flat = vec![0.0f32; chunk_cap * area];
            let mut coords: Vec<PatchCoord> = Vec::with_capacity(chunk_cap);

            #[derive(Clone, Copy, PartialEq, Eq)]
            enum Stop {
                Continue,
                Cancelled,
                Advanced,
            }

            let mut flush = |coords: &mut Vec<PatchCoord>,
                             x_flat: &[f32],
                             m_flat: &[f32],
                             patches_done: &mut usize,
                             touched: &mut Vec<(u32, u32)>,
                             acc: &mut SweepAccum|
             -> Result<Stop> {
                touched.clear();
                flush_patch_batch(
                    model,
                    x_flat,
                    m_flat,
                    coords,
                    p,
                    h,
                    w,
                    &perm_r,
                    &perm_c,
                    acc,
                    touched,
                    with_confidence,
                )?;
                *patches_done += coords.len();
                coords.clear();
                let frac = *patches_done as f64 / total_patches.max(1) as f64;
                // Confidence-bearing progress is exposed only by the full path.
                if with_confidence {
                    on_chunk(ChunkReport {
                        sweep: k,
                        sweeps: plan.sweeps,
                        frac,
                        touched,
                        accum: acc,
                    });
                }
                if sink.is_cancelled() {
                    Ok(Stop::Cancelled)
                } else if sink.take_advance_request(0) {
                    Ok(Stop::Advanced)
                } else {
                    Ok(Stop::Continue)
                }
            };

            for r in (0..h_p).step_by(p) {
                for c in (0..w_p).step_by(p) {
                    let offset = coords.len() * area;
                    fill_one_patch(
                        &mut x_flat,
                        &mut m_flat,
                        offset,
                        r,
                        c,
                        p,
                        h,
                        w,
                        xu8,
                        &perm_r,
                        &perm_c,
                        pad_u8,
                        quant_scale,
                    );
                    coords.push(PatchCoord { r, c });
                    if coords.len() == chunk_cap {
                        match flush(
                            &mut coords,
                            &x_flat,
                            &m_flat,
                            &mut patches_done,
                            &mut touched,
                            &mut acc,
                        )? {
                            Stop::Continue => {}
                            Stop::Cancelled => {
                                cancelled = true;
                                break 'sweeps;
                            }
                            Stop::Advanced => {
                                advanced = true;
                                break 'sweeps;
                            }
                        }
                    }
                }
            }
            if !coords.is_empty() {
                match flush(
                    &mut coords,
                    &x_flat,
                    &m_flat,
                    &mut patches_done,
                    &mut touched,
                    &mut acc,
                )? {
                    Stop::Continue => {}
                    Stop::Cancelled => {
                        cancelled = true;
                        break 'sweeps;
                    }
                    Stop::Advanced => {
                        advanced = true;
                        break 'sweeps;
                    }
                }
            }
        }
    }

    let n_cells = h * w;
    // (1) Per-cell MEAN embedding = cell_sum / cell_count (covered cells; others stay 0).
    let mut cellmap = vec![0.0f32; n_cells * EMB_KEEP];
    for cell in 0..n_cells {
        let cnt = acc.cell_count[cell];
        if cnt > 0 {
            let inv = 1.0 / cnt as f32;
            let base = cell * EMB_KEEP;
            for d in 0..EMB_KEEP {
                cellmap[base + d] = acc.cell_sum[base + d] * inv;
            }
        }
    }
    // (2) Per-dim min-max normalize to [0,1] over the covered cells.
    let mut lo = vec![f32::INFINITY; EMB_KEEP];
    let mut hi = vec![f32::NEG_INFINITY; EMB_KEEP];
    for cell in 0..n_cells {
        if acc.cell_count[cell] > 0 {
            let base = cell * EMB_KEEP;
            for d in 0..EMB_KEEP {
                let v = cellmap[base + d];
                if v < lo[d] {
                    lo[d] = v;
                }
                if v > hi[d] {
                    hi[d] = v;
                }
            }
        }
    }
    // Env escape hatch for parity/debugging: JAIN_HYPERION_NO_NORMALIZE=1 keeps the raw per-cell
    // mean (skips the per-dim [0,1] rescale) so the encoder output can be compared without the
    // normalization amplifying tiny cross-framework diffs.
    let skip_norm = std::env::var("JAIN_HYPERION_NO_NORMALIZE").is_ok_and(|v| v == "1");
    if !skip_norm {
        for cell in 0..n_cells {
            if acc.cell_count[cell] > 0 {
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
    }
    // (3) Per-row features = col-mean of the normalized cell-map over covered cells (the v* columns).
    let mut row_emb = vec![0.0f32; h * EMB_KEEP];
    let mut row_covered = vec![false; h];
    for (r, covered) in row_covered.iter_mut().enumerate().take(h) {
        let mut nc = 0usize;
        for c in 0..w {
            let cell = r * w + c;
            if acc.cell_count[cell] > 0 {
                nc += 1;
                let base = cell * EMB_KEEP;
                let ro = r * EMB_KEEP;
                for d in 0..EMB_KEEP {
                    row_emb[ro + d] += cellmap[base + d];
                }
            }
        }
        if nc > 0 {
            *covered = true;
            let inv = 1.0 / nc as f32;
            let ro = r * EMB_KEEP;
            for d in 0..EMB_KEEP {
                row_emb[ro + d] *= inv;
            }
        }
    }
    let confidence =
        with_confidence.then(|| acc.cell_logit_mean.iter().map(|&l| sigmoid(l)).collect());
    // (4) Dataset vector = mean over covered rows of row_emb.
    let mut dataset_vec = vec![0.0f32; EMB_KEEP];
    let mut n_rows = 0usize;
    for (r, covered) in row_covered.iter().enumerate().take(h) {
        if *covered {
            n_rows += 1;
            let ro = r * EMB_KEEP;
            for d in 0..EMB_KEEP {
                dataset_vec[d] += row_emb[ro + d];
            }
        }
    }
    if n_rows > 0 {
        let inv = 1.0 / n_rows as f32;
        dataset_vec.iter_mut().for_each(|v| *v *= inv);
    }
    Ok((
        EmbeddingSweepOutput {
            cellmap,
            h,
            w,
            row_emb,
            dataset_vec,
            cancelled,
            advanced,
        },
        confidence,
    ))
}
