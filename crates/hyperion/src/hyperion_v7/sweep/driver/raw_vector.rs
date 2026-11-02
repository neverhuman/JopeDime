use super::*;

/// Fast raw dataset-vector path. This is equivalent to `run_composite_sweep(...).dataset_vec`
/// when `JAIN_HYPERION_NO_NORMALIZE=1`: every real cell is covered exactly once for each
/// sweep/scale, so mean-over-cells of the per-cell running mean equals the real-cell-weighted mean
/// of patch embeddings.
pub fn run_raw_dataset_vector_sweep(
    model: &dyn PatchModel,
    xu8: &U8Grid,
    pad_u8: u8,
    quant_scale: f32,
    max_patch: usize,
    plan: &SweepPlan,
) -> Result<Vec<f32>> {
    validate_plan(plan, max_patch)?;
    let (h, w) = (xu8.h, xu8.w);
    let emb_dim = model.emb_dim();
    if emb_dim < EMB_KEEP {
        anyhow::bail!("Hyperion model emb_dim {emb_dim} is below EMB_KEEP {EMB_KEEP}");
    }

    let chunk_cap = plan.batch.max(1);
    let mut dataset = vec![0.0f32; EMB_KEEP];
    let mut total_weight = 0usize;

    for k in 0..plan.sweeps {
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
            let mut rng = seeded(plan.seed ^ mix64(k as u64));
            perm_r.shuffle(&mut rng);
            perm_c.shuffle(&mut rng);
        }

        for &p in &plan.scales {
            let area = p * p;
            let h_p = h.div_ceil(p) * p;
            let w_p = w.div_ceil(p) * p;
            let mut x_flat = vec![0.0f32; chunk_cap * area];
            let mut m_flat = vec![0.0f32; chunk_cap * area];
            let mut coords: Vec<PatchCoord> = Vec::with_capacity(chunk_cap);

            let flush = |coords: &mut Vec<PatchCoord>,
                         x_flat: &[f32],
                         m_flat: &[f32],
                         dataset: &mut [f32],
                         total_weight: &mut usize|
             -> Result<()> {
                let batch = coords.len();
                if batch == 0 {
                    return Ok(());
                }
                let emb = model.forward_emb_batch(
                    &x_flat[..batch * area],
                    &m_flat[..batch * area],
                    batch,
                    p,
                )?;
                if emb.len() != batch * emb_dim {
                    anyhow::bail!(
                        "Hyperion model returned {} emb floats for {batch} patches, expected {}",
                        emb.len(),
                        batch * emb_dim
                    );
                }

                for (idx, coord) in coords.iter().enumerate() {
                    if coord.r >= h || coord.c >= w {
                        continue;
                    }
                    let len_r = std::cmp::min(coord.r + p, h) - coord.r;
                    let len_c = std::cmp::min(coord.c + p, w) - coord.c;
                    let weight = len_r * len_c;
                    if weight == 0 {
                        continue;
                    }
                    *total_weight += weight;
                    let emb_slice = &emb[idx * emb_dim..idx * emb_dim + EMB_KEEP];
                    for (dst, &src) in dataset.iter_mut().zip(emb_slice) {
                        *dst += src * weight as f32;
                    }
                }
                coords.clear();
                Ok(())
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
                        flush(
                            &mut coords,
                            &x_flat,
                            &m_flat,
                            &mut dataset,
                            &mut total_weight,
                        )?;
                    }
                }
            }
            flush(
                &mut coords,
                &x_flat,
                &m_flat,
                &mut dataset,
                &mut total_weight,
            )?;
        }
    }

    if total_weight > 0 {
        let inv = 1.0 / total_weight as f32;
        dataset.iter_mut().for_each(|v| *v *= inv);
    }
    Ok(dataset)
}
