use anyhow::{bail, Result};

use super::super::EMB_KEEP;
use super::driver::SweepAccum;
use super::types::{PatchCoord, PatchModel};

/// Run one model batch and scatter BOTH outputs back to ORIGINAL coordinates with
/// order-independent weighted running means (the veox flush rule):
/// - per-row embedding (dims `0..EMB_KEEP`), weight = number of in-range columns the patch covers;
/// - per-cell confidence logit, weight = 1 per covering patch (the reference `sum/count` rule).
///
/// Touched original cells are appended to `touched` for the streaming frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn flush_patch_batch(
    model: &dyn PatchModel,
    x_flat: &[f32],
    m_flat: &[f32],
    coords: &[PatchCoord],
    p: usize,
    h: usize,
    w: usize,
    perm_r: &[usize],
    perm_c: &[usize],
    acc: &mut SweepAccum,
    touched: &mut Vec<(u32, u32)>,
) -> Result<()> {
    let batch = coords.len();
    let area = p * p;
    let out = model.forward_batch(&x_flat[..batch * area], &m_flat[..batch * area], batch, p)?;
    let emb_dim = model.emb_dim();
    if out.emb.len() != batch * emb_dim {
        bail!(
            "Hyperion model returned {} emb floats for {batch} patches, expected {}",
            out.emb.len(),
            batch * emb_dim
        );
    }
    if out.logits.len() != batch * area {
        bail!(
            "Hyperion model returned {} logits for {batch} patches of area {area}, expected {}",
            out.logits.len(),
            batch * area
        );
    }
    if emb_dim < EMB_KEEP {
        bail!("Hyperion model emb_dim {emb_dim} is below EMB_KEEP {EMB_KEEP}");
    }

    for (idx, coord) in coords.iter().enumerate() {
        if coord.r >= h || coord.c >= w {
            continue; // patch fully in the padded remainder
        }
        let r_end = std::cmp::min(coord.r + p, h);
        let c_end = std::cmp::min(coord.c + p, w);
        let len_r = r_end - coord.r;
        let len_c = c_end - coord.c;
        if len_c == 0 {
            continue;
        }

        // Scatter the patch's 128-d embedding into EVERY original cell it covers (running SUM),
        // and the per-cell confidence LOGIT (running mean). One shared cell_count per cell.
        let emb_slice = &out.emb[idx * emb_dim..idx * emb_dim + EMB_KEEP];
        let logits = &out.logits[idx * area..(idx + 1) * area];
        for i in 0..len_r {
            let r_orig = perm_r[coord.r + i];
            for j in 0..len_c {
                let c_orig = perm_c[coord.c + j];
                let cell = r_orig * w + c_orig;
                let n = acc.cell_count[cell] + 1;
                // confidence logit: plain running mean (weight 1 per covering patch)
                let curr = acc.cell_logit_mean[cell];
                acc.cell_logit_mean[cell] = curr + (logits[i * p + j] - curr) / n as f32;
                // embedding: running SUM (finalized to mean / min-max in run_composite_sweep)
                let base = cell * EMB_KEEP;
                for (dst, &v) in acc.cell_sum[base..base + EMB_KEEP]
                    .iter_mut()
                    .zip(emb_slice)
                {
                    *dst += v;
                }
                acc.cell_count[cell] = n;
                touched.push((r_orig as u32, c_orig as u32));
            }
        }
    }
    Ok(())
}
