// jankurai:allow repo-rot.path.fake-versioned-source reason=canonical-hyperion-orig-v2-sweep-owned-by-jain-core-proof-required-no-migration-while-profile-supported expires=2027-08-11
//! Old-compatible cell sweep used by the `.vs.orig.v2` profile.
//!
//! This intentionally differs from the interactive composite sweep: every sweep is shuffled,
//! two tiling origins are evaluated, padding is sampled, and the returned surface is the plain
//! arithmetic mean of raw `h_real` logits (no sigmoid, percentile, EWMA, or embedding collapse).

use anyhow::{bail, Result};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::hyperion_v7::grid::{normalize_u8_to_model, U8Grid};
use crate::rng::seeded;

use super::types::PatchModel;

pub const ORIG_V2_SWEEPS: u32 = 100;
pub const ORIG_V2_WINDOWS: [usize; 4] = [3, 6, 9, 12];
pub const ORIG_V2_SHIFTS: [(usize, usize); 2] = [(0, 0), (1, 1)];
pub const ORIG_V2_BASE_SEED: u64 = 42;
pub const ORIG_V2_SEED_STRIDE: u64 = 777;
pub const ORIG_V2_BATCH: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrigV2SweepPlan {
    pub sweeps: u32,
    pub windows: Vec<usize>,
    pub shifts: Vec<(usize, usize)>,
    pub batch: usize,
    pub base_seed: u64,
    pub seed_stride: u64,
}

impl Default for OrigV2SweepPlan {
    fn default() -> Self {
        Self {
            sweeps: ORIG_V2_SWEEPS,
            windows: ORIG_V2_WINDOWS.to_vec(),
            shifts: ORIG_V2_SHIFTS.to_vec(),
            batch: ORIG_V2_BATCH,
            base_seed: ORIG_V2_BASE_SEED,
            seed_stride: ORIG_V2_SEED_STRIDE,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrigV2SweepOutput {
    /// Column-agnostic row-major `h × w` arithmetic mean of raw cell logits.
    pub logits: Vec<f32>,
    pub h: usize,
    pub w: usize,
    /// Coverage is retained for receipts/tests. A complete run has
    /// `sweeps × windows × shifts` observations for every real cell.
    pub counts: Vec<u32>,
}

fn sampled_canvas(
    grid: &U8Grid,
    perm_r: &[usize],
    perm_c: &[usize],
    window: usize,
    shift: (usize, usize),
    rng: &mut crate::rng::Rng64,
) -> (Vec<u8>, usize, usize, usize, usize) {
    let (pad_top, pad_left) = (shift.1, shift.0);
    let pad_bottom = (window - (grid.h + pad_top) % window) % window;
    let pad_right = (window - (grid.w + pad_left) % window) % window;
    let height = grid.h + pad_top + pad_bottom;
    let width = grid.w + pad_left + pad_right;
    let mut canvas = vec![0u8; height * width];

    for row in 0..grid.h {
        for col in 0..grid.w {
            canvas[(pad_top + row) * width + pad_left + col] = grid.get(perm_r[row], perm_c[col]);
        }
    }
    for row in 0..pad_top {
        let sampled = rng.gen_range(0..grid.h);
        for col in 0..grid.w {
            canvas[row * width + pad_left + col] = grid.get(perm_r[sampled], perm_c[col]);
        }
    }
    for row in 0..pad_bottom {
        let sampled = rng.gen_range(0..grid.h);
        let dst = pad_top + grid.h + row;
        for col in 0..grid.w {
            canvas[dst * width + pad_left + col] = grid.get(perm_r[sampled], perm_c[col]);
        }
    }
    for col in 0..pad_left {
        let sampled = rng.gen_range(0..grid.w);
        for row in 0..height {
            canvas[row * width + col] = canvas[row * width + pad_left + sampled];
        }
    }
    for col in 0..pad_right {
        let sampled = rng.gen_range(0..grid.w);
        let dst = pad_left + grid.w + col;
        for row in 0..height {
            canvas[row * width + dst] = canvas[row * width + pad_left + sampled];
        }
    }
    (canvas, height, width, pad_top, pad_left)
}

#[allow(clippy::too_many_arguments)]
fn flush(
    model: &dyn PatchModel,
    x: &[f32],
    mask: &[f32],
    patch_origins: &[(usize, usize)],
    window: usize,
    pad_top: usize,
    pad_left: usize,
    grid: &U8Grid,
    perm_r: &[usize],
    perm_c: &[usize],
    sums: &mut [f64],
    counts: &mut [u32],
) -> Result<()> {
    if patch_origins.is_empty() {
        return Ok(());
    }
    let area = window * window;
    let batch = patch_origins.len();
    let out = model.forward_batch(&x[..batch * area], &mask[..batch * area], batch, window)?;
    if out.logits.len() != batch * area {
        bail!(
            "Hyperion orig-v2 model returned {} logits for {batch} patches of area {area}",
            out.logits.len()
        );
    }
    for (patch, &(origin_r, origin_c)) in patch_origins.iter().enumerate() {
        for local_r in 0..window {
            let canvas_r = origin_r + local_r;
            if canvas_r < pad_top || canvas_r >= pad_top + grid.h {
                continue;
            }
            let shuffled_r = canvas_r - pad_top;
            for local_c in 0..window {
                let canvas_c = origin_c + local_c;
                if canvas_c < pad_left || canvas_c >= pad_left + grid.w {
                    continue;
                }
                let shuffled_c = canvas_c - pad_left;
                let original = perm_r[shuffled_r] * grid.w + perm_c[shuffled_c];
                let value = out.logits[patch * area + local_r * window + local_c];
                if !value.is_finite() {
                    bail!("Hyperion orig-v2 emitted a non-finite raw logit");
                }
                sums[original] += value as f64;
                counts[original] = counts[original].saturating_add(1);
            }
        }
    }
    Ok(())
}

pub fn run_orig_v2_cell_sweep(
    model: &dyn PatchModel,
    grid: &U8Grid,
    quant_scale: f32,
    max_patch: usize,
    plan: &OrigV2SweepPlan,
) -> Result<OrigV2SweepOutput> {
    if grid.h == 0 || grid.w == 0 {
        bail!("Hyperion orig-v2 requires a non-empty matrix");
    }
    if plan.sweeps == 0 || plan.batch == 0 || plan.windows.is_empty() || plan.shifts.is_empty() {
        bail!("Hyperion orig-v2 sweep plan must be non-empty");
    }
    if plan
        .windows
        .iter()
        .any(|&window| window < 2 || window > max_patch)
    {
        bail!("Hyperion orig-v2 window is outside model patch geometry");
    }

    let mut sums = vec![0.0f64; grid.h * grid.w];
    let mut counts = vec![0u32; grid.h * grid.w];
    // Historical row/column permutations had a per-sweep generator. Sampled padding used a
    // separate global generator seeded once for the whole run.
    let mut padding_rng = seeded(plan.base_seed);

    for sweep in 0..plan.sweeps {
        let permutation_seed = plan
            .base_seed
            .wrapping_add(plan.seed_stride.wrapping_mul(sweep as u64));
        let mut permutation_rng = seeded(permutation_seed);
        let mut perm_r: Vec<usize> = (0..grid.h).collect();
        let mut perm_c: Vec<usize> = (0..grid.w).collect();
        perm_r.shuffle(&mut permutation_rng);
        perm_c.shuffle(&mut permutation_rng);

        for &shift in &plan.shifts {
            for &window in &plan.windows {
                let (canvas, height, width, pad_top, pad_left) =
                    sampled_canvas(grid, &perm_r, &perm_c, window, shift, &mut padding_rng);
                let area = window * window;
                let mut x = vec![0.0f32; plan.batch * area];
                let mask = vec![1.0f32; plan.batch * area];
                let mut origins = Vec::with_capacity(plan.batch);

                for origin_r in (0..height).step_by(window) {
                    for origin_c in (0..width).step_by(window) {
                        let offset = origins.len() * area;
                        for local_r in 0..window {
                            for local_c in 0..window {
                                let value =
                                    canvas[(origin_r + local_r) * width + origin_c + local_c];
                                x[offset + local_r * window + local_c] =
                                    normalize_u8_to_model(value, quant_scale);
                            }
                        }
                        origins.push((origin_r, origin_c));
                        if origins.len() == plan.batch {
                            flush(
                                model,
                                &x,
                                &mask,
                                &origins,
                                window,
                                pad_top,
                                pad_left,
                                grid,
                                &perm_r,
                                &perm_c,
                                &mut sums,
                                &mut counts,
                            )?;
                            origins.clear();
                        }
                    }
                }
                flush(
                    model,
                    &x,
                    &mask,
                    &origins,
                    window,
                    pad_top,
                    pad_left,
                    grid,
                    &perm_r,
                    &perm_c,
                    &mut sums,
                    &mut counts,
                )?;
            }
        }
    }

    let expected = plan.sweeps * plan.windows.len() as u32 * plan.shifts.len() as u32;
    if counts.iter().any(|&count| count != expected) {
        bail!("Hyperion orig-v2 did not cover every real cell exactly {expected} times");
    }
    let logits = sums
        .iter()
        .zip(&counts)
        .map(|(&sum, &count)| (sum / count as f64) as f32)
        .collect();
    Ok(OrigV2SweepOutput {
        logits,
        h: grid.h,
        w: grid.w,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperion_v7::sweep::{PatchModel, PatchOutputs};

    struct ConstantLogitModel(f32);

    impl PatchModel for ConstantLogitModel {
        fn forward_batch(
            &self,
            _x: &[f32],
            _m: &[f32],
            batch: usize,
            p: usize,
        ) -> Result<PatchOutputs> {
            Ok(PatchOutputs {
                emb: vec![0.0; batch * 128],
                logits: vec![self.0; batch * p * p],
            })
        }

        fn emb_dim(&self) -> usize {
            128
        }
    }

    #[test]
    fn orig_v2_returns_raw_all_sweep_mean_and_exact_cell_coverage() {
        let grid = U8Grid {
            data: (0..35).map(|value| value as u8).collect(),
            h: 7,
            w: 5,
        };
        let plan = OrigV2SweepPlan {
            sweeps: 3,
            windows: vec![3, 6],
            shifts: ORIG_V2_SHIFTS.to_vec(),
            batch: 2,
            ..OrigV2SweepPlan::default()
        };
        let output =
            run_orig_v2_cell_sweep(&ConstantLogitModel(-2.5), &grid, 3.0, 16, &plan).unwrap();
        assert_eq!((output.h, output.w), (7, 5));
        assert!(output.logits.iter().all(|&value| value == -2.5));
        assert!(output.counts.iter().all(|&count| count == 12));
    }

    #[test]
    fn orig_v2_is_deterministic_with_sampled_padding() {
        let grid = U8Grid {
            data: (0..30).map(|value| (value * 7) as u8).collect(),
            h: 6,
            w: 5,
        };
        let plan = OrigV2SweepPlan {
            sweeps: 2,
            windows: vec![3],
            shifts: ORIG_V2_SHIFTS.to_vec(),
            batch: 3,
            ..OrigV2SweepPlan::default()
        };
        let first =
            run_orig_v2_cell_sweep(&ConstantLogitModel(4.0), &grid, 3.0, 16, &plan).unwrap();
        let second =
            run_orig_v2_cell_sweep(&ConstantLogitModel(4.0), &grid, 3.0, 16, &plan).unwrap();
        assert_eq!(first, second);
    }
}
