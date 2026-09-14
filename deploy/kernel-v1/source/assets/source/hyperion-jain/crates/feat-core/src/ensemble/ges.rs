use super::LCB_T;
use crate::metrics::{auc, fold_auc_lcb, rank01};
use crate::rng::{seeded, Rng};
use rayon::prelude::*;

/// Bagged rank-aware Caruana greedy ensemble selection.
pub struct GesConfig {
    pub bags: usize,
    pub max_steps: usize,
    pub sorted_init: usize,
    pub use_rank: bool,
    pub seed: u64,
}

impl Default for GesConfig {
    fn default() -> Self {
        GesConfig {
            bags: 16,
            max_steps: 50,
            sorted_init: 2,
            use_rank: true,
            seed: 20260616,
        }
    }
}

pub(super) fn onehot(m: usize, i: usize) -> Vec<f64> {
    let mut v = vec![0.0; m];
    v[i] = 1.0;
    v
}

/// Weighted blend of member columns -> length-n vector.
pub(crate) fn blend(cols: &[Vec<f32>], w: &[f64]) -> Vec<f32> {
    let n = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut out = vec![0f32; n];
    for (m, col) in cols.iter().enumerate() {
        let wm = w[m] as f32;
        if wm == 0.0 {
            continue;
        }
        for i in 0..n {
            out[i] += wm * col[i];
        }
    }
    out
}

/// Rank-transform WITHIN each fold's validation rows. OOF predictions come from K different fold-models
/// with different output ranges, so ranks must be fold-local to be comparable in a blend (a global rank
/// would inject fold identity as noise and corrupt the ensemble).
pub(crate) fn rank01_per_fold(oof: &[f32], folds: &[(Vec<usize>, Vec<usize>)]) -> Vec<f32> {
    let mut out = vec![0f32; oof.len()];
    for (_, va) in folds {
        let vals: Vec<f32> = va.iter().map(|&i| oof[i]).collect();
        let r = rank01(&vals);
        for (k, &i) in va.iter().enumerate() {
            out[i] = r[k];
        }
    }
    out
}

fn auc_on_rows(col: &[f32], rows: &[usize], yr: &[f32]) -> f64 {
    let pv: Vec<f32> = rows.iter().map(|&i| col[i]).collect();
    auc(yr, &pv)
}

/// Greedy add-with-replacement on a bootstrap row sample; returns per-member selection counts.
fn greedy_counts(cols: &[Vec<f32>], y: &[f32], rows: &[usize], cfg: &GesConfig) -> Vec<f64> {
    let m = cols.len();
    let yr: Vec<f32> = rows.iter().map(|&i| y[i]).collect();
    let aucs: Vec<f64> = (0..m).map(|j| auc_on_rows(&cols[j], rows, &yr)).collect();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        aucs[b]
            .partial_cmp(&aucs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut counts = vec![0f64; m];
    let mut sum = vec![0f32; rows.len()];
    let mut nsel = 0usize;
    for &s in order.iter().take(cfg.sorted_init.max(1)) {
        for (k, &i) in rows.iter().enumerate() {
            sum[k] += cols[s][i];
        }
        counts[s] += 1.0;
        nsel += 1;
    }
    let mut best = auc(
        &yr,
        &sum.iter().map(|v| v / nsel as f32).collect::<Vec<f32>>(),
    );
    for _ in 0..cfg.max_steps {
        let cand_auc: Vec<f64> = (0..m)
            .into_par_iter()
            .map(|c| {
                let denom = (nsel + 1) as f32;
                let blended: Vec<f32> = (0..rows.len())
                    .map(|k| (sum[k] + cols[c][rows[k]]) / denom)
                    .collect();
                auc(&yr, &blended)
            })
            .collect();
        let (bc, ba) = cand_auc
            .iter()
            .enumerate()
            .fold(
                (usize::MAX, best),
                |acc, (c, &a)| {
                    if a > acc.1 {
                        (c, a)
                    } else {
                        acc
                    }
                },
            );
        if bc == usize::MAX || ba <= best + 1e-9 {
            break;
        }
        for (k, &i) in rows.iter().enumerate() {
            sum[k] += cols[bc][i];
        }
        counts[bc] += 1.0;
        nsel += 1;
        best = ba;
    }
    counts
}

/// Run bagged GES; returns (weights, OOF mean AUC, OOF LCB, active member count).
pub fn bagged_ges(
    cols: &[Vec<f32>],
    y: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cfg: &GesConfig,
) -> (Vec<f64>, f64, f64, usize) {
    let n = y.len();
    let m = cols.len();
    let mut rng = seeded(cfg.seed);
    let mut counts = vec![0f64; m];
    let mut total = 0f64;
    for _ in 0..cfg.bags {
        let rows: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let c = greedy_counts(cols, y, &rows, cfg);
        let s: f64 = c.iter().sum();
        for (j, cj) in c.iter().enumerate() {
            counts[j] += cj;
        }
        total += s;
    }
    let weights: Vec<f64> = if total > 0.0 {
        counts.iter().map(|c| c / total).collect()
    } else {
        vec![1.0 / m as f64; m]
    };
    let blended = blend(cols, &weights);
    let summ = fold_auc_lcb(&blended, y, folds, LCB_T);
    let active = weights.iter().filter(|w| **w > 1e-9).count();
    (weights, summ.mean, summ.lcb, active)
}
