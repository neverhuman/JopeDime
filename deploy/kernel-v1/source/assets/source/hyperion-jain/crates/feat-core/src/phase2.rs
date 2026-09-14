//! Phase 2 — importance epochs. Over E shuffled passes, partition the F features into blocks,
//! fit a quick model per block on a stratified subsample, take PredictionValuesChange importance
//! (normalized per block to sum 1), and accumulate into a `memory_vector` at canonical indices.
//! Finish with an epsilon floor so every feature keeps a small nonzero weight.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::stratified_subsample;
use crate::progress::{Event, Sink};
use crate::rng::{seeded, Rng};
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone)]
pub struct P2Config {
    pub epochs: usize,
    pub block: usize,
    pub subsample: usize,
    pub iters: usize,
    pub depth: usize,
    pub thread_count: usize,
    pub floor: f32,
    pub seed: u64,
}

impl Default for P2Config {
    fn default() -> Self {
        P2Config {
            epochs: 5,
            block: 100,
            subsample: 25000,
            iters: 150,
            depth: 6,
            thread_count: 2,
            floor: 0.05,
            seed: 20260616,
        }
    }
}

pub struct ImportanceEpochs {
    pub memory_vector: Vec<f32>,
    pub importance_percentile: Vec<f32>,
}

/// Returns the memory_vector (length F, sums to 1).
pub fn importance_epochs(
    x: &Matrix,
    y: &[f32],
    be: &dyn Backend,
    cfg: &P2Config,
    objective: crate::backend::ObjectiveSpec,
    sink: &dyn Sink,
    t0: Instant,
) -> Vec<f32> {
    importance_epochs_detailed(x, y, be, cfg, objective, sink, t0).memory_vector
}

/// Returns both the normalized memory vector and the smoothed per-feature importance percentile.
pub fn importance_epochs_detailed(
    x: &Matrix,
    y: &[f32],
    be: &dyn Backend,
    cfg: &P2Config,
    objective: crate::backend::ObjectiveSpec,
    sink: &dyn Sink,
    t0: Instant,
) -> ImportanceEpochs {
    let f_total = x.c;
    if f_total == 0 {
        return ImportanceEpochs {
            memory_vector: Vec::new(),
            importance_percentile: Vec::new(),
        };
    }
    let block = cfg.block.max(1);
    let mut spec = TrainSpec::from_basic_fields(cfg.iters, cfg.depth, 0.1, cfg.thread_count, 0);
    spec.objective = objective; // task-aware (RMSE for regression, etc.); Binary -> Logloss unchanged
    let bpe = f_total.div_ceil(block);
    let blocks_total = (bpe * cfg.epochs) as u64;

    let mut mv = vec![0f64; f_total];
    let mut percentile = vec![0f64; f_total];

    {
        let mut e = Event::new(2, "start", t0.elapsed().as_secs_f64());
        e.blocks_total = Some(blocks_total);
        e.done = Some(0);
        e.done_frac = Some(0.0);
        sink.emit(&e);
    }

    let mut done = 0u64;
    for epoch in 0..cfg.epochs {
        if sink.is_cancelled() {
            break;
        }
        let mut rng = seeded(cfg.seed.wrapping_add(epoch as u64));
        // random feature permutation
        let mut perm: Vec<u32> = (0..f_total as u32).collect();
        for i in (1..f_total).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let rows = stratified_subsample(y, cfg.subsample.min(x.n), &mut rng);
        let ysub: Vec<f32> = rows.iter().map(|&i| y[i]).collect();

        let blocks: Vec<Vec<u32>> = perm.chunks(block).map(|c| c.to_vec()).collect();
        // train one model per block in parallel; return (block_cols, normalized importance)
        let results: Vec<(Vec<u32>, Vec<f32>)> = blocks
            .par_iter()
            .map(|cols| {
                let bp = cols.len();
                let xb = x.submatrix(&rows, cols);
                let m = be.train_single(&xb, rows.len(), bp, &ysub, &spec);
                let mut imp = m.importance(bp);
                let s: f32 = imp.iter().sum::<f32>().max(1e-12);
                for v in &mut imp {
                    *v /= s;
                }
                (cols.clone(), imp)
            })
            .collect();

        for (cols, imp) in &results {
            let pct = block_percentiles(imp);
            for (k, &c) in cols.iter().enumerate() {
                mv[c as usize] += imp[k] as f64;
                percentile[c as usize] += pct[k] as f64;
            }
        }
        done += bpe as u64;

        // top features so far (for the TUI histogram)
        let mut e = Event::new(2, "block", t0.elapsed().as_secs_f64());
        e.blocks_total = Some(blocks_total);
        e.done = Some(done);
        e.done_frac = Some(done as f64 / blocks_total as f64);
        e.top = Some(top_k(&mv, 6));
        sink.emit(&e);
    }

    // normalize to sum 1, apply epsilon floor, renormalize
    let s: f64 = mv.iter().sum::<f64>().max(1e-12);
    for v in &mut mv {
        *v /= s;
    }
    let floor = cfg.floor as f64;
    let f = f_total as f64;
    for v in &mut mv {
        *v = (1.0 - floor) * *v + floor / f;
    }
    let s2: f64 = mv.iter().sum::<f64>().max(1e-12);
    let out: Vec<f32> = mv.iter().map(|&v| (v / s2) as f32).collect();
    let pct_den = cfg.epochs.max(1) as f64;
    let pct_out: Vec<f32> = percentile.iter().map(|&v| (v / pct_den) as f32).collect();

    {
        let mut e = Event::new(2, "done", t0.elapsed().as_secs_f64());
        e.blocks_total = Some(blocks_total);
        e.done = Some(blocks_total);
        e.done_frac = Some(1.0);
        e.top = Some(top_k(&out, 6));
        sink.emit(&e);
    }
    ImportanceEpochs {
        memory_vector: out,
        importance_percentile: pct_out,
    }
}

pub fn block_percentiles(importance: &[f32]) -> Vec<f32> {
    let n = importance.len();
    if n == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        importance[a]
            .partial_cmp(&importance[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut out = vec![0.0; n];
    let denom = (n + 1) as f32;
    for (rank_from_bottom, idx) in order.into_iter().enumerate() {
        out[idx] = (rank_from_bottom as f32 + 1.0) / denom;
    }
    out
}

fn top_k<T: Into<f64> + Copy + PartialOrd>(mv: &[T], k: usize) -> Vec<(u32, f64)> {
    let mut idx: Vec<usize> = (0..mv.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        mv[b]
            .partial_cmp(&mv[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter()
        .take(k)
        .map(|i| (i as u32, mv[i].into()))
        .collect()
}
