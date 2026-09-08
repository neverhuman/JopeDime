//! Shared EDA/PBIL feature selector for Phase 3 (single-target) and Phase 4 (MIMO). Ported from
//! gpfeat_eda_select.py / gpfeat_mimo_select.py.
//!
//! One per-feature inclusion-probability vector `q` (seeded from the Phase-2 memory_vector) is
//! evolved. Each generation samples masks ~ Bernoulli(q) (+ champion, neighbours, random
//! immigrants), scores them with a LOW-VARIANCE fixed-fold CV AUC, ranks by LCB, and enriches q
//! toward features over-represented in the elites. A keep-best champion ratchets up; the blind
//! holdout is reported each generation but NEVER used for selection.
use crate::progress::{Event, Sink};
use crate::rng::{seeded, Rng, Rng64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Scores a feature mask. The only difference between Phase 3 and Phase 4 is the implementation.
pub trait MaskScorer: Sync {
    fn n_features(&self) -> usize;
    /// Fixed-fold CV: returns (mean AUC, std AUC) of the scored target.
    fn score_cv(&self, mask: &[u32]) -> (f64, f64);
    /// Honest blind-holdout AUC (firewalled: reported only, never used for selection).
    fn score_holdout(&self, mask: &[u32]) -> f64;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EdaConfig {
    pub selection_mode: crate::mvp::SelectionMode,
    pub mvp: crate::mvp::MvpFastConfig,
    pub pop: usize,
    pub gens: usize,
    pub folds: usize,
    pub elite_frac: f32,
    pub lr: f32,
    pub random_frac: f32,
    pub target_size: f32,
    pub qmin: f32,
    pub qmax: f32,
    pub lcb_t: f32,
    pub warm_q: f32,
    pub seed: u64,
}

impl Default for EdaConfig {
    fn default() -> Self {
        EdaConfig {
            selection_mode: crate::mvp::SelectionMode::MvpFast,
            mvp: crate::mvp::MvpFastConfig::default(),
            pop: 80,
            gens: 40,
            folds: 5,
            elite_frac: 0.35,
            lr: 0.15,
            random_frac: 0.15,
            target_size: 150.0,
            qmin: 0.0,
            qmax: 0.9,
            lcb_t: 1.0,
            warm_q: 0.6,
            seed: 20260616,
        }
    }
}

#[derive(Clone)]
pub struct Champion {
    pub cols: Vec<u32>,
    pub cv_mean_auc: f64,
    pub holdout_auc: f64,
    pub gen: i64,
}

struct Best {
    cols: Vec<u32>,
    cvm: f64,
    lcb: f64,
    gen: i64,
}

fn weighted_pick(q: &[f32], sum: f32, rng: &mut Rng64) -> usize {
    let r = rng.gen::<f32>() * sum;
    let mut acc = 0f32;
    for (j, &qj) in q.iter().enumerate() {
        acc += qj;
        if acc >= r {
            return j;
        }
    }
    q.len() - 1
}

fn random_distinct(k: usize, f: usize, rng: &mut Rng64) -> Vec<u32> {
    let k = k.min(f);
    if k * 2 >= f {
        let mut idx: Vec<u32> = (0..f as u32).collect();
        for i in 0..k {
            let j = rng.gen_range(i..f);
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx.sort_unstable();
        return idx;
    }
    let mut set = std::collections::HashSet::new();
    while set.len() < k {
        set.insert(rng.gen_range(0..f as u32));
    }
    let mut v: Vec<u32> = set.into_iter().collect();
    v.sort_unstable();
    v
}

/// Run the EDA selector. `warm_champ` (Phase 4) seeds the champion + boosts q on its features.
pub fn run_eda(
    memory_vector: &[f32],
    warm_champ: Option<&[u32]>,
    scorer: &dyn MaskScorer,
    cfg: &EdaConfig,
    sink: &dyn Sink,
    phase: u8,
    t0: Instant,
) -> Champion {
    let f = scorer.n_features();
    let qmin = if cfg.qmin > 0.0 {
        cfg.qmin
    } else {
        0.5 / f as f32
    };
    let qmax = cfg.qmax;
    let mv_sum: f32 = memory_vector.iter().sum::<f32>().max(1e-12);
    let mut q: Vec<f32> = memory_vector
        .iter()
        .map(|&v| (v / mv_sum * cfg.target_size).clamp(qmin, qmax))
        .collect();
    if let Some(champ) = warm_champ {
        for &c in champ {
            q[c as usize] = q[c as usize].max(cfg.warm_q);
        }
    }

    let n_elite = ((cfg.elite_frac * cfg.pop as f32) as usize).max(2);
    let n_rand = (cfg.random_frac * cfg.pop as f32) as usize;
    let sqk = (cfg.folds as f64).sqrt();
    let mut rng = seeded(cfg.seed);

    let mut best: Option<Best> = warm_champ.map(|c| Best {
        cols: c.to_vec(),
        cvm: -1.0,
        lcb: -1.0,
        gen: -1,
    });

    {
        let mut e = Event::new(phase, "start", t0.elapsed().as_secs_f64());
        e.gens = Some(cfg.gens as u64);
        sink.emit(&e);
    }

    let mut gen_secs: Vec<f64> = Vec::new();
    for gen in 0..cfg.gens {
        if sink.is_cancelled() {
            break;
        }
        let gen_start = std::time::Instant::now();
        let qsum: f32 = q.iter().sum();
        let mut masks: Vec<Vec<u32>> = Vec::with_capacity(cfg.pop);
        if let Some(b) = &best {
            masks.push(b.cols.clone()); // champion at index 0 (re-scored on the fixed folds)
            for _ in 0..2 {
                let mut set: std::collections::HashSet<u32> = b.cols.iter().copied().collect();
                let nops = rng.gen_range(1..6);
                for _ in 0..nops {
                    if rng.gen::<f32>() < 0.5 && set.len() > 6 {
                        let victim = *set.iter().nth(rng.gen_range(0..set.len())).unwrap();
                        set.remove(&victim);
                    } else {
                        set.insert(weighted_pick(&q, qsum, &mut rng) as u32);
                    }
                }
                let mut v: Vec<u32> = set.into_iter().collect();
                v.sort_unstable();
                masks.push(v);
            }
        }
        while masks.len() < cfg.pop {
            let i = masks.len();
            let cols = if i < n_rand {
                let k = rng.gen_range(20..500);
                random_distinct(k, f, &mut rng)
            } else {
                let mut c: Vec<u32> = (0..f as u32)
                    .filter(|&j| rng.gen::<f32>() < q[j as usize])
                    .collect();
                if c.len() < 5 {
                    let mut idx: Vec<usize> = (0..f).collect();
                    idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap());
                    c = idx.into_iter().take(5).map(|x| x as u32).collect();
                    c.sort_unstable();
                }
                c
            };
            masks.push(cols);
        }

        // Evaluate the population in chunks (parallel within a chunk) so the TUI gets fine-grained
        // intra-generation progress + a live ETA as masks complete.
        let mut res: Vec<(f64, f64)> = Vec::with_capacity(masks.len());
        let chunk = (cfg.pop / 10).max(1);
        for cm in masks.chunks(chunk) {
            let part: Vec<(f64, f64)> = cm.par_iter().map(|m| scorer.score_cv(m)).collect();
            res.extend(part);
            let done = res.len();
            let cur = gen_start.elapsed().as_secs_f64();
            let per_mask = if done > 0 { cur / done as f64 } else { 0.0 };
            let est_gen = if gen_secs.is_empty() {
                per_mask * cfg.pop as f64
            } else {
                gen_secs.iter().sum::<f64>() / gen_secs.len() as f64
            };
            let eta = per_mask * cfg.pop.saturating_sub(done) as f64
                + est_gen * cfg.gens.saturating_sub(gen + 1) as f64;
            let mut e = Event::new(phase, "genprog", t0.elapsed().as_secs_f64());
            e.gen = Some(gen as u64);
            e.gens = Some(cfg.gens as u64);
            e.done = Some(done as u64);
            e.blocks_total = Some(cfg.pop as u64);
            e.done_frac = Some(done as f64 / cfg.pop as f64);
            e.eta = Some(eta);
            sink.emit(&e);
        }
        let cvm: Vec<f64> = res.iter().map(|r| r.0).collect();
        let cvs: Vec<f64> = res.iter().map(|r| r.1).collect();
        let lcb: Vec<f64> = (0..masks.len())
            .map(|i| cvm[i] - cfg.lcb_t as f64 * cvs[i] / sqk)
            .collect();
        let mut order: Vec<usize> = (0..masks.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            lcb[b]
                .partial_cmp(&lcb[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let elites = &order[..n_elite.min(order.len())];

        // enrichment update: q += lr * (elite_freq - pop_freq)
        let mut fe = vec![0f32; f];
        for &e in elites {
            for &c in &masks[e] {
                fe[c as usize] += 1.0;
            }
        }
        for v in &mut fe {
            *v /= elites.len() as f32;
        }
        let mut fp = vec![0f32; f];
        for m in &masks {
            for &c in m {
                fp[c as usize] += 1.0;
            }
        }
        for v in &mut fp {
            *v /= masks.len() as f32;
        }
        for j in 0..f {
            q[j] = (q[j] + cfg.lr * (fe[j] - fp[j])).clamp(qmin, qmax);
        }

        // champion update (fair: champion is masks[0] when present, so lcb[0] is its current score)
        let bi = order[0];
        let champ_present = best.is_some();
        let champ_lcb_now = if champ_present { lcb[0] } else { -1e9 };
        if !champ_present || lcb[bi] > champ_lcb_now {
            best = Some(Best {
                cols: masks[bi].clone(),
                cvm: cvm[bi],
                lcb: lcb[bi],
                gen: gen as i64,
            });
        } else if let Some(b) = best.as_mut() {
            b.cvm = cvm[0];
            b.lcb = lcb[0];
        }

        let bref = best.as_ref().unwrap();
        let champ_ho = scorer.score_holdout(&bref.cols);
        let mean_size = masks.iter().map(|m| m.len()).sum::<usize>() / masks.len();
        let this_gen = gen_start.elapsed().as_secs_f64();
        let avg_gen = (gen_secs.iter().sum::<f64>() + this_gen) / (gen_secs.len() + 1) as f64;
        {
            let mut e = Event::new(phase, "gen", t0.elapsed().as_secs_f64());
            e.gen = Some(gen as u64);
            e.gens = Some(cfg.gens as u64);
            e.best_val_auc = Some(bref.cvm);
            e.best_holdout_auc = Some(champ_ho);
            e.mean_val_auc = Some(cvm.iter().sum::<f64>() / cvm.len() as f64);
            e.best_size = Some(bref.cols.len() as u64);
            e.mean_size = Some(mean_size as u64);
            e.eta = Some(avg_gen * cfg.gens.saturating_sub(gen + 1) as f64);
            sink.emit(&e);
        }
        gen_secs.push(this_gen);
    }

    let b = best.unwrap();
    let holdout = scorer.score_holdout(&b.cols);
    {
        let mut e = Event::new(phase, "done", t0.elapsed().as_secs_f64());
        e.best_val_auc = Some(b.cvm);
        e.best_holdout_auc = Some(holdout);
        e.best_size = Some(b.cols.len() as u64);
        sink.emit(&e);
    }
    Champion {
        cols: b.cols,
        cv_mean_auc: b.cvm,
        holdout_auc: holdout,
        gen: b.gen,
    }
}
