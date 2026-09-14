use crate::canary::CanarySpec;
use crate::data::Matrix;
use crate::dsl::Expr;
use crate::metrics::{sse, standardize};
use crate::rng::{seeded, Rng};
use serde::Serialize;

/// Inputs the GP engine searches over.
pub struct GpData {
    pub x: Matrix, // base features, column-major (NaN preserved)
    pub names: Vec<String>,
    pub r: Vec<f32>, // residual proxy target (finite), e.g. y - oof
    pub r_std: Vec<f32>,
    pub r_sse: f32,
    pub y_std: Vec<f32>,        // standardized target (for reporting target_corr)
    pub lib_std: Vec<Vec<f32>>, // standardized base columns (decorrelation library)
    pub sub: Vec<usize>,        // fixed subsample for fast correlation sketches
    pub holdout: Vec<usize>,    // holdout indices for elite selection (prevents leakage)
    // Anti-overfit CV fitness. Each GENERATION reshuffles `sub` into `cv_k` disjoint folds (fresh
    // seed per gen -> a feature must generalize across MANY random partitions to keep surviving,
    // preventing lock-in to one slice). Fitness gain = mean_k(signal) - cv_penalty*std_k(signal),
    // a harsh penalty on cross-fold inconsistency. cv_k<=1 && cv_penalty==0 == legacy single-slice.
    pub cv_k: usize,
    pub cv_penalty: f32,
    /// Negative-control sentinels present in the GP search matrix. Expressions that reference one
    /// may be surfaced live as flagged candidates, but are rejected before survivor admission.
    pub canary_specs: Vec<CanarySpec>,
}

impl GpData {
    pub fn new(
        x: Matrix,
        names: Vec<String>,
        r: Vec<f32>,
        y: &[f32],
        seed: u64,
        m_full: usize,
    ) -> Self {
        Self::new_with_holdout(x, names, r, y, seed, m_full, 0.2)
    }

    pub fn new_with_holdout(
        x: Matrix,
        names: Vec<String>,
        r: Vec<f32>,
        y: &[f32],
        seed: u64,
        m_full: usize,
        holdout_ratio: f32,
    ) -> Self {
        let n = x.n;
        let lib_std: Vec<Vec<f32>> = (0..x.c).map(|j| standardize(x.col(j)).0).collect();
        let r_std = standardize(&r).0;
        let r_sse = sse(&r);
        let y_std = standardize(y).0;
        let msub = m_full.min(n);
        let sub = {
            let mut sr = seeded(seed ^ 0x5EED_5AB5);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = sr.gen_range(0..=i);
                idx.swap(i, j);
            }
            idx.truncate(msub);
            idx
        };
        let holdout_size = ((msub as f32 * holdout_ratio).round() as usize).min(sub.len());
        let holdout = sub[..holdout_size].to_vec();
        // Legacy defaults; the pipeline sets the task-adaptive CV-fitness policy (see run_pipeline).
        let cv_k = 1;
        let cv_penalty = 0.0;
        GpData {
            x,
            names,
            r,
            r_std,
            r_sse,
            y_std,
            lib_std,
            sub,
            holdout,
            cv_k,
            cv_penalty,
            canary_specs: Vec::new(),
        }
    }
}

/// Phase-1 configuration.
#[derive(Clone)]
pub struct GpConfig {
    pub pop: usize,
    pub gens: usize,
    pub depth: usize,
    pub islands: usize,
    pub tau: f32,
    pub mu: f32,
    pub lin_w: f32,
    pub max_surv: usize,
    pub migrate_every: usize,
    pub subsample: usize,     // m_full (sketch width)
    pub subsample_fit: usize, // m_fit (coarse prefix)
    pub seed: u64,
    pub holdout_ratio: f32,
}

impl GpConfig {
    pub fn new(
        pop: usize,
        gens: usize,
        depth: usize,
        islands: usize,
        tau: f32,
        mu: f32,
        lin_w: f32,
        max_surv: usize,
        migrate_every: usize,
        subsample: usize,
        subsample_fit: usize,
        seed: u64,
        holdout_ratio: f32,
    ) -> Self {
        GpConfig {
            pop,
            gens,
            depth,
            islands,
            tau,
            mu,
            lin_w,
            max_surv,
            migrate_every,
            subsample,
            subsample_fit,
            seed,
            holdout_ratio,
        }
    }
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig {
            pop: 4000,
            gens: 10,
            depth: 5,
            islands: 8,
            tau: 0.5,
            mu: 0.000003,
            lin_w: 0.05,
            max_surv: 60000,
            migrate_every: 2,
            subsample: 8192,
            subsample_fit: 1024,
            seed: 20260616,
            holdout_ratio: 0.2,
        }
    }
}

/// A surviving feature + its provenance.
#[derive(Clone, Serialize)]
pub struct SurvivorRec {
    pub expr: Expr,
    pub formula: String,
    pub signal: f32,
    pub proxy_gain: f32,
    pub lin_corr: f32,
    pub maxcorr_library: f32,
    pub target_corr: f32,
    pub depth: usize,
    pub nodes: usize,
}
