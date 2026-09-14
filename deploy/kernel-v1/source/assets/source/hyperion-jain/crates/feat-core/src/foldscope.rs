//! FoldScope — the cross-cutting fold-safety context.
//!
//! Every "fit on train only" operation takes a scope of GLOBAL row indices into the full train
//! matrices. Because the existing scorers already index global rows (via `Matrix::submatrix` over
//! `folds` of global indices), the flat *root* scope reproduces the pipeline's current single split
//! (drawn once at `pipeline.rs`, reused across phases) **byte-for-byte** — so the default path is
//! numerically identical to before, and nested/grouped evaluation is a drop-in extension.
use crate::hashing::fold_hash;
use crate::metrics::{stratified_kfold, stratified_subsample};
use crate::rng::seeded;

/// How the inner CV folds are formed. Only `Stratified` is constructed on the default path today;
/// `Group`/`Time` are wired by the eval-rigor track (B8/B9).
#[derive(Clone, Debug, PartialEq)]
pub enum FoldPolicy {
    Stratified {
        k: usize,
    },
    Group {
        k: usize,
        group_col: usize,
    },
    Time {
        k: usize,
        time_col: usize,
        embargo_frac: f64,
    },
}

impl FoldPolicy {
    pub fn k(&self) -> usize {
        match self {
            FoldPolicy::Stratified { k }
            | FoldPolicy::Group { k, .. }
            | FoldPolicy::Time { k, .. } => *k,
        }
    }
    /// Stable discriminant for the fold hash (policy-sensitive: different policy => different hash).
    pub fn tag(&self) -> u64 {
        match self {
            FoldPolicy::Stratified { k } => 0x5111 ^ (*k as u64),
            FoldPolicy::Group { k, group_col } => 0x6222 ^ (*k as u64) ^ ((*group_col as u64) << 8),
            FoldPolicy::Time { k, time_col, .. } => {
                0x7333 ^ (*k as u64) ^ ((*time_col as u64) << 8)
            }
        }
    }
}

/// Which rows of the FULL train matrices the current context may touch, and how those rows are
/// partitioned into inner CV folds. All indices are GLOBAL (into `train.x` / `train.y`).
#[derive(Clone)]
pub struct FoldScope {
    /// Rows fitting is allowed to touch ("outer train").
    pub active: Vec<usize>,
    /// Inner CV partition over `active` (global indices).
    pub folds: Vec<(Vec<usize>, Vec<usize>)>,
    /// Outer-valid / audit rows (empty for the flat root scope).
    pub audit: Vec<usize>,
    pub policy: FoldPolicy,
    pub fold_hash: u64,
}

impl FoldScope {
    /// Flat root scope == today's behavior: `stratified_subsample(select_subsample)` with the SAME
    /// `seed.wrapping_add(777)` the pipeline uses, then `stratified_kfold`. Reproduces the existing
    /// split exactly (the A1 numeric-identity gate).
    pub fn root_stratified(y: &[f32], select_subsample: usize, k: usize, seed: u64) -> Self {
        let mut rng = seeded(seed.wrapping_add(777));
        let active = stratified_subsample(y, select_subsample.min(y.len()), &mut rng);
        Self::from_active_stratified(active, y, k, seed)
    }

    pub fn from_active_stratified(active: Vec<usize>, y: &[f32], k: usize, seed: u64) -> Self {
        let folds = stratified_kfold(&active, y, k, seed);
        let policy = FoldPolicy::Stratified { k };
        let mut act_sorted = active.clone();
        act_sorted.sort_unstable();
        let fh = fold_hash(policy.tag(), seed, &act_sorted, &folds);
        FoldScope {
            active,
            folds,
            audit: Vec::new(),
            policy,
            fold_hash: fh,
        }
    }

    /// Restrict to an outer-train row set, building a fresh inner stratified partition; the outer-valid
    /// rows become this scope's `audit` set. Used by the nested-CV runners.
    pub fn restrict(
        &self,
        outer_train: &[usize],
        outer_valid: &[usize],
        y: &[f32],
        seed: u64,
    ) -> Self {
        let folds = stratified_kfold(outer_train, y, self.policy.k(), seed);
        let mut act_sorted = outer_train.to_vec();
        act_sorted.sort_unstable();
        let fh = fold_hash(self.policy.tag(), seed, &act_sorted, &folds);
        FoldScope {
            active: outer_train.to_vec(),
            folds,
            audit: outer_valid.to_vec(),
            policy: self.policy.clone(),
            fold_hash: fh,
        }
    }
}
