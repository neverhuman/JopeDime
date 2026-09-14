//! Phase 1 — genetic-programming feature synthesis with islands + an exact-admission survivor
//! universe. Ported from tools/gpfeat/src/main.rs.
//!
//! Fitness = signal * diversity(1 - maxcorr/tau) - parsimony, where
//!   signal  = split_gain(feature, residual) + lin_w * |corr(feature, residual)|.
//! Survivors are pairwise |corr| < tau (exact, full-data check) and decorrelated from the base
//! "library" columns — i.e. the set of high-value, mutually-novel engineered features.

mod eval;
mod random;
mod run;
mod types;

pub use run::run_gp;
pub use types::{GpConfig, GpData, SurvivorRec};
