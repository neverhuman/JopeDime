//! Diverse pure-Rust base learners (no FFI, no external crates) that add inductive-bias DIVERSITY to the
//! GBDT-heavy ensemble — the lever for "beat any single model." Each implements the `Backend`/`Model`
//! traits, so it becomes a fold-pure OOF member via `ensemble::build_oof_member` (inheriting its leak
//! proofs); dense learners fit their `StandardScaler` INSIDE `train_single`, so they're leak-safe for free.
pub mod bma;
pub mod glm;
pub mod math;
pub mod rff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmFamily {
    Logistic,
    Gaussian,
}
