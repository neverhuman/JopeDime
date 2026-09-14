//! feat-core — the 100%-Rust feature-engineering pipeline library.
//!
// Intentional: erf coefficients are exact (Python-parity); the GP hot path takes many params by
// design; the negated float comparison in `standardize` is deliberate for NaN-safety.
#![allow(
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::neg_cmp_op_on_partial_ord
)]
//!
//! Phase 1: GP feature synthesis (gp + dsl, residual proxy target). Phase 2: importance epochs.
//! Phase 3/4: EDA/PBIL selection (single-target / MIMO). `pipeline` orchestrates all phases and
//! reports lift; every phase emits `progress::Event`s and calls a `Backend` (CatBoost or Mock).
pub use domain::{ErrorCode, JainError, RepairHint, Result as JainResult};

pub mod advval;
pub mod artifact;
pub mod backend;
pub mod bakeoff;
pub mod bench;
pub mod buildmatrix;
pub mod calibration;
pub mod canary;
pub mod data;
pub mod dominance;
pub mod dsl;
pub mod encode;
pub mod ensemble;
pub mod foldscope;
pub mod gp;
pub mod hashing;
pub mod hyperion;
pub mod hyperion_v7;
pub mod ingest;
pub mod invention;
pub mod leakage;
pub mod learners;
pub mod linalg;
pub mod meta;
pub mod metrics;
pub mod missing;
pub mod mvp;
pub mod naming;
pub mod phase2;
pub mod phase3;
pub mod phase4;
pub mod pipeline;
pub mod preprocess;
pub mod progress;
pub mod residual;
pub mod rng;
pub mod router;
pub mod select;
pub mod stacking;
pub mod starforge_integration;
pub mod targets;
pub mod task;
