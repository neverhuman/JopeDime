//! Rust-first JOPE self-play control plane.
//!
//! Python is intentionally restricted to the PyTorch training/proposal
//! boundary. Queueing, provenance, fitness bookkeeping, freshness, datasets,
//! and deployment policy live in this crate.

pub mod selfplay_core;
pub mod selfplay_data;
pub mod selfplay_expansion;
pub mod selfplay_ingest;
pub mod selfplay_registry;
pub mod selfplay_runtime;
pub mod selfplay_shadow;
/// WS-2: the single source of truth for scq job specs, and the drift test that
/// fails when two renderers disagree.
pub mod spec_drift;
pub mod spec_policy;
pub mod timeseries;
pub mod training_history;
