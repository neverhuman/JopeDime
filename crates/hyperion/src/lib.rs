//! Hyperion v5/v7 candle inference, ported from jain-core `feat-core` (see README.md for the
//! file-by-file provenance).
//!
//! Modules `hyperion`, `hyperion_v7` and `hyperion_router` are feat-core's, byte-for-byte except
//! for the edits listed in the README. `data`, `progress`, `rng` and `hashing` are the feat-core
//! support modules the encoder needs; `battle_gpu` is jain-battle-gpu's std-only shared-GPU lease
//! queue, kept under its old path so `battle_gpu::queue::…` keeps resolving.

/// Retrain hint feat-core's `progress` events carry; kept verbatim so `progress.rs` is unchanged.
pub const RETRAIN_WITH_JAIN_10: &str = "retrain with Jain 10.0.0";

pub mod battle_gpu;
pub mod data;
pub mod diagnostics;
pub mod hashing;
pub mod hyperion;
pub mod hyperion_router;
pub mod hyperion_v7;
pub mod progress;
pub mod rng;
