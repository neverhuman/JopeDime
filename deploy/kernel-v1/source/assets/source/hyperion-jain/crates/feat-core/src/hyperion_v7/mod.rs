//! Hyperion v7 — the TRUE model forward + composite randomized sweep.
//!
//! Vendored and adapted from the veox-bootstrap reference (`veox-split/veox-enclave/apps/
//! veox-bootstrap/src/hyperion`, sweep driver/patch/flush + candle model); the per-cell confidence
//! head is reconstructed from the training reference (`MQOTRCheat.py::MatrixGhostUnified`).
//! Workspace license applies; the model weights are own-trained ("Hyperion Entanglement" family).
//!
//! Layering:
//! - `grid`, `config`, `frames`, `sweep` are **candle-free** (compiled always, unit-tested in the
//!   fast CI lane via a fake patch model).
//! - `ops`, `mha`, `pooling`, `model`, `backend`, `weights` hold the candle inference stack and are
//!   gated on `hyperion-cpu` (+`hyperion-cuda` for GPU device selection).
//!
//! The composite sweep: K sweeps (default 30). Sweep 0 uses IDENTITY row/col permutations (the UI
//! shows a moving front); sweeps 1 through K-1 use seeded shuffles (updates appear as random dots at
//! their ORIGINAL positions). Each sweep tiles the matrix at every configured scale (areas, e.g. 3,5,7,9),
//! batches patches through the model, and scatters BOTH outputs back to original coordinates with
//! order-independent weighted running means:
//! - per-row 128-dim embeddings (the `hyperion::v000..127` feature columns), and
//! - per-cell confidence logits (sigmoid applied on the AVERAGED logit, mirroring the reference
//!   `P_map = sum_logits / count`).

pub mod config;
pub mod frames;
pub mod grid;
pub mod sweep;

#[cfg(feature = "hyperion-cpu")]
pub mod backend;
#[cfg(feature = "hyperion-cpu")]
pub mod mha;
#[cfg(feature = "hyperion-cpu")]
pub mod model;
#[cfg(feature = "hyperion-cpu")]
pub mod ops;
#[cfg(feature = "hyperion-cpu")]
pub mod pooling;
#[cfg(feature = "hyperion-cpu")]
pub mod weights;

/// Embedding dims kept for the pipeline feature columns: the shared head's first 128 dims (the
/// band-gated remainder of the 512-dim embedding is scale-sparse and mostly zero after averaging).
pub const EMB_KEEP: usize = 128;
