//! jope dataset-embedding assembly (P2): reproduce jope's 1536-d input vector in Rust
//! with parity (~1e-5) to the Python `embedding_from_vs` path.
//!
//! z_in(1536) = concat[ hyperion-512 (signature over the .vs rows, standardized),
//!                      foundation-1024 (PCA affine of the cached 2048-d foundation raw) ]
//! target(512) = mean over the target .vs rows.
//!
//! Deterministic, pure-Rust: reads the Hyperion `.vs` sidecars (CSV), the cached foundation
//! `<hash>.npy` (2048 f32), the `foundation_projection.npz` PCA arrays, and `standardization.json`.
//! Mirrors `jope/src/jope/propose.py:embedding_from_vs`, `data/vs.py:dataset_signature_from_vs`,
//! `encoders/projection.py:block_for_hash`, `data/target_encoding.py:target_vs_embedding`.

mod preprocessing;
mod projection;
#[cfg(test)]
mod tests;

pub use preprocessing::{
    JopeDatasetInputV1, JopeEmbeddingV1, JopeEncodeOptions, JopeEncoder, JopePreprocessingV1,
    JOPE_EMBEDDING_V1,
};
pub use projection::{
    jope_embedding, jope_embedding_from_parts, load_npy_f32, load_projection, load_std_cfg,
    FoundationProj, StdCfg,
};
