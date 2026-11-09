//! Read-only fused-ranker components.

pub mod adapter;
pub mod admission;
pub mod bank;
pub mod bundle;
pub mod cuda;
pub mod featurizer;
pub mod h256;
#[path = "landmarks.rs"]
pub mod landmarks;
pub mod lm48;
pub mod proj16;
pub mod stage1b;

/// File names inside an authenticated internal ranker package. The public jAIn CLI edition
/// carries no model weights and must not name them either (owner rule 2026-11-06); there the
/// names are placeholders that no package can contain, so every lookup reports a missing asset.
pub mod files {
    #[cfg(not(feature = "public-edition"))]
    pub const DOPE: &str = "dope.safetensors";
    #[cfg(not(feature = "public-edition"))]
    pub const JOPE: &str = "jope.safetensors";
    /// The extension of every tensor file.
    #[cfg(not(feature = "public-edition"))]
    pub const TENSORS: &str = ".safetensors";
    #[cfg(feature = "public-edition")]
    pub const DOPE: &str = "internal/encoder";
    #[cfg(feature = "public-edition")]
    pub const JOPE: &str = "internal/ranker";
    #[cfg(feature = "public-edition")]
    pub const TENSORS: &str = ".internal";
}
