//! Optional Starforge final-stage candidate and serving support.
//!
//! The types in this module are always available so manifests remain readable without the
//! optional Candle dependency. Actual inference is compiled only under `starforge-cpu` or
//! `starforge-cuda`.

mod config;
mod context;
mod session;

pub use config::*;
pub use context::*;
pub use session::*;
