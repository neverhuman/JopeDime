//! Deterministic projections of immutable research evidence.

mod capsule;
mod compile;
mod render;

pub mod state;

pub use capsule::{EvidenceLocator, EvidenceValue, ResearchCapsuleV2, RoundInput, Stage};
pub use compile::{compile, CompileError, EvidenceResolver};
pub use render::{render_capsule, CapsuleBytes, RenderError, CAPSULE_BYTE_LIMIT};
