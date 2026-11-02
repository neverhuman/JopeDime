//! Deterministic projections of immutable research evidence.

mod capsule;
mod compile;
mod render;

pub mod record_reader;

pub mod state;
pub mod wave_git;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResearchEvidenceStage {
    Synthetic,
    Shadow,
    Adaptive,
    Confirmatory,
}

/// Compatibility name for the evidence stage used by research projections.
pub type Stage = ResearchEvidenceStage;

impl ResearchEvidenceStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Shadow => "shadow",
            Self::Adaptive => "adaptive",
            Self::Confirmatory => "confirmatory",
        }
    }
}

pub use capsule::{EvidenceLocator, EvidenceValue, ResearchCapsuleV2, RoundInput};
pub use compile::{compile, CompileError, EvidenceResolver};
pub use render::{render_capsule, CapsuleBytes, RenderError, CAPSULE_BYTE_LIMIT};
pub use state::*;
