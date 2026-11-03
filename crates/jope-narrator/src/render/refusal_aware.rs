//! Renderer v2: explicit refusal decisions and corrected sentence grammar.
//! The v1 implementation and transcripts remain immutable for retained replay.
use super::v1::{implementation_notes, resolution, Claim, Narration};
use crate::{diff_programs, digest, AtomicEdit, LineageFact, Links, ProgramDelta, ProgramSnapshot};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
pub const RENDERER: &str = "controlled-search-commentary/2";

pub(crate) struct Renderer {
    recent: VecDeque<usize>,
}
impl Renderer {
    pub(crate) fn new() -> Self {
        Self {
            recent: VecDeque::new(),
        }
    }
    pub(crate) fn render(
        &mut self,
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
    ) -> Narration {
        self.render_with_variant(fact, links, evidence, None)
    }
    pub(crate) fn rederive(
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
        variant: Option<u8>,
    ) -> Result<Narration> {
        ensure!(variant.is_none_or(|v| v < 8), "invalid grammar variant");
        ensure!(
            matches!(fact, LineageFact::Drawn { .. }) || variant.is_none(),
            "unexpected grammar variant"
        );
        Ok(Self::new().render_with_variant(fact, links, evidence, variant))
    }
}

mod events;
mod text;

pub use text::safe;
use text::{append_boundary_note, edit_text, evidence_refs, hypothesis, short};
