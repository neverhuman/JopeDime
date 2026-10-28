//! Renderer v3 corrects two scope descriptions while preserving v2 replay.
use super::v1::{safe, Narration};
use super::v2;
use crate::{LineageFact, Links};
use anyhow::Result;

pub const RENDERER: &str = "controlled-search-commentary/3";

pub(crate) struct Renderer(v2::Renderer);
impl Renderer {
    pub(crate) fn new() -> Self {
        Self(v2::Renderer::new())
    }
    pub(crate) fn render(
        &mut self,
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
    ) -> Narration {
        correct_scope(fact, self.0.render(fact, links, evidence))
    }
    pub(crate) fn rederive(
        fact: &LineageFact,
        links: &Links,
        evidence: &str,
        variant: Option<u8>,
    ) -> Result<Narration> {
        Ok(correct_scope(
            fact,
            v2::Renderer::rederive(fact, links, evidence, variant)?,
        ))
    }
}

fn correct_scope(fact: &LineageFact, mut narration: Narration) -> Narration {
    match fact {
        LineageFact::Admitted {
            cut,
            program_sha256,
            disposition,
            ..
        } if disposition == "evaluations" => {
            // Multi-cut exploration reserves part of the cut budget for comparison.
            // This fact carries no phase, so describe only the reached bound.
            let handle = safe(&program_sha256.chars().take(12).collect::<String>());
            narration.text = format!(
                "Cut {cut}, candidate {handle}: not admitted: the evaluation bound was reached."
            );
        }
        LineageFact::ExplorationStopped {
            cut,
            reason,
            evaluations,
            ..
        } if reason == "search_space_exhausted" => {
            narration.text = format!(
                "Cut {cut}: exploration stopped after {evaluations} evaluations; consecutive generations spent no evaluations."
            );
        }
        _ => {}
    }
    narration
}
