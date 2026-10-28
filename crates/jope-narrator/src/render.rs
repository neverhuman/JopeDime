//! Versioned realization and validation. Keep v1 unchanged when introducing a new
//! renderer: add a module and dispatch arm, and retain the old golden journal.
mod v1;
use crate::{LineageFact, Links};
pub(crate) use v1::Renderer;
pub use v1::{Claim, Metric, Narration, RENDERER};

pub(crate) fn rederive(
    revision: &str,
    fact: &LineageFact,
    links: &Links,
    evidence: &str,
    variant: Option<u8>,
) -> anyhow::Result<Narration> {
    match revision {
        "controlled-search-commentary/1" => v1::Renderer::rederive(fact, links, evidence, variant),
        _ => anyhow::bail!("unsupported renderer revision"),
    }
}
pub use v1::safe;
