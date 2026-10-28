//! Versioned realization and validation. Keep v1 unchanged when introducing a new
//! renderer: add a module and dispatch arm, and retain the old golden journal.
#[expect(
    dead_code,
    reason = "retain the frozen v1 writing method alongside replay"
)]
mod v1;
mod v2;
mod v3;
use crate::{LineageFact, Links};
pub use v1::{Claim, Metric, Narration};
pub(crate) use v3::Renderer;
pub use v3::RENDERER;

pub(crate) fn rederive(
    revision: &str,
    fact: &LineageFact,
    links: &Links,
    evidence: &str,
    variant: Option<u8>,
) -> anyhow::Result<Narration> {
    match revision {
        v1::RENDERER => v1::Renderer::rederive(fact, links, evidence, variant),
        v2::RENDERER => v2::Renderer::rederive(fact, links, evidence, variant),
        v3::RENDERER => v3::Renderer::rederive(fact, links, evidence, variant),
        _ => anyhow::bail!("unsupported renderer revision"),
    }
}
pub use v1::safe;
