//! Versioned realization and validation. Keep v1 unchanged when introducing a new
//! renderer: add a module and dispatch arm, and retain the old golden journal.
mod refusal_aware;
mod scope_corrected;
#[expect(
    dead_code,
    reason = "retain the frozen v1 writing method alongside replay"
)]
mod v1;
use crate::{LineageFact, Links};
pub(crate) use scope_corrected::Renderer;
pub use scope_corrected::RENDERER;
pub use v1::{Claim, Metric, Narration};

pub(crate) fn rederive(
    revision: &str,
    fact: &LineageFact,
    links: &Links,
    evidence: &str,
    variant: Option<u8>,
) -> anyhow::Result<Narration> {
    match revision {
        v1::RENDERER => v1::Renderer::rederive(fact, links, evidence, variant),
        refusal_aware::RENDERER => {
            refusal_aware::Renderer::rederive(fact, links, evidence, variant)
        }
        scope_corrected::RENDERER => {
            scope_corrected::Renderer::rederive(fact, links, evidence, variant)
        }
        _ => anyhow::bail!("unsupported renderer revision"),
    }
}
pub use v1::safe;
